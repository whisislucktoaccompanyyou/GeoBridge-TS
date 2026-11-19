import math
import torch
from torch import nn
from torch.nn import functional as F


class TimeEmbedding(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.mlp = nn.Sequential(nn.Linear(channels, channels*4), nn.SiLU(), nn.Linear(channels*4, channels*4))

    def forward(self, t):
        freq = torch.exp(torch.arange(self.channels//2, device=t.device)*(-math.log(10000)/max(self.channels//2-1, 1)))
        angles = t[:, None]*freq[None, :]*1000
        return self.mlp(torch.cat([angles.sin(), angles.cos()], 1))


class Residual(nn.Module):
    def __init__(self, inputs, outputs, time_channels):
        super().__init__()
        self.a = nn.Conv2d(inputs, outputs, 3, padding=1)
        self.norm1 = nn.GroupNorm(math.gcd(outputs, 8), outputs)
        self.norm2 = nn.GroupNorm(math.gcd(outputs, 8), outputs)
        self.b = nn.Conv2d(outputs, outputs, 3, padding=1)
        self.time = nn.Linear(time_channels, outputs)
        self.skip = nn.Conv2d(inputs, outputs, 1) if inputs != outputs else nn.Identity()

    def forward(self, x, t):
        h = self.a(x)
        h = self.norm1(h)+self.time(t)[:, :, None, None]
        return self.skip(x)+self.b(F.silu(self.norm2(F.silu(h))))


class EndpointUNet(nn.Module):
    """Six spatial channels plus explicit time; inject geometry at every scale."""
    def __init__(self, channels=32, mults=(1, 2, 4, 4)):
        super().__init__()
        self.time = TimeEmbedding(channels)
        widths = [channels*m for m in mults]
        self.entry = nn.Conv2d(6, widths[0], 3, padding=1)
        self.down = nn.ModuleList([Residual(widths[max(i-1, 0)], w, channels*4) for i, w in enumerate(widths)])
        self.geo = nn.ModuleList([nn.Conv2d(4, w, 1) for w in widths])
        self.mid = Residual(widths[-1], widths[-1], channels*4)
        self.up = nn.ModuleList([Residual(widths[i+1]+widths[i], widths[i], channels*4) for i in reversed(range(len(widths)-1))])
        self.output = nn.Conv2d(widths[0], 1, 3, padding=1)

    def predict_endpoint(self, x_t, t, x0, geometry):
        if x_t.shape != x0.shape or geometry.shape[1] != 4 or t.shape != (len(x_t),):
            raise ValueError('桥输入形状错误')
        if (t < 0).any() or (t >= 1).any():
            raise ValueError('桥时间必须在[0,1)')
        embedding = self.time(t)
        h = self.entry(torch.cat([x_t, x0, geometry], 1))
        skips = []
        for i, layer in enumerate(self.down):
            if i:
                h = F.avg_pool2d(h, 2)
            h = layer(h, embedding)+self.geo[i](F.interpolate(geometry, h.shape[-2:], mode='nearest'))
            skips.append(h)
        h = self.mid(h, embedding)
        for layer, skip in zip(self.up, reversed(skips[:-1])):
            h = F.interpolate(h, skip.shape[-2:], mode='bilinear', align_corners=False)
            h = layer(torch.cat([h, skip], 1), embedding)
        return self.output(h).tanh()

    forward = predict_endpoint


def bridge_state(x0, x1, t, reference_variance, noise=None):
    if reference_variance <= 0 or (t <= 0).any() or (t >= 1).any():
        raise ValueError('桥训练时间须避开端点，epsilon必须正')
    noise = torch.randn_like(x0) if noise is None else noise
    time = t[:, None, None, None]
    return (1-time)*x0+time*x1+(reference_variance*time*(1-time)).sqrt()*noise


def endpoint_to_drift(endpoint, state, t, delta):
    return (endpoint-state)/(1-t[:, None, None, None]).clamp_min(delta)


def sample_bridge(network, x0, geometry, mask, valid, label_weight, teacher, bridge_cfg, gpg_cfg, generator=None):
    """Inference intentionally has no x1/target-photo argument. No inference_mode around GPG."""
    from model.ot_bridge.gpg import gpg_correct
    state = x0.detach().clone()
    delta, count = bridge_cfg['delta'], bridge_cfg['sampling_steps']
    dt = (1-delta)/count
    records = []
    for k in range(count):
        time = k*dt
        t = state.new_full((len(state),), time)
        with torch.no_grad():
            endpoint = network.predict_endpoint(state, t, x0, geometry)
        drift = endpoint_to_drift(endpoint, state, t, delta)
        if (k+1) % gpg_cfg['every'] == 0:
            corrected, diag = gpg_correct(endpoint, teacher, mask, geometry, valid*label_weight, gpg_cfg)
            drift = drift+gpg_cfg['kappa_max']*time*(corrected-endpoint)/max(1-time, delta)
            records.append(dict(step=k, time=time, **diag))
        with torch.no_grad():
            noise = torch.randn(state.shape, device=state.device, dtype=state.dtype, generator=generator)
            state = state+dt*drift+math.sqrt(bridge_cfg['reference_variance']*dt)*noise
    with torch.no_grad():
        t = state.new_full((len(state),), 1-delta)
        endpoint = network.predict_endpoint(state, t, x0, geometry)
    final, diag = gpg_correct(endpoint, teacher, mask, geometry, valid*label_weight, gpg_cfg)
    records.append(dict(step=count, time=1-delta, final=True, **diag))
    return final.detach(), records
