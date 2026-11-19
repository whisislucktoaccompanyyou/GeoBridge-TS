"""Differentiable core: never decorate forward_probability with no_grad."""
import torch
from torch import nn
from torch.nn import functional as F


class TinyCore(nn.Module):
    def __init__(self, inputs=1, channels=8):
        super().__init__()
        self.net = nn.Sequential(nn.Conv2d(inputs, channels, 3, padding=1), nn.SiLU(),
                                 nn.Conv2d(channels, channels, 3, padding=1), nn.SiLU(), nn.Conv2d(channels, 1, 1))

    def forward(self, x):
        return self.net(x)


def legacy_modules(cfg):
    from model.legacy.unet import UNet
    from model.legacy.spade.generator import SPADEGenerator
    c = cfg['model']
    unet = UNet(in_channel=1, out_channel=1, inner_channel=c['inner_channels'],
                channel_mults=c['channel_mults'], attn_res=c['attention_resolutions'],
                res_blocks=c['res_blocks'], dropout=0, image_size=cfg['data']['image_size'])
    return unet, SPADEGenerator(2, 1)


class SegmentationCore(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.backend = cfg['model']['backend']
        if self.backend == 'tiny':
            self.net = TinyCore()
        else:
            self.denoise_fn, self.segment_fn = legacy_modules(cfg)

    def forward_probability(self, image):
        if self.backend == 'tiny':
            return self.net(image).sigmoid()
        t = torch.zeros(image.shape[0], dtype=torch.long, device=image.device)
        feature = self.denoise_fn(image, t)
        return ((self.segment_fn(torch.cat([image, feature], 1)) + 1) / 2).clamp(1e-6, 1-1e-6)

    def forward(self, image):
        return self.forward_probability(image)


class InitialGenerator(nn.Module):
    """Original B source synthesis: q_sample(reference,t) -> denoiser -> SPADE(P)."""
    def __init__(self, cfg):
        super().__init__()
        c = cfg['model']
        self.backend, self.timestep = c['backend'], c['g0_timestep']
        if not 0 <= self.timestep < c['diffusion_steps']:
            raise ValueError('g0_timestep 超出扩散 schedule')
        alphas = (1-torch.linspace(c['beta_start'], c['beta_end'], c['diffusion_steps'], dtype=torch.float64)).cumprod(0)
        self.register_buffer('alpha_bar', alphas.float())
        if self.backend == 'tiny':
            self.net = TinyCore(inputs=3)
        else:
            self.denoise_fn, self.segment_fn = legacy_modules(cfg)

    def forward(self, mask, reference, noise=None, return_denoising=False):
        noise = torch.randn_like(reference) if noise is None else noise
        a = self.alpha_bar[self.timestep]
        noisy = a.sqrt()*reference + (1-a).sqrt()*noise
        if self.backend == 'tiny':
            out = self.net(torch.cat([mask*2-1, reference, noise], 1)).tanh()
            denoise = out.sum()*0
        else:
            t = torch.full((reference.shape[0],), self.timestep, dtype=torch.long, device=reference.device)
            latent = self.denoise_fn(noisy, t)
            condition = torch.cat([1-mask, mask], 1)
            out = self.segment_fn(torch.cat([noisy, latent], 1), condition)
            denoise = F.mse_loss(latent, noise)
        return (out, denoise) if return_denoising else out


def make_discriminator(cfg):
    if cfg['model']['backend'] == 'tiny':
        return TinyCore()
    from model.legacy.discriminators import NLayerDiscriminator
    return NLayerDiscriminator(1)


def freeze(model):
    model.eval()
    model.requires_grad_(False)
    return model


def assert_independent(*models):
    seen = set()
    for model in models:
        pointers = {p.untyped_storage().data_ptr() for p in list(model.parameters())+list(model.buffers()) if p.numel()}
        if pointers & seen:
            raise ValueError('模型参数/缓冲存储共享，不允许同时冻结与更新')
        seen |= pointers
