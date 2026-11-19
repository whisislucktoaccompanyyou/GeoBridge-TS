import torch
from training.losses import geometry_energy


def gpg_correct(endpoint, teacher, mask, geometry, weight, cfg):
    """Optimize only detached image y; teacher fixed, differentiable w.r.t. y."""
    if cfg['prox_weight'] <= 0:
        raise ValueError('近端权重必须为正')
    if any(p.requires_grad for p in teacher.parameters()) or teacher.training:
        raise ValueError('GPG 要求固定且 eval 的教师')
    if not weight.any():
        return endpoint.detach(), dict(skipped=True, before=0., after=0., max_change=0., gradient_norm=0.)
    origin = endpoint.detach().float()
    w = weight.float()
    def geo(y):
        return geometry_energy(teacher.forward_probability(y), mask.float(), geometry.float(), w,
                               cfg['mask_weight'], cfg['boundary_weight'])
    with torch.enable_grad():
        y = origin.clone().requires_grad_(True)
        before = float(geo(y).detach())
        max_norm = 0.
        for _ in range(cfg['iterations']):
            proximal = cfg['prox_weight']*((y-origin).square()*w).sum()/w.sum().clamp_min(1)
            energy = geo(y)+proximal
            grad, = torch.autograd.grad(energy, y)
            if not torch.isfinite(grad).all():
                raise RuntimeError('GPG 非有限图像梯度')
            max_norm = max(max_norm, float(grad.norm()))
            # Mean losses have O(1/pixels) derivatives; explicit scale invariant step.
            direction = grad*w.sum().clamp_min(1)/len(y)
            step = cfg['learning_rate']
            for _backtrack in range(12):
                candidate = y.detach()-step*direction.detach()
                candidate = torch.maximum(torch.minimum(candidate, origin+cfg['max_pixel_change']), origin-cfg['max_pixel_change'])
                candidate = candidate.clamp(-1, 1)
                candidate = torch.where(w > 0, candidate, origin)
                with torch.no_grad():
                    candidate_energy = geo(candidate)+cfg['prox_weight']*((candidate-origin).square()*w).sum()/w.sum().clamp_min(1)
                if candidate_energy <= energy.detach():
                    y = candidate.requires_grad_(True)
                    break
                step *= .5
            else:
                break
        after = float(geo(y).detach())
    return y.detach(), dict(skipped=False, before=before, after=after,
                            max_change=float((y.detach()-origin).abs().max()), gradient_norm=max_norm)
