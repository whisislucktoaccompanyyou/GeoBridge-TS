import torch
from torch.nn import functional as F


def masked_bce(prob, target, weight):
    with torch.autocast(device_type=prob.device.type, enabled=False):
        return (F.binary_cross_entropy(prob.float().clamp(1e-6, 1-1e-6), target.float(), reduction='none')*weight.float()).sum()/weight.float().sum().clamp_min(1)


def segmentation_loss(prob, target, weight):
    dims = (1, 2, 3)
    w = weight.float()
    dice = 1-(2*(prob*target*w).sum(dims)+1e-6)/((prob*w).sum(dims)+(target*w).sum(dims)+1e-6)
    active = (w.sum(dims)>0).float()
    return masked_bce(prob, target, w)+(dice*active).sum()/active.sum().clamp_min(1)


def soft_boundary(prob):
    return F.max_pool2d(prob, 3, 1, 1) + F.max_pool2d(-prob, 3, 1, 1)


def geometry_energy(prob, mask, geometry, weight, mask_weight=1., boundary_weight=.1):
    # Target distance/boundary are fixed; prediction branch remains differentiable.
    edge = soft_boundary(prob)
    boundary = ((edge-geometry[:, 1:2]).abs()*weight*(1+geometry[:, 3:4].abs())).sum()/weight.sum().clamp_min(1)
    return mask_weight*segmentation_loss(prob, mask, weight)+boundary_weight*boundary
