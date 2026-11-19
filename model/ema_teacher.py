import copy
import torch
from data.paired_pseudo import appearance
from model.segmentation_core import freeze


def make_teacher(student):
    return freeze(copy.deepcopy(student))


@torch.no_grad()
def update_ema(student, teacher, decay):
    if not 0 <= decay < 1:
        raise ValueError('EMA decay 必须在[0,1)')
    source = dict(student.named_parameters())
    for name, p in teacher.named_parameters():
        p.mul_(decay).add_(source[name], alpha=1-decay)
    source_buffers = dict(student.named_buffers())
    for name, value in teacher.named_buffers():
        value.copy_(source_buffers[name])
    teacher.eval()


@torch.no_grad()
def uncertainty_target(teacher, image, valid, repeats, confidence, variance_threshold):
    predictions = []
    for i in range(repeats):
        # Independent mild appearance draws; alternating invertible horizontal flips.
        weak = appearance(image)
        if i % 2:
            weak = weak.flip(-1)
        p = teacher.forward_probability(weak)
        if i % 2:
            p = p.flip(-1)
        predictions.append(p)
    stack = torch.stack(predictions)
    q, variance = stack.mean(0), stack.var(0, unbiased=False)
    reliable = ((torch.maximum(q, 1-q) >= confidence) & (variance <= variance_threshold)).to(valid)*valid
    count = reliable.sum().clamp_min(1)
    return q.detach(), reliable, dict(reliable_fraction=float(reliable.sum()/valid.sum().clamp_min(1)),
                                    reliable_foreground_fraction=float((reliable*(q>=0.5)).sum()/count),
                                    reliable_background_fraction=float((reliable*(q<0.5)).sum()/count))
