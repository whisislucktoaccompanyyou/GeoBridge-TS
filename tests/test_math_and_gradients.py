import copy
import inspect
import numpy as np
import pytest
import torch
from torch import nn
from data.paired_pseudo import build_geometry, load_sample, restore_probability
from data.fundus_manifest import validate_splits
from model.ot_bridge.transport import solve_conditional_ot, log_sinkhorn
from model.ot_bridge.bridge import bridge_state, endpoint_to_drift, sample_bridge
from model.ot_bridge.gpg import gpg_correct
from model.ema_teacher import make_teacher, update_ema, uncertainty_target
from model.segmentation_core import SegmentationCore, freeze, assert_independent
from training.losses import segmentation_loss, geometry_energy


def test_ot_marginals_scale_and_condition_restriction():
    x = torch.tensor([[[[0., 1.]]], [[[1., 0.]]]])
    y = torch.tensor([[[[.2, .7]]], [[[.7, .2]]]])
    a = solve_conditional_ot(x, y, ['p', 'p'], ['p', 'p'], .2)
    b = solve_conditional_ot(x, y, ['p', 'p'], ['p', 'p'], .2, reduction='mean')
    torch.testing.assert_close(a.plan, b.plan, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(a.plan.sum(0), torch.full((2,), .5, dtype=torch.float64))
    torch.testing.assert_close(a.plan.sum(1), torch.full((2,), .5, dtype=torch.float64))
    with pytest.raises(ValueError, match='condition_id'):
        solve_conditional_ot(x, y, ['p', 'q'], ['p', 'p'], .2)
    with pytest.raises(RuntimeError, match='未收敛'):
        log_sinkhorn(torch.tensor([[0., 4.], [1., 2.]]), .1, tolerance=1e-20, max_iterations=1)


def test_bridge_variance_endpoint_formula_and_inference_signature():
    n = 40000
    x0, x1 = torch.zeros(n, 1, 1, 1), torch.ones(n, 1, 1, 1)
    t = torch.full((n,), .3)
    torch.manual_seed(77)
    state = bridge_state(x0, x1, t, .2)
    residual = state-.3
    assert abs(float(residual.var())-.2*.3*.7) < .002
    drift = endpoint_to_drift(x1, state, t, .001)
    torch.testing.assert_close(state+(1-t[:, None, None, None])*drift, x1)
    assert 'x1' not in inspect.signature(sample_bridge).parameters
    with pytest.raises(ValueError):
        bridge_state(x0[:1], x1[:1], torch.ones(1), .2)


class PixelTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.))

    def forward_probability(self, x):
        return (x*self.scale).sigmoid()


def test_gpg_decreases_objective_with_image_grad_and_fixed_teacher(base):
    teacher = freeze(PixelTeacher())
    before_weights = copy.deepcopy(teacher.state_dict())
    mask = torch.zeros(1, 1, 16, 16)
    mask[:, :, 3:13, 6:9] = 1
    geometry = build_geometry(mask)
    image = torch.zeros_like(mask)
    weight = torch.ones_like(mask)
    cfg = dict(base['gpg'], iterations=3, max_pixel_change=.2, boundary_weight=0., learning_rate=.1)
    corrected, diag = gpg_correct(image, teacher, mask, geometry, weight, cfg)
    assert diag['gradient_norm'] > 0 and diag['after'] < diag['before']
    assert float((corrected-image).abs().max()) <= .200001
    for k, v in teacher.state_dict().items():
        torch.testing.assert_close(v, before_weights[k], atol=0, rtol=0)
    assert teacher.scale.grad is None


def test_empty_regions_zero_losses_and_gpg_skip(base):
    p = torch.full((1, 1, 8, 8), .5, requires_grad=True)
    target = torch.zeros_like(p)
    zero = torch.zeros_like(p)
    loss = segmentation_loss(p, target, zero)
    loss.backward()
    assert loss == 0 and p.grad.abs().sum() == 0
    _, diag = gpg_correct(target, freeze(PixelTeacher()), target, build_geometry(target), zero, base['gpg'])
    assert diag['skipped']


def test_ema_buffers_copied_and_storage_independent():
    student = nn.Sequential(nn.Conv2d(1, 2, 1), nn.BatchNorm2d(2))
    teacher = make_teacher(student)
    assert_independent(student, teacher)
    old = [p.detach().clone() for p in teacher.parameters()]
    with torch.no_grad():
        for p in student.parameters():
            p.add_(2)
        student[1].running_mean.fill_(3)
        student[1].num_batches_tracked.fill_(17)
    update_ema(student, teacher, .9)
    for original, now in zip(old, teacher.parameters()):
        torch.testing.assert_close(now, original+.2)
    assert teacher[1].num_batches_tracked == 17
    torch.testing.assert_close(teacher[1].running_mean, student[1].running_mean)
    assert not teacher.training


def test_weak_prediction_alignment_not_dropout(mock_data):
    c, _, _ = mock_data
    x = torch.linspace(-1, 1, 32*32).reshape(1, 1, 32, 32)
    teacher = freeze(PixelTeacher())
    q, w, _ = uncertainty_target(teacher, x, torch.ones_like(x), 3, .5, 1.)
    assert torch.corrcoef(torch.stack([q.flatten(), x.flatten()]))[0, 1] > .99
    assert not q.requires_grad and w.sum() == x.numel()


def test_label_flip_and_letterbox_are_aligned(mock_data):
    c, rows, _ = mock_data
    a = load_sample(rows[0], 32, target='gt_mask_path')
    b = load_sample(rows[0], 32, target='gt_mask_path', flip=True)
    for k in ('image', 'mask', 'valid', 'weight'):
        torch.testing.assert_close(a[k].flip(-1), b[k])
    torch.testing.assert_close(build_geometry(b['mask']), b['geometry'])
    assert restore_probability(a['mask'], a['info']).shape == (rows[0]['height'], rows[0]['width'])


@pytest.mark.parametrize('duplicate', ['subject', 'image'])
def test_cross_split_leakage_rejected(duplicate):
    rows = [dict(sample_id='a', subject_id='person_a', image_sha256='hash_a', split='train'),
            dict(sample_id='b', subject_id='person_b', image_sha256='hash_b', split='test')]
    field = 'subject_id' if duplicate == 'subject' else 'image_sha256'
    rows[1][field] = rows[0][field]
    with pytest.raises(ValueError, match='泄漏'):
        validate_splits(rows)
