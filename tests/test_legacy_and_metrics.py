import copy
import numpy as np
import torch
from model.segmentation_core import SegmentationCore, InitialGenerator, assert_independent, freeze
from training.evaluation import segmentation_metrics, pr_auc


def test_reused_legacy_core_has_gpg_input_gradient(base):
    c = copy.deepcopy(base)
    c['data']['image_size'] = 64
    c['model'].update(inner_channels=32, channel_mults=[1, 2], attention_resolutions=[], res_blocks=1)
    student = SegmentationCore(c)
    generator = InitialGenerator(c)
    assert_independent(student, generator)
    teacher = freeze(student)
    x = torch.randn(1, 1, 64, 64, requires_grad=True)
    probability = teacher.forward_probability(x)
    gradient, = torch.autograd.grad(probability.mean(), x)
    assert probability.shape == x.shape and float(gradient.abs().sum()) > 0
    assert all(p.grad is None for p in teacher.parameters())
    # Real original G0 source path, not the tiny smoke backend.
    y, denoise = generator((x.detach()>0).float(), x.detach(), return_denoising=True)
    (y.square().mean()+denoise).backward()
    assert any(p.grad is not None and float(p.grad.abs().sum()) > 0 for p in generator.parameters())


def test_metrics_fov_and_zero_denominator():
    mask = np.array([[255, 255, 0], [0, 0, 0]], np.uint8)
    prob = np.array([[.9, .1, .8], [.1, .2, .9]])
    fov = np.array([[255, 255, 255], [255, 255, 0]], np.uint8)
    r = segmentation_metrics(prob, mask, fov, .5)
    assert (r['tp'], r['fp'], r['fn'], r['tn']) == (1, 1, 1, 2)
    assert r['dice'] == .5 and r['score'] == 50
    empty = segmentation_metrics(np.zeros((2, 2)), np.zeros((2, 2)), np.ones((2, 2)), .5)
    assert empty['dice'] is None and empty['precision'] is None and empty['recall'] is None
    assert abs(pr_auc([False, True, False, True], [.1, .9, .2, .8])-1.) < 1e-12
