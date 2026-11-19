import copy
import json
from pathlib import Path
import numpy as np
import pytest
import torch
from PIL import Image
from configuration import config_version, sha256
from training.smoke import run_smoke
from training.checkpoints import read_checkpoint


def assert_nested_equal(a, b):
    if isinstance(a, torch.Tensor):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    elif isinstance(a, np.ndarray):
        np.testing.assert_array_equal(a, b)
    elif isinstance(a, dict):
        assert a.keys() == b.keys()
        for k in a:
            assert_nested_equal(a[k], b[k])
    elif isinstance(a, (tuple, list)):
        assert len(a) == len(b)
        for x, y in zip(a, b):
            assert_nested_equal(x, y)
    else:
        assert a == b


def test_complete_mock_pipeline_and_exact_cpu_resume(base, tmp_path, monkeypatch):
    import training.stages as stages
    original_save = stages.save_checkpoint
    step_one = []
    def capture(*args, **kwargs):
        original_save(*args, **kwargs)
        if args[1] == 'train-student-teacher' and args[2] == 1 and Path(args[0]).name == 'student_last.pt':
            step_one.append(Path(args[0]).read_bytes())
    monkeypatch.setattr(stages, 'save_checkpoint', capture)
    holder = {}
    result = run_smoke(base, tmp_path, before_student=lambda p: holder.update(pipeline=p))
    p = holder['pipeline']
    assert result['api_calls'] == 2 and result['student']['ema_steps'] == 2
    assert step_one
    last_path = p.ckpt('train-student-teacher')
    uninterrupted = torch.load(last_path, weights_only=False)
    # Restore the real checkpoint taken after the first update and execute the second.
    last_path.write_bytes(step_one[0])
    p.train('train-student-teacher', resume=True)
    resumed = torch.load(last_path, weights_only=False)
    for key in ('models', 'optimizers', 'schedulers', 'samplers', 'rng', 'scaler', 'ema_steps', 'step', 'generation_seed'):
        assert_nested_equal(uninterrupted[key], resumed[key])
    assert resumed['step'] == 2 and resumed['ema_steps'] == 2
    logs = [json.loads(s) for s in (tmp_path/'training.jsonl').read_text(encoding='utf-8').splitlines()]
    final_logs = [r for r in logs if r['stage'] == 'train-student-teacher']
    assert all(all(k in log for k in ('loss_a', 'loss_c', 'loss_u', 'reliable_fraction')) for log in final_logs)
    # S optimizer only owns S; fixed G0/bridge retain their pre-C fingerprints.
    g0 = read_checkpoint(p.ckpt('train-g0'), p.cfg)
    bridge = read_checkpoint(p.ckpt('train-bridge'), p.cfg)
    assert resumed['model_fingerprints']['g0'] == g0['model_fingerprints']['g0']
    assert resumed['model_fingerprints']['bridge'] == bridge['model_fingerprints']['bridge']
    assert set(resumed['optimizers']) == {'student'}
    for row in p.paired:
        native = Image.open(tmp_path/'synthetic'/(row['sample_id']+'.png'))
        assert native.size == (row['width'], row['height'])
        meta = json.loads((tmp_path/'synthetic'/(row['sample_id']+'.json')).read_text(encoding='utf-8'))
        assert meta['cache_row']['reference_sample_id'] != row['sample_id']
        assert any(x['gradient_norm'] > 0 for x in meta['gpg'])
    test = next(r for r in p.rows if r['split'] == 'test')
    with Image.open(tmp_path/'evaluation/test'/(test['sample_id']+'.png')) as prediction:
        assert prediction.size == (test['width'], test['height'])
    # Caches can recover an interrupted manifest write from committed per-image metadata.
    cache_manifest = tmp_path/'synthetic/manifest.jsonl'
    cache_manifest.unlink()
    p.synthesize(resume=True)
    assert len(cache_manifest.read_text(encoding='utf-8').splitlines()) == 2
    # Manual image-quality approval is a runtime gate, not a new scientific data version.
    other = copy.deepcopy(p.cfg)
    other['synthesis']['quality_approved'] = False
    assert config_version(other) == config_version(p.cfg)
    # None of the persisted artifacts contains the virtual test credential.
    for file in tmp_path.rglob('*'):
        if file.is_file():
            assert b'mock-secret-do-not-log' not in file.read_bytes()


def test_formal_config_rejects_mock_and_missing_budgets(mock_data):
    from training.stages import Pipeline
    c, _, _ = mock_data
    c['data']['allow_mock'] = False
    with pytest.raises(ValueError, match='模拟'):
        Pipeline(c)
    c['data']['allow_mock'] = True
    c['training']['warmup_steps'] = 0
    with pytest.raises(ValueError, match='预算'):
        Pipeline(c).train('warmup-student')


def test_binary_png_fixture_is_supported(base):
    from pseudo_labels.parsers import parse_response
    path = Path(__file__).parent/'fixtures/binary_png_response.json'
    response = json.loads(path.read_text(encoding='utf-8'))
    cfg = dict(base['gateway'], response_profile='binary_png_base64', response_field='artifact.mask_base64')
    result = parse_response(response, dict(sample_id='fixture', width=5, height=3, coordinate_frame='exif_transposed_original'), cfg)
    assert result.status == 'decoded'
    expected = np.array([[0, 0, 255, 0, 0], [0, 255, 255, 0, 0], [0, 255, 0, 0, 0]], np.uint8)
    np.testing.assert_array_equal(result.mask, expected)
