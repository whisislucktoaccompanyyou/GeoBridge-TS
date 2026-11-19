"""A fully marked mock pipeline; never contacts the configured remote endpoint."""
import copy
from datetime import datetime
import hashlib
from pathlib import Path
import uuid
import numpy as np
import torch
from PIL import Image
from configuration import sha256
from data.fundus_manifest import write_manifest, atomic_json
from pseudo_labels.gateway import load_prompt, run_gateway
from pseudo_labels.mock_server import mock_gateway


def make_mock_data(cfg, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    rows, lookup = [], {}
    for i, split in enumerate(('train', 'train', 'val', 'test')):
        height, width = 37+i, 43+i
        mask = np.zeros((height, width), np.uint8)
        mask[4:height-5, 8+i:10+i] = 255
        mask[height//2:height//2+2, 8+i:width-5] = 255
        mask[6:9, width-13:width-8] = 255
        rng = np.random.default_rng(2026+i)
        rgb = np.stack([np.full((height, width), 150+i*3), np.full((height, width), 90+i*2),
                        np.full((height, width), 60+i)], -1).astype(np.int16)
        rgb += rng.integers(-4, 5, size=rgb.shape)
        rgb[mask > 0] -= 35
        rgb = rgb.clip(0, 255).astype(np.uint8)
        sid = f'mock_{i}'
        image_path, mask_path, fov_path = [folder/(sid+s) for s in ('.png', '.mask.png', '.fov.png')]
        Image.fromarray(rgb).save(image_path)
        Image.fromarray(mask).save(mask_path)
        Image.new('L', (width, height), 255).save(fov_path)
        row = dict(sample_id=sid, subject_id=f'mock_subject_{i}', split=split, image_path=str(image_path.resolve()),
                   image_sha256=sha256(image_path), width=width, height=height, coordinate_frame='exif_transposed_original',
                   gt_mask_path=str(mask_path.resolve()), annotation_source='synthetic_fixture_not_human',
                   fov_path=str(fov_path.resolve()), mock=True)
        rows.append(row)
        lookup[hashlib.sha256(rgb.tobytes()).hexdigest()] = mask
    write_manifest(cfg['paths']['input_manifest'], rows)
    return rows, lookup


def smoke_config(base, folder):
    c = copy.deepcopy(base)
    folder = Path(folder).resolve()
    c['mock'] = True
    c['paths'].update(data_root=str(folder), input_manifest=str(folder/'input.jsonl'),
                      accepted_manifest=str(folder/'accepted.jsonl'), run_dir=str(folder))
    c['data'].update(image_size=32, allow_mock=True)
    c['model'].update(backend='tiny', bridge_channels=8, bridge_mults=[1, 2], diffusion_steps=20, g0_timestep=19)
    c['training'].update(device='cpu', amp=False, g0_steps=2, warmup_steps=2, bridge_steps=2, student_steps=2,
                          checkpoint_every=1, validate_every=1, confidence_threshold=.5, uncertainty_threshold=1., u_ramp_steps=1)
    c['bridge'].update(reference_variance=10., sampling_steps=3, sinkhorn_max_iterations=5000)
    c['gpg'].update(every=1, iterations=2, learning_rate=.1)
    c['synthesis']['quality_approved'] = True
    c['gateway'].update(api_key='mock-secret-do-not-log', response_profile='rle', response_field='output.0.content.0.text',
                         profile_verified=True, max_requests=10, requests_per_minute=60000)
    return c


def run_smoke(base, folder=None, before_student=None):
    from training.stages import Pipeline
    from training.checkpoints import read_checkpoint, load_model
    from model.segmentation_core import SegmentationCore
    from training.evaluation import evaluate
    if folder is None:
        project = Path(__file__).resolve().parents[1]
        folder = project/'runs/geobridge_ts'/('smoke_'+datetime.now().strftime('%Y%m%d_%H%M%S')+'_'+uuid.uuid4().hex[:6])
    c = smoke_config(base, folder)
    rows, lookup = make_mock_data(c, Path(folder)/'mock_inputs')
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with mock_gateway(lookup, load_prompt(c['paths']['prompt_file']), c['gateway']['model']) as (url, calls):
            c['gateway']['request_url'] = url
            dry = run_gateway(c, rows, 1, probe=True, dry_run=True)
            api = run_gateway(c, rows, 2)
            before = len(calls)
            run_gateway(c, rows, 2)
            assert len(calls) == before == 2, 'API续跑重复请求'
            pipeline = Pipeline(c)
            result = dict(mock=True, real_network=False, loopback_mock_http=True, dry_run=dry, api=api, api_calls=len(calls),
                          prepared=pipeline.prepare())
            result['g0'] = pipeline.train('train-g0')
            result['warmup'] = pipeline.train('warmup-student')
            result['teacher'] = pipeline.initialize_teacher()
            result['bridge'] = pipeline.train('train-bridge')
            result['synthesis'] = pipeline.synthesize()
            pipeline.synthesize(resume=True)
            if before_student:
                before_student(pipeline)
            result['student'] = pipeline.train('train-student-teacher')
            state = read_checkpoint(pipeline.ckpt('train-student-teacher'), c, pipeline.data_version)
            student = load_model(state, 'student', SegmentationCore(c))
            test_rows = [r for r in rows if r['split'] == 'test']
            result['evaluation'] = evaluate(student, test_rows, pipeline.size, 'cpu', c['evaluation']['threshold'], Path(folder)/'evaluation/test')
            result['limitations'] = '仅模拟功能检查，不是实际眼底生成质量、真实中转站能力或医学分割性能结果'
            atomic_json(Path(folder)/'smoke_report.json', result)
            return result
    finally:
        torch.set_num_threads(previous_threads)
