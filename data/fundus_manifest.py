import json
import os
import re
from pathlib import Path
from PIL import Image, ImageOps
from configuration import sha256

PATH_FIELDS = ('image_path', 'gt_mask_path', 'pseudo_mask_path', 'fov_path', 'label_weight_path', 'transform_path')


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    os.replace(temp, path)


def write_manifest(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    os.replace(tmp, path)


def read_manifest(path, root, allow_mock=False, require_fov=True):
    path, root = Path(path), Path(root)
    if not path.is_file():
        raise FileNotFoundError(f'缺少 manifest: {path}；用 scripts/build_manifest.py 显式建立清单')
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8-sig').splitlines() if line.strip()]
    if not rows:
        raise ValueError(f'空 manifest: {path}')
    required = ('sample_id', 'subject_id', 'split', 'image_path', 'image_sha256', 'width', 'height', 'coordinate_frame')
    ids = set()
    for r in rows:
        if any(k not in r or r[k] is None or r[k] == '' for k in required):
            raise ValueError(f'清单必需字段不全: {r.get("sample_id")}')
        if not re.fullmatch(r'[A-Za-z0-9_.-]+', r['sample_id']) or r['sample_id'] in ids:
            raise ValueError('sample_id 必须是唯一且安全的 ASCII 文件标识')
        ids.add(r['sample_id'])
        if r['split'] not in ('train', 'val', 'test') or r['coordinate_frame'] != 'exif_transposed_original':
            raise ValueError('必须显式指定 train/val/test 和 exif_transposed_original 坐标')
        if r.get('mock', False) and not allow_mock:
            raise ValueError('模拟数据不能进入正式流程')
        for k in PATH_FIELDS:
            if r.get(k):
                r[k] = str((root / r[k]).resolve())
        if sha256(r['image_path']) != r['image_sha256']:
            raise ValueError(f'原图哈希不匹配: {r["sample_id"]}')
        with Image.open(r['image_path']) as im:
            if ImageOps.exif_transpose(im).size != (r['width'], r['height']):
                raise ValueError('清单原图尺寸与统一方向坐标不匹配')
        if require_fov and not r.get('fov_path'):
            raise ValueError(f'缺少 FOV: {r["sample_id"]}')
        if r.get('gt_mask_path') and not r.get('annotation_source'):
            raise ValueError('人工标注必须记录 annotation_source')
        if r.get('pseudo_mask_path'):
            fields = ('pseudo_status', 'pseudo_version', 'prompt_sha256', 'model', 'request_job_key', 'pseudo_sha256', 'transform_path')
            if any(not r.get(k) for k in fields) or r['pseudo_status'] != 'accepted':
                raise ValueError('训练伪标签必须 accepted 且有完整来源')
            if sha256(r['pseudo_mask_path']) != r['pseudo_sha256']:
                raise ValueError('伪标签哈希已变化')
    validate_splits(rows)
    return rows


def validate_splits(rows):
    subjects, hashes = {}, {}
    for r in rows:
        for key, table in [(r['subject_id'], subjects), (r['image_sha256'], hashes)]:
            if key in table and table[key] != r['split']:
                raise ValueError(f'受试者或重复图跨划分泄漏: {r["sample_id"]}')
            table[key] = r['split']


def merge_manifests(original, accepted):
    base = {r['sample_id']: dict(r) for r in original}
    for r in accepted:
        if r['sample_id'] not in base or r['image_sha256'] != base[r['sample_id']]['image_sha256']:
            raise ValueError('accepted 伪标签与原清单不匹配')
        if r['split'] != 'train' or r['subject_id'] != base[r['sample_id']]['subject_id']:
            raise ValueError('accepted 只能来自相同受试者的训练图')
        base[r['sample_id']].update(r)
    return list(base.values())
