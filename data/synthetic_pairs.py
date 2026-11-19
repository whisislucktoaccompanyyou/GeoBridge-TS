import json
from pathlib import Path
from configuration import fingerprint, sha256


def read_synthetic_pairs(path, pseudo_rows, expected_version, allow_mock=False):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('缺少完整B合成缓存；先执行synthesize-cache')
    rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
    targets = {r['sample_id']: r for r in pseudo_rows}
    if not rows:
        raise ValueError('合成缓存为空')
    seen = set()
    for r in rows:
        sid = r['source_sample_id']
        if r.get('mock', False) and not allow_mock:
            raise ValueError('模拟缓存不能进入正式训练')
        if sid not in targets or r['sample_id'] in seen or r['generation_version'] != expected_version:
            raise ValueError('合成缓存来源/生成版本不匹配或重复')
        seen.add(r['sample_id'])
        p = targets[sid]
        if r['pseudo_sha256'] != p['pseudo_sha256'] or sha256(r['pseudo_mask_path']) != p['pseudo_sha256']:
            raise ValueError('Y目标不是输入B的同一份P')
        if r['pseudo_mask_path'] != p['pseudo_mask_path'] or sha256(r['image_path']) != r['image_sha256']:
            raise ValueError('合成图或目标路径/哈希变化')
        if sha256(r['generation_metadata']) != r['generation_metadata_sha256']:
            raise ValueError('生成元数据指纹变化')
        if r['split'] != 'train':
            raise ValueError('合成缓存只能属于训练集')
    if set(targets) != {r['source_sample_id'] for r in rows}:
        raise ValueError('缓存尚未覆盖全部accepted条件')
    return rows
