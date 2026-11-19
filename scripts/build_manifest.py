"""Index only, then require an explicit subject/split CSV; no inferred patient identities."""
import argparse
import csv
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from PIL import Image, ImageOps
from configuration import load_config, sha256
from data.fundus_manifest import read_manifest, write_manifest

FIELDS = ['sample_id', 'subject_id', 'split', 'image_pool', 'image_path', 'gt_mask_path', 'fov_path', 'annotation_source']


def build(cfg, index=False):
    sources = cfg['sources']
    if index:
        rows = []
        for pool in ('unlabeled', 'labeled'):
            directory = sources[f'{pool}_images']
            if not directory:
                continue
            root = Path(directory)
            if not root.is_dir():
                raise FileNotFoundError(f'不存在的图像目录: {root}')
            for image in sorted(root.rglob('*')):
                if image.suffix.lower() in ('.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'):
                    # Stable safe IDs; they are image IDs, never assumed patient IDs.
                    rows.append(dict.fromkeys(FIELDS, '') | dict(sample_id=pool+'_'+sha256(image)[:16],
                                image_pool=pool, image_path=image.relative_to(root).as_posix()))
        if not rows:
            raise ValueError('请填写sources.unlabeled_images或labeled_images')
        output = Path(sources['index_output_csv'])
        output.parent.mkdir(parents=True, exist_ok=True)
        if output.exists():
            raise FileExistsError('索引模板已存在，不覆盖已填写的对应表')
        with output.open('w', newline='', encoding='utf-8-sig') as f:
            w = csv.DictWriter(f, FIELDS)
            w.writeheader()
            w.writerows(rows)
        return dict(indexed=len(rows), template=str(output), next='填写subject_id/split/FOV/标注对应，另存为sources.metadata_csv')
    metadata = Path(sources['metadata_csv'])
    with metadata.open(encoding='utf-8-sig', newline='') as f:
        rows = list(csv.DictReader(f))
    result = []
    for row in rows:
        if row['image_pool'] not in ('labeled', 'unlabeled'):
            raise ValueError('image_pool必须为labeled或unlabeled')
        pool = row.pop('image_pool')
        for key, base in [('image_path', f'{pool}_images'), ('gt_mask_path', 'labeled_masks'), ('fov_path', f'{pool}_fov')]:
            if row.get(key):
                value = Path(row[key])
                if not value.is_absolute() and not sources[base]:
                    raise ValueError(f'请配置sources.{base}')
                row[key] = str((Path(sources[base])/value).resolve())
        for key in list(row):
            if row[key] == '':
                del row[key]
        with Image.open(row['image_path']) as im:
            width, height = ImageOps.exif_transpose(im).size
        row.update(width=width, height=height, image_sha256=sha256(row['image_path']),
                   coordinate_frame='exif_transposed_original', mock=False)
        result.append(row)
    output = Path(cfg['paths']['input_manifest'])
    if output.exists():
        raise FileExistsError('input_manifest已存在；为新数据版本指定新文件，不覆盖旧清单')
    # Validate in a temporary manifest before publishing the final file.
    tmp = output.with_name(output.name+'.checking')
    write_manifest(tmp, result)
    try:
        read_manifest(tmp, cfg['paths']['data_root'], False, cfg['data']['require_fov'])
        tmp.replace(output)
    finally:
        if tmp.is_file():
            tmp.unlink()
    return dict(samples=len(result), manifest=str(output))


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--index', action='store_true')
    args = parser.parse_args()
    import json
    print(json.dumps(build(load_config(args.config), args.index), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
