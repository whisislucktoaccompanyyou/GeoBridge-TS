import os
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import label


def save_png(path, mask):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    Image.fromarray(mask.astype(np.uint8)).save(tmp, format='PNG')
    os.replace(tmp, path)


def check_quality(mask, row, cfg):
    reasons = []
    if mask.shape != (row['height'], row['width']) or not np.isin(mask, [0, 255]).all():
        return 'rejected', dict(reasons=['错误尺寸或像素值'])
    if not row.get('fov_path'):
        return 'needs_review', dict(reasons=['缺少有效视野文件'])
    with Image.open(row['fov_path']) as f:
        if f.size != (row['width'], row['height']):
            return 'rejected', dict(reasons=['FOV尺寸不匹配'])
        arr = np.asarray(f.convert('L'))
        if not np.isin(arr, [0, 255]).all():
            return 'rejected', dict(reasons=['FOV必须二值'])
        valid = arr > 0
    if not valid.any():
        return 'rejected', dict(reasons=['有效视野为空'])
    fraction = float((mask[valid] > 0).mean())
    components = int(label((mask > 0) & valid)[1])
    if not cfg['min_foreground_fraction'] <= fraction <= cfg['max_foreground_fraction']:
        reasons.append('前景比例异常')
    if components > cfg['max_components']:
        reasons.append('连通分量数量异常')
    if ((mask > 0) & ~valid).any():
        reasons.append('视野外存在血管像素')
    return ('needs_review' if reasons else 'accepted'), dict(foreground_fraction=fraction, components=components, reasons=reasons)


def save_overlay(path, row, mask):
    with Image.open(row['image_path']) as im:
        rgb = np.asarray(ImageOps.exif_transpose(im).convert('RGB')).copy()
    foreground = mask > 0
    rgb[foreground] = (rgb[foreground]*.5 + np.array([0, 255, 0])*.5).astype(np.uint8)
    path = Path(path)
    temp = path.with_name(path.name+'.tmp')
    Image.fromarray(rgb).save(temp, format='PNG')
    os.replace(temp, path)
