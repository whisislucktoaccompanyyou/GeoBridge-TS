"""One letterbox transform shared by X/P/M/V/W. Masks are never independently cropped."""
import numpy as np
import torch
from PIL import Image, ImageOps
from data.geometry import geometry_from_mask


def build_geometry(mask):
    """0/1 [B,1,H,W] target -> [B,4,H,W]; target-only, intentionally no autograd."""
    return torch.from_numpy(np.stack([geometry_from_mask(m[0] > 0.5) for m in mask.detach().cpu().numpy()])).to(mask)


def transform_info(width, height, size):
    scale = min(size / width, size / height)
    w, h = max(1, round(width * scale)), max(1, round(height * scale))
    return dict(original_width=width, original_height=height, width=w, height=h,
                left=(size-w)//2, top=(size-h)//2, canvas=size,
                scale_x=w/width, scale_y=h/height)


def letterbox(im, info, nearest=False, fill=0):
    im = im.resize((info['width'], info['height']), Image.Resampling.NEAREST if nearest else Image.Resampling.BILINEAR)
    out = Image.new('L', (info['canvas'], info['canvas']), fill)
    out.paste(im, (info['left'], info['top']))
    return out


def restore_probability(prob, info):
    arr = prob.detach().float().cpu().numpy().squeeze()
    t, l = info['top'], info['left']
    arr = arr[t:t+info['height'], l:l+info['width']]
    return np.asarray(Image.fromarray(arr).resize((info['original_width'], info['original_height']), Image.Resampling.BILINEAR))


def load_sample(row, size, device='cpu', target='pseudo_mask_path', flip=False):
    with Image.open(row['image_path']) as raw:
        im = ImageOps.exif_transpose(raw).convert('L')
    info = transform_info(*im.size, size)
    x = np.asarray(letterbox(im, info), dtype=np.float32) / 127.5 - 1
    result = {'image': x}
    for name, field in [('mask', target), ('valid', 'fov_path'), ('weight', 'label_weight_path')]:
        if row.get(field):
            with Image.open(row[field]) as raw:
                if raw.size != im.size or int(raw.getexif().get(274, 1)) != 1:
                    raise ValueError(f'标签尺寸或方向不匹配: {field}')
                if raw.mode not in ('1', 'L', 'P'):
                    raise ValueError('标签和有效视野必须为单通道')
                raw = raw.convert('L')
                arr = np.asarray(raw)
                if name != 'weight' and not np.isin(arr, [0, 255]).all():
                    raise ValueError(f'{field} 必须为0/255二值图')
                result[name] = np.asarray(letterbox(raw, info, nearest=True), dtype=np.float32) / 255
        elif name in ('valid', 'weight'):
            result[name] = np.asarray(letterbox(Image.new('L', im.size, 255), info, nearest=True), dtype=np.float32) / 255
    for k, v in result.items():
        if flip:
            v = np.flip(v, 1).copy()
        result[k] = torch.from_numpy(v.copy()).unsqueeze(0).unsqueeze(0).to(device)
    if 'mask' in result:
        result['geometry'] = build_geometry(result['mask'])
    result['info'], result['row'] = info, row
    return result


def appearance(x, strength=0.06, generator=None):
    # Same geometry; mild affine intensity changes, no image-based vessel extraction.
    gain = 1 + (torch.rand((), device=x.device, generator=generator) * 2 - 1) * strength
    bias = (torch.rand((), device=x.device, generator=generator) * 2 - 1) * strength
    return (x * gain + bias).clamp(-1, 1)


class StatefulSampler:
    def __init__(self, length, seed):
        if length < 1:
            raise ValueError('阶段数据池为空')
        self.length = length
        self.generator = torch.Generator().manual_seed(seed)
        self.order, self.position = [], 0

    def next(self):
        if self.position == len(self.order):
            self.order = torch.randperm(self.length, generator=self.generator).tolist()
            self.position = 0
        i = self.order[self.position]
        self.position += 1
        return i

    def state_dict(self):
        return dict(length=self.length, generator=self.generator.get_state(), order=self.order, position=self.position)

    def load_state_dict(self, s):
        if s['length'] != self.length:
            raise ValueError('采样数据数量发生变化，不能恢复')
        self.generator.set_state(s['generator'])
        self.order, self.position = s['order'], s['position']
