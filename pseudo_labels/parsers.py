"""Strict explicit profiles. Never execute or heuristically search model content."""
import base64
import io
import json
from dataclasses import dataclass, field
import numpy as np
from PIL import Image


@dataclass
class ParsedSegmentationResult:
    sample_id: str
    request_id: str | None
    raw_response_type: str
    mask: np.ndarray | None = None
    coordinates: dict = field(default_factory=dict)
    status: str = 'failed'
    failure_reason: str | None = None
    artifact_paths: dict = field(default_factory=dict)


class UnusableResponse(ValueError):
    pass


def select_field(obj, path):
    if not path:
        return obj
    for key in path.split('.'):
        if isinstance(obj, list):
            obj = obj[int(key)]
        elif isinstance(obj, dict):
            obj = obj[key]
        else:
            raise UnusableResponse('response_field 不匹配实际响应结构')
    return obj


def decode_rle(payload, width, height, order):
    if not isinstance(payload, dict) or payload.get('width') != width or payload.get('height') != height or payload.get('order') != order:
        raise UnusableResponse('RLE 尺寸或明确的坐标顺序不匹配')
    counts = payload.get('counts')
    if order not in ('C', 'F') or not isinstance(counts, list) or not counts:
        raise UnusableResponse('仅支持明确C/F顺序的非压缩RLE')
    if any(type(c) is not int or c < 0 for c in counts) or sum(counts) != width*height:
        raise UnusableResponse('RLE游程无效')
    flat = np.repeat(np.arange(len(counts), dtype=np.uint64) % 2, counts).astype(np.uint8)*255
    return flat.reshape((height, width), order=order)


def parse_response(response, row, cfg, download=None, request_id=None):
    result = ParsedSegmentationResult(row['sample_id'], request_id, cfg.get('response_profile', ''),
                                      coordinates=dict(width=row['width'], height=row['height'], frame=row['coordinate_frame']))
    try:
        profile = cfg['response_profile']
        if not profile:
            raise UnusableResponse('response_profile 未核实；只保存探测响应，不准入训练')
        payload = select_field(response, cfg['response_field'])
        if profile in ('rle', 'pixel_array') and isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                result.status = 'text_only_unusable'
                raise UnusableResponse('只有文字或非完整JSON，无法获得掩膜')
        w, h = row['width'], row['height']
        if w*h > 100_000_000:
            raise UnusableResponse('响应画布超出解码上限')
        if profile == 'rle':
            arr = decode_rle(payload, w, h, cfg['rle_order'])
        elif profile == 'pixel_array':
            if not isinstance(payload, dict) or payload.get('width') != w or payload.get('height') != h:
                raise UnusableResponse('像素数组尺寸不匹配')
            arr = np.asarray(payload.get('pixels'))
            if arr.shape != (h, w):
                raise UnusableResponse('像素数组必须[height,width]')
        elif profile in ('binary_png_base64', 'binary_png_url'):
            if not isinstance(payload, str):
                raise UnusableResponse('图片字段必须字符串')
            if profile == 'binary_png_url':
                if download is None:
                    raise UnusableResponse('未配置下载适配器')
                blob = download(payload)
            else:
                if payload.startswith('data:'):
                    if not payload.startswith('data:image/png;base64,'):
                        raise UnusableResponse('掩膜必须无损PNG')
                    payload = payload.split(',', 1)[1]
                blob = base64.b64decode(payload, validate=True)
            if len(blob) > cfg['max_response_bytes']:
                raise UnusableResponse('图片产物超过大小上限')
            with Image.open(io.BytesIO(blob)) as im:
                if im.format != 'PNG' or im.mode not in ('1', 'L') or im.size != (w, h):
                    raise UnusableResponse('叠加彩色图、非PNG或错误尺寸不能作为标签')
                if im.getexif().get(274, 1) != 1:
                    raise UnusableResponse('标签坐标方向不明确')
                arr = np.asarray(im.convert('L'))
        else:
            raise UnusableResponse(f'不支持的response_profile: {profile}')
        if not np.isin(arr, [0, 255]).all():
            raise UnusableResponse('不是0/255二值掩膜；不自动阈值化图片')
        if cfg['foreground_value'] not in (0, 255):
            raise UnusableResponse('必须显式指定foreground_value=0或255')
        result.mask = np.where(arr == cfg['foreground_value'], 255, 0).astype(np.uint8)
        result.status = 'decoded'
    except (ValueError, TypeError, KeyError, IndexError, OSError, OverflowError) as exc:
        result.failure_reason = str(exc)
    return result
