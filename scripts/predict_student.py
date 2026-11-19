import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image, ImageOps
from configuration import load_config
from model.segmentation_core import SegmentationCore
from training.checkpoints import load_model
from data.paired_pseudo import transform_info, letterbox, restore_probability
from pseudo_labels.quality import save_png


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='部署仅需要学生检查点、配置和新的真实图')
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    c = load_config(args.config)
    source = c['inference']['input_image']
    if not source:
        raise ValueError('请配置inference.input_image')
    ckpt = c['inference']['student_checkpoint'] or str(Path(c['paths']['run_dir'])/'checkpoints'/c['evaluation']['checkpoint'])
    state = torch.load(ckpt, map_location='cpu', weights_only=False)
    if state['stage'] != 'train-student-teacher' or state['mock'] != c.get('mock', False):
        raise ValueError('需要正式学生教师训练阶段的学生权重，模拟和正式不能混用')
    # Inference can move to a new machine/root; architecture and canvas must match.
    if state['config']['model'] != c['model'] or state['config']['data']['image_size'] != c['data']['image_size']:
        raise ValueError('推理结构或训练画布与检查点不匹配')
    device = torch.device(c['training']['device'])
    student = load_model(state, 'student', SegmentationCore(c).to(device)).eval()
    with Image.open(source) as raw:
        im = ImageOps.exif_transpose(raw).convert('L')
    info = transform_info(*im.size, c['data']['image_size'])
    x = torch.from_numpy(np.asarray(letterbox(im, info), dtype=np.float32)/127.5-1)[None, None].to(device)
    with torch.no_grad():
        p = restore_probability(student.forward_probability(x), info)
    folder = Path(c['inference']['output_dir'])
    folder.mkdir(parents=True, exist_ok=True)
    stem = Path(source).stem
    save_png(folder/(stem+'.mask.png'), (p >= c['evaluation']['threshold']).astype(np.uint8)*255)
    np.save(folder/(stem+'.prob.npy'), p.astype(np.float32))
    print(json.dumps(dict(mask=str(folder/(stem+'.mask.png')), width=im.width, height=im.height), ensure_ascii=False))


if __name__ == '__main__':
    main()
