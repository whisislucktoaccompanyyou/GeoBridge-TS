import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configuration import load_config
from training.stages import Pipeline
from training.checkpoints import read_checkpoint, load_model
from model.segmentation_core import SegmentationCore
from training.evaluation import evaluate


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='仅独立真实标注图评估；阈值使用配置固定值')
    parser.add_argument('--config', required=True)
    parser.add_argument('--split', choices=('val', 'test'), default='test')
    args = parser.parse_args()
    c = load_config(args.config)
    pipeline = Pipeline(c)
    rows = [r for r in pipeline.rows if r['split'] == args.split and r.get('gt_mask_path')]
    checkpoint = pipeline.run/'checkpoints'/c['evaluation']['checkpoint']
    state = read_checkpoint(checkpoint, c, pipeline.data_version, 'train-student-teacher')
    student = load_model(state, 'student', SegmentationCore(c).to(pipeline.device()))
    result = evaluate(student, rows, pipeline.size, pipeline.device(), c['evaluation']['threshold'], pipeline.run/'evaluation'/args.split)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
