import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from configuration import load_config

COMMANDS = ('api-probe', 'api-segment', 'pseudo-qc', 'prepare-data', 'train-g0', 'warmup-student',
            'init-teacher', 'train-bridge', 'synthesize-cache', 'train-student-teacher', 'smoke-run')


def main():
    if hasattr(sys.stdout, 'reconfigure'):
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    parser = argparse.ArgumentParser(description='GeoBridge-TS 配置驱动的单系统分阶段流程')
    parser.add_argument('stage', choices=COMMANDS)
    parser.add_argument('--config', default=str(Path(__file__).resolve().parents[1]/'config/geobridge_ts.toml'))
    parser.add_argument('--manifest', help='可选：覆盖输入清单，仅限API阶段；路径相对此config')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--resume', action='store_true')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--dry-run', action='store_true')
    mode.add_argument('--live', action='store_true')
    parser.add_argument('--smoke-run', action='store_true', help='运行完整隔离模拟流程，不请求真实API')
    args = parser.parse_args()
    c = load_config(args.config)
    if args.smoke_run or args.stage == 'smoke-run':
        if args.live:
            parser.error('smoke-run禁止--live')
        from training.smoke import run_smoke
        result = run_smoke(c)
    elif args.stage in ('api-probe', 'api-segment'):
        from data.fundus_manifest import read_manifest
        from pseudo_labels.gateway import run_gateway
        if args.stage == 'api-probe' and not (args.live or args.dry_run):
            parser.error('api-probe必须显式选择--dry-run或--live')
        if args.manifest:
            c['paths']['input_manifest'] = str((Path(c['_config_path']).parent/args.manifest).resolve())
        # Segmentation requests need only the source image and prompt. FOV remains mandatory for QC and training.
        rows = read_manifest(c['paths']['input_manifest'], c['paths']['data_root'], c['data']['allow_mock'], require_fov=False)
        result = run_gateway(c, rows, args.limit if args.limit is not None else (1 if args.stage == 'api-probe' else None),
                             probe=args.stage == 'api-probe', dry_run=args.dry_run)
    elif args.stage == 'pseudo-qc':
        from pseudo_labels.gateway import rerun_qc
        result = rerun_qc(c)
    else:
        if args.live or args.dry_run:
            parser.error('--live/--dry-run仅限API阶段')
        from training.stages import Pipeline
        from pseudo_labels.jobs import RunLock
        with RunLock(Path(c['paths']['run_dir'])/'training.lock'):
            pipeline = Pipeline(c)
            if args.stage == 'prepare-data':
                result = pipeline.prepare()
            elif args.stage == 'init-teacher':
                result = pipeline.initialize_teacher(args.resume)
            elif args.stage == 'synthesize-cache':
                result = pipeline.synthesize(args.resume)
            else:
                result = pipeline.train(args.stage, args.resume)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except (ValueError, FileNotFoundError, RuntimeError) as exc:
        print(f'停止: {exc}', file=sys.stderr)
        sys.exit(2)
