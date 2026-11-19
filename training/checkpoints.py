import os
import random
from pathlib import Path
import numpy as np
import torch
from configuration import fingerprint, public_config, sha256, config_version


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [])


def restore_rng(state):
    random.setstate(state['python'])
    np.random.set_state(state['numpy'])
    torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state['cuda']])


def model_fingerprint(model):
    import hashlib
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode())
        h.update(value.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def save_checkpoint(path, stage, step, models, optimizers, schedulers, scaler, samplers, cfg, data_version, dependencies, ema_steps=0, best_score=None):
    state = dict(format_version=1, stage=stage, step=step, mock=cfg.get('mock', False),
                 config=public_config(cfg), config_version=config_version(cfg),
                 data_version=data_version, dependencies=dependencies, rng=rng_state(),
                 models={k: m.state_dict() for k, m in models.items()},
                 model_fingerprints={k: model_fingerprint(m) for k, m in models.items()},
                 optimizers={k: o.state_dict() for k, o in optimizers.items()},
                 schedulers={k: s.state_dict() for k, s in schedulers.items()},
                 scaler=scaler.state_dict(), samplers={k: s.state_dict() for k, s in samplers.items()},
                 ema_steps=ema_steps, best_score=best_score, generation_seed=cfg['synthesis']['seed'])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name+'.tmp')
    torch.save(state, tmp)
    os.replace(tmp, path)


def read_checkpoint(path, cfg, data_version=None, stage=None, dependencies=None):
    if not Path(path).is_file():
        raise FileNotFoundError(f'缺少依赖检查点: {path}')
    # Only trust checkpoints produced locally/by a trusted training run.
    state = torch.load(path, map_location='cpu', weights_only=False)
    if state['mock'] != cfg.get('mock', False):
        raise ValueError('模拟与正式检查点禁止混用')
    if state['config_version'] != config_version(cfg):
        raise ValueError('配置指纹变化，使用新的run_dir重新执行依赖阶段')
    if data_version is not None and state['data_version'] != data_version:
        raise ValueError('数据指纹变化，禁止恢复旧训练状态')
    if stage and state['stage'] != stage:
        raise ValueError('检查点训练阶段不匹配')
    if dependencies is not None and state['dependencies'] != dependencies:
        raise ValueError('上游模型/缓存指纹变化，禁止恢复')
    return state


def restore_training(state, models, optimizers, schedulers, scaler, samplers):
    for k, m in models.items():
        m.load_state_dict(state['models'][k], strict=True)
        if model_fingerprint(m) != state['model_fingerprints'][k]:
            raise ValueError('模型指纹损坏')
    for k, o in optimizers.items():
        o.load_state_dict(state['optimizers'][k])
    for k, s in schedulers.items():
        s.load_state_dict(state['schedulers'][k])
    scaler.load_state_dict(state['scaler'])
    for k, s in samplers.items():
        s.load_state_dict(state['samplers'][k])
    restore_rng(state['rng'])


def load_model(state, name, model):
    model.load_state_dict(state['models'][name], strict=True)
    if model_fingerprint(model) != state['model_fingerprints'][name]:
        raise ValueError('依赖模型指纹损坏')
    return model
