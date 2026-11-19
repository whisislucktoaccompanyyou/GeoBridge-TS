"""Config paths are anchored to the config file, never the shell working directory."""
import copy
import hashlib
import json
from pathlib import Path
import tomllib


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, default=str).encode()).hexdigest()


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def sanitized(value):
    if isinstance(value, dict):
        def private(k):
            name = k.lower()
            return name in ('api_key', 'authorization', 'token', 'password') or name.endswith(('_secret', '_token'))
        return {k: ('[REDACTED]' if private(k) else sanitized(v))
                for k, v in value.items() if not k.startswith('_')}
    if isinstance(value, list):
        return [sanitized(v) for v in value]
    return value


def load_config(path):
    path = Path(path).resolve()
    with path.open('rb') as f:
        cfg = tomllib.load(f)
    cfg['_config_path'] = str(path)
    for key, value in cfg['paths'].items():
        if value:
            p = Path(value).expanduser()
            cfg['paths'][key] = str((path.parent / p).resolve())
    for section in ('sources', 'inference'):
        for key, value in cfg.get(section, {}).items():
            if value:
                cfg[section][key] = str((path.parent / Path(value).expanduser()).resolve())
    # Credentials and endpoint come only from the explicitly selected TOML.
    validate_local(cfg)
    return cfg


def validate_local(c):
    if c['data']['image_size'] < 16 or c['data']['image_size'] % 8:
        raise ValueError('data.image_size 必须 >=16 且可被8整除')
    if c['training']['batch_size'] != 1 or c['training']['num_workers'] != 0:
        raise ValueError('首版仅支持 batch_size=1 / num_workers=0，以保证同步采样恢复')
    b, g = c['bridge'], c['gpg']
    if b['reference_variance'] <= 0 or not 0 < b['delta'] < 0.5:
        raise ValueError('reference_variance>0 且 0<delta<0.5')
    if min(b['source_candidates'], b['target_candidates']) < 2:
        raise ValueError('每个条件至少2×2候选')
    if b['sampling_steps'] < 1 or b['sinkhorn_tolerance'] <= 0 or b['sinkhorn_max_iterations'] < 1:
        raise ValueError('无效的桥步数或 Sinkhorn 参数')
    if g['prox_weight'] <= 0 or g['every'] < 1 or g['iterations'] < 1 or g['learning_rate'] <= 0 or g['max_pixel_change'] <= 0:
        raise ValueError('无效的 GPG 参数')
    if not 0 <= c['training']['ema_decay'] < 1 or c['training']['weak_predictions'] < 2:
        raise ValueError('无效的 EMA 或弱增强次数')
    if c['model']['backend'] not in ('legacy', 'tiny'):
        raise ValueError('model.backend 仅支持 legacy/tiny')
    if not c.get('mock', False) and c['model']['backend'] == 'tiny':
        raise ValueError('tiny 网络只允许显式 mock=true 的功能测试')
    if c['data']['allow_mock'] and not c.get('mock', False):
        raise ValueError('allow_mock只能与mock=true同时启用')
    model = c['model']
    if model['backend'] == 'legacy':
        if model['inner_channels'] % 32 or c['data']['image_size'] % (2**(len(model['channel_mults'])-1)):
            raise ValueError('原UNet要求inner_channels为32倍数且画布可被下采样倍率整除')
    if model['bridge_channels'] < 2 or model['bridge_channels'] % 2 or not model['bridge_mults'] or min(model['bridge_mults']) < 1:
        raise ValueError('桥基础通道必须为正偶数、倍率为正')
    if c['data']['image_size'] < 2**len(model['bridge_mults']):
        raise ValueError('画布过小，无法容纳配置的桥下采样层数')
    train = c['training']
    if min(train['checkpoint_every'], train['validate_every'], train['gradient_clip']) <= 0:
        raise ValueError('检查点/验证频率与梯度裁剪必须为正')
    if any(train[k] <= 0 for k in ('g0_lr', 'discriminator_lr', 'warmup_lr', 'bridge_lr', 'student_lr')):
        raise ValueError('学习率必须为正')
    if not .5 <= train['confidence_threshold'] <= 1 or train['uncertainty_threshold'] < 0:
        raise ValueError('不确定性/置信度阈值无效')


def public_config(cfg):
    result = sanitized(copy.deepcopy(cfg))
    # Secret URL query/userinfo are rejected before any network operation.
    return result


def scientific_config(cfg):
    """Manual release gates/inference destinations do not change learned data or models."""
    result = public_config(cfg)
    result.get('synthesis', {}).pop('quality_approved', None)
    result.pop('inference', None)
    return result


def config_version(cfg):
    return fingerprint(scientific_config(cfg))
