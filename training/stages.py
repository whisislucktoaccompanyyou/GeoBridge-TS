import copy
import json
import time
from pathlib import Path
import numpy as np
import torch
from torch.nn import functional as F
from configuration import fingerprint, public_config, sha256, scientific_config, config_version
from data.fundus_manifest import read_manifest, merge_manifests, atomic_json, write_manifest
from data.paired_pseudo import load_sample, StatefulSampler, appearance, restore_probability
from data.synthetic_pairs import read_synthetic_pairs
from model.segmentation_core import SegmentationCore, InitialGenerator, make_discriminator, freeze, assert_independent
from model.ema_teacher import make_teacher, update_ema, uncertainty_target
from model.ot_bridge.bridge import EndpointUNet, bridge_state, sample_bridge
from model.ot_bridge.transport import solve_conditional_ot, sample_pair
from pseudo_labels.quality import save_png
from training.losses import segmentation_loss, masked_bce
from training.checkpoints import (seed_everything, save_checkpoint, read_checkpoint, load_model,
                                  restore_training, model_fingerprint)


STAGE_PATHS = {'train-g0': 'g0_last.pt', 'warmup-student': 'warmup_last.pt', 'init-teacher': 'teacher_init.pt',
               'train-bridge': 'bridge_last.pt', 'train-student-teacher': 'student_last.pt'}


class Pipeline:
    def __init__(self, cfg):
        self.cfg = cfg
        self.run = Path(cfg['paths']['run_dir'])
        self.run.mkdir(parents=True, exist_ok=True)
        original = read_manifest(cfg['paths']['input_manifest'], cfg['paths']['data_root'],
                                 cfg['data']['allow_mock'], cfg['data']['require_fov'])
        accepted_path = Path(cfg['paths']['accepted_manifest'])
        accepted = read_manifest(accepted_path, cfg['paths']['data_root'], cfg['data']['allow_mock'], True) if accepted_path.is_file() else []
        self.rows = merge_manifests(original, accepted)
        # Include every relevant file, not only paths or the image hash, in data version.
        self.data_version = fingerprint([dict(row=r, files={k: sha256(r[k]) for k in
            ('image_path', 'gt_mask_path', 'pseudo_mask_path', 'fov_path', 'label_weight_path', 'transform_path') if r.get(k)}) for r in self.rows])
        self.paired = [r for r in self.rows if r['split'] == 'train' and r.get('pseudo_status') == 'accepted']
        self.labeled = [r for r in self.rows if r['split'] == 'train' and r.get('gt_mask_path')]
        self.unlabeled = [r for r in original if r['split'] == 'train']
        self.validation = [r for r in self.rows if r['split'] == 'val' and r.get('gt_mask_path')]
        self.size = cfg['data']['image_size']

    def ckpt(self, stage):
        return self.run/'checkpoints'/STAGE_PATHS[stage]

    def prepare(self):
        snapshot = self.run/'prepared.json'
        if snapshot.is_file():
            previous = json.loads(snapshot.read_text(encoding='utf-8'))
            if previous['data_version'] != self.data_version or previous['config_version'] != config_version(self.cfg):
                raise ValueError('run_dir已经绑定其他数据/配置版本；请使用新运行目录')
        atomic_json(snapshot, dict(data_version=self.data_version, config_version=config_version(self.cfg),
                                  train_labeled=len(self.labeled), train_paired=len(self.paired),
                                  train_unlabeled=len(self.unlabeled), validation=len(self.validation), mock=self.cfg.get('mock', False)))
        atomic_json(self.run/'config.sanitized.json', public_config(self.cfg))
        write_manifest(self.run/'manifest.snapshot.jsonl', self.rows)
        for row in self.paired:
            batch = load_sample(row, self.size)
            folder = self.run/'geometry'
            folder.mkdir(exist_ok=True)
            temp = folder/(row['sample_id']+'.tmp')
            with temp.open('wb') as f:
                np.save(f, batch['geometry'].numpy()[0])
            temp.replace(folder/(row['sample_id']+'.npy'))
        return dict(data_version=self.data_version, labeled=len(self.labeled), pseudo=len(self.paired), validation=len(self.validation))

    def device(self):
        device = torch.device(self.cfg['training']['device'])
        if device.type == 'cuda' and not torch.cuda.is_available():
            raise ValueError('配置要求CUDA，但当前环境没有CUDA；修改training.device或在GPU服务器执行')
        return device

    def bridge_model(self):
        c = self.cfg['model']
        return EndpointUNet(c['bridge_channels'], c['bridge_mults'])

    def dependency(self, stage):
        path = self.ckpt(stage)
        state = read_checkpoint(path, self.cfg, self.data_version, stage)
        budget_key = {'train-g0': 'g0_steps', 'warmup-student': 'warmup_steps', 'train-bridge': 'bridge_steps'}.get(stage)
        if budget_key and state['step'] < self.cfg['training'][budget_key]:
            raise ValueError(f'依赖阶段尚未完成配置预算: {stage}')
        return state, sha256(path)

    def generation_dependencies(self):
        states, hashes = {}, {}
        for stage in ('train-g0', 'train-bridge', 'init-teacher'):
            states[stage], hashes[stage] = self.dependency(stage)
        return states, hashes

    def initialize_teacher(self, resume=False):
        source, digest = self.dependency('warmup-student')
        target = self.ckpt('init-teacher')
        if target.is_file():
            read_checkpoint(target, self.cfg, self.data_version, 'init-teacher', {'warmup': digest})
            if not resume:
                raise ValueError('teacher_init已存在；使用--resume复用')
            return dict(reused=True)
        student = load_model(source, 'student', SegmentationCore(self.cfg))
        teacher = make_teacher(student)
        assert_independent(student, teacher)
        scaler = torch.amp.GradScaler('cpu', enabled=False)
        save_checkpoint(target, 'init-teacher', 0, {'student': student, 'teacher': teacher}, {}, {}, scaler, {},
                        self.cfg, self.data_version, {'warmup': digest})
        return dict(teacher=model_fingerprint(teacher))

    def train(self, stage, resume=False):
        self.prepare()
        c, tcfg = self.cfg, self.cfg['training']
        budget_key = {'train-g0': 'g0_steps', 'warmup-student': 'warmup_steps',
                      'train-bridge': 'bridge_steps', 'train-student-teacher': 'student_steps'}[stage]
        max_steps = tcfg[budget_key]
        if max_steps < 1:
            raise ValueError(f'training.{budget_key}尚未填写正数正式训练预算')
        if stage in ('warmup-student', 'train-student-teacher') and not self.validation:
            raise ValueError('学生训练需要独立人工标注验证集以选择检查点')
        device = self.device()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        seed_everything(c['seed'])
        models, frozen, optimizers, samplers, dependencies = {}, {}, {}, {}, {}
        pools = {}
        if stage == 'train-g0':
            models = {'g0': InitialGenerator(c).to(device), 'discriminator': make_discriminator(c).to(device)}
            pools['paired'] = self.paired
            optimizers = {k: torch.optim.Adam(m.parameters(), lr=tcfg['g0_lr' if k == 'g0' else 'discriminator_lr']) for k, m in models.items()}
        elif stage == 'warmup-student':
            models = {'student': SegmentationCore(c).to(device)}
            pools['labeled'] = self.labeled
            optimizers['student'] = torch.optim.Adam(models['student'].parameters(), lr=tcfg['warmup_lr'])
        elif stage == 'train-bridge':
            state, digest = self.dependency('train-g0')
            teacher_state, tdigest = self.dependency('init-teacher')
            dependencies = {'g0': digest, 'teacher': tdigest}
            frozen['g0'] = freeze(load_model(state, 'g0', InitialGenerator(c).to(device)))
            frozen['teacher'] = freeze(load_model(teacher_state, 'teacher', SegmentationCore(c).to(device)))
            models = {'bridge': self.bridge_model().to(device)}
            pools['paired'] = self.paired
            optimizers['bridge'] = torch.optim.Adam(models['bridge'].parameters(), lr=tcfg['bridge_lr'])
        else:
            states, hashes = self.generation_dependencies()
            version = fingerprint(dict(dependencies=hashes, data=self.data_version, config=scientific_config(c)))
            if not c['synthesis']['quality_approved']:
                raise ValueError('请检查G0/SB/GPG实际输出后设置synthesis.quality_approved=true；此人工准入开关不改变训练指纹')
            synthetic = read_synthetic_pairs(self.run/'synthetic/manifest.jsonl', self.paired, version, c['data']['allow_mock'])
            dependencies = dict(hashes, cache=sha256(self.run/'synthetic/manifest.jsonl'))
            models['student'] = load_model(states['init-teacher'], 'student', SegmentationCore(c).to(device))
            models['teacher'] = freeze(load_model(states['init-teacher'], 'teacher', SegmentationCore(c).to(device)))
            frozen['g0'] = freeze(load_model(states['train-g0'], 'g0', InitialGenerator(c).to(device)))
            frozen['bridge'] = freeze(load_model(states['train-bridge'], 'bridge', self.bridge_model().to(device)))
            pools = dict(labeled=self.labeled, synthetic=synthetic, unlabeled=self.unlabeled)
            optimizers['student'] = torch.optim.Adam(models['student'].parameters(), lr=tcfg['student_lr'])
        assert_independent(*models.values(), *frozen.values())
        atomic_json(self.run/(stage+'.models.json'), dict(
            mock=c.get('mock', False), device=str(device), torch_version=str(torch.__version__),
            canvas=[self.size, self.size], image_channels=1,
            models={k: dict(parameters=sum(p.numel() for p in m.parameters()),
                            trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad))
                    for k, m in dict(models, **frozen).items()},
            bridge_inputs='x_t[B,1,H,W], t[B], x0[B,1,H,W], geometry[B,4,H,W]',
            bridge_output='clean_endpoint[B,1,H,W] in [-1,1]',
            segmentation_output='probability[B,1,H,W] in [0,1]',
            memory_measurement='每步training.jsonl记录实际peak_memory_bytes；CPU为0'))
        for i, (name, pool) in enumerate(pools.items()):
            samplers[name] = StatefulSampler(len(pool), c['seed']+i)
        schedulers = {k: torch.optim.lr_scheduler.LambdaLR(o, lambda step: 1.) for k, o in optimizers.items()}
        use_amp = bool(tcfg['amp'] and device.type == 'cuda')
        scaler = torch.amp.GradScaler(device.type, enabled=use_amp)
        path = self.ckpt(stage)
        start, ema_steps, best_score = 0, 0, None
        if path.is_file():
            if not resume:
                raise ValueError('阶段检查点已存在；使用--resume，不静默覆盖')
            state = read_checkpoint(path, c, self.data_version, stage, dependencies)
            restore_training(state, models, optimizers, schedulers, scaler, samplers)
            start, ema_steps, best_score = state['step'], state['ema_steps'], state['best_score']
        elif resume:
            raise FileNotFoundError(f'--resume但缺少阶段检查点: {path}')
        def next_batch(name, target='pseudo_mask_path'):
            pool = pools[name]
            row = pool[samplers[name].next()]
            return load_sample(row, self.size, device, target=target, flip=bool(torch.rand(())<.5))
        all_models = dict(models, **frozen)
        frozen_versions = {k: model_fingerprint(m) for k, m in frozen.items()}
        def save(step, dest=path):
            save_checkpoint(dest, stage, step, all_models, optimizers, schedulers, scaler, samplers,
                            c, self.data_version, dependencies, ema_steps, best_score)
        # Save even before the first update: interrupted runs have a deterministic restart point.
        if start == 0 and not path.is_file():
            save(0)
        for step in range(start, max_steps):
            tick = time.perf_counter()
            for name, m in models.items():
                if name != 'teacher':
                    m.train()
            for o in optimizers.values():
                o.zero_grad(set_to_none=True)
            log = dict(stage=stage, step=step+1)
            if stage == 'train-g0':
                b = next_batch('paired')
                d, g = models['discriminator'], models['g0']
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    y, denoise = g(b['mask'], appearance(b['image']), return_denoising=True)
                    d_loss = .5*((d(b['image'])-1).square().mean()+d(y.detach()).square().mean())
                scaler.scale(d_loss).backward()
                d.requires_grad_(False).eval()
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    rec = ((y-b['image']).abs()*b['valid']).sum()/b['valid'].sum().clamp_min(1)
                    adv = (d(y)-1).square().mean()
                    loss = tcfg['g0_reconstruction_weight']*rec+tcfg['g0_adversarial_weight']*adv+tcfg['g0_denoising_weight']*denoise
                scaler.scale(loss).backward()
                d.requires_grad_(True)
                log.update(reconstruction=float(rec.detach()), adversarial=float(adv.detach()), discriminator=float(d_loss.detach()), denoising=float(denoise.detach()))
            elif stage == 'train-bridge':
                b = next_batch('paired')
                bc = c['bridge']
                source_seeds, target_seeds = [], []
                sources, targets = [], []
                with torch.no_grad():
                    for _ in range(bc['source_candidates']):
                        candidate_seed = int(torch.randint(0, 2**31-1, (1,)))
                        rng = torch.Generator(device=device).manual_seed(candidate_seed)
                        noise = torch.randn(b['image'].shape, generator=rng, device=device)
                        sources.append(frozen['g0'](b['mask'], b['image'], noise))
                        source_seeds.append(candidate_seed)
                    for _ in range(bc['target_candidates']):
                        candidate_seed = int(torch.randint(0, 2**31-1, (1,)))
                        rng = torch.Generator(device=device).manual_seed(candidate_seed)
                        targets.append(appearance(b['image'], generator=rng))
                        target_seeds.append(candidate_seed)
                    sources, targets = torch.cat(sources), torch.cat(targets)
                sid = b['row']['sample_id']
                ot = solve_conditional_ot(sources, targets, [sid]*len(sources), [sid]*len(targets), bc['reference_variance'],
                                         bc['cost_reduction'], bc['sinkhorn_tolerance'], bc['sinkhorn_max_iterations'])
                i, j = sample_pair(ot)
                x0, x1 = sources[i:i+1], targets[j:j+1]
                time_t = torch.rand((1,), device=device)*(1-2*bc['delta'])+bc['delta']
                x_t = bridge_state(x0, x1, time_t, bc['reference_variance'])
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    endpoint = models['bridge'].predict_endpoint(x_t, time_t, x0, b['geometry'])
                    loss = F.mse_loss(endpoint.float(), x1.float())
                scaler.scale(loss).backward()
                log.update(bridge=float(loss.detach()), ot=ot.diagnostics, condition_id=sid,
                           pseudo_sha256=b['row']['pseudo_sha256'], source_seeds=source_seeds,
                           target_appearance_seeds=target_seeds, selected_source=i, selected_target=j,
                           g0_fingerprint=frozen_versions['g0'], appearance_strength=.06)
            else:
                student = models['student']
                a = next_batch('labeled', 'gt_mask_path')
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    loss_a = segmentation_loss(student.forward_probability(a['image']), a['mask'], a['valid']*a['weight'])
                scaler.scale(tcfg['lambda_a']*loss_a).backward()
                log['loss_a'] = float(loss_a.detach())
                if stage == 'train-student-teacher':
                    synthetic = next_batch('synthetic')
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        loss_c = segmentation_loss(student.forward_probability(synthetic['image']), synthetic['mask'], synthetic['valid']*synthetic['weight'])
                    scaler.scale(tcfg['lambda_c']*loss_c).backward()
                    u = next_batch('unlabeled', target='unused')
                    q, reliable, stats = uncertainty_target(models['teacher'], u['image'], u['valid'], tcfg['weak_predictions'],
                                                           tcfg['confidence_threshold'], tcfg['uncertainty_threshold'])
                    weight_u = tcfg['lambda_u']*min(1, ema_steps/max(tcfg['u_ramp_steps'], 1))
                    with torch.autocast(device_type=device.type, enabled=use_amp):
                        loss_u = masked_bce(student.forward_probability(appearance(u['image'], .12)), q, reliable)
                    scaler.scale(weight_u*loss_u).backward()
                    log.update(loss_c=float(loss_c.detach()), loss_u=float(loss_u.detach()), lambda_u=weight_u, **stats)
            for name, optimizer in optimizers.items():
                scaler.unscale_(optimizer)
                params = [p for group in optimizer.param_groups for p in group['params']]
                torch.nn.utils.clip_grad_norm_(params, tcfg['gradient_clip'])
            # Detect each optimizer's actual skip independently (G0/D can differ).
            finite = {k: all(p.grad is None or bool(torch.isfinite(p.grad).all()) for group in o.param_groups for p in group['params']) for k, o in optimizers.items()}
            if not use_amp and not all(finite.values()):
                raise RuntimeError('非有限梯度，未执行更新；从最近检查点恢复')
            for k, optimizer in optimizers.items():
                scaler.step(optimizer)
            scaler.update()
            for k, scheduler in schedulers.items():
                if finite[k]:
                    scheduler.step()
            if stage == 'train-student-teacher' and finite['student']:
                update_ema(models['student'], models['teacher'], tcfg['ema_decay'])
                ema_steps += 1
            if device.type == 'cuda':
                torch.cuda.synchronize(device)
            log.update(optimizer_updated=finite, ema_steps=ema_steps, seconds=time.perf_counter()-tick,
                       peak_memory_bytes=torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else 0)
            if stage in ('warmup-student', 'train-student-teacher') and ((step+1) % tcfg['validate_every'] == 0 or step+1 == max_steps):
                from training.evaluation import validate_student
                score = validate_student(models['student'], self.validation, self.size, device, c['evaluation']['threshold'])
                log['validation_dice'] = score
                if best_score is None or score > best_score:
                    best_score = score
                    if stage == 'train-student-teacher':
                        save(step+1, self.run/'checkpoints/student_best.pt')
            with (self.run/'training.jsonl').open('a', encoding='utf-8') as f:
                f.write(json.dumps(log, ensure_ascii=False)+'\n')
            if (step+1) % tcfg['checkpoint_every'] == 0 or step+1 == max_steps:
                save(step+1)
        return dict(stage=stage, steps=max_steps, checkpoint=str(path), ema_steps=ema_steps)

    def synthesize(self, resume=False):
        self.prepare()
        if not self.paired:
            raise ValueError('没有accepted训练伪标签')
        device = self.device()
        states, hashes = self.generation_dependencies()
        version = fingerprint(dict(dependencies=hashes, data=self.data_version, config=scientific_config(self.cfg)))
        g0 = freeze(load_model(states['train-g0'], 'g0', InitialGenerator(self.cfg).to(device)))
        bridge = freeze(load_model(states['train-bridge'], 'bridge', self.bridge_model().to(device)))
        teacher = freeze(load_model(states['init-teacher'], 'teacher', SegmentationCore(self.cfg).to(device)))
        assert_independent(g0, bridge, teacher)
        frozen_versions = {'g0': model_fingerprint(g0), 'bridge': model_fingerprint(bridge), 'teacher': model_fingerprint(teacher)}
        folder = self.run/'synthetic'
        folder.mkdir(exist_ok=True)
        path = folder/'manifest.jsonl'
        old = {}
        if path.is_file():
            if not resume:
                raise ValueError('缓存已存在；使用--resume校验后复用')
            old = {r['source_sample_id']: r for r in (json.loads(s) for s in path.read_text(encoding='utf-8').splitlines() if s)}
        rows = []
        for index, p in enumerate(self.paired):
            sid = p['sample_id']
            seed = self.cfg['synthesis']['seed']+int(fingerprint(sid)[:8], 16)
            image_path, meta_path = folder/(sid+'.png'), folder/(sid+'.json')
            previous = old.get(sid)
            if previous is None and meta_path.is_file() and resume:
                previous = json.loads(meta_path.read_text(encoding='utf-8')).get('cache_row')
                if previous:
                    previous['generation_metadata'] = str(meta_path)
                    previous['generation_metadata_sha256'] = sha256(meta_path)
            if previous:
                if previous['generation_version'] != version or sha256(previous['image_path']) != previous['image_sha256'] or sha256(previous['pseudo_mask_path']) != p['pseudo_sha256'] or sha256(previous['generation_metadata']) != previous['generation_metadata_sha256']:
                    raise ValueError('已有合成产物指纹变化；使用新run_dir，不覆盖旧版本')
                rows.append(previous)
                continue
            seed_everything(seed)
            generator = torch.Generator(device=device).manual_seed(seed)
            batch = load_sample(p, self.size, device)
            if self.cfg['synthesis']['reference_policy'] != 'different_training_subject':
                raise ValueError('首版仅支持不同训练受试者的参考外观，避免访问待生成目标X')
            references = [r for r in self.unlabeled if r['subject_id'] != p['subject_id'] and r['image_sha256'] != p['image_sha256']]
            if not references:
                raise ValueError('缺少不同训练受试者的参考图，不能用目标照片X冒充推理参考')
            reference_row = references[seed % len(references)]
            reference = load_sample(reference_row, self.size, device, target='unused')['image']
            # Reference is a different training subject. SB receives no target x1.
            with torch.no_grad():
                noise = torch.randn(reference.shape, device=device, generator=generator)
                x0 = g0(batch['mask'], reference, noise)
            start = time.perf_counter()
            y, gpg_log = sample_bridge(bridge, x0, batch['geometry'], batch['mask'], batch['valid'], batch['weight'], teacher,
                                       self.cfg['bridge'], self.cfg['gpg'], generator)
            y = torch.where(batch['valid'] > 0, y, -torch.ones_like(y))
            output = np.clip((restore_probability(y, batch['info'])+1)*127.5, 0, 255).round().astype(np.uint8)
            save_png(image_path, output)
            r = dict(p, image_path=str(image_path), image_sha256=sha256(image_path), source_sample_id=sid,
                     synthetic_id=sid+'_'+version[:12], reference_sample_id=reference_row['sample_id'],
                     generation_version=version, seed=seed, mock=self.cfg.get('mock', False))
            r.pop('gt_mask_path', None)
            atomic_json(meta_path, dict(cache_row=r, dependencies=hashes, model_fingerprints=frozen_versions,
                geometry_transform=batch['info'], config=public_config(self.cfg), gpg=gpg_log,
                seconds=time.perf_counter()-start, mock=self.cfg.get('mock', False)))
            r['generation_metadata'], r['generation_metadata_sha256'] = str(meta_path), sha256(meta_path)
            rows.append(r)
            write_manifest(path, rows)
        write_manifest(path, rows)
        read_synthetic_pairs(path, self.paired, version, self.cfg['data']['allow_mock'])
        return dict(generated=len(rows), generation_version=version, manifest=str(path))
