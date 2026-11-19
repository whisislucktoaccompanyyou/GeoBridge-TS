"""Explicit HTTP only. Imports, training and tests never contact the real gateway."""
import base64
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
import http.client
import io
import hashlib
import json
import random
import socket
import time
from pathlib import Path
from urllib.parse import urlsplit
from PIL import Image, ImageOps
from configuration import fingerprint, sha256, sanitized
from data.fundus_manifest import atomic_json, write_manifest
from pseudo_labels.jobs import JobLedger, RunLock
from pseudo_labels.parsers import parse_response
from pseudo_labels.quality import check_quality, save_png, save_overlay


def checked_url(url, allow_http_local=True):
    p = urlsplit(url)
    if p.scheme != 'https' and not (allow_http_local and p.scheme == 'http' and p.hostname in ('127.0.0.1', 'localhost')):
        raise ValueError('请求必须HTTPS（离线测试允许本机HTTP）')
    if not p.hostname or p.username or p.password or p.query or p.fragment:
        raise ValueError('URL不能含凭据、query或fragment；使用独立配置字段')
    return p


def http_request(url, method, body, headers, cfg):
    p = checked_url(url)
    cls = http.client.HTTPSConnection if p.scheme == 'https' else http.client.HTTPConnection
    connection = cls(p.hostname, port=p.port, timeout=cfg['connect_timeout_seconds'])
    try:
        connection.connect()
        connection.sock.settimeout(cfg['read_timeout_seconds'])
        connection.request(method, p.path or '/', body=body, headers=headers)
        response = connection.getresponse()
        blob = response.read(cfg['max_response_bytes']+1)
        if len(blob) > cfg['max_response_bytes']:
            raise ValueError('响应超过max_response_bytes，停止解析')
        return response.status, dict((k.lower(), v) for k, v in response.getheaders()), blob
    finally:
        connection.close()


def load_prompt(path):
    text = Path(path).read_text(encoding='utf-8-sig')
    if not text.strip() or text.strip().lower() in ('todo', 'placeholder', '请填写提示词'):
        raise ValueError('prompt_file为空或仍是占位符')
    return text


def request_image(row, cfg):
    with Image.open(row['image_path']) as raw:
        orientation = int(raw.getexif().get(274, 1))
        image = ImageOps.exif_transpose(raw).convert('RGB')
        if image.size != (row['width'], row['height']):
            raise ValueError('请求图尺寸不匹配manifest')
        buffer = io.BytesIO()
        image.save(buffer, format='PNG')
    blob = buffer.getvalue()
    if len(blob) > cfg['max_image_bytes']:
        raise ValueError('原分辨率PNG超过限制；请显式制定坐标变换后再实现适配，不自动压缩或裁剪')
    transform = dict(preprocessing_version='exif_rgb_full_png_v1', exif_orientation=orientation,
                     frame='exif_transposed_original', width=row['width'], height=row['height'],
                     request_width=row['width'], request_height=row['height'], resize=False, crop=False)
    return 'data:image/png;base64,'+base64.b64encode(blob).decode('ascii'), hashlib.sha256(blob).hexdigest(), transform


def make_body(cfg, prompt, data_url):
    protocol = cfg['protocol']
    if protocol == 'openai_responses':
        body = dict(model=cfg['model'], input=[dict(role='user', content=[
            dict(type='input_text', text=prompt), dict(type='input_image', image_url=data_url)])])
    elif protocol == 'openai_chat_completions':
        body = dict(model=cfg['model'], messages=[dict(role='user', content=[
            dict(type='text', text=prompt), dict(type='image_url', image_url=dict(url=data_url))])])
    elif protocol == 'custom_json':
        template = cfg.get('request_template')
        if not template or any(tag not in json.dumps(template) for tag in ('{{model}}', '{{prompt}}', '{{image}}')):
            raise ValueError('custom_json缺少含{{model}}/{{prompt}}/{{image}}的request_template')
        def fill(v):
            if isinstance(v, dict):
                return {k: fill(x) for k, x in v.items()}
            if isinstance(v, list):
                return [fill(x) for x in v]
            return {'{{model}}': cfg['model'], '{{prompt}}': prompt, '{{image}}': data_url}.get(v, v) if isinstance(v, str) else v
        body = fill(template)
    else:
        raise ValueError('未知gateway.protocol')
    extra = cfg.get('extra_body', {})
    forbidden = {'model', 'input', 'messages', 'authorization', 'headers', 'api_key', 'request_url'}
    if any(k.lower() in forbidden or k in body for k in extra):
        raise ValueError('extra_body不允许覆盖模型、输入、提示词或认证')
    return dict(body, **extra)


def validate_gateway(cfg, live=False, batch=False):
    g = cfg['gateway']
    if not g['request_url'] or not g['model']:
        raise ValueError('缺少gateway.request_url或model')
    checked_url(g['request_url'])
    if cfg.get('mock', False) and urlsplit(g['request_url']).hostname not in ('127.0.0.1', 'localhost'):
        raise ValueError('mock配置仅允许本机模拟HTTP，禁止请求真实中转站')
    if g['concurrency'] != 1 or g['max_attempts'] < 1 or g['max_requests'] < 1 or g['requests_per_minute'] <= 0:
        raise ValueError('首版串行运行，必须设置正数重试/限流/总请求预算')
    if live and not g['api_key']:
        raise ValueError('缺少API密钥：在所选TOML配置的[gateway]中填写api_key')
    if batch and (not g['profile_verified'] or not g['response_profile']):
        raise ValueError('先live probe并核实实际响应，再固定response_profile/response_field/profile_verified')
    return load_prompt(cfg['paths']['prompt_file'])


def retry_delay(header, attempt, maximum):
    seconds = 0.
    if header:
        try:
            seconds = float(header)
        except ValueError:
            try:
                seconds = (parsedate_to_datetime(header)-datetime.now(timezone.utc)).total_seconds()
            except (ValueError, TypeError):
                pass
    return min(maximum, max(seconds, min(maximum, 2**attempt+random.random())))


def scrub_response(value, secret):
    value = sanitized(value)
    if isinstance(value, dict):
        return {k: scrub_response(v, secret) for k, v in value.items()}
    if isinstance(value, list):
        return [scrub_response(v, secret) for v in value]
    if isinstance(value, str) and secret:
        return value.replace(secret, '[REDACTED]')
    return value


def prepare_job(row, prompt, g):
    data_url, request_hash, transform = request_image(row, g)
    body = make_body(g, prompt, data_url)
    signature = fingerprint(dict(image=row['image_sha256'], request_image=request_hash,
                                 prompt=fingerprint(prompt), namespace=g['request_url'], protocol=g['protocol'],
                                 model=g['model'], transform=transform, body=body))
    key = fingerprint(dict(request_signature=signature, profile=g['response_profile'], field=g['response_field'],
                           polarity=g['foreground_value'], order=g['rle_order']))
    return key, signature, body, transform


class GatewayClient:
    def __init__(self, cfg, transport=http_request):
        self.cfg, self.g = cfg, cfg['gateway']
        self.transport, self.sent, self.last_request = transport, 0, 0.
        self.run = Path(cfg['paths']['run_dir'])

    def download(self, url):
        p = checked_url(url, allow_http_local=False)
        if p.hostname not in self.g['download_allowed_hosts']:
            raise ValueError('下载主机不在download_allowed_hosts中')
        # No authentication forwarded, no automatic redirects.
        status, _, blob = self.transport(url, 'GET', None, {}, self.g)
        if status != 200:
            raise ValueError('掩膜下载失败（重定向也不自动跟随）')
        return blob

    def segment(self, row, prompt, ledger, probe=False):
        g = self.g
        key, signature, body, transform = prepare_job(row, prompt, g)
        ledger.ensure(key, row['sample_id'])
        job = ledger.get(key)
        if job['status'] == 'failed' and g.get('retry_failed', False):
            ledger.set(key, 'pending')
            job = ledger.get(key)
        folder = self.run/'pseudo_labels'/key
        folder.mkdir(parents=True, exist_ok=True)
        response_path, mask_path = folder/'response.json', folder/'mask.png'
        transform_path = folder/'transform.json'
        atomic_json(transform_path, transform)
        if job['status'] == 'accepted':
            meta = job['metadata']
            if mask_path.is_file() and sha256(mask_path) == meta.get('mask_sha256'):
                return job
            ledger.set(key, 'needs_review', dict(reason='已完成产物损坏或哈希不匹配，不重新付费请求'))
            return ledger.get(key)
        if job['status'] in ('needs_review', 'decoded', 'failed') and not response_path.is_file():
            return job
        if job['status'] == 'ambiguous' and not g['retry_ambiguous']:
            return job
        request_id = job['metadata'].get('request_id')
        if response_path.is_file() and job['status'] not in ('pending', 'ambiguous'):
            if sha256(response_path) != job['metadata'].get('response_sha256'):
                ledger.set(key, 'needs_review', dict(reason='响应缓存哈希不匹配'))
                return ledger.get(key)
            response = json.loads(response_path.read_text(encoding='utf-8'))
        else:
            response = None
            for attempt in range(g['max_attempts']):
                if self.sent >= g['max_requests']:
                    raise RuntimeError('达到本次运行gateway.max_requests，未继续发请求')
                pause = 60/g['requests_per_minute']-(time.monotonic()-self.last_request)
                if pause > 0:
                    # Long configured intervals are split; no single sleep exceeds 30s.
                    while pause > 0:
                        part = min(pause, 30)
                        time.sleep(part)
                        pause -= part
                headers = {g['auth_header']: g['auth_prefix']+g['api_key'], 'Content-Type': 'application/json', 'Accept': 'application/json'}
                if g['idempotency_header']:
                    headers[g['idempotency_header']] = key
                ledger.set(key, 'inflight', dict(row=row, request_signature=signature, transform_path=str(transform_path)), attempt=True)
                self.sent += 1
                self.last_request = time.monotonic()
                started = time.monotonic()
                try:
                    status, response_headers, blob = self.transport(g['request_url'], 'POST', json.dumps(body).encode(), headers, g)
                except (TimeoutError, socket.timeout, ConnectionError, OSError, http.client.HTTPException):
                    ledger.set(key, 'ambiguous', dict(reason='网络异常或读超时，无法确认服务是否完成；默认不重发'))
                    return ledger.get(key)
                except ValueError:
                    ledger.set(key, 'ambiguous', dict(reason='响应过大或传输异常，无法确认结果'))
                    return ledger.get(key)
                elapsed = time.monotonic()-started
                if status in (429, 500, 502, 503, 504):
                    ledger.set(key, 'pending', dict(http_status=status, reason='有限退避后重试'))
                    if attempt+1 < g['max_attempts']:
                        time.sleep(min(30, retry_delay(response_headers.get('retry-after'), attempt, g['max_retry_delay_seconds'])))
                        continue
                    ledger.set(key, 'failed', dict(http_status=status))
                    return ledger.get(key)
                if status != 200:
                    ledger.set(key, 'failed', dict(http_status=status, reason='配置或服务错误；不输出可能含凭据的响应正文'))
                    raise RuntimeError(f'中转站HTTP {status}，核实完整端点、协议和认证；详情仅记录状态码')
                request_id = response_headers.get('x-request-id')
                try:
                    response = json.loads(blob)
                except (json.JSONDecodeError, UnicodeDecodeError):
                    ledger.set(key, 'needs_review', dict(reason='HTTP200但响应不是JSON；可能地址指向网页，需核实完整API路由'))
                    return ledger.get(key)
                response = scrub_response(response, g['api_key'])
                atomic_json(response_path, response)
                ledger.set(key, 'received', dict(response_path=str(response_path), response_sha256=sha256(response_path),
                                                request_id=request_id, elapsed_seconds=elapsed,
                                                usage=response.get('usage') if isinstance(response, dict) else None,
                                                cost='unknown'))
                break
        if response is None:
            return ledger.get(key)
        parsed = parse_response(response, row, g, self.download, request_id)
        if parsed.mask is None:
            ledger.set(key, 'needs_review', dict(parse_status=parsed.status, reason=parsed.failure_reason))
            return ledger.get(key)
        ledger.set(key, 'decoded', dict(parse_status=parsed.status))
        save_png(mask_path, parsed.mask)
        save_overlay(folder/'overlay.png', row, parsed.mask)
        status, quality = check_quality(parsed.mask, row, self.cfg['quality'])
        if probe or not g['profile_verified']:
            status = 'needs_review'
        accepted = dict(row, pseudo_mask_path=str(mask_path), pseudo_status=status, pseudo_version=key,
                        pseudo_sha256=sha256(mask_path), prompt_sha256=sha256(self.cfg['paths']['prompt_file']),
                        model=g['model'], request_job_key=key, transform_path=str(transform_path))
        ledger.set(key, status, dict(mask_path=str(mask_path), mask_sha256=sha256(mask_path),
                                     quality=quality, accepted_row=accepted))
        return ledger.get(key)


def export_accepted(ledger, path):
    rows = []
    for job in ledger.rows():
        if job['status'] == 'accepted':
            row = job['metadata']['accepted_row']
            if sha256(row['pseudo_mask_path']) != row['pseudo_sha256']:
                raise ValueError('accepted产物哈希不匹配')
            rows.append(row)
    # Multiple versions of the same sample must not silently overwrite each other.
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('同一样本存在多个accepted版本；请为新配置设置独立run_dir')
    write_manifest(path, rows)
    return rows


def run_gateway(cfg, rows, limit, probe=False, dry_run=False):
    prompt = validate_gateway(cfg, live=not dry_run, batch=not probe)
    if limit is None or limit < 1 or (probe and limit != 1):
        raise ValueError('必须显式给出正数--limit；probe只允许1张')
    selected = [r for r in rows if r['split'] == 'train'][:limit]
    if not selected:
        raise ValueError('没有训练图可请求')
    if dry_run:
        image_url, _, transform = request_image(selected[0], cfg['gateway'])
        body = make_body(cfg['gateway'], prompt, image_url)
        return dict(network=False, url=cfg['gateway']['request_url'], protocol=cfg['gateway']['protocol'],
                    model=cfg['gateway']['model'], prompt_sha256=sha256(cfg['paths']['prompt_file']),
                    body_fields=list(body), image_mime='image/png', transform=transform)
    run = Path(cfg['paths']['run_dir'])
    with RunLock(run/'api.lock'):
        ledger = JobLedger(run/'api_jobs.sqlite3')
        try:
            ledger.recover_inflight()
            client = GatewayClient(cfg)
            result = []
            for row in selected:
                job = client.segment(row, prompt, ledger, probe)
                result.append(dict(sample_id=row['sample_id'], status=job['status'], attempts=job['attempts']))
            if not probe:
                export_accepted(ledger, cfg['paths']['accepted_manifest'])
            return result
        finally:
            ledger.close()


def rerun_qc(cfg):
    run = Path(cfg['paths']['run_dir'])
    if not (run/'api_jobs.sqlite3').is_file():
        raise FileNotFoundError('没有API账本可质检')
    with RunLock(run/'api.lock'):
        ledger = JobLedger(run/'api_jobs.sqlite3')
        try:
            # Promote a verified probe locally: changing only the parser profile must
            # not require paying for the same request again.
            if cfg['gateway']['profile_verified'] and cfg['gateway']['response_profile']:
                prompt = load_prompt(cfg['paths']['prompt_file'])
                def offline_transport(*args, **kwargs):
                    raise ValueError('pseudo-qc不联网；URL产物应先在live probe下载并保存')
                client = GatewayClient(cfg, transport=offline_transport)
                for job in ledger.rows():
                    meta = job['metadata']
                    if not meta.get('row') or not meta.get('response_path') or job['status'] in ('accepted', 'inflight', 'ambiguous'):
                        continue
                    if sha256(meta['response_path']) != meta.get('response_sha256'):
                        ledger.set(job['job_key'], 'needs_review', dict(reason='响应缓存哈希不匹配'))
                        continue
                    row = meta['row']
                    key, signature, _, _ = prepare_job(row, prompt, cfg['gateway'])
                    if signature != meta.get('request_signature'):
                        continue
                    current = ledger.get(key)
                    if current and current['status'] == 'accepted':
                        continue
                    folder = run/'pseudo_labels'/key
                    folder.mkdir(parents=True, exist_ok=True)
                    path = folder/'response.json'
                    if Path(meta['response_path']).resolve() != path.resolve():
                        # Copy only already-sanitized, hash-verified bytes.
                        tmp = path.with_name(path.name+'.tmp')
                        tmp.write_bytes(Path(meta['response_path']).read_bytes())
                        tmp.replace(path)
                    ledger.ensure(key, row['sample_id'])
                    ledger.set(key, 'received', dict(meta, response_path=str(path), response_sha256=sha256(path), source_probe_key=job['job_key']))
                    client.segment(row, prompt, ledger, probe=False)
            for job in ledger.rows():
                meta = job['metadata']
                if 'accepted_row' not in meta:
                    continue
                row = meta['accepted_row']
                if sha256(row['pseudo_mask_path']) != row['pseudo_sha256']:
                    ledger.set(job['job_key'], 'needs_review', dict(reason='产物哈希改变'))
                    continue
                with Image.open(row['pseudo_mask_path']) as im:
                    import numpy as np
                    status, quality = check_quality(np.asarray(im), row, cfg['quality'])
                if not cfg['gateway']['profile_verified']:
                    status = 'needs_review'
                row['pseudo_status'] = status
                ledger.set(job['job_key'], status, dict(quality=quality, accepted_row=row))
            return dict(accepted=len(export_accepted(ledger, cfg['paths']['accepted_manifest'])))
        finally:
            ledger.close()
