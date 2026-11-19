import base64
import copy
import io
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from configuration import sha256
from pseudo_labels.gateway import (GatewayClient, load_prompt, make_body, rerun_qc, validate_gateway,
                                   scrub_response, prepare_job)
from pseudo_labels.jobs import JobLedger
from pseudo_labels.mock_server import encode_rle
from pseudo_labels.parsers import parse_response
from pseudo_labels.quality import check_quality


def response_for(row):
    with Image.open(row['gt_mask_path']) as im:
        rle = encode_rle(np.asarray(im))
    return dict(output=[dict(content=[dict(type='output_text', text=json.dumps(rle))])], usage=dict(input_tokens=12, output_tokens=15))


def test_url_body_mime_auth_and_no_secret_on_disk(mock_data, tmp_path):
    c, rows, _ = mock_data
    prompt = load_prompt(c['paths']['prompt_file'])
    calls = []
    def transport(url, method, body, headers, g):
        assert url == 'http://127.0.0.1:9999/responses' and method == 'POST'
        assert headers['Authorization'] == 'Bearer mock-secret-do-not-log'
        b = json.loads(body)
        assert b['model'] == c['gateway']['model']
        assert b['input'][0]['content'][0]['text'] == prompt
        image = b['input'][0]['content'][1]['image_url']
        assert image.startswith('data:image/png;base64,')
        with Image.open(io.BytesIO(base64.b64decode(image.split(',')[1]))) as im:
            assert im.mode == 'RGB' and im.size == (rows[0]['width'], rows[0]['height'])
        calls.append(url)
        return 200, {'x-request-id': 'fixture'}, json.dumps(response_for(rows[0])).encode()
    ledger = JobLedger(tmp_path/'jobs.sqlite3')
    client = GatewayClient(c, transport)
    first = client.segment(rows[0], prompt, ledger)
    second = client.segment(rows[0], prompt, ledger)
    assert first['status'] == second['status'] == 'accepted' and len(calls) == 1
    assert first['metadata']['usage']['input_tokens'] == 12
    ledger.close()
    for file in tmp_path.rglob('*'):
        if file.is_file():
            assert b'mock-secret-do-not-log' not in file.read_bytes()


def test_retry_429_is_bounded_and_timeout_is_ambiguous(mock_data, tmp_path):
    c, rows, _ = mock_data
    c['gateway']['max_retry_delay_seconds'] = 0
    prompt = load_prompt(c['paths']['prompt_file'])
    attempts = []
    def retrying(*args):
        attempts.append(1)
        if len(attempts) == 1:
            return 429, {'retry-after': '0'}, b'{}'
        return 200, {}, json.dumps(response_for(rows[0])).encode()
    ledger = JobLedger(tmp_path/'jobs.sqlite3')
    job = GatewayClient(c, retrying).segment(rows[0], prompt, ledger)
    assert job['attempts'] == 2 and job['status'] == 'accepted'
    timeout_calls = []
    def timeout(*args):
        timeout_calls.append(1)
        raise TimeoutError()
    client = GatewayClient(c, timeout)
    a = client.segment(rows[1], prompt, ledger)
    b = client.segment(rows[1], prompt, ledger)
    assert a['status'] == b['status'] == 'ambiguous' and len(timeout_calls) == 1
    ledger.close()


def test_request_budget_counts_retries(mock_data, tmp_path):
    c, rows, _ = mock_data
    c['gateway'].update(max_requests=1, max_retry_delay_seconds=0)
    ledger = JobLedger(tmp_path/'jobs.sqlite3')
    client = GatewayClient(c, lambda *args: (429, {}, b'{}'))
    with pytest.raises(RuntimeError, match='max_requests'):
        client.segment(rows[0], load_prompt(c['paths']['prompt_file']), ledger)
    assert client.sent == 1
    ledger.close()


def test_crashed_inflight_not_pending(tmp_path):
    ledger = JobLedger(tmp_path/'jobs.sqlite3')
    ledger.ensure('a', 'sample')
    ledger.set('a', 'inflight', attempt=True)
    ledger.close()
    ledger = JobLedger(tmp_path/'jobs.sqlite3')
    ledger.recover_inflight()
    assert ledger.get('a')['status'] == 'ambiguous'
    ledger.close()


def test_probe_promoted_locally_without_second_request(mock_data, tmp_path):
    c, rows, _ = mock_data
    c['gateway'].update(response_profile='', response_field='', profile_verified=False)
    ledger = JobLedger(tmp_path/'api_jobs.sqlite3')
    client = GatewayClient(c, lambda *args: (200, {}, json.dumps(response_for(rows[0])).encode()))
    probe = client.segment(rows[0], load_prompt(c['paths']['prompt_file']), ledger, probe=True)
    assert probe['status'] == 'needs_review'
    ledger.close()
    c['gateway'].update(response_profile='rle', response_field='output.0.content.0.text', profile_verified=True)
    assert rerun_qc(c)['accepted'] == 1
    assert len(Path(c['paths']['accepted_manifest']).read_text(encoding='utf-8').splitlines()) == 1


@pytest.mark.parametrize('kind', ['text', 'overlay', 'wrong_size', 'corrupt', 'nonbinary'])
def test_unusable_response_rejected(mock_data, kind):
    c, rows, _ = mock_data
    row, g = rows[0], copy.deepcopy(c['gateway'])
    g['response_field'] = 'mask'
    if kind == 'text':
        g['response_profile'] = 'rle'
        response = {'mask': '分割完成'}
    else:
        g['response_profile'] = 'binary_png_base64'
        im = Image.new('RGB' if kind == 'overlay' else 'L',
                       (2, 3) if kind == 'wrong_size' else (row['width'], row['height']), 100 if kind == 'nonbinary' else 0)
        buf = io.BytesIO()
        im.save(buf, 'PNG')
        response = {'mask': base64.b64encode(b'broken' if kind == 'corrupt' else buf.getvalue()).decode()}
    parsed = parse_response(response, row, g)
    assert parsed.mask is None and parsed.status != 'decoded'


def test_rle_non_symmetric_coordinates_and_explicit_polarity(mock_data):
    c, rows, _ = mock_data
    row = rows[0]
    with Image.open(row['gt_mask_path']) as im:
        target = np.asarray(im)
    g = dict(c['gateway'], response_field='mask')
    parsed = parse_response({'mask': encode_rle(target)}, row, g)
    np.testing.assert_array_equal(parsed.mask, target)
    wrong_order = encode_rle(target)
    wrong_order['order'] = 'C'
    assert parse_response({'mask': wrong_order}, row, g).mask is None
    g['foreground_value'] = 0
    reversed_result = parse_response({'mask': encode_rle(target)}, row, g)
    np.testing.assert_array_equal(reversed_result.mask, 255-target)
    assert check_quality(reversed_result.mask, row, c['quality'])[0] == 'needs_review'


def test_extra_body_cannot_override_and_custom_requires_mapping(mock_data):
    c, _, _ = mock_data
    g = dict(c['gateway'], extra_body={'model': 'different'})
    with pytest.raises(ValueError):
        make_body(g, 'exact prompt', 'data:image/png;base64,xxx')
    g = dict(c['gateway'], protocol='custom_json')
    with pytest.raises(ValueError, match='request_template'):
        make_body(g, 'prompt', 'image')


def test_response_redaction_preserves_usage():
    scrubbed = scrub_response({'usage': {'input_tokens': 10}, 'access_token': 'secret', 'text': 'prefix-secret'}, 'secret')
    assert scrubbed['usage']['input_tokens'] == 10
    assert 'secret' not in json.dumps(scrubbed)


def test_response_profile_required_for_batch_but_not_local_training(mock_data):
    c, _, _ = mock_data
    c['gateway']['profile_verified'] = False
    with pytest.raises(ValueError, match='profile'):
        validate_gateway(c, batch=True)
    c['gateway']['api_key'] = ''
    from training.stages import Pipeline
    assert Pipeline(c).labeled
