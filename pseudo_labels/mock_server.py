"""Offline acceptance fixture: only binds loopback; masks are supplied, not inferred."""
import base64
from contextlib import contextmanager
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import io
import json
import threading
import numpy as np
from PIL import Image


def encode_rle(mask):
    flat = (mask.flatten(order='F') > 0).astype(np.uint8)
    counts, value, count = [], 0, 0
    for pixel in flat:
        if pixel == value:
            count += 1
        else:
            counts.append(count)
            value, count = int(pixel), 1
    counts.append(count)
    return dict(width=mask.shape[1], height=mask.shape[0], order='F', counts=counts)


@contextmanager
def mock_gateway(image_masks, prompt, model, key='mock-secret-do-not-log'):
    calls = []
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            try:
                assert self.path == '/responses'
                assert self.headers.get('Authorization') == 'Bearer '+key
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                assert body['model'] == model
                content = body['input'][0]['content']
                assert content[0] == dict(type='input_text', text=prompt)
                assert content[1]['type'] == 'input_image'
                data_url = content[1]['image_url']
                assert data_url.startswith('data:image/png;base64,')
                blob = base64.b64decode(data_url.split(',', 1)[1], validate=True)
                with Image.open(io.BytesIO(blob)) as im:
                    assert im.mode == 'RGB' and im.format == 'PNG'
                    image_key = hashlib.sha256(im.tobytes()).hexdigest()
                    mask = image_masks[image_key]
                    assert im.size == (mask.shape[1], mask.shape[0])
                payload = dict(id='mock-response', output=[dict(content=[dict(type='output_text', text=json.dumps(encode_rle(mask)))])],
                               usage=dict(input_tokens=1, output_tokens=1), mock=True)
                encoded = json.dumps(payload).encode()
                calls.append(dict(path=self.path, model=model, image_key=image_key, mime='image/png'))
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('x-request-id', 'mock-request')
                self.send_header('Content-Length', str(len(encoded)))
                self.end_headers()
                self.wfile.write(encoded)
            except Exception:
                self.send_response(400)
                self.end_headers()
    server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f'http://127.0.0.1:{server.server_port}/responses', calls
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
