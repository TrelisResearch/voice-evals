#!/usr/bin/env python3
"""
Generic transcript review server.
Serves review.html + audio files + rows.json from a data directory.
Accepts POST /save to write corrections back to rows.json.

Usage:
    uv run tools/review/server.py --data <data_dir> [--port 8765]
"""
import argparse
import http.server
import json
import os
from pathlib import Path
from urllib.parse import urlparse

DEFAULT_PORT = 8765
HERE = Path(__file__).parent


class ReviewHandler(http.server.SimpleHTTPRequestHandler):
    data_dir: Path = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(HERE), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        # Serve audio files from data_dir/audio/
        if parsed.path.startswith('/audio/'):
            fname = os.path.basename(parsed.path)
            fpath = self.data_dir / 'audio' / fname
            if fpath.exists():
                data = fpath.read_bytes()
                self.send_response(200)
                self.send_header('Content-Type', 'audio/wav')
                self.send_header('Content-Length', len(data))
                self.end_headers()
                self.wfile.write(data)
                return
            self.send_error(404)
            return
        # Serve rows.json
        if parsed.path == '/rows.json':
            data = (self.data_dir / 'rows.json').read_bytes()
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', len(data))
            self.end_headers()
            self.wfile.write(data)
            return
        # Default: serve static files from tools/review/
        super().do_GET()

    def do_POST(self):
        if self.path == '/save':
            length = int(self.headers.get('Content-Length', 0))
            body = json.loads(self.rfile.read(length))
            rows_path = self.data_dir / 'rows.json'
            rows = json.loads(rows_path.read_text())
            # Update matching row by id
            for row in rows:
                if row['id'] == body['id']:
                    row['consensus_text'] = body['consensus_text']
                    row['reviewed'] = True
                    break
            rows_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
            self._json(200, {'ok': True})
            print(f"  Saved review for row {body['id']}")
            return
        self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def _json(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', len(data))
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, fmt, *args):
        pass  # suppress request logs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Path to data directory (contains rows.json + audio/)')
    parser.add_argument('--port', type=int, default=DEFAULT_PORT)
    args = parser.parse_args()

    data_dir = Path(args.data).resolve()
    ReviewHandler.data_dir = data_dir

    url = f'http://localhost:{args.port}/review.html'
    print(f'Review UI: {url}')
    print(f'Data dir:  {data_dir}')
    print('Ctrl+C to stop.\n')

    server = http.server.HTTPServer(('127.0.0.1', args.port), ReviewHandler)
    server.serve_forever()
