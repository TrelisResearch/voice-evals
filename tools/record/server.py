"""Minimal local server for the phase 4 recording app.

Serves record.html and saves uploaded audio to tricky-tts/phase4/audio/.
Run from repo root: uv run tricky-tts/phase4/server.py
"""

import http.server
import json
import os
from pathlib import Path

PORT = 8234
AUDIO_DIR = Path("tricky-tts/phase4/audio")


class Handler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory="tricky-tts/phase4", **kwargs)

    def do_POST(self):
        if self.path == "/save":
            content_length = int(self.headers["Content-Length"])
            body = self.rfile.read(content_length)

            # Filename comes from query param
            from urllib.parse import urlparse, parse_qs
            params = parse_qs(urlparse(self.requestline.split()[1]).query)
            filename = params.get("name", ["recording.webm"])[0]

            # Sanitise
            filename = os.path.basename(filename)
            AUDIO_DIR.mkdir(parents=True, exist_ok=True)
            out_path = AUDIO_DIR / filename
            out_path.write_bytes(body)

            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps({"saved": str(out_path)}).encode())
            print(f"Saved {out_path} ({len(body)} bytes)")
        else:
            self.send_error(404)

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


if __name__ == "__main__":
    AUDIO_DIR.mkdir(parents=True, exist_ok=True)
    server = http.server.HTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Recording server at http://127.0.0.1:{PORT}/record.html")
    print(f"Audio saves to {AUDIO_DIR.resolve()}")
    server.serve_forever()
