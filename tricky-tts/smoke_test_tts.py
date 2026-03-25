"""
Quick TTS smoke test using Trelis Router /api/v1/synthesize.
Tests one text per category from the public split.
"""

import os
import json
import requests
from pathlib import Path

env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

ROUTER_KEY = os.environ["TRELIS_ROUTER_API_KEY"]
BASE_URL = "https://router.trelis.com"
HEADERS = {"Authorization": f"Bearer {ROUTER_KEY}"}

data = json.loads(Path("/home/claude/TR/voice-evals/tricky-tts/generated_texts.json").read_text())
public = [r for r in data if r["split"] == "public"]

# Pick one text per category
from collections import defaultdict
by_cat = defaultdict(list)
for r in public:
    by_cat[r["category"]].append(r)

samples = [rows[0] for rows in by_cat.values()]

output_dir = Path("/home/claude/TR/voice-evals/tricky-tts/smoke_test_audio")
output_dir.mkdir(exist_ok=True)

for sample in samples:
    text = sample["text"]
    category = sample["category"]
    print(f"\n[{category}] {text[:80]}...", flush=True)

    resp = requests.post(
        f"{BASE_URL}/api/v1/synthesize",
        headers=HEADERS,
        data={
            "text": text,
            "model": "elevenlabs/eleven-multilingual-v2",
            "output_format": "mp3_44100_128",
        },
        timeout=30,
    )

    print(f"  Status: {resp.status_code}", flush=True)
    if resp.status_code == 200:
        out_file = output_dir / f"{category}.mp3"
        out_file.write_bytes(resp.content)
        chars = resp.headers.get("X-Character-Count", "?")
        cost = resp.headers.get("X-Cost-Dollars", "?")
        provider = resp.headers.get("X-Provider", "?")
        size_kb = len(resp.content) / 1024
        print(f"  Saved: {out_file.name} ({size_kb:.1f} KB) | chars={chars} cost=${cost} provider={provider}", flush=True)
    else:
        print(f"  Error: {resp.text[:300]}", flush=True)

print("\nDone.", flush=True)
