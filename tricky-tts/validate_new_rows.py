"""
Run round-trip ASR on the new rows only (those not in original results).
Then print combined stats for the full curated dataset.
"""

import os
import json
import re
import io
import time
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
ASR_MODEL = "assemblyai/universal-3-pro"

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def wer(ref, hyp):
    r, h = normalize(ref).split(), normalize(hyp).split()
    if not r: return 0.0
    d = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1,len(r)+1):
        for j in range(1,len(h)+1):
            cost = 0 if r[i-1]==h[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(r)][len(h)] / len(r)

def cer(ref, hyp):
    r = normalize(ref).replace(" ","")
    h = normalize(hyp).replace(" ","")
    if not r: return 0.0
    d = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1,len(r)+1):
        for j in range(1,len(h)+1):
            cost = 0 if r[i-1]==h[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(r)][len(h)] / len(r)

def synthesize(text):
    r = requests.post(f"{BASE_URL}/api/v1/synthesize", headers=HEADERS,
        data={"text": text, "model": "elevenlabs/eleven-multilingual-v2", "output_format": "mp3_44100_128"},
        timeout=30)
    return r.content if r.status_code == 200 else None

def transcribe(audio, retries=2):
    for attempt in range(retries+1):
        try:
            r = requests.post(f"{BASE_URL}/api/v1/transcribe", headers=HEADERS,
                files={"file": ("audio.mp3", io.BytesIO(audio), "audio/mpeg")},
                data={"model": ASR_MODEL, "language": "en", "output_format": "json"},
                timeout=120)
            if r.status_code == 200:
                return r.json().get("text","").strip()
            elif r.status_code == 429 and attempt < retries:
                time.sleep(3)
            else:
                print(f"  ASR error {r.status_code}: {r.text[:100]}", flush=True)
                return None
        except requests.exceptions.Timeout:
            if attempt < retries: time.sleep(3)
            else: return None
    return None

# Load data
curated = json.loads(Path("tricky-tts/phase1/curated_public.json").read_text())
original_results = {r["text"]: r for r in json.loads(Path("tricky-tts/phase1/roundtrip_results.json").read_text())}

out_path = Path("tricky-tts/phase1/curated_results.json")
if out_path.exists():
    curated_results = json.loads(out_path.read_text())
    done = {r["text"] for r in curated_results}
else:
    curated_results = []
    done = set()

# First: carry over already-tested rows
for row in curated:
    if row["text"] in original_results and row["text"] not in done:
        r = original_results[row["text"]]
        curated_results.append({"text": r["text"], "category": r["category"], "transcript": r.get("transcript",""), "wer": r["wer"], "cer": r["cer"]})
        done.add(row["text"])

# Then: test new rows
new_rows = [row for row in curated if row["text"] not in done]
print(f"Testing {len(new_rows)} new rows...\n", flush=True)

for i, row in enumerate(new_rows):
    text, cat = row["text"], row["category"]
    print(f"[{i+1:02d}/{len(new_rows)}] [{cat}] {text[:70]}...", flush=True)

    audio = synthesize(text)
    if not audio:
        print("  TTS failed", flush=True)
        continue

    transcript = transcribe(audio)
    if not transcript:
        print("  ASR failed", flush=True)
        continue

    w, c = wer(text, transcript), cer(text, transcript)
    print(f"  WER={w:.2f} CER={c:.2f}  → {transcript[:80]}", flush=True)

    curated_results.append({"text": text, "category": cat, "transcript": transcript, "wer": round(w,3), "cer": round(c,3)})
    done.add(text)
    out_path.write_text(json.dumps(curated_results, indent=2, ensure_ascii=False))

# Summary
print("\n" + "="*65, flush=True)
print("CURATED DATASET — FINAL STATS", flush=True)
print("="*65, flush=True)

from collections import defaultdict
cat_wers = defaultdict(list)
for r in curated_results:
    if "wer" in r:
        cat_wers[r["category"]].append(r["wer"])

print(f"\n{'Category':<20} {'Rows':>5} {'Avg WER':>8} {'Max WER':>8} {'Min WER':>8}", flush=True)
print("-"*55, flush=True)
for cat in sorted(cat_wers):
    ws = cat_wers[cat]
    print(f"{cat:<20} {len(ws):>5} {sum(ws)/len(ws):>8.3f} {max(ws):>8.3f} {min(ws):>8.3f}", flush=True)

all_wers = [r["wer"] for r in curated_results if "wer" in r]
print(f"\n{'Overall':<20} {len(all_wers):>5} {sum(all_wers)/len(all_wers):>8.3f}", flush=True)
easy = sum(1 for w in all_wers if w < 0.05)
hard = sum(1 for w in all_wers if w > 0.30)
print(f"\nEasy (WER<0.05): {easy}/{len(all_wers)} | Hard (WER>0.30): {hard}/{len(all_wers)}", flush=True)
