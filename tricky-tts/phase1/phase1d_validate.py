"""
Round-trip validation for Phase 1d public split.
TTS: elevenlabs/eleven-multilingual-v2
ASR: assemblyai/universal-3-pro
"""
import os, json, re, io, time, requests
from pathlib import Path
from collections import defaultdict

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
        except requests.exceptions.Timeout:
            if attempt < retries: time.sleep(3)
    return None

data = json.loads(Path("tricky-tts/phase1/phase1d_generated.json").read_text())
public = [r for r in data if r["split"] == "public"]

out_path = Path("tricky-tts/phase1/phase1d_results.json")
results = json.loads(out_path.read_text()) if out_path.exists() else []
done = {r["text"] for r in results}

for i, row in enumerate(public):
    text, cat = row["text"], row["category"]
    if text in done:
        print(f"[{i+1:02d}] SKIP [{cat}]", flush=True)
        continue
    print(f"\n[{i+1:02d}/48] [{cat}] {text[:70]}...", flush=True)
    audio = synthesize(text)
    if not audio:
        print("  TTS failed", flush=True)
        continue
    transcript = transcribe(audio)
    if not transcript:
        print("  ASR failed", flush=True)
        continue
    w = wer(text, transcript)
    print(f"  WER={w:.3f}  → {transcript[:80]}", flush=True)
    results.append({"text": text, "category": cat, "transcript": transcript, "wer": round(w,3)})
    done.add(text)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

# Stats
print("\n" + "="*65, flush=True)
print("PHASE 1D — RESULTS", flush=True)
print("="*65, flush=True)
cat_wers = defaultdict(list)
for r in results:
    cat_wers[r["category"]].append(r["wer"])

print(f"\n{'Category':<20} {'N':>4} {'Avg':>7} {'Max':>7} {'Min':>7}", flush=True)
print("-"*50, flush=True)
for cat in sorted(cat_wers):
    ws = cat_wers[cat]
    print(f"{cat:<20} {len(ws):>4} {sum(ws)/len(ws):>7.3f} {max(ws):>7.3f} {min(ws):>7.3f}", flush=True)
all_wers = [r["wer"] for r in results]
print(f"\n{'Overall':<20} {len(all_wers):>4} {sum(all_wers)/len(all_wers):>7.3f}", flush=True)
easy = sum(1 for w in all_wers if w < 0.05)
hard = sum(1 for w in all_wers if w > 0.30)
print(f"Easy (WER<0.05): {easy}/{len(all_wers)} | Hard (WER>0.30): {hard}/{len(all_wers)}", flush=True)

print("\n── All rows sorted by WER ────────────────────────────────────────", flush=True)
for r in sorted(results, key=lambda x: x["wer"]):
    print(f"  [{r['category']:15s}] {r['wer']:.3f}  {r['text'][:70]}", flush=True)
