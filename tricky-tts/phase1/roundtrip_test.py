"""
Round-trip ASR test for tricky-tts public split.
Pipeline: text → TTS (elevenlabs) → ASR (3 models) → WER/CER → difficulty analysis

ASR models used (diverse architectures):
  - fireworks/whisper-v3-turbo  (cheap, fast)
  - assemblyai/universal-3-pro  (strong general)
  - deepgram/nova-3             (different engine)
"""

import os
import json
import re
import io
import time
import requests
from pathlib import Path

# ── env ────────────────────────────────────────────────────────────────────────
env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

ROUTER_KEY = os.environ["TRELIS_ROUTER_API_KEY"]
BASE_URL = "https://router.trelis.com"
HEADERS = {"Authorization": f"Bearer {ROUTER_KEY}"}

# Single strong ASR model — used as a consistent measuring instrument for TTS quality.
# Not testing the ASR; standardizing on one removes inconsistency noise.
ASR_MODEL = "assemblyai/universal-3-pro"

# ── text normalization for WER ─────────────────────────────────────────────────
def normalize(text: str) -> str:
    """Basic normalization: lowercase, strip punctuation, collapse whitespace."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def wer(ref: str, hyp: str) -> float:
    """Word error rate."""
    ref_words = normalize(ref).split()
    hyp_words = normalize(hyp).split()
    if not ref_words:
        return 0.0
    # Dynamic programming edit distance
    d = [[0] * (len(hyp_words) + 1) for _ in range(len(ref_words) + 1)]
    for i in range(len(ref_words) + 1):
        d[i][0] = i
    for j in range(len(hyp_words) + 1):
        d[0][j] = j
    for i in range(1, len(ref_words) + 1):
        for j in range(1, len(hyp_words) + 1):
            cost = 0 if ref_words[i-1] == hyp_words[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[len(ref_words)][len(hyp_words)] / len(ref_words)

def cer(ref: str, hyp: str) -> float:
    """Character error rate."""
    ref_n = normalize(ref).replace(" ", "")
    hyp_n = normalize(hyp).replace(" ", "")
    if not ref_n:
        return 0.0
    d = [[0] * (len(hyp_n) + 1) for _ in range(len(ref_n) + 1)]
    for i in range(len(ref_n) + 1):
        d[i][0] = i
    for j in range(len(hyp_n) + 1):
        d[0][j] = j
    for i in range(1, len(ref_n) + 1):
        for j in range(1, len(hyp_n) + 1):
            cost = 0 if ref_n[i-1] == hyp_n[j-1] else 1
            d[i][j] = min(d[i-1][j] + 1, d[i][j-1] + 1, d[i-1][j-1] + cost)
    return d[len(ref_n)][len(hyp_n)] / len(ref_n)

# ── API helpers ────────────────────────────────────────────────────────────────
def synthesize(text: str) -> bytes | None:
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
    if resp.status_code == 200:
        return resp.content
    print(f"    TTS error {resp.status_code}: {resp.text[:200]}", flush=True)
    return None

def transcribe(audio_bytes: bytes, model: str, retries: int = 2) -> str | None:
    for attempt in range(retries + 1):
        try:
            resp = requests.post(
                f"{BASE_URL}/api/v1/transcribe",
                headers=HEADERS,
                files={"file": ("audio.mp3", io.BytesIO(audio_bytes), "audio/mpeg")},
                data={"model": model, "language": "en", "output_format": "json"},
                timeout=120,
            )
        except requests.exceptions.Timeout:
            print(f"    ASR timeout [{model}] attempt {attempt+1}", flush=True)
            if attempt < retries:
                time.sleep(3)
                continue
            return None
        if resp.status_code == 200:
            try:
                return resp.json().get("text", "").strip()
            except Exception:
                return resp.text.strip()
        elif resp.status_code == 429 and attempt < retries:
            time.sleep(3)
        else:
            print(f"    ASR error {resp.status_code} [{model}]: {resp.text[:200]}", flush=True)
            return None
    return None

# ── main ───────────────────────────────────────────────────────────────────────
def main():
    data = json.loads(Path("/home/claude/TR/voice-evals/tricky-tts/phase1/generated_texts.json").read_text())
    public = [r for r in data if r["split"] == "public"]

    out = Path("/home/claude/TR/voice-evals/tricky-tts/phase1/roundtrip_results.json")

    # Resume: load previously saved results
    if out.exists():
        results = json.loads(out.read_text())
        done_texts = {r["text"] for r in results}
        print(f"Resuming — {len(results)} rows already done.", flush=True)
    else:
        results = []
        done_texts = set()

    for i, row in enumerate(public):
        text = row["text"]
        category = row["category"]

        if text in done_texts:
            print(f"[{i+1:02d}/{len(public)}] [{category}] SKIP (already done)", flush=True)
            continue

        print(f"\n[{i+1:02d}/{len(public)}] [{category}] {text[:70]}...", flush=True)

        audio = synthesize(text)
        if audio is None:
            print("  TTS failed, skipping", flush=True)
            continue

        transcript = transcribe(audio, ASR_MODEL)
        if transcript is None:
            print("  ASR failed, skipping", flush=True)
            continue

        w = wer(text, transcript)
        c = cer(text, transcript)
        print(f"  WER={w:.2f} CER={c:.2f}  → {transcript[:80]}", flush=True)

        row_result = {
            "text": text,
            "category": category,
            "transcript": transcript,
            "wer": round(w, 3),
            "cer": round(c, 3),
        }

        results.append(row_result)
        done_texts.add(text)

        # Incremental save after each row
        out.write_text(json.dumps(results, indent=2, ensure_ascii=False))

    print(f"\nSaved {len(results)} rows to {out}", flush=True)

    # ── analysis ──────────────────────────────────────────────────────────────
    print("\n" + "="*70, flush=True)
    print("DIFFICULTY ANALYSIS", flush=True)
    print("="*70, flush=True)

    from collections import defaultdict
    cat_wers = defaultdict(list)
    cat_cers = defaultdict(list)

    for r in results:
        if "wer" in r:
            cat_wers[r["category"]].append(r["wer"])
            cat_cers[r["category"]].append(r["cer"])

    print(f"\n{'Category':<20} {'Rows':>5} {'Avg WER':>8} {'Avg CER':>8} {'Max WER':>8} {'Min WER':>8}", flush=True)
    print("-" * 60, flush=True)
    for cat in sorted(cat_wers):
        ws = cat_wers[cat]
        cs = cat_cers[cat]
        print(f"{cat:<20} {len(ws):>5} {sum(ws)/len(ws):>8.3f} {sum(cs)/len(cs):>8.3f} {max(ws):>8.3f} {min(ws):>8.3f}", flush=True)

    all_wers = [r["wer"] for r in results if "wer" in r]
    print(f"\n{'Overall':<20} {len(all_wers):>5} {sum(all_wers)/len(all_wers):>8.3f}", flush=True)

    print("\n── Easiest rows (WER < 0.05) ─────────────────────────────────────────", flush=True)
    easy = [r for r in results if r.get("wer", 1) < 0.05]
    for r in sorted(easy, key=lambda x: x["wer"]):
        print(f"  [{r['category']}] WER={r['wer']:.3f}  {r['text'][:80]}", flush=True)

    print("\n── Hardest rows (WER > 0.30) ─────────────────────────────────────────", flush=True)
    hard = [r for r in results if r.get("wer", 0) > 0.30]
    for r in sorted(hard, key=lambda x: -x["wer"]):
        print(f"  [{r['category']}] WER={r['wer']:.3f}  {r['text'][:80]}", flush=True)

    print(f"\nEasy rows (WER<0.05): {len(easy)} | Hard rows (WER>0.30): {len(hard)}", flush=True)

if __name__ == "__main__":
    main()
