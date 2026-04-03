#!/usr/bin/env python3
"""
Step 14: Consensus ground-truth estimation for high-density medical rows.
Sends audio + 3 transcripts (YT, Whisper, Gemini) to Gemini 3 Flash.
Exports data for tools/review UI. Set TEST_ONLY=False for full run.
"""
import os, json, re, io, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')

import soundfile as sf
import numpy as np
from google import genai
from google.genai import types as gentypes
from datasets import load_dataset, Dataset, Audio
from huggingface_hub import HfApi

HF_TOKEN = os.environ['HF_TOKEN']
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
client = genai.Client(api_key=GEMINI_API_KEY)
api = HfApi(token=HF_TOKEN)

TEST_ONLY = True   # flip to False for full run
N_THREADS = 10
REVIEW_DIR = Path('tools/review/data')

PROMPT = """You are a medical transcription expert. Listen to the audio clip and produce the most accurate transcript of what was said.

Three automatic transcriptions are provided for reference — each has known weaknesses:
- YouTube auto-captions (often garbles rare medical terms and drug names):
  {yt}

- Whisper large-v3 (phonetically accurate but may misspell specialist terms):
  {whisper}

- Gemini 2.5 Pro ASR (strong medical vocabulary but may add filler words or rephrase):
  {gemini}

Output only the corrected transcript. No explanation, no quotes."""


def audio_to_wav_bytes(audio_dict):
    raw = audio_dict.get('bytes')
    if not raw:
        return None
    try:
        array, sr = sf.read(io.BytesIO(raw))
        if array.ndim > 1:
            array = array.mean(axis=1)
        out = io.BytesIO()
        sf.write(out, array.astype(np.float32), sr, format='WAV', subtype='PCM_16')
        return out.getvalue()
    except Exception as e:
        print(f"  Audio decode error: {e}")
        return None


def consensus_row(idx, row, yt_text, gemini_text):
    wav_bytes = audio_to_wav_bytes(row['audio'])
    prompt = PROMPT.format(yt=yt_text, whisper=row['text'], gemini=gemini_text)
    for attempt in range(3):
        try:
            parts = []
            if wav_bytes:
                parts.append(gentypes.Part.from_bytes(data=wav_bytes, mime_type='audio/wav'))
            parts.append(prompt)
            response = client.models.generate_content(
                model='gemini-3-flash-preview',
                contents=parts,
                config=gentypes.GenerateContentConfig(temperature=0.1),
            )
            return idx, response.text.strip()
        except Exception as e:
            if attempt == 2:
                print(f"  Row {idx} failed: {e}")
                return idx, None
            time.sleep(2 ** attempt)


# ── Load datasets ─────────────────────────────────────────────────
print("Loading datasets...")
med_raw = load_dataset('ronanarraig/multimed-sentences-medical', split='test', token=HF_TOKEN)
med_raw = med_raw.cast_column('audio', med_raw.features['audio'].__class__(decode=False))
high = sorted(
    [dict(r) for r in med_raw if r['medical_density'] == 'high'],
    key=lambda r: r['gemini_cer'], reverse=True
)
print(f"  {len(high)} high-density rows")

gemini_ds = load_dataset(
    'ronanarraig/eval-gemini-2.5-pro-multimed-sentences-transcribed-20260403-0658',
    split='test', token=HF_TOKEN
)
gemini_ds = gemini_ds.cast_column('audio', gemini_ds.features['audio'].__class__(decode=False))
whisper_to_gemini = {row['reference']: row['prediction'] for row in gemini_ds}

yt_ds = load_dataset('ronanarraig/multimed-sentences-500', split='test', token=HF_TOKEN)
yt_ds = yt_ds.cast_column('audio', yt_ds.features['audio'].__class__(decode=False))
yt_list = list(yt_ds)

if TEST_ONLY:
    high = high[:10]
    print("=== TEST MODE: 10 rows ===")

# ── Run consensus ─────────────────────────────────────────────────
print(f"\nRunning consensus on {len(high)} row(s) ({N_THREADS} threads)...")
results = [None] * len(high)

with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
    futures = {}
    for i, row in enumerate(high):
        sample_idx = int(re.search(r'sample_(\d+)', row['source_file']).group(1))
        yt_text = yt_list[sample_idx]['text'] if sample_idx < len(yt_list) else '(unavailable)'
        gemini_text = whisper_to_gemini.get(row['text'], '(unavailable)')
        futures[executor.submit(consensus_row, i, row, yt_text, gemini_text)] = i

    done = 0
    for future in as_completed(futures):
        i, consensus = future.result()
        results[i] = consensus
        done += 1
        print(f"  {done}/{len(high)} done")

# ── Export for review UI ──────────────────────────────────────────
print(f"\nExporting to {REVIEW_DIR}...")
audio_dir = REVIEW_DIR / 'audio'
audio_dir.mkdir(parents=True, exist_ok=True)

rows_out = []
for i, (row, consensus) in enumerate(zip(high, results)):
    sample_idx = int(re.search(r'sample_(\d+)', row['source_file']).group(1))
    yt_text = yt_list[sample_idx]['text'] if sample_idx < len(yt_list) else '(unavailable)'
    gemini_text = whisper_to_gemini.get(row['text'], '(unavailable)')

    # Save audio as WAV
    wav = audio_to_wav_bytes(row['audio'])
    if wav:
        (audio_dir / f'{i}.wav').write_bytes(wav)

    rows_out.append({
        'id': i,
        'source_file': row['source_file'],
        'gemini_cer': row['gemini_cer'],
        'medical_density': row['medical_density'],
        'entities': row['entities'],
        'yt_text': yt_text,
        'whisper_text': row['text'],
        'gemini_text': gemini_text,
        'consensus_text': consensus or row['text'],
        'reviewed': False,
    })

    print(f"\nRow {i}  CER={row['gemini_cer']:.3f}")
    print(f"  YT:        {yt_text[:100]}")
    print(f"  Whisper:   {row['text'][:100]}")
    print(f"  Gemini:    {gemini_text[:100]}")
    print(f"  CONSENSUS: {(consensus or row['text'])[:100]}")

(REVIEW_DIR / 'rows.json').write_text(json.dumps(rows_out, indent=2, ensure_ascii=False))
print(f"\nExported {len(rows_out)} rows to {REVIEW_DIR}")
print(f"\nStart review UI with:")
print(f"  uv run tools/review/server.py --data {REVIEW_DIR}")
