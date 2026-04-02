#!/usr/bin/env python3
"""Extract a few audio samples from each pilot dataset for manual listening."""
import os, json
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')

import pathlib, io
import soundfile as sf
import numpy as np
from datasets import load_dataset, Audio

HF_TOKEN = os.environ['HF_TOKEN']
OUT = pathlib.Path('medical-asr/phase1/audio_samples')
OUT.mkdir(exist_ok=True)

N = 5  # samples per dataset

# ── EKA ──────────────────────────────────────────────────────────
print("EKA samples...")
eka = load_dataset('ronanarraig/medical-pilot-eka', split='test', token=HF_TOKEN)
# Pick 1 conversation + 2 narration_sentence + 2 narration_entity
by_ctx = {}
for row in eka:
    ctx = row['recording_context']
    if ctx not in by_ctx:
        by_ctx[ctx] = []
    by_ctx[ctx].append(row)

picks = (by_ctx.get('conversation', [])[:1] +
         by_ctx.get('narration_sentence', [])[:2] +
         by_ctx.get('narration_entity', [])[:2])

for i, row in enumerate(picks):
    audio = row['audio']
    arr = np.array(audio['array'], dtype=np.float32)
    sr = audio['sampling_rate']
    fname = OUT / f'eka_{i+1}_{row["recording_context"]}.wav'
    sf.write(fname, arr, sr)
    import json as _json
    ents = _json.loads(row['entities']) if row['entities'] else []
    ent_str = ', '.join(f"{e['text']} ({e['category']})" for e in ents[:3])
    print(f"  [{i+1}] {fname.name}")
    print(f"       text: {row['text'][:80]}")
    print(f"       entities: {ent_str}")

# ── MultiMed ─────────────────────────────────────────────────────
print("\nMultiMed samples...")
mm = load_dataset('ronanarraig/medical-pilot-multimed', split='test', token=HF_TOKEN)
# Pick rows with agreed entities
rows_with_ents = [r for r in mm if r['entities'] and r['entities'] != '[]']
picks_mm = rows_with_ents[:3] + [r for r in mm if r['entities'] == '[]'][:2]

for i, row in enumerate(picks_mm[:5]):
    audio = row['audio']
    arr = np.array(audio['array'], dtype=np.float32)
    sr = audio['sampling_rate']
    fname = OUT / f'multimed_{i+1}.wav'
    sf.write(fname, arr, sr)
    ents = json.loads(row['entities']) if row['entities'] else []
    ent_str = ', '.join(f"{e['text']} ({e['category']})" for e in ents[:3])
    print(f"  [{i+1}] {fname.name}")
    print(f"       text: {row['text'][:80]}")
    print(f"       entities: {ent_str or '(none)'}")

# ── United ───────────────────────────────────────────────────────
print("\nUnited-Syn-Med samples...")
united = load_dataset('ronanarraig/medical-pilot-united', split='test', token=HF_TOKEN)
# Pick rows with high drug entity density
rows_with_ents = [r for r in united if r['entities'] and r['entities'] != '[]']

for i, row in enumerate(rows_with_ents[:5]):
    audio = row['audio']
    arr = np.array(audio['array'], dtype=np.float32)
    sr = audio['sampling_rate']
    fname = OUT / f'united_{i+1}.wav'
    sf.write(fname, arr, sr)
    ents = json.loads(row['entities']) if row['entities'] else []
    ent_str = ', '.join(f"{e['text']} ({e['category']})" for e in ents[:3])
    print(f"  [{i+1}] {fname.name}")
    print(f"       text: {row['text'][:80]}")
    print(f"       entities: {ent_str}")

print(f"\nAll samples saved to: {OUT.resolve()}")
