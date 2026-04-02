#!/usr/bin/env python3
"""
Step 4: Build and push the three 50-row pilot datasets to HuggingFace.
- ronanarraig/medical-pilot-eka (EKA EN, entity annotations converted)
- ronanarraig/medical-pilot-multimed (MultiMed EN, dual-LLM entity annotations)
- ronanarraig/medical-pilot-united (United-Syn-Med, dual-LLM entity annotations)
All pushed as private, split='test'.
"""
import os, json
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')
load_dotenv('/home/claude/TR/voice-evals/.env', override=False)

import pathlib
import numpy as np
from datasets import Dataset, Audio, Features, Value, Sequence
from huggingface_hub import HfApi

HF_TOKEN = os.environ['HF_TOKEN']
TMP = pathlib.Path('medical-asr/phase1/tmp')
VOL = pathlib.Path('/mnt/HC_Volume_105102660/voice-evals-data')
api = HfApi(token=HF_TOKEN)

def ensure_repo(repo_id):
    try:
        api.create_repo(repo_id=repo_id, repo_type='dataset', private=True, exist_ok=True)
        print(f"  Repo ready: {repo_id}")
    except Exception as e:
        print(f"  Repo error: {e}")

def push_dataset(ds, repo_id):
    ensure_repo(repo_id)
    ds.push_to_hub(repo_id, split='test', token=HF_TOKEN, private=True)
    print(f"  Pushed {len(ds)} rows → {repo_id}")

# ─────────────────────────────────────────────
# 1. EKA pilot
# ─────────────────────────────────────────────
print("\n=== Building EKA pilot ===")
from datasets import load_dataset

eka_raw = load_dataset('ekacare/eka-medical-asr-evaluation-dataset', 'en', split='test', token=HF_TOKEN)
eka_raw = eka_raw.cast_column('audio', eka_raw.features['audio'].__class__(decode=False))

with open(TMP / 'eka_pilot_rows.json') as f:
    eka_meta = json.load(f)  # non-audio metadata

# Re-select same rows by matching text
text_to_meta = {r['text']: r for r in eka_meta}
indices = []
for i, row in enumerate(eka_raw):
    if row['text'] in text_to_meta:
        indices.append(i)
    if len(indices) == len(eka_meta):
        break

eka_sample = eka_raw.select(indices)

def row_to_eka(row):
    meta = text_to_meta.get(row['text'], {})
    return {
        'text': row['text'],
        'audio': row['audio'],
        'duration': float(row['duration']),
        'speaker': row['speaker'],
        'recording_context': row['recording_context'],
        'type_concept': row['type_concept'],
        'entities': meta.get('entities', '[]'),
        'source': 'ekacare/eka-medical-asr-evaluation-dataset',
    }

eka_rows = [row_to_eka(r) for r in eka_sample]
eka_ds = Dataset.from_list(eka_rows).cast_column('audio', Audio(sampling_rate=16000))
print(f"EKA pilot: {len(eka_ds)} rows, columns: {eka_ds.column_names}")
push_dataset(eka_ds, 'ronanarraig/medical-pilot-eka')

# ─────────────────────────────────────────────
# 2. MultiMed EN pilot
# ─────────────────────────────────────────────
print("\n=== Building MultiMed EN pilot ===")
from datasets import load_from_disk

mm_ds = load_from_disk(str(TMP / 'multimed_pilot_ds'))
mm_ds = mm_ds.cast_column('audio', mm_ds.features['audio'].__class__(decode=False))

with open(TMP / 'multimed_entities.json') as f:
    mm_ents = json.load(f)
idx_to_ents = {r['idx']: r['entities'] for r in mm_ents}

def add_mm_entities(row, idx):
    # entities = only agreed-by-both
    return {'entities': idx_to_ents.get(idx, '[]'), 'source': 'leduckhai/MultiMed'}

mm_rows = []
for i, row in enumerate(mm_ds):
    mm_rows.append({
        'text': row['text'],
        'audio': row['audio'],
        'duration': float(row['duration']) if row['duration'] else 0.0,
        'entities': idx_to_ents.get(i, '[]'),
        'source': 'leduckhai/MultiMed',
    })

mm_out = Dataset.from_list(mm_rows).cast_column('audio', Audio(sampling_rate=16000))
print(f"MultiMed pilot: {len(mm_out)} rows, columns: {mm_out.column_names}")
push_dataset(mm_out, 'ronanarraig/medical-pilot-multimed')

# ─────────────────────────────────────────────
# 3. United-Syn-Med pilot
# ─────────────────────────────────────────────
print("\n=== Building United-Syn-Med pilot ===")
import soundfile as sf
import io

with open(TMP / 'united_pilot_rows.json') as f:
    united_rows = json.load(f)

with open(TMP / 'united_entities.json') as f:
    united_ents = json.load(f)
fn_to_ents = {r['file_name']: r['entities'] for r in united_ents}

united_out_rows = []
for row in united_rows:
    audio_path = pathlib.Path(row['audio_path'])
    if not audio_path.exists():
        # Try volume path
        audio_path = VOL / 'united_audio' / row['file_name']
    if not audio_path.exists():
        print(f"  WARNING: audio not found for {row['file_name']}")
        continue
    audio_bytes = audio_path.read_bytes()
    united_out_rows.append({
        'text': row['text'],
        'audio': {'bytes': audio_bytes, 'path': row['file_name']},
        'entities': fn_to_ents.get(row['file_name'], '[]'),
        'source': 'united-we-care/United-Syn-Med',
    })

united_ds = Dataset.from_list(united_out_rows).cast_column('audio', Audio(sampling_rate=16000))
print(f"United pilot: {len(united_ds)} rows, columns: {united_ds.column_names}")
push_dataset(united_ds, 'ronanarraig/medical-pilot-united')

print("\nAll three pilot datasets pushed.")
print("  ronanarraig/medical-pilot-eka")
print("  ronanarraig/medical-pilot-multimed")
print("  ronanarraig/medical-pilot-united")
