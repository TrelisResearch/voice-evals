#!/usr/bin/env python3
"""Resume from Step 4: rank top 100 using already-completed diff jobs."""
import os, json, requests
import numpy as np
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')

import pathlib
from datasets import load_dataset, Dataset, Audio
from huggingface_hub import HfApi

HF_TOKEN = os.environ['HF_TOKEN']
API_KEY = os.environ['TRELIS_STUDIO_API_KEY']
BASE = 'https://studio.trelis.com/api/v1'
HEADERS = {'Authorization': f'Bearer {API_KEY}'}
TMP = pathlib.Path('medical-asr/phase1/tmp')
api = HfApi(token=HF_TOKEN)

DIFF_MODELS = ['nvidia/canary-1b-v2', 'mistralai/Voxtral-Mini-3B-2507']

with open(TMP / 'eka_sentences_diff_jobs.json') as f:
    diff_jobs = json.load(f)

# Load otsu dataset (already pushed)
print("Loading eka-sentences-otsu...")
otsu_ds = load_dataset('ronanarraig/eka-sentences-otsu', split='test', token=HF_TOKEN)
otsu_ds = otsu_ds.cast_column('audio', otsu_ds.features['audio'].__class__(decode=False))
print(f"  {len(otsu_ds)} rows")

# Load diff results
print("\nLoading difficulty job results...")
model_cer_maps = {}
for model_id, job_id in diff_jobs.items():
    r = requests.get(f'{BASE}/evaluation/jobs/{job_id}', headers=HEADERS)
    data = r.json()
    pushed_id = data['result']['pushed_dataset_id']
    print(f"  {model_id.split('/')[-1]}: loading from {pushed_id}")
    res_ds = load_dataset(pushed_id, split='test', token=HF_TOKEN)
    res_ds = res_ds.cast_column('audio', res_ds.features['audio'].__class__(decode=False))
    ref_col = 'reference' if 'reference' in res_ds.column_names else 'text'
    cer_col = next(c for c in res_ds.column_names if 'cer' in c.lower() and 'entity' not in c.lower())
    model_cer_maps[model_id] = {row[ref_col]: float(row[cer_col]) for row in res_ds}

# Rank top 100
print("\nRanking top 100...")
rows_scored = []
for row in otsu_ds:
    w_cer = float(row['whisper_cer'])
    model_cers = [w_cer]
    for model_id in DIFF_MODELS:
        m_cer = model_cer_maps.get(model_id, {}).get(row['text'])
        if m_cer is not None:
            model_cers.append(m_cer)
    d = dict(row)
    d['median_cer'] = float(np.median(model_cers))
    d['n_models'] = len(model_cers)
    rows_scored.append(d)

rows_scored.sort(key=lambda r: r['median_cer'], reverse=True)
# Only 77 rows total, take all or top 100 whichever is smaller
top_n = min(100, len(rows_scored))
top_rows = rows_scored[:top_n]
print(f"  {top_n} rows, median CER range: {top_rows[-1]['median_cer']:.3f} – {top_rows[0]['median_cer']:.3f}")

hard_ds = Dataset.from_list(top_rows)
if 'audio' in hard_ds.column_names:
    hard_ds = hard_ds.cast_column('audio', Audio(sampling_rate=16000))
api.create_repo('ronanarraig/eka-sentences-hard-100', repo_type='dataset', private=True, exist_ok=True)
hard_ds.push_to_hub('ronanarraig/eka-sentences-hard-100', split='test', token=HF_TOKEN, private=True)
print(f"  Pushed ronanarraig/eka-sentences-hard-100 ({len(hard_ds)} rows)")
print("\nDone.")
