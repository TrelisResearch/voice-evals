#!/usr/bin/env python3
"""
Step 8c: Final CER + length filtering for EKA and MultiMed.
- EKA: text length >= 10 AND 0.05 <= CER <= 0.588 (Otsu threshold)
- MultiMed: 0.05 <= CER <= 0.30
Overwrites ronanarraig/eka-otsu and ronanarraig/multimed-otsu.
"""
import os, json
import numpy as np
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')

import pathlib
from datasets import load_dataset, Dataset, Audio
from huggingface_hub import HfApi

HF_TOKEN = os.environ['HF_TOKEN']
TMP = pathlib.Path('medical-asr/phase1/tmp')
api = HfApi(token=HF_TOKEN)

configs = {
    'eka': {
        'results_id': 'ronanarraig/eval-whisper-v3-eka-500-20260402-1447',
        'source_id': 'ronanarraig/eka-500',
        'out_id': 'ronanarraig/eka-otsu',
        'cer_floor': 0.05,
        'cer_ceiling': 0.588,  # Otsu threshold
        'min_length': 10,
    },
    'multimed': {
        'results_id': 'ronanarraig/eval-whisper-v3-multimed-500-20260402-1448',
        'source_id': 'ronanarraig/multimed-500',
        'out_id': 'ronanarraig/multimed-otsu',
        'cer_floor': 0.05,
        'cer_ceiling': 0.30,
        'min_length': 0,
    },
}

final_datasets = {}

for name, cfg in configs.items():
    print(f"\n=== {name} ===")
    results_ds = load_dataset(cfg['results_id'], split='test', token=HF_TOKEN)
    results_ds = results_ds.cast_column('audio', results_ds.features['audio'].__class__(decode=False))

    ref_col = 'reference' if 'reference' in results_ds.column_names else 'text'
    text_to_cer = {row[ref_col]: float(row['cer']) for row in results_ds}

    orig_ds = load_dataset(cfg['source_id'], split='test', token=HF_TOKEN)
    orig_ds = orig_ds.cast_column('audio', orig_ds.features['audio'].__class__(decode=False))

    filtered = []
    n_short, n_cer_low, n_cer_high, n_no_match = 0, 0, 0, 0
    for row in orig_ds:
        cer = text_to_cer.get(row['text'])
        if cer is None:
            n_no_match += 1
            continue
        if len(row['text']) < cfg['min_length']:
            n_short += 1
            continue
        if cer < cfg['cer_floor']:
            n_cer_low += 1
            continue
        if cer > cfg['cer_ceiling']:
            n_cer_high += 1
            continue
        row_dict = dict(row)
        row_dict['whisper_cer'] = cer
        filtered.append(row_dict)

    print(f"  Kept: {len(filtered)}/{len(orig_ds)}")
    print(f"  Dropped — too short (<{cfg['min_length']} chars): {n_short}")
    print(f"  Dropped — CER too low (<{cfg['cer_floor']}): {n_cer_low}")
    print(f"  Dropped — CER too high (>{cfg['cer_ceiling']}): {n_cer_high}")
    if n_no_match:
        print(f"  Dropped — no CER match: {n_no_match}")

    ds = Dataset.from_list(filtered)
    if 'audio' in ds.column_names:
        ds = ds.cast_column('audio', Audio(sampling_rate=16000))
    api.create_repo(repo_id=cfg['out_id'], repo_type='dataset', private=True, exist_ok=True)
    ds.push_to_hub(cfg['out_id'], split='test', token=HF_TOKEN, private=True)
    print(f"  Pushed {cfg['out_id']} ({len(ds)} rows)")

    cer_vals = [r['whisper_cer'] for r in filtered]
    final_datasets[name] = {
        'repo_id': cfg['out_id'],
        'n_rows': len(filtered),
        'cer_floor': cfg['cer_floor'],
        'cer_ceiling': cfg['cer_ceiling'],
        'min_length': cfg['min_length'],
        'cer_stats': {
            'mean': float(np.mean(cer_vals)),
            'median': float(np.median(cer_vals)),
            'p25': float(np.percentile(cer_vals, 25)),
            'p75': float(np.percentile(cer_vals, 75)),
        }
    }

with open(TMP / 'final_filtered_datasets.json', 'w') as f:
    json.dump(final_datasets, f, indent=2)

print("\n=== Done ===")
for name, info in final_datasets.items():
    print(f"  {name}: {info['n_rows']} rows → {info['repo_id']}")
