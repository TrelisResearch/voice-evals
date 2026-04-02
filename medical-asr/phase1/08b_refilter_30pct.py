#!/usr/bin/env python3
"""
Step 8b: Re-filter EKA and MultiMed at 30% CER hard threshold (instead of Otsu).
Also push boundary slices (CER 0.25-0.40) for manual inspection.
"""
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

CER_CEILING = 0.30
CER_FLOOR = 0.05
BOUNDARY_LOW = 0.20   # rows to show around the threshold
BOUNDARY_HIGH = 0.45

def get_job_result(job_id, name):
    r = requests.get(f'{BASE}/evaluation/jobs/{job_id}', headers=HEADERS)
    data = r.json()
    status = data.get('status')
    if status != 'completed':
        print(f"  WARNING: {name} job status = {status}")
        return None
    return data

def get_per_sample_results(job_data):
    pushed_id = job_data['result'].get('pushed_dataset_id')
    if not pushed_id:
        print("  WARNING: no pushed_dataset_id in result")
        return None
    print(f"  Loading per-sample results from {pushed_id}...")
    ds = load_dataset(pushed_id, split='test', token=HF_TOKEN)
    if 'audio' in ds.column_names:
        ds = ds.cast_column('audio', ds.features['audio'].__class__(decode=False))
    return ds

with open(TMP / 'whisper_500_jobs.json') as f:
    jobs = json.load(f)

refiltered = {}

for name, job_id in jobs.items():
    print(f"\n=== {name} ===")
    job_data = get_job_result(job_id, name)
    if not job_data:
        continue

    results_ds = get_per_sample_results(job_data)
    if results_ds is None:
        continue

    cer_col = next((c for c in results_ds.column_names if 'cer' in c.lower() and 'entity' not in c.lower()), None)
    if not cer_col:
        print(f"  ERROR: no CER column in {results_ds.column_names}")
        continue

    cer_values = [float(v) for v in results_ds[cer_col]]
    text_col = 'reference' if 'reference' in results_ds.column_names else 'text'

    # Stats around threshold
    n_below_30 = sum(1 for c in cer_values if c <= CER_CEILING)
    n_with_floor = sum(1 for c in cer_values if CER_FLOOR <= c <= CER_CEILING)
    n_boundary = sum(1 for c in cer_values if BOUNDARY_LOW <= c <= BOUNDARY_HIGH)
    print(f"  CER ≤ {CER_CEILING}: {n_below_30}/{len(cer_values)}")
    print(f"  CER {CER_FLOOR}–{CER_CEILING} (floor applied): {n_with_floor}/{len(cer_values)}")
    print(f"  CER {BOUNDARY_LOW}–{BOUNDARY_HIGH} (boundary slice): {n_boundary}")

    # Load original 500-row dataset
    orig_ds = load_dataset(f'ronanarraig/{name}-500', split='test', token=HF_TOKEN)
    orig_ds = orig_ds.cast_column('audio', orig_ds.features['audio'].__class__(decode=False))

    text_to_cer = {results_ds[i][text_col]: cer_values[i] for i in range(len(results_ds))}

    filtered_rows = []
    boundary_rows = []
    for row in orig_ds:
        cer = text_to_cer.get(row['text'])
        if cer is None:
            continue
        row_dict = dict(row)
        row_dict['whisper_cer'] = cer
        if CER_FLOOR <= cer <= CER_CEILING:
            filtered_rows.append(row_dict)
        if BOUNDARY_LOW <= cer <= BOUNDARY_HIGH:
            boundary_rows.append(row_dict)

    print(f"  Matched filtered rows: {len(filtered_rows)}")
    print(f"  Matched boundary rows: {len(boundary_rows)}")

    # Sort boundary by CER so user can see progression
    boundary_rows.sort(key=lambda r: r['whisper_cer'])

    # Push filtered dataset (overwrite eka-otsu / multimed-otsu)
    repo_id = f'ronanarraig/{name}-otsu'
    filtered_ds = Dataset.from_list(filtered_rows)
    if 'audio' in filtered_ds.column_names:
        filtered_ds = filtered_ds.cast_column('audio', Audio(sampling_rate=16000))
    api.create_repo(repo_id=repo_id, repo_type='dataset', private=True, exist_ok=True)
    filtered_ds.push_to_hub(repo_id, split='test', token=HF_TOKEN, private=True)
    print(f"  Pushed {repo_id} ({len(filtered_ds)} rows)")

    # Push boundary slice for manual inspection
    boundary_id = f'ronanarraig/{name}-boundary'
    if boundary_rows:
        boundary_ds = Dataset.from_list(boundary_rows)
        if 'audio' in boundary_ds.column_names:
            boundary_ds = boundary_ds.cast_column('audio', Audio(sampling_rate=16000))
        api.create_repo(repo_id=boundary_id, repo_type='dataset', private=True, exist_ok=True)
        boundary_ds.push_to_hub(boundary_id, split='test', token=HF_TOKEN, private=True)
        print(f"  Pushed {boundary_id} ({len(boundary_ds)} rows, CER {BOUNDARY_LOW}–{BOUNDARY_HIGH})")

    # Save updated stats
    stats = {
        'n_original': len(cer_values),
        'cer_ceiling': CER_CEILING,
        'cer_floor': CER_FLOOR,
        'n_kept': len(filtered_rows),
        'n_boundary': len(boundary_rows),
        'cer_distribution': {
            'min': float(min(cer_values)),
            'max': float(max(cer_values)),
            'mean': float(np.mean(cer_values)),
            'median': float(np.median(cer_values)),
            'p25': float(np.percentile(cer_values, 25)),
            'p75': float(np.percentile(cer_values, 75)),
            'p90': float(np.percentile(cer_values, 90)),
        }
    }
    with open(TMP / f'{name}_30pct_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)
    refiltered[name] = {'repo_id': repo_id, 'boundary_id': boundary_id, 'stats': stats}

with open(TMP / 'refiltered_datasets.json', 'w') as f:
    json.dump(refiltered, f, indent=2)

print("\n=== Summary ===")
for name, info in refiltered.items():
    s = info['stats']
    print(f"  {name}: {s['n_kept']} rows kept (CER {s['cer_floor']}–{s['cer_ceiling']})")
    print(f"    Boundary slice (CER {s['cer_floor']}–{BOUNDARY_HIGH}): {s['n_boundary']} rows → {info['boundary_id']}")
    print(f"    Filtered dataset → {info['repo_id']}")
print("\nStep 8b done.")
