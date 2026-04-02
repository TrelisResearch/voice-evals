#!/usr/bin/env python3
"""
Step 8: Poll for Whisper 500-row eval jobs, fetch per-sample CER results,
apply Otsu threshold + 5% CER floor, push filtered datasets.
"""
import os, json, time, requests
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

def otsu_threshold(values):
    """Find Otsu threshold for a list of CER values."""
    values = np.array(values)
    # Try thresholds across the range
    thresholds = np.linspace(values.min(), values.max(), 500)
    best_t, best_var = 0, -1
    for t in thresholds:
        w0 = np.mean(values <= t)
        w1 = 1 - w0
        if w0 == 0 or w1 == 0:
            continue
        var = w0 * w1 * (np.mean(values[values <= t]) - np.mean(values[values > t])) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t

def poll_until_done(job_id, name, interval=20):
    print(f"  Polling {name} ({job_id})...")
    while True:
        r = requests.get(f'{BASE}/evaluation/jobs/{job_id}', headers=HEADERS)
        data = r.json()
        status = data.get('status')
        if status == 'completed':
            print(f"  {name}: completed")
            return data
        elif status == 'failed':
            print(f"  {name}: FAILED — {data.get('error', '')[:100]}")
            return None
        else:
            print(f"  {name}: {status}... waiting {interval}s")
            time.sleep(interval)

def get_per_sample_results(job_data):
    """Fetch per-sample results from the pushed dataset."""
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

filtered_datasets = {}

for name, job_id in jobs.items():
    print(f"\n=== {name} ===")
    job_data = poll_until_done(job_id, name)
    if not job_data:
        continue

    # Get per-sample results
    results_ds = get_per_sample_results(job_data)
    if results_ds is None:
        continue

    print(f"  Per-sample columns: {results_ds.column_names}")
    print(f"  Sample row keys: {results_ds.column_names}")

    # Extract CER per sample — column may be 'cer', 'sample_cer', etc.
    cer_col = next((c for c in results_ds.column_names if 'cer' in c.lower() and 'entity' not in c.lower()), None)
    if not cer_col:
        print(f"  ERROR: no CER column found in {results_ds.column_names}")
        continue

    cer_values = [float(v) for v in results_ds[cer_col]]
    print(f"  CER stats: min={min(cer_values):.3f} max={max(cer_values):.3f} "
          f"mean={np.mean(cer_values):.3f} median={np.median(cer_values):.3f}")

    # Otsu threshold
    t_otsu = otsu_threshold(cer_values)
    print(f"  Otsu threshold: {t_otsu:.3f}")

    # Apply: keep CER <= Otsu AND CER >= 0.05 (not too easy)
    CER_FLOOR = 0.05
    keep_mask = [(cer_values[i] <= t_otsu and cer_values[i] >= CER_FLOOR)
                 for i in range(len(cer_values))]
    n_keep = sum(keep_mask)
    print(f"  After Otsu (≤{t_otsu:.3f}) + floor (≥{CER_FLOOR}): {n_keep}/{len(cer_values)} rows kept")

    # Load original 500-row dataset and filter
    orig_ds = load_dataset(f'ronanarraig/{name}-500', split='test', token=HF_TOKEN)
    orig_ds = orig_ds.cast_column('audio', orig_ds.features['audio'].__class__(decode=False))
    # Align by text (results_ds may be reordered)
    # results_ds uses 'reference' for the transcript text
    text_col = 'reference' if 'reference' in results_ds.column_names else 'text'
    text_to_cer = {results_ds[i][text_col]: cer_values[i] for i in range(len(results_ds))}

    filtered_rows = []
    for row in orig_ds:
        cer = text_to_cer.get(row['text'])
        if cer is not None and cer <= t_otsu and cer >= CER_FLOOR:
            row_dict = dict(row)
            row_dict['whisper_cer'] = cer
            filtered_rows.append(row_dict)

    print(f"  Matched and filtered: {len(filtered_rows)} rows")

    # Save stats
    stats = {
        'n_original': len(cer_values),
        'otsu_threshold': float(t_otsu),
        'cer_floor': CER_FLOOR,
        'n_kept': len(filtered_rows),
        'cer_distribution': {
            'min': float(min(cer_values)),
            'max': float(max(cer_values)),
            'mean': float(np.mean(cer_values)),
            'median': float(np.median(cer_values)),
        }
    }
    with open(TMP / f'{name}_otsu_stats.json', 'w') as f:
        json.dump(stats, f, indent=2)

    # Push filtered dataset
    filtered_ds = Dataset.from_list(filtered_rows)
    if 'audio' in filtered_ds.column_names:
        filtered_ds = filtered_ds.cast_column('audio', Audio(sampling_rate=16000))
    repo_id = f'ronanarraig/{name}-otsu'
    api.create_repo(repo_id=repo_id, repo_type='dataset', private=True, exist_ok=True)
    filtered_ds.push_to_hub(repo_id, split='test', token=HF_TOKEN, private=True)
    print(f"  Pushed {repo_id} ({len(filtered_ds)} rows)")
    filtered_datasets[name] = {'repo_id': repo_id, 'n_rows': len(filtered_ds), 'stats': stats}

with open(TMP / 'otsu_filtered_datasets.json', 'w') as f:
    json.dump(filtered_datasets, f, indent=2)
print(f"\nStep 8 done. Filtered datasets: {list(filtered_datasets.keys())}")
