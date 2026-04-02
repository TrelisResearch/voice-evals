#!/usr/bin/env python3
"""
Step 10: Difficulty filter on CER-filtered datasets.
- Submit Canary 1B v2 + Voxtral Mini 3B eval jobs on eka-otsu and multimed-otsu
- Whisper CER already available as whisper_cer column
- Rank by median CER across 3 models (entity CER for EKA, overall CER for MultiMed)
- Take top 100 hardest rows per dataset → push eka-hard-100, multimed-hard-100
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
HEADERS = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
TMP = pathlib.Path('medical-asr/phase1/tmp')
api = HfApi(token=HF_TOKEN)

MODELS = ['nvidia/canary-1b-v2', 'mistralai/Voxtral-Mini-3B-2507']
DATASETS = ['ronanarraig/eka-otsu', 'ronanarraig/multimed-otsu']
TOP_N = 100

def check_balance():
    r = requests.get(f'{BASE}/balance', headers=HEADERS)
    return r.json().get('balance', '?')

def get_active_jobs():
    r = requests.get(f'{BASE}/evaluation/jobs?limit=50', headers=HEADERS)
    data = r.json()
    jobs = data if isinstance(data, list) else data.get('jobs', data.get('items', []))
    active = [j for j in jobs if j.get('status') in ('running', 'queued', 'pending', 'in_progress')]
    return len(active)

def submit_job(model_id, dataset_id, n_samples):
    # Wait if at concurrency limit
    while True:
        active = get_active_jobs()
        if active < 9:
            break
        print(f"  [{active} active jobs] waiting 20s...")
        time.sleep(20)

    r = requests.post(f'{BASE}/evaluation/jobs', headers=HEADERS, json={
        'model_id': model_id,
        'dataset_id': dataset_id,
        'split': 'test',
        'num_samples': n_samples,
        'normalizer': 'generic',
        'language': 'en',
        'push_results': True,
    })
    data = r.json()
    job_id = data.get('job_id') or data.get('id')
    print(f"  Submitted {model_id} on {dataset_id}: job_id={job_id}")
    return job_id

def poll_job(job_id, label, interval=20):
    while True:
        r = requests.get(f'{BASE}/evaluation/jobs/{job_id}', headers=HEADERS)
        data = r.json()
        status = data.get('status')
        if status == 'completed':
            print(f"  {label}: completed")
            return data
        elif status == 'failed':
            print(f"  {label}: FAILED — {data.get('error', '')[:100]}")
            return None
        else:
            print(f"  {label}: {status}... waiting {interval}s")
            time.sleep(interval)

def load_per_sample_results(job_data, label):
    pushed_id = job_data['result'].get('pushed_dataset_id')
    if not pushed_id:
        print(f"  WARNING: no pushed_dataset_id for {label}")
        return None
    print(f"  Loading results from {pushed_id}...")
    ds = load_dataset(pushed_id, split='test', token=HF_TOKEN)
    if 'audio' in ds.column_names:
        ds = ds.cast_column('audio', ds.features['audio'].__class__(decode=False))
    return ds

# ── Submit all jobs ───────────────────────────────────────────────
print(f"Balance before: {check_balance()}")

jobs = {}  # (model, dataset) -> job_id
for ds_id in DATASETS:
    # Get row count
    ds_info = load_dataset(ds_id, split='test', token=HF_TOKEN)
    n = len(ds_info)
    print(f"\n{ds_id}: {n} rows")
    for model_id in MODELS:
        label = f"{model_id.split('/')[-1]} / {ds_id.split('/')[-1]}"
        job_id = submit_job(model_id, ds_id, n)
        jobs[(model_id, ds_id)] = job_id

with open(TMP / 'difficulty_jobs.json', 'w') as f:
    json.dump({f"{m}|{d}": jid for (m, d), jid in jobs.items()}, f, indent=2)
print(f"\nAll jobs submitted. Polling for completion...")

# ── Poll all jobs ─────────────────────────────────────────────────
results = {}  # (model, dataset) -> per-sample ds
for (model_id, ds_id), job_id in jobs.items():
    label = f"{model_id.split('/')[-1]} / {ds_id.split('/')[-1]}"
    job_data = poll_job(job_id, label)
    if job_data:
        results_ds = load_per_sample_results(job_data, label)
        results[(model_id, ds_id)] = results_ds

# ── Build difficulty ranking ──────────────────────────────────────
print("\n=== Building difficulty rankings ===")

for ds_id in DATASETS:
    name = ds_id.split('/')[-1].replace('-otsu', '')
    print(f"\n--- {name} ---")

    # Load the CER-filtered dataset (has whisper_cer column)
    base_ds = load_dataset(ds_id, split='test', token=HF_TOKEN)
    base_ds = base_ds.cast_column('audio', base_ds.features['audio'].__class__(decode=False))

    # For EKA: try entity CER; for MultiMed: use overall CER
    use_entity_cer = (name == 'eka')

    # Build text → whisper_cer map from base dataset
    text_to_whisper = {row['text']: float(row['whisper_cer']) for row in base_ds}

    # Build text → model CER maps from new eval results
    model_cer_maps = {}
    for model_id in MODELS:
        res_ds = results.get((model_id, ds_id))
        if res_ds is None:
            print(f"  WARNING: no results for {model_id}")
            continue
        ref_col = 'reference' if 'reference' in res_ds.column_names else 'text'

        if use_entity_cer:
            # Look for entity CER column
            ent_cer_col = next((c for c in res_ds.column_names if 'entity' in c.lower() and 'cer' in c.lower()), None)
            cer_col = ent_cer_col if ent_cer_col else next(c for c in res_ds.column_names if 'cer' in c.lower())
            if ent_cer_col:
                print(f"  {model_id.split('/')[-1]}: using {ent_cer_col} for ranking")
            else:
                print(f"  {model_id.split('/')[-1]}: no entity CER found, falling back to {cer_col}")
        else:
            cer_col = next(c for c in res_ds.column_names if 'cer' in c.lower() and 'entity' not in c.lower())

        model_cer_maps[model_id] = {row[ref_col]: float(row[cer_col]) for row in res_ds}

    # Compute median CER across all 3 models per row
    # For whisper: use entity_cer from original eval results if EKA
    if use_entity_cer:
        whisper_results = load_dataset(
            'ronanarraig/eval-whisper-v3-eka-500-20260402-1447', split='test', token=HF_TOKEN)
        whisper_results = whisper_results.cast_column('audio', whisper_results.features['audio'].__class__(decode=False))
        ent_cer_col = next((c for c in whisper_results.column_names if 'entity' in c.lower() and 'cer' in c.lower()), None)
        if ent_cer_col:
            print(f"  Whisper: using {ent_cer_col}")
            text_to_whisper_ent = {row['reference']: float(row[ent_cer_col]) for row in whisper_results}
        else:
            print(f"  Whisper: no entity CER, using overall CER")
            text_to_whisper_ent = None
    else:
        text_to_whisper_ent = None

    rows_with_scores = []
    for row in base_ds:
        text = row['text']
        w_cer = text_to_whisper_ent.get(text) if text_to_whisper_ent else text_to_whisper.get(text)
        if w_cer is None:
            w_cer = text_to_whisper.get(text, 0.0)

        model_cers = [w_cer]
        for model_id in MODELS:
            m_map = model_cer_maps.get(model_id, {})
            m_cer = m_map.get(text)
            if m_cer is not None:
                model_cers.append(m_cer)

        median_cer = float(np.median(model_cers))
        row_dict = dict(row)
        row_dict['median_cer'] = median_cer
        row_dict['n_models'] = len(model_cers)
        rows_with_scores.append(row_dict)

    # Sort by median CER descending (hardest first)
    rows_with_scores.sort(key=lambda r: r['median_cer'], reverse=True)
    top_100 = rows_with_scores[:TOP_N]

    print(f"  Total scored rows: {len(rows_with_scores)}")
    print(f"  Top {TOP_N} median CER range: {top_100[-1]['median_cer']:.3f} – {top_100[0]['median_cer']:.3f}")

    # Push
    out_id = f'ronanarraig/{name}-hard-100'
    hard_ds = Dataset.from_list(top_100)
    if 'audio' in hard_ds.column_names:
        hard_ds = hard_ds.cast_column('audio', Audio(sampling_rate=16000))
    api.create_repo(repo_id=out_id, repo_type='dataset', private=True, exist_ok=True)
    hard_ds.push_to_hub(out_id, split='test', token=HF_TOKEN, private=True)
    print(f"  Pushed {out_id} ({len(hard_ds)} rows)")

print(f"\nBalance after: {check_balance()}")
print("\nStep 10 done.")
