#!/usr/bin/env python3
"""
Steps 8d–10b: Full pipeline for eka-sentences-500.
1. Poll Whisper job → CER filter (Otsu + 5% floor) → push eka-sentences-otsu
2. Submit Canary + Voxtral difficulty jobs → poll → rank top 100 → push eka-sentences-hard-100
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

with open(TMP / 'eka_sentences_whisper_job.json') as f:
    whisper_job_id = json.load(f)['job_id']

def poll_job(job_id, label, interval=20):
    while True:
        r = requests.get(f'{BASE}/evaluation/jobs/{job_id}', headers=HEADERS)
        data = r.json()
        status = data.get('status')
        if status == 'completed':
            print(f"  {label}: completed")
            return data
        elif status == 'failed':
            print(f"  {label}: FAILED — {data.get('error','')[:100]}")
            return None
        else:
            print(f"  {label}: {status}... waiting {interval}s")
            time.sleep(interval)

def get_active_jobs():
    r = requests.get(f'{BASE}/evaluation/jobs?limit=50', headers=HEADERS)
    jobs = r.json().get('jobs', [])
    return len([j for j in jobs if j.get('status') in ('running','queued','pending','in_progress')])

def submit_job(model_id, dataset_id, n):
    while get_active_jobs() >= 9:
        print(f"  Concurrency limit, waiting 20s...")
        time.sleep(20)
    r = requests.post(f'{BASE}/evaluation/jobs', headers=HEADERS, json={
        'model_id': model_id, 'dataset_id': dataset_id,
        'split': 'test', 'num_samples': n,
        'normalizer': 'generic', 'language': 'en', 'push_results': True,
    })
    data = r.json()
    job_id = data.get('job_id') or data.get('id')
    print(f"  Submitted {model_id.split('/')[-1]}: {job_id}")
    return job_id

def otsu_threshold(values):
    values = np.array(values)
    thresholds = np.linspace(values.min(), values.max(), 500)
    best_t, best_var = 0, -1
    for t in thresholds:
        w0 = np.mean(values <= t)
        w1 = 1 - w0
        if w0 == 0 or w1 == 0: continue
        var = w0 * w1 * (np.mean(values[values <= t]) - np.mean(values[values > t])) ** 2
        if var > best_var:
            best_var, best_t = var, t
    return best_t

# ── Step 1: Poll Whisper job ──────────────────────────────────────
print("=== Step 1: Polling Whisper job ===")
job_data = poll_job(whisper_job_id, 'Whisper/eka-sentences-500')
if not job_data:
    raise SystemExit("Whisper job failed")

pushed_id = job_data['result']['pushed_dataset_id']
print(f"  Loading results from {pushed_id}...")
results_ds = load_dataset(pushed_id, split='test', token=HF_TOKEN)
results_ds = results_ds.cast_column('audio', results_ds.features['audio'].__class__(decode=False))

cer_col = next(c for c in results_ds.column_names if 'cer' in c.lower() and 'entity' not in c.lower())
ref_col = 'reference' if 'reference' in results_ds.column_names else 'text'
cer_values = [float(r[cer_col]) for r in results_ds]
text_to_cer = {r[ref_col]: float(r[cer_col]) for r in results_ds}

print(f"  CER stats: min={min(cer_values):.3f} max={max(cer_values):.3f} "
      f"mean={np.mean(cer_values):.3f} median={np.median(cer_values):.3f}")

# ── Step 2: CER filter ────────────────────────────────────────────
print("\n=== Step 2: Otsu + 5% floor CER filter ===")
t_otsu = otsu_threshold(cer_values)
CER_FLOOR = 0.05
print(f"  Otsu threshold: {t_otsu:.3f}")
n_kept = sum(1 for c in cer_values if CER_FLOOR <= c <= t_otsu)
print(f"  Rows passing (floor {CER_FLOOR} – Otsu {t_otsu:.3f}): {n_kept}/{len(cer_values)}")

orig_ds = load_dataset('ronanarraig/eka-sentences-500', split='test', token=HF_TOKEN)
orig_ds = orig_ds.cast_column('audio', orig_ds.features['audio'].__class__(decode=False))

filtered = []
for row in orig_ds:
    cer = text_to_cer.get(row['text'])
    if cer is not None and CER_FLOOR <= cer <= t_otsu:
        d = dict(row); d['whisper_cer'] = cer
        filtered.append(d)

print(f"  Matched filtered rows: {len(filtered)}")

otsu_ds = Dataset.from_list(filtered)
if 'audio' in otsu_ds.column_names:
    otsu_ds = otsu_ds.cast_column('audio', Audio(sampling_rate=16000))
api.create_repo('ronanarraig/eka-sentences-otsu', repo_type='dataset', private=True, exist_ok=True)
otsu_ds.push_to_hub('ronanarraig/eka-sentences-otsu', split='test', token=HF_TOKEN, private=True)
print(f"  Pushed ronanarraig/eka-sentences-otsu ({len(otsu_ds)} rows)")

with open(TMP / 'eka_sentences_otsu_stats.json', 'w') as f:
    json.dump({'otsu_threshold': float(t_otsu), 'cer_floor': CER_FLOOR,
               'n_original': len(cer_values), 'n_kept': len(filtered)}, f, indent=2)

# ── Step 3: Difficulty filter ─────────────────────────────────────
print("\n=== Step 3: Difficulty filter (Canary + Voxtral) ===")
DIFF_MODELS = ['nvidia/canary-1b-v2', 'mistralai/Voxtral-Mini-3B-2507']
diff_jobs = {}
for model_id in DIFF_MODELS:
    jid = submit_job(model_id, 'ronanarraig/eka-sentences-otsu', len(filtered))
    diff_jobs[model_id] = jid

with open(TMP / 'eka_sentences_diff_jobs.json', 'w') as f:
    json.dump(diff_jobs, f, indent=2)

diff_results = {}
for model_id, jid in diff_jobs.items():
    label = f"{model_id.split('/')[-1]}/eka-sentences-otsu"
    job_data = poll_job(jid, label)
    if job_data:
        pushed = job_data['result']['pushed_dataset_id']
        ds = load_dataset(pushed, split='test', token=HF_TOKEN)
        ds = ds.cast_column('audio', ds.features['audio'].__class__(decode=False))
        diff_results[model_id] = ds

# ── Step 4: Rank top 100 ──────────────────────────────────────────
print("\n=== Step 4: Ranking top 100 hardest rows ===")
model_cer_maps = {}
for model_id, res_ds in diff_results.items():
    ref_col = 'reference' if 'reference' in res_ds.column_names else 'text'
    cer_col = next(c for c in res_ds.column_names if 'cer' in c.lower() and 'entity' not in c.lower())
    model_cer_maps[model_id] = {row[ref_col]: float(row[cer_col]) for row in res_ds}

otsu_ds = otsu_ds.cast_column('audio', otsu_ds.features['audio'].__class__(decode=False))
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
top_100 = rows_scored[:100]
print(f"  Top 100 median CER range: {top_100[-1]['median_cer']:.3f} – {top_100[0]['median_cer']:.3f}")

hard_ds = Dataset.from_list(top_100)
if 'audio' in hard_ds.column_names:
    hard_ds = hard_ds.cast_column('audio', Audio(sampling_rate=16000))
api.create_repo('ronanarraig/eka-sentences-hard-100', repo_type='dataset', private=True, exist_ok=True)
hard_ds.push_to_hub('ronanarraig/eka-sentences-hard-100', split='test', token=HF_TOKEN, private=True)
print(f"  Pushed ronanarraig/eka-sentences-hard-100 ({len(hard_ds)} rows)")

print("\nPipeline steps 8d–10b done. Ready for entity extraction (script 09b).")
