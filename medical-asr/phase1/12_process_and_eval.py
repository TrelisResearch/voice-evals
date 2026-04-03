#!/usr/bin/env python3
"""Resume from process step using the transcribed file store b556a69a."""
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

TRANSCRIBED_STORE_ID = 'b556a69a-0de0-49d7-aab6-6f308c4522d1'

def poll_eval_job(job_id, label, interval=20):
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

def otsu_threshold(values):
    values = np.array(values)
    thresholds = np.linspace(values.min(), values.max(), 500)
    best_t, best_var = 0, -1
    for t in thresholds:
        w0 = np.mean(values <= t); w1 = 1 - w0
        if w0 == 0 or w1 == 0: continue
        var = w0 * w1 * (np.mean(values[values <= t]) - np.mean(values[values > t])) ** 2
        if var > best_var: best_var, best_t = var, t
    return best_t

# ── Step 4: Process → push to HF ─────────────────────────────────
print("=== Step 4: Process transcribed store → HF ===")
r = requests.post(f'{BASE}/file-stores/{TRANSCRIBED_STORE_ID}/process', headers=HEADERS, json={
    'output_dataset_name': 'multimed-sentences-transcribed',
    'output_org': 'ronanarraig',
    'hf_token': HF_TOKEN,
    'split_option': 'test_only',
    'language': 'eng',
    'enable_quality_checks': False,
})
resp = r.json()
print(f"  Response: {resp}")
process_job_id = resp.get('job_id') or resp.get('id')
print(f"  Process job: {process_job_id}")

# Poll the process job directly if we got a job_id
if process_job_id:
    print("  Polling process job...")
    for _ in range(180):
        time.sleep(15)
        r = requests.get(f'{BASE}/file-stores/{TRANSCRIBED_STORE_ID}', headers=HEADERS)
        data = r.json()
        pushed_id = data.get('output_dataset_id') or data.get('pushed_dataset_id')
        process_status = data.get('process_status') or data.get('processing_status')
        print(f"  status={process_status} pushed={pushed_id}")
        if pushed_id or process_status in ('completed', 'done'):
            print("  Process complete")
            break
        if process_status in ('failed', 'error'):
            raise SystemExit(f"Process failed: {data}")
else:
    # No job id — might have completed synchronously or inline
    print("  No job_id in response, checking if dataset already exists...")
    time.sleep(30)

# Try loading the output dataset
transcribed_id = 'ronanarraig/multimed-sentences-transcribed'
print(f"\n  Checking {transcribed_id}...")
for attempt in range(20):
    try:
        transcribed_ds = load_dataset(transcribed_id, split='test', token=HF_TOKEN)
        transcribed_ds = transcribed_ds.cast_column('audio', transcribed_ds.features['audio'].__class__(decode=False))
        print(f"  Dataset ready: {len(transcribed_ds)} rows, columns: {transcribed_ds.column_names}")
        break
    except Exception as e:
        print(f"  Not ready yet ({e}), waiting 30s...")
        time.sleep(30)
else:
    raise SystemExit("Dataset never appeared")

with open(TMP / 'multimed_transcribed_id.json', 'w') as f:
    json.dump({'dataset_id': transcribed_id}, f)

# ── Step 5: Gemini Pro eval + Otsu ───────────────────────────────
print("\n=== Step 5: Gemini Pro eval + Otsu CER filter ===")
n_rows = len(transcribed_ds)

r = requests.post(f'{BASE}/evaluation/jobs', headers=HEADERS, json={
    'model_id': 'google/gemini-2.5-pro',
    'dataset_id': transcribed_id,
    'split': 'test',
    'num_samples': n_rows,
    'normalizer': 'generic',
    'language': 'en',
    'push_results': True,
})
data = r.json()
job_id = data.get('job_id') or data.get('id')
print(f"  Gemini Pro eval job: {job_id}")

job_data = poll_eval_job(job_id, 'Gemini Pro')
if not job_data:
    raise SystemExit("Gemini eval failed")

results_ds = load_dataset(job_data['result']['pushed_dataset_id'], split='test', token=HF_TOKEN)
results_ds = results_ds.cast_column('audio', results_ds.features['audio'].__class__(decode=False))

cer_col = next(c for c in results_ds.column_names if 'cer' in c.lower() and 'entity' not in c.lower())
ref_col = 'reference' if 'reference' in results_ds.column_names else 'text'
cer_values = [float(row[cer_col]) for row in results_ds]
text_to_cer = {row[ref_col]: float(row[cer_col]) for row in results_ds}

print(f"  CER — min:{min(cer_values):.3f} mean:{np.mean(cer_values):.3f} "
      f"median:{np.median(cer_values):.3f} max:{max(cer_values):.3f}")

t_otsu = otsu_threshold(cer_values)
CER_FLOOR = 0.05
print(f"  Otsu threshold: {t_otsu:.3f}")

filtered = []
for row in transcribed_ds:
    cer = text_to_cer.get(row['text'])
    if cer is not None and CER_FLOOR <= cer <= t_otsu:
        d = dict(row); d['gemini_cer'] = cer
        filtered.append(d)
filtered.sort(key=lambda r: r['gemini_cer'], reverse=True)
print(f"  Rows passing filter: {len(filtered)}")

for name, rows in [('multimed-sentences-otsu', filtered),
                   ('multimed-sentences-inspect', filtered[:20])]:
    out_ds = Dataset.from_list(rows)
    if 'audio' in out_ds.column_names:
        out_ds = out_ds.cast_column('audio', Audio(sampling_rate=16000))
    api.create_repo(f'ronanarraig/{name}', repo_type='dataset', private=True, exist_ok=True)
    out_ds.push_to_hub(f'ronanarraig/{name}', split='test', token=HF_TOKEN, private=True)
    print(f"  Pushed ronanarraig/{name} ({len(out_ds)} rows)")

print("\nTop 10 hardest rows:")
for row in filtered[:10]:
    print(f"  {row['gemini_cer']:.3f}  {row['text'][:80]}")

print("\nDone.")
