#!/usr/bin/env python3
"""Step 5: Gemini Pro eval on multimed-sentences-transcribed + Otsu filter."""
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

def poll_eval_job(job_id, label, interval=20):
    while True:
        r = requests.get(f'{BASE}/evaluation/jobs/{job_id}', headers=HEADERS)
        data = r.json()
        status = data.get('status')
        if status == 'completed':
            print(f"  {label}: completed"); return data
        elif status == 'failed':
            print(f"  {label}: FAILED — {data.get('error','')[:100]}"); return None
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

TRANSCRIBED_ID = 'ronanarraig/multimed-sentences-transcribed'

print(f"Loading {TRANSCRIBED_ID}...")
ds = load_dataset(TRANSCRIBED_ID, split='test', token=HF_TOKEN)
ds = ds.cast_column('audio', ds.features['audio'].__class__(decode=False))
print(f"  {len(ds)} rows, columns: {ds.column_names}")

# Show samples
print("\nSample transcribed rows:")
for row in list(ds)[:5]:
    print(f"  [{row.get('duration', '?'):.1f}s] {row['text'][:90]}" if 'duration' in row else f"  {row['text'][:90]}")

print(f"\nSubmitting Gemini Pro eval on {len(ds)} rows...")
r = requests.post(f'{BASE}/evaluation/jobs', headers=HEADERS, json={
    'model_id': 'google/gemini-2.5-pro',
    'dataset_id': TRANSCRIBED_ID,
    'split': 'test',
    'num_samples': len(ds),
    'normalizer': 'generic',
    'language': 'en',
    'push_results': True,
})
data = r.json()
job_id = data.get('job_id') or data.get('id')
print(f"  Job: {job_id}")

job_data = poll_eval_job(job_id, 'Gemini Pro')
if not job_data:
    raise SystemExit("Gemini eval failed")

results_ds = load_dataset(job_data['result']['pushed_dataset_id'], split='test', token=HF_TOKEN)
results_ds = results_ds.cast_column('audio', results_ds.features['audio'].__class__(decode=False))

cer_col = next(c for c in results_ds.column_names if 'cer' in c.lower() and 'entity' not in c.lower())
ref_col = 'reference' if 'reference' in results_ds.column_names else 'text'
cer_values = [float(row[cer_col]) for row in results_ds]
text_to_cer = {row[ref_col]: float(row[cer_col]) for row in results_ds}

print(f"\nCER — min:{min(cer_values):.3f} mean:{np.mean(cer_values):.3f} "
      f"median:{np.median(cer_values):.3f} max:{max(cer_values):.3f}")

t_otsu = otsu_threshold(cer_values)
CER_FLOOR = 0.05
print(f"Otsu threshold: {t_otsu:.3f}")

filtered = []
for row in ds:
    cer = text_to_cer.get(row['text'])
    if cer is not None and CER_FLOOR <= cer <= t_otsu:
        d = dict(row); d['gemini_cer'] = cer
        filtered.append(d)
filtered.sort(key=lambda r: r['gemini_cer'], reverse=True)
print(f"Rows passing filter (CER {CER_FLOOR}–{t_otsu:.3f}): {len(filtered)}")

for name, rows in [('multimed-sentences-otsu', filtered),
                   ('multimed-sentences-inspect', filtered[:20])]:
    out_ds = Dataset.from_list(rows)
    if 'audio' in out_ds.column_names:
        out_ds = out_ds.cast_column('audio', Audio(sampling_rate=16000))
    api.create_repo(f'ronanarraig/{name}', repo_type='dataset', private=True, exist_ok=True)
    out_ds.push_to_hub(f'ronanarraig/{name}', split='test', token=HF_TOKEN, private=True)
    print(f"Pushed ronanarraig/{name} ({len(out_ds)} rows)")

print("\nTop 10 hardest rows:")
for row in filtered[:10]:
    print(f"  {row['gemini_cer']:.3f}  {row['text'][:80]}")

print("\nDone.")
