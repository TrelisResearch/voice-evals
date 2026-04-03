#!/usr/bin/env python3
"""Resume MultiMed pipeline from draft-transcribe onwards."""
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

STORE_ID = 'd1cbb62a-76d3-4868-9d6e-a7fba46bf86b'

def poll_store(label, interval=20):
    while True:
        r = requests.get(f'{BASE}/file-stores/{STORE_ID}', headers=HEADERS)
        data = r.json()
        # Check for any job status fields
        statuses = {k: v for k, v in data.items() if 'status' in k.lower()}
        print(f"  {label}: {statuses or data.get('status', 'unknown')}... waiting {interval}s")
        # If no pending jobs indicated, assume done
        all_done = all(v in ('completed', 'done', 'ready', None)
                       for v in statuses.values()) if statuses else True
        if all_done and statuses:
            print(f"  {label}: done")
            return data
        elif not statuses:
            # No status fields at all — try checking for output
            return data
        time.sleep(interval)

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

# ── Step 3: Draft-transcribe ──────────────────────────────────────
print("=== Step 3: Draft-transcribe with Whisper large-v3 ===")
r = requests.post(f'{BASE}/file-stores/{STORE_ID}/draft-transcribe', headers=HEADERS, json={
    'model_id': 'openai/whisper-large-v3',
    'language': 'en',
})
resp = r.json()
print(f"  Response: {resp}")

# Poll for completion
print("  Polling for completion...")
for attempt in range(120):  # max 40 min
    time.sleep(20)
    r = requests.get(f'{BASE}/file-stores/{STORE_ID}', headers=HEADERS)
    data = r.json()
    print(f"  Store state: { {k:v for k,v in data.items() if k not in ('files',)} }")
    # Check if transcription completed
    transcript_status = data.get('draft_transcribe_status') or data.get('transcription_status')
    if transcript_status in ('completed', 'done'):
        print("  Draft-transcribe complete")
        break
    elif transcript_status in ('failed', 'error'):
        print(f"  Draft-transcribe failed: {data}")
        break
    # If no job status field but store looks ready, proceed
    if attempt > 3 and not transcript_status:
        print("  No transcription status field — checking if files have transcripts...")
        break

# ── Step 4: Process ───────────────────────────────────────────────
print("\n=== Step 4: Process → push to HF ===")
r = requests.post(f'{BASE}/file-stores/{STORE_ID}/process', headers=HEADERS, json={
    'output_dataset_name': 'multimed-sentences-transcribed',
    'output_org': 'ronanarraig',
    'hf_token': HF_TOKEN,
    'split_option': 'test_only',
    'language': 'eng',
    'enable_quality_checks': False,
})
resp = r.json()
print(f"  Response: {resp}")

# Poll for process completion
print("  Polling for completion...")
for attempt in range(180):  # max 60 min
    time.sleep(20)
    r = requests.get(f'{BASE}/file-stores/{STORE_ID}', headers=HEADERS)
    data = r.json()
    process_status = data.get('process_status') or data.get('processing_status')
    pushed_id = data.get('output_dataset_id') or data.get('pushed_dataset_id')
    print(f"  process_status={process_status} pushed_id={pushed_id}")
    if process_status in ('completed', 'done') or pushed_id:
        print("  Process complete")
        break
    elif process_status in ('failed', 'error'):
        print(f"  Process failed: {data}")
        raise SystemExit("Process failed")

transcribed_id = pushed_id or 'ronanarraig/multimed-sentences-transcribed'
print(f"  Dataset: {transcribed_id}")
with open(TMP / 'multimed_transcribed_id.json', 'w') as f:
    json.dump({'dataset_id': transcribed_id}, f)

# ── Step 5: Gemini Pro eval + Otsu ───────────────────────────────
print("\n=== Step 5: Gemini Pro eval + Otsu CER filter ===")
transcribed_ds = load_dataset(transcribed_id, split='test', token=HF_TOKEN)
transcribed_ds = transcribed_ds.cast_column('audio', transcribed_ds.features['audio'].__class__(decode=False))
n_rows = len(transcribed_ds)
print(f"  {n_rows} rows in transcribed dataset")
print(f"  Columns: {transcribed_ds.column_names}")

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
