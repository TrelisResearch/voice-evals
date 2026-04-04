#!/usr/bin/env python3
"""
Phase 1D: EKA difficulty filter
1. Push 920 high-density EKA rows to HF (ronanarraig/eka-hard-candidates)
2. Run 3 eval jobs (Whisper v3, Canary 1B v2, Voxtral Mini) with push_results=True
3. Download per-sample CER from result datasets (by index order)
4. Rank by median CER → top-100 → export updated rows.json

Checkpoint: saves job IDs to tmp/diff_filter_jobs.json after submission.
Re-run will skip push+submit and go straight to polling if jobs already submitted.

Run from: /home/claude/TR/voice-evals
"""
import os, json, time, statistics, tempfile
import soundfile as sf
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')
from datasets import load_dataset, Dataset, Audio
from huggingface_hub import HfApi

HF_TOKEN   = os.environ['HF_TOKEN']
API_KEY    = os.environ['TRELIS_STUDIO_API_KEY']
BASE       = 'https://studio.trelis.com/api/v1'
H          = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}

ROWS_JSON   = 'tools/review/data-eka/rows.json'
AUDIO_DIR   = 'tools/review/data-eka/audio'
REVIEW_DIR  = 'tools/review/data-eka'
HF_DATASET  = 'ronanarraig/eka-hard-candidates'
JOBS_FILE   = 'medical-asr/phase1/tmp/diff_filter_jobs.json'
EVAL_MODELS = [
    'openai/whisper-large-v3',
    'nvidia/canary-1b-v2',
    'mistralai/Voxtral-Mini-3B-2507',
]

t0 = time.time()
def elapsed(): return f'{(time.time()-t0)/60:.1f}min'
def tick(label): print(f'  [{elapsed()}] {label}', flush=True)

def poll_eval(job_id, label, interval=30, timeout=7200):
    for _ in range(timeout // interval):
        r = requests.get(f'{BASE}/evaluation/jobs/{job_id}', headers=H)
        data = r.json()
        status = data.get('status', '?')
        print(f'  {label}: {status}', flush=True)
        if status == 'completed':
            return data
        if status in ('failed', 'error'):
            print(f'  ERROR: {data.get("error","")[:200]}')
            return data
        time.sleep(interval)
    return {'status': 'timeout'}

# ── Check for checkpoint ────────────────────────────────────────────
if os.path.exists(JOBS_FILE):
    print(f'\n=== Resuming from checkpoint {JOBS_FILE} ===')
    job_ids = json.load(open(JOBS_FILE))
    print(f'  Jobs: {job_ids}')
    rows = json.load(open(ROWS_JSON))
    print(f'  {len(rows)} rows in rows.json')
else:
    # ── Step 1: Push 920 rows to HF ────────────────────────────────────
    print(f'\n=== Step 1: Push EKA rows to HF ({HF_DATASET}) ===')
    rows = json.load(open(ROWS_JSON))
    print(f'  {len(rows)} rows to push')

    audio_bytes_list, texts, ids, durations = [], [], [], []
    for r in rows:
        path = f'{AUDIO_DIR}/{r["id"]}.wav'
        audio_bytes_list.append(open(path, 'rb').read())
        texts.append(r['transcript'])
        ids.append(str(r['id']))
        durations.append(float(r.get('duration', 0)))
    tick(f'loaded {len(rows)} audio files')

    # Build Arrow table, cast to proper Audio feature (no torch needed)
    audio_col = pa.array(audio_bytes_list, type=pa.binary())
    table = pa.table({
        'id':       pa.array(ids),
        'audio':    audio_col,
        'text':     pa.array(texts),
        'duration': pa.array(durations, type=pa.float32()),
    })
    ds = Dataset(table).cast_column('audio', Audio(sampling_rate=16000))
    tick('dataset built, pushing to HF...')

    ds.push_to_hub(HF_DATASET, split='test', private=True, token=HF_TOKEN)
    tick(f'pushed {len(rows)} rows to {HF_DATASET}')

    # ── Step 2: Submit 3 eval jobs ──────────────────────────────────────
    print(f'\n=== Step 2: Submit eval jobs ===')
    job_ids = {}
    for model_id in EVAL_MODELS:
        r = requests.post(f'{BASE}/evaluation/jobs', headers=H, json={
            'model_id':     model_id,
            'dataset_id':   HF_DATASET,
            'split':        'test',
            'num_samples':  len(rows),
            'push_results': True,
            'language':     'english',
            'normalizer':   'generic',
        })
        resp = r.json()
        job_id = resp.get('job_id') or resp.get('id')
        print(f'  {model_id}: {job_id}  (status={r.status_code})')
        if job_id:
            job_ids[model_id] = job_id

    os.makedirs(os.path.dirname(JOBS_FILE), exist_ok=True)
    json.dump(job_ids, open(JOBS_FILE, 'w'), indent=2)
    tick(f'submitted {len(job_ids)} eval jobs, checkpoint saved')

# ── Step 3: Poll all eval jobs ─────────────────────────────────────────
print(f'\n=== Step 3: Poll eval jobs ===')
results = {}
for model_id, job_id in job_ids.items():
    print(f'\n  Polling {model_id} ({job_id})')
    r = poll_eval(job_id, model_id, interval=30, timeout=7200)
    results[model_id] = r
    res = r.get('result') or {}
    print(f'  → status={r.get("status")} WER={res.get("wer")} CER={res.get("cer")} n={res.get("samples_evaluated")}')
    print(f'  → pushed_dataset_id={res.get("pushed_dataset_id")}')

tick('all eval jobs done')

# ── Step 4: Download per-sample CER ────────────────────────────────────
print(f'\n=== Step 4: Download per-sample CER ===')
per_sample_cer = {}  # index → list of CER values

for model_id, result in results.items():
    res = result.get('result') or {}
    result_ds_id = res.get('pushed_dataset_id')
    if not result_ds_id:
        print(f'  {model_id}: no pushed_dataset_id, skipping')
        continue
    try:
        print(f'  Loading {result_ds_id}...')
        rds = load_dataset(result_ds_id, split='test', token=HF_TOKEN)
        rds = rds.cast_column('audio', Audio(decode=False)) if 'audio' in rds.column_names else rds
        print(f'  {len(rds)} rows, cols: {rds.column_names}')
        for i, row in enumerate(rds):
            cer = row.get('cer')
            if cer is not None:
                per_sample_cer.setdefault(i, []).append(float(cer))
    except Exception as e:
        print(f'  {model_id}: failed to load results — {e}')

tick(f'loaded per-sample CER for {len(per_sample_cer)} rows')

# ── Step 5: Rank by median CER → top-100 ───────────────────────────────
print(f'\n=== Step 5: Rank top-100 ===')
if not per_sample_cer:
    print('  WARNING: no per-sample CER, check eval results above')
else:
    ranked = []
    for idx, cers in per_sample_cer.items():
        median_cer = statistics.median(cers)
        ranked.append((median_cer, idx))
    ranked.sort(reverse=True)

    print(f'  CER range: {ranked[-1][0]:.3f} – {ranked[0][0]:.3f}')
    print(f'  Top 10: {[f"{c:.3f}" for c,_ in ranked[:10]]}')

    top100_indices = [idx for _, idx in ranked[:100]]
    top100_rows = []
    for rank, (median_cer, idx) in enumerate(ranked[:100]):
        r = dict(rows[idx])
        r['median_cer'] = median_cer
        r['model_cers'] = {m: per_sample_cer[idx][i] for i, m in enumerate(EVAL_MODELS) if idx in per_sample_cer and i < len(per_sample_cer[idx])}
        r['difficulty_rank'] = rank + 1
        top100_rows.append(r)

    out_path = f'{REVIEW_DIR}/rows.json'
    json.dump(top100_rows, open(out_path, 'w'), indent=2)
    tick(f'saved top-100 to {out_path}')
    print(f'  Top-100 median CER range: {top100_rows[-1]["median_cer"]:.3f} – {top100_rows[0]["median_cer"]:.3f}')

    # Clean up checkpoint
    os.remove(JOBS_FILE)
    print(f'  Checkpoint removed.')

tick('DONE')
