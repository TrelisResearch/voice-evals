#!/usr/bin/env python3
"""
Phase 1D: EKA difficulty filter
1. Push 920 high-density EKA rows to HF (ronanarraig/eka-hard-candidates)
2. Run 3 eval jobs (Whisper v3, Canary 1B v2, Voxtral Mini) with push_results=True
3. Download per-sample CER from result datasets
4. Rank by median CER → top-100 → export to review dir
"""
import os, json, time, requests
from dotenv import load_dotenv

load_dotenv('/home/claude/TR/.env')
HF_TOKEN = os.environ['HF_TOKEN']
API_KEY = os.environ['TRELIS_STUDIO_API_KEY']
BASE = 'https://studio.trelis.com/api/v1'
H = {'Authorization': f'Bearer {API_KEY}'}

ROWS_JSON   = '/home/claude/TR/voice-evals/tools/review/data-eka/rows.json'
AUDIO_DIR   = '/home/claude/TR/voice-evals/tools/review/data-eka/audio'
REVIEW_DIR  = '/home/claude/TR/voice-evals/tools/review/data-eka'
HF_DATASET  = 'ronanarraig/eka-hard-candidates'
EVAL_MODELS = [
    'openai/whisper-large-v3',
    'nvidia/canary-1b-v2',
    'mistralai/Voxtral-Mini-3B-2507',
]

t0 = time.time()
def tick(label):
    print(f'  [{(time.time()-t0)/60:.1f}min] {label}', flush=True)

def poll_job(job_id, label='', interval=15, timeout=1800):
    for _ in range(timeout // interval):
        r = requests.get(f'{BASE}/data-prep/jobs/{job_id}', headers=H).json()
        status = r.get('status', '?')
        print(f'  {label}: {status}', flush=True)
        if status in ('completed', 'failed', 'error'): return r
        time.sleep(interval)
    return {'status': 'timeout'}

def poll_eval(job_id, label='', interval=20, timeout=3600):
    for _ in range(timeout // interval):
        r = requests.get(f'{BASE}/evaluation/jobs/{job_id}', headers=H).json()
        status = r.get('status', '?')
        print(f'  {label}: {status}', flush=True)
        if status in ('completed', 'failed', 'error'): return r
        time.sleep(interval)
    return {'status': 'timeout'}

# ── Step 1: Push EKA rows to HuggingFace ───────────────────────────────────
print('\n=== Step 1: Push EKA 920 rows to HF ===')
rows = json.load(open(ROWS_JSON))
print(f'  {len(rows)} rows loaded')

from datasets import Dataset, Audio as AudioFeature
import soundfile as sf
import numpy as np

def load_audio(row_id):
    path = f'{AUDIO_DIR}/{row_id}.wav'
    data, sr = sf.read(path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    return {'array': data.astype(np.float32), 'sampling_rate': sr, 'path': path}

tick('loading audio...')
records = []
for r in rows:
    records.append({
        'id': str(r['id']),
        'audio': load_audio(r['id']),
        'text': r['transcript'],
        'whisper_sentence': r.get('whisper_sentence', ''),
        'duration': r.get('duration', 0.0),
        'medical_density': r.get('medical_density', ''),
        'entities': json.dumps(r.get('entities', [])),
    })
tick(f'loaded {len(records)} audio files')

ds = Dataset.from_list(records).cast_column('audio', AudioFeature())
tick('pushing to HF...')
ds.push_to_hub(HF_DATASET, split='test', private=True, token=HF_TOKEN)
tick(f'pushed to {HF_DATASET}')

# ── Step 2: Submit 3 eval jobs ──────────────────────────────────────────────
print('\n=== Step 2: Submit eval jobs ===')
job_ids = {}
for model_id in EVAL_MODELS:
    r = requests.post(f'{BASE}/evaluation/jobs', headers=H, json={
        'model_id': model_id,
        'dataset_id': HF_DATASET,
        'split': 'test',
        'num_samples': len(rows),
        'push_results': True,
        'language': 'english',
        'normalizer': 'generic',
    }).json()
    job_id = r.get('job_id') or r.get('id')
    print(f'  {model_id}: job_id={job_id}  raw={r}')
    if job_id:
        job_ids[model_id] = job_id
tick(f'submitted {len(job_ids)} eval jobs')

# ── Step 3: Poll eval jobs ──────────────────────────────────────────────────
print('\n=== Step 3: Poll eval jobs ===')
results = {}
for model_id, job_id in job_ids.items():
    print(f'\n  Polling {model_id} ({job_id})')
    r = poll_eval(job_id, model_id, interval=20, timeout=3600)
    results[model_id] = r
    print(f'  Final: {r.get("status")} — WER={r.get("wer"):.3f}, CER={r.get("cer"):.3f}' if r.get('wer') else f'  Final: {r}')
    # Save result dataset ID for per-sample download
    result_ds = r.get('results_dataset_id') or r.get('output_dataset_id') or r.get('dataset_id')
    print(f'  Results dataset: {result_ds}')

tick('eval jobs complete')
print(json.dumps({m: {'status': r.get('status'), 'wer': r.get('wer'), 'cer': r.get('cer'),
                       'results_dataset': r.get('results_dataset_id') or r.get('output_dataset_id')}
                  for m, r in results.items()}, indent=2))

# ── Step 4: Download per-sample CER and rank ────────────────────────────────
print('\n=== Step 4: Per-sample CER → rank → top-100 ===')
from datasets import load_dataset

per_sample_cer = {}  # id → list of CER values
for model_id, result in results.items():
    result_ds = result.get('results_dataset_id') or result.get('output_dataset_id') or result.get('dataset_id')
    if not result_ds:
        print(f'  {model_id}: no results dataset, skipping')
        continue
    try:
        print(f'  Loading {result_ds}...')
        rds = load_dataset(result_ds, split='test', token=HF_TOKEN)
        print(f'  {len(rds)} rows, columns: {rds.column_names}')
        for row in rds:
            row_id = str(row.get('id', ''))
            cer = row.get('cer')
            if row_id and cer is not None:
                per_sample_cer.setdefault(row_id, []).append(float(cer))
    except Exception as e:
        print(f'  {model_id}: failed to load results — {e}')

tick(f'loaded per-sample CER for {len(per_sample_cer)} rows')

if per_sample_cer:
    import statistics
    # Compute median CER per row
    row_lookup = {str(r['id']): r for r in rows}
    ranked = []
    for row_id, cers in per_sample_cer.items():
        median_cer = statistics.median(cers)
        ranked.append((median_cer, row_id, cers))
    ranked.sort(reverse=True)

    print(f'\n  CER distribution:')
    print(f'  Top 10: {[f"{c:.3f}" for c,_,_ in ranked[:10]]}')
    print(f'  Bottom 10: {[f"{c:.3f}" for c,_,_ in ranked[-10:]]}')

    # Top-100
    top100_ids = [row_id for _, row_id, _ in ranked[:100]]
    top100_rows = []
    for _, row_id, cers in ranked[:100]:
        r = row_lookup.get(row_id, {})
        r = dict(r)
        r['median_cer'] = statistics.median(cers)
        r['model_cers'] = {m: cers[i] for i, m in enumerate(EVAL_MODELS) if i < len(cers)}
        top100_rows.append(r)

    # Save updated rows.json with top-100 only
    out_path = f'{REVIEW_DIR}/rows.json'
    json.dump(top100_rows, open(out_path, 'w'), indent=2)
    tick(f'saved top-100 to {out_path}')
    print(f'  Top-100 median CER range: {top100_rows[-1]["median_cer"]:.3f} – {top100_rows[0]["median_cer"]:.3f}')
else:
    print('  WARNING: no per-sample CER data — saving raw results for manual inspection')
    json.dump({'results': results, 'note': 'per-sample CER not available from eval jobs'},
              open(f'{REVIEW_DIR}/eval_results.json', 'w'), indent=2)

tick('DONE')
