#!/usr/bin/env python3
"""
Phase 2 Step 4: Difficulty calibration + push final medical-terms-public.
1. Run 3 eval jobs (Whisper large-v3, Canary 1B v2, Voxtral Mini) on medical-terms-tts-raw
2. Download per-sample CER from result datasets
3. Rank by median CER → top-50
4. Push to ronanarraig/medical-terms-public (test split)
"""
import os, json, time, statistics
import pyarrow as pa
import requests
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')
from datasets import load_dataset, Dataset, Audio

HF_TOKEN = os.environ['HF_TOKEN']
API_KEY  = os.environ['TRELIS_STUDIO_API_KEY']
BASE     = 'https://studio.trelis.com/api/v1'
H        = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}

HF_INPUT    = 'ronanarraig/medical-terms-tts-raw'
HF_OUTPUT   = 'ronanarraig/medical-terms-public'
JOBS_FILE   = 'medical-asr/phase2/tmp/eval_jobs.json'
SENTENCES_FILE = 'medical-asr/phase2/tmp/sentences_with_audio.json'

EVAL_MODELS = [
    'openai/whisper-large-v3',
    'nvidia/canary-1b-v2',
    'mistralai/Voxtral-Mini-3B-2507',
]

t0 = time.time()
def elapsed(): return f'{(time.time()-t0)/60:.1f}min'
def tick(msg): print(f'  [{elapsed()}] {msg}', flush=True)

# ── Step 1: Submit eval jobs ────────────────────────────────────────
if os.path.exists(JOBS_FILE):
    job_ids = json.load(open(JOBS_FILE))
    print(f'Resuming from checkpoint: {job_ids}')
else:
    print('=== Step 1: Submit eval jobs ===')
    job_ids = {}
    for model_id in EVAL_MODELS:
        r = requests.post(f'{BASE}/evaluation/jobs', headers=H, json={
            'model_id': model_id,
            'dataset_id': HF_INPUT,
            'split': 'test',
            'num_samples': 52,
            'push_results': True,
            'language': 'english',
            'normalizer': 'generic',
        })
        resp = r.json()
        job_id = resp.get('job_id') or resp.get('id')
        print(f'  {model_id}: job_id={job_id} (status={r.status_code})')
        if job_id:
            job_ids[model_id] = job_id
    json.dump(job_ids, open(JOBS_FILE, 'w'), indent=2)
    tick(f'Submitted {len(job_ids)} eval jobs')

# ── Step 2: Poll eval jobs ─────────────────────────────────────────
print(f'\n=== Step 2: Poll eval jobs ===')
results = {}
for model_id, job_id in job_ids.items():
    print(f'\n  Polling {model_id} ({job_id})...')
    for _ in range(240):
        r = requests.get(f'{BASE}/evaluation/jobs/{job_id}', headers=H)
        data = r.json()
        status = data.get('status', '?')
        if status == 'completed':
            res = data.get('result', {})
            print(f'  → WER={res.get("wer"):.3f} CER={res.get("cer"):.3f} n={res.get("samples_evaluated")}')
            pushed = res.get('pushed_dataset_id')
            print(f'  → pushed_dataset_id={pushed}')
            results[model_id] = {'result': res, 'pushed_dataset_id': pushed}
            break
        elif status in ('failed', 'error'):
            print(f'  → FAILED: {data.get("error","")[:200]}')
            results[model_id] = {'status': 'failed'}
            break
        else:
            print(f'  {model_id}: {status}', flush=True)
            time.sleep(30)

tick('All eval jobs done')

# ── Step 3: Download per-sample CER ─────────────────────────────────
print(f'\n=== Step 3: Download per-sample CER ===')
per_sample_cer = {}  # index → {model: cer}
index_to_ref = {}    # index → reference text

for model_id, res in results.items():
    pushed = res.get('pushed_dataset_id')
    if not pushed:
        print(f'  {model_id}: no pushed dataset, skipping')
        continue
    try:
        print(f'  Loading {pushed}...')
        rds = load_dataset(pushed, split='test', token=HF_TOKEN)
        rds = rds.cast_column('audio', Audio(decode=False)) if 'audio' in rds.column_names else rds
        print(f'  {len(rds)} rows, cols: {rds.column_names}')
        for i, row in enumerate(rds):
            cer = row.get('cer')
            if cer is not None:
                if i not in per_sample_cer:
                    per_sample_cer[i] = {}
                per_sample_cer[i][model_id] = float(cer)
            if model_id == 'openai/whisper-large-v3':
                index_to_ref[i] = row.get('reference', '')
    except Exception as e:
        print(f'  FAILED: {e}')

tick(f'Loaded per-sample CER for {len(per_sample_cer)} rows')

# ── Step 4: Rank and select top-50 ──────────────────────────────────
print(f'\n=== Step 4: Rank by median CER ===')
ranked = []
for idx, model_cers in per_sample_cer.items():
    cers = list(model_cers.values())
    median_cer = statistics.median(cers)
    ranked.append((median_cer, idx, model_cers))
ranked.sort(reverse=True)

print(f'CER range: {ranked[-1][0]:.3f} – {ranked[0][0]:.3f}')
print(f'Top 10: {[f"{c:.3f}" for c,_,_ in ranked[:10]]}')

# Load sentences metadata
swa = json.load(open(SENTENCES_FILE))
idx_to_meta = {r['idx']: r for r in swa}

# Build top-50 rows
top50 = []
for rank, (median_cer, idx, model_cers) in enumerate(ranked[:50]):
    meta = idx_to_meta.get(idx, {})
    audio_path = meta.get('audio_path', '')
    top50.append({
        'idx': idx,
        'text': meta.get('text', index_to_ref.get(idx, '')),
        'keyword': meta.get('keyword', ''),
        'category': meta.get('category', ''),
        'voice': meta.get('voice', ''),
        'median_cer': median_cer,
        'model_cers': model_cers,
        'difficulty_rank': rank + 1,
        'audio_path': audio_path,
    })

print(f'\nTop-50 CER range: {top50[-1]["median_cer"]:.3f} – {top50[0]["median_cer"]:.3f}')
print('\nSample rows:')
for r in top50[:5]:
    print(f'  [{r["difficulty_rank"]}] cer={r["median_cer"]:.3f} [{r["category"]}] {r["text"][:70]}')

# ── Step 5: Push final dataset ───────────────────────────────────────
print(f'\n=== Step 5: Push to {HF_OUTPUT} ===')
audio_bytes_list, texts, keywords, categories, voices = [], [], [], [], []
median_cers, diff_ranks, entities_list = [], [], []

for r in top50:
    ap = r['audio_path']
    if ap and os.path.exists(ap):
        audio_bytes_list.append(open(ap, 'rb').read())
    else:
        audio_bytes_list.append(b'')
    texts.append(r['text'])
    keywords.append(r['keyword'])
    categories.append(r['category'])
    voices.append(r['voice'])
    median_cers.append(float(r['median_cer']))
    diff_ranks.append(int(r['difficulty_rank']))

audio_col = pa.array(audio_bytes_list, type=pa.binary())
table = pa.table({
    'audio':           audio_col,
    'text':            pa.array(texts),
    'keyword':         pa.array(keywords),
    'category':        pa.array(categories),
    'voice':           pa.array(voices),
    'median_cer':      pa.array(median_cers, type=pa.float32()),
    'difficulty_rank': pa.array(diff_ranks, type=pa.int32()),
})
out_ds = Dataset(table).cast_column('audio', Audio(sampling_rate=24000))
print(f'  Dataset: {len(out_ds)} rows')
out_ds.push_to_hub(HF_OUTPUT, split='test', private=True, token=HF_TOKEN)
tick(f'Pushed to {HF_OUTPUT}')

# Save results
json.dump(top50, open('medical-asr/phase2/tmp/top50.json', 'w'), indent=2)
print('\nDONE')
