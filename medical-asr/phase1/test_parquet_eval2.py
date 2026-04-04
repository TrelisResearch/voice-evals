#!/usr/bin/env python3
"""
Test approach 2: load matching EKA rows from HF (decode=False, no torch),
replace text with Gemini transcripts, push to HF, run eval.
"""
import os, json, time
import requests
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')
from datasets import load_dataset, Audio, Dataset

HF_TOKEN = os.environ['HF_TOKEN']
API_KEY  = os.environ['TRELIS_STUDIO_API_KEY']
BASE     = 'https://studio.trelis.com/api/v1'
H        = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}

ROWS_JSON  = 'tools/review/data-eka/rows.json'
HF_DATASET = 'ronanarraig/eka-mini-test-v2'
N = 5

t0 = time.time()
def elapsed(): return f'{(time.time()-t0):.0f}s'

# ── Step 1: Load our rows.json ───────────────────────────────────────
print(f'\n=== Step 1: Load our rows ===')
all_rows = json.load(open(ROWS_JSON))
rows = all_rows[:N]
source_to_gemini = {r['source_file']: r['transcript'] for r in rows}
print(f'  source files: {list(source_to_gemini.keys())}')

# ── Step 2: Load matching EKA rows from HF (decode=False) ───────────
print(f'\n=== Step 2: Load EKA from HF (decode=False) ===')
eka = load_dataset('ekacare/eka-medical-asr-evaluation-dataset', 'en',
                   split='test', token=HF_TOKEN)
eka = eka.cast_column('audio', Audio(decode=False))
print(f'  [{elapsed()}] loaded {len(eka)} rows, features: {eka.features}')

# Filter to matching rows
def matches(row):
    path = (row['audio'] or {}).get('path', '')
    return path in source_to_gemini

matched = eka.filter(matches)
print(f'  [{elapsed()}] {len(matched)} matching rows')
print(f'  matched paths: {[(r["audio"] or {}).get("path") for r in matched]}')

# Replace text with Gemini transcripts
def replace_text(row):
    path = (row['audio'] or {}).get('path', '')
    row['text'] = source_to_gemini.get(path, row['text'])
    return row

matched = matched.map(replace_text)
# Keep only id, audio, text
matched = matched.select_columns(['audio', 'text'])
print(f'  [{elapsed()}] text replaced, features: {matched.features}')
print(f'  sample texts: {[r["text"][:50] for r in matched]}')

# ── Step 3: Push to HF ───────────────────────────────────────────────
print(f'\n=== Step 3: Push to HF ===')
matched.push_to_hub(HF_DATASET, split='test', private=True, token=HF_TOKEN)
print(f'  [{elapsed()}] pushed to {HF_DATASET}')

# ── Step 4: Submit eval job ──────────────────────────────────────────
print(f'\n=== Step 4: Submit eval job ===')
r = requests.post(f'{BASE}/evaluation/jobs', headers=H, json={
    'model_id': 'openai/whisper-large-v3',
    'dataset_id': HF_DATASET,
    'split': 'test',
    'num_samples': N,
    'push_results': True,
    'language': 'english',
    'normalizer': 'generic',
})
print(f'  status={r.status_code} response={r.json()}')
job_id = r.json().get('job_id')

# ── Step 5: Poll ─────────────────────────────────────────────────────
print(f'\n=== Step 5: Poll {job_id} ===')
for _ in range(120):
    r = requests.get(f'{BASE}/evaluation/jobs/{job_id}', headers=H)
    data = r.json()
    status = data.get('status', '?')
    print(f'  [{elapsed()}] {status}', flush=True)
    if status == 'completed':
        result = data.get('result', {})
        print(f'  WER={result.get("wer"):.3f} CER={result.get("cer"):.3f} n={result.get("samples_evaluated")}')
        result_ds = result.get('pushed_dataset_id') or data.get('results_dataset_id')
        print(f'  results_dataset={result_ds}')
        if result_ds:
            rds = load_dataset(result_ds, split='test', token=HF_TOKEN)
            rds = rds.cast_column('audio', Audio(decode=False)) if 'audio' in rds.column_names else rds
            print(f'  {len(rds)} result rows, cols={rds.column_names}')
            for row in rds:
                print(f'    cer={row.get("cer"):.3f} pred={str(row.get("prediction",""))[:60]}')
        break
    elif status in ('failed', 'error'):
        print(f'  FAILED\n  logs: {data.get("logs","")[-600:]}')
        break
    time.sleep(20)

print(f'\n[{elapsed()}] DONE')
