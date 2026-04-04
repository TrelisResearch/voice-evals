#!/usr/bin/env python3
"""
Mini pipeline test — 5 rows from data-eka through:
1. Push 5 rows to ronanarraig/eka-mini-test (HF, private)
2. Submit 1 eval job (whisper-large-v3, push_results=True)
3. Poll until complete
4. Check result dataset for per-sample CER
5. Print results

Run from: /home/claude/TR/voice-evals
"""
import os, json, time, statistics
import soundfile as sf
import numpy as np
import requests
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')

HF_TOKEN  = os.environ['HF_TOKEN']
API_KEY   = os.environ['TRELIS_STUDIO_API_KEY']
BASE      = 'https://studio.trelis.com/api/v1'
H         = {'Authorization': f'Bearer {API_KEY}'}

ROWS_JSON  = 'tools/review/data-eka/rows.json'
AUDIO_DIR  = 'tools/review/data-eka/audio'
HF_DATASET = 'ronanarraig/eka-mini-test'
N          = 5

t0 = time.time()
def elapsed(): return f'{(time.time()-t0):.0f}s'

# ── Step 1: Push 5 rows to HF ──────────────────────────────────────
print(f'\n=== Step 1: Push {N} rows to {HF_DATASET} ===')
all_rows = json.load(open(ROWS_JSON))
rows = all_rows[:N]
print(f'  rows: {[r["id"] for r in rows]}')
print(f'  texts: {[r["transcript"][:50] for r in rows]}')

import pyarrow as pa
from datasets import Dataset, Audio

audio_bytes_list, texts, ids, durations = [], [], [], []
for r in rows:
    path = f'{AUDIO_DIR}/{r["id"]}.wav'
    audio_bytes_list.append(open(path, 'rb').read())
    texts.append(r['transcript'])
    ids.append(str(r['id']))
    durations.append(float(r.get('duration', 0)))

# Build Arrow table with bytes/path struct, then cast_column (no torch needed)
audio_struct = pa.array(
    [{'bytes': b, 'path': p} for b, p in zip(audio_bytes_list, ids)],
    type=pa.struct([pa.field('bytes', pa.binary()), pa.field('path', pa.string())])
)
table = pa.table({
    'id': pa.array(ids),
    'audio': audio_struct,
    'text': pa.array(texts),
    'duration': pa.array(durations, type=pa.float32()),
})
# cast_column uses cast_storage which handles bytes struct without torch
ds = Dataset(table).cast_column('audio', Audio(sampling_rate=16000, decode=False))
print(f'  features: {ds.features}')
ds.push_to_hub(HF_DATASET, split='test', private=True, token=HF_TOKEN)
print(f'  [{elapsed()}] pushed')
print(f'  [{elapsed()}] pushed')

# ── Step 2: Submit eval job ────────────────────────────────────────
print(f'\n=== Step 2: Submit eval job ===')
r = requests.post(f'{BASE}/evaluation/jobs', headers=H, json={
    'model_id': 'openai/whisper-large-v3',
    'dataset_id': HF_DATASET,
    'split': 'test',
    'num_samples': N,
    'push_results': True,
    'language': 'english',
    'normalizer': 'generic',
})
print(f'  status={r.status_code}')
print(f'  response={json.dumps(r.json(), indent=2)}')
job_id = r.json().get('job_id') or r.json().get('id')
print(f'  job_id={job_id}')

# ── Step 3: Poll ───────────────────────────────────────────────────
print(f'\n=== Step 3: Poll eval job ===')
for i in range(120):  # 40 min max
    r = requests.get(f'{BASE}/evaluation/jobs/{job_id}', headers=H)
    data = r.json()
    status = data.get('status', '?')
    print(f'  [{elapsed()}] {status}', flush=True)
    if status == 'completed':
        print(f'  WER={data.get("wer")}, CER={data.get("cer")}')
        print(f'  Full response keys: {list(data.keys())}')
        print(f'  Full response: {json.dumps({k:v for k,v in data.items() if k != "samples"}, indent=2)}')
        # Find results dataset
        result_ds = (data.get('results_dataset_id') or
                     data.get('output_dataset_id') or
                     data.get('dataset_id') or
                     data.get('result_dataset_id'))
        print(f'  results_dataset = {result_ds}')
        break
    elif status in ('failed', 'error'):
        print(f'  FAILED: {data}')
        break
    time.sleep(20)

# ── Step 4: Download per-sample CER ───────────────────────────────
print(f'\n=== Step 4: Per-sample results ===')
if result_ds:
    from datasets import load_dataset
    try:
        rds = load_dataset(result_ds, split='test', token=HF_TOKEN)
        print(f'  {len(rds)} rows, columns: {rds.column_names}')
        for row in rds:
            print(f'  id={row.get("id")} cer={row.get("cer")} wer={row.get("wer")} text={str(row.get("text",""))[:50]}')
    except Exception as e:
        print(f'  load failed: {e}')
else:
    print('  no result_ds found — check full response above')

print(f'\n[{elapsed()}] DONE')
