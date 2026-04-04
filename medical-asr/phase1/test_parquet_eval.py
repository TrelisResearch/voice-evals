#!/usr/bin/env python3
"""
Test the new parquet_url eval workflow with 5 rows.
1. Build parquet (PyArrow, no torch)
2. Upload to Studio file store → get presigned PUT URL
3. PUT parquet to S3
4. Submit eval job with parquet_url
5. Poll + check per-sample results
"""
import os, json, time, tempfile
import soundfile as sf
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import requests
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')

HF_TOKEN = os.environ['HF_TOKEN']
API_KEY  = os.environ['TRELIS_STUDIO_API_KEY']
BASE     = 'https://studio.trelis.com/api/v1'
H        = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}

ROWS_JSON = 'tools/review/data-eka/rows.json'
AUDIO_DIR = 'tools/review/data-eka/audio'
N = 5

t0 = time.time()
def elapsed(): return f'{(time.time()-t0):.0f}s'

# ── Step 1: Build parquet ────────────────────────────────────────────
print(f'\n=== Step 1: Build parquet ({N} rows) ===')
all_rows = json.load(open(ROWS_JSON))
rows = all_rows[:N]

audio_bytes_list, texts, ids = [], [], []
for r in rows:
    path = f'{AUDIO_DIR}/{r["id"]}.wav'
    audio_bytes_list.append(open(path, 'rb').read())
    texts.append(r['transcript'])
    ids.append(str(r['id']))

table = pa.table({
    'id': pa.array(ids),
    'audio': pa.array(audio_bytes_list, type=pa.binary()),  # flat binary, not struct
    'text': pa.array(texts),
})

with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
    pq.write_table(table, f.name)
    parquet_path = f.name
    parquet_size = os.path.getsize(f.name)

print(f'  parquet: {parquet_path} ({parquet_size/1024:.0f} KB)')
print(f'  texts: {[t[:50] for t in texts]}')

# ── Step 2: Upload parquet to HF Hub as raw file ────────────────────
print(f'\n=== Step 2: Upload parquet to HF Hub (raw file) ===')
from huggingface_hub import HfApi
api = HfApi(token=HF_TOKEN)
PARQUET_REPO = 'ronanarraig/eval-parquets'
api.create_repo(PARQUET_REPO, repo_type='dataset', private=True, exist_ok=True)
api.upload_file(
    path_or_fileobj=parquet_path,
    path_in_repo='eka_mini_test_v2.parquet',
    repo_id=PARQUET_REPO,
    repo_type='dataset',
    token=HF_TOKEN,
)
# HF private download URL with token
parquet_url = f'https://huggingface.co/datasets/{PARQUET_REPO}/resolve/main/eka_mini_test_v2.parquet?token={HF_TOKEN}'
print(f'  [{elapsed()}] uploaded')
print(f'  URL (token hidden): https://huggingface.co/datasets/{PARQUET_REPO}/resolve/main/eka_mini_test_v2.parquet?token=...')

# Verify it's accessible
head_r = requests.head(parquet_url)
print(f'  HEAD status={head_r.status_code}')

# ── Step 3: Submit eval job with parquet_url ─────────────────────────
print(f'\n=== Step 3: Submit eval job with parquet_url ===')
r = requests.post(f'{BASE}/evaluation/jobs', headers=H, json={
    'model_id': 'openai/whisper-large-v3',
    'parquet_url': parquet_url,
    'num_samples': N,
    'push_results': True,
    'language': 'english',
    'normalizer': 'generic',
})
print(f'  status={r.status_code}')
resp = r.json()
print(f'  response={json.dumps(resp, indent=2)}')
job_id = resp.get('job_id') or resp.get('id')

if not job_id:
    print('  ERROR: no job_id in response')
else:
    # ── Step 4: Poll ─────────────────────────────────────────────────
    print(f'\n=== Step 4: Poll eval job {job_id} ===')
    for i in range(60):
        r = requests.get(f'{BASE}/evaluation/jobs/{job_id}', headers=H)
        data = r.json()
        status = data.get('status', '?')
        print(f'  [{elapsed()}] {status}', flush=True)
        if status == 'completed':
            print(f'  Full response: {json.dumps({k:v for k,v in data.items() if k != "samples"}, indent=2)}')
            print(f'  WER={data.get("wer")}, CER={data.get("cer")}')
            result_ds = (data.get('results_dataset_id') or data.get('output_dataset_id'))
            print(f'  results_dataset={result_ds}')
            if result_ds:
                from datasets import load_dataset
                rds = load_dataset(result_ds, split='test', token=HF_TOKEN)
                rds_plain = rds.cast_column('audio', rds.features['audio'].__class__(decode=False)) if 'audio' in rds.column_names else rds
                print(f'  {len(rds_plain)} result rows, columns={rds_plain.column_names}')
                for row in rds_plain:
                    print(f'    id={row.get("id")} cer={row.get("cer"):.3f} pred={str(row.get("prediction",""))[:60]}')
            break
        elif status in ('failed', 'error'):
            print(f'  FAILED: {json.dumps(data.get("error","")[:300])}')
            print(f'  logs: {data.get("logs","")[-500:]}')
            break
        time.sleep(20)

print(f'\n[{elapsed()}] DONE')
