#!/usr/bin/env python3
"""Quick test: can Studio import leduckhai/MultiMed via from-hf-dataset?
This was broken (bug 2bef1bbf) - aborted after ~2h at 1400/4751 rows.
Test with max_rows=10 to verify the fix."""
import os, time, requests
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')

API_KEY = os.environ['TRELIS_STUDIO_API_KEY']
BASE = 'https://studio.trelis.com/api/v1'
H = {'Authorization': f'Bearer {API_KEY}'}

t0 = time.time()

print('=== HF import test: leduckhai/MultiMed (max_rows=10) ===')
r = requests.post(f'{BASE}/file-stores/from-hf-dataset', headers=H, json={
    'dataset_id': 'leduckhai/MultiMed',
    'split': 'test',
    'config': 'English',
    'max_rows': 10,
    'name': 'multimed-import-test-10rows',
    'audio_column': 'audio',
    'text_column': 'text',
}).json()
print(f'Response: {r}')

job_id = r.get('job_id') or r.get('id')
file_store_id = r.get('file_store_id')
print(f'job_id={job_id}  file_store_id={file_store_id}')

if not job_id:
    print('ERROR: no job_id returned')
    raise SystemExit(1)

# Poll
print('\nPolling...')
for i in range(60):
    time.sleep(10)
    r2 = requests.get(f'{BASE}/data-prep/jobs/{job_id}', headers=H).json()
    status = r2.get('status', '?')
    elapsed = (time.time() - t0) / 60
    print(f'  [{elapsed:.1f}min] {status}  {r2.get("message","") or ""}', flush=True)
    if status in ('completed', 'failed', 'error'):
        print(f'\nFinal: {r2}')
        break
else:
    print('Timed out after 10min')
