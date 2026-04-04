#!/usr/bin/env python3
"""Test full data prep pipeline using a HF dataset as input:
  from-hf-dataset → process → push to HF

Key findings so far:
- from-hf-dataset stores data as parquet (embedded audio), not raw audio files
- draft-transcribe expects raw audio files → skip it for HF-sourced datasets
- process should handle parquet format directly

Previously reported issues:
  - bug 2bef1bbf: HF import slow/fails (fixed: pass config='English' for multi-config datasets)
  - user report: using HF dataset as input to data prep was slow (testing now)

Uses 10 rows of leduckhai/MultiMed for speed.
"""
import os, time, requests
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')

API_KEY = os.environ['TRELIS_STUDIO_API_KEY']
BASE = 'https://studio.trelis.com/api/v1'
H = {'Authorization': f'Bearer {API_KEY}'}

t_global = time.time()

def elapsed():
    return (time.time() - t_global) / 60

def poll_job(job_id, label, interval=10, timeout=600):
    t0 = time.time()
    for _ in range(timeout // interval):
        r = requests.get(f'{BASE}/data-prep/jobs/{job_id}', headers=H).json()
        status = r.get('status', '?')
        print(f'  [{elapsed():.1f}min] {label}: {status}', flush=True)
        if status in ('completed', 'failed', 'error'):
            return r, time.time() - t0
        time.sleep(interval)
    return {'status': 'timeout'}, time.time() - t0

# ── Step 1: from-hf-dataset ──────────────────────────────────────────────────
print(f'\n=== Step 1: from-hf-dataset (10 rows) [{elapsed():.1f}min] ===')
t1 = time.time()
r = requests.post(f'{BASE}/file-stores/from-hf-dataset', headers=H, json={
    'dataset_id': 'leduckhai/MultiMed',
    'split': 'test',
    'config': 'English',
    'max_rows': 10,
    'name': 'multimed-pipe-test-10rows',
    'audio_column': 'audio',
    'text_column': 'text',
}).json()
print(f'  Response: {r}')
job_id = r.get('job_id') or r.get('id')
file_store_id = r.get('file_store_id')
if not job_id:
    print('ERROR: no job_id'); raise SystemExit(1)

import_result, import_time = poll_job(job_id, 'import')
print(f'  Import: {import_result.get("status")} in {import_time:.0f}s')
print(f'  Files: {import_result.get("result", {}).get("file_count")} ({import_result.get("result", {}).get("format")})')
if import_result.get('status') != 'completed':
    print('Import failed — stopping'); raise SystemExit(1)

# Check what's in the file store
files_r = requests.get(f'{BASE}/file-stores/{file_store_id}/files', headers=H).json()
print(f'  File store contents: {[f["filename"] for f in files_r.get("files", [])]}')

# NOTE: draft-transcribe does NOT work on parquet-format file stores (no raw audio files).
# HF-sourced file stores already have transcripts → go straight to process.
print(f'\n  Note: skipping draft-transcribe — parquet format has embedded audio+text, no raw audio files')

# ── Step 2: process → push to HF ─────────────────────────────────────────────
print(f'\n=== Step 2: process → push to HF [{elapsed():.1f}min] ===')
r3 = requests.post(f'{BASE}/file-stores/{file_store_id}/process', headers=H, json={
    'output_dataset_name': 'multimed-pipe-test',
    'split_option': 'test_only',
    'language': 'english',
    'enable_quality_checks': False,
    'min_chunk_duration': 1.0,
}).json()
print(f'  Response: {r3}')
proc_job_id = r3.get('job_id') or r3.get('id')
if not proc_job_id:
    print(f'ERROR: no job_id from process — response: {r3}'); raise SystemExit(1)

proc_result, proc_time = poll_job(proc_job_id, 'process', interval=15)
print(f'  Process: {proc_result.get("status")} in {proc_time:.0f}s')
if proc_result.get('status') != 'completed':
    print(f'  Error: {proc_result.get("error")}')
    print(f'  Logs: {proc_result.get("logs","")[-500:]}')

# ── Summary ───────────────────────────────────────────────────────────────────
total = elapsed()
print(f'\n=== Summary [{total:.1f}min total for 10 rows] ===')
print(f'  Step 1 from-hf-dataset:  {import_result.get("status")} in {import_time:.0f}s')
print(f'  Step 2 process:          {proc_result.get("status")} in {proc_time:.0f}s')
if proc_result.get('status') == 'completed':
    out_ds = proc_result.get('result', {})
    print(f'  Output dataset: {out_ds}')
    print(f'\n  Pipeline is WORKING. Estimated time for 4,751 rows: ~{import_time * 475 / 60:.0f}min import + ~{proc_time * 475 / 60:.0f}min process')
