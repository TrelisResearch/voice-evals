#!/usr/bin/env python3
"""Check eval job statuses and collect results when complete."""
import os, json, requests
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')
from collections import defaultdict

API_KEY = os.environ['TRELIS_STUDIO_API_KEY']
BASE = 'https://studio.trelis.com/api/v1'
HEADERS = {'Authorization': f'Bearer {API_KEY}'}

with open('medical-asr/phase1/tmp/eval_jobs.json') as f:
    jobs = json.load(f)

print(f"Checking {len(jobs)} jobs...\n")

results = []
status_counts = defaultdict(int)

for job in jobs:
    r = requests.get(f'{BASE}/evaluation/jobs/{job["job_id"]}', headers=HEADERS)
    if r.status_code != 200:
        print(f"  ERROR fetching {job['job_id']}: {r.status_code}")
        continue
    data = r.json()
    status = data.get('status', 'unknown')
    status_counts[status] += 1

    row = {
        'dataset': job['dataset'],
        'model': job['model'],
        'job_id': job['job_id'],
        'status': status,
    }

    if status == 'completed':
        metrics = data.get('result', data.get('results', data.get('metrics', {}))) or {}
        row['wer'] = metrics.get('wer')
        row['cer'] = metrics.get('cer')
        row['entity_cer'] = metrics.get('entity_cer')
        row['category_cer'] = metrics.get('category_cer', {})
        row['samples_evaluated'] = metrics.get('samples_evaluated')
        results.append(row)
    elif status == 'failed':
        row['error'] = data.get('error', str(data))[:100]
        results.append(row)
        print(f"  FAILED: {job['dataset']} x {job['model']}: {row['error']}")

print(f"Status summary: {dict(status_counts)}")

completed = [r for r in results if r['status'] == 'completed']
if completed:
    print(f"\n=== Results ({len(completed)} completed) ===")
    # Group by dataset
    by_ds = defaultdict(list)
    for r in completed:
        by_ds[r['dataset']].append(r)

    for ds_name, rows in sorted(by_ds.items()):
        print(f"\n--- {ds_name} ---")
        rows.sort(key=lambda x: (x.get('wer') or 1.0))
        print(f"{'Model':<40} {'WER':>6} {'CER':>6} {'EntCER':>8}  category_cer")
        print("-" * 90)
        for r in rows:
            wer = f"{r['wer']:.3f}" if r['wer'] is not None else "  n/a"
            cer = f"{r['cer']:.3f}" if r['cer'] is not None else "  n/a"
            ecer = f"{r['entity_cer']:.3f}" if r['entity_cer'] is not None else "  n/a"
            cats = r.get('category_cer', {})
            cat_str = '  '.join(f"{k}:{v:.2f}" for k, v in sorted(cats.items())) if cats else ''
            print(f"{r['model']:<40} {wer:>6} {cer:>6} {ecer:>8}  {cat_str}")

with open('medical-asr/phase1/tmp/eval_results.json', 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved to eval_results.json")
