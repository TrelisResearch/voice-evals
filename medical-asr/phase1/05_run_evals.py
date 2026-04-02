#!/usr/bin/env python3
"""
Step 5: Submit Trelis Studio eval jobs, respecting 10-concurrent-job limit.
Polls for completions and submits remaining jobs as slots free up.
"""
import os, json, time, requests
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')

API_KEY = os.environ['TRELIS_STUDIO_API_KEY']
BASE = 'https://studio.trelis.com/api/v1'
HEADERS = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
MAX_CONCURRENT = 10
POLL_INTERVAL = 30  # seconds

DATASETS = [
    {'hf_id': 'ronanarraig/medical-pilot-eka',      'name': 'eka'},
    {'hf_id': 'ronanarraig/medical-pilot-multimed', 'name': 'multimed'},
    {'hf_id': 'ronanarraig/medical-pilot-united',   'name': 'united'},
]

OPEN_SOURCE_MODELS = [
    'openai/whisper-large-v3',
    'nvidia/canary-1b-v2',
    'mistralai/Voxtral-Mini-3B-2507',
    'Qwen/Qwen3-ASR-1.7B',
    'UsefulSensors/moonshine-tiny',
]

PROPRIETARY_MODELS = [
    'google/gemini-2.5-pro',
    'speechmatics/ursa-2-enhanced',
    'deepgram/nova-3',
    'assemblyai/universal-3-pro',
    'elevenlabs/scribe-v2',
]

# Build full job queue
queue = []
for ds in DATASETS:
    for model in OPEN_SOURCE_MODELS + PROPRIETARY_MODELS:
        queue.append({'dataset': ds['name'], 'hf_id': ds['hf_id'], 'model': model})

print(f"Total jobs to submit: {len(queue)}")

# Load any already-submitted jobs
jobs_file = 'medical-asr/phase1/tmp/eval_jobs.json'
try:
    with open(jobs_file) as f:
        submitted = json.load(f)
except FileNotFoundError:
    submitted = []

already_done = {(j['dataset'], j['model']) for j in submitted}
queue = [j for j in queue if (j['dataset'], j['model']) not in already_done]
print(f"Already submitted: {len(already_done)}, remaining: {len(queue)}")

active_job_ids = {j['job_id'] for j in submitted if j.get('job_id')}

def get_active_count():
    r = requests.get(f'{BASE}/evaluation/jobs?limit=50', headers=HEADERS)
    if r.status_code != 200:
        return MAX_CONCURRENT  # assume full if can't check
    jobs = r.json().get('jobs', [])
    running = [j for j in jobs if j.get('status') in ('pending', 'running', 'processing')]
    return len(running)

def submit_job(hf_id, model):
    payload = {
        'model_id': model,
        'dataset_id': hf_id,
        'split': 'test',
        'num_samples': 50,
        'normalizer': 'generic',
        'language': 'en',
    }
    r = requests.post(f'{BASE}/evaluation/jobs', headers=HEADERS, json=payload)
    if r.status_code in (200, 201):
        data = r.json()
        return data.get('job_id') or data.get('id')
    elif r.status_code == 429:
        return 'LIMIT'
    else:
        print(f"  ERROR {r.status_code}: {r.text[:150]}")
        return None

while queue:
    active = get_active_count()
    slots = MAX_CONCURRENT - active
    print(f"\n[{time.strftime('%H:%M:%S')}] Active jobs: {active}, slots: {slots}, remaining in queue: {len(queue)}")

    if slots <= 0:
        print(f"  No slots — waiting {POLL_INTERVAL}s...")
        time.sleep(POLL_INTERVAL)
        continue

    # Submit up to available slots
    to_submit = queue[:slots]
    for job in to_submit:
        job_id = submit_job(job['hf_id'], job['model'])
        if job_id == 'LIMIT':
            print(f"  Hit limit submitting {job['dataset']} x {job['model']}, will retry")
            break
        elif job_id:
            print(f"  ✓ {job['dataset']} x {job['model']} → {job_id}")
            submitted.append({'dataset': job['dataset'], 'model': job['model'], 'job_id': job_id})
            queue.remove(job)
            with open(jobs_file, 'w') as f:
                json.dump(submitted, f, indent=2)
        else:
            print(f"  ✗ Failed: {job['dataset']} x {job['model']} — skipping")
            queue.remove(job)
        time.sleep(0.3)

    if queue:
        print(f"  Waiting {POLL_INTERVAL}s before checking slots again...")
        time.sleep(POLL_INTERVAL)

print(f"\nAll {len(submitted)} jobs submitted.")
print("Saved to", jobs_file)
