#!/usr/bin/env python3
"""
Step 7: Sample 500 rows from EKA and MultiMed EN, push to HF as private datasets,
submit Whisper large-v3 eval jobs to get per-sample CER for Otsu filtering.
"""
import os, json, random, requests
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')

import pathlib
from datasets import load_dataset, Dataset, Audio
from collections import defaultdict

HF_TOKEN = os.environ['HF_TOKEN']
API_KEY = os.environ['TRELIS_STUDIO_API_KEY']
BASE = 'https://studio.trelis.com/api/v1'
HEADERS = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
TMP = pathlib.Path('medical-asr/phase1/tmp')
TMP.mkdir(exist_ok=True)
random.seed(42)
N = 500

# ── EKA: stratified by recording_context ─────────────────────────
print("=== EKA: sampling 500 rows ===")
eka_raw = load_dataset('ekacare/eka-medical-asr-evaluation-dataset', 'en',
                       split='test', token=HF_TOKEN)
eka_raw = eka_raw.cast_column('audio', eka_raw.features['audio'].__class__(decode=False))

def convert_eka_entities(raw_str):
    import json as _j
    try:
        ents = _j.loads(raw_str) if isinstance(raw_str, str) else (raw_str or [])
    except Exception:
        return []
    result = []
    for e in ents:
        if isinstance(e, list) and len(e) >= 3:
            for span in e[2]:
                result.append({'text': e[0], 'category': e[1],
                                'char_start': span[0], 'char_end': span[1]})
    return result

# Stratified: proportional to context distribution
by_ctx = defaultdict(list)
for i, row in enumerate(eka_raw):
    if row['duration'] >= 1.5:
        by_ctx[row['recording_context']].append(i)

# conversation:110, narration_entity:2206, narration_sentence:1303 → proportional to 500
total_pool = sum(len(v) for v in by_ctx.values())
targets = {ctx: max(5, round(N * len(idxs) / total_pool))
           for ctx, idxs in by_ctx.items()}
# Adjust to exactly N
diff = N - sum(targets.values())
largest = max(targets, key=targets.get)
targets[largest] += diff

selected = []
for ctx, n in targets.items():
    selected.extend(random.sample(by_ctx[ctx], min(n, len(by_ctx[ctx]))))

print(f"  Selected {len(selected)} rows: {dict((c, len([i for i in selected if eka_raw[i]['recording_context']==c])) for c in by_ctx)}")

eka_500_rows = []
for i in selected:
    row = eka_raw[i]
    ents = convert_eka_entities(row['medical_entities'])
    eka_500_rows.append({
        'text': row['text'],
        'audio': row['audio'],
        'duration': float(row['duration']),
        'speaker': row['speaker'],
        'recording_context': row['recording_context'],
        'entities': json.dumps(ents),
        'source': 'ekacare/eka-medical-asr-evaluation-dataset',
    })

eka_500_ds = Dataset.from_list(eka_500_rows).cast_column('audio', Audio(sampling_rate=16000))
eka_500_ds.push_to_hub('ronanarraig/eka-500', split='test', token=HF_TOKEN, private=True)
print(f"  Pushed ronanarraig/eka-500 ({len(eka_500_ds)} rows)")

# ── MultiMed EN: random 500 ───────────────────────────────────────
print("\n=== MultiMed EN: sampling 500 rows ===")
mm_raw = load_dataset('leduckhai/MultiMed', 'English', split='test', token=HF_TOKEN)
mm_raw = mm_raw.cast_column('audio', mm_raw.features['audio'].__class__(decode=False))

# Filter: duration >= 3s (removes very short clips)
pool = [i for i, d in enumerate(mm_raw['duration']) if d >= 3.0]
selected_mm = random.sample(pool, min(N, len(pool)))

mm_500_rows = []
for i in selected_mm:
    row = mm_raw[i]
    mm_500_rows.append({
        'text': row['text'],
        'audio': row['audio'],
        'duration': float(row['duration']),
        'entities': '[]',  # will be extracted after Otsu filter
        'source': 'leduckhai/MultiMed',
    })

mm_500_ds = Dataset.from_list(mm_500_rows).cast_column('audio', Audio(sampling_rate=16000))
mm_500_ds.push_to_hub('ronanarraig/multimed-500', split='test', token=HF_TOKEN, private=True)
print(f"  Pushed ronanarraig/multimed-500 ({len(mm_500_ds)} rows)")

# ── Submit Whisper eval jobs ──────────────────────────────────────
print("\n=== Submitting Whisper large-v3 eval jobs ===")
jobs = {}
for ds_id, name in [('ronanarraig/eka-500', 'eka'), ('ronanarraig/multimed-500', 'multimed')]:
    r = requests.post(f'{BASE}/evaluation/jobs', headers=HEADERS, json={
        'model_id': 'fireworks/whisper-v3',  # router version — cheaper/faster for Otsu CER filter
        'dataset_id': ds_id,
        'split': 'test',
        'num_samples': N,
        'normalizer': 'generic',
        'language': 'en',
        'push_results': True,
    })
    data = r.json()
    job_id = data.get('job_id') or data.get('id')
    jobs[name] = job_id
    print(f"  {name}: job_id={job_id}, balance={data.get('balance', '?'):.1f}" if isinstance(data.get('balance'), float) else f"  {name}: job_id={job_id}")

with open(TMP / 'whisper_500_jobs.json', 'w') as f:
    json.dump(jobs, f, indent=2)
print(f"\nJob IDs saved to tmp/whisper_500_jobs.json")
print("Waiting for jobs to complete before running Otsu filter...")
