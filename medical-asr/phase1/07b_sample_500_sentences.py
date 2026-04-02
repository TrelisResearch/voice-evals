#!/usr/bin/env python3
"""
Step 7b: Re-sample 500 EKA rows, filtered to sentence-length utterances first.
Filter: recording_context == 'narration_sentence' OR len(text) >= 60 chars.
Then stratified sample 500 from that pool, push to ronanarraig/eka-sentences-500,
submit Whisper large-v3 eval job.
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
MIN_LEN = 60

def convert_eka_entities(raw_str):
    try:
        ents = json.loads(raw_str) if isinstance(raw_str, str) else (raw_str or [])
    except Exception:
        return []
    result = []
    for e in ents:
        if isinstance(e, list) and len(e) >= 3:
            for span in e[2]:
                result.append({'text': e[0], 'category': e[1],
                                'char_start': span[0], 'char_end': span[1]})
    return result

print("Loading EKA dataset...")
eka_raw = load_dataset('ekacare/eka-medical-asr-evaluation-dataset', 'en',
                       split='test', token=HF_TOKEN)
eka_raw = eka_raw.cast_column('audio', eka_raw.features['audio'].__class__(decode=False))
print(f"  Total rows: {len(eka_raw)}")

# Filter to sentence pool
sentence_pool = []
for i, row in enumerate(eka_raw):
    is_sentence_context = row['recording_context'] == 'narration_sentence'
    is_long_enough = len(row['text']) >= MIN_LEN
    if row['duration'] >= 1.5 and (is_sentence_context or is_long_enough):
        sentence_pool.append(i)

print(f"  Sentence pool (narration_sentence OR len≥{MIN_LEN}): {len(sentence_pool)} rows")

# Show context breakdown of pool
from collections import Counter
ctx_counts = Counter(eka_raw[i]['recording_context'] for i in sentence_pool)
print(f"  Context breakdown: {dict(ctx_counts)}")

# Stratified sample 500 by recording_context
by_ctx = defaultdict(list)
for i in sentence_pool:
    by_ctx[eka_raw[i]['recording_context']].append(i)

total_pool = sum(len(v) for v in by_ctx.values())
targets = {ctx: max(5, round(N * len(idxs) / total_pool))
           for ctx, idxs in by_ctx.items()}
diff = N - sum(targets.values())
largest = max(targets, key=targets.get)
targets[largest] += diff

selected = []
for ctx, n in targets.items():
    selected.extend(random.sample(by_ctx[ctx], min(n, len(by_ctx[ctx]))))

print(f"\n  Sampled {len(selected)} rows:")
for ctx in by_ctx:
    n = sum(1 for i in selected if eka_raw[i]['recording_context'] == ctx)
    print(f"    {ctx}: {n}")

# Show text length stats
lengths = [len(eka_raw[i]['text']) for i in selected]
import numpy as np
print(f"\n  Text length — min:{min(lengths)} median:{np.median(lengths):.0f} max:{max(lengths)}")
print(f"  Sample texts:")
for i in selected[:5]:
    print(f"    [{eka_raw[i]['recording_context']}] {eka_raw[i]['text'][:90]}")

# Build dataset
rows = []
for i in selected:
    row = eka_raw[i]
    ents = convert_eka_entities(row['medical_entities'])
    rows.append({
        'text': row['text'],
        'audio': row['audio'],
        'duration': float(row['duration']),
        'speaker': row['speaker'],
        'recording_context': row['recording_context'],
        'entities': json.dumps(ents),
        'source': 'ekacare/eka-medical-asr-evaluation-dataset',
    })

ds = Dataset.from_list(rows).cast_column('audio', Audio(sampling_rate=16000))
ds.push_to_hub('ronanarraig/eka-sentences-500', split='test', token=HF_TOKEN, private=True)
print(f"\n  Pushed ronanarraig/eka-sentences-500 ({len(ds)} rows)")

# Submit Whisper eval job
r = requests.post(f'{BASE}/evaluation/jobs', headers=HEADERS, json={
    'model_id': 'fireworks/whisper-v3',
    'dataset_id': 'ronanarraig/eka-sentences-500',
    'split': 'test',
    'num_samples': N,
    'normalizer': 'generic',
    'language': 'en',
    'push_results': True,
})
data = r.json()
job_id = data.get('job_id') or data.get('id')
print(f"  Whisper eval job submitted: {job_id}")

with open(TMP / 'eka_sentences_whisper_job.json', 'w') as f:
    json.dump({'job_id': job_id}, f)
print("Done. Waiting for Whisper job before running CER filter.")
