#!/usr/bin/env python3
"""
Step 12: MultiMed sentence-filtered pipeline.
1. Filter MultiMed to sentence-length rows → push multimed-sentences-500
2. Create Studio file store from HF dataset
3. Draft-transcribe with Whisper large-v3
4. Process → push multimed-sentences-transcribed
5. Run Gemini Pro eval → Otsu CER filter → push inspection subset
"""
import os, json, time, re, random, requests
import numpy as np
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')

import pathlib
from datasets import load_dataset, Dataset, Audio
from huggingface_hub import HfApi

HF_TOKEN = os.environ['HF_TOKEN']
API_KEY = os.environ['TRELIS_STUDIO_API_KEY']
BASE = 'https://studio.trelis.com/api/v1'
HEADERS = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}
TMP = pathlib.Path('medical-asr/phase1/tmp')
api = HfApi(token=HF_TOKEN)
random.seed(42)
N = 500

def is_sentence(text, duration):
    """Heuristic: complete phrase/sentence."""
    if len(text) < 60: return False
    if duration < 5.0: return False
    # Ends with sentence-ending punctuation
    if re.search(r'[.!?]\s*$', text.strip()): return True
    # Or has multiple clauses (commas/semicolons suggest phrase-level content)
    if text.count(',') >= 2: return True
    # Or long enough to be a complete phrase even without punctuation
    if len(text) >= 100: return True
    return False

def poll_file_store(store_id, interval=15):
    while True:
        r = requests.get(f'{BASE}/file-stores/{store_id}', headers=HEADERS)
        data = r.json()
        status = data.get('status', 'unknown')
        if status in ('ready', 'completed', 'done'):
            print(f"  File store ready")
            return data
        elif status in ('failed', 'error'):
            print(f"  File store FAILED: {data}")
            return None
        else:
            print(f"  File store status: {status}... waiting {interval}s")
            time.sleep(interval)

def poll_job(store_id, job_type, interval=20):
    """Poll draft-transcribe or process job on a file store."""
    while True:
        r = requests.get(f'{BASE}/file-stores/{store_id}', headers=HEADERS)
        data = r.json()
        job_status = data.get(f'{job_type}_status') or data.get('status', 'unknown')
        if job_status in ('completed', 'done', 'ready'):
            print(f"  {job_type}: completed")
            return data
        elif job_status in ('failed', 'error'):
            print(f"  {job_type}: FAILED — {data}")
            return None
        else:
            print(f"  {job_type}: {job_status}... waiting {interval}s")
            time.sleep(interval)

def poll_eval_job(job_id, label, interval=20):
    while True:
        r = requests.get(f'{BASE}/evaluation/jobs/{job_id}', headers=HEADERS)
        data = r.json()
        status = data.get('status')
        if status == 'completed':
            print(f"  {label}: completed")
            return data
        elif status == 'failed':
            print(f"  {label}: FAILED — {data.get('error','')[:100]}")
            return None
        else:
            print(f"  {label}: {status}... waiting {interval}s")
            time.sleep(interval)

def otsu_threshold(values):
    values = np.array(values)
    thresholds = np.linspace(values.min(), values.max(), 500)
    best_t, best_var = 0, -1
    for t in thresholds:
        w0 = np.mean(values <= t); w1 = 1 - w0
        if w0 == 0 or w1 == 0: continue
        var = w0 * w1 * (np.mean(values[values <= t]) - np.mean(values[values > t])) ** 2
        if var > best_var: best_var, best_t = var, t
    return best_t

# ── Step 1: Filter + sample 500 ───────────────────────────────────
print("=== Step 1: Filter MultiMed to sentence-length rows ===")
mm_raw = load_dataset('leduckhai/MultiMed', 'English', split='test', token=HF_TOKEN)
mm_raw = mm_raw.cast_column('audio', mm_raw.features['audio'].__class__(decode=False))
print(f"  Total rows: {len(mm_raw)}")

pool = [i for i, row in enumerate(mm_raw) if is_sentence(row['text'], row.get('duration', 0))]
print(f"  Sentence pool (duration≥5s, len≥60, sentence heuristic): {len(pool)} rows")

# Show length stats
lengths = [len(mm_raw[i]['text']) for i in pool]
durations = [mm_raw[i].get('duration', 0) for i in pool]
print(f"  Text length — median:{np.median(lengths):.0f} chars")
print(f"  Duration — median:{np.median(durations):.1f}s")
print(f"  Sample texts:")
for i in random.sample(pool, min(5, len(pool))):
    print(f"    [{mm_raw[i].get('duration',0):.1f}s] {mm_raw[i]['text'][:90]}")

selected = random.sample(pool, min(N, len(pool)))
rows = []
for i in selected:
    row = mm_raw[i]
    rows.append({
        'text': row['text'],
        'audio': row['audio'],
        'duration': float(row.get('duration', 0)),
        'source': 'leduckhai/MultiMed',
    })

ds = Dataset.from_list(rows).cast_column('audio', Audio(sampling_rate=16000))
ds.push_to_hub('ronanarraig/multimed-sentences-500', split='test', token=HF_TOKEN, private=True)
print(f"\n  Pushed ronanarraig/multimed-sentences-500 ({len(ds)} rows)")

# ── Step 2: Create file store from HF dataset ─────────────────────
print("\n=== Step 2: Create Studio file store from HF dataset ===")
r = requests.post(f'{BASE}/file-stores/from-hf-dataset', headers=HEADERS, json={
    'dataset_id': 'ronanarraig/multimed-sentences-500',
    'split': 'test',
    'audio_column': 'audio',
    'text_column': 'text',
    'name': 'multimed-sentences-500',
    'max_rows': N,
})
data = r.json()
print(f"  Response: {json.dumps({k: v for k, v in data.items() if k != 'files'}, indent=2)[:300]}")
store_id = data.get('file_store_id') or data.get('id') or data.get('store_id')
print(f"  Store ID: {store_id}")

with open(TMP / 'multimed_file_store.json', 'w') as f:
    json.dump({'store_id': store_id}, f)

# Poll until ready
poll_file_store(store_id)

# ── Step 3: Draft-transcribe with Whisper large-v3 ────────────────
print("\n=== Step 3: Draft-transcribe with Whisper large-v3 ===")
r = requests.post(f'{BASE}/file-stores/{store_id}/draft-transcribe', headers=HEADERS, json={
    'model_id': 'openai/whisper-large-v3',
    'language': 'en',
})
print(f"  Draft-transcribe response: {r.json()}")
# Poll until transcription done
poll_job(store_id, 'draft_transcribe', interval=20)

# ── Step 4: Process → push to HF ─────────────────────────────────
print("\n=== Step 4: Process → push to HF ===")
r = requests.post(f'{BASE}/file-stores/{store_id}/process', headers=HEADERS, json={
    'output_dataset_name': 'multimed-sentences-transcribed',
    'output_org': 'ronanarraig',
    'hf_token': HF_TOKEN,
    'split_option': 'test_only',
    'language': 'eng',
    'enable_quality_checks': False,
})
print(f"  Process response: {r.json()}")
result = poll_job(store_id, 'process', interval=30)
pushed_id = None
if result:
    pushed_id = result.get('output_dataset_id') or result.get('pushed_dataset_id') or 'ronanarraig/multimed-sentences-transcribed'
    print(f"  Output dataset: {pushed_id}")

with open(TMP / 'multimed_transcribed_id.json', 'w') as f:
    json.dump({'dataset_id': pushed_id or 'ronanarraig/multimed-sentences-transcribed'}, f)

# ── Step 5: Gemini Pro eval → Otsu CER filter ────────────────────
print("\n=== Step 5: Gemini Pro eval + Otsu CER filter ===")
transcribed_id = pushed_id or 'ronanarraig/multimed-sentences-transcribed'

# Load transcribed dataset to get row count
transcribed_ds = load_dataset(transcribed_id, split='test', token=HF_TOKEN)
transcribed_ds = transcribed_ds.cast_column('audio', transcribed_ds.features['audio'].__class__(decode=False))
n_rows = len(transcribed_ds)
print(f"  Transcribed dataset: {n_rows} rows")

r = requests.post(f'{BASE}/evaluation/jobs', headers=HEADERS, json={
    'model_id': 'google/gemini-2.5-pro',
    'dataset_id': transcribed_id,
    'split': 'test',
    'num_samples': n_rows,
    'normalizer': 'generic',
    'language': 'en',
    'push_results': True,
})
data = r.json()
job_id = data.get('job_id') or data.get('id')
print(f"  Gemini Pro eval job: {job_id}")

job_data = poll_eval_job(job_id, 'Gemini Pro eval')
if not job_data:
    raise SystemExit("Gemini eval failed")

results_ds = load_dataset(job_data['result']['pushed_dataset_id'], split='test', token=HF_TOKEN)
results_ds = results_ds.cast_column('audio', results_ds.features['audio'].__class__(decode=False))

cer_col = next(c for c in results_ds.column_names if 'cer' in c.lower() and 'entity' not in c.lower())
ref_col = 'reference' if 'reference' in results_ds.column_names else 'text'
cer_values = [float(r[cer_col]) for r in results_ds]
text_to_cer = {r[ref_col]: float(r[cer_col]) for r in results_ds}

print(f"  CER stats: min={min(cer_values):.3f} max={max(cer_values):.3f} "
      f"mean={np.mean(cer_values):.3f} median={np.median(cer_values):.3f}")

t_otsu = otsu_threshold(cer_values)
CER_FLOOR = 0.05
print(f"  Otsu threshold: {t_otsu:.3f}")

# Filter
filtered = []
for row in transcribed_ds:
    cer = text_to_cer.get(row['text'])
    if cer is not None and CER_FLOOR <= cer <= t_otsu:
        d = dict(row); d['gemini_cer'] = cer
        filtered.append(d)

filtered.sort(key=lambda r: r['gemini_cer'], reverse=True)
print(f"  Rows passing CER filter: {len(filtered)}")

# Push filtered + inspection subset (top 20 hardest)
for name, rows in [('multimed-sentences-otsu', filtered),
                   ('multimed-sentences-inspect', filtered[:20])]:
    out_ds = Dataset.from_list(rows)
    if 'audio' in out_ds.column_names:
        out_ds = out_ds.cast_column('audio', Audio(sampling_rate=16000))
    api.create_repo(f'ronanarraig/{name}', repo_type='dataset', private=True, exist_ok=True)
    out_ds.push_to_hub(f'ronanarraig/{name}', split='test', token=HF_TOKEN, private=True)
    print(f"  Pushed ronanarraig/{name} ({len(out_ds)} rows)")

print("\nSample hard rows (top 10):")
print(f"{'CER':>6}  TEXT")
print("-" * 90)
for r in filtered[:10]:
    print(f"  {r['gemini_cer']:.3f}  {r['text'][:80]}")

print("\nStep 12 done.")
