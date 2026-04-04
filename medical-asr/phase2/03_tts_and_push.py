#!/usr/bin/env python3
"""
Phase 2 Step 3: Generate TTS audio for medical sentences using Kokoro via Studio.

Flow per voice:
1. Upload .txt files to S3-backed file store (batch upload URLs)
2. POST /file-stores/{store_id}/tts with engine=kokoro, kokoro_voice=X
3. Poll data-prep job until complete → get output file store ID
4. POST /file-stores/{output_store_id}/process → push to HF split
5. Download audio from HF → save locally to phase2/tmp/audio/

Output:
  - phase2/tmp/tts_jobs.json (checkpoint: {voice: {store_id, tts_job_id, output_store_id, process_job_id}})
  - HF dataset: ronanarraig/medical-terms-tts-raw (test split with all voices merged)
  - phase2/tmp/sentences_with_audio.json (sentences + local audio paths)

Run from: /home/claude/TR/voice-evals
"""
import os, json, time, requests, math
import pyarrow as pa
from pathlib import Path
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')
from datasets import load_dataset, Dataset, Audio

HF_TOKEN = os.environ['HF_TOKEN']
API_KEY  = os.environ['TRELIS_STUDIO_API_KEY']
BASE     = 'https://studio.trelis.com/api/v1'
H        = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}

SENTENCES_FILE = 'medical-asr/phase2/tmp/sentences.json'
JOBS_FILE      = 'medical-asr/phase2/tmp/tts_jobs.json'
AUDIO_DIR      = Path('medical-asr/phase2/tmp/audio')
HF_DATASET     = 'ronanarraig/medical-terms-tts-raw'

KOKORO_VOICES = ['af_heart', 'am_michael', 'bf_emma', 'af_bella']

t0 = time.time()
def elapsed(): return f'{(time.time()-t0)/60:.1f}min'
def tick(msg): print(f'  [{elapsed()}] {msg}', flush=True)

AUDIO_DIR.mkdir(parents=True, exist_ok=True)

sentences = json.load(open(SENTENCES_FILE))
print(f'Loaded {len(sentences)} sentences')

# Distribute sentences evenly across voices
voice_groups = {v: [] for v in KOKORO_VOICES}
for i, s in enumerate(sentences):
    voice = KOKORO_VOICES[i % len(KOKORO_VOICES)]
    voice_groups[voice].append((i, s))

for v, grp in voice_groups.items():
    print(f'  {v}: {len(grp)} sentences')

# ── Load or init checkpoint ─────────────────────────────────────────
if os.path.exists(JOBS_FILE):
    jobs = json.load(open(JOBS_FILE))
    print(f'\nResuming from checkpoint: {list(jobs.keys())}')
else:
    jobs = {}

def poll_datprep_job(job_id, label, interval=20, timeout=3600):
    for _ in range(timeout // interval):
        r = requests.get(f'{BASE}/data-prep/jobs/{job_id}', headers=H)
        data = r.json()
        status = data.get('status', '?')
        print(f'  {label}: {status}', flush=True)
        if status in ('completed', 'done', 'ready'):
            return data
        if status in ('failed', 'error', 'aborted'):
            print(f'  ERROR: {data.get("error",data.get("message",""))[:200]}')
            return data
        time.sleep(interval)
    return {'status': 'timeout'}

# ── Step 1: Upload .txt files + submit TTS jobs ────────────────────��─
for voice, grp in voice_groups.items():
    if voice in jobs and jobs[voice].get('tts_job_id'):
        print(f'\n[{voice}] Already submitted TTS job {jobs[voice]["tts_job_id"]}, skipping upload')
        continue

    print(f'\n=== [{voice}] Uploading {len(grp)} .txt files ===')
    filenames = [f'{voice}_{i:03d}.txt' for i, _ in grp]
    texts_for_voice = [s['tts_text'] for _, s in grp]
    text_bytes = [t.encode('utf-8') for t in texts_for_voice]

    # Get presigned upload URLs
    r = requests.post(f'{BASE}/file-stores/upload-urls', headers=H, json={
        'files': [
            {'filename': fn, 'size_bytes': len(tb), 'content_type': 'text/plain'}
            for fn, tb in zip(filenames, text_bytes)
        ],
        'name': f'medical-terms-tts-{voice}',
    })
    resp = r.json()
    store_id = resp.get('file_store_id')
    upload_entries = resp.get('files', [])
    tick(f'got file_store_id={store_id}, {len(upload_entries)} upload URLs')

    # PUT each file
    for entry, tb in zip(upload_entries, text_bytes):
        url = entry['upload_url']
        ct = entry.get('content_type', 'text/plain')
        put_r = requests.put(url, data=tb, headers={'Content-Type': ct}, timeout=30)
        if put_r.status_code not in (200, 204):
            print(f'  WARNING: upload status {put_r.status_code} for {entry["filename"]}')

    tick(f'uploaded {len(filenames)} files')

    # Submit TTS job
    r = requests.post(f'{BASE}/file-stores/{store_id}/tts', headers=H, json={
        'engine': 'kokoro',
        'kokoro_voice': voice,
    })
    tts_resp = r.json()
    tts_job_id = tts_resp.get('job_id') or tts_resp.get('id')
    # output_file_store_id is returned in the TTS response directly
    output_store_id_initial = tts_resp.get('output_file_store_id')
    tick(f'TTS job submitted: {tts_job_id}, output_store={output_store_id_initial}')

    jobs[voice] = {
        'store_id': store_id,
        'tts_job_id': tts_job_id,
        'output_store_id': output_store_id_initial,  # capture from initial response
        'sentences': [(i, s['text']) for i, s in grp],
    }
    json.dump(jobs, open(JOBS_FILE, 'w'), indent=2)

# ── Step 2: Poll all TTS jobs ─────────────────────────────────────────
print(f'\n=== Polling TTS jobs ===')
for voice in KOKORO_VOICES:
    job_info = jobs.get(voice, {})
    tts_job_id = job_info.get('tts_job_id')
    if not tts_job_id:
        print(f'[{voice}] No TTS job ID, skipping')
        continue
    if job_info.get('output_store_id'):
        print(f'[{voice}] Already have output_store_id={job_info["output_store_id"]}, skipping poll')
        continue

    print(f'\n[{voice}] Polling TTS job {tts_job_id}...')
    result = poll_datprep_job(tts_job_id, f'TTS/{voice}')
    status = result.get('status', '?')
    output_store_id = (result.get('result') or {}).get('output_file_store_id') or result.get('output_file_store_id')
    print(f'  → status={status} output_store_id={output_store_id}')
    if output_store_id:
        jobs[voice]['output_store_id'] = output_store_id
        json.dump(jobs, open(JOBS_FILE, 'w'), indent=2)

# ── Step 3: Process each output store → push to HF ──────────────────
print(f'\n=== Processing TTS output stores ===')
for voice in KOKORO_VOICES:
    job_info = jobs.get(voice, {})
    output_store_id = job_info.get('output_store_id')
    if not output_store_id:
        print(f'[{voice}] No output_store_id, skipping process')
        continue
    if job_info.get('process_job_id'):
        print(f'[{voice}] Already submitted process job, skipping')
        continue

    print(f'\n[{voice}] Processing output store {output_store_id}...')
    r = requests.post(f'{BASE}/file-stores/{output_store_id}/process', headers=H, json={
        'output_dataset_name': f'medical-terms-tts-{voice}',
        'split_option': 'test_only',
        'language': 'eng',
    })
    proc_resp = r.json()
    proc_job_id = proc_resp.get('job_id') or proc_resp.get('id')
    tick(f'process job submitted: {proc_job_id}')

    jobs[voice]['process_job_id'] = proc_job_id
    json.dump(jobs, open(JOBS_FILE, 'w'), indent=2)

# ── Step 4: Poll process jobs + collect HF dataset IDs ───────────────
print(f'\n=== Polling process jobs ===')
for voice in KOKORO_VOICES:
    job_info = jobs.get(voice, {})
    proc_job_id = job_info.get('process_job_id')
    if not proc_job_id:
        print(f'[{voice}] No process job ID, skipping')
        continue
    if job_info.get('hf_dataset_id'):
        print(f'[{voice}] Already have hf_dataset_id={job_info["hf_dataset_id"]}, skipping')
        continue

    print(f'\n[{voice}] Polling process job {proc_job_id}...')
    result = poll_datprep_job(proc_job_id, f'process/{voice}', interval=20)
    status = result.get('status', '?')
    # Find pushed dataset ID
    res = result.get('result') or {}
    hf_ds_id = (res.get('pushed_dataset_id') or res.get('dataset_id') or
                result.get('pushed_dataset_id') or result.get('dataset_id'))
    print(f'  → status={status} hf_dataset_id={hf_ds_id}')
    if hf_ds_id:
        jobs[voice]['hf_dataset_id'] = hf_ds_id
        json.dump(jobs, open(JOBS_FILE, 'w'), indent=2)

# ── Step 5: Download audio + merge into one local dataset ─────────────
print(f'\n=== Downloading audio from HF datasets ===')
all_rows = []

for voice in KOKORO_VOICES:
    job_info = jobs.get(voice, {})
    hf_ds_id = job_info.get('hf_dataset_id')
    if not hf_ds_id:
        print(f'[{voice}] No HF dataset ID, skipping download')
        continue

    print(f'[{voice}] Loading {hf_ds_id}...')
    try:
        ds = load_dataset(hf_ds_id, split='test', token=HF_TOKEN)
        ds_plain = ds.cast_column('audio', Audio(decode=False)) if 'audio' in ds.column_names else ds
        print(f'  {len(ds_plain)} rows, cols: {ds_plain.column_names}')

        voice_sentences = job_info.get('sentences', [])
        # Match by filename/text ordering
        for hf_idx, row in enumerate(ds_plain):
            # Original sentence idx (0-based in full list)
            if hf_idx < len(voice_sentences):
                orig_idx, orig_text = voice_sentences[hf_idx]
            else:
                orig_idx = -1
                orig_text = row.get('text', '')

            audio_bytes = (row.get('audio') or {}).get('bytes', b'')
            if audio_bytes:
                wav_path = AUDIO_DIR / f'{orig_idx}_{voice}.wav'
                wav_path.write_bytes(audio_bytes)

            # Use text from our sentences list (not the VTT-derived text from Studio)
            text = sentences[orig_idx]['text'] if orig_idx >= 0 else orig_text
            all_rows.append({
                'idx': orig_idx,
                'voice': voice,
                'keyword': sentences[orig_idx]['keyword'] if orig_idx >= 0 else '',
                'category': sentences[orig_idx]['category'] if orig_idx >= 0 else '',
                'text': text,
                'audio_path': str(AUDIO_DIR / f'{orig_idx}_{voice}.wav') if audio_bytes else '',
                'duration': row.get('duration', 0.0),
            })
    except Exception as e:
        print(f'  [{voice}] Load failed: {e}')
        import traceback; traceback.print_exc()

tick(f'Downloaded {len(all_rows)} rows total, audio in {AUDIO_DIR}')

if not all_rows:
    print('ERROR: No rows downloaded. Check TTS + process jobs above.')
    exit(1)

# ── Step 6: Build merged HF dataset ──────────────────────────────────
print(f'\n=== Pushing merged dataset to {HF_DATASET} ===')
audio_bytes_list, texts, voices, keywords, categories, durations = [], [], [], [], [], []
for r in all_rows:
    ap = r['audio_path']
    audio_bytes_list.append(open(ap, 'rb').read() if ap and os.path.exists(ap) else b'')
    texts.append(r['text'])
    voices.append(r['voice'])
    keywords.append(r['keyword'])
    categories.append(r['category'])
    durations.append(float(r.get('duration', 0)))

audio_col = pa.array(audio_bytes_list, type=pa.binary())
table = pa.table({
    'audio':    audio_col,
    'text':     pa.array(texts),
    'voice':    pa.array(voices),
    'keyword':  pa.array(keywords),
    'category': pa.array(categories),
    'duration': pa.array(durations, type=pa.float32()),
})
merged_ds = Dataset(table).cast_column('audio', Audio(sampling_rate=24000))
print(f'  Dataset: {len(merged_ds)} rows')
merged_ds.push_to_hub(HF_DATASET, split='test', private=True, token=HF_TOKEN)
tick(f'Pushed to {HF_DATASET}')

# Save sentences_with_audio
json.dump(all_rows, open('medical-asr/phase2/tmp/sentences_with_audio.json', 'w'), indent=2)
print(f'Saved sentences_with_audio.json')
print('\nDONE')
