#!/usr/bin/env python3
"""
Phase 1D: Build EKA-hard or MultiMed-hard baseline eval set.

Pipeline:
1. Filter source dataset (duration >= 3s, text len >= 60 chars)
2. Push filtered rows to temp HF dataset
3. Studio: file store from HF → draft-transcribe → process → push HF with word_timestamps
4. Load processed dataset → NLTK sentence detection + audio trim (MultiMed only)
5. Gemini 2.5 Pro ASR (audio only) → completeness check
6. Gemini Flash tagging → keep medical_density == high
7. Export to review UI

Usage:
    uv run python -u medical-asr/phase1/16_build_hard_eval.py --source eka
    uv run python -u medical-asr/phase1/16_build_hard_eval.py --source multimed
"""
import argparse, os, io, json, re, time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')

import nltk
import numpy as np
import soundfile as sf
import pyarrow as pa
import pyarrow.parquet as pq
import tempfile
from google import genai
from google.genai import types as gentypes
from datasets import load_dataset
from huggingface_hub import HfApi

# ── Config ────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--source', choices=['eka', 'multimed'], required=True)
args = parser.parse_args()

if args.source == 'eka':
    SRC_DATASET = 'ekacare/eka-medical-asr-evaluation-dataset'
    SRC_SPLIT = 'test'
    TEMP_DATASET = 'ronanarraig/eka-filtered-temp'
    TRANSCRIBED_DATASET = 'ronanarraig/eka-transcribed'
    REVIEW_DIR = Path('tools/review/data-eka')
    APPLY_NLTK_TRIM = False  # EKA rows are already sentence-level
else:
    SRC_DATASET = 'leduckhai/MultiMed'
    SRC_SPLIT = 'test'
    TEMP_DATASET = 'ronanarraig/multimed-test-filtered-temp'
    TRANSCRIBED_DATASET = 'ronanarraig/multimed-test-transcribed'
    REVIEW_DIR = Path('tools/review/data-multimed')
    APPLY_NLTK_TRIM = True   # MultiMed rows are lecture chunks needing sentence extraction

HF_TOKEN = os.environ['HF_TOKEN']
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
STUDIO_KEY = os.environ['TRELIS_STUDIO_API_KEY']
BASE = 'https://studio.trelis.com/api/v1'
HEADERS = {'Authorization': f'Bearer {STUDIO_KEY}', 'Content-Type': 'application/json'}

client = genai.Client(api_key=GEMINI_API_KEY)
api = HfApi(token=HF_TOKEN)
t0 = time.time()
timings = {}

def tick(label):
    timings[label] = time.time() - t0

MIN_DURATION = 3.0
MIN_CHARS_SOURCE = 60   # filter on source dataset
MIN_CHARS_NLTK = 40     # filter on NLTK output sentences
MAX_DURATION = 25.0
N_ASR_THREADS = 20
N_TAG_THREADS = 100
ASR_PROMPT = "Transcribe this audio. Output ONLY the transcription."
TAG_PROMPT = """You are a medical NLP expert. Given a transcribed sentence, return JSON:
- "is_medical": true if substantively about medicine, health, biology, or clinical topics
- "medical_density": "high" (dense clinical/scientific jargon throughout), "medium" (some medical terms), "low" (barely medical), "none" (not medical)
- "entities": list of medical named entities. Each: {"text": str, "category": str, "char_start": int, "char_end": int}. Categories: drug, condition, procedure, anatomy, organisation, measurement.

Return ONLY valid JSON."""


# ── Helpers ───────────────────────────────────────────────────────

def poll_data_prep_job(job_id, desc='job', interval=15):
    while True:
        r = requests.get(f'{BASE}/data-prep/jobs/{job_id}', headers=HEADERS)
        data = r.json()
        status = data.get('status')
        if status == 'completed':
            print(f"  {desc}: complete")
            return data
        elif status == 'failed':
            print(f"  {desc}: FAILED — {data.get('error','')[:200]}")
            return None
        else:
            print(f"  {desc}: {status}... waiting {interval}s")
            time.sleep(interval)


def find_clean_sentences(text, word_timestamps):
    sentences = nltk.sent_tokenize(text)
    if len(sentences) < 2:
        return []

    first_partial = not text[0].isupper() if text else True
    last_partial = not re.search(r'[.!?]\s*$', text.strip())

    inner = sentences[:]
    if first_partial and len(inner) > 1:
        inner = inner[1:]
    if last_partial and len(inner) > 1:
        inner = inner[:-1]
    if not inner:
        return []

    wts = json.loads(word_timestamps) if isinstance(word_timestamps, str) else word_timestamps
    if not wts:
        return []

    results = []
    search_start = 0
    for sent in inner:
        sent_words = sent.split()
        if not sent_words:
            continue

        first_w = re.sub(r'[^\w]', '', sent_words[0]).lower()
        last_w = re.sub(r'[^\w]', '', sent_words[-1]).lower()

        start_idx = None
        for i in range(search_start, len(wts)):
            w = re.sub(r'[^\w]', '', wts[i]['word']).lower()
            if w == first_w:
                start_idx = i
                break
        if start_idx is None:
            continue

        expected_end = start_idx + len(sent_words) - 1
        end_idx = start_idx
        for i in range(max(start_idx, expected_end - 3), min(expected_end + 6, len(wts))):
            w = re.sub(r'[^\w]', '', wts[i]['word']).lower()
            if w == last_w:
                end_idx = i
                break

        # Contextual padding
        if start_idx > 0:
            gap = wts[start_idx]['start'] - wts[start_idx - 1]['end']
            pad_start = min(gap / 2, 0.2)
        else:
            pad_start = 0.3
        if end_idx < len(wts) - 1:
            gap = wts[end_idx + 1]['start'] - wts[end_idx]['end']
            pad_end = min(gap / 2, 0.2)
        else:
            pad_end = 0.3

        start_sec = max(0.0, wts[start_idx]['start'] - pad_start)
        end_sec = wts[end_idx]['end'] + pad_end
        duration = end_sec - start_sec

        if duration < MIN_DURATION or duration > MAX_DURATION:
            continue
        if len(sent) < MIN_CHARS_NLTK:
            continue

        results.append((sent, start_sec, end_sec))
        search_start = end_idx
    return results


def trim_audio_array(audio_dict, start_sec, end_sec):
    raw = audio_dict.get('bytes')
    if not raw:
        return None, None
    try:
        array, sr = sf.read(io.BytesIO(raw))
        if array.ndim > 1:
            array = array.mean(axis=1)
        s = max(0, int(start_sec * sr))
        e = min(len(array), int(end_sec * sr))
        return array[s:e].astype(np.float32), sr
    except Exception:
        return None, None


def array_to_wav_bytes(array, sr):
    out = io.BytesIO()
    sf.write(out, array, sr, format='WAV', subtype='PCM_16')
    return out.getvalue()


def is_complete_transcript(text):
    text = text.strip()
    if len(text) < MIN_CHARS_NLTK:
        return False
    if not text[0].isupper():
        return False
    if not re.search(r'[.!?]["\')]*\s*$', text):
        return False
    return True


def transcribe_with_gemini_pro(idx, audio_bytes):
    for attempt in range(3):
        f = None
        try:
            f = client.files.upload(
                file=io.BytesIO(audio_bytes),
                config=gentypes.UploadFileConfig(mime_type='audio/wav', display_name=f'row_{idx}')
            )
            response = client.models.generate_content(
                model='gemini-2.5-pro',
                contents=[gentypes.Part.from_uri(file_uri=f.uri, mime_type='audio/wav'), ASR_PROMPT],
                config=gentypes.GenerateContentConfig(temperature=0.0),
            )
            return idx, response.text.strip()
        except Exception as e:
            if attempt == 2:
                return idx, None
            time.sleep(2 ** attempt)
        finally:
            if f:
                try: client.files.delete(name=f.name)
                except: pass


def tag_transcript(idx, text):
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=f"{TAG_PROMPT}\n\nText: {text}",
                config=gentypes.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type='application/json',
                ),
            )
            result = json.loads(response.text.strip())
            if isinstance(result, list):
                result = result[0] if result and isinstance(result[0], dict) else {}
            return idx, result
        except Exception:
            if attempt == 2:
                return idx, {}
            time.sleep(2 ** attempt)


# ── Step 1: Filter source dataset ────────────────────────────────
print(f"\n{'='*60}")
print(f"Source: {args.source.upper()} — {SRC_DATASET} ({SRC_SPLIT})")
print(f"{'='*60}")

print(f"\nStep 1: Loading and filtering {SRC_DATASET}...")
ds_kwargs = {'split': SRC_SPLIT, 'token': HF_TOKEN}
if args.source == 'multimed':
    ds_kwargs['name'] = 'English'
ds = load_dataset(SRC_DATASET, **ds_kwargs)
ds = ds.cast_column('audio', ds.features['audio'].__class__(decode=False))
print(f"  {len(ds)} total rows")

filtered = []
for row in ds:
    dur = row.get('duration') or 0
    txt = row.get('text') or ''
    if dur >= MIN_DURATION and len(txt) >= MIN_CHARS_SOURCE:
        filtered.append(row)

print(f"  {len(filtered)} rows after filter (duration >= {MIN_DURATION}s, text >= {MIN_CHARS_SOURCE} chars)")
tick('1_filter')

# ── Step 2: Push filtered rows to HF ─────────────────────────────
print(f"\nStep 2: Pushing {len(filtered)} rows to {TEMP_DATASET}...")

audio_bytes_list, texts, source_files, durations = [], [], [], []
for r in filtered:
    audio_bytes_list.append(r['audio']['bytes'])
    texts.append(r['text'])
    source_files.append(r['audio'].get('path') or r.get('file_name') or '')
    durations.append(float(r.get('duration') or 0))

audio_struct = pa.array(
    [{'bytes': b, 'path': p} for b, p in zip(audio_bytes_list, source_files)],
    type=pa.struct([pa.field('bytes', pa.binary()), pa.field('path', pa.string())])
)
table = pa.table({
    'audio': audio_struct,
    'text': pa.array(texts),
    'source_file': pa.array(source_files),
    'duration': pa.array(durations, type=pa.float32()),
})

with tempfile.NamedTemporaryFile(suffix='.parquet', delete=False) as f:
    pq.write_table(table, f.name)
    parquet_path = f.name

api.create_repo(TEMP_DATASET, repo_type='dataset', private=True, exist_ok=True)
api.upload_file(
    path_or_fileobj=parquet_path,
    path_in_repo='data/test-00000-of-00001.parquet',
    repo_id=TEMP_DATASET,
    repo_type='dataset',
    token=HF_TOKEN,
)
print(f"  Pushed {len(filtered)} rows")
tick('2_hf_push')

# ── Step 3: Studio — file store from HF ──────────────────────────
print(f"\nStep 3a: Creating file store from {TEMP_DATASET}...")
r = requests.post(f'{BASE}/file-stores/from-hf-dataset', headers=HEADERS, json={
    'dataset_id': TEMP_DATASET,
    'split': 'test',
    'name': f'{args.source}-filtered-temp',
    'max_rows': len(filtered),
})
data = r.json()
print(f"  {r.status_code} {data}")
fs_job_id = data.get('job_id')
store_id = data.get('file_store_id')

job = poll_data_prep_job(fs_job_id, 'HF import')
if not job:
    raise SystemExit("File store import failed")
tick('3a_filestore_import')

# ── Step 3b: Draft-transcribe ─────────────────────────────────────
print(f"\nStep 3b: Draft-transcribing {len(filtered)} files...")
r = requests.post(f'{BASE}/file-stores/{store_id}/draft-transcribe', headers=HEADERS,
    json={'language': 'en'})
data = r.json()
print(f"  {r.status_code} {data}")
dt_job_id = data.get('job_id')
output_store_id = data.get('output_file_store_id')

job = poll_data_prep_job(dt_job_id, 'draft-transcribe', interval=20)
if not job:
    raise SystemExit("Draft-transcribe failed")
print(f"  Cost: ${job['result'].get('cost_charged', 0):.3f}")
tick('3b_draft_transcribe')

# Capture signed file URLs from draft-transcribe job config (valid 4h)
dt_full_job = requests.get(f'{BASE}/data-prep/jobs/{dt_job_id}', headers=HEADERS).json()
dt_file_urls = dt_full_job.get('config', {}).get('file_urls', [])
print(f"  Captured {len(dt_file_urls)} signed file URL pairs")

# ── Step 3c: Load VTT+WAV from draft-transcribe output ───────────
# Note: Studio process step ignores output_org/hf_token (bug filed: 907b979b).
# Workaround: parse VTT+WAV directly from draft-transcribe signed URLs.
# For MultiMed (APPLY_NLTK_TRIM=True), we also try the process step to get
# word_timestamps for sentence-level trimming.

def parse_vtt(vtt_text):
    """Parse Whisper VTT → (full_text, start_sec, end_sec)."""
    segments = []
    lines = vtt_text.strip().split('\n')
    i = 0
    while i < len(lines):
        if '-->' in lines[i]:
            times = re.findall(r'(\d+:\d+:\d+\.\d+)', lines[i])
            if len(times) == 2:
                def to_sec(t):
                    h, m, s = t.split(':')
                    return int(h)*3600 + int(m)*60 + float(s)
                start, end = to_sec(times[0]), to_sec(times[1])
                i += 1
                text_lines = []
                while i < len(lines) and lines[i].strip() and '-->' not in lines[i]:
                    text_lines.append(lines[i].strip())
                    i += 1
                text = ' '.join(text_lines)
                if text:
                    segments.append((text, start, end))
            continue
        i += 1
    if not segments:
        return None, 0, 0
    return ' '.join(s[0] for s in segments), segments[0][1], segments[-1][2]


print(f"\nStep 3c: Downloading {len(dt_file_urls)} VTT+WAV pairs from draft-transcribe output...")
chunks = []
n_download_fail = 0

with ThreadPoolExecutor(max_workers=50) as executor:
    def download_pair(entry):
        try:
            vtt = requests.get(entry['transcript_url'], timeout=30).text
            wav_bytes = requests.get(entry['audio_url'], timeout=60).content
            text, start, end = parse_vtt(vtt)
            if not text:
                return None
            return {
                'audio': {'bytes': wav_bytes},
                'text': text,
                'source_file': entry['audio_filename'],
                'duration': end - start,
                'word_timestamps': '[]',  # segment-level only; word-level via process step
            }
        except Exception:
            return None

    futures = {executor.submit(download_pair, e): i for i, e in enumerate(dt_file_urls)}
    done = 0
    for future in as_completed(futures):
        result = future.result()
        chunks.append(result)
        done += 1
        if done % 200 == 0 or done == len(dt_file_urls):
            print(f"  {done}/{len(dt_file_urls)}")

chunks = [c for c in chunks if c is not None]
n_download_fail = len(dt_file_urls) - len(chunks)
print(f"  {n_download_fail} download failures")
print(f"  {len(chunks)} chunks loaded")
tick('3c_download_vtt')

# For MultiMed: also run process step to get word_timestamps, then merge
if APPLY_NLTK_TRIM:
    print(f"\nStep 3d: Process step for word_timestamps (MultiMed)...")
    r = requests.post(f'{BASE}/file-stores/{output_store_id}/process', headers=HEADERS, json={
        'output_org': 'ronanarraig',
        'output_dataset_name': TRANSCRIBED_DATASET.split('/')[-1],
        'split_option': 'test_only',
        'max_test_rows': 10000,
        'language': 'english',
        'enable_quality_checks': False,
        'min_chunk_duration': MIN_DURATION,
        'hf_token': HF_TOKEN,
    })
    data = r.json()
    proc_job_id = data.get('job_id') or data.get('id')
    print(f"  {r.status_code} job={proc_job_id}")
    proc_job = poll_data_prep_job(proc_job_id, 'process', interval=20)
    if proc_job and proc_job.get('result', {}).get('dataset_id'):
        # Process step successfully pushed to HF — load word_timestamps from there
        ds_id = proc_job['result']['dataset_id']
        print(f"  Loading word_timestamps from {ds_id}...")
        wts_ds = load_dataset(ds_id, split='test', token=HF_TOKEN)
        # Build map: source_file → word_timestamps
        wts_map = {row.get('source_file',''): row.get('word_timestamps','[]') for row in wts_ds}
        for c in chunks:
            c['word_timestamps'] = wts_map.get(c['source_file'], '[]')
        print(f"  Merged word_timestamps for {sum(1 for c in chunks if c[\"word_timestamps\"] != \"[]\")}/{len(chunks)} chunks")
    else:
        print(f"  Process step did not push to HF (known bug) — NLTK trim will be skipped")
        APPLY_NLTK_TRIM_EFFECTIVE = False
    tick('3d_process_word_timestamps')
else:
    APPLY_NLTK_TRIM_EFFECTIVE = False

# For EKA, no NLTK trim needed
if not APPLY_NLTK_TRIM:
    APPLY_NLTK_TRIM_EFFECTIVE = False

# ── Step 5: NLTK sentence detection + audio trim ─────────────────
print(f"\nStep 5: {'NLTK sentence detection + trim' if APPLY_NLTK_TRIM_EFFECTIVE else 'Direct extraction (no NLTK trim)'}...")
trimmed_rows = []
n_no_sentence = 0

for row in chunks:
    text = row.get('text') or row.get('transcript') or ''
    audio = row.get('audio') or {}
    dur = row.get('duration') or 0

    if APPLY_NLTK_TRIM_EFFECTIVE:
        wts = row.get('word_timestamps', '[]')
        sents = find_clean_sentences(text, wts)
        if not sents:
            n_no_sentence += 1
            continue
        for sent_text, start_sec, end_sec in sents:
            arr, sr = trim_audio_array(audio, start_sec, end_sec)
            if arr is None:
                continue
            trimmed_rows.append({
                'source_file': row.get('source_file', ''),
                'whisper_chunk': text,
                'whisper_sentence': sent_text,
                'start_sec': round(start_sec, 3),
                'end_sec': round(end_sec, 3),
                'duration': round(end_sec - start_sec, 2),
                'audio_array': arr,
                'sr': sr,
            })
    else:
        # EKA: each chunk is already a complete sentence
        if len(text) < MIN_CHARS_NLTK or dur < MIN_DURATION or dur > MAX_DURATION:
            n_no_sentence += 1
            continue
        arr, sr = trim_audio_array(audio, 0, dur)
        if arr is None:
            n_no_sentence += 1
            continue
        trimmed_rows.append({
            'source_file': row.get('source_file', ''),
            'whisper_chunk': text,
            'whisper_sentence': text,
            'start_sec': 0.0,
            'end_sec': round(dur, 3),
            'duration': round(dur, 2),
            'audio_array': arr,
            'sr': sr,
        })

print(f"  {n_no_sentence} chunks dropped")
print(f"  {len(trimmed_rows)} rows to transcribe")
tick('5_nltk_trim')

# ── Step 6: Gemini 2.5 Pro ASR ────────────────────────────────────
print(f"\nStep 6: Gemini 2.5 Pro ASR ({N_ASR_THREADS} threads) on {len(trimmed_rows)} rows...")
asr_results = [None] * len(trimmed_rows)

with ThreadPoolExecutor(max_workers=N_ASR_THREADS) as executor:
    futures = {executor.submit(transcribe_with_gemini_pro, i, array_to_wav_bytes(r['audio_array'], r['sr'])): i
               for i, r in enumerate(trimmed_rows)}
    done = 0
    for future in as_completed(futures):
        i, text = future.result()
        asr_results[i] = text
        done += 1
        if done % 50 == 0 or done == len(trimmed_rows):
            print(f"  {done}/{len(trimmed_rows)}")

n_asr_failed = sum(1 for t in asr_results if not t)
print(f"  {n_asr_failed} failed ASR calls")
tick('6_gemini_asr')

# ── Step 7: Completeness check ────────────────────────────────────
print(f"\nStep 7: Completeness check...")
complete_rows = []
n_incomplete = 0

for r, prediction in zip(trimmed_rows, asr_results):
    if not prediction or not is_complete_transcript(prediction):
        n_incomplete += 1
        continue
    r['gemini_transcript'] = prediction
    complete_rows.append(r)

print(f"  {n_incomplete} dropped (incomplete)")
print(f"  {len(complete_rows)} complete")
tick('7_completeness')

# ── Step 8: Gemini Flash tagging ─────────────────────────────────
print(f"\nStep 8: Tagging {len(complete_rows)} rows ({N_TAG_THREADS} threads)...")
tag_results = [None] * len(complete_rows)

with ThreadPoolExecutor(max_workers=N_TAG_THREADS) as executor:
    futures = {executor.submit(tag_transcript, i, r['gemini_transcript']): i
               for i, r in enumerate(complete_rows)}
    done = 0
    for future in as_completed(futures):
        i, tags = future.result()
        tag_results[i] = tags
        done += 1
        if done % 100 == 0 or done == len(complete_rows):
            print(f"  {done}/{len(complete_rows)}")

for r, tags in zip(complete_rows, tag_results):
    tags = tags or {}
    if isinstance(tags, list):
        tags = tags[0] if tags and isinstance(tags[0], dict) else {}
    r['is_medical'] = tags.get('is_medical', False)
    r['medical_density'] = tags.get('medical_density', 'none')
    r['entities'] = json.dumps(tags.get('entities', []))

# ── Drop log ──────────────────────────────────────────────────────
density_counts = {}
for r in complete_rows:
    density_counts[r['medical_density']] = density_counts.get(r['medical_density'], 0) + 1
n_high = density_counts.get('high', 0)
n_medical = sum(1 for r in complete_rows if r['is_medical'])

print(f"\n{'='*60}")
print(f"DROP LOG — {args.source.upper()}")
print(f"{'='*60}")
print(f"Source rows:                  {len(ds)}")
print(f"  → after duration/len filter: {len(filtered)}")
print(f"  → after Studio process:      {len(chunks)}")
print(f"  → dropped (no sentence):     {n_no_sentence}")
print(f"  → to transcribe:             {len(trimmed_rows)}")
print(f"  → incomplete Gemini output:  {n_incomplete} dropped")
print(f"  → complete transcripts:      {len(complete_rows)}")
print(f"  → is_medical=True:           {n_medical} ({100*n_medical/max(len(complete_rows),1):.0f}%)")
print(f"  → medical_density=high:      {n_high} ({100*n_high/max(len(complete_rows),1):.0f}%)")
print(f"  density breakdown: {density_counts}")
tick('8_tagging')

# ── Step 9: Export high-density rows ─────────────────────────────
high_rows = [r for r in complete_rows if r['medical_density'] == 'high']
print(f"\nStep 9: Exporting {len(high_rows)} high-density rows to {REVIEW_DIR}...")
audio_dir = REVIEW_DIR / 'audio'
audio_dir.mkdir(parents=True, exist_ok=True)

rows_out = []
for i, item in enumerate(high_rows):
    wav = array_to_wav_bytes(item['audio_array'], item['sr'])
    (audio_dir / f'{i}.wav').write_bytes(wav)
    rows_out.append({
        'id': i,
        'source_file': item['source_file'],
        'whisper_chunk': item['whisper_chunk'],
        'whisper_sentence': item['whisper_sentence'],
        'start_sec': item['start_sec'],
        'end_sec': item['end_sec'],
        'duration': item['duration'],
        'transcript': item['gemini_transcript'],
        'is_medical': item['is_medical'],
        'medical_density': item['medical_density'],
        'entities': item['entities'],
        'reviewed': False,
        'dropped': False,
    })

(REVIEW_DIR / 'rows.json').write_text(json.dumps(rows_out, indent=2, ensure_ascii=False))
print(f"Exported {len(rows_out)} rows")

print("\nSample high-density rows:")
for r in rows_out[:8]:
    ents = [e['text'] for e in json.loads(r['entities'] or '[]')]
    print(f"  [{r['duration']:.1f}s] {r['transcript'][:85]}")
    if ents:
        print(f"    entities: {', '.join(ents[:5])}")

tick('9_export')

print(f"\nStart review UI:")
print(f"  uv run tools/review/server.py --data {REVIEW_DIR}")

print(f"\n{'='*60}")
print(f"TIMINGS")
print(f"{'='*60}")
prev = 0
for label, t in timings.items():
    print(f"  {label}: {t-prev:.0f}s (cumulative {t:.0f}s)")
    prev = t
print("\nDone.")
