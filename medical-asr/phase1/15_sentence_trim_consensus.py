#!/usr/bin/env python3
"""
Phase 1C: MultiMed sentence extraction + Gemini 2.5 Pro (via Studio) consensus + tagging.

Pipeline:
1. NLTK sentence detection on Whisper text → clean inner sentences
2. Trim audio via word timestamps
3. Push trimmed clips as temp HF dataset
4. Studio eval with google/gemini-2.5-pro → clean transcriptions
5. Post-NLTK completeness check on Gemini output → discard incomplete
6. Gemini Flash tagging (is_medical, medical_density, entities) on clean transcripts
7. Keep medical_density == high, export to review UI

Reports drop counts at each step.
"""
import os, json, re, io, time, requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')

import nltk
import numpy as np
import soundfile as sf
from google import genai
from google.genai import types as gentypes
from datasets import load_dataset, Dataset, Audio
from huggingface_hub import HfApi

HF_TOKEN = os.environ['HF_TOKEN']
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
API_KEY = os.environ['TRELIS_STUDIO_API_KEY']
BASE = 'https://studio.trelis.com/api/v1'
HEADERS = {'Authorization': f'Bearer {API_KEY}', 'Content-Type': 'application/json'}

client = genai.Client(api_key=GEMINI_API_KEY)
api = HfApi(token=HF_TOKEN)

N_TAG_THREADS = 100
N_ASR_THREADS = 20
ASR_PROMPT = "Transcribe this audio. Output ONLY the transcription."
MIN_DURATION = 3.0
MIN_CHARS = 40
MAX_DURATION = 25.0
TEMP_DATASET = 'ronanarraig/multimed-trimmed-temp'
REVIEW_DIR = Path('tools/review/data')

TAG_PROMPT = """You are a medical NLP expert. Given a transcribed sentence, return JSON:
- "is_medical": true if substantively about medicine, health, biology, or clinical topics
- "medical_density": "high" (dense clinical/scientific jargon throughout), "medium" (some medical terms), "low" (barely medical), "none" (not medical)
- "entities": list of medical named entities. Each: {"text": str, "category": str, "char_start": int, "char_end": int}. Categories: drug, condition, procedure, anatomy, organisation, measurement.

Return ONLY valid JSON."""


# ── Sentence detection + audio trim ──────────────────────────────

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

        start_sec = wts[start_idx]['start']
        end_sec = wts[end_idx]['end']
        duration = end_sec - start_sec

        if duration < MIN_DURATION or duration > MAX_DURATION:
            continue
        if len(sent) < MIN_CHARS:
            continue

        results.append((sent, start_sec, end_sec))
        search_start = end_idx

    return results


def trim_audio_array(audio_dict, start_sec, end_sec):
    """Returns (numpy_array, sample_rate) for the trimmed segment."""
    raw = audio_dict.get('bytes')
    if not raw:
        return None, None
    try:
        array, sr = sf.read(io.BytesIO(raw))
        if array.ndim > 1:
            array = array.mean(axis=1)
        pad = int(0.05 * sr)
        s = max(0, int(start_sec * sr) - pad)
        e = min(len(array), int(end_sec * sr) + pad)
        return array[s:e].astype(np.float32), sr
    except Exception:
        return None, None


def array_to_wav_bytes(array, sr):
    out = io.BytesIO()
    sf.write(out, array, sr, format='WAV', subtype='PCM_16')
    return out.getvalue()


# ── Post-Gemini completeness check ───────────────────────────────

def is_complete_transcript(text):
    """True if text looks like a complete sentence/phrase (not partial)."""
    text = text.strip()
    if len(text) < MIN_CHARS:
        return False
    if not text[0].isupper():
        return False
    # Must end with sentence-final punctuation
    if not re.search(r'[.!?]["\')]*\s*$', text):
        return False
    return True


# ── Gemini Pro ASR ────────────────────────────────────────────────

def transcribe_with_gemini_pro(idx, audio_bytes):
    """Upload audio to Gemini Files API and transcribe with 2.5 Pro."""
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


# ── Gemini tagging ────────────────────────────────────────────────

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
        except Exception as e:
            if attempt == 2:
                return idx, {}
            time.sleep(2 ** attempt)


# ── Load dataset ──────────────────────────────────────────────────
print("Loading multimed-sentences-transcribed...")
ds = load_dataset('ronanarraig/multimed-sentences-transcribed', split='test', token=HF_TOKEN)
ds = ds.cast_column('audio', ds.features['audio'].__class__(decode=False))
chunks = list(ds)
print(f"  {len(chunks)} chunks")

# ── Step 1: NLTK sentence detection + audio trim ─────────────────
print("\nStep 1: NLTK sentence detection + trim...")
trimmed_rows = []
n_no_sentence = 0

for i, row in enumerate(chunks):
    sents = find_clean_sentences(row['text'], row.get('word_timestamps', '[]'))
    if not sents:
        n_no_sentence += 1
        continue
    for sent_text, start_sec, end_sec in sents:
        array, sr = trim_audio_array(row['audio'], start_sec, end_sec)
        if array is None:
            continue
        trimmed_rows.append({
            'source_file': row.get('source_file', ''),
            'whisper_chunk': row['text'],
            'whisper_sentence': sent_text,
            'start_sec': round(start_sec, 3),
            'end_sec': round(end_sec, 3),
            'duration': round(end_sec - start_sec, 2),
            'audio_array': array,
            'sr': sr,
        })

print(f"  {n_no_sentence} chunks dropped (no clean sentence)")
print(f"  {len(trimmed_rows)} trimmed sentences")

# ── Step 2: Push trimmed clips to HF ─────────────────────────────
print(f"\nStep 2: Pushing {len(trimmed_rows)} trimmed clips to {TEMP_DATASET}...")
import pyarrow as pa
import pyarrow.parquet as pq
import tempfile

audio_bytes_list, texts, source_files, whisper_chunks, durations = [], [], [], [], []
for r in trimmed_rows:
    audio_bytes_list.append(array_to_wav_bytes(r['audio_array'], r['sr']))
    texts.append(r['whisper_sentence'])
    source_files.append(r['source_file'])
    whisper_chunks.append(r['whisper_chunk'])
    durations.append(r['duration'])

audio_struct = pa.array(
    [{'bytes': b, 'path': p} for b, p in zip(audio_bytes_list, source_files)],
    type=pa.struct([pa.field('bytes', pa.binary()), pa.field('path', pa.string())])
)
table = pa.table({
    'audio': audio_struct,
    'text': pa.array(texts),
    'source_file': pa.array(source_files),
    'whisper_chunk': pa.array(whisper_chunks),
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
print(f"  Pushed {len(trimmed_rows)} rows")

# ── Step 3: Gemini 2.5 Pro ASR (direct API, audio only) ──────────
# Note: Studio router model eval broken as of 2026-04-03 (bug filed: 8788af6d)
print(f"\nStep 3: Transcribing {len(trimmed_rows)} clips with Gemini 2.5 Pro ({N_ASR_THREADS} threads)...")
asr_results = [None] * len(trimmed_rows)

with ThreadPoolExecutor(max_workers=N_ASR_THREADS) as executor:
    futures = {executor.submit(transcribe_with_gemini_pro, i, array_to_wav_bytes(r['audio_array'], r['sr'])): i
               for i, r in enumerate(trimmed_rows)}
    done = 0
    for future in as_completed(futures):
        i, text = future.result()
        asr_results[i] = text
        done += 1
        if done % 20 == 0 or done == len(trimmed_rows):
            print(f"  {done}/{len(trimmed_rows)}")

n_asr_failed = sum(1 for t in asr_results if not t)
print(f"  {n_asr_failed} failed ASR calls")

# ── Step 4: Post-NLTK completeness check ─────────────────────────
print(f"\nStep 4: Post-NLTK completeness check...")
complete_rows = []
n_incomplete = 0

for r, prediction in zip(trimmed_rows, asr_results):
    if not prediction:
        n_incomplete += 1
        continue
    if not is_complete_transcript(prediction):
        n_incomplete += 1
        continue
    r['gemini_transcript'] = prediction
    complete_rows.append(r)

print(f"  {n_incomplete} dropped (no prediction or incomplete Gemini output)")
print(f"  {len(complete_rows)} complete transcripts")

# ── Step 5: Tag with Gemini Flash ────────────────────────────────
print(f"\nStep 5: Tagging {len(complete_rows)} rows with Gemini Flash ({N_TAG_THREADS} threads)...")
tag_results = [None] * len(complete_rows)

with ThreadPoolExecutor(max_workers=N_TAG_THREADS) as executor:
    futures = {executor.submit(tag_transcript, i, r['gemini_transcript']): i
               for i, r in enumerate(complete_rows)}
    done = 0
    for future in as_completed(futures):
        i, tags = future.result()
        tag_results[i] = tags
        done += 1
        if done % 50 == 0 or done == len(complete_rows):
            print(f"  {done}/{len(complete_rows)}")

# Merge tags
for r, tags in zip(complete_rows, tag_results):
    tags = tags or {}
    if isinstance(tags, list):
        tags = tags[0] if tags and isinstance(tags[0], dict) else {}
    r['is_medical'] = tags.get('is_medical', False)
    r['medical_density'] = tags.get('medical_density', 'none')
    r['entities'] = json.dumps(tags.get('entities', []))

# ── Drop log ──────────────────────────────────────────────────────
n_medical_high = sum(1 for r in complete_rows if r['medical_density'] == 'high')
n_medical_any = sum(1 for r in complete_rows if r['is_medical'])
density_counts = {}
for r in complete_rows:
    density_counts[r['medical_density']] = density_counts.get(r['medical_density'], 0) + 1

print(f"\n{'='*60}")
print(f"DROP LOG")
print(f"{'='*60}")
print(f"Input chunks:                  {len(chunks)}")
print(f"  → no clean Whisper sentence: {n_no_sentence} dropped ({100*n_no_sentence/len(chunks):.0f}%)")
print(f"  → trimmed sentences:         {len(trimmed_rows)}")
print(f"  → incomplete Gemini output:  {n_incomplete} dropped ({100*n_incomplete/max(len(trimmed_rows),1):.0f}%)")
print(f"  → complete transcripts:      {len(complete_rows)}")
print(f"  → is_medical=True:           {n_medical_any} ({100*n_medical_any/max(len(complete_rows),1):.0f}%)")
print(f"  → medical_density=high:      {n_medical_high} kept ({100*n_medical_high/max(len(complete_rows),1):.0f}%)")
print(f"  density breakdown: {density_counts}")

# ── Export high-density rows to review UI ─────────────────────────
high_rows = [r for r in complete_rows if r['medical_density'] == 'high']
print(f"\nExporting {len(high_rows)} high-density rows to {REVIEW_DIR}...")
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
    })

(REVIEW_DIR / 'rows.json').write_text(json.dumps(rows_out, indent=2, ensure_ascii=False))
print(f"Exported {len(rows_out)} rows")

print("\nSample high-density rows:")
for r in rows_out[:8]:
    ents = [e['text'] for e in json.loads(r['entities'] or '[]')]
    print(f"  [{r['duration']:.1f}s] {r['transcript'][:85]}")
    if ents:
        print(f"    entities: {', '.join(ents[:5])}")

print(f"\nStart review UI:")
print(f"  uv run tools/review/server.py --data {REVIEW_DIR}")
print("\nDone.")
