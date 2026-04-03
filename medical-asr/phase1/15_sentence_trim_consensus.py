#!/usr/bin/env python3
"""
Phase 1C: MultiMed sentence extraction + Gemini 3 Flash consensus + tagging.

Pipeline per chunk:
1. NLTK sentence detection on Whisper text → clean inner sentences
2. Trim audio via word timestamps
3. Gemini 3 Flash (trimmed audio + full Whisper chunk context) →
   transcript + is_medical + medical_density + entities (single call)
4. Keep medical_density == high

Reports drop counts at each step. Exports to tools/review/data for inspection.
"""
import os, json, re, io, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')

import nltk
import numpy as np
import soundfile as sf
from google import genai
from google.genai import types as gentypes
from datasets import load_dataset

HF_TOKEN = os.environ['HF_TOKEN']
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
client = genai.Client(api_key=GEMINI_API_KEY)

N_THREADS = 100
MIN_DURATION = 3.0
MIN_CHARS = 40
MAX_DURATION = 25.0
REVIEW_DIR = Path('tools/review/data')

PROMPT = """You are a medical transcription expert. Listen carefully to the audio clip and return a JSON object.

Context: this audio is a trimmed segment from a longer chunk. The full Whisper transcript of the source chunk is:
  "{whisper_chunk}"

The audio corresponds approximately to this sentence within that chunk:
  "{whisper_sentence}"

Return JSON with these fields:
- "transcript": accurate transcript of what is spoken in the audio (only what is in the audio, not the full chunk)
- "is_medical": true if the content is substantively about medicine, health, biology, or clinical topics
- "medical_density": "high" (dense clinical/scientific terminology throughout), "medium" (some medical terms), "low" (barely medical), or "none" (not medical)
- "entities": list of medical named entities in the transcript. Each: {{"text": str, "category": str, "char_start": int, "char_end": int}}. Categories: drug, condition, procedure, anatomy, organisation, measurement.

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


def trim_audio(audio_dict, start_sec, end_sec):
    raw = audio_dict.get('bytes')
    if not raw:
        return None
    try:
        array, sr = sf.read(io.BytesIO(raw))
        if array.ndim > 1:
            array = array.mean(axis=1)
        pad = int(0.05 * sr)
        s = max(0, int(start_sec * sr) - pad)
        e = min(len(array), int(end_sec * sr) + pad)
        out = io.BytesIO()
        sf.write(out, array[s:e].astype(np.float32), sr, format='WAV', subtype='PCM_16')
        return out.getvalue()
    except Exception as ex:
        return None


# ── Gemini combined call ──────────────────────────────────────────

def gemini_call(wav_bytes, whisper_chunk, whisper_sentence):
    prompt = PROMPT.format(
        whisper_chunk=whisper_chunk[:500],  # truncate very long chunks
        whisper_sentence=whisper_sentence,
    )
    for attempt in range(3):
        try:
            parts = []
            if wav_bytes:
                parts.append(gentypes.Part.from_bytes(data=wav_bytes, mime_type='audio/wav'))
            parts.append(prompt)
            response = client.models.generate_content(
                model='gemini-3-flash-preview',
                contents=parts,
                config=gentypes.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type='application/json',
                ),
            )
            return json.loads(response.text.strip())
        except Exception as e:
            if attempt == 2:
                print(f"    Gemini failed: {str(e)[:80]}")
                return None
            time.sleep(2 ** attempt)


def process_chunk(chunk_idx, row):
    sents = find_clean_sentences(row['text'], row.get('word_timestamps', '[]'))
    if not sents:
        return chunk_idx, 'no_sentences', []

    results = []
    for sent_text, start_sec, end_sec in sents:
        wav = trim_audio(row['audio'], start_sec, end_sec)
        tags = gemini_call(wav, row['text'], sent_text)
        if tags is None:
            tags = {'transcript': sent_text, 'is_medical': False, 'medical_density': 'none', 'entities': []}

        results.append({
            'chunk_idx': chunk_idx,
            'source_file': row.get('source_file', ''),
            'whisper_chunk': row['text'],
            'whisper_sentence': sent_text,
            'start_sec': round(start_sec, 3),
            'end_sec': round(end_sec, 3),
            'duration': round(end_sec - start_sec, 2),
            'transcript': tags.get('transcript', sent_text),
            'is_medical': tags.get('is_medical', False),
            'medical_density': tags.get('medical_density', 'none'),
            'entities': json.dumps(tags.get('entities', [])),
            'wav': wav,
        })
    return chunk_idx, 'ok', results


# ── Load dataset ──────────────────────────────────────────────────
print("Loading multimed-sentences-transcribed...")
ds = load_dataset('ronanarraig/multimed-sentences-transcribed', split='test', token=HF_TOKEN)
ds = ds.cast_column('audio', ds.features['audio'].__class__(decode=False))
chunks = list(ds)
print(f"  {len(chunks)} chunks")

# ── Process all chunks ────────────────────────────────────────────
print(f"\nProcessing {len(chunks)} chunks ({N_THREADS} threads)...")

all_results = {}
with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
    futures = {executor.submit(process_chunk, i, row): i for i, row in enumerate(chunks)}
    done = 0
    for future in as_completed(futures):
        idx, status, results = future.result()
        all_results[idx] = (status, results)
        done += 1
        if done % 50 == 0 or done == len(chunks):
            print(f"  {done}/{len(chunks)}")

# ── Drop log ──────────────────────────────────────────────────────
n_chunks = len(chunks)
n_no_sentence = sum(1 for s, _ in all_results.values() if s == 'no_sentences')
all_sentences = [r for _, (s, results) in all_results.items() for r in results]
n_sentences = len(all_sentences)
n_medical_high = sum(1 for r in all_sentences if r['medical_density'] == 'high')
n_medical_any = sum(1 for r in all_sentences if r['is_medical'])

print(f"\n{'='*60}")
print(f"DROP LOG")
print(f"{'='*60}")
print(f"Input chunks:              {n_chunks}")
print(f"  → no clean sentence:     {n_no_sentence} dropped ({100*n_no_sentence/n_chunks:.0f}%)")
print(f"  → sentences extracted:   {n_sentences} from {n_chunks - n_no_sentence} chunks")
print(f"  → is_medical=True:       {n_medical_any} ({100*n_medical_any/max(n_sentences,1):.0f}%)")
print(f"  → medical_density=high:  {n_medical_high} kept ({100*n_medical_high/max(n_sentences,1):.0f}%)")

density_counts = {}
for r in all_sentences:
    density_counts[r['medical_density']] = density_counts.get(r['medical_density'], 0) + 1
print(f"  density breakdown: {density_counts}")

# ── Export high-density rows to review UI ─────────────────────────
high_rows = [r for r in all_sentences if r['medical_density'] == 'high']

print(f"\nExporting {len(high_rows)} high-density rows to {REVIEW_DIR}...")
audio_dir = REVIEW_DIR / 'audio'
audio_dir.mkdir(parents=True, exist_ok=True)

rows_out = []
for i, item in enumerate(high_rows):
    if item['wav']:
        (audio_dir / f'{i}.wav').write_bytes(item['wav'])
    row_data = {k: v for k, v in item.items() if k != 'wav'}
    row_data['id'] = i
    rows_out.append(row_data)

(REVIEW_DIR / 'rows.json').write_text(json.dumps(rows_out, indent=2, ensure_ascii=False))
print(f"Exported {len(rows_out)} rows")

print("\nSample high-density rows:")
for r in rows_out[:8]:
    ents = [e['text'] for e in json.loads(r['entities'] or '[]')]
    print(f"  [{r['duration']:.1f}s] {r['transcript'][:80]}")
    if ents:
        print(f"    entities: {', '.join(ents[:5])}")

print(f"\nStart review UI:")
print(f"  uv run tools/review/server.py --data {REVIEW_DIR}")
print("\nDone.")
