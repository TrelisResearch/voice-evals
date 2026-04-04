#!/usr/bin/env python3
"""
Phase 2 Step 3b: Merge TTS outputs from all 4 voices into one dataset.
Downloads from the reprocessed HF datasets (min_chunk_duration=1).
Pushes merged dataset to ronanarraig/medical-terms-tts-raw.
"""
import os, json
import pyarrow as pa
from pathlib import Path
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')
from datasets import load_dataset, Dataset, Audio

HF_TOKEN = os.environ['HF_TOKEN']
SENTENCES_FILE = 'medical-asr/phase2/tmp/sentences.json'
AUDIO_DIR = Path('medical-asr/phase2/tmp/audio')
HF_DATASET = 'ronanarraig/medical-terms-tts-raw'

# Reprocessed datasets (v2 = with min_chunk_duration=1)
VOICE_DATASETS = {
    'af_heart':    'ronanarraig/medical-terms-tts-af_heart-v2',
    'am_michael':  'ronanarraig/medical-terms-tts-am_michael-v2',
    'bf_emma':     'ronanarraig/medical-terms-tts-bf_emma-v2',
    'af_bella':    'ronanarraig/medical-terms-tts-af_bella',
}

sentences = json.load(open(SENTENCES_FILE))
print(f'Loaded {len(sentences)} sentences')
AUDIO_DIR.mkdir(parents=True, exist_ok=True)

# Load checkpoint for voice→sentence mapping
jobs = json.load(open('medical-asr/phase2/tmp/tts_jobs.json'))

all_rows = []
for voice, ds_id in VOICE_DATASETS.items():
    print(f'\n[{voice}] Loading {ds_id}...')
    try:
        ds = load_dataset(ds_id, split='test', token=HF_TOKEN)
        ds_plain = ds.cast_column('audio', Audio(decode=False)) if 'audio' in ds.column_names else ds
        print(f'  {len(ds_plain)} rows, cols: {ds_plain.column_names}')

        voice_sentences = jobs[voice].get('sentences', [])
        for hf_idx, row in enumerate(ds_plain):
            if hf_idx < len(voice_sentences):
                orig_idx, orig_text = voice_sentences[hf_idx]
            else:
                orig_idx = -1
                orig_text = row.get('text', '')

            audio_bytes = (row.get('audio') or {}).get('bytes', b'')
            if audio_bytes:
                wav_path = AUDIO_DIR / f'{orig_idx}_{voice}.wav'
                wav_path.write_bytes(audio_bytes)

            text = sentences[orig_idx]['text'] if 0 <= orig_idx < len(sentences) else orig_text
            keyword = sentences[orig_idx]['keyword'] if 0 <= orig_idx < len(sentences) else ''
            category = sentences[orig_idx]['category'] if 0 <= orig_idx < len(sentences) else ''

            all_rows.append({
                'idx': orig_idx,
                'voice': voice,
                'keyword': keyword,
                'category': category,
                'text': text,
                'audio_path': str(AUDIO_DIR / f'{orig_idx}_{voice}.wav') if audio_bytes else '',
                'duration': float(row.get('speech_duration', row.get('duration', 0))),
            })
    except Exception as e:
        print(f'  FAILED: {e}')
        import traceback; traceback.print_exc()

print(f'\nTotal rows: {len(all_rows)}')
# Check coverage
idx_set = set(r['idx'] for r in all_rows if r['idx'] >= 0)
print(f'Unique sentence indices: {len(idx_set)}/52')
missing = set(range(52)) - idx_set
if missing:
    print(f'Missing indices: {sorted(missing)}')

# Push merged dataset
print(f'\nPushing to {HF_DATASET}...')
audio_bytes_list, texts, voices, keywords, categories, durations = [], [], [], [], [], []
for r in all_rows:
    ap = r['audio_path']
    if ap and os.path.exists(ap):
        audio_bytes_list.append(open(ap, 'rb').read())
    else:
        audio_bytes_list.append(b'')
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
print(f'  Pushed to {HF_DATASET}')

json.dump(all_rows, open('medical-asr/phase2/tmp/sentences_with_audio.json', 'w'), indent=2)
print('\nDONE')
