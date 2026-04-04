#!/usr/bin/env python3
"""
Phase 1D: Push top-50 EKA hard rows to ronanarraig/eka-hard-public.
Selects rows 1-50 by difficulty_rank from rows.json.
"""
import os, json
import pyarrow as pa
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')
from datasets import Dataset, Audio

HF_TOKEN   = os.environ['HF_TOKEN']
ROWS_JSON  = 'tools/review/data-eka/rows.json'
AUDIO_DIR  = 'tools/review/data-eka/audio'
HF_DATASET = 'ronanarraig/eka-hard-public'

print('Loading rows.json...')
rows = json.load(open(ROWS_JSON))
top50 = [r for r in rows if r['difficulty_rank'] <= 50]
top50.sort(key=lambda r: r['difficulty_rank'])
print(f'  {len(top50)} rows selected (ranks 1-50)')
print(f'  CER range: {top50[-1]["median_cer"]:.3f} – {top50[0]["median_cer"]:.3f}')

# Load audio bytes
audio_bytes_list, texts, ids, durations, median_cers, difficulty_ranks = [], [], [], [], [], []
for r in top50:
    idx = r['id']
    path = f'{AUDIO_DIR}/{idx}.wav'
    audio_bytes_list.append(open(path, 'rb').read())
    texts.append(r['transcript'])
    ids.append(str(idx))
    durations.append(float(r.get('duration', 0)))
    median_cers.append(float(r['median_cer']))
    difficulty_ranks.append(int(r['difficulty_rank']))

print(f'  Loaded {len(audio_bytes_list)} audio files')

# Build dataset
audio_col = pa.array(audio_bytes_list, type=pa.binary())
table = pa.table({
    'id':               pa.array(ids),
    'audio':            audio_col,
    'text':             pa.array(texts),
    'duration':         pa.array(durations, type=pa.float32()),
    'median_cer':       pa.array(median_cers, type=pa.float32()),
    'difficulty_rank':  pa.array(difficulty_ranks, type=pa.int32()),
})
ds = Dataset(table).cast_column('audio', Audio(sampling_rate=16000))
print(f'  Dataset built: {len(ds)} rows, features: {list(ds.features.keys())}')

print(f'\nPushing to {HF_DATASET}...')
ds.push_to_hub(HF_DATASET, split='test', private=True, token=HF_TOKEN)
print(f'Done! Pushed {len(ds)} rows to {HF_DATASET} (split=test)')
print('\nSample rows:')
for r in top50[:5]:
    print(f'  [{r["difficulty_rank"]}] cer={r["median_cer"]:.3f} {r["transcript"][:80]}')
