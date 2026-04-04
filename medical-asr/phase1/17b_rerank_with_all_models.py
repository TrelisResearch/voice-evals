#!/usr/bin/env python3
"""
Re-rank top-100 using all 3 model result datasets.
The 3 eval jobs already completed; we just couldn't download 2 due to disk space.
Downloads Canary + Voxtral results, merges with Whisper, re-ranks.

Text comes from Whisper result dataset's 'reference' column (= original Gemini transcript).
Audio comes from local tools/review/data-eka/audio/{idx}.wav files.
This avoids re-downloading eka-hard-candidates (415MB).
"""
import os, json, statistics
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')
from datasets import load_dataset, Audio

HF_TOKEN   = os.environ['HF_TOKEN']
REVIEW_DIR = 'tools/review/data-eka'
AUDIO_DIR  = f'{REVIEW_DIR}/audio'

# The 3 result dataset IDs from the completed eval jobs
RESULT_DATASETS = {
    'openai/whisper-large-v3':        'ronanarraig/eval-whisper-large-v3-eka-hard-candidates-20260404-1656',
    'nvidia/canary-1b-v2':            'ronanarraig/eval-canary-1b-v2-eka-hard-candidates-20260404-1656',
    'mistralai/Voxtral-Mini-3B-2507': 'ronanarraig/eval-Voxtral-Mini-3B-2507-eka-hard-candidates-20260404-1655',
}

print('Loading result datasets...')
per_sample_cer = {}   # index → {model: cer}
index_to_text  = {}   # index → reference transcript (from Whisper results)

for model_id, ds_id in RESULT_DATASETS.items():
    try:
        print(f'  Loading {ds_id}...')
        rds = load_dataset(ds_id, split='test', token=HF_TOKEN)
        rds_plain = rds.cast_column('audio', Audio(decode=False)) if 'audio' in rds.column_names else rds
        print(f'  {len(rds_plain)} rows, cols: {rds_plain.column_names}')
        for i, row in enumerate(rds_plain):
            cer = row.get('cer')
            if cer is not None:
                if i not in per_sample_cer:
                    per_sample_cer[i] = {}
                per_sample_cer[i][model_id] = float(cer)
            # Capture reference text from Whisper results (ground truth transcript)
            if model_id == 'openai/whisper-large-v3':
                ref = row.get('reference') or row.get('text') or row.get('transcript') or ''
                index_to_text[i] = ref
    except Exception as e:
        print(f'  FAILED: {e}')
        import traceback; traceback.print_exc()

print(f'\nPer-sample CER loaded for {len(per_sample_cer)} rows')
print(f'Reference texts captured: {len(index_to_text)}')
models_per_row = [len(v) for v in per_sample_cer.values()]
if models_per_row:
    print(f'Models per row: min={min(models_per_row)} max={max(models_per_row)} avg={sum(models_per_row)/len(models_per_row):.1f}')

# Rank all 920 by median CER across available models
ranked = []
for idx, model_cers in per_sample_cer.items():
    cers = list(model_cers.values())
    median_cer = statistics.median(cers)
    ranked.append((median_cer, idx, model_cers))
ranked.sort(reverse=True)

print(f'\nCER range: {ranked[-1][0]:.3f} – {ranked[0][0]:.3f}')
print(f'Top 10: {[f"{c:.3f}" for c,_,_ in ranked[:10]]}')

# Build top-100 rows using local audio + reference text
print('\nBuilding top-100 rows from local audio + reference text...')
top100_rows = []
missing_audio = 0
missing_text = 0

for rank, (median_cer, idx, model_cers) in enumerate(ranked[:100]):
    audio_path = f'{AUDIO_DIR}/{idx}.wav'
    text = index_to_text.get(idx, '')
    if not os.path.exists(audio_path):
        missing_audio += 1
    if not text:
        missing_text += 1

    # Try to get duration from WAV file
    duration = 0.0
    if os.path.exists(audio_path):
        try:
            import soundfile as sf
            info = sf.info(audio_path)
            duration = info.duration
        except Exception:
            pass

    row = {
        'id': idx,
        'source_file': f'{idx}.wav',
        'text': text,
        'duration': duration,
        'transcript': text,
        'is_medical': True,
        'medical_density': 'high',
        'entities': '[]',
        'reviewed': False,
        'dropped': False,
        'median_cer': median_cer,
        'model_cers': model_cers,
        'difficulty_rank': rank + 1,
    }
    top100_rows.append(row)

print(f'Missing audio: {missing_audio}/100, missing text: {missing_text}/100')

out_path = f'{REVIEW_DIR}/rows.json'
json.dump(top100_rows, open(out_path, 'w'), indent=2)
print(f'\nSaved {len(top100_rows)} rows to {out_path}')
print(f'Top-100 median CER range: {top100_rows[-1]["median_cer"]:.3f} – {top100_rows[0]["median_cer"]:.3f}')
print('\nSample rows:')
for r in top100_rows[:5]:
    print(f'  [{r["difficulty_rank"]}] cer={r["median_cer"]:.3f} {r["transcript"][:70]}')
print('DONE')
