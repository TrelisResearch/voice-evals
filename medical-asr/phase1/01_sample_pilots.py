#!/usr/bin/env python3
"""
Step 1: Sample 50 rows from EKA and MultiMed EN test splits.
- EKA: stratified by recording_context; convert entity format; keep embedded audio
- MultiMed EN: random 50; no entity annotations yet (done in step 3)
Saves intermediate JSONs for step 3 (entity extraction).
"""
import os, json, random
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')
load_dotenv('/home/claude/TR/voice-evals/.env', override=False)

from datasets import load_dataset, Dataset, Audio, Features, Value
from collections import defaultdict
import pathlib

HF_TOKEN = os.environ['HF_TOKEN']
random.seed(42)
OUT = pathlib.Path('medical-asr/phase1/tmp')
OUT.mkdir(exist_ok=True)

# ─────────────────────────────────────────────
# EKA: stratified sample, convert entity format
# ─────────────────────────────────────────────
print("=== EKA: loading EN test ===")
eka_raw = load_dataset(
    'ekacare/eka-medical-asr-evaluation-dataset', 'en',
    split='test', token=HF_TOKEN
)
eka_raw = eka_raw.cast_column('audio', eka_raw.features['audio'].__class__(decode=False))

def convert_eka_entities(raw_str):
    """Convert EKA format [text, type, [[s,e]]] → standard dict list."""
    try:
        ents = json.loads(raw_str) if isinstance(raw_str, str) else raw_str
    except Exception:
        return []
    result = []
    for e in ents:
        if isinstance(e, list) and len(e) >= 3:
            text, etype, spans = e[0], e[1], e[2]
            for span in spans:
                result.append({
                    "text": text,
                    "category": etype,
                    "char_start": span[0],
                    "char_end": span[1]
                })
    return result

# Stratified by recording_context
# conversation:110, narration_entity:2206, narration_sentence:1303
# Target: ~5 conversation, ~30 narration_entity, ~15 narration_sentence
# Filter: duration >= 2s, must have entities
by_context = defaultdict(list)
for i, row in enumerate(eka_raw):
    if row['duration'] < 2.0:
        continue
    ents = convert_eka_entities(row['medical_entities'])
    if not ents:
        continue
    by_context[row['recording_context']].append(i)

targets = {'conversation': 5, 'narration_entity': 30, 'narration_sentence': 15}
selected_idx = []
for ctx, n in targets.items():
    pool = by_context[ctx]
    selected_idx.extend(random.sample(pool, min(n, len(pool))))

print(f"EKA selected: {len(selected_idx)} rows")
eka_sample = eka_raw.select(selected_idx)

# Build output rows (keep audio bytes, convert entities)
eka_rows = []
for row in eka_sample:
    ents = convert_eka_entities(row['medical_entities'])
    eka_rows.append({
        'text': row['text'],
        'audio': row['audio'],  # raw bytes dict when decode=False
        'duration': float(row['duration']),
        'speaker': row['speaker'],
        'recording_context': row['recording_context'],
        'type_concept': row['type_concept'],
        'entities': json.dumps(ents),
        'source': 'ekacare/eka-medical-asr-evaluation-dataset',
    })

# Save texts for entity extraction (step 3 doesn't need to re-extract for EKA)
with open(OUT / 'eka_pilot_rows.json', 'w') as f:
    # save without audio for inspection
    json.dump([{k: v for k, v in r.items() if k != 'audio'} for r in eka_rows], f, indent=2)
print(f"EKA pilot: {len(eka_rows)} rows saved to tmp/eka_pilot_rows.json")
print(f"  contexts: { {r['recording_context'] for r in eka_rows} }")

# ─────────────────────────────────────────────
# MultiMed EN: random 50
# ─────────────────────────────────────────────
print("\n=== MultiMed EN: loading test ===")
mm_raw = load_dataset('leduckhai/MultiMed', 'English', split='test', token=HF_TOKEN)
mm_raw = mm_raw.cast_column('audio', mm_raw.features['audio'].__class__(decode=False))
print(f"Columns: {mm_raw.column_names}")

# Filter: prefer longer utterances (>3s if duration col exists, else by text length)
has_duration = 'duration' in mm_raw.column_names
if has_duration:
    pool = [i for i, d in enumerate(mm_raw['duration']) if d >= 3.0]
else:
    pool = [i for i, t in enumerate(mm_raw['text']) if len(t.split()) >= 5]

selected_mm = random.sample(pool, min(50, len(pool)))
mm_sample = mm_raw.select(selected_mm)

# Save texts for LLM entity extraction
mm_texts = [{'idx': i, 'text': row['text']} for i, row in enumerate(mm_sample)]
with open(OUT / 'multimed_pilot_texts.json', 'w') as f:
    json.dump(mm_texts, f, indent=2)
print(f"MultiMed EN pilot: {len(mm_sample)} rows")
print(f"Sample texts:")
for t in mm_texts[:3]:
    print(f"  [{t['idx']}] {t['text'][:100]}")

# Save dataset for later push step
mm_sample.save_to_disk(str(OUT / 'multimed_pilot_ds'))
print("MultiMed pilot saved to tmp/multimed_pilot_ds")

print("\nStep 1 done.")
