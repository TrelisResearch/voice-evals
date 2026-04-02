#!/usr/bin/env python3
"""Inspect EKA dataset entity format and recording context distribution."""
import os, json
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')
load_dotenv('/home/claude/TR/voice-evals/.env', override=False)

from datasets import load_dataset
from collections import Counter

HF_TOKEN = os.environ['HF_TOKEN']

print("Loading EKA EN test split...")
ds = load_dataset('ekacare/eka-medical-asr-evaluation-dataset', 'en', split='test', token=HF_TOKEN, trust_remote_code=False)
ds = ds.cast_column('audio', ds.features['audio'].__class__(decode=False))
print(f"Rows: {len(ds)}")
print(f"Columns: {ds.column_names}")
print(f"Features: {ds.features}\n")

# Recording context distribution
contexts = Counter(ds['recording_context'])
print(f"recording_context: {dict(contexts)}")

# Speaker distribution
speakers = Counter(ds['speaker'])
print(f"Unique speakers: {len(speakers)}")
print(f"Top 5 speakers: {dict(speakers.most_common(5))}")

# Type concept
type_concepts = Counter(ds['type_concept'])
print(f"type_concept: {dict(type_concepts)}")

# Duration stats
durations = ds['duration']
import statistics
print(f"\nDuration (s): min={min(durations):.1f} max={max(durations):.1f} median={statistics.median(durations):.1f} mean={statistics.mean(durations):.1f}")

# Entity annotation format
print("\n=== Entity annotation samples ===")
entity_type_counts = Counter()
rows_with_entities = 0
for i, row in enumerate(ds):
    entities = row['medical_entities']
    if isinstance(entities, str):
        try:
            entities = json.loads(entities)
        except Exception:
            entities = []
    if not entities:
        entities = []
    # entities may be a list of dicts or a list of lists — inspect raw first
    if i < 3:
        print(f"\nRow {i} raw medical_entities: {repr(row['medical_entities'])[:300]}")
        print(f"  text: {row['text'][:100]}")
        continue

    if entities:
        rows_with_entities += 1
        for e in entities:
            if isinstance(e, dict):
                entity_type_counts[e.get('type') or e.get('category', 'unknown')] += 1
            elif isinstance(e, list) and len(e) >= 2:
                entity_type_counts[str(e[1])] += 1

print(f"\nRows with entities: {rows_with_entities}/{len(ds)-3}")
print(f"Entity type distribution: {dict(entity_type_counts.most_common())}")
