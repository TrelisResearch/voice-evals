#!/usr/bin/env python3
"""Show 10 EKA samples around the Otsu threshold (0.589) for inspection."""
import os, json
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')
from datasets import load_dataset

HF_TOKEN = os.environ['HF_TOKEN']

# Load per-sample whisper results
results_ds = load_dataset('ronanarraig/eval-whisper-v3-eka-500-20260402-1447', split='test', token=HF_TOKEN)
results_ds = results_ds.cast_column('audio', results_ds.features['audio'].__class__(decode=False))

cer_col = next(c for c in results_ds.column_names if 'cer' in c.lower() and 'entity' not in c.lower())
hyp_col = next((c for c in results_ds.column_names if 'hypothesis' in c.lower() or 'transcription' in c.lower() or 'predicted' in c.lower()), None)
print(f"Columns: {results_ds.column_names}")
print(f"CER col: {cer_col}, Hyp col: {hyp_col}\n")

# Get rows around 0.589 threshold
TARGET = 0.589
WINDOW = 0.15

rows = []
for row in results_ds:
    cer = float(row[cer_col])
    if abs(cer - TARGET) <= WINDOW:
        rows.append(row)

# Sort by CER, take 10 closest to threshold
rows.sort(key=lambda r: abs(float(r[cer_col]) - TARGET))
rows = rows[:10]
rows.sort(key=lambda r: float(r[cer_col]))

ref_col = 'reference' if 'reference' in results_ds.column_names else 'text'
print(f"{'CER':>6}  {'REFERENCE':<60}  HYPOTHESIS")
print("-" * 120)
for r in rows:
    cer = float(r[cer_col])
    ref = r[ref_col][:58]
    hyp = (r[hyp_col][:58] if hyp_col and r.get(hyp_col) else '—')
    print(f"{cer:>6.3f}  {ref:<60}  {hyp}")
