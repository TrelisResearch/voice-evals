#!/usr/bin/env python3
"""Plot CER histogram for multimed-sentences-transcribed Gemini eval results."""
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')
from datasets import load_dataset

HF_TOKEN = os.environ['HF_TOKEN']

# Load the Gemini eval results dataset (pushed by the eval job)
# The eval job pushed results to a dataset — we need to find it.
# We know: job d1ca7c5e, and the otsu dataset has the gemini_cer column
print("Loading multimed-sentences-otsu for CER values...")
otsu_ds = load_dataset('ronanarraig/multimed-sentences-otsu', split='test', token=HF_TOKEN)
otsu_ds = otsu_ds.cast_column('audio', otsu_ds.features['audio'].__class__(decode=False))
print(f"  {len(otsu_ds)} rows, columns: {otsu_ds.column_names}")

cer_filtered = [row['gemini_cer'] for row in otsu_ds]

print(f"\nFiltered CER stats (0.05–0.553):")
print(f"  n={len(cer_filtered)}, min={min(cer_filtered):.3f}, max={max(cer_filtered):.3f}, mean={np.mean(cer_filtered):.3f}")

# Try to get the full distribution from the eval results dataset
# The job result dataset is typically named with the eval job id
eval_result_id = None
try:
    from huggingface_hub import HfApi
    api = HfApi(token=HF_TOKEN)
    datasets = list(api.list_datasets(author='ronanarraig', token=HF_TOKEN))
    for d in datasets:
        if 'multimed' in d.id.lower() and 'eval' in d.id.lower():
            print(f"  Found candidate: {d.id}")
            eval_result_id = d.id
except Exception as e:
    print(f"  Could not search datasets: {e}")

# Try loading full eval results
full_cer = None
if eval_result_id:
    try:
        full_ds = load_dataset(eval_result_id, split='test', token=HF_TOKEN)
        full_ds = full_ds.cast_column('audio', full_ds.features['audio'].__class__(decode=False))
        cer_col = next(c for c in full_ds.column_names if 'cer' in c.lower() and 'entity' not in c.lower())
        full_cer = [float(row[cer_col]) for row in full_ds]
        print(f"  Full eval dataset: {len(full_cer)} rows")
    except Exception as e:
        print(f"  Could not load full eval results: {e}")

# Plot
fig, axes = plt.subplots(1, 2 if full_cer else 1, figsize=(14 if full_cer else 7, 5))
if full_cer is None:
    axes = [axes]

OTSU = 0.553
CER_FLOOR = 0.05

if full_cer:
    ax = axes[0]
    ax.hist(full_cer, bins=60, color='steelblue', edgecolor='white', linewidth=0.3)
    ax.axvline(OTSU, color='red', linestyle='--', linewidth=1.5, label=f'Otsu = {OTSU:.3f}')
    ax.axvline(CER_FLOOR, color='orange', linestyle='--', linewidth=1.5, label=f'Floor = {CER_FLOOR:.2f}')
    ax.set_xlabel('CER')
    ax.set_ylabel('Count')
    ax.set_title('Full CER distribution (501 rows)\nGemini Pro eval on multimed-sentences-transcribed')
    ax.legend()

# Filtered distribution
ax = axes[-1]
ax.hist(cer_filtered, bins=40, color='seagreen', edgecolor='white', linewidth=0.3)
ax.axvline(OTSU, color='red', linestyle='--', linewidth=1.5, label=f'Otsu ceiling = {OTSU:.3f}')
ax.set_xlabel('CER')
ax.set_ylabel('Count')
ax.set_title(f'Filtered CER distribution ({len(cer_filtered)} rows)\nCER 0.05–0.553')
ax.legend()

plt.suptitle('MultiMed sentences — Gemini Pro CER (Whisper large-v3 transcripts as reference)', fontsize=11)
plt.tight_layout()

out_path = 'medical-asr/reports/multimed_cer_histogram.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")
