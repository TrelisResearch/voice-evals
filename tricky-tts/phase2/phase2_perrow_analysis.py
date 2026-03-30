"""
Phase 2 Step 5: Download per-row results from HF eval datasets and compute
per-row difficulty metrics for median-of-N filtering.
"""

import os, json
from pathlib import Path

env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

HF_TOKEN = os.environ["HF_TOKEN"]

import requests
import numpy as np
from huggingface_hub import hf_hub_download

MODEL_DATASETS = {
    "elevenlabs": "ronanarraig/tricky-tts-v2-eval-elevenlabs",
    "cartesia": "ronanarraig/tricky-tts-v2-eval-cartesia",
    "gpt4o_mini": "ronanarraig/tricky-tts-v2-eval-gpt4o-mini",
    "kokoro": "ronanarraig/tricky-tts-v2-eval-kokoro",
    "orpheus": "ronanarraig/tricky-tts-v2-eval-orpheus",
}

def load_parquet_no_audio(repo_id: str, token: str) -> list[dict]:
    """Load parquet file from HF dataset, excluding audio column."""
    import pyarrow.parquet as pq

    # Download the parquet file
    local_path = hf_hub_download(
        repo_id=repo_id,
        filename="data/train-00000-of-00001.parquet",
        repo_type="dataset",
        token=token,
    )
    table = pq.read_table(local_path)
    # Drop audio column to avoid torch dependency
    cols_to_keep = [c for c in table.column_names if c != "generated_audio"]
    table = table.select(cols_to_keep)
    return table.to_pydict()

# Load per-row data from each model
print("Loading per-row data from HF datasets...", flush=True)
model_rows = {}

for model_key, dataset_id in MODEL_DATASETS.items():
    print(f"  {model_key}: {dataset_id}", flush=True)
    try:
        data = load_parquet_no_audio(dataset_id, HF_TOKEN)
        n = len(data.get("text_prompt", []))
        print(f"    {n} rows. Columns: {list(data.keys())}", flush=True)
        model_rows[model_key] = data
    except Exception as e:
        print(f"    ERROR: {e}", flush=True)

# Build per-row difficulty table
# text_prompt → {model → {cer, wer, asr_transcription}}
print("\n=== Building per-row difficulty table ===", flush=True)

# Create index from text to row position using phase2_data
phase2_data = json.loads(Path("tricky-tts/phase2/phase2_data.json").read_text())
text_to_meta = {r["text"]: r for r in phase2_data}

rows_output = []

# Use ElevenLabs as the canonical ordering (all 48 rows)
el_data = model_rows.get("elevenlabs")
if not el_data:
    print("ERROR: ElevenLabs data not loaded", flush=True)
    exit(1)

n_rows = len(el_data["text_prompt"])
print(f"Processing {n_rows} rows...", flush=True)

for i in range(n_rows):
    text = el_data["text_prompt"][i]
    meta = text_to_meta.get(text, {})
    category = meta.get("category", "unknown")
    spoken_form = meta.get("spoken_form", "")
    cer_reliable = meta.get("cer_reliable", True)

    row_data = {
        "text": text,
        "category": category,
        "spoken_form": spoken_form,
        "cer_reliable": cer_reliable,
        "models": {},
    }

    for model_key, data in model_rows.items():
        # Find this text in the model's data
        if "text_prompt" in data:
            try:
                idx = data["text_prompt"].index(text)
                row_data["models"][model_key] = {
                    "cer": data.get("asr_cer", [None])[idx],
                    "wer": data.get("asr_wer", [None])[idx],
                    "asr_transcription": data.get("asr_transcription", [None])[idx],
                }
            except ValueError:
                row_data["models"][model_key] = None

    # Compute median CER across models (only for cer_reliable rows)
    cer_values = [
        v["cer"] for v in row_data["models"].values()
        if v is not None and v.get("cer") is not None
    ]

    if cer_values:
        row_data["median_cer"] = float(np.median(cer_values))
        row_data["max_cer"] = float(np.max(cer_values))
        row_data["min_cer"] = float(np.min(cer_values))
        row_data["mean_cer"] = float(np.mean(cer_values))
        row_data["n_models"] = len(cer_values)
    else:
        row_data["median_cer"] = None
        row_data["n_models"] = 0

    rows_output.append(row_data)

# Save full per-row data
out_path = Path("tricky-tts/phase2/phase2_perrow_results.json")
out_path.write_text(json.dumps(rows_output, indent=2))
print(f"\nSaved {len(rows_output)} rows to {out_path}", flush=True)

# Summary by category
print("\n=== Per-category median CER (across all models) ===")
from collections import defaultdict
by_cat = defaultdict(list)
for row in rows_output:
    if row["median_cer"] is not None:
        by_cat[row["category"]].append((row["text"][:60], row["median_cer"]))

for cat, rows in sorted(by_cat.items()):
    cers = [r[1] for r in rows]
    print(f"\n{cat} (n={len(cers)}, avg_median_cer={np.mean(cers):.4f}):")
    for text, cer in sorted(rows, key=lambda x: x[1]):
        flag = " ← EASY" if cer < 0.05 else ""
        print(f"  {cer:.4f}  {text}{flag}")

# Identify easy rows (median CER < 0.05 across all models)
EASY_THRESHOLD = 0.05
easy_rows = [r for r in rows_output if r["median_cer"] is not None and r["median_cer"] < EASY_THRESHOLD]
print(f"\n=== Easy rows (median CER < {EASY_THRESHOLD}): {len(easy_rows)} ===")
for r in easy_rows:
    print(f"  [{r['category']}] median_cer={r['median_cer']:.4f}  {r['text'][:80]}")

print(f"\nDataset would shrink: {len(rows_output)} → {len(rows_output) - len(easy_rows)} rows after filtering")
