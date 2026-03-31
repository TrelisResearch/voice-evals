"""
Phase 3: Poll Kokoro v2 eval job, then build and push ronanarraig/tricky-tts-prototype.
"""

import os, json, time, requests
from pathlib import Path
from datasets import load_dataset, Dataset

env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

API_KEY = os.environ["TRELIS_STUDIO_API_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]
TRELIS_API = "https://studio.trelis.com/api/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

info = json.loads(Path("tricky-tts/phase3/phase3_kokoro_v2_job_id.json").read_text())
job_id = info["job_id"]
print(f"Polling job {job_id}...", flush=True)

while True:
    resp = requests.get(f"{TRELIS_API}/tts-evaluation/jobs/{job_id}", headers=HEADERS)
    resp.raise_for_status()
    data = resp.json()
    status = data.get("status", "unknown")
    print(f"  Status: {status}", flush=True)
    if status in ("completed", "failed", "error"):
        break
    time.sleep(30)

if status != "completed":
    print(f"Job did not complete cleanly: {json.dumps(data, indent=2)[:500]}", flush=True)
    exit(1)

# Load updated proto data
print("\nLoading updated proto data...", flush=True)
proto_rows = json.loads(Path("tricky-tts/phase3/phase3_proto_updated.json").read_text())

# Load eval output
eval_dataset_id = f"ronanarraig/{info['output_name']}"
print(f"Loading eval output: {eval_dataset_id}...", flush=True)
eval_ds = load_dataset(eval_dataset_id, split="train", token=HF_TOKEN)
print(f"  Columns: {eval_ds.column_names}", flush=True)

text_cols = [c for c in ["text_prompt", "text", "asr_transcription"] if c in eval_ds.column_names]
eval_ds = eval_ds.select_columns(text_cols)
text_col = "text_prompt" if "text_prompt" in eval_ds.column_names else "text"

asr_by_spoken = {row[text_col]: row.get("asr_transcription", "") for row in eval_ds}

# Build final dataset
final_rows = []
for row in proto_rows:
    ref_asr = asr_by_spoken.get(row["spoken_form"], "")
    if not ref_asr:
        print(f"  WARNING: no asr_transcription for: {row['spoken_form'][:60]}", flush=True)
    final_rows.append({
        "text": row["text"],
        "spoken_form": row["spoken_form"],
        "category": row["category"],
        "reference_asr": ref_asr,
    })

final_ds = Dataset.from_list(final_rows)
print(f"\nFinal dataset: {len(final_ds)} rows, columns: {final_ds.column_names}", flush=True)

# Print for inspection
for row in final_ds:
    print(f"  [{row['category']}] ref_asr: {row['reference_asr'][:80]}", flush=True)

print(f"\nPushing ronanarraig/tricky-tts-prototype...", flush=True)
final_ds.push_to_hub("ronanarraig/tricky-tts-prototype", split="train", token=HF_TOKEN, private=True)
print("Done → ronanarraig/tricky-tts-prototype", flush=True)
