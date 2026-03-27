"""
Phase 3 Step 1: Push spoken_form input dataset, then submit Orpheus + Kokoro TTS eval jobs.

Each job:
- Input: spoken_form text (pushed as 'text' column in temp dataset)
- ASR: openai/whisper-large-v3
- Output: per-row audio + asr_transcription pushed to HF
  - ronanarraig/tricky-tts-eval-ref-orpheus
  - ronanarraig/tricky-tts-eval-ref-kokoro

After polling, phase3_build_datasets.py joins results back and pushes:
  - ronanarraig/tricky-tts-public-orpheus
  - ronanarraig/tricky-tts-public-kokoro
"""

import os, json, requests
from pathlib import Path
from datasets import Dataset

env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

API_KEY = os.environ["TRELIS_STUDIO_API_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]
TRELIS_API = "https://studio.trelis.com/api/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

PROTO_DATASET = "ronanarraig/tricky-tts-proto-v4"
INPUT_DATASET = "ronanarraig/tricky-tts-proto-spoken-form-input"

# --- Step 1: Push spoken_form as 'text' ---
print("Loading proto-v4 dataset...", flush=True)
from datasets import load_dataset
proto = load_dataset(PROTO_DATASET, split="train")

# Build input dataset: spoken_form → text, keep original_text and category for join later
input_rows = [
    {"text": row["spoken_form"], "original_text": row["text"], "category": row["category"]}
    for row in proto
]
input_ds = Dataset.from_list(input_rows)

print(f"Pushing input dataset ({len(input_ds)} rows) to {INPUT_DATASET}...", flush=True)
input_ds.push_to_hub(INPUT_DATASET, split="train", token=HF_TOKEN, private=True)
print("  Done.", flush=True)

# --- Step 2: Submit TTS eval jobs ---
MODELS = [
    {
        "label": "Orpheus",
        "model_id": "unsloth/orpheus-3b-0.1-ft",
        "tts_model_type": "orpheus",
        "speaker_name": "tara",
        "output_name": "tricky-tts-eval-ref-orpheus",
    },
    {
        "label": "Kokoro",
        "model_id": "kokoro",
        "tts_model_type": "kokoro",
        "kokoro_voice": "af_heart",
        "output_name": "tricky-tts-eval-ref-kokoro",
    },
]

job_ids = {}

for m in MODELS:
    payload = {
        "model_id": m["model_id"],
        "dataset_id": INPUT_DATASET,
        "split": "train",
        "num_samples": 10,
        "asr_model_id": "openai/whisper-large-v3",
        "language": "auto",
        "tts_model_type": m["tts_model_type"],
        "max_new_tokens": 4000,
        "push_results": True,
        "output_org": "ronanarraig",
        "output_name": m["output_name"],
        "private": True,
    }
    if "speaker_name" in m:
        payload["speaker_name"] = m["speaker_name"]
    if "kokoro_voice" in m:
        payload["kokoro_voice"] = m["kokoro_voice"]

    print(f"Submitting: {m['label']}...", flush=True)
    resp = requests.post(f"{TRELIS_API}/tts-evaluation/jobs", headers=HEADERS, json=payload)

    if resp.status_code in (200, 201):
        job = resp.json()
        job_id = job.get("id") or job.get("job_id")
        job_ids[m["label"]] = {
            "job_id": job_id,
            "model_id": m["model_id"],
            "output_name": m["output_name"],
        }
        print(f"  → Job ID: {job_id}", flush=True)
    else:
        print(f"  ERROR {resp.status_code}: {resp.text[:300]}", flush=True)

out_path = Path("tricky-tts/phase3_job_ids.json")
out_path.write_text(json.dumps(job_ids, indent=2))
print(f"\nJob IDs saved to {out_path}", flush=True)
print(json.dumps(job_ids, indent=2), flush=True)
