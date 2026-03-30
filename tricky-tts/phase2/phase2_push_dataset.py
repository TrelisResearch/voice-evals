"""
Phase 2 Step 2: Push updated public dataset to HuggingFace with spoken_form + cer_reliable columns.
Target: ronanarraig/tricky-tts-v2-public
"""

import os, json
from pathlib import Path
from datasets import Dataset
from huggingface_hub import HfApi

env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

HF_TOKEN = os.environ["HF_TOKEN"]

data = json.loads(Path("tricky-tts/phase2/phase2_data.json").read_text())
print(f"Loaded {len(data)} rows", flush=True)

# Validate all rows have required columns
for row in data:
    assert "text" in row and "category" in row and "spoken_form" in row and "cer_reliable" in row, f"Missing column in: {row}"

repo_id = "ronanarraig/tricky-tts-v2-public"
api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)

ds = Dataset.from_list(data)
print(f"Pushing to {repo_id}...", flush=True)
print(f"Columns: {ds.column_names}", flush=True)
ds.push_to_hub(repo_id, token=HF_TOKEN, private=True, split="train")
print(f"Done. {len(data)} rows at https://huggingface.co/datasets/{repo_id}", flush=True)
