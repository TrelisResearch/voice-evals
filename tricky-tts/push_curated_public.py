"""
Push the final curated public split to HuggingFace.
Replaces the previous placeholder version.
"""

import os
import json
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
REPO_ID = "ronanarraig/tricky-tts-public"

rows = json.loads(Path("tricky-tts/curated_public.json").read_text())
ds_rows = [{"text": r["text"], "category": r["category"]} for r in rows]

api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=True, exist_ok=True)

ds = Dataset.from_list(ds_rows)
print(f"Pushing {len(ds_rows)} rows to {REPO_ID}...", flush=True)
print(f"Category counts: { {c: sum(1 for r in ds_rows if r['category']==c) for c in set(r['category'] for r in ds_rows)} }", flush=True)
ds.push_to_hub(REPO_ID, token=HF_TOKEN, private=True)
print("Done.", flush=True)
