"""
Push tricky-tts dataset splits to HuggingFace as separate private repos.
Repos: ronanarraig/tricky-tts-public, ronanarraig/tricky-tts-semi-private, ronanarraig/tricky-tts-private
"""

import os
import json
from pathlib import Path
from datasets import Dataset
from huggingface_hub import HfApi

# Load env
env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

HF_TOKEN = os.environ["HF_TOKEN"]
ORG = "ronanarraig"

SPLIT_MAP = {
    "public": "tricky-tts-public",
    "semi_private": "tricky-tts-semi-private",
    "private": "tricky-tts-private",
}

api = HfApi(token=HF_TOKEN)

data = json.loads(Path("/home/claude/TR/voice-evals/tricky-tts/phase1/generated_texts.json").read_text())

for split_key, repo_name in SPLIT_MAP.items():
    repo_id = f"{ORG}/{repo_name}"
    rows = [{"text": r["text"], "category": r["category"]} for r in data if r["split"] == split_key]

    print(f"\nPushing {len(rows)} rows to {repo_id} ...", flush=True)

    # Create repo (private)
    try:
        api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
        print(f"  Repo ready: {repo_id}", flush=True)
    except Exception as e:
        print(f"  Repo create warning: {e}", flush=True)

    ds = Dataset.from_list(rows)
    print(f"  Dataset: {ds}", flush=True)

    ds.push_to_hub(repo_id, token=HF_TOKEN, private=True)
    print(f"  Pushed successfully.", flush=True)

print("\nAll splits pushed.", flush=True)
