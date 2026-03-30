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

# Push all three splits
data = json.loads(Path("tricky-tts/phase1/phase1d_generated.json").read_text())
# Override public split with calibrated version
public_final = json.loads(Path("tricky-tts/phase1/phase1d_final.json").read_text())
public_texts = {r["text"] for r in public_final}

api = HfApi(token=HF_TOKEN)
split_map = {"public": "tricky-tts-v2-public", "semi_private": "tricky-tts-v2-semi-private", "private": "tricky-tts-v2-private"}

# Public: use calibrated version
repo_id = f"ronanarraig/{split_map['public']}"
api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
ds = Dataset.from_list(public_final)
print(f"Pushing {len(public_final)} rows to {repo_id}...", flush=True)
ds.push_to_hub(repo_id, token=HF_TOKEN, private=True)
print("Done.", flush=True)

# Semi-private and private: use generated versions
for split_key in ["semi_private", "private"]:
    rows = [{"text": r["text"], "category": r["category"]} for r in data if r["split"] == split_key]
    repo_id = f"ronanarraig/{split_map[split_key]}"
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)
    ds = Dataset.from_list(rows)
    print(f"Pushing {len(rows)} rows to {repo_id}...", flush=True)
    ds.push_to_hub(repo_id, token=HF_TOKEN, private=True)
    print("Done.", flush=True)
