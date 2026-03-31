"""
Transfer tricky-tts phase 4 eval datasets from ronanarraig/ to Trelis/ (private).
Drops 'ph4-v3' from the target name. Then updates the dataset card on
Trelis/tricky-tts-public so all links point to the new Trelis/ names.
"""

import os
import sys
from dotenv import load_dotenv
from datasets import load_dataset
from huggingface_hub import HfApi

load_dotenv()

READ_TOKEN = os.environ.get("HF_TOKEN_RONANARRAIG")
WRITE_TOKEN = os.environ.get("HF_TOKEN")
if not READ_TOKEN:
    print("Error: HF_TOKEN_RONANARRAIG not found. Check .env", file=sys.stderr)
    sys.exit(1)
if not WRITE_TOKEN:
    print("Error: HF_TOKEN not found. Check .env", file=sys.stderr)
    sys.exit(1)

api = HfApi(token=WRITE_TOKEN)

# (source_name, target_name) — target drops "ph4-v3"
DATASETS = [
    ("tricky-tts-public",                  "tricky-tts-public"),
    ("tricky-tts-ph4-v3-elevenlabs",       "tricky-tts-elevenlabs"),
    ("tricky-tts-ph4-v3-gpt-4o-mini-tts",  "tricky-tts-gpt-4o-mini-tts"),
    ("tricky-tts-ph4-v3-cartesia-sonic-3",  "tricky-tts-cartesia-sonic-3"),
    ("tricky-tts-ph4-v3-gemini-flash-tts",  "tricky-tts-gemini-flash-tts"),
    ("tricky-tts-ph4-v3-gemini-pro-tts",    "tricky-tts-gemini-pro-tts"),
    ("tricky-tts-ph4-v3-orpheus",           "tricky-tts-orpheus"),
    ("tricky-tts-ph4-v3-kokoro",            "tricky-tts-kokoro"),
    ("tricky-tts-ph4-v3-piper-en-gb",       "tricky-tts-piper-en-gb"),
    ("tricky-tts-ph4-v3-chatterbox",        "tricky-tts-chatterbox"),
    ("tricky-tts-ph4-v3-mistral",           "tricky-tts-mistral"),
]

# Mapping from old ronanarraig links to new Trelis links (for README update)
LINK_REPLACEMENTS = {
    "ronanarraig/tricky-tts-ph4-v3-elevenlabs":       "Trelis/tricky-tts-elevenlabs",
    "ronanarraig/tricky-tts-ph4-v3-gpt-4o-mini-tts":  "Trelis/tricky-tts-gpt-4o-mini-tts",
    "ronanarraig/tricky-tts-ph4-v3-cartesia-sonic-3":  "Trelis/tricky-tts-cartesia-sonic-3",
    "ronanarraig/tricky-tts-ph4-v3-gemini-flash-tts":  "Trelis/tricky-tts-gemini-flash-tts",
    "ronanarraig/tricky-tts-ph4-v3-gemini-pro-tts":    "Trelis/tricky-tts-gemini-pro-tts",
    "ronanarraig/tricky-tts-ph4-v3-orpheus":           "Trelis/tricky-tts-orpheus",
    "ronanarraig/tricky-tts-ph4-v3-kokoro":            "Trelis/tricky-tts-kokoro",
    "ronanarraig/tricky-tts-ph4-v3-piper-en-gb":       "Trelis/tricky-tts-piper-en-gb",
    "ronanarraig/tricky-tts-ph4-v3-chatterbox":        "Trelis/tricky-tts-chatterbox",
    "ronanarraig/tricky-tts-ph4-v3-mistral":           "Trelis/tricky-tts-mistral",
    "ronanarraig/tricky-tts-public":                   "Trelis/tricky-tts-public",
}

# --- Step 1: Transfer all datasets ---
for source_name, target_name in DATASETS:
    source = f"ronanarraig/{source_name}"
    target = f"Trelis/{target_name}"
    print(f"\n{'='*60}")
    print(f"Transferring: {source} → {target}")
    try:
        ds = load_dataset(source, token=READ_TOKEN)
        ds.push_to_hub(target, token=WRITE_TOKEN, private=True)
        print(f"  Done")
    except Exception as e:
        print(f"  Failed: {e}")

# --- Step 2: Update the dataset card on Trelis/tricky-tts-public ---
print(f"\n{'='*60}")
print("Updating dataset card on Trelis/tricky-tts-public...")

try:
    readme_path = api.hf_hub_download(
        repo_id="Trelis/tricky-tts-public",
        filename="README.md",
        repo_type="dataset",
        token=WRITE_TOKEN,
    )
    with open(readme_path, "r") as f:
        readme = f.read()

    # Apply each specific replacement (handles renamed datasets)
    for old, new in LINK_REPLACEMENTS.items():
        readme = readme.replace(old, new)

    # Catch any remaining ronanarraig/ references
    readme = readme.replace("ronanarraig/", "Trelis/")

    api.upload_file(
        path_or_fileobj=readme.encode(),
        path_in_repo="README.md",
        repo_id="Trelis/tricky-tts-public",
        repo_type="dataset",
        token=WRITE_TOKEN,
        commit_message="Update links from ronanarraig to Trelis (renamed datasets)",
    )
    print("  Dataset card updated")
except Exception as e:
    print(f"  Card update failed: {e}")

print(f"\n{'='*60}")
print("All done.")
