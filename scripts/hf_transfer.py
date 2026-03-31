"""
Transfer a model or dataset from ronanarraig/ to Trelis/ on HuggingFace.
Always pushes to Trelis as a private repo.

Usage:
    uv run scripts/hf_transfer.py --type dataset --name ai-terms-v2-public
    uv run scripts/hf_transfer.py --type model --name my-whisper-lora
"""

import argparse
import sys
import os
from dotenv import load_dotenv

load_dotenv()


def transfer_dataset(name: str, token: str):
    from datasets import load_dataset
    from huggingface_hub import HfApi

    source = f"ronanarraig/{name}"
    target = f"Trelis/{name}"

    print(f"Pulling dataset: {source}")
    ds = load_dataset(source, token=token)

    print(f"Pushing dataset: {target} (private)")
    ds.push_to_hub(target, token=token, private=True)

    print("Done.")


def transfer_model(name: str, token: str):
    from huggingface_hub import HfApi, snapshot_download

    source = f"ronanarraig/{name}"
    target = f"Trelis/{name}"

    api = HfApi(token=token)

    print(f"Pulling model: {source}")
    local_dir = snapshot_download(repo_id=source, token=token)

    print(f"Creating private repo: {target}")
    api.create_repo(repo_id=target, private=True, exist_ok=True)

    print(f"Pushing model: {target}")
    api.upload_folder(folder_path=local_dir, repo_id=target)

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Transfer HF repo from ronanarraig to Trelis (private)")
    parser.add_argument("--type", required=True, choices=["model", "dataset"], help="Repo type")
    parser.add_argument("--name", required=True, help="Repo name (without org prefix)")
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")
    if not token:
        print("Error: HF_TOKEN not found in environment. Check your .env file.", file=sys.stderr)
        sys.exit(1)

    if args.type == "dataset":
        transfer_dataset(args.name, token)
    else:
        transfer_model(args.name, token)


if __name__ == "__main__":
    main()
