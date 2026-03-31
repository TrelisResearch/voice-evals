"""
Push tricky-tts-phase4 as public dataset ronanarraig/tricky-tts-public
with a proper dataset card.
"""

import os
from pathlib import Path
from datasets import load_dataset
from huggingface_hub import HfApi

env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

HF_TOKEN = os.environ["HF_TOKEN"]
api = HfApi(token=HF_TOKEN)

REPO_ID = "ronanarraig/tricky-tts-public"

README = """\
---
license: mit
tags:
  - tts
  - text-to-speech
  - evaluation
  - benchmark
  - english
language:
  - en
---

# Tricky TTS

A benchmark dataset for evaluating text-to-speech (TTS) models on linguistically and
typographically challenging English text. Each row is designed to stress-test a specific
failure mode that separates capable TTS systems from weaker ones.

## Dataset

4 rows covering four challenge categories:

| Category | What it tests |
|---|---|
| `symbol_expansion` | Unicode symbols, units, operators — `≥`, `μL`, `±`, `×10⁶` |
| `abbreviation_reading` | Acronyms, initialisms, roman numerals, dotted titles — `IEEE`, `Vol. XII`, `F.A.C.C.` |
| `proper_nouns` | Irish/Celtic names, HuggingFace model paths, brand names |
| `prosody_and_punctuation` | Em-dashes, ellipses, onomatopoeia, rhythm — `zzz`, `Psst`, `whoosh` |

Columns: `text`, `category`, `spoken_form` (normalised reference transcription), `reference_audio` (human voice recording, webm), `reference_asr` (Whisper large-v3 transcription of reference audio).

## Usage

```python
from datasets import load_dataset
ds = load_dataset("ronanarraig/tricky-tts-public", split="train")
for row in ds:
    print(row["category"], row["text"])
```

## Leaderboard

Evaluated with round-trip ASR (Whisper large-v3 human reference, `fireworks/whisper-v3` scoring).
MOS from UTMOS. Human reference audio scored at 4.22 MOS.

| Rank | Model | MOS ↑ | CER ↓ | Eval dataset |
|---|---|---|---|---|
| 1 | Gemini Pro TTS | 4.227 | 0.112 | [ronanarraig/tricky-tts-ph4-v3-gemini-pro-tts](https://huggingface.co/datasets/ronanarraig/tricky-tts-ph4-v3-gemini-pro-tts) |
| 2 | GPT-4o mini TTS | 4.330 | 0.121 | [ronanarraig/tricky-tts-ph4-v3-gpt-4o-mini-tts](https://huggingface.co/datasets/ronanarraig/tricky-tts-ph4-v3-gpt-4o-mini-tts) |
| 3 | Gemini Flash TTS | 4.184 | 0.122 | [ronanarraig/tricky-tts-ph4-v3-gemini-flash-tts](https://huggingface.co/datasets/ronanarraig/tricky-tts-ph4-v3-gemini-flash-tts) |
| 4 | ElevenLabs | 4.273 | 0.192 | [ronanarraig/tricky-tts-ph4-v3-elevenlabs](https://huggingface.co/datasets/ronanarraig/tricky-tts-ph4-v3-elevenlabs) |
| 5 | Kokoro | 4.511 | 0.209 | [ronanarraig/tricky-tts-ph4-v3-kokoro](https://huggingface.co/datasets/ronanarraig/tricky-tts-ph4-v3-kokoro) |
| 6 | Orpheus | 4.152 | 0.229 | [ronanarraig/tricky-tts-ph4-v3-orpheus](https://huggingface.co/datasets/ronanarraig/tricky-tts-ph4-v3-orpheus) |
| 7 | Cartesia Sonic-3 | 4.019 | 0.259 | [ronanarraig/tricky-tts-ph4-v3-cartesia-sonic-3](https://huggingface.co/datasets/ronanarraig/tricky-tts-ph4-v3-cartesia-sonic-3) |
| 8 | Piper (en-gb) | 3.777 | 0.323 | [ronanarraig/tricky-tts-ph4-v3-piper-en-gb](https://huggingface.co/datasets/ronanarraig/tricky-tts-ph4-v3-piper-en-gb) |
| 9 | Mistral Voxtral-Mini | 4.289 | 0.569 | [ronanarraig/tricky-tts-ph4-v3-mistral](https://huggingface.co/datasets/ronanarraig/tricky-tts-ph4-v3-mistral) |
| 10 | Chatterbox | 4.100 | 0.583 | [ronanarraig/tricky-tts-ph4-v3-chatterbox](https://huggingface.co/datasets/ronanarraig/tricky-tts-ph4-v3-chatterbox) |

## Evaluation methodology

Evaluations were run via [Trelis Voice AI Services](https://trelis.com/voice-ai-services/):
- **Round-trip ASR CER**: TTS model generates audio → Whisper transcribes back → CER vs human reference
- **MOS (naturalness)**: UTMOS score on generated audio

## License

MIT
"""

# Load source dataset
print("Loading ronanarraig/tricky-tts-phase4...", flush=True)
ds = load_dataset("ronanarraig/tricky-tts-phase4", split="train", token=HF_TOKEN)

# Keep all columns
cols_to_keep = [c for c in ["text", "category", "spoken_form", "reference_asr", "reference_audio"] if c in ds.column_names]
ds = ds.select_columns(cols_to_keep)
print(f"  {len(ds)} rows | columns: {ds.column_names}", flush=True)

# Create repo if needed (public)
try:
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", private=True, exist_ok=True)
    print(f"  Repo ready: {REPO_ID}", flush=True)
except Exception as e:
    print(f"  create_repo: {e}", flush=True)

# Push dataset
print("Pushing dataset...", flush=True)
ds.push_to_hub(REPO_ID, split="train", token=HF_TOKEN, private=True)
print("  Pushed.", flush=True)

# Push README
print("Uploading dataset card...", flush=True)
api.upload_file(
    path_or_fileobj=README.encode(),
    path_in_repo="README.md",
    repo_id=REPO_ID,
    repo_type="dataset",
    token=HF_TOKEN,
    commit_message="Add dataset card",
)
print(f"  Done. https://huggingface.co/datasets/{REPO_ID}", flush=True)
