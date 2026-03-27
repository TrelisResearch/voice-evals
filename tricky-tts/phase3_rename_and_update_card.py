"""
Rename ronanarraig/tricky-tts-prototype → ronanarraig/tricky-tts-public
and update dataset card with Trelis Voice AI Services link.
"""

import os
from pathlib import Path
from huggingface_hub import HfApi

env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

HF_TOKEN = os.environ["HF_TOKEN"]
api = HfApi(token=HF_TOKEN)

# Copy data from prototype to tricky-tts-public, then delete prototype
from datasets import load_dataset
print("Loading tricky-tts-prototype...", flush=True)
ds = load_dataset("ronanarraig/tricky-tts-prototype", split="train", token=HF_TOKEN)
from datasets import Audio
ds = ds.cast_column("reference_audio", Audio(decode=False))

print("Pushing to ronanarraig/tricky-tts-public...", flush=True)
ds = ds.cast_column("reference_audio", Audio())
ds.push_to_hub("ronanarraig/tricky-tts-public", split="train", token=HF_TOKEN, private=True)
print("  Done.", flush=True)

print("Deleting tricky-tts-prototype...", flush=True)
api.delete_repo(repo_id="ronanarraig/tricky-tts-prototype", repo_type="dataset")
print("  Done.", flush=True)

# Update dataset card
DATASET_CARD = """\
---
language:
- en
license: cc-by-4.0
task_categories:
- text-to-speech
tags:
- tts
- benchmark
- evaluation
- spoken-language
- audio
size_categories:
- n<1K
---

# Tricky TTS — Public

A 10-row prototype benchmark for evaluating text-to-speech models on linguistically and typographically challenging English text.

Evaluation is powered by [Trelis Voice AI Services](https://trelis.com/voice-ai-services/).

## Purpose

TTS models are typically evaluated on clean, simple prose. This dataset targets known failure modes: abbreviations, scientific notation, Celtic names, foreign loanwords, punctuation-driven prosody, and robustness edge cases.

Evaluation is performed by Trelis Studio using:
- **UTMOS** — neural naturalness / MOS score
- **Round-trip CER** — TTS output is transcribed by Whisper large-v3 and compared against `reference_asr` (Kokoro's ASR transcript of the `spoken_form` text), giving a pronunciation accuracy score that is model-agnostic

## Schema

| Column | Description |
|---|---|
| `text` | Original written text fed to the TTS model |
| `spoken_form` | Canonicalised spoken form — abbreviations, symbols, and special characters fully expanded |
| `category` | One of 6 difficulty categories (see below) |
| `reference_asr` | Whisper large-v3 transcript of Kokoro TTS output on `spoken_form` — used as CER reference |
| `reference_audio` | Kokoro TTS audio of `spoken_form` — for reference listening |

## Categories

| Category | Description |
|---|---|
| `edge_cases` | Abbreviations, units, scientific notation, journal references |
| `domain_specific` | Medical/microbiology terminology, IUPAC compounds, clinical standards |
| `ai_tech` | AI model paths (`org/model`), training hyperparameters, optimisers |
| `number_format` | Mixed number formats — blood pressure, temperatures, ratios, scientific notation |
| `phonetic` | Celtic names, foreign loanwords, non-obvious pronunciations |
| `paralinguistics` | Onomatopoeia, interjections, ellipses, em-dashes |

## Usage

This is a text-only benchmark — TTS models generate audio at evaluation time via [Trelis Voice AI Services](https://trelis.com/voice-ai-services/).
The `spoken_form` column is the recommended TTS input; `reference_asr` is the CER reference.

## Notes

- This is a 10-row public split. Larger semi-private and private splits are in development.
- All splits are kept private until methodology is finalised.
- Dataset created as part of the Trelis voice evaluation project.
"""

print("Updating dataset card...", flush=True)
api.upload_file(
    path_or_fileobj=DATASET_CARD.encode(),
    path_in_repo="README.md",
    repo_id="ronanarraig/tricky-tts-public",

    repo_type="dataset",
)
print("Done → ronanarraig/tricky-tts-public", flush=True)
