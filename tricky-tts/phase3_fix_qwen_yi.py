"""
Phase 3 fix: Regenerate spoken forms for Qwen and Yi rows (Chwen→Kwen, Yi→Yee),
re-push datasets, re-run Kokoro eval, rebuild tricky-tts-prototype with dataset card.
"""

import json, os, re, time, requests
from pathlib import Path
from datasets import Dataset, load_dataset
from huggingface_hub import HfApi
from openai import OpenAI

env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

HF_TOKEN = os.environ["HF_TOKEN"]
API_KEY = os.environ["TRELIS_STUDIO_API_KEY"]
TRELIS_API = "https://studio.trelis.com/api/v1"
TRELIS_HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
MODEL = "anthropic/claude-sonnet-4-5"
FALLBACK_MODEL = "google/gemini-2.5-flash"

RULES_PATH = Path("tricky-tts/spoken_form_rules.md")
SYSTEM_PROMPT = RULES_PATH.read_text()

UNICODE_SUBS = [
    ("×", " times "), ("±", " plus or minus "), ("≥", " greater than or equal to "),
    ("≤", " less than or equal to "), ("→", " to "), ("°C", " degrees Celsius"),
    ("°F", " degrees Fahrenheit"), ("μg", " micrograms"), ("μL", " microlitres"),
    ("μ", " micro"), ("°", " degrees"), ("⁻⁴", " to the minus four"),
    ("⁻", " minus "), ("⁹", " to the ninth"), ("—", ", "), ("–", " to "), ("…", "..."),
    ("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"'),
]

def sanitise_unicode(text):
    for src, dst in UNICODE_SUBS:
        text = text.replace(src, dst)
    return re.sub(r" {2,}", " ", text).strip()

def generate_spoken_form(text, category):
    for model in [MODEL, FALLBACK_MODEL]:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Category: {category}\nText: {text}"}
            ],
            max_tokens=1000, temperature=0.1,
        )
        content = resp.choices[0].message.content
        if content:
            return sanitise_unicode(content.strip())
        print(f"  ⚠ {model} returned None, trying fallback...", flush=True)
        time.sleep(1)
    raise ValueError(f"All models failed for: {text[:60]}")

# Load current results
data_path = Path("tricky-tts/phase3_proto_updated.json")
rows = json.loads(data_path.read_text())

# Regenerate rows 4 (Qwen) and 5 (Yi)
for i in [4, 5]:
    row = rows[i]
    print(f"[{i}] [{row['category']}] {row['text'][:70]}", flush=True)
    spoken = generate_spoken_form(row["text"], row["category"])
    print(f"  → {spoken[:120]}", flush=True)
    rows[i]["spoken_form"] = spoken
    time.sleep(0.3)

data_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False))
print("Updated spoken forms saved.", flush=True)

# Push proto-v4
proto_ds = Dataset.from_list([
    {"text": r["text"], "spoken_form": r["spoken_form"], "category": r["category"]}
    for r in rows
])
print(f"\nPushing ronanarraig/tricky-tts-proto-v4...", flush=True)
proto_ds.push_to_hub("ronanarraig/tricky-tts-proto-v4", split="train", token=HF_TOKEN, private=True)

# Push spoken_form_input
input_ds = Dataset.from_list([
    {"text": r["spoken_form"], "original_text": r["text"], "category": r["category"]}
    for r in rows
])
print(f"Pushing ronanarraig/tricky-tts-proto-spoken-form-input...", flush=True)
input_ds.push_to_hub("ronanarraig/tricky-tts-proto-spoken-form-input", split="train", token=HF_TOKEN, private=True)

# Submit Kokoro eval
print(f"\nSubmitting Kokoro eval job...", flush=True)
payload = {
    "model_id": "kokoro",
    "dataset_id": "ronanarraig/tricky-tts-proto-spoken-form-input",
    "split": "train",
    "num_samples": 10,
    "asr_model_id": "openai/whisper-large-v3",
    "language": "auto",
    "tts_model_type": "kokoro",
    "kokoro_voice": "af_heart",
    "max_new_tokens": 4000,
    "push_results": True,
    "output_org": "ronanarraig",
    "output_name": "tricky-tts-eval-ref-kokoro-v3",
    "private": True,
}
resp = requests.post(f"{TRELIS_API}/tts-evaluation/jobs", headers=TRELIS_HEADERS, json=payload)
resp.raise_for_status()
job = resp.json()
job_id = job.get("id") or job.get("job_id")
print(f"  → Job ID: {job_id}", flush=True)
Path("tricky-tts/phase3_kokoro_v3_job_id.json").write_text(json.dumps({"job_id": job_id, "output_name": "tricky-tts-eval-ref-kokoro-v3"}))

# Poll
print("Polling...", flush=True)
while True:
    r = requests.get(f"{TRELIS_API}/tts-evaluation/jobs/{job_id}", headers={"Authorization": f"Bearer {API_KEY}"})
    r.raise_for_status()
    status = r.json().get("status", "unknown")
    print(f"  Status: {status}", flush=True)
    if status in ("completed", "failed", "error"):
        break
    time.sleep(30)

if status != "completed":
    print("Job failed — aborting.", flush=True)
    exit(1)

# Build final dataset
print("\nBuilding tricky-tts-prototype...", flush=True)
eval_ds = load_dataset("ronanarraig/tricky-tts-eval-ref-kokoro-v3", split="train", token=HF_TOKEN)
text_cols = [c for c in ["text_prompt", "text", "asr_transcription"] if c in eval_ds.column_names]
eval_ds = eval_ds.select_columns(text_cols)
text_col = "text_prompt" if "text_prompt" in eval_ds.column_names else "text"
asr_by_spoken = {row[text_col]: row.get("asr_transcription", "") for row in eval_ds}

final_rows = []
for row in rows:
    ref_asr = asr_by_spoken.get(row["spoken_form"], "")
    if not ref_asr:
        print(f"  WARNING: no asr_transcription for: {row['spoken_form'][:60]}", flush=True)
    final_rows.append({
        "text": row["text"],
        "spoken_form": row["spoken_form"],
        "category": row["category"],
        "reference_asr": ref_asr,
    })
    print(f"  [{row['category']}] ref_asr: {ref_asr[:80]}", flush=True)

final_ds = Dataset.from_list(final_rows)
print(f"\nPushing ronanarraig/tricky-tts-prototype ({len(final_ds)} rows)...", flush=True)
final_ds.push_to_hub("ronanarraig/tricky-tts-prototype", split="train", token=HF_TOKEN, private=True)
print("Done → ronanarraig/tricky-tts-prototype", flush=True)

# Push dataset card
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

# Tricky TTS — Prototype

A 10-row prototype benchmark for evaluating text-to-speech models on linguistically and typographically challenging English text.

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

This is a text-only dataset. TTS models generate audio at evaluation time via Trelis Studio.
The `spoken_form` column is the recommended TTS input; `reference_asr` is the CER reference.

## Notes

- This is a 10-row prototype. Larger semi-private and private splits are in development.
- All splits are kept private until methodology is finalised.
- Dataset created as part of the Trelis voice evaluation project.
"""

print("\nPushing dataset card...", flush=True)
api = HfApi(token=HF_TOKEN)
api.upload_file(
    path_or_fileobj=DATASET_CARD.encode(),
    path_in_repo="README.md",
    repo_id="ronanarraig/tricky-tts-prototype",
    repo_type="dataset",
)
print("Dataset card pushed.", flush=True)
print("\nAll done.", flush=True)
