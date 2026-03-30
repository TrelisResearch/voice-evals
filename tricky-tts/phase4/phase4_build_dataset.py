"""
Build ronanarraig/tricky-tts-phase4 dataset from rows.json + recorded .webm audio.
Columns: text, category, spoken_form, reference_audio (Audio feature)
"""

import os, json
from pathlib import Path
from datasets import Dataset, Audio

env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

HF_TOKEN = os.environ["HF_TOKEN"]

PHASE4_DIR = Path(__file__).parent
rows = json.loads((PHASE4_DIR / "rows.json").read_text())
audio_dir = PHASE4_DIR / "audio"

audio_map = {
    0: "row0_symbol_expansion.webm",
    1: "row1_abbreviation_reading.webm",
    2: "row2_proper_nouns.webm",
    3: "row3_prosody_and_punctuation.webm",
}

texts, categories, spoken_forms, audio_data = [], [], [], []

for row in rows:
    texts.append(row["text"])
    categories.append(row["category"])
    spoken_forms.append(row["spoken_form"])
    audio_path = audio_dir / audio_map[row["id"]]
    audio_data.append({"bytes": audio_path.read_bytes(), "path": str(audio_path)})

ds = Dataset.from_dict({
    "text": texts,
    "category": categories,
    "spoken_form": spoken_forms,
    "reference_audio": audio_data,
})
ds = ds.cast_column("reference_audio", Audio())

print(f"Dataset: {len(ds)} rows", flush=True)
print(f"Columns: {ds.column_names}", flush=True)
for i, row in enumerate(rows):
    print(f"  [{row['category']}] {row['text'][:80]}...", flush=True)

ds.push_to_hub(
    "ronanarraig/tricky-tts-phase4",
    split="train",
    token=HF_TOKEN,
    private=True,
)
print("\nPushed to ronanarraig/tricky-tts-phase4 (train split, private)", flush=True)
