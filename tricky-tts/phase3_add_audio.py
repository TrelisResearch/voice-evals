"""
Add Kokoro reference audio column to ronanarraig/tricky-tts-prototype.
Pulls generated_audio from tricky-tts-eval-ref-kokoro-v3, joins on spoken_form,
and pushes updated dataset.
"""

import os, json
from pathlib import Path
from datasets import load_dataset, Dataset, Audio

env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

HF_TOKEN = os.environ["HF_TOKEN"]

# Load prototype (text only)
print("Loading tricky-tts-prototype...", flush=True)
proto = load_dataset("ronanarraig/tricky-tts-prototype", split="train", token=HF_TOKEN)
print(f"  Columns: {proto.column_names}", flush=True)

# Load Kokoro eval output with audio decoded=False (avoids torch requirement)
print("Loading tricky-tts-eval-ref-kokoro-v3 (audio decode=False)...", flush=True)
eval_ds = load_dataset("ronanarraig/tricky-tts-eval-ref-kokoro-v3", split="train", token=HF_TOKEN)
eval_ds = eval_ds.cast_column("generated_audio", Audio(decode=False))
print(f"  Columns: {eval_ds.column_names}", flush=True)

# Build lookup: spoken_form → audio dict {bytes, path}
text_col = "text_prompt" if "text_prompt" in eval_ds.column_names else "text"
audio_by_spoken = {}
for row in eval_ds:
    audio_by_spoken[row[text_col]] = row["generated_audio"]

print(f"  Audio entries: {len(audio_by_spoken)}", flush=True)

# Build updated rows
final_rows = []
for row in proto:
    audio = audio_by_spoken.get(row["spoken_form"])
    if audio is None:
        print(f"  WARNING: no audio for spoken_form: {row['spoken_form'][:60]}", flush=True)
    final_rows.append({
        "text": row["text"],
        "spoken_form": row["spoken_form"],
        "category": row["category"],
        "reference_asr": row["reference_asr"],
        "reference_audio": audio,
    })

final_ds = Dataset.from_list(final_rows)
# Cast reference_audio column as Audio feature so HF renders it properly
final_ds = final_ds.cast_column("reference_audio", Audio())
print(f"\nFinal columns: {final_ds.column_names}", flush=True)

print("Pushing ronanarraig/tricky-tts-prototype...", flush=True)
final_ds.push_to_hub("ronanarraig/tricky-tts-prototype", split="train", token=HF_TOKEN, private=True)
print("Done.", flush=True)
