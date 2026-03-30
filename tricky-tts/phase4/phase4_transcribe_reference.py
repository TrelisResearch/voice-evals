"""
Transcribe the 4 human-recorded reference audio files using Whisper large-v3,
then update ronanarraig/tricky-tts-phase4 with a reference_asr column.
"""

import os, json
from pathlib import Path
from datasets import load_dataset, Audio, Dataset

env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

HF_TOKEN = os.environ["HF_TOKEN"]

PHASE4_DIR = Path(__file__).parent
audio_files = [
    PHASE4_DIR / "audio" / "row0_symbol_expansion.webm",
    PHASE4_DIR / "audio" / "row1_abbreviation_reading.webm",
    PHASE4_DIR / "audio" / "row2_proper_nouns.webm",
    PHASE4_DIR / "audio" / "row3_prosody_and_punctuation.webm",
]

print("Loading Whisper large-v3 (this may take a moment)...", flush=True)
from faster_whisper import WhisperModel
model = WhisperModel("large-v3", device="cpu", compute_type="int8")
print("Model loaded.\n", flush=True)

transcripts = []
for path in audio_files:
    print(f"Transcribing {path.name}...", flush=True)
    segments, info = model.transcribe(str(path), language="en", beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments)
    transcripts.append(text)
    print(f"  -> {text[:120]}", flush=True)

print(f"\nAll transcribed. Saving...", flush=True)
Path(PHASE4_DIR / "phase4_reference_asr.json").write_text(json.dumps(transcripts, indent=2))

# Load existing dataset and add reference_asr column
print("\nLoading ronanarraig/tricky-tts-phase4 from HF...", flush=True)
rows_meta = json.loads((PHASE4_DIR / "rows.json").read_text())

# Rebuild dataset with reference_asr column
texts = [r["text"] for r in rows_meta]
categories = [r["category"] for r in rows_meta]
spoken_forms = [r["spoken_form"] for r in rows_meta]
audio_data = []
for path in audio_files:
    audio_data.append({"bytes": path.read_bytes(), "path": str(path)})

ds = Dataset.from_dict({
    "text": texts,
    "category": categories,
    "spoken_form": spoken_forms,
    "reference_asr": transcripts,
    "reference_audio": audio_data,
})
ds = ds.cast_column("reference_audio", Audio())

print(f"\nDataset with reference_asr:", flush=True)
for i, row in enumerate(rows_meta):
    print(f"  [{row['category']}]", flush=True)
    print(f"    text:          {row['text'][:80]}", flush=True)
    print(f"    reference_asr: {transcripts[i][:80]}", flush=True)

ds.push_to_hub(
    "ronanarraig/tricky-tts-phase4",
    split="train",
    token=HF_TOKEN,
    private=True,
)
print("\nPushed updated dataset to ronanarraig/tricky-tts-phase4", flush=True)
