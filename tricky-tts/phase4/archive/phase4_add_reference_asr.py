"""
1. Push temp dataset with reference audio as 'audio' column for ASR eval
2. Run ASR eval (fireworks/whisper-v3) to transcribe human reference audio
3. Pull asr_transcription from results
4. Update ronanarraig/tricky-tts-phase4 with reference_asr column
5. Re-submit all 9 TTS eval jobs with reference_column="reference_asr"
"""

import os, json, time, requests
from pathlib import Path
from datasets import Dataset, Audio
from huggingface_hub import hf_hub_download, list_repo_files
import pyarrow.parquet as pq

env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

HF_TOKEN = os.environ["HF_TOKEN"]
API_KEY = os.environ["TRELIS_STUDIO_API_KEY"]
TRELIS_API = "https://studio.trelis.com/api/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

PHASE4_DIR = Path(__file__).parent
rows_meta = json.loads((PHASE4_DIR / "rows.json").read_text())
audio_files_webm = [
    PHASE4_DIR / "audio" / "row0_symbol_expansion.webm",
    PHASE4_DIR / "audio" / "row1_abbreviation_reading.webm",
    PHASE4_DIR / "audio" / "row2_proper_nouns.webm",
    PHASE4_DIR / "audio" / "row3_prosody_and_punctuation.webm",
]
# Use WAV for ASR eval (libsndfile doesn't support webm)
audio_files = [
    PHASE4_DIR / "audio_wav" / "row0_symbol_expansion.wav",
    PHASE4_DIR / "audio_wav" / "row1_abbreviation_reading.wav",
    PHASE4_DIR / "audio_wav" / "row2_proper_nouns.wav",
    PHASE4_DIR / "audio_wav" / "row3_prosody_and_punctuation.wav",
]

# --- Step 1: Push temp dataset with audio column ---
print("Step 1: Pushing temp ASR-input dataset...", flush=True)
audio_data = [{"bytes": p.read_bytes(), "path": str(p)} for p in audio_files]
temp_ds = Dataset.from_dict({
    "audio": audio_data,
    "text": [r["text"] for r in rows_meta],
    "category": [r["category"] for r in rows_meta],
})
temp_ds = temp_ds.cast_column("audio", Audio())
temp_ds.push_to_hub(
    "ronanarraig/tricky-tts-phase4-asr-input",
    split="train",
    token=HF_TOKEN,
    private=True,
)
print("  Pushed ronanarraig/tricky-tts-phase4-asr-input", flush=True)

# --- Step 2: Submit ASR eval job ---
print("\nStep 2: Submitting ASR eval job (fireworks/whisper-v3)...", flush=True)
payload = {
    "model_id": "openai/whisper-large-v3",
    "dataset_id": "ronanarraig/tricky-tts-phase4-asr-input",
    "split": "train",
    "num_samples": 4,
    "output_org": "ronanarraig",
    "push_results": True,
    "private": True,
    "language": "en",
}
r = requests.post(f"{TRELIS_API}/evaluation/jobs", json=payload, headers=HEADERS)
if r.status_code != 200:
    print(f"  ERROR {r.status_code}: {r.text}", flush=True)
    exit(1)
job_id = r.json().get("job_id") or r.json().get("id")
print(f"  Job ID: {job_id}", flush=True)

# --- Step 3: Poll until done ---
print("\nStep 3: Polling ASR eval job...", flush=True)
while True:
    r = requests.get(f"{TRELIS_API}/evaluation/jobs/{job_id}", headers=HEADERS)
    r.raise_for_status()
    data = r.json()
    status = data.get("status", "unknown")
    print(f"  Status: {status}", flush=True)
    if status in ("completed", "failed", "error"):
        break
    time.sleep(15)

if status != "completed":
    print(f"  Job {status}. Exiting.", flush=True)
    exit(1)

# --- Step 4: Pull asr_transcription from results ---
print("\nStep 4: Pulling ASR transcriptions from HF...", flush=True)
# Discover result dataset from job response
result_repo = data.get("output_dataset") or data.get("hf_dataset_id") or data.get("dataset_id")
print(f"  Result dataset: {result_repo}", flush=True)
if not result_repo:
    print(f"  Full job result: {json.dumps(data, indent=2)[:1000]}", flush=True)
    exit(1)
files = list(list_repo_files(result_repo, repo_type="dataset", token=HF_TOKEN))
parquet_files = [f for f in files if f.endswith(".parquet")]
print(f"  Found parquet files: {parquet_files}", flush=True)

local = hf_hub_download(repo_id=result_repo, filename=parquet_files[0], repo_type="dataset", token=HF_TOKEN)
table = pq.read_table(local)
rows_dict = table.to_pydict()
print(f"  Columns: {list(rows_dict.keys())}", flush=True)

transcripts = rows_dict.get("asr_transcription", [])
print(f"\n  Reference ASR transcriptions:", flush=True)
for i, (meta, t) in enumerate(zip(rows_meta, transcripts)):
    print(f"  [{meta['category']}]", flush=True)
    print(f"    text: {meta['text'][:80]}", flush=True)
    print(f"    asr:  {str(t)[:80]}", flush=True)

# Save transcripts
Path(PHASE4_DIR / "phase4_reference_asr.json").write_text(json.dumps(transcripts, indent=2))

# --- Step 5: Update phase4 dataset with reference_asr column ---
print("\nStep 5: Rebuilding ronanarraig/tricky-tts-phase4 with reference_asr...", flush=True)
audio_data2 = [{"bytes": p.read_bytes(), "path": str(p)} for p in audio_files_webm]
ds = Dataset.from_dict({
    "text": [r["text"] for r in rows_meta],
    "category": [r["category"] for r in rows_meta],
    "spoken_form": [r["spoken_form"] for r in rows_meta],
    "reference_asr": transcripts,
    "reference_audio": audio_data2,
})
ds = ds.cast_column("reference_audio", Audio())
ds.push_to_hub(
    "ronanarraig/tricky-tts-phase4",
    split="train",
    token=HF_TOKEN,
    private=True,
)
print("  Pushed updated ronanarraig/tricky-tts-phase4 with reference_asr column", flush=True)
print("\nDone! Now re-run TTS evals with reference_column='reference_asr'.", flush=True)
