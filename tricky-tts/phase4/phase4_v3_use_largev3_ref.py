"""
Pull Whisper large-v3 reference_asr from the completed Studio ASR eval,
update ronanarraig/tricky-tts-phase4, and re-submit 9 TTS eval jobs.
"""

import os, json, requests
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
webm_files = [
    PHASE4_DIR / "audio" / "row0_symbol_expansion.webm",
    PHASE4_DIR / "audio" / "row1_abbreviation_reading.webm",
    PHASE4_DIR / "audio" / "row2_proper_nouns.webm",
    PHASE4_DIR / "audio" / "row3_prosody_and_punctuation.webm",
]

# --- Step 1: Pull large-v3 transcripts ---
print("Step 1: Pulling Whisper large-v3 transcripts from HF...", flush=True)
result_repo = "ronanarraig/eval-whisper-large-v3-tricky-tts-phase4-asr-input-20260330-1200"
files = list(list_repo_files(result_repo, repo_type="dataset", token=HF_TOKEN))
parquet_files = [f for f in files if f.endswith(".parquet")]
print(f"  Files: {parquet_files}", flush=True)

local = hf_hub_download(repo_id=result_repo, filename=parquet_files[0], repo_type="dataset", token=HF_TOKEN)
table = pq.read_table(local)
rows_dict = table.to_pydict()
print(f"  Columns: {list(rows_dict.keys())}", flush=True)

transcripts = rows_dict.get("prediction", rows_dict.get("asr_transcription", []))
print(f"\n  Whisper large-v3 reference_asr:", flush=True)
for i, (meta, t) in enumerate(zip(rows_meta, transcripts)):
    print(f"  Row {i} [{meta['category']}]", flush=True)
    print(f"    text: {meta['text'][:90]}", flush=True)
    print(f"    asr:  {str(t)[:90]}", flush=True)

Path(PHASE4_DIR / "phase4_reference_asr_largev3.json").write_text(json.dumps(transcripts, indent=2))

# --- Step 2: Update dataset ---
print("\nStep 2: Updating ronanarraig/tricky-tts-phase4 with large-v3 reference_asr...", flush=True)
audio_data = [{"bytes": p.read_bytes(), "path": str(p)} for p in webm_files]
ds = Dataset.from_dict({
    "text": [r["text"] for r in rows_meta],
    "category": [r["category"] for r in rows_meta],
    "spoken_form": [r["spoken_form"] for r in rows_meta],
    "reference_asr": transcripts,
    "reference_audio": audio_data,
})
ds = ds.cast_column("reference_audio", Audio())
ds.push_to_hub("ronanarraig/tricky-tts-phase4", split="train", token=HF_TOKEN, private=True)
print("  Pushed.", flush=True)

# --- Step 3: Re-submit 9 TTS eval jobs ---
print("\nStep 3: Submitting TTS eval jobs (v3, large-v3 reference_asr)...", flush=True)
MODELS = [
    {"label": "ElevenLabs",       "model_id": "elevenlabs/eleven-multilingual-v2", "tts_model_type": "auto",       "extra": {}},
    {"label": "GPT-4o mini TTS",  "model_id": "openai/gpt-4o-mini-tts",           "tts_model_type": "auto",       "extra": {}},
    {"label": "Cartesia Sonic-3", "model_id": "cartesia/sonic-3",                  "tts_model_type": "auto",       "extra": {}},
    {"label": "Gemini Flash TTS", "model_id": "google/gemini-2.5-flash-tts",       "tts_model_type": "auto",       "extra": {}},
    {"label": "Gemini Pro TTS",   "model_id": "google/gemini-2.5-pro-tts",         "tts_model_type": "auto",       "extra": {}},
    {"label": "Orpheus",          "model_id": "unsloth/orpheus-3b-0.1-ft",         "tts_model_type": "orpheus",    "extra": {"max_new_tokens": 4000}},
    {"label": "Kokoro",           "model_id": "kokoro",                             "tts_model_type": "kokoro",     "extra": {}},
    {"label": "Piper (en-gb)",    "model_id": "Trelis/piper-en-gb-cori-high",      "tts_model_type": "piper",      "extra": {"language": "en"}},
    {"label": "Chatterbox",       "model_id": "ResembleAI/chatterbox",             "tts_model_type": "chatterbox", "extra": {"language": "en"}},
]

job_ids = {}
for m in MODELS:
    slug = m["label"].lower().replace(" ", "-").replace("(", "").replace(")", "").replace("/", "-")
    payload = {
        "model_id": m["model_id"],
        "dataset_id": "ronanarraig/tricky-tts-phase4",
        "split": "train",
        "num_samples": 4,
        "tts_model_type": m["tts_model_type"],
        "asr_model_id": "fireworks/whisper-v3",
        "reference_column": "reference_asr",
        "output_org": "ronanarraig",
        "output_name": f"tricky-tts-ph4-v3-{slug}",
        "push_results": True,
        "private": True,
        **m["extra"],
    }
    r = requests.post(f"{TRELIS_API}/tts-evaluation/jobs", json=payload, headers=HEADERS)
    if r.status_code == 200:
        job_id = r.json().get("job_id") or r.json().get("id")
        job_ids[m["label"]] = {"job_id": job_id, "model_id": m["model_id"]}
        print(f"  [submitted] {m['label']}: {job_id}", flush=True)
    else:
        print(f"  [ERROR {r.status_code}] {m['label']}: {r.text[:200]}", flush=True)
        job_ids[m["label"]] = {"job_id": None, "error": r.text[:200]}

Path(PHASE4_DIR / "phase4_v3_eval_job_ids.json").write_text(json.dumps(job_ids, indent=2))
print(f"\nSaved {len([v for v in job_ids.values() if v.get('job_id')])} job IDs.", flush=True)
