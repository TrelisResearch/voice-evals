"""
Transcribe reference audio locally with faster-whisper large-v3,
then update tricky-tts-phase4 with reference_asr column and resubmit evals.
"""

import os, json, requests
from pathlib import Path
from datasets import Dataset, Audio

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

wav_files = [
    PHASE4_DIR / "audio_wav" / "row0_symbol_expansion.wav",
    PHASE4_DIR / "audio_wav" / "row1_abbreviation_reading.wav",
    PHASE4_DIR / "audio_wav" / "row2_proper_nouns.wav",
    PHASE4_DIR / "audio_wav" / "row3_prosody_and_punctuation.wav",
]
webm_files = [
    PHASE4_DIR / "audio" / "row0_symbol_expansion.webm",
    PHASE4_DIR / "audio" / "row1_abbreviation_reading.webm",
    PHASE4_DIR / "audio" / "row2_proper_nouns.webm",
    PHASE4_DIR / "audio" / "row3_prosody_and_punctuation.webm",
]

# --- Step 1: Transcribe with faster-whisper large-v3 ---
print("Step 1: Transcribing with Whisper large-v3 (local, int8)...", flush=True)
from faster_whisper import WhisperModel
model = WhisperModel("medium", device="cpu", compute_type="int8")

transcripts = []
for path in wav_files:
    print(f"  {path.name}...", flush=True)
    segments, _ = model.transcribe(str(path), language="en", beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments)
    transcripts.append(text)
    print(f"    -> {text[:120]}", flush=True)

Path(PHASE4_DIR / "phase4_reference_asr.json").write_text(json.dumps(transcripts, indent=2))
print("", flush=True)

# --- Step 2: Rebuild dataset with reference_asr ---
print("Step 2: Rebuilding ronanarraig/tricky-tts-phase4 with reference_asr...", flush=True)
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
print("  Pushed ronanarraig/tricky-tts-phase4", flush=True)

# --- Step 3: Submit 9 TTS eval jobs with reference_column="reference_asr" ---
print("\nStep 3: Submitting TTS eval jobs with reference_column='reference_asr'...", flush=True)

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
    payload = {
        "model_id": m["model_id"],
        "dataset_id": "ronanarraig/tricky-tts-phase4",
        "split": "train",
        "num_samples": 4,
        "tts_model_type": m["tts_model_type"],
        "asr_model_id": "fireworks/whisper-v3",
        "reference_column": "reference_asr",
        "output_org": "ronanarraig",
        "output_name": f"tricky-tts-ph4-v2-{m['label'].lower().replace(' ', '-').replace('(', '').replace(')', '').replace('/', '-')}",
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

Path(PHASE4_DIR / "phase4_v2_eval_job_ids.json").write_text(json.dumps(job_ids, indent=2))
print(f"\nSaved {len([v for v in job_ids.values() if v.get('job_id')])} job IDs.", flush=True)
