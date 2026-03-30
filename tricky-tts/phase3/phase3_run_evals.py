"""
Phase 3 Step 3: Submit TTS eval jobs for all target models on tricky-tts-public.

Input dataset: ronanarraig/tricky-tts-eval-input (spoken_form as text + reference_asr)
Reference column: reference_asr
ASR model: openai/whisper-large-v3
"""

import os, json, requests
from pathlib import Path

env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

HF_TOKEN = os.environ["HF_TOKEN"]
API_KEY = os.environ["TRELIS_STUDIO_API_KEY"]
TRELIS_API = "https://studio.trelis.com/api/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

# Build eval-input dataset: text=spoken_form, reference_asr from tricky-tts-public
EVAL_DATASET = "ronanarraig/tricky-tts-public"

MODELS = [
    {"label": "ElevenLabs",       "model_id": "elevenlabs/eleven-multilingual-v2", "tts_model_type": "auto",       "output_name": "tricky-tts-pub-eval-elevenlabs"},
    {"label": "GPT-4o mini TTS",  "model_id": "openai/gpt-4o-mini-tts",           "tts_model_type": "auto",       "output_name": "tricky-tts-pub-eval-gpt4o-mini"},
    {"label": "Cartesia Sonic-3", "model_id": "cartesia/sonic-3",                 "tts_model_type": "auto",       "output_name": "tricky-tts-pub-eval-cartesia"},
    {"label": "Gemini Flash TTS", "model_id": "google/gemini-2.5-flash-tts",      "tts_model_type": "auto",       "output_name": "tricky-tts-pub-eval-gemini-flash"},
    {"label": "Gemini Pro TTS",   "model_id": "google/gemini-2.5-pro-tts",        "tts_model_type": "auto",       "output_name": "tricky-tts-pub-eval-gemini-pro"},
    {"label": "Orpheus",          "model_id": "unsloth/orpheus-3b-0.1-ft",        "tts_model_type": "orpheus",    "output_name": "tricky-tts-pub-eval-orpheus",    "speaker_name": "tara"},
    {"label": "Kokoro",           "model_id": "kokoro",                           "tts_model_type": "kokoro",     "output_name": "tricky-tts-pub-eval-kokoro",     "kokoro_voice": "af_heart"},
    {"label": "Piper",            "model_id": "piper",                            "tts_model_type": "piper",      "output_name": "tricky-tts-pub-eval-piper"},
    {"label": "Chatterbox",       "model_id": "chatterbox",                       "tts_model_type": "chatterbox", "output_name": "tricky-tts-pub-eval-chatterbox"},
]

job_ids = {}

for m in MODELS:
    payload = {
        "model_id": m["model_id"],
        "dataset_id": EVAL_DATASET,
        "split": "train",
        "num_samples": 10,
        "asr_model_id": "openai/whisper-large-v3",
        "language": "auto",
        "tts_model_type": m["tts_model_type"],
        "max_new_tokens": 4000,
        "reference_column": "reference_asr",
        "push_results": True,
        "output_org": "ronanarraig",
        "output_name": m["output_name"],
        "private": True,
    }
    if "speaker_name" in m:
        payload["speaker_name"] = m["speaker_name"]
    if "kokoro_voice" in m:
        payload["kokoro_voice"] = m["kokoro_voice"]

    print(f"Submitting: {m['label']}...", flush=True)
    resp = requests.post(f"{TRELIS_API}/tts-evaluation/jobs", headers=HEADERS, json=payload)

    if resp.status_code in (200, 201):
        job = resp.json()
        job_id = job.get("id") or job.get("job_id")
        job_ids[m["label"]] = {"job_id": job_id, "model_id": m["model_id"], "output_name": m["output_name"]}
        print(f"  → Job ID: {job_id}", flush=True)
    else:
        print(f"  ERROR {resp.status_code}: {resp.text[:300]}", flush=True)

out_path = Path("tricky-tts/phase3/phase3_eval_job_ids.json")
out_path.write_text(json.dumps(job_ids, indent=2))
print(f"\nAll jobs submitted. IDs saved to {out_path}", flush=True)
print(json.dumps(job_ids, indent=2), flush=True)
