"""
Submit TTS eval jobs for all 9 Studio models on ronanarraig/tricky-tts-phase4.
No reference_column — CER is computed against raw 'text' column directly.
"""

import os, json, requests
from pathlib import Path

env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

API_KEY = os.environ["TRELIS_STUDIO_API_KEY"]
TRELIS_API = "https://studio.trelis.com/api/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

DATASET_ID = "ronanarraig/tricky-tts-phase4"
SPLIT = "train"

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
        "dataset_id": DATASET_ID,
        "split": SPLIT,
        "num_samples": 4,
        "tts_model_type": m["tts_model_type"],
        "asr_model_id": "fireworks/whisper-v3",
        "output_org": "ronanarraig",
        "output_name": f"tricky-tts-ph4-eval-{m['label'].lower().replace(' ', '-').replace('(', '').replace(')', '')}",
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

Path("tricky-tts/phase4/phase4_eval_job_ids.json").write_text(json.dumps(job_ids, indent=2))
print(f"\nSaved {len([v for v in job_ids.values() if v.get('job_id')])} job IDs to phase4_eval_job_ids.json", flush=True)
