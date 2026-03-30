"""Submit remaining 5 TTS eval jobs: Gemini Pro, Orpheus, Kokoro, Chatterbox, Piper."""

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
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

EVAL_DATASET = "ronanarraig/tricky-tts-public"

MODELS = [
    {"label": "Gemini Pro TTS",  "model_id": "google/gemini-2.5-pro-tts",     "tts_model_type": "auto",       "output_name": "tricky-tts-pub-eval-gemini-pro"},
    {"label": "Orpheus",         "model_id": "unsloth/orpheus-3b-0.1-ft",      "tts_model_type": "orpheus",    "output_name": "tricky-tts-pub-eval-orpheus",   "speaker_name": "tara"},
    {"label": "Kokoro",          "model_id": "kokoro",                         "tts_model_type": "kokoro",     "output_name": "tricky-tts-pub-eval-kokoro",    "kokoro_voice": "af_heart"},
    {"label": "Chatterbox",      "model_id": "ResembleAI/chatterbox",          "tts_model_type": "chatterbox", "output_name": "tricky-tts-pub-eval-chatterbox"},
    {"label": "Piper (en-gb)",   "model_id": "Trelis/piper-en-gb-cori-high",   "tts_model_type": "piper",      "output_name": "tricky-tts-pub-eval-piper"},
]

job_ids = json.loads(Path("tricky-tts/phase3/phase3_eval_job_ids.json").read_text())

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
        print(f"  ERROR {resp.status_code}: {resp.text[:400]}", flush=True)

Path("tricky-tts/phase3/phase3_eval_job_ids.json").write_text(json.dumps(job_ids, indent=2))
print(f"\nDone. All job IDs saved.", flush=True)
print(json.dumps(job_ids, indent=2), flush=True)
