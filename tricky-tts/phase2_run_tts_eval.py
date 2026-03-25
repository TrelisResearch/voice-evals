"""
Phase 2 Step 3: Submit TTS evaluation jobs via Trelis Studio.

Models (3 proprietary + Kokoro, no Orpheus per user instruction):
- elevenlabs/eleven-multilingual-v2  (Router, used in Phase 1)
- cartesia/sonic-3                   (Router, quality model)
- openai/gpt-4o-mini-tts             (Router, widely used)
- kokoro                             (open-source, via Studio)

Each job:
- dataset_id: ronanarraig/tricky-tts-v2-public
- split: train (48 rows)
- asr_model_id: openai/whisper-large-v3-turbo
- push_results: true → per-row audio + transcripts pushed to HF
- output_org: ronanarraig
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
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

DATASET_ID = "ronanarraig/tricky-tts-v2-public"

MODELS = [
    {
        "model_id": "elevenlabs/eleven-multilingual-v2",
        "tts_model_type": "auto",
        "output_name": "tricky-tts-v2-eval-elevenlabs",
        "label": "ElevenLabs",
    },
    {
        "model_id": "cartesia/sonic-3",
        "tts_model_type": "auto",
        "output_name": "tricky-tts-v2-eval-cartesia",
        "label": "Cartesia Sonic-3",
    },
    {
        "model_id": "openai/gpt-4o-mini-tts",
        "tts_model_type": "auto",
        "output_name": "tricky-tts-v2-eval-gpt4o-mini",
        "label": "GPT-4o mini TTS",
    },
    {
        "model_id": "kokoro",
        "tts_model_type": "kokoro",
        "kokoro_voice": "af_heart",
        "output_name": "tricky-tts-v2-eval-kokoro",
        "label": "Kokoro",
    },
]

job_ids = {}

for m in MODELS:
    payload = {
        "model_id": m["model_id"],
        "dataset_id": DATASET_ID,
        "split": "train",
        "num_samples": 48,
        "asr_model_id": "openai/whisper-large-v3-turbo",
        "language": "auto",
        "tts_model_type": m["tts_model_type"],
        "push_results": True,
        "output_org": "ronanarraig",
        "output_name": m["output_name"],
        "private": True,
    }
    if "kokoro_voice" in m:
        payload["kokoro_voice"] = m["kokoro_voice"]

    print(f"Submitting: {m['label']} ({m['model_id']})...", flush=True)
    resp = requests.post(f"{TRELIS_API}/tts-evaluation/jobs", headers=HEADERS, json=payload)

    if resp.status_code in (200, 201):
        job = resp.json()
        job_id = job.get("id") or job.get("job_id")
        job_ids[m["label"]] = {"job_id": job_id, "model_id": m["model_id"], "output_name": m["output_name"]}
        print(f"  → Job ID: {job_id}", flush=True)
    else:
        print(f"  ERROR {resp.status_code}: {resp.text[:300]}", flush=True)

# Save job IDs for polling
out_path = Path("tricky-tts/phase2_job_ids.json")
out_path.write_text(json.dumps(job_ids, indent=2))
print(f"\nJob IDs saved to {out_path}", flush=True)
print(json.dumps(job_ids, indent=2), flush=True)
