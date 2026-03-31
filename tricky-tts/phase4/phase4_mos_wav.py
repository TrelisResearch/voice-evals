"""
Push WAV version of reference audio and run MOS eval on it.
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
wav_files = [
    PHASE4_DIR / "audio_wav" / "row0_symbol_expansion.wav",
    PHASE4_DIR / "audio_wav" / "row1_abbreviation_reading.wav",
    PHASE4_DIR / "audio_wav" / "row2_proper_nouns.wav",
    PHASE4_DIR / "audio_wav" / "row3_prosody_and_punctuation.wav",
]

# Push WAV audio dataset
print("Pushing WAV reference audio dataset...", flush=True)
audio_data = [{"bytes": p.read_bytes(), "path": str(p)} for p in wav_files]
ds = Dataset.from_dict({
    "audio": audio_data,
    "text": [r["text"] for r in rows_meta],
    "category": [r["category"] for r in rows_meta],
})
ds = ds.cast_column("audio", Audio())
ds.push_to_hub("ronanarraig/tricky-tts-phase4-wav", split="train", token=HF_TOKEN, private=True)
print("  Pushed ronanarraig/tricky-tts-phase4-wav", flush=True)

# Submit MOS eval
print("\nSubmitting MOS eval on WAV reference audio...", flush=True)
r = requests.post(f"{TRELIS_API}/mos-evaluation/jobs", headers=HEADERS, json={
    "dataset_id": "ronanarraig/tricky-tts-phase4-wav",
    "split": "train",
    "audio_column": "audio",
    "num_samples": 4,
    "output_org": "ronanarraig",
    "output_name": "tricky-tts-phase4-reference-mos",
    "push_results": True,
})
if r.status_code != 200:
    print(f"  ERROR {r.status_code}: {r.text}", flush=True)
    exit(1)
job_id = r.json().get("job_id") or r.json().get("id")
print(f"  Job: {job_id}", flush=True)

# Poll
print("\nPolling...", flush=True)
while True:
    r = requests.get(f"{TRELIS_API}/mos-evaluation/jobs/{job_id}", headers=HEADERS)
    data = r.json()
    status = data.get("status", "unknown")
    print(f"  {status}", flush=True)
    if status in ("completed", "failed", "error"):
        break
    time.sleep(20)

print(f"\nResult: {json.dumps(data.get('result', {}), indent=2)[:300]}", flush=True)
print(f"Logs: {data.get('logs','')[-300:]}", flush=True)

Path(PHASE4_DIR / "phase4_mos_result.json").write_text(json.dumps(data, indent=2))

# Pull per-row MOS
if status == "completed":
    print("\nPulling per-row MOS scores...", flush=True)
    mos_repo = "ronanarraig/tricky-tts-phase4-reference-mos"
    try:
        files = list(list_repo_files(mos_repo, repo_type="dataset", token=HF_TOKEN))
        parquet_files = [f for f in files if f.endswith(".parquet")]
        local = hf_hub_download(repo_id=mos_repo, filename=parquet_files[0], repo_type="dataset", token=HF_TOKEN)
        table = pq.read_table(local)
        d = table.to_pydict()
        print(f"  Columns: {list(d.keys())}", flush=True)
        mos_col = next((c for c in d if "mos" in c.lower() or "utmos" in c.lower()), None)
        scores = []
        for i, meta in enumerate(rows_meta):
            score = d[mos_col][i] if mos_col and i < len(d[mos_col]) else None
            scores.append(score)
            print(f"  Row {i} [{meta['category']}]: MOS={score:.3f}" if score else f"  Row {i}: N/A", flush=True)
        Path(PHASE4_DIR / "phase4_reference_mos_scores.json").write_text(json.dumps(scores, indent=2))
    except Exception as e:
        print(f"  ERROR: {e}", flush=True)
