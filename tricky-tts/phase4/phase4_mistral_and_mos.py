"""
1. Submit MOS eval on tricky-tts-phase4 reference_audio column
2. Submit Mistral TTS eval (mistral/voxtral-mini-tts-2603)
3. Poll both until complete
4. Pull per-row data and print results
"""

import os, json, time, requests
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
import pyarrow.parquet as pq

env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

API_KEY = os.environ["TRELIS_STUDIO_API_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]
TRELIS_API = "https://studio.trelis.com/api/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

PHASE4_DIR = Path(__file__).parent
rows_meta = json.loads((PHASE4_DIR / "rows.json").read_text())

# --- Submit MOS eval on reference audio ---
print("Submitting MOS eval on reference_audio...", flush=True)
r = requests.post(f"{TRELIS_API}/mos-evaluation/jobs", headers=HEADERS, json={
    "dataset_id": "ronanarraig/tricky-tts-phase4",
    "split": "train",
    "audio_column": "reference_audio",
    "num_samples": 4,
    "output_org": "ronanarraig",
    "output_name": "tricky-tts-phase4-reference-mos",
    "push_results": True,
})
if r.status_code == 200:
    mos_job_id = r.json().get("job_id") or r.json().get("id")
    print(f"  MOS job: {mos_job_id}", flush=True)
else:
    print(f"  MOS ERROR {r.status_code}: {r.text[:300]}", flush=True)
    mos_job_id = None

# --- Submit Mistral TTS eval ---
print("\nSubmitting Mistral TTS eval...", flush=True)
r = requests.post(f"{TRELIS_API}/tts-evaluation/jobs", headers=HEADERS, json={
    "model_id": "mistral/voxtral-mini-tts-2603",
    "tts_model_type": "auto",
    "dataset_id": "ronanarraig/tricky-tts-phase4",
    "split": "train",
    "num_samples": 4,
    "asr_model_id": "fireworks/whisper-v3",
    "reference_column": "reference_asr",
    "output_org": "ronanarraig",
    "output_name": "tricky-tts-ph4-v3-mistral",
    "push_results": True,
    "private": True,
})
if r.status_code == 200:
    mistral_job_id = r.json().get("job_id") or r.json().get("id")
    print(f"  Mistral job: {mistral_job_id}", flush=True)
else:
    print(f"  Mistral ERROR {r.status_code}: {r.text[:300]}", flush=True)
    mistral_job_id = None

job_ids = {"MOS": mos_job_id, "Mistral": mistral_job_id}
Path(PHASE4_DIR / "phase4_mistral_mos_job_ids.json").write_text(json.dumps(job_ids, indent=2))

# --- Poll both jobs ---
print("\nPolling jobs...", flush=True)
results = {}
pending = {k: v for k, v in job_ids.items() if v}

while pending:
    still_pending = {}
    for label, job_id in pending.items():
        endpoint = "mos-evaluation" if label == "MOS" else "tts-evaluation"
        r = requests.get(f"{TRELIS_API}/{endpoint}/jobs/{job_id}", headers=HEADERS)
        r.raise_for_status()
        data = r.json()
        status = data.get("status", "unknown")
        if status in ("completed", "failed", "error"):
            print(f"  [{status}] {label}", flush=True)
            results[label] = data
        else:
            still_pending[label] = job_id
    if still_pending:
        print(f"  Waiting: {list(still_pending.keys())}", flush=True)
        time.sleep(30)
    pending = still_pending

Path(PHASE4_DIR / "phase4_mistral_mos_results.json").write_text(json.dumps(results, indent=2))

# --- Pull MOS per-row results ---
print("\n--- MOS scores for human reference audio ---", flush=True)
mos_repo = "ronanarraig/tricky-tts-phase4-reference-mos"
try:
    files = list(list_repo_files(mos_repo, repo_type="dataset", token=HF_TOKEN))
    parquet_files = [f for f in files if f.endswith(".parquet")]
    if parquet_files:
        local = hf_hub_download(repo_id=mos_repo, filename=parquet_files[0], repo_type="dataset", token=HF_TOKEN)
        table = pq.read_table(local)
        d = table.to_pydict()
        print(f"  Columns: {list(d.keys())}", flush=True)
        mos_col = next((c for c in d if "mos" in c.lower() or "utmos" in c.lower()), None)
        if mos_col:
            for i, meta in enumerate(rows_meta):
                score = d[mos_col][i] if i < len(d[mos_col]) else None
                print(f"  Row {i} [{meta['category']}]: MOS={score:.3f}" if score else f"  Row {i}: N/A", flush=True)
except Exception as e:
    print(f"  ERROR: {e}", flush=True)

# --- Pull Mistral per-row results ---
print("\n--- Mistral TTS per-row results ---", flush=True)
mistral_repo = "ronanarraig/tricky-tts-ph4-v3-mistral"
try:
    files = list(list_repo_files(mistral_repo, repo_type="dataset", token=HF_TOKEN))
    parquet_files = [f for f in files if f.endswith(".parquet")]
    if parquet_files:
        local = hf_hub_download(repo_id=mistral_repo, filename=parquet_files[0], repo_type="dataset", token=HF_TOKEN)
        schema_names = pq.read_schema(local).names
        wanted = [c for c in ["text_prompt", "asr_transcription", "asr_wer", "asr_cer", "duration_s", "utmos_score"] if c in schema_names]
        table = pq.read_table(local, columns=wanted)
        d = table.to_pydict()
        n = len(d.get("text_prompt", []))
        print(f"  {n} rows | cols: {wanted}", flush=True)
        cers = [d["asr_cer"][i] for i in range(n) if d.get("asr_cer") and d["asr_cer"][i] is not None]
        moss = [d["utmos_score"][i] for i in range(n) if d.get("utmos_score") and d["utmos_score"][i] is not None]
        if cers: print(f"  avg CER={sum(cers)/len(cers):.3f}", flush=True)
        if moss: print(f"  avg MOS={sum(moss)/len(moss):.3f}", flush=True)
        for i, meta in enumerate(rows_meta):
            if i >= n: break
            asr = str(d.get("asr_transcription", [None]*n)[i] or "N/A")
            cer = d.get("asr_cer", [None]*n)[i]
            print(f"\n  Row {i} [{meta['category']}] CER={f'{cer:.3f}' if cer else 'N/A'}", flush=True)
            print(f"    TEXT: {meta['text'][:90]}", flush=True)
            print(f"    ASR:  {asr[:90]}", flush=True)
        # Save for report
        perrow_mistral = [{k: d[k][i] if k in d else None for k in ["text_prompt","asr_transcription","asr_wer","asr_cer","duration_s","utmos_score"]} for i in range(n)]
        Path(PHASE4_DIR / "phase4_v3_mistral_perrow.json").write_text(json.dumps(perrow_mistral, indent=2))
except Exception as e:
    print(f"  ERROR: {e}", flush=True)

print("\nDone.", flush=True)
