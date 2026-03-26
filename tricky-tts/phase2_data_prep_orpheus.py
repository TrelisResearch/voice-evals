"""
Data prep TTS pipeline: Orpheus on spoken_form texts → HF dataset with audio column.

Workflow:
1. Load spoken_form texts from tricky-tts-proto-v4 HF dataset
2. Upload one .txt file per row to a new file store
3. Run Orpheus TTS via data prep → output file store (WAV + VTT)
4. Process output file store → push to ronanarraig/tricky-tts-proto-ref-orpheus-datap
"""

import os, time, requests
from pathlib import Path

env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

API_KEY = os.environ["TRELIS_STUDIO_API_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]
TRELIS_API = "https://studio.trelis.com/api/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

def poll_datap_job(job_id: str, interval: int = 15):
    while True:
        time.sleep(interval)
        resp = requests.get(f"{TRELIS_API}/data-prep/jobs/{job_id}", headers=HEADERS)
        if not resp.text.strip():
            print("  empty response, retrying...", flush=True)
            continue
        j = resp.json()
        r = j.get("result") or {}
        print(f"  {j['status']}  {r}", flush=True)
        if j["status"] in ("completed", "failed", "stopped"):
            return j

# ── Load spoken_form texts ────────────────────────────────────────────────────
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

local = hf_hub_download("ronanarraig/tricky-tts-proto-v4",
    "data/train-00000-of-00001.parquet", repo_type="dataset", token=HF_TOKEN)
t = pq.read_table(local, columns=["text", "spoken_form", "category"])
d = t.to_pydict()
rows = list(zip(d["text"], d["spoken_form"], d["category"]))
print(f"Loaded {len(rows)} rows", flush=True)

# ── Upload .txt files to a new file store ─────────────────────────────────────
print("\nRequesting batch upload URLs...", flush=True)
filenames = [f"row_{i:02d}_{cat}.txt" for i, (_, _, cat) in enumerate(rows)]
file_contents = [sf.encode("utf-8") for _, sf, _ in rows]

resp = requests.post(f"{TRELIS_API}/file-stores/upload-urls", headers=HEADERS, json={
    "files": [{"filename": fn, "size_bytes": len(fc), "content_type": "text/plain"}
              for fn, fc in zip(filenames, file_contents)],
    "name": "tricky-tts-proto-spoken-form-txt",
})
print(f"  Status: {resp.status_code}", flush=True)
upload_resp = resp.json()
print(f"  file_store_id: {upload_resp.get('file_store_id')}", flush=True)
file_store_id = upload_resp["file_store_id"]
upload_urls = upload_resp["files"]  # list of {filename, upload_url}

# Upload each file
url_map = {u["filename"]: u["upload_url"] for u in upload_urls}
for fn, fc in zip(filenames, file_contents):
    put_resp = requests.put(url_map[fn], data=fc,
        headers={"Content-Type": "text/plain"})
    print(f"  Uploaded {fn}: {put_resp.status_code}", flush=True)

# ── Run Orpheus TTS on file store ─────────────────────────────────────────────
print(f"\nSubmitting Orpheus TTS data prep job on store {file_store_id}...", flush=True)
resp = requests.post(f"{TRELIS_API}/file-stores/{file_store_id}/tts", headers=HEADERS, json={
    "model_id": "unsloth/orpheus-3b-0.1-ft",
    "engine": "orpheus",
    "speaker_name": "tara",
    "max_new_tokens": 4000,
    "language": "en",
})
print(f"  Status: {resp.status_code}  Body: {resp.text[:200]}", flush=True)
tts_job = resp.json()
tts_job_id = tts_job.get("job_id") or tts_job.get("id")
tts_store_id = tts_job.get("output_file_store_id")
print(f"  TTS job_id: {tts_job_id}  output_store: {tts_store_id}", flush=True)

j = poll_datap_job(tts_job_id)
if j["status"] != "completed":
    print(f"TTS job FAILED: {j.get('error')}"); exit(1)

# Get output store id from result if not in initial response
if not tts_store_id:
    tts_store_id = (j.get("result") or {}).get("output_file_store_id") or \
                   (j.get("result") or {}).get("file_store_id")
print(f"Output file store: {tts_store_id}", flush=True)

# ── Process → push to HF ──────────────────────────────────────────────────────
print(f"\nProcessing output store {tts_store_id} → pushing to HF...", flush=True)
resp = requests.post(f"{TRELIS_API}/file-stores/{tts_store_id}/process", headers=HEADERS, json={
    "output_org": "ronanarraig",
    "output_dataset_name": "tricky-tts-proto-ref-orpheus-datap",
    "hf_token": HF_TOKEN,
    "private": True,
    "split_option": "train_only",
    "enable_quality_checks": False,
    "language": "english",
    "target_chunk_duration": 30.0,
    "max_chunk_duration": 30.0,
})
print(f"  Status: {resp.status_code}  Body: {resp.text[:200]}", flush=True)
proc_job = resp.json()
proc_job_id = proc_job.get("job_id") or proc_job.get("id")
print(f"  Process job_id: {proc_job_id}", flush=True)

j = poll_datap_job(proc_job_id)
if j["status"] != "completed":
    print(f"Process job FAILED: {j.get('error')}"); exit(1)

hf_dataset = (j.get("result") or {}).get("hf_dataset_id", "ronanarraig/tricky-tts-proto-ref-orpheus-datap")
print(f"\nDone. Dataset: {hf_dataset}", flush=True)
