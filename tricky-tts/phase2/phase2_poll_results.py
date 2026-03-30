"""
Phase 2 Step 4: Poll TTS evaluation jobs until complete, then collect per-row results
from pushed HuggingFace datasets and compute difficulty metrics.
"""

import os, json, time, requests
from pathlib import Path

env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

API_KEY = os.environ["TRELIS_STUDIO_API_KEY"]
TRELIS_API = "https://studio.trelis.com/api/v1"
HF_TOKEN = os.environ["HF_TOKEN"]
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

job_ids = json.loads(Path("tricky-tts/phase2/phase2_job_ids.json").read_text())

def poll_job(job_id: str) -> dict:
    resp = requests.get(f"{TRELIS_API}/tts-evaluation/jobs/{job_id}", headers=HEADERS)
    resp.raise_for_status()
    return resp.json()

def poll_all_until_done(max_wait_s: int = 3600) -> dict:
    results = {}
    pending = dict(job_ids)
    start = time.time()

    while pending and time.time() - start < max_wait_s:
        time.sleep(30)
        done_keys = []
        for label, info in pending.items():
            job = poll_job(info["job_id"])
            status = job["status"]
            elapsed = int(time.time() - start)
            print(f"  [{elapsed:4d}s] {label}: {status}", flush=True)

            if status == "completed":
                results[label] = {**info, "result": job["result"], "status": "completed"}
                done_keys.append(label)
                print(f"    MOS={job['result'].get('mos'):.3f}  WER={job['result'].get('wer'):.4f}  CER={job['result'].get('cer'):.4f}  dataset={job['result'].get('dataset_url')}", flush=True)
            elif status == "failed":
                results[label] = {**info, "error": job.get("error"), "status": "failed"}
                done_keys.append(label)
                print(f"    FAILED: {job.get('error')}", flush=True)

        for k in done_keys:
            del pending[k]

        if pending:
            print(f"  Still pending: {list(pending.keys())}", flush=True)

    return results

print("Polling TTS evaluation jobs...", flush=True)
print(f"Jobs: {list(job_ids.keys())}", flush=True)

all_results = poll_all_until_done()

# Save aggregate results
out_path = Path("tricky-tts/phase2/phase2_eval_results.json")
out_path.write_text(json.dumps(all_results, indent=2))
print(f"\nResults saved to {out_path}", flush=True)

# Summary table
print("\n=== AGGREGATE RESULTS ===")
print(f"{'Model':<25} {'Status':<12} {'MOS':>6} {'WER':>7} {'CER':>7}")
print("-" * 65)
for label, info in all_results.items():
    if info["status"] == "completed":
        r = info["result"]
        print(f"{label:<25} {'completed':<12} {r.get('mos', 0):.3f}  {r.get('wer', 0):.4f}  {r.get('cer', 0):.4f}")
    else:
        print(f"{label:<25} {'FAILED':<12} {'':>6} {'':>7} {'':>7}")
