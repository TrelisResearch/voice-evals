"""
Test whether ellipsis (...) triggers early EOS in Mistral Voxtral-Mini.
Re-runs row 3 (prosody_and_punctuation) with ... replaced by , then compares.
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
HEADERS = {"Authorization": f"Bearer {API_KEY}"}

PHASE4_DIR = Path(__file__).parent

row3_original = (
    'He started snoring \u2014 zzz, zzz \u2014 right in the middle of the lecture. '
    '"Psst," she hissed, nudging him. "Wake up!" He jolted awake. '
    '"Huh? What... what happened?" She sighed. "Shhh \u2014 just pay attention." '
    'Outside, the wind went whoosh through the open window, and somewhere far off... drip, drip, drip.'
)

# Replace ellipsis with comma+space to test EOS hypothesis
row3_no_ellipsis = row3_original.replace("...", ",")

print("Original:    ", row3_original)
print("No-ellipsis: ", row3_no_ellipsis)
print()

prompts = [row3_original, row3_no_ellipsis]
labels = ["original (with ...)", "no-ellipsis (... → ,)"]

# Submit single job with both prompts
print("Submitting Mistral TTS job with 2 prompts...", flush=True)
r = requests.post(f"{TRELIS_API}/tts-evaluation/jobs", headers=HEADERS, json={
    "model_id": "mistral/voxtral-mini-tts-2603",
    "tts_model_type": "auto",
    "prompts": prompts,
    "asr_model_id": "fireworks/whisper-v3",
    "output_org": "ronanarraig",
    "output_name": "tricky-tts-ph4-mistral-ellipsis-test",
    "push_results": True,
    "private": True,
})
if r.status_code != 200:
    print(f"ERROR {r.status_code}: {r.text}")
    exit(1)
job_id = r.json().get("job_id") or r.json().get("id")
print(f"  Job: {job_id}", flush=True)

# Poll
print("\nPolling...", flush=True)
while True:
    r = requests.get(f"{TRELIS_API}/tts-evaluation/jobs/{job_id}", headers=HEADERS)
    data = r.json()
    status = data.get("status", "unknown")
    print(f"  {status}", flush=True)
    if status in ("completed", "failed", "error"):
        break
    time.sleep(15)

print(f"\nResult: {json.dumps(data.get('result', {}), indent=2)}", flush=True)

# Pull per-row results from HF
if status == "completed":
    from huggingface_hub import hf_hub_download, list_repo_files
    import pyarrow.parquet as pq

    HF_TOKEN = os.environ["HF_TOKEN"]
    repo = "ronanarraig/tricky-tts-ph4-mistral-ellipsis-test"
    files = list(list_repo_files(repo, repo_type="dataset", token=HF_TOKEN))
    parquet_files = [f for f in files if f.endswith(".parquet")]
    local = hf_hub_download(repo_id=repo, filename=parquet_files[0], repo_type="dataset", token=HF_TOKEN)
    table = pq.read_table(local)
    d = table.to_pydict()
    print(f"\nColumns: {list(d.keys())}", flush=True)

    results = []
    for i, label in enumerate(labels):
        asr = str(d.get("asr_transcription", [None]*2)[i] or "N/A")
        cer = d.get("asr_cer", [None]*2)[i]
        dur = d.get("duration_s", [None]*2)[i]
        print(f"\n  [{label}]")
        print(f"    CER={f'{cer:.3f}' if cer is not None else 'N/A'}  dur={f'{dur:.1f}s' if dur else 'N/A'}")
        print(f"    ASR: {asr}")
        results.append({"label": label, "prompt": prompts[i], "asr_transcription": asr, "asr_cer": cer, "duration_s": dur})

    Path(PHASE4_DIR / "phase4_mistral_ellipsis_results.json").write_text(json.dumps(results, indent=2))
    print("\nSaved to phase4_mistral_ellipsis_results.json", flush=True)

print("\nDone.", flush=True)
