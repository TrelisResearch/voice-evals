"""
Phase 2 Reference Pipeline (Prototype — 10 rows)

Steps:
1. Push a spoken_form-as-text dataset (10 rows) to HF
2. Run Orpheus TTS on spoken_form + AssemblyAI ASR → reference audio + reference_asr_transcript
3. Download per-row asr_transcription from pushed eval dataset
4. Build prototype dataset with reference_asr_transcript column
5. Run ElevenLabs test eval with reference_column="reference_asr_transcript"
6. Print side-by-side inspection: text | spoken_form | ref_asr | test_asr | cer
"""

import os, json, time, requests
from pathlib import Path
from collections import defaultdict

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

# ── Step 1: Select 10 prototype rows ──────────────────────────────────────────
data = json.loads(Path("tricky-tts/phase2/phase2_final.json").read_text())
perrow = json.loads(Path("tricky-tts/phase2/phase2_perrow_results.json").read_text())
cer_by_text = {r["text"]: r.get("median_cer", 0) for r in perrow}

by_cat = defaultdict(list)
for r in data:
    by_cat[r["category"]].append(r)

sample = []
for cat, take in [("edge_cases",2),("domain_specific",2),("ai_tech",2),
                   ("number_format",2),("phonetic",1),("paralinguistics",1)]:
    rows = sorted(by_cat[cat], key=lambda r: -cer_by_text.get(r["text"], 0))
    sample.extend(rows[:take])
sample = sample[:10]
print(f"Prototype: {len(sample)} rows", flush=True)

# ── Step 2: Push spoken_form-as-text dataset ──────────────────────────────────
from datasets import Dataset
from huggingface_hub import HfApi

# Dataset for reference TTS: text=spoken_form, keep original text in orig_text
ref_rows = [{"text": r["spoken_form"], "orig_text": r["text"], "category": r["category"]} for r in sample]
ref_repo = "ronanarraig/tricky-tts-proto-spoken-form"
api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=ref_repo, repo_type="dataset", private=True, exist_ok=True)
Dataset.from_list(ref_rows).push_to_hub(ref_repo, token=HF_TOKEN, private=True, split="train")
print(f"Pushed spoken_form dataset to {ref_repo}", flush=True)

# ── Step 3: Run Orpheus TTS on spoken_form + AssemblyAI ASR ──────────────────
print("\nSubmitting Orpheus reference TTS job...", flush=True)
resp = requests.post(f"{TRELIS_API}/tts-evaluation/jobs", headers=HEADERS, json={
    "model_id": "unsloth/orpheus-3b-0.1-ft",
    "dataset_id": ref_repo,
    "split": "train",
    "num_samples": 10,
    "asr_model_id": "assemblyai/universal-3-pro",
    "language": "auto",
    "tts_model_type": "orpheus",
    "push_results": True,
    "output_org": "ronanarraig",
    "output_name": "tricky-tts-proto-ref-orpheus",
    "private": True,
})
ref_job_id = resp.json().get("id") or resp.json().get("job_id")
print(f"Reference job ID: {ref_job_id}", flush=True)

# Poll
print("Polling reference job...", flush=True)
while True:
    time.sleep(30)
    j = requests.get(f"{TRELIS_API}/tts-evaluation/jobs/{ref_job_id}", headers=HEADERS).json()
    status = j["status"]
    r = j.get("result") or {}
    print(f"  {status}  MOS={r.get('mos','')}  CER={r.get('cer','')}", flush=True)
    if status in ("completed", "failed", "stopped"):
        break

if j["status"] != "completed":
    print(f"FAILED: {j.get('error')}")
    exit(1)

print(f"Reference dataset: {j['result']['dataset_url']}", flush=True)

# ── Step 4: Download reference ASR transcripts ────────────────────────────────
from huggingface_hub import hf_hub_download
import pyarrow.parquet as pq

local = hf_hub_download("ronanarraig/tricky-tts-proto-ref-orpheus",
    "data/train-00000-of-00001.parquet", repo_type="dataset", token=HF_TOKEN)
t = pq.read_table(local, columns=["text_prompt", "asr_transcription", "asr_cer"])
ref_data = t.to_pydict()

# Map spoken_form → reference ASR transcript
spoken_to_ref_asr = dict(zip(ref_data["text_prompt"], ref_data["asr_transcription"]))
spoken_to_ref_cer = dict(zip(ref_data["text_prompt"], ref_data["asr_cer"]))

print(f"\nGot reference ASR for {len(spoken_to_ref_asr)} rows", flush=True)

# ── Step 5: Build prototype dataset with reference_asr_transcript column ──────
proto_rows = []
for r in sample:
    ref_asr = spoken_to_ref_asr.get(r["spoken_form"], "")
    proto_rows.append({
        "text": r["text"],
        "category": r["category"],
        "spoken_form": r["spoken_form"],
        "cer_reliable": r["cer_reliable"],
        "reference_asr_transcript": ref_asr,
    })

proto_repo = "ronanarraig/tricky-tts-proto-with-reference"
api.create_repo(repo_id=proto_repo, repo_type="dataset", private=True, exist_ok=True)
Dataset.from_list(proto_rows).push_to_hub(proto_repo, token=HF_TOKEN, private=True, split="train")
print(f"Pushed prototype dataset (with reference_asr_transcript) to {proto_repo}", flush=True)

# Save reference data locally
Path("tricky-tts/phase2/phase2_proto_reference.json").write_text(
    json.dumps(proto_rows, indent=2, ensure_ascii=False))

# ── Step 6: Run ElevenLabs test eval with reference_column ────────────────────
print("\nSubmitting ElevenLabs test eval with reference_column...", flush=True)
resp = requests.post(f"{TRELIS_API}/tts-evaluation/jobs", headers=HEADERS, json={
    "model_id": "elevenlabs/eleven-multilingual-v2",
    "dataset_id": proto_repo,
    "split": "train",
    "num_samples": 10,
    "asr_model_id": "assemblyai/universal-3-pro",
    "language": "auto",
    "tts_model_type": "auto",
    "push_results": True,
    "output_org": "ronanarraig",
    "output_name": "tricky-tts-proto-test-elevenlabs",
    "reference_column": "reference_asr_transcript",
    "private": True,
})
test_job_id = resp.json().get("id") or resp.json().get("job_id")
print(f"Test job ID: {test_job_id}", flush=True)

# Poll
print("Polling test job...", flush=True)
while True:
    time.sleep(30)
    j = requests.get(f"{TRELIS_API}/tts-evaluation/jobs/{test_job_id}", headers=HEADERS).json()
    status = j["status"]
    r = j.get("result") or {}
    print(f"  {status}  MOS={r.get('mos','')}  CER={r.get('cer','')}", flush=True)
    if status in ("completed", "failed", "stopped"):
        break

if j["status"] != "completed":
    print(f"FAILED: {j.get('error')}")
    exit(1)

# ── Step 7: Download test results and inspect ─────────────────────────────────
local2 = hf_hub_download("ronanarraig/tricky-tts-proto-test-elevenlabs",
    "data/train-00000-of-00001.parquet", repo_type="dataset", token=HF_TOKEN)
t2 = pq.read_table(local2, columns=["text_prompt", "asr_transcription", "asr_cer", "asr_wer"])
test_data = t2.to_pydict()
test_by_text = dict(zip(test_data["text_prompt"], zip(
    test_data["asr_transcription"], test_data["asr_cer"])))

# Also load old CER (vs written text) for comparison
old_perrow = {r["text"]: r.get("median_cer") for r in perrow}

print("\n" + "="*100)
print("PROTOTYPE INSPECTION: reference_asr pipeline vs old CER-vs-written-text")
print("="*100)

rows_sorted = sorted(proto_rows, key=lambda r: -(test_by_text.get(r["text"], ("",0))[1] or 0))

for r in rows_sorted:
    test_asr, test_cer = test_by_text.get(r["text"], ("N/A", None))
    ref_asr = r["reference_asr_transcript"]
    old_cer = old_perrow.get(r["text"])
    ref_cer_vs_spoken = spoken_to_ref_cer.get(r["spoken_form"], None)

    print(f"\n[{r['category']}]")
    print(f"  text:      {r['text'][:90]}")
    print(f"  spoken:    {r['spoken_form'][:90]}")
    print(f"  ref_asr:   {ref_asr[:90]}")
    print(f"  test_asr:  {test_asr[:90]}")
    if test_cer is not None and old_cer is not None:
        delta = test_cer - old_cer
        print(f"  CER: old={old_cer:.3f}  ref_pipeline={test_cer:.3f}  Δ={delta:+.3f}")
    elif test_cer is not None:
        print(f"  CER (ref_pipeline): {test_cer:.3f}")

print("\nDone.", flush=True)
