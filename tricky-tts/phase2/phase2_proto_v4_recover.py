"""Poll the already-submitted v4 jobs and print results."""
import json, os, time, requests
from pathlib import Path
import editdistance
from huggingface_hub import hf_hub_download
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
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

JOBS = {
    "elevenlabs":   "7b7dcbff-61a1-4531-ae16-13baae750b33",
    "orpheus":      "a5932df5-40bc-4bf5-b605-79c216585d87",
    "gemini-flash": "6b66b8d3-f02f-4a0a-8ba9-d33722bdbf91",
}

def compute_cer(ref: str, hyp: str) -> float:
    ref, hyp = ref.strip().lower(), hyp.strip().lower()
    if not ref:
        return 0.0
    return editdistance.eval(ref, hyp) / len(ref)

# Poll until all done
results = {}
pending = dict(JOBS)
while pending:
    time.sleep(15)
    done = []
    for name, job_id in pending.items():
        try:
            resp = requests.get(f"{TRELIS_API}/tts-evaluation/jobs/{job_id}", headers=HEADERS)
            if not resp.text.strip():
                print(f"  [{name}] empty response, retrying...", flush=True)
                continue
            j = resp.json()
        except Exception as e:
            print(f"  [{name}] error: {e}, retrying...", flush=True)
            continue
        r = j.get("result") or {}
        print(f"  [{name}] {j['status']}  MOS={r.get('mos','')}  CER={r.get('cer','')}", flush=True)
        if j["status"] in ("completed", "failed", "stopped"):
            results[name] = j
            done.append(name)
    for name in done:
        del pending[name]

# Download results
test_rows = json.loads(Path("tricky-tts/phase2/phase2_proto_v4_rows.json").read_text()) \
    if Path("tricky-tts/phase2/phase2_proto_v4_rows.json").exists() else None

perrow = json.loads(Path("tricky-tts/phase2/phase2_perrow_results.json").read_text())
old_cer = {r["text"]: r.get("median_cer") for r in perrow}

model_results = {}
for short_name in JOBS:
    if results.get(short_name, {}).get("status") != "completed":
        print(f"  {short_name} FAILED: {results.get(short_name, {}).get('error')}", flush=True)
        continue
    try:
        local = hf_hub_download(
            f"ronanarraig/tricky-tts-proto-test-{short_name}-v4",
            "data/train-00000-of-00001.parquet", repo_type="dataset", token=HF_TOKEN)
        t = pq.read_table(local, columns=["text_prompt", "asr_transcription", "asr_cer"])
        d = t.to_pydict()
        model_results[short_name] = dict(zip(d["text_prompt"],
            zip(d["asr_transcription"], d["asr_cer"])))
        print(f"  Downloaded {short_name}: {len(model_results[short_name])} rows", flush=True)
    except Exception as e:
        print(f"  Failed to download {short_name}: {e}", flush=True)

# Need test_rows — reconstruct from one of the result datasets
if test_rows is None:
    # Use elevenlabs text_prompts as row order
    name = next(iter(model_results))
    texts = list(model_results[name].keys())
    # Load spoken_form from the pushed HF dataset
    from huggingface_hub import hf_hub_download as hfd
    local_test = hfd("ronanarraig/tricky-tts-proto-v4",
        "data/train-00000-of-00001.parquet", repo_type="dataset", token=HF_TOKEN)
    t_test = pq.read_table(local_test)
    d_test = t_test.to_pydict()
    test_rows = [{"text": tx, "spoken_form": sf, "category": cat, "cer_reliable": cr}
        for tx, sf, cat, cr in zip(d_test["text"], d_test["spoken_form"],
                                    d_test["category"], d_test["cer_reliable"])]

print("\n" + "="*110)
print("PROTOTYPE v4 — CER(spoken_form, test_asr)  |  ASR: whisper-large-v3")
print("="*110)

for r in sorted(test_rows, key=lambda x: x["category"]):
    print(f"\n[{r['category']}]")
    print(f"  text:    {r['text'][:90]}")
    print(f"  spoken:  {r['spoken_form'][:90]}")
    o_cer = old_cer.get(r["text"])
    for name in ["elevenlabs", "orpheus", "gemini-flash"]:
        if name not in model_results:
            continue
        test_asr, _ = model_results[name].get(r["text"], ("N/A", None))
        ref_cer = compute_cer(r["spoken_form"], test_asr)
        delta = f"  Δ={ref_cer - o_cer:+.3f}" if o_cer is not None else ""
        print(f"  [{name:14s}] CER={ref_cer:.3f}{delta}  asr: {test_asr[:65]}")

print("\nDone.", flush=True)
