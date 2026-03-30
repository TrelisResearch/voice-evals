"""
Poll all 9 phase4-v2 eval jobs (with reference_asr), pull per-row data,
print leaderboard, save results.
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
job_ids = json.loads((PHASE4_DIR / "phase4_v2_eval_job_ids.json").read_text())
rows_meta = json.loads((PHASE4_DIR / "rows.json").read_text())
reference_asr = json.loads((PHASE4_DIR / "phase4_reference_asr.json").read_text())

print(f"Polling {len(job_ids)} jobs...\n", flush=True)

completed = {}
pending = {k: v for k, v in job_ids.items() if v.get("job_id")}

while pending:
    still_pending = {}
    for label, info in pending.items():
        r = requests.get(f"{TRELIS_API}/tts-evaluation/jobs/{info['job_id']}", headers=HEADERS)
        r.raise_for_status()
        data = r.json()
        status = data.get("status", "unknown")
        if status in ("completed", "failed", "error"):
            print(f"  [{status}] {label}", flush=True)
            completed[label] = {"info": info, "result": data, "status": status}
        else:
            still_pending[label] = info
    if still_pending:
        print(f"  Waiting ({len(still_pending)}): {list(still_pending.keys())}", flush=True)
        time.sleep(30)
    pending = still_pending

print(f"\nAll done. {sum(1 for v in completed.values() if v['status']=='completed')}/{len(completed)} completed.\n", flush=True)
Path(PHASE4_DIR / "phase4_v2_eval_results.json").write_text(json.dumps(completed, indent=2))

# Pull per-row data from HF
LABEL_ORDER = ["ElevenLabs", "GPT-4o mini TTS", "Cartesia Sonic-3", "Gemini Flash TTS",
               "Gemini Pro TTS", "Orpheus", "Kokoro", "Piper (en-gb)", "Chatterbox"]

print("Pulling per-row data from HF datasets...\n", flush=True)
perrow = {}

for label in LABEL_ORDER:
    info = job_ids.get(label, {})
    if not info.get("job_id"):
        continue
    # Find the result dataset from job result
    result_data = completed.get(label, {}).get("result", {})
    repo_id = result_data.get("output_dataset") or result_data.get("hf_dataset_id")
    if not repo_id:
        # Try the output_name convention
        slug = label.lower().replace(' ', '-').replace('(', '').replace(')', '').replace('/', '-')
        repo_id = f"ronanarraig/tricky-tts-ph4-v2-{slug}"
    print(f"  {label} -> {repo_id}", flush=True)
    try:
        files = list(list_repo_files(repo_id, repo_type="dataset", token=HF_TOKEN))
        parquet_files = [f for f in files if f.endswith(".parquet")]
        if not parquet_files:
            print(f"    no parquet files", flush=True)
            continue
        local = hf_hub_download(repo_id=repo_id, filename=parquet_files[0], repo_type="dataset", token=HF_TOKEN)
        schema_names = pq.read_schema(local).names
        wanted = [c for c in ["text_prompt", "asr_transcription", "asr_wer", "asr_cer", "duration_s", "utmos_score"] if c in schema_names]
        table = pq.read_table(local, columns=wanted)
        rows_dict = table.to_pydict()
        n = len(rows_dict.get("text_prompt", []))
        perrow[label] = [
            {k: rows_dict[k][i] if k in rows_dict else None
             for k in ["text_prompt", "asr_transcription", "asr_wer", "asr_cer", "duration_s", "utmos_score"]}
            for i in range(n)
        ]
        cers = [r["asr_cer"] for r in perrow[label] if r.get("asr_cer") is not None]
        mos = [r["utmos_score"] for r in perrow[label] if r.get("utmos_score") is not None]
        print(f"    {n} rows | avg CER={sum(cers)/len(cers):.3f}" + (f" | avg MOS={sum(mos)/len(mos):.3f}" if mos else ""), flush=True)
    except Exception as e:
        print(f"    ERROR: {e}", flush=True)

Path(PHASE4_DIR / "phase4_v2_perrow.json").write_text(json.dumps(perrow, indent=2))

# Leaderboard
print("\n" + "="*80, flush=True)
print(f"{'Model':<22} {'MOS':>6} {'WER':>7} {'CER':>7}  (CER vs reference_asr)", flush=True)
print("-"*80, flush=True)

summary = []
for label in LABEL_ORDER:
    data = completed.get(label, {})
    r = data.get("result", {})
    inner = r.get("result", {}) or {}
    mos = inner.get("mos") or r.get("utmos_score") or r.get("mos_score")
    wer = inner.get("wer") or r.get("average_wer") or r.get("wer")
    cer = inner.get("cer") or r.get("average_cer") or r.get("cer")
    if label in perrow:
        cers = [x["asr_cer"] for x in perrow[label] if x.get("asr_cer") is not None]
        wers = [x["asr_wer"] for x in perrow[label] if x.get("asr_wer") is not None]
        moss = [x["utmos_score"] for x in perrow[label] if x.get("utmos_score") is not None]
        if cers: cer = sum(cers)/len(cers)
        if wers: wer = sum(wers)/len(wers)
        if moss: mos = sum(moss)/len(moss)
    mos_s = f"{mos:.3f}" if mos is not None else " N/A"
    wer_s = f"{wer:.3f}" if wer is not None else " N/A"
    cer_s = f"{cer:.3f}" if cer is not None else " N/A"
    print(f"{label:<22} {mos_s:>6} {wer_s:>7} {cer_s:>7}", flush=True)
    summary.append({"model": label, "mos": mos, "wer": wer, "cer": cer})

print("="*80, flush=True)

# Per-row CER table
models_with_cer = [l for l in LABEL_ORDER if l in perrow and any(r.get("asr_cer") is not None for r in perrow[l])]
if models_with_cer:
    print(f"\n{'Row':<3} {'Category':<25} {'Reference ASR (human voice)':<50}", flush=True)
    print(f"     " + " ".join(f"{m[:9]:>9}" for m in models_with_cer), flush=True)
    print("-"*150, flush=True)
    for i, meta in enumerate(rows_meta):
        ref = reference_asr[i][:48] if i < len(reference_asr) else ""
        row_cers = []
        for label in models_with_cer:
            cer = perrow[label][i].get("asr_cer") if i < len(perrow[label]) else None
            row_cers.append(f"{cer:.3f}" if cer is not None else "  N/A")
        print(f"{i:<3} {meta['category']:<25} {ref:<50}", flush=True)
        print(f"     " + " ".join(f"{c:>9}" for c in row_cers), flush=True)

# Per-row ASR details
print("\n--- Per-row ASR transcriptions ---", flush=True)
for i, meta in enumerate(rows_meta):
    print(f"\nRow {i} [{meta['category']}]", flush=True)
    print(f"  TEXT:          {meta['text'][:120]}", flush=True)
    print(f"  REFERENCE_ASR: {reference_asr[i][:120] if i < len(reference_asr) else 'N/A'}", flush=True)
    for label in models_with_cer:
        if i < len(perrow.get(label, [])):
            asr = perrow[label][i].get("asr_transcription", "N/A") or "N/A"
            cer = perrow[label][i].get("asr_cer")
            cer_s = f"{cer:.3f}" if cer is not None else "N/A"
            print(f"  {label:<22} CER={cer_s}  ASR: {str(asr)[:100]}", flush=True)

Path(PHASE4_DIR / "phase4_v2_summary.json").write_text(json.dumps(summary, indent=2))
print("\nDone. Saved phase4_v2_eval_results.json, phase4_v2_perrow.json, phase4_v2_summary.json", flush=True)
