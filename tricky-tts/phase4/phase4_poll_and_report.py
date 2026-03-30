"""
Poll all 9 phase4 eval jobs, pull per-row data from HF, print leaderboard,
save results JSON. Loads parquet directly to avoid torch dependency.
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

job_ids = json.loads(Path("tricky-tts/phase4/phase4_eval_job_ids.json").read_text())
rows_meta = json.loads(Path("tricky-tts/phase4/rows.json").read_text())

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
Path("tricky-tts/phase4/phase4_eval_results.json").write_text(json.dumps(completed, indent=2))

# Pull per-row data from HF eval datasets
DATASET_MAP = {
    "ElevenLabs":       "ronanarraig/tricky-tts-ph4-eval-elevenlabs",
    "GPT-4o mini TTS":  "ronanarraig/tricky-tts-ph4-eval-gpt-4o-mini-tts",
    "Cartesia Sonic-3": "ronanarraig/tricky-tts-ph4-eval-cartesia-sonic-3",
    "Gemini Flash TTS": "ronanarraig/tricky-tts-ph4-eval-gemini-flash-tts",
    "Gemini Pro TTS":   "ronanarraig/tricky-tts-ph4-eval-gemini-pro-tts",
    "Orpheus":          "ronanarraig/tricky-tts-ph4-eval-orpheus",
    "Kokoro":           "ronanarraig/tricky-tts-ph4-eval-kokoro",
    "Piper (en-gb)":    "ronanarraig/tricky-tts-ph4-eval-piper-en-gb-",
    "Chatterbox":       "ronanarraig/tricky-tts-ph4-eval-chatterbox",
}

print("Pulling per-row data from HF datasets...\n", flush=True)
perrow = {}

for label, repo_id in DATASET_MAP.items():
    try:
        files = list(list_repo_files(repo_id, repo_type="dataset", token=HF_TOKEN))
        parquet_files = [f for f in files if f.endswith(".parquet")]
        if not parquet_files:
            print(f"  {label}: no parquet files found in {repo_id}", flush=True)
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
        mos_vals = [r["utmos_score"] for r in perrow[label] if r.get("utmos_score") is not None]
        print(f"  {label}: {n} rows | cols: {wanted}", flush=True)
        if cers: print(f"    avg CER={sum(cers)/len(cers):.3f}", flush=True)
        if mos_vals: print(f"    avg MOS={sum(mos_vals)/len(mos_vals):.3f}", flush=True)
    except Exception as e:
        print(f"  {label}: ERROR — {e}", flush=True)

Path("tricky-tts/phase4/phase4_perrow.json").write_text(json.dumps(perrow, indent=2))

# Aggregate leaderboard
print("\n" + "="*80, flush=True)
print(f"{'Model':<22} {'MOS':>6} {'WER':>7} {'CER':>7}", flush=True)
print("-"*80, flush=True)

summary = []
for label in DATASET_MAP:
    data = completed.get(label, {})
    r = data.get("result", {})
    # Try nested result.result structure
    inner = r.get("result", {}) or {}
    mos = inner.get("mos") or r.get("utmos_score") or r.get("mos_score") or r.get("average_utmos_score")
    wer = inner.get("wer") or r.get("average_wer") or r.get("wer")
    cer = inner.get("cer") or r.get("average_cer") or r.get("cer")
    # Override with per-row averages if available
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
models_with_cer = [l for l in DATASET_MAP if l in perrow and any(r.get("asr_cer") is not None for r in perrow[l])]
if models_with_cer:
    print(f"\n{'Row':<3} {'Category':<25} " + " ".join(f"{m[:9]:>9}" for m in models_with_cer), flush=True)
    print("-"*140, flush=True)
    for i, meta in enumerate(rows_meta):
        cat = meta["category"]
        row_cers = []
        for label in models_with_cer:
            cer = perrow[label][i].get("asr_cer") if i < len(perrow[label]) else None
            row_cers.append(f"{cer:.3f}" if cer is not None else "  N/A")
        print(f"{i:<3} {cat:<25} " + " ".join(f"{c:>9}" for c in row_cers), flush=True)

# Per-row ASR transcriptions
print("\n--- Per-row ASR transcriptions ---", flush=True)
for i, meta in enumerate(rows_meta):
    print(f"\nRow {i} [{meta['category']}]", flush=True)
    print(f"  TEXT:  {meta['text'][:120]}", flush=True)
    for label in models_with_cer:
        if i < len(perrow.get(label, [])):
            asr = perrow[label][i].get("asr_transcription", "N/A") or "N/A"
            cer = perrow[label][i].get("asr_cer")
            cer_s = f"{cer:.3f}" if cer is not None else "N/A"
            print(f"  {label:<22} CER={cer_s}  ASR: {str(asr)[:100]}", flush=True)

Path("tricky-tts/phase4/phase4_summary.json").write_text(json.dumps(summary, indent=2))
print("\nDone. Saved phase4_eval_results.json, phase4_perrow.json, phase4_summary.json", flush=True)
