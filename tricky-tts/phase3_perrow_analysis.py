"""
Pull per-row results from all pushed HF eval datasets and produce a full analysis.
Loads parquet directly to avoid torch/torchcodec dependency.
"""

import os, json
from pathlib import Path
from huggingface_hub import hf_hub_download, list_repo_files
import pyarrow.parquet as pq

env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

HF_TOKEN = os.environ["HF_TOKEN"]

DATASETS = {
    "ElevenLabs":       "ronanarraig/tricky-tts-pub-eval-elevenlabs",
    "GPT-4o mini TTS":  "ronanarraig/tricky-tts-pub-eval-gpt4o-mini",
    "Cartesia Sonic-3": "ronanarraig/tricky-tts-pub-eval-cartesia",
    "Gemini Flash TTS": "ronanarraig/tricky-tts-pub-eval-gemini-flash",
    "Gemini Pro TTS":   "ronanarraig/tricky-tts-pub-eval-gemini-pro",
    "Orpheus":          "ronanarraig/tricky-tts-pub-eval-orpheus",
    "Kokoro":           "ronanarraig/tricky-tts-pub-eval-kokoro",
    "Piper (en-gb)":    "ronanarraig/tricky-tts-pub-eval-piper",
    "Chatterbox":       "ronanarraig/tricky-tts-pub-eval-chatterbox",
}

# Aggregate metrics from results file
all_results = json.loads(Path("tricky-tts/phase3_eval_results.json").read_text())
# Update Chatterbox with completed result
all_results["Chatterbox"] = {
    "status": "completed",
    "result": {
        "result": {"mos": 4.3825, "wer": None, "cer": None, "num_successful": 10}
    }
}
Path("tricky-tts/phase3_eval_results.json").write_text(json.dumps(all_results, indent=2))

print("Pulling per-row data from HF datasets...\n", flush=True)
perrow = {}

for label, repo_id in DATASETS.items():
    try:
        files = list(list_repo_files(repo_id, repo_type="dataset", token=HF_TOKEN))
        parquet_files = [f for f in files if f.endswith(".parquet")]
        if not parquet_files:
            print(f"  {label}: no parquet files", flush=True)
            continue
        local = hf_hub_download(repo_id=repo_id, filename=parquet_files[0], repo_type="dataset", token=HF_TOKEN)
        schema_names = pq.read_schema(local).names
        wanted = [c for c in ["text_prompt", "asr_transcription", "asr_wer", "asr_cer", "duration_s", "utmos_score"] if c in schema_names]
        table = pq.read_table(local, columns=wanted)
        rows = table.to_pydict()
        n = len(rows.get("text_prompt", []))
        perrow[label] = [
            {k: rows[k][i] if k in rows else None for k in ["text_prompt", "asr_transcription", "asr_wer", "asr_cer", "duration_s", "utmos_score"]}
            for i in range(n)
        ]
        cers = [r["asr_cer"] for r in perrow[label] if r["asr_cer"] is not None]
        mos_vals = [r["utmos_score"] for r in perrow[label] if r.get("utmos_score") is not None]
        print(f"  {label}: {n} rows | cols: {wanted}", flush=True)
        if cers: print(f"    avg CER={sum(cers)/len(cers):.3f}", flush=True)
        if mos_vals: print(f"    avg MOS={sum(mos_vals)/len(mos_vals):.3f}", flush=True)
    except Exception as e:
        print(f"  {label}: ERROR — {e}", flush=True)

Path("tricky-tts/phase3_perrow.json").write_text(json.dumps(perrow, indent=2))

# Load original proto for category mapping
proto_rows = json.loads(Path("tricky-tts/phase3_proto_updated.json").read_text())
spoken_to_cat = {r["spoken_form"]: r["category"] for r in proto_rows}
spoken_to_text = {r["spoken_form"]: r["text"] for r in proto_rows}

# Print aggregate leaderboard
print("\n" + "="*80, flush=True)
print(f"{'Model':<22} {'MOS':>6} {'WER':>7} {'CER':>7}", flush=True)
print("-"*80, flush=True)

summary = []
for label, data in all_results.items():
    r = data.get("result", {}).get("result", {}) or {}
    mos = r.get("mos")
    wer = r.get("wer")
    cer = r.get("cer")
    # Override with per-row averages if available
    if label in perrow:
        cers = [x["asr_cer"] for x in perrow[label] if x.get("asr_cer") is not None]
        wers = [x["asr_wer"] for x in perrow[label] if x.get("asr_wer") is not None]
        moss = [x["utmos_score"] for x in perrow[label] if x.get("utmos_score") is not None]
        if cers: cer = sum(cers)/len(cers)
        if wers: wer = sum(wers)/len(wers)
        if moss: mos = sum(moss)/len(moss)
    summary.append({"model": label, "mos": mos, "wer": wer, "cer": cer})
    print(f"{label:<22} {mos:.3f if mos else 'N/A':>6} {wer:.3f if wer else ' N/A':>7} {cer:.3f if cer else ' N/A':>7}", flush=True)

print("="*80, flush=True)

# Per-row CER table
models_with_cer = [l for l in DATASETS if l in perrow and any(r.get("asr_cer") is not None for r in perrow[l])]
if models_with_cer:
    ref_rows = perrow[models_with_cer[0]]
    print(f"\n{'Category':<18} {'Text (abbrev)':<35} " + " ".join(f"{m[:7]:>7}" for m in models_with_cer), flush=True)
    print("-"*120, flush=True)
    for i, ref_row in enumerate(ref_rows):
        cat = spoken_to_cat.get(ref_row.get("text_prompt"), "?")
        text = spoken_to_text.get(ref_row.get("text_prompt"), "?")[:33]
        row_cers = []
        for label in models_with_cer:
            cer = perrow[label][i].get("asr_cer") if i < len(perrow[label]) else None
            row_cers.append(f"{cer:.3f}" if cer is not None else "  N/A")
        print(f"{cat:<18} {text:<35} " + " ".join(f"{c:>7}" for c in row_cers), flush=True)

Path("tricky-tts/phase3_summary.json").write_text(json.dumps(summary, indent=2))
print("\nDone. Saved to phase3_summary.json and phase3_perrow.json", flush=True)
