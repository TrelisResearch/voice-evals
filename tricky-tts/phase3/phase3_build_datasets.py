"""
Phase 3 Step 1b: Poll jobs, then build and push final reference datasets.

Polls Orpheus + Kokoro eval jobs, pulls asr_transcription from results,
joins back to original proto data, and pushes:
  - ronanarraig/tricky-tts-public-orpheus  (reference_asr = Orpheus Whisper transcript)
  - ronanarraig/tricky-tts-public-kokoro   (reference_asr = Kokoro Whisper transcript)

Schema: text, spoken_form, category, reference_asr
(cer_reliable dropped — not useful)
"""

import os, json, time, requests
from pathlib import Path
from datasets import load_dataset, Dataset

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

job_ids_path = Path("tricky-tts/phase3/phase3_job_ids.json")
job_ids = json.loads(job_ids_path.read_text())
print(f"Loaded jobs: {list(job_ids.keys())}", flush=True)

# --- Poll until both jobs complete ---
def poll_job(job_id):
    while True:
        resp = requests.get(f"{TRELIS_API}/tts-evaluation/jobs/{job_id}", headers=HEADERS)
        resp.raise_for_status()
        data = resp.json()
        status = data.get("status", "unknown")
        print(f"  Job {job_id}: {status}", flush=True)
        if status in ("completed", "failed", "error"):
            return data
        time.sleep(30)

print("\nPolling jobs...", flush=True)
results = {}
for label, info in job_ids.items():
    print(f"\n{label} (job {info['job_id']}):", flush=True)
    result = poll_job(info["job_id"])
    results[label] = {"info": info, "result": result}
    status = result.get("status")
    print(f"  Final status: {status}", flush=True)
    if status not in ("completed",):
        print(f"  WARNING: job did not complete cleanly. Full response:", flush=True)
        print(json.dumps(result, indent=2)[:500], flush=True)

# Save raw results
Path("tricky-tts/phase3/phase3_job_results.json").write_text(json.dumps(results, indent=2))
print("\nRaw results saved.", flush=True)

# --- Load original proto data ---
print("\nLoading original proto-v4 data...", flush=True)
proto = load_dataset("ronanarraig/tricky-tts-proto-v4", split="train")
# Index by spoken_form for joining
proto_by_spoken = {row["spoken_form"]: row for row in proto}
original_texts = [row["text"] for row in proto]
spoken_forms = [row["spoken_form"] for row in proto]
categories = [row["category"] for row in proto]

# --- Build and push each dataset ---
for label, data in results.items():
    info = data["info"]
    result = data["result"]
    output_hf = f"ronanarraig/{info['output_name'].replace('eval-ref', 'public')}"
    # output_name is e.g. tricky-tts-eval-ref-orpheus → tricky-tts-public-orpheus
    suffix = "orpheus" if "orpheus" in info["output_name"] else "kokoro"
    output_hf = f"ronanarraig/tricky-tts-public-{suffix}"

    print(f"\nBuilding dataset for {label} → {output_hf}", flush=True)

    # Load the eval output dataset
    eval_dataset_id = f"ronanarraig/{info['output_name']}"
    print(f"  Loading eval output: {eval_dataset_id}", flush=True)
    eval_ds = load_dataset(eval_dataset_id, split="train", token=HF_TOKEN)
    print(f"  Eval output columns: {eval_ds.column_names}", flush=True)
    print(f"  Eval output rows: {len(eval_ds)}", flush=True)

    # Select only text columns to avoid audio decoding (requires torch)
    text_cols = [c for c in ["text_prompt", "text", "asr_transcription"] if c in eval_ds.column_names]
    eval_ds = eval_ds.select_columns(text_cols)

    # text_prompt or text column holds the spoken_form input
    text_col = "text_prompt" if "text_prompt" in eval_ds.column_names else "text"
    asr_by_spoken = {}
    for row in eval_ds:
        asr_by_spoken[row[text_col]] = row.get("asr_transcription", "")

    # Build final rows
    final_rows = []
    for orig_text, spoken, category in zip(original_texts, spoken_forms, categories):
        ref_asr = asr_by_spoken.get(spoken, "")
        if not ref_asr:
            print(f"  WARNING: no asr_transcription found for spoken_form: {spoken[:60]}...", flush=True)
        final_rows.append({
            "text": orig_text,
            "spoken_form": spoken,
            "category": category,
            "reference_asr": ref_asr,
        })

    final_ds = Dataset.from_list(final_rows)
    print(f"  Final dataset: {len(final_ds)} rows, columns: {final_ds.column_names}", flush=True)

    # Spot-check ref_self_cer
    try:
        import jiwer
        print("  ref_self_cer spot-check:", flush=True)
        for row in final_ds:
            if row["reference_asr"]:
                cer = jiwer.cer(row["spoken_form"], row["reference_asr"])
                flag = " *** HIGH ***" if cer > 0.5 else ""
                print(f"    [{row['category']}] cer={cer:.3f}{flag} | ref: {row['reference_asr'][:60]}", flush=True)
    except ImportError:
        print("  (jiwer not available — skipping CER spot-check)", flush=True)

    print(f"  Pushing to {output_hf}...", flush=True)
    final_ds.push_to_hub(output_hf, split="train", token=HF_TOKEN, private=True)
    print(f"  Done → {output_hf}", flush=True)

print("\nAll done. Datasets pushed:", flush=True)
for suffix in ["orpheus", "kokoro"]:
    print(f"  ronanarraig/tricky-tts-public-{suffix}", flush=True)
print("\nListen and pick one → rename to ronanarraig/tricky-tts-prototype", flush=True)
