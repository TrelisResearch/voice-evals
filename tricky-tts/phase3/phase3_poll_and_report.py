"""
Phase 3: Poll all 9 eval jobs, collect per-row and aggregate results,
print leaderboard, and save results JSON for report writing.
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

job_ids = json.loads(Path("tricky-tts/phase3/phase3_eval_job_ids.json").read_text())
print(f"Polling {len(job_ids)} jobs...\n", flush=True)

# Poll all jobs until all complete
completed = {}
pending = dict(job_ids)

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
        print(f"  Waiting: {list(still_pending.keys())}", flush=True)
        time.sleep(30)
    pending = still_pending

print(f"\nAll jobs done. {sum(1 for v in completed.values() if v['status']=='completed')}/{len(completed)} completed successfully.\n", flush=True)

# Save raw results
Path("tricky-tts/phase3/phase3_eval_results.json").write_text(json.dumps(completed, indent=2))

# Extract aggregate metrics
print("=" * 70, flush=True)
print(f"{'Model':<22} {'MOS':>6} {'WER':>7} {'CER':>7} {'Samples':>8}", flush=True)
print("-" * 70, flush=True)

summary = []
for label, data in completed.items():
    if data["status"] != "completed":
        print(f"{label:<22}  FAILED", flush=True)
        continue
    result = data["result"]
    mos = result.get("utmos_score") or result.get("mos_score") or result.get("average_utmos_score")
    wer = result.get("average_wer") or result.get("wer")
    cer = result.get("average_cer") or result.get("cer")
    n   = result.get("num_samples") or result.get("total_samples") or result.get("samples_completed")

    # Try nested metrics dict
    if mos is None and "metrics" in result:
        m = result["metrics"]
        mos = m.get("utmos_score") or m.get("mos_score")
        wer = m.get("average_wer") or m.get("wer")
        cer = m.get("average_cer") or m.get("cer")

    mos_s = f"{mos:.3f}" if mos is not None else "N/A"
    wer_s = f"{wer:.3f}" if wer is not None else "N/A"
    cer_s = f"{cer:.3f}" if cer is not None else "N/A"
    n_s   = str(n) if n is not None else "?"

    print(f"{label:<22} {mos_s:>6} {wer_s:>7} {cer_s:>7} {n_s:>8}", flush=True)
    summary.append({"model": label, "mos": mos, "wer": wer, "cer": cer, "samples": n, "status": data["status"]})

print("=" * 70, flush=True)

# Sort by CER then WER
ranked = sorted([s for s in summary if s["cer"] is not None], key=lambda x: x["cer"])
print(f"\nRanked by CER (lower = better pronunciation accuracy):", flush=True)
for i, s in enumerate(ranked, 1):
    print(f"  {i}. {s['model']:<22} CER={s['cer']:.3f}  WER={s['wer']:.3f}  MOS={s['mos']:.3f if s['mos'] else 'N/A'}", flush=True)

ranked_mos = sorted([s for s in summary if s["mos"] is not None], key=lambda x: -x["mos"])
print(f"\nRanked by MOS (higher = better naturalness):", flush=True)
for i, s in enumerate(ranked_mos, 1):
    print(f"  {i}. {s['model']:<22} MOS={s['mos']:.3f}  CER={s['cer']:.3f if s['cer'] else 'N/A'}", flush=True)

Path("tricky-tts/phase3/phase3_summary.json").write_text(json.dumps(summary, indent=2))
print(f"\nResults saved to phase3_eval_results.json and phase3_summary.json", flush=True)
