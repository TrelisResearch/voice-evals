"""
Regenerate spoken_form for the 10 prototype rows using updated rules,
then re-run the reference pipeline prototype.
"""

import json, os, re, time, requests
from pathlib import Path
from collections import defaultdict
import editdistance

env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from openai import OpenAI
from datasets import Dataset
from huggingface_hub import HfApi
import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
API_KEY = os.environ["TRELIS_STUDIO_API_KEY"]
HF_TOKEN = os.environ["HF_TOKEN"]
TRELIS_API = "https://studio.trelis.com/api/v1"
HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

MODEL = "anthropic/claude-sonnet-4-5"
FALLBACK = "google/gemini-2.5-flash"

SYSTEM_PROMPT = Path("tricky-tts/spoken_form_rules.md").read_text()

UNICODE_SUBS = [
    ("×", " times "), ("±", " plus or minus "), ("≥", " greater than or equal to "),
    ("≤", " less than or equal to "), ("→", " to "), ("°C", " degrees Celsius"),
    ("°F", " degrees Fahrenheit"), ("μg", " micrograms"), ("μL", " microlitres"),
    ("μM", " micromolar"), ("μ", " micro"), ("°", " degrees"),
    ("⁻⁴", " to the minus four"), ("⁻³", " to the minus three"),
    ("⁻²", " to the minus two"), ("⁻¹", " to the minus one"), ("⁻", " minus "),
    ("⁰"," zero"),("¹"," one"),("²"," squared"),("³"," cubed"),
    ("⁴"," to the fourth"),("⁵"," to the fifth"),("⁶"," to the sixth"),
    ("⁷"," to the seventh"),("⁸"," to the eighth"),("⁹"," to the ninth"),
    ("½"," one half"),("⅓"," one third"),("⅔"," two thirds"),
    ("¼"," one quarter"),("¾"," three quarters"),("⅘"," four fifths"),
    ("£"," pounds "),("€"," euros "),("—",", "),("–"," to "),("…","..."),
    ("\u2019","'"),("\u2018","'"),("\u201c",'"'),("\u201d",'"'),
]

def sanitise(text: str) -> str:
    for src, dst in UNICODE_SUBS:
        text = text.replace(src, dst)
    text = re.sub(r" {2,}", " ", text).strip()
    non_ascii = [c for c in text if ord(c) > 127]
    if non_ascii:
        print(f"    ⚠ Non-ASCII remains: {set(non_ascii)}", flush=True)
    return text

def gen_spoken(text: str, category: str) -> str:
    for model in [MODEL, FALLBACK]:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Category: {category}\nText: {text}"}
            ],
            max_tokens=1000, temperature=0.1,
        )
        c = resp.choices[0].message.content
        if c:
            return sanitise(c.strip())
    raise ValueError(f"All models failed: {text[:50]}")

def poll_job(job_type: str, job_id: str, interval: int = 15):
    while True:
        time.sleep(interval)
        j = requests.get(f"{TRELIS_API}/{job_type}/jobs/{job_id}", headers=HEADERS).json()
        r = j.get("result") or {}
        print(f"  {j['status']}  MOS={r.get('mos','')}  CER={r.get('cer','')}", flush=True)
        if j["status"] in ("completed", "failed", "stopped"):
            return j

# ── Select same 10 prototype rows ────────────────────────────────────────────
data = json.loads(Path("tricky-tts/phase2_final.json").read_text())
perrow = json.loads(Path("tricky-tts/phase2_perrow_results.json").read_text())
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

# ── Regenerate spoken_form with new rules ─────────────────────────────────────
print("Regenerating spoken_form with updated rules...\n", flush=True)
for r in sample:
    print(f"[{r['category']}] {r['text'][:70]}", flush=True)
    old = r.get("spoken_form", "")
    new = gen_spoken(r["text"], r["category"])
    r["spoken_form"] = new
    print(f"  OLD: {old[:80]}", flush=True)
    print(f"  NEW: {new[:80]}", flush=True)
    print(flush=True)

# ── Push updated spoken_form dataset ─────────────────────────────────────────
api = HfApi(token=HF_TOKEN)
ref_rows = [{"text": r["spoken_form"], "orig_text": r["text"], "category": r["category"]} for r in sample]
ref_repo = "ronanarraig/tricky-tts-proto-spoken-form"
api.create_repo(repo_id=ref_repo, repo_type="dataset", private=True, exist_ok=True)
Dataset.from_list(ref_rows).push_to_hub(ref_repo, token=HF_TOKEN, private=True, split="train")
print(f"Pushed updated spoken_form dataset to {ref_repo}", flush=True)

# ── Run Orpheus reference TTS + AssemblyAI ASR ───────────────────────────────
print("\nSubmitting Orpheus reference job...", flush=True)
resp = requests.post(f"{TRELIS_API}/tts-evaluation/jobs", headers=HEADERS, json={
    "model_id": "unsloth/orpheus-3b-0.1-ft",
    "dataset_id": ref_repo, "split": "train", "num_samples": 10,
    "asr_model_id": "openai/whisper-large-v3", "language": "auto",
    "tts_model_type": "orpheus", "push_results": True,
    "output_org": "ronanarraig", "output_name": "tricky-tts-proto-ref-orpheus-v3",
    "private": True,
})
ref_job_id = resp.json().get("id") or resp.json().get("job_id")
print(f"Reference job: {ref_job_id}", flush=True)
j = poll_job("tts-evaluation", ref_job_id)
if j["status"] != "completed":
    print(f"FAILED: {j.get('error')}"); exit(1)

# ── Extract reference ASR transcripts ────────────────────────────────────────
local = hf_hub_download("ronanarraig/tricky-tts-proto-ref-orpheus-v3",
    "data/train-00000-of-00001.parquet", repo_type="dataset", token=HF_TOKEN)
t = pq.read_table(local, columns=["text_prompt", "asr_transcription", "asr_cer"])
ref_d = t.to_pydict()
spoken_to_ref_asr = dict(zip(ref_d["text_prompt"], ref_d["asr_transcription"]))
spoken_to_ref_cer = dict(zip(ref_d["text_prompt"], ref_d["asr_cer"]))

# ── Build prototype dataset with reference_asr_transcript ────────────────────
proto_rows = []
for r in sample:
    proto_rows.append({
        "text": r["text"], "category": r["category"],
        "spoken_form": r["spoken_form"], "cer_reliable": r["cer_reliable"],
        "reference_asr_transcript": spoken_to_ref_asr.get(r["spoken_form"], ""),
    })

proto_repo = "ronanarraig/tricky-tts-proto-with-reference"
api.create_repo(repo_id=proto_repo, repo_type="dataset", private=True, exist_ok=True)
Dataset.from_list(proto_rows).push_to_hub(proto_repo, token=HF_TOKEN, private=True, split="train")
print(f"Pushed proto dataset (with reference_asr_transcript) to {proto_repo}", flush=True)
Path("tricky-tts/phase2_proto_reference_v2.json").write_text(
    json.dumps(proto_rows, indent=2, ensure_ascii=False))

# ── Run ElevenLabs test eval with reference_column ───────────────────────────
print("\nSubmitting ElevenLabs test eval...", flush=True)
resp = requests.post(f"{TRELIS_API}/tts-evaluation/jobs", headers=HEADERS, json={
    "model_id": "elevenlabs/eleven-multilingual-v2",
    "dataset_id": proto_repo, "split": "train", "num_samples": 10,
    "asr_model_id": "openai/whisper-large-v3", "language": "auto",
    "tts_model_type": "auto", "push_results": True,
    "output_org": "ronanarraig", "output_name": "tricky-tts-proto-test-elevenlabs-v3",
    "reference_column": "reference_asr_transcript", "private": True,
})
test_job_id = resp.json().get("id") or resp.json().get("job_id")
print(f"Test job: {test_job_id}", flush=True)
j = poll_job("tts-evaluation", test_job_id)
if j["status"] != "completed":
    print(f"FAILED: {j.get('error')}"); exit(1)

# ── Download and inspect ──────────────────────────────────────────────────────
local2 = hf_hub_download("ronanarraig/tricky-tts-proto-test-elevenlabs-v3",
    "data/train-00000-of-00001.parquet", repo_type="dataset", token=HF_TOKEN)
t2 = pq.read_table(local2, columns=["text_prompt","asr_transcription","asr_cer"])
test_d = t2.to_pydict()
test_by_text = dict(zip(test_d["text_prompt"],
    zip(test_d["asr_transcription"], test_d["asr_cer"])))

REF_SELF_CER_THRESHOLD = 0.3

def compute_cer(ref: str, hyp: str) -> float:
    """Character error rate: edit_distance(ref, hyp) / len(ref)."""
    ref, hyp = ref.strip().lower(), hyp.strip().lower()
    if not ref:
        return 0.0
    return editdistance.eval(ref, hyp) / len(ref)

old_cer = {r["text"]: r.get("median_cer") for r in perrow}

print("\n" + "="*100)
print("INSPECTION: old CER-vs-text  |  ref_pipeline CER (computed)  |  spoken_form  |  ref_asr  |  test_asr")
print("="*100)

rows_sorted = sorted(proto_rows, key=lambda r: -compute_cer(
    r["reference_asr_transcript"], test_by_text.get(r["text"], ("",))[0] or ""))
for r in rows_sorted:
    test_asr, _ = test_by_text.get(r["text"], ("N/A", None))
    ref_asr = r["reference_asr_transcript"]
    o_cer = old_cer.get(r["text"])
    ref_self_cer = spoken_to_ref_cer.get(r["spoken_form"])

    ref_ok = ref_self_cer is not None and ref_self_cer <= REF_SELF_CER_THRESHOLD
    ref_pipeline_cer = compute_cer(ref_asr, test_asr) if (ref_ok and test_asr != "N/A") else None
    delta = (ref_pipeline_cer - o_cer) if (ref_pipeline_cer is not None and o_cer is not None) else None

    print(f"\n[{r['category']}]")
    print(f"  text:     {r['text'][:90]}")
    print(f"  spoken:   {r['spoken_form'][:90]}")
    print(f"  ref_asr:  {ref_asr[:90]}")
    print(f"  test_asr: {test_asr[:90]}")
    if ref_self_cer is not None and ref_self_cer > REF_SELF_CER_THRESHOLD:
        print(f"  CER: ⚠ REFERENCE FAILED (ref_self_cer={ref_self_cer:.3f} > {REF_SELF_CER_THRESHOLD}) — skipping")
    elif ref_pipeline_cer is not None:
        cer_str = f"old={o_cer:.3f}  ref_pipeline={ref_pipeline_cer:.3f}  delta={delta:+.3f}" if delta is not None else f"ref_pipeline={ref_pipeline_cer:.3f}"
        print(f"  CER: {cer_str}  ref_self_cer={ref_self_cer:.3f}")
    else:
        print(f"  CER: N/A  ref_self_cer={ref_self_cer}")

print("\nDone.", flush=True)
