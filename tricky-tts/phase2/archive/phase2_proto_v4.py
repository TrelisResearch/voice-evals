"""
Prototype v4: spoken_form as direct CER reference + Gemini Pro reference audio for inspection.

Steps:
1. Regenerate spoken_form for 10 prototype rows with latest rules
2. Push spoken_form-as-text dataset → run Gemini Pro TTS → reference audio for inspection
3. Push test dataset (text + spoken_form columns)
4. Run 3 test model evals (ElevenLabs, Orpheus, Gemini Flash) with reference_column="spoken_form"
5. Print side-by-side inspection
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
from huggingface_hub import HfApi, hf_hub_download
import pyarrow.parquet as pq

llm = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
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
        resp = llm.chat.completions.create(
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
    raise ValueError(f"All models failed for: {text[:50]}")

def poll_job(job_type: str, job_id: str, interval: int = 15):
    while True:
        time.sleep(interval)
        j = requests.get(f"{TRELIS_API}/{job_type}/jobs/{job_id}", headers=HEADERS).json()
        r = j.get("result") or {}
        print(f"  {j['status']}  MOS={r.get('mos','')}  CER={r.get('cer','')}", flush=True)
        if j["status"] in ("completed", "failed", "stopped"):
            return j

def compute_cer(ref: str, hyp: str) -> float:
    ref, hyp = ref.strip().lower(), hyp.strip().lower()
    if not ref:
        return 0.0
    return editdistance.eval(ref, hyp) / len(ref)

# ── Select 10 prototype rows ──────────────────────────────────────────────────
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

# ── Regenerate spoken_form ────────────────────────────────────────────────────
print("Regenerating spoken_form...\n", flush=True)
for r in sample:
    print(f"[{r['category']}] {r['text'][:70]}", flush=True)
    old = r.get("spoken_form", "")
    new = gen_spoken(r["text"], r["category"])
    r["spoken_form"] = new
    print(f"  OLD: {old[:80]}", flush=True)
    print(f"  NEW: {new[:80]}", flush=True)
    print(flush=True)

api = HfApi(token=HF_TOKEN)

# ── Step 1: Gemini Pro TTS reference audio (for inspection) ──────────────────
# Push spoken_form-as-text → run Gemini Pro TTS eval → audio dataset
print("Pushing spoken_form dataset for Gemini Pro TTS reference...", flush=True)
ref_rows = [{"text": r["spoken_form"], "orig_text": r["text"], "category": r["category"]} for r in sample]
ref_repo = "ronanarraig/tricky-tts-proto-spoken-form"
api.create_repo(repo_id=ref_repo, repo_type="dataset", private=True, exist_ok=True)
Dataset.from_list(ref_rows).push_to_hub(ref_repo, token=HF_TOKEN, private=True, split="train")
print(f"Pushed to {ref_repo}", flush=True)

print("\nSubmitting Gemini Pro TTS reference job...", flush=True)
resp = requests.post(f"{TRELIS_API}/tts-evaluation/jobs", headers=HEADERS, json={
    "model_id": "google/gemini-2.5-pro-tts",
    "dataset_id": ref_repo, "split": "train", "num_samples": 10,
    "asr_model_id": "openai/whisper-large-v3", "language": "auto",
    "tts_model_type": "auto", "push_results": True,
    "output_org": "ronanarraig", "output_name": "tricky-tts-proto-ref-gemini-pro",
    "private": True,
})
ref_job_id = resp.json().get("id") or resp.json().get("job_id")
print(f"Reference job: {ref_job_id}", flush=True)
j = poll_job("tts-evaluation", ref_job_id)
if j["status"] != "completed":
    print(f"FAILED: {j.get('error')}"); exit(1)
print(f"Reference audio at: ronanarraig/tricky-tts-proto-ref-gemini-pro", flush=True)

# ── Step 2: Push test dataset (text + spoken_form as CER reference) ───────────
print("\nPushing test dataset with spoken_form reference column...", flush=True)
test_rows = [{
    "text": r["text"],
    "spoken_form": r["spoken_form"],
    "category": r["category"],
    "cer_reliable": r["cer_reliable"],
} for r in sample]
test_repo = "ronanarraig/tricky-tts-proto-v4"
api.create_repo(repo_id=test_repo, repo_type="dataset", private=True, exist_ok=True)
Dataset.from_list(test_rows).push_to_hub(test_repo, token=HF_TOKEN, private=True, split="train")
print(f"Pushed to {test_repo}", flush=True)

# ── Step 3: Run 3 test model evals with reference_column="spoken_form" ────────
TEST_MODELS = [
    ("elevenlabs/eleven-multilingual-v2", "auto",    "elevenlabs"),
    ("unsloth/orpheus-3b-0.1-ft",         "orpheus", "orpheus"),
    ("google/gemini-2.5-flash-tts",        "auto",    "gemini-flash"),
]

jobs = {}
for model_id, model_type, short_name in TEST_MODELS:
    print(f"\nSubmitting {short_name} eval...", flush=True)
    resp = requests.post(f"{TRELIS_API}/tts-evaluation/jobs", headers=HEADERS, json={
        "model_id": model_id,
        "dataset_id": test_repo, "split": "train", "num_samples": 10,
        "asr_model_id": "openai/whisper-large-v3", "language": "auto",
        "tts_model_type": model_type, "push_results": True,
        "output_org": "ronanarraig",
        "output_name": f"tricky-tts-proto-test-{short_name}-v4",
        "reference_column": "spoken_form", "private": True,
    })
    job_id = resp.json().get("id") or resp.json().get("job_id")
    print(f"  Job ID: {job_id}", flush=True)
    jobs[short_name] = job_id

# Poll all jobs
results = {}
pending = dict(jobs)
while pending:
    time.sleep(15)
    done = []
    for name, job_id in pending.items():
        j = requests.get(f"{TRELIS_API}/tts-evaluation/jobs/{job_id}", headers=HEADERS).json()
        r = j.get("result") or {}
        print(f"  [{name}] {j['status']}  MOS={r.get('mos','')}  CER={r.get('cer','')}", flush=True)
        if j["status"] in ("completed", "failed", "stopped"):
            results[name] = j
            done.append(name)
    for name in done:
        del pending[name]

# ── Step 4: Download results and compare ─────────────────────────────────────
model_results = {}
for short_name, _ in [(n, None) for n in jobs]:
    if results.get(short_name, {}).get("status") != "completed":
        print(f"  {short_name} FAILED: {results.get(short_name, {}).get('error')}", flush=True)
        continue
    local = hf_hub_download(
        f"ronanarraig/tricky-tts-proto-test-{short_name}-v4",
        "data/train-00000-of-00001.parquet", repo_type="dataset", token=HF_TOKEN)
    t = pq.read_table(local, columns=["text_prompt", "asr_transcription", "asr_cer"])
    d = t.to_pydict()
    model_results[short_name] = dict(zip(d["text_prompt"],
        zip(d["asr_transcription"], d["asr_cer"])))

old_cer = {r["text"]: r.get("median_cer") for r in perrow}

print("\n" + "="*110)
print("PROTOTYPE v4 — CER reference: spoken_form text  |  ASR: whisper-large-v3")
print("="*110)

for r in sorted(test_rows, key=lambda r: r["category"]):
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
        print(f"  [{name}] CER={ref_cer:.3f}{delta}  asr: {test_asr[:70]}")

print("\nDone.", flush=True)
print(f"\nReference audio (Gemini Pro, for inspection): ronanarraig/tricky-tts-proto-ref-gemini-pro")
print(f"Test eval datasets: ronanarraig/tricky-tts-proto-test-{{elevenlabs,orpheus,gemini-flash}}-v4")
