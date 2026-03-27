"""
Phase 3: Update proto dataset with shortened texts, regenerate spoken forms,
push updated proto + spoken_form_input datasets, re-run Kokoro eval job.

Changes:
- Row 0 (edge_cases): remove page range to shorten
- Row 1 (edge_cases): simplify preamble + remove page range
- Row 2 (domain_specific): remove second organism
- Row 3 (domain_specific): remove second organism, simplify
- Row 4 (ai_tech): spoken form rules updated (ZeRO→zero, 80GB, Qwen→Chwen)
- Rows 5-9: unchanged text, but spoken forms regenerated with updated rules
"""

import json, os, re, time, requests
from pathlib import Path
from datasets import Dataset, load_dataset
from openai import OpenAI

env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

HF_TOKEN = os.environ["HF_TOKEN"]
API_KEY = os.environ["TRELIS_STUDIO_API_KEY"]
TRELIS_API = "https://studio.trelis.com/api/v1"
TRELIS_HEADERS = {"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"}

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)
MODEL = "anthropic/claude-sonnet-4-5"
FALLBACK_MODEL = "google/gemini-2.5-flash"

RULES_PATH = Path("tricky-tts/spoken_form_rules.md")
SYSTEM_PROMPT = RULES_PATH.read_text()

UNICODE_SUBS = [
    ("×", " times "), ("±", " plus or minus "), ("≥", " greater than or equal to "),
    ("≤", " less than or equal to "), ("→", " to "), ("°C", " degrees Celsius"),
    ("°F", " degrees Fahrenheit"), ("μg", " micrograms"), ("μL", " microlitres"),
    ("μM", " micromolar"), ("μ", " micro"), ("°", " degrees"),
    ("⁻⁴", " to the minus four"), ("⁻³", " to the minus three"),
    ("⁻²", " to the minus two"), ("⁻¹", " to the minus one"), ("⁻", " minus "),
    ("⁹", " to the ninth"), ("—", ", "), ("–", " to "), ("…", "..."),
    ("\u2019", "'"), ("\u2018", "'"), ("\u201c", '"'), ("\u201d", '"'),
]

def sanitise_unicode(text: str) -> str:
    for src, dst in UNICODE_SUBS:
        text = text.replace(src, dst)
    text = re.sub(r" {2,}", " ", text).strip()
    non_ascii = [c for c in text if ord(c) > 127]
    if non_ascii:
        print(f"  ⚠ Non-ASCII remains: {set(non_ascii)}", flush=True)
    return text

def generate_spoken_form(text: str, category: str) -> str:
    for model in [MODEL, FALLBACK_MODEL]:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": f"Category: {category}\nText: {text}"}
            ],
            max_tokens=1000,
            temperature=0.1,
        )
        content = resp.choices[0].message.content
        if content:
            return sanitise_unicode(content.strip())
        print(f"  ⚠ {model} returned None, trying fallback...", flush=True)
        time.sleep(1)
    raise ValueError(f"All models returned None for: {text[:60]}")

# Updated texts (shortened where needed)
UPDATED_ROWS = [
    {
        "category": "edge_cases",
        "text": "The manuscript cited Vol. 47, No. 3 of the J. Phys. Chem., noting samples required ≥99.9% purity and storage at −80°C±2°C in 50μg/mL aliquots.",
    },
    {
        "category": "edge_cases",
        "text": "Per IEEE Trans. Vol. 15, the circuit required 3.3V±5%, 150mA max., capacitance ≤10μF and operating temps −40°C to +85°C per MIL-STD-883.",
    },
    {
        "category": "domain_specific",
        "text": "Administer 0.25mL of 1×10⁻⁴ mol/L (2R,3S,4R,5R)-2,3,4,5,6-pentahydroxyhexanal (CAS 50-99-7) to Saccharomyces cerevisiae S288C at 30.0°C ± 0.2°C, pH 5.5 ± 0.05 per ASTM E2149-13a.",
    },
    {
        "category": "domain_specific",
        "text": "Mycobacterium tuberculosis H37Rv expressing rpoB S531L demonstrates rifampicin MIC >128μg/mL per CLSI M24-A2:2011 at pH 6.8 ± 0.1 and 37°C ± 0.5°C.",
    },
    # Rows 4-9: unchanged text
    {
        "category": "ai_tech",
        "text": "We fine-tuned deepseek-ai/DeepSeek-R1-Distill-Qwen-32B on 4×A100 80GB SXM with bf16, ZeRO-3, gradient_checkpointing=True, and lr=2e-5.",
    },
    {
        "category": "ai_tech",
        "text": "We benchmarked 01-ai/Yi-1.5-34B-Chat-16K against meta-llama/Llama-3.1-70B-Instruct using MMLU, GSM8K@4-shot, and HumanEval benchmarks.",
    },
    {
        "category": "number_format",
        "text": "The spacecraft traveled 2.5×10⁹ miles at temperatures from -40°C to +120°C, covering the distance in approximately 3:2 the predicted time.",
    },
    {
        "category": "number_format",
        "text": "For participants ages 18–65, blood pressure should range from 90/60 to 120/80 mmHg, with temperatures between 97.8°F and 99.1°F.",
    },
    {
        "category": "phonetic",
        "text": "Eithne and Caoilfhinn debated the mise en scène while eating Worcestershire sauce with Dr. Nguyen.",
    },
    {
        "category": "paralinguistics",
        "text": "He started snoring—zzz, zzz—and I just thought... you know what, forget it. Not worth it.",
    },
]

# --- Generate spoken forms ---
output_path = Path("tricky-tts/phase3_proto_updated.json")

# Resume from partial output if it exists
if output_path.exists():
    results = json.loads(output_path.read_text())
    done_texts = {r["text"] for r in results}
    print(f"Resuming: {len(results)}/{len(UPDATED_ROWS)} already done", flush=True)
else:
    results = []
    done_texts = set()

for i, row in enumerate(UPDATED_ROWS):
    if row["text"] in done_texts:
        print(f"[{i}] skip (already done)", flush=True)
        continue
    print(f"[{i}] [{row['category']}] {row['text'][:70]}...", flush=True)
    spoken = generate_spoken_form(row["text"], row["category"])
    print(f"  → {spoken[:100]}", flush=True)
    results.append({"text": row["text"], "category": row["category"], "spoken_form": spoken})
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    time.sleep(0.3)

print(f"\nAll spoken forms generated.", flush=True)

# --- Push updated proto dataset ---
proto_ds = Dataset.from_list([
    {"text": r["text"], "spoken_form": r["spoken_form"], "category": r["category"]}
    for r in results
])
print(f"\nPushing ronanarraig/tricky-tts-proto-v4 ({len(proto_ds)} rows)...", flush=True)
proto_ds.push_to_hub("ronanarraig/tricky-tts-proto-v4", split="train", token=HF_TOKEN, private=True)
print("  Done.", flush=True)

# --- Push spoken_form_input dataset (spoken_form as 'text' for TTS eval) ---
input_ds = Dataset.from_list([
    {"text": r["spoken_form"], "original_text": r["text"], "category": r["category"]}
    for r in results
])
print(f"\nPushing ronanarraig/tricky-tts-proto-spoken-form-input ({len(input_ds)} rows)...", flush=True)
input_ds.push_to_hub("ronanarraig/tricky-tts-proto-spoken-form-input", split="train", token=HF_TOKEN, private=True)
print("  Done.", flush=True)

# --- Submit Kokoro eval job ---
print(f"\nSubmitting Kokoro eval job...", flush=True)
payload = {
    "model_id": "kokoro",
    "dataset_id": "ronanarraig/tricky-tts-proto-spoken-form-input",
    "split": "train",
    "num_samples": 10,
    "asr_model_id": "openai/whisper-large-v3",
    "language": "auto",
    "tts_model_type": "kokoro",
    "kokoro_voice": "af_heart",
    "max_new_tokens": 4000,
    "push_results": True,
    "output_org": "ronanarraig",
    "output_name": "tricky-tts-eval-ref-kokoro-v2",
    "private": True,
}

resp = requests.post(f"{TRELIS_API}/tts-evaluation/jobs", headers=TRELIS_HEADERS, json=payload)
if resp.status_code in (200, 201):
    job = resp.json()
    job_id = job.get("id") or job.get("job_id")
    print(f"  → Job ID: {job_id}", flush=True)
    Path("tricky-tts/phase3_kokoro_v2_job_id.json").write_text(json.dumps({"job_id": job_id, "output_name": "tricky-tts-eval-ref-kokoro-v2"}))
else:
    print(f"  ERROR {resp.status_code}: {resp.text[:300]}", flush=True)

print("\nDone. Run phase3_build_kokoro_v2.py after job completes.", flush=True)
