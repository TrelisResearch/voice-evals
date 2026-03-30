"""
Phase 1d calibration: replace easy WER=0 rows in ai_tech and number_format.
Paralinguistics WER=0 rows are kept (UTMOS-only by design).
"""
import os, json, re, io, time, requests
from pathlib import Path
from openai import OpenAI
from collections import defaultdict

env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])
ROUTER_KEY = os.environ["TRELIS_ROUTER_API_KEY"]
BASE_URL = "https://router.trelis.com"
HEADERS = {"Authorization": f"Bearer {ROUTER_KEY}"}
ASR_MODEL = "assemblyai/universal-3-pro"

def normalize(text):
    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def wer(ref, hyp):
    r, h = normalize(ref).split(), normalize(hyp).split()
    if not r: return 0.0
    d = [[0]*(len(h)+1) for _ in range(len(r)+1)]
    for i in range(len(r)+1): d[i][0] = i
    for j in range(len(h)+1): d[0][j] = j
    for i in range(1,len(r)+1):
        for j in range(1,len(h)+1):
            cost = 0 if r[i-1]==h[j-1] else 1
            d[i][j] = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+cost)
    return d[len(r)][len(h)] / len(r)

def synthesize(text):
    r = requests.post(f"{BASE_URL}/api/v1/synthesize", headers=HEADERS,
        data={"text": text, "model": "elevenlabs/eleven-multilingual-v2", "output_format": "mp3_44100_128"},
        timeout=30)
    return r.content if r.status_code == 200 else None

def transcribe(audio, retries=2):
    for attempt in range(retries+1):
        try:
            r = requests.post(f"{BASE_URL}/api/v1/transcribe", headers=HEADERS,
                files={"file": ("audio.mp3", io.BytesIO(audio), "audio/mpeg")},
                data={"model": ASR_MODEL, "language": "en", "output_format": "json"},
                timeout=120)
            if r.status_code == 200:
                return r.json().get("text","").strip()
            elif r.status_code == 429 and attempt < retries:
                time.sleep(3)
        except requests.exceptions.Timeout:
            if attempt < retries: time.sleep(3)
    return None

# ── Easy rows to replace (non-paralinguistics WER=0) ──────────────────────────
REPLACE = {
    # ai_tech — standard citations/version strings, too clean
    "According to Vaswani et al. (2017), the transformer architecture enabled GPT-4o to achieve state-of-the-art performance across multiple benchmarks.",
    "The model trained on 1.4T tokens reached convergence at epoch-12 using the v2.1.3-rc checkpoint with AdamW optimizer.",
    "Our RAG pipeline uses Mistral 7B v0.3 with INT8 quantization and KV-cache optimization, achieving 12ms p50 latency.",
    # number_format — ElevenLabs handled Super Bowl LVIII correctly
    "Super Bowl LVIII drew over 123.7 million viewers, making it the most-watched program in US television history.",
}

# ── Generate harder replacements ──────────────────────────────────────────────
harder_prompts = {
    "ai_tech": """Generate 8 English sentences about AI/ML that are MAXIMALLY HARD for TTS to pronounce correctly.
The easy patterns to AVOID: clean citations like "Vaswani et al. (2017)", simple version strings like "v2.1.3".
Focus on patterns that actually trip up TTS:
- Unusual model names with non-obvious casing/hyphens: "Mixtral-8×7B-Instruct-v0.1", "DeepSeek-R1-Distill-Qwen-32B", "Qwen2.5-72B-Instruct", "Phi-3.5-MoE-instruct", "Yi-1.5-34B-Chat-16K"
- HuggingFace org/repo paths with special chars: "Qwen/Qwen2.5-72B-Instruct", "deepseek-ai/DeepSeek-R1", "microsoft/Phi-3.5-MoE-instruct"
- Mixed benchmark notation: "89.3% on MMLU", "92.1 on HumanEval", "67.3 on GSM8K@4-shot"
- Training configs dense with abbreviations: "4×A100 80GB SXM with bf16, ZeRO-3, gradient_checkpointing=True, lr=2e-5"
- Model version chains: "based on LLaMA-3.1-8B → fine-tuned on ShareGPT4 → RLHF'd with DPO"
- Obscure acronyms in natural sentences: "We use MoE with 8 experts, top-2 routing, and FoT attention"
Return as JSON array of strings only.""",

    "number_format": """Generate 8 English sentences where numbers are in formats TTS commonly gets wrong.
Avoid sentences ElevenLabs handled correctly: Super Bowl LVIII (standard Roman numeral context).
Focus on:
- Roman numerals in LESS common contexts: "the IVth century BC", "Act III Scene iv", "the XXIst Amendment", "Volume MCMXCIX"
- Fractions as Unicode: "⅔ of respondents", "¾ teaspoon", "⅛ note in music"
- Ranges with unusual notation: "ages 18–65 (n=1,984)", "pp. iv–xii", "§§3–7 of the Act"
- Ambiguous year-as-number: "1,984 participants were enrolled in the 1984 Act study"
- Mixed number systems in one sentence: "the 3rd (III) King Henry, the 8th (VIII), and the XXIInd"
- Scientific notation edge cases: "10⁻⁶ mol/L", "6.022×10²³ molecules", "2⁸ = 256 memory addresses"
- Time formats: "0800hrs", "23:59:59 UTC+5:30", "T-minus 00:02:37"
Return as JSON array of strings only.""",
}

def generate(prompt):
    response = client.chat.completions.create(
        model="anthropic/claude-sonnet-4-5",
        messages=[
            {"role": "system", "content": "Generate challenging TTS evaluation texts focused on what genuinely trips up neural TTS systems."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
    )
    content = response.choices[0].message.content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        return [str(t).strip() for t in json.loads(content) if str(t).strip()]
    except:
        return []

print("=== GENERATING HARDER REPLACEMENTS ===\n", flush=True)
candidates = {}
for cat, prompt in harder_prompts.items():
    texts = generate(prompt)
    candidates[cat] = texts
    print(f"[{cat}] Generated {len(texts)}:", flush=True)
    for t in texts:
        print(f"  {t[:85]}", flush=True)

print("\n=== ROUND-TRIP TESTING CANDIDATES ===\n", flush=True)
tested = []
cat_map = {"ai_tech": "ai_tech", "number_format": "number_format"}
for cat, texts in candidates.items():
    for text in texts:
        print(f"[{cat}] {text[:70]}...", flush=True)
        audio = synthesize(text)
        if not audio:
            print("  TTS failed", flush=True)
            continue
        transcript = transcribe(audio)
        if not transcript:
            print("  ASR failed", flush=True)
            continue
        w = wer(text, transcript)
        print(f"  WER={w:.3f}  → {transcript[:75]}", flush=True)
        tested.append({"text": text, "category": cat, "wer": round(w,3), "transcript": transcript})

# Sort by WER descending per category
by_cat = defaultdict(list)
for r in tested:
    by_cat[r["category"]].append(r)
for cat in by_cat:
    by_cat[cat].sort(key=lambda r: -r["wer"])
    print(f"\n  [{cat}] best candidates:", flush=True)
    for r in by_cat[cat][:4]:
        print(f"    WER={r['wer']:.3f}  {r['text'][:75]}", flush=True)

# ── Apply replacements ─────────────────────────────────────────────────────────
results = json.loads(Path("tricky-tts/phase1/phase1d_results.json").read_text())
used = defaultdict(int)
final = []
for row in results:
    if row["text"] in REPLACE:
        cat = row["category"]
        idx = used[cat]
        if idx < len(by_cat.get(cat, [])):
            best = by_cat[cat][idx]
            used[cat] += 1
            print(f"\n  REPLACING [{cat}] {row['wer']:.3f}→{best['wer']:.3f}", flush=True)
            print(f"    OLD: {row['text'][:70]}", flush=True)
            print(f"    NEW: {best['text'][:70]}", flush=True)
            final.append(best)
        else:
            print(f"  KEEPING [{cat}] (no better candidate)", flush=True)
            final.append(row)
    else:
        final.append(row)

# ── Final stats ────────────────────────────────────────────────────────────────
print("\n" + "="*65, flush=True)
print("PHASE 1D FINAL STATS (after calibration)", flush=True)
print("="*65, flush=True)
cat_wers = defaultdict(list)
for r in final:
    cat_wers[r["category"]].append(r["wer"])
print(f"\n{'Category':<20} {'N':>4} {'Avg':>7} {'Max':>7} {'Min':>7}", flush=True)
print("-"*50, flush=True)
for cat in sorted(cat_wers):
    ws = cat_wers[cat]
    print(f"{cat:<20} {len(ws):>4} {sum(ws)/len(ws):>7.3f} {max(ws):>7.3f} {min(ws):>7.3f}", flush=True)
all_wers = [r["wer"] for r in final]
print(f"\n{'Overall':<20} {len(all_wers):>4} {sum(all_wers)/len(all_wers):>7.3f}", flush=True)
easy = sum(1 for w in all_wers if w < 0.05)
hard = sum(1 for w in all_wers if w > 0.30)
print(f"Easy (WER<0.05): {easy}/{len(all_wers)} | Hard (WER>0.30): {hard}/{len(all_wers)}", flush=True)

Path("tricky-tts/phase1/phase1d_final.json").write_text(
    json.dumps([{"text": r["text"], "category": r["category"]} for r in final], indent=2, ensure_ascii=False))
Path("tricky-tts/phase1/phase1d_final_results.json").write_text(json.dumps(final, indent=2, ensure_ascii=False))
print(f"\nSaved {len(final)} rows to phase1d_final.json", flush=True)
