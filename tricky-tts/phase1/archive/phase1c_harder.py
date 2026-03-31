"""
Phase 1c: Replace remaining WER=0 rows (excluding prosody, which is UTMOS-only)
with harder alternatives. Keep improvements only.

Target replacements:
- punctuation: 5 WER=0 rows → embed numbers/abbreviations within complex punctuation
- robustness: 3 WER=0 rows → harder repeated patterns / dense technical robustness
- phonetic: 1 WER=0 row → harder heteronym
- edge_cases: add more St./Dr./abbreviation ambiguity density
- domain_specific: denser compound technical strings
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

# ── helpers ────────────────────────────────────────────────────────────────────
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

def cer(ref, hyp):
    r = normalize(ref).replace(" ","")
    h = normalize(hyp).replace(" ","")
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
            else:
                print(f"  ASR error {r.status_code}: {r.text[:100]}", flush=True)
                return None
        except requests.exceptions.Timeout:
            if attempt < retries: time.sleep(3)
            else: return None
    return None

def generate(prompt, n=6):
    response = client.chat.completions.create(
        model="anthropic/claude-sonnet-4-5",
        messages=[
            {"role": "system", "content": "Generate challenging TTS evaluation texts. Focus on what genuinely trips up text-to-speech systems, not just what sounds clever."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
    )
    content = response.choices[0].message.content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        texts = json.loads(content)
        return [str(t).strip() for t in texts if str(t).strip()][:n]
    except:
        return []

# ── Rows to target ─────────────────────────────────────────────────────────────
# WER=0 non-prosody rows worth replacing
REPLACE_TEXTS = {
    # Punctuation rows — replace with punctuation + embedded numbers/abbrevs
    "The discovery—unprecedented in its scope—the analysis—which took months to complete—and the implications—far-reaching beyond anyone's imagination—forced the committee to reconsider everything.",
    "'I understand your concern,' he interrupted coldly, 'but the decision, frankly, isn't yours to make.'",
    "She wondered, what's the point of trying?; he believed effort mattered regardless of outcome; they would never agree.",
    "He started to explain... then stopped, realizing the truth was worse than the lie... much worse.",
    "The defendant (who had previously denied everything (even under oath)) suddenly admitted his role—partial though it was—in the conspiracy.",
    # Robustness — replace garden-path novelties with something genuinely harder
    "The fact that that that that that student used in his essay was grammatically incorrect prompted the professor to explain that that that is different from that that.",
    "James, while John had had had, had had had had; had had had had the teacher's approval.",
    # Phonetic — replace easy heteronym
    "Lachlan tried to read the minute details in the minute handwriting, feeling schadenfreude when his colleague couldn't discern the subtle differences either.",
}

# ── Harder generation prompts ──────────────────────────────────────────────────
HARDER = {
    "punctuation_with_numbers": """Generate 6 English sentences combining COMPLEX PUNCTUATION with ABBREVIATIONS or NUMBERS.
The goal: TTS must correctly handle both the punctuation (pauses, intonation) AND the abbreviation/number expansion simultaneously.
Examples of what we want:
- "The contract—signed on 03/04/2025 by Dr. J. Smith, Ph.D.—stipulates £1,200/mo. until Dec. 31st."
- "'We expect approx. 1,500 attendees,' said the CEO of TechCorp Ltd., 'which is 3× our 2024 figure.'"
Be creative. Mix: em-dashes or parentheses WITH currencies, dates, credentials, abbreviated titles, ordinals.
Return as JSON array of strings only.""",

    "robustness_hard": """Generate 6 English sentences that genuinely stress-test TTS robustness and produce INCONSISTENT output across models.
Focus on:
- Sequences of numbers that must be read as individual digits vs as a number: "Call 0800 123 4567 ext. 89", "PIN: 4-4-2-8", "the year 1984 vs. the number 1,984"
- Unit clusters: "the 5ft 11in, 185lb athlete ran 5km in 19min 32sec"
- All-caps acronyms mid-sentence followed by numbers: "The NATO 2B14 mortar", "NASA's SLS Block 1B"
- Hyphenated compound modifiers with numbers: "a 3-to-1 ratio", "the well-known 80/20 rule", "a 40-year-old, 6-foot-tall, award-winning author"
Return as JSON array of strings only.""",

    "phonetic_hard": """Generate 6 English sentences with MULTIPLE phonetically ambiguous or difficult elements per sentence.
Focus on combinations:
- Multiple heteronyms in one sentence where BOTH must be read correctly by context
- Names from Celtic/Welsh/Polish/Irish tradition combined with unusual English words: Niamh, Caoilfhinn, Saoirse, Caoimhe, Tadhg, Eithne, Siobhán, Aoife
- Foreign terms used naturally in English academic/culinary sentences: schadenfreude, Weltanschauung, gemütlichkeit, mise en scène, trompe-l'œil, eau de vie
- Words where stress pattern changes meaning: "object/object", "record/record", "permit/permit", "content/content"
Each sentence should have at least 2–3 phonetically tricky elements.
Return as JSON array of strings only.""",

    "edge_cases_denser": """Generate 6 English sentences with VERY HIGH DENSITY of abbreviation and symbol ambiguity.
Build on what works: "Dr. St. James" and "Prof. O'Brien, Ph.D., M.D." scored highest.
Target:
- Multiple ambiguous "St." instances (Street? Saint? Saint's?): "St. James's St." "St. Peter's Sq."
- Chains of credentials: "Dr. Prof. R.J. O'Sullivan, B.Sc.(Hons), M.A., D.Phil., F.R.S."
- Dense mixed units and symbols in one sentence: "±0.5°C", "≥95%", "CO₂", "$1.2M", "No. 3"
- Ambiguous abbreviations mid-number: "The vol. 3 No. 12 issue of J. Med. Chem. (pp. 1234–56)"
- Email + URL + abbreviation in one sentence
Return as JSON array of strings only.""",

    "domain_denser": """Generate 6 English sentences with MAXIMUM technical density — multiple hard-to-pronounce elements per sentence.
Target what tripped up TTS before: dosages, IUPAC names, standards codes.
Now combine them:
- Drug + dose + route + frequency + indication in one sentence: "Administer 2.5mg/kg IV q8h of meropenem for Pseudomonas aeruginosa sepsis per IDSA 2023 guidelines"
- IUPAC + CAS number + standard in one sentence
- Multiple Latin binomials + gene names: "Staphylococcus aureus strain MRSA252 expressing mecA and pvl genes"
- Mixed units and notation: "inject 0.5mL of 10⁻³ mol/L solution at 37°C ± 0.5°C"
- Legal citations with Latin: "per In re Winship, 397 U.S. 358 (1970) applying mens rea beyond reasonable doubt"
Return as JSON array of strings only.""",
}

# ── Generate candidates ────────────────────────────────────────────────────────
print("=== GENERATING HARDER CANDIDATES ===\n", flush=True)
candidates = {}
for key, prompt in HARDER.items():
    print(f"Generating [{key}]...", flush=True)
    texts = generate(prompt, n=6)
    candidates[key] = texts
    for t in texts:
        print(f"  {t[:90]}", flush=True)

# ── Test all candidates ────────────────────────────────────────────────────────
print("\n=== ROUND-TRIP TESTING CANDIDATES ===\n", flush=True)
tested = []
for key, texts in candidates.items():
    cat_map = {
        "punctuation_with_numbers": "punctuation",
        "robustness_hard": "robustness",
        "phonetic_hard": "phonetic",
        "edge_cases_denser": "edge_cases",
        "domain_denser": "domain_specific",
    }
    cat = cat_map[key]
    for text in texts:
        print(f"[{cat}] {text[:75]}...", flush=True)
        audio = synthesize(text)
        if not audio:
            print("  TTS failed", flush=True)
            continue
        transcript = transcribe(audio)
        if not transcript:
            print("  ASR failed", flush=True)
            continue
        w = wer(text, transcript)
        c = cer(text, transcript)
        print(f"  WER={w:.3f} CER={c:.3f}  → {transcript[:75]}", flush=True)
        tested.append({"text": text, "category": cat, "wer": round(w,3), "cer": round(c,3), "transcript": transcript, "source_key": key})

# ── Select best candidates per replacement slot ────────────────────────────────
print("\n=== SELECTING BEST CANDIDATES ===\n", flush=True)
by_cat = defaultdict(list)
for r in tested:
    by_cat[r["category"]].append(r)

# Sort by WER descending (hardest first)
for cat in by_cat:
    by_cat[cat].sort(key=lambda r: -r["wer"])
    print(f"  [{cat}] top candidates:", flush=True)
    for r in by_cat[cat][:3]:
        print(f"    WER={r['wer']:.3f}  {r['text'][:75]}", flush=True)

# ── Load current dataset and apply replacements ────────────────────────────────
curated = json.loads(Path("tricky-tts/phase1/curated_results.json").read_text())
curated_texts = {r["text"] for r in curated}

# Build replacement map: replace REPLACE_TEXTS with best new candidates per category
new_dataset = []
used_replacements = defaultdict(int)

# How many slots to fill per category (replace WER=0 non-prosody rows)
SLOTS = {"punctuation": 5, "robustness": 2, "phonetic": 1}

replacements_done = defaultdict(list)

for row in curated:
    if row["text"] in REPLACE_TEXTS:
        cat = row["category"]
        # Find best unused candidate for this category
        slot_idx = used_replacements[cat]
        if slot_idx < len(by_cat.get(cat, [])):
            best = by_cat[cat][slot_idx]
            used_replacements[cat] += 1
            print(f"  REPLACING [{cat}] WER {row['wer']:.3f}→{best['wer']:.3f}", flush=True)
            print(f"    OLD: {row['text'][:70]}", flush=True)
            print(f"    NEW: {best['text'][:70]}", flush=True)
            new_dataset.append(best)
            replacements_done[cat].append(best)
        else:
            # No better candidate found, keep original
            print(f"  KEEPING [{cat}] WER={row['wer']:.3f} (no better candidate) {row['text'][:60]}", flush=True)
            new_dataset.append(row)
    else:
        new_dataset.append(row)

# Also try to upgrade the weakest edge_cases and domain_specific rows
# by adding the best new candidates if they beat existing rows
for cat in ["edge_cases", "domain_specific"]:
    if not by_cat.get(cat):
        continue
    # Find the weakest existing row in this category
    cat_rows = [(i, r) for i, r in enumerate(new_dataset) if r["category"] == cat]
    cat_rows.sort(key=lambda x: x[1].get("wer", 0))
    weakest_idx, weakest_row = cat_rows[0]  # lowest WER
    best_new = by_cat[cat][0]  # highest WER new candidate
    if best_new["wer"] > weakest_row.get("wer", 0) + 0.03 and best_new["text"] not in {r["text"] for r in new_dataset}:
        print(f"\n  UPGRADING [{cat}] weakest WER={weakest_row['wer']:.3f}→{best_new['wer']:.3f}", flush=True)
        new_dataset[weakest_idx] = best_new

# ── Final stats ────────────────────────────────────────────────────────────────
print("\n" + "="*65, flush=True)
print("PHASE 1C — FINAL STATS", flush=True)
print("="*65, flush=True)

cat_wers = defaultdict(list)
for r in new_dataset:
    if "wer" in r:
        cat_wers[r["category"]].append(r["wer"])

print(f"\n{'Category':<20} {'Rows':>5} {'Avg WER':>8} {'Max WER':>8} {'Min WER':>8}", flush=True)
print("-"*55, flush=True)
for cat in sorted(cat_wers):
    ws = cat_wers[cat]
    print(f"{cat:<20} {len(ws):>5} {sum(ws)/len(ws):>8.3f} {max(ws):>8.3f} {min(ws):>8.3f}", flush=True)

all_wers = [r["wer"] for r in new_dataset if "wer" in r]
print(f"\n{'Overall':<20} {len(all_wers):>5} {sum(all_wers)/len(all_wers):>8.3f}", flush=True)
easy = sum(1 for w in all_wers if w < 0.05)
hard = sum(1 for w in all_wers if w > 0.30)
print(f"\nEasy (WER<0.05): {easy}/{len(all_wers)} | Hard (WER>0.30): {hard}/{len(all_wers)}", flush=True)

# Save final dataset
final_rows = [{"text": r["text"], "category": r["category"]} for r in new_dataset]
Path("tricky-tts/phase1/phase1c_public.json").write_text(json.dumps(final_rows, indent=2, ensure_ascii=False))
Path("tricky-tts/phase1/phase1c_results.json").write_text(json.dumps(new_dataset, indent=2, ensure_ascii=False))
print(f"\nSaved {len(final_rows)} rows to phase1c_public.json", flush=True)
