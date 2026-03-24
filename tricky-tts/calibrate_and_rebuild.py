"""
Analyze round-trip results, identify weak rows, generate harder replacements,
and rebuild the final curated public dataset.

Strategy:
- Keep rows with WER > 0 (proven tricky for ElevenLabs — likely discriminating across models)
- Keep some WER=0 rows that are valuable for UTMOS naturalness testing (complex prosody)
- Replace generic/trivial WER=0 rows with harder texts targeting known TTS failure modes
- Target: avg WER ~0.10–0.20 with good category coverage
"""

import os
import json
import re
from pathlib import Path
from openai import OpenAI

env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

results = json.loads(Path("tricky-tts/roundtrip_results.json").read_text())

# ── Step 1: Categorise existing rows ──────────────────────────────────────────
print("=== EXISTING ROW ANALYSIS ===\n")
keep = []        # keep as-is
replace = []     # replace with harder version

# Decision rules:
# - WER >= 0.05: keep (genuinely tricky, discriminating)
# - WER < 0.05, prosody: keep only long/complex ones for UTMOS value; replace simple ones
# - WER < 0.05, other categories: replace with harder version

PROSODY_UTMOS_KEEPERS = {
    # Long complex sentences worth keeping for naturalness/rhythm testing
    "Have you ever wondered",
    "The manuscript revealed secrets",
    "The breakthrough discovery wasn't made",
    "Why would anyone choose to spend",
    "When the afternoon sun finally broke",
    "First, we visited the ancient cathedral",
}

for r in results:
    wer = r.get("wer", 0)
    cat = r["category"]
    text = r["text"]
    decision = None

    if wer >= 0.05:
        decision = "KEEP"
        reason = f"WER={wer:.3f} — proven tricky"
    elif cat == "prosody":
        if any(text.startswith(k) for k in PROSODY_UTMOS_KEEPERS):
            decision = "KEEP"
            reason = f"WER={wer:.3f} — long/complex, UTMOS value"
        else:
            decision = "REPLACE"
            reason = f"WER={wer:.3f} — too simple for discrimination"
    elif wer == 0.0:
        decision = "REPLACE"
        reason = f"WER={wer:.3f} — trivially easy, replace with harder version"
    else:
        decision = "KEEP"
        reason = f"WER={wer:.3f} — borderline, keep"

    print(f"  [{decision}] [{cat}] {reason}")
    print(f"         {text[:90]}")

    if decision == "KEEP":
        keep.append(r)
    else:
        replace.append(r)

print(f"\nKeeping: {len(keep)} | Replacing: {len(replace)}")

# ── Step 2: Count gaps per category ───────────────────────────────────────────
from collections import Counter
keep_cats = Counter(r["category"] for r in keep)
target_per_cat = 8  # aim for ~8 per category in a 50-row dataset (6 cats = 48 + 2 flex)

gaps = {}
for cat in ["prosody", "edge_cases", "phonetic", "punctuation", "robustness", "domain_specific"]:
    have = keep_cats.get(cat, 0)
    need = max(0, target_per_cat - have)
    gaps[cat] = need
    print(f"  {cat}: have {have}, need {need} more")

# ── Step 3: Generate harder replacements ──────────────────────────────────────

HARDER_PROMPTS = {
    "prosody": """Generate {n} English sentences specifically designed to challenge TTS prosody.
Focus on patterns that trip up text-to-speech systems:
- Garden-path sentences requiring precise pause placement
- Nested appositives: "My brother, the one who moved to Berlin last year, called."
- Contrastive stress sentences where word emphasis changes meaning: "SHE didn't say he STOLE it."
- Lists with irregular cadence (not just "and finally")
- Direct address mid-sentence: "I need you, John, to listen carefully."
- Sentences where comma placement dramatically changes prosody
- Long relative clauses that require natural breath pauses

Return as a JSON array of strings only. No other text.""",

    "edge_cases": """Generate {n} English sentences with AMBIGUOUS or DIFFICULT-TO-PRONOUNCE text elements.
Focus on things that trip up TTS systems:
- Ambiguous abbreviations: "He lives on Oak St." (Street or Saint?), "Call Dr. Cook at St. Mary's"
- Mixed number formats: "£1,234.56", "€5,000.00", "The temperature was -15°C or 5°F"
- Ambiguous date formats: "The deadline is 04/05/2026"
- Degree/unit symbols: "98.6°F", "5km²", "200μg"
- Mixed-case tech brands in natural sentences: "I use VS Code on macOS", "My iPhone 15 Pro runs iOS 17"
- Academic credentials: "She holds a Ph.D. and an M.B.A. from MIT"
- Tricky ordinals and ranges: "Chapters 3–7", "the 1,002nd visitor"

Return as a JSON array of strings only. No other text.""",

    "phonetic": """Generate {n} English sentences where correct pronunciation is non-obvious or context-dependent.
Focus on:
- Multiple heteronyms in one sentence: "I read that she will read the lead article about lead poisoning."
- Less common loanwords: "schadenfreude", "weltanschauung", "Zeitgeist", "Bildungsroman", "ouroboros"
- Difficult names from multiple cultures: "Aadhya", "Caoilfhinn", "Przemysław", "Lachlan", "Mairéad"
- Words with silent letters in context: "The subtle nuance in his paradigm was acknowledged by the colonel."
- Proper nouns with unexpected stress: "Versailles", "Cannes", "Leicester", "Edinburgh", "Gloucester"

Return as a JSON array of strings only. No other text.""",

    "punctuation": """Generate {n} English sentences where punctuation is CRITICAL to correct prosody.
Focus on cases where TTS systems commonly fail:
- Multiple em-dashes: "The result—shocking as it was—and the aftermath—which dragged on for years—changed everything."
- Interrupted speech or parenthetical within parenthetical
- Semicolons used to join contrasting ideas that need distinct pauses
- Rhetorical questions embedded in statements: "She asked herself, why bother?, and went back to sleep."
- Complex quoted speech with attribution mid-quote: "'I never,' she said firmly, 'agreed to that.'"
- Trailing ellipsis vs mid-sentence ellipsis used differently
- Colons followed by long lists requiring sustained intonation

Return as a JSON array of strings only. No other text.""",

    "robustness": """Generate {n} English sentences that stress-test TTS robustness.
Focus on:
- Strings of identical or near-identical words that must be spoken distinctly: "that that that clause", "had had had"
- Tongue twisters embedded in sentences: "She sells seashells by the seashore, but the shells she sells aren't cheap."
- Very long single sentences (200+ words) that require sustained coherent delivery
- Single word or very short utterances that must sound natural: "Hmm.", "No.", "Perhaps."
- Intentional repetition for emphasis: "slowly, very slowly, almost imperceptibly slowly"
- Sentences with awkward consonant clusters or alliteration

Return as a JSON array of strings only. No other text.""",

    "domain_specific": """Generate {n} English sentences with highly technical jargon that TTS systems often mispronounce.
Focus on:
- Drug names/dosages: "The patient received 2.5mg of warfarin sodium and 40mg pantoprazole"
- Chemical IUPAC names: "2-acetyloxybenzoic acid" (aspirin), "ethylenediaminetetraacetic acid (EDTA)"
- Latin taxonomic binomials: "Homo sapiens", "Escherichia coli", "Staphylococcus aureus"
- Legal Latin in context: "The court applied the doctrine of res ipsa loquitur"
- Engineering standards/codes: "The IEC 60601 standard", "per RFC 2616", "ASTM D638 testing"
- Mathematical notation read aloud: "x squared plus 2x minus 15 equals zero"
- Medical acronyms and initialisms mixed with full terms

Return as a JSON array of strings only. No other text.""",
}


def generate_harder(category: str, n: int) -> list[str]:
    if n == 0:
        return []
    print(f"\n  Generating {n} harder [{category}] texts...", flush=True)
    prompt = HARDER_PROMPTS[category].format(n=n)
    response = client.chat.completions.create(
        model="anthropic/claude-sonnet-4-5",
        messages=[
            {"role": "system", "content": "Generate evaluation texts for a TTS benchmark. Be creative, precise, and focused on edge cases that genuinely challenge TTS systems."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
    )
    content = response.choices[0].message.content.strip()
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)
    try:
        texts = json.loads(content)
        texts = [str(t).strip() for t in texts if str(t).strip()]
        print(f"    Got {len(texts)}", flush=True)
        return texts
    except Exception as e:
        print(f"    Parse error: {e}", flush=True)
        return []


print("\n=== GENERATING HARDER REPLACEMENTS ===")
new_rows = []

for cat, n_needed in gaps.items():
    # Generate a bit extra in case some are weak
    texts = generate_harder(cat, max(n_needed, 3) if n_needed > 0 else 0)
    for text in texts[:n_needed]:
        new_rows.append({"text": text, "category": cat, "split": "public"})

# ── Step 4: Assemble final dataset ────────────────────────────────────────────
final = []
for r in keep:
    final.append({"text": r["text"], "category": r["category"], "split": "public"})
for r in new_rows:
    final.append(r)

# Trim to 50, ensuring category balance
from collections import defaultdict
by_cat = defaultdict(list)
for r in final:
    by_cat[r["category"]].append(r)

# Take up to 9 per category, total 50
balanced = []
for cat in ["prosody", "edge_cases", "phonetic", "punctuation", "robustness", "domain_specific"]:
    rows = by_cat[cat][:9]
    balanced.extend(rows)

# Pad to 50 if needed (take leftovers from any category)
if len(balanced) < 50:
    all_used = set(r["text"] for r in balanced)
    for cat in by_cat:
        for r in by_cat[cat]:
            if len(balanced) >= 50:
                break
            if r["text"] not in all_used:
                balanced.append(r)
                all_used.add(r["text"])

balanced = balanced[:50]

print(f"\n=== FINAL DATASET: {len(balanced)} rows ===")
cat_counts = Counter(r["category"] for r in balanced)
for cat, count in sorted(cat_counts.items()):
    print(f"  {cat}: {count}")

# Save
out = Path("tricky-tts/curated_public.json")
out.write_text(json.dumps(balanced, indent=2, ensure_ascii=False))
print(f"\nSaved to {out}")
