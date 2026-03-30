"""
Phase 1d: New research-backed categories for TTS difficulty.

New category set (6 categories × 8 rows = 48 rows):
  edge_cases       — dense symbol/abbreviation ambiguity (proven high WER)
  domain_specific  — dosage chains, IUPAC, legal Latin (proven high WER)
  phonetic         — Celtic names, heteronyms, loanwords (proven WER + naturalness)
  ai_tech          — NEW: AI model names, ML jargon, HuggingFace paths, version strings
  number_format    — NEW: Roman numerals, fractions, ranges, LaTeX, ambiguous number reading
  paralinguistics  — NEW: interjections, elongation, stutters, onomatopoeia (UTMOS + WER)

Sources: EmergentTTS-Eval (NeurIPS 2025), ElevenLabs help docs, NVIDIA NeMo hallucination paper
"""

import os, json, re
from pathlib import Path
from openai import OpenAI

env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

PROMPTS = {
    "edge_cases": """Generate 10 English sentences with very HIGH DENSITY of ambiguous symbols, abbreviations, and mixed notation.
Build on proven high-WER patterns:
- Multiple "St." instances with different meanings in one sentence
- Dense credential chains: D.Phil., F.R.S., B.Sc.(Hons)
- Chemical/physics notation: ±0.5°C, ≥95%, CO₂, μg, m², H₂SO₄
- Phone numbers in various formats: 0800 vs +44(0)20, PIN digit-by-digit
- Mixed unit clusters: 6ft 2in, 220lb, 4.58sec
- Email + URL + abbreviation in one sentence
- Currency ambiguity: $1.2M, £0.5B, €1,234.56
- Ordinal+abbreviation combos: "the 3rd Int'l Conf. on AI, Vol. 12, pp. 345–67"

Each sentence must contain at least 3 distinct hard elements. Make them realistic (not artificially jammed together).
Return as JSON array of strings only.""",

    "domain_specific": """Generate 10 English sentences with MAXIMUM technical density combining multiple hard-to-pronounce elements.
Focus on patterns that trip up TTS (proven from testing):
- Drug dosage chains: "15mg/kg IV q6h of piperacillin-tazobactam 4.5g"
- IUPAC names + CAS numbers + standards: "(2S,5R,6R)-6-amino... per ASTM E70"
- Latin binomials + gene names: "Staphylococcus aureus strain MRSA252 harboring mecA"
- Mixed notation: "0.75mL of 5×10⁻⁶ mol/L at 36.8°C ± 0.3°C maintaining pH 7.4"
- Legal Latin + case citations: "per Daubert v. Merrell Dow, 509 U.S. 579 (1993)"
- Engineering specs: "IEC 60601-1:2005+AMD1:2012 Ed. 3.1", "per RFC 2616 §14.9"
- Multiple binomials in one sentence
- Combine legal + medical + chemical in one realistic sentence

Return as JSON array of strings only.""",

    "phonetic": """Generate 10 English sentences that force difficult pronunciation choices.
Focus on combinations of:
- Celtic/Irish/Welsh names (BOTH forms of heteronyms must appear and be unambiguous):
  Caoilfhinn, Caoimhe, Tadhg, Eithne, Aoife, Siobhán, Saoirse, Niamh, Mairéad
- Stress-shifting heteronyms where BOTH forms appear in one sentence:
  object/object, permit/permit, record/record, present/present, content/content, conduct/conduct
- Rare French/German loanwords in natural academic context:
  schadenfreude, weltanschauung, gemütlichkeit, trompe-l'œil, mise en scène, Bildungsroman, ouroboros
- Words with counter-intuitive pronunciation: Worcestershire, Leicester, Loughborough, Gloucester, Magdalen
- Names of non-English origin used naturally: Nguyen, Mbeki, Laocoön, Ptolemy, Nahuatl

Each sentence should contain 2–3 phonetically difficult elements. Sentences must be natural.
Return as JSON array of strings only.""",

    "ai_tech": """Generate 10 English sentences containing AI/ML technical text that TTS systems commonly mispronounce.
This is a NEW category targeting neural TTS weaknesses on AI jargon. Focus on:
- Model names with mixed case, digits, hyphens: "GPT-4o", "LLaMA 3.1 405B", "Gemma-2-27B-it", "claude-sonnet-4-5", "Mistral 7B v0.3"
- ML abbreviations read aloud: "LoRA fine-tuning", "RLHF", "KV-cache", "FP16 quantization", "INT8", "RAG pipeline"
- HuggingFace-style paths: "meta-llama/Llama-2-7b-chat-hf", "mistralai/Mistral-7B-Instruct-v0.1"
- Academic citation patterns: "Vaswani et al. (2017)", "Brown et al. (2020)", "as per §3.2 of the GPT-4 technical report"
- Training metrics: "3.2B parameters", "trained on 1.4T tokens", "achieved 89.3% on MMLU"
- Version strings and hyphens: "v2.1.3-rc", "checkpoint-5000", "epoch-12"
- Mixed AI org names: "OpenAI", "DeepMind", "Hugging Face", "Mistral AI", "Cohere"

Each sentence must be realistic (something an AI researcher or engineer might actually say/write).
Return as JSON array of strings only.""",

    "number_format": """Generate 10 English sentences where numbers must be read in a specific, non-obvious way.
This is a NEW category targeting number normalization failures — a top TTS weakness per research:
- Roman numerals in context: "King George V", "Chapter XIV", "Super Bowl LVIII", "World War II", "Apollo XI"
  (TTS often reads these as acronyms: "L-V-I-I-I" instead of "fifty-eight")
- Fractions and ratios: "⅔ of the population", "a 3:2 aspect ratio", "the 80/20 rule"
- Ambiguous year vs. number: "1984 vs. 1,984 people", "the 1984 Act requires 15 days notice"
- Ranges and spans: "pp. 234–89", "ages 18–65", "Chapters 3–7", "temperatures of -40 to +120°C"
- Ordinal + context: "the IVth dynasty", "XXIII Olympiad", "Henry VIII's 6th wife"
- Mixed systems: "5 miles (8.05km)", "98.6°F (37°C)", "6 feet 2 inches (188cm)"
- Large number edge cases: "1.5 billion" vs "1,500,000,000" vs "1.5×10⁹"
- Phone/ID number formats: "+44 (0)20 7946 0958", "NI: AB 12 34 56 C", "DOB: 15/04/1989"

Return as JSON array of strings only.""",

    "paralinguistics": """Generate 10 English sentences/utterances that test TTS handling of non-standard expressive text.
This is a NEW category from EmergentTTS-Eval (NeurIPS 2025) targeting paralinguistic features:
- Interjections requiring specific delivery: "Ugh, not again.", "Hmm... interesting.", "Eww, that's disgusting!"
- Vowel elongation for emphasis: "Noooo, that can't be right!", "I'm sooo tired of this."
- Stuttering/hesitation markers: "I-I-I don't know what to say.", "She... she actually did it."
- Onomatopoeia that should sound like the thing: "The clock went tick-tock, tick-tock.", "He went zzz almost immediately."
- ALL CAPS for intensity: "DO NOT touch that under ANY circumstances.", "I TOLD you this would happen."
- Trailing off vs hard stop: "I just thought... never mind.", "She almost said it. Almost."
- Sarcasm/irony markers: "Oh sure, THAT went really well.", "Yeah... great idea."
- Exclamation vs flat delivery: "Fire! Everyone out now!" vs "The building is on fire." (same meaning, different register)
- Compound emotional shifts in one utterance: "'No,' she said quietly. Then: 'NO!' she screamed."

Each must be a realistic utterance. Mix short and long.
Return as JSON array of strings only.""",
}

SPLITS = ["public", "semi_private", "private"]
ROWS_PER_CAT = 8  # target per split

def generate(category, split, n=10):
    split_context = {
        "public": "These are for a publicly released TTS benchmark.",
        "semi_private": "These are for a semi-private eval set — use completely different examples from a public set.",
        "private": "These are for a private held-out test set — use completely different examples from both public and semi-private sets.",
    }
    prompt = PROMPTS[category]
    system = f"You are generating challenging TTS evaluation texts. {split_context[split]} Be creative and precise."
    print(f"  [{category}] / [{split}]...", flush=True)
    response = client.chat.completions.create(
        model="anthropic/claude-sonnet-4-5",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": prompt}],
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

all_rows = []
for split in SPLITS:
    print(f"\n=== {split} ===", flush=True)
    for cat in PROMPTS:
        texts = generate(cat, split, n=10)
        # Take best ROWS_PER_CAT, shuffle slightly
        import random
        random.shuffle(texts)
        for text in texts[:ROWS_PER_CAT]:
            all_rows.append({"text": text, "category": cat, "split": split})

out = Path("tricky-tts/phase1/phase1d_generated.json")
out.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False))
print(f"\nTotal rows: {len(all_rows)}")
from collections import Counter
for split in SPLITS:
    rows = [r for r in all_rows if r["split"] == split]
    cats = Counter(r["category"] for r in rows)
    print(f"  {split}: {dict(sorted(cats.items()))} = {len(rows)} rows")
