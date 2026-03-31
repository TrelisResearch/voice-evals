"""
Generate tricky-tts dataset texts using OpenRouter LLM.
Produces 50 rows per split (public, semi-private, private) across 6 categories.
"""

import os
import json
import random
import re
import sys
from pathlib import Path
from openai import OpenAI

# Load env
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

CATEGORIES = [
    "prosody",
    "edge_cases",
    "phonetic",
    "punctuation",
    "robustness",
    "domain_specific",
]

SPLITS = ["public", "semi_private", "private"]

# 9 rows per category per split (54 total per split, trim to 50)
ROWS_PER_CATEGORY_PER_SPLIT = 9

PROMPTS = {
    "prosody": """Generate {n} English sentences to test TTS prosody and naturalness. Each should be a distinct test case. Include a mix of:
- Long complex sentences that require natural rhythm (30–50 words)
- Questions vs statements testing pitch variation
- Lists ending with "and finally..." or "lastly..." cadence
- Sentences with natural stress patterns on key words

Rules: English only. Each sentence must be standalone and self-contained. Varied topics. No duplicates.
Return as a JSON array of strings. Just the array, no other text.""",

    "edge_cases": """Generate {n} English sentences to test TTS handling of edge cases. Each should contain at least one tricky element. Include a mix of:
- Numbers, dates, currencies: e.g. "£1,234.56", "$0.99", "03/04/2025", "the 1980s"
- Abbreviations and acronyms: "CEO", "NASA", "Dr.", "St.", "Prof.", "Ltd.", "approx."
- URLs and email addresses (short, realistic ones)
- Mixed-case proper nouns: "iPhone", "ChatGPT", "macOS", "YouTube", "PayPal"
- Ordinal numbers: "21st", "3rd", "42nd"

Rules: English only. Each sentence must feel natural in context. No duplicates.
Return as a JSON array of strings. Just the array, no other text.""",

    "phonetic": """Generate {n} English sentences to test TTS phonetic difficulty. Each should contain at least one tricky word. Include a mix of:
- Heteronyms (same spelling, different pronunciation depending on context): read/read, lead/lead, bow/bow, wound/wound, tear/tear, live/live
- Foreign loanwords: croissant, naive, café, fiancé, résumé, genre, debris, façade, entrepreneur
- Unusual names: Siobhan, Wojciech, Niamh, Ngozi, Saoirse, Ptolemy, Goethe
- Silent letters or unusual spellings: Wednesday, colonel, queue, phlegm, yacht

Rules: English only. Context must make the intended pronunciation unambiguous. Varied sentences. No duplicates.
Return as a JSON array of strings. Just the array, no other text.""",

    "punctuation": """Generate {n} English sentences to test TTS punctuation handling. Each should rely on punctuation for natural delivery. Include a mix of:
- Em-dashes mid-sentence: "The result—unexpected as it was—changed everything."
- Ellipses indicating trailing off or pause: "Well... I suppose that could work."
- Parenthetical asides: "The package (which arrived late) was damaged."
- Quoted speech within a sentence: She said, "I'll be there at noon," and hung up.
- Semicolons connecting related clauses
- Colons introducing lists or explanations

Rules: English only. Punctuation must feel natural, not forced. Varied topics. No duplicates.
Return as a JSON array of strings. Just the array, no other text.""",

    "robustness": """Generate {n} English sentences to test TTS robustness at extremes. Include a mix of:
- Very short utterances (1–5 words): "Yes.", "Okay, thanks.", "Absolutely not.", "See you then."
- Very long utterances (150–250 words): a single coherent paragraph on any topic
- Repeated words that must be read correctly: "the the", "very very very", "no no no, I said no"
- Sentences with unusual word order or fragments
- All-caps words for emphasis: "This is URGENT"

Rules: English only. Short and long sentences should both be realistic speech. No duplicates.
Return as a JSON array of strings. Just the array, no other text.""",

    "domain_specific": """Generate {n} English sentences containing technical, medical, or legal jargon to test TTS domain handling. Include a mix of:
- Medical: drug names (acetaminophen, metformin, amoxicillin), anatomical terms, diagnoses
- Legal: Latin phrases (habeas corpus, mens rea, prima facie), legal terminology
- Technical/engineering: specific component names, protocols, units of measurement
- Financial: specific instruments, terms (amortization, fiduciary, collateral)
- Scientific: chemical names, taxonomy, physics terms

Rules: English only. Each sentence must be realistic (something a professional might actually say). Varied domains. No duplicates.
Return as a JSON array of strings. Just the array, no other text.""",
}


def generate_texts_for_category(category: str, n: int, split: str, attempt: int = 0) -> list[str]:
    """Generate n texts for a given category, adding split context to encourage variation."""
    split_context = {
        "public": "These will be in a publicly released dataset.",
        "semi_private": "These are for a semi-private evaluation set — use different examples from the public set.",
        "private": "These are for a private held-out test set — use completely different examples from the public and semi-private sets.",
    }

    prompt = PROMPTS[category].format(n=n)
    system = f"You are generating evaluation data for a TTS benchmark. {split_context[split]} Be creative and vary sentence topics, structures, and difficulty levels."

    print(f"  Generating {n} texts for [{category}] / [{split}]...", flush=True)

    response = client.chat.completions.create(
        model="anthropic/claude-sonnet-4-5",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.9,
    )

    content = response.choices[0].message.content.strip()

    # Strip markdown code blocks if present
    content = re.sub(r"^```(?:json)?\s*", "", content)
    content = re.sub(r"\s*```$", "", content)

    try:
        texts = json.loads(content)
        if not isinstance(texts, list):
            raise ValueError("Expected a list")
        texts = [str(t).strip() for t in texts if str(t).strip()]
        print(f"    Got {len(texts)} texts", flush=True)
        return texts
    except Exception as e:
        print(f"    Parse error: {e}\n    Raw: {content[:200]}", flush=True)
        if attempt < 2:
            return generate_texts_for_category(category, n, split, attempt + 1)
        return []


def main():
    all_rows = []

    for split in SPLITS:
        print(f"\n=== Generating split: {split} ===", flush=True)
        split_rows = []

        for category in CATEGORIES:
            texts = generate_texts_for_category(category, ROWS_PER_CATEGORY_PER_SPLIT, split)
            for text in texts:
                split_rows.append({"text": text, "category": category, "split": split})

        # Trim or pad to 50
        random.shuffle(split_rows)
        split_rows = split_rows[:50]
        print(f"  Total rows for {split}: {len(split_rows)}", flush=True)
        all_rows.extend(split_rows)

    # Save to JSON for review
    output_path = Path("/home/claude/TR/voice-evals/tricky-tts/phase1/generated_texts.json")
    output_path.write_text(json.dumps(all_rows, indent=2, ensure_ascii=False))
    print(f"\nSaved {len(all_rows)} rows to {output_path}", flush=True)

    # Print category distribution per split
    print("\n=== Category distribution ===", flush=True)
    for split in SPLITS:
        rows = [r for r in all_rows if r["split"] == split]
        cats = {}
        for r in rows:
            cats[r["category"]] = cats.get(r["category"], 0) + 1
        print(f"  {split}: {dict(sorted(cats.items()))} (total: {len(rows)})", flush=True)


if __name__ == "__main__":
    main()
