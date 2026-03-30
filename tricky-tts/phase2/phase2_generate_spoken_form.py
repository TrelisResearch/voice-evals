"""
Phase 2 Step 1: Generate spoken_form and cer_reliable columns for each row.

Loads rewrite rules from spoken_form_rules.md as the system prompt.
Includes post-processing Unicode sanitisation pass.
"""

import json, os, re, time
from pathlib import Path

# Load env
env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from openai import OpenAI

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ["OPENROUTER_API_KEY"],
)

MODEL = "anthropic/claude-sonnet-4-5"
FALLBACK_MODEL = "google/gemini-2.5-flash"

# Load rules from markdown file
RULES_PATH = Path("tricky-tts/spoken_form_rules.md")
SYSTEM_PROMPT = RULES_PATH.read_text()

# Categories where CER comparison against written text is unreliable
CER_UNRELIABLE_CATS = {"edge_cases", "number_format", "ai_tech"}

# Unicode → ASCII sanitisation map (post-processing safety net)
UNICODE_SUBS = [
    ("×", " times "),
    ("±", " plus or minus "),
    ("≥", " greater than or equal to "),
    ("≤", " less than or equal to "),
    ("→", " to "),
    ("°C", " degrees Celsius"),
    ("°F", " degrees Fahrenheit"),
    ("μg", " micrograms"),
    ("μL", " microlitres"),
    ("μM", " micromolar"),
    ("μ", " micro"),
    ("°", " degrees"),
    ("⁻⁴", " to the minus four"),
    ("⁻³", " to the minus three"),
    ("⁻²", " to the minus two"),
    ("⁻¹", " to the minus one"),
    ("⁻", " minus "),
    ("⁰", " zero"),
    ("¹", " one"),
    ("²", " squared"),
    ("³", " cubed"),
    ("⁴", " to the fourth"),
    ("⁵", " to the fifth"),
    ("⁶", " to the sixth"),
    ("⁷", " to the seventh"),
    ("⁸", " to the eighth"),
    ("⁹", " to the ninth"),
    ("½", " one half"),
    ("⅓", " one third"),
    ("⅔", " two thirds"),
    ("¼", " one quarter"),
    ("¾", " three quarters"),
    ("⅘", " four fifths"),
    ("£", " pounds "),
    ("€", " euros "),
    ("—", ", "),
    ("–", " to "),
    ("…", "..."),
    ("\u2019", "'"),  # right single quote
    ("\u2018", "'"),  # left single quote
    ("\u201c", '"'),  # left double quote
    ("\u201d", '"'),  # right double quote
]

def sanitise_unicode(text: str) -> str:
    """Replace any remaining Unicode with ASCII equivalents."""
    for src, dst in UNICODE_SUBS:
        text = text.replace(src, dst)
    # Collapse multiple spaces
    text = re.sub(r" {2,}", " ", text).strip()
    # Warn if non-ASCII remains
    non_ascii = [c for c in text if ord(c) > 127]
    if non_ascii:
        print(f"    ⚠ Non-ASCII remains after sanitise: {set(non_ascii)}", flush=True)
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
            spoken = content.strip()
            spoken = sanitise_unicode(spoken)
            return spoken
    raise ValueError(f"All models returned None for: {text[:60]}")

def main():
    data = json.loads(Path("tricky-tts/phase1/phase1d_final.json").read_text())
    output_path = Path("tricky-tts/phase2/phase2_data.json")

    # Resume from partial output if it exists
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        done_texts = {r["text"] for r in existing}
        print(f"Resuming: {len(existing)}/{len(data)} already done", flush=True)
    else:
        existing = []
        done_texts = set()

    results = list(existing)

    for i, row in enumerate(data):
        if row["text"] in done_texts:
            continue

        cat = row["category"]
        cer_reliable = cat not in CER_UNRELIABLE_CATS

        print(f"[{i+1}/{len(data)}] [{cat}] {row['text'][:60]}...", flush=True)

        try:
            spoken = generate_spoken_form(row["text"], cat)
            print(f"  → {spoken[:80]}", flush=True)

            results.append({
                "text": row["text"],
                "category": cat,
                "spoken_form": spoken,
                "cer_reliable": cer_reliable,
            })

            output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
            time.sleep(0.5)

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            time.sleep(2)

    print(f"\nDone. {len(results)} rows saved to {output_path}", flush=True)

    from collections import Counter
    cats = Counter(r["category"] for r in results)
    reliable = sum(1 for r in results if r["cer_reliable"])
    print(f"cer_reliable: {reliable}/{len(results)} rows", flush=True)
    print("Categories:", dict(cats), flush=True)

if __name__ == "__main__":
    main()
