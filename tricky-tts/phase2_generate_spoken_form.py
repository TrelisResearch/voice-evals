"""
Phase 2 Step 1: Generate spoken_form and cer_reliable columns for each row.

Uses OpenRouter LLM to convert written text to its canonical spoken form,
handling model paths, abbreviations, numbers, symbols, etc.
"""

import json, os, time
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

# Categories where CER comparison against written text is unreliable
CER_UNRELIABLE_CATS = {"edge_cases", "number_format", "ai_tech"}

SYSTEM_PROMPT = """You are a text-to-speech normalization expert. Given a text snippet, produce its canonical SPOKEN FORM — the way a professional narrator would read it aloud.

Rules:
- Expand abbreviations: "Dr." → "Doctor", "Ph.D." → "P H D", "F.A.C.C." → "F A C C", "Vol." → "Volume", "pp." → "pages", "No." → "Number", "Ann." → "Annual", "Conf." → "Conference", "Int'l" → "International"
- Numbers and units: "42%" → "forty-two percent", "20mg/dL" → "twenty milligrams per deciliter", "3.3V±5%" → "three point three volts plus or minus five percent", "−80°C" → "minus eighty degrees Celsius", "50μg/mL" → "fifty micrograms per milliliter", "2.5×10⁹" → "two point five times ten to the ninth"
- Currency: "£875,000.50" → "eight hundred seventy-five thousand pounds and fifty pence", "$1.25M" → "one point two five million dollars"
- Roman numerals: "Vol. XXIII" → "Volume twenty-three", "King Henry VIII" → "King Henry the Eighth", "IVth century" → "fourth century"
- Phone numbers: "+44 (0)20 7946 0958" → "plus forty-four zero twenty seven nine four six zero nine five eight", "0800-PRAYERS" → "zero eight hundred PRAYERS"
- ISBN/reference numbers: "ISBN: 978-0-12-345678-9" → "I S B N nine seven eight zero twelve three four five six seven eight nine"
- Account numbers: "GB29-NWBK-6016-1331-9268-19" → "G B twenty-nine N W B K six zero one six one three three one nine two six eight nineteen"
- Dates: "Feb. 14th" → "February fourteenth", "15/04/1989" → "the fifteenth of April nineteen eighty-nine"
- Times: "10:30am" → "ten thirty a m", "2:15pm" → "two fifteen p m"
- Standards codes: "MIL-STD-883" → "MIL standard eight eight three", "ASTM E2149-13a" → "A S T M E two one four nine dash thirteen a", "ISO 80601-2-61:2017" → "I S O eight zero six zero one dash two dash sixty-one colon twenty seventeen"
- CAS numbers: "CAS 50-99-7" → "C A S fifty ninety-nine seven"
- IUPAC names: read each part as written but pronounce chemical suffixes naturally
- AI model paths: org/model-name format — "meta-llama/Llama-3.1-405B-Instruct-hf" → "meta-llama slash Llama three point one four oh five B Instruct HF". The slash IS pronounced as "slash". Dashes in model names are usually silent (read as spaces). Dashes between org and model indicate the org name itself: "01-ai/Yi-1.5-34B-Chat-16K" → "zero one A I slash Yi one point five thirty-four B Chat sixteen K"
- Mixed case brands: "iPhone" → "iPhone", "ChatGPT" → "Chat G P T", "GPT-4o" → "G P T four oh"
- URLs: "www.st-aug.org.uk" → "www dot st dash aug dot org dot U K"
- Extensions: "x5847" → "extension five eight four seven"
- Special symbols: "≥" → "greater than or equal to", "≤" → "less than or equal to", "±" → "plus or minus", "→" → "to", "×" → "times", "⁰¹²³⁴⁵⁶⁷⁸⁹" → digit names, "⅘" → "four fifths"
- Punctuation that affects speech: em-dashes (—) become natural pauses, ellipses (...) become pauses, parentheses content is read naturally
- Repeated words: "the the the" stays as "the the the" (it's intentional)
- Interjections and informal speech: preserve as-is ("Ugh", "Hmm", "Eww", "waaah", "zzz", "reeeally")
- Ratios: "3:2" → "three to two", "blood pressure 90/60" → "ninety over sixty"
- Greek letters in context: maintain as "alpha", "beta", etc.

IMPORTANT: For AI model paths with slash notation, ALWAYS include "slash" when reading org/model names.

Output ONLY the spoken form text, nothing else. No explanation, no quotes."""

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
            return content.strip()
    raise ValueError(f"All models returned None for: {text[:60]}")

def main():
    data = json.loads(Path("tricky-tts/phase1d_final.json").read_text())
    output_path = Path("tricky-tts/phase2_data.json")

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

            # Save incrementally
            output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))

            # Rate limit: ~3 req/s
            time.sleep(0.5)

        except Exception as e:
            print(f"  ERROR: {e}", flush=True)
            time.sleep(2)

    print(f"\nDone. {len(results)} rows saved to {output_path}", flush=True)

    # Summary
    from collections import Counter
    cats = Counter(r["category"] for r in results)
    reliable = sum(1 for r in results if r["cer_reliable"])
    print(f"cer_reliable: {reliable}/{len(results)} rows", flush=True)
    print("Categories:", dict(cats), flush=True)

if __name__ == "__main__":
    main()
