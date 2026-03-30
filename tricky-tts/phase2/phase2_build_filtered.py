"""
Phase 2 Step 7: Build filtered dataset by:
1. Removing easy rows from phonetic and number_format categories
2. Adding harder replacement rows
3. Generating spoken_form + cer_reliable for new rows
4. Pushing final filtered dataset to HF

Paralinguistics rows kept as-is (UTMOS-focused, CER naturally low).
"""

import os, json, time
from pathlib import Path

env_path = Path("/home/claude/TR/.env")
for line in env_path.read_text().splitlines():
    line = line.strip()
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())

from openai import OpenAI
client = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=os.environ["OPENROUTER_API_KEY"])

MODEL = "anthropic/claude-sonnet-4-5"
FALLBACK = "google/gemini-2.5-flash"

# Load all data
phase2_data = json.loads(Path("tricky-tts/phase2/phase2_data.json").read_text())
perrow = json.loads(Path("tricky-tts/phase2/phase2_perrow_results.json").read_text())
replacements = json.loads(Path("tricky-tts/phase2/phase2_replacements.json").read_text())

# Build set of easy rows to drop (excluding paralinguistics)
EASY_THRESHOLD = 0.05
easy_texts = set()
for row in perrow:
    if row["category"] == "paralinguistics":
        continue  # Keep: UTMOS-focused
    if row["median_cer"] is not None and row["median_cer"] < EASY_THRESHOLD:
        easy_texts.add(row["text"])

print(f"Dropping {len(easy_texts)} easy rows:", flush=True)
for t in sorted(easy_texts):
    print(f"  {t[:80]}", flush=True)

# Build base filtered dataset (keep non-easy rows)
filtered = [r for r in phase2_data if r["text"] not in easy_texts]
print(f"\nKept {len(filtered)} rows (from {len(phase2_data)})", flush=True)

from collections import Counter
cats = Counter(r["category"] for r in filtered)
print("Category counts after dropping:", dict(cats), flush=True)

# Spoken form generation
SYSTEM_PROMPT = """You are a text-to-speech normalization expert. Given a text snippet, produce its canonical SPOKEN FORM — the way a professional narrator would read it aloud.

Rules:
- Expand abbreviations: "Dr." → "Doctor", "Ph.D." → "P H D", "F.A.C.C." → "F A C C", "Vol." → "Volume", "pp." → "pages", "No." → "Number"
- Numbers to words: "42%" → "forty-two percent", "20mg" → "twenty milligrams", "10⁻³" → "ten to the minus third"
- Currency: "£875,000" → "eight hundred seventy-five thousand pounds", "$1.25M" → "one point two five million dollars"
- Roman numerals: "XVI" → "sixteenth", "XI" → "eleven", "XXI" → "twenty-first"
- Phone numbers spelled out digit by digit
- Mixed alphanumeric codes: "4B" → "four B", "BA2107" → "B A two one zero seven", "21C" → "twenty-one C"
- Flight gates, room numbers: natural spoken form
- Ratios: "3:2" → "three to two", "10⁻³" → "ten to the minus three"
- Special symbols: "°C" → "degrees Celsius", "±" → "plus or minus", "→" → "to", "⁻" → "minus"
- AI model paths: org/model notation — slash is pronounced "slash"
- Heteronyms: preserve as written (pronunciation depends on context)
- Interjections (Ugh, Hmm, Eww): preserve as-is
Output ONLY the spoken form text, nothing else."""

def generate_spoken_form(text: str, category: str) -> str:
    for model in [MODEL, FALLBACK]:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": f"Category: {category}\nText: {text}"}
                ],
                max_tokens=800,
                temperature=0.1,
            )
            content = resp.choices[0].message.content
            if content:
                return content.strip()
        except Exception as e:
            print(f"  {model} error: {e}", flush=True)
    raise ValueError(f"All models failed for: {text[:60]}")

CER_UNRELIABLE = {"edge_cases", "number_format", "ai_tech"}

# Generate spoken forms for replacement rows
print("\nGenerating spoken forms for replacement rows...", flush=True)
new_rows = []

for category, texts in replacements.items():
    cer_reliable = category not in CER_UNRELIABLE
    for text in texts:
        print(f"  [{category}] {text[:70]}...", flush=True)
        spoken = generate_spoken_form(text, category)
        print(f"    → {spoken[:80]}", flush=True)
        new_rows.append({
            "text": text,
            "category": category,
            "spoken_form": spoken,
            "cer_reliable": cer_reliable,
        })
        time.sleep(0.3)

# Combine
final_dataset = filtered + new_rows
print(f"\nFinal dataset: {len(final_dataset)} rows", flush=True)
cats_final = Counter(r["category"] for r in final_dataset)
print("Category breakdown:", dict(cats_final), flush=True)

# Save
out_path = Path("tricky-tts/phase2/phase2_final.json")
out_path.write_text(json.dumps(final_dataset, indent=2, ensure_ascii=False))
print(f"Saved to {out_path}", flush=True)

# Push to HF
from datasets import Dataset
from huggingface_hub import HfApi

HF_TOKEN = os.environ["HF_TOKEN"]
repo_id = "ronanarraig/tricky-tts-v2-public"
api = HfApi(token=HF_TOKEN)
api.create_repo(repo_id=repo_id, repo_type="dataset", private=True, exist_ok=True)

ds = Dataset.from_list(final_dataset)
print(f"\nPushing {len(final_dataset)} rows to {repo_id}...", flush=True)
ds.push_to_hub(repo_id, token=HF_TOKEN, private=True, split="train")
print("Done.", flush=True)
