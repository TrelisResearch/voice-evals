"""
Phase 2 Step 6: Generate harder replacement rows for easy phonetic and number_format rows.

Easy rows to replace (median CER < 0.05, excluding paralinguistics which are UTMOS-focused):
Phonetic (5 rows):
  - "The conduct of the Leicester conference..." (CER=0.0240)
  - "The present study will present evidence that Caoimhe..." (CER=0.0488)
  - "When they present the present to Aoife at Loughborough..." (CER=0.0154)
  - "Dr. Nguyen's Bildungsroman..." (CER=0.0250)
  - "Aoife needed a permit to film..." (CER=0.0086)

Number format (2 rows):
  - "Pope Benedict XVI resigned in 2013..." (CER=0.0000)
  - "The Apollo XI mission landed on July 20, 1969..." (CER=0.0215)
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

PHONETIC_PROMPT = """Generate 5 new short text snippets for a TTS benchmark (phonetic difficulty category).
These must be HARDER than the easy rows that TTS models already handle well.

Easy rows to replace (models pronounced these correctly, CER < 0.05):
- "The conduct of the Leicester conference will conduct attendees through Tadhg's analysis of schadenfreude in German literature."
- "The present study will present evidence that Caoimhe experienced schadenfreude watching her rival's failure in Loughborough."
- "When they present the present to Aoife at Loughborough, they'll record her record collection featuring Siobhán's Welsh compositions."
- "Dr. Nguyen's Bildungsroman explores how Tadhg must conduct himself with proper conduct while studying at Magdalen College."
- "Aoife needed a permit to film, but the Leicester council wouldn't permit shooting near the Ptolemy manuscript exhibit."

Requirements for NEW harder rows:
1. Each must include at least one PHONETICALLY CHALLENGING element that TTS models frequently mispronounce:
   - Celtic names with unusual phonology: Saoirse (SEER-sha), Niamh (NEEV), Caoilfhinn (KEELIN), Eithne (ETH-na), Oisín (UH-sheen), Diarmuid (DER-mid), Fionnuala (fi-NOO-la), Catrìona (kah-TREE-na), Pádraig (PAW-drig), Fearghal (FAR-gul)
   - Rare/unusual place names: Clachnacuddin, Auchterarder, Llangefni, Cnoc an Doire, Blaenavon, Knightsbridge (silent K?), Worcestershire, Loughborough
   - Technical/foreign loanwords with unintuitive pronunciations: quinoa, açaí, pho, chipotle, gnocchi, pommes dauphine, bourguignon, ptarmigan
   - Heteronyms (same spelling, different pronunciation based on context): "The wound wound around the pole", "The soldier decided to desert his desert dessert", "I object to this object"
   - Stress-pattern heteronyms: perfect (adj vs verb), rebel (n vs v), conflict (n vs v), increase (n vs v)

2. Must be NATURAL sentences that occur in real text (not a word list)
3. Length: 15–40 words each
4. Each sentence must be meaningfully different from the others
5. Include at least 2–3 phonetically challenging elements per sentence

Output exactly 5 sentences, one per line. No numbering, no explanation."""

NUMBER_FORMAT_PROMPT = """Generate 2 new short text snippets for a TTS benchmark (number_format difficulty category).
These must be HARDER than the easy rows that TTS models already handle well.

Easy rows to replace (models handled these correctly, CER < 0.03):
- "Pope Benedict XVI resigned in 2013, the first pope to do so since Gregory XII in 1415."
- "The Apollo XI mission landed on July 20, 1969, fulfilling Kennedy's goal set in the early 1960s."

Requirements for NEW harder rows that TTS models struggle with:
1. Must test AMBIGUOUS number reading patterns:
   - Mixed ordinal/cardinal: "the 3rd floor of building 3, room 303, at 3pm on 3/3"
   - Mixed Roman/Arabic: "Chapter XIV has 14 sections, each citing 14th century texts"
   - Uncommon ratio/proportion formats: "a 3:2 aspect ratio" vs "3 to 2 odds" ambiguity
   - Telephone numbers: "+353 (0)1 896 4321" or "+1-800-CALL-NOW"
   - Reference numbers with mixed alphanumerics: "case no. 23-CV-04567-DLF", "Flight BA172 departed at 17:45 from Terminal 2B"
   - Combined ordinal systems: "the XVIth arrondissement" or "Queen Elizabeth II's 70th jubilee"
   - Scientific notation with ambiguity: "a 10⁻⁴ M solution" vs "10 to the minus 4 molar"
   - Fraction + percentage combinations: "⅓ of the 33.3% voted, with a 0.1% margin"

2. Each sentence must be 20–40 words
3. Must be natural, realistic text (e.g. from a medical record, legal document, news article)
4. Include at least 2–3 different number format challenges per sentence

Output exactly 2 sentences, one per line. No numbering, no explanation."""

def call_llm(prompt: str) -> str:
    for model in [MODEL, FALLBACK]:
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=800,
                temperature=0.8,
            )
            content = resp.choices[0].message.content
            if content:
                return content.strip()
        except Exception as e:
            print(f"  {model} error: {e}", flush=True)
    raise ValueError("All models failed")

print("Generating 5 harder phonetic rows...", flush=True)
phonetic_text = call_llm(PHONETIC_PROMPT)
phonetic_rows = [l.strip() for l in phonetic_text.strip().split("\n") if l.strip()]
print(f"Got {len(phonetic_rows)} phonetic rows:", flush=True)
for r in phonetic_rows:
    print(f"  {r[:100]}", flush=True)

time.sleep(1)

print("\nGenerating 2 harder number_format rows...", flush=True)
number_text = call_llm(NUMBER_FORMAT_PROMPT)
number_rows = [l.strip() for l in number_text.strip().split("\n") if l.strip()]
print(f"Got {len(number_rows)} number_format rows:", flush=True)
for r in number_rows:
    print(f"  {r[:100]}", flush=True)

replacements = {
    "phonetic": phonetic_rows[:5],
    "number_format": number_rows[:2],
}

out_path = Path("tricky-tts/phase2/phase2_replacements.json")
out_path.write_text(json.dumps(replacements, indent=2))
print(f"\nSaved to {out_path}", flush=True)
