#!/usr/bin/env python3
"""
Step 13: Filter multimed-sentences-otsu to CER 0.05-0.20, then tag with Gemini Flash:
  - is_medical (bool)
  - entities (JSON list: text, category, char_start, char_end)
  - medical_density (low/medium/high)
Push tagged dataset, then push medical-only subset.
"""
import os, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')

from google import genai
from google.genai import types as gentypes
from datasets import load_dataset, Dataset, Audio
from huggingface_hub import HfApi

HF_TOKEN = os.environ['HF_TOKEN']
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
client = genai.Client(api_key=GEMINI_API_KEY)

api = HfApi(token=HF_TOKEN)

CER_CEILING = 0.20
N_THREADS = 20

SYSTEM_PROMPT = """You are a medical NLP expert. Given a sentence of transcribed speech, return a JSON object with:
- "is_medical": true if the sentence is substantively about medicine, health, biology, or clinical topics; false otherwise
- "medical_density": "high" (dense clinical/scientific jargon throughout), "medium" (some medical terms or general health discussion), or "low" (barely medical, mostly general language) — only relevant if is_medical is true, else use "none"
- "entities": list of medical named entities found. Each entity: {"text": str, "category": str, "char_start": int, "char_end": int}
  Categories: drug, condition, procedure, anatomy, organisation, measurement
  Only include clearly medical named entities. char_start/char_end are character offsets into the input text.

Return ONLY valid JSON, no explanation."""

def tag_row(idx, text):
    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model='gemini-2.5-flash',
                contents=f"{SYSTEM_PROMPT}\n\nText: {text}",
                config=gentypes.GenerateContentConfig(
                    temperature=0.1,
                    response_mime_type="application/json",
                ),
            )
            raw = response.text.strip()
            result = json.loads(raw)
            return idx, result
        except Exception as e:
            if attempt == 2:
                print(f"  Row {idx} failed after 3 attempts: {e}")
                return idx, {"is_medical": False, "medical_density": "none", "entities": []}
            time.sleep(2 ** attempt)

# ── Load + filter ─────────────────────────────────────────────────
print("Loading multimed-sentences-otsu...")
ds = load_dataset('ronanarraig/multimed-sentences-otsu', split='test', token=HF_TOKEN)
ds = ds.cast_column('audio', ds.features['audio'].__class__(decode=False))
print(f"  {len(ds)} rows total")

rows = [dict(r) for r in ds if r['gemini_cer'] <= CER_CEILING]
print(f"  {len(rows)} rows with CER <= {CER_CEILING}")

# ── Tag with Gemini Flash ─────────────────────────────────────────
print(f"\nTagging {len(rows)} rows with Gemini Flash ({N_THREADS} threads)...")
results = [None] * len(rows)
with ThreadPoolExecutor(max_workers=N_THREADS) as executor:
    futures = {executor.submit(tag_row, i, row['text']): i for i, row in enumerate(rows)}
    done = 0
    for future in as_completed(futures):
        idx, result = future.result()
        results[idx] = result
        done += 1
        if done % 20 == 0 or done == len(rows):
            print(f"  {done}/{len(rows)} tagged")

# ── Merge tags into rows ──────────────────────────────────────────
tagged = []
for row, tags in zip(rows, results):
    d = dict(row)
    d['is_medical'] = tags.get('is_medical', False)
    d['medical_density'] = tags.get('medical_density', 'none')
    d['entities'] = json.dumps(tags.get('entities', []))
    tagged.append(d)

n_medical = sum(1 for r in tagged if r['is_medical'])
density_counts = {}
for r in tagged:
    density_counts[r['medical_density']] = density_counts.get(r['medical_density'], 0) + 1

print(f"\nTagging results:")
print(f"  is_medical=True: {n_medical}/{len(tagged)}")
print(f"  medical_density: {density_counts}")

n_with_entities = sum(1 for r in tagged if json.loads(r['entities'] or '[]'))
print(f"  rows with entities: {n_with_entities}")

# ── Push full tagged dataset ──────────────────────────────────────
tagged_ds = Dataset.from_list(tagged)
if 'audio' in tagged_ds.column_names:
    tagged_ds = tagged_ds.cast_column('audio', Audio(sampling_rate=16000))
api.create_repo('ronanarraig/multimed-sentences-tagged', repo_type='dataset', private=True, exist_ok=True)
tagged_ds.push_to_hub('ronanarraig/multimed-sentences-tagged', split='test', token=HF_TOKEN, private=True)
print(f"\nPushed ronanarraig/multimed-sentences-tagged ({len(tagged_ds)} rows)")

# ── Push medical-only subset ──────────────────────────────────────
medical = [r for r in tagged if r['is_medical']]
medical_ds = Dataset.from_list(medical)
if 'audio' in medical_ds.column_names:
    medical_ds = medical_ds.cast_column('audio', Audio(sampling_rate=16000))
api.create_repo('ronanarraig/multimed-sentences-medical', repo_type='dataset', private=True, exist_ok=True)
medical_ds.push_to_hub('ronanarraig/multimed-sentences-medical', split='test', token=HF_TOKEN, private=True)
print(f"Pushed ronanarraig/multimed-sentences-medical ({len(medical_ds)} rows)")

# ── Sample output ─────────────────────────────────────────────────
print("\nSample medical rows:")
print(f"{'CER':>6}  {'DENSITY':8}  TEXT")
print("-" * 90)
for r in sorted(medical, key=lambda x: x['gemini_cer'], reverse=True)[:15]:
    ents = [e['text'] for e in json.loads(r['entities'] or '[]')]
    print(f"  {r['gemini_cer']:.3f}  {r['medical_density']:8}  {r['text'][:65]}")
    if ents:
        print(f"         Entities: {', '.join(ents[:5])}")

print("\nNon-medical rows (sample):")
non_med = [r for r in tagged if not r['is_medical']]
for r in non_med[:5]:
    print(f"  {r['gemini_cer']:.3f}  {r['text'][:80]}")

print("\nDone.")
