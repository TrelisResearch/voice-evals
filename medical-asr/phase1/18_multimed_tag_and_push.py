#!/usr/bin/env python3
"""
Phase 1D: Tag multimed-hard-100 with Gemini Flash medical density,
select top-50 by median_cer, push to ronanarraig/multimed-hard-public.
"""
import os, json, time
from concurrent.futures import ThreadPoolExecutor, as_completed
import pyarrow as pa
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')

from google import genai
from google.genai import types as gentypes
from datasets import load_dataset, Dataset, Audio

HF_TOKEN      = os.environ['HF_TOKEN']
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
client = genai.Client(api_key=GEMINI_API_KEY)

HF_INPUT   = 'ronanarraig/multimed-hard-100'
HF_OUTPUT  = 'ronanarraig/multimed-hard-public'
N_THREADS  = 20

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

# ── Load dataset ────────────────────────────────────────────────────
print(f'Loading {HF_INPUT}...')
ds = load_dataset(HF_INPUT, split='test', token=HF_TOKEN)
ds_plain = ds.cast_column('audio', Audio(decode=False)) if 'audio' in ds.column_names else ds
rows = list(ds_plain)
print(f'  {len(rows)} rows, cols: {ds_plain.column_names}')

# ── Run Flash tagging ───────────────────────────────────────────────
print(f'\nRunning Gemini Flash tagging ({N_THREADS} threads)...')
tag_results = {}
with ThreadPoolExecutor(max_workers=N_THREADS) as ex:
    futures = {ex.submit(tag_row, i, r['text']): i for i, r in enumerate(rows)}
    done = 0
    for fut in as_completed(futures):
        idx, tags = fut.result()
        tag_results[idx] = tags
        done += 1
        if done % 20 == 0:
            print(f'  {done}/{len(rows)} tagged', flush=True)

print(f'  Tagging complete: {len(tag_results)} rows')

# Merge tags into rows
tagged_rows = []
for i, r in enumerate(rows):
    tags = tag_results.get(i, {})
    tagged_rows.append({
        **r,
        'is_medical': tags.get('is_medical', False),
        'medical_density': tags.get('medical_density', 'none'),
        'entities': json.dumps(tags.get('entities', [])),
    })

# ── Density breakdown ────────────────────────────────────────────────
density_counts = {}
for r in tagged_rows:
    d = r['medical_density']
    density_counts[d] = density_counts.get(d, 0) + 1
print(f'\nMedical density breakdown: {density_counts}')
print(f'is_medical=True: {sum(1 for r in tagged_rows if r["is_medical"])}/{len(tagged_rows)}')

# ── Select top-50 ────────────────────────────────────────────────────
# Prefer high-density, then sort by median_cer descending
high_density = sorted(
    [r for r in tagged_rows if r.get('medical_density') == 'high'],
    key=lambda r: r['median_cer'], reverse=True
)
print(f'\nHigh-density rows: {len(high_density)}')

if len(high_density) >= 50:
    selected = high_density[:50]
    print(f'Using top-50 high-density rows')
elif len(high_density) >= 25:
    # Fill with medium density rows
    medium = sorted(
        [r for r in tagged_rows if r.get('medical_density') == 'medium'],
        key=lambda r: r['median_cer'], reverse=True
    )
    selected = high_density + medium[:50 - len(high_density)]
    print(f'Using {len(high_density)} high + {len(selected)-len(high_density)} medium density rows')
else:
    # Just take top-50 by CER regardless of density
    selected = sorted(tagged_rows, key=lambda r: r['median_cer'], reverse=True)[:50]
    print(f'Insufficient high/medium density rows, using top-50 by CER')

# Add difficulty_rank
for rank, r in enumerate(selected):
    r['difficulty_rank'] = rank + 1

print(f'Selected {len(selected)} rows, CER range: {selected[-1]["median_cer"]:.3f} – {selected[0]["median_cer"]:.3f}')
print('\nSample rows:')
for r in selected[:5]:
    print(f'  [{r["difficulty_rank"]}] cer={r["median_cer"]:.3f} density={r["medical_density"]} {r["text"][:70]}')

# ── Build and push dataset ────────────────────────────────────────────
print(f'\nBuilding dataset...')
audio_bytes_list = [(r.get('audio') or {}).get('bytes', b'') for r in selected]
texts      = [r['text'] for r in selected]
durations  = [float(r.get('duration', 0)) for r in selected]
med_cers   = [float(r['median_cer']) for r in selected]
diff_ranks = [int(r['difficulty_rank']) for r in selected]
entities   = [r.get('entities', '[]') for r in selected]
densities  = [r.get('medical_density', 'none') for r in selected]

audio_col = pa.array(audio_bytes_list, type=pa.binary())
table = pa.table({
    'audio':            audio_col,
    'text':             pa.array(texts),
    'duration':         pa.array(durations, type=pa.float32()),
    'median_cer':       pa.array(med_cers, type=pa.float32()),
    'difficulty_rank':  pa.array(diff_ranks, type=pa.int32()),
    'entities':         pa.array(entities),
    'medical_density':  pa.array(densities),
})
out_ds = Dataset(table).cast_column('audio', Audio(sampling_rate=16000))
print(f'  Dataset: {len(out_ds)} rows, features: {list(out_ds.features.keys())}')

print(f'\nPushing to {HF_OUTPUT}...')
out_ds.push_to_hub(HF_OUTPUT, split='test', private=True, token=HF_TOKEN)
print(f'Done! Pushed {len(out_ds)} rows to {HF_OUTPUT}')
print('DONE')
