#!/usr/bin/env python3
"""
Step 9: Entity extraction on difficulty-filtered hard-100 datasets.
- EKA: entities already exist (human-annotated) — just reformat
- MultiMed: dual-LLM extraction (Gemini 2.5 Flash + Claude Sonnet 4.6)
Also extract context templates (sentence with entity slots masked) from both.
Runs on eka-hard-100 and multimed-hard-100 (not full otsu sets) to save LLM costs.
"""
import os, json, re
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')

import pathlib
from datasets import load_dataset, Dataset, Audio
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

HF_TOKEN = os.environ['HF_TOKEN']
GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
OPENROUTER_API_KEY = os.environ['OPENROUTER_API_KEY']
TMP = pathlib.Path('medical-asr/phase1/tmp')

SYSTEM_PROMPT = """You are a medical named entity recognition system.
Extract medical entities from the given text and return them as a JSON array.

Entity categories:
- drugs: drug names, medications, brand names, generics
- procedures: medical procedures, tests, imaging, surgeries
- conditions: diagnoses, diseases, symptoms, syndromes
- anatomy: body parts, organs, anatomical structures
- organisations: hospitals, pharma companies, regulatory bodies

Rules:
- Only extract entities clearly present in the text
- For each entity: text (exact substring), category, char_start, char_end
- char_start/char_end must be exact character offsets in the original text
- Return empty array [] if no entities found
- Return ONLY the JSON array, no explanation"""

def find_offset(text, entity_text):
    idx = text.lower().find(entity_text.lower())
    if idx == -1:
        return None, None
    return idx, idx + len(entity_text)

def validate_offsets(text, entities):
    valid = []
    for e in entities:
        s, en = e.get('char_start', 0), e.get('char_end', 0)
        t = e.get('text', '')
        if 0 <= s < en <= len(text) and text[s:en].lower() == t.lower():
            valid.append(e)
        else:
            s2, e2 = find_offset(text, t)
            if s2 is not None:
                e_fixed = dict(e); e_fixed['char_start'] = s2; e_fixed['char_end'] = e2
                valid.append(e_fixed)
    return valid

def extract_templates(text, entities):
    """Extract sentence templates by replacing entity spans with slot tags."""
    if not entities:
        return []
    templates = []
    # Replace each entity with its category slot
    for e in sorted(entities, key=lambda x: x['char_start'], reverse=True):
        slot = f"[{e['category'].upper()}]"
        template = text[:e['char_start']] + slot + text[e['char_end']:]
        templates.append(template.strip())
    return templates

def call_gemini(text, idx):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_API_KEY)
    try:
        resp = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=f"{SYSTEM_PROMPT}\n\nText: {text}",
            config=types.GenerateContentConfig(temperature=0),
        )
        raw = resp.text.strip().lstrip('```json').lstrip('```').rstrip('```').strip()
        return ('gemini', idx, validate_offsets(text, json.loads(raw)), None)
    except Exception as ex:
        return ('gemini', idx, [], str(ex))

def call_claude(text, idx):
    from openai import OpenAI
    client = OpenAI(api_key=OPENROUTER_API_KEY, base_url='https://openrouter.ai/api/v1')
    try:
        resp = client.chat.completions.create(
            model='anthropic/claude-sonnet-4.6',
            messages=[{'role': 'system', 'content': SYSTEM_PROMPT},
                      {'role': 'user', 'content': f'Text: {text}'}],
            temperature=0, max_tokens=1024,
        )
        raw = resp.choices[0].message.content.strip().lstrip('```json').lstrip('```').rstrip('```').strip()
        return ('claude', idx, validate_offsets(text, json.loads(raw)), None)
    except Exception as ex:
        return ('claude', idx, [], str(ex))

def merge_entities(g_ents, c_ents):
    def key(e): return (e['text'].lower(), e['category'])
    g_map = {key(e): e for e in g_ents}
    c_map = {key(e): e for e in c_ents}
    merged = []
    for k in set(g_map) | set(c_map):
        e = dict(g_map.get(k) or c_map.get(k))
        e['agreement'] = 'both' if (k in g_map and k in c_map) else ('gemini_only' if k in g_map else 'claude_only')
        merged.append(e)
    return sorted(merged, key=lambda x: x.get('char_start', 0))

# ── EKA: reformat existing annotations ───────────────────────────
print("=== EKA: reformatting existing entity annotations ===")
eka_ds = load_dataset('ronanarraig/eka-hard-100', split='test', token=HF_TOKEN)
eka_ds = eka_ds.cast_column('audio', eka_ds.features['audio'].__class__(decode=False))

all_templates = []
eka_rows_out = []
for row in eka_ds:
    ents = json.loads(row['entities']) if row['entities'] else []
    templates = extract_templates(row['text'], ents)
    all_templates.extend(templates)
    row_dict = dict(row)
    row_dict['entities'] = json.dumps(ents)
    row_dict['context_templates'] = json.dumps(templates)
    eka_rows_out.append(row_dict)

print(f"  EKA: {len(eka_rows_out)} rows, {len(all_templates)} templates extracted")

# ── MultiMed: dual-LLM extraction ────────────────────────────────
print("\n=== MultiMed: dual-LLM entity extraction ===")
mm_ds = load_dataset('ronanarraig/multimed-hard-100', split='test', token=HF_TOKEN)
mm_ds = mm_ds.cast_column('audio', mm_ds.features['audio'].__class__(decode=False))

texts = [(i, row['text']) for i, row in enumerate(mm_ds)]
gemini_results = {}
claude_results = {}

with ThreadPoolExecutor(max_workers=10) as ex:
    futures = []
    for idx, text in texts:
        futures.append(ex.submit(call_gemini, text, idx))
        futures.append(ex.submit(call_claude, text, idx))
    for i, fut in enumerate(as_completed(futures)):
        model, idx, ents, err = fut.result()
        if err: print(f"  [{model}] row {idx} error: {err[:60]}")
        if model == 'gemini': gemini_results[idx] = ents
        else: claude_results[idx] = ents
        if (i+1) % 50 == 0: print(f"  {i+1}/{len(futures)} done")

mm_rows_out = []
mm_templates = []
agree_count = 0
for i, row in enumerate(mm_ds):
    g = gemini_results.get(i, [])
    c = claude_results.get(i, [])
    merged = merge_entities(g, c)
    agreed = [e for e in merged if e['agreement'] == 'both']
    agree_count += len(agreed)
    templates = extract_templates(row['text'], agreed)
    mm_templates.extend(templates)
    row_dict = dict(row)
    row_dict['entities'] = json.dumps([{k: v for k, v in e.items() if k != 'agreement'} for e in agreed])
    row_dict['context_templates'] = json.dumps(templates)
    mm_rows_out.append(row_dict)

all_templates.extend(mm_templates)
print(f"  MultiMed: {len(mm_rows_out)} rows, {agree_count} agreed entities, {len(mm_templates)} templates")

# Save combined context templates
with open(TMP / 'context_templates.json', 'w') as f:
    json.dump(list(set(all_templates)), f, indent=2)
print(f"\n  Total context templates saved: {len(set(all_templates))}")

# Push annotated datasets
print("\n=== Pushing annotated datasets ===")
for name, rows in [('eka-hard-100-annotated', eka_rows_out), ('multimed-hard-100-annotated', mm_rows_out)]:
    ds = Dataset.from_list(rows)
    if 'audio' in ds.column_names:
        ds = ds.cast_column('audio', Audio(sampling_rate=16000))
    ds.push_to_hub(f'ronanarraig/{name}', split='test', token=HF_TOKEN, private=True)
    print(f"  Pushed ronanarraig/{name} ({len(ds)} rows)")

print("\nStep 9 done.")
