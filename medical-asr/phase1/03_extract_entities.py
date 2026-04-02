#!/usr/bin/env python3
"""
Step 3: Dual-LLM entity extraction for MultiMed EN and United-Syn-Med pilots.
Runs Gemini 2.5 Flash and Claude Sonnet in parallel, then compares outputs.
EKA already has annotations — skipped here.

Entity categories: drugs, procedures, conditions, anatomy, organisations
Output: merged annotations with agreement flag.
"""
import os, json, time
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')

import pathlib
from concurrent.futures import ThreadPoolExecutor, as_completed

GEMINI_API_KEY = os.environ['GEMINI_API_KEY']
OPENROUTER_API_KEY = os.environ['OPENROUTER_API_KEY']
OUT = pathlib.Path('medical-asr/phase1/tmp')

SYSTEM_PROMPT = """You are a medical named entity recognition system.
Extract medical entities from the given text and return them as a JSON array.

Entity categories:
- drugs: drug names, medications, brand names, generics (e.g. "metformin", "Lipitor", "azithromycin")
- procedures: medical procedures, tests, imaging, surgeries (e.g. "MRI", "appendectomy", "CBC")
- conditions: diagnoses, diseases, symptoms, syndromes (e.g. "type 2 diabetes", "pneumonia", "hypertension")
- anatomy: body parts, organs, anatomical structures (e.g. "left ventricle", "femur", "hippocampus")
- organisations: hospitals, pharma companies, regulatory bodies (e.g. "Mayo Clinic", "FDA", "Pfizer")

Rules:
- Only extract entities clearly present in the text
- For each entity include: text (exact substring), category, char_start, char_end
- char_start/char_end must be the exact character offsets in the original text
- Return empty array [] if no entities found
- Return ONLY the JSON array, no explanation

Example output:
[{"text": "metformin", "category": "drugs", "char_start": 12, "char_end": 21},
 {"text": "type 2 diabetes", "category": "conditions", "char_start": 45, "char_end": 60}]"""

def find_char_offsets(text, entity_text):
    """Find all occurrences of entity_text in text, return list of (start, end)."""
    offsets = []
    start = 0
    while True:
        idx = text.lower().find(entity_text.lower(), start)
        if idx == -1:
            break
        offsets.append((idx, idx + len(entity_text)))
        start = idx + 1
    return offsets

def validate_and_fix_offsets(text, entities):
    """Ensure char offsets match entity text; fix if possible."""
    valid = []
    for e in entities:
        t = e.get('text', '')
        s, en = e.get('char_start', 0), e.get('char_end', 0)
        # Check if offset is correct
        if 0 <= s < en <= len(text) and text[s:en].lower() == t.lower():
            valid.append(e)
        else:
            # Try to find correct offset
            offsets = find_char_offsets(text, t)
            if offsets:
                e2 = dict(e)
                e2['char_start'], e2['char_end'] = offsets[0]
                valid.append(e2)
            # else: drop entity (can't locate it)
    return valid

def call_gemini(text, row_idx):
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=GEMINI_API_KEY)
    prompt = f"{SYSTEM_PROMPT}\n\nText: {text}"
    try:
        resp = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0),
        )
        raw = resp.text.strip()
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
        entities = json.loads(raw)
        entities = validate_and_fix_offsets(text, entities)
        return ('gemini', row_idx, entities, None)
    except Exception as ex:
        return ('gemini', row_idx, [], str(ex))

def call_claude(text, row_idx):
    from openai import OpenAI
    client = OpenAI(
        api_key=OPENROUTER_API_KEY,
        base_url='https://openrouter.ai/api/v1'
    )
    try:
        resp = client.chat.completions.create(
            model='anthropic/claude-sonnet-4.6',
            messages=[
                {'role': 'system', 'content': SYSTEM_PROMPT},
                {'role': 'user', 'content': f'Text: {text}'}
            ],
            temperature=0,
            max_tokens=1024,
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith('```'):
            raw = raw.split('```')[1]
            if raw.startswith('json'):
                raw = raw[4:]
        entities = json.loads(raw)
        entities = validate_and_fix_offsets(text, entities)
        return ('claude', row_idx, entities, None)
    except Exception as ex:
        return ('claude', row_idx, [], str(ex))

def merge_annotations(gemini_ents, claude_ents, text):
    """Merge two entity lists. Tag each with agreement status."""
    def key(e):
        return (e['text'].lower(), e['category'])

    g_map = {key(e): e for e in gemini_ents}
    c_map = {key(e): e for e in claude_ents}
    all_keys = set(g_map) | set(c_map)

    merged = []
    for k in all_keys:
        in_g = k in g_map
        in_c = k in c_map
        e = g_map.get(k) or c_map.get(k)
        e = dict(e)
        if in_g and in_c:
            e['agreement'] = 'both'
        elif in_g:
            e['agreement'] = 'gemini_only'
        else:
            e['agreement'] = 'claude_only'
        merged.append(e)
    merged.sort(key=lambda x: x.get('char_start', 0))
    return merged

def process_dataset(texts_path, dataset_name):
    with open(texts_path) as f:
        texts = json.load(f)
    print(f"\n=== {dataset_name}: {len(texts)} rows ===")

    gemini_results = {}
    claude_results = {}

    tasks = []
    with ThreadPoolExecutor(max_workers=8) as ex:
        for item in texts:
            idx = item['idx']
            text = item['text']
            tasks.append(ex.submit(call_gemini, text, idx))
            tasks.append(ex.submit(call_claude, text, idx))

        for i, fut in enumerate(as_completed(tasks)):
            model, idx, ents, err = fut.result()
            if err:
                print(f"  [{model}] row {idx} ERROR: {err}")
            if model == 'gemini':
                gemini_results[idx] = ents
            else:
                claude_results[idx] = ents
            if (i+1) % 20 == 0:
                print(f"  {i+1}/{len(tasks)} done")

    # Merge
    merged_all = []
    agree_count = 0
    for item in texts:
        idx = item['idx']
        g = gemini_results.get(idx, [])
        c = claude_results.get(idx, [])
        merged = merge_annotations(g, c, item['text'])
        both = [e for e in merged if e['agreement'] == 'both']
        agree_count += len(both)
        merged_all.append({
            'idx': idx,
            'text': item.get('text', ''),
            'file_name': item.get('file_name', ''),
            'gemini_entities': g,
            'claude_entities': c,
            'merged_entities': merged,
            'entities': json.dumps([{k: v for k, v in e.items() if k != 'agreement'}
                                     for e in merged if e['agreement'] == 'both']),
        })

    total_ents = sum(len(r['merged_entities']) for r in merged_all)
    total_both = sum(len([e for e in r['merged_entities'] if e['agreement'] == 'both']) for r in merged_all)
    total_gemini_only = sum(len([e for e in r['merged_entities'] if e['agreement'] == 'gemini_only']) for r in merged_all)
    total_claude_only = sum(len([e for e in r['merged_entities'] if e['agreement'] == 'claude_only']) for r in merged_all)

    print(f"\nResults for {dataset_name}:")
    print(f"  Total unique entities: {total_ents}")
    print(f"  Both agree: {total_both} ({100*total_both/max(total_ents,1):.0f}%)")
    print(f"  Gemini only: {total_gemini_only}")
    print(f"  Claude only: {total_claude_only}")

    # Show sample disagreements
    print("\nSample disagreements:")
    shown = 0
    for r in merged_all:
        diffs = [e for e in r['merged_entities'] if e['agreement'] != 'both']
        if diffs and shown < 5:
            print(f"  Text: {r['text'][:80]}")
            for d in diffs[:3]:
                print(f"    [{d['agreement']}] {d['text']!r} ({d['category']})")
            shown += 1

    out_path = OUT / f'{dataset_name}_entities.json'
    with open(out_path, 'w') as f:
        json.dump(merged_all, f, indent=2)
    print(f"\nSaved to {out_path}")
    return merged_all

if __name__ == '__main__':
    process_dataset(OUT / 'multimed_pilot_texts.json', 'multimed')
    process_dataset(OUT / 'united_pilot_texts.json', 'united')
    print("\nStep 3 done.")
