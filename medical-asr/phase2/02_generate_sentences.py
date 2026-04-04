#!/usr/bin/env python3
"""
Phase 2 Step 2: Generate clinical sentences from (keyword, context) pairs.

For each pair, Claude Sonnet generates a fresh clinical sentence that:
- Features the keyword naturally in a clinical context
- Uses the mined context as a style anchor
- Varies register: prescription note, discharge summary, radiology report,
  referral letter, clinical trial description, nursing note
- Contains 1-2 entities, 15-35 words, ≤25s when spoken

Output: phase2/tmp/sentences.json
"""
import os, json, time, random
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

OPENROUTER_API_KEY = os.environ['OPENROUTER_API_KEY']
PAIRS_FILE = 'medical-asr/phase2/tmp/pairs.json'
OUT = 'medical-asr/phase2/tmp/sentences.json'

client = OpenAI(api_key=OPENROUTER_API_KEY, base_url='https://openrouter.ai/api/v1')

REGISTERS = [
    'prescription note',
    'discharge summary',
    'radiology report',
    'referral letter',
    'clinical trial description',
    'nursing note',
    'surgical operative report',
    'clinical consultation note',
]

SYSTEM = """You are a medical language expert. Generate a single realistic clinical sentence featuring a specific medical keyword.

Rules:
- The sentence must be 15-35 words long
- Use the keyword exactly as given (same spelling and capitalisation)
- Write in a real clinical register (not lay language, not a definition)
- The sentence should feel like it belongs in an actual clinical document
- Include 1-2 medical entities total (the keyword + at most 1 other)
- Do NOT start with the keyword — embed it naturally mid-sentence
- Do NOT use abbreviations that need decoding (e.g. write "twice daily" not "b.i.d.")
- Return ONLY the sentence, nothing else"""

def generate_sentence(pair, register, idx):
    keyword = pair['keyword']
    context = pair['context']
    cat = pair['category']

    prompt = f"""Generate a clinical sentence for a {register} that naturally includes the keyword: "{keyword}"

Medical category: {cat}
Style reference (from real clinical literature — use as tone/register inspiration, DO NOT copy):
"{context[:200]}"

Return only the sentence."""

    try:
        resp = client.chat.completions.create(
            model='anthropic/claude-sonnet-4-6',
            messages=[
                {'role': 'system', 'content': SYSTEM},
                {'role': 'user', 'content': prompt},
            ],
            temperature=0.7,
            max_tokens=200,
        )
        sentence = resp.choices[0].message.content.strip().strip('"')
        # Basic validation
        words = sentence.split()
        if len(words) < 10 or len(words) > 50:
            return None
        if keyword.lower() not in sentence.lower():
            return None
        return sentence
    except Exception as e:
        print(f'  Error generating sentence for {keyword}: {e}')
        return None

def process_pair(item):
    idx, pair = item
    register = REGISTERS[idx % len(REGISTERS)]
    sentence = generate_sentence(pair, register, idx)
    if sentence:
        return {
            'idx': idx,
            'keyword': pair['keyword'],
            'category': pair['category'],
            'context': pair['context'],
            'source': pair['source'],
            'register': register,
            'text': sentence,
            'tts_text': sentence,  # will be updated with spoken form if needed
        }
    return None

pairs = json.load(open(PAIRS_FILE))
print(f'Loaded {len(pairs)} pairs')

results = []
failed = []

with ThreadPoolExecutor(max_workers=10) as ex:
    futures = {ex.submit(process_pair, (i, p)): i for i, p in enumerate(pairs)}
    for i, fut in enumerate(as_completed(futures)):
        r = fut.result()
        if r:
            results.append(r)
        else:
            failed.append(futures[fut])
        if (i+1) % 20 == 0:
            print(f'  {i+1}/{len(pairs)} done, {len(results)} generated', flush=True)

print(f'\nGenerated: {len(results)}, failed: {len(failed)}')

# Category breakdown
from collections import Counter
cats = Counter(r['category'] for r in results)
for cat, n in sorted(cats.items()):
    print(f'  {cat}: {n}')

# Sample output
print('\nSample sentences:')
for r in results[:5]:
    print(f'  [{r["category"]}] {r["text"]}')

json.dump(results, open(OUT, 'w'), indent=2)
print(f'\nSaved {len(results)} sentences to {OUT}')
