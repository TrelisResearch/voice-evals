#!/usr/bin/env python3
"""
Step 9b: Entity extraction on eka-sentences-hard-100 (77 rows).
EKA entities already exist (human-annotated) — reformat + extract templates.
Then split into public/private (up to 50 each, with entity dedup).
"""
import os, json
import numpy as np
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')

import pathlib
from datasets import load_dataset, Dataset, Audio
from huggingface_hub import HfApi

HF_TOKEN = os.environ['HF_TOKEN']
TMP = pathlib.Path('medical-asr/phase1/tmp')
api = HfApi(token=HF_TOKEN)

def extract_templates(text, entities):
    templates = []
    for e in sorted(entities, key=lambda x: x['char_start'], reverse=True):
        slot = f"[{e['category'].upper()}]"
        template = text[:e['char_start']] + slot + text[e['char_end']:]
        templates.append(template.strip())
    return templates

def get_entity_texts(row):
    try:
        ents = json.loads(row['entities']) if row['entities'] else []
        return {e['text'].lower() for e in ents if isinstance(e, dict) and 'text' in e}
    except Exception:
        return set()

# ── Load + annotate ───────────────────────────────────────────────
print("Loading eka-sentences-hard-100...")
ds = load_dataset('ronanarraig/eka-sentences-hard-100', split='test', token=HF_TOKEN)
ds = ds.cast_column('audio', ds.features['audio'].__class__(decode=False))
print(f"  {len(ds)} rows")

rows_out = []
all_templates = []
for row in ds:
    ents = json.loads(row['entities']) if row['entities'] else []
    templates = extract_templates(row['text'], ents)
    all_templates.extend(templates)
    d = dict(row)
    d['entities'] = json.dumps(ents)
    d['context_templates'] = json.dumps(templates)
    rows_out.append(d)

print(f"  {len(all_templates)} context templates extracted")

# Push annotated
ann_ds = Dataset.from_list(rows_out)
if 'audio' in ann_ds.column_names:
    ann_ds = ann_ds.cast_column('audio', Audio(sampling_rate=16000))
api.create_repo('ronanarraig/eka-sentences-hard-annotated', repo_type='dataset', private=True, exist_ok=True)
ann_ds.push_to_hub('ronanarraig/eka-sentences-hard-annotated', split='test', token=HF_TOKEN, private=True)
print(f"  Pushed ronanarraig/eka-sentences-hard-annotated ({len(ann_ds)} rows)")

# ── Split public/private ──────────────────────────────────────────
print("\nSplitting public/private...")
rows_sorted = sorted(rows_out, key=lambda r: float(r.get('median_cer', 0)), reverse=True)
n_total = len(rows_sorted)
n_public = min(50, n_total // 2 + n_total % 2)  # slightly more to public if odd
n_private = n_total - n_public
print(f"  Total: {n_total} → public: {n_public}, private: {n_private}")

public_rows, private_rows = [], []
public_ents, private_ents = set(), set()

for row in rows_sorted:
    ents = get_entity_texts(row)
    if len(public_rows) < n_public and not (ents & public_ents):
        public_rows.append(row); public_ents |= ents
    elif len(private_rows) < n_private and not (ents & private_ents):
        private_rows.append(row); private_ents |= ents
    elif len(public_rows) < n_public:
        public_rows.append(row); public_ents |= ents
    elif len(private_rows) < n_private:
        private_rows.append(row); private_ents |= ents

overlap = public_ents & private_ents
print(f"  Entity overlap: {len(overlap)} entities")
pub_cers = [float(r.get('median_cer', 0)) for r in public_rows]
priv_cers = [float(r.get('median_cer', 0)) for r in private_rows]
print(f"  Public CER range: {min(pub_cers):.3f} – {max(pub_cers):.3f}")
print(f"  Private CER range: {min(priv_cers):.3f} – {max(priv_cers):.3f}")

for split_name, split_rows in [('public', public_rows), ('private', private_rows)]:
    repo_id = f'ronanarraig/eka-sentences-hard-{split_name}'
    split_ds = Dataset.from_list(split_rows)
    if 'audio' in split_ds.column_names:
        split_ds = split_ds.cast_column('audio', Audio(sampling_rate=16000))
    api.create_repo(repo_id=repo_id, repo_type='dataset', private=True, exist_ok=True)
    split_ds.push_to_hub(repo_id, split='test', token=HF_TOKEN, private=True)
    print(f"  Pushed {repo_id} ({len(split_ds)} rows)")

# Show sample rows
print("\nSample hard rows (top 10 by median CER):")
print(f"{'CER':>6}  TEXT")
print("-" * 90)
for r in rows_sorted[:10]:
    ents = [e['text'] for e in json.loads(r['entities'] or '[]')]
    print(f"  {r.get('median_cer',0):.3f}  {r['text'][:75]}")
    if ents:
        print(f"         Entities: {', '.join(ents[:5])}")

print("\nDone.")
