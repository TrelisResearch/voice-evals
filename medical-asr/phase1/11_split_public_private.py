#!/usr/bin/env python3
"""
Step 11: Split eka-hard-100-annotated and multimed-hard-100-annotated
into 50 public + 50 private with entity deduplication.
Entity dedup: no entity text appears in both public and private splits.
Rank: hardest 50 → public (will also get proprietary evals), next 50 → private (open-source only).
"""
import os, json, random
from dotenv import load_dotenv
load_dotenv('/home/claude/TR/.env')

import pathlib
from datasets import load_dataset, Dataset, Audio
from huggingface_hub import HfApi
from collections import defaultdict

HF_TOKEN = os.environ['HF_TOKEN']
TMP = pathlib.Path('medical-asr/phase1/tmp')
api = HfApi(token=HF_TOKEN)
random.seed(42)

def get_entity_texts(row):
    try:
        ents = json.loads(row['entities']) if row['entities'] else []
        return {e['text'].lower() for e in ents if isinstance(e, dict) and 'text' in e}
    except Exception:
        return set()

def split_with_entity_dedup(rows):
    """
    Split 100 rows into 50 public + 50 private.
    Rows are pre-sorted hardest-first (by median_cer descending).
    Assign rows greedily: try to put in public first, then private,
    skipping if it would add an entity already seen in the other split.
    Any unassigned rows fill remaining slots ignoring entity constraint.
    """
    public, private = [], []
    public_entities, private_entities = set(), set()

    for row in rows:
        ents = get_entity_texts(row)
        in_public = len(public) < 50
        in_private = len(private) < 50

        if not in_public and not in_private:
            break

        # Check overlap
        overlaps_public = bool(ents & public_entities)
        overlaps_private = bool(ents & private_entities)

        if in_public and not overlaps_public:
            public.append(row)
            public_entities |= ents
        elif in_private and not overlaps_private:
            private.append(row)
            private_entities |= ents
        elif in_public:
            # Accept overlap rather than leave slot empty
            public.append(row)
            public_entities |= ents
        else:
            private.append(row)
            private_entities |= ents

    # Fill any remaining slots (shouldn't happen with 100 rows → 50+50)
    assigned = set(id(r) for r in public + private)
    remaining = [r for r in rows if id(r) not in assigned]
    for row in remaining:
        if len(public) < 50:
            public.append(row)
        elif len(private) < 50:
            private.append(row)

    overlap = public_entities & private_entities
    return public, private, overlap

for source in ['eka', 'multimed']:
    print(f"\n=== {source} ===")
    ds = load_dataset(f'ronanarraig/{source}-hard-100-annotated', split='test', token=HF_TOKEN)
    ds = ds.cast_column('audio', ds.features['audio'].__class__(decode=False))

    # Sort hardest first by median_cer (already sorted, but be explicit)
    rows = sorted(list(ds), key=lambda r: float(r.get('median_cer', 0)), reverse=True)

    public_rows, private_rows, overlap = split_with_entity_dedup(rows)

    print(f"  Public: {len(public_rows)} rows")
    print(f"  Private: {len(private_rows)} rows")
    print(f"  Entity overlap between splits: {len(overlap)} entities")
    if overlap:
        print(f"    Overlapping: {sorted(overlap)[:10]}")

    # CER ranges
    pub_cers = [float(r.get('median_cer', 0)) for r in public_rows]
    priv_cers = [float(r.get('median_cer', 0)) for r in private_rows]
    print(f"  Public median CER range: {min(pub_cers):.3f} – {max(pub_cers):.3f}")
    print(f"  Private median CER range: {min(priv_cers):.3f} – {max(priv_cers):.3f}")

    for split_name, split_rows in [('public', public_rows), ('private', private_rows)]:
        repo_id = f'ronanarraig/{source}-hard-{split_name}'
        split_ds = Dataset.from_list(split_rows)
        if 'audio' in split_ds.column_names:
            split_ds = split_ds.cast_column('audio', Audio(sampling_rate=16000))
        api.create_repo(repo_id=repo_id, repo_type='dataset', private=True, exist_ok=True)
        split_ds.push_to_hub(repo_id, split='test', token=HF_TOKEN, private=True)
        print(f"  Pushed {repo_id} ({len(split_ds)} rows)")

print("\nStep 11 done.")
