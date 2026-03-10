# Entity-Level CER Computation Guide

Guide for computing per-entity and per-category Character Error Rate (CER) from ASR transcriptions, using the entity annotations in the `ai-terms-*` datasets.

## Dataset Format

Each row in the dataset has:
- `text` — reference transcription (string)
- `audio` — audio file
- `entities` — JSON string containing entity annotations:

```json
[
  {"text": "MiniMax", "category": "companies", "char_start": 0, "char_end": 7},
  {"text": "M2.5", "category": "models", "char_start": 120, "char_end": 124}
]
```

Categories: `companies`, `models`, `products`, `benchmarks`, `people`, `technical`

## Algorithm

### Step 1: Get character-level alignment

Use `jiwer.process_characters()` to align reference and hypothesis at the character level:

```python
import jiwer

result = jiwer.process_characters(reference_text, hypothesis_text)
```

`result` contains:
- `result.references` — list of reference character sequences
- `result.hypotheses` — list of hypothesis character sequences
- `result.alignments` — list of alignment chunks, each a list of `(op, ref_start, ref_end, hyp_start, hyp_end)` tuples

Operations (`op`):
- `'equal'` — characters match
- `'substitute'` — characters differ
- `'delete'` — characters in reference missing from hypothesis
- `'insert'` — extra characters in hypothesis not in reference

### Step 2: Build a reference-position error map

From the alignment, build an array marking each reference character position as correct or errored:

```python
alignment = result.alignments[0]  # first (only) sentence pair

ref_len = len(reference_text)
ref_errors = [False] * ref_len  # True = this ref char has an error

for chunk in alignment:
    op = chunk.type      # 'equal', 'substitute', 'delete', 'insert'
    ref_s = chunk.ref_start_idx
    ref_e = chunk.ref_end_idx

    if op in ('substitute', 'delete'):
        for pos in range(ref_s, ref_e):
            ref_errors[pos] = True
    # 'insert' doesn't consume reference chars, but you may want to
    # attribute insertions to nearby entities — see note below.
```

### Step 3: Compute per-entity CER

For each entity, count errors within its character span:

```python
import json

entities = json.loads(row["entities"])

for entity in entities:
    span_start = entity["char_start"]
    span_end = entity["char_end"]
    span_len = span_end - span_start

    errors_in_span = sum(ref_errors[span_start:span_end])
    entity_cer = errors_in_span / span_len if span_len > 0 else 0.0

    entity["cer"] = entity_cer
    entity["n_errors"] = errors_in_span
    entity["n_chars"] = span_len
```

### Step 4: Aggregate by category

```python
from collections import defaultdict

category_stats = defaultdict(lambda: {"n_errors": 0, "n_chars": 0})

for entity in entities:
    cat = entity["category"]
    category_stats[cat]["n_errors"] += entity["n_errors"]
    category_stats[cat]["n_chars"] += entity["n_chars"]

for cat, stats in category_stats.items():
    stats["cer"] = stats["n_errors"] / stats["n_chars"] if stats["n_chars"] > 0 else 0.0
```

### Step 5: Compare entity CER to overall CER

```python
overall_cer = result.cer  # from jiwer

print(f"Overall CER: {overall_cer:.3f}")
for cat, stats in sorted(category_stats.items()):
    print(f"  {cat}: CER={stats['cer']:.3f} ({stats['n_errors']}/{stats['n_chars']} chars)")
```

## Handling Insertions

Insertions (extra characters in hypothesis not in reference) don't consume reference positions, so they won't appear in the `ref_errors` array above. Two options:

1. **Ignore insertions for entity CER** (simpler, recommended for v1) — entity CER only reflects substitutions and deletions within the entity span.
2. **Attribute insertions to nearest entity** — if an insertion occurs between `ref_end` of one alignment chunk and `ref_start` of the next, attribute it to the nearest entity span. This is more complex but captures cases where the model hallucinates extra characters within an entity name.

## Handling Multiple Occurrences

An entity like "Claude" may appear multiple times in a text. The dataset stores each occurrence as a separate entry with different `char_start`/`char_end`. Each occurrence gets its own CER score, which is correct — the model may transcribe the same entity correctly in one position but not another.

## Example: Full Pipeline

```python
import json
import jiwer
from collections import defaultdict
from datasets import load_dataset

ds = load_dataset("Trelis/ai-terms-public", split="test")

model_results = []  # assume you have ASR hypotheses for each row

for i, row in enumerate(ds):
    ref = row["text"]
    hyp = model_results[i]  # hypothesis from ASR model

    # Character alignment
    result = jiwer.process_characters(ref, hyp)
    alignment = result.alignments[0]

    # Build error map
    ref_errors = [False] * len(ref)
    for chunk in alignment:
        if chunk.type in ('substitute', 'delete'):
            for pos in range(chunk.ref_start_idx, chunk.ref_end_idx):
                ref_errors[pos] = True

    # Score entities
    entities = json.loads(row["entities"])
    for entity in entities:
        s, e = entity["char_start"], entity["char_end"]
        n_errors = sum(ref_errors[s:e])
        entity["cer"] = n_errors / (e - s) if e > s else 0.0

    # Aggregate
    cat_stats = defaultdict(lambda: {"errors": 0, "chars": 0})
    for ent in entities:
        cat_stats[ent["category"]]["errors"] += sum(ref_errors[ent["char_start"]:ent["char_end"]])
        cat_stats[ent["category"]]["chars"] += ent["char_end"] - ent["char_start"]

    print(f"\nRow {i}: overall CER={result.cer:.3f}")
    for cat, s in sorted(cat_stats.items()):
        cer = s["errors"] / s["chars"] if s["chars"] > 0 else 0
        print(f"  {cat}: {cer:.3f} ({s['errors']}/{s['chars']})")
```

## Output Format Suggestion

Per-evaluation results could include:

```json
{
  "overall_cer": 0.045,
  "entity_cer": 0.12,
  "category_cer": {
    "companies": 0.08,
    "models": 0.18,
    "products": 0.05,
    "benchmarks": 0.15,
    "people": 0.03,
    "technical": 0.22
  },
  "worst_entities": [
    {"text": "gated-delta", "category": "technical", "cer": 1.0, "occurrences": 2},
    {"text": "M2.5", "category": "models", "cer": 0.75, "occurrences": 1}
  ]
}
```

This helps identify which entity types (and specific entities) are hardest for each ASR model, enabling targeted fine-tuning.
