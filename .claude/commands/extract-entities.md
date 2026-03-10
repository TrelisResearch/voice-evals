# Entity Extraction for Voice Eval Datasets

Extract named entities from a HuggingFace dataset, compute character offsets, analyze split overlap, and push annotated splits.

## Arguments
- `$ARGUMENTS` — HuggingFace dataset ID (e.g. `Trelis/ai-terms`). If empty, prompt the user.

## Workflow

### 1. Extract entities with LLM

Run `tmp/llm_entity_extraction.py` against the dataset. This uses Claude Haiku to extract entities in these categories: companies, models, products, benchmarks, people, technical.

```
uv run tmp/llm_entity_extraction.py
```

The script:
- Loads the dataset from HuggingFace (requires `HF_TOKEN` in `.env`)
- Calls Claude Haiku for each row (requires `ANTHROPIC_API_KEY` in `.env`)
- Saves results to `tmp/entities_by_row.json`

If the dataset ID differs from `Trelis/ai-terms`, update the `load_dataset()` call in the script before running.

Review the output: check that entities look reasonable, no JSON parse failures, good coverage across categories.

### 2. Analyze split overlap

Run `tmp/split_analysis.py` to compute entity Jaccard similarity and n-gram overlap between candidate splits.

```
uv run tmp/split_analysis.py
```

**Key thresholds:**
- Entity Jaccard similarity between any two splits should be **< 0.20** (on distinctive entities, excluding ubiquitous terms like "OpenAI", "Anthropic", "Claude")
- Trigram Jaccard should be **< 0.01**

If overlap is too high, reassign source files between splits in the `splits` config within the script.

### 3. Build splits with entity annotations

Run `tmp/build_splits.py` to create balanced splits with character-offset entity annotations.

```
uv run tmp/build_splits.py
```

This:
- Assigns rows to splits based on source file
- Balances splits to ~12 rows each (drops shortest rows if needed)
- Computes character offsets for each entity occurrence in the text
- Saves splits locally to `tmp/splits/{public,semi_private,private}`

**Verify:** Check that entity char offsets match the actual text (the script prints alignment checks). Fix any MISMATCH entries.

### 4. Clean and push to HuggingFace

Run `tmp/push_splits.py` to deduplicate entities, remove generic terms, fix categories, and push.

```
uv run tmp/push_splits.py
```

This pushes to `Trelis/ai-terms-{public,semi-private,private}` as private datasets with `split="test"`.

**Before pushing:** Review the entity cleanup config in the script:
- `GENERIC_TERMS` — terms to filter out (not useful for entity CER)
- `CATEGORY_FIXES` — manual category corrections

### 5. Verify pushed datasets

After pushing, spot-check a few rows:
```python
from datasets import load_dataset
ds = load_dataset("Trelis/ai-terms-public", split="test", token=HF_TOKEN)
```

Check that:
- Entity char offsets are valid (`text[char_start:char_end] == entity["text"]`)
- No duplicate entities per row
- Category distribution looks reasonable

## Reference Scripts

All scripts are in `tmp/`:
- `llm_entity_extraction.py` — LLM-based entity extraction
- `split_analysis.py` — overlap analysis (Jaccard, n-grams)
- `build_splits.py` — split creation with char offset annotations
- `push_splits.py` — cleanup and HF push

## Entity Annotation Format

```json
[
  {"text": "MiniMax", "category": "companies", "char_start": 0, "char_end": 7},
  {"text": "M2.5", "category": "models", "char_start": 120, "char_end": 124}
]
```

## Entity CER Guidance

See `docs/entity-cer-guidance.md` for how Trelis Studio should compute per-entity and per-category CER using these annotations.
