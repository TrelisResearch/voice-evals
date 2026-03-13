# AI Terms v2 Dataset Build Report

## Phase 1: Source Texts (Complete)
- Compiled **60 candidate text passages** from AI news articles (Jan 1 – Feb 28, 2026)
- Sources: web searches of AI news aggregators, model release trackers, tech blogs
- Topics cover: model releases, funding rounds, benchmark scores, hardware, safety, geopolitics
- Saved to `text/v2_candidate_texts.json`

## Phase 2: Entity Extraction (Complete)
- Extracted entities using Claude Haiku via OpenRouter
- **322 total entities** across 60 rows
- Category breakdown:
  - Companies: 145
  - Models: 67
  - Products: 57
  - Technical: 27
  - Benchmarks: 15
  - People: 11
- Average: ~5.4 entities per row
- Saved to `tmp/v2_entities_by_row.json`
- Script: `tmp/extract_entities_v2.py`

## Phase 3: TTS Audio Generation (In Progress)
- Using Trelis Studio TTS evaluation endpoint
- Will push text+entities as HF dataset, then generate audio

## Phase 4: Difficulty Filtering (Pending)
- Filter models: Whisper Large-v3-Turbo, Parakeet TDT 0.6B v3, Qwen3-ASR 1.7B
- Metric: median per-row entity CER across 3 models
- Drop rows below difficulty threshold

## Phase 5: Split Assignment & Dedup (Pending)
- Assign filtered rows to public/semi-private/private splits
- Entity overlap and n-gram deduplication checks

## Phase 6: Push to HuggingFace (Pending)
- Target: `Trelis/ai-terms-v2-{public,semi-private,private}`
