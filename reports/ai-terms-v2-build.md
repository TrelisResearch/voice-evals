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

## Phase 3: TTS Audio Generation (Complete)
- Used Trelis Studio TTS evaluation endpoint (Orpheus 3B model, speaker: tara)
- Generated audio for 34/60 rows (remaining rows still generating)
- **Bug filed**: TTS `prompts` array only processes first element (workaround: 1 job per prompt)
- **Bug filed**: Studio evaluation can't access private HF datasets
- Individual TTS datasets at `Trelis/ai-terms-v2-tts-{001..034}`
- Merged dataset at `ronanarraig/ai-terms-v2-candidates-with-audio` (private)

## Phase 4: Difficulty Filtering (Complete)
- Evaluated 34 rows with 3 ASR models:
  - Whisper Large-v3-Turbo: 31.7% CER, 24.9% entity CER
  - Parakeet TDT 0.6B v3: 31.6% CER, 25.4% entity CER
  - Qwen3-ASR 1.7B: 29.6% CER, 24.8% entity CER
- High CER partly due to TTS pronunciation issues and number normalization (written-out numbers vs digits)
- Per-row median CER range: 0.071 – 0.580
- Dropped bottom 25% (9 easiest rows, median CER < 0.213)
- Kept 25 harder rows

### Key observations
- Hardest rows: funding amounts with large written-out numbers, Chinese company/model names
- Easiest rows: simple entity-light content (corporate announcements, regulatory news)
- TTS mispronounces many entity names (e.g. "Zhipu AI" → "GPU AI", "K-EXAONE" → "KX01")

## Phase 5: Split Assignment & Dedup (Complete)
- Greedy assignment minimizing entity overlap between splits
- **Entity overlap**: Jaccard < 0.02 between all split pairs
  - public vs semi_private: 0 overlapping entities
  - private vs public: 1 overlap ("claude opus 4.5")
  - private vs semi_private: 1 overlap ("gemini 3 pro")

### Split contents
| Split | Rows | Median CER range |
|-------|------|-----------------|
| public | 9 | 0.239 – 0.580 |
| semi_private | 8 | 0.250 – 0.556 |
| private | 8 | 0.250 – 0.535 |

## Phase 6: Push to HuggingFace (Complete)
- `ronanarraig/ai-terms-v2-public` (9 rows, private, split=test)
- `ronanarraig/ai-terms-v2-semi-private` (8 rows, private, split=test)
- `ronanarraig/ai-terms-v2-private` (8 rows, private, split=test)
- All datasets include: audio, text, entities (JSON with char offsets), duration_s

## Known Issues & Next Steps
1. **TTS audio quality**: Orpheus model mispronounces many entity names → consider re-recording hard rows manually
2. **Number normalization**: Written-out numbers ("one hundred and ten billion") cause inflated CER when ASR outputs digits — need normalizer that handles this
3. **Remaining 26 rows**: TTS generation for rows v2-035 to v2-060 was interrupted; can be resumed later to expand the dataset
4. **Studio bugs filed**:
   - TTS prompts array only generates first prompt
   - Evaluation can't access private HF datasets
