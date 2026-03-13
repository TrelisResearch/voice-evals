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
- Generated audio for all **60/60 rows**
- **Bug filed**: TTS `prompts` array only processes first element (workaround: 1 job per prompt)
- **Bug filed**: Studio evaluation can't access private HF datasets
- Individual TTS datasets at `Trelis/ai-terms-v2-tts-{001..060}`
- Data-prep upload created `Trelis/ai-terms-v2-eval-ready` (60 rows, Studio-accessible)

## Phase 4: Difficulty Filtering (Complete)
- Evaluated all **60 rows** with 3 ASR models on `Trelis/ai-terms-v2-eval-ready`:
  - Whisper Large-v3-Turbo: 23.1% CER
  - Parakeet TDT 0.6B v3: 22.8% CER
  - Qwen3-ASR 1.7B: 21.5% CER
- Entity CER computed locally using entity annotations from `v2_entities_by_row.json`
- Per-row median CER range: 0.031 – 0.540
- Dropped bottom 25% (**15 easiest rows**, median CER < 0.100)
- Kept **45 harder rows**

### Key observations
- Hardest rows: funding amounts with large written-out numbers, Chinese company/model names
- Easiest rows: simple entity-light content (corporate announcements, regulatory news)
- TTS mispronounces many entity names (e.g. "Zhipu AI" → "GPU AI", "K-EXAONE" → "KX01")

## Phase 5: Split Assignment & Dedup (Complete)
- Greedy assignment minimizing entity overlap between splits
- **Entity overlap** (Jaccard, excluding ubiquitous terms):
  - public vs semi_private: 0.076 (9 overlapping entities)
  - private vs public: 0.060 (7 overlapping entities)
  - private vs semi_private: 0.057 (6 overlapping entities)

### Split contents
| Split | Rows | Median CER range |
|-------|------|-----------------|
| public | 16 | 0.100 – 0.540 |
| semi_private | 15 | 0.108 – 0.510 |
| private | 14 | 0.127 – 0.490 |

## Phase 6: Push to HuggingFace (Complete)
- `ronanarraig/ai-terms-v2-public` (16 rows, private, split=test)
- `ronanarraig/ai-terms-v2-semi-private` (15 rows, private, split=test)
- `ronanarraig/ai-terms-v2-private` (14 rows, private, split=test)
- All datasets include: audio, text, entities (JSON with char offsets), duration_s

## Known Issues & Next Steps
1. **TTS audio quality**: Orpheus model mispronounces many entity names → consider re-recording hard rows manually
2. **Number normalization**: Written-out numbers ("one hundred and ten billion") cause inflated CER when ASR outputs digits — need normalizer that handles this
3. **Studio bugs filed**:
   - TTS prompts array only generates first prompt
   - Evaluation can't access private HF datasets
