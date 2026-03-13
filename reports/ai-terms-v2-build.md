# AI Terms v2 Dataset Build Report

## Phase 1: Source Texts (Complete)
- Compiled **100 candidate text passages** from AI news articles (Jan 1 – Feb 28, 2026)
- Initial 60 rows, expanded to 100 for larger filtering pool
- Sources: web searches of AI news aggregators, model release trackers, tech blogs
- Topics cover: model releases, funding rounds, benchmark scores, hardware, safety, geopolitics
- Saved to `text/v2_candidate_texts.json`

## Phase 2: Entity Extraction (Complete)
- Extracted entities using Claude Haiku via OpenRouter
- **~540 total entities** across 100 rows
- Category breakdown (original 60): Companies: 145, Models: 67, Products: 57, Technical: 27, Benchmarks: 15, People: 11
- Average: ~5.4 entities per row
- Saved to `tmp/v2_entities_by_row.json`
- Script: `tmp/extract_entities_v2.py`

## Phase 3: TTS Audio Generation (Complete)
- Used Trelis Studio TTS evaluation endpoint (Orpheus 3B model, speaker: tara)
- Generated audio for all **100/100 rows**
- **Bug filed**: TTS `prompts` array only processes first element (workaround: 1 job per prompt)
- **Bug filed**: Studio evaluation can't access private HF datasets
- Individual TTS datasets at `Trelis/ai-terms-v2-tts-{001..100}`
- Full 100-row dataset: `ronanarraig/ai-terms-v2-all100` → `Trelis/ai-terms-v2-all100-eval` (93 rows survived data-prep)

## Phase 4: Difficulty Filtering (Complete)
- Evaluated all **93 rows** (100 generated, 7 lost in data-prep) with 3 ASR models on `Trelis/ai-terms-v2-all100-eval`:
  - Whisper Large-v3-Turbo: 30.7% CER
  - Parakeet TDT 0.6B v3: 30.2% CER
  - Qwen3-ASR 1.7B: 29.1% CER
- Entity CER computed locally using entity annotations from `v2_entities_by_row.json`
- Per-row median CER range: 0.042 – 0.657
- **Aggressive filtering**: dropped bottom 30 easiest rows (median CER < 0.198)
- Kept **63 harder rows**

### Key observations
- Hardest rows: funding amounts with large written-out numbers, Chinese company/model names
- Easiest rows: simple entity-light content (corporate announcements, regulatory news)
- TTS mispronounces many entity names (e.g. "Zhipu AI" → "GPU AI", "K-EXAONE" → "KX01")

## Phase 5: Split Assignment & Dedup (Complete)
- Greedy assignment minimizing entity overlap between splits
- **Entity overlap** (Jaccard, excluding ubiquitous terms):
  - public vs semi_private: 0.102 (21 overlapping entities)
  - public vs private: 0.116 (23 overlapping entities)
  - semi_private vs private: 0.068 (14 overlapping entities)

### Split contents
| Split | Rows | Median CER range |
|-------|------|-----------------|
| public | 20 | 0.218 – 0.562 |
| semi_private | 22 | 0.198 – 0.552 |
| private | 21 | 0.202 – 0.657 |

## Phase 6: Push to HuggingFace (Complete)
- `ronanarraig/ai-terms-v2-public` (20 rows, private, split=test)
- `ronanarraig/ai-terms-v2-semi-private` (22 rows, private, split=test)
- `ronanarraig/ai-terms-v2-private` (21 rows, private, split=test)
- All datasets include: audio, text, entities (JSON with char offsets), duration_s

## Phase 7: Benchmark Results (Complete)
Evaluated 12 ASR models on the original 60-row splits via `Trelis/ai-terms-v2-{public,semi-private,private}-eval`:

| Model | Public CER | Semi-Priv CER | Private CER |
|---|---|---|---|
| elevenlabs/scribe-v2 | **16.4%** | **19.6%** | N/A |
| facebook/omniASR-LLM-1B | 19.4% | 27.3% | 25.5% |
| microsoft/VibeVoice-ASR-HF | 19.7% | 22.3% | **23.0%** |
| Qwen/Qwen3-ASR-0.6B | 20.1% | 25.2% | 26.3% |
| Qwen/Qwen3-ASR-1.7B | 20.3% | 26.4% | 26.5% |
| nvidia/canary-1b-v2 | 23.9% | 26.6% | 27.3% |
| deepgram/nova-3 | 24.3% | 31.5% | N/A |
| openai/whisper-large-v3 | 24.6% | 27.6% | 27.1% |
| fireworks/whisper-v3 | 24.5% | 27.7% | N/A |
| nvidia/parakeet-tdt-0.6b-v3 | 25.2% | 27.2% | 27.5% |
| google/gemini-2.5-pro | 25.2% | 27.6% | N/A |
| openai/whisper-large-v3-turbo | 25.6% | 27.6% | 27.6% |

**Notes**: Proprietary models (deepgram, gemini, elevenlabs, fireworks) only evaluated on public + semi-private splits (never on private, per leakage prevention policy). Benchmark was run on the earlier 60-row splits before the expanded 100-row re-filtering.

## Known Issues & Next Steps
1. **TTS audio quality**: Orpheus model mispronounces many entity names — re-record hard rows manually
2. **Number normalization**: Written-out numbers ("one hundred and ten billion") cause inflated CER when ASR outputs digits — need normalizer that handles this
3. **Re-benchmark on final splits**: The benchmark results above are from the original 60-row splits. Need to re-upload the new 63-row splits via data-prep and re-run benchmarks.
4. **Studio bugs filed**:
   - TTS prompts array only generates first prompt
   - Evaluation can't access private HF datasets
5. **Suggestion filed**: Batch evaluation endpoint to run multiple models on same dataset in one call
