# AI Terms v2 Dataset Build Report

## Phase 1: Source Texts (Complete)
- Compiled **100 candidate text passages** from AI news articles (Jan 1 – Feb 28, 2026)
- Initial 60 rows, expanded to 100 for larger filtering pool
- Sources: web searches of AI news aggregators, model release trackers, tech blogs
- Topics cover: model releases, funding rounds, benchmark scores, hardware, safety, geopolitics
- Texts use standard digit formatting (not spelled-out numbers)
- Saved to `text/v2_candidate_texts.json`

## Phase 2: Entity Extraction (Complete)
- Extracted entities using Claude Sonnet via OpenRouter
- **571 total entities** across 100 rows
- Categories: companies, models, products, technical, benchmarks, people
- Average: ~5.7 entities per row
- Verified char positions with auto-correction
- Saved to `tmp/v2_entities_by_row.json`

## Phase 3: TTS Audio Generation (Complete)
- Used Trelis Studio TTS (Orpheus 3B model, speaker: tara, `max_new_tokens=4000`)
- Generated audio for all **100/100 rows**
- `max_new_tokens=4000` critical to avoid audio truncation (default 1200 was cutting audio short)
- Individual TTS datasets at `ronanarraig/ai-terms-v2-tts-regen-{001..100}`
- Full 100-row dataset: `ronanarraig/ai-terms-v2-all100`
- Studio eval dataset: `ronanarraig/ai-terms-v2-all100-eval-v3` (95 rows survived max_duration filter)

## Phase 4: Difficulty Filtering (Complete)
- Evaluated **95 rows** with 3 filter models on `ronanarraig/ai-terms-v2-all100-eval-v3`:
  - Whisper Large-v3-Turbo: 5.3% CER
  - Parakeet TDT 0.6B v3: 5.5% CER
  - Qwen3-ASR 1.7B: 7.2% CER
- Entity CER computed locally using entity annotations
- Per-row median CER range: 0.010 – 0.113
- **Kept 63 harder rows** (median CER >= 0.045), **dropped 30 easiest**
- 2 rows unmatched (text mismatch), 5 rows skipped by max_duration=30s

## Phase 5: Split Assignment & Dedup (Complete)
- Greedy assignment minimizing entity overlap between splits
- **Entity overlap** (Jaccard, excluding ubiquitous terms):
  - public vs semi_private: 0.077 (16 overlapping entities)
  - public vs private: 0.070 (15 overlapping entities)
  - semi_private vs private: 0.079 (16 overlapping entities)

### Split contents
| Split | Rows | CER range (median) |
|-------|------|--------------------|
| public | 21 | 0.045 – 0.113 |
| semi_private | 21 | 0.046 – 0.101 |
| private | 21 | 0.045 – 0.098 |

## Phase 6: Push to HuggingFace (Complete)
- `ronanarraig/ai-terms-v2-public` (21 rows, private, split=test)
- `ronanarraig/ai-terms-v2-semi-private` (21 rows, private, split=test)
- `ronanarraig/ai-terms-v2-private` (21 rows, private, split=test)
- All datasets include: audio, text, entities (JSON with char offsets), duration_s
- Studio eval datasets:
  - `ronanarraig/ai-terms-v2-public-eval`
  - `ronanarraig/ai-terms-v2-semi-private-eval`
  - `ronanarraig/ai-terms-v2-private-eval`

## Phase 7: Benchmark Results (Complete)
Evaluated 12 ASR models on final 21-row splits:

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

**Notes**:
- Proprietary models (deepgram, gemini, elevenlabs, fireworks) only on public + semi-private (never private, per leakage prevention policy)
- Sample counts per split: public=9, semi_private=15, private=14 (some samples exceed max_duration=30s)
- Best overall: ElevenLabs Scribe v2, followed by VibeVoice and OmniASR
- CERs are significantly higher than v1, confirming difficulty filtering worked

## Known Issues & Next Steps
1. **Sample dropout**: Only 9-15 of 21 samples evaluated per split due to 30s max_duration limit. Consider requesting longer max_duration or trimming audio.
2. **TTS audio quality**: Orpheus model mispronounces some entity names — consider re-recording hard rows manually
3. **Studio bugs filed**:
   - TTS prompts array only generates first prompt
   - Trelis/ namespace write access lost (workaround: use ronanarraig/)
4. **Suggestion filed**: Batch evaluation endpoint to run multiple models on same dataset in one call

## Pipeline Scripts (in `tmp/`)
- `rewrite_texts_digits.py` — Convert spelled-out numbers to digits
- `extract_entities_v2_regen.py` — Entity extraction with char offsets
- `generate_tts_v2_full.py` — TTS generation with max_new_tokens=4000
- `push_v2_all100.py` — Push 100-row dataset to HF
- `upload_all100_dataprep.py` — Upload to Studio via data-prep
- `run_filter_evals.py` — Run 3 filter models
- `filter_and_split_all100.py` — Difficulty filtering + split assignment
- `push_v2_final_splits.py` — Push final 3 splits to HF
- `upload_splits_dataprep.py` — Upload splits to Studio
- `benchmark_v2_retry.py` — Full 12-model benchmark with retry logic
