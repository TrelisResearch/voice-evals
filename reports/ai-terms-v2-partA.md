# AI Terms v2 — Part A: Build Report

Dataset series: `ronanarraig/ai-terms-v2-{public,semi-private,private}`
Build date: 2026-03-18
Status: **Pilot complete — pending QA and re-recording pass**

---

## Overview

v2 is a harder version of the AI Technical Terms benchmark (v1: `Trelis/ai-terms-{public,semi-private,private}`). The core change is a difficulty filtering step that drops rows current models handle easily, leaving only samples where entity recognition is genuinely challenging. Audio was generated with TTS (Orpheus 3B) rather than recorded manually, trading some audio naturalness for rapid iteration speed.

---

## Phase 1: Source Texts

- **100 candidate text passages** from AI news articles published **Jan 1 – Feb 28, 2026**
- Date window chosen to minimise contamination: most model training cuts off before 2026
- Topics: model releases, funding rounds, benchmark scores, hardware, safety, AI geopolitics
- Numbers formatted as digits (not spelled-out words) — see Part B for TTS implications
- Initial pool of 60 rows expanded to 100 to allow aggressive filtering
- Saved to `text/v2_candidate_texts.json`

## Phase 2: Entity Extraction

- Extracted using Claude Sonnet via OpenRouter
- **571 total entities** across 100 rows (~5.7 per row)
- 6 categories: companies, models, products, technical, benchmarks, people
- Char offsets verified with auto-correction
- Saved to `tmp/v2_entities_by_row.json`

## Phase 3: TTS Audio Generation

- **Tool**: Trelis Studio TTS — Orpheus 3B, speaker: tara
- `max_new_tokens=4000` required (default 1200 truncated audio)
- Generated audio for all **100/100 rows**
- Individual TTS datasets: `ronanarraig/ai-terms-v2-tts-regen-{001..100}`
- Merged dataset: `ronanarraig/ai-terms-v2-all100`
- Studio eval-ready (filtered by max_duration=30s): `ronanarraig/ai-terms-v2-all100-eval-v3` (95 rows)

## Phase 4: Difficulty Filtering

Evaluated all 95 rows with 3 open-source filter models:

| Model | Aggregate Entity CER |
|-------|----------------------|
| openai/whisper-large-v3-turbo | 5.3% |
| nvidia/parakeet-tdt-0.6b-v3 | 5.5% |
| Qwen/Qwen3-ASR-1.7B | 7.2% |

**Filtering logic:** computed per-row median entity CER across the 3 models; dropped rows below threshold 0.045.

| Outcome | Count |
|---------|-------|
| Rows kept (hard) | 63 |
| Rows dropped (easy) | 30 |
| Rows unmatched (text mismatch) | 2 |
| Rows skipped (max_duration > 30s) | 5 |

## Phase 5: Split Assignment & Deduplication

Greedy assignment minimising entity overlap across splits:

| Pair | Jaccard overlap | Overlapping entities |
|------|----------------|----------------------|
| public vs semi_private | 0.077 | 16 |
| public vs private | 0.070 | 15 |
| semi_private vs private | 0.079 | 16 |

Final split sizes:

| Split | Rows | Median CER range |
|-------|------|-----------------|
| public | 21 | 0.045 – 0.113 |
| semi_private | 21 | 0.046 – 0.101 |
| private | 21 | 0.045 – 0.098 |

## Phase 6: HuggingFace Upload

Datasets pushed (all under `ronanarraig/`, private visibility):

| Dataset | Rows | Split |
|---------|------|-------|
| `ronanarraig/ai-terms-v2-public` | 21 | test |
| `ronanarraig/ai-terms-v2-semi-private` | 21 | test |
| `ronanarraig/ai-terms-v2-private` | 21 | test |

Columns: `audio`, `text`, `entities` (JSON with char offsets), `duration_s`

Studio eval datasets:
- `ronanarraig/ai-terms-v2-public-eval`
- `ronanarraig/ai-terms-v2-semi-private-eval`
- `ronanarraig/ai-terms-v2-private-eval`

## Phase 7: Benchmark Results (12 models)

Sample counts per split were 9–15 of 21 rows due to the 30s max_duration constraint in Studio.

| Model | Public CER | Semi-Priv CER | Private CER |
|-------|-----------|--------------|------------|
| elevenlabs/scribe-v2 | **16.4%** | **19.6%** | — |
| facebook/omniASR-LLM-1B | 19.4% | 27.3% | 25.5% |
| microsoft/VibeVoice-ASR-HF | 19.7% | 22.3% | **23.0%** |
| Qwen/Qwen3-ASR-0.6B | 20.1% | 25.2% | 26.3% |
| Qwen/Qwen3-ASR-1.7B | 20.3% | 26.4% | 26.5% |
| nvidia/canary-1b-v2 | 23.9% | 26.6% | 27.3% |
| deepgram/nova-3 | 24.3% | 31.5% | — |
| openai/whisper-large-v3 | 24.6% | 27.6% | 27.1% |
| fireworks/whisper-v3 | 24.5% | 27.7% | — |
| nvidia/parakeet-tdt-0.6b-v3 | 25.2% | 27.2% | 27.5% |
| google/gemini-2.5-pro | 25.2% | 27.6% | — |
| openai/whisper-large-v3-turbo | 25.6% | 27.6% | 27.6% |

*Proprietary models (ElevenLabs, Deepgram, Gemini, Fireworks) only on public + semi-private — never private, per leakage prevention policy.*

**Key observations:**
- CERs are substantially higher than v1 (e.g. Whisper-v3-Turbo: 4.1% → 25.6% on public entity CER) — difficulty filtering worked
- ElevenLabs Scribe v2 is the clear leader, well ahead of all others
- Best open-source: VibeVoice ASR-HF, followed by OmniASR 1B
- The spread between models is larger than v1, making v2 more discriminative

---

## Known Issues

| Issue | Impact | Status |
|-------|--------|--------|
| Sample dropout: only 9–15 of 21 rows evaluated per split due to 30s max_duration cap | Reduces benchmark reliability | Pending — see Part B |
| Orpheus TTS mispronounces some entity names (numbers, acronyms, novel model names) | Audio quality / reference transcript mismatch | Pending — see Part B |
| Trelis/ namespace write access unavailable on the droplet; datasets under `ronanarraig/` | Naming only, no data issue | Intentional workaround |
| Studio bug: TTS prompts array only generates first prompt | Required manual row-by-row generation | Filed with Studio |
| Studio bug: data-prep pipeline issues with Orpheus audio | Mostly resolved | Resolved |

---

## Pipeline Scripts (in `tmp/`)

| Script | Purpose |
|--------|---------|
| `rewrite_texts_digits.py` | Convert spelled-out numbers to digits |
| `extract_entities_v2_regen.py` | Entity extraction with char offsets |
| `generate_tts_v2_full.py` | TTS generation (max_new_tokens=4000) |
| `push_v2_all100.py` | Push 100-row pool to HF |
| `upload_all100_dataprep.py` | Upload pool to Studio via data-prep |
| `run_filter_evals.py` | Run 3 filter models |
| `filter_and_split_all100.py` | Difficulty filtering + split assignment |
| `push_v2_final_splits.py` | Push final 3 splits to HF |
| `upload_splits_dataprep.py` | Upload splits to Studio |
| `benchmark_v2_retry.py` | Full 12-model benchmark with retry logic |
