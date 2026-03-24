# Voice Evals

## Project Overview
Voice evaluation datasets and benchmarks for ASR models. Uses [Trelis Studio](https://studio.trelis.com) for dataset creation and evaluation.

## Environment
- `.env` in root contains `TRELIS_STUDIO_API_KEY` and `HF_TOKEN`
- `GITHUB_PAT` is available in `/home/claude/TR/.env` for pushing to GitHub
- Use `uv` for all Python operations

## Trelis Studio API
- Base URL: `https://studio.trelis.com`
- Auth: `Authorization: Bearer $TRELIS_API_KEY`
- Full OpenAPI spec: `https://studio.trelis.com/openapi.json`
- Human-readable docs: `https://studio.trelis.com/api/skills.md`

### Key Workflows
1. **Data Prep**: Create session → upload audio → transcribe (draft-transcribe for VAD+chunking) → process (forced alignment + push to HF)
2. **Evaluation**: POST `/api/v1/evaluation/jobs` with `model_id`, `dataset_id`, `split`, `num_samples`. Returns WER/CER.
3. **Training**: POST `/api/v1/training/jobs` for Whisper LoRA fine-tuning on H100.
4. **Filtering**: POST `/api/v1/filtering/jobs` to remove bad samples by CER threshold.

### Important API Parameters
- `process` endpoint: `split_option` (create_validation/train_only/validation_only/test_only), `enable_quality_checks`, `language` (ISO 639-3), `target_chunk_duration` (default 20s), `max_chunk_duration` (30s), `min_chunk_duration` (5s)
- `evaluation` endpoint: `normalizer` (auto/generic/none/language-name), `language` (default auto, supports per-sample multilingual via dataset 'language' column)
- Draft transcription supports VAD silence stripping and smart chunking

## Dataset Architecture

### Three tiers:
1. **Public** — fully open, published on HuggingFace
2. **Semi-private** — shared dataset, proprietary + open-source models evaluated privately via Studio
3. **Private (Trelis-OOD)** — never shared online, only Trelis evaluates using open-source models

### Planned Datasets

#### Technical AI v1 (`Trelis/ai-terms-{public,semi-private,private}`)
- **Status: Complete (v0 eval results in `docs/eval-results-ai-terms-v0.md`)**
- 12 samples per split, 6 entity categories, 25+ models evaluated
- Sources: AI News articles (with permission), YouTube transcripts, rewrites of transcripts

#### Technical AI v2 (`ai-terms-v2-{public,semi-private,private}`)
- **Status: Planning**
- Goal: harder dataset — filter out "easy" rows that most models already handle well
- Sources: AI News articles from **Jan 1 – Feb 28, 2026** only (contamination avoidance)
- **Difficulty filtering**: run 3 diverse open-source models, compute per-row entity CER, take median, drop rows below threshold
- Filter models: Whisper Large-v3-Turbo, Parakeet TDT 0.6B v3, Qwen3-ASR 1.7B
- **Audio pipeline (hybrid TTS + human)**:
  1. Generate TTS audio for candidate texts (once Studio TTS integration is available)
  2. Run 3 filter models on TTS audio to identify hard rows
  3. If TTS audio quality is sufficient, keep it; otherwise re-record hard rows manually
- Requires entity-based deduplication across splits to prevent leakage
- Larger initial pool (~50+ rows) to allow filtering down to ~15-20 per split

#### Tricky TTS (`ronanarraig/tricky-tts-{public,semi-private,private}`)
- **Status: Phase 1 Complete** — see `tricky-tts/roadmap.md`
- Purpose: evaluate TTS models on linguistically and typographically challenging English text
- Text-only dataset (no audio); TTS models generate audio at eval time
- ~50 rows per split, with a `text` + `category` column
- Categories: prosody, edge_cases, phonetic, punctuation, robustness, domain_specific
- Evaluation via Trelis Studio: UTMOS (naturalness) + Round Trip ASR (accuracy)
- Phase 2: median-of-N filtering to ensure difficulty and avoid unfairly penalising any single model
- English only; will migrate to `Trelis/` org once on Trelis infrastructure

#### Code-Switching (`voice-evals-code-switching`)
- Languages: French, English, German, Spanish
- Synthetic data mixing languages within utterances
- Leakage concern: ensure no identical synthetic prompts/patterns across public and semi-private splits

#### Trelis-OOD (private, held out)
- ATC (air traffic control) audio
- English-Irish code-switching
- Technical domain data
- Goal: non-web data to minimize contamination risk
- **Never publish online**

## Leakage Prevention
- Run entity/n-gram overlap detection between public, semi-private, and private splits before publishing
- For synthetic code-switching: use different topic seeds / prompt templates per split
- For Trelis-OOD: source non-web data (radio recordings, ATC, in-person recordings) to avoid web contamination
- **NEVER run evaluation with proprietary/closed-source models on `-private` splits.** The private split must only be evaluated with open-source models run locally or via Trelis. This prevents any risk of private data leaking to third-party APIs.
