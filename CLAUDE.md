# Voice Evals

## Project Overview
Voice evaluation datasets and benchmarks for ASR models. Uses [Trelis Studio](https://studio.trelis.com) for dataset creation and evaluation.

## Environment
- `.env` in root contains `TRELIS_STUDIO_API_KEY` and `HF_TOKEN`
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

#### Technical AI (`voice-evals-ai-technical`)
- Sources: AI News articles (with permission), YouTube transcripts, rewrites of transcripts
- Requires entity-based deduplication across splits to prevent leakage

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
