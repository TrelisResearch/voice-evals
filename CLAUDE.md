# Voice Evals

## Project Overview
Voice evaluation datasets and benchmarks for ASR models. Uses [Trelis Studio](https://studio.trelis.com) for dataset creation and evaluation.

## Environment
- `.env` in root contains `TRELIS_STUDIO_API_KEY` and `HF_TOKEN`
- Use `uv` for all Python operations
- HF datasets are pushed under `ronanarraig/` namespace (Trelis namespace write access not available on build machines — intentional)
- On a Hetzner VPS: `GITHUB_PAT` is available in `/home/claude/TR/.env` for pushing to GitHub

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
5. **TTS**: Orpheus 3B, speaker: tara, `max_new_tokens=4000` (default 1200 truncates audio)

### Important API Parameters
- `process` endpoint: `split_option` (create_validation/train_only/validation_only/test_only), `enable_quality_checks`, `language` (ISO 639-3), `target_chunk_duration` (default 20s), `max_chunk_duration` (30s), `min_chunk_duration` (5s)
- `evaluation` endpoint: `normalizer` (auto/generic/none/language-name), `language` (default auto, supports per-sample multilingual via dataset 'language' column)
- Draft transcription supports VAD silence stripping and smart chunking

## Dataset Architecture

### Three tiers:
1. **Public** — fully open, published on HuggingFace
2. **Semi-private** — shared dataset, proprietary + open-source models evaluated privately via Studio
3. **Private** — never shared online, only evaluated with open-source models (via Studio or locally)

### Datasets

#### AI Terms v1 (`Trelis/ai-terms-{public,semi-private,private}`)
- **Status: Complete**
- 12 samples/split, 6 entity categories, 25+ models evaluated
- Full results: `docs/eval-results-ai-terms-v0.md`
- Sources: AI News articles, YouTube transcripts, rewrites

#### AI Terms v2 (`ronanarraig/ai-terms-v2-{public,semi-private,private}`)
- **Status: Pilot complete — small-scale validation next**
- 21 rows/split, difficulty-filtered (median entity CER >= 0.045 across 3 filter models)
- TTS audio: Orpheus 3B/tara via Studio; data-prep pipeline issues mostly resolved
- Full build details: `reports/ai-terms-v2-partA.md`
- Next steps: `reports/ai-terms-v2-partB-roadmap.md`
- Key open issue: **number formatting** — need to confirm Orpheus reads digits correctly, else add a separate TTS transcript column with numbers spelled out
- Key open issue: **30s max_duration cap** causes sample dropout (9–15 of 21 rows evaluated); request limit increase from Studio

#### Code-Switching (`voice-evals-code-switching`)
- **Status: Planned**
- Languages: French, English, German, Spanish
- Synthetic data mixing languages within utterances
- Different topic seeds / prompt templates per split to prevent leakage

#### Medical Terms
- **Status: Planned** (start after ai-terms v2 pipeline is stable)
- Entity categories: drugs, procedures, conditions, anatomy, organisations

#### Legal Terms
- **Status: Planned** (start after ai-terms v2 pipeline is stable)
- Entity categories: case names, legal terms, jurisdictions, statutes

#### Trelis-OOD (private, held out)
- **Status: Planned**
- ATC audio, English-Irish code-switching, technical recordings
- Non-web data to minimise contamination
- **Never publish online**

## Leakage Prevention
- Run entity/n-gram overlap detection between splits before publishing
- For synthetic code-switching: use different topic seeds / prompt templates per split
- For Trelis-OOD: source non-web data (radio recordings, ATC, in-person recordings)
- **NEVER run evaluation with proprietary/closed-source models on `-private` splits.** Private splits are open-source models only (local or via Trelis). This prevents private data leaking to third-party APIs.

## Pending Studio Feature Requests
- **Re-recording rows via UI** — needed for final QA pass (flag row, record replacement, re-run forced alignment)
- **Import HF datasets into data viewer** — reduce friction during iterative builds
- **Drop rows in data viewer** — row-level delete for inline QA cleanup
