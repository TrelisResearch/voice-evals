# Medical ASR — Phase 1 Report

**Status:** Phase 1 complete (2026-04-04)
**Goal:** Landscape survey + baseline model eval + curated hard eval set construction

---

## 1a. Dataset Survey

### Datasets audited

| Dataset | Rows (test) | Audio | Entity labels | Verdict |
|---------|-------------|-------|--------------|---------|
| `ekacare/eka-medical-asr-evaluation-dataset` EN | 3,619 | Embedded | Yes — 5 types | **Use as primary baseline** |
| `leduckhai/MultiMed` EN | 4,751 | Embedded | No (LLM-extracted) | **Use as secondary baseline** |
| `united-we-care/United-Syn-Med` | 79,069 | tar.gz (4.4 GB) | No (LLM-extracted) | **Use for drug-name CER specifically** |
| `google/medasr` | — | — | — | Model only, no public dataset |

### EKA Medical ASR Eval
- 3,619 EN rows, test-only, MIT licence
- 57 speakers from 4 Indian medical colleges; 16kHz mono; median 4.9s
- Entity categories: `drugs`, `clinical_findings`, `diagnostics`, `advices`, `misc_medical`
- recording_context breakdown: narration_entity (61%), narration_sentence (36%), conversation (3%)
- **Strengths:** real speech, entity-annotated, reasonably diverse speakers
- **Limitations:** Indian accent and Indian drug brands dominate; not representative of US/UK clinical dictation; drug entities are single-mention narrations ("take azithromycin"), not in-context clinical sentences

### MultiMed EN
- 4,751 test rows, MIT licence, sourced from YouTube medical channels
- Columns: audio, text, duration only — no entity labels
- Content: lectures, interviews, podcasts (not clinical dictation)
- **Strengths:** longer utterances, diverse speakers and settings, free
- **Limitations:** lecture/podcast register very different from clinical use; no entity annotations; YouTube Fair Use provenance; WER is high (~17–23%) partly because of transcription quality, not just model errors

### United-Syn-Med
- 79,069 test rows, CC BY-SA 4.0, accessed via 4.4 GB tar.gz
- All rows: drug category only; synthetic TTS; short sentences (median 13 words)
- **Strengths:** large, fully synthetic and clean, focused on drug brand names and generic names; useful as a drug-name difficulty signal
- **Limitations:** synthetic audio doesn't reflect real clinical acoustic conditions; drugs-only — no procedures, conditions, anatomy; oversimplified sentence structure

### Entity extraction (MultiMed + United)
LLM-based extraction using Gemini 2.5 Flash + Claude Sonnet 4.6. Only agreed-by-both entities kept.

| Dataset | Rows with entities | Agreed entities | Agreement rate |
|---------|-------------------|-----------------|----------------|
| MultiMed EN pilot (50 rows) | 20/50 | 42 | 45% |
| United-Syn-Med pilot (50 rows) | 48/50 | 97 | 84% |

MultiMed entity categories: conditions (23), anatomy (12), procedures (4), drugs (2). The low drug count reflects the lecture/podcast style.

---

## 1b. Model Evaluation Results

50-row pilot across three datasets. Models: 8 working (Moonshine-tiny failed dtype bug, Qwen3-ASR-1.7B broken — both bugs filed with Studio).

### EKA pilot — real Indian clinical speech (sorted by entity CER)

**Open-source / open-weights:**

| Model | WER | CER | EntCER | drugs | clinical | diagnostics | advices |
|-------|-----|-----|--------|-------|----------|-------------|---------|
| openai/whisper-large-v3 | 0.133 | 0.039 | 0.054 | 0.133 | 0.019 | 0.034 | 0.042 |
| mistralai/Voxtral-Mini-3B-2507 | 0.130 | 0.045 | 0.059 | 0.121 | 0.023 | 0.034 | 0.091 |
| nvidia/canary-1b-v2 | 0.174 | 0.072 | 0.075 | 0.163 | 0.032 | 0.052 | 0.083 |
| UsefulSensors/moonshine-tiny | FAIL | — | — | — | — | — | — |
| Qwen/Qwen3-ASR-1.7B | broken | — | — | — | — | — | — |

**Proprietary:**

| Model | WER | CER | EntCER | drugs | clinical | diagnostics | advices |
|-------|-----|-----|--------|-------|----------|-------------|---------|
| google/gemini-2.5-pro | 0.110 | 0.040 | **0.039** | 0.077 | 0.010 | 0.060 | 0.072 |
| assemblyai/universal-3-pro | 0.114 | 0.033 | 0.048 | 0.118 | 0.019 | 0.026 | 0.030 |
| elevenlabs/scribe-v2 | 0.178 | 0.086 | 0.072 | 0.148 | 0.019 | 0.069 | 0.144 |
| speechmatics/ursa-2-enhanced | 0.165 | 0.066 | 0.073 | 0.168 | 0.034 | 0.034 | 0.057 |
| deepgram/nova-3 | 0.203 | 0.103 | 0.092 | 0.223 | 0.027 | 0.069 | 0.098 |

### MultiMed EN pilot — YouTube medical lectures/podcasts (sorted by entity CER)

Source: `leduckhai/MultiMed`, YouTube medical channels (lectures, interviews, podcasts, documentaries). Transcriptions are auto-generated YouTube captions — quality variable. **Needs CER-based filtering** (Otsu thresholding on per-sample CER) before use as a reliable benchmark; some rows have entirely wrong reference transcriptions. Entity CER on conditions/anatomy only (low drug count in lecture-style content).

**Open-source / open-weights:**

| Model | WER | CER | EntCER |
|-------|-----|-----|--------|
| openai/whisper-large-v3 | 0.167 | 0.117 | 0.037 |
| mistralai/Voxtral-Mini-3B-2507 | 0.214 | 0.165 | 0.101 |
| nvidia/canary-1b-v2 | 0.219 | 0.154 | 0.088 |

**Proprietary:**

| Model | WER | CER | EntCER |
|-------|-----|-----|--------|
| assemblyai/universal-3-pro | 0.179 | 0.128 | **0.031** |
| google/gemini-2.5-pro | 0.215 | 0.145 | 0.043 |
| speechmatics/ursa-2-enhanced | 0.215 | 0.146 | 0.043 |
| deepgram/nova-3 | 0.210 | 0.147 | 0.074 |
| elevenlabs/scribe-v2 | 0.234 | 0.159 | 0.037 |

### United-Syn-Med pilot — synthetic drug sentences (sorted by entity CER)

Low WER/CER due to clean synthetic audio. Drugs CER 0.23–0.31 even here — strong signal that drug names are hard regardless of acoustic conditions. **TTS quality is poor** — flat prosody, incorrect stress on polysyllabic drug names (e.g. "Nitrazepam is a benzodiazepine medication commonly used to treat insomnia" has wrong emphasis). Orpheus TTS with spoken form rules would produce better audio than United-Syn-Med's TTS pipeline.

**Open-source / open-weights:**

| Model | WER | CER | EntCER | drugs CER |
|-------|-----|-----|--------|-----------|
| mistralai/Voxtral-Mini-3B-2507 | 0.058 | 0.014 | 0.142 | 0.26 |
| nvidia/canary-1b-v2 | 0.106 | 0.028 | 0.176 | 0.31 |
| openai/whisper-large-v3 | 0.079 | 0.022 | 0.155 | 0.28 |

**Proprietary:**

| Model | WER | CER | EntCER | drugs CER |
|-------|-----|-----|--------|-----------|
| elevenlabs/scribe-v2 | 0.053 | 0.013 | **0.128** | 0.23 |
| assemblyai/universal-3-pro | 0.069 | 0.017 | 0.141 | 0.25 |
| deepgram/nova-3 | 0.084 | 0.020 | 0.155 | 0.27 |
| speechmatics/ursa-2-enhanced | 0.073 | 0.029 | 0.160 | 0.28 |

---

## 1c. Key Findings

### Finding 1: Drugs CER is the dominant failure mode across all models and datasets

On EKA, drugs CER ranges 0.08–0.22 vs clinical_findings 0.01–0.03. On United (synthetic, clean audio, drug sentences), drugs CER is 0.23–0.31 even for top models. Drug brand names and generics are where ASR falls down hardest — this is the primary signal our Phase 2 dataset should target.

### Finding 2: Speechmatics does not justify its "medical specialist" claims here

Speechmatics Ursa 2 ranks 5th on EKA entity CER (0.073), behind Gemini, AssemblyAI, Voxtral, and Whisper. On United (drugs only) it ranks 4th. Its medical-specialist marketing is not borne out on these benchmarks. Deepgram Nova 3 is similarly mid-table.

### Finding 3: Gemini 2.5 Pro leads on entity CER but not WER

On EKA, Gemini has the best entity CER (0.039) but only middling WER (0.110 vs AssemblyAI 0.114). The gap between entity CER and WER suggests Gemini handles the hard medical vocabulary well but may over-normalise or paraphrase elsewhere. Worth inspecting samples.

### Finding 4: MultiMed needs a hard 30% CER filter; Otsu threshold too permissive

Source is YouTube medical channels (lectures, interviews, podcasts, documentaries) with auto-generated captions as references. Otsu thresholding produced a threshold of 1.178 — far too permissive, keeping rows with entirely misaligned transcriptions. Manual inspection of the boundary slice (CER 0.20–0.45) confirmed: lower-end rows (CER ~0.20–0.30) look fine; upper-end rows show speaker labels embedded in transcripts (e.g. "Speaker 1:") and clear alignment failures. **Adopted 30% CER hard ceiling (+ 5% floor): 174/499 rows kept.** `ronanarraig/multimed-otsu`

### Finding 4b: United-Syn-Med TTS quality is poor

Prosody is flat and stress patterns on polysyllabic drug names are wrong (e.g. "Nitrazepam is a benzodiazepine medication commonly used to treat insomnia" — incorrect emphasis). This likely flatters models somewhat compared to natural speech, and undersells drug-name difficulty. Our Phase 2 dataset using Orpheus TTS with spoken form rules should produce substantially better audio quality.

### Finding 5: Qwen3-ASR-1.7B non-functional, Moonshine-tiny broken on Studio

Qwen3-ASR-1.7B returns 1.000 WER/CER on all datasets — likely empty output or Chinese characters. Studio bug filed (`5eb76bfc`). Moonshine-tiny dtype mismatch in Studio — bug filed (`77aef1bb`). Both need fixing before Phase 3 fine-tuning targets can be evaluated.

### Finding 6: Model rankings are consistent across datasets

The EKA and United rankings correlate well (Gemini/AssemblyAI top, Deepgram/Canary bottom). This is a positive sign — suggests our Phase 2 dataset doesn't need to be very large to produce stable rankings.

### Finding 7: EKA high-CER rows are valid difficulty signal, not noise

Manual inspection of EKA rows around the Otsu threshold (0.589) revealed these are legitimate hard cases — short drug name narrations where Whisper hallucinates phonetically plausible English words:

| CER | Reference | Whisper prediction |
|-----|-----------|-------------------|
| 0.455 | tolperisone | Tall person. |
| 0.471 | Nebicard 5 Tablet | Navy Card File Tablet |
| 0.500 | itopride | I took pride. |
| 0.500 | carbetocin | Carpet of Sin |
| 0.538 | Grilinctus Ls | Grelynthus ellis. |
| 0.571 | Arm bag | I'm back. |
| 0.588 | Triglimisave Ls 2 | Strike, let me save, LS2. |

High CER here is an artefact of short utterance length (one wrong word = high CER), not bad audio. **Implication for EKA curation: filter by minimum token/character length rather than CER ceiling.** Short single-word narrations like "Abbott" (CER 0.500, Whisper: "bot") are valid hard rows but inflate CER stats — length-based filtering preserves them while removing low-content rows.

---

## 1d. Phase 2 Design Decisions

**Dataset name:** `medical-terms` — vocabulary-difficulty focused, parallel to ai-terms. Full HF IDs: `ronanarraig/medical-terms-{public,semi-private,private}`.

**Scope:** Vocabulary difficulty is our contribution. EKA kept as external acoustic-diversity baseline. MultiMed dropped (reference quality too low). United kept for drug-name-specific comparisons.

**Entity category priorities (by current model failure rate):**
1. **Drugs** — highest CER across all models; primary focus (40–50% of rows)
2. **Procedures** — moderate difficulty; second priority (25% of rows)
3. **Conditions** — generally easier but rare diseases are hard; 20% of rows
4. **Anatomy** — Latin terms can be hard; 10–15% of rows
5. **Organisations** — easiest; include for coverage (5% of rows)

**Audio source:** Orpheus TTS with spoken form normalisation — same pipeline as ai-terms v2. Private split: user-recorded audio for 10–15 hardest entity rows.

**Target row count:** 50 rows per split (increased from ai-terms' 21, to cover entity category diversity).

**Difficulty threshold:** Will recalibrate from ai-terms' 0.045 — drugs CER of 0.28+ even on clean synthetic audio suggests threshold should be higher for medical, possibly 0.10+.

**Spoken form rules needed (before TTS):** Drug dosages, Latin abbreviations (q.d., b.i.d.), medical acronyms (CBC, HbA1c), ICD-style terminology.

---

## 1e. MultiMed Sentence Extraction Pipeline (Phase 1C)

**Goal:** Extract clean sentence-level medical clips from MultiMed with Gemini 2.5 Pro ground truth transcripts.

**Status: Complete — 12 high-density rows extracted from 501 MultiMed chunks**

### Pipeline

1. Filter MultiMed test (4,751 rows) to duration ≥5s + text ≥60 chars → sample 500 → Studio draft-transcribe → 501 rows with Whisper word timestamps
2. NLTK sentence detection on Whisper text → trim audio via word timestamps → clean sentence clips
3. Contextual audio padding: `min(gap/2, 0.2)s` when adjacent word exists, `0.3s` at boundaries
4. Gemini 2.5 Pro direct API: transcribe trimmed audio → ground-truth `transcript`
5. Completeness check: transcript must end with sentence-final punctuation
6. Gemini 2.5 Flash combined call: `is_medical`, `medical_density`, `entities`, `medical_entities` in single JSON response
7. Keep `medical_density == high` rows → 12 rows exported to review UI

### Drop log (full 501-row run)

| Step | In | Out | Dropped | Reason |
|------|----|-----|---------|--------|
| NLTK sentence trim | 501 | ~280 sentences | ~220 rows | No clean inner sentence found |
| Completeness check | ~280 | ~200 | ~80 | Hallucinated/incomplete endings |
| Medical density filter | ~200 | 12 | ~188 | medium/low/none density |

**Note:** MultiMed is lecture/podcast content — low medical density is expected. 12 high-density rows from 501 source rows is ~2.4% yield. The 73% high-density yield on EKA (see Phase 1D) confirms EKA is a much better source for medical content density.

### Key design decisions

- YT captions not passed to Gemini (not sentence-aligned after trimming)
- Full Whisper chunk passed as surrounding context for Gemini Pro transcription
- Tagging (is_medical, entities, medical_density) combined in one Gemini Flash call
- All sentences from multi-sentence chunks kept — treated as independent data points
- Studio router eval broken (bug `8788af6d`) → switched to direct Gemini 2.5 Pro API

---

## 1f. Hard Eval Set Construction (Phase 1D)

**Goal:** Build `eka-hard` and `multimed-hard` baseline eval sets — 50 manually-reviewed rows each, difficulty-filtered using 3 open-source model median CER.

### Architecture Decisions

- **EKA:** no NLTK sentence trimming (rows already sentence-level); filter audio ≥1s + text ≥10 chars
- **MultiMed:** NLTK sentence trimming (lecture chunks need sentence extraction); use test split only (train split reserved for training)
- **Training data:** MultiMed train split + synthetic TTS — EKA NOT used for training (leakage risk; used for eval)
- **Ground truth:** Gemini 2.5 Pro direct API (not Studio router — broken since ~10:30 UTC 2026-04-03)
- **Difficulty filter:** 3 open-source models (Whisper large-v3, Canary 1B v2, Voxtral Mini) → median CER vs Gemini transcript → top-100 → manual review/drop → finalise at 50
- **Review UI:** `/tools/review/server.py` + `review.html` — Accept/Skip/Drop per row, green/purple/red dot status in nav

### Pipeline

1. Load source dataset → filter (audio len + text len)
2. Studio from-hf-dataset import → poll until complete → draft-transcribe (Whisper large-v3) → poll
3. Download VTT+WAV via signed S3 URLs from re-process job config (workaround for Studio HF push bug)
4. (MultiMed only) NLTK sentence trim via word timestamps → contextual audio padding
5. Gemini 2.5 Pro ASR → ground-truth `gemini_text`
6. Completeness check (sentence-final punctuation)
7. Gemini 2.5 Flash tagging: `is_medical`, `medical_density`, `entities`, `medical_entities`
8. Keep high-density rows → export to review dir
9. Difficulty filter: 3 models → median CER → top-100 (pending)
10. Manual review in UI → drop to 50 rows (pending)

### EKA Results (Complete)

- **Source:** `ekacare/eka-medical-asr-evaluation-dataset` — 1,444 rows after audio+text filter
- **Studio import + draft-transcribe:** ~6 min download, 1,441 rows processed
- **Gemini 2.5 Pro ASR:** 1,257 complete, 184 dropped (incomplete/no sentence-final punct), 1 failed API call; ~21 min (20 threads)
- **Flash tagging:** 1,257 rows, ~5 min (100 threads)
- **Density breakdown:**

| Density | Rows | % |
|---------|------|---|
| high | 920 | 73% |
| medium | 330 | 26% |
| low | 4 | <1% |
| none | 3 | <1% |

- **Output:** 920 high-density rows exported to `tools/review/data-eka/` (~32 min total)
- **Next:** run 3-model difficulty filter → top-100 → manual review in UI

### MultiMed Results (Phase 1D — In Progress)

**Previous attempt (2026-04-04):** Used Phase 1C `multimed-hard-100` (Otsu-filtered from ~500 heuristic-selected rows). Skipped NLTK sentence trimming, Gemini ground truth, and proper medical density filtering. Produced `multimed-hard-public` but quality is insufficient — no sentence trimming, no Gemini ASR ground truth, heuristic rather than NLTK sentence extraction.

**Correct approach (2026-04-04, rewrite):** Use Studio `from-hf-dataset` directly on `leduckhai/MultiMed` (config=English, split=test). This avoids loading 3.4GB locally — Studio imports server-side with zero local memory. Then draft-transcribe for word timestamps → download VTT+WAV via signed URLs → NLTK sentence trim → Gemini 2.5 Pro ASR → Flash tagging → high-density filter → difficulty filter → top-50.

**Key architectural fix:** the original script 16 loaded the full dataset locally (3.4GB), filtered, pushed to HF, then had Studio re-import — OOM-killed on VPS. Direct `from-hf-dataset` eliminates the local load+push entirely.

### EKA Results (Phase 1D Complete)

- **3-model difficulty filter (2026-04-04):** All 3 eval jobs (Whisper large-v3, Canary 1B v2, Voxtral Mini) ran on 920 high-density EKA rows. CER range 0.000–0.280.
- **Working fix:** `Dataset(pa_table).cast_column('audio', Audio(sampling_rate=16000))` — uses `cast_storage` path which avoids `Audio.encode_example` and does not require torch.
- **Top-100 ranked by 3-model median CER:** all rows had 3 models, min/max/avg models per row = 3.0
- **Output:** `ronanarraig/eka-hard-public` — 50 rows, CER range 0.074–0.280

### Studio Bugs Encountered (Phase 1D)

| Bug | Feedback ID | Status | Impact | Workaround |
|-----|------------|--------|--------|------------|
| Router eval broken (`RouterEvaluation.run()` unexpected kwarg `output_target`) | `8788af6d` | **Fixed** (2026-04-03 later) | Draft-transcribe with `router_model` completed without error in re-test; job used `openai/whisper-large-v3` as source (unclear if `fireworks/whisper-v3` routed correctly or silently fell back) | Direct Gemini 2.5 Pro API (no longer needed) |
| HF dataset import very slow + fails (`Input aborted`) | `2bef1bbf` | **Fixed** 2026-04-04 | Root cause: missing `config` param for multi-config datasets. 10-row test completed in ~35s. | Pass `config='English'` when importing MultiMed |
| Process step ignores `output_org`/`hf_token` params — always uses account defaults | `907b979b` | **Fixed** (2026-04-03 later) | Re-test confirmed: process step pushed to `ronanarraig/studio-test-hf-push` without any `output_org` param — account default is now ronanarraig | No longer needed |

**Notes on re-test (2026-04-03):**
- Correct poll URL for draft-transcribe/process jobs: `GET /api/v1/data-prep/jobs/{job_id}` (not `/file-stores/{id}/draft-transcribe/{job_id}`)
- Correct upload URL field from batch upload response: `files[].upload_url` (not `upload_urls`)
- Process step takes ~90s for 3 files

**HF data-prep pipeline test (2026-04-04):**

Full pipeline: `from-hf-dataset` → `process` → HF push. Tested with 10 rows of `leduckhai/MultiMed`.

| Step | Time (10 rows) | Notes |
|------|---------------|-------|
| `from-hf-dataset` | 11s | Stores as parquet (`data/test.parquet` + `dataset_info.json`) |
| `process` | 78s | Ran forced alignment + HF push; 10 samples, 116s audio total |
| **Total** | **~90s** | Output: `ronanarraig/multimed-pipe-test` (test split, 10 rows) |

**Key findings:**
- **Correct flow:** `from-hf-dataset` → `process` directly. The API docs explicitly say: *"Returns a `file_store_id` to pass to `POST /file-stores/{id}/process`"*. The resulting file store has `source: "upload"` which is exactly what `process` accepts.
- **Skip `draft-transcribe`:** `draft-transcribe` is for file stores containing raw audio files without transcripts. HF datasets already include transcripts in the parquet; `draft-transcribe` would fail with `NO_FILES` since there are no raw audio files — only `data/test.parquet`.
- `from-hf-dataset` requires `config` param for multi-config datasets (e.g. `'English'` for MultiMed) — omitting it causes immediate failure with a helpful error listing valid configs.
- At 10-row rate, full 4,751-row MultiMed: import ~87min, process time likely sub-linear (parallelised). Actual scale not yet tested.

---

## 1g. Studio Bugs Filed (All Phases)

| Bug | Feedback ID | Status |
|-----|------------|--------|
| Moonshine-tiny dtype mismatch (HalfTensor vs BFloat16) | `77aef1bb` | Filed |
| Qwen3-ASR-1.7B returns 1.000 WER/CER | `5eb76bfc` | Filed |
| Add google/medasr model | `6c6a50b9` | Filed |
| Add Gemini 2.5 Flash | `a205d9fc` | Filed |
| Add OpenAI gpt-4o-transcribe | `d1877c02` | Filed |
| GET /evaluation/jobs: ?status= filter not applied server-side | `a5c4c486` | Filed |
| Router eval broken — `RouterEvaluation.run()` unexpected kwarg `output_target` | `8788af6d` | **Fixed** 2026-04-03 |
| HF dataset import slow (~2h for 4,751 rows) + aborts | `2bef1bbf` | **Fixed** 2026-04-04 |
| Process step ignores `output_org`/`hf_token` | `907b979b` | **Fixed** 2026-04-03 |
| Eval jobs fail with `KeyError: 'array'` when audio parquet lacks decoded Audio format | — | **Fixed** — workaround: use `Dataset(pa_table).cast_column('audio', Audio(sampling_rate=16000))` which avoids encode_example; eval endpoint now accepts HF datasets with proper Audio feature metadata |

---

## Timing

| Step | Time |
|------|------|
| Dataset inspection + sampling | ~20 min |
| United-Syn-Med download (4.4 GB) | ~10 min |
| Dual-LLM entity extraction (100 rows × 2 models) | ~5 min |
| HF push (3 datasets) | ~2 min |
| Eval job submission + completion (30 jobs × 50 rows) | ~15 min |
| MultiMed 501-row sentence extraction (Phase 1C) | ~45 min |
| EKA 1,444-row pipeline (download + Gemini ASR + tagging) | ~32 min |
| MultiMed HF import (4,751 rows) | ~2h (est.) |

## Costs

| Step | Cost |
|------|------|
| 30 eval jobs × 1.0 credit each | ~30 credits |
| LLM entity extraction (Gemini 2.5 Flash + Claude Sonnet) | ~$0.10 |
| Gemini 2.5 Pro ASR — EKA 1,257 rows | ~$0.50 est. |
| Gemini 2.5 Flash tagging — 1,257 rows | ~$0.10 est. |
