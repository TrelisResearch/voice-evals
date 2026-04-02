# Medical ASR — Phase 1 Report

**Status:** Complete (2026-04-02)
**Goal:** Landscape survey of existing medical ASR datasets + baseline model evaluation

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

### Finding 4: MultiMed is usable but needs Otsu filtering first

Source is YouTube medical channels (lectures, interviews, podcasts, documentaries) with auto-generated captions as references. Some rows have entirely wrong reference transcriptions, inflating apparent model WER. However the audio itself is realistic — diverse speakers, natural conditions, genuine medical content. **Plan: run Otsu thresholding on per-sample CER (using Whisper as reference model) to separate good from garbage rows, then use the filtered subset as a secondary baseline.** The high overall WER (17–23%) is partly reference noise, not purely model error.

### Finding 4b: United-Syn-Med TTS quality is poor

Prosody is flat and stress patterns on polysyllabic drug names are wrong (e.g. "Nitrazepam is a benzodiazepine medication commonly used to treat insomnia" — incorrect emphasis). This likely flatters models somewhat compared to natural speech, and undersells drug-name difficulty. Our Phase 2 dataset using Orpheus TTS with spoken form rules should produce substantially better audio quality.

### Finding 5: Qwen3-ASR-1.7B non-functional, Moonshine-tiny broken on Studio

Qwen3-ASR-1.7B returns 1.000 WER/CER on all datasets — likely empty output or Chinese characters. Studio bug filed (`5eb76bfc`). Moonshine-tiny dtype mismatch in Studio — bug filed (`77aef1bb`). Both need fixing before Phase 3 fine-tuning targets can be evaluated.

### Finding 6: Model rankings are consistent across datasets

The EKA and United rankings correlate well (Gemini/AssemblyAI top, Deepgram/Canary bottom). This is a positive sign — suggests our Phase 2 dataset doesn't need to be very large to produce stable rankings.

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

## 1e. Studio Bugs Filed

| Bug | Feedback ID | Status |
|-----|------------|--------|
| Moonshine-tiny dtype mismatch (HalfTensor vs BFloat16) | `77aef1bb` | Filed |
| Qwen3-ASR-1.7B returns 1.000 WER/CER | `5eb76bfc` | Filed |
| Add google/medasr model | `6c6a50b9` | Filed |
| Add Gemini 2.5 Flash | `a205d9fc` | Filed |
| Add OpenAI gpt-4o-transcribe | `d1877c02` | Filed |

---

## Timing

| Step | Time |
|------|------|
| Dataset inspection + sampling | ~20 min |
| United-Syn-Med download (4.4 GB) | ~10 min |
| Dual-LLM entity extraction (100 rows × 2 models) | ~5 min |
| HF push (3 datasets) | ~2 min |
| Eval job submission + completion (30 jobs × 50 rows) | ~15 min |

## Costs

| Step | Cost |
|------|------|
| 30 eval jobs × 1.0 credit each | ~30 credits |
| LLM entity extraction (Gemini 2.5 Flash + Claude Sonnet) | ~$0.10 |
