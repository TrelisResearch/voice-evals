# Medical ASR — Project Roadmap

**Goal:** Build an entity-aware medical ASR benchmark, evaluate leading models, and fine-tune lightweight models on medical speech data to demonstrate improvement.

**Entity categories:** drugs/medications, procedures, conditions/diagnoses, anatomy, organisations

**Dataset architecture:** Three-tier (public / semi-private / private), following the ai-terms pattern.
- Public: fully open, published on HuggingFace
- Semi-private: proprietary models evaluated privately via Studio; dataset shared
- Private: open-source models only; never published

**Dataset naming:** Deferred to end of Phase 1. Name should reflect actual scope — if vocabulary-difficulty focused, something like `medical-vocab-{public,semi-private,private}`; if broader, `medical-terms-{public,semi-private,private}`. Will decide once Phase 1 reveals what gap we're actually filling.

---

## Test Set Strategy

Our test set can contribute on two axes:

| Axis | Can we contribute? | How |
|------|--------------------|-----|
| **Vocabulary difficulty** — rare drug names, procedures, anatomy terms models haven't seen | Yes | Synthetic TTS + spoken form normalisation + difficulty filtering |
| **Acoustic/contextual diversity** — noise, accents, disfluencies, speaker variation | Limited | Potentially user recording for private split; otherwise rely on external datasets |

**Plan:** Our dataset = vocabulary-difficulty layer. External datasets (EKA, MultiMed) = acoustic diversity anchor. Phase 1 determines which external sets are good enough to keep as permanent comparison baselines alongside our own.

**Audio source decision (deferred to Phase 2):**
- Primary option: Orpheus TTS + spoken form normalisation (scalable, consistent, same pipeline as ai-terms)
- Private split option: user-recorded audio (single speaker, no PII, real acoustics, most credible for hardest entities)
- Decision depends on Phase 1 findings — if existing synthetic datasets already cover vocabulary difficulty adequately, we shift focus to acoustic diversity contributions

---

## Training Data Strategy (Phases 3–5)

**Preferred approach:** Find unlabeled real medical audio → run strong teacher model → pseudo-labeled training set. Far superior to synthetic for generalisation; real acoustic conditions; diverse speakers. **Blocker: PII.** Worth searching for de-identified sources (MIMIC has notes, not audio; some hospital systems have released under research agreements).

**Fallback:** Synthetic TTS training data (same Orpheus pipeline as test set, different entity partition). Less credible but fully controllable.

**Train/test separation** (critical for credibility):
- Entity non-overlap: partition entity lists at generation time — train entities and test entities are disjoint
- Text source separation: different source documents for train vs test
- N-gram overlap check before any fine-tuning
- Document separation method publicly in dataset card

---

## Phase 1 — Landscape Survey + Baseline Evaluation

**Goal:** Run evals on existing public medical ASR test sets; understand current model performance; identify gaps that our Phase 2 dataset should fill.

### 1a. Build 50-row pilot test sets

Create three small eval sets (random 50-row samples), push to `ronanarraig/` on HF as private datasets:

| Pilot set | Source | HF ID | Audio notes | Entity labels |
|-----------|--------|-------|-------------|--------------|
| `eka-pilot` | `ekacare/eka-medical-asr-evaluation-dataset` EN test | `ronanarraig/eka-pilot` | Embedded parquet | Yes — 5 types |
| `multimed-en-pilot` | `leduckhai/MultiMed` EN test | `ronanarraig/multimed-en-pilot` | Embedded parquet | No |
| `united-synmed-pilot` | `united-we-care/United-Syn-Med` test | `ronanarraig/united-synmed-pilot` | Download test.tar.gz (4.4 GB), extract 50 files | No |

Sampling: stratified where possible (EKA: across entity types and recording contexts; MultiMed: across speaker roles; United: random).

### 1b. Calibrated datasets — EKA rebuild + CC-BY second source

**Goal:** Get the real-speech test sets right before running any further evals. Two workstreams.

#### 1b-i. EKA rebuild (sentence-filtered)

EKA has a `recording_context` column. `narration_entity` rows (2,206) are mostly single drug/entity narrations — not useful as sentences. `narration_sentence` rows (1,303) are full clinical sentences — this is the target pool. Also include any row with text ≥ 60 chars regardless of context.

**Pipeline:**
1. Filter full EKA to sentence pool (`narration_sentence` OR len ≥ 60 chars)
2. Stratified-sample 500 from that pool (by recording_context)
3. Whisper CER filter: Otsu ceiling + 5% floor
4. Difficulty filter: Canary + Voxtral; median CER top 100
5. Entity extraction (dual-LLM) on top 100 only
6. Split 50 public + 50 private with entity dedup
7. Push `ronanarraig/eka-hard-public` v2, `ronanarraig/eka-hard-private` v2

#### 1b-ii. CC-BY second real-speech source

**Step 1** — Identify medical audio categories and CC-BY/public-domain sources per category. Wide shortlist, quick licence + quality inspection.

**Step 2** — Download at most 3 sample audios across different categories.

**Step 3** — Trelis Studio draft-transcribe + data prep on each sample.

**Step 4** — LLM-clean a 10-segment subset: review one segment at a time, passing ~2 surrounding segments as context. Produce clean ground-truth references.

**Step 5** — User listens to and inspects the 10 cleaned samples; decide which source(s) to proceed with for full processing.

### 1c. MultiMed sentence extraction pipeline

**Goal:** Extract clean, sentence-level medical audio clips from MultiMed with high-quality ground truth transcripts. Prototype on test split (4,751 rows); scale to train split (25,512 rows) if results are good.

**Pipeline:**

1. **Filter**: duration ≥5s + text ≥60 chars (no sentence heuristic — YT captions too unreliable for boundary detection). Sample 500.
2. **Push to HF**: `multimed-sentences-500`
3. **Trelis Studio draft-transcribe**: Whisper large-v3 → word timestamps + transcript → `multimed-sentences-transcribed`
4. **NLTK sentence detection on Whisper text**: detect clean inner sentences (drop partial sentences at chunk start/end). Trim audio using word timestamps. Keep sentences ≥3s, ≥40 chars. Multiple sentences per chunk all kept.
5. **Gemini 3 Flash combined call** (audio + full Whisper chunk + trimmed Whisper sentence → single JSON response):
   - `transcript`: clean ground truth (Gemini listens to trimmed audio; Whisper chunk provides context for what surrounds the target sentence)
   - `is_medical`: bool
   - `medical_density`: low/medium/high
   - `entities`: list with text, category, char offsets
6. **Filter**: keep `medical_density == high` only
7. **Review UI**: human inspect/correct consensus transcripts
8. **Public/private split**: entity dedup → target 50 public + 50 private

**Drop points (observed on 45-row prototype):**
- Step 4: ~42% rows have no clean inner sentence
- Step 6: ~40% of surviving rows not high-density medical

**Scale-up path**: run steps 1–7 on full 25k train split → estimated ~14k clean sentences before density filtering.

**Key design decisions:**
- YT captions not passed to Gemini (not sentence-aligned after trimming)
- Full Whisper chunk (not just trimmed sentence) passed as context, with note that audio only covers the trimmed portion
- Tagging combined with consensus in one Gemini call to save cost/latency
- Gemini 3 Flash used throughout (not 2.5)

### 1d. Sample inspection

After eval runs, inspect high-CER samples from each dataset and model. Key questions:
- What is Whisper actually getting wrong — entity terms specifically, or general transcription quality?
- Are medical-specialist proprietary models (Speechmatics, Deepgram) measurably better on entity CER, or just on WER?
- What failure modes appear: hallucination, entity substitution, phonetically similar confusion?
- Is EKA too India-specific (drug brands, accents) to generalise as a baseline?
- Is MultiMed lecture-style too different from clinical dictation?

### 1d. Gap analysis + Phase 2 decisions

Based on evals:
- Which external datasets pass quality bar as permanent comparison anchors?
- What entity categories are hardest — prioritise those in Phase 2 build
- What's missing from existing sets that our vocabulary-difficulty dataset can fill?
- Finalise dataset name and scope
- Decide audio source: Orpheus TTS vs user-recorded vs hybrid
- Set target row count per split (likely 30–50 given entity diversity needed)

### 1d. EKA-hard + MultiMed-hard baseline eval sets

Build two public benchmark splits using an identical pipeline, so results are directly comparable on the leaderboard. Target: 50 public rows each. Private splits deferred.

**Datasets:**
- EKA: full 3,619 rows (`ekacare/eka-medical-asr-evaluation-dataset`)
- MultiMed: **test split only** (4,751 rows) — train split reserved for fine-tuning data

**Pipeline (identical for both):**

1. **Sentence filter** — keep rows with duration ≥ 3s AND text len ≥ 60 chars
2. **Draft-transcribe via Studio** — Whisper large-v3 router (Fireworks) → word timestamps
3. **NLTK sentence detection + audio trim** — find clean inner sentences, contextual padding (half inter-word gap, cap 0.2s; 0.3s at boundaries), drop clips < 3s or < 40 chars
4. **Gemini 2.5 Pro ASR** — audio only, no text context → completeness check → drop incomplete
5. **Gemini Flash tagging** — `is_medical`, `medical_density`, `entities` → keep `medical_density == high` only
6. **Difficulty filtering** — run Whisper large-v3, Canary 1B v2, Voxtral Mini via Studio → median CER vs Gemini transcript → take top-100 by median CER
7. **Manual review** — review UI with drop functionality; reviewer drops rows until 50 remain; correct any ground truth errors
8. **Push** — `ronanarraig/eka-hard-public`, `ronanarraig/multimed-hard-public`

**Difficulty metric rationale:** Gemini 2.5 Pro used as pseudo-ground-truth (best available reference). Filter models are open-source only. Gemini errors are invisible to the CER signal but caught by manual review on the final top-100.

**Review tool:** drop button added to review UI — reviewer works through top-100, drops low-quality or borderline rows, finalises at 50.

**Private split rules:** open-source models only. Never submit private splits to proprietary APIs.

### 1g. Phase 1 report

Document in `reports/phase1.md`:
- Dataset quality verdicts (keep / conditionally use / discard)
- Model leaderboard across all three pilot sets
- Key failure modes with examples
- Phase 2 design decisions with rationale
- Timing and costs

---

## Phase 2 — Our Medical ASR Test Set

**Goal:** Build a small (50-row) but high-quality, hard, vocabulary-difficulty medical speech benchmark — entity-annotated, difficulty-filtered, good breadth across medical domains.

**Name:** `ronanarraig/medical-terms-{public,semi-private,private}`

**Philosophy:** quality over volume. Inspect data at every step. Start with ~100 candidate rows, filter to 50 that are hard and representative.

---

### 2a. Define categories + mine (keyword, context) pairs

**Step 1 — Define high-level categories (~6–8):**
e.g. cardiology, oncology, neurology, pharmacology (drugs/dosing), infectious disease, surgery/procedures, anatomy, metabolic/endocrinology.

**Step 2 — Identify 2–3 key sources per category:**
For each category, find active-use sources where medical terminology appears in natural clinical context. Examples:
- Drug formularies, FDA drug labels (pharmacology)
- Clinical practice guidelines (cardiology, oncology, etc.)
- PubMed abstracts, open case reports
- Surgical procedure coding descriptions (CPT)
- Medical textbook excerpts (public domain)

**Step 3 — Mine (keyword, context) pairs:**
From each source, extract pairs: `{keyword, context_sentence, category, source}`. The context sentence is a real sentence from the source where the keyword appears naturally. Target ~15–20 pairs per category, ~100–150 total.

Inspect pairs before proceeding — check they represent real active use, not definition lists.

---

### 2b. LLM sentence generation

For each `(keyword, context)` pair, prompt an LLM to generate a fresh clinical sentence featuring the keyword naturally in context. Use the mined context sentence as a style anchor (not a template). Vary clinical register across rows (prescription note, discharge summary, radiology report, referral letter, clinical trial description).

Constraints per sentence: 1–2 entities, at least one rare/difficult term, ≤ 30s when spoken.

Inspect a sample (~20 rows) before full run.

---

### 2c. TTS audio — Kokoro via Studio

**Quick test first:** generate audio for 5–10 sentences, listen to output — check pronunciation of drug names, Latin terms, acronyms. Apply spoken form normalisation beforehand where needed (dosages, abbreviations, acronyms).

**Full run:** evenly distribute rows across all Kokoro voices supported by Studio (not random — even allocation per voice). This gives speaker diversity without bias.

Inspect a sample per voice before proceeding to difficulty filtering.

---

### 2d. Difficulty calibration

Run 3 open-source models (Whisper large-v3, Canary 1B v2, Voxtral Mini) via Studio on all ~100 rows. Compute median entity CER per row vs ground-truth text. Inspect distribution — check hard rows are hard for the right reasons (entity errors, not garbled audio). Set threshold; keep top ~50 by median entity CER.

---

### 2e. Manual review

Review UI — inspect each of the ~50 hard rows. Drop any with TTS pronunciation failures, awkward generated text, or borderline entity difficulty. Correct any ground truth errors.

---

### 2f. Proprietary model spot-check

Once 50 rows are finalised, run 1–2 top proprietary models (e.g. Gemini 2.5 Pro + AssemblyAI Universal 3 Pro) via Studio. Report overall CER and entity CER per category. This gives a quick benchmark anchor before full leaderboard evaluation.

---

### 2g. Phase 2 report

Document in `reports/phase2.md`: sources used, (keyword, context) mining approach, TTS voice distribution, difficulty calibration results, proprietary model spot-check, drop log at each step.

---

## Phase 3 — Training Data (~10 hours) + Fine-tuning

**Goal:** Train on medical speech data, demonstrate entity CER improvement.

### 3a. Training data source decision

Try in order of preference:
1. Unlabeled real medical audio + teacher-model pseudo-labels (if source found — PII-free)
2. Synthetic TTS (Orpheus, different entity partition from test set)

~10 hours = ~3,600 utterances at ~10s avg.

### 3b. Fine-tuning via Trelis Studio

- **Moonshine-tiny** — edge/mobile target; check if Studio supports or run locally
- **Whisper large-v3** — highest-quality; Studio LoRA training on H100

### 3c. Evaluation

Fine-tuned vs base on Phase 2 test set + any external baselines kept from Phase 1. Report entity CER per category.

**Report:** `reports/phase3.md`

---

## Phase 4 — Scale to 100 Hours

10x data, same pipeline. Decision gate: proceed to Phase 5 only if ≥10% relative entity CER reduction vs Phase 3.

**Report:** `reports/phase4.md`

---

## Phase 5 — Scale to 1,000 Hours (Conditional)

May need additional text sources (de-identified MIMIC, medical podcasts, CME recordings). Consider speaker diversity augmentation.

**Report:** `reports/phase5.md`

---

## Open Questions

| Question | Resolves in |
|----------|-------------|
| Which external datasets pass Phase 1 quality bar? | Phase 1d |
| Final dataset name and scope? | Phase 1d |
| Audio source: Orpheus TTS / user-recorded / hybrid? | Phase 1d |
| Target row count per split? | Phase 1d |
| Real unlabeled medical audio source available? | Phase 3a |
| Does Studio support Moonshine fine-tuning? | Phase 3b |
| Difficulty threshold for medical (vs 0.045 for ai-terms)? | Phase 2d |

---

## Key References

- **Google MedASR** — `google/medasr`, 105M Conformer, 5,000h physician dictations, 4.6% WER radiology vs 25.3% Whisper large-v3
- **EKA Medical ASR Eval** — `ekacare/eka-medical-asr-evaluation-dataset` — 3,619 EN rows, entity annotations, Indian accent focus
- **MultiMed EN** — `leduckhai/MultiMed` — 4,751 EN test rows, lecture/podcast style, MIT
- **United-Syn-Med** — `united-we-care/United-Syn-Med` — 79k rows, drugs only, synthetic, CC BY-SA 4.0
- **arXiv:2406.12387** — "Performant ASR Models for Medical Entities in Accented Speech" — entity CER methodology
- **PROFASR-BENCH** — entity-aware metrics, slice-wise reporting (OpenReview 2024)
