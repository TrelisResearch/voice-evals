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

### 1b. Model evaluation via Trelis Studio

Run all open-source models on all three pilot sets. Run proprietary models on all three — United-Syn-Med is drugs-only synthetic but that's still a useful signal: if medical-specialist proprietary models (Speechmatics, Deepgram) aren't beating general models on simple drug-name audio, that's a meaningful finding.

**Open-source models:**

| Model | Studio ID | Notes |
|-------|-----------|-------|
| Whisper large-v3 | `openai/whisper-large-v3` | Universal baseline |
| Canary 1B v2 | `nvidia/canary-1b-v2` | Top open-source on HF leaderboard |
| Voxtral Mini 3B | `mistralai/Voxtral-Mini-3B-2507` | Recent multimodal |
| Qwen3-ASR 1.7B | `Qwen/Qwen3-ASR-1.7B` | Recent; strong at size |
| Moonshine-tiny | `UsefulSensors/moonshine-tiny` | Pre-fine-tune baseline; Phase 3 target |
| Google MedASR | `google/medasr` | Medical specialist; *pending Studio support* |

**Proprietary models (semi-private pilot sets only):**

| Model | Studio ID | Notes |
|-------|-----------|-------|
| Gemini 2.5 Pro | `google/gemini-2.5-pro` | Strong general + medical |
| Gemini 2.5 Flash | *(pending Studio support)* | Faster/cheaper |
| OpenAI transcribe | *(pending Studio support)* | gpt-4o-transcribe |
| Speechmatics Ursa 2 | `speechmatics/ursa-2-enhanced` | Medical-specialist claim |
| Deepgram Nova 3 | `deepgram/nova-3` | Medical-specialist claim |
| AssemblyAI Universal 3 | `assemblyai/universal-3-pro` | Competitive general |
| ElevenLabs Scribe v2 | `elevenlabs/scribe-v2` | Worth including |

*Studio model requests submitted 2026-04-02: google/medasr (`6c6a50b9`), Gemini 2.5 Flash (`a205d9fc`), OpenAI gpt-4o-transcribe (`d1877c02`).*

**Metrics:** WER, CER, entity CER on EKA (only set with entity annotations). Entity CER is primary signal.

### 1c. Sample inspection

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

### 1f. Curated baselines: EKA-hard and MultiMed-hard

Build permanent public + private benchmark splits from EKA and MultiMed using the same difficulty-filtering approach as ai-terms v2. Target: 50 public + 50 private rows per dataset.

**Key lesson from Phase 1 run:** EKA contains a large proportion of single-word and short-phrase narrations (single drug names, single symptoms). These inflate CER artifically and are less interesting than sentence-level utterances — a model getting "carbetocin" wrong tells you less than a model getting it wrong in a clinical sentence. **Next run should pre-filter EKA to sentence-length rows before sampling 500**, to ensure the pool reflects realistic clinical speech rather than isolated term narrations. This will also reveal broader model failure modes (not just drug name recall but in-context transcription errors).

**Revised pipeline (next run):**

1. **Pre-filter EKA to sentence-length rows** — filter the full 3,619-row EKA pool to proper sentences before sampling. Heuristic: ≥ 5 tokens AND length ≥ 60 chars. Sample 500 from this filtered pool (stratified by recording_context). Single-word/short-phrase narrations excluded upfront, not post-hoc.
2. **CER filter + floor** — Whisper CER on the 500. Otsu ceiling + 5% floor. No char-length floor needed since pool is already sentence-filtered.
3. **Difficulty filter before entity extraction** — Canary 1B v2 + Voxtral Mini 3B; median CER across 3 models; top 100. Open-source only. Saves ~70% LLM cost vs extracting on full filtered set.
4. **Entity extraction** — dual-LLM (Gemini 2.5 Flash + Claude Sonnet, agreed-only) on top 100 only.
5. **Split 50 public + 50 private** — entity deduplication across splits.
6. **Push to HF** — `ronanarraig/eka-hard-public`, `ronanarraig/eka-hard-private`

**MultiMed: dropped.** Reference quality not reliably fixable via CER filtering — auto-caption misalignment and speaker labels persist. Standard YouTube content is not fair use for ML eval data.

**Second real-speech source (to find):** audit CC-BY audio sources — medical YouTube CC-BY channels, open courseware (MIT, Johns Hopkins), CC-BY medical podcasts. Pick 1–2 that have natural clinical speech, reliable transcripts, and clean licences. Process through Trelis Studio.

**Private split rules:** open-source models only. Never submit private splits to proprietary APIs (Gemini, Speechmatics, Deepgram, etc.).

### 1g. Phase 1 report

Document in `reports/phase1.md`:
- Dataset quality verdicts (keep / conditionally use / discard)
- Model leaderboard across all three pilot sets
- Key failure modes with examples
- Phase 2 design decisions with rationale
- Timing and costs

---

## Phase 2 — Our Medical ASR Test Set

**Goal:** Build our own vocabulary-difficulty medical speech benchmark — difficulty-filtered, entity-annotated, three-tier.

**Name:** Decided at end of Phase 1. Working assumption: `ronanarraig/medical-terms-{public,semi-private,private}`.

### 2a. Spoken form rules (before any TTS)

Medical text requires more normalisation than ai-terms before TTS will render it correctly:

| Pattern | Example | Spoken form |
|---------|---------|-------------|
| Drug dosages | `10mg`, `250mcg` | "ten milligrams", "two hundred fifty micrograms" |
| Latin abbreviations | `q.d.`, `b.i.d.`, `p.r.n.` | "once daily", "twice daily", "as needed" |
| Medical acronyms | `T2DM`, `CBC`, `BP` | "type two diabetes mellitus", "complete blood count", "blood pressure" |
| Lab values | `HbA1c 7.2%` | "HbA1c seven point two percent" |
| ICD codes | leave out or spell context | handle case by case |

Build `spoken_form_rules.md` (parallel to tricky-tts) before generating any audio.

### 2b. Entity seed lists

Build entity lists from public domain sources — no licence concerns:

| Entity type | Source | Notes |
|-------------|--------|-------|
| Drugs (approved) | FDA Orange Book + NDC database | Generic + brand names; bias toward rare/novel |
| Pipeline drugs | Pharmaceutical earnings call transcripts (names only — not copyrightable) | Novel Phase 2/3 drugs not yet in FDA labels; hardest for models |
| Conditions | ICD-10 code descriptions | Systematic; sample across specialties |
| Procedures | CPT code descriptions | Surgical, diagnostic, imaging |
| Anatomy | Standard anatomical terms | Gray's Anatomy out of copyright |
| Organisations | Public hospital/pharma/regulator lists | FDA, NICE, named hospital systems |

Target ~300–500 candidate entities total (need buffer since most will be filtered out as too easy).

### 2c. Context seed library

**Why context diversity matters:** uniform sentence templates make the benchmark gameable and unrepresentative. Entities should appear in varied clinical registers.

**Approach: mine entity + context together from the same source.**

Slot-filling templates (entity replaced with `[DRUG]` etc.) are not useful unless the replacement term is semantically compatible with the surrounding sentence — "particularly the [ANATOMY] information as Otitis media impacts the middle ear" filled with "femur" is nonsense. The Phase 1 context templates extracted from EKA/MultiMed are retained as **register examples only** (to show the LLM what clinical language looks like), not as fill-in-the-blank templates.

**Actual approach:** when building the entity seed list (step 2b), simultaneously mine real example sentences containing each entity from public-domain sources (FDA drug labels, PubMed abstracts, open case reports). Each seed entry = `{entity, category, example_sentence}`. The example sentence provides natural context specific to that entity. LLM generation in step 2d then uses the example sentence as a style anchor for that entity, rather than a generic template.

This ensures context is always semantically coherent with the entity.

### 2d. Text corpus generation

Generate ~150–200 candidate utterances using LLM, one per seed entity:
- For each entity: use its mined example sentence as context anchor
- LLM generates a fresh clinical utterance in a similar register, with the entity naturally embedded
- Vary clinical register across rows (prescription, discharge summary, radiology, referral, etc.) by instruction in the prompt
- Each utterance: 1–2 entities, at least one rare/difficult term, ≤ 30s when spoken
- Entity distribution: 40% drugs, 30% procedures, 20% conditions, 10% anatomy

### 2e. TTS audio generation

Orpheus 3B / tara via Studio. Apply spoken form rules before generation. Audit 5–10 samples per entity category before full run (check pronunciation of drug names, Latin terms).

### 2f. Difficulty filtering

Median entity CER ≥ threshold across 3 filter models. Threshold calibrated from Phase 1 findings (may differ from ai-terms' 0.045).

### 2g. Split assignment + leakage check

30–50 rows per split after filtering. Entity deduplication across splits. N-gram overlap check.

### 2h. Full benchmark evaluation

~15–20 models on final splits. Publish leaderboard.

### 2i. Phase 2 report

Document in `reports/phase2.md`.

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
