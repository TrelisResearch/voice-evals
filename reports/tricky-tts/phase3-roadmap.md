# Tricky TTS — Phase 3 Roadmap

## Goal

Complete the reference pipeline, evaluate all target models on the 10-row prototype, validate results, then scale to the full 48-row public split. Finish with a production-ready benchmark dataset ready for semi-private and private split calibration.

---

## Context: Where Phase 2 Left Off

**What works:**
- 48-row public split (`ronanarraig/tricky-tts-v2-public`) — difficulty filtered, spoken_form generated
- 10-row prototype subset (`ronanarraig/tricky-tts-proto-v4`) — ready for reference pipeline
- Orpheus reference audio via data prep (`ronanarraig/tricky-tts-proto-ref-orpheus-datap`) — 10 samples, 2.9 min, audio quality confirmed good
- `spoken_form_rules.md` — complete, covers all categories including chemical compounds

**What's blocked:**
- Orpheus eval endpoint recurring 504s (bug `6248f9de`) — needed to get Whisper ASR of Orpheus reference audio
- Gemini Flash TTS eval producing 0 samples (bug `e2bc5701`)

**The reference pipeline design (not yet complete):**
```
spoken_form text
    → Orpheus TTS (data prep endpoint — works)
    → Orpheus audio
    → Whisper large-v3 ASR          ← BLOCKED (need ASR on data prep audio)
    → reference_asr_transcript
    → reference_column in eval jobs
```

---

## Step 1: Unblock the Reference Pipeline

**Option A — Wait for Orpheus eval endpoint fix:**
Once bug `6248f9de` is resolved, re-run Orpheus eval on `ronanarraig/tricky-tts-proto-spoken-form` with `asr_model_id="openai/whisper-large-v3"` to get `asr_transcription` column = reference_asr_transcript.

**Option B — ASR on data prep audio directly:**
Download audio from `ronanarraig/tricky-tts-proto-ref-orpheus-datap`, run through Trelis ASR eval endpoint to get Whisper transcripts. Map back to original texts and build the reference dataset.

Option A is cleaner (one job, keeps alignment with existing scripts). Option B unblocks us now if the eval endpoint remains broken.

**Deliverable:** `ronanarraig/tricky-tts-proto-with-reference` — 10 rows:
- `text` (original written text)
- `spoken_form`
- `reference_asr_transcript` (Whisper ASR of Orpheus audio)
- `category`, `cer_reliable`

---

## Step 2: Validate Reference Quality

Before running all model evals, spot-check the reference_asr_transcript quality:

- Compute `ref_self_cer = CER(spoken_form, reference_asr_transcript)` for each row
- Flag rows where ref_self_cer > 0.5 (likely Orpheus loop/garble — raise threshold from previous 0.3 which was too strict)
- Inspect flagged rows manually — is the audio garbled or is it ASR formatting noise?
- Confirm that CER reference is cleaner than v4's spoken_form-direct approach

---

## Step 3: Evaluate All Target Models on Prototype

Run TTS eval jobs on `ronanarraig/tricky-tts-proto-with-reference` with `reference_column="reference_asr_transcript"` and `asr_model_id="openai/whisper-large-v3"`.

**Target models (discuss with user before running):**

| Model | Type | Priority | Notes |
|---|---|---|---|
| `elevenlabs/eleven-multilingual-v2` | proprietary | high | baseline from phase 2 |
| `unsloth/orpheus-3b-0.1-ft` | open-source | high | used as reference TTS — still evaluate as test model |
| `google/gemini-2.5-flash-tts` | proprietary | high | bug `e2bc5701` may be fixed |
| `openai/gpt-4o-mini-tts` | proprietary | high | strong performer in phase 2 |
| `cartesia/sonic-3` | proprietary | medium | phase 2 baseline |
| `kokoro` | open-source | medium | phase 2 baseline |
| `google/gemini-2.5-pro-tts` | proprietary | low | reference audio model — may bias if also tested |

Open question: should Orpheus be both the reference TTS and a test model? It's fine — the reference is its ASR transcript, not raw audio. Orpheus being evaluated against its own transcript is a valid "upper bound" data point.

---

## Step 4: Inspect Prototype Results

Before scaling:
- Compare CER rankings across models — do they match intuition?
- Check per-row ASR snippets for interesting failure modes
- Confirm phonetic rows still show meaningful signal (ASR prior concern)
- Confirm domain_specific rows don't loop/overflow for any models
- Confirm paralinguistics rows are appropriately low CER and check UTMOS

---

## Step 5: Scale to Full 48 Rows

Once prototype is validated:

1. Generate `spoken_form` for remaining 38 rows (already done in phase 2 — `ronanarraig/tricky-tts-v2-public` has `spoken_form` column)
2. Generate Orpheus reference audio for all 48 rows via data prep
3. Run Whisper ASR → `reference_asr_transcript` for all 48 rows
4. Push full eval dataset: `ronanarraig/tricky-tts-public` — 48 rows with `text`, `spoken_form`, `reference_asr_transcript`, `category`, `cer_reliable`
5. Run all target models
6. Produce final leaderboard

---

## Step 6: Semi-Private and Private Splits

Apply the same methodology to semi-private and private splits:
- Semi-private: same model list, different text rows
- Private: open-source models only (never send to proprietary APIs)
- Run entity/n-gram overlap check between splits before publishing

---

## Open Questions for Discussion

1. **Model list** — which models do we want in the final leaderboard? Any additions beyond the list above?
2. **Orpheus as reference + test** — comfortable with this, or use a different reference TTS (e.g. Gemini Pro)?
3. **Phonetic category** — CER is inherently noisy for this category (ASR spelling prior). Do we want to add a UTMOS-only flag for phonetic rows, similar to paralinguistics?
4. **Timing** — scale to full 48 rows immediately after prototype validation, or do semi-private/private in parallel?
5. **Dataset naming** — current names are under `ronanarraig/tricky-tts-*`. When do we migrate to `Trelis/` org?

---

## Artefacts Reference

| Dataset | Description | Status |
|---|---|---|
| `ronanarraig/tricky-tts-v2-public` | Full 48-row public split, text + spoken_form | ✅ Ready |
| `ronanarraig/tricky-tts-proto-v4` | 10-row prototype, text + spoken_form | ✅ Ready |
| `ronanarraig/tricky-tts-proto-ref-orpheus-datap` | Orpheus audio of spoken_form (data prep) | ✅ Ready |
| `ronanarraig/tricky-tts-proto-ref-gemini-pro` | Gemini Pro audio of spoken_form (inspection) | ✅ Ready |
| `ronanarraig/tricky-tts-proto-with-reference` | Prototype + reference_asr_transcript | ⏳ Phase 3 Step 1 |
| `ronanarraig/tricky-tts-public` | Full 48-row eval dataset with reference | ⏳ Phase 3 Step 5 |
