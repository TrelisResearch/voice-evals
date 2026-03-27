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

**Previously blocked (now resolved):**
- ~~Orpheus eval endpoint recurring 504s (bug `6248f9de`)~~ — **fixed as of 2026-03-27**
- ~~Gemini Flash TTS eval producing 0 samples (bug `e2bc5701`)~~ — **fixed as of 2026-03-27**

**The reference pipeline design (not yet complete):**
```
spoken_form text
    → Orpheus TTS (data prep endpoint — works)
    → Orpheus audio
    → Whisper large-v3 ASR          ← next step (Option A now unblocked)
    → reference_asr
    → reference_column="reference_asr" in eval jobs
```

---

## Step 1: Build the Reference Pipeline ✅ Complete

Both blockers are resolved. Use Option A (cleaner path):

Run Orpheus TTS eval on `ronanarraig/tricky-tts-proto-v4` with `asr_model_id="openai/whisper-large-v3"` to get Whisper ASR transcripts of Orpheus audio. Store result as column `reference_asr`.

**Deliverable:** `ronanarraig/tricky-tts-proto-with-reference` — 10 rows:
- `text` (original written text)
- `spoken_form`
- `reference_asr` (Whisper ASR of Orpheus audio)
- `category`, `cer_reliable`

---

## Step 2: Validate Reference Quality

Before running all model evals, spot-check the `reference_asr` quality:

- Compute `ref_self_cer = CER(spoken_form, reference_asr)` for each row
- Flag rows where ref_self_cer > 0.5 (likely Orpheus loop/garble — raise threshold from previous 0.3 which was too strict)
- Inspect flagged rows manually — is the audio garbled or is it ASR formatting noise?
- Confirm that CER reference is cleaner than v4's spoken_form-direct approach

---

## Step 3: Evaluate All Target Models on Prototype ✅ Complete

Run TTS eval jobs on `ronanarraig/tricky-tts-proto-with-reference` with `reference_column="reference_asr"` and `asr_model_id="openai/whisper-large-v3"`.

**Target models (confirmed):**

| Model | `tts_model_type` | Notes |
|---|---|---|
| `elevenlabs/eleven-multilingual-v2` | `auto` | baseline from phase 2 |
| `openai/gpt-4o-mini-tts` | `auto` | strong performer in phase 2 |
| `cartesia/sonic-3` | `auto` | baseline from phase 2 |
| `google/gemini-2.5-flash-tts` | `auto` | bug `e2bc5701` resolved — ready |
| `google/gemini-2.5-pro-tts` | `auto` | also used as reference audio; fine to test |
| `unsloth/orpheus-3b-0.1-ft` | `orpheus` | reference TTS + test model (valid — ref is its ASR transcript) |
| `kokoro` | `kokoro` | baseline from phase 2 |
| `piper` | `piper` | new in phase 3 |
| `chatterbox` | `chatterbox` | new — not evaluated in phase 2 |

All models: `asr_model_id="openai/whisper-large-v3"`, `reference_column="reference_asr"`, UTMOS enabled.

---

## Step 4: Inspect Prototype Results ✅ Complete (see phase3.md)

**Phase 3 status: complete pending Piper/Chatterbox ASR fix in Studio.**
Re-run those two jobs once fixed — no dataset changes needed.

Before scaling:
- Compare CER rankings across models — do they match intuition?
- Check per-row ASR snippets for interesting failure modes
- Confirm phonetic rows still show meaningful signal (ASR prior concern)
- Confirm domain_specific rows don't loop/overflow for any models
- Confirm paralinguistics rows are appropriately low CER and check UTMOS

---

## Step 5: Scale to Full 48 Rows

> Deferred — next phase will be a **private 48-row split** using the same reference pipeline methodology rather than scaling the public split.

Once prototype is validated:

1. Generate `spoken_form` for remaining 38 rows (already done in phase 2 — `ronanarraig/tricky-tts-v2-public` has `spoken_form` column)
2. Generate Orpheus reference audio for all 48 rows via data prep
3. Run Whisper ASR → `reference_asr` for all 48 rows
4. Push full eval dataset: `ronanarraig/tricky-tts-public` — 48 rows with `text`, `spoken_form`, `reference_asr`, `category`, `cer_reliable`
5. Run all target models
6. Produce final leaderboard

---

## Step 6: Semi-Private and Private Splits

Apply the same methodology to semi-private and private splits:
- Semi-private: same model list, different text rows
- Private: open-source models only (never send to proprietary APIs)
- Run entity/n-gram overlap check between splits before publishing

---

## Decisions Made

1. **Model list** — ElevenLabs, GPT-4o-mini-tts, Cartesia Sonic-3, Gemini Flash-TTS, Gemini Pro-TTS, Orpheus, Kokoro, Piper, Chatterbox (all models Studio supports)
2. **Orpheus as reference + test** — confirmed fine
3. **CER for all categories** — yes, including phonetic; UTMOS for all rows
4. **Dataset naming** — TBD when migrating to `Trelis/` org

---

## Artefacts Reference

| Dataset | Description | Status |
|---|---|---|
| `ronanarraig/tricky-tts-v2-public` | Full 48-row public split, text + spoken_form | ✅ Ready |
| `ronanarraig/tricky-tts-proto-v4` | 10-row prototype, text + spoken_form | ✅ Ready |
| `ronanarraig/tricky-tts-proto-ref-orpheus-datap` | Orpheus audio of spoken_form (data prep) | ✅ Ready |
| `ronanarraig/tricky-tts-proto-ref-gemini-pro` | Gemini Pro audio of spoken_form (inspection) | ✅ Ready |
| `ronanarraig/tricky-tts-public` | 10-row public split, text + spoken_form + reference_asr + audio | ✅ Ready |
| `ronanarraig/tricky-tts-public` | Full 48-row eval dataset with reference | ⏳ Phase 3 Step 5 |
