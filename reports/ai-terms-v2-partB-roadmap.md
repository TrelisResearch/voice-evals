# AI Terms v2 — Part B: Roadmap

Status as of 2026-03-18. Picks up from Part A (build complete, known issues outstanding).

---

## Immediate Priority: Small-Scale Pilot (Public Split Only)

Before rebuilding the full v2 dataset, run a small pilot to validate the TTS pipeline end-to-end. The goal is to catch issues cheaply on a handful of rows before committing to 100.

### Step 1: Number formatting audit

The candidate texts use digits (e.g. "3.1", "70B", "$4 million") rather than spelled-out words. It is not confirmed whether Orpheus 3B reads these correctly and consistently.

**Test:**
1. Pick ~5 representative rows from the existing 100 that contain numbers in different forms (plain integers, decimals, dollar amounts, parameter counts like "0.6B")
2. Generate TTS audio via Studio
3. Listen to each and check: does Orpheus read "0.6B" as "zero point six billion"? "3.1" as "three point one"? "$4 million" as "four million dollars"?

**Expected outcomes and actions:**

| Outcome | Action |
|---------|--------|
| Orpheus reads digits correctly and naturally | Keep digit format in both reference transcript and TTS input. No change needed. |
| Orpheus mis-reads some digit patterns | Add a **TTS transcript** column: identical to the reference text except numbers are written as words. TTS uses the word version; CER evaluation uses the digit version. |

**If a TTS transcript column is needed**, the rewrite must be minimal — only numbers converted to words, no other changes to wording, punctuation, or entity names. This preserves alignment between the two versions and avoids introducing new entity-name differences.

### Step 2: Small pilot build

Once number formatting is resolved:

1. Select ~10 candidate rows from the existing 100-row pool (a mix of easy and hard by median CER)
2. Generate TTS audio with the confirmed number format
3. Run Studio data-prep pipeline (draft-transcribe → process with forced alignment)
4. Inspect the public-split dataset in Studio: listen to audio, check reference transcript alignment, check entity char offsets
5. Run 2–3 models on this mini dataset as a sanity check

This is the acceptance gate. Only proceed to full rebuild if audio quality and pipeline output look clean.

---

## Phase 2: Full v2 Rebuild

Once the pilot passes:

1. **Re-generate TTS audio** for all 100 candidate rows using the confirmed transcript format
2. **Manual QA pass**: listen to each row; flag rows where entity names are mispronounced — these get dropped from the candidate pool (do not attempt text-to-speech correction of proper nouns)
3. **Re-run difficulty filtering** on the cleaned audio pool (same 3 filter models)
4. **Re-assign splits** (public 21 / semi-private 21 / private 21) with entity deduplication
5. **Re-run full benchmark** (25+ models) on the final splits
6. Address the **30s max_duration dropout**: either request Studio to raise the limit, or trim audio during data-prep

### Target dataset naming (final)
- `ronanarraig/ai-terms-v2-public`
- `ronanarraig/ai-terms-v2-semi-private`
- `ronanarraig/ai-terms-v2-private`

*(Move to `Trelis/` namespace once Trelis namespace write access is available on the build machine.)*

---

## Studio Feature Requests

The following Studio features are needed to complete v2 properly and will also benefit all future datasets:

### 1. Re-recording rows via UI (high priority)

After TTS generation, some rows will have entity mispronunciations that can only be fixed by human re-recording. Studio needs a UI workflow to:
- Flag a row for re-recording
- Record a replacement take directly in the browser (or upload an audio file)
- Replace the TTS audio for that row while keeping all metadata (text, entities, timestamps) intact
- Re-run forced alignment on the new audio

This is essential for a final QA pass before publishing any v2 split.

### 2. Import HuggingFace datasets into Studio data viewer (high priority)

Currently datasets must be uploaded via the API. A UI flow to import a dataset directly from a HF dataset ID (with optional split selection) would significantly reduce friction during iterative builds.

### 3. Drop rows from datasets in Studio data viewer (medium priority)

After inspecting rows in the data viewer, there is no way to delete individual rows without rebuilding the dataset externally. A row-level delete action in the data viewer would allow inline cleanup during QA.

---

## Future Datasets

Once the v2 pipeline is stable and validated:

| Dataset | Description | Notes |
|---------|-------------|-------|
| **Medical Terms** | Clinical/medical vocabulary benchmark — drug names, procedures, anatomy | Similar pipeline to ai-terms; entity categories: drugs, procedures, conditions, anatomy, organisations |
| **Legal Terms** | Legal vocabulary benchmark — case law, statutes, Latin terms | Entity categories: case names, legal terms, jurisdictions, statutes |
| **Code-Switching** | FR/EN/DE/ES mixed-language utterances | Synthetic TTS data; per-split prompt template isolation to prevent leakage |
| **Trelis-OOD** | Non-web audio: ATC, English-Irish code-switching, technical recordings | Never publish; only evaluate with open-source models |

Medical and legal datasets follow the same three-tier architecture (public / semi-private / private) and the same difficulty-filtering pipeline. Start these only after v2 ai-terms is fully published and the pipeline is documented.

---

## Open Questions

- **Digit vs word CER scoring**: If a TTS transcript column is used, confirm Studio's CER evaluation uses the digit-format reference (not the TTS input). Clarify with Studio team.
- **Per-row entity CER from Studio API**: Confirm the eval API returns per-sample entity CER (not just aggregate) — needed for difficulty filtering in future builds.
- **Filtering threshold calibration**: The 0.045 median entity CER threshold was set heuristically. After the pilot, re-calibrate using v1 per-row entity CER as a baseline.
- **Minimum post-filter rows**: v1 had 12 rows per split; v2 has 21. Decide target size for medical/legal.
