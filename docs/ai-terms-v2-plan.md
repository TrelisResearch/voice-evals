# AI Terms v2 — Dataset Plan

## Goal
Create a harder version of the ai-terms benchmark by filtering out rows that current ASR models handle easily, using articles from a recent time window to minimize training data contamination.

## Source Data

- **AI News articles** published **Jan 1 – Feb 28, 2026**
- Same sourcing approach as v1 (with permission), plus YouTube transcripts and rewrites
- Target: **~50+ candidate rows** before filtering (to end up with ~15-20 per split after filtering)

### Why this date window?
- Most model training data cuts off before 2026, so Jan–Feb 2026 content is unlikely to be in training sets
- Gives ~6-12 months of useful benchmark life before contamination risk rises from newer models

## Difficulty Filtering Pipeline

### Step 1: Text Preparation
1. Source articles from Jan–Feb 2026
2. Extract entities using LLM (same 6 categories: companies, models, products, benchmarks, people, technical)
3. Run entity/n-gram overlap detection to assign rows to splits without leakage

### Step 2: TTS Audio Generation
- **Blocked on**: Trelis Studio TTS integration (in progress)
- Generate synthetic audio for all ~50+ candidate texts
- Manual QA pass: listen to TTS output, discard rows where entities are mispronounced
- This avoids recording a large pool of audio manually

### Step 3: Difficulty Filtering
Run 3 diverse open-source ASR models on the TTS audio:

| Model | Why chosen |
|-------|-----------|
| openai/whisper-large-v3-turbo | Strong baseline, 4.8% entity CER on v1 semi-private |
| nvidia/parakeet-tdt-0.6b-v3 | Different architecture (CTC/TDT), 5.0% entity CER |
| Qwen/Qwen3-ASR-1.7B | Newer model family, 6.3% entity CER |

All open-source — safe for private splits.

**Filtering logic:**
1. Compute per-row entity CER for each of the 3 models
2. Take the **median** entity CER per row
3. Drop rows where median entity CER is below threshold (TBD — calibrate using v1 data first)
4. Candidate threshold: ~3-4% median entity CER (rows easier than this are dropped)

### Step 4: Audio Decision
- If TTS audio quality is good (validated in Step 2 QA): **keep TTS audio** for the filtered rows
- If TTS audio has issues: **re-record only the hard rows** manually (~15-20 rows per split instead of 50+)

### Step 5: Finalize Dataset
1. Assign filtered rows to splits (public, semi-private, private)
2. Run entity deduplication and overlap checks (same as v1)
3. Build entity annotations with char offsets
4. Push to HuggingFace as `Trelis/ai-terms-v2-{public,semi-private,private}`
5. Run full eval suite (25+ models) on v2

## Calibration (TODO)
Before building v2, calibrate the difficulty threshold using v1 data:
- Run per-row entity CER for the 3 filter models on v1 semi-private split
- Determine what threshold separates "easy" from "hard" rows
- Use this to set the v2 filtering cutoff

## Blockers
1. **TTS integration in Trelis Studio** — needed for Step 2
2. **Per-row entity CER from Studio** — need to confirm eval API returns per-sample entity CER (not just aggregate)

## Open Questions
- Exact entity CER threshold for filtering (calibrate from v1)
- Whether to weight certain entity categories higher (e.g. benchmark names are hardest)
- Minimum dataset size per split after filtering (12 like v1, or larger?)
