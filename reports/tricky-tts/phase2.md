# Tricky TTS — Phase 2 Report

## Summary

Phase 2 completed difficulty validation and median-of-N filtering on the public split of `ronanarraig/tricky-tts-v2-public` (48 rows). The benchmark is now meaningfully harder, with aggregate CER improving from 0.1358 → 0.2032 (+50%) post-filtering.

---

## What Was Done

### Step 1: Spoken Form Generation

Generated `spoken_form` column for all 48 rows via LLM (Claude claude-sonnet-4-5 / Gemini 2.5 Flash fallback), converting written text to canonical spoken form:
- Abbreviation expansion: `Dr.` → `Doctor`, `Ph.D.` → `P H D`
- Number normalization: `10⁻³ M` → `ten to the minus three molar`
- AI model paths: `meta-llama/Llama-3.1-405B` → `meta-llama slash Llama three point one four oh five B`
- Celtic names preserved phonetically: Saoirse, Niamh, Caoilfhinn, etc.

Tagged `cer_reliable = False` for categories where CER vs written text is inherently unreliable: `edge_cases`, `number_format`, `ai_tech`.

### Step 2: TTS Evaluation (5 models)

Submitted TTS eval jobs via Trelis Studio against the initial 48-row dataset:

| Model | MOS | WER | CER |
|---|---|---|---|
| `kokoro` | 4.526 | 0.658 | 0.399 |
| `openai/gpt-4o-mini-tts` | 4.465 | 0.292 | 0.154 |
| `elevenlabs/eleven-multilingual-v2` | 4.355 | 0.281 | 0.136 |
| `unsloth/orpheus-3b-0.1-ft` | 4.281 | 0.363 | 0.184 |
| `cartesia/sonic-3` | 4.042 | 0.410 | 0.226 |

**ASR model:** `openai/whisper-large-v3-turbo` for all. All jobs used `push_results: true` to get per-row audio + transcripts.

### Step 3: Per-Row Difficulty Analysis

Downloaded per-row `asr_cer` from all 5 pushed HF datasets and computed median CER per row across models.

**Key findings by category:**

| Category | Avg Median CER | Easy rows (< 0.05) | Notes |
|---|---|---|---|
| edge_cases | 0.245 | 0 | All challenging |
| domain_specific | 0.189 | 0 | All challenging |
| ai_tech | 0.110 | 0 | All challenging |
| number_format | 0.164 | 2 | Pope Benedict + Apollo XI too simple |
| phonetic | 0.045 | 5 | Irish names too well-known to models |
| paralinguistics | 0.033 | 6 | Low CER by design; UTMOS-focused category |

**Paralinguistics kept entirely** — these rows test naturalness/expressiveness (UTMOS), not transcription accuracy. Their low CER is expected and correct.

### Step 4: Difficulty Filtering

Removed 7 easy rows (median CER < 0.05, excluding paralinguistics):

**Phonetic (5 removed):**
- "The conduct of the Leicester conference..." (CER 0.024)
- "When they present the present to Aoife at Loughborough..." (CER 0.015)
- "Dr. Nguyen's Bildungsroman explores how Tadhg must conduct himself..." (CER 0.025)
- "Aoife needed a permit to film, but the Leicester council..." (CER 0.009)
- "The present study will present evidence that Caoimhe..." (CER 0.049)

**Number format (2 removed):**
- "Pope Benedict XVI resigned in 2013..." (CER 0.000)
- "The Apollo XI mission landed on July 20, 1969..." (CER 0.022)

### Step 5: Replacement Row Generation

Generated 7 harder replacement rows targeting known TTS failure modes:

**Phonetic (5 new rows):**
1. "The rebel Saoirse decided to rebel against serving quinoa and gnocchi at the Auchterarder conference..." — heteronym stress + Celtic + foreign loanwords
2. "Niamh's research on ptarmigan migration patterns will complement Oisín's conflict analysis..." — rare bird name + heteronym
3. "At the Worcestershire estate, Catrìona tried to perfect her perfect pronunciation of bourguignon..." — place name + Celtic + French loanword + heteronym
4. "Eithne's close friend felt close to tears when the Clachnacuddin council refused to permit the açaí..." — rare place + Celtic + heteronym + Portuguese
5. "The Polish workers at Blaenavon will polish Fionnuala's collection of Cnoc an Doire artifacts..." — heteronym + Welsh place + Irish place + Celtic name

**Number format (2 new rows):**
1. "The patient in room 4B received 2.5mg of medication at 14:30, showing a 10⁻³ molar concentration..." — mixed alphanumeric + 24h time + scientific notation
2. "Flight BA2107 departed gate 21C at 21:07 hours, carrying 207 passengers to the XXIst century's busiest hub..." — alphanumeric codes + 24h time + Roman ordinal

### Step 6: Validation Eval (ElevenLabs on filtered dataset)

| Metric | Initial (48 rows) | Filtered (48 rows) | Change |
|---|---|---|---|
| MOS | 4.355 | 4.360 | +0.1% |
| WER | 0.281 | 0.440 | **+56%** |
| CER | 0.136 | 0.203 | **+50%** |

The filtered dataset is substantially harder while maintaining the same MOS range.

---

## Per-Row Results (ElevenLabs, Filtered Dataset)

Sorted hardest first:

| Category | CER | New? | Text |
|---|---|---|---|
| domain_specific | 2.722 | | Titrate 2.5mg/kg/min of (±)-4-(2-aminoethyl)... |
| edge_cases | 0.399 | | According to IEEE Trans. Vol. 15, pp. 67–142... |
| edge_cases | 0.313 | | The manuscript cited Vol. 47, No. 3, pp. 1234–89... |
| number_format | 0.293 | | King Henry the 3rd (III), King Henry the 8th... |
| number_format | 0.246 | | The spacecraft traveled 2.5×10⁹ miles... |
| edge_cases | 0.194 | | The property at 789 St. Clair Ave... |
| edge_cases | 0.189 | | Dr. Zhang, M.D., Ph.D., F.A.C.C... |
| domain_specific | 0.188 | | Administer 15mg/kg IV q6h of piperacillin... |
| number_format | 0.173 | ★ | Flight BA2107 departed gate 21C at 21:07... |
| ... | ... | | ... |

Note: `domain_specific` "Titrate..." row has CER > 1.0, indicating ElevenLabs produced extra/looped audio that the ASR transcribed with significant drift from reference text. This is a genuine failure mode worth keeping.

---

## Dataset State

**Final public split:** `ronanarraig/tricky-tts-v2-public` — 48 rows, 8 per category

| Category | Role | Avg CER |
|---|---|---|
| edge_cases | CER + UTMOS | 0.245 |
| domain_specific | CER + UTMOS | 0.189 |
| ai_tech | CER + UTMOS | 0.110 |
| number_format | CER + UTMOS | 0.164 |
| phonetic | CER + UTMOS | 0.045 → harder after filtering |
| paralinguistics | UTMOS only | low CER by design |

Schema: `text`, `category`, `spoken_form`, `cer_reliable`

---

## CER Reliability Notes

### CER is inflated for technical categories

CER is currently computed as `CER(original_written_text, asr_transcript_of_tts_audio)`. For `edge_cases`, `number_format`, and `ai_tech`, there are many equally valid spoken forms — e.g. `"£1,234.56"` might be spoken as *"one thousand two hundred thirty-four pounds fifty-six pence"* or *"twelve thirty-four point five six pounds"*. The ASR transcribes whichever form the TTS chose, so CER measures disagreement between two valid spoken forms rather than TTS failure. This inflates CER for these categories.

### CER may be deflated for phonetic categories

For phonetic rows, if a TTS model mispronounces "Saoirse" as "Sa-oir-se" but ASR still outputs "Saoirse" (strong spelling prior), the error is invisible to round-trip CER. This understates phonetic difficulty.

### Fix: reference-TTS ground truth pipeline (Phase 3)

The `spoken_form` column was generated this phase but not yet used as the CER reference. Phase 3 should implement:
1. Run Kokoro TTS on `spoken_form` → reference audio
2. Run ASR on reference audio → ground truth transcript
3. CER(ground_truth, test_asr_transcript)

This replaces the ambiguous written text reference with a consistent spoken-form reference, making CER meaningful for technical rows.

**Studio feature request filed** (ID: `6814128c`): add `reference_column` parameter to TTS eval API so Studio can compute CER against `spoken_form` instead of `text` natively.

---

## Trelis Studio Experience

**Positives:**
- TTS eval API is smooth: `dataset_id` + `asr_model_id` + `push_results: true` does exactly what's needed
- Per-row `asr_cer` and `asr_transcription` in pushed HF datasets is very useful for difficulty analysis
- Kokoro and all 3 Router proprietary models worked well
- Job submission and polling are reliable

**Issues:**

**Orpheus temporary failure** — Orpheus was failing in evaluation at session start; fix went live mid-session.

**CER > 1.0 for complex domain_specific rows** — expected when TTS loops or generates extra tokens (e.g. "Titrate 2.5mg/kg/min..." scored CER=2.72 for ElevenLabs). The ASR transcript is much longer than the reference text. Not a bug — a genuine failure mode worth keeping in the dataset.

**Torch dependency in pushed eval datasets** (bug filed, ID: `5e8a3902`) — Loading a pushed TTS eval dataset via `datasets` library raises `No module named torch` even when only accessing non-audio columns (`asr_cer`, `asr_transcription`, etc.), because the Audio feature type registers at schema load time. Workaround: load the parquet directly with `pyarrow.parquet` and select only needed columns:
```python
import pyarrow.parquet as pq
table = pq.read_table(local_path, columns=["text_prompt", "asr_cer", "asr_transcription"])
```

---

## Phase 3 Next Steps

- Calibrate semi-private and private splits with same methodology
- Apply entity-based n-gram overlap check between splits (leakage prevention)
- Migrate from `ronanarraig/` to `Trelis/` org once on Trelis infrastructure
- Implement reference-TTS ground truth pipeline once Studio supports `reference_column` parameter (or manually via spoken_form → Kokoro → ASR)
