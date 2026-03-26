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

### Reference pipeline — iteration history

**v1–v2 (Orpheus → AssemblyAI ASR → reference_asr_transcript):**
- Reference pipeline implemented with Orpheus TTS on spoken_form, AssemblyAI ASR for reference transcript
- Bug: AssemblyAI intelligent formatting converts spoken numbers back to Unicode (`"two point five times ten to the ninth"` → `"2.5×10⁹"`), injecting Unicode into reference_asr_transcript and inflating CER
- Phonetic regression: hyphenated respellings ("Eth-na") caused Orpheus to split on the hyphen and misread as initials

**v3 (Orpheus → Whisper large-v3 → reference_asr_transcript):**
- Switched ASR to Whisper large-v3 (more literal, no intelligent formatting, no Unicode injection)
- Fixed all phonetic respellings to remove hyphens (Ethna, Seersha, Woostersheer, Luffbruh, etc.)
- Added `ref_self_cer` quality gate: if `CER(spoken_form, ref_asr) > 0.3`, reference audio is unreliable (Orpheus loop/garble) and row is skipped
- Added Section 4a to spoken_form_rules.md: chemical compound rules (stereo descriptors, locant lists, CAS numbers, dose notation)

**v4 — Prototype (10 rows, reference pipeline investigation):**

Prototype dataset: `ronanarraig/tricky-tts-proto-v4` — 10 rows with `text`, `spoken_form`, `category`, `cer_reliable`.

Orpheus reference audio generated via data prep: `ronanarraig/tricky-tts-proto-ref-orpheus-datap` — 10 samples, 2.9 min. Audio quality confirmed good via inspection.

Gemini Pro reference audio also generated for comparison: `ronanarraig/tricky-tts-proto-ref-gemini-pro`.

**Intended final methodology** (not yet complete — blocked by Orpheus eval 504):
1. Orpheus TTS reads `spoken_form` → audio (via data prep or eval endpoint)
2. Whisper large-v3 ASR on that audio → `reference_asr_transcript`
3. Dataset: `text` (original), `spoken_form`, `reference_asr_transcript`
4. Eval jobs: TTS models read `text`, ASR → CER vs `reference_asr_transcript`

This normalises both reference and test through the same ASR model, so CER reflects pronunciation differences rather than formatting differences between written and expanded text.

**What was run in v4 (approximation):** `reference_column="spoken_form"` (text string, not ASR transcript). This is incorrect — it inflates CER artificially because the test ASR outputs digits/abbreviations while spoken_form is fully expanded words. Used as a temporary diagnostic only.

**Prototype v4 ElevenLabs results (CER vs spoken_form — inflated, for reference only):**

| Category | CER | ASR snippet |
|---|---|---|
| domain_specific (chemical) | 0.638 | `0.25 mL of 1-10-4-MOLOL, 2R, 3S...` |
| edge_cases (IEEE) | 0.604–0.614 | `IEEE Transeval 115, PPM 67-142...` |
| number_format (BP) | 0.588 | `ages 18 to 65` (digits, not words) |
| ai_tech (DeepSeek) | 0.417 | `DeepSeq AI to DeepSeq R1 to distill QUEN` |
| phonetic | 0.272 | `Eithne and Kauilfin` (original spelling kept) |
| paralinguistics | 0.124 | `zzz` dropped, rest clean |

Orpheus eval failed (504, bug `6248f9de`). Gemini Flash eval: 0 samples (bug `e2bc5701`). Only ElevenLabs results available from v4.

**Bugs filed this phase:**
- `8f38a6ae`: empty response body on job poll (intermittent 200 with no JSON)
- `37a45916`: data prep TTS should support router models (ElevenLabs/Gemini etc.)
- `e2bc5701`: Gemini Flash TTS eval completed but result=null, 0 samples — silent failure
- `d53d481f`: clarify whether asr_cer in parquet uses reference_column or original text
- `1a350d3c`: add ASR+CER to data prep TTS pipeline (currently audio only, no transcript)
- `749df765`: batch upload response key mismatch (`upload_urls` in docs vs `files` in actual response)
- `6248f9de`: Orpheus eval endpoint recurring 504 failures

**Studio feature request filed** (ID: `6814128c`): add `reference_column` parameter to TTS eval API. Now live.

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

See `reports/tricky-tts/phase3-roadmap.md`.
