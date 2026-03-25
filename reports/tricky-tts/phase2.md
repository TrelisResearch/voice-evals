# Tricky TTS — Phase 2 Plan

## Scope

Phase 2 focuses on the **public split only**. Semi-private and private splits will be calibrated in Phase 3, once the public pipeline is working well.

**Starting dataset:** `ronanarraig/tricky-tts-v2-public` (48 rows, Phase 1d)

---

## Evaluation Methodology

### Metric: CER (not WER)

Studio uses CER (character error rate) for round-trip evaluation. This is the right choice:
- CER gives partial credit for near-correct transcriptions ("Keelin" vs "Caoilfhinn" scores better than total failure)
- For technical strings (model paths, IUPAC names), single character differences are meaningful
- WER over-penalises multi-word expansions of single tokens

All Phase 1 figures were WER (proxy only). Phase 2 uses CER throughout.

### Reference-TTS ground truth pipeline

To resolve ASR OOV and reference mismatch problems identified in Phase 1:

1. **Spoken form generation** — use an LLM to convert each row's written text to its canonical spoken form
   - `"£1,234.56"` → `"one thousand two hundred and thirty-four pounds and fifty-six pence"`
   - `"01-ai/Yi-1.5-34B-Chat-16K"` → spoken expansion (note: model paths need context-aware prompting — dashes/slashes in org/model names are not pronounced the same way as in natural text; the LLM prompt must include examples for this)
   - `"Caoilfhinn"` → `"Keelin"`
2. **Reference TTS** — run spoken form through an open-source TTS model (Kokoro or Orpheus, both available via Studio). Determine which is the better reference choice before running at scale.
3. **Reference ASR** — run `assemblyai/universal-3-pro` on reference audio → ground truth transcript
4. **Test TTS** — run each test model on the original written text → test audio
5. **Test ASR** — run same ASR on test audio → test transcript
6. **CER(ground_truth, test_transcript)**

### UTMOS for naturalness

- Covers `paralinguistics` rows where CER is uninformative
- Covers prosody quality across all categories

---

## Models

### Open-source (via Trelis Studio)
- Orpheus
- Kokoro
- *(skip Piper and Chatterbox for now)*

### Proprietary (via Trelis Router)
- ~4 available; use ~3 for calibration

Median-of-N difficulty filtering will be applied once results are in from ≥3 models.

---

## Dataset Schema Updates

Add two columns to the dataset before running evaluation:

| Column | Type | Description |
|---|---|---|
| `spoken_form` | string | LLM-generated canonical spoken form of the reference text |
| `cer_reliable` | bool | False for `edge_cases`, `number_format`, `ai_tech` where reference mismatch is expected |

`cer_reliable` rows tagged False are excluded from CER aggregation and evaluated via reference-TTS ground truth or UTMOS only.

---

## Checklist

- [ ] Decide reference TTS: compare Kokoro vs Orpheus on a small sample, pick the cleaner one
- [ ] Write LLM prompt for spoken form generation; include examples for model paths and ambiguous abbreviations
- [ ] Generate `spoken_form` column for all 48 public rows
- [ ] Tag `cer_reliable` boolean per row
- [ ] Implement reference TTS → ASR → ground truth pipeline
- [ ] Run evaluation via Trelis Studio across Orpheus, Kokoro, and ~3 proprietary models
- [ ] Apply median-of-N difficulty filtering (drop rows that are easy across all models)
- [ ] Review results → feedback → adjust before proceeding to Phase 3
