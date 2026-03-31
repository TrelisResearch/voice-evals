# Tricky TTS — Phase 3 Report

## Summary

Phase 3 built the reference pipeline, constructed the 10-row `ronanarraig/tricky-tts-public` prototype dataset, and evaluated 9 TTS models via Trelis Studio. **Gemini Flash TTS leads on CER (0.164) and WER (0.272); Kokoro leads on MOS (4.533).** Chatterbox and Piper returned MOS-only (no ASR round-trip). Orpheus produced one extreme outlier (CER=1.198) on the number_format row.

---

## What Was Done

### Step 1: Reference Pipeline

Built `ronanarraig/tricky-tts-public` (10 rows) with a Kokoro-based reference ASR column:

1. Shortened 4 long rows to reduce ASR truncation risk (rows 0–3 were 347–495 spoken-form chars)
2. Fixed spoken form rules: `ZeRO` → `zero`, `80GB` → `eighty gigabytes`, `Qwen` → `Kwen`, `Yi` → `Yee`
3. Ran Kokoro TTS on `spoken_form` text via Studio eval endpoint with `asr_model_id="openai/whisper-large-v3"`
4. Used Kokoro ASR transcript as `reference_asr` column — normalises test and reference through the same Whisper model

**Dataset schema:** `text`, `spoken_form`, `category`, `reference_asr`, `reference_audio` (Kokoro audio for listening)

**Key decision:** TTS models receive the original `text` column (raw written form with abbreviations, symbols etc.) — this is intentional. The benchmark tests whether models can handle challenging written text. `spoken_form` is only used to generate the reference audio.

### Step 2: Studio API Discoveries

- `reference_column` defaults to `reference_asr` if that column exists — no need to pass it explicitly
- `text` column is hardcoded as TTS input — no `text_column` parameter
- Piper requires a real HF repo with `model.onnx` (used `Trelis/piper-en-gb-cori-high`)
- Chatterbox requires explicit `language="en"` — `language="auto"` fails silently (bug filed: `e3fa6b66`)
- ASR truncation for long audio is silent with no per-sample warning (bug filed: `50f34fff`)

### Step 3: Model Evaluations

**9 models submitted** against `ronanarraig/tricky-tts-public` with `reference_column="reference_asr"`, `asr_model_id="openai/whisper-large-v3"`.

---

## Results

### Aggregate Leaderboard

| Model | MOS | WER | CER |
|---|---|---|---|
| **Kokoro** | **4.533** | 0.319 | 0.203 |
| Gemini Flash TTS | 4.438 | **0.272** | **0.164** |
| Gemini Pro TTS | 4.403 | 0.350 | 0.209 |
| Chatterbox | 4.383 | N/A | N/A |
| GPT-4o mini TTS | 4.362 | 0.306 | 0.179 |
| ElevenLabs | 4.346 | 0.428 | 0.297 |
| Piper (en-gb) | 4.115 | N/A | N/A |
| Cartesia Sonic-3 | 3.990 | 0.437 | 0.252 |
| Orpheus | 3.918 | 0.557 | 0.389 |

*Ranked by MOS. Piper and Chatterbox have MOS only — ASR round-trip was not applied (no `asr_cer`/`asr_wer` columns in output).*

**Key observations:**
- Gemini Flash leads on accuracy (CER 0.164, WER 0.272) — best at normalising challenging text
- Kokoro tops naturalness (MOS 4.533) while competitive on CER (0.203)
- GPT-4o mini is the best balanced proprietary model (MOS 4.362, CER 0.179)
- Orpheus scores lowest on both MOS and CER — expected for a 3B open model vs proprietary APIs
- Cartesia Sonic-3 underperforms relative to its reputation — lowest MOS of the API models

### Per-Row CER by Category

| Category | ElevenLabs | GPT-4o | Cartesia | Gem Flash | Gem Pro | Orpheus | Kokoro |
|---|---|---|---|---|---|---|---|
| edge_cases (Vol/J.Phys.Chem) | 0.496 | 0.208 | 0.376 | 0.208 | 0.348 | 0.264 | 0.228 |
| edge_cases (IEEE Trans) | 0.433 | 0.143 | 0.382 | 0.176 | 0.206 | 0.223 | 0.256 |
| domain_specific (IUPAC) | 0.425 | 0.349 | 0.325 | 0.222 | 0.353 | 0.432 | 0.369 |
| domain_specific (Mycobacterium) | 0.394 | 0.231 | 0.303 | 0.267 | 0.267 | 0.520 | 0.299 |
| ai_tech (DeepSeek/Kwen) | 0.348 | 0.337 | 0.376 | 0.253 | 0.219 | 0.421 | 0.270 |
| ai_tech (Yi/Llama benchmarks) | 0.133 | 0.104 | 0.281 | 0.074 | 0.104 | 0.111 | 0.044 |
| number_format (2.5×10⁹/temps) | 0.297 | 0.115 | 0.104 | 0.057 | 0.088 | **1.198** | 0.193 |
| number_format (BP/°F ranges) | 0.342 | 0.153 | 0.179 | 0.337 | 0.337 | 0.347 | 0.268 |
| phonetic (Eithne/Caoilfhinn) | 0.077 | 0.055 | 0.099 | 0.044 | 0.143 | 0.231 | 0.099 |
| paralinguistics (snoring/zzz) | 0.024 | 0.098 | 0.098 | 0.000 | 0.024 | 0.146 | 0.000 |
| **MEAN** | **0.297** | **0.179** | **0.252** | **0.164** | **0.209** | **0.389** | **0.203** |

### Qualitative Row-by-Row Analysis

**Row 0 — edge_cases: journal abbreviations (J. Phys. Chem., Vol. 47, ≥99.9%, −80°C)**
All models struggle with `J. Phys. Chem.` — transcribed as "J. Fisantabur Chem." (ElevenLabs), "JFIS Chem" (GPT-4o, Gemini Flash/Pro), "J-phys. Chem." (Kokoro). None correctly expanded the journal abbreviation. The `≥` symbol is handled well by Gemini Flash ("greater than") and Kokoro ("greater or equal to") but ElevenLabs garbles it as "Noriju". ElevenLabs also hallucinated "Fisantabur" — a clear phonetic confusion from the `J. Phys.` cluster.

**Row 1 — edge_cases: IEEE Trans., 3.3V±5%, 150mA, ≤10μF, MIL-STD-883**
GPT-4o mini best here (CER 0.143) — correctly reads "3.3 volts", "milliamps", "microfarads". ElevenLabs garbled "IEEE Trans V1115" and "150 MW max". Orpheus produced "10 lieutenant fai" for `≤10μF` — a hallucination. Gemini Flash introduced "Celeste" for `≤` which is phonetically plausible but wrong.

**Row 2 — domain_specific: IUPAC stereo compound, CAS number, Saccharomyces**
Hardest single-model row outside the Orpheus outlier. All models struggle with `(2R,3S,4R,5R)-2,3,4,5,6-pentahydroxyhexanal`. Gemini Flash best (CER 0.222) — correctly reads "1 times 10 to the negative fourth" and "pentahydroxyhexanol". Orpheus worst (CER 0.432): "1 x 10 mitofoxamol L" — severe hallucination on the scientific notation. CAS number `50-99-7` comes out as "50997" (Gemini Flash/Pro) or "50 and 99 to 7" (ElevenLabs) — dashes are dropped by most.

**Row 3 — domain_specific: Mycobacterium tuberculosis, rpoB S531L, CLSI M24-A2**
Orpheus worst (CER 0.520): completely garbles the middle — "7-3-Cithelesaz 0.5-C-MSK, I plus lomelantations". All models read `H37Rv` as "H37RV" or "H37-RV". `rpoB` → "RPOB" universally (letter-by-letter). `MIC` correctly spelled out by GPT-4o ("greater than 128 micrograms per milliliter") but ElevenLabs reads it as "128 mbar mLs".

**Row 4 — ai_tech: DeepSeek/Kwen model path, ZeRO-3, bf16, lr=2e-5**
`Qwen` consistently comes out as "Quen", "QN", "QEN" — not "Kwen" as intended by our spoken_form. This reveals that the reference_asr (Kokoro's transcript of spoken_form) itself says something different to what the raw-text TTS models produce — a known limitation of this reference design for ai_tech rows. Gemini Pro best (CER 0.219, read the slash correctly). Cartesia hallucinated "Eyelash" for deepseek-ai. Orpheus produced "check the kiting train" for `gradient_checkpointing=True`.

**Row 5 — ai_tech: 01-ai/Yi-1.5, meta-llama/Llama-3.1, MMLU, GSM8K@4-shot**
Easiest ai_tech row — shorter, less exotic. Kokoro best (CER 0.044) — reads model paths letter-by-letter cleanly. Cartesia worst (CER 0.281) — literally spells out each character with hyphens: "0-1-A-I-slash-Y-1-dot-5, 3-4-B-C-hat-16-K". `Yi` → "Yai" (GPT-4o), "Eero 1 AI Y" (ElevenLabs), "ye" (Gemini Flash).

**Row 6 — number_format: 2.5×10⁹, −40°C to +120°C, 3:2 ratio**
Orpheus outlier (CER 1.198): "2.5 times 10 nines out of two quarters from 40 degrees Celsius to plus 20, 20 degrees" — severe hallucination/loop, completely broken. Gemini Flash best (CER 0.057): "2.5 times 10 to the 9th miles... minus 40 degrees Celsius to plus 120 degrees Celsius". GPT-4o introduced "muion miles" — a curious phonetic confabulation. The `3:2` ratio trips ElevenLabs ("3.2") and Kokoro ("two, five... covering" drops the ratio entirely).

**Row 7 — number_format: blood pressure 90/60–120/80, 97.8°F–99.1°F**
Blood pressure fractions are hard — the `/` is ambiguous (division vs ratio). GPT-4o and Cartesia handle it best, reading "90-60 to 120-80 mmHg" or "90 over 60". Cartesia misread `mmHg` as "MHG". Gemini Flash and Pro both scored 0.337 — interestingly identical, both producing "90-60 to 120-80 mmHg" which has hyphens rather than "over", differing from the reference.

**Row 8 — phonetic: Eithne, Caoilfhinn, mise en scène, Worcestershire, Dr. Nguyen**
Closest to ground truth across all models. Gemini Flash best (CER 0.044): "Ethna and Kalefin" — close to the phonetic reference. Orpheus worst (CER 0.231): "Solilfen" for Caoilfhinn and "maison's saying" for mise en scène — a creative but wrong hallucination. Gemini Pro dropped "Dr." → "Dr. Duen" (Nguyen confusion). Worcestershire handled correctly by all.

**Row 9 — paralinguistics: snoring, zzz, ellipsis, em-dash**
Best category overall. Gemini Flash and Kokoro: CER 0.000 — perfect. ElevenLabs handled the `zzz` and em-dash well. GPT-4o mini and Cartesia both silently dropped the `zzz` sounds entirely — "He started snoring. And I just thought..." — meaning they failed on the core paralinguistic challenge but were otherwise clean. Orpheus also dropped the zzz.

---

## Piper & Chatterbox — MOS Only (pending re-run)

Both models completed successfully but returned no ASR round-trip metrics. `asr_model_id="openai/whisper-large-v3"` and `reference_column="reference_asr"` were confirmed present in the stored job config — the ASR step is simply not wired through for these model types in Studio.

**Action required:** Once Studio fixes ASR for Piper/Chatterbox evaluation paths, re-run both jobs against `ronanarraig/tricky-tts-public` with the same parameters. No dataset changes needed.

| Model | MOS | Note |
|---|---|---|
| Chatterbox (`ResembleAI/chatterbox`) | 4.383 | MOS only; re-run needed for WER/CER |
| Piper (`Trelis/piper-en-gb-cori-high`) | 4.115 | MOS only; re-run needed for WER/CER |

---

## Bugs & Friction

| Bug ID | Description |
|---|---|
| `50f34fff` | ASR silently truncates long audio — no per-sample warning |
| `e3fa6b66` | Chatterbox: `language="auto"` fails all samples silently; upload hung on retry before completing |
| New (unfiled) | Piper/Chatterbox: `asr_model_id` + `reference_column` not applied — no WER/CER output |

---

## Trelis Studio Experience

**Positives:**
- `reference_column` parameter works cleanly for the 7 models that support ASR round-trip
- All 7 round-trip eval jobs completed without errors; per-row results pushed to HF as expected
- MOS (UTMOS) returned for all 9 models including Piper and Chatterbox
- Concurrent job limit (10) is adequate for sequential batching

**Issues:**
- Piper and Chatterbox don't output `asr_cer`/`asr_wer` despite `asr_model_id` being set — round-trip ASR not wired for these model types
- Chatterbox `language="auto"` fails silently with 0 samples instead of validating at submission
- Chatterbox upload hung mid-transfer on first successful run (eventually completed)
- `/api/v1/models` endpoint only returns 6 models — Piper, Chatterbox, Kokoro not listed

---

## Listening Links

All eval output datasets are private on HuggingFace with `generated_audio` columns — open in the dataset viewer to listen row by row.

| Model | Dataset | Highlight rows to listen |
|---|---|---|
| ElevenLabs | [tricky-tts-pub-eval-elevenlabs](https://huggingface.co/datasets/ronanarraig/tricky-tts-pub-eval-elevenlabs) | Row 0 ("Fisantabur"), Row 6 (spacecraft) |
| GPT-4o mini TTS | [tricky-tts-pub-eval-gpt4o-mini](https://huggingface.co/datasets/ronanarraig/tricky-tts-pub-eval-gpt4o-mini) | Row 1 (IEEE best), Row 9 (zzz dropped) |
| Cartesia Sonic-3 | [tricky-tts-pub-eval-cartesia](https://huggingface.co/datasets/ronanarraig/tricky-tts-pub-eval-cartesia) | Row 5 (character-by-character), Row 4 ("Eyelash") |
| Gemini Flash TTS | [tricky-tts-pub-eval-gemini-flash](https://huggingface.co/datasets/ronanarraig/tricky-tts-pub-eval-gemini-flash) | Row 6 (best scientific notation), Row 9 (perfect zzz) |
| Gemini Pro TTS | [tricky-tts-pub-eval-gemini-pro](https://huggingface.co/datasets/ronanarraig/tricky-tts-pub-eval-gemini-pro) | Row 4 (slash handling), Row 8 ("Dr. Duen") |
| Orpheus | [tricky-tts-pub-eval-orpheus](https://huggingface.co/datasets/ronanarraig/tricky-tts-pub-eval-orpheus) | Row 6 (CER=1.198 loop), Row 3 ("Cithelesaz") |
| Kokoro | [tricky-tts-pub-eval-kokoro](https://huggingface.co/datasets/ronanarraig/tricky-tts-pub-eval-kokoro) | Row 5 (best ai_tech), Row 9 (perfect zzz) |
| Piper (en-gb) | [tricky-tts-pub-eval-piper](https://huggingface.co/datasets/ronanarraig/tricky-tts-pub-eval-piper) | All rows — British accent, no CER available |
| Chatterbox | [tricky-tts-pub-eval-chatterbox](https://huggingface.co/datasets/ronanarraig/tricky-tts-pub-eval-chatterbox) | All rows — no CER available |

**Recommended listening order if doing a subset:**
1. **Orpheus Row 6** — most dramatic failure (loop/hallucination on 2.5×10⁹)
2. **Gemini Flash Row 6** — the same row done well, for contrast
3. **Cartesia Row 5** — character-by-character robotic reading of model paths
4. **ElevenLabs Row 0** — "Fisantabur" hallucination
5. **GPT-4o vs Gemini Flash Row 9** — zzz dropped vs preserved

---

## Dataset State

| Dataset | Description | Status |
|---|---|---|
| `ronanarraig/tricky-tts-public` | 10-row prototype, text + spoken_form + reference_asr + reference_audio | ✅ Complete |
| `ronanarraig/tricky-tts-pub-eval-elevenlabs` | Per-row eval results | ✅ |
| `ronanarraig/tricky-tts-pub-eval-gpt4o-mini` | Per-row eval results | ✅ |
| `ronanarraig/tricky-tts-pub-eval-cartesia` | Per-row eval results | ✅ |
| `ronanarraig/tricky-tts-pub-eval-gemini-flash` | Per-row eval results | ✅ |
| `ronanarraig/tricky-tts-pub-eval-gemini-pro` | Per-row eval results | ✅ |
| `ronanarraig/tricky-tts-pub-eval-orpheus` | Per-row eval results | ✅ |
| `ronanarraig/tricky-tts-pub-eval-kokoro` | Per-row eval results | ✅ |
| `ronanarraig/tricky-tts-pub-eval-piper` | MOS only | ✅ |
| `ronanarraig/tricky-tts-pub-eval-chatterbox` | MOS only | ✅ |

---

## Next Steps

- **Waiting on Studio:** Re-run Piper and Chatterbox once ASR is wired for those model types — same job parameters, no dataset changes needed
- **Phase 4:** Build a private 48-row split using the same reference pipeline methodology
