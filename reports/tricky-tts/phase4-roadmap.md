# Tricky TTS — Phase 4 & 5 Roadmap

## New Taxonomy

Four categories replacing the previous six:

| Category | What it tests |
|---|---|
| `symbol_expansion` | Unicode, units, operators — `≥`, `μg`, `±`, `×10⁹` |
| `abbreviation_reading` | Acronyms, initialisms, titles — `IEEE`, `rpoB`, `bf16`, `Vol.` |
| `proper_nouns` | Names, brands, model paths — Celtic names, `deepseek-ai/DeepSeek-R1` |
| `prosody_and_punctuation` | Em-dashes, ellipses, onomatopoeia, rhythm |

---

## Phase 4 — Minimal Human-Referenced Eval

**Goal:** 4 rows (one per category), human reference audio, qualitative + quantitative eval across all Studio TTS models.

### Step 1: Write 4 rows
One carefully chosen example per category. No `spoken_form` needed.

### Step 2: Record reference audio
User records their own voice reading each of the 4 rows. Push as `ronanarraig/tricky-tts-phase4` with `text`, `category`, `reference_audio` columns.

### Step 3: Run eval on all supported Studio models
- `asr_model_id="openai/whisper-large-v3"`
- `reference_column` = not set (compare ASR output to `text` directly)
- All 9 models: ElevenLabs, GPT-4o mini, Cartesia, Gemini Flash, Gemini Pro, Orpheus, Kokoro, Piper, Chatterbox

### Step 4: Review results
- User listens to each model's audio per row and gives qualitative commentary
- Inspect CER per row — note where round-trip ASR signal is strong vs weak
- Document failure modes

---

## Phase 5 — Reference ASR Model

**Goal:** Train an ASR model that maps TTS audio back to original written text, enabling fully automated eval without a `reference_asr` column.

**Approach:**
- Generate synthetic training pairs: original text → open-source TTS (multiple voices) → audio, with original text as ASR target
- `spoken_form` column useful here for symbol/abbreviation rows — TTS reads spoken_form, target label is original text
- Fine-tune Whisper on this data
- Validate: does CER(whisper_finetuned(tts_audio), original_text) correctly penalise mispronunciations?
- Expected to work well for `symbol_expansion` + `abbreviation_reading`; partial coverage for `proper_nouns`; `prosody_and_punctuation` likely still needs human eval

**Limitations to resolve:**
- Proper nouns: LLM-generated phonetic transcriptions are unreliable for obscure names — may need pronunciation dictionary augmentation
- Prosody: no automated metric captures naturalness of pauses/rhythm — remains human-eval territory
