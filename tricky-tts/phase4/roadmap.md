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

### Step 1: Write 4 rows -- DONE
One carefully chosen example per category, with `spoken_form` for each.
See `tricky-tts/phase4/rows.json`.

### Step 2: Record reference audio -- DONE
User recorded voice reading each of the 4 rows (~20s each).
Audio files in `tricky-tts/phase4/audio/`.

### Step 3: Upload to Trelis Studio
Create a session in Trelis Studio and upload the 4 audio files + text/category metadata.
Use the data prep pipeline (upload audio, process with forced alignment) to build a HuggingFace dataset with `text`, `category`, `reference_audio` columns.

### Step 4: Run eval on all supported Studio models
- `asr_model_id="openai/whisper-large-v3"`
- `reference_column` = not set (compare ASR output to `text` directly)
- All supported TTS models: ElevenLabs, GPT-4o mini, Cartesia, Gemini Flash, Gemini Pro, Orpheus, Kokoro, Piper, Chatterbox

### Step 5: Review results
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
