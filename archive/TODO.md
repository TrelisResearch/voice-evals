# Trelis Transcribe AI Terms

Specialized Whisper models for accurate transcription of AI/ML terminology.

Notes:
- Always push to Trelis org on HF. Always push private, not public.

## Project Outcomes

### Models
> Each model should have an associated huggingface space so that users can try it out.
- [ ] **Transcribe AI Terms Tiny** [base: openai/whisper-tiny]
  - Private repo: `Trelis/transcribe-ai-terms-v1-tiny`
  - CTranslate2 variant: `Trelis/transcribe-ai-terms-v1-tiny-ctranslate2`
- [ ] **Transcribe AI Terms Turbo** [base: openai/whisper-large-v3-turbo]
  - Private repo: `Trelis/transcribe-ai-terms-v1-turbo`
  - CTranslate2 variant: `Trelis/transcribe-ai-terms-v1-turbo-ctranslate2`

### Datasets
- [ ] **Transcribe AI Terms Train** - Training split
  - Private repo: `Trelis/transcribe-ai-terms-train-{timestamp}`
- [ ] **Transcribe AI Terms Test** - validation split (~50 rows)
  - Private repo: `Trelis/transcribe-ai-terms-test-{timestamp}`
  - Public variant: `Trelis/transcribe-ai-terms-v1` (reduced columns)

### Later
- [ ] **AI Terms Text-to-Speech Model** - Specialized TTS for AI terminology

## Training Data Pipeline

### 1. Data Fetching [NEEDS BUILDING]

**Source:** Daily Smol AI newsletters (permission granted)
- Example: https://news.smol.ai/issues/25-12-08-not-much
- Historical issues + ongoing fetches

**Fetching Options:**
- [ ] **Option A:** Direct fetch + parse to markdown
- [ ] **Option B:** Firecrawler for markdown extraction (recommended)

**Script to build:** `ai-terms/fetch_smol_ai.py`
- Fetch historical archive
- Incremental fetch for latest issues
- Output: Raw markdown files

Start by just trying to get one smol ai news bulletin to be processed (and go all the way through fine-tuning), then add a script to get more history. Then finally, add a script that allows us to pull in the latest data.

### 2. Text Parsing [NEEDS BUILDING]

**Script to build:** `ai-terms/parse_text.py`
- Parse markdown into sentence groups
- Constraints:
  - 2-3 sentences per chunk
  - Max ~150 characters per chunk
  - Preserve AI/ML terminology formatting
- Output: `ai-terms/data/generated/text_chunks.parquet`

**Note:** Additional synthetic data generation may be needed based on fine-tuned model performance.

### 3. Synthetic Voice Data Generation [USE EXISTING]
Note that the endpoint right now only really supports a concurrency of one (see the kokoro.py on running the endpoint).

**Use:** `common/voice_samplers/kokoro.py` (already built)

**Config file to create:** `ai-terms/config/voice_config.yaml`
```yaml
tts_provider: kokoro
base_url: "http://KOKORO_SERVER:11376/v1"
speed_range: [0.85, 1.15]
max_duration: 30  # seconds
voices:
  train: [...]  # Select voices
  test: [...]   # Held-out voices
```

**Command:**
```bash
uv run python common/voice_samplers/kokoro.py \
  --config ai-terms/config/voice_config.yaml \
  --input ai-terms/data/generated/text_chunks.parquet \
  --output-dir ai-terms/data/generated/audio/
```

**Output:** Audio files + metadata parquet

### 4. Dataset Building [USE EXISTING]

**Use:** Similar to `english_variants/build_hf_dataset.py` or create `ai-terms/build_hf_dataset.py`

**Config file:** Extend `ai-terms/config/voice_config.yaml` with dataset building parameters:

```yaml
# ai-terms/config/voice_config.yaml
tts_provider: kokoro
base_url: "http://KOKORO_SERVER:11376/v1"
speed_range: [0.85, 1.15]
max_duration: 30
voices:
  train: [...]
  test: [...]  # Held-out voices

# Dataset building config
dataset:
  audio_dir: data/generated/audio/
  train:
    audio_metadata: data/generated/audio_metadata_train.parquet
    hf_repo: Trelis/transcribe-ai-terms-train
    split: train
    private: true
  test:
    audio_metadata: data/generated/audio_metadata_test.parquet
    hf_repo: Trelis/transcribe-ai-terms-test
    split: test
    private: true
    num_samples: 50  # Random selection
  public_test:
    source: Trelis/transcribe-ai-terms-test
    target: Trelis/transcribe-ai-terms-v1
    readme: datasets/test-dataset-README.md

  # Critical columns to include
  columns:
    - audio
    - text
    - voice
    - speed
    - gender
```

**Commands:**
```bash
# Build train dataset
uv run python ai-terms/build_hf_dataset.py --config ai-terms/config/voice_config.yaml --split train

# Build test dataset
uv run python ai-terms/build_hf_dataset.py --config ai-terms/config/voice_config.yaml --split test

# Push reduced public test dataset
uv run python ai-terms/build_hf_dataset.py --config ai-terms/config/voice_config.yaml --split public_test
```

**Note:** The config-based approach keeps all parameters in one place and ensures consistency with voice generation settings.

## Training [EXISTING SCRIPTS]

**Use:** `finetuning/Trelis_transcribe_finetuning.ipynb`

**Steps:**
1. Load base model (whisper-tiny or whisper-large-v3-turbo)
2. Load training dataset from HuggingFace
3. Fine-tune with Unsloth LoRA
4. Export formats:
   - Merged 16bit (HuggingFace format)
   - OpenAI format (.bin)
   - CTranslate2 format
5. Push to HuggingFace (private repos)

**Models to train:**
- [ ] Tiny model
- [ ] Turbo model

## Publishing [EXISTING SCRIPTS]

**Config file to create:** `ai-terms/config/readme_targets.yaml`
```yaml
template: ../model_info/README.md
output_dir: ../model_info/generated_readmes
repos:
  - repo_id: Trelis/transcribe-ai-terms-v1-tiny
    base_model: openai/whisper-tiny
    stripe_link: TBD  # Update later
  - repo_id: Trelis/transcribe-ai-terms-v1-turbo
    base_model: openai/whisper-large-v3-turbo
    stripe_link: TBD  # Update later
  # Add ctranslate2 variants
```

**Command:**
```bash
uv run --with pyyaml --with huggingface_hub python common/push_readmes.py \
  --config ai-terms/config/readme_targets.yaml --push
```

**Note:** Stripe links can be updated later

## Evaluation [EXISTING SCRIPTS]

**Use:** `common/eval_whisper.py`

### LibriSpeech Baseline
```bash
# Evaluate on LibriSpeech test.other split
uv run python common/eval_whisper.py \
  --model Trelis/transcribe-ai-terms-v1-turbo \
  --datasets librispeech \
  --num-samples 50
```

**Purpose:** WER baseline + manual inspection for degradation

### validation split Evaluation
```bash
# Evaluate on AI Terms validation split
uv run python common/eval_whisper.py \
  --model Trelis/transcribe-ai-terms-v1-turbo \
  --datasets Trelis/transcribe-ai-terms-v1 \
  --num-samples 0  # All samples
```

**Purpose:** WER on domain-specific data + manual inspection of AI terminology accuracy

### Manual Inspection Focus
- [ ] Correct spelling of AI/ML terms (e.g., "GPT", "LoRA", "CUDA")
- [ ] Proper capitalization (e.g., "PyTorch" not "pytorch")
- [ ] Acronym handling (e.g., "LLM", "RAG", "API")
- [ ] Common model names (e.g., "Claude", "Gemini", "Llama")

## Scripts to Build

Priority order:
1. [ ] `ai-terms/fetch_smol_ai.py` - Fetch newsletter content
2. [ ] `ai-terms/parse_text.py` - Parse into sentence chunks
3. [ ] `ai-terms/config/voice_config.yaml` - Voice sampling config
4. [ ] `ai-terms/build_hf_dataset.py` - Dataset builder (or adapt existing)
5. [ ] `ai-terms/config/readme_targets.yaml` - Model README config
6. [ ] `ai-terms/model_info/README.md` - README template (adapt from english_variants)

## Dependencies

### Existing (Ready to Use)
- ✓ `common/voice_samplers/kokoro.py` - TTS voice generation
- ✓ `common/eval_whisper.py` - Model evaluation
- ✓ `common/push_readmes.py` - README publishing
- ✓ `common/reduce_dataset_columns.py` - Public dataset creation
- ✓ `finetuning/Trelis_transcribe_finetuning.ipynb` - Training workflow

### To Be Built
- Smol AI fetcher
- Text parser
- Config files
- Dataset builder (may reuse with modifications)

## Environment

Same as other use cases:
```
OPENROUTER_API_KEY=your_key  # Not needed for this use case
OPENAI_API_KEY=your_key      # Optional: alternative TTS
```

Kokoro TTS server: Deploy via [RunPod template](https://console.runpod.io/deploy?template=grwfixzu60&ref=jmfkcdio)

## Notes

- Start with historical Smol AI data for initial training
- Monitor model performance - may need additional synthetic data if coverage is insufficient
- AI terminology list should be curated for evaluation metrics
- Consider adding custom WER metric that weights AI terms more heavily
