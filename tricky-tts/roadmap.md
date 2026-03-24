# Tricky TTS Dataset Roadmap

## Goal
A text-only benchmark for evaluating TTS model quality on linguistically and typographically challenging English inputs. Models are evaluated by Trelis Studio using UTMOS (naturalness score) and Round Trip ASR (transcribe TTS output and measure WER/CER).

## Dataset IDs
- `ronanarraig/tricky-tts-public`
- `ronanarraig/tricky-tts-semi-private`
- `ronanarraig/tricky-tts-private`

> Note: currently pushed to `ronanarraig` (this VPS). Will migrate to `Trelis/` org once on Trelis infrastructure.

## Schema
| Column | Type | Notes |
|---|---|---|
| `text` | string | The input text fed to the TTS model |
| `category` | string | One of the 6 categories below |

## Categories & Target Distribution
~50 rows per split, ~8–9 rows per category:

| Category | Description | Examples |
|---|---|---|
| `prosody` | Long/complex sentences, questions vs statements, lists with cadence | "And finally..." lists, rhetorical questions |
| `edge_cases` | Numbers, dates, currencies, abbreviations, URLs, mixed-case proper nouns | "£1,234.56", "Dr.", "iPhone", "ChatGPT" |
| `phonetic` | Homophones, foreign loanwords, unusual names | "read/read", "croissant", "Siobhan", "Wojciech" |
| `punctuation` | Em-dashes, ellipses, parentheses, quoted speech | Does punctuation produce natural pauses? |
| `robustness` | Very short, very long (200+ words), repeated words | "Yes.", "the the the" |
| `domain_specific` | Technical, medical, or legal jargon | Drug names, legal Latin, engineering terms |

## Leakage Prevention
- Each split should use **different instantiations** of each category pattern — same category types but distinct texts
- No identical sentences across splits
- Run n-gram overlap check before publishing

## Phases

### Phase 1 — Data Creation ✅ Complete
- [x] Generated candidate texts via Claude Sonnet (OpenRouter); smoke tested TTS via Router — all 6 categories OK, ~$0.03/100 chars
- [x] Round-trip calibration: initial draft avg WER 0.026 (too easy, 37/50 rows at WER=0)
- [x] Replaced 29 trivially-easy rows with harder texts targeting known TTS failure modes
- [x] Validated curated dataset: avg WER 0.067 (2.5× harder), hard rows (WER>0.30) up from 0→2
- [x] Phase 1b public split pushed: `ronanarraig/tricky-tts-public` (avg WER 0.067)
- [x] **Phase 1c hardening**: replaced 10 more rows; pushed final public split (avg WER 0.127, 4.9× harder than initial)
  - Replaced 5 punctuation WER=0 rows with punctuation+numbers/abbreviation hybrids
  - Replaced 2 robustness rows with phone numbers and unit-cluster strings
  - Replaced 1 phonetic row with harder Celtic name + contrastive stress combo
  - Upgraded weakest edge_cases (chemistry notation, WER=0.771) and domain_specific rows
- [x] Placeholder semi-private + private splits exist — need same calibration treatment before Phase 2
- Scripts: `generate_texts.py`, `roundtrip_test.py`, `calibrate_and_rebuild.py`, `validate_new_rows.py`, `phase1c_harder.py`

**Round-trip WER by category (public split, ElevenLabs TTS + AssemblyAI ASR):**
| Category | Avg WER | Notes |
|---|---|---|
| edge_cases | 0.160 | Best WER discriminator — abbrev. ambiguity, units, mixed-case brands |
| domain_specific | 0.105 | Drug names, IUPAC, legal Latin, standards codes |
| phonetic | 0.065 | Difficult names (Caoilfhinn, Przemysław), loanwords, heteronyms |
| punctuation | 0.046 | Complex nested dashes/colons; some rows primarily UTMOS-valued |
| robustness | 0.020 | Repeated words, long utterances — primarily UTMOS-valued |
| prosody | 0.003 | Long complex sentences — WER near-zero by design; UTMOS-discriminating |

### Phase 2 — Difficulty Validation & Evaluation (future)
- [ ] Run UTMOS + Round Trip ASR via Trelis Studio across N diverse TTS models (e.g. Orpheus + 2 others)
- [ ] Compute per-row median score across models — rows where *all* models perform well are candidates for removal
- [ ] Filter out "easy" rows (below difficulty threshold) to ensure the benchmark is genuinely challenging
- [ ] Use median-of-N approach to avoid unfairly penalising any single model's quirks
- [ ] Identify which categories are hardest; expand or resample underperforming categories as needed
- [ ] Re-run final evaluation on filtered, stable splits

## Notes
- Evaluation is entirely handled by Trelis Studio — no audio files stored in the dataset
- Private split: open-source models only (no closed-source APIs)
- Phase 2 deferred until Trelis Studio TTS evaluation pipeline is faster/more stable
