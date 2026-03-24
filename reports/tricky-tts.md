# Tricky TTS Dataset — Project Report

## Overview
Text-only benchmark for evaluating TTS models on linguistically and typographically challenging English inputs. 48 rows per split across 6 categories. Models are evaluated at inference time; no audio is stored in the dataset.

**v1 dataset:** `ronanarraig/tricky-tts-{public,semi-private,private}` — 48 rows, 8 per category (Phase 1a–1c)
**v2 dataset:** `ronanarraig/tricky-tts-v2-{public,semi-private,private}` — 48 rows, 6 research-backed categories (Phase 1d)
**ASR measurement model:** `assemblyai/universal-3-pro` (standardized; single model to avoid inconsistency)
**TTS used for calibration:** `elevenlabs/eleven-multilingual-v2` via Trelis Router
**Filtering note:** Single TTS model calibration only (ElevenLabs). Median-of-N filtering across multiple TTS models deferred to Phase 2 (only one TTS model currently available via Router).

---

## Calibration Journey: Three Phases

### Phase 1a — Initial Draft
- Generated 50 texts across 6 categories (9 per category × 3 splits) via Claude Sonnet (OpenRouter)
- Ran smoke test of Trelis Router TTS — all 6 categories synthesized successfully, ~$0.03/100 chars
- BYOKs pre-configured on Router key — no extra setup needed

**Round-trip results (ElevenLabs TTS → AssemblyAI ASR):**
- Overall avg WER: **0.026** — far too easy
- 37/50 rows at WER=0.00
- 0/50 rows above WER=0.30
- **Finding:** ElevenLabs v2 + AssemblyAI is an excellent pipeline. Most natural prose, domain jargon, and even unusual names went through perfectly.

### Phase 1b — First Calibration
Replaced 29 trivially-easy rows with harder texts targeting known TTS failure modes:
- Ambiguous abbreviations (St./Dr./Prof. with multiple valid expansions)
- Dense technical strings (dosages, IUPAC names, legal citations)
- Celtic/Irish proper names (Siobhán, Saoirse, Caoilfhinn)
- Punctuation combined with numbers

**Round-trip results:**
- Overall avg WER: **0.067** — 2.5× improvement
- Easy rows (WER<0.05): 37 → 24
- Hard rows (WER>0.30): 0 → 2

**Category breakdown:**
| Category | Avg WER | Notes |
|---|---|---|
| edge_cases | 0.160 | Best discriminator — St./Dr. ambiguity, units |
| domain_specific | 0.105 | Drug names, IUPAC, legal Latin |
| phonetic | 0.065 | Caoilfhinn (0.14), Saoirse (0.07), résumé (0.08) |
| punctuation | 0.046 | Most rows still near 0 |
| robustness | 0.020 | Grammar novelties handled cleanly |
| prosody | 0.003 | Near-zero by design — UTMOS-discriminating |

### Phase 1c — Second Calibration (Hardening)
Key insight from Phase 1b: **punctuation rows with WER=0 were clever structurally but contained no abbreviations or numbers**, so ASR just stripped the punctuation and matched perfectly. **Robustness rows** like "that that that" and "James while John had had had" — while linguistically interesting — were handled cleanly by TTS+ASR.

**Strategy:**
1. Replace 5 punctuation WER=0 rows with punctuation+numbers/abbreviations hybrids
2. Replace 2 robustness rows with unit-dense, phone-number, and compound-measurement strings
3. Replace 1 easy phonetic row with harder Celtic name + contrastive stress combo
4. Upgrade weakest edge_cases and domain_specific rows with denser symbol/notation strings

**Standout new rows (highest WER):**
| Row | WER | Why it's hard |
|---|---|---|
| `"The pH was 7.2±0.1 @ 25°C; add ≥50 mL H₂SO₄ (conc. ≈98% w/w) per Fig. 3(b)..."` | 0.771 | Special characters (±, ≥, subscripts) broke TTS — garbled output. Edge case for symbol handling. |
| `"Please call 0800 759 3421 extension 47 and enter PIN 9-1-7-3..."` | 0.375 | Phone number format ambiguity: TTS said "one-eight-hundred" vs "zero-eight-hundred" |
| `"The 6ft 2in, 220lb linebacker completed the 40-yard dash in 4.58sec..."` | 0.353 | Dense unit clusters (ft, in, lb, sec) each with different expansion conventions |
| `"Schedule updates (effective Sept. 1st): Mon.–Wed. sessions run 9:00 AM–5:30 PM..."` | 0.317 | Abbreviation chain + range notation; ASR couldn't recover "Mon–Wed" |
| `"St. Peter's Sq. redevelopment (est. $12.5M–$15.3M) spans ≈45,000 ft²..."` | 0.296 | Multiple symbol types, area notation, M for million |

**Round-trip results (Phase 1c):**
- Overall avg WER: **0.127** — 1.9× improvement over 1b, 4.9× over initial draft
- Easy rows (WER<0.05): 37 → 24 → **14**
- Hard rows (WER>0.30): 0 → 2 → **6**

**Final category breakdown:**
| Category | Avg WER | Max WER | Notes |
|---|---|---|---|
| edge_cases | **0.251** | 0.771 | Hardest category — symbols, abbreviation chains, emails |
| punctuation | **0.176** | 0.317 | Transformed by embedding numbers/abbrevs into complex structures |
| domain_specific | **0.130** | 0.237 | Dosage chains, IUPAC names, legal citations |
| robustness | **0.111** | 0.375 | Phone numbers, unit clusters, compound measurements |
| phonetic | **0.090** | 0.200 | Celtic names, German loanwords, heteronym combos |
| prosody | **0.003** | 0.026 | Intentionally low — evaluates naturalness (UTMOS), not accuracy |

---

## Key Findings

### What actually trips up TTS (learned empirically):
1. **Abbreviation ambiguity at highest density** — "St." as Street/Saint/Saint's, "Dr. Prof." chains, credential suffixes (D.Phil., F.R.C.P.)
2. **Phone numbers** — 0800 vs 1-800, digit-by-digit vs grouped, extension notation
3. **Unit clusters** — ft/in/lb/sec/kg all in one sentence, each with different expansion rules
4. **Special Unicode characters** — ±, ≥, ≈, subscripts (H₂SO₄), superscripts (m²) cause TTS to garble
5. **Mixed number formats** — M for million, μg for micrograms, g/m² together
6. **Celtic/Irish names** — Caoilfhinn, Saoirse, Tadhg, Eithne consistently mispronounced
7. **Heteronyms need strong context** — "permit/permit", "object/object" work best when BOTH forms appear in one sentence
8. **German academic loanwords** — Weltanschauung, Bildungsroman handled better than expected

### What does NOT trip up ElevenLabs (potential easy rows for other models too):
- Plain prose, even very long (200+ words)
- Standard medical/legal jargon (prima facie, polytetrafluoroethylene, subcutaneous hematoma)
- Common repeated-word novelties ("that that that", garden-path sentences)
- Standard punctuation (em-dashes, ellipses, parentheses) without embedded tricky content
- Most URLs and email addresses when well-formed

### Category roles in a dual-metric benchmark (WER + UTMOS):
- **edge_cases, domain_specific, phonetic**: primarily WER-discriminating
- **punctuation**: now hybrid — WER-discriminating with embedded numbers, UTMOS-valued for pause/rhythm
- **prosody**: primarily UTMOS-discriminating (WER=0 expected)
- **robustness**: now hybrid — unit-dense rows are WER-discriminating; short/long utterances are UTMOS-valued

---

## Trelis Router / Studio Experience

### Router (used throughout)
- TTS: `elevenlabs/eleven-multilingual-v2` — BYOKs pre-configured, no extra setup needed. Reliable, ~$0.03/100 chars.
- ASR: `assemblyai/universal-3-pro` — strong, reliable, 120s timeout sufficient for longest audio. Standardized on single model to avoid inconsistency noise.
- Sync endpoints worked well for short texts; 120s timeout needed for 200+ word robustness rows.

### Studio
- Orpheus TTS training/eval endpoints available; UTMOS and Round-Trip ASR not yet in API.
- Phase 2 blocked on Studio TTS eval pipeline maturity.

---

## Dataset Files
| File | Description |
|---|---|
| `tricky-tts/phase1c_public.json` | Final curated 48-row public dataset |
| `tricky-tts/phase1c_results.json` | Full round-trip results with WER/CER/transcripts |
| `tricky-tts/curated_results.json` | Phase 1b results |
| `tricky-tts/roundtrip_results.json` | Phase 1a initial results |
| `tricky-tts/generate_texts.py` | Initial text generation |
| `tricky-tts/roundtrip_test.py` | Round-trip evaluation script |
| `tricky-tts/calibrate_and_rebuild.py` | Phase 1b calibration |
| `tricky-tts/validate_new_rows.py` | Phase 1b validation |
| `tricky-tts/phase1c_harder.py` | Phase 1c hardening |

---

## Phase 2 — Planned
- Run UTMOS + Round-Trip ASR across N diverse TTS models via Trelis Studio
- Median-of-N filtering to ensure difficulty holds across models (not just ElevenLabs)
- Semi-private and private splits need the same calibration treatment (currently still initial draft quality)
- Blocked on: Trelis Studio UTMOS/RT-ASR API availability and pipeline stability

---

## Phase 1d — Research-Backed Category Redesign

### Motivation
Phase 1c showed that `prosody` (WER=0.003) and `robustness` (grammar novelties) were weak WER discriminators. Online research into TTS failure modes, including the **EmergentTTS-Eval benchmark (NeurIPS 2025)** and ElevenLabs/NVIDIA documentation, revealed three well-evidenced categories not yet covered:
- **AI/ML technical text** — model names, HuggingFace paths, version strings (highly domain-relevant)
- **Number format ambiguity** — Roman numerals, Unicode fractions, scientific notation, mixed systems
- **Paralinguistics** — interjections, elongation, stutters, onomatopoeia (from EmergentTTS-Eval)

### Category redesign (v2 dataset)
| Category | Replaces | Rationale |
|---|---|---|
| `edge_cases` | kept | Proven strongest WER discriminator |
| `domain_specific` | kept | Consistently hard |
| `phonetic` | kept | Good WER + naturalness signal |
| `ai_tech` | NEW (replaces `prosody`) | Domain-relevant; model name/path formats trip up TTS |
| `number_format` | NEW (replaces `robustness`) | Roman numerals, fractions, ambiguous number reading |
| `paralinguistics` | NEW (replaces `punctuation`) | UTMOS-focused; interjections, elongation, stutters |

### Phase 1d results (public split, ElevenLabs + AssemblyAI)

**First-pass WER (before calibration): 0.142** — already better than Phase 1c's post-calibration 0.127.

| Category | Avg WER | Max WER | Notes |
|---|---|---|---|
| edge_cases | **0.315** | 0.444 | St./credential chains, unit clusters, IBAN numbers |
| domain_specific | **0.241** | 0.381 | Dosage chains, legal citations, microbial strain names |
| ai_tech | 0.217 | 0.389 | HuggingFace paths, Mixtral/DeepSeek/Yi names; some clean rows needed replacement |
| number_format | 0.153 | 0.450 | King Henry III/VIII/XXIInd (0.45!), ⅔/¾ fractions, Act III Scene iv |
| phonetic | 0.096 | 0.200 | Celtic names + loanwords + heteronym combos |
| paralinguistics | **0.022** | 0.067 | Intentionally low — UTMOS-only |
| **Overall** | **0.174** | | After calibration replacing 4 easy rows |

Easy (WER<0.05): 9/48 | Hard (WER>0.30): 10/48

**Improvement over v1 (Phase 1c):**
| Metric | v1 (Phase 1c) | v2 (Phase 1d) |
|---|---|---|
| Overall avg WER | 0.127 | **0.174** |
| Easy rows (WER<0.05) | 14/48 | **9/48** |
| Hard rows (WER>0.30) | 6/48 | **10/48** |

### Standout rows
| Row | WER | Why |
|---|---|---|
| `King Henry the 3rd (III), King Henry the 8th (VIII), and the XXIInd amendment...` | 0.450 | TTS confused by parallel Roman + ordinal forms |
| `The property at 789 St. Clair Ave., measuring 2,450ft² (227.6m²), sold for $875K` | 0.444 | Unit cluster + St. ambiguity + currency |
| `Dr. Zhang, M.D., Ph.D., F.A.C.C., presented at the 5th Ann. Conf., ISBN 978-0...` | 0.417 | Dense credential chain + ISBN |
| `We benchmarked 01-ai/Yi-1.5-34B-Chat-16K against meta-llama/Llama-3.1-70B-Instruct` | 0.389 | HuggingFace paths with unusual org names |
| `Titrate 2.5mg/kg/min of (±)-4-(2-aminoethyl)benzene-1,2-diol (CAS 51-61-6)...` | 0.381 | IUPAC + CAS + dosage notation |

### Key new findings
1. **`ai_tech` is a strong category** — HuggingFace-style paths (org/model-version) are genuinely tricky because TTS must infer how to split and pronounce e.g. `01-ai/Yi-1.5-34B-Chat-16K`. Arrow chains (`LLaMA → fine-tuned → RLHF'd`) also scored high (0.37).
2. **Roman numerals are harder than expected in mixed context** — "Henry the 3rd (III)...XXIInd" hit WER=0.45 because parallel ordinal + Roman forms created confusion about which to expand.
3. **Standard citations are easy** — "Vaswani et al. (2017)" scored WER=0.00. The hard cases are paths and version strings, not conventional academic prose.
4. **`paralinguistics` confirms UTMOS-only role** — WER=0 rows like "DO NOT open that attachment" and "Oh sure, that went REALLY well" are handled perfectly by pipeline. They exist for naturalness/affect scoring.
5. **`number_format` needs a calibration split** — some rows (blood pressure ranges, Henry VIII) are easy for ElevenLabs; fractions and mixed systems are much harder.

### Category disclosure question
You raised whether to include the `category` column in published datasets. Assessment:
- **Arguments for hiding:** Prevents evaluators from cherry-picking easy categories; categories may be inferable from text anyway
- **Arguments for keeping:** Enables per-category analysis; useful for debugging model weaknesses; EmergentTTS-Eval publishes its categories
- **Recommendation:** Keep `category` column in the dataset. The texts are distinctive enough that categories would be obvious. Withholding adds friction without meaningful blinding benefit — unlike held-out test sets, the category label doesn't help a TTS model perform better.

---

## Datasets
| Repo | Version | Rows | Avg WER |
|---|---|---|---|
| `ronanarraig/tricky-tts-public` | v1 (Phase 1c) | 48 | 0.127 |
| `ronanarraig/tricky-tts-v2-public` | v2 (Phase 1d) | 48 | 0.174 |
| `ronanarraig/tricky-tts-v2-semi-private` | v2 (Phase 1d) | 48 | uncalibrated |
| `ronanarraig/tricky-tts-v2-private` | v2 (Phase 1d) | 48 | uncalibrated |

---

## Round-Trip ASR Evaluation — Fundamental Limitations & Mitigations

### The core problem

Round-trip ASR (TTS → ASR → WER against written reference) conflates two independent failure modes:

1. **TTS failure**: the TTS model mispronounced the text
2. **ASR failure**: the ASR model can't transcribe the spoken word even when the TTS pronounced it correctly

These are indistinguishable with a single WER measurement. High WER on a row like "Caoilfhinn" or "01-ai/Yi-1.5-34B-Chat-16K" may reflect excellent TTS that simply exceeds AssemblyAI's vocabulary. A second problem is **reference mismatch**: the written reference "£1,234.56" bears no lexical resemblance to the spoken form "one thousand two hundred and thirty four pounds fifty six pence" — so any ASR-transcribed output will produce artificially high WER even when TTS handled it perfectly.

This is well-documented in the TTS evaluation literature. From research:

> "Traditional ASR-based metrics fundamentally cannot distinguish between correct pronunciation that ASR still fails to recognize, incorrect pronunciation that ASR happens to recognize, and rare words pronounced correctly but outside ASR's training vocabulary."

### How it affects our specific categories

| Category | Problem type | Severity |
|---|---|---|
| `edge_cases` | Reference mismatch (£, ±, abbreviation expansion) | High — WER is systematically inflated by format differences |
| `ai_tech` | ASR OOV (out-of-vocabulary model paths/names) | Medium — some rows WER=0, others high, hard to tell why |
| `domain_specific` | Both OOV + reference mismatch | Medium |
| `phonetic` | ASR OOV (Celtic names like Caoilfhinn) | Medium — ASR may consistently fail regardless of TTS quality |
| `number_format` | Reference mismatch (Roman numerals, fractions) | High — "LVIII" vs "fifty-eight" always high WER |
| `paralinguistics` | Neither — spoken form ≈ written form | Low |

### Mitigation options (evaluated)

**Option 1: Spoken form normalization (recommended for Phase 2)**
Convert the written reference to its expected spoken form *before* computing WER. Tools:
- **PolyNorm** (arxiv 2511.03080) — few-shot LLM-based text normalizer designed for TTS; handles currencies, abbreviations, dates, symbols
- **NVIDIA NeMo text normalization** — WFST + hybrid LM approach; includes audio-based TN that uses CER comparisons against ASR transcripts to pick the acoustically appropriate normalization
- **Custom LLM prompt** — cheapest to implement; prompt an LLM with "convert this text to how it would be spoken aloud" and use that as the WER reference

This directly fixes the reference mismatch problem. Instead of comparing ASR("one thousand two hundred pounds") against "£1,200", compare it against "one thousand two hundred pounds".

Limitation: doesn't fix ASR OOV failures (Caoilfhinn, Yi-1.5-34B); and for ambiguous abbreviations (St. = street vs saint), we'd need context-aware normalization.

**Option 2: Model-as-judge (recommended for ambiguous rows)**
EmergentTTS-Eval (NeurIPS 2025) uses a large audio language model (LALM) as the evaluator instead of WER. The judge receives the audio and the original text, then uses chain-of-thought reasoning to assess whether pronunciation is correct — achieving 90.5% Spearman correlation with human preferences.

This sidesteps both problems: ASR OOV doesn't matter (the judge "hears" the audio directly), and reference mismatch doesn't matter (the judge understands context).

Practical options:
- Gemini 2.5 Pro with audio input (most capable)
- GPT-4o audio
- A smaller LALM fine-tuned for TTS assessment

Cost is the main concern — this is per-sample inference at frontier model prices.

**Option 3: Multiple ASR models + consensus**
Run 3 diverse ASR models. If all 3 produce similar (high or low) WER for a given row, confidence is higher that it reflects TTS quality rather than a single model's OOV. Useful as a noise reduction layer, but doesn't solve the structural issues above. Also: ASR models share vocabulary gaps for rare proper names.

**Option 4: SP-MCQA (comprehension-based)**
SP-MCQA (arxiv 2510.26190) evaluates whether listeners (human or model) can correctly answer multiple-choice questions about the content of synthesized speech. Shown to reveal failures invisible to WER — "low WER does not guarantee high key-information accuracy". Works well for domain-specific content but harder to apply to short utterances.

**Option 5: Per-row WER reliability tagging**
Label each dataset row with its expected WER reliability:
- `wer_reliable: true` — reference has unambiguous spoken form (e.g., simple prose, paralinguistics)
- `wer_reliable: false` — reference mismatch expected (e.g., edge_cases, number_format)

Rows tagged `false` are excluded from WER aggregation and evaluated only via UTMOS or model-as-judge. Simple to implement; honest about limitations.

### Recommended Phase 2 approach

A pragmatic two-layer strategy:

1. **Spoken form normalization for WER reference**: Use an LLM (or NeMo) to generate expected spoken forms for all rows before running WER. This fixes reference mismatch for currencies, abbreviations, numbers — the biggest source of WER inflation — at low cost.

2. **Model-as-judge for phonetically hard rows**: For categories where ASR OOV is likely (`phonetic`, `ai_tech`, rows with Celtic names or model paths), supplement WER with a LALM judge call. Gemini 2.5 Pro with audio input is the practical choice given Router/Studio access.

3. **UTMOS for naturalness**: Unchanged from current plan — covers prosody and paralinguistics where WER is uninformative.

The combination of (1) + (2) + UTMOS gives three independently-motivated signals that together cover all six dataset categories without the confounds in pure round-trip WER.

### Implementation notes for Phase 2

- **Spoken form normalization**: can be scripted with OpenRouter (LLM prompt) before running evaluation
- **LALM judge**: needs audio files as input; could use Router TTS output cached to disk temporarily
- **Per-row reliability tag**: add `wer_reliable` boolean column to dataset schema now, populate based on category
- **WER reference column**: optionally add `spoken_reference` column to dataset for pre-computed spoken forms

Sources: EmergentTTS-Eval (arxiv 2505.23009), SP-MCQA (arxiv 2510.26190), PolyNorm (arxiv 2511.03080), NVIDIA NeMo TN blog, "An ASR Guided Speech Intelligibility Measure for TTS" (arxiv 2006.01463)

---

## Phase 2 Evaluation Methodology (Final Plan)

### On CER vs WER
Studio uses CER (character error rate) for round-trip evaluation. This is the right choice for this benchmark:
- CER gives partial credit for near-correct transcriptions ("Keelin" vs "Caoilfhinn" scores better than total failure)
- For technical strings (model paths, IUPAC names), single character differences are meaningful
- WER over-penalises multi-word expansions of single tokens ("£1,234.56" → one word in WER, many words in spoken form)

All WER figures in this report are proxies only; Phase 2 will use CER throughout.

### The reference-TTS ground truth approach

The three-layer fix described earlier (spoken form normalization + LALM judge + UTMOS) is replaced by a simpler and more principled single approach:

**Pipeline:**
1. For each row, use an LLM to generate the canonical **spoken form** of the reference text
   - "£1,234.56" → "one thousand two hundred and thirty-four pounds and fifty-six pence"
   - "01-ai/Yi-1.5-34B-Chat-16K" → "zero one dash A I slash Yi one point five dash thirty-four B dash chat dash sixteen K"
   - "Caoilfhinn" → "Keelin" (its pronunciation)
2. Run a **strong reference TTS** (e.g. ElevenLabs) on the spoken form → reference audio
3. Run the **fixed ASR model** (AssemblyAI Universal-3 Pro) on reference audio → **ground truth transcript**
4. Run the **test TTS** on the original written text → test audio
5. Run the **same ASR** on test audio → test transcript
6. Compute **CER(ground_truth, test_transcript)**

**Why this works:**
- ASR OOV errors cancel: if ASR consistently transcribes a rare name as "Keelin" regardless of TTS, both ground truth and test paths produce "Keelin", CER = 0 for a correct pronunciation
- Reference mismatch is resolved: the LLM spoken form bridges the written/spoken gap before any audio is involved
- The reference TTS reads plain English (the spoken form), which any competent TTS handles correctly — so reference TTS errors are negligible
- No LALM judge needed for most rows; UTMOS still covers naturalness

**What it doesn't solve:**
- If the test TTS is genuinely ambiguous (e.g., "St." could be "Street" or "Saint"), the LLM must pick one canonical form. The test TTS may choose differently — this is a legitimate failure to penalise.
- Rows where even the spoken form is hard to generate unambiguously (e.g., mathematical notation) may still need a judge call.

**Note on novelty:** A search of the TTS evaluation literature found no prior work using this exact "reference TTS → ASR → ground truth" structure. Related approaches use human transcriptions as multi-reference ground truths (Style-agnostic evaluation, arxiv 2412.07937) or WER of ASR-trained-on-TTS-data, but the specific cancellation of ASR vocabulary gaps via a shared ASR path appears to be a novel contribution. PolyNorm (arxiv 2511.03080) covers the spoken form generation step independently.

### Phase 2 checklist
- [ ] Add `spoken_form` column to dataset: LLM-generated canonical spoken form for each row
- [ ] Tag `wer_reliable` boolean per row (false for edge_cases, number_format, ai_tech where reference mismatch was observed)
- [ ] Implement reference TTS → ASR → ground truth pipeline
- [ ] Run evaluation via Trelis Studio (CER, UTMOS) across multiple TTS models
- [ ] Apply median-of-N difficulty filtering once ≥3 TTS models are available on Router
- [ ] Calibrate semi-private and private splits using same round-trip methodology
