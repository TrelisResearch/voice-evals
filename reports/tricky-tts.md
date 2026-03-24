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
