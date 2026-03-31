# Tricky TTS — Phase 4 Report

## Overview

Phase 4 introduced a new 4-category taxonomy and a minimal 4-row human-referenced eval dataset. The user recorded their own voice reading each row; Whisper large-v3 (via Studio ASR eval) transcribed that audio to produce a `reference_asr` column used as the CER target. All 9 supported Studio TTS models were evaluated.

**Dataset:** `ronanarraig/tricky-tts-phase4` (private, 4 rows)
**ASR model for round-trip:** `fireworks/whisper-v3`
**Reference column:** `reference_asr` (Whisper large-v3 transcription of human-recorded audio, via Studio ASR eval)

---

## New Taxonomy

| Category | What it tests |
|---|---|
| `symbol_expansion` | Unicode symbols, units, operators — `≥`, `μL`, `±`, `×10⁶` |
| `abbreviation_reading` | Acronyms, initialisms, roman numerals, dotted titles — `IEEE`, `rpoB`, `Vol. XII`, `F.A.C.C.` |
| `proper_nouns` | Irish/Celtic names, HF model paths, brand names — `Saoirse Ní Chaoilfhinn`, `deepseek-ai/DeepSeek-R1` |
| `prosody_and_punctuation` | Em-dashes, ellipses, onomatopoeia, rhythm — `zzz`, `Psst`, `whoosh`, `drip, drip, drip` |

---

## Reference Audio & ASR Quality

The user recorded ~20s of audio per row (`.webm` format). The Studio ASR eval job (`openai/whisper-large-v3`) eventually completed after 43 minutes total — the HF push alone took 33 minutes for a 2.63MB file (~1.3 KB/s). See Bugs section.

| Row | Category | Reference ASR (Whisper large-v3) | Notes |
|---|---|---|---|
| 0 | symbol_expansion | "fires greater than or equal to 500 microlitres of H2O2, 30% weight by volume, at 37 ± 2°C, yielding approximately 2.5 × 10⁶ CFU per milliliter, a 3× improvement over the control at pH 7.4 ± 0.1." | "fires" is a mishearing at the very start — but the rest is accurate. |
| 1 | abbreviation_reading | "Dr. K. Orlew, MD, PhD, FACC, presented in Volume 12 of IEEE Transactions on Bioinformatics, pages 89-104, arguing that the RPOB gene's S531L mutation remains the gold standard versus the newer 4G assay endorsed by CLSI." | "Orlew" for "K.R. Liu" — garbled initials. "4G assay" for "katG assay". Otherwise solid. |
| 2 | proper_nouns | "Kirsia Ní Chíoláin from Dún Laoghaire benchmarked DeepSeq AI/DeepSeq R1-0528 against QEN QEN3-235B/A22B on MMLU Pro while her colleague Siobhán O'Rhean from Inishmann evaluated Mistral AI Mixtral 8x22B Instruct V0.1 on GSM 8K" | "Kirsia" for "Saoirse", "Chíoláin" for "Chaoilfhinn" — even large-v3 struggles with Irish names. Still the best available reference. |
| 3 | prosody_and_punctuation | "He started snoring, zzz, zzz, right in the middle of the lecture. Psst, she hissed, nudging him. Wake up! He jolted awake. Huh? What? What happened? She sighed. Shhh, just pay attention. Outside, the wind went whoosh through the open window and somewhere far off, drip, drip, drip." | Clean. Excellent reference. |

**Key implication:** Row 3 (prosody) has the cleanest reference and most reliable CER signal. Row 0 reference still has a single-word error at the start ("fires") but is otherwise accurate. Rows 1–2 have some garbling of proper names/initials, which inflates CER for those rows across all models.

---

## Reference Audio MOS

Human-recorded audio scored with UTMOS (WAV format, via `ronanarraig/tricky-tts-phase4-wav`).

| Row | Category | MOS |
|---|---|---|
| 0 | symbol_expansion | 4.129 |
| 1 | abbreviation_reading | 4.278 |
| 2 | proper_nouns | 4.276 |
| 3 | prosody_and_punctuation | 4.207 |
| **avg** | | **4.223** |

Human reference scores 4.22 overall — a ceiling for naturalness comparison. All TTS models that score above ~4.1 are competitive in that dimension.

---

## Leaderboard

CER measured against `reference_asr` column (Whisper large-v3 of human voice). MOS from UTMOS. All 10 models returned 4/4 rows. Mistral MOS is job-level aggregate (per-row `utmos_score` is null in output parquet; aggregate available in job result JSON).

| Rank | Model | MOS ↑ | WER | CER ↓ |
|---|---|---|---|---|
| 1 | Gemini Pro TTS | 4.227 | 0.212 | **0.112** |
| 2 | GPT-4o mini TTS | 4.330 | 0.200 | **0.121** |
| 3 | Gemini Flash TTS | 4.184 | 0.222 | 0.122 |
| 4 | ElevenLabs | 4.273 | 0.361 | 0.192 |
| 5 | Kokoro | **4.511** | 0.383 | 0.209 |
| 6 | Orpheus | 4.152 | 0.375 | 0.229 |
| 7 | Cartesia Sonic-3 | 4.019 | 0.548 | 0.259 |
| 8 | Piper (en-gb) | 3.777 | 0.533 | 0.323 |
| 9 | Chatterbox | 4.100 | 0.928 | 0.583 |
| 10 | Mistral Voxtral-Mini | 4.289 | 0.710 | **0.569** |
| — | Human reference | 4.223 | — | — |

**MOS winner:** Kokoro (4.511) — consistently highest naturalness across all phases.
**CER winner:** Gemini Pro TTS (0.112) — also excellent WER (0.212).
**Balanced best:** GPT-4o mini TTS — strong across both accuracy and naturalness, consistent across all 4 rows.
**Mistral Voxtral-Mini:** Last place by a wide margin (CER 0.569) despite competitive naturalness (MOS 4.289) — garbles symbol_expansion entirely and truncates prosody row mid-sentence. High MOS with low accuracy is a pattern typical of fluent-but-hallucinating TTS.

---

## Per-Row CER

Reference: Whisper large-v3 of human-recorded audio.

| Row | Category | ElevenLabs | GPT-4o | Cartesia | Gemini Flash | Gemini Pro | Orpheus | Kokoro | Piper | Chatterbox | Mistral |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | symbol_expansion | 0.450 | 0.286 | 0.513 | 0.254 | **0.138** | 0.476 | 0.471 | 0.503 | 0.693 | 0.741 |
| 1 | abbreviation_reading | 0.133 | **0.071** | 0.171 | 0.114 | **0.071** | 0.133 | 0.085 | 0.389 | 0.346 | 0.630 |
| 2 | proper_nouns | 0.185 | **0.122** | 0.306 | 0.113 | 0.176 | 0.243 | 0.270 | 0.270 | 0.423 | 0.369 |
| 3 | prosody_and_punctuation | **0.000** | 0.004 | 0.046 | 0.008 | 0.065 | 0.065 | 0.008 | 0.131 | 0.869 | 0.535 |

---

## Row-by-Row Analysis

### Row 0 — `symbol_expansion`
**Text:** `The reaction requires ≥500 μL of H₂O₂ (30% w/v) at 37±2°C, yielding ≈2.5×10⁶ CFU/mL — a 3× improvement over the control at pH 7.4±0.1.`

The hardest row. All models show high CER (0.138–0.693). Gemini Pro TTS is clear best (0.138), rendering "greater than or equal to 500 microliters", "30% weight per volume", and "approximately 2.5 times 10 to the 6th". GPT-4o and Gemini Flash also reasonable. Orpheus, Kokoro, Piper all garble unit symbols. Chatterbox almost completely fails (0.693). Note: reference starts with "fires" (mishearing) which inflates all CERs slightly.

**Gemini Pro TTS CER=0.138:** "The reaction requires greater than or equal to 500 microliters of H2O2, 30% weight per volume, at 37 ± 2°C, yielding approximately 2.5 × 10⁶..."
**Chatterbox CER=0.693:** "The reaction requires 500mL of H-PAL-02-O2 at 3d7-2-OdA, yielding 2-PAL-5-tondrin-6-Cl..."

### Row 1 — `abbreviation_reading`
**Text:** `Dr. K.R. Liu, M.D., Ph.D., F.A.C.C., presented in Vol. XII of IEEE Trans. on Bioinformatics (pp. 89–104), arguing that the rpoB gene's S531L mutation remains the gold standard vs. the newer katG assay endorsed by CLSI.`

GPT-4o mini TTS and Gemini Pro TTS joint best (CER 0.071). Both correctly read "Volume 12", "IEEE Trans on Bioinformatics", "pages 89-104" and most abbreviations. Kokoro also strong (0.085). Piper and Chatterbox fail badly — Piper hallucinates entire phrases ("Dr. K.R. Liu and D. Hego-Fleisch, Dr. Verhorn"), Chatterbox garbles abbreviations ("M.A.D., Peyate 8th ACC").

**GPT-4o mini TTS CER=0.071:** "Dr. K. R. Liu, MD, PhD, FACC, presented in Volume 12 of IEEE Trans on Bioinformatics, pages 89-104..."
**Piper CER=0.389:** "Dr. K.R. Liu and D. Hego-Fleisch, Dr. Verhorn, FACC presented in vol. Sorrel-Vervieres et Trolls..."

### Row 2 — `proper_nouns`
**Text:** `Saoirse Ní Chaoilfhinn from Dún Laoghaire benchmarked deepseek-ai/DeepSeek-R1-0528 against Qwen/Qwen3-235B-A22B on MMLU-Pro, while her colleague Siobhán Ó Riain from Inis Meáin evaluated mistralai/Mixtral-8x22B-Instruct-v0.1 on GSM8K.`

GPT-4o mini TTS best (0.122), Gemini Flash close (0.113). Irish names universally mispronounced by all models — even the reference itself has "Kirsia" for "Saoirse". HF model paths (slashes, version numbers) handled reasonably by commercial models. Chatterbox worst (0.423) — hallucinates model names entirely.

**GPT-4o mini TTS CER=0.122:** "Sirsha Ni Hielthin from DUN Laogira benchmarked DeepSeq AI slash DeepSeq R1-0528 against QEN-QEN3-235B-A22B..."
**Chatterbox CER=0.423:** "Sawarisni Kowalfin from Dunlau Hair Benchmark DeepSeek, iDeepSeek R1, ADSeek R1 8528..."

### Row 3 — `prosody_and_punctuation`
**Text:** `He started snoring — zzz, zzz — right in the middle of the lecture. "Psst," she hissed, nudging him. "Wake up!" He jolted awake. "Huh? What... what happened?" She sighed. "Shhh — just pay attention." Outside, the wind went whoosh through the open window, and somewhere far off... drip, drip, drip.`

Clearest signal row — reference is clean. ElevenLabs achieves perfect CER=0.000. GPT-4o, Gemini Flash, Kokoro all near-zero. Orpheus (0.065) skips the zzz sounds entirely. Piper (0.131) renders "zzz" as "said, said, said". Chatterbox catastrophically fails (0.869) — completely garbles the second half of the text.

**ElevenLabs CER=0.000:** "He started snoring, zzz, zzz, right in the middle of the lecture. Psst, she hissed, nudging him. Wake up!..."
**Chatterbox CER=0.869:** "He started snoring. See, Zee? Right in the middle of the lecture. See! Hissed, nudging him. Wak..."

---

## Mistral Voxtral-Mini Results

Model: `mistral/voxtral-mini-tts-2603`. Evaluated after the main 9-model run. UTMOS not available (null). CER=0.569 average — worst of all models tested.

| Row | Category | CER | Notes |
|---|---|---|---|
| 0 | symbol_expansion | 0.741 | Completely garbles: "an inquiry greater than the epis-fertilum occurrence. 40% at the slash V at 2 fring plus 3 hum yield" — no symbol expansion at all |
| 1 | abbreviation_reading | 0.630 | Reads initials/abbreviations partially ("K. R. Liu, MD, PhD, f.a.c.c") but truncates at "PP 89-109" and drops second half |
| 2 | proper_nouns | 0.369 | Best row — handles HF model paths reasonably ("DeepSeek Eye slash DeepSeek R10528"), though Irish names still wrong |
| 3 | prosody_and_punctuation | 0.535 | Truncates mid-sentence ("Huh? What?") — drops ~40% of the text |

**Key failure modes:** No symbol expansion (treats `≥`, `μL`, `×10⁶` as unparseable), early truncation of longer sentences, and inconsistent abbreviation reading. Not yet production-quality for technical text.

**On truncation — ellipsis EOS hypothesis tested:** Row 3 was re-run with `...` replaced by `,` to test whether ellipsis triggers early EOS. Result: **hypothesis not confirmed**. Both versions truncated at roughly the same point ("She sighed."), with near-identical CER (original: 0.444, no-ellipsis: 0.429) and duration (11.7s vs 12.1s). The truncation is a general length/token-budget issue rather than `...` specifically. Voxtral-Mini simply stops generating before completing longer inputs — likely a `max_new_tokens` cap or a model-level length bias. Dataset: `ronanarraig/tricky-tts-ph4-mistral-ellipsis-test`.

---

## Eval Result Datasets

Final run (v3, Whisper large-v3 reference):

| Model | HF Dataset |
|---|---|
| ElevenLabs | `ronanarraig/tricky-tts-ph4-v3-elevenlabs` |
| GPT-4o mini TTS | `ronanarraig/tricky-tts-ph4-v3-gpt-4o-mini-tts` |
| Cartesia Sonic-3 | `ronanarraig/tricky-tts-ph4-v3-cartesia-sonic-3` |
| Gemini Flash TTS | `ronanarraig/tricky-tts-ph4-v3-gemini-flash-tts` |
| Gemini Pro TTS | `ronanarraig/tricky-tts-ph4-v3-gemini-pro-tts` |
| Orpheus | `ronanarraig/tricky-tts-ph4-v3-orpheus` |
| Kokoro | `ronanarraig/tricky-tts-ph4-v3-kokoro` |
| Piper (en-gb) | `ronanarraig/tricky-tts-ph4-v3-piper-en-gb` |
| Chatterbox | `ronanarraig/tricky-tts-ph4-v3-chatterbox` |
| Mistral Voxtral-Mini | `ronanarraig/tricky-tts-ph4-v3-mistral` |
| Human reference (WAV) | `ronanarraig/tricky-tts-phase4-wav` |
| Human reference MOS | `ronanarraig/tricky-tts-phase4-reference-mos` |

---

## Trelis Studio Experience / Bugs

### Bug: ASR eval HF push extremely slow + job status not updating during push
- Job `cd1fd2e4` ran Whisper large-v3 on the 4-row WAV dataset
- Inference completed (CER 16%, WER 29%) at 12:00:15
- HF shard upload then took **33 minutes for a 2.63MB file** (~1.3 KB/s) — total runtime 2593.6s (~43 min)
- During the entire push phase, the job API returned `status: running` with `result: null` — no indication of progress
- Appeared hung; worked around by transcribing locally with Whisper-medium, then re-ran properly once the job completed
- Filed as bug `7a7266ff` (slow push + status not updating during push)

### Bug: ASR eval fails on `.webm` audio
- First ASR eval attempt used the webm-format audio dataset
- Studio's libsndfile cannot decode webm: `LibsndfileError: Format not recognised`
- Workaround: converted to WAV with ffmpeg before pushing
- Should be caught with a clear error message, not a remote traceback

### Note: `.webm` works fine in TTS eval (HF Audio decoding)
- The TTS eval pipeline reads the audio column via HuggingFace's audio decoder (which uses soundfile internally but for a different purpose)
- In practice, the TTS eval only reads `text` column — `reference_audio` column is ignored by the eval engine

### Bug: MOS eval fails on `.webm` audio
- MOS eval submitted on `ronanarraig/tricky-tts-phase4` `reference_audio` column (webm format) — same libsndfile error as ASR eval
- Workaround: pushed separate WAV dataset `ronanarraig/tricky-tts-phase4-wav` with `audio` column in PCM WAV format
- Filed as bug `edef0e0a`

### Friction: Whisper large-v3 not accepted in TTS round-trip ASR
- Phase 3 used `openai/whisper-large-v3` in `asr_model_id` — this now returns `ASR_ROUTER_ONLY` error
- Must use Router models (e.g. `fireworks/whisper-v3`)
- Phase 3 results were run before this restriction was added — worth noting for reproducibility

---

## Timing

| Step | Time |
|---|---|
| Record audio (user) | ~5 min |
| Convert webm→wav | <1 min |
| Push ASR-input dataset to HF | ~5 min |
| ASR eval job (4 samples, Whisper large-v3) | ~10 min (inference only; push hung) |
| Local transcription (Whisper medium, CPU) | ~3 min |
| Push tricky-tts-phase4 with reference_asr | ~5 min |
| Submit 9 TTS eval jobs | <1 min |
| All 9 jobs complete | ~6 min |
| Total | ~35 min |
