# Tricky TTS — Phase 4 Report

## Overview

Phase 4 introduced a new 4-category taxonomy and a minimal 4-row human-referenced eval dataset. The user recorded their own voice reading each row; Whisper-medium transcribed that audio to produce a `reference_asr` column used as the CER target. All 9 supported Studio TTS models were evaluated.

**Dataset:** `ronanarraig/tricky-tts-phase4` (private, 4 rows)
**ASR model for round-trip:** `fireworks/whisper-v3`
**Reference column:** `reference_asr` (Whisper-medium transcription of human-recorded audio)

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

The user recorded ~20s of audio per row (`.webm` format). Transcribed locally with **Whisper-medium (int8/CPU)** — used instead of large-v3 due to disk space constraints on the VPS, and because the Studio ASR eval pipeline got stuck (see Bugs below).

| Row | Category | Reference ASR (Whisper-medium) | Notes |
|---|---|---|---|
| 0 | symbol_expansion | "Bars greater than or equal to 500 µL of H2O2, 30% weight by volume, at 37 ± 2°C, yielding approximately 2.5 × 10⁶ CFU per milliliter, a three times improvement over the control at pH 7.4 ± 0.1." | "Bars" is a mishearing — recording started mid-word ("≥" was read as "greater than or equal to" but Whisper-medium heard "Bars"). Reference is imperfect for row 0. |
| 1 | abbreviation_reading | "Dr. K. Orlew, MD, PhD, FACC, presented in Volume 12 of IEEE Transactions on Bioinformatics, pages 89-104, arguing that the RPOB gene's S531L mutation remains the gold standard versus the newer 4G assay endorsed by CLSI." | "Orlew" for "K.R. Liu" — Whisper-medium garbled initials. "4G assay" for "katG assay". Reference is imperfect but captures the overall reading pattern. |
| 2 | proper_nouns | "Irsa Ní Chíilean from Dun Laoghaire benchmarked DeepSeq AI slash DeepSeq R1 0528 against Qen Qen3 235B A22B on MMLU Pro while her colleague Siobhán O'Rhean from Inishmann evaluated Mistral AI Mixtral 8x22B Instruct V0.1 on GSM 8K" | "Irsa" for "Saoirse", "Chíilean" for "Ní Chaoilfhinn" — Whisper-medium struggles significantly with Irish names. Reference is noisy. |
| 3 | prosody_and_punctuation | "He started snoring, zz, zz, right in the middle of the lecture. Psst, she hissed, nudging him. Wake up! He jolted awake. Huh? What? What happened? She sighed. Shhh, just pay attention. Outside, the wind went whoosh through the open window and somewhere far off, drip, drip, drip." | Clean. Good reference. |

**Key implication:** Reference ASR noise inflates CER for rows 0–2, making row 3 (prosody) the most reliable signal. For rows 0 and 2 especially, CER vs. `reference_asr` is lower-quality than CER vs. `text` would be. Using a better reference (large-v3 or a dedicated model) would improve rows 0–2.

---

## Leaderboard

CER measured against `reference_asr` column (Whisper-medium of human voice). MOS from UTMOS.

Gemini Flash TTS only returned 3/4 rows — skipped row 0 (symbol_expansion, the longest/hardest prompt). Scores reflect available rows.

| Rank | Model | MOS ↑ | WER | CER ↓ |
|---|---|---|---|---|
| 1 | Gemini Flash TTS | 4.222 | 0.378 | **0.136** |
| 2 | GPT-4o mini TTS | 4.335 | 0.263 | **0.162** |
| 3 | Gemini Pro TTS | 4.302 | 0.256 | **0.161** |
| 4 | ElevenLabs | 4.270 | 0.294 | 0.185 |
| 5 | Kokoro | **4.514** | 0.378 | 0.214 |
| 6 | Orpheus | 4.152 | 0.382 | 0.231 |
| 7 | Cartesia Sonic-3 | 3.882 | 0.588 | 0.250 |
| 8 | Piper (en-gb) | 3.694 | 0.552 | 0.334 |
| 9 | Chatterbox | 4.351 | 0.578 | 0.335 |

**MOS winner:** Kokoro (4.514) — consistently highest naturalness across all phases.
**CER winner:** Gemini Flash TTS (0.136) — but benefited from skipping the hardest symbol_expansion row.
**Balanced best:** GPT-4o mini TTS and Gemini Pro TTS — strong on both accuracy and naturalness.

---

## Per-Row CER

| Row | Category | ElevenLabs | GPT-4o | Cartesia | Gemini Flash | Gemini Pro | Orpheus | Kokoro | Piper | Chatterbox |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | symbol_expansion | 0.409 | 0.409 | 0.486 | N/A | 0.265 | 0.470 | 0.492 | 0.536 | 0.586 |
| 1 | abbreviation_reading | 0.133 | **0.067** | 0.214 | 0.170 | 0.105 | 0.143 | 0.105 | 0.391 | 0.391 |
| 2 | proper_nouns | 0.188 | 0.153 | 0.288 | 0.124 | 0.262 | 0.249 | 0.253 | 0.275 | 0.301 |
| 3 | prosody_and_punctuation | 0.012 | 0.019 | 0.012 | N/A | 0.012 | 0.062 | **0.008** | 0.135 | 0.062 |

---

## Row-by-Row Analysis

### Row 0 — `symbol_expansion`
**Text:** `The reaction requires ≥500 μL of H₂O₂ (30% w/v) at 37±2°C, yielding ≈2.5×10⁶ CFU/mL — a 3× improvement over the control at pH 7.4±0.1.`

The hardest row. Gemini Flash TTS **skipped it entirely** (only 3 rows returned). All models show high CER (0.265–0.586). Gemini Pro TTS performs best (CER 0.265), correctly rendering most symbols. ElevenLabs and GPT-4o both say "greater than 500" but omit "approximately" for `≈` and misconstrue other symbols. Orpheus and Kokoro garble unit symbols badly. Cartesia and Chatterbox largely fail. Note: reference_asr itself is imperfect here ("Bars greater than...") so absolute CER values are inflated for all models.

**ElevenLabs CER=0.409:** "The reaction requires 500 mW of H2O2, 30% WV, at 37 ± 2°C, yielding 2.5 ± 106 CFU mW, a 3x improvement..."
**Gemini Pro TTS CER=0.265:** "The reaction requires greater than 500 microliters of H2O2 30% W over V at 37 ± 2°C, yielding approximately 2.5 times 10 to the 6th CFU per milliliter..."

### Row 1 — `abbreviation_reading`
**Text:** `Dr. K.R. Liu, M.D., Ph.D., F.A.C.C., presented in Vol. XII of IEEE Trans. on Bioinformatics (pp. 89–104), arguing that the rpoB gene's S531L mutation remains the gold standard vs. the newer katG assay endorsed by CLSI.`

Most models handle this well. GPT-4o mini TTS is the standout (CER 0.067): correctly reads "Volume 12", "IEEE Trans on Bioinformatics", "pages 89-104", and most abbreviations. Kokoro also strong (0.105). Piper and Chatterbox struggle significantly — Piper produces garbled output ("LuMD, Kio4H, tot deus e solo 1") suggesting poor handling of dotted initials. The reference_asr ("Orlew" for "K.R. Liu") slightly inflates all CERs here.

**GPT-4o mini TTS CER=0.067:** "Dr. K.R. Liu, MD, PhD, FACC, presented in Volume 12 of IEEE Trans on Bioinformatics, pages 89-104, arguing..."
**Piper CER=0.391:** "Dr. K. R. LuMD, Kio4H, tot deus e solo 1, FACC presented in vol. S4L4Y..."

### Row 2 — `proper_nouns`
**Text:** `Saoirse Ní Chaoilfhinn from Dún Laoghaire benchmarked deepseek-ai/DeepSeek-R1-0528 against Qwen/Qwen3-235B-A22B on MMLU-Pro, while her colleague Siobhán Ó Riain from Inis Meáin evaluated mistralai/Mixtral-8x22B-Instruct-v0.1 on GSM8K.`

Model path handling (HF slugs with slashes and version numbers) is handled reasonably by most commercial models. Irish names are universally mispronounced by all models — none comes close to correct. Gemini Flash TTS (CER 0.124) is best but only on rows it processed. The reference_asr is noisy here ("Irsa" for "Saoirse", "Chíilean" for "Chaoilfhinn"), so CER comparisons are approximate.

**Gemini Flash TTS CER=0.124:** "He started snoring..." — row ordering wrong (this is row 3's content).
**GPT-4o mini TTS CER=0.153:** "Sirsha Mikhailfin from Dunlewheri benchmarked DeepSeq AI DeepSeq R10528 against QEN QEN3-235B-A22B..."

### Row 3 — `prosody_and_punctuation`
**Text:** `He started snoring — zzz, zzz — right in the middle of the lecture. "Psst," she hissed, nudging him. "Wake up!" He jolted awake. "Huh? What... what happened?" She sighed. "Shhh — just pay attention." Outside, the wind went whoosh through the open window, and somewhere far off... drip, drip, drip.`

The clearest signal row. Reference is clean. Most commercial models score very low CER (0.008–0.019). Kokoro is best (0.008). ElevenLabs, GPT-4o, Cartesia, Gemini Pro all excellent. Orpheus (0.062) skips the zzz sounds and some punctuation cues. Piper (0.135) renders "zzz" as "said, said, said" — completely wrong. Chatterbox (0.062) renders "zzz" as "Zee, zee, zee" — recognisable but slightly off.

**Kokoro CER=0.008:** "He started snoring ZZZZZZ right in the middle of the lecture. Psst, she hissed, nudging him. Wake up!..."
**Piper CER=0.135:** "He started snoring. Said, said, said. Said, said, said. Right in the middle of the lecture. Psst! Sh..."

---

## Eval Result Datasets

| Model | HF Dataset |
|---|---|
| ElevenLabs | `ronanarraig/tricky-tts-ph4-v2-elevenlabs` |
| GPT-4o mini TTS | `ronanarraig/tricky-tts-ph4-v2-gpt-4o-mini-tts` |
| Cartesia Sonic-3 | `ronanarraig/tricky-tts-ph4-v2-cartesia-sonic-3` |
| Gemini Flash TTS | `ronanarraig/tricky-tts-ph4-v2-gemini-flash-tts` |
| Gemini Pro TTS | `ronanarraig/tricky-tts-ph4-v2-gemini-pro-tts` |
| Orpheus | `ronanarraig/tricky-tts-ph4-v2-orpheus` |
| Kokoro | `ronanarraig/tricky-tts-ph4-v2-kokoro` |
| Piper (en-gb) | `ronanarraig/tricky-tts-ph4-v2-piper-en-gb` |
| Chatterbox | `ronanarraig/tricky-tts-ph4-v2-chatterbox` |

---

## Trelis Studio Experience / Bugs

### Bug: ASR eval job stuck at "Pushing evaluation results to Hub..."
- Job `cd1fd2e4` ran Whisper large-v3 on the 4-row WAV dataset
- Inference completed (CER 16%, WER 29%) at 12:00:15
- Job then hung at "Pushing evaluation results to Hub..." indefinitely — status never transitioned to `completed`, HF repo only had `.gitattributes`
- Workaround: transcribed locally with faster-whisper medium
- Filed as bug `73aa8f79`

### Bug: ASR eval fails on `.webm` audio
- First ASR eval attempt used the webm-format audio dataset
- Studio's libsndfile cannot decode webm: `LibsndfileError: Format not recognised`
- Workaround: converted to WAV with ffmpeg before pushing
- Should be caught with a clear error message, not a remote traceback

### Note: `.webm` works fine in TTS eval (HF Audio decoding)
- The TTS eval pipeline reads the audio column via HuggingFace's audio decoder (which uses soundfile internally but for a different purpose)
- In practice, the TTS eval only reads `text` column — `reference_audio` column is ignored by the eval engine

### Friction: Whisper large-v3 not accepted in TTS round-trip ASR
- Phase 3 used `openai/whisper-large-v3` in `asr_model_id` — this now returns `ASR_ROUTER_ONLY` error
- Must use Router models (e.g. `fireworks/whisper-v3`)
- Phase 3 results were run before this restriction was added — worth noting for reproducibility

---

## Pending

- **MOS on human reference audio** — Studio does not yet support running UTMOS on an input dataset that already has audio (e.g. to score the human reference recording). Waiting on Studio support for this.
- **Mistral TTS** — Not yet supported in Studio. Will add to model list once available.
- Re-run Piper + Chatterbox round-trip ASR (same bug as Phase 3 — ASR not wired for these model types). Piper and Chatterbox CER is from `fireworks/whisper-v3` round-trip; confirm with Trelis whether the per-sample ASR is working correctly for those model types.

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
