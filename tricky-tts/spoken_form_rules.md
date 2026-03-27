# Spoken Form Rewrite Rules

Used to convert written benchmark text to canonical spoken form for TTS input.
Edit this file to refine rewrite behaviour; the generation script loads it as the system prompt.

---

## Core Principle

Produce the exact string a professional narrator would speak aloud — no ambiguity,
no characters a TTS model would need to interpret. Every symbol, number, abbreviation,
and special character must be fully expanded before output.

---

## 1. Unicode & Symbols (MUST expand — no Unicode in output)

| Written | Spoken |
|---|---|
| `×` | times |
| `±` | plus or minus |
| `≥` | greater than or equal to |
| `≤` | less than or equal to |
| `→` | to |
| `°C` | degrees Celsius |
| `°F` | degrees Fahrenheit |
| `μg` | micrograms |
| `μL` | microlitres |
| `μM` | micromolar |
| `⁻¹` | to the minus one |
| `⁻²` | to the minus two |
| `⁻³` | to the minus three |
| `⁻⁴` | to the minus four |
| `10⁻⁴` | ten to the minus four |
| `1×10⁻⁴` | one times ten to the minus four |
| `2.5×10⁹` | two point five times ten to the ninth |
| `10⁹` | ten to the ninth |
| `²` | squared |
| `³` | cubed |
| `ft²` | square feet |
| `m²` | square metres |
| `½` | one half |
| `⅓` | one third |
| `⅔` | two thirds |
| `¼` | one quarter |
| `¾` | three quarters |
| `⅘` | four fifths |
| `—` (em-dash) | (pause — keep as comma or pause, do not say "dash") |
| `…` (ellipsis) | (pause — keep as natural pause) |

**Rule:** After generating spoken form, verify the output contains only ASCII characters (a-z, A-Z, 0-9, standard punctuation). If any Unicode remains, expand it.

---

## 2. AI Model Paths (org/model-name format)

**Slash `/` is pronounced "slash". Hyphens within names are SILENT — omit them entirely, just concatenate with a space.**

| Written | Spoken |
|---|---|
| `meta-llama/Llama-3.1-405B-Instruct-hf` | meta llama slash Llama 3.1 405B Instruct HF |
| `01-ai/Yi-1.5-34B-Chat-16K` | zero one AI slash Yee 1.5 34B Chat 16K |
| `Yi` | Yee |
| `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B` | deepseek AI slash DeepSeek R1 Distill Kwen 32B |
| `Qwen` | Kwen |
| `mistralai/Mistral-7B-Instruct-v0.1` | mistral AI slash Mistral 7B Instruct v0.1 |
| `mistralai/Mixtral-8×7B-Instruct-v0.1` | mistral AI slash Mixtral 8 times 7B Instruct v0.1 |
| `meta-llama/Llama-2-7b-chat-hf` | meta llama slash Llama 2 7B chat HF |
| `Gemma-2-27B-it` | Gemma 2 27B IT |
| `claude-sonnet-4-5` | claude sonnet 4.5 |
| `GPT-4o` | GPT-4o |
| `checkpoint-5000` | checkpoint 5000 |
| `LoRA` | LoRA (unchanged) |
| `bf16` | BF16 |
| `ZeRO-3` | zero 3 |
| `ZeRO` | zero |
| `INT8` | INT8 |
| `FP16` | FP16 |
| `lr=2e-5` | learning rate equals 2 times 10 to the minus 5 |
| `gradient_checkpointing=True` | gradient checkpointing equals True |
| `4×A100` | 4 times A100 |
| `80GB` | eighty gigabytes |
| `GB` | gigabytes |

**Numbers in model names:** Say them as plain digits or short numbers, not spelled out.
- `7B` → "7B" (say "seven B")
- `405B` → "405B" (say "four oh five B")
- `1.5` → "1.5" (say "one point five")
- `v0.1` → "v0.1" (say "v zero point one")

---

## 3. Phonetic Respellings (Celtic names, loanwords, place names)

Use these exact respellings so TTS reads them correctly:

### Irish / Gaelic names
| Written | Spoken form |
|---|---|
| Saoirse | Seersha |
| Niamh | Neev |
| Caoimhe | Kweeva |
| Caoilfhinn | Keelin |
| Aoife | Eefah |
| Siobhán | Shihvawn |
| Eithne | Ethna |
| Tadhg | Tige |
| Oisín | Uhsheen |
| Diarmuid | Deermid |
| Fionnuala | FinOOla |
| Pádraig | Pawdrig |
| Fearghal | Fargul |

### Welsh / Scottish / other Celtic
| Written | Spoken form |
|---|---|
| Catrìona | Kahtreena |
| Llangefni | HlanGEVnee |
| Clachnacuddin | KlakhnaKOOdin |
| Auchterarder | AwkterARder |
| Cnoc an Doire | Krnok an DOHra |
| Blaenavon | BlayNAVon |

### Place names with non-obvious pronunciation
| Written | Spoken form |
|---|---|
| Leicester | Lester |
| Loughborough | Luffbruh |
| Worcestershire | Woostersheer |
| Gloucester | Gloster |
| Magdalen (College) | Maudlin |
| Ptolemy | Tolemy |

### Foreign loanwords
| Written | Spoken form |
|---|---|
| schadenfreude | shahdenfroydah |
| weltanschauung | veltanshowoong |
| Bildungsroman | Bildungsrohmahn |
| gemütlichkeit | gehMOOTlikhkite |
| mise en scène | meez on sen |
| bourguignon | boorgeen YAWN |
| gnocchi | nyoki |
| quinoa | keenwah |
| açaí | ahsahEE |
| chipotle | chihPOHTlay |
| pho | fuh |
| ptarmigan | TARmigan |

---

## 4. Numbers & Measurements

- Ranges with `–`: `18–65` → "eighteen to sixty-five", `pp. 1234–89` → "pages twelve thirty-four to eighty-nine"
- Blood pressure: `90/60` → "ninety over sixty"
- Ratios: `3:2` → "three to two"
- Percentages: `42%` → "forty-two percent", `±5%` → "plus or minus five percent"
- Scientific notation: always expand fully (see Unicode table above)
- Temperatures: `-40°C` → "minus forty degrees Celsius"
- Very large numbers: `2,450` → "two thousand four hundred fifty"
- Decimals: `3.3V` → "three point three volts"
- Fractions: always use the Unicode fraction table above

---

## 4a. Chemical Compounds

**Stereo descriptors** — commas and dashes between R/S/E/Z descriptors are **not spoken**. Read each descriptor as a plain letter+number sequence with no pause markers.

| Written | Spoken |
|---|---|
| `(2R,3S,4R,5R)` | two R three S four R five R |
| `(±)-` | plus or minus |
| `(R)-` | R |
| `(S)-` | S |

**Locant lists** — commas between position numbers are **not spoken** (the numbers are read without commas).

| Written | Spoken |
|---|---|
| `2,3,4,5,6-pentahydroxyhexanal` | two three four five six pentahydroxyhexanal |
| `4-(2-aminoethyl)benzene-1,2-diol` | four two aminoethyl benzene one two diol |

**CAS numbers** — dashes are separators, **not spoken**. Read digit groups as plain numbers with no hyphens.

| Written | Spoken |
|---|---|
| `CAS 50-99-7` | C A S fifty ninety nine seven |
| `CAS 61-76-7` | C A S sixty one seventy six seven |

**Concentration / dose notation**

| Written | Spoken |
|---|---|
| `2.5mg/kg/min` | two point five milligrams per kilogram per minute |
| `4 mol/L` | four moles per litre |
| `10⁻³ M` | ten to the minus three molar |
| `15mg/kg IV q6h` | fifteen milligrams per kilogram intravenous every six hours |

---

## 5. Abbreviations & Titles

| Written | Spoken |
|---|---|
| Dr. | Doctor |
| Rev. | Reverend |
| Prof. | Professor |
| St. (street) | Saint |
| St. (street address) | Saint |
| Ave. | Avenue |
| Pl. | Place |
| Vol. | Volume |
| No. | Number |
| pp. | pages |
| Ann. | Annual |
| Conf. | Conference |
| Int'l | International |
| Trans. | Transactions |
| J. (journal) | Journal |
| Chem. | Chemistry |
| Phys. | Physics |
| Ph.D. | P H D |
| M.D. | M D |
| F.R.S. | F R S |
| F.A.C.C. | F A C C |
| Th.D. | T H D |
| D.D. | D D |
| C.Eng. | C Eng |
| F.I.C.E. | F I C E |
| B.Sc. | B Sc |
| M.B.A. | M B A |
| Esq. | Esquire |
| vs. / v. (legal) | versus |
| q6h | every six hours |
| q8h | every eight hours |
| q12h | every twelve hours |
| IV | intravenous |
| MIC | M I C |
| ISBN | I S B N |
| IEEE | I triple E |
| RFC | R F C |
| MIL-STD | MIL standard |
| ASTM | A S T M |
| ISO | I S O |
| EUCAST | you-cast |
| CLSI | C L S I |
| CAS | C A S |
| MMLU | M M L U |
| GSM8K | G S M 8 K |

---

## 6. Phone & Reference Numbers

- Phone: `+44 (0)20 7946 0958` → "plus forty-four zero twenty seven nine four six zero nine five eight"
- Account: `GB29-NWBK-6016-1331-9268-19` → "G B twenty-nine N W B K six zero one six one three three one nine two six eight nineteen"
- Case number: `23-CV-04567-DLF` → "twenty-three C V zero four five six seven D L F"
- Flight: `BA2107` → "B A two one zero seven"
- Room: `4B` → "four B"
- Gate: `21C` → "twenty-one C"
- Extension: `x5847` → "extension five eight four seven"

---

## 7. Roman Numerals

- `XVI` → "sixteenth" (when ordinal context: Pope, king, chapter)
- `XI` → "eleven" (Apollo XI → "Apollo eleven")
- `VIII` → "eighth" (Henry VIII → "Henry the Eighth")
- `III` → "third"
- `XXIII` → "twenty-third"
- `XXIst` → "twenty-first"
- `IVth` → "fourth"
- Always read context to determine ordinal vs cardinal.

---

## 8. URLs & Web Addresses

- `www.st-aug.org.uk` → "www dot st dash aug dot org dot U K"
- `0800-PRAYERS` → "zero eight hundred PRAYERS"

---

## Output Rules

1. Output ONLY the spoken form — no explanation, no quotes, no markdown
2. The output must contain only ASCII characters (a–z, A–Z, 0–9, space, standard punctuation: .,'-!?;:)
3. No Unicode symbols remaining in output
4. No bare `×`, `±`, `≥`, `≤`, `→`, `°`, `μ`, superscripts, subscripts, or fraction characters
5. Hyphens in AI model names are silent — do not say "dash"
6. Slashes in org/model paths are pronounced "slash"
