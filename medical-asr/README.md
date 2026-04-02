# Medical ASR

Entity-aware benchmark and fine-tuning project for medical speech recognition.

## Roadmap

See [roadmap.md](roadmap.md) for full phase-by-phase plan.

## Reports

- [Phase 1](reports/phase1.md) — landscape survey + baseline evaluation *(in progress)*
- [Phase 2](reports/phase2.md) — medical-terms test set build
- [Phase 3](reports/phase3.md) — 10h training data + fine-tuning (Moonshine-tiny, Whisper large-v3)
- [Phase 4](reports/phase4.md) — scale to 100 hours
- [Phase 5](reports/phase5.md) — scale to 1,000 hours (conditional)

## Entity Categories

| Category | Examples |
|----------|----------|
| Drugs / Medications | Brand names, generics, biologics, chemotherapy agents |
| Procedures | Surgical, diagnostic imaging, endoscopy, lab tests |
| Conditions / Diagnoses | ICD-10 range — infectious, oncology, cardiology, neurology, rare |
| Anatomy | Latin terms, common and obscure body regions |
| Organisations | Hospital systems, pharma companies, regulatory bodies |

## Dataset Architecture

Three-tier (follows ai-terms pattern):
- `ronanarraig/medical-terms-public` — fully open
- `ronanarraig/medical-terms-semi-private` — proprietary models evaluated privately via Studio
- `ronanarraig/medical-terms-private` — open-source models only; never published
