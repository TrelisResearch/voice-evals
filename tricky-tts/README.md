# Tricky TTS

Text-only benchmark for evaluating TTS models on linguistically challenging English inputs.

## Reports
- [Phase 1](reports/phase1.md) — calibration journey (1a-1d), key findings, dataset files
- [Phase 2](reports/phase2.md) — difficulty validation, spoken_form, 5-model eval
- [Phase 3](reports/phase3.md) — reference pipeline, 9-model eval, qualitative analysis
- [Phase 4+5 Roadmap](reports/phase4-roadmap.md) — new 4-category taxonomy, human-referenced eval

## Reference
- [Spoken Form Rules](spoken_form_rules.md) — rules for generating spoken_form column

## Scripts
- [`phase1/`](phase1/) — text generation, calibration, round-trip testing, HF push
- [`phase2/`](phase2/) — spoken_form generation, difficulty filtering, eval
- [`phase3/`](phase3/) — reference pipeline, multi-model eval, dataset builds
