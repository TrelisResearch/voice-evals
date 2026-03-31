# Tricky TTS

Text-only benchmark for evaluating TTS models on linguistically challenging English inputs.

## Reports
- [Phase 1](phase1/report.md) — calibration journey (1a-1d), key findings, dataset files
- [Phase 2](phase2/report.md) — difficulty validation, spoken_form, 5-model eval
- [Phase 3](phase3/report.md) — reference pipeline, 9-model eval, qualitative analysis
- [Phase 4](phase4/report.md) — new 4-category taxonomy, human-referenced eval, 10-model leaderboard
- [Phase 4 Roadmap](phase4/roadmap.md) — upcoming phases

## Reference
- [Spoken Form Rules](spoken_form_rules.md) — rules for generating spoken_form column

## Scripts
- [`phase1/`](phase1/) — text generation, calibration, round-trip testing, HF push
- [`phase2/`](phase2/) — spoken_form generation, difficulty filtering, eval
- [`phase3/`](phase3/) — reference pipeline, multi-model eval, dataset builds
- [`phase4/`](phase4/) — 4-row human-referenced eval (new taxonomy), recording tools
