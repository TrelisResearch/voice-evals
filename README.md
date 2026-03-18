# Voice Evals

Voice evaluation datasets and benchmarks for ASR models, built with [Trelis Studio](https://studio.trelis.com).

## Datasets

| Dataset | Tier | Status | HF |
|---------|------|--------|----|
| AI Terms v1 | Public + Semi-private + Private | Complete | `Trelis/ai-terms-{public,semi-private,private}` |
| AI Terms v2 | Public + Semi-private + Private | Pilot in progress | `ronanarraig/ai-terms-v2-{public,semi-private,private}` |
| Code-Switching (FR/EN/DE/ES) | Public + Semi-private | Planned | — |
| Medical Terms | Public + Semi-private + Private | Planned | — |
| Legal Terms | Public + Semi-private + Private | Planned | — |
| Trelis-OOD | Private only | Planned | Never published |

## Reports

- `reports/ai-terms-v2-partA.md` — v2 build report (what was done, benchmark results, known issues)
- `reports/ai-terms-v2-partB-roadmap.md` — v2 roadmap (pilot plan, full rebuild, Studio feature requests, future datasets)
- `docs/eval-results-ai-terms-v0.md` — v1 full eval results (25+ models)
- `docs/entity-cer-guidance.md` — guidance on entity CER methodology

## Setup

```bash
cp sample.env .env
# Add TRELIS_STUDIO_API_KEY and HF_TOKEN
```

## Key Principles

- **Three-tier architecture**: public (fully open) / semi-private (shared, proprietary models OK) / private (open-source models only — never send to third-party APIs)
- **Leakage prevention**: entity/n-gram overlap detection between splits before publishing
- **Difficulty filtering**: use 3 open-source models to drop rows current models handle easily
