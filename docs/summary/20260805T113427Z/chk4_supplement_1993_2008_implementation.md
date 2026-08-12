# chk4 1993–2008 Decision supplement implementation

## Implemented lineage

```text
archived meeting_date/rate_change + exact ALFRED D-1 evidence
  → deterministic per-meeting evidence compression
  → DeepSeek target-neutral meeting_decision_brief
  → DeepSeek summary-only blind decision audit
  → DeepSeek hidden-gold grounded SFT reasoning
  → core v2 + supplement combined Decision-SFT/GRPO release
```

The blind prediction is never used as a label or admission filter. Canonical
training decisions are derived locally from `rate_change` and reconciled with
the same- or next-day change in `ffr_hist.xlsx`.

## Frozen supplement contract

- Population: `decision_supplement_1993_2008_v1`.
- Grain: one meeting per unique manifest row; 109 candidates, train only.
- Date range: 1993-03-23 through 2008-12-16.
- Actions: hold-0 (67), hike-25 (20), hike-50 (3), cut-25 (9), cut-50 (8), cut-75 (2).
- Source populations: five 13-meeting and four 11-meeting sealed populations.
- Evidence: exact ALFRED `meeting_date - 1 calendar day` vintage, sparse fail-closed acquisition, 24-month source lookback.
- Admission: at least 98 meetings, at least 11 valid topics per meeting, all three macro categories, and all six action classes.
- Summary input: at most two series per topic with latest values and relative short/medium/long changes; absolute observation and meeting dates remain in manifests.
- Provider: `deepseek-v4-pro`, thinking enabled, high reasoning effort, JSON output, concurrency 8, no fallback, one format repair.

SOFR is retained in the sealed source inventory so the sparse acquisition can
prove it unavailable for this period. It is never replaced with current-
vintage or synthetic evidence.

## Entry point

Preflight without ALFRED or DeepSeek calls:

```bash
conda run -n fomc_trainer bash \
  run/generation/run_chk4_supplement_pipeline.sh --dry-run
```

Full pipeline:

```bash
conda run -n fomc_trainer bash \
  run/generation/run_chk4_supplement_pipeline.sh
```

Resume an interrupted acquisition or provider stage:

```bash
conda run -n fomc_trainer bash \
  run/generation/run_chk4_supplement_pipeline.sh --resume
```

The shell locks the release directory, verifies `DEEPSEEK_API_KEY` inside the
`fomc_trainer` environment, and runs source acquisition, compression,
summaries, blind prediction/audit, hidden-gold targets, combined materialization,
and release QA in order.

## Artifacts and current state

Artifacts live under:

```text
output/data/retrain_v2/chk4/decision_supplement_1993_2008_v1/
```

The implementation dry-run completed on 2026-08-05 with zero network and API
requests. It froze and validated the 109 candidate manifest, separate pre-2009
roster/registry, label audit, and nine-population source plan. Exact-D-1 ALFRED
acquisition and all DeepSeek stages have not been started by this implementation
run.

Formal chk4 DAG/training integration remains separate and must bind the final
combined training release only after the active chk2 stage is sealed.
