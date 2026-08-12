# Chart map

## repetition_prefix_chart

- Question: Did the model need a longer token budget, or was it already looping well before the old 8,192-token cap?
- Source: `output/evaluation/diagnostics/chk2_tp2_long_tokens_20260809T185409Z/artifact_attempt2/trajectory_analysis.json`
- Grain: one reviewed prefix checkpoint per row.
- Encoding: prefix checkpoint on x; word-trigram repetition percentage on y.
- Decision use: repetition already above 99% at 8,192 falsifies “raise the cap” as a remedy.

## loop_phase_chart

- Question: Where were the 52,000 generated tokens spent?
- Source: the same trajectory analysis.
- Grain: one mutually exclusive trajectory phase per row; phase token counts sum to 52,000.
- Encoding: phase on x; token count on y.
- Decision use: only 58 tokens precede the first exact cycle, locating the failure near the start rather than the context boundary.

## prompt_scale_chart

- Question: How far is the old Minutes stress input from the SFT/GRPO training distribution?
- Source: `docs/summary/20260809T181157Z/chk2_optimization_diagnosis/prompt_contract_comparison.csv`
- Grain: corpus × statistic (median or maximum).
- Encoding: corpus on x; rendered prompt tokens on y; statistic as the meaningful grouped dimension.
- Decision use: separates physical context capacity from task/distribution mismatch.

## reward_loophole_chart

- Question: Did boundaryless, token-capped completions receive the intended fail-closed reward?
- Source: checkpoint steps 1–150 parquet files under `output/training/retrain_v2/chk2_compressed_chk1_v1_reward_v3_long4096_fresh_20260807/adapters/chk2/completions`.
- Grain: completion status group (`no boundary, truncated` vs `boundary present`).
- Encoding: status on x; mean `grounded_analysis_v3` reward on y; row count and positive-advantage rate in tooltips.
- Decision use: demonstrates that the parser/reward fallback rewards the exact failure class instead of rejecting it.
