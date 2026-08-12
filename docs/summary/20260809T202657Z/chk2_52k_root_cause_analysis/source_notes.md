# Source notes

## Primary executed evidence

- `output/evaluation/diagnostics/chk2_tp2_long_tokens_20260809T185409Z/artifact_attempt2/summary.json` is the completed TP=2, 52,000-token run receipt.
- `output/evaluation/diagnostics/chk2_tp2_long_tokens_20260809T185409Z/artifact_attempt2/trajectory_analysis.json` retokenizes the saved output and locates exact periodic runs. It records 52,000 reencoded tokens, matching the generation receipt.
- `docs/summary/20260809T181157Z/chk2_optimization_diagnosis/diagnostic_summary.json` and its CSV companions summarize the frozen 3×33 stress set, recent in-domain chk2 completions, and SFT/GRPO contract statistics.

## Configuration and implementation evidence

- `configs/retrain_v2/chk2_analysis_grpo_compressed_chk1_v1_20260805.yaml` confirms `temperature=0.7/top_p=0.9` in GRPO, `max_completion_length=4096`, and `mask_truncated_completions=true`.
- The installed TRL implementation in the `fomc_trainer` environment was inspected read-only: when truncation masking is enabled, any completion ending without EOS/PAD has its entire completion mask zeroed before loss computation. The portable report cites the bounded derived statement in `evidence_snapshot.json` because the environment-owned package path is not a portable repository source.
- `src/open_r1/structured_response.py` and `src/open_r1/trainer/rewards/reward_funcs/analysis_reward_v3.py` confirm that text without `</think>` is parsed as a plain answer and remains eligible for Judge/numeric/concision reward; the token 4-gram reasoning penalty is bypassed because reasoning is set to empty.
- The frozen step 1–150 parquet files were recomputed with the local chk0 tokenizer: 600 rows total, 14 without a boundary, all 14 exactly 4,096 tokens, all 14 nonzero reward, mean 0.2586976 versus 0.1755188 for the 586 boundary-present rows, and 12/14 positive advantage.
- `models/DeepSeek-R1-Distill-Llama-8B/README.md` recommends temperature 0.5–0.7 (0.6 specifically) to prevent endless repetition and recommends avoiding a system prompt.
- `docs/summary/20260809T152718Z/chk2_checkpoint150_merge_eval/checkpoint_manifest.json` records the verified merged checkpoint hash and parent lineage.

## Interpretation boundaries

- Direct observations: output length/finish reason, lack of boundary/answer, repetition, exact-cycle locations, context headroom, and GPU completion without OOM.
- Mechanism confirmed but historical effect not yet quantified: truncated completions receive no policy-gradient tokens under the current TRL mask.
- Inference requiring A/B: the relative contribution of greedy decoding, task mismatch, system-prompt placement, TP floating-point order, and 8B long-synthesis capacity.
- The TP=1/TP=2 token-25 divergence cannot be attributed solely to tensor parallelism because other vLLM runtime parameters differed.

## Packaging QA

- `artifact.json` is the canonical source-backed report.
- `report.html` is generated from that artifact with the packaged portable report builder.
- If the delivery receipt reports `structural_only`, the artifact payload, runtime roots, and semantic fallback passed exact structural checks, but no compatible installed Chromium was available for per-report browser geometry/interaction verification.
