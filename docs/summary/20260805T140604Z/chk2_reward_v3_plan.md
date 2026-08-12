# chk2 Reward v3 / v10 implementation plan

Created: 2026-08-05T14:06:04Z

## Immutable baseline

- Let the active v9 `grounded_analysis_v2` run finish unchanged.
- Preserve `analysis_reward_v2.py`, the v9 resolved config, checkpoints, logs, and reward records.
- Build v10 from the same merged chk1, base release, seeds, and GRPO settings so the reward and completion-contract changes are isolated.

## Reward v3 contract

- Register a new `grounded_analysis_v3` reward without changing v2 behavior.
- Send the complete reconstructed `<think> + <answer>` response to Qwen3.5-9B.
- Require a plain-text final answer; JSON, evidence IDs, schema commentary, Markdown, and output-format rehearsal are not part of the chk2 answer contract.
- Replace the universal 0.25 hard cap with a continuous reward:

  ```text
  base =
      0.50 * judge_score
    + 0.25 * answer_numeric_score
    + 0.15 * structure_score
    + 0.05 * answer_concision
    + 0.05 * reasoning_efficiency

  penalty = min(
      0.50,
      0.20 * major_answer_error
    + 0.08 * minor_answer_error
    + 0.04 * major_think_error
    + 0.02 * minor_think_error
  )

  reward = clip(base - penalty, 0, 1)
  ```

- Return zero only for an empty answer or a quote-validated target-decision/Minutes leakage violation.
- Score numbers in the final answer; retain think numeric diagnostics without allowing meta/schema numbers to hard-cap the sample.
- Separate Judge violations by section, kind, severity, exact candidate quote, and explanation. Formatting, omission, and style violations never contribute factual penalty.

## Runtime and hardware

- Start v10 with `max_completion_length=1536`, `max_prompt_length=2560`, and a 4096-token policy budget.
- Increase the conservative Qwen candidate reserve to 2304 tokens while retaining the 8192-token Judge context and 2048-token Judge response budget.
- Keep four generations, group reward scaling, truncated-completion masking, QLoRA, GPU1 for policy, and GPU0 for the Judge.
- Run a 12-step smoke first. Require no OOM/Judge/context error, GPU1 peak reserved memory at most 22 GiB, clipped ratio at most 25%, and fully-clipped/zero-gradient steps at most 15%.
- On OOM, retry at 1280/1920 policy/Judge reserve. If clipping remains high with at most 20 GiB reserved, test 1792/2688. Do not launch the full run unless one profile passes all gates.

## Validation and promotion

- Add strict unit tests for the v3 Judge schema, quote validation, answer-only numeric grounding, derived numbers, continuous penalties, leakage/empty-answer zeroing, context gates, registry, DAG, and execution receipts while retaining v2 regression coverage.
- Replay v3 over every saved rank-0 v9 completion with no Minutes, teacher target, external review, exact-200 gate, or independent approval.
- Require finite `[0,1]` rewards, less than 30% concentration at any one reward, at most 5% zero-variance groups, and zero factual penalties from format/omission/style categories.
- Compare v9 and v10 on identical validation prompts and seeds using the same v3 evaluator. Promote v10 only if mean and median reward do not regress, data fidelity and trend reasoning do not regress, complete plain answers reach 75%, clipping stays at or below 25%, target leakage is zero, and Judge request errors are zero.

