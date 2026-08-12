# Chart map

| Figure | Decision supported | Dataset | Source | Caveat |
|---|---|---|---|---|
| Rendered prompt scale | Whether the old common test is comparable to chk1/chk2 training | `prompt_scale` | Executed diagnostic notebook and `prompt_contract_comparison.csv` | Token counts use the exact local chk0 tokenizer in `fomc_trainer`; the old test remains valid only as an OOD stress benchmark. |
| Termination health | Whether checkpoint-150 itself has lost the ability to reach the answer | `termination_health` | Frozen common-test generations and chk2 completions from steps 131–150 | Training-domain completions are not an independent quality evaluation; they only diagnose termination behavior. |
| Rolling reward through step 150 | Whether checkpoint-150 is weaker than earlier observed checkpoints on its training-domain signal | `rolling_reward` | chk2 completion parquet files through step 150 | Reward is the training evaluator and must not be the sole promotion metric. |
| chk1 answer-contract violations | Whether the next SFT release needs deterministic cleanup | `sft_contract` | Sealed compressed chk1 SFT release | Regex screening is intentionally broad; inspect affected rows before transforming them. |

