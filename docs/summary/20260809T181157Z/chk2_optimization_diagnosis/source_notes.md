# Source notes

- Snapshot time: 2026-08-09 UTC. The diagnosis is read-only with respect to models, datasets, checkpoints, and running processes.
- The frozen common test contains 33 prompts for each of chk0, chk1, and chk2 checkpoint-150. All 99 raw completions and their generation metadata are used.
- In-domain termination statistics use all 80 stored chk2 completions from steps 131–150. The rolling series uses every available completion parquet from steps 1–150.
- SFT contract checks cover the immutable compressed release's 1,743 rows: train/eval/test = 1,354/199/190.
- Token counts are recomputed with the exact local `DeepSeek-R1-Distill-Llama-8B` tokenizer under the `fomc_trainer` environment. The executed notebook records the interpreter path.
- A strict periodic suffix requires at least 128 final tokens to repeat an exact fixed token period. This is conservative: near-periodic drift is not included in the 93/99 count.
- `</think>` is the reasoning/answer boundary. The chat template pre-fills `<think>\n`; therefore raw completions should not contain another opening tag.
- Training reward is useful for checkpoint diagnosis but is not judge-independent held-out evidence. Promotion should use the proposed task-aligned test split and answer-only metrics.
- The old 33-row test should remain immutable and be labeled as a secondary long-context raw-evidence-to-Minutes stress benchmark.

