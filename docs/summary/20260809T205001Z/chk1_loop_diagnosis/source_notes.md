# Source notes

## Required technical-report structure

- Title: preserved as the first visible block and manifest title.
- Technical summary: direct answer and implication.
- Key findings: prompt/task distribution, binding collapse, sampling comparison, and early-loop trajectory.
- Scope and definitions: sample, baselines, repetition and periodicity definitions.
- Methodology: metadata reconciliation, trajectory recomputation, evidence-value binding, SFT target scan.
- Limitations/robustness: one sampled temperature-0.6 row and non-identical TP/runtime settings.
- Recommended next steps: task-aligned comparison, hierarchical synthesis, insufficiency contract, boundary A/B, then training cleanup.
- Further questions: included because task-aligned behavior and factor contributions remain unresolved.

## Source boundaries

- The completed `chk1_retry/result.json` owns the temperature-0.6 termination and full-output metrics.
- The frozen common-generation JSONL files own the temperature-zero same-row comparison.
- The frozen prompt row owns the 651 evidence facts and exact value/date/series joins.
- The prompt-contract CSV owns SFT/GRPO/stress distribution comparisons.
- The sealed SFT JSONL files own target-boundary and periodic-suffix checks.

## Visual QA

- All charts use native artifact bar charts because the comparisons are small discrete categories, not time trends.
- `checkpoint_comparison` and `prefix_repetition` deliberately use the same family: both are magnitude comparisons across semantic ordered categories; a line would overstate continuity.
- The portable builder's semantic chart tables are retained. If packaging reports `structural_only`, no installed Chromium was available for per-report browser geometry/interaction checks.
