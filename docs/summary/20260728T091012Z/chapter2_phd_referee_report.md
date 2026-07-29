# Formal PhD Referee Report: Chapter 2

## Review decision

**Recommendation: Major Revision — not ready for submission or PhD examination in its current form.**

The chapter addresses a timely and potentially valuable question: whether a post-trained language model can transform macro-financial indicators into FOMC-Minutes-style text and support related policy-action inference. The project also reflects substantial engineering work. Its strongest features are the explicit checkpoint vocabulary, the attempt to split data at meeting level, and the effort to evaluate the system using more than a single text metric.

The present evidence, however, does not yet support the chapter's headline claims. The most serious problems are not matters of prose. They concern target leakage, experiment identity, sample accounting, metric implementation, statistical inference, temporal information availability, and decision-task comparability. In particular, the current executable pipeline passes reference-conditioned teacher analysis into held-out Minutes-rewriting prompts; the available leave-one-out evaluator stores cosine distance while the chapter labels it cosine similarity; and the same textual-result numbers are attributed to different checkpoint comparisons in two manuscript trees. Until those issues are resolved from row-level artifacts, even the chapter's claimed “strongest empirical evidence” must be treated as provisional.

This is a promising applied research chapter, but it is not yet a closed doctoral argument.

## Review scope and evidential status

The primary manuscript reviewed was:

- `docs/Chapter2/chapter2.tex`
- `docs/Chapter2/sections/intro.tex`
- `docs/Chapter2/sections/literature_review.tex`
- `docs/Chapter2/sections/dataset_construction.tex`
- `docs/Chapter2/sections/model_training.tex`

I also inspected:

- the alternative manuscript tree in `docs/report`;
- archived Chapter 2 configurations and reported outputs;
- the active data-building, training, reward, generation, masking, and evaluation code;
- active configuration and metadata files;
- current manifests and pipeline audit summaries;
- the prior review in `docs/Chapter2/chapter2_review.md`.

The review snapshot is Git commit `2e5fc6d8b68693391b660ab2dc20a5117aff9979`. The SHA-256 of the primary `chapter2.tex` was:

```text
a59c5acf0436c34cda0936ee9d60b7bf84d9584517bf89f01f46de01a790ab6d
```

I did not rerun expensive model training or external teacher generation. Several claimed historical result artifacts and merged checkpoints are not present in a form that resolves them to a unique run. Findings about the current code therefore distinguish carefully between:

1. a defect demonstrated in the current executable pipeline;
2. a contradiction in the manuscript or archive;
3. a risk that must be excluded for the historical reported run.

The current pipeline cannot by itself prove that every historical result was affected. Conversely, absent immutable historical lineage, the historical result cannot be presumed clean merely because it predates the current code.

## Severity convention

| Level | Meaning |
|---|---|
| **Critical** | Threatens the validity or identity of a headline result and must be resolved before the result is cited as evidence. |
| **Major** | Materially weakens the doctoral contribution, reproducibility, or interpretation and requires substantive revision. |
| **Minor** | A technical, presentation, or language problem that should be corrected after the design is stabilized. |

## Principal strengths

1. **Important research setting.** Central-bank communication is consequential, text is sparse, and conditional data-to-text generation is a defensible research object.
2. **Ambitious system design.** The chapter attempts to connect data construction, supervised adaptation, reward-guided training, structured generation, input-sensitivity analysis, econometrics, and policy-action inference.
3. **Improved checkpoint exposition.** The `chk-0` through `chk-4` table is a useful intended organizing device, even though the actual lineage is not yet reconciled.
4. **Meeting-level leakage awareness.** The manuscript recognizes that sections from the same meeting must not be split across train and test.
5. **More careful leave-one-out prose than earlier drafts.** The current draft no longer calls the implemented intervention a full Shapley value and acknowledges mixed external evidence.
6. **Recognition of information-set differences.** The benchmark section now warns that ex-ante, same-meeting, and ex-post rows should not be read as one unified leaderboard.
7. **A salvageable core contribution.** With a clean data lineage and redesigned evaluation, this could become a strong applied chapter on historical, indicator-conditioned central-bank text generation.

# I. Critical findings

## C1. Held-out Minutes-rewriting inputs currently contain target-derived information

### Evidence

The manuscript states that the `reference excerpt` is used only for training-set distillation and excluded from evaluation and test preparation:

- `docs/Chapter2/sections/dataset_construction.tex:383`
- `docs/Chapter2/sections/dataset_construction.tex:575`

The current pipeline does something different:

1. `build_analysis_teacher_prompts.py:40-66` builds teacher prompts for `train`, `eval`, and `test` with `include_reference=True`.
2. `common/analysis.py:124-139` inserts each row's `reference_excerpt` whenever that flag is true.
3. `build_minutes_rewrite_dataset.py:37-61` loads the teacher answer for every split as `raw_analysis`.
4. `build_minutes_rewrite_dataset.py:70-87` inserts that `raw_analysis` into the downstream Minutes-rewriting prompt.
5. `build_minutes_rewrite_dataset.py:125-170` uses the official `reference_excerpt` as the supervised response.

The current artifacts confirm the path:

| Split | Teacher prompts | Prompts containing their own reference excerpt | Minutes prompts with non-empty teacher analysis |
|---|---:|---:|---:|
| Train | 3,890 | 3,890 | 3,883 |
| Evaluation | 535 | 535 | 533 |
| Test | 462 | 462 | 461 |

The active materializer currently labels these responses `archived_master_seed` rather than issuing a fresh teacher request (`generate_analysis_teacher_responses.py:101-119`). This does not remove the concern: all 4,887 prompts in the archived QA master contain the reference-excerpt instruction, and the archived responses are joined back to the newly hashed reference-bearing rows by sample ID.

The official target appearing as an evaluation label is normal in reference-based evaluation. The validity problem is that a representation generated **from that target** is also passed into the model as an evaluation input. Removing the literal excerpt string from the student-facing template does not remove information already encoded in the teacher-generated analysis.

### Impact

This creates an indirect target-leakage path:

```text
official held-out excerpt
    -> reference-conditioned teacher prompt
    -> teacher analysis
    -> held-out Minutes-rewriting input
    -> generated text compared with the same official excerpt
```

Consequently, the reported cosine/BERTScore advantage cannot presently be interpreted as clean generation from macro-financial indicators. It may partly measure how well the downstream model reconstructs information already supplied by the target-conditioned teacher.

### Required modification

Rebuild evaluation and test inputs so that no target text, target-derived summary, target-selected reasoning trace, or target-based retrieval result is an ancestor of the model input. Recommended clean conditions are:

1. direct indicator tables plus date and section;
2. a teacher analysis produced only from those indicator tables;
3. optionally, a leaked/reference-conditioned condition retained solely as a diagnostic upper bound.

Add explicit fields to every row:

```text
reference_used_in_model_input
reference_used_in_teacher_input
reference_used_for_candidate_selection
reference_used_as_evaluation_target
teacher_model_revision
teacher_prompt_hash
source_row_ids
```

### Acceptance test

- A machine-readable provenance graph demonstrates that no evaluation/test target is an ancestor of any evaluated input.
- A literal and semantic leakage audit is run on every held-out prompt and intermediate representation.
- All headline results are rerun on the clean condition.
- The reference-conditioned condition, if retained, is labelled as an oracle or diagnostic condition and is not used for the main claim.

## C2. The reported checkpoint graph and comparator identities are not recoverable

### Evidence

The manuscript says `chk-3` and `chk-4` branch from `chk-2`:

- `docs/Chapter2/chapter2.tex:44-63`
- `docs/Chapter2/sections/model_training.tex:122-180`

The active and archived artifacts disagree:

- `configs/main/minutes_alignment_sft.yaml:1` starts from `output/training/main/merged/analysis_sft`, i.e. the SFT checkpoint rather than analysis GRPO.
- `configs/main/decision_sft.yaml:1` starts from analysis SFT.
- `configs/main/decision_grpo.yaml:1` starts from an additional decision-SFT checkpoint that is absent from the manuscript's five-node graph.
- `metadata/main/checkpoint_lineage.json:16-24` also makes the downstream checkpoints children of `checkpoint_1`, not `checkpoint_2`, and omits decision SFT.
- The archived synthetic-text configuration starts from `llama_sft_20250522`, not the GRPO checkpoint: `docs/Chapter2/archive/configs/sft_synthetic_20250521.yaml:4-6`.
- The archived decision configuration also starts from `llama_sft_20250522`: `docs/Chapter2/archive/configs/grpo_decision_20250601.yaml:4-7`.
- The primary chapter identifies a DeepSeek-R1-Distill-Llama-8B base (`chapter2.tex:122-134`), while the default active SFT configuration uses `google/gemma-4-E2B-it` (`configs/main/analysis_sft.yaml:1`).
- That active SFT configuration writes `analysis_sft_e2b_it` (`analysis_sft.yaml:38,44`), whereas downstream configurations request `analysis_sft`.

The same text-similarity numbers are attributed to different comparisons:

- `chk-3` versus `chk-0` in `docs/Chapter2/chapter2.tex:591-594,642-672`;
- Checkpoint-3 versus Checkpoint-2 in `docs/report/report.tex:902-913,961-1015`.

### Impact

The causal interpretation of every training stage depends on parentage. A Minutes model initialized from SFT is not evidence about the incremental contribution of prior GRPO. Likewise, a `chk-3` versus `chk-0` comparison answers a different question from `chk-3` versus `chk-2`.

At present, a reader cannot resolve a table row to:

- a unique base model and revision;
- a parent checkpoint;
- a data and split manifest;
- a configuration;
- a selected training step;
- a generation configuration;
- a result artifact.

### Required modification

Create one immutable experiment registry. Every checkpoint must record:

```text
checkpoint_id
base_model_id and immutable revision
parent_checkpoint_hash
adapter_hash
merged_model_hash
data_manifest_hash
split_manifest_hash
config_hash
code_commit
selected_step
selection_metric
prompt/template hash
result artifact hashes
```

Generate the checkpoint diagram and methods table from this registry rather than editing them manually.

### Acceptance test

- Every number, figure, and table row resolves to existing immutable artifacts.
- The same result is never assigned to two comparator pairs.
- A clean rerun reproduces the registered metrics within a stated numerical tolerance.
- The chapter's checkpoint diagram exactly matches the registry.

## C3. Corpus counts, task units, and split definitions are internally contradictory

### Evidence

The chapter reports incompatible counts for apparently related populations:

- 123 Minutes and 486 sections in `dataset_construction.tex:20,35-41`;
- 128 meetings and 718 sections in `dataset_construction.tex:490,573`;
- 5,781 main labels and 44,178 supplementary labels in prose (`:84`) versus 5,327 and 44,116 in the table (`:94-125`);
- 4,889 analytical paragraphs, Q&A pairs, and training samples used in different places (`:84,420,573`);
- 718 section samples yielding 1,392 pairs with “two” reasoning processes (`:490`), although two per section would imply 1,436;
- 578/72/72 downstream sections (`:601`), totalling 722 rather than 718;
- 1,113 Minutes-training pairs in `chapter2.tex:532`;
- 3,129 SFT rows in the split table (`dataset_construction.tex:587`) versus 3,128 in the results (`chapter2.tex:481`).

The active canonical builder instead fixes:

```text
master rows:                 4,887
raw split rows:              3,890 / 535 / 462
analysis SFT rows:           3,112 / 533 / 461
analysis GRPO rows:          1,082 / 533 / 461
Minutes-alignment rows:      3,883 / 533 / 461
Minutes meeting-sections:      577 /  72 /  70
```

The active builder also creates a chronological meeting split:

- `src/process_fomc_report/build_qa_master.py:272-307`
- train: 2009-01-28 to 2021-11-03;
- evaluation: 2021-12-15 to 2023-06-14;
- test: 2023-07-26 to 2025-01-29.

In contrast, `metadata/main/split_policy.json:2-25` specifies a seed-42 random meeting split under a different path scheme.

The manuscript says SFT and GRPO use 81 and 21 training meetings (`dataset_construction.tex:575-588`). The active assembler sorts **rows** by prompt hash, takes an SFT slice, assigns remaining rows to GRPO, and replays SFT rows (`assemble_analysis_training_data.py:88-126`). These sets span the same 102 training meetings rather than disjoint 81/21 meeting groups.

### Impact

The unit of observation, training exposure, evaluation denominator, and generalization estimand are all unclear. A chronological test evaluates future-period transfer; a random meeting split estimates interpolation across the same historical regime. They are not interchangeable.

### Required modification

Construct a single row-level sample-flow ledger with:

```text
source document
meeting_id
section_id
paragraph_id
indicator_id
sample_id
split_id
stage membership
teacher status
drop reason
target_id
duplicate/weight group
```

Choose one primary split. For a forecasting or future-transfer claim, a chronological split should normally be primary; a random meeting split can be a secondary robustness check. All descendants in the model DAG must inherit the same meeting assignment.

### Acceptance test

- Every count in prose and tables is generated from the ledger.
- Counts balance at each transition and every exclusion has a reason.
- Train/evaluation/test meeting overlap is zero across all ancestors and descendants.
- The manuscript names the split strategy, exact date ranges, and task unit.

## C4. The leave-one-out tables mislabel the implemented metric and use invalid inference

### Evidence

The manuscript defines:

\[
\Delta_i =
\operatorname{Cos}(o_{\mathrm{full}},o_{\mathrm{target}})
-
\operatorname{Cos}(o_{\mathrm{masked},i},o_{\mathrm{target}})
\]

at `docs/Chapter2/chapter2.tex:312-319`.

The available external-target evaluator instead:

- sets the actual Minutes as `base_answer`;
- passes that text as both `subset_with_p` and `generated`;
- compares the masked output as `subset`.

See `jobs/eval/eval_mask.py:538-550`. Because `shapley_value_calc` returns \(f_{\mathrm{with}}-f_{\mathrm{without}}\) (`src/open_r1/validator/shapley/shapley_calc.py:83-97`), the stored statistic is:

\[
1-\operatorname{Cos}(o_{\mathrm{masked},i},o_{\mathrm{actual}}),
\]

which is cosine distance, not the cosine similarity claimed in `chapter2.tex:745,758-795,849-856`.

This resolves an otherwise confusing sign pattern. For example:

- Participants/Labour: displayed unmasked 0.2401 and masked 0.2631 correspond to similarities 0.7599 and 0.7369; masking Labour worsens alignment.
- Participants/PCE: displayed unmasked 0.2401 and masked 0.2314 correspond to similarities 0.7599 and 0.7686; masking PCE improves alignment.

The post-processing code then calculates a percentage difference on aggregate means and applies an independent Welch test (`archive/code/reformat_shapley_result/reformat_shapley_result.py:135-154`), even though full and masked outputs are paired by meeting, section, and generation condition.

Additional problems are:

- the internal synthetic target is the full-prompt output, so the statistic largely tests whether masking changes the text, not whether the indicator is used correctly;
- observations are nested within only 128 meetings;
- dozens of indicator-by-section tests are conducted without a stated multiplicity policy;
- generation randomness and prompt-length changes are not controlled transparently.

### Impact

The displayed levels, signs, captions, and inferential tests cannot be interpreted together as written. The result does not establish sufficiency or factual grounding.

### Required modification

For every meeting \(m\), section \(s\), indicator \(i\), and decoding seed \(r\), store:

\[
s_{\mathrm{full},msr},\qquad
s_{\mathrm{masked},msir},\qquad
\Delta_{msir}=s_{\mathrm{full},msr}-s_{\mathrm{masked},msir}.
\]

Use direct similarities rather than an ambiguously named “Shapley” column. Pair generation seeds or use deterministic decoding. Add:

- meeting-clustered or hierarchical paired bootstrap intervals;
- Holm or false-discovery-rate adjustment;
- prompt-length-preserving neutral masks;
- shuffled-value placebos;
- economically directional counterfactuals.

### Acceptance test

- Captions state unambiguously whether a level is similarity, distance, or a delta.
- Every contribution is a row-level paired delta with a 95% meeting-clustered interval.
- Adjusted p- or q-values are reported.
- The main text calls the result indicator sensitivity or target-relative alignment, not proof of grounding.

## C5. Statistical significance for the headline text result is not reproducible

### Evidence

`chapter2.tex:637-675` reports t-statistics in the thousands and significance stars for cosine and BERTScore, but:

- the sample size and unit of inference are not stated;
- the null hypothesis and standard-error construction are not stated;
- no confidence intervals are reported;
- the active evaluator in `jobs/main/eval_text_similarity.py:59-83,104-133` returns only means and deltas;
- no active row-level text-similarity result artifact exists under the documented evaluation output path.

The available generic helper in `src/utils.py:113-174` creates bootstrap means and then treats the number of bootstrap replicates as observations in a one-sample t-test. If used for these tables, this would make the t-statistic depend incorrectly on the number of resamples. Because the exact historical table-generation lineage is absent, this is a serious risk rather than a conclusive attribution.

The reported examples are modest absolute gains:

- answer cosine: \(0.9290-0.9031=0.0259\);
- answer BERTScore-F1: \(0.8688-0.8558=0.0130\).

Their practical relevance cannot be inferred from significance stars alone.

### Impact

The chapter's central comparison is statistically unauditable. Repeated rows within meetings and sections also invalidate naive row-level standard errors.

### Required modification

Use the same held-out rows and generations for every comparator. Report paired per-row deltas, but conduct uncertainty estimation at the meeting level. Pre-specify primary endpoints, for example:

1. answer-level BERTScore-F1;
2. answer-level cosine similarity from an independent sentence encoder.

Treat other segment-by-metric combinations as secondary and adjust for multiplicity. Add factual and numerical evaluations because semantic similarity cannot establish data correctness.

### Acceptance test

- The result file contains row-level scores, meeting IDs, section IDs, prompt hashes, model hashes, and generation seeds.
- The table reports meetings, sections, rows, absolute effects, 95% paired meeting-clustered intervals, and adjusted tests.
- The comparator identity is unique.
- Re-running the evaluation script reproduces the complete table, including uncertainty.

## C6. The decision result is not a valid like-for-like held-out comparison

### Evidence

The headline table uses different samples:

- `chk-4`: 142/180;
- `chk-0`: 120/178.

The post-2009, pre-2009, and class denominators also differ (`chapter2.tex:939-949`). The exclusions, invalid parses, and missing outputs are not reconciled.

The claim that the models were not directly trained on realized decisions (`chapter2.tex:929`) is false for `chk-4`: the decision GRPO reward explicitly compares the prediction with the ground-truth vote (`sections/model_training.tex:642-653`). Excluding the Action paragraph avoids one type of textual leakage; it does not remove decision-label supervision.

The archive contains another common-sample diagnostic over 237 meetings:

| Method | Accuracy | Balanced accuracy | Macro-F1 |
|---|---:|---:|---:|
| Backbone-0 | 0.5190 | 0.5434 | 0.5591 |
| Checkpoint-4 | 0.6034 | 0.5550 | 0.5995 |
| Majority | 0.7089 | 0.3333 | 0.2765 |
| Lag-1 action | 0.7384 | 0.6424 | 0.6424 |

See `docs/report/report.tex:1348-1406`. This is informative but not a publishable held-out result because it appears to include training meetings.

The reconstructed decision pipeline has an additional transitive split problem. In the available 2026 reconstruction, decision-SFT training meetings overlap with 9/13 decision-GRPO evaluation meetings and 13/13 decision-GRPO test meetings. Because decision GRPO descends from decision SFT, that test is not held out under this lineage.

The synthetic-Minutes decision experiment is also under-defined: “500 synthetic Minutes,” “20 random samples,” and a “sampling interval: 2,000” appear together at `chapter2.tex:966`, followed by enormous t-statistics without denominators or a paired model-difference test.

### Impact

The reported `chk-4` advantage confounds model differences, sample composition, parsing failures, and possibly ancestor exposure. Moreover, the simple lag-1 rule is a strong benchmark that the current conclusion does not confront consistently.

### Required modification

Evaluate all models and baselines on one frozen held-out meeting set. Missing, malformed, or abstaining outputs must remain in the denominator as errors. Report:

- exact basis-point accuracy;
- coarse Cut/No-change/Raise accuracy;
- balanced accuracy and macro-F1;
- class precision and recall;
- confusion matrices;
- invalid-output rate;
- paired bootstrap intervals and McNemar tests.

Separate:

1. training-fit diagnostics;
2. chronological held-out results;
3. pre-2009 robustness;
4. same-meeting/ex-post inference;
5. any genuinely ex-ante forecast.

### Acceptance test

- All comparators use identical meeting IDs and target definitions.
- A transitive split audit passes for every ancestor checkpoint.
- The primary table is generated from one immutable prediction ledger.
- Claims acknowledge when a simple persistence baseline wins.

## C7. The information set does not support an ex-ante forecasting claim

### Evidence

The current indicator filter retains observations using only:

```text
observation_date <= meeting_date
```

See `common/indicators.py:66-71`. It has no publication timestamp or real-time vintage. This permits revised data and period-dated observations that were unavailable at the decision cutoff.

The chapter's 2023-07-26 example uses July 2023 PCE data (`sections/model_training.tex:250-313`). The FOMC meeting occurred on July 25–26, while BEA released the July 2023 Personal Income and Outlays data on August 31. See the [Federal Reserve meeting record](https://www.federalreserve.gov/monetarypolicy/fomcminutes20230726.htm) and the [BEA July 2023 release](https://www.bea.gov/news/2023/personal-income-and-outlays-july-2023).

The helper `IndicatorRepository.load_target_rate` deliberately reads the day after the meeting and returns the resulting rate/action (`common/indicators.py:149-158`). That behavior may be appropriate for target construction, but it is not appropriate in a pre-decision input feature.

Finally, the decision task uses non-action sections of the same meeting's Minutes (`chapter2.tex:926-929`). Minutes are released after the policy decision; the chapter itself acknowledges this at `chapter2.tex:1036-1040`.

### Impact

The implemented task is historical reconstruction or same-meeting/ex-post action inference, not real-time policy forecasting. Revised-vintage and post-decision information may also create look-ahead bias.

### Required modification

Choose one of two defensible paths:

1. **Reframe:** describe the task consistently as ex-post action inference from historical analytical discussion.
2. **Build a true forecast:** establish a pre-announcement cutoff and use only real-time vintages with `release_timestamp <= cutoff`.

For an ex-ante task, the pre-meeting target rate, market data cutoff, timezone, release lags, and document availability must be explicit.

### Acceptance test

- Every input cell has source, vintage, observation period, release timestamp, and cutoff timestamp.
- An automated audit finds zero post-cutoff inputs.
- The 2023-07-26 example contains only information available by the chosen cutoff.
- “Forecast” appears only for experiments passing this audit.

# II. Major methodological and scholarly findings

## M1. The research questions and measurements are not aligned

The central question and five sub-questions ask about grounding, deliberation, trade-offs, information content, market impact, policy outcomes, and interpretability (`intro.tex:34-50`). The evaluation principally measures semantic similarity, a masking diagnostic, one under-specified regression, and action classification (`chapter2.tex:159-162`).

There is no independent held-out test of:

- numerical correctness;
- economic directional correctness;
- structured deliberation or trade-off treatment;
- faithful reasoning;
- interpretability;
- practical decision support.

**Modification:** reduce the chapter to a small set of operational questions and add an RQ-to-evidence table containing task, information cutoff, comparator, primary endpoint, uncertainty method, result, and limitation.

**Acceptance:** every RQ has a pre-specified endpoint, a corresponding result, and an explicit answer in the conclusion.

## M2. The contribution of individual training stages is not identified

The chapter claims that the checkpoint design separates the effects of domain SFT, analytical GRPO, and downstream alignment (`chapter2.tex:44-45`). It does not evaluate the full ladder on the same held-out data. Training loss and in-training reward are not substitutes for held-out ablations.

**Modification:** run:

- Minutes generation: `chk-0`, `chk-1`, `chk-2`, `chk-3`;
- action inference: `chk-0`, `chk-2`, `chk-4`;
- prompting-only, retrieval/copy, and SFT-only baselines where appropriate.

**Acceptance:** one common-sample ablation table reports each incremental effect with paired confidence intervals.

## M3. Weak-label, teacher, and judge validity are not established

The manuscript names ChatGPT-4o as labeler and DeepSeek-R1 as teacher (`dataset_construction.tex:57-84,383`). The active configuration uses DeepSeek/archive sources and null model revisions. No stratified human validation, agreement statistic, error taxonomy, or frozen request/response provenance is reported.

The LLM-as-judge reward is also uncalibrated. A high in-training judge score does not demonstrate faithful or correct reasoning. Service failures are returned as reward zero in the active implementation, making infrastructure failure indistinguishable from bad output (`online_reward.py:142-158`).

**Modification:**

- freeze labeler/teacher/judge model revisions, prompts, dates, temperatures, and hashes;
- validate a stratified sample across period, section, and indicator;
- report agreement and error rates;
- distinguish judge failure, parser failure, invalid output, and substantive score;
- calibrate judge scores against blinded expert ratings.

**Acceptance:** all generated labels and targets resolve to immutable provenance, and judge-human agreement plus failure rates are reported.

## M4. The econometric validation is under-specified and does not show market impact

The chapter omits the sentiment model/version, sample dates and \(n\), Minutes release timing, event window, futures contract and rollover rule, outcome transformation, missing-data protocol, covariance estimator, and bootstrap unit/seed (`chapter2.tex:324-346,870-916`).

The reported t-statistics of 0.6169 and 0.7152 are both statistically weak. Moving a coefficient closer to a benchmark is not a test of equality or model difference. Synthetic text that was never released cannot have caused historical returns; the exercise can at most test whether its sentiment preserves a historical association.

**Modification:** estimate authentic Minutes, `chk-0`, and `chk-3` on the same sample and use paired meeting/time-block resampling. Report \(\beta_3-\beta_0\) and \(\beta_3-\beta_{\mathrm{actual}}\), confidence intervals, HAC or clustered standard errors, placebo dates, and alternative sentiment models/windows. Consider moving this analysis to an appendix if it remains exploratory.

**Acceptance:** a complete regression table and reproducible specification replace the current coefficient/t-statistic narrative; no causal “market impact” wording remains.

## M5. The stated training method and hyperparameters mix incompatible versions

Examples include:

- the prose says QLoRA/FP4 (`model_training.tex:730-734`), while active SFT configs specify `load_in_4bit: false`;
- general learning rate \(3\times10^{-6}\) in prose (`:741`) conflicts with the table's \(10^{-5}\) or \(10^{-6}\) (`:774-777`);
- `chk-3` is one epoch in prose (`:746`) but three epochs in the table and results;
- decision training is described as five epochs/1,940 steps (`chapter2.tex:563`) while the active config specifies ten epochs;
- the chapter's `chk-2` reward weights are reasoning 0.4 and answer 0.6 (`model_training.tex:775`), while the active config uses format/answer/reasoning weights 0.2/0.4/0.4;
- active configurations and manuscript prompt formats mix legacy XML, DeepSeek-native, and Gemma thought-channel conventions.

**Modification:** generate the methods table from the resolved runtime manifest. If the reported run used bf16 LoRA, call it LoRA rather than QLoRA. Publish one tokenizer-rendered golden prompt/response and reward-parser test per model family.

**Acceptance:** prose, table, runtime config, trainer state, and checkpoint metadata agree exactly.

## M6. Automated similarity cannot support the claimed constructs by itself

Cosine and BERTScore measure resemblance. They do not establish factual consistency, correct use of numerical evidence, deliberative realism, faithful reasoning, or interpretability. The result at `chapter2.tex:680` therefore over-interprets automated similarity as “analytical reasoning,” “structural coherence,” and “high fidelity.”

**Modification:** add a blinded expert evaluation on held-out meetings. Separate:

- numerical and directional correctness;
- unsupported claims;
- omissions;
- section/genre appropriateness;
- policy trade-offs;
- internal coherence;
- usefulness of the answer, independently of the generated rationale.

Report inter-rater agreement and adjudication. Add deterministic claim-to-table checks where possible.

**Acceptance:** factual or reasoning fidelity is never inferred solely from semantic similarity.

## M7. Multiple seeds, model selection, and reward robustness are missing

Both generation and reward-guided training are stochastic, yet the chapter relies mainly on one seed and in-training curves. GRPO configurations disable evaluation, while the prose discusses convergence and checkpoint choice. The selection rule is not pre-specified.

**Modification:** define an evaluation-only selection set, selection metric, and stopping rule before touching test data. Report at least several training/generation seeds for core comparisons, plus reward ablations and parser/judge failure rates.

**Acceptance:** the chosen checkpoint is reproducibly selected without test-set inspection, and uncertainty includes seed variation where material.

## M8. The literature review is too thin for a PhD chapter

The literature review is only 29 source lines and approximately 520 words. It does not establish the closest prior task or distinguish:

- central-bank communication analysis;
- monetary-policy forecasting versus ex-post inference;
- controlled data-to-text generation;
- style transfer;
- synthetic-text validity;
- factuality and grounding evaluation;
- reasoning distillation and rationale faithfulness;
- RLHF/GRPO and LLM-as-judge limitations.

**Modification:** organize a critical synthesis around those strands and add a closest-work comparison table with input, output, information timing, method, evaluation, and remaining gap.

**Acceptance:** every novelty claim follows from a demonstrated literature gap rather than a broad “one of the first” assertion.

## M9. The project is not currently reproducible from its documented canonical entry point

The repository contains two incompatible active path schemes:

- the README and `jobs/main/*` expect `dataset/processed/main/...`;
- `configs/main/prompt_pipeline.yaml` and the existing artifacts use `dataset/processed/...`.

The documented audit command fails because `dataset/processed/main/manifests/analysis_minutes_split.json` is absent. The full test suite also fails during collection in the present environment. A focused set produced 32 passes and two errors caused by a missing archived source file required by the canonical master regression test.

The thesis root, bibliography, and clean LaTeX build entry point are not present in the reviewed repository, so the chapter cannot be independently compiled here.

Finally, the tracked prompt-pipeline configuration contains plaintext API credentials. The values are intentionally not reproduced in this report.

**Modification:**

- choose one canonical path scheme and migrate all commands/configs;
- provide a locked environment and smoke-test dataset;
- make the split/data audit runnable from a clean checkout;
- package or document the required archive dependency;
- add a thesis build entry point or a standalone Chapter 2 harness;
- immediately rotate exposed credentials, remove them from the working tree and history, and use environment variables or a secret manager.

**Acceptance:** a clean-checkout CI job builds data manifests, runs tests, reproduces a small evaluation, and compiles the chapter without secrets in tracked files.

# III. Framing and claim calibration

## Recommended primary framing

The current evidence best supports:

> Historical, indicator-conditioned generation of FOMC-Minutes-style sections, followed by same-meeting or ex-post policy-action inference.

It does not presently support real-time forecasting, observed market impact, interpretable reasoning, or operational decision support.

## Claim-replacement matrix

| Current or implied claim | Evidential judgment | Recommended wording |
|---|---|---|
| “Input-grounded synthetic Minutes” | Not established | “Indicator-conditioned Minutes-style text; input sensitivity is evaluated separately.” |
| “Policy forecasting” | Wrong information timing for current task | “Same-meeting/ex-post policy-action inference,” unless a release-vintage forecast is added. |
| “Market impact of synthetic text” | Synthetic text was never released | “Replication of a historical sentiment–market association.” |
| “Interpretable reasoning paths” | Generated rationales are not validated explanations | “Generated or distilled rationales.” |
| “Methodological innovation: SFT + GRPO” | Techniques are established | “A staged application of SFT and GRPO to central-bank data-to-text generation.” |
| “Continuous distribution of qualitative information” | Undefined | Delete or define a precise estimand and demonstrate it. |
| “Actionable decision-support insights” | No user or expert study | “Potential research use,” explicitly subject to validation. |
| “High fidelity” from cosine/BERTScore | Over-broad | “Higher semantic similarity on the specified metric.” |
| “Strongest empirical evidence” | Currently blocked by leakage and lineage | “Provisional automated-similarity result pending clean reconstruction.” |

## Recommended research questions

1. **Generation quality:** Does staged adaptation improve held-out FOMC-style generation relative to specified baselines?
2. **Factual consistency and input sensitivity:** Are generated claims numerically consistent with the supplied indicators, and do controlled input changes produce economically coherent output changes?
3. **Downstream validity:** Do generated texts preserve useful associations for a pre-specified econometric replication and historical policy-action inference?

Delete the deliberation and interpretability questions unless corresponding expert and faithfulness studies are added.

# IV. Required revision programme

## Phase 0: Preserve and secure

1. Freeze all current manuscript, data, configuration, checkpoint, and result artifacts as `legacy`.
2. Do not overwrite legacy metrics during reconstruction.
3. Rotate and remove tracked credentials.
4. Create a result registry before any new run.

**Deliverable:** immutable legacy manifest with hashes and an incident note for credentials.

## Phase 1: Close the data and lineage audit

1. Define the unit of analysis for each task: meeting, meeting-section, paragraph, or indicator-paragraph.
2. Build the sample-flow ledger described in C3.
3. Choose and freeze one global meeting split.
4. Enforce the split transitively through every checkpoint ancestor.
5. Add release timestamps and real-time vintages if any forecast claim remains.
6. Eliminate target-derived features from held-out inputs.
7. Freeze labeler, teacher, and judge provenance.

**Deliverables:**

- sample-flow table;
- split manifest and hash;
- checkpoint DAG;
- target-ancestry leakage report;
- temporal-availability audit;
- teacher/label provenance report.

## Phase 2: Run the minimum decisive experiment set

### A. Clean generation ablation

Evaluate `chk-0`, `chk-1`, `chk-2`, and `chk-3` on the same clean chronological test meetings and identical prompts.

Report:

- semantic similarity;
- numerical/directional consistency;
- unsupported-claim rate;
- section-style expert ratings;
- invalid/truncated output rate;
- paired meeting-clustered intervals.

### B. Clean leave-one-out and counterfactual tests

Use direct paired deltas with controlled decoding, meeting-clustered inference, multiplicity correction, neutral-mask placebos, and directional value interventions.

### C. Common-sample decision inference

Evaluate `chk-0`, `chk-2`, `chk-4`, majority, lag-1, and statistical/text baselines on one held-out set. Keep invalid outputs in the denominator.

### D. Expert evaluation

Use blinded, randomized outputs on held-out meetings. At least two qualified raters should independently score the dimensions in M6, followed by adjudication.

### E. Econometric replication

Either complete the paired, fully specified replication in M4 or move the current analysis to an explicitly exploratory appendix.

**Deliverable:** one immutable prediction-and-score ledger from which all results tables are generated.

## Phase 3: Rewrite the chapter around the verified evidence

Recommended structure:

1. Introduction, narrow research questions, and auditable contributions
2. Critical literature synthesis and conceptual distinctions
3. Task definitions and information sets
4. Data, real-time availability, sample flow, and split protocol
5. Models, checkpoint DAG, and research-specific design decisions
6. Pre-specified evaluation protocol and hypotheses
7. Main results
8. Ablations, robustness, and expert evaluation
9. Limitations, ethics, and artifact governance
10. Conclusion answering each research question

Move standard architecture tutorials, full prompts, extended equations, and engineering details to appendices.

# V. Tables and figures required in the revised chapter

1. **Sample-flow table:** every corpus and task transition with exclusions.
2. **Information-set table:** task, cutoff, available documents/data, and whether ex-ante or ex-post.
3. **Checkpoint registry table:** exact parent, model revision, data/split hash, config hash, and selected step.
4. **Label-quality table:** human validation sample, agreement, and errors by period/section/topic.
5. **Generation ablation table:** common-sample `chk-0` through `chk-3`.
6. **Expert-evaluation table:** factuality, style, coherence, and unsupported claims.
7. **Leave-one-out table:** direct paired deltas with adjusted inference.
8. **Econometric table:** full specifications and paired coefficient differences.
9. **Decision table:** common-sample accuracy, balanced accuracy, macro-F1, invalid rate, and paired tests.
10. **Threats-to-validity table:** leakage, memorization, revised vintages, teacher bias, stochasticity, and external validity.

# VI. Minor and editorial comments

1. A more idiomatic title would be **“Generating Synthetic FOMC Minutes with a Post-Trained Large Language Model.”**
2. The contribution list should be reduced from seven broad claims to approximately three auditable contributions.
3. The “Training Plan” and “Model Training Details” substantially repeat one another.
4. The reference-based/reference-free evaluation discussion is duplicated at `chapter2.tex:153-156`.
5. “Chain of Though” should be “Chain of Thought.”
6. “ROGUE” should be “ROUGE”; “Bag-of-Word (BOG)” should be “bag of words (BoW).”
7. F1 is not classification accuracy, and probabilistic generation does not necessarily select the highest-probability token at every step.
8. Cosine similarity is generally bounded by \([-1,1]\), not necessarily \([0,1]\).
9. The chapter's Llama 3 description conflates releases. Meta's [Llama 3 model card](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) describes 8B and 70B variants with an 8k context, whereas `chapter2.tex:80-81` adds a 405B model and 128k context.
10. The SwiGLU equations and tensor shapes at `chapter2.tex:93-106` require technical correction or removal.
11. The daily EFFR-change frequency table is not a meeting-level distribution of FOMC target actions.
12. The Vote Format Reward equation at `model_training.tex:662-667` describes correctness rather than valid formatting.
13. “Federal Fund Rate” should normally be “federal funds rate”; distinguish the effective rate from the target range.
14. The manuscript should not call GRPO “unsupervised” when it is trained with explicit reward supervision.
15. Remove TODO markers, commented revision history, duplicated passages, and stale authoring notes from the final source.
16. Use numbered lower-level sections where navigation matters; nearly all current subsections are starred.
17. The chapter requires a full technical and academic-English edit after the substantive redesign.

# VII. Submission-readiness gates

The chapter should not be treated as examination-ready until all of the following are true.

## Data and leakage

- [ ] One reconciled sample-flow ledger exists.
- [ ] One immutable primary split exists.
- [ ] No meeting appears in a held-out set and any ancestor training set.
- [ ] No held-out target or target-derived representation appears in an evaluated input.
- [ ] Every ex-ante input passes a release-time and vintage audit.

## Model and result identity

- [ ] Every checkpoint has an immutable parent and hash.
- [ ] Every table row names one comparator pair and one common sample.
- [ ] All prompts, model revisions, generation settings, and seeds are frozen.
- [ ] Every headline number is generated from an existing row-level artifact.

## Statistical validity

- [ ] Paired designs are analysed as paired.
- [ ] Inference is clustered or bootstrapped at meeting level.
- [ ] Multiplicity is addressed.
- [ ] Effect sizes and 95% intervals accompany p-values.
- [ ] Invalid outputs remain visible and in the denominator.

## Construct validity

- [ ] Similarity is not presented as factual grounding.
- [ ] Generated rationale is not presented as faithful interpretation without validation.
- [ ] Forecasting is separated from ex-post inference.
- [ ] Market association is not described as causal impact.
- [ ] Expert or deterministic factual validation accompanies text similarity.

## Reproducibility and presentation

- [ ] The canonical data, training, audit, and evaluation commands run from a clean checkout.
- [ ] Tests and a standalone chapter build pass.
- [ ] No credentials are tracked.
- [ ] The literature review and contribution statement are rewritten.
- [ ] All technical terms, tables, equations, and prose pass specialist copyediting.

# Final recommendation

The chapter should receive **Major Revision**. I would not recommend submitting it for examination in its current form, because the central result is not yet traceable to a clean held-out design and several downstream conclusions rely on non-canonical or statistically invalid comparisons.

The project is nevertheless worth continuing. The core idea is relevant, the engineering base is substantial, and the manuscript has already improved in how it discusses leave-one-out evidence and information-set differences. The most productive revision is not to add more narrative around the existing numbers. It is to freeze one experiment identity, rebuild leakage-free held-out inputs, rerun a minimal checkpoint ablation on common meetings, and rewrite the chapter around the claims that survive those tests.

If those gates are satisfied, the chapter could become a credible PhD contribution on conditional central-bank text generation and historical policy-action inference.

---

## Appendix A. Local verification notes

### Current artifact counts

```text
Canonical QA rows:                 4,887
Meetings:                            128
Raw split:                  3,890 / 535 / 462
Analysis SFT:               3,112 / 533 / 461
Analysis GRPO:              1,082 / 533 / 461
Minutes alignment:          3,883 / 533 / 461
Minutes meeting-sections:     577 /  72 /  70
```

### Local tests

- Full `pytest -q`: stopped at collection with four errors, including missing/incompatible PyTorch components and a stale slow-test import.
- Focused lightweight suite: 32 passed, 2 errored because `archive/data/dataset/raw_data/generate_prompt/data_to_analysis/merged_response_20250512.jsonl` is missing.
- Documented canonical audit: failed because `dataset/processed/main/manifests/analysis_minutes_split.json` is absent.

These checks are diagnostic of the reviewed snapshot, not claims that every historical training environment failed.

### Document build

No thesis root, bibliography, or standalone Chapter 2 build harness was present in the reviewed repository. I therefore performed source-level label and structure checks but could not certify a clean full LaTeX build.

## Appendix B. Snapshot hashes

```text
docs/Chapter2/chapter2.tex
a59c5acf0436c34cda0936ee9d60b7bf84d9584517bf89f01f46de01a790ab6d

docs/Chapter2/sections/intro.tex
45b4f5b3a79645ecb02639fa4978381e09f01d46897c69cb954300070441581d

docs/Chapter2/sections/literature_review.tex
32546151c6bc3f49b9478405d1ad0ab7f5d11324681791d0fae7fe16fe7c56a9

docs/Chapter2/sections/dataset_construction.tex
055a8a4bf5acf1aac69aa2f567f3a63111aa416ba2103d2d2042b20a2f493359

docs/Chapter2/sections/model_training.tex
8a8d0031863e61d7a4c4ef695fae635d97101ad8e1c3a5fc9298cd9a4c7794fa
```
