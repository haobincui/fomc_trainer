# FOMC Trainer: Research, Reproduction, and Maintenance Plan

## 1. Purpose of This Document

This file is the living project contract for the codebase and the dissertation chapter on synthetic FOMC text generation. It is intended to help the author and future maintainers:

- understand the research question and the intended experiment design;
- identify the active code, data, model, and document interfaces;
- reproduce an experiment without silently changing its scientific meaning;
- distinguish historical evidence from current development work;
- plan modifications in a controlled and auditable way; and
- decide when a result is sufficiently supported to enter the dissertation.

This plan was prepared from a local audit on **2026-07-27**, at Git commit `2e5fc6d`. The directory named in the original request is present as `docs/Chapter2` (capitalized), not `docs/chatper2`. The audit also examined the divergent `docs/report` manuscript tree, active source code, configurations, scripts, tests, local datasets, metadata, and available output artifacts.

> **Current status:** the repository contains substantial research work, but it is not yet a clean end-to-end reproduction package. Do not launch a new canonical training run or promote new dissertation claims until the P0 gates in Section 10 are closed.

---

## 2. Research Contract

### 2.1 Research objective

The chapter asks whether a post-trained reasoning-capable large language model can transform macroeconomic and financial indicators into:

1. domain-consistent macro-financial analysis;
2. institutionally styled FOMC Minutes text; and
3. decision-relevant representations from which an FOMC policy action can be inferred.

The motivation is that monetary-policy communication affects financial markets, but historical qualitative observations are sparse and discontinuous. Conditional text generation may expand the set of scenarios available for macro-financial analysis and risk management.

The central research question is:

> Can a fine-tuned reasoning LLM generate FOMC-Minutes-style sections that are faithful to macro-financial inputs, and do those synthetic texts preserve policy-relevant signals?

### 2.2 Intended task decomposition

The dissertation narrative uses five named checkpoints:

| Name | Intended role |
|---|---|
| `chk-0` / `Backbone-0` | Reasoning-capable base model |
| `chk-1` / `Checkpoint-1` | Analysis supervised fine-tuning (SFT) |
| `chk-2` / `Checkpoint-2` | Analysis Group Relative Policy Optimization (GRPO) |
| `chk-3` / `Checkpoint-3` | Minutes-style alignment SFT |
| `chk-4` / `Checkpoint-4` | Policy-action inference adaptation |

This vocabulary is useful only if every name maps to exactly one model revision, parent checkpoint, dataset manifest, configuration, and output artifact. That mapping is currently inconsistent and must be frozen in an experiment manifest before the names are used in future tables.

### 2.3 Evidence dimensions

The chapter evaluates four different properties. They must not be collapsed into a single claim of “model quality.”

| Dimension | Main methods | Valid interpretation |
|---|---|---|
| Textual fidelity | Sentence-level cosine similarity and token-level BERTScore | Closeness to reference Minutes text |
| Indicator reliance | Indicator-block leave-one-out masking | Local prompt sensitivity under a specified masking intervention |
| Economic relevance | Sentiment–market regression and bootstrap summaries | Exploratory directional evidence |
| Policy-action inference | Exact/coarse action metrics and historical/econometric baselines | Ability to infer an action from the stated information set |

### 2.4 Scientific claim boundaries

The following language is mandatory in code comments, tables, and prose:

- **Leave-one-out masking is not full Shapley attribution.** It does not average marginal effects across coalitions and does not prove grounding or sufficiency.
- A synthetic unmasked target measures **internal reliance or self-consistency**. Authentic Minutes as the target measure **external alignment**.
- The current sentiment–market evidence is **exploratory**. Reported mean t-statistics remain far below conventional significance thresholds.
- Inference from non-decision sections of the same meeting is **same-meeting policy-action inference**. Inference from Minutes released after the decision is **ex-post inference**. Neither is real-time forecasting.
- The word **forecasting** is reserved for experiments using only information available before the policy decision.
- Results from different information sets must not be presented as one unified leaderboard.
- Meeting, document, section, paragraph, label assignment, indicator prompt, Q&A pair, reasoning variant, and evaluation row are different sample units and must never be used interchangeably.

### 2.5 Current evidence summary

The most defensible result in `docs/Chapter2` is that Minutes-style adaptation improves the reported textual-similarity metrics. The leave-one-out results show heterogeneous and section-dependent sensitivity, with mixed alignment to authentic Minutes. The econometric result is directional but not conventionally significant.

Decision results are not yet publication-ready:

- `docs/Chapter2` reports `chk-4` at 78.89% versus `chk-0` at 67.42%, but the models are evaluated on different denominators.
- Per the author's scope decision, `docs/report` is excluded from the manuscript review and maintenance path; values found only there are not evidence for this chapter.

Until the underlying row-level artifacts and checkpoint identities are reconciled, these values are historical draft evidence, not canonical claims.

---

## 3. Provisional Source-of-Truth Policy

Authority depends on the type of information:

1. **Research intent and author-approved exposition:** `docs/Chapter2` is the provisional manuscript because it is the source directory identified by the author.
2. **Executable behavior:** active `src/`, `jobs/`, `configs/main/`, and `run/` define what the current code attempts to execute.
3. **Numerical evidence:** only a frozen result artifact linked to row IDs, split manifest, exact checkpoint, command, config hash, code commit, and model revision is authoritative.
4. **`docs/report`:** out of scope by author instruction. Do not use it as manuscript evidence, reconcile it into Chapter 2, or modify it unless the author explicitly reactivates it.
5. **`docs/Chapter2/archive/` and root `archive/`:** provenance only. They are read-only and must not be active runtime dependencies.
6. **README and comments:** explanatory aids, not evidence. They must be updated when the executable contract changes.

When these sources disagree, record the conflict in Section 9 and block claim promotion. Executed, traceable artifacts determine what happened; author approval determines how a verified result is presented.

### 3.1 Document versioning problem

At present, `docs/`, `dataset/`, `models/`, `output/`, and `metadata/` are ignored and contain no Git-tracked files. The local workspace is much larger than the tracked repository, and a clean Git status therefore does **not** mean that dissertation inputs, documents, or results are preserved.

The target policy is:

- track text-based manuscript sources, small manifests, schemas, and checksums in Git;
- use an immutable artifact store, DVC-like registry, institutional repository, or release archive for large data, models, figures, and results;
- use Git LFS only where appropriate and supported;
- record immutable URIs and SHA-256 hashes for every external artifact; and
- document access-controlled reconstruction for WRDS or other restricted data rather than redistributing it without authorization.

---

## 4. Repository Map

| Path | Intended responsibility | Maintenance note |
|---|---|---|
| `configs/main/` | Active data, training, judge, and evaluation settings | Must contain no secret values |
| `configs/accelerate/` | DeepSpeed/FSDP/Accelerate launch settings | Validate duplicate YAML keys |
| `src/process_fomc_report/` | QA reconstruction, labeling, prompt construction, teacher generation, and task dataset assembly | Main data-preparation implementation |
| `src/process_fomc_report/generate_prompt_and_response/templates/` | Versioned prompt interfaces | A template change invalidates prompt hashes and caches |
| `src/open_r1/` | Model loading, TRL trainers, LoRA, response parsing, rewards, and utilities | Adapted from Hugging Face Open-R1 |
| `jobs/train/` | Low-level SFT and GRPO entry points | Training and merging are separate operations |
| `jobs/main/` | High-level routing, canonical dataset tools, audits, baselines, and evaluation | Several defaults currently use the wrong data namespace |
| `jobs/generation/` | Analysis, Minutes, decision, and masking generation | Generation settings and seeds must be recorded |
| `jobs/eval/` | Vote and leave-one-out evaluation | Must use shared parsers and frozen row IDs |
| `run/generate_input/` | Operational data-build wrappers | `chk1` and `chk3` start background jobs |
| `run/train_gemma/`, `run/train_llama/` | Model-family-specific wrappers | Must not share ambiguous output names |
| `dataset/raw_data/` | Immutable upstream or licensed source material | Never rewrite in place |
| `dataset/processed/input_sources/` | Reusable normalized inputs | Derived, checksummed, and reproducible |
| `dataset/processed/pipeline/` | Prompts, teacher outputs, audits, and intermediates | Run-scoped or content-addressed |
| `dataset/processed/train/` | Minimal train/eval/test JSONL datasets | Current row interface is `{prompt, response, provided_data}` |
| `dataset/processed/manifests/` | Row IDs and split membership | Small manifests should be versioned |
| `output/training/main/` | Adapters, merged models, runtime metadata, and curves | Never silently overwrite dissertation evidence |
| `output/evaluation/main/` | Predictions, per-row scores, summaries, tables, and figures | Every chapter result must link here |
| `metadata/main/` | Lineage, external-model roles, split policy, and provenance | Currently ignored and partly stale |
| `docs/Chapter2/` | Provisional chapter source and review material | Must be reconciled and versioned |
| `docs/report/` | Competing repository-aligned manuscript draft | Not a byte-for-byte mirror |
| `archive/` | Frozen historical material | No active reads or writes after P0 migration |

Inherited Open-R1 code-execution utilities and `src/README.md` are not, by themselves, part of the FOMC research mainline.

---

## 5. Intended Data and Model Flow

```text
FOMC Minutes + macro/market series + rate decisions
                         |
                         v
       normalize -> label -> reconcile -> QA master
                         |
                         v
          frozen meeting-level split manifest
                         |
              +----------+-----------+
              |                      |
              v                      v
     teacher analysis prompts     decision inputs
              |                      |
              v                      v
     teacher reasoning/answers    decision datasets
              |
              v
        analysis SFT dataset
              |
              v
          analysis SFT
              |
      +-------+----------+----------------+
      |                  |                |
      v                  v                v
 analysis GRPO     Minutes alignment   decision SFT
                         SFT                |
                                            v
                                      decision GRPO
```

This diagram describes the current configuration DAG more closely than a simple five-step chain. The dissertation version instead says that both downstream branches originate from analysis GRPO. The canonical parentage must be selected from actual experiment evidence and encoded once in a machine-readable lineage registry.

### 5.1 Required row identity

Every derived row must preserve a stable `sample_id` and, where applicable:

- normalized `meeting_date`;
- `section_name` and `topic`;
- source document and source-row identifiers;
- split and split-manifest version;
- prompt/template hash;
- source-data hash;
- target/reference provenance;
- quality flags and exclusion reason;
- teacher/judge model and exact revision;
- response-format version; and
- generation seed and decoding settings.

Minimal training files may contain only the fields required by the trainer, but a one-to-one manifest must retain the full provenance.

### 5.2 Split policy

Meeting-level isolation is non-negotiable. All rows from one meeting must remain in one split, and all tasks that inherit data or checkpoints must use compatible frozen manifests.

Two incompatible implementations currently exist:

- the populated prompt pipeline uses a **chronological** 102/13/13 split:
  - train: 2009-01-28 through 2021-11-03;
  - evaluation: 2021-12-15 through 2023-06-14;
  - test: 2023-07-26 through 2025-01-29;
- `jobs/main/build_datasets.py`, ignored metadata, and `docs/report` specify a **seed-42 random** 102/13/13 meeting split.

Changing the split is a new experiment, not a refactor. Name it with a new `split_id`, rebuild all descendants, and do not compare metrics as if only the code changed. A chronological split is required for any forward-generalization claim. A random meeting holdout may be retained for controlled interpolation experiments if labeled clearly.

### 5.3 Leakage controls

Before training or evaluation, automated checks must verify:

- no meeting overlaps across train, evaluation, and test;
- no target `Committee Policy Action` language appears in policy-inference inputs;
- reference excerpts used for teacher distillation are absent from student evaluation/test prompts;
- no decision-GRPO test meeting appeared in decision-SFT training;
- the same sample or near-duplicate paragraph is not present across splits;
- preprocessing is fit on training data only where applicable; and
- model selection never uses the final test set.

---

## 6. Current Repository Snapshot

The following facts describe the audited local workspace, not a promised stable release.

### 6.1 Data state

| Artifact | Observed state |
|---|---|
| QA master | 4,887 rows across 128 meetings |
| Raw meeting split rows | 3,890 train / 535 evaluation / 462 test |
| Analysis teacher responses | 3,883 / 533 / 461, with 7 / 2 / 1 failed rows |
| Compat analysis SFT | 3,112 / 533 / 461 |
| Compat analysis GRPO | 1,082 / 533 / 461; 771 primary + 311 replay rows |
| Minutes teacher responses | 3,890 / 535 / 462; generation marked complete |
| Minutes training rows | 3,883 / 533 / 461 after the same ten missing analyses are dropped |
| Unique meeting-section units in Minutes manifests | 577 / 72 / 70, 719 total |
| Decision inputs and datasets | Missing |
| `dataset/processed/main/` | Does not exist |
| Llama-converted datasets | Analysis SFT only |

The analysis resume checker remains incomplete because ten teacher rows failed, even though downstream compatible datasets were materialized after dropping them. A retry, accepted-exclusion, and completion policy must be made explicit.

### 6.2 Training and evaluation state

- Local adapters and runtime logs represent several heterogeneous experiments with unsuffixed, `_0`, `_1`, `_e2b_it`, and `_llama` names.
- Only older suffixed analysis merged-model directories are present.
- No canonical merged `analysis_sft`, Minutes-alignment, or decision model is available at the paths expected by current downstream configs.
- The evaluation directory contains only a smoke-level decision-baseline JSON; the chapter’s tables and figures are not linked to canonical per-row evaluation outputs.

### 6.3 Verification state

The codebase does not currently have a green reproducibility baseline:

- most offline unit tests pass;
- two QA reconstruction tests fail because `build_qa_master.py` references an archived path that no longer exists;
- the default shell interpreter also has an incompatible/partial PyTorch-vLLM installation that prevents full test collection;
- the safe test command is currently `pytest tests --ignore=tests/slow`;
- there is no CI workflow or pytest marker configuration;
- YAML validation finds a duplicate `distributed_type` key in `configs/accelerate/fsdp.yaml`; and
- linting reports a significant backlog, including undefined names in active generation code.

GPU training, external API generation, and destructive rebuilds were not used to prepare this plan.

### 6.4 Known implementation hazards

The following observed behaviors should be treated as defects, not as stable interfaces:

- `build_qa_master.py` depends on a stale hard-coded location under `archive/`; a similar source exists elsewhere locally, but a clean rebuild fails.
- `jobs.merge_model` edits `adapter_config.json` before merging. A release merge should operate on a copy and preserve the original adapter evidence.
- Analysis dataset “expectation” inspection can materialize outputs. Planning and inspection must be pure read-only operations.
- Resume markers rely too heavily on row counts instead of source and content hashes.
- Teacher generation repeatedly rewrites whole split files and needs atomic incremental checkpointing.
- Online judge request failures can become reward zero, contaminating training without clearly invalidating the run.
- Some evaluation code manually parses only legacy `<think>/<answer>` text even though current data use the Gemma thought-channel format.
- Several evaluation defaults point to absent `/processed/main/` or obsolete input locations.
- Active generation code contains at least one undefined `Path` reference.
- Direct runtime dependencies are incomplete or only transitively installed.

### 6.5 Chapter review status

`docs/Chapter2/chapter2_review.md` is useful but partially stale. The live TeX has already corrected the previously mismatched section labels, the dataset-table column declaration, and duplicate rate-choice labels. It also now separates internal reliance from external alignment and describes leave-one-out and econometric evidence more cautiously.

Still open are the sample-count ledger, comparator identity, decision denominators, task framing, model/config lineage, reward and quantization descriptions, econometric specification, language editing, and a clean full-thesis LaTeX build.

---

## 7. Known Conflicts Requiring an Authoritative Decision

| Topic | Conflicting states | Recommended resolution |
|---|---|---|
| Manuscript source | `docs/Chapter2` and `docs/report` diverge | Keep `docs/Chapter2` provisional; reconcile useful `docs/report` changes through reviewed commits |
| Data namespace | Populated/tested `dataset/processed/...` versus documented `dataset/processed/main/...` | Standardize on the populated `dataset/processed/...` layout unless a migration is explicitly approved |
| Backbone | DeepSeek-R1-Distill-Llama-8B in chapter/metadata, Gemma-4-E2B in active config | Preserve a named Llama reproduction family and isolate Gemma as a separate experiment family |
| Downstream parentage | Chapter branches from `chk-2`; report/metadata branch from `chk-1`; code adds decision SFT | Derive parentage from the frozen run artifacts, then encode it once in a DAG registry |
| Gemma output name | Analysis SFT writes `analysis_sft_e2b_it`; descendants expect `analysis_sft` | Use one exact stage output path generated from the DAG registry |
| Split policy | Populated chronological split versus planned random seed-42 split | Assign explicit split IDs; use chronological data for forecasting/generalization claims |
| Minutes unit | Chapter reports section-level pairs; active dataset contains indicator-level rows mapped to repeated section targets | Generate a sample-flow ledger and decide whether training is per indicator, paragraph, or aggregated section |
| Response format | Legacy XML, Gemma thought-channel, and DeepSeek formats coexist | Version the format and use the shared response parser in all training and evaluation code |
| Decision task | Forecasting language versus same-meeting/ex-post inputs | Rename the current task “policy-action inference”; add a separate pre-meeting forecasting experiment if needed |
| Judge service | Gemma-4/vLLM and older Gemma-3/Ollama endpoints coexist | Freeze judge ID, revision, server type, endpoint contract, and health check per run |
| Reported results | Unequal denominators and different comparator identities | Recompute every comparison on common frozen row IDs |
| Quantization | Chapter says QLoRA/FP4; active configs disable four-bit loading | Report the executed runtime config, not the intended technique |

No training configuration should be called “main” until these decisions are recorded in a tracked release manifest.

---

## 8. Security and Data-Governance Rules

### 8.1 Immediate credential incident

A plaintext API credential is present in the tracked prompt-pipeline configuration and its Git history. The value must never be copied into documentation, issues, logs, or new commits.

Required response:

1. revoke or rotate the credential immediately;
2. replace config values with empty/null values and environment-variable references;
3. clean Git history if the repository has been shared;
4. scan existing logs, datasets, output artifacts, and all Git revisions;
5. add automated secret scanning to pre-commit and CI; and
6. document the incident without preserving the credential value.

The trainer also has a potential secondary leak: resolved online-judge settings can include the actual API key and are serialized to `resolved_runtime_config.json`. Runtime manifests must record only the environment-variable name and a boolean indicating availability.

### 8.2 External services

Teacher and judge services must have:

- an explicit health check before a run;
- bounded retries with exponential backoff;
- request timeout and rate-limit handling;
- model name, immutable revision if available, provider, endpoint type, and serving date;
- cost, request count, latency, and failure summaries; and
- a fail-closed threshold.

Judge outages must not be silently converted to reward zero, because that confounds infrastructure failure with model quality.

### 8.3 Data governance

- Raw data are immutable.
- Restricted-source data are never redistributed without permission.
- Each series records provider, series identifier, retrieval date, vintage, frequency, units, transformations, access restrictions, and checksum.
- Generated reasoning traces may contain target-derived or proprietary content and must follow the same retention policy as their sources.
- Cleanup commands are dry-run first and must not remove evidence referenced by a dissertation release.

---

## 9. Open Decision Log

Update this table before changing the associated implementation.

| ID | Decision | Current default | Required evidence / owner action | Status |
|---|---|---|---|---|
| D-01 | Canonical manuscript tree | `docs/Chapter2` provisional | Author approves reconciled tree | Open |
| D-02 | Canonical processed-data namespace | Recommend `dataset/processed/` | Update code, docs, metadata, and tests together | Open |
| D-03 | Canonical chapter model family | Historical Llama vs active Gemma unresolved | Freeze reported checkpoint artifacts and model revisions | Open |
| D-04 | Downstream checkpoint parents | Conflicting narratives | Resolve from run-specific evidence | Open |
| D-05 | Primary split policy | Populated chronological split | Author chooses experiment role for chronological and random splits | Open |
| D-06 | Minutes training unit | Active indicator-row dataset | Confirm intended section aggregation | Open |
| D-07 | Decision task label | Same-meeting inference | Remove forecasting claims or add an ex-ante experiment | Open |
| D-08 | Canonical decision result | No canonical table | Common-row reevaluation with invalid-output accounting | Open |
| D-09 | Artifact storage | Local ignored files | Select durable storage and checksum registry | Open |
| D-10 | Restricted-data release policy | Undocumented | Confirm WRDS and other licenses | Open |

---

## 10. Prioritized Roadmap

### P0 — Secure and freeze the scientific identity

#### P0.1 Contain credentials

- [ ] Revoke/rotate the exposed credential.
- [ ] Remove secret values from current configuration and, if necessary, Git history.
- [ ] Redact runtime snapshots and add a regression test.
- [ ] Pass secret scanning on the worktree and Git revisions.

**Acceptance:** no secret value is present in code, config, logs, runtime metadata, datasets, artifacts, or Git history.

#### P0.2 Freeze one Chapter 2 experiment release

- [ ] Create a tracked manifest containing code commit, manuscript hash, config hashes, data hashes, exact model revisions, checkpoint parents, prompt hashes, split ID, seeds, hardware, environment, teacher/judge identity, and result hashes.
- [ ] Map each `chk-*` name to one parent, input manifest, config, adapter, merged model, and evaluation artifact.
- [ ] Distinguish historical reproduction configs from current development configs.

**Acceptance:** every numerical claim, table, and figure maps to an immutable result artifact; no unexplained model, count, reward, quantization, or hyperparameter discrepancy remains.

#### P0.3 Consolidate the runnable path and lineage

- [ ] Select one processed-data namespace.
- [ ] Replace hard-coded duplicate paths with a typed central path/config registry.
- [ ] Fix the Gemma parent/output-name mismatch.
- [ ] Restore decision sources or remove unavailable stages from the active mainline.
- [ ] Move active source dependencies out of `archive/`.
- [ ] Fix the duplicate FSDP YAML key.
- [ ] Add a read-only preflight command for required files, config schema, parent artifacts, disk, GPU, credentials, and model access.

**Acceptance:** two clean data-only builds produce identical sample IDs, split memberships, and hashes; every child stage consumes its declared parent’s output.

#### P0.4 Reconcile samples, splits, and document claims

- [ ] Generate one sample-flow ledger from raw Minutes through every task dataset.
- [ ] Explain every dropped, failed, backfilled, duplicated, and excluded row.
- [ ] Assign explicit split IDs and freeze their meeting lists.
- [ ] Replace hand-entered chapter counts with generated tables.
- [ ] Recompute decision comparisons on identical row IDs.

**Acceptance:** counts in manifests, chapter tables, training logs, and evaluation outputs agree exactly.

#### P0.5 Make a fresh clone restorable

- [ ] Track manuscript text, schemas, small manifests, and checksum registries.
- [ ] Publish or register all permissible large artifacts in immutable storage.
- [ ] Add restore and verification commands.

**Acceptance:** a fresh clone plus documented data/model access passes preflight and restores all required artifacts with matching hashes.

### P1 — Build a reproducible engineering baseline

#### P1.1 Freeze the environment

- [ ] Add a lockfile, versioned Conda environment, or container.
- [ ] Pin Python patch version, CUDA, PyTorch, Transformers, TRL, Accelerate, DeepSpeed, vLLM, FlashAttention, metric models, and all direct dependencies.
- [ ] Pin remote model revisions.
- [ ] Record sanitized package, GPU, driver, and runtime details for every run.

**Acceptance:** installation and `pip check` pass in a clean environment, and the offline test suite is reproducible.

#### P1.2 Make builders immutable and auditable

- [ ] Stop rewriting reusable inputs in place.
- [ ] Merge models from a copy without mutating the source adapter.
- [ ] Separate pure planning/inspection from materialization.
- [ ] Use atomic, run-scoped outputs and promote them only after validation.
- [ ] Replace row-count-only resume checks with content hashes.
- [ ] Make teacher checkpointing incremental and efficient.
- [ ] Define whether the ten current teacher failures are retried or accepted via an exclusion manifest.

**Acceptance:** dry-run and inspection commands perform no writes; interrupted stages resume without corrupting or duplicating outputs.

#### P1.3 Establish CI and test layers

- [ ] Add offline CPU CI for strict YAML loading, lint, unit tests, split checks, config-DAG checks, CLI smoke tests, and secret scanning.
- [ ] Mark GPU, network, external-API, slow, and large-data tests explicitly.
- [ ] Replace stale archive-dependent tests with small committed fixtures or verified artifact fixtures.
- [ ] Guard `model_test.py` so imports never load a large CUDA model.
- [ ] Add a tiny end-to-end fixture covering data build, config parsing, response parsing, generation stub, and evaluation.

**Acceptance:** zero duplicate YAML keys, zero undefined-name lint failures in active code, and all non-integration tests pass in CI.

#### P1.4 Harden external generation and rewards

- [ ] Pin teacher and judge identities.
- [ ] Add service health checks, retry/backoff, and outage thresholds.
- [ ] Record per-request failures without treating infrastructure errors as model scores.
- [ ] Use central response and vote parsers everywhere.

**Acceptance:** a simulated judge outage stops or clearly invalidates the run; it cannot silently lower rewards.

### P2 — Strengthen the empirical chapter

#### P2.1 Dataset and label validation

- [ ] Manually validate a stratified sample of weakly supervised paragraph labels.
- [ ] Report agreement/error rates by section, indicator, and time period.
- [ ] Compare `compat` and `strict` profiles, including all retained quality flags.
- [ ] Audit pre-2009 versus post-2009 comparability.

#### P2.2 Training ablations

- [ ] Evaluate Backbone, analysis SFT, analysis GRPO, Minutes SFT, and decision adaptation on fixed row IDs.
- [ ] Ablate GRPO reward components and weights.
- [ ] Ablate reference-assisted teacher distillation.
- [ ] Compare downstream parentage from analysis SFT versus analysis GRPO.
- [ ] Run multiple seeds and report uncertainty.

#### P2.3 Text and grounding evaluation

- [ ] Pin embedding and BERTScore models and revisions.
- [ ] Use paired confidence intervals and per-section breakdowns.
- [ ] Fix generation seeds and retain per-row outputs.
- [ ] Extend leave-one-out analysis to interaction-aware subset sampling if calling the result Shapley-like.
- [ ] Add factual entailment, numerical consistency, and expert/human review.

Implementation note (2026-07-28): the LOO code path now supports seeded,
replicate-aware generation with input/output hashes and completeness-checked
manifests. A tracked roster freezes the expected indicator and context
universe; pre-generation validation requires identical row-key coverage and
proves that every masked prompt differs from its paired full prompt by one
exact contiguous deletion. The frozen roster supplies indicator markers;
validation additionally requires complete line/block boundaries and distinct
masked prompts across indicators. The named marker must appear in the
removed block's first semantic line, and deletion spans must be pairwise
non-overlapping within a sample, preventing one intervention from absorbing a
second indicator block. Generated rows are bound back to the attested prompt
hash and row identity. Both the generation checkpoint tree
and local embedding-model tree receive immutable inventory fingerprints. The
canonical paired evaluator verifies non-empty meeting/section identity,
generation position, context, model/decoding provenance, and matched seeds;
it writes per-row
`similarity_full`, `similarity_masked`, signed `delta`, exclusion, and audit
records. Summaries average within meeting--replicate before equal-weighting
replicates at the meeting level, keep evaluation contexts separate, and use a
predeclared global Holm family.

The sibling `synthetic_text` audit recovers a seven-indicator row-level pilot,
but its unseeded stochastic generation, unbalanced random sampling, changed
prompt template, and nonmatching indicator/section population exclude it from
the canonical Chapter 2 analysis. It is retained only for a separately
labelled legacy prompt-perturbation diagnostic or provenance checks. A
dedicated re-scorer (`jobs.eval.eval_legacy_7_indicator_pilot`) and detached
launcher (`run/eval_legacy_7_indicator_pilot.sh`) recover the 2,655 complete
triplets, cache unique-text embeddings, and emit both target-similarity
components plus the signed delta. They use
`data/training/input_qa.json` as the historical scoring target and the
index-aligned tagged QA file only for meeting/section metadata; substituting
the cleaned tagged output would change the target text. The same audit
validates the actual-Minutes content in the 697-row
`synthetic_text_20250518.jsonl` candidate against all 3,481 observed rows of
the 2025-05-20 reason file. The candidate must still be frozen as a new
versioned reference; it is not evidence of byte identity with the missing
historical input.

Generation-only follow-up (2026-07-28): the repository now freezes separate
13-meeting pilot-eval and formal-test populations and constructs each
population's 338-row (`13 meetings × 26 indicators`) input ledger from keyless
ALFRED historical-vintage CSV responses. The only information cutoff is the
previous calendar day: the requested vintage and availability-as-of date both
equal `meeting_date - 1 day`, every included observation is dated no later
than that day, and all meeting-day data are forbidden. The tracked source
registry includes explicit source metadata, disabled-source reasons, license
controls, and a 102-file legacy crosswalk; legacy and `synthetic_text` values
are mapping-only. Raw request bytes, aligned 12/12/2 vintage batches, hashes,
exclusions, source evidence, coverage, and both ledger manifests are
independently replay-validated.

The generation stage emits three section families with exact-deletion and
tokenizer-length-matched neutral interventions. The downstream wrapper derives
every row seed from only `replicate_seed` and `sample_id`, performs exact
chat-template context preflight, rejects prompt-token drift and length-limited
completions, and binds outputs to sealed model, tokenizer, source, ledger, and
snapshot specifications. `run/generate_loo_end_to_end.sh all` is the primary
background entrypoint; it performs no training or model merge and does not
require `FRED_API_KEY`. The P2.3 checklist remains open until both population
generation release manifests and the enclosing workflow release manifest have
completed successfully.

The checklist remains open until the missing masking prompts and fixed split
manifest are restored or reconstructed, the validated actual-Minutes
candidate is frozen in a release manifest, a new run is completed, and its
model/data revisions and generation/intervention manifest hashes are pinned
in the Chapter 2 release manifest. The historical aggregate workbook is
retained only for descriptive recovery.

#### P2.4 Decision evaluation

- [ ] Define separate pre-meeting, same-meeting, and ex-post information sets.
- [ ] Report accuracy, balanced accuracy, macro-F1, confusion matrices, class support, invalid outputs, and paired confidence intervals.
- [ ] Compare identical meetings against majority, lag-1, multinomial-logit, text, and market-implied baselines.
- [ ] Treat the small Cut class and format failures as first-class failure modes.
- [ ] Add a chronological out-of-sample forecasting experiment if forecasting remains a contribution.

#### P2.5 Econometric validation

- [ ] Specify sentiment construction, futures contract, sample period, controls, standard errors, bootstrap unit, and confidence intervals.
- [ ] Publish full regression tables and robustness specifications.
- [ ] Prevent generated text for one meeting from using future-vintage indicators.
- [ ] Keep the section explicitly exploratory unless conventional inferential evidence is obtained.

**P2 acceptance:** all comparisons use frozen row IDs and artifacts, all stochastic results report uncertainty, and every claim states its information set and evidence boundary.

### P3 — Manuscript, release, and long-term stewardship

- [ ] Keep `docs/report` outside the active thesis workflow unless the author explicitly reactivates it; maintain `docs/Chapter2` as the sole manuscript source.
- [ ] Add the full thesis root, bibliography, and a documented LaTeX build.
- [ ] Generate chapter tables and figures from frozen result files.
- [ ] Resolve remaining count, comparator, reward, QLoRA, learning-rate, and epoch inconsistencies.
- [ ] Complete a technical and language-editing pass.
- [ ] Remove residual TODOs and duplicated exposition.
- [ ] Add LICENSE, third-party attribution, `CITATION.cff`, SECURITY, CONTRIBUTING, and maintainer ownership.
- [ ] Replace inherited package metadata with project-specific metadata.
- [ ] Document GPU memory, runtime, API request count/cost, recovery, and storage needs.
- [ ] Create a DOI-bearing dissertation release with permissible code, manifests, data instructions, tables, figures, and a reproducibility report.

**Acceptance:** the thesis builds without missing files, undefined references, duplicate labels, or LaTeX errors, and a release command regenerates all chapter outputs from frozen artifacts.

---

## 11. Operational Workflow After P0

The commands below describe the intended interface. Commands marked “safe now” are read-only or lightweight. Expensive stages must not be treated as canonical until P0 is complete.

### 11.1 Environment target

```bash
conda create -n fomc_trainer python=3.10
conda activate fomc_trainer
pip install -r requirements.txt
pip install -e .
pip install flash-attn==2.5.6 --no-build-isolation
```

Replace this free-form installation with the locked environment required by P1.1.

### 11.2 Safe inspection

```bash
python -m jobs.main.run_pipeline --help
python -m jobs.main.run_pipeline train analysis_sft --dry-run
python -m process_fomc_report.build_qa_master --help
python -m process_fomc_report.generate_prompt_and_response.run_generate_prompt_pipeline --help
pytest tests --ignore=tests/slow
```

The test command is a target baseline, not currently a guaranteed green run in every local interpreter.

### 11.3 Data stages

For debugging, prefer foreground module calls so failures are visible:

```bash
python -m process_fomc_report.generate_prompt_and_response.run_generate_prompt_pipeline \
  --config configs/main/prompt_pipeline.yaml \
  --scope after_2009 \
  --profile compat \
  --stage <stage>
```

Operational wrappers are:

```bash
./run/generate_input/chk1.sh
./run/generate_input/chk2.sh
./run/generate_input/chk3.sh
./run/generate_input/chk4.sh
```

`chk1` and `chk3` return after spawning background jobs; inspect the timestamped log and PID under `logs/generate_input/`. The current `chk4` cannot complete because its configured decision sources are absent.

Do not present `jobs.main.run_pipeline build-datasets`, `audit`, or evaluation commands as canonical until their `/processed/main/` defaults are reconciled.

### 11.4 Training and merge

After preflight and data validation:

```bash
python -m jobs.main.run_pipeline train <stage> --dry-run
python -m jobs.main.run_pipeline train <stage>
python -m jobs.main.run_pipeline merge <stage>
```

Training does not imply merge. Before each stage:

1. verify the dataset and split hashes;
2. verify the declared parent merged model;
3. assign a unique run ID and output directory;
4. save a sanitized resolved configuration;
5. record GPU, package, teacher/judge, and model revisions; and
6. confirm that the test set will not be used for checkpoint selection.

### 11.5 Evaluation and chapter generation

Evaluation must consume a release manifest, not ad-hoc paths. The paired LOO
evaluator now implements the row-alignment, audit, and clustered-summary
parts of this interface; other evaluation branches and release-manifest
enforcement remain open. A complete canonical command should:

1. generate or load frozen per-row predictions;
2. validate exact row-ID alignment;
3. compute metrics and uncertainty;
4. write per-row and aggregate outputs;
5. create tables and figures; and
6. emit a machine-readable mapping from chapter labels to artifacts.

---

## 12. Change-Control Checklist

### Before a change

- [ ] State whether the change affects scientific meaning, only implementation, or only prose.
- [ ] Record the current Git commit and working-tree state.
- [ ] Identify the experiment family, split ID, response format, and affected descendants.
- [ ] Run secret, config, path, schema, split, and parent-artifact preflight.
- [ ] Preserve any thesis-referenced artifact before modifying cleanup or output code.

### During a change

- [ ] Never edit raw data or archive material in place.
- [ ] Never reuse an output directory for a different config or model.
- [ ] Keep sample IDs stable unless the data definition changes.
- [ ] Treat prompt/template changes as versioned interface changes.
- [ ] Make randomness explicit and record all seeds.
- [ ] Keep inspections and dry runs read-only.
- [ ] Update tests with the behavior change.

### After a change

- [ ] Run offline tests, lint, strict YAML validation, and split/leakage checks.
- [ ] Compare row counts and hashes with the expected migration plan.
- [ ] Verify that runtime metadata contains no secrets.
- [ ] Verify parent-child checkpoint lineage.
- [ ] Retain failure and exclusion manifests.
- [ ] Update README, PLANS, metadata, config documentation, and manuscript claims together.
- [ ] Record the decision and close or update the relevant entry in Section 9.

---

## 13. Run and Artifact Naming

Use explicit, immutable run IDs, for example:

```text
20260727_llama_analysis_sft_chronological-v1_seed42_2e5fc6d
```

Recommended identity fields:

```text
<date>_<model-family>_<stage>_<split-id>_seed<seed>_<git-sha>
```

Each run directory should contain:

- `experiment_manifest.json`;
- sanitized `resolved_runtime_config.json`;
- source/config/template hashes;
- dataset and split-manifest hashes;
- environment and hardware snapshot;
- checkpoints or adapter references;
- training/evaluation logs;
- per-row predictions or generation references;
- aggregate metrics with uncertainty;
- failure/exclusion summaries; and
- artifact checksums.

Never use unexplained `_0`, `_1`, or manually renamed “final” directories for dissertation evidence.

---

## 14. Definition of Done for Chapter 2

Chapter 2 is reproducible and submission-ready only when all of the following are true:

- **Security:** no credential is present in the worktree, history, logs, or artifacts.
- **Identity:** every checkpoint and external model has an exact ID, revision, parent, and role.
- **Data:** the sample-flow ledger reconciles every unit and exclusion.
- **Splits:** meeting membership is frozen, leakage-audited, and described accurately.
- **Environment:** a clean locked environment passes the offline test suite.
- **Pipeline:** a fresh restore can rebuild derived data and reproduce run manifests.
- **Evaluation:** all compared systems use common row IDs and report failures and uncertainty.
- **Evidence:** every table and figure is generated from a frozen result artifact.
- **Claims:** forecasting, inference, grounding, sufficiency, and significance language matches the implemented evidence.
- **Document:** one canonical manuscript builds cleanly and contains no unresolved numerical or methodological contradiction.
- **Release:** code, permissible artifacts, provenance, access instructions, and checksums are archived durably.
