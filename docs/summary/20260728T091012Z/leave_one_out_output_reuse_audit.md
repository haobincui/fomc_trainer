# Leave-One-Out Output Reuse Audit and Implementation Decision

## Purpose

This audit determines which historical leave-one-out (LOO) artifacts can be
reused after redefining the primary estimand as

\[
\Delta_i =
\cos(o_{\mathrm{full}}, y)
-
\cos(o_{\mathrm{masked},i}, y),
\]

where \(y\) is fixed to the same target text for the full and masked outputs.
With cosine distance \(d=1-\cos\), the identical estimand is
\(\Delta_i=d_{\mathrm{masked},i}-d_{\mathrm{full}}\).

The repository, Git history, active `output/` tree, archived code, historical
workbooks, manuscript tables, and the sibling
`../synthetic_text/synthetic_text` repository were inspected on 2026-07-28.
The sibling repository was treated as read-only.

## Relation to the earlier review documents

This audit supersedes the LOO implementation descriptions in
`chapter2_phd_referee_report.md` and
`chapter2_revision_recommendations_zh.md` where they conflict. Those documents
correctly identified the old absolute-distance and unmatched-test problems,
but they were written before the surviving workbooks and the revised
canonical evaluator were inspected in detail. In particular:

- the unmatched Welch procedure belongs to `shapley_diff.xlsx`;
- `shapley_result_with_diff.xlsx` retains historical paired-difference
  statistics, although those statistics still cannot be re-clustered by
  meeting;
- the new canonical path is `jobs.eval.eval_leave_one_out`, while
  `jobs.eval.eval_mask` is legacy-only.

## Methodological basis and claim boundary

The design is a removal-based explanation: it measures the change in a fixed
utility after one input block is removed. This framing follows the taxonomy
in [Covert, Lundberg, and Lee (2021)](https://www.jmlr.org/papers/v22/20-1316.html).
Subtracting the removed-input score from the full-input score is also
analogous to the comprehensiveness contrast in
[DeYoung et al. (2020)](https://aclanthology.org/2020.acl-main.408/).
Full Shapley attribution instead averages marginal contributions over
coalitions, as formalized for model explanation by
[Lundberg and Lee (2017)](https://proceedings.neurips.cc/paper_files/paper/2017/hash/8a20a8621978632d76c43dfd28b67767-Abstract.html).

Accordingly, the reported \(\Delta_i\) is a local, target-relative,
leave-one-indicator-out sensitivity under the stated deletion intervention.
It is not a Shapley value, causal effect, proof of factual grounding, or proof
that indicator \(i\) is necessary or sufficient. Redundant indicators may
make a single deletion look unimportant, and interactions can make the
effect depend on which other indicators remain.

## Reuse classification

| Artifact | Surviving content | Decision |
|---|---|---|
| `archive/code/reformat_shapley_result/shapley_result_with_diff.xlsx` | 1,726 aggregate indicator-by-section rows; 78 rows cover the three principal sections. It retains indicator-specific `mean_base`, `mean_indicator`, paired-difference SD/SE, test statistics, and intervals. | Reuse the point estimate `mean_indicator - mean_base` as a legacy descriptive \(\Delta\). Retain old inferential columns for provenance only; do not publish them as current inference because meeting IDs and seed metadata are unavailable. |
| `archive/code/reformat_shapley_result/shapley_filter.xlsx` | 75 aggregate cells for 25 indicators and three sections, using the unmasked generated output as target. | Reuse only as a historical description of internal output sensitivity. Because the target is the full output, its distance equals the LOO delta algebraically. Meeting-clustered inference cannot be recovered. |
| `archive/code/reformat_shapley_result/shapley_by_raw_filter.xlsx` | 81 aggregate cells, including the `None` baseline; values are `1-cos(masked, actual)`. | Reuse only as an audit of absolute masked-output distance. Do not use its global `None` mean to construct inference: indicator-specific sample sizes differ. |
| `archive/code/reformat_shapley_result/shapley_diff.xlsx` | 78 aggregate percentage differences with unmatched Welch tests. | Retire. It does not implement paired LOO inference. |
| Historical generated `*.tex` tables and `archive/data/output/chapter2/shapley_*` summaries | Presentation-only derivatives of the workbooks. | Regenerate or retain as provenance only. |
| `output/evaluation/main/` | Only a decision-baseline smoke artifact. | No LOO output is reusable from the active evaluation tree. |
| `../synthetic_text/synthetic_text/output/archive/masked/ft_20250330/` | 2,660 row-level full/masked pairs for seven indicators, stored in 133 workbooks. Corresponding internal cosine scores also survive. | Reuse only as a separately labelled exploratory pilot or for provenance checks. It is not the row-level source of the 25/26-indicator Chapter 2 workbooks and must not be merged with them. |
| `../synthetic_text/synthetic_text/output/synthetic_text_reason/synthetic_text_20250520_reason.jsonl` | 3,481 generated reasoning rows carrying 696 distinct meeting-section target texts. | Reuse to validate the actual-Minutes target text and source family. Do not use it unchanged as the canonical reference manifest because it omits one of 697 source records and contains one duplicated replicate identifier. |

The recovered external-alignment point estimates are now materialized as:

- `docs/Chapter2/Chapter2Results/loo_external_legacy_descriptive.csv`
- `docs/Chapter2/Chapter2Results/loo_external_legacy_descriptive.tex`
- `docs/Chapter2/Chapter2Results/loo_internal_legacy_descriptive.csv`
- `docs/Chapter2/Chapter2Results/loo_internal_legacy_descriptive.tex`

The CSV files preserve the old unclustered fields with an explicit
`inference_status` warning. The TeX tables report signed raw deltas without
significance stars.

They are regenerated with:

```bash
python -m jobs.eval.recover_legacy_loo \
  --input-xlsx archive/code/reformat_shapley_result/shapley_result_with_diff.xlsx \
  --output-csv docs/Chapter2/Chapter2Results/loo_external_legacy_descriptive.csv \
  --output-tex docs/Chapter2/Chapter2Results/loo_external_legacy_descriptive.tex \
  --internal-input-xlsx archive/code/reformat_shapley_result/shapley_filter.xlsx \
  --internal-output-csv docs/Chapter2/Chapter2Results/loo_internal_legacy_descriptive.csv \
  --internal-output-tex docs/Chapter2/Chapter2Results/loo_internal_legacy_descriptive.tex
```

## Sibling `synthetic_text` audit

### Row-level seven-indicator pilot

The sibling repository contains a complete file inventory for the
`ft_20250330` pilot:

| Property | Observed value |
|---|---:|
| Indicators | 7 |
| Indicator names | `fed_rate`, `gdp`, `inflation_rate`, `sp500`, `unemployment_rate`, `us_t10`, `yield_curve` |
| Draw-batch files per indicator | 19 |
| Rows per indicator-batch file | 20 |
| Full/masked indicator pairs | 2,660 |
| Full generation units before indicator expansion | 380 |
| Unique source indices covered | 256 of 486 |
| Meetings covered | 114 of 123 |
| Section types covered | 5 |
| Date range | 2009-01-28 to 2024-05-01 |

The file
`output/archive/cosine/mask/ft_20250330/tagged_cos_ft_20250330.xlsx`
has SHA-256
`002472ae93530b9ba1d35ea70e75ef9fc8a1c24a00027b40fbb73a6ad2558ec8`.
Its stored `cos` field is
\(\operatorname{Cos}(o_{\mathrm{full}},o_{\mathrm{masked}})\), not either
component of the external actual-Minutes delta. Its complement can therefore
be recovered as an internal self-distance only. Across the seven historical
tags, mean self-distances range from 0.1204 to 0.1364.

The individual workbooks preserve the draw-batch number, which is required
to distinguish repeated source indices. A stable recovery key is
`meeting_date::section_name::qa_index::legacy_batch`. The batch number is a
random draw occurrence, not a seeded generation replicate.

The pilot is not the source of the Chapter 2 aggregate workbooks:

- the pilot has seven indicators, whereas the manuscript workbooks have 25
  or 26 indicators;
- the pilot covers five older section types and has no `Participants' Views`
  section, whereas the manuscript tables report three different principal
  sections;
- the pilot supplies only 380 rows per indicator before sectioning, while a
  manuscript indicator-section cell contains 897--1,337 pairs;
- the pilot covers 256 meeting-section rows from 114 meetings, whereas the
  recovered target source contains 697 rows from 128 meetings; and
- several old indicator labels have no unique mapping to the later 26-item
  taxonomy.

The generation scripts record stochastic decoding with `do_sample=True`,
`temperature=0.6`, and `top_p=0.9`, but do not set or retain a seed. The 380
rows are repeated random samples rather than a balanced replicate panel:
source rows occur between one and five times. The referenced checkpoint
`./models/llama3_qlora_20250330_pt` no longer survives at that path, and the
`./models/llama3-8b` checkpoint used by the historical cosine script is also
missing. The sampling universe is also referenced by an SFT data
configuration, so training-set overlap cannot be ruled out without a
reconstructed split and lineage audit.

Most importantly, the historical masking operation changed the prompt
template as well as the indicator value. None of the 2,660 masked prompts can
be obtained from its full prompt by one exact contiguous block deletion;
2,571 masked prompts are not even shorter than their paired full prompts.
Across the full 486-row QA universe, none of the 3,402
row-by-seven-indicator prompt pairs preserves the full template exactly. For
`Staff-Economic-Outlook` and `Committee-Policy-Action`, all seven indicator
labels produce the same masked prompt within a sample, so those contrasts do
not remove different indicators at all. The Economic Situation prompts have
five variants and the Financial Situation prompts have four; indicators that
are irrelevant to a template collapse to shared null-mask prompts.
Consequently, the old contrast confounds indicator removal with prompt
rewriting and cannot satisfy the canonical intervention manifest.

All 486 cleaned metadata texts in
`output/archive/qa_json/input_qa_tagged.json` can nevertheless be matched
exactly and uniquely to the dated section files under
`data/processed/sections/`. The historical generation script used the
index-aligned `output` field in `data/training/input_qa.json` as its actual
target. The training and tagged files have identical instructions at all 486
indices but identical output text at only 82 indices, so the tagged file must
not be substituted as the scoring target. Of the 380 targets copied into the
full generation workbooks, 334 exactly match the training JSON and 46 hit
Excel's 32,767-character cell limit; every truncated value is an unambiguous
prefix of its complete training-JSON target. Full responses are non-empty in
all 380 draw occurrences. Five of the 2,660 masked responses also hit the
Excel limit and cannot be parsed completely; they must be excluded rather
than silently scored:

- `fed_rate`, batch 6, source index 128;
- `gdp`, batch 6, source index 21;
- `gdp`, batch 11, source index 5;
- `unemployment_rate`, batch 7, source index 81; and
- `us_t10`, batch 6, source index 303.

Meeting identity and actual target text are therefore recoverable for a
separately labelled pilot reanalysis. Such a reanalysis may be useful as a
robustness appendix, but it must be called a **legacy prompt-perturbation
sensitivity diagnostic**, report the old intervention, unbalanced sampling,
unseeded decoding, truncation exclusions, and possible training overlap, and
must not supply the Chapter 2 primary table or formal significance claims.

The earlier `ft_20250313` run is less usable: only 116 masked workbooks
survive and GDP has one batch while other indicators have approximately 19.
It is classified as incomplete and retired.

### Actual-Minutes target provenance

The strongest reusable finding is the recovery of the target text family:

- `synthetic_text_20250520_reason.jsonl` has 3,481 rows, 128 meetings, and
  696 distinct meeting-section keys. Its SHA-256 is
  `5d4bcb4b0724b122f291a7470c76d6b10aae067e18bf47f700a751c6f6f31164`.
- Its 67 section names exactly equal the section-name universe in
  `shapley_result_with_diff.xlsx`.
- Every one of its 3,481 rows has an exact match on meeting, section, and
  actual-Minutes text in
  `archive/data/dataset_20260421/raw_data/archive/synthetic_text_20250518.jsonl`.
- The latter file contains 697 unique source rows from 128 meetings and has
  SHA-256
  `0edc4c44ce13870933a461d3507cc24c40e792667e54bda97a0150b1ebc3ac06`.
  The 2025-05-20 file covers 696 of them: source index `153` is absent and
  replicate identifier `0-1` occurs twice.

This establishes content equivalence for the observed target rows and
recovers the actual-Minutes source family with high confidence. It does not
prove that the 2025-05-18 file is byte-for-byte identical to the missing
historical LOO input, nor does it reconstruct the historical LOO row
manifest. The 697-row file is suitable as a reference for a newly versioned
run after its path, hash, schema, and row universe are frozen in the Chapter
2 release manifest.

The machine-readable inventory accompanying this audit is
`synthetic_text_loo_reuse_inventory.json`.

## Remaining missing artifacts

No row-level file for the historical 25/26-indicator Chapter 2 experiment
containing the exact `full_output`, `masked_output`, meeting identifier,
section identifier, and generation replicate survives in the current
repository, its Git history, or the sibling pilot. The historical generated
folders under `output/validation/...` are absent.

The exact frozen inputs required for a bit-for-bit reproduction are also
missing:

- masking prompts under
  `dataset/processed/main/evaluation_inputs/source_prompts/mask_indicator/`;
- the fixed split manifest
  `dataset/processed/main/manifests/analysis_minutes_split.json`, which is
  required by the prompt-filtering command;
- the historical model/checkpoint referenced by the old run scripts;
- the original 25/26-indicator unmasked and masked generations; and
- the exact row manifest that linked those generations to the historical
  target file.

The target text and source family are no longer classified as wholly missing:
the 697-row 2025-05-18 archive is content-equivalent on all 3,481 observed
2025-05-20 rows. What remains missing is proof of byte identity and exact
row-level linkage for the historical 25/26-indicator run.

## Adopted canonical implementation

New runs use `jobs.eval.eval_leave_one_out` rather than the historical
`jobs.eval.eval_mask` Shapley path.

The evaluator:

1. keeps every replicate instead of overwriting files in an indicator map;
2. requires a generation manifest and hash-validates every declared artifact
   against it; the manifest itself still requires an external hash anchor in
   the Chapter 2 release manifest;
3. verifies a frozen roster and complete indicator/context/replicate coverage
   and fails on any empty, invalid, extra, missing, or modified generation
   artifact;
4. verifies that the deleted block has complete line/block boundaries,
   begins with a roster-declared marker for the named indicator, has no
   overlap with another indicator's deletion span, and produces a distinct
   masked prompt for every indicator within a sample;
5. binds every generated row's key, prompt hash, meeting, and section back to
   the intervention manifest;
6. fingerprints the complete local generation-checkpoint and embedding-model
   file trees rather than recording mutable paths alone;
7. pairs masked and full artifacts with the same context and replicate;
8. joins rows by `sample_id`, or by a validated legacy
   `(meeting_date, section_name, source_index/index)` key;
9. rejects full/masked pairs whose meeting, section, generation position,
   seed, model, batch size, decoding parameters, or masking strategy conflict;
10. requires an explicit reference file for actual-Minutes evaluation;
11. extracts the answer from legacy XML and Gemma thought-channel responses;
12. writes `similarity_full`, `similarity_masked`, signed `delta`,
   `distance_full`, `distance_masked`, and internal `self_distance`;
13. emits row-level, paired-row exclusion-ledger, summary, and audit files;
14. averages rows within `meeting × replicate`, then gives each replicate
    equal weight in the meeting mean;
15. keeps evaluation contexts separate, obtains confidence intervals by
    resampling meetings, and applies Holm correction across all
    indicator–section–context cells in one run.

The primary output schema includes:

```text
schema_version
metric
target_mode
indicator
context
meeting_date
section_name
sample_id
source_index
row_key
replicate_id
generation_seed
full_generation_seed
masked_generation_seed
pair_id
full_output
masked_output
target_text
similarity_full
similarity_masked
delta
distance_full
distance_masked
self_similarity
self_distance
embedding_model_path
full_artifact
masked_artifact
full_artifact_sha256
masked_artifact_sha256
full_source_prompt_sha256
masked_source_prompt_sha256
generation_model_sha256
embedding_model_sha256
pairing_quality
status
```

New masking generation defaults to deterministic decoding. Before model
inference, it checks a frozen roster, identical sample-key coverage, non-empty
meeting/section identities, and an exact single contiguous deletion from each
full prompt to its indicator-masked counterpart. It additionally checks
roster-declared indicator markers, complete line/block boundaries, and
pairwise non-overlapping deletion spans as well as cross-indicator prompt
uniqueness. It writes a row-level
`intervention_manifest.json` containing prompt and removed-block hashes. The
generation outputs record the replicate ID, actual batch seed, model path,
model-tree fingerprint, batch size, decoding parameters, masking strategy,
original source index, generation position, source-prompt SHA-256, input
SHA-256, expected row count, and a versioned
`generation_manifest.json` that hashes the intervention manifest. Existing
output files are reused only when every expected row is present and their
input identity, prompt binding, and complete run metadata match. Evaluation
repeats the prompt/intervention binding checks and fingerprints the local
embedding-model tree. Partial generations fail fast and remain available for
diagnosis rather than being silently accepted.

The `.exclusions.jsonl` file covers row-level pairing and reference failures.
Artifact-level corruption or manifest incompleteness is intentionally a hard
error surfaced by the failed command rather than being converted into a
publishable partial result.

The planned percentile bootstrap treats meetings as the independent,
exchangeable clustering unit. It does not model serial dependence across
adjacent FOMC meetings; a chronological block-bootstrap or HAC-style
robustness analysis remains advisable if serial dependence is substantively
important.

## Required rerun boundary

The legacy workbook is sufficient to repair the definition and recover
descriptive aggregate point estimates. It is not sufficient for:

- per-meeting or per-sample deltas;
- meeting-clustered standard errors or bootstrap intervals;
- a repeated-seed robustness analysis;
- alternative embedding or factuality metrics;
- a complete sample-flow ledger linking every exclusion to a source row.

Those results require restoration or reconstruction of the masking prompts
and fixed split manifest, freezing the validated 697-row actual-Minutes
reference as a newly versioned artifact, and a new run under the paired
evaluator. The resulting experiment is a new reproducible specification, not
a bit-for-bit recreation of the historical experiment. The seven-indicator
pilot cannot fill this gap.
