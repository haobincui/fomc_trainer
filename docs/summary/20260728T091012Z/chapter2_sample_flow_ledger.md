# Chapter 2 Sample-Flow Ledger

## 1. Purpose and status

This document defines the authoritative sample-flow accounting for Chapter 2 and specifies how the paper's data-flow table should be generated from machine-readable manifests.

The ledger has four purposes:

1. distinguish meetings, meeting-sections, QA rows, traces, prompts, and rewrite rows;
2. reconcile candidate, retained, excluded, allocated, and quality-flagged observations;
3. ensure every downstream task inherits a named meeting split rather than creating an independent split;
4. eliminate manually typed sample counts from the thesis.

### Current release status

> **Status: INCOMPLETE — suitable for data auditing, but not yet suitable as the final paper table.**

The post-2009 generation branch can be audited from existing artifacts. The active Decision inputs and outputs are missing, and its archived split is incompatible with the current chronological QA split. The final LaTeX table must therefore not be released until the Decision branch has been rebuilt against the canonical split.

---

## 2. Authoritative scope and precedence

### 2.1 Main population

The primary Chapter 2 population is the post-2009 corpus:

- raw meeting files: January 2009–January 2025;
- 129 raw meeting documents;
- 128 eligible meetings after parsing and labeling;
- one chronological meeting split shared by the analysis, Minutes, and main Decision tasks.

The canonical retained-meeting split is:

| Split | Meetings | Date range |
|---|---:|---|
| Train | 102 | 2009-01-28 to 2021-11-03 |
| Eval | 13 | 2021-12-15 to 2023-06-14 |
| Test | 13 | 2023-07-26 to 2025-01-29 |
| **Total** | **128** | 2009-01-28 to 2025-01-29 |

The current source of this assignment is:

```text
dataset/processed/manifests/qa_meeting_split_manifest.json
```

Its implementation is the chronological split in:

```text
src/process_fomc_report/build_qa_master.py
```

### 2.2 Source precedence

Until the new manifests are implemented, evidence should be read in this order:

1. row manifests under `dataset/processed/manifests/` and `dataset/processed/train/`;
2. audit files under `dataset/processed/pipeline/audit/`;
3. canonical master data under `dataset/processed/pipeline/`;
4. archived artifacts, used only for provenance or diagnosis;
5. manuscript counts, which are claims to be checked rather than sources of truth.

The following competing path is not authoritative for the present ledger:

```text
dataset/processed/main/
```

`jobs/main/build_datasets.py` defines an independent seed-42 random split under that path, but the required inputs and manifests do not currently exist. It must not be combined with the existing chronological artifacts under `dataset/processed/`.

---

## 3. Counting vocabulary

Every generated table and manifest must use the following terms consistently.

| Term | Definition |
|---|---|
| Candidate | An upstream unit presented to a stage before that stage's quality filters and allocation rules. |
| Eligible | A candidate that passes quality filters, whether or not it is selected for this particular training stage. |
| Retained | A unit physically included in the stage's final dataset. |
| Excluded | A candidate removed for a documented data-quality or scope reason. |
| Allocated elsewhere | An eligible unit intentionally reserved for another training stage; this is not a quality exclusion. |
| Replay | An eligible unit deliberately reused in another stage; replay creates physical rows but not a new underlying QA unit. |
| Physical row | One serialized record in a dataset file. |
| Analysis unit | The unit on which a reported count is based, such as meeting, meeting-section, or QA row. |
| Quality flag | A retained unit with a known issue. A flag is not an exclusion unless an explicit filter removes it. |

Counts may only be added or subtracted when they refer to the same unit. For example, four duplicate source rows and 109 excluded meetings cannot be reported as 113 excluded meetings.

---

## 4. Verified current sample-flow table

The following table reports the current auditable state. Parentheses in the Minutes row report unique meeting-sections nested within the physical rewrite rows.

| Stage | Analysis unit | Candidate total | Train | Eval | Test | Retained total | Exclusion/allocation summary | Current manifest or evidence | Status |
|---|---|---:|---:|---:|---:|---:|---|---|---|
| Raw meetings ingested | meeting | 129 | 103 | 13 | 13 | 129 | One Train-period meeting is later excluded | Raw files; no complete raw-meeting manifest yet | Partial |
| Eligible meetings | meeting | 129 | 102 | 13 | 13 | 128 | 1 meeting excluded: 2009-11-04 produced no usable core section | `qa_meeting_split_manifest.json` plus raw-file reconciliation | Verified |
| Labeled section candidates | meeting-section | 726 | 584 | 72 | 70 | 726 | Derived after paragraph-label filtering; before analysis-topic filtering | Labeled files and labeling rules; no row manifest yet | Reconstructed |
| Analysis-ready meeting sections | meeting-section | 726 | 577 | 72 | 70 | 719 | 7 Train section groups had only `Other` labels; the excluded raw meeting contributes no usable section | QA manifests grouped by meeting and normalized section | Verified |
| QA construction | QA row | 4,888 | 3,890 | 535 | 462 | 4,887 | 1 Train row excluded for `empty_response_without_backfill`; 2 other empty responses were backfilled but remain an integrity warning | `qa_{split}_manifest.jsonl` and source-reconciliation audit | Verified with warning |
| Analysis SFT | analysis trace | 4,887 | 3,112 | 533 | 461 | 4,106 | 7/2/1 teacher responses missing; 771 eligible Train rows allocated outside SFT | `train/analysis_sft/*_manifest.jsonl` and dataset-profile audit | Verified |
| Analysis GRPO | GRPO prompt | 4,887 | 1,082 | 533 | 461 | 2,076 | 7/2/1 teacher responses missing; Train consists of 771 core + 311 intended replay rows; 2,801 eligible Train rows not selected | `train/analysis_grpo/*_manifest.jsonl` and dataset-profile audit | Verified with allocation conflict |
| Minutes SFT | QA-derived rewrite row | 4,887 | 3,883 **(577 sections)** | 533 **(72 sections)** | 461 **(70 sections)** | 4,877 rows **(719 sections)** | 7/2/1 rows excluded for missing `raw_analysis`; current code does not aggregate to one row per section | `train/minutes_alignment/*_manifest.jsonl` and Minutes audit | Verified; unit differs from manuscript |
| Decision task, active | meeting | Unknown | — | — | — | — | Active inputs, prompts, datasets, and audit are missing | Expected under `processed/input_sources/decision_*`; not present | **INCOMPLETE** |
| Decision task, archived diagnosis | meeting | 237 | 102* | 13* | 13* | 128 post-2009 | 241 source rows deduplicate to 237 meetings; 109 pre-2009 meetings removed. Asterisk: archived post-2009 split is independent random assignment, not the canonical chronological split | Archived Decision artifacts | Diagnostic only |

### Main-table reading rule

For the paper:

- use the unparenthesized Minutes counts only when the unit is named `QA-derived rewrite row`;
- use the parenthesized Minutes counts only when the unit is named `unique meeting-section`;
- do not label 3,883/533/461 as meeting-section counts;
- do not publish the archived Decision 102/13/13 as if it inherited the QA split.

---

## 5. Target paper table after canonical rebuild

The final generated LaTeX table should have the following structure. Values must be read from manifests at generation time rather than hardcoded in the renderer.

| Stage | Analysis unit | Train | Eval | Test | Exclusion/allocation summary | Manifest |
|---|---|---:|---:|---:|---|---|
| Raw meetings | meeting | 103 | 13 | 13 | 1/129 meeting excluded after parsing/labeling | `raw_meetings_manifest.jsonl` |
| Eligible meetings | meeting | 102 | 13 | 13 | 2009-11-04 excluded: no usable core section | `canonical_meeting_split.json` |
| Meeting sections | meeting-section | 577 | 72 | 70 | 726 labeled candidates → 719 analysis-ready sections; 7 `Other`-only groups excluded | `meeting_sections_manifest.jsonl` |
| QA construction | QA row | 3,890 | 535 | 462 | 4,888 candidates → 4,887 retained; 1 empty response without backfill | `qa_construction_manifest.jsonl` |
| Analysis SFT | analysis trace | 3,112 | 533 | 461 | 10 missing teacher responses; 771 eligible Train rows allocated to GRPO core | `analysis_sft_manifest.jsonl` |
| Analysis GRPO | GRPO prompt | 1,082 | 533 | 461 | Train = 771 core + 311 replay; 10 missing teacher responses | `analysis_grpo_manifest.jsonl` |
| Minutes SFT | rewrite row | 3,883 | 533 | 461 | 10 missing `raw_analysis`; rows nest within 577/72/70 unique meeting-sections | `minutes_sft_manifest.jsonl` |
| Decision task | meeting | 102 | 13 | 13 | Rebuilt from deduplicated post-2009 meetings and inherited from the canonical split | `decision_task_manifest.jsonl` |

This target table is not permission to type the Decision values manually. The last row becomes valid only after all 128 meeting IDs in the rebuilt Decision manifest match the canonical split manifest.

---

## 6. Stage-by-stage reconciliation

### 6.1 Raw meetings

#### Candidate population

- 129 post-2009 meeting files;
- date range: 2009-01-28 to 2025-01-29;
- the `.xlsx` and `.csv` files under `dataset/raw_data/labeled_text/after_2009/` are two representations of the same 129 meetings and must not be counted twice.

#### Excluded meeting

The only raw meeting absent from the 128-meeting QA population is:

```text
2009-11-04
```

Its labeled file has 74 paragraph rows, all marked `Pre-Non-Core`. Most text was incorrectly associated with the date-like section name `November 3-4, 2009`, indicating a section-parsing failure rather than a genuine absence of analytical content.

The new raw manifest must retain this meeting with:

```json
{
  "meeting_id": "fomc-2009-11-04",
  "meeting_date": "2009-11-04",
  "split": "train",
  "included": false,
  "exclusion_reason_code": "no_usable_core_section",
  "parser_status": "failed_section_segmentation",
  "paragraph_count": 74
}
```

An excluded meeting must remain visible in the ledger; it must not disappear because it was removed before split-manifest creation.

### 6.2 Meeting sections

There are three different section counts:

| Section representation | Count | Suitable for paper? |
|---|---:|---|
| Raw parser-assigned `(meeting, section name)` candidates | approximately 5,562 | No; contains dates, names, markup, and other noise |
| Labeled candidate meeting-sections after logical paragraph deduplication and label filtering | 726 | Useful as candidate population |
| Analysis-ready meeting-sections linked to at least one QA row | 719 | Yes; 577/72/70 |

The 726 labeled candidates are split 584/72/70. Seven Train-period meeting-sections are absent from the QA population because all retained labels were `Other`:

1. 2009-04-29 — `Staff Economic Outlook`
2. 2012-03-13 — `Monetary Policy Communications`
3. 2013-05-01 — `Review of Exit Strategy Principles`
4. 2013-09-18 — `Staff Economic Outlook`
5. 2013-10-30 — `Developments in Financial Markets and the Federal Reserve's Balance Sheet`
6. 2017-11-01 — `October 31-November 1, 2017`
7. 2018-01-31 — `Developments in Financial Markets and Open Market Operations`

The manuscript's value of 718 sections is not supported by the current artifacts.

#### Duplicate merged workbook warning

The current merged workbook contains 10,666 physical rows because the merger reads both 5,333 logical `.xlsx` rows and their 5,333 `.csv` duplicates. The ledger must never use 10,666 as the paragraph population.

The canonical rule should be:

1. use `.xlsx` as the authoritative labeled representation;
2. treat a matching `.csv` as a redundant export;
3. fail if duplicate logical keys disagree;
4. key logical paragraph rows by:

```text
(meeting_date, line_id, canonical_section_name, normalized_raw_text)
```

### 6.3 QA construction

#### Reconciliation

```text
4,888 prompt/response candidates
  - 1 empty response without a usable backfill
= 4,887 canonical QA rows
```

The retained split is:

```text
Train 3,890
Eval    535
Test    462
Total 4,887
```

The dropped row is:

```text
meeting: 2012-06-20
section: Participants' Views on Current Conditions and the Economic Outlook
topic: GDP Growth
reason: empty_response_without_backfill
```

Two other empty response rows were matched to a backfill source. Their normalized canonical responses are still reported as empty by the current audit, so the new ledger must flag them as `retained_with_empty_response` until they are corrected or explicitly excluded.

#### Unit hierarchy

The 4,887 QA rows are not 4,887 sections:

| Split | QA rows | Unique meeting-sections | Unique meeting-section-topic combinations |
|---|---:|---:|---:|
| Train | 3,890 | 577 | 3,016 |
| Eval | 535 | 72 | 359 |
| Test | 462 | 70 | 323 |
| **Total** | **4,887** | **719** | **3,698** |

### 6.4 Analysis SFT

#### Reconciliation

```text
Candidate prompts:       3,890 / 535 / 462
Teacher-response missing:    7 /   2 /   1
Eligible rows:           3,883 / 533 / 461
Train allocated to SFT:  3,112
Train allocated away:      771
Final SFT:               3,112 / 533 / 461
```

The 771 eligible Train rows not selected for SFT are an allocation decision, not a data-quality exclusion.

The analysis unit is:

> one QA-derived prompt–teacher-response trace.

It is not a meeting and not a meeting-section.

### 6.5 Analysis GRPO

#### Intended reconciliation

```text
Eligible Train rows: 3,883
  SFT-side slice:     3,112
  GRPO core:            771

GRPO final Train:
  GRPO core:            771
  intended replay:      311
  total:              1,082
```

Eval and Test retain 533 and 461 usable prompts.

#### Allocation conflict

The present Analysis SFT and Analysis GRPO builders independently order rows using hashes of different prompt templates. Therefore the recorded `311` replay count does not describe the true cross-dataset overlap.

Observed current overlap:

- Analysis SFT Train ∩ Analysis GRPO Train: 866 sample IDs;
- GRPO rows marked `grpo_core` that overlap actual SFT: 616;
- GRPO rows marked `sft_mix` that overlap actual SFT: 250.

The canonical fix is a shared QA-level allocation manifest created before either template-specific prompt is rendered. Both builders must read the same allocation role for each `sample_id`.

### 6.6 Minutes SFT

#### Actual unit

The builder creates one rewrite row for each source QA row. It does not group all topics or excerpts into one record per meeting-section.

The correct analysis unit is:

> QA-derived rewrite row identified by meeting, section, topic, and `sample_id`.

#### Reconciliation

```text
Candidate rewrite prompts: 3,890 / 535 / 462
Missing raw_analysis:          7 /   2 /   1
Final rewrite rows:         3,883 / 533 / 461
Unique meeting-sections:      577 /  72 /  70
```

Multiplicity within section:

| Split | Rewrite rows | Unique meeting-sections | Sections with multiple rows | Maximum rows in one section |
|---|---:|---:|---:|---:|
| Train | 3,883 | 577 | 480 | 18 |
| Eval | 533 | 72 | 65 | 17 |
| Test | 461 | 70 | 63 | 15 |

The paper must either:

1. report the actual rewrite-row counts and disclose nesting; or
2. implement a genuine aggregation stage and retrain the Minutes model.

It must not call 3,883/533/461 meeting-section samples.

#### Provenance warning

The final target is the official `reference_excerpt`, while `raw_analysis` is produced by an analysis teacher that is currently reference-conditioned on all splits. The ledger must record:

```text
reference_used_in_teacher_prompt
analysis_teacher_model
analysis_teacher_prompt_hash
analysis_teacher_response_hash
minutes_target_hash
```

This warning is separate from sample counting but necessary for interpreting the resulting dataset.

### 6.7 Decision task

#### Active state

The configured active inputs do not exist:

```text
dataset/processed/input_sources/decision_making/*.jsonl
dataset/processed/input_sources/decision_grpo/*.jsonl
```

Consequently, the active prompt, training, and audit directories are also absent.

#### Archived source diagnosis

The archived Decision source supports the following provenance calculation:

```text
241 physical source rows
  - 4 duplicate meeting rows
= 237 unique meetings
  - 109 pre-2009 meetings
= 128 post-2009 meetings
```

The archived 128-meeting GRPO split is random. Its membership relative to the canonical chronological split is:

| Archived Decision split | Canonical Train | Canonical Eval | Canonical Test |
|---|---:|---:|---:|
| Train | 83 | 11 | 8 |
| Eval | 10 | 1 | 2 |
| Test | 9 | 1 | 3 |

Identical marginal counts of 102/13/13 do not imply identical samples.

The archived Decision SFT data also contain meeting overlap:

| Check | Meetings |
|---|---:|
| Train–Eval overlap | 38 |
| Train–Test overlap | 38 |
| Eval–Test overlap | 9 |
| SFT Train overlap with GRPO Eval | 9 of 13 |
| SFT Train overlap with GRPO Test | 13 of 13 |

The Decision row can enter the final paper table only after it is rebuilt using the canonical chronological meeting manifest.

#### Full-history Decision population

The 237-meeting source population should be recorded as a separate provenance branch, not silently mixed into the 128-meeting generation flow:

```text
population_id: decision-full-history-v1
source period: 1993–2025
unique meetings: 237
role: supplemental Decision source/evaluation population
```

The main post-2009 Decision task should use:

```text
population_id: chapter2-post2009-v1
unique meetings: 128
split_id: post2009-chronological-v1
```

---

## 7. Manuscript-count corrections

The following values in the current manuscript must not be copied into the new ledger.

| Manuscript statement | Auditable value | Required correction |
|---|---|---|
| 123 Minutes | 129 raw files; 128 eligible meetings | Distinguish raw and retained meeting populations |
| 486 or 718 sections | 726 labeled candidates; 719 analysis-ready sections | Report the stage and unit explicitly |
| 4,889 QA pairs | 4,887 canonical QA rows | Replace and cite manifest |
| 3,911 / 489 / 489 QA split | 3,890 / 535 / 462 | Replace with chronological manifest counts |
| 3,129 or 3,128 Analysis SFT samples | 3,112 Train traces | Replace |
| 1,095 Analysis GRPO samples | 1,082 Train prompts | Replace |
| 1,113 Minutes training pairs | 3,883 Train rewrite rows in current data | Explain the actual row construction or rebuild |
| 578 / 72 / 72 Minutes sections | 577 / 72 / 70 unique meeting-sections | Replace; also disclose 3,883/533/461 physical rows |
| Decision 102 / 13 / 13 | Same marginal archived counts but incompatible membership | Rebuild and validate split hashes before reporting |

---

## 8. Canonical manifest design

### 8.1 Directory contract

The unique machine-readable source should be:

```text
dataset/processed/manifests/
├── canonical_meeting_split.json
├── analysis_allocation_manifest.jsonl
├── sample_flow/
│   ├── raw_meetings_manifest.jsonl
│   ├── meeting_sections_manifest.jsonl
│   ├── qa_construction_manifest.jsonl
│   ├── analysis_sft_manifest.jsonl
│   ├── analysis_grpo_manifest.jsonl
│   ├── minutes_sft_manifest.jsonl
│   ├── decision_task_manifest.jsonl
│   └── exclusions_manifest.jsonl
└── sample_flow.json
```

No renderer may obtain a count from a `.tex` file, README, hardcoded expected-count constant, or archived summary.

### 8.2 Stable identifiers

Use stable identifiers independent of row order:

```text
meeting_id = "fomc-" + meeting_date

meeting_section_key =
    meeting_id + "\0" + NFC(canonical_section_name)

meeting_section_id =
    "section-" + sha256(meeting_section_key)

qa_id =
    existing stable sample_id, preserved across all downstream stages
```

Section-name normalization must repair or reject encoding differences such as mojibake `â` versus Unicode en dash before generating the section ID.

### 8.3 Aggregate stage schema

Each stage in `sample_flow.json` should follow:

```json
{
  "schema_version": 1,
  "ledger_id": "chapter2-sample-flow-v1",
  "stage_id": "minutes_sft",
  "stage_label": "Minutes SFT",
  "population_id": "chapter2-post2009-v1",
  "analysis_unit": "qa_derived_rewrite_row",
  "unit_key": ["sample_id"],
  "parent_stage_id": "qa_construction",
  "split_id": "post2009-chronological-v1",
  "split_manifest": "dataset/processed/manifests/canonical_meeting_split.json",
  "split_manifest_sha256": "<sha256>",
  "candidate_counts_by_split": {
    "train": 3890,
    "eval": 535,
    "test": 462
  },
  "retained_unit_counts_by_split": {
    "train": 3883,
    "eval": 533,
    "test": 461
  },
  "physical_rows_by_split": {
    "train": 3883,
    "eval": 533,
    "test": 461
  },
  "nested_unit_counts": {
    "analysis_unit": "meeting_section",
    "train": 577,
    "eval": 72,
    "test": 70
  },
  "exclusion_counts_by_reason": {
    "missing_raw_analysis": {
      "train": 7,
      "eval": 2,
      "test": 1
    }
  },
  "retained_quality_flags": {},
  "row_manifest": "dataset/processed/manifests/sample_flow/minutes_sft_manifest.jsonl",
  "source_artifacts": [],
  "code_commit": "<git-commit>",
  "manifest_sha256": "<sha256>",
  "status": "complete"
}
```

### 8.4 Row-level exclusion schema

All exclusions must be row-level and machine-readable:

```json
{
  "schema_version": 1,
  "stage_id": "qa_construction",
  "population_id": "chapter2-post2009-v1",
  "analysis_unit": "qa_row",
  "sample_key": "qa-after-2009-01085",
  "meeting_id": "fomc-2012-06-20",
  "split": "train",
  "reason_code": "empty_response_without_backfill",
  "source_file": "<repo-relative-path>",
  "source_row_index": 1085
}
```

Allowed reason codes should be enumerated in code. At minimum:

```text
no_usable_core_section
other_only_section
duplicate_source_row
pre_2009_meeting
unparseable_meeting_date
target_conflict
empty_response_without_backfill
teacher_response_missing
missing_raw_analysis
missing_reference_excerpt
teacher_prompt_hash_mismatch
teacher_rewrite_empty_response
```

Quality flags retained in a dataset must be stored separately from exclusions.

---

## 9. Automatic table generation

### 9.1 Code organization

Implement:

```text
src/open_r1/utils/sample_flow.py
jobs/main/generate_sample_flow.py
tests/test_sample_flow.py
```

Add a top-level pipeline command:

```bash
python -m jobs.main.run_pipeline sample-flow
```

### 9.2 CLI contract

The complete command should be:

```bash
python -m jobs.main.run_pipeline sample-flow \
  --manifest-root dataset/processed/manifests \
  --json-out dataset/processed/manifests/sample_flow.json \
  --markdown-out docs/summary/20260728T091012Z/chapter2_sample_flow_ledger.generated.md \
  --latex-out docs/Chapter2/generated/sample_flow_table.tex
```

Supported behavior:

- strict mode is the default;
- strict mode exits nonzero when a required stage is missing or inconsistent;
- `--allow-incomplete` may generate a diagnostic JSON and Markdown report;
- incomplete mode must not emit a publication-ready LaTeX table;
- all displayed counts come from stage manifests;
- all paths written to outputs are repository-relative;
- stage order and reason-code order are deterministic;
- no wall-clock timestamp is embedded, so identical inputs produce byte-identical output;
- the aggregate ledger records the code commit, split ID, split hash, stage manifest hashes, and its own content hash.

### 9.3 LaTeX integration

Generate:

```text
docs/Chapter2/generated/sample_flow_table.tex
```

The first line should state:

```latex
% Generated from sample_flow.json. Do not edit manually.
```

The table should use:

```latex
\label{tab:ch2:sample_flow}
```

The hand-written allocation table in:

```text
docs/Chapter2/sections/dataset_construction.tex
```

should be replaced by:

```latex
\input{Chapter2/generated/sample_flow_table.tex}
```

Narrative text must refer to this table and avoid repeating every count manually.

---

## 10. Validation rules

The strict renderer must fail if any of the following checks fails.

### 10.1 Split integrity

- every retained meeting has exactly one split;
- all main post-2009 stages use `post2009-chronological-v1`;
- all stages record the same split-manifest SHA;
- each downstream row's split matches its meeting's canonical assignment;
- matching 102/13/13 totals are insufficient without matching meeting IDs.

### 10.2 Lineage integrity

- every section has a valid retained parent meeting;
- every QA row has a valid meeting and meeting-section parent;
- every SFT, GRPO, and Minutes row has a valid QA parent;
- every Decision row has a valid meeting parent;
- no excluded key appears in a retained stage manifest;
- all referenced source artifacts exist and match recorded hashes.

### 10.3 Count reconciliation

For non-replay stages:

```text
candidate = retained + excluded + allocated_elsewhere
```

For replay stages, the ledger must separately report:

```text
unique underlying units
physical rows
core rows
replay rows
not-selected eligible rows
```

### 10.4 Unit integrity

- `meeting`: unique by `meeting_id`;
- `meeting-section`: unique by `meeting_section_id`;
- `QA row`: unique by `sample_id`;
- `analysis trace`: unique by stage and `sample_id`;
- `GRPO prompt`: unique by physical row ID, with parent `sample_id`;
- `rewrite row`: unique by `sample_id`, with nested meeting-section count;
- `Decision task`: unique by `meeting_id`.

### 10.5 Publication gate

The LaTeX table may be emitted only when:

- every required stage has `status=complete`;
- Decision data have been rebuilt from active, hashed sources;
- no meeting overlap exists across splits;
- the Analysis SFT/GRPO allocation comes from a shared allocation manifest;
- candidate/retained/excluded arithmetic closes;
- all manifest hashes validate.

---

## 11. Required implementation sequence

1. **Freeze current artifacts**
   - record the repository commit and hashes of all current QA, training, and audit manifests;
   - label the current output as `pre-ledger-audit`.
2. **Create the raw-meeting manifest**
   - include all 129 meetings;
   - retain 2009-11-04 as an excluded Train-period unit.
3. **Fix labeled-source duplication**
   - use `.xlsx` as canonical;
   - verify matching `.csv` files rather than concatenating them;
   - create logical paragraph and meeting-section manifests.
4. **Create the canonical split manifest**
   - preserve the current 102/13/13 chronological membership;
   - assign a stable split ID and hash.
5. **Backfill parent identifiers**
   - add `meeting_id` and `meeting_section_id` to QA and downstream row manifests.
6. **Create a shared Analysis allocation manifest**
   - assign 3,112 Train QA IDs to SFT;
   - assign 771 to GRPO core;
   - select 311 replay IDs from the SFT allocation;
   - make both prompt builders consume this manifest.
7. **Materialize stage manifests**
   - QA, Analysis SFT, Analysis GRPO, and Minutes SFT;
   - record exclusions and retained quality flags separately.
8. **Rebuild the Decision branch**
   - restore and hash the original Decision sources;
   - deduplicate by meeting;
   - fail on conflicting targets;
   - retain the 237-meeting full-history population as a separate provenance branch;
   - filter the main task to the 128 post-2009 meetings;
   - inherit the canonical chronological split;
   - prevent Decision SFT Train from using Eval/Test meetings.
9. **Generate aggregate outputs**
   - `sample_flow.json`;
   - diagnostic Markdown;
   - publication LaTeX after strict validation.
10. **Replace the manuscript table**
    - use the generated `\input`;
    - correct surrounding prose and remove conflicting manually entered numbers.

---

## 12. Test plan

### Unit tests

- stable meeting and section IDs;
- Unicode normalization of section names;
- `.xlsx`/`.csv` logical duplicate detection;
- analysis-unit deduplication;
- exclusion arithmetic;
- shared SFT/GRPO allocation;
- Markdown rendering;
- LaTeX escaping;
- deterministic JSON/Markdown/LaTeX output.

### Failure tests

- missing manifest;
- missing source artifact;
- incorrect artifact hash;
- unknown exclusion reason;
- one meeting assigned to two splits;
- downstream row assigned differently from its parent meeting;
- retained/excluded key overlap;
- candidate totals that do not reconcile;
- Decision target conflict;
- independent Decision split with matching marginal counts;
- publication rendering requested while any stage is incomplete.

### Integration tests

1. build manifests in a temporary directory from fixture data;
2. generate aggregate JSON and Markdown twice;
3. confirm byte-identical output;
4. generate LaTeX in strict complete mode;
5. verify the LaTeX table values equal the JSON values;
6. run the top-level `sample-flow --dry-run` command;
7. confirm incomplete mode marks the report and suppresses publication LaTeX.

---

## 13. Acceptance checklist

- [ ] All 129 raw meetings appear in a manifest, including the excluded meeting.
- [ ] The retained split contains exactly 102/13/13 meetings with the documented date boundaries.
- [ ] The section flow reconciles 726 candidates to 719 analysis-ready units.
- [ ] The QA flow reconciles 4,888 candidates to 4,887 retained rows.
- [ ] Analysis SFT reports 3,112/533/461 traces.
- [ ] Analysis GRPO reports 1,082/533/461 prompts and exactly identifies core and replay rows.
- [ ] Actual SFT/GRPO overlap matches the shared allocation manifest.
- [ ] Minutes reports both 3,883/533/461 rewrite rows and 577/72/70 unique meeting-sections.
- [ ] Decision uses the canonical meeting IDs and split hash.
- [ ] The 237-meeting Decision source is a separate population, not silently merged with the 128-meeting flow.
- [ ] Exclusions and retained quality flags are reported separately.
- [ ] Every count in the paper table is generated from `sample_flow.json`.
- [ ] The manuscript imports the generated LaTeX table.
- [ ] Strict generation fails on missing or inconsistent evidence.

---

## 14. Final paper wording

After the Decision rebuild and strict ledger validation, the dataset paragraph can be written as:

> We began with 129 post-2009 FOMC meeting documents. One meeting was excluded after parsing and labeling produced no usable core section, leaving 128 meetings. Meetings were divided chronologically into 102 training, 13 evaluation, and 13 test meetings, and this assignment was inherited by all downstream tasks. The retained meetings yielded 719 analysis-ready meeting-sections and 4,887 QA rows. After teacher-response filtering and stage-specific allocation, the Analysis SFT datasets contained 3,112/533/461 traces and the Analysis GRPO datasets contained 1,082/533/461 prompts. The Minutes-alignment datasets contained 3,883/533/461 QA-derived rewrite rows nested within 577/72/70 unique meeting-sections. The Decision task was constructed at meeting level from the same 128-meeting population and inherited the identical chronological split. All counts were generated from immutable row manifests rather than entered manually.

This paragraph must not be used until the final sentence about Decision split inheritance is demonstrated by the completed active manifest.
