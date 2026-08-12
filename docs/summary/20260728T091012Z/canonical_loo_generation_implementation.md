# Canonical Leave-One-Out Generation Implementation

## Status

The repository now contains a generation-only implementation for the new
Chapter 2 leave-one-out (LOO) experiment. It does not train, continue-train,
merge, or modify either model.

The frozen model roles are:

| Role | Frozen artifact |
|---|---|
| Indicator analysis model | `../fomc_trainer_back/fomc_trainer/output/merged/llama_grpo_20250515` |
| Indicator analysis tokenizer | `models/DeepSeek-R1-Distill-Llama-8B` |
| Minutes generation model | `output/checkpoints/recovered/llama_sft_synthetic_20250526_2_cp1668_recovered_v1_20260729/model` |
| Minutes generation tokenizer | the tokenizer bundled with the Minutes model |

The Minutes default is the provenance-verified recovery of the surviving
checkpoint-1668 adapter merged with its declared analysis-SFT parent. The
historical `llama_sft_synthetic_20250526` directory is excluded because its
model payload was overwritten by decision-GRPO weights. The paths are defaults,
not identity claims. At run time, the complete model
and tokenizer directory trees are hashed and written to the frozen experiment
specification.

## Scientific boundary

The resulting experiment is a prompt intervention on frozen models and
prespecified chronological populations. It may be described as a canonical LOO
prompt-intervention result after its generation release manifest passes.

It must not be described as a held-out model result until checkpoint lineage
has independently established that the relevant meetings did not enter
training, selection, or teacher construction. This implementation deliberately
does not attempt to repair checkpoint lineage by retraining.

## Frozen populations

| Population | Split label | Meetings | Date range | Config |
|---|---|---:|---|---|
| `pilot_eval_13` | eval | 13 | 2021-12-15 to 2023-06-14 | `configs/main/loo_population_pilot_eval_13.json` |
| `formal_test_13` | test | 13 | 2023-07-26 to 2025-01-29 | `configs/main/loo_population_formal_test_13.json` |

The two populations are ordered, non-overlapping, and fixed in
`configs/main/canonical_loo_generation.json`.

Every meeting generates the following three section families:

1. `Participants' Views on Current Conditions and the Economic Outlook`
2. `Staff Review of the Economic Situation`
3. `Staff Review of the Financial Situation`

## Required release-safe input

Historical generated analyses in `synthetic_text` are not promoted into this
run. They do not provide a reliable meeting-time information boundary, and the
audited 2023-07-26 example contains post-meeting information.

The implementation now builds the input rather than accepting an
unsubstantiated hand-written ledger. Its only timing rule is:

```text
information_as_of_date = requested_vintage_date
                       = meeting_date - 1 calendar day
observation_date <= information_as_of_date
```

All meeting-day observations are excluded. The implementation does not apply
an 08:30-versus-14:00 exception, and it does not infer an intraday release
time. “D-1 data” means the data visible in the D-1 vintage snapshot; it does
not mean that every observation must itself be dated D-1.

`configs/main/loo_indicator_sources.json` is the canonical source registry. It
contains the 26-indicator source map, source metadata, disabled-source reasons,
license controls, and an exact crosswalk for the 102 legacy files under
`dataset/raw_data/input_data/us_data`. Legacy and `synthetic_text` values are
mapping-only and are never copied into a canonical ledger.

Some public series are declared proxies rather than exact replacements. The
most material substitutions are NFCICREDIT for unavailable ICE
rating/maturity yield ladders and OECD country share-price indices for
unavailable MSCI series. Results using these constructs must call them proxies,
not equivalent reconstructions of the legacy inputs.

The snapshot fetcher uses the keyless ALFRED Graph CSV endpoint. It does not
read or require `FRED_API_KEY`. For the union of 26 meetings it makes three
requests per enabled series, using 12, 12, and 2 aligned
`vintage_date`/`cosd`/`coed` values. It rejects a response unless its header
contains exactly the requested historical-vintage columns and all non-empty
values are finite and no later than their corresponding D-1 cutoff. Raw
responses, request parameters, retrieval timestamps, byte counts, and SHA-256
hashes are retained in a sealed snapshot manifest.

The builder replays those raw responses and emits one row per meeting and
roster indicator. Each population therefore contains `13 × 26 = 338` rows. An
abridged row is:

```json
{
  "schema_version": "canonical-loo-indicator-input-v2",
  "meeting_id": "2024-01-31",
  "sample_id": "2024-01-31::GDP-Growth",
  "meeting_timestamp": "2024-01-31T19:00:00Z",
  "meeting_date": "2024-01-31",
  "indicator": "GDP-Growth",
  "source_id": "canonical-loo-d1:formal_test_13:2024-01-31:GDP-Growth",
  "source_sha256": "64-character-lowercase-sha256",
  "source_timestamp": "2026-07-28T00:00:00Z",
  "information_as_of_date": "2024-01-30",
  "requested_vintage_date": "2024-01-30",
  "availability_as_of_date": "2024-01-30",
  "availability_evidence_type": "alfred_vintage_snapshot",
  "source_interface": "alfred-graph-csv-v1",
  "observation_date": "2023-10-01",
  "source_payload": {
    "sampling_policy": {"version": "d1-frequency-aware-v1"},
    "series": [
      {
        "source_key": "alfred__GDPC1",
        "series_id": "GDPC1",
        "requested_vintage_date": "2024-01-30",
        "availability_as_of_date": "2024-01-30",
        "observations": [
          {"date": "2023-10-01", "value": "example only"}
        ]
      }
    ]
  }
}
```

`source_timestamp` is retrieval provenance only. It is retained in the ledger
and manifest but deliberately omitted from the analysis-model prompt, so the
model never receives a post-meeting retrieval date.

For prompt-size stability, the frozen sampling policy retains up to 25
month-end observations for daily/weekly series, 24 monthly observations, 8
quarterly observations, 6 semiannual observations, or 5 annual observations.
It records every discarded raw record and reason in the exclusion ledger.

The builder and independent validator reject:

- unstable or duplicate sample IDs;
- indicators outside the frozen 26-indicator roster;
- incomplete meeting-by-indicator coverage;
- inconsistent meeting timestamps;
- anything other than the exact previous-calendar-day vintage;
- any observation dated after D-1;
- malformed, current-vintage fallback, or mismatched ALFRED responses;
- a changed registry, population, roster, request, raw response, or output hash;
- meetings outside the selected population; and
- empty source payloads or malformed source hashes.

The ledger deliberately stores no manufactured `release_timestamp`.
Date-granular ALFRED vintage availability is the evidence used by this
simplified design. Sources whose terms permit local research but not
redistribution can participate locally; raw snapshots and ledgers remain
gitignored, and the ledger manifest declares
`local_research_use_only=true` when such a source is active.

### Live integration verification

On 2026-07-28, the finalized registry and both populations were exercised
against the real keyless endpoint. All 279 requests returned validated HTTP
200 CSV responses (93 series × `12 + 12 + 2` vintage batches). Independent
raw-byte replay produced 338 input rows and 338 evidence rows for each
population, with zero top-level or nested post-D-1 observations. The temporary
integration artifacts were removed after validation; this check proves the
pipeline is operable but is not itself a dissertation release.

## Generation stages

### 1. Indicator analysis

`jobs.generation.canonical_indicator_analysis` creates one deterministic
analysis block for every meeting-indicator pair. It:

- uses temperature `0`, top-p `1`, and seed `20260728`;
- excludes actual Minutes from the prompt;
- fingerprints the source ledger, population, roster, model, and tokenizer
  before and after generation;
- rejects context overflow, silent prompt-token changes, input truncation, and
  incomplete output;
- uses a frozen 8,192-token technical generation limit while its dedicated
  system prompt requires concise final-answer-only prose to stay within 4,096
  tokens, leaving overflow tolerance without requesting a longer answer;
- excludes chain-of-thought, padding, headings, and preambles from the
  requested response;
- permits at most two otherwise valid token-limit finishes, recording their
  sample IDs and token metadata in every affected row and in the analysis
  manifest; a third such finish fails the stage before artifacts are published;
- writes `indicator_analysis.jsonl` and `analysis_manifest.json`; and
- refuses to overwrite an existing run.

### 2. Deterministic final-answer projection

`jobs.generation.project_indicator_analysis` preserves the immutable raw
analysis JSONL and creates a separate `minutes_analysis.jsonl`. It accepts only
a well-formed DeepSeek completion with exactly one `</think>` delimiter,
non-empty reasoning before it, and a non-empty final answer after it. Only the
final answer is projected. Plain-text fallback, token truncation, a second
generation model, and generative summarization are forbidden.

The projection manifest binds the raw analysis manifest and output, every
canonical raw-row hash, every extracted-answer hash, the frozen Minutes
tokenizer, and per-answer token counts. The release finalizer independently
repeats the extraction and rejects any mismatch.

The existing 13-meeting pilot was replayed through this implementation without
model inference: all 338 analyses projected successfully, final answers ranged
from 51 to 312 Minutes-tokenizer tokens, and all 2,106 exported prompt rows
passed the frozen context gate. The maximum chat-template prompt was 5,091
tokens, leaving a minimum 3,101-token context margin.

### 3. Full, deletion, and neutral prompts

`jobs.generation.loo_prompt_builder` constructs 26 fixed-order indicator blocks
for each meeting and then creates:

- one full prompt;
- 26 exact single-block deletion prompts; and
- 26 label-preserving neutral replacements.

The neutral replacement is matched using the frozen Minutes tokenizer. Both
the replaced block and the complete prompt must have exactly the same token
count as their full-prompt counterparts. The builder also proves that only one
declared span changed and that its prefix, suffix, indicator label, and block
delimiters were preserved.

The builder uses only the projected `minutes_analysis` field. It applies the
real Minutes tokenizer chat template, including the frozen system prompt and
generation prefix, to every exported row. Prompt construction fails before
artifact publication unless:

```text
chat_prompt_tokens + 8,192 <= 16,384
```

Actual Minutes, references, gold answers, and target-text fields are forbidden
from these prompt artifacts.

For one population, the prompt inventory is:

| Artifact | Count |
|---|---:|
| Indicator analyses | 338 |
| Full meeting-section prompts | 39 |
| Exact-deletion prompts | 1,014 |
| Neutral-replacement prompts | 1,014 |

### 4. Frozen experiment specification

`jobs.generation.build_loo_generation_spec` fingerprints both models, both
tokenizers, the source ledger, analysis artifacts, prompt artifacts, rosters,
population, and configuration. It seals the payload with a canonical JSON
digest and writes an immutable `loo-generation-spec-v1` file.

### 5. Minutes generation

`jobs.generation.mask_generation` runs four generation cells:

| Intervention | Regime | Replicates | Temperature | Top-p |
|---|---|---:|---:|---:|
| Exact deletion | primary | 1 | 0.0 | 1.0 |
| Neutral replacement | primary | 1 | 0.0 | 1.0 |
| Exact deletion | stochastic robustness | 5 | 0.6 | 0.9 |
| Neutral replacement | stochastic robustness | 5 | 0.6 | 0.9 |

Each row seed is derived only from `replicate_seed` and `sample_id`. It is
therefore invariant to batch size, prompt order, and indicator identity. Full
and intervened prompts use a common row seed within a replicate.

The five stochastic replicate seeds are:

```text
20260729
21260729
22260729
23260729
24260729
```

Before inference, the exact chat-template token count must satisfy:

```text
prompt_tokens + max_new_tokens <= max_model_len
```

After inference, the consumed prompt-token count must equal the preflight
count. Unknown finish reasons are always rejected. Length-limited finishes are
rejected by default; the explicitly documented operational partial-shard
policy may retain and exclude them from scoring.

### 5. Generation release validation

`jobs.generation.finalize_loo_generation` re-hashes every declared prompt and
generation artifact, revalidates per-row seeds and completion safety, checks
the required `1/1/5/5` replicate design, and requires the full baseline output
to match exactly between deletion and neutral runs for every
regime-replicate-sample combination. It then writes the sealed
`canonical-loo-generation-release-v1` manifest. A run is not complete merely
because the four generation processes exited.

## Background launcher

The end-to-end launcher runs in the background by default:

```bash
./run/generate_loo_end_to_end.sh all
```

It performs source fetch, snapshot sealing, both 338-row ledger builds,
raw-evidence replay validation, pilot generation, formal generation, and final
workflow sealing. It prints the run ID, PID, log path, and workflow directory
before returning. No API key is needed. Before any download or background
launch, it resolves physical GPU `1` through `nvidia-smi`, sets
`CUDA_VISIBLE_DEVICES=1`, and requires exactly one logical CUDA device. It also
verifies that logical `cuda:0` has the UUID reported by `nvidia-smi` for
physical GPU `1`. The numeric selector is required by the pinned vLLM runtime.
This pin is reapplied by the inner launcher, propagates to all `nohup` workers
and vLLM children, and cannot be overridden by a caller-supplied CUDA device
list. vLLM tensor parallelism is fixed at `1`. The launcher then imports
PyTorch, Transformers, vLLM, and the project generation module. On this host it selects
`~/.conda/envs/llama_factory/bin/python` when `LOO_PYTHON` is unset because the
base Python does not contain a complete PyTorch installation.

Run only one population with:

```bash
./run/generate_loo_end_to_end.sh pilot

LOO_PILOT_RELEASE_MANIFEST=/absolute/path/to/pilot_release.json \
  ./run/generate_loo_end_to_end.sh formal
```

Useful optional environment variables are:

```text
LOO_RUN_ID
LOO_WORKFLOW_BASE
LOO_BATCH_SIZE
LOO_PYTHON                  # override the validated generation environment
LOO_FETCH_WORKERS            # 1 or 2
LOO_FETCH_RATE               # at most 2 requests/second
LOO_RESUME                   # reuse only verified immutable source cache
LOO_OFFLINE                  # require an already complete snapshot manifest
LOO_ANALYSIS_MODEL
LOO_ANALYSIS_TOKENIZER
LOO_MINUTES_MODEL            # must equal the manifest-registered frozen path
LOO_MINUTES_TOKENIZER        # must equal the manifest-registered frozen path
LOO_REUSE_ANALYSIS_MANIFEST  # optional immutable raw pilot analysis manifest
LOO_INTERVENTION_INDICATORS  # comma-separated pilot compute shard
LOO_MINUTES_TOKEN_LIMIT_POLICY  # error (default) or exclude
```

The inner worker performs a fail-closed checkpoint provenance preflight before
Minutes generation. It verifies the sealed manifest file and payload digests,
requires the selected `eval-minutes-sft-from-chk1` record to be usable, hashes
the complete runtime model and tokenizer directory trees, and matches those
hashes to the frozen manifest record. Consequently, a path override cannot
silently select the corrupted historical `llama_sft_synthetic_20250526`
directory or another merely loadable checkpoint.

Set `LOO_FOREGROUND=1` to run synchronously for debugging. The launcher refuses
to reuse an existing workflow unless `LOO_RESUME=1`; partial generation
directories are never silently resumed.

`LOO_INTERVENTION_INDICATORS` filters only generated deletion and neutral
cells. It does not shrink the evidence roster: raw analysis, projected
analysis, and the `None` baseline retain all 26 indicator blocks. The launcher
automatically includes the baseline, validates IDs against the frozen roster,
restricts this mode to the pilot population, and writes
`intervention_shard_manifest.json` instead of a canonical release. That
manifest records `status=partial_complete` and
`standalone_canonical_release=false`. Omitted cells must later be generated
under the identical specification before the ordinary finalizer can seal a
complete canonical release. Non-canonical legacy seven-indicator workbooks
cannot substitute for omitted canonical cells.

The default Minutes token-limit policy is `error`. An operational partial shard
may opt in to `LOO_MINUTES_TOKEN_LIMIT_POLICY=exclude`. This retains the raw
token-limit completion, marks it `excluded_token_limit_finish`, continues
later generation, and propagates a sample-level exclusion inventory into the
generation and shard manifests. The paired evaluator never sends a marked row
to the embedding scorer. If any exclusion is observed, the shard status is
`partial_complete_with_generation_exclusions`; it is a complete-case,
non-canonical result and cannot be sealed by the full canonical finalizer.

## Run layout

```text
output/evaluation/main/canonical_loo/workflows/<run_id>/
├── run.log
├── run.pid                         # present only while the worker is active
├── workflow_status.jsonl
├── workflow_release_manifest.json
├── inputs/
│   ├── loo_indicator_sources.json
│   ├── leave_one_out_roster.json
│   ├── loo_sections.json
│   ├── canonical_loo_generation.json
│   ├── loo_population_pilot_eval_13.json
│   └── loo_population_formal_test_13.json
├── source_snapshots/
│   ├── snapshot_manifest.json
│   └── raw/alfred/...
├── ledgers/
│   ├── pilot_eval_13/
│   │   ├── indicator_inputs.jsonl
│   │   ├── source_evidence.jsonl
│   │   ├── excluded_records.jsonl
│   │   ├── coverage.json
│   │   └── ledger_manifest.json
│   └── formal_test_13/...
└── generations/
    ├── pilot_eval_13/
    │   ├── analysis_raw/
    │   ├── analysis_projection/
    │   ├── prompts/
    │   ├── generations/
    │   ├── generation_spec.json
    │   └── generation_release_manifest.json
    └── formal_test_13/...
```

Every generated JSONL is tied to its source prompt hash, sample ID, generation
position, row seed, replicate, model and tokenizer hashes, decoding settings,
system-prompt hash, token counts, finish reason, and output hash.

## What is intentionally not done

- No checkpoint is trained, continued, selected, merged, or edited.
- No historical masked output is relabelled as canonical.
- No actual Minutes text is used during indicator analysis or prompt
  construction.
- No cosine score or causal claim is produced by the generation launcher.
- No test population is used for checkpoint selection.

Scoring remains a separate downstream step. The intended estimand remains:

```text
Delta = cos(full output, target) - cos(intervened output, target)
```

where `target` is a separately frozen actual-Minutes artifact and never an
input to generation.
