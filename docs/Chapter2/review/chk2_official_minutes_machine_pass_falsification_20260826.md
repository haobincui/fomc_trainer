# Machine-PASS Falsification Review: 0 of 16 Rows Survived All Three Axes

- **Review type:** Independent claim-level review
- **Reviewer status:** Separate model-assisted review; not human adjudication
- **Review date:** 2026-08-26
- **Release:** `chk2_official_minutes_semantic_qwen3_14b_v1_20260826`

## Result

None of the 16 fixed machine-PASS rows passed all three required semantic axes under independent claim-level review (**0/16 complete passes**). All 16 official Minutes targets contain at least one factual claim not supported by their paired analysis. All 16 remain within the same specific topic under a conservative topic rule, while 14 contain reasoning plans incompatible with the target actually produced.

This result falsifies the proposition that the current machine-PASS gate reliably establishes “reasoning + official Minutes final-answer” alignment. A second failure is reproducibility: rerunning the same judge configuration over byte-identical judged payloads changed 26 rows' axis verdicts and 11 rows' final PASS/FAIL status. The fixed review set does **not** estimate the error prevalence among all 128 current machine-PASS rows because it is deterministic and non-random.

## Byte-Identical Payloads Produced Different Verdicts

The earlier screen was created at `2026-08-26T22:55:21Z`; the current screen was created at `2026-08-26T23:25:25Z`. The source-release manifest changed when `training_ready=false` was added, but its six data/manifest JSONL files were byte-identical. All 332 IDs also retained identical `analysis_sha256`, `reasoning_sha256`, `official_minutes_sha256`, and `source_response_sha256` values. The builder hash, judge contract, model digest, seed (`20260826`), temperature (`0`), and batch size (`4`) were unchanged. All 84 serialized judge request hashes matched, while **62 of 84 response hashes differed**.

| Screen result | Earlier run | Current run | Change |
|---|---:|---:|---:|
| Machine PASS, train | 101 | 104 | +3 |
| Machine PASS, validation | 6 | 6 | 0 |
| Machine PASS, test | 16 | 18 | +2 |
| **Machine PASS, total** | **123** | **128** | **+5** |
| TS failure rows | 209 | 204 | -5 |
| ST failure rows | 124 | 113 | -11 |
| RC failure rows | 205 | 199 | -6 |
| Confidence-below-threshold rows | 63 | 60 | -3 |

The net `+5` PASS count masks **11 final-status flips**: 8 FAIL-to-PASS and 3 PASS-to-FAIL. Across the 332 rows, 26 changed at least one of TS/ST/RC; TS changed on 11 rows (8 F→T, 3 T→F), ST on 13 (12 F→T, 1 T→F), and RC on 16 (11 F→T, 5 T→F). Numeric confidence changed on 39 rows. This is run-to-run judge nondeterminism, not a payload revision, and independently invalidates the screen as a reproducible training-data gate.

The drift also changed this report's fixed projection: the current set retains 12 of the prior 16 rows. `qa-after-2009-03832`, `qa-after-2009-00836`, `qa-after-2009-00846`, and `qa-after-2009-00849` left the first-8/4/4 PASS set; `qa-after-2009-03370`, `qa-after-2009-00336`, `qa-after-2009-02775`, and `qa-after-2009-04109` entered it.

## Scope and Exact Selection

The reviewed set was selected directly from the machine-PASS projection, preserving each manifest's stored order:

- first 8 rows of `minutes_alignment/manifests/train.jsonl`;
- first 4 rows of `minutes_alignment/manifests/validation.jsonl`; and
- first 4 rows of `minutes_alignment/manifests/test.jsonl`.

The current release contains 128 machine-PASS rows: 104 train, 6 validation, and 18 test. The selected 8/4/4 rows form a fixed non-random falsification set. The set can demonstrate admitted counterexamples, but it cannot support a population precision or prevalence estimate.

The machine judge marked every selected row `target_supported=true`, `same_topic=true`, and `reasoning_compatible=true`. Fifteen carry confidence `0.95`; `qa-after-2009-02052` carries `0.90`.

## Axis Definitions

- **TS — target supported:** every factual claim in the official target is entailed by the analysis without outside knowledge.
- **ST — same topic:** target and analysis concern the same specific economic or financial phenomenon; generic overlap is insufficient.
- **RC — reasoning compatible:** the reasoning is grounded in the analysis and compatible with the target actually produced, including promised facts, quantities, timing, directions, and coverage.

`T` means the axis clearly passes; `F` means at least one concrete counterexample defeats it. Calls are conservative: omissions alone do not fail TS, and topic was retained as `T` where the pair remained within the same specific phenomenon despite factual changes.

## Aggregate Findings

| Measure | Machine verdict | Independent claim-level verdict | Machine-positive failures |
|---|---:|---:|---:|
| Complete three-axis pass | 16/16 | **0/16** | **16/16** |
| TS: target supported | 16/16 | **0/16** | **16/16** |
| ST: same topic | 16/16 | 16/16 | 0/16 |
| RC: reasoning compatible | 16/16 | 2/16 | 14/16 |

These denominators describe only the fixed 16-row set.

## Row-Level Evidence

| Sample ID | Split | TS | ST | RC | Short counterevidence |
|---|---|:---:|:---:|:---:|---|
| `qa-after-2009-02356` | train | F | T | F | Target adds appliances, furniture/carpeting, aircraft, a strike, and outsourced-component problems absent from the analysis. Reasoning promises percentages, index values, and timing; target gives no numbers. |
| `qa-after-2009-03817` | train | F | T | F | Target adds capacity utilization, vehicle inventory-to-sales conditions, communications equipment, and near-term indicators. Reasoning promises nondurables, mining, and utilities coverage that the target omits. |
| `qa-after-2009-04273` | train | F | T | F | Target adds non-auto manufacturing/materials demand, business-survey gains, and August utilization claims. Reasoning promises September and sector coverage not delivered by the target. |
| `qa-after-2009-04370` | train | F | T | T | Target adds funding and corporate spreads, implied volatility, liquidity-facility wind-down, and bank-loan-rate increases absent from the analysis. The generic stylistic reasoning remains compatible. |
| `qa-after-2009-01236` | train | F | T | F | Target adds small firms as a major employment-growth source and regional-bank vulnerability to deteriorating CRE. Reasoning plans commercial-paper and cyclical-investment coverage missing from the target. |
| `qa-after-2009-02078` | train | F | T | F | Target adds exports, computer/communications equipment, business equipment, construction supplies, and lean dealer inventories. Reasoning promises full sector preservation that the target does not provide. |
| `qa-after-2009-02104` | train | F | T | F | Target adds workweek/temp-hiring signals, disappointing employment, claims, policy uncertainty, and productivity. Reasoning plans unemployment-rate timing and quantities omitted from the target. |
| `qa-after-2009-03370` | train | F | T | F | Target adds capital-market access and cash holdings of large firms, small-firm survey results, CRE/mortgage standards, improving credit quality, and bankers seeking growth. Reasoning promises SLOOS quantities, saving rates, consumer-credit levels, and the KBW index; target omits them. |
| `qa-after-2009-00636` | validation | F | T | F | Target turns a possible near-term CUSIP step into a definite plan and asserts prior cost/complexity reductions. Reasoning promises balance-sheet quantities and composition absent from the operations-only target. |
| `qa-after-2009-00648` | validation | F | T | F | Target adds a February auto decline, aluminum shortage, input-price elevation, port congestion, and driver shortage. Reasoning promises the March 0.8% value and other numbers; target supplies neither March nor numbers. |
| `qa-after-2009-02052` | validation | F | T | F | Target adds C&I/CRE balance growth, a CMBS slowdown, and small-business credit trends; the analysis's strong balance data concern consumer credit. Planned equity, survey-percentage, consumer-credit, and saving evidence is omitted. |
| `qa-after-2009-02926` | validation | F | T | T | Target adds quits, District contacts, household-survey/QCEW evidence, retention behavior, immigration/transport costs, and demographic effects absent from the analysis. The core labor-tightness reasoning remains compatible. |
| `qa-after-2009-00018` | test | F | T | F | Analysis does not support “economic outlook” as the most-cited reason or expected tightening through year-end. Reasoning promises categories, quantities, consumer credit, and saving; target contains only three qualitative C&I sentences. |
| `qa-after-2009-00336` | test | F | T | F | Within the shared outlook-risk topic, target adds an upside inflation-risk balance, adverse supply shocks, and a financial-market reaction absent from the analysis. Reasoning promises solid GDP growth and delayed demand-supply alignment, neither delivered. |
| `qa-after-2009-02775` | test | F | T | F | Target adds upside inflation risk, adverse supply shocks, and policy tightening necessitated by persistent inflation, none supported by the GDP analysis. Reasoning promises observed growth moderation, figures, gradual deceleration, and upside growth risks; target omits them. |
| `qa-after-2009-04109` | test | F | T | F | Target adds October capital-goods shipments, construction, home sales, and a strike effect. It says the trade deficit widened as exports fell and imports rose, contradicting the analysis's narrowing deficit, growing service exports, and contained imports; reasoning also promises the opposite trade direction. |

## Interpretation and Release Recommendation

The dominant claim-level failure mode is substitution of broad topic overlap for strict entailment: unsupported entities, indicators, causes, timing, and directions pass despite the judge contract. The fixed set also shows that reasoning compatibility is not being checked against the target's actual coverage. Confidence of at least `0.90` on every selected false complete pass is not credible calibration evidence for this set. Separately, the 26 changed axis verdicts show that one nominally deterministic run cannot produce a stable gate even when every judged payload hash is unchanged.

Keep `training_ready=false` and do not use the 128-row machine-PASS projection as training supervision without further adjudication. Review all 128 rows at claim level; require evidence spans from each target sentence to its supporting analysis; fail closed when any claim lacks evidence; require repeated runs to produce zero verdict flips before treating the gate as reproducible; and retain these 16 rows plus the 26 drift rows as regression tests.

## Reproducibility Record

- Release manifest: `dataset/processed/retrain_v2/chk2_official_minutes_semantic_qwen3_14b_v1_20260826/release_manifest.json`
- Current release-manifest SHA-256: `d075315f65b71d5421b1fcf1ba33a2144facd4d138d0b706c3cac325a7e07e98`
- Current source-release-manifest SHA-256: `12fc545f217bd6b45dfd713c8d34012488e1d4683c323ffb3cfb88f2d807f02d`
- Current semantic-quality SHA-256: `a52c8f0ce475fa1baf3aa6df80234092a6f6d90f7c6c1bf9419e8e8921598999`
- Judge contract SHA-256: `7d73d937335a300773ae6f53f97b1db4b26bec439bf335a3e0cdd6d4397b1c49`
- Builder SHA-256: `014d6483dea10d3960bbf68215bdb7ac4f087b9302bbfc6e354d1111b2427df2`
- Current train manifest SHA-256: `be5ecf6a9d11d2038b7db1d8c38cbd93e4092c3976dd076c22bf03b884cf3cce`
- Current validation manifest SHA-256: `da40939853884537340056fa7c71972e3dfe720a718a3b1da5b2ac5ed73a19fb`
- Current test manifest SHA-256: `dc19003325b4da53d737978a74d91b2ab6116b2562ef5be9c365e909032d3007`
- Archived earlier release: `dataset/processed/retrain_v2/chk2_official_minutes_semantic_qwen3_14b_v1_20260826_pre_training_ready_field/release_manifest.json`
- Archived earlier release-manifest SHA-256: `f27ee01c394fe0b62a82a27f5c6a974b2d759034bdc9844d3348c020e28b4b60`
