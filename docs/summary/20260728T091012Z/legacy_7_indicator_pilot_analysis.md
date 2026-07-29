# Analysis of the Legacy Seven-Indicator Pilot (`ft_20250330`)

**Analysis date:** 2026-07-28

**Source repository:** `../synthetic_text/synthetic_text` (read-only)

**Recommended status:** legacy prompt-perturbation sensitivity diagnostic only

## Executive conclusion

The surviving pilot is sufficiently complete for a descriptive reanalysis of
**internal output drift**, but it is not a valid source for the canonical
Chapter 2 leave-one-out (LOO) result.

The historical score is

\[
D_{\mathrm{self},i}
=
1-\cos(o_{\mathrm{full}},o_{\mathrm{masked},i}),
\]

not the target-relative score

\[
\Delta_i
=
\cos(o_{\mathrm{full}},y)
-
\cos(o_{\mathrm{masked},i},y).
\]

After excluding five masked responses truncated at Excel's 32,767-character
cell limit, 2,655 of 2,660 indicator-output pairs remain. Their pooled mean
self-distances range only from 0.1189 to 0.1350. This apparent indicator
ranking is not reliable: two sections used an identical masked prompt for all
seven indicator labels, stochastic decoding was unseeded, and none of the
2,660 interventions is an exact single-block deletion.

A source-matched indicator-inactive/template-control analysis gives a more
informative pilot diagnostic:

- In the Economic Situation section, all four active macro indicators have
  negative adjusted self-distance relative to the three
  financial-indicator-inactive prompts (pooled adjusted mean: -0.0078).
- In the Financial Situation section, `sp500` and `us_t10` do not exceed the
  macro-indicator-inactive benchmark. `yield_curve` has a positive
  exploratory signal (adjusted mean: 0.0291; adjusted median: 0.0103;
  meeting-balanced mean: 0.0276).
- These are descriptive internal-drift results, not target-alignment effects,
  causal effects, or formal significance findings.

The existing full and masked texts can be re-embedded to calculate a
target-relative signed delta for this old prompt-perturbation experiment; the
texts do **not** need to be generated again for that limited purpose. A
canonical thesis-quality LOO experiment does require new masked generations,
because the historical prompts do not implement the required exact deletion
intervention.

## 1. Reconstructed analysis population

The analysis was rebuilt from the 133 individual cosine workbooks rather than
the merged workbook. This preserves the batch number required to distinguish
repeated source rows.

| Property | Reconstructed value |
|---|---:|
| Indicators | 7 |
| Indicator-batch workbooks | 133 |
| Batches per indicator | 19 |
| Rows per workbook | 20 |
| Full/masked pairs | 2,660 |
| Retained pairs after integrity exclusions | 2,655 |
| Full-generation draw occurrences | 380 |
| Unique QA source rows selected | 256 of 486 |
| Meetings selected | 114 of 123 |
| Date range | 2009-01-28 to 2024-05-01 |
| Section types | 5 |

The seven historical labels are:

`fed_rate`, `gdp`, `inflation_rate`, `sp500`, `unemployment_rate`,
`us_t10`, and `yield_curve`.

The 380 draw occurrences are not a balanced replicate panel. Among the 256
selected source rows, 162 occur once, 70 occur twice, 19 occur three times,
four occur four times, and one occurs five times. The stable recovery key is

```text
indicator :: legacy_batch :: qa_index
```

Meeting and section identity were recovered by matching the cleaned metadata
text in `input_qa_tagged.json` to the dated section files. All 486 metadata
texts have exactly one exact match. The actual scoring target is instead the
index-aligned `output` field in `data/training/input_qa.json`, which is the
source used by the historical generation script. The longer stable identifier
for future exports should therefore be

```text
meeting_date :: section_name :: qa_index :: legacy_batch :: indicator
```

### Source artifacts

- Individual internal-cosine workbooks:
  `output/archive/cosine/mask/ft_20250330/generations/`
- Raw full/masked workbooks:
  `output/archive/masked/ft_20250330/`
- Complete historical scoring target:
  `data/training/input_qa.json`
- Tagged QA metadata source:
  `output/archive/qa_json/input_qa_tagged.json`
- Merged tagged cosine artifact:
  `output/archive/cosine/mask/ft_20250330/tagged_cos_ft_20250330.xlsx`
- Full-output-to-target batch summary:
  `output/archive/cosine/unmask/cos_ft_20250330/cos_ft_20250330.csv`

All paths above are relative to `../synthetic_text/synthetic_text`.

For provenance, the complete scoring-target file has SHA-256
`3c2b1a94c31c4328328725764e8471a49b062af7ea67d67756ebe5bee7249c3a`.
The merged tagged artifact has SHA-256
`002472ae93530b9ba1d35ea70e75ef9fc8a1c24a00027b40fbb73a6ad2558ec8`,
and the tagged metadata source has SHA-256
`e204d7dad881364545d8b6cd44a86d4429cb44404cff23d3bc8823931c2c0c61`.
The raw 133-workbook inventory has SHA-256
`4dd5dff569381e356b8a769ad694d919234aa35c1991c70f6d707c56c1bb4a5b`,
calculated over sorted lines of
`<filename>\t<file_sha256>\n`.

## 2. Metric that can actually be recovered

The historical cosine script explicitly embeds `unmask_response` and
`mask_response`, producing

\[
c_{\mathrm{internal},i}
=
\cos(o_{\mathrm{full}},o_{\mathrm{masked},i}).
\]

This report analyzes its complement, \(D_{\mathrm{self},i}=1-c\). Algebraically,
this is the LOO drop obtained when the full output itself is treated as the
reference:

\[
\cos(o_{\mathrm{full}},o_{\mathrm{full}})
-
\cos(o_{\mathrm{masked},i},o_{\mathrm{full}})
=1-\cos(o_{\mathrm{masked},i},o_{\mathrm{full}}).
\]

It answers “how much did the generated text move?” It does not answer “how
much did removal reduce similarity to the actual Minutes?”

No row-level `cos(masked output, actual Minutes)` or signed external delta
survives. The only target-relative pilot artifact contains 99 aggregate
full-output cosine values—one per 20-row generation workbook for batches
1–99—with no QA index. Only batches 1–19 correspond to the masked pilot
analyzed here; their mean is 0.8207 (SD 0.0260; range 0.7579–0.8579). These
aggregates cannot be joined to row-level masked scores and are not used in
the analysis below. Cosine geometry also does not permit the missing
masked-to-target cosine to be inferred from full-to-target and
full-to-masked cosine values.

## 3. Integrity checks and exclusions

The tuple `(indicator, legacy_batch, qa_index)` is unique for all 2,660
records. Within each `(legacy_batch, qa_index)`, the same full response is
carried across all seven indicators.

Five masked responses are exactly 32,767 characters long and therefore
truncated rather than complete model outputs. All five belong to the
Financial Situation section and were excluded before summarization.

| Indicator | Batch | QA index | Meeting | Stored self-distance |
|---|---:|---:|---|---:|
| `fed_rate` | 6 | 128 | 2021-01-27 | 0.582716 |
| `gdp` | 6 | 21 | 2014-10-29 | 0.509110 |
| `gdp` | 11 | 5 | 2021-09-22 | 0.690458 |
| `unemployment_rate` | 7 | 81 | 2010-03-16 | 0.660311 |
| `us_t10` | 6 | 303 | 2014-09-17 | 0.708318 |

All 380 full responses are non-empty. Of the 380 target cells copied into the
full-generation workbooks, 46 are themselves Excel-truncated, but every one
is an unambiguous prefix of the corresponding complete `output` in
`data/training/input_qa.json`; the other 334 match exactly. The training and
tagged JSON files are index-aligned by identical instructions, but their
cleaned output texts are identical for only 82 of 486 QA rows. The tagged
file must therefore be used for meeting/section metadata only, not as the
historical scoring target.

The merged tagged workbook should not be used for a new row-level join: it
does not retain the source filename or legacy batch, and its `masked_tag`
values truncate several indicator names. This analysis derived indicator and
batch directly from each original filename.

## 4. Overall descriptive results

The row mean weights every retained draw occurrence equally. The
meeting-balanced mean first averages all retained occurrences within an
indicator and meeting, then gives each meeting equal weight. SD and quartiles
are descriptive row-level summaries; they are not cluster-robust uncertainty
estimates.

| Indicator | N | Meetings | Mean | Median | SD | Q1 | Q3 | Meeting-balanced mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `fed_rate` | 379 | 114 | 0.128294 | 0.082215 | 0.126563 | 0.028881 | 0.183575 | 0.126551 |
| `gdp` | 378 | 114 | 0.118882 | 0.073134 | 0.114650 | 0.032410 | 0.181679 | 0.118811 |
| `inflation_rate` | 380 | 114 | 0.123418 | 0.075851 | 0.123486 | 0.030753 | 0.168790 | 0.117913 |
| `sp500` | 380 | 114 | 0.120427 | 0.081977 | 0.116392 | 0.030807 | 0.160750 | 0.115251 |
| `unemployment_rate` | 379 | 114 | 0.135007 | 0.090793 | 0.132119 | 0.033030 | 0.195345 | 0.135127 |
| `us_t10` | 379 | 114 | 0.124324 | 0.077906 | 0.126146 | 0.032478 | 0.166573 | 0.122790 |
| `yield_curve` | 380 | 114 | 0.134136 | 0.079377 | 0.138971 | 0.031594 | 0.187691 | 0.132659 |

Three features matter for interpretation:

1. The distributions are strongly right-skewed. Every median is materially
   below its mean, and individual retained distances reach 0.736.
2. Meeting balancing changes the indicator means by at most 0.0056. Repeated
   source-row sampling is therefore not the principal reason the overall
   means are close, although it remains a design defect.
3. The largest-minus-smallest row mean is only 0.0161. It is not credible to
   interpret this raw difference as an importance ranking when identical
   masked prompts in other sections produce indicator-labelled mean spans of
   0.0407–0.0499.

The unseeded batch means also vary substantially. Across indicator-specific
19-batch series, the SD of batch means ranges from 0.0223 to 0.0354, and
individual batch means range from 0.0580 to 0.2064.

## 5. Section and intervention audit

The section composition before seven-indicator expansion is:

| Section | Draw occurrences | Unique QA rows (= meetings) | Masked prompt variants per `(batch, index)` | Interpretation |
|---|---:|---:|---:|---|
| Committee Policy Action (CPA) | 100 | 65 | 1 | All seven labels use the same prompt; no indicator-specific intervention |
| Staff Economic Outlook (SEO) | 102 | 70 | 1 | All seven labels use the same prompt; no indicator-specific intervention |
| Staff Review of the Economic Situation (SRE) | 89 | 60 | 5 | Four active macro masks plus one shared financial-indicator-inactive prompt |
| Staff Review of the Financial Situation (SRF) | 88 | 60 | 4 | Three active financial masks plus one shared macro-indicator-inactive prompt |
| Combined Economic and Financial Situation (SREF) | 1 | 1 | 7 | Seven variants, but only one observation |

The retained row-level section means are:

| Section | `fed_rate` | `gdp` | `inflation_rate` | `sp500` | `unemployment_rate` | `us_t10` | `yield_curve` |
|---|---:|---:|---:|---:|---:|---:|---:|
| CPA | 0.145197 | 0.139406 | 0.158121 | 0.137571 | 0.170289 | 0.135137 | 0.129615 |
| SEO | 0.150116 | 0.124756 | 0.115816 | 0.127991 | 0.152911 | 0.131481 | 0.165687 |
| SRE | 0.036361 | 0.034608 | 0.034616 | 0.038201 | 0.033914 | 0.055023 | 0.034823 |
| SREF | 0.052223 | 0.104410 | 0.061422 | 0.059600 | 0.051561 | 0.100815 | 0.055225 |
| SRF | 0.178202 | 0.175434 | 0.183312 | 0.176028 | 0.177839 | 0.174667 | 0.204041 |

CPA and SEO cannot identify indicator effects because the indicator-labelled
prompts are identical within a source draw. Their across-label spans—0.0407
and 0.0499 respectively—are direct evidence that unseeded independent
generation can create an apparent indicator ranking even when no
indicator-specific prompt difference exists. SREF has only one source draw
and should be omitted from any substantive comparison.

More fundamentally, none of the 2,660 stored masked prompts is obtainable
from its full prompt by one exact contiguous deletion. In 2,571 cases the
masked prompt is not even shorter than the full prompt; the remaining 89 are
other non-deletion rewrites. Across the complete 486-row-by-seven-indicator
prompt universe, zero of 3,402 pairs preserves the full template exactly.
The old intervention therefore combines content removal, prompt rewriting,
and stochastic generation.

## 6. Source-matched indicator-inactive/template-control diagnostic

The SRE and SRF template structure permits a limited negative-control
analysis. For each active indicator and the same `(legacy_batch, qa_index)`,
define

\[
D_{\mathrm{adjusted},i}
=
D_{\mathrm{self},i}
-
\frac{1}{|\mathcal N|}
\sum_{j\in\mathcal N}D_{\mathrm{self},j},
\]

where \(\mathcal N\) is the set of indicator labels that are absent from that
section's prompt content and therefore lead to the shared
indicator-inactive/template variant.

- SRE active set:
  `fed_rate`, `gdp`, `inflation_rate`, `unemployment_rate`
- SRE indicator-inactive control set:
  `sp500`, `us_t10`, `yield_curve`
- SRF active set:
  `sp500`, `us_t10`, `yield_curve`
- SRF indicator-inactive control set:
  `fed_rate`, `gdp`, `inflation_rate`, `unemployment_rate`

Strict complete cases require every active row and every inactive-control row
for the same `(batch, index)`. This is a source-matched comparison, not an
RNG-paired experiment: the corresponding generations were independently
sampled and no common random seed survives. All 89 SRE keys are complete. In
SRF, the five truncated rows occur on five different keys, reducing the
matched set from 88 to 83 keys and from 60 to 59 meetings. This excludes the
five corrupt rows and 30 otherwise valid companion rows solely to preserve
the source-matched comparison.

| Section | Active indicator | Complete N | Meetings | Active raw mean | Matched inactive-control mean | Adjusted mean | Adjusted median | Adjusted SD | Meeting-balanced adjusted mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SRE | `fed_rate` | 89 | 60 | 0.036361 | 0.042683 | -0.006322 | -0.001659 | 0.072263 | -0.005723 |
| SRE | `gdp` | 89 | 60 | 0.034608 | 0.042683 | -0.008075 | -0.003089 | 0.065083 | -0.011539 |
| SRE | `inflation_rate` | 89 | 60 | 0.034616 | 0.042683 | -0.008066 | -0.001229 | 0.064040 | -0.010305 |
| SRE | `unemployment_rate` | 89 | 60 | 0.033914 | 0.042683 | -0.008769 | -0.001909 | 0.059477 | -0.010960 |
| SRF | `sp500` | 83 | 59 | 0.175265 | 0.179285 | -0.004020 | -0.014254 | 0.109756 | 0.001695 |
| SRF | `us_t10` | 83 | 59 | 0.170985 | 0.179285 | -0.008300 | -0.010916 | 0.125448 | -0.010796 |
| SRF | `yield_curve` | 83 | 59 | 0.208344 | 0.179285 | 0.029060 | 0.010290 | 0.124922 | 0.027580 |

The corresponding pooled active-indicator results are:

| Section | Active rows | Meetings | Raw mean | Inactive-control mean | Adjusted mean | Adjusted median | Adjusted SD | Meeting-balanced adjusted mean |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SRE | 356 | 60 | 0.034875 | 0.042683 | -0.007808 | -0.001904 | 0.065106 | -0.009632 |
| SRF | 249 | 59 | 0.184865 | 0.179285 | 0.005580 | -0.007771 | 0.120939 | 0.006159 |

A negative adjusted value does not mean that removal improves factual
alignment. It only means that the active-mask output moved less from the full
output than independently sampled indicator-inactive/template outputs did.
The only directionally consistent positive pilot signal is `yield_curve` in SRF:
its mean, median, and meeting-balanced adjusted values are all positive.
However, the adjusted row SD is 0.1249, decoding seeds were not controlled,
and the intervention is not canonical. It should therefore be described as a
hypothesis-generating observation, not a confirmed indicator effect.

## 7. Why this pilot cannot support the Chapter 2 primary claim

The following limitations are jointly decisive:

1. **Wrong estimand in the surviving score.** The stored cosine compares the
   two generated outputs, not either output with the actual Minutes.
2. **Prompt rewriting rather than exact deletion.** All 2,660 interventions
   fail the canonical exact single-block deletion test.
3. **Placebo labels in multiple sections.** CPA and SEO do not apply different
   indicator interventions at all.
4. **Unseeded stochastic decoding.** Generation used `do_sample=True`,
   `temperature=0.6`, and `top_p=0.9` with no recorded seed. Identical prompts
   therefore produce materially different output distributions.
5. **Unbalanced sampling.** Source rows appear one to five times, and batch is
   a random draw occurrence rather than a controlled generation replicate.
6. **Missing model provenance.** The recorded generation checkpoint
   `./models/llama3_qlora_20250330_pt` and historical embedding checkpoint
   `./models/llama3-8b` do not survive at those paths, and no immutable model
   fingerprint was recorded.
7. **Taxonomy and population mismatch.** Seven indicators and five old section
   types do not match the later 25/26-indicator Chapter 2 experiment.
8. **Possible training overlap.** The sampled QA universe is referenced by
   historical training configuration, and the split lineage cannot currently
   rule out overlap.
9. **No valid inferential structure.** Repeated source rows and shared
   meetings require clustered or hierarchical treatment, while common random
   numbers or deterministic decoding would be needed to isolate the prompt
   intervention cleanly.

No p-values or confidence claims are reported here. Applying an ordinary
row-level test would treat dependent records as independent and would attach
formal uncertainty to an invalid intervention contrast.

## 8. Reuse and regeneration decision

| Intended use | Reuse old generated text? | Re-embed? | Regenerate masked output? | Decision |
|---|---|---|---|---|
| Reproduce the internal self-distance summaries in this report | Yes | No | No | Use the stored cosine after the five exclusions |
| Compute target-relative signed delta for the old perturbation design | Yes | Yes, both full and masked text with one frozen encoder | No | Feasible as a separately labelled legacy re-score |
| Produce a canonical Chapter 2 LOO result | No | Yes, within the new run | Yes | Required because the old prompts are not exact deletions |
| Infer signed delta from the old batch-level full-target CSV | No | N/A | N/A | Mathematically and relationally impossible |
| Use the merged tagged workbook as the canonical row ledger | No | N/A | N/A | Batch identity is missing; rebuild from individual files |

For a legacy target-relative re-score, the existing text is adequate except
for the five truncated masked outputs. The procedure should:

1. read the 133 individual workbooks and retain `(indicator, batch, index)`;
2. restore the complete actual-Minutes target from the index-aligned `output`
   field in `data/training/input_qa.json`, while using
   `input_qa_tagged.json` only for meeting and section metadata;
3. normalize the full and masked response wrappers identically;
4. freeze and fingerprint one embedding model and tokenizer;
5. calculate both `cos(full, target)` and `cos(masked, target)` in the same
   run;
6. emit both components and their signed difference on every row;
7. exclude the five truncated masked outputs with explicit reason codes; and
8. summarize descriptively by meeting, while labelling the result as a
   historical prompt-perturbation analysis.

For the canonical experiment, masked outputs must instead be regenerated from
the new exact-deletion manifest with the full indicator roster, fixed split,
model and tokenizer fingerprints, prompt hashes, generation seeds or
deterministic decoding, and meeting-aware inference.

## Appendix: complete section-level descriptive table

The table below uses all retained rows available in each indicator-section
cell. Its SRF values therefore differ slightly from the strict 83-key
complete-case table above.

| Indicator | Section | N | Meetings | Mean | Median | SD | Meeting-balanced mean |
|---|---|---:|---:|---:|---:|---:|---:|
| `fed_rate` | CPA | 100 | 65 | 0.145197 | 0.126595 | 0.107075 | 0.145885 |
| `fed_rate` | SEO | 102 | 70 | 0.150116 | 0.084080 | 0.147840 | 0.146884 |
| `fed_rate` | SRE | 89 | 60 | 0.036361 | 0.020640 | 0.071361 | 0.035952 |
| `fed_rate` | SREF | 1 | 1 | 0.052223 | 0.052223 | N/A | 0.052223 |
| `fed_rate` | SRF | 87 | 60 | 0.178202 | 0.151357 | 0.118729 | 0.182646 |
| `gdp` | CPA | 100 | 65 | 0.139406 | 0.104958 | 0.106509 | 0.141321 |
| `gdp` | SEO | 102 | 70 | 0.124756 | 0.070229 | 0.128048 | 0.127139 |
| `gdp` | SRE | 89 | 60 | 0.034608 | 0.018459 | 0.067883 | 0.030136 |
| `gdp` | SREF | 1 | 1 | 0.104410 | 0.104410 | N/A | 0.104410 |
| `gdp` | SRF | 86 | 59 | 0.175434 | 0.161470 | 0.098095 | 0.179671 |
| `inflation_rate` | CPA | 100 | 65 | 0.158121 | 0.126015 | 0.117620 | 0.154079 |
| `inflation_rate` | SEO | 102 | 70 | 0.115816 | 0.073696 | 0.119119 | 0.115605 |
| `inflation_rate` | SRE | 89 | 60 | 0.034616 | 0.019854 | 0.069142 | 0.031370 |
| `inflation_rate` | SREF | 1 | 1 | 0.061422 | 0.061422 | N/A | 0.061422 |
| `inflation_rate` | SRF | 88 | 60 | 0.183312 | 0.148234 | 0.127225 | 0.184100 |
| `sp500` | CPA | 100 | 65 | 0.137571 | 0.100260 | 0.107018 | 0.135311 |
| `sp500` | SEO | 102 | 70 | 0.127991 | 0.083195 | 0.130932 | 0.127063 |
| `sp500` | SRE | 89 | 60 | 0.038201 | 0.019847 | 0.071216 | 0.036376 |
| `sp500` | SREF | 1 | 1 | 0.059600 | 0.059600 | N/A | 0.059600 |
| `sp500` | SRF | 88 | 60 | 0.176028 | 0.152975 | 0.102274 | 0.183139 |
| `unemployment_rate` | CPA | 100 | 65 | 0.170289 | 0.137760 | 0.125712 | 0.168570 |
| `unemployment_rate` | SEO | 102 | 70 | 0.152911 | 0.083930 | 0.154366 | 0.160082 |
| `unemployment_rate` | SRE | 89 | 60 | 0.033914 | 0.020464 | 0.064250 | 0.030715 |
| `unemployment_rate` | SREF | 1 | 1 | 0.051561 | 0.051561 | N/A | 0.051561 |
| `unemployment_rate` | SRF | 87 | 60 | 0.177839 | 0.141519 | 0.109455 | 0.177641 |
| `us_t10` | CPA | 100 | 65 | 0.135137 | 0.105493 | 0.101385 | 0.130731 |
| `us_t10` | SEO | 102 | 70 | 0.131481 | 0.069692 | 0.143091 | 0.127500 |
| `us_t10` | SRE | 89 | 60 | 0.055023 | 0.019314 | 0.113774 | 0.061095 |
| `us_t10` | SREF | 1 | 1 | 0.100815 | 0.100815 | N/A | 0.100815 |
| `us_t10` | SRF | 87 | 60 | 0.174667 | 0.141374 | 0.114372 | 0.173653 |
| `yield_curve` | CPA | 100 | 65 | 0.129615 | 0.095331 | 0.110310 | 0.129150 |
| `yield_curve` | SEO | 102 | 70 | 0.165687 | 0.086659 | 0.164680 | 0.165460 |
| `yield_curve` | SRE | 89 | 60 | 0.034823 | 0.019970 | 0.063754 | 0.027554 |
| `yield_curve` | SREF | 1 | 1 | 0.055225 | 0.055225 | N/A | 0.055225 |
| `yield_curve` | SRF | 88 | 60 | 0.204041 | 0.166486 | 0.135635 | 0.208631 |
