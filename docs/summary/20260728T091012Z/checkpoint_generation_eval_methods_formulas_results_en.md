# Provenance-Controlled Four-Artifact Common-Test Evaluation

## Manuscript-ready methods, formulas, and results

This document provides a self-contained English description of the
provenance-controlled checkpoint evaluation. It is written so that the methods,
equations, tables, and interpretation can be adapted directly for the
dissertation. It distinguishes the pre-specified strict final-answer analysis
from the subsequently requested length-tolerant robustness analysis. The latter
scores every non-empty completion even when generation reached the output-token
limit. No model was retrained and no stored completion was regenerated for
either analysis.

## 1. Scope and evidentiary status

The intended post-training design is the sequential chain

$$
\texttt{chk-0}
\longrightarrow
\texttt{chk-1}
\longrightarrow
\texttt{chk-2}
\longrightarrow
\texttt{chk-3}.
$$

In that design, `chk-1` is obtained by analysis supervised fine-tuning (SFT)
from `chk-0`; `chk-2` is obtained by applying GRPO to `chk-1`; and `chk-3` is
obtained by Minutes SFT from `chk-2`. The intended `chk-3` task takes generated
analysis as input and uses the corresponding original official FOMC Minutes
excerpt as the supervised target, thereby strengthening synthetic-text
generation and Minutes-style rewriting.

The surviving archived artifacts do not instantiate the last two edges of this
chain. Their verified lineage is:

| Evaluation artifact | Design role | Intended parent | Verified parent in the surviving archive | Interpretation |
|---|---|---|---|---|
| `eval-base` | `chk-0` | None | None | DeepSeek-R1-Distill-Llama-8B baseline |
| `eval-analysis-sft` | `chk-1` | `chk-0` | `eval-base` | Recovered analysis-SFT branch |
| `eval-legacy-grpo-from-chk0` | archived `chk-2` role | `chk-1` | `eval-base` | Archived GRPO branch; not the intended `chk-1→chk-2` artifact |
| `eval-minutes-sft-from-chk1` | archived `chk-3` role | `chk-2` | `eval-analysis-sft` | Recovered Minutes-SFT branch; not the intended `chk-2→chk-3` artifact |

Consequently, the evaluation supports common-test comparisons among four
archived artifacts, but it does **not** identify the incremental effect of the
intended `chk-1→chk-2` or `chk-2→chk-3` training stages. This distinction should
be retained in any manuscript table or causal interpretation.

## 2. Frozen common-test design

Let $\mathcal{M}$ denote the set of meetings and $\mathcal{S}$ the set of
requested Minutes sections. The full test contains

$$
|\mathcal{M}|=11,\qquad |\mathcal{S}|=3,\qquad
N=|\mathcal{M}||\mathcal{S}|=33
$$

rows per artifact and $4\times 33=132$ generation rows in total. The three
sections are:

1. Participants' Views on Current Conditions and the Economic Outlook;
2. Staff Review of the Economic Situation; and
3. Staff Review of the Financial Situation.

The meeting dates are March 19, May 7, June 18, July 30, September 17,
October 29, and December 10, 2025, and January 28, March 18, April 29, and
June 17, 2026. A prospective-only robustness subset excludes the first two
meetings and contains nine meetings and 27 rows per artifact.

Each prompt was constructed from a frozen macro-financial evidence packet
available on the calendar day before the relevant meeting ($D-1$). The 33
prompts contain 648--651 structured evidence facts each. The official Minutes
section is stored separately as the reference and is not interpolated into the
generation prompt. In addition to manifest-field assertions, the leakage audit
compared normalized contiguous 20-token sequences for every prompt/reference
pair and found zero overlapping sequences in all 33 rows. This rules out direct
copying at the tested span length; it is not proof against every possible form
of shorter-span or semantic leakage. All four artifacts receive byte-identical
prompts.

Generation is deterministic: temperature is 0, top-$p$ is 1, the maximum
completion length is 8,192 model tokens, the maximum model context is 24,576
tokens, and the row seed is derived deterministically from the sample ID using
base seed 20260729. Input truncation is forbidden and was not observed for any
of the 132 rows.

This common-input benchmark maps raw $D-1$ evidence directly to a requested
Minutes section. It is therefore an end-to-end stress test, not an exact
replication of the intended `chk-3` analysis-to-Minutes rewriting task.

## 3. Notation

For artifact $a$, meeting $m$, and section $s$, let

- $x_{ams}$ be the raw model completion;
- $c_{ams}$ be the text selected for scoring;
- $r_{ms}$ be the held-out official Minutes reference;
- $E_{ms}$ be the frozen evidence set; and
- $z_{ams}$ be an arbitrary row-level metric.

Metrics may be undefined for a row when the generated text contains no
eligible claim. Undefined values are recorded as `NA` and are not converted to
zero in the length-tolerant analysis.

## 4. Candidate extraction and output validity

### 4.1 Strict final-answer policy

Under the pre-specified strict policy, the generation post-processor attempts
to recover a final answer from a supported structured response or from a
non-empty plain-text completion. Define $U_{ams}$ as the indicator that
upstream `valid_generation` is not false, any reported validation status is in
the accepted set (`passed`, `valid`, `validated`, `ok`, or `success`), and
neither explicit parseability field is false. Define $L_{ams}$ as the indicator
that the finish reason is one of `length`, `max_tokens`, or `token_limit`. The
strict validity rule is

$$
V^{\mathrm{strict}}_{ams}
=
\mathbf{1}\{c^{\mathrm{strict}}_{ams}\neq\varnothing\}
\,U_{ams}\,
\mathbf{1}\{L_{ams}=0\}
\,\mathbf{1}\{\mathrm{inputTrunc}_{ams}\ \text{is not true}\}.
$$

Invalid rows remain in the declared $4\times33$ matrix. Their semantic and
ROUGE scores are set to 0, repetition to 1, format compliance to 0, and
rule-coverable factual headline metrics to their finite worst-case values.
Length remains descriptive and is not overwritten.

### 4.2 Length-tolerant open-tag policy

The subsequent robustness analysis uses the raw completion and does not
require a `</think>` closing tag. Let $p(x)$ be the position immediately after
the first case-insensitive `<answer>` opening tag, if one exists. Candidate
extraction is

$$
C(x)=
\begin{cases}
\operatorname{trim}\!\left(
\operatorname{stripTrailingAnswerClose}(x_{p(x):})
\right),
& \text{if an `<answer>` opening tag exists},\\[4pt]
\operatorname{trim}(x),
& \text{otherwise}.
\end{cases}
$$

The optional operation `stripTrailingAnswerClose` removes one terminal
`</answer>` tag and surrounding whitespace. The `<think>` opening tag is used
only for audit metadata. The presence, absence, or position of `</think>` does
not affect extraction or validity. If an `<answer>` opening tag exists but no
non-whitespace text follows it, the extracted candidate is empty and the row is
invalid; extraction does not fall back to the complete raw completion.

Length-tolerant validity is

$$
V^{\mathrm{LT}}_{ams}
=
\mathbf{1}\{
C(x_{ams})\neq\varnothing,\,
\mathrm{inputTrunc}_{ams}\ \text{is not true}
\}.
$$

Thus, `finish_reason=length` is retained as an audit flag but is not a fatal
condition. In the present data, none of the 132 raw completions contains an
`<answer>` opening tag, and none contains a returned `<think>` opening tag.
Every row therefore uses the complete non-empty raw completion. This point is
important: the length-tolerant semantic scores are **raw-completion scores**,
not final-answer-only scores.

## 5. Semantic-similarity metrics

### 5.1 BERTScore

BERTScore is computed using the locally frozen
`FacebookAI/roberta-large` encoder at resolved revision
`722cf37b1afa9454edce342e7895e588b6ff1d59`, layer 17. Baseline rescaling is
disabled. For contextualized candidate embeddings
$\{\mathbf{h}^{c}_{i}\}_{i=1}^{n}$ and reference embeddings
$\{\mathbf{h}^{r}_{j}\}_{j=1}^{q}$, define token similarity

$$
s_{ij}
=
\cos(\mathbf{h}^{c}_{i},\mathbf{h}^{r}_{j})
=
\frac{
(\mathbf{h}^{c}_{i})^\top\mathbf{h}^{r}_{j}
}{
\|\mathbf{h}^{c}_{i}\|_2\|\mathbf{h}^{r}_{j}\|_2
}.
$$

With no inverse-document-frequency weighting, chunk-level BERTScore precision
and recall are

$$
P_{\mathrm{BERT}}
=
\frac{1}{n}\sum_{i=1}^{n}\max_{1\le j\le q}s_{ij},
\qquad
R_{\mathrm{BERT}}
=
\frac{1}{q}\sum_{j=1}^{q}\max_{1\le i\le n}s_{ij},
$$

and

$$
F_{1,\mathrm{BERT}}
=
\frac{2P_{\mathrm{BERT}}R_{\mathrm{BERT}}}
{P_{\mathrm{BERT}}+R_{\mathrm{BERT}}}.
$$

BERTScore precision and recall are retained in the row-level results; F1 is
the headline measure.

### 5.2 Independent MPNet embedding cosine

The independent semantic encoder is the locally frozen
`sentence-transformers/all-mpnet-base-v2` model at resolved revision
`e8c3b32edf5434bc2275fc9bab85f82640a19130`. This encoder is independent of
the training reward models.

For a token-level hidden-state matrix
$H=(\mathbf{h}_1,\ldots,\mathbf{h}_T)$ and attention mask
$a_t\in\{0,1\}$, the chunk embedding is attention-mask mean pooled:

$$
\bar{\mathbf{h}}
=
\frac{\sum_{t=1}^{T}a_t\mathbf{h}_t}
{\sum_{t=1}^{T}a_t},
\qquad
\tilde{\mathbf{h}}
=
\frac{\bar{\mathbf{h}}}{\|\bar{\mathbf{h}}\|_2}.
$$

The chunk-level similarity is

$$
\operatorname{MPNetCos}
=
(\tilde{\mathbf{h}}^{c})^\top\tilde{\mathbf{h}}^{r}.
$$

Cosine similarity has the mathematical range $[-1,1]$; it is not constrained
to $[0,1]$.

### 5.3 Long-text chunking and aggregation

Neither semantic metric silently truncates long text. Candidate and reference
texts are greedily split at sentence boundaries under each tokenizer's
512-token total limit, reserving two special tokens and leaving a 510-token
content budget. A sentence longer than the budget is split deterministically
through its complete token-ID sequence, so no input token is discarded.

Let candidate chunks be
$c^{(1)},\ldots,c^{(K_c)}$ and reference chunks
$r^{(1)},\ldots,r^{(K_r)}$. Chunks are paired by ordinal position using
`zip_longest`. For pair $j$,

$$
w_j=\max\{n^{c}_j,n^{r}_j\},
$$

where a missing-side token count is zero. If both sides exist, let $q_j$ be
the BERTScore or MPNet chunk score. If one side is missing, define $q_j=0$.
The document-level semantic metric is

$$
Q_{\mathrm{doc}}
=
\frac{\sum_{j=1}^{\max(K_c,K_r)}w_jq_j}
{\sum_{j=1}^{\max(K_c,K_r)}w_j}.
$$

This rule penalizes excessive candidate length explicitly because unmatched
candidate chunks receive zero. In the length-tolerant run, the BERTScore audit
records 392 non-empty paired chunks and 1,396 missing-side zero pairs; the
MPNet audit records 396 non-empty paired chunks and 1,398 missing-side zero
pairs. BERTScore precision, recall, and F1 are each aggregated separately with
this rule; document-level F1 is not recomputed from the aggregated document
precision and recall.

## 6. Surface-text metrics

Surface metrics use the common lexical tokenizer

$$
\tau(x)
=
\left[
\operatorname{casefold}(m):
m\in\operatorname{findall}\!\left(
\texttt{[A-Za-z]+(?:['’][A-Za-z]+)?|[-+]?\textbackslash d+(?:\textbackslash.\textbackslash d+)?},
x
\right)
\right].
$$

It retains alphabetic words with an optional internal apostrophe and signed
integer or decimal numbers, while discarding punctuation. This evaluator
tokenization is separate from the RoBERTa and MPNet model tokenizers.

### 6.1 ROUGE-L F1

Let $g=(g_1,\ldots,g_n)$ and $r=(r_1,\ldots,r_q)$ be the generated and
reference token sequences, and let $L(g,r)$ be their longest-common-subsequence
length. Then

$$
P_L=\frac{L(g,r)}{n},
\qquad
R_L=\frac{L(g,r)}{q},
\qquad
\operatorname{ROUGE\mbox{-}L}_{F1}
=
\frac{2P_LR_L}{P_L+R_L}
=
\frac{2L(g,r)}{n+q}.
$$

The value is zero if either token sequence is empty. The implementation uses
an exact bit-parallel longest-common-subsequence algorithm; it changes
computational complexity in Python but not the ROUGE-L estimand.

### 6.2 Length and length ratio

Generated token count is $n$. The row-level length ratio is

$$
\operatorname{LengthRatio}_{ams}
=
\frac{|g_{ams}|}{|r_{ms}|}.
$$

Length is descriptive: a larger value is not automatically better or worse.

### 6.3 Trigram repetition

For $n\ge3$, let $\mathcal{T}_3(g)$ be the multiset of consecutive generated
trigrams and let $U_3(g)$ be its set of unique trigrams. The repetition rate is

$$
\operatorname{Rep}_3(g)
=
1-
\frac{|U_3(g)|}{|\mathcal{T}_3(g)|}.
$$

The rate is zero when fewer than three tokens are present. A value near 1
indicates that almost all trigram positions repeat a previously observed
trigram.

### 6.4 Format compliance

Format compliance is a binary row-level indicator:

$$
\operatorname{Format}_{ams}
=
\mathbf{1}\{\text{no declared format violation is detected}\}.
$$

The body-only contract prohibits reasoning/answer tags, copied evidence
delimiters, and explicit section headings. It also enforces any declared
required or forbidden patterns, literal tags or sections, character bounds,
and balanced known tag pairs. This is a narrow contract check rather than a
holistic writing-quality score. Although `</think>` is ignored for
length-tolerant extraction and validity, a returned reasoning tag can still
violate the independent body-only format contract.

## 7. Factual and numerical consistency

### 7.1 Number extraction and deterministic matching

Generated number mentions include signed integers, comma-grouped integers, and
decimals. Scientific notation, spelled-out numbers, and leading-decimal forms
such as `.5` are not recognized. Mentions contained within recognized time
expressions are marked as time values.

Let $y$ be a generated number displayed with $d(y)$ decimal places and let
$v$ be an allowed evidence value. A match occurs only if

$$
\operatorname{ROUND\_HALF\_UP}\!\left(v,d(y)\right)=y.
$$

This is a deterministic display-precision rule. It is not a relative fuzzy
tolerance and performs no unit conversion. For example, a generated statement
in basis points is not numerically converted to percentage points.

The allowed-number set contains structured evidence values, numeric values in
declared evidence time ranges, explicitly declared allowed values and times,
and numeric mentions extracted from the frozen evidence text. The complete
allowed set is hashed and stored with every numeric claim audit.

### 7.2 Numeric-value accuracy

Let $\mathcal{G}^{\mathrm{num}}$ be the non-time generated number mentions and
$\mathcal{A}$ the allowed-number set. Define

$$
\operatorname{NumAcc}
=
\frac{
\sum_{y\in\mathcal{G}^{\mathrm{num}}}
\mathbf{1}\{\exists v\in\mathcal{A}:v\sim y\}
}{
|\mathcal{G}^{\mathrm{num}}|
},
$$

where $v\sim y$ denotes the display-precision match above. The metric is `NA`
when no non-time number is generated. Numeric-value accuracy evaluates the
value only; unit correctness is reported separately.

### 7.3 Evidence-value coverage

Let $\mathcal{F}^{\mathrm{num}}$ be the distinct structured evidence facts with
numeric values. A fact is covered if at least one non-time generated number
matches its value and, when the fact declares a unit, the generated unit also
matches. Then

$$
\operatorname{EvidenceCoverage}
=
\frac{
|\{f\in\mathcal{F}^{\mathrm{num}}:
\exists y,\ v_f\sim y
\ \land\
(u_f=\varnothing\ \lor\ u_y=u_f)\}|
}{
|\mathcal{F}^{\mathrm{num}}|
}.
$$

This fact-level recall measure must be read jointly with conditional
numeric-value accuracy.

### 7.4 Unit accuracy

A generated number enters the unit denominator only when it matches at least
one structured fact whose unit is declared. If $\mathcal{G}^{u}$ is this
eligible set and $\mathcal{U}(y)$ is the set of expected normalized units for
mention $y$, then

$$
\operatorname{UnitAcc}
=
\frac{
\sum_{y\in\mathcal{G}^{u}}
\mathbf{1}\{u_y\in\mathcal{U}(y)\}
}{
|\mathcal{G}^{u}|
}.
$$

Normalized units include percent, percentage points, basis points, currency
and exchange-rate units, scale units such as thousand/million/billion, and
index points. Missing units are incorrect when an expected unit exists.

### 7.5 Time accuracy

Recognized time expressions include ISO dates and ranges, year ranges, years,
quarters, month-year expressions, and specified relative periods. After
normalizing case, whitespace, and dash characters, a generated time is correct
only if it exactly matches an allowed time extracted from the evidence:

$$
\operatorname{TimeAcc}
=
\frac{
\sum_{t\in\mathcal{G}^{time}}
\mathbf{1}\{t\in\mathcal{A}^{time}\}
}{
|\mathcal{G}^{time}|
}.
$$

The metric is `NA` when no time expression is generated.

### 7.6 Novel-number rate

Using all generated number mentions, including those inside recognized time
expressions,

$$
\operatorname{NovelNumRate}
=
\frac{
\sum_{y\in\mathcal{G}^{allnum}}
\mathbf{1}\{\nexists v\in\mathcal{A}:v\sim y\}
}{
|\mathcal{G}^{allnum}|
}.
$$

Because time-derived numbers are also added to the allowed set, a year or date
present in the evidence is not automatically classified as novel.

### 7.7 Rule-covered unsupported-claim rate

The deterministic rule scope contains:

- non-time numeric claims, supported only when the number is present in the
  evidence and no applicable unit check conflicts;
- time claims, supported only by an exact normalized evidence-time match;
- direction claims;
- policy-stance claims; and
- any explicit custom claim rules.

Let $\mathcal{C}^{rule}$ be these detected claims and $S(c)\in\{0,1\}$ their
rule-based support indicator. Then

$$
\operatorname{UnsupportedRate}
=
\frac{
\sum_{c\in\mathcal{C}^{rule}}[1-S(c)]
}{
|\mathcal{C}^{rule}|
}.
$$

Standalone unit checks do not enter this denominator because the corresponding
numeric claim already incorporates a unit conflict. The metric is `NA` when no
rule-covered claim is detected. It is not an exhaustive natural-language
inference or factuality metric. A detected direction or stance claim for which
the evidence does not yield an expected class is counted as unsupported within
this rule scope. The rate therefore combines rule-detected contradiction with
the absence of rule-adjudicable evidence.

## 8. Direction and policy-stance consistency

### 8.1 Expected directions

The four core topics are inflation, growth, employment, and unemployment.
Evidence directions are normalized to `up`, `down`, or `flat`. When no
explicit expected direction is supplied, the evaluator gives priority to
short-run `derived_absolute_change` facts, then year-over-year
`derived_year_absolute_change` facts, and then other directional facts. For a
derived numeric fact without an explicit direction, the sign of the value
implies `up`, `down`, or `flat`. A topic receives an expected direction only
when the selected evidence agrees; explicitly mixed or conflicting evidence
makes the topic unavailable for consistency scoring.

Within each generated sentence, the nearest direction expression is attached
to each recognized topic mention. Let $\mathcal{C}^{dir}_{cov}$ be generated
direction claims for which an expected evidence direction exists. Direction
consistency is

$$
\operatorname{DirConsistency}
=
\frac{
\sum_{c\in\mathcal{C}^{dir}_{cov}}
\mathbf{1}\{d_c=d^{*}_{topic(c)}\}
}{
|\mathcal{C}^{dir}_{cov}|
}.
$$

If no coverable claim is generated, the metric is `NA`, not zero.

Let $\mathcal{T}^{*}$ be the set of topics with an unambiguous expected
direction and $\mathcal{T}^{gen}$ the set mentioned with a direction in the
generated text. Direction coverage is

$$
\operatorname{DirCoverage}
=
\frac{|\mathcal{T}^{gen}\cap\mathcal{T}^{*}|}
{|\mathcal{T}^{*}|}.
$$

### 8.2 Expected policy stance

Policy stance is normalized to `hawkish`, `dovish`, or `neutral`. An explicitly
declared evidence stance takes priority. Otherwise, a unique policy-rate
direction is mapped as

$$
\mathrm{up}\mapsto\mathrm{hawkish},\qquad
\mathrm{down}\mapsto\mathrm{dovish},\qquad
\mathrm{flat}\mapsto\mathrm{neutral}.
$$

For coverable generated stance claims $\mathcal{C}^{stance}_{cov}$,

$$
\operatorname{StanceConsistency}
=
\frac{
\sum_{c\in\mathcal{C}^{stance}_{cov}}
\mathbf{1}\{s_c=s^{*}\}
}{
|\mathcal{C}^{stance}_{cov}|
}.
$$

Row-level stance coverage is binary:

$$
\operatorname{StanceCoverage}
=
\mathbf{1}\{|\mathcal{C}^{stance}|>0\},
$$

conditional on an expected stance being available. Consistency is `NA` if no
coverable stance claim is generated.

## 9. Aggregation, uncertainty, and paired inference

### 9.1 Meeting-equal-weight aggregation

Metrics are not pooled over all claims or rows. For metric $z$, let
$\mathcal{S}^{z}_{am}$ be the sections with a finite row-level value for
artifact $a$ and meeting $m$. The meeting mean is

$$
\bar z_{am}
=
\frac{1}{|\mathcal{S}^{z}_{am}|}
\sum_{s\in\mathcal{S}^{z}_{am}}z_{ams}.
$$

The reported artifact mean gives each eligible meeting equal weight:

$$
\bar z_a
=
\frac{1}{|\mathcal{M}^{z}_{a}|}
\sum_{m\in\mathcal{M}^{z}_{a}}\bar z_{am}.
$$

Therefore, reported factual rates are means of row-level ratios, first averaged
within meeting and then across meetings; they are not ratios formed by pooling
all detected claims. Tables report the number of finite row-level observations
as the eligibility count $n$.

### 9.2 Cluster-bootstrap confidence intervals

For each metric and artifact, the evaluator draws $B=10{,}000$ bootstrap
samples of $|\mathcal{M}^{z}_{a}|$ meeting means with replacement using seed
20260729. If $\bar z^{*(b)}_a$ is the mean in bootstrap draw $b$, the 95%
percentile interval is

$$
\left[
Q_{0.025}\{\bar z^{*(b)}_a\}_{b=1}^{B},
Q_{0.975}\{\bar z^{*(b)}_a\}_{b=1}^{B}
\right].
$$

At least two eligible meeting clusters are required.

### 9.3 Paired contrasts

Contrasts are allowed only for verified archived parent-child edges:

1. analysis SFT minus base;
2. archived GRPO minus base; and
3. recovered Minutes SFT minus analysis SFT.

For each eligible matched row, the candidate and baseline use the same sample,
prompt, seed, and decoding configuration. Let $\mathcal{P}^{z}_m$ be the
sections in meeting $m$ for which both artifacts have a finite value for
metric $z$. The meeting-level paired difference is

$$
\Delta_m
=
\frac{1}{|\mathcal{P}^{z}_m|}
\sum_{s\in\mathcal{P}^{z}_m}
\left(
z_{\mathrm{candidate},ms}
-
z_{\mathrm{baseline},ms}
\right),
$$

and the reported effect is

$$
\bar\Delta
=
\frac{1}{M_{\Delta}}\sum_m\Delta_m.
$$

Confidence intervals resample the meeting-level differences. Two-sided
sign-flip tests use

$$
p
=
\Pr_{\boldsymbol{\epsilon}}
\left(
\left|
\frac{1}{M_{\Delta}}
\sum_m\epsilon_m\Delta_m
\right|
\ge
|\bar\Delta|
\right),
\qquad
\epsilon_m\in\{-1,+1\}.
$$

Because there are only 11 or nine meeting clusters, all sign configurations
are enumerated exactly (the implementation uses a numerical comparison
tolerance of $10^{-15}$). Holm correction is applied separately within each
evaluation subset. If ordered raw $p$-values are
$p_{(1)}\le\cdots\le p_{(K)}$, then

$$
p^{Holm}_{(k)}
=
\max_{1\le j\le k}
\min\{1,(K-j+1)p_{(j)}\}.
$$

The length-tolerant analysis contains 58 estimable contrast-metric tests in
each subset. The bootstrap intervals are not multiplicity-adjusted; therefore,
an interval that excludes zero can coexist with a Holm-adjusted
$p$-value above 0.05.

## 10. Results

### 10.1 Output validity and extraction audit

| Artifact | Strict valid, all 11 | Strict valid, prospective 9 | Length-tolerant valid, all 11 | Length-tolerant valid, prospective 9 | Token-limit rows, all / prospective | Length-tolerant extraction |
|---|---:|---:|---:|---:|---:|---|
| Base | 0/33 | 0/27 | 33/33 | 27/27 | 33 / 27 | `full_completion` for 33/33 |
| Analysis SFT | 0/33 | 0/27 | 33/33 | 27/27 | 33 / 27 | `full_completion` for 33/33 |
| Archived GRPO | 32/33 | 27/27 | 33/33 | 27/27 | 1 / 0 | `full_completion` for 33/33 |
| Recovered Minutes SFT | 0/33 | 0/27 | 33/33 | 27/27 | 33 / 27 | `full_completion` for 33/33 |

The strict result primarily diagnoses completion reliability. The
length-tolerant result answers a different question: how much measurable
content is present in every non-empty raw completion, irrespective of whether
generation terminated normally.

### 10.2 Strict final-answer results

Each cell reports the all-11 mean followed by the prospective-only mean.
Zeros for fatal rows are pre-specified penalties and are not encoder
assessments of the discarded or incomplete text.

| Metric | Base | Analysis SFT | Archived GRPO | Recovered Minutes SFT |
|---|---:|---:|---:|---:|
| Valid output | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.9697 / 1.0000 | 0.0000 / 0.0000 |
| BERTScore F1 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.5148 / 0.5294 | 0.0000 / 0.0000 |
| MPNet cosine | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.4401 / 0.4562 | 0.0000 / 0.0000 |
| ROUGE-L F1 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0999 / 0.1049 | 0.0000 / 0.0000 |
| Length ratio, descriptive | 7.5872 / 7.7822 | 8.0275 / 8.1563 | 0.7383 / 0.4964 | 6.9199 / 7.5093 |
| Trigram repetition | 1.0000 / 1.0000 | 1.0000 / 1.0000 | 0.0464 / 0.0168 | 1.0000 / 1.0000 |
| Format compliance | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.2424 / 0.2963 | 0.0000 / 0.0000 |

The strict policy has 66 estimable contrast-metric tests per subset. No strict
contrast survives Holm adjustment: the minimum adjusted $p$-value is
0.064453125 for all 11 meetings and 0.2578125 for the prospective-only subset.

### 10.3 Length-tolerant semantic and surface results: all 11 meetings

Cells report meeting-equal-weight mean [95% meeting-cluster bootstrap CI]
and the number of eligible rows.

| Metric | Base | Analysis SFT | Archived GRPO | Recovered Minutes SFT |
|---|---:|---:|---:|---:|
| Valid output | 1.0000 [1.0000, 1.0000] (n=33) | 1.0000 [1.0000, 1.0000] (n=33) | 1.0000 [1.0000, 1.0000] (n=33) | 1.0000 [1.0000, 1.0000] (n=33) |
| BERTScore F1 | 0.1448 [0.1366, 0.1535] (n=33) | 0.1448 [0.1361, 0.1540] (n=33) | 0.6444 [0.5721, 0.7110] (n=33) | 0.1472 [0.1375, 0.1577] (n=33) |
| MPNet cosine | 0.0425 [0.0359, 0.0491] (n=33) | 0.0426 [0.0333, 0.0526] (n=33) | 0.4518 [0.4065, 0.4922] (n=33) | 0.0474 [0.0381, 0.0571] (n=33) |
| ROUGE-L F1 | 0.0268 [0.0248, 0.0288] (n=33) | 0.0355 [0.0336, 0.0374] (n=33) | 0.1163 [0.1101, 0.1206] (n=33) | 0.0328 [0.0293, 0.0365] (n=33) |
| Generated tokens | 6261.6 [6019.4, 6502.6] (n=33) | 6689.7 [6314.9, 7008.3] (n=33) | 1271.6 [1060.1, 1608.3] (n=33) | 5671.0 [5121.4, 6212.8] (n=33) |
| Length ratio | 7.5872 [6.9494, 8.2844] (n=33) | 8.0275 [7.3394, 8.7387] (n=33) | 1.5655 [1.2116, 2.1096] (n=33) | 6.9199 [5.9457, 7.8518] (n=33) |
| Trigram repetition | 0.9840 [0.9814, 0.9862] (n=33) | 0.9868 [0.9839, 0.9892] (n=33) | 0.1151 [0.0893, 0.1577] (n=33) | 0.9849 [0.9787, 0.9902] (n=33) |
| Format compliance | 1.0000 [1.0000, 1.0000] (n=33) | 0.9697 [0.9091, 1.0000] (n=33) | 0.0000 [0.0000, 0.0000] (n=33) | 1.0000 [1.0000, 1.0000] (n=33) |

### 10.4 Length-tolerant semantic and surface results: prospective-only subset

| Metric | Base | Analysis SFT | Archived GRPO | Recovered Minutes SFT |
|---|---:|---:|---:|---:|
| Valid output | 1.0000 [1.0000, 1.0000] (n=27) | 1.0000 [1.0000, 1.0000] (n=27) | 1.0000 [1.0000, 1.0000] (n=27) | 1.0000 [1.0000, 1.0000] (n=27) |
| BERTScore F1 | 0.1439 [0.1347, 0.1538] (n=27) | 0.1429 [0.1331, 0.1535] (n=27) | 0.6626 [0.6025, 0.7226] (n=27) | 0.1444 [0.1347, 0.1546] (n=27) |
| MPNet cosine | 0.0420 [0.0340, 0.0500] (n=27) | 0.0378 [0.0295, 0.0466] (n=27) | 0.4626 [0.4292, 0.4968] (n=27) | 0.0442 [0.0348, 0.0550] (n=27) |
| ROUGE-L F1 | 0.0259 [0.0240, 0.0280] (n=27) | 0.0345 [0.0327, 0.0360] (n=27) | 0.1189 [0.1162, 0.1213] (n=27) | 0.0317 [0.0279, 0.0359] (n=27) |
| Generated tokens | 6267.8 [5980.3, 6543.7] (n=27) | 6720.7 [6264.6, 7100.9] (n=27) | 1132.3 [1059.8, 1220.9] (n=27) | 5974.2 [5563.2, 6429.8] (n=27) |
| Length ratio | 7.7822 [7.0789, 8.5391] (n=27) | 8.1563 [7.3478, 8.9812] (n=27) | 1.3480 [1.2000, 1.5036] (n=27) | 7.5093 [6.7496, 8.2701] (n=27) |
| Trigram repetition | 0.9838 [0.9809, 0.9864] (n=27) | 0.9874 [0.9841, 0.9900] (n=27) | 0.0945 [0.0855, 0.1027] (n=27) | 0.9892 [0.9865, 0.9917] (n=27) |
| Format compliance | 1.0000 [1.0000, 1.0000] (n=27) | 0.9630 [0.8889, 1.0000] (n=27) | 0.0000 [0.0000, 0.0000] (n=27) | 1.0000 [1.0000, 1.0000] (n=27) |

Archived GRPO's zero format-compliance mean does not revoke its
length-tolerant validity. Every complete raw GRPO completion triggered the
separate body-only prohibition on reasoning/answer tags and the known-tag
balance check. These are nonfatal format violations; extraction and validity
continue to depend only on the open-tag rule defined in Section 4.2.

### 10.5 Length-tolerant factual and directional results: all 11 meetings

| Metric | Base | Analysis SFT | Archived GRPO | Recovered Minutes SFT |
|---|---:|---:|---:|---:|
| Numeric-value accuracy | 0.9993 [0.9985, 1.0000] (n=11) | 0.9603 [0.9103, 0.9940] (n=11) | 0.9519 [0.9115, 0.9804] (n=29) | 0.9835 [0.9611, 1.0000] (n=15) |
| Evidence-value coverage | 0.0015 [0.0000, 0.0041] (n=33) | 0.0059 [0.0006, 0.0153] (n=33) | 0.0062 [0.0039, 0.0085] (n=33) | 0.0047 [0.0011, 0.0094] (n=33) |
| Unit accuracy | 0.0428 [0.0001, 0.1157] (n=11) | 0.3569 [0.1493, 0.5962] (n=11) | 0.1458 [0.0933, 0.1943] (n=29) | 0.2706 [0.1143, 0.4310] (n=14) |
| Time accuracy | 1.0000 [1.0000, 1.0000] (n=3) | 1.0000 [1.0000, 1.0000] (n=3) | 0.5379 [0.4278, 0.6677] (n=33) | 0.7500 [0.3750, 1.0000] (n=8) |
| Novel-number rate | 0.0007 [0.0000, 0.0015] (n=13) | 0.0222 [0.0008, 0.0439] (n=12) | 0.0345 [0.0136, 0.0648] (n=33) | 0.0165 [0.0000, 0.0389] (n=15) |
| Rule-covered unsupported rate | 0.8619 [0.6715, 0.9938] (n=13) | 0.5456 [0.3114, 0.7730] (n=12) | 0.8215 [0.7765, 0.8596] (n=33) | 0.4824 [0.3242, 0.6547] (n=15) |
| Direction consistency | NA (n=0) | NA (n=0) | 0.3464 [0.1495, 0.5707] (n=29) | NA (n=0) |
| Direction coverage | 0.0000 [0.0000, 0.0000] (n=33) | 0.0000 [0.0000, 0.0000] (n=33) | 0.8485 [0.7071, 0.9596] (n=33) | 0.0000 [0.0000, 0.0000] (n=33) |
| Policy-stance consistency | NA (n=0) | NA (n=0) | 0.0000 [0.0000, 0.0000] (n=17) | NA (n=0) |
| Policy-stance coverage | 0.0000 [0.0000, 0.0000] (n=21) | 0.0000 [0.0000, 0.0000] (n=21) | 0.8095 [0.6190, 1.0000] (n=21) | 0.0000 [0.0000, 0.0000] (n=21) |

### 10.6 Length-tolerant factual and directional results: prospective-only subset

| Metric | Base | Analysis SFT | Archived GRPO | Recovered Minutes SFT |
|---|---:|---:|---:|---:|
| Numeric-value accuracy | 0.9991 [0.9980, 1.0000] (n=7) | 0.9551 [0.8889, 0.9996] (n=8) | 0.9665 [0.9530, 0.9816] (n=23) | 0.9921 [0.9763, 1.0000] (n=11) |
| Evidence-value coverage | 0.0018 [0.0000, 0.0049] (n=27) | 0.0015 [0.0002, 0.0035] (n=27) | 0.0055 [0.0030, 0.0081] (n=27) | 0.0035 [0.0003, 0.0088] (n=27) |
| Unit accuracy | 0.0598 [0.0000, 0.1618] (n=7) | 0.2867 [0.0784, 0.5832] (n=8) | 0.1262 [0.0718, 0.1755] (n=23) | 0.2746 [0.0924, 0.4785] (n=10) |
| Time accuracy | 1.0000 [1.0000, 1.0000] (n=3) | 1.0000 [1.0000, 1.0000] (n=3) | 0.5796 [0.4593, 0.7216] (n=27) | 0.8333 [0.5000, 1.0000] (n=6) |
| Novel-number rate | 0.0009 [0.0000, 0.0020] (n=9) | 0.0216 [0.0004, 0.0476] (n=9) | 0.0233 [0.0128, 0.0336] (n=27) | 0.0079 [0.0000, 0.0237] (n=11) |
| Rule-covered unsupported rate | 0.8068 [0.5488, 0.9913] (n=9) | 0.5733 [0.2857, 0.8587] (n=9) | 0.8153 [0.7618, 0.8613] (n=27) | 0.4341 [0.2600, 0.6259] (n=11) |
| Direction consistency | NA (n=0) | NA (n=0) | 0.3122 [0.1342, 0.5242] (n=25) | NA (n=0) |
| Direction coverage | 0.0000 [0.0000, 0.0000] (n=27) | 0.0000 [0.0000, 0.0000] (n=27) | 0.8889 [0.7901, 0.9753] (n=27) | 0.0000 [0.0000, 0.0000] (n=27) |
| Policy-stance consistency | NA (n=0) | NA (n=0) | 0.0000 [0.0000, 0.0000] (n=11) | NA (n=0) |
| Policy-stance coverage | 0.0000 [0.0000, 0.0000] (n=15) | 0.0000 [0.0000, 0.0000] (n=15) | 0.7333 [0.4667, 1.0000] (n=15) | 0.0000 [0.0000, 0.0000] (n=15) |

### 10.7 Eligibility counts for conditional metrics

Because the confidence intervals resample meeting-level means, both eligible
row and meeting counts are required to interpret conditional metrics. Each
cell below gives `all-11 eligible rows/meetings; prospective-9 eligible
rows/meetings`.

| Metric | Base | Analysis SFT | Archived GRPO | Recovered Minutes SFT |
|---|---:|---:|---:|---:|
| Numeric-value accuracy | 11/7; 7/5 | 11/8; 8/6 | 29/11; 23/9 | 15/10; 11/8 |
| Evidence-value coverage | 33/11; 27/9 | 33/11; 27/9 | 33/11; 27/9 | 33/11; 27/9 |
| Unit accuracy | 11/7; 7/5 | 11/8; 8/6 | 29/11; 23/9 | 14/9; 10/7 |
| Time accuracy | 3/2; 3/2 | 3/3; 3/3 | 33/11; 27/9 | 8/8; 6/6 |
| Novel-number rate | 13/7; 9/5 | 12/9; 9/7 | 33/11; 27/9 | 15/10; 11/8 |
| Rule-covered unsupported rate | 13/7; 9/5 | 12/9; 9/7 | 33/11; 27/9 | 15/10; 11/8 |
| Direction consistency | 0/0; 0/0 | 0/0; 0/0 | 29/11; 25/9 | 0/0; 0/0 |
| Direction coverage | 33/11; 27/9 | 33/11; 27/9 | 33/11; 27/9 | 33/11; 27/9 |
| Policy-stance consistency | 0/0; 0/0 | 0/0; 0/0 | 17/7; 11/5 | 0/0; 0/0 |
| Policy-stance coverage | 21/7; 15/5 | 21/7; 15/5 | 21/7; 15/5 | 21/7; 15/5 |

### 10.8 BERTScore precision and recall

| Subset | Artifact | Precision | Recall | F1 |
|---|---|---:|---:|---:|
| All 11 | Base | 0.1404 | 0.1496 | 0.1448 |
| All 11 | Analysis SFT | 0.1406 | 0.1494 | 0.1448 |
| All 11 | Archived GRPO | 0.6427 | 0.6464 | 0.6444 |
| All 11 | Recovered Minutes SFT | 0.1422 | 0.1527 | 0.1472 |
| Prospective 9 | Base | 0.1396 | 0.1487 | 0.1439 |
| Prospective 9 | Analysis SFT | 0.1386 | 0.1475 | 0.1429 |
| Prospective 9 | Archived GRPO | 0.6605 | 0.6648 | 0.6626 |
| Prospective 9 | Recovered Minutes SFT | 0.1394 | 0.1499 | 0.1444 |

### 10.9 Key paired contrasts: all 11 meetings

The table below reports candidate-minus-baseline effects for the full
11-meeting sample. Confidence intervals are meeting-cluster bootstrap
intervals. The raw $p$-values use exact meeting-level sign flips, and
$p_{\mathrm{Holm}}$ adjusts over all 58 estimable tests in the subset.

| Contrast | Metric | Difference | 95% CI | Raw p | Holm p |
|---|---|---:|---:|---:|---:|
| Analysis SFT minus base | BERTScore F1 | 0.000013 | [-0.002606, 0.002624] | 0.992188 | 1.000000 |
| Analysis SFT minus base | MPNet cosine | 0.000152 | [-0.011941, 0.011884] | 0.983398 | 1.000000 |
| Analysis SFT minus base | ROUGE-L F1 | 0.008697 | [0.006430, 0.010733] | 0.000977 | 0.056641 |
| Analysis SFT minus base | Repetition rate | 0.002819 | [-0.001088, 0.006927] | 0.236328 | 1.000000 |
| Archived GRPO minus base | BERTScore F1 | 0.499642 | [0.429897, 0.563620] | 0.000977 | 0.056641 |
| Archived GRPO minus base | MPNet cosine | 0.409305 | [0.363479, 0.449173] | 0.000977 | 0.056641 |
| Archived GRPO minus base | ROUGE-L F1 | 0.089551 | [0.082569, 0.094423] | 0.000977 | 0.056641 |
| Archived GRPO minus base | Length ratio | -6.021664 | [-6.776775, -5.250061] | 0.000977 | 0.056641 |
| Archived GRPO minus base | Repetition rate | -0.868863 | [-0.895026, -0.826736] | 0.000977 | 0.056641 |
| Archived GRPO minus base | Format compliance | -1.000000 | [-1.000000, -1.000000] | 0.000977 | 0.056641 |
| Recovered Minutes SFT minus analysis SFT | BERTScore F1 | 0.002391 | [-0.001177, 0.006661] | 0.334961 | 1.000000 |
| Recovered Minutes SFT minus analysis SFT | MPNet cosine | 0.004783 | [-0.006801, 0.015972] | 0.456055 | 1.000000 |
| Recovered Minutes SFT minus analysis SFT | ROUGE-L F1 | -0.002639 | [-0.006245, 0.000823] | 0.204102 | 1.000000 |
| Recovered Minutes SFT minus analysis SFT | Repetition rate | -0.001939 | [-0.007887, 0.003357] | 0.563477 | 1.000000 |

### 10.10 Key paired contrasts: prospective-only nine meetings

| Contrast | Metric | Difference | 95% CI | Raw p | Holm p |
|---|---|---:|---:|---:|---:|
| Analysis SFT minus base | BERTScore F1 | -0.001052 | [-0.003653, 0.001605] | 0.492188 | 1.000000 |
| Analysis SFT minus base | MPNet cosine | -0.004198 | [-0.016888, 0.007932] | 0.531250 | 1.000000 |
| Analysis SFT minus base | ROUGE-L F1 | 0.008580 | [0.006014, 0.010963] | 0.003906 | 0.226562 |
| Analysis SFT minus base | Repetition rate | 0.003614 | [-0.001134, 0.008313] | 0.214844 | 1.000000 |
| Archived GRPO minus base | BERTScore F1 | 0.518654 | [0.461739, 0.576924] | 0.003906 | 0.226562 |
| Archived GRPO minus base | MPNet cosine | 0.420559 | [0.388238, 0.454345] | 0.003906 | 0.226562 |
| Archived GRPO minus base | ROUGE-L F1 | 0.092960 | [0.090101, 0.095475] | 0.003906 | 0.226562 |
| Archived GRPO minus base | Length ratio | -6.434180 | [-7.116027, -5.796408] | 0.003906 | 0.226562 |
| Archived GRPO minus base | Repetition rate | -0.889347 | [-0.899344, -0.879890] | 0.003906 | 0.226562 |
| Archived GRPO minus base | Format compliance | -1.000000 | [-1.000000, -1.000000] | 0.003906 | 0.226562 |
| Recovered Minutes SFT minus analysis SFT | BERTScore F1 | 0.001540 | [-0.001306, 0.004982] | 0.484375 | 1.000000 |
| Recovered Minutes SFT minus analysis SFT | MPNet cosine | 0.006368 | [-0.005681, 0.017988] | 0.375000 | 1.000000 |
| Recovered Minutes SFT minus analysis SFT | ROUGE-L F1 | -0.002727 | [-0.006819, 0.000945] | 0.250000 | 1.000000 |
| Recovered Minutes SFT minus analysis SFT | Repetition rate | 0.001768 | [-0.001679, 0.005495] | 0.398438 | 1.000000 |

No contrast has $p_{\mathrm{Holm}}<0.05$. The minimum adjusted $p$-value is
0.056640625 in the all-meeting analysis and 0.2265625 in the prospective-only
analysis.

## 11. Interpretation

Under the length-tolerant definition, all four artifacts provide non-empty
text for all 33 common-test rows. This should not be interpreted as a 100%
normal-completion or high-quality-answer rate. The base, analysis-SFT, and
recovered Minutes-SFT artifacts reach the 8,192-token completion limit on all
rows. Their mean trigram repetition rates are approximately 0.985, and their
mean lengths are 6.9--8.0 times the reference length. They predominantly
contain extended, repetitive reasoning rather than concise final Minutes
sections.

The archived GRPO branch is materially different descriptively. Its all-meeting
BERTScore F1 is 0.6444, MPNet cosine is 0.4518, and ROUGE-L F1 is 0.1163,
while trigram repetition is 0.1151. These means are substantially better than
the other archived artifacts. However, this does not establish the benefit of
the intended `chk-1→chk-2` stage because the evaluated GRPO artifact was
actually initialized from `chk-0`.

Semantic similarity also does not imply factual grounding. For archived GRPO,
evidence-value coverage is only 0.0062, the rule-covered unsupported rate is
0.8215, direction consistency is 0.3464, and policy-stance consistency is
zero. The other artifacts have similarly very low evidence-value coverage.
Their high conditional numeric-value accuracy is based on relatively few rows
that contain eligible numeric claims and must not be described as broad
coverage of the input evidence.

The recovered Minutes-SFT branch cannot be used to reject the intended
`chk-3` capability. Its verified parent is analysis SFT rather than GRPO, and
the common test supplies raw evidence rather than the generated analysis input
used by the intended Minutes-rewriting task.

## 12. Limitations and manuscript claim boundary

1. **Archived rather than intended lineage.** The last two evaluated artifacts
   do not identify the intended sequential training effects.
2. **Task mismatch for the Minutes branch.** The common prompt is raw
   evidence-to-Minutes, whereas the intended `chk-3` task is
   generated-analysis-to-Minutes.
3. **Post-hoc length-tolerant estimand.** Length-tolerant scores are a
   robustness analysis and should not replace normal-completion statistics.
4. **Raw-completion scoring.** Because no output contains `<answer>`, reasoning
   and answer-like text are not separable under the requested open-tag rule.
5. **Conditional factual denominators.** Numeric, unit, time, direction, and
   stance accuracies may be based on a small subset of rows.
6. **Rule-limited factuality.** The unsupported-claim rate does not cover all
   possible natural-language claims and is not an NLI score.
7. **Narrow format metric.** Format compliance checks explicit contract
   violations; it does not measure coherence, style, or usefulness.
8. **Multiplicity and limited meeting clusters.** There are 11 or nine
   meeting clusters, and no paired contrast survives family-wise Holm
   adjustment at 5%.
9. **No human or LLM-judge evaluation.** The comparison is restricted to
   frozen automatic metrics and deterministic evidence rules.
10. **Ordinal chunk alignment.** Candidate and reference chunks are paired by
    position rather than by an optimized semantic alignment; unmatched chunks
    receive weighted zero scores.
11. **Single stored generation per row.** Inference captures variation across
    meetings but not training-seed or decoding-seed uncertainty.
12. **Cluster assumptions.** The bootstrap treats meetings as exchangeable
    clusters and does not model possible serial dependence between adjacent
    FOMC meetings.
13. **Partially sealed software environment.** Encoder revisions, model
    directory hashes, and scoring parameters are sealed, but the active
    `bert-score` package version (0.3.12) is not recorded inside the sealed
    audit object.

The strongest supported manuscript conclusion is:

> On the frozen raw-$D-1$-evidence-to-Minutes benchmark, the archived
> GRPO-from-base branch exhibited the best observed reference-similarity and
> repetition profile. It nevertheless retained low evidence coverage, a high
> rule-covered unsupported-claim rate, and weak directional and policy-stance
> agreement. These comparisons characterize surviving archived artifacts and
> do not identify the incremental effects of the intended
> `chk-1→chk-2→chk-3` sequence.

## 13. Recommended methodological citations

The manuscript should cite Zhang et al. (2020) for BERTScore, Lin (2004) for
ROUGE, Song et al. (2020) for MPNet, Reimers and Gurevych (2019) for the
sentence-embedding framework, Holm (1979) for the sequential multiplicity
correction, and Efron and Tibshirani (1993) for bootstrap inference. These
citations describe the metric families and inferential procedures; the exact
implementation choices remain those specified above.

## 14. Reproducibility record

The length-tolerant evaluation can be reproduced without training or
regeneration:

```bash
run/eval_checkpoint_generation.sh score-length-tolerant
```

The sealed output directory is
`output/evaluation/main/checkpoint_generation/checkpoint_eval_11_v1/scores_length_tolerant`.

| Output | Rows | SHA-256 |
|---|---:|---|
| `summary.jsonl` | 320 | `437d6c7e6fb48a303d8bb1a60bb2c7f73bac8eb25de2e0960f85cd114f9bae6a` |
| `contrasts.jsonl` | 132 | `138c54170c58f1afe316d4ae3324e221c7d0d284fd6abf4342f73812d640482c` |
| `row_scores.jsonl` | 132 | `aa5852e20a12b9b08856e9fd2a820257d74389c0a67d405c2bf4b9a15cf3c6a5` |
| `claim_checks.jsonl` | 45,522 | `7ce972e6c19aaf1a36516af8e17202b761e71050a67850d534a108d7cc34caea` |
| `audit.json` | 1 object | `37fa80ea65fa5cf2020ca1f5391ea7323858a89c3919dcb2733c5ea33fca84ba` |

The sealed audit payload SHA-256 is
`1c30f3462f3e0c7f6c903283e3e774b7f350c9d5f53247098849eb76439df6d7`,
and the run-spec SHA-256 is
`4bef2d18028c15a948c753542e625488edb7cb737b4e206134cdbeb84af5b9a9`.
The original strict audit remains unchanged, with file SHA-256
`4919cbeb2ef6f450925c4a47fe6b1926cbfd6429bbb1213f9d75291c619a58be`.

The frozen semantic-model directory hashes are:

- RoBERTa-large BERTScore encoder:
  `b970a47c99ab5d994a7bcd689ad86b48fb709076c5d309f6a39d5763478eab88`;
- independent MPNet encoder:
  `1c8bfc2c3cb29e484b3ac3585c5166de44389c32cbfdc33d1f2b51634e37403f`.

The immutable reuse check succeeds without rerunning semantic inference.
