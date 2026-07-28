You are helping with a PhD thesis on meeting-level FOMC rate-decision classification.

Your job is NOT to invent new baselines. Your job is to implement, as faithfully as possible, a literature-grounded baseline suite for predicting the FOMC committee action at each meeting.

======================================================================
0. PRIMARY TASK
======================================================================

Task:
Predict the FOMC committee action at each meeting.

Primary label:
action_class in {cut, hold, hike}

Define the label using the midpoint of the target range:

mid_before = (target_low_before + target_high_before) / 2
mid_after  = (target_low_after  + target_high_after ) / 2
delta_mid_bps = 100 * (mid_after - mid_before)

If delta_mid_bps < 0: action_class = cut
If delta_mid_bps = 0: action_class = hold
If delta_mid_bps > 0: action_class = hike

Optional robustness label:
delta_mid_bps_class in {-50, -25, 0, +25, +50}

======================================================================
1. HARD INFORMATION-SET RULES
======================================================================

You MUST keep different information sets in separate panels.
Do NOT mix them in one leaderboard.

Panel A: PRE-MEETING FORECASTING
Allowed inputs:
- macro / financial / survey variables known before the meeting
- fed funds futures
- fed funds futures options
- any text that was public before the meeting (if provided)

Forbidden:
- same-meeting official statement
- same-meeting minutes
- transcripts

Panel B: ANNOUNCEMENT-TIME TEXT INFERENCE
Allowed inputs:
- same-meeting official FOMC statement ONLY AFTER stripping all sentences that explicitly reveal the decision or vote record

Forbidden:
- raw statement sentences that directly reveal the action
- minutes
- transcripts

Panel C: MINUTES-BASED INFERENCE (optional, ex post)
Allowed inputs:
- same-meeting minutes, but only as a separate panel
Reason:
- FOMC minutes are generally released three weeks after the policy decision

Panel D: HISTORICAL-RESEARCH PANEL (optional)
Allowed inputs:
- alternative statements
- transcripts
Reason:
- transcripts are public with about a five-year lag
- alternative statements are also historical/internal materials

Official Fed references for timing and leakage constraints:
- Meeting calendars and minutes timing:
  https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm
- Example official statement showing the decision and vote record in the statement itself:
  https://www.federalreserve.gov/newsevents/pressreleases/monetary20260318a.htm
- Example official minutes release note:
  https://www.federalreserve.gov/newsevents/pressreleases/monetary20260408a.htm
- FOMC transcripts availability with about a five-year lag:
  https://www.federalreserve.gov/monetarypolicy/fomc_historical.htm
  https://www.fedsearch.org/fomc-docs/resources/category_text.jsp

Interpretation rule:
Models are compared only within the same information set.

======================================================================
2. DATA FILES EXPECTED
======================================================================

Required:
- data/meetings.csv
  columns:
  meeting_id
  meeting_date
  target_low_before
  target_high_before
  target_low_after
  target_high_after
  delta_mid_bps
  action_class

- data/statements.csv
  columns:
  meeting_id
  meeting_date
  statement_text_raw
  statement_text_stripped

Optional:
- data/minutes.csv
  meeting_id
  meeting_date
  minutes_release_date
  minutes_text_raw
  minutes_text_stripped

- data/macro_features.csv
  meeting_id
  [macro and financial predictors known before the meeting]

- data/futures_features.csv
  meeting_id
  zq_price_or_implied_rate
  fff_expected_change_bps
  fff_p_cut
  fff_p_hold
  fff_p_hike

- data/futures_options_features.csv
  meeting_id
  [discrete outcome probabilities if available]

- data/alt_statements.csv
  meeting_id
  alt_a_text
  alt_b_text
  alt_c_text
  alt_d_text (nullable)

======================================================================
3. TEXT STRIPPING RULES
======================================================================

For Panel B and Panel C, you must either verify or construct stripped text.

Remove or mask any sentence matching patterns like:
- "The Committee decided to ..."
- "decided to raise"
- "decided to lower"
- "decided to maintain"
- "target range for the federal funds rate"
- "Voting for the monetary policy action"
- "Voting against this action"
- "preferred to lower"
- "preferred to raise"
- "preferred no change"

Rationale:
official FOMC statements frequently contain the answer directly.

Do not silently skip this.
Persist both raw and stripped text.

======================================================================
4. LITERATURE REFERENCE PACK
======================================================================

Your implementation should be grounded in the following baseline families.

----------------------------------------------------------------------
B1. MARKET-IMPLIED BASELINES: FED FUNDS FUTURES / OPTIONS
----------------------------------------------------------------------

Core papers:
1) Kuttner (2001), Journal of Monetary Economics
   "Monetary Policy Surprises and Interest Rates: Evidence from the Fed Funds Futures Market"
   Link:
   https://www.sciencedirect.com/science/article/pii/S0304393201000551
   Also:
   https://ideas.repec.org/a/eee/moneco/v47y2001i3p523-544.html

2) Carlson, Craig, and Melick (2005), Journal of Futures Markets / Cleveland Fed WP
   "Recovering Market Expectations of FOMC Rate Changes with Options on Federal Funds Futures"
   Links:
   https://onlinelibrary.wiley.com/doi/10.1002/fut.20187
   https://www.clevelandfed.org/publications/working-paper/2005/wp-0507-recovering-market-expectations-of-fomc-rate-changes

3) Keasler and Goff (2007)
   "Using Fed Funds Futures to Predict a Federal Reserve Rate Hike"
   Link:
   https://www.economics-finance.org/jefe/fin/KeaslerGoffpaper.pdf

4) CME FedWatch methodology
   Links:
   https://www.cmegroup.com/markets/interest-rates/cme-fedwatch-tool.html
   https://www.cmegroup.com/articles/2023/understanding-the-cme-group-fedwatch-tool-methodology.html
   https://www.cmegroup.com/tools-information/quikstrike/cme-fedwatch-tool-user-guide.html

5) Piazzesi and Swanson (2008), Journal of Monetary Economics
   "Futures prices as risk-adjusted forecasts of monetary policy"
   Links:
   https://www.sciencedirect.com/science/article/abs/pii/S0304393208000494
   https://ideas.repec.org/a/eee/moneco/v55y2008i4p677-691.html

Implement two versions:

B1a. Simple futures-implied expected change
If fed funds futures price is quoted as:
r_fut = 100 - Price

For a meeting month with D calendar days total and d days up to and including the meeting date,
a standard weighted-average identity is:

r_fut_month = (d / D) * r_current + ((D - d) / D) * E[r_post_meeting]

Therefore:
E[r_post_meeting] = (D * r_fut_month - d * r_current) / (D - d)

If only two outcomes are assumed, e.g. hold or +25bp hike:
p(hike) = (E[r_post_meeting] - r_current) / 25bp

General multi-outcome version:
r_fut_month = sum_j p_j * r_j
subject to:
sum_j p_j = 1, p_j >= 0

B1b. Kuttner-style surprise measure
For high-frequency change around the announcement, use the Kuttner rescaling:

Surprise_t = (FF_t^+ - FF_t^-) * d_m / (d_m - d)

where:
FF_t^+ = current-month fed funds futures implied rate just after the announcement
FF_t^- = current-month fed funds futures implied rate just before the announcement
d_m    = number of days in the month
d      = day of month on which the meeting occurs

This exact implementation formula is reproduced in Lucca and Trebbi’s appendix/discussion citing Kuttner.

Important caveat:
Raw fed funds futures are not pure expectations at longer horizons because of risk premia.
Use them as a strong benchmark, but not as a perfect probability oracle.

Deliverables:
- class prediction via argmax if p_cut/p_hold/p_hike available
- expected bp move
- probability outputs for evaluation

----------------------------------------------------------------------
B2. DISCRETE-CHOICE MACRO BASELINES: HAMILTON / ORDERED PROBIT
----------------------------------------------------------------------

Core papers:
1) Hamilton (2002), Journal of Money, Credit and Banking
   "A Model of the Federal Funds Rate Target"
   Links:
   https://www.jstor.org/stable/10.1086/341872
   https://www.nber.org/papers/w7847

2) van den Hauwe, Paap, and van Dijk (2013), Journal of Macroeconomics
   "Bayesian forecasting of federal funds target rate decisions"
   Links:
   https://www.sciencedirect.com/science/article/abs/pii/S0164070413000876
   https://ideas.repec.org/a/eee/jmacro/v37y2013icp19-40.html

Operational baseline:
Use a dynamic ordered probit / ordered response model.

Latent variable:
y_t^* = x_t' beta + rho * g(y_{t-1}) + epsilon_t
epsilon_t ~ N(0,1)

Observed class:
y_t = cut   if y_t^* <= kappa_1
y_t = hold  if kappa_1 < y_t^* <= kappa_2
y_t = hike  if y_t^* > kappa_2

where x_t includes only pre-meeting variables.

Implementation notes:
- Use statsmodels OrderedModel if available
- If dynamic lagged class is included, do it carefully with strictly past information
- Estimate class probabilities and save them

Metrics:
Accuracy, Macro-F1, Balanced Accuracy, Brier, LogLoss

----------------------------------------------------------------------
B3. NONLINEAR MACRO BASELINE: RANDOM FOREST / ORDINAL FOREST
----------------------------------------------------------------------

Core paper:
Yoon and Fan (2024), Journal of Forecasting
"Forecasting the direction of the Fed's monetary policy decisions using random forest"
Links:
https://onlinelibrary.wiley.com/doi/full/10.1002/for.3144
https://ideas.repec.org/a/wly/jforec/v43y2024i7p2848-2859.html

Goal:
Provide a nonlinear pre-meeting macro benchmark.

Preferred implementation:
- ordinal forest if package support exists

Fallback implementation if no proper ordinal forest package:
Cumulative binary decomposition:
q1_t = P(y_t > cut | x_t)
q2_t = P(y_t > hold | x_t)

Then recover:
P(cut)  = 1 - q1_t
P(hold) = q1_t - q2_t
P(hike) = q2_t

Use random forest classifiers or gradient boosting classifiers for q1_t and q2_t.

If this decomposition is numerically inconsistent:
- clip probabilities to [0,1]
- renormalize so probabilities sum to 1

Be explicit in logs:
"ordinal approximation used; not an exact replication of Yoon and Fan (2024)"

----------------------------------------------------------------------
B4. TRADITIONAL TEXT BASELINE: LUCCA–TREBBI SEMANTIC ORIENTATION
----------------------------------------------------------------------

Core paper:
Lucca and Trebbi (2009), NBER Working Paper 15367
"Measuring Central Bank Communication: An Automated Approach with Application to FOMC Statements"
Links:
https://www.nber.org/papers/w15367
https://www.nber.org/system/files/working_papers/w15367/w15367.pdf

Two key formulas from the paper:

(1) Semantic orientation using PMI:
SO(x) = PMI(x, hawkish) - PMI(x, dovish)

(2) Factiva semantic orientation score:
FSO_t = log(
    [sum_{s in T_t} I[s, R, P]] /
    [sum_{s in T_t} I[s, R, N]]
)

where:
- T_t = set of relevant news sentences around meeting t
- R = relevance words, e.g. Rates, Policy, Statement, Fed, FOMC, Federal Reserve
- P = positive-rate-move words, e.g. hawkish, tighten, hike, raise, increase, boost
- N = negative-rate-move words, e.g. dovish, ease, cut, lower, decrease, loose

Unexpected change in semantic orientation:
Delta_FSO_t = FSO_t^+ - FSO_t^-

For this thesis baseline:
If Factiva-style news corpus is unavailable, implement a stripped-statement proxy:
lex_score_t = (hawk_count_t - dove_count_t) / token_count_t
or
tfidf_lex_score_t = sum_j tfidf_{j,t} * polarity_j

Important:
- This is a literature-inspired operational baseline
- It is NOT a literal replication unless you have the external news corpus

Model variant:
- Use lex_score_t directly as a stance feature
- Or combine TF-IDF features with multinomial logistic regression:
  p(y_t = k | x_t) = softmax(W x_t + b)_k

Save:
- raw lexicon score
- class prediction
- probability outputs if logistic is used

----------------------------------------------------------------------
B5. ALTERNATIVE-STATEMENT TEXT BASELINE: DOH–SONG–YANG
----------------------------------------------------------------------

Core paper:
Doh, Song, and Yang (2020/2025 update)
"Deciphering Federal Reserve Communication via Text Analysis of Alternative FOMC Statements"
Links:
https://www.kansascityfed.org/documents/5642/rwp20-14dohsongyang.pdf
https://www.aeaweb.org/articles?from=f&id=10.1257/mac.20220312

This paper is highly relevant if historical alternative statements are available.

Exact formulas reported in the paper:

Tone:
Tone_t =
[ sim(F_t^c, F_t) - sim(F_t^a, F_t) ] /
[ 1 - sim(F_t^a, F_t^c) ]

where:
- F_t   = released official statement embedding
- F_t^a = dovish alternative embedding
- F_t^c = hawkish alternative embedding
- sim(.) = cosine similarity or stated semantic similarity metric

Novelty:
Novelty_t = 1 - sim(F_t, F_{t-1})

Stance:
Stance_t = Novelty_t * Tone_t

They also define expected stance and monetary-policy surprise:
E_{t-Δ}[Stance_t] = (1 - 2 p_{t-Δ}) * Novelty_{t-Δ}
MPS_t = Stance_t - E_{t-Δ}[Stance_t]

Implementation plan:
If alt_statements.csv exists:
1. Compute embeddings for official statement and alternatives
2. Compute Tone_t
3. Compute Novelty_t
4. Compute Stance_t
5. Use Stance_t as:
   - a continuous baseline feature
   - a thresholded classifier into cut / hold / hike
   - or an explanatory predictor combined with macro features

Thresholding rule:
Estimate thresholds on the training set only.
Use either:
- terciles / quantiles
- or optimize thresholds on training data for Macro-F1

Be explicit:
This baseline is ex post / historical-research only if alternatives are not available in real time.

----------------------------------------------------------------------
B6. DOMAIN-SPECIFIC CENTRAL-BANK LM BASELINE: CB-LMs
----------------------------------------------------------------------

Core paper:
Gambacorta et al. (2024), BIS Working Papers
"CB-LMs: language models for central banking"
Link:
https://www.bis.org/publ/work1215.pdf

Relevant findings:
- The paper uses FOMC statement sentences manually labeled as dovish / hawkish / neutral
- 1,243 labeled sentences from 1997–2010
- 80/20 train-test sentence split
- top-performing CB-LM reaches mean accuracy around 84%

Operational baseline:
Use a domain-specific encoder if available, otherwise use a strong sentence-transformer or finance LM.

Formally:
e_t = Encoder(text_t)
p(y_t = k | e_t) = softmax(W e_t + b)_k

Recommended variants:
- FinBERT embedding + linear probe
- sentence-transformer embedding + linear probe
- RoBERTa/DeBERTa/CB-LM embedding + linear probe

Use only stripped same-meeting statement text in Panel B.

Optional meeting-level aggregation if using sentence-level classification:
- classify each sentence
- aggregate by average hawkish probability minus dovish probability
- map aggregate stance to cut / hold / hike using training-set thresholds

----------------------------------------------------------------------
B7. SELF-SUPERVISED LLM STANCE BASELINE: DCS
----------------------------------------------------------------------

Core paper:
Tang and Yang (2026)
"Mind the Shift: Decoding Monetary Policy Stance from FOMC Statements with Large Language Models"
Links:
https://arxiv.org/abs/2603.14313
https://github.com/yixuantt/DeltaConsistentScoring

Key idea:
Stance should be modeled relatively across consecutive FOMC statements, not only as isolated classification.

Exact formulas from the paper:

Absolute and relative projections:
z_t^abs = theta_abs' h_t^abs + b_abs
z_t^rel = theta_rel' h_t^rel + b_rel

Stance score:
s_t = sigmoid(z_t^abs),  s_t in [0,1]

Delta-consistency loss:
L_delta = E_t [ ((z_t^abs - z_{t-1}^abs) - alpha * tanh(z_t^rel / tau))^2 ]

Confidence regularizer:
L_conf = - E_t [ s_t log s_t + (1 - s_t) log(1 - s_t) ]

Total objective:
L = L_delta + lambda * L_conf

Implementation rule:
If full DCS repo is not practical, implement a simplified DCS-style baseline:
1. Frozen LLM embeddings for each statement
2. Absolute projection head
3. Relative projection head on consecutive pairs
4. Train with the delta-consistency objective
5. Convert s_t to cut / hold / hike using training-set thresholds

Be explicit:
This is a DCS-style baseline if you do not literally run the authors’ code.

----------------------------------------------------------------------
B8. RECENT ADVANCED LLM COMPARATORS: MINIFED / FEDSIGHT
----------------------------------------------------------------------

These are optional advanced comparators, not mandatory first-round baselines.

MiniFed (2024)
"MiniFed: Integrating LLM-based Agentic-Workflow for Simulating FOMC Meeting"
Link:
https://arxiv.org/abs/2410.18012

FedSight AI (2025)
"FedSight AI: Multi-Agent System Architecture for Federal Funds Target Rate Prediction"
Links:
https://arxiv.org/abs/2512.15728
https://openreview.net/pdf/2e9f074511a9793a451212db90eaa3395e36e66c.pdf

Use these only as optional advanced comparators because:
- they are new
- they are architecturally heavy
- they are less standard than futures / ordered probit / RF / text baselines

If implemented:
- each agent returns a class or rate suggestion
- final committee action is the aggregation / majority / structured vote

Do not prioritize this before the classical baselines.

======================================================================
5. REQUIRED BASELINES TO IMPLEMENT
======================================================================

You must attempt these in order, conditional on available data.

Panel A: PRE-MEETING
A0. Naive-Hold
A1. Persistence (same as last action_class)
A2. Futures-Prob / Futures-Expected-Change
A3. Ordered-Probit
A4. Ordinal-Forest or ordinal approximation

Panel B: ANNOUNCEMENT-TIME STRIPPED STATEMENT
B1. TFIDF-Logit
B2. Embedding-Linear
B3. Lexicon / semantic-orientation score
B4. DCS-style relative shift baseline (optional but strongly preferred if feasible)

Panel C: MINUTES (optional)
C1. TFIDF-Logit on stripped minutes
C2. Embedding-Linear on stripped minutes

Panel D: HISTORICAL RESEARCH ONLY (optional)
D1. Alternative-statement tone / novelty / stance
D2. MiniFed / FedSight if resources allow

======================================================================
6. EVALUATION
======================================================================

For every panel, report:
- Accuracy
- Macro-F1
- Balanced Accuracy

If probabilities are available, also report:
- Brier Score
- Log Loss

Save:
- raw out-of-sample probabilities
- predicted classes
- meeting_id
- meeting_date
- panel
- train_end_date
- test_date

Only use time-ordered expanding-window evaluation:
train = all meetings <= t-1
test  = meeting t

Forbidden:
- random split
- shuffled k-fold
- any look-ahead leakage

======================================================================
7. OUTPUT FILES
======================================================================

Required outputs:
results/panelA_premeeting_metrics.csv
results/panelB_statement_metrics.csv
results/panelC_minutes_metrics.csv   (if applicable)

results/predictions_panelA.csv
results/predictions_panelB.csv
results/predictions_panelC.csv       (if applicable)

results/latex/table_panelA_premeeting.tex
results/latex/table_panelB_statement.tex
results/latex/table_panelC_minutes.tex  (if applicable)

Also create:
results/run_log.txt

The log must explicitly state:
- which baselines ran
- which baselines were skipped
- why they were skipped
- what information set each panel uses
- what exact formulas / approximations were used
- what train/test date ranges were used

======================================================================
8. IMPLEMENTATION PRIORITIES
======================================================================

Round 1:
- Naive-Hold
- Persistence
- Futures-Prob
- Ordered-Probit
- TFIDF-Logit on stripped statements

Round 2:
- Ordinal-Forest
- Embedding-Linear
- Lexicon / semantic orientation score

Round 3:
- DCS-style baseline
- Alternative-statement baseline
- Minutes panel
- MiniFed / FedSight

======================================================================
9. FINAL CAUTIONS
======================================================================

1. Do not claim to replicate a paper unless the implementation is genuinely close.
2. If you use an operational approximation, say so explicitly.
3. Keep all panels separate by information set.
4. Do not use the official statement’s answer-bearing sentences in Panel B.
5. Prefer transparent failure over silent fabrication.

End of prompt.