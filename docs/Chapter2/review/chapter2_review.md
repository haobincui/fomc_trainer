# Chapter 2 Review

## Overall Assessment

This revised Chapter 2 is stronger than the previous version in two important respects. First, the overall training pipeline is now easier to follow, especially the separation between the core post-training checkpoints (`chk-0` to `chk-2`) and the two downstream branches (`chk-3`, `chk-4`). Second, the treatment of the “Sufficient Information / grounding” section is materially improved: the current text in `Chapter2/chapter2.tex:289-316`, `:683-858`, and `:1066-1074` is much more careful in distinguishing internal reliance, external alignment, and mixed evidence.

That said, my overall recommendation is still **major revision**. The main reason is no longer the grounding claim itself, but the fact that the chapter still has several unresolved problems at the level of dataset accounting, task framing, evaluation comparability, and document integrity. At present, the chapter is close to a defensible empirical chapter, but it is not yet methodologically closed enough for submission in its current form.

## Main Strengths

1. The chapter now has a clearer checkpoint-based training narrative, and Table `tab:ch2:model_checkpoints` is a useful organizing device.
2. The revised leave-one-out section is substantially more defensible than before. The text now correctly treats the test as a local diagnostic rather than as full grounding proof.
3. The decision-task benchmark discussion in `Chapter2/chapter2.tex:1004-1057` is better than before because it explicitly distinguishes real-time forecasting, same-meeting text inference, and ex post Minutes-based inference.

## Major Findings

### 1. Dataset accounting is still internally inconsistent, and the sample pipeline is not reproducible end-to-end

**Evidence:** The chapter gives multiple incompatible counts for what appears to be the same underlying corpus. In `Chapter2/sections/dataset_construction.tex:20`, the post-2009 corpus yields `486` section-level samples. In `:35-41`, Table `tab:ch2:summary_statis` reports `123` Minutes and `486` sections. But later, `:489-490` states that the same post-2009 corpus contains `718` section-level samples from `128` meetings, and `:572-574` says the full set contains `4,889` Q\&A pairs from `128` meetings with `718` sections. In addition, `:84` says the main dataset has `5,781` standardized labels and the supplementary dataset has `44,178`, while Table `tab:ch2:indicator_count` at `:124-128` totals only `5,327` and `44,116`. The downstream synthetic-text branch then reports `1,392` Q\&A pairs at `:489-490`, but the training-results section in `Chapter2/chapter2.tex:529` says `Model chk-3` was trained on `1,113` Q\&A pairs, and the dataset split at `dataset_construction.tex:600` reports `578/72/72` sections instead.

**Impact:** A reader cannot reconstruct what exactly is being split, trained, or evaluated at each stage. This is not a cosmetic issue. It affects reproducibility, the interpretation of model performance, and the credibility of all downstream results.

**Recommendation:** Add one explicit sample-flow table that maps every stage of the pipeline. At minimum, the chapter needs to reconcile: raw Minutes count, parsed section count, analytical paragraph count, standardized-label count, distilled Q\&A count, section-level rewriting pairs, and final train/eval/test splits for each task.

### 2. The chapter still overstates the nature of the decision task in several places: this is mostly same-meeting inference, not monetary-policy forecasting

**Evidence:** The introduction and contribution framing still use forecasting language and benchmark claims that are too strong for the implemented task. See `Chapter2/sections/intro.tex:26` on “forecasting monetary policy decisions,” `:62` on comparison with FedWatch-style forecasting methods, and `:71` on “grounding” and “realism.” But the actual task definition in `Chapter2/chapter2.tex:365` says that `current_analysis` is composed of “the other sections from the FOMC minutes,” and the evaluation text in `:926-927` states that non-decision sections from the Minutes are used as inputs. The benchmark discussion in `:1004-1037` itself acknowledges that Panel B is same-meeting statement inference and Panel C is explicitly ex post Minutes-based inference.

**Impact:** The chapter currently asks the reader to compare unlike information sets. That weakens the credibility of the contribution claim, especially where the text suggests comparability to pre-meeting market-implied forecasting tools.

**Recommendation:** Reframe the entire `chk-4` task consistently as **policy-decision inference from same-meeting analytical discussion**, unless you introduce a truly ex ante input set. If you want to keep a forecasting contribution, you need a separate experiment built only on pre-meeting information.

### 3. The core decision-accuracy comparison is not currently interpretable because the compared models are evaluated on different denominators

**Evidence:** In Table `tab:ch2:decision_acc` (`Chapter2/chapter2.tex:929-950`), `Model chk-4` is evaluated on `180` total Minutes, while `Model chk-0` is evaluated on `178`. The split by period is also inconsistent: `128/52` for `chk-4` versus `131/47` for `chk-0`. If the purpose is to compare the two models’ accuracy, they should be evaluated on the same observation set. Otherwise, the chapter must explain which observations were dropped, why they were dropped, and whether drops are correlated with model failure modes.

**Impact:** The reported superiority of `chk-4` over `chk-0` is directionally plausible, but the current table does not permit a clean like-for-like comparison. Unequal denominators can materially distort headline accuracy.

**Recommendation:** Recompute the table on a common evaluation subset, or add a transparent failure-accounting appendix that reports invalid outputs, parsing failures, abstentions, or excluded meetings for each model.

### 4. Chapter 2 still contains unresolved structural and LaTeX integrity problems

**Evidence:** The document now compiles to PDF, but the Chapter 2 log is still not clean. The introduction references `ch2:sec:the-dataset` and `ch2:sec:application` in `Chapter2/sections/intro.tex:77`, but the actual labels are `ch2:sec:the dataset` in `Chapter2/sections/dataset_construction.tex:5` and `ch2:sec:applictaion` in `Chapter2/chapter2.tex:142`. `thesis.log` reports these as undefined references. The dataset split table at `Chapter2/sections/dataset_construction.tex:581-594` declares only three columns (`{l l r}`) while the header has four fields, which produces repeated `Extra alignment tab has been changed to \cr` errors. In addition, `tab:ch2:ffr_changes` and `tab:ch2:vote_choice` are defined twice, once in `Chapter2/sections/model_training.tex:672-714` and again in `Chapter2/chapter2.tex:368-410`, which generates multiply-defined-label warnings.

**Impact:** These are hard document-quality problems. Even when they do not stop compilation, they signal poor chapter hygiene and make the thesis look unfinished.

**Recommendation:** Fix all label/ref mismatches, remove duplicate table labels by consolidating the duplicated content, and correct the column specification in the dataset partition table before treating the chapter as submission-ready.

### 5. The econometric validation remains under-specified and too thin to carry the weight placed on it

**Evidence:** The empirical-model test is framed as an important validation layer in `Chapter2/chapter2.tex:321-344`, but the implementation details remain too sparse. The chapter does not clearly specify the sentiment extraction method used on the synthetic Minutes, the exact futures contract definition, the sample period of the regression, the bootstrap resampling unit, or the confidence-interval reporting protocol. In the results section, `:867-913` reports only bootstrap mean coefficients and mean `t`-statistics, then concludes that the gap to the benchmark coefficient narrows by `20.97%`.

**Impact:** Even with the revised “exploratory” wording, this section is still methodologically under-documented. A reader cannot evaluate whether the regression evidence is stable, significant, or sensitive to specification choices.

**Recommendation:** Either expand this section into a properly documented robustness exercise with tables and interval estimates, or explicitly downgrade it to a brief exploratory appendix-style check rather than one of the chapter’s four core evaluation pillars.

### 6. Chapter-level framing is still not fully aligned with the now more cautious evidence base

**Evidence:** The leave-one-out and conclusion sections are now relatively careful, but the chapter-level framing still contains stronger language. `Chapter2/sections/intro.tex:20` says the chapter evaluates whether the text is “input-grounded.” `:58` describes the method as “data-grounded macro-financial reasoning.” `:71` still presents grounding and economics-based validation as establishing a broader notion of realism. These claims now sit uneasily beside the revised wording in `Chapter2/chapter2.tex:309-316`, `:742-858`, and `:1066-1074`, which correctly describe the evidence as partial, section-dependent, and mixed.

**Impact:** The chapter currently sets stronger expectations in the Introduction than it fulfills in the results. Reviewers will notice this mismatch immediately.

**Recommendation:** Do one final consistency pass across `Introduction`, `Contribution`, and `Conclusion`, and align all chapter-level claims to the revised evidence standard: `textual fidelity` is strong, `indicator reliance` is partial and section-dependent, `external alignment` is mixed, and `econometric validation` is exploratory.

## Minor Comments

### A. The evaluation-method section still contains duplicated exposition and avoidable prose noise

`Chapter2/chapter2.tex:149-153` essentially repeats the previous paragraph on reference-based versus reference-free evaluation. This should be collapsed into one clean paragraph.

### B. Some technical exposition in the base-model section is imprecise or mathematically weak

Examples include the GQA discussion in `Chapter2/chapter2.tex:83`, the unusual parameter shapes in the SwiGLU exposition at `:95-103`, and the broader tendency to mix tutorial-style background with chapter-specific method description. This does not break the chapter, but it weakens technical rigor.

### C. The chapter still needs a full language-editing pass

There are still many grammar and phrasing issues, for example `Chapter2/chapter2.tex:145`, `:169`, `:475`, `Chapter2/sections/model_training.tex:13-17`, and `Chapter2/sections/dataset_construction.tex:13`. These no longer obscure the core argument, but they are numerous enough to affect readability and professionalism.

### D. Benchmark positioning is improved but should still be tightened

The three-panel benchmark discussion in `Chapter2/chapter2.tex:1004-1057` is much better than before, but it would be stronger if the chapter explicitly stated, in one sentence near the table, that the rows are **not** comparable as a unified forecasting leaderboard because the information sets differ materially.

## Bottom Line

This version is a meaningful improvement over the previous draft. The most important positive change is that the chapter no longer overclaims the leave-one-out evidence. However, the chapter still requires a substantial revision round before it is genuinely submission-ready. The highest-priority fixes are:

1. close all dataset and split accounting,
2. reframe the `chk-4` task consistently as decision inference unless a true forecasting setup is added,
3. repair the unequal-denominator accuracy comparison,
4. clean all label/ref/table integrity issues,
5. either strengthen or clearly demote the econometric validation section.

If those five items are fixed, Chapter 2 will be much closer to a solid PhD-thesis empirical chapter.
