# chk3 and chk4 checkpoint-selection probe update

Created: 2026-08-18 10:55 UTC

## Decisions

- **chk3:** retain `checkpoint-318`. The new `cp280` and `cp300`
  neighbourhood probes each pass only 10/12 preregistered core cases, whereas
  `cp318` passes 12/12 in both adapter-overlay screening and exact-merged
  replay. This supports `cp318` as the hard-gate-qualified late-plateau
  checkpoint, not as a statistically unique optimum.
- **chk4:** keep `checkpoint-450` only as the **best-observed exploratory
  candidate**. The new local neighbours `cp440` and `cp460` are both worse
  under the frozen selection panel, but none of the three reaches the
  preregistered delivery-valid threshold. The formal selection remains
  `selected_checkpoint_step=null`, and chk4 quality has not been demonstrated.

The new runs are retrospective, post-hoc robustness diagnostics. They do not
make either historical selection preregistered.

## Frozen contracts

### chk3

The new `cp280` and `cp300` runs use the same adapter-overlay contract as the
existing `cp170`, `cp200`, `cp230`, and `cp318` screen:

- the same 12-case native analysis-to-Minutes manifest
  (`sha256=371d29601e343acf98cac6173663d842ead9acaa4a454991d512b3f00bf5ee77`);
- the same parent model, prompt and chat template, ordered cases, per-case
  seeds, greedy decoding, 4-bit loading, SDPA, 3,072-token cap, and
  1,024-token tail audit;
- the preregistered core gate: delivery, native structure, numeric-multiset
  fidelity, date-set fidelity, and degeneration limits.

The ordered case-key digest for `cp280`, `cp300`, and the historical overlay
`cp318` is identical:
`82bed6bbcf8946b5fde706262535c58fcc0811316c70045a8b060aac4ce62ed6`.
The full contract digest is also identical:
`48d7983138e8352080e86dfc9535b9b2b622e9ac4811650156fb2e028db7021d`.

The runner's stored `quality_valid` additionally includes a signed-number
surface diagnostic. That diagnostic is not substituted for the preregistered
core gate below.

### chk4

The new `cp440` and `cp460` runs replay the original chk4 selection panel:

- the same nine validation prompts, each with one greedy and one fixed sampled
  decode (18 completions);
- identical ordered cases, targets, seeds, generation modes and parameters,
  tokenizer, parent SFT model, and scoring implementation;
- the original selection eligibility threshold: delivery-valid at least
  0.75, cap rate at most 0.25, and no periodic tail.

The ordered case-key digest for `cp440`, `cp450`, and `cp460` is identical:
`fd18b9ff351086627d04aa9c41e1ddaf66d1216ab95c9b55693b9a385419b7f6`.
This post-hoc run did not read the test set. The sealed test had already been
opened historically on 2026-08-14, so this cannot be described as a new
preregistered or still-sealed selection.

## chk3 results

### Adapter-overlay checkpoint screen

| checkpoint | epoch | validation loss | core hard-valid | EOS | cap / periodic | numeric | date | mean full 4-gram | max tail 4-gram | result |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 170 | 1.608 | 1.128256 | 10/12 | 11/12 | 1 / 1 | 11/12 | 10/12 | 0.288701 | 0.996082 | reject |
| 200 | 1.893 | 1.122187 | 12/12 | 12/12 | 0 / 0 | 12/12 | 12/12 | 0.190638 | 0.384917 | advance |
| 230 | 2.171 | 1.121265 | 11/12 | 12/12 | 0 / 0 | 12/12 | 11/12 | 0.196301 | 0.461810 | reject |
| **280 (new)** | **2.646** | **1.121306** | **10/12** | **11/12** | **1 / 1** | **11/12** | **10/12** | **0.253836** | **0.944172** | **reject** |
| **300 (new)** | **2.836** | **1.121313** | **10/12** | **12/12** | **0 / 0** | **12/12** | **11/12** | **0.238718** | **0.514202** | **reject** |
| **318** | **3.000** | **1.121280** | **12/12** | **12/12** | **0 / 0** | **12/12** | **12/12** | **0.215215** | **0.350954** | **advance / selected** |

`cp280` has one capped, strict-periodic generation and a second date-fidelity
failure. `cp300` terminates every case, but one case crosses the full-output
4-gram repetition limit (`0.526316 >= 0.50`) and another omits required date
surfaces. Neither is an exact-merge finalist.

### Exact-merged deployment replay

Adapter screening and exact-merged replay are kept separate because
quantization and merge form can change boundary cases.

| checkpoint | validation loss | core hard-valid | EOS | cap / periodic | numeric | date | mean full 4-gram | max tail 4-gram |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 200 | 1.122187 | 11/12 | 12/12 | 0 / 0 | 12/12 | 11/12 | 0.150280 | 0.346863 |
| **250 (minimum loss)** | **1.121204** | **10/12** | **11/12** | **1 / 1** | **11/12** | **10/12** | **0.267695** | **0.947111** |
| **318 (retained)** | **1.121280** | **12/12** | **12/12** | **0 / 0** | **12/12** | **12/12** | **0.144111** | **0.335354** |

The `cp318` loss is only `+0.00007641` (`+0.0068%`) above the trajectory
minimum at `cp250`, while `cp250` has one capped periodic loop and additional
numeric/date failures. `cp318` is the only exact-merged finalist to pass all
12 core cases. On the non-identity N=11 semantic diagnostic, `cp318` also
improves BERTScore F1 (`0.975129` versus `0.968356`) and MPNet cosine
(`0.972733` versus `0.944044`) relative to chk1, although N=11 does not support
population inference.

**Interpretation:** the selection rule reasonably prioritizes deployment
validity over a negligible loss difference. The evidence supports `cp318` as
a defensible hard-gate-qualified checkpoint on this late plateau. It does not
show that step 318 is a statistically unique optimum.

## chk4 results

### Selection panel

| checkpoint | epoch | reward mean | direction | exact action | delivery-valid | EOS | cap | eligible |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 310 | 1.987 | 0.227083 | 0.277778 | 0.222222 | 0.444444 | 0.888889 | 0.111111 | no |
| 360 | 2.308 | 0.106944 | 0.222222 | 0.166667 | 0.388889 | 1.000000 | 0.000000 | no |
| 410 | 2.628 | 0.093750 | 0.111111 | 0.111111 | 0.555556 | 0.944444 | 0.055556 | no |
| **440 (new)** | **2.821** | **0.122917** | **0.166667** | **0.111111** | **0.388889** | **0.833333** | **0.166667** | **no** |
| **450** | **2.885** | **0.143750** | **0.222222** | **0.166667** | **0.333333** | **0.833333** | **0.166667** | **no** |
| **460 (new)** | **2.949** | **0.078819** | **0.111111** | **0.055556** | **0.333333** | **0.888889** | **0.111111** | **no** |
| 468 | 3.000 | 0.069444 | 0.055556 | 0.055556 | 0.333333 | 0.833333 | 0.166667 | no |

Every checkpoint has zero periodic-tail cases, but every checkpoint fails the
0.75 delivery-valid eligibility threshold. Within the new local neighbourhood,
the original ranking rule gives `cp450 > cp440 > cp460`; both neighbours also
trail `cp450` in reward, direction accuracy, and exact-action accuracy. Paired
mean reward differences are `cp450-cp440=+0.020833` (5 wins, 7 ties, 6 losses)
and `cp450-cp460=+0.064931` (6 wins, 7 ties, 5 losses). These tiny panels are
diagnostic, not significance tests.

### Why cp450 was the best observed candidate

In the original broad screen, `cp310` and `cp450` advanced to blind and
train-only retention checks. `cp450` had higher blind reward (`0.1725` versus
`0.0878`), blind direction accuracy (`0.30` versus `0.15`), and combined
validation reward (`0.158882` versus `0.153783`). It also produced at least one
correct direction for cut, hold, and hike on the train-only retention panel;
`cp310` had zero cut recall. This explains the stored
`best_observed_checkpoint_step=450`.

However, `cp450` blind delivery-valid is only `0.30`, below the preregistered
`0.75` threshold. The authoritative selection artifact therefore states:

```text
selected_checkpoint_step = null
status = provisional_no_candidate_passed_all_gates
best_observed_checkpoint_step = 450
```

The later sealed test also reports `verdict=quality_not_demonstrated`:
greedy N=13 direction/exact accuracy is `0.3077`, balanced accuracy is
`0.4444`, delivery-valid is `0`, and cut recall is `0`. Thus the neighbour
probes support only a local/best-observed checkpoint rationale; they do not
repair chk4's model-quality evidence.

## Limitations

- The chk3 N=12 set is a deterministic checkpoint diagnostic, not an
  inferential test; the reference Minutes are synthetic teacher targets.
- The chk3 result belongs to the standalone direct `chk1 -> chk3` lineage and
  is evaluation-only. It does not authorize canonical-DAG promotion or
  downstream training.
- The chk4 validation pool has 13 unique meetings (10 hike, 3 hold, no cut).
  The selection subset used here has nine prompts; cut evidence comes only
  from a train-only retention panel.
- `cp440` and `cp460` were not advanced to blind/retention because both were
  ineligible and worse than `cp450` on the selection panel.
- Both updates are single-seed, post-hoc diagnostics. Neither establishes
  global checkpoint optimality or population-level generalization.
- The runtime emitted the same known upstream tokenizer-regex warning as the
  historical probes. Vocabulary, chat-template, prompt, case, and generation
  hashes remained identical; changing tokenizer semantics mid-comparison
  would have broken comparability.

## Suggested paper wording

### chk3

> Within the standalone direct chk1-to-chk3 trajectory, checkpoint 318 was
> retained using deployment-oriented hard gates rather than validation loss
> alone. Although checkpoint 250 attained the minimum validation loss
> (1.121204 versus 1.121280 at checkpoint 318), its exact-merged N=12 replay
> contained a capped periodic loop and passed only 10/12 core cases, whereas
> checkpoint 318 passed 12/12. Retrospective late-neighbour probes at
> checkpoints 280 and 300 also passed only 10/12. We therefore treat
> checkpoint 318 as a defensible hard-gate-qualified representative of the
> late training plateau, not as a statistically unique optimum.

### chk4

> Checkpoint 450 was retained only as the best-observed exploratory candidate,
> not as a gate-selected optimum. It outperformed checkpoint 310 on the blind
> and retention tie-breakers, and retrospective local probes at checkpoints
> 440 and 460 did not surpass it. Nevertheless, no candidate met the
> preregistered delivery-valid threshold, the formal selected checkpoint is
> null, and the subsequent sealed test reported that quality was not
> demonstrated. Chk4 results are therefore reported as exploratory and are
> excluded from headline quality claims.

## Evidence and QA

- chk3 frozen N=12 manifest:
  `docs/summary/20260811T003000Z/chk3_native_analysis_to_minutes_cp250_eval/samples_n12.json`
- chk3 new probes:
  `output/evaluation/main/chk3_native_cp_neighbors_n12_20260818_v1/`
  - `cp280` manifest payload SHA:
    `401037f3242fb057752e0d1086900ec2684c8f146cb246923399fb774b740995`
  - `cp300` manifest payload SHA:
    `63285725c55ada5410f43e1539233fad763c7e8cf28f892f5ef89c0e6c11a1b0`
  - generation file SHAs: `fd3ec9079698959094558513d83d9bcfce595ae127ece5ebc6e0922ed18ee4b9`
    (`cp280`) and `dcdbde72ce6b07adba90847d8851a4ce462b597878d701337186f5815dd0ea26`
    (`cp300`)
- chk3 authoritative prior selection:
  `docs/summary/20260811T104604Z/chk3_checkpoint_selection_native_n12/selection_receipt.json`
- chk4 original broad probe and selection:
  `output/evaluation/retrain_v2/chk4_cp38_direct_grpo_checkpoint_probe_v2_batch8_20260814/`
- chk4 new neighbour probes:
  `output/evaluation/retrain_v2/chk4_cp450_neighbor_probe_cp440_cp460_20260818_v1/`
  - manifest SHA:
    `c7fc726c5697443e4112bdf584dfca9b54292ffea76e227875206ed9e4ada6f5`
  - result SHAs: `b98c3b9e2e5f1717d835c0a1bb3becbf0f3e8bb6aceb8d3946c310eeaf4b03bc`
    (`cp440`) and `1d782e31b8636378cc71a3a8eed46d55871f042f50bdfcd27bf0fa01bef8e337`
    (`cp460`)
- chk4 final sealed-test status:
  `output/evaluation/retrain_v2/chk4_cp450_final_test_v1_20260814/run/summary.json`
- Integrity QA: stored and independently recomputed hashes match; all result
  rows align with their frozen cases and contracts; chk4 completion parsing,
  reward replay, and aggregates were independently reproduced.
- Regression tests:
  `23 passed` across the chk3 native evaluator and chk3/chk4 probe test suites.
