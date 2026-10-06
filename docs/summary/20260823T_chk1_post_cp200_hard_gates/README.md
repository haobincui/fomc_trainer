# Model chk-1 post-cp200 hard-gate replay

This directory records a retrospective matched replay of Model chk-1
checkpoints 210, 220, 230, 240, and 250. The comparison uses the original
eight-case sample manifest, decoding settings, Model chk-0 baseline, and
software versions used by the contemporaneous checkpoint-200 and
checkpoint-255 probes.

The intermediate checkpoints did not participate in the original selection
decision. The results are therefore descriptive stability diagnostics, not a
new prospective checkpoint-selection experiment.

| Checkpoint | Training / evaluation loss | Finite / EOS / contract-valid | Cap / catastrophic / periodic | Mean full-text 4-gram repetition | Delta vs chk-0, mean / maximum | Gate |
|---|---:|---:|---:|---:|---:|---|
| cp200 (retained) | 1.729560 / 1.686099 | 8 / 8 / 8 | 0 / 0 / 0 | 0.134664 | 0.051945 / 0.144158 | Passed |
| cp210 (retrospective) | 1.694860 / 1.686223 | 8 / 8 / 8 | 0 / 0 / 0 | 0.148501 | 0.065781 / 0.135519 | Passed |
| cp220 (retrospective) | 1.709530 / 1.686188 | 8 / 8 / 8 | 0 / 0 / 0 | 0.142607 | 0.059888 / 0.161819 | Passed |
| cp230 (retrospective) | 1.724720 / 1.686177 | 8 / 8 / 8 | 0 / 0 / 0 | 0.116562 | 0.033842 / 0.158795 | Passed |
| cp240 (retrospective) | 1.704090 / 1.686112 | 8 / 8 / 8 | 0 / 0 / 0 | 0.111440 | 0.028720 / 0.146266 | Passed |
| cp250 (retrospective) | 1.705740 / 1.686216 | 8 / 8 / 8 | 0 / 0 / 0 | 0.150136 | 0.067417 / 0.141370 | Passed |
| cp255 (terminal) | 1.722860 / 1.686131 | 8 / 8 / 8 | 0 / 0 / 0 | 0.140996 | 0.058276 / 0.171121 | Passed |

Training loss is the ten-step trailing mean ending at each checkpoint; the
evaluation loss is the logged checkpoint value, except that cp255 uses the
terminal evaluation loss from the archived run summary.

The frozen sample-manifest SHA-256 is
`3675e72983f6f235015b80460c1895f662bb8dbfe3fdd28bb1ac4634ecf65246`.
All probes used PyTorch 2.10.0+cu128 and Transformers 4.57.6. The comparator
thresholds are stored in each `*_vs_chk0_comparison.json` receipt.
