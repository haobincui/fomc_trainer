# chk1 checkpoint-200 merge for chk2

Status: `ready_for_chk2_parent_under_explicit_override`

Merged model: `output/training/retrain_v2/chk1_clean_v2_lr1e6_selected_cp200_for_chk2_20260810/merged/chk1`

The model is an exact BF16 merge of `models/DeepSeek-R1-Distill-Llama-8B` plus `output/training/retrain_v2/chk1_clean_v2_override_full3ep_lr1e6_maxlen7168_20260810/adapters/chk1/checkpoint-200`.  The CPU tensor verifier checked all 291 model tensors: 224 adapted tensors and 67 unchanged tensors, with zero mismatches.  A 4-bit NF4 load and finite-logits forward smoke passed in the `fomc_trainer` environment.

The user explicitly authorized using checkpoint-200 as the chk2 parent.  The source SFT semantic audit remains `failed` (234 failed rows, 2 judge errors, 611 blocking violations); this promotion does not claim otherwise.  The override is limited to chk2 and does not authorize chk3 or chk4.

This historical nested checkpoint cannot honestly receive a retroactive canonical training receipt.  `promotion_manifest.json`, the in-model `merge_attestation.json`, the exact tensor proof, and `checkpoint_manifest.json` form the truthful lineage chain.

Key evidence:

- `chk1_cp200_to_chk2_authorization.json`
- `promotion_manifest.json`
- `merge_result.json`
- `chk1_cp200_exact_merge_lineage.json`
- `load_smoke.json`
- `checkpoint_manifest.json`
- `SHA256SUMS`

Known non-blocking warnings: Transformers reports a tokenizer regex advisory and a rope-scaling advisory.  Base/merged vocabulary, special-token IDs, four fixed text encodings, and chat-template token IDs are exactly equal under the actual training environment.
