# Source and QA notes

- Decision: use checkpoint-200 as the chk1 input to chk2.
- Execution blocker: the current semantic override authorization is chk1-only with `downstream_stages_allowed=[]`; semantic audit remains failed and canonical promotion/merge lineage is absent.
- Audience: technical.
- Delivery mode: portable HTML because no MCP report renderer is callable in this runtime.
- Loss source: `output/training/retrain_v2/chk1_clean_v2_override_full3ep_lr1e6_maxlen7168_20260810/adapters/chk1/trainer_state.json` (SHA256 `ea7143be36657dfff23e5871ee642f4df166af428a86c433d5665575fded45ca`).
- Final metrics source: `output/training/retrain_v2/chk1_clean_v2_override_full3ep_lr1e6_maxlen7168_20260810/adapters/chk1/eval_results.json` (SHA256 `d18cf2dd93426aa535ff837a676f9b89b8ba5d9dcc7cf6f611e27fc9411de5a9`).
- Probe sample manifest SHA256: `3675e72983f6f235015b80460c1895f662bb8dbfe3fdd28bb1ac4634ecf65246`.
- Frozen chk0 baseline results SHA256: `26063038ea3258fb17ba7500a3af321389a6b39a394694a15c7c982229d72391`.
- Chart map:
  - Loss section: multi-series line; x=step, y=loss, series=train trailing-10/eval; supports convergence-platform claim.
  - Behavior section: category bar; x=model checkpoint, y=mean 4-gram repetition; supports soft-risk comparison.
- Palette: shared reader defaults; line uses meaningful series encoding, bar uses no redundant category legend.
- Required technical report sections are all present. Validation details are included under methodology and limitations.
- No chart was omitted. The final HTML must be packaged by the shared portable builder and its receipt retained.
