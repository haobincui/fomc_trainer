# Chart map

## `eval_loss_curve`

- Question: where did validation loss stop improving, and which retained checkpoint achieved the minimum?
- Dataset: `eval_curve` from `analysis_snapshot.json`.
- Encoding: optimizer step on x; eval loss on y; one line for the completed run.
- Decision use: shows the plateau beginning around step 220 and the minimum at step 250.

## `repetition_comparison`

- Question: among probed checkpoints, which stable candidate adds the least repetition relative to chk1?
- Dataset: `repetition_compare` from `analysis_snapshot.json`.
- Encoding: artifact/checkpoint on x; mean token 4-gram repetition across the same three frozen cases on y.
- Decision use: checkpoint 250 is better than checkpoints 200 and 318 on this soft stability metric while also owning the lowest eval loss.
