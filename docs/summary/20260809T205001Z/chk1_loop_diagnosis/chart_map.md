# Chart map

## prompt_scale

- Section: Old Minutes stress is outside the chk1 task.
- Question: How different is the rendered prompt scale from SFT/GRPO training?
- Family/type: grouped categorical bar.
- Fields: corpus, statistic, tokens; facts and task retained in tooltips.
- Takeaway: the stress input is about 8–10× longer and changes the task to multi-topic Minutes synthesis.
- Palette: two-root grouped comparison; native portable-report chart.

## checkpoint_comparison

- Section: Sampling does not rescue the old stress row.
- Question: Is the loop unique to chk1 or to temperature zero?
- Family/type: categorical bar.
- Fields: run label and word-trigram repetition; tokens, finish and temperature retained.
- Takeaway: all four controlled observations hit the cap without a boundary; chk1 at temperature 0.6 still repeats.
- Palette: single-root; native portable-report chart.

## prefix_repetition

- Section: The loop forms early.
- Question: At what output scale is repetition already dominant?
- Family/type: ordered categorical bar.
- Fields: output-prefix tokens and repetition percentage.
- Takeaway: 85.8% repetition at 512 tokens rules out insufficient output budget.
- Palette: single-root; native portable-report chart.
