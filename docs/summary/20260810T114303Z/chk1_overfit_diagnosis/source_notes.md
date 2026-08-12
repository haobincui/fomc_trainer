# Source notes

- Audience: technical.
- Snapshot cutoff: checkpoint-210, generated 2026-08-10T11:43:03Z.
- Primary question: whether train and eval loss show sustained divergence consistent with overfitting.
- Chart contract: Trend / highlighted multi-series line; 21 evaluation points; x=optimizer step, y=loss, series=train trailing-10 mean vs full eval loss; hard two-root palette; full-width report block.
- Train/eval absolute gaps are not treated as calibrated generalization gaps because dropout and evaluation mode differ.
- Robustness evidence: fixed 8-case degeneration probes at checkpoints 80, 150, and 200.
- Omitted richer test-loss comparison: no final independent test evaluation exists at this training snapshot.

