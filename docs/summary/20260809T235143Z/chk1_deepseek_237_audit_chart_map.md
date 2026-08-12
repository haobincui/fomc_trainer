# Chart map

- `violation_kind_chart`: comparison bar; fields `kind`, `confirmed_violations`, with rows affected and severity counts retained for tooltip auditability. Takeaway: causal violations are the largest confirmed category. Palette policy: single-root blue, no redundant legend.
- `repair_group_chart`: comparison bar; fields `repair_group`, `failure_rate`, with group size, failed count and Wilson interval bounds retained. Takeaway: failure rates are similar across clean-v2 repair groups, so the deterministic repair is not an obvious failure driver. Palette policy: single-root blue, percent axis from zero.

Both charts are native report-artifact charts backed by reviewed snapshot rows. Exact counts remain available in the snapshot and supporting analysis JSON.
