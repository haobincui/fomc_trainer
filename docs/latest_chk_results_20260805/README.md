# chk1–chk4 最新实验结果汇总

本目录收集了截至 2026-08-05 的 chk1–chk4 最新有效结果和审计材料。文件从 `docs/summary/` 原样复制，原目录中的历史记录仍然保留。

## 状态概览

| Checkpoint | 当前状态 | 结论边界 |
|---|---|---|
| chk1 | 训练完成并 sealed | 最新包含 SFT 结果、模型血缘、审计程序及离线 HTML 报告。 |
| chk2 | 训练运行快照 | 快照时进度为 175/247 optimizer steps，尚未 sealed，不是最终训练结果。 |
| chk3 | 数据 release complete / quality passed | 已完成 2,072 条 Minutes-SFT 数据发布，但 `dag_bindable=false`，尚未绑定 sealed chk2。 |
| chk4 | 核心数据完成；supplement 仅 dry-run | 核心 128/128 teacher targets 已通过；1993–2008 supplement 只完成无网络、无 API 请求的实现预检。 |

## 文件索引

### chk1

原始来源：`docs/summary/20260805T124140Z/`

- [README.md](chk1/README.md)：文档包说明和关键结论。
- [chk1_summary.html](chk1/chk1_summary.html)：最终离线技术报告。
- [artifact.json](chk1/artifact.json)：报告的规范化输入和图表数据。
- [chk1_audit.py](chk1/chk1_audit.py)：只读审计程序。
- [chk1_audit.ipynb](chk1/chk1_audit.ipynb)：已执行的审计 Notebook。
- [source_inventory.md](chk1/source_inventory.md)：报告证据和口径清单。

### chk2

- [chk2_technical_handoff.md](chk2/chk2_technical_handoff.md)：2026-08-05 12:41:52 UTC 的运行快照与验收交接。  
  原始来源：`docs/summary/20260805T124042Z/chk2_technical_handoff.md`

### chk3

- [chk3_training_release_result.md](chk3/chk3_training_release_result.md)：数据 release 整理结果。  
  原始来源：`docs/summary/20260805T111745Z/chk3_training_release_result.md`
- [chk3_training_data_technical_report.md](chk3/chk3_training_data_technical_report.md)：最新数据处理技术报告。  
  原始来源：`docs/summary/20260805T123700Z/chk3_training_data_technical_report.md`
- [chk3_regenerated_sample_inventory.md](chk3/chk3_regenerated_sample_inventory.md)：52 条重生成样本的逐条审计清单。  
  原始来源：`docs/summary/20260805T123700Z/chk3_regenerated_sample_inventory.md`

### chk4

- [chk4_teacher_target_failure_repair.md](chk4/chk4_teacher_target_failure_repair.md)：核心 128 条 teacher target 的修复和最终运行结果。  
  原始来源：`docs/summary/20260804T233805Z/chk4_teacher_target_failure_repair.md`
- [chk4_supplement_1993_2008_implementation.md](chk4/chk4_supplement_1993_2008_implementation.md)：1993–2008 supplement 的最新实现与 dry-run 状态。  
  原始来源：`docs/summary/20260805T113427Z/chk4_supplement_1993_2008_implementation.md`

## 选取口径

- “最新”按文档生成时间、完成状态以及是否被后续结果取代综合判断。
- 本包排除旧计划、旧实施记录、`.pyc` 和 `__pycache__`。
- 本包不复制数据集、模型权重或 `output/` 下的大型实验产物。
