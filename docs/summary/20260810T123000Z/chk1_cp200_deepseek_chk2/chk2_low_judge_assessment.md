# DeepSeek low-effort Judge 评估与 chk2 恢复训练准备

评估时间：2026-08-10 14:30 UTC

## 结论

`deepseek-v4-flash` 的 low reasoning effort 已通过本轮 chk2 trial 门槛，可以用于版本化试跑。该结论只授权 chk2 candidate，不授权自动合并模型，也不授权进入 chk3/chk4。

## Judge 测试结果

- 共执行 10 个真实 logical API requests，覆盖 40 个候选；10/10 均第一次成功，没有 retry。
- 固定参数为 `effort=low`、`max_output_tokens=16384`；所有请求都报告了非零 reasoning tokens。
- 延迟中位数为 26.69 秒，P95 为 43.83 秒；输出预算最大利用率为 31.81%。
- 与已有 high 基线的 8 个相同候选相比，reward MAE 为 0.0352，最大绝对差为 0.105，排序一致率为 1.0。
- 缺少 `</think>`、think/answer target leakage 均得到 0 reward。
- answer/think 错误定位、TOTALSL billions/trillions 等价换算、错误单位、因果错误、格式元话语均通过预设门槛。
- 原顺序与固定置换顺序之间的 reward MAE 为 0.0241，最大差为 0.1125，低于门槛。
- 未保存 API key、隐藏 reasoning、provider 原始输出、原始 evidence 或候选正文。

完整机器可读结果位于 `chk2_low_judge_benchmark_v1/benchmark_summary.json`，其 SHA-256 为 `30172fd9d30579b0d3d8dd1d0ed27e651eaef9b75284b6167c53f89d5ea24794`。

## 恢复训练设计

- 从 high run 的完整 `checkpoint-1` 恢复模型、optimizer、scheduler、RNG 和 trainer state。
- 恢复后使用 `grounded_analysis_v3_deepseek_low`；DeepSeek 预算仍为 16,384、timeout 420 秒、最多 2 次总尝试。
- policy、数据、seed、4096 completion 上限、4 generations、gradient accumulation 8 和 checkpoint 保留策略与 high 配置完全一致。
- 新输出目录为 `chk2_clean_v2_cp200_deepseek_low_totalsl_v1_resume_cp1_20260810`，必须 fresh；不会写回或覆盖 high run。
- 仅在物理 GPU1 上运行，world size 为 1，不启动本地 Qwen Judge，也不调用 canonical DAG。

需要注意：optimizer step 1 使用的是 high Judge，后续 step 使用 low Judge。因此这是版本化恢复 trial，不是 reward 全程同质的严格消融实验。启动后仍应观察前两个恢复 step 的 API 成功率、reward 分布、OOM、NaN 与 checkpoint 落盘情况。

