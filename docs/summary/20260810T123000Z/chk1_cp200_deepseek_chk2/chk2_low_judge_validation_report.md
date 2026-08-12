# DeepSeek low-effort Judge 验证报告

## 结论：可进入版本化 chk2 试跑

`deepseek-v4-flash` 的 `effort=low` 在本轮固定基准中通过全部质量、稳定性、速度和隐私门槛。它保留了 high Judge 对候选的排序，并显著降低了 reasoning token 使用量和尾延迟，适合替代本次因 high reasoning 接近 16,384-token 上限而中断的在线 Judge。

## 方法与范围

- 数据截至：2026-08-10 UTC。
- 只改变 `reasoning.effort`：`high -> low`；模型、Responses API、system prompt、严格 JSON schema、四候选分组、`max_output_tokens=16,384`、timeout 和 reward 数学保持一致。
- 5 个 evidence group，每组运行原顺序和固定置换 `[2,0,3,1]`，共 10 个 logical requests、40 个候选判断。
- 覆盖 FFR 数字错误定位、上一轮真实 chk2 completion、TOTALSL billions/trillions 换算、因果过推、格式元话语、缺失 `</think>` 和目标会议泄漏。
- 8 个候选可与已保存的 high Judge 结果逐项比较；其余样本使用确定性注入错误和本地 reward 合同作为预期。

## 验证结果

- API：10/10 请求成功，全部首次成功；无 429、timeout、schema retry 或最终错误。
- 延迟：均值 `25.82s`，中位数 `26.69s`，P95 `43.83s`。
- 输出预算：平均 output `3,367.9` tokens，平均 reasoning `2,837.4` tokens；最大预算利用率 `31.81%`。
- 与 high 的 8 候选比较：reward MAE `0.0352`，最大绝对差 `0.1050`，pairwise 排序一致率 `100%`，weighted Judge score MAE `0.0453`。
- 置换稳定性：同候选 reward MAE `0.0241`，最大差 `0.1125`；penalty 完全一致率 `95%`。
- 安全语义：think/answer 重大错误定位、TOTALSL 等价尺度、错误尺度、因果过推、target leakage、缺边界本地归零均通过两轮门槛。
- GRPO 可用性：真实 completion 组标准差 `0.1764`，至少 3 个不同 reward；所有 10 个请求组均非 zero-std。
- 隐私：落盘产物不含 API key、原始 evidence、原始 candidate、可见 provider JSON 或隐藏 reasoning。

## 重要 caveat

置换后的 rubric cell 完全一致率为 `84%`，说明 low Judge 的细粒度 0–4 rubric 仍有位置/采样波动；但 penalty 一致率为 `95%`，reward 最大差仍在预设 `0.12` 门槛内。另一个限制是仅有一组真实训练 completion 可重放；10 次请求不能证明完整约 494 次请求绝不失败。因此结论只授权版本化恢复和前两个 optimizer steps 的严密监控，不代表自动提升 chk2 或进入 chk3/chk4。

## 可追溯产物

- 基准摘要：`chk2_low_judge_benchmark_v1/benchmark_summary.json`，SHA-256 `30172fd9d30579b0d3d8dd1d0ed27e651eaef9b75284b6167c53f89d5ea24794`
- 输入/合同 manifest：`chk2_low_judge_benchmark_v1/benchmark_manifest.json`，SHA-256 `4b126d5e783ac94dd1e7824ece1e02336fd2da17d46a4e0c8e65aad66475abc6`
- 脱敏逐候选记录：`chk2_low_judge_benchmark_v1/benchmark_records.jsonl`，SHA-256 `6c855da65a09beea19dd84a36da07aa4633d9aa8de23e11020575bcc90fc2f52`
- 可复现脚本：`jobs/retrain_v2/benchmark_chk2_deepseek_low_judge.py`，执行时 SHA-256 `bda9176ce5f75b0ffbcbbfd08dfa08dfe13e3f78d5d94648138e2c8a308dce6f`

验证评级：**Ready for a monitored two-step chk2 trial**，不授权自动 merge 或下游 promotion。
