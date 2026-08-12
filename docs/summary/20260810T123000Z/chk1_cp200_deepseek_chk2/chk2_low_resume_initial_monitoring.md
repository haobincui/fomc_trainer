# chk2 DeepSeek Low 初始运行验证

记录时间：2026-08-10 14:59:53 UTC

## 结论

`deepseek-v4-flash` 的 `low` effort 已通过独立 Judge 水平测试，并通过 chk2 恢复训练的前两个 optimizer step 验证。4 次真实训练请求、16 个候选 reward 全部首试成功；checkpoint-2 和 checkpoint-3 完整落盘。训练保持运行，记录时已进入 step 4。

当前结果支持继续把 `low` 用作这个版本化 chk2 candidate run 的在线 Judge。它明显缓解了 `high` 模式接近 16,384 输出上限而失败的问题，同时保留了测试集上的错误排序能力。

## Judge 水平测试

独立 benchmark 使用 5 组样本及固定置换复测，共 10 次四候选请求、40 次判断：

- 10/10 请求均在第一次尝试完成；
- 延迟中位数 26.69 秒，P95 43.83 秒；
- 相对 high 的同候选 reward MAE 为 0.0352，最大绝对差为 0.105；
- 候选两两排序一致率为 100%；
- missing-boundary、target leakage、数值尺度错误和 answer/think 错误门禁全部通过；
- 置换复测仍有少量 rubric 离散差异，因此 low 适合在线训练，但不能视为逐字段完全确定的标注器。

详细结果见 `chk2_low_judge_benchmark_v1/benchmark_summary.json` 和 `chk2_low_judge_validation_report.md`。

## 训练态验证

恢复来源为 high run 的 checkpoint-1；step 2 起改用 `grounded_analysis_v3_deepseek_low`。两个已验证 step 的结果如下：

| Step | Judge 请求 | Reward mean ± std | Loss | Grad norm | 截断率 | Step time |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 2/2 首试成功 | 0.7790 ± 0.3087 | 0.2188 | 1.0726 | 0 | 251.10 s |
| 3 | 2/2 首试成功 | 0.4618 ± 0.2851 | 0.0007 | 1.1254 | 0 | 263.00 s |

四次 API 延迟为 46.54、33.95、77.21 和 40.85 秒。最大 provider 输出为 10,956 tokens，其中 reasoning 9,768 tokens，占 16,384 预算的 66.9%；没有出现 `incomplete`、超时或重试。

累计 16 个 reward 范围为 0.1496–0.9774，均为有限值；没有 target leakage、空结果、拒绝、零方差组或零梯度窗口。峰值 allocated/reserved 显存为 11.52/12.22 GiB。

step 3 的 loss 接近零不是零梯度：其 grad norm 为 1.1254，reward std 为 0.2851，运行安全日志中的 zero-gradient fraction 为 0。

## 可比性限制

本次是有意保留证据链的混合 Judge 试验：

- checkpoint-1 来自 `high` effort；
- checkpoint-2 及之后来自 `low` effort；
- 因此不能把这个 run 当作从 step 0 开始的纯 low/high 消融实验；
- 它可以作为当前 chk2 candidate 继续训练，但训练完成后仍需基于固定 validation prompts、截断率、重复率和 reward 分布选择 checkpoint；
- 不自动进入 chk3/chk4。

## 运行状态

- tmux：`fomc_chk2_cp200_deepseek_low_totalsl_v1_resume_cp1_20260810`
- GPU：物理 GPU1，单进程；GPU0 未加载本地 Qwen Judge
- 配置：`configs/retrain_v2/chk2_analysis_grpo_cp200_deepseek_low_totalsl_v1_resume_cp1_20260810.yaml`
- 输出：`output/training/retrain_v2/chk2_clean_v2_cp200_deepseek_low_totalsl_v1_resume_cp1_20260810`
- 初始验证收据：`chk2_low_resume_initial_monitoring_receipt.json`

启动阶段曾因另一进程占用绝大部分主机 RAM 而触发大量 swap，导致 trainer 初始化变慢；数据加载完成后训练正常推进，这不是 GPU OOM 或 DeepSeek API 故障。
