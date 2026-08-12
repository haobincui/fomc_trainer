# chk2 DeepSeek Low / 600 秒启动状态

记录时间：2026-08-10 15:25:58 UTC

## 结果

chk2 已使用 `deepseek-v4-flash`、`reasoning.effort=low` 和 600 秒单次响应超时，从完整 checkpoint-4 恢复。checkpoint-5 已成功保存，训练继续进入 step 6。

旧 420 秒 run 在 step 5 的第二组连续两次出现 `_RetryableProviderResponseError` 后按设计 fail closed。旧脱敏日志无法区分 timeout、provider incomplete 或 schema-invalid，因此 600 秒不是对所有同类错误的保证；本次重新执行时，两组均在第一次尝试成功。

## Step 5 验证

| 请求 | Attempts | 延迟 | Input | Output | Reasoning |
|---:|---:|---:|---:|---:|---:|
| 1 | 1 | 10.53 s | 5,579 | 1,461 | 1,186 |
| 2 | 1 | 61.98 s | 4,890 | 9,598 | 8,354 |

8 个 reward 全部有限且在 `[0,1]`，step reward 为 `0.8375 ± 0.1048`。loss 为 `0.0382`，grad norm 为 `1.0079`，completion 截断率为 0，step 用时 191.49 秒。没有 OOM、NaN、context error、API 最终失败、候选分组错误或 traceback。

checkpoint-5 包含 adapter、optimizer、scheduler、RNG、trainer state 和 tokenizer，共 12 个文件，目录指纹为 `a8c362cc228af8762039d9522c313b62ac4fb6129e32644aad56c032ad8e5465`。

## 当前合同

- reward：`grounded_analysis_v3_deepseek_low_timeout600`
- model：`deepseek-v4-flash`
- effort：`low`
- timeout：600 秒/attempt
- 最多 2 次总尝试，backoff 2 秒
- Judge 输出预算：16,384 tokens
- policy：物理 GPU1、单进程
- GPU0 不启动本地 Qwen Judge
- 仍为 chk2 candidate，不自动进入 chk3/chk4

两次请求最坏可能让一个 logical group 等待约 1,202 秒；训练仍保持 fail closed，不会在同一 run 中静默改 timeout、reward 或 token 配置。

相关证据：

- `chk2_low_timeout600_transition_plan.md`
- `chk2_low420_failure_transition_receipt.json`
- `chk2_low600_resume_cp4_authorization.json`
- `chk2_low600_resume_cp4_launch_binding.json`
- `chk2_low600_resume_cp4_launch_receipt.json`
- `chk2_low600_launch_and_step5_receipt.json`
