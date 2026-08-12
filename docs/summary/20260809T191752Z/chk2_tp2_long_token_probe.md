# chk2 TP=2 / 52k-token 极限探针

## 结论

单纯扩大 token 上限不能解决旧 Minutes stress prompt 的终止问题。当前选定的 chk2 checkpoint-150 在最长 prompt 上使用两张 A30、单模型 tensor parallel 2，将输出上限从 8,192 提高到 52,000 后，仍完整耗尽上限，未生成 `</think>`、answer 或 EOS，且输出几乎全部为循环。

## 实验合同

- 模型：`output/training/retrain_v2/chk2_compressed_chk1_v1_reward_v3_selected_cp150_20260809/merged/chk2`
- 样本：`2025-07-30::participants_views`（旧 33-row 测试中唯一最长 prompt）
- prompt tokens：12,475
- tensor parallel：2（单一模型跨 GPU0/GPU1）
- dtype：bfloat16
- `max_model_len`：65,536
- `max_new_tokens`：52,000
- 请求总量：64,475 tokens
- context 余量：1,061 tokens
- decoding：`temperature=0.0, top_p=1.0, repetition_penalty=1.0`
- input truncation：false
- vLLM：0.8.5.post1；Torch：2.6.0+cu124；Transformers：4.52.3

`fomc_trainer` 环境没有 vLLM，且其 Torch 2.10 训练栈不能安全地原地安装旧 vLLM 0.8.5。因此数据预检和结果复核使用 `fomc_trainer`，GPU 推理复用此前共测所用的 `llama_factory` vLLM 环境，没有修改训练环境依赖。

## 结果

| 指标 | 结果 |
|---|---:|
| finish reason | `length` |
| output tokens | 52,000 / 52,000 |
| `</think>` | 0 |
| non-empty answer | 0 |
| full trigram repetition | 99.8600% |
| tail-2,048-word trigram repetition | 99.9022% |
| generation wall time | 863.52 s |
| generation throughput | 60.22 output tokens/s |
| end-to-end time（含初始化） | 955.94 s |
| 每卡显存 | 约 22,633 MiB |
| OOM / 非有限输出 | 0 |

## 循环轨迹

对完整输出用本地 chk0 tokenizer 重新分词，记录 token 数仍严格为 52,000。输出仅在前 58 tokens 有实质性规划，随后经历三段退化：

1. tokens 58–31,709：约 31,652 tokens 的 12-token 周期，核心句为 `labor market conditions index is 0.1 percent`，约 2,638 个周期。
2. tokens 31,721–45,038：约 13,318 tokens 的 9-token 周期，退化为 `labor market conditions index is 0`，约 1,480 个周期。
3. tokens 45,106–51,999：6,894 tokens 的 2-token 周期 `. The`，共 3,447 个周期。

在 2,048、4,096、8,192、16,384-token 截面均无 `</think>`；8,192-token 截面的 trigram repetition 已为 99.2010%。因此额外增加的 43,808 tokens 没有换来 answer，只让循环继续并进一步坍缩。

TP=2 与旧 TP=1 greedy 轨迹在第 25 个重编码 token 后分叉，说明跨卡数值差异改变了落入的具体循环，但没有改变失败类型。旧轨迹最终是 34-token 周期；新轨迹从 token 58 起即进入更强的 12-token 周期。

## 硬件与启动修复

两张 A30 的模型权重各占约 7.51 GiB，vLLM 为每卡提供约 198,128-token KV cache，65,536 context 的理论并发约 3.02，因此本次不是显存容量不足。

第一次启动在 NCCL 初始化阶段无进展，权重尚未加载、每卡仅约 0.4 GiB，已主动终止。独立 NCCL all-reduce smoke 通过后，第二次通过 `NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1` 强制本机 shared-memory 通道，随后 TP=2 权重加载、KV cache、预检和生成全部成功。

## 决策

- 不再继续提高 greedy 的 completion 上限，也不扩展到全部 33 条；最长样本已经直接否定“8,192 不够长”的假设。
- TP=2 可作为长上下文容量方案，但它不会自行修复重复或终止行为。
- 下一项最有信息量的实验应保持相同模型和样本，改为 `temperature=0.6, top_p=0.95`，并比较 `repetition_penalty=1.0/1.05`；诊断输出上限 2,304 足够，无需再次生成 52k。
- 正式 chk2 主评测仍应改回 task-aligned atomic evidence → analysis，而不是旧 raw-evidence → Minutes 合同。

## 工件

- 完整结果：`output/evaluation/diagnostics/chk2_tp2_long_tokens_20260809T185409Z/artifact_attempt2/result.json`
- 精简摘要：`output/evaluation/diagnostics/chk2_tp2_long_tokens_20260809T185409Z/artifact_attempt2/summary.json`
- 启动与预检：`output/evaluation/diagnostics/chk2_tp2_long_tokens_20260809T185409Z/artifact_attempt2/{launch,preflight}.json`
- 运行日志：`output/evaluation/diagnostics/chk2_tp2_long_tokens_20260809T185409Z/probe_attempt2.log`
- 可复用探针：`jobs/eval/run_tp2_long_generation_probe.py`
