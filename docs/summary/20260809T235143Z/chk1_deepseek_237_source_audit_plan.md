# chk1 clean-v2：237 条 DeepSeek source-only 审核计划

生成时间：2026-08-09T23:51:43Z

## 目标与边界

- 使用 `fomc_trainer` conda 环境中已配置的 `DEEPSEEK_API_KEY`，调用固定模型 `deepseek-v4-flash` 审核 clean-v2 candidate 的全部 237 条变更样本。
- 审核对象为完整 reasoning 与 final answer；唯一事实权威为每条样本自己的 point-in-time `prompt` 和 `provided_data`。不发送 Minutes、teacher target、目标会议决策或外部参考答案。
- 本次结果写入新的 DeepSeek 审核目录，不覆盖旧 Qwen 审核，也不把 DeepSeek 结果伪装成本地 Qwen weight-attested 收据。
- API 使用 Responses 接口、`reasoning.effort=max`、`max_output_tokens=32768`。隐藏 reasoning 永不落盘，只记录 SHA-256 和 token usage。

## 固定输入

- candidate manifest：`output/data/retrain_v2/chk1/chk1_reasoning_compressed_flash_max_v2_clean_20260809_candidate/candidate_manifest.json`
- candidate manifest SHA-256：`41a5111875b052a44daec4bde5d622e25e19429158819aec43e1ec85d1945700`
- semantic audit input：`output/data/retrain_v2/chk1/chk1_reasoning_compressed_flash_max_v2_clean_20260809_candidate/audits/semantic_audit_input.jsonl`
- semantic audit input SHA-256：`34f63a7ac8b94a885f44851acbb5188c729aee8fcc5d90cbd2f8fdb592ecfdcb`
- 样本数：237；split 为 train/eval/test=`183/25/29`；237 个 `sample_id` 全部唯一；逐行 prompt/provided_data/candidate 哈希均已复算通过。

## 审核合同

- 将 raw target 按唯一的 `\n</think>\n` 分为 `think` 与 `answer`；DeepSeek 同时审查两部分。
- 输出五项 `0..4` rubric：data fidelity、trend reasoning、policy relevance、uncertainty calibration、FOMC style。
- 最多八条结构化 violation：section、kind、severity、candidate quote、explanation。
- quote 必须通过大小写和空白归一化后的 section 内原文子串校验；无法匹配的 violation 保留为 invalid，但不得进入 blocking 统计。
- 只有通过 quote 校验的 factual、numerical、causal、target leakage 属于 blocking；format、omission、style 永不作为事实错误。
- 对第一阶段提出的每条 blocking violation 再运行一次独立的 DeepSeek `try-to-disprove` 复核。只有复核为 `confirmed` 才计入最终 blocking；`false_positive` 完整留痕但不惩罚；`ambiguous` 或复核 API/schema 错误使该行进入 `review_required`，使整个 run 标为 incomplete，绝不能被误算为数据通过或失败。
- 第一阶段若给出“correctly / matches / consistent / supported”等明显肯定性理由，却仍将该 quote 标为事实错误，则视为 semantic-schema-invalid 并重试，避免复刻旧 Qwen 的自相矛盾误报。
- 每条请求独立缓存，并绑定 input SHA、sample ID、三项内容 SHA 与 request-contract SHA；可中断恢复。缓存和日志不得保存原始 prompt、provided data、candidate 或隐藏 reasoning。

## 执行与验收

1. 新增 DeepSeek 专用 audit CLI 和单元测试，保持现有 Qwen audit 代码与输出不变。
2. 先用相同正式合同跑 5 条 canary，验证 API、严格 JSON、max reasoning、usage、quote 校验、缓存和无敏感正文落盘。
3. canary 成功后通过 `--resume` 扩展到全部 237 条；并发不超过样本数，provider/schema/incomplete 错误最多重试 3 次。
4. 要求最终 completed=237、provider errors=0、所有数值有限、缓存合同完全一致；语义失败不会被隐藏，而是按 section/kind/severity 汇总并列出 hash-bound sample ID。
5. 生成机器可读 summary、逐行审计结果和技术审核报告。报告明确说明 DeepSeek 结论能否支持 clean-v2 数据继续用于 chk1，以及与旧 Qwen 结果的差异和局限。

## 已知限制

- DeepSeek 是外部 API，本轮会将这 237 条 point-in-time 输入与候选回答发送给服务方；用户已明确要求执行该外部审核。
- 当前正式 clean-release publisher 固定要求本地 `Qwen3.5-9B` health/weight attestation。即使 DeepSeek 全部通过，本轮也只产生独立审核证据；是否替换发布门禁需另行修改明确的发布合同。
