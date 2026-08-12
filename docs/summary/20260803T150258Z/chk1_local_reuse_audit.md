# chk1 本地数据复用审计

审计日期：2026-08-03（UTC）

## 结论

本地旧数据可以复用的是 point-in-time 输入来源、meeting split、topic 映射、Minutes 风格统计和同样本泄漏检查语料；旧 `deepseek-reasoner` 回答不能作为 retrain-v2 的直接监督目标。

原因不是旧回答数量不足，而是它们不满足新合同中“teacher 只看 point-in-time 数据、topic 和风格指南”的因果隔离要求。因此不对旧 response 做“看起来合理就复用”的宽松筛选。

## 只读对账结果

| split | 旧成功 response | 与 teacher prompt hash 一致 | teacher prompt 含 Minutes reference | 含 current rate/rate change | 有固定 model revision/response ID | 有 evidence IDs |
|---|---:|---:|---:|---:|---:|---:|
| train | 3,883 | 3,883 | 3,883 | 3,883 | 0 | 0 |
| eval | 533 | 533 | 533 | 533 | 0 | 0 |
| test | 461 | 461 | 461 | 461 | 0 | 0 |

其他绑定事实：

- 旧 response 只记录可变 alias `deepseek-reasoner`，没有日期版本、model SHA、provider response ID 或生成时间，无法建立不可变 teacher 血缘。
- 旧 teacher prompt 与新 prepared prompt 不同；前者的 `reference_excerpt`、`current_rate` 和 `rate_change` 全部为非空。
- 新数据以 `meeting_date + atomic_topic` 为主键，而旧 response 以 legacy QA row 为单位；一个新样本可对应多个 legacy row，目标语义不是字节级等价。
- 旧 response 没有显式 evidence ID，无法通过 retrain-v2 的数字/趋势声明支撑审计。
- 当前 shell 环境没有 `DEEPSEEK_API_KEY`、`OPENAI_API_KEY` 或 `FOMC_REPORT_API_KEY`；不能在不新增外部凭证和 API 成本的情况下重现远程 `deepseek-reasoner` 生成。

## 执行决策

- 保留并继续哈希绑定新 session 已经复用的本地输入、split、topic 和风格资产。
- 旧 response 仅作为历史审计证据，不进入 SFT target。
- 当前无外部凭证的可执行方案是：使用固定本地 Qwen3.5-9B 生成两个候选，用确定性 evidence verifier 和固定 chk0 critic 双重验证，并对 200 条做人工审计。teacher 的完整本地模型目录、tokenizer、prompt 和生成配置都纳入 SHA-256 血缘。
- Qwen 同时担任 chk2 judge 会带来 teacher/judge 相关性；因此 chk1 必须保留独立 chk0 critic、确定性支撑检查和人工审计，chk2 验证不能只报 Qwen reward，还要报 unsupported-claim 和盲评指标。

如果后续使用者明确提供了可固定到不变 revision 的 DeepSeek teacher 和所需凭证，应使用新 release ID 重跑 smoke/pilot/full，不覆盖本地 Qwen 生成血缘。
