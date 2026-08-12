# chk1 SFT 数据污染与单 BOS 修复计划

## Summary

- 保留现有 v1 release、teacher cache、chk1/chk2 模型不变。
- 创建不可变 release `chk1_reasoning_compressed_flash_max_v2_clean_20260809`，保持 train/eval/test 为 `1354/199/190`。
- 全程复用本地封存数据，不调用 DeepSeek API：1,542 条保持原样，159 条移除内联 evidence ID，42 条 JSON-like answer 从 v5 teacher cache 恢复。
- 清理 reasoning 元话语及 answer 复写，并修复 SFT 训练器的双 BOS。
- 生成新的 train-ready chk1 配置，但本轮不启动训练。

## Data and interfaces

- 新增版本化修复 CLI，以 canonical `sample_id` 和哈希关联 compressed release、base manifest、generation manifest 与 teacher cache，禁止依赖发布后行号。
- 42 条 JSON-like answer 依次通过 nested `content.answer`、完整 answer JSON string、reasoning 中的 final-answer JSON、明确标记的 final-answer 引文恢复；只提取 final answer，不复制周围 teacher reasoning。
- 逐行 manifest 保存 stable ID、修复来源、cache/path SHA 以及 old/new reasoning、answer、response SHA。
- answer 必须是 16–512 个 chk0 tokens 的纯文本；reasoning 保持 512–2,400 tokens。
- 新 SFT 配置指向 clean release 和独立输出目录，并绑定 release manifest；旧 v1 配置仅用于历史复现。

## Pipeline fixes

- teacher 原始响应继续无损缓存，但只有 `finish_reason=stop`、严格 JSON schema、非空 reasoning、纯文本 answer、独立 evidence-ID 列表才可 accepted；其他情况重试并 fail closed。
- compressed publisher 同时验证 reasoning 和 answer，不再把 reasoning-only semantic audit 当作完整发布依据。
- Trainer 与 token auditor 共用 prompt renderer：移除模板字符串中已有的一个 literal BOS，让 TRL 只添加一个 BOS，并要求 token IDs 与 `apply_chat_template(tokenize=True)` 完全一致。
- 数据质量与本地 Qwen 审核只在不可变数据目录构建时执行；训练启动只验证封存 manifest 和哈希。

## Validation

- 维持 1,743 条及原 split 顺序，prompt/provided_data 哈希 100% 不变。
- 发布数据中 JSON-like answer、evidence ID、空/过短 answer、meta phrase、answer-in-reasoning 复写均为 0。
- 每条恰有一个 `\n</think>\n`，无 raw opening `<think>`、跨 split 重复或严格周期尾部。
- 对全部 237 条变化样本运行本地 Qwen3.5-9B source-only 审核；不得出现经过 quote 校验的 factual、numerical、causal 或 target-leakage violation，Judge 错误率为 0。
- 使用真实 DeepSeek tokenizer 验证单 BOS、单 EOS、completion label mask、`max_length=4608` 和零截断。
- 全部通过后使用 staging 与 `RENAME_NOREPLACE` 封存 release；任一门禁失败均不发布，也不自动改用 API 或排除样本。

## Assumptions

- 本轮不改变 reasoning 的 512–2,400 token 长度策略。
- 不覆盖现有数据、cache、adapter、merged model 或历史配置。
- clean 数据只有重新训练 chk1 后才会影响模型；本轮只完成数据、训练路径和配置准备。
