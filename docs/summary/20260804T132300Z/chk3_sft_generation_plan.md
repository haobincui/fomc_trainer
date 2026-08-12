# chk3 SFT 数据生成方案

## 目标与映射

chk3 学习任务固定为：

```text
chk1 final answer -> DeepSeek reasoning -> formal Minutes paragraph
```

DeepSeek V4 Pro 的 `message.reasoning_content` 映射为监督 reasoning，
`message.content.answer` 映射为正式 Minutes 段落。由于 DeepSeek tokenizer 会在
assistant 开头自动加入 `<think>\n`，数据中的 completion 固定为：

```text
reasoning_content
</think>
Minutes paragraph
```

不额外写入 `<think>`，也不使用 `<answer>` 标签。

## 输入数据

- 唯一输入是 canonical chk1 release
  `chk1_full_v7_automated_v2_20260804`。
- 保持 train/eval/test `1683/199/190`、sample ID 和原始顺序不变。
- 从 chk1 `response` 的唯一 `</think>` 后提取 final answer。
- teacher/student user prompt 只包含该 final answer 及固定改写指令。
- 不读取 chk2，不发送 fact card、provided_data、chk1 reasoning、原始 Minutes、
  decision 或 vote。

## 生成合同

- teacher 固定为 `deepseek-v4-pro`，thinking enabled、reasoning effort high、
  JSON output、`max_tokens=4096`、默认并发 8。
- content 必须是只有 `answer` 的 JSON object；reasoning_content 与 answer 都必须
  非空且不含模型控制标签。
- 最终 completion 恰好包含一个 `</think>`，使用本地 DeepSeek tokenizer 验证完整
  prompt+completion 不超过 4096 tokens，禁止截断。
- Minutes 中的数字和日期不得超出输入 analysis；格式失败允许一次同模型修复。
- chk1 answer 中的 `ev-...` 内部证据 ID 可作为输入，但不得出现在正式 Minutes target；
  数字门禁在比较前忽略这些 citation token。
- 缓存绑定 sample ID、输入/prompt/合同/代码哈希以及 provider response provenance，
  支持不可变断点续跑，不允许模型 fallback 或静默丢行。

## 产物与验收

新增 `jobs/generation/generate_chk3_sft_targets.py`，支持 `--dry-run`、全量生成和
`--resume`。产物写到 `output/data/retrain_v2/chk3/deepseek_v4_pro_v1/` 下的
prompt contract、prepared、teacher responses、SFT splits、manifests、cache、failure
和 summary 文件。实现使用 mock backend 覆盖映射、全量 split、缓存/恢复、格式修复、
模型漂移、数字日期和 token budget；实现与测试期间不发送真实 DeepSeek 请求。
