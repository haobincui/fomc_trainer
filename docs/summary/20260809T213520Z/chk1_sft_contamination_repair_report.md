# chk1 SFT 数据污染与单 BOS 修复报告

## 当前状态

确定性修复、训练管线加固和真实 tokenizer 全量门禁均已完成。强制的本地 Qwen3.5-9B source-only 审核已结束，但未通过：237 条预期样本中完成 235 条，其中 1 条通过、234 条失败，另有 2 条 Judge 错误。

因此本次执行按预先约定的 fail-closed 规则停在候选数据：**没有发布正式 clean v2 release，没有生成 train-ready 配置，没有启动训练**。旧 v1 release、teacher cache、历史配置以及 chk1/chk2 模型均未被覆盖。

## 修复范围

源数据保持为 `chk1_reasoning_compressed_flash_max_v1_20260805`，新数据目标为 `chk1_reasoning_compressed_flash_max_v2_clean_20260809`。所有恢复均来自本地封存 release、generation manifest 和 v5 teacher cache，没有调用 DeepSeek API。

Answer 的 1,743 条处理路径为：

| 处理路径 | 行数 |
|---|---:|
| 完全保持原 answer | 1,542 |
| 删除内联 `ev-*` 引用 | 159 |
| 提取 nested `content.answer` | 4 |
| 从截断外层 content 解码完整 answer 字符串 | 24 |
| 从 cache reasoning 的完整 final-answer JSON 恢复 | 11 |
| 从明确标记的 final-answer 引文恢复 | 3 |

Reasoning 的修复为：37 条替换 `fact card` 元话语、3 条替换 `provided data`、1 条重写 `fixed final answer`、1 条删除复写完整 answer 的末尾段。Answer 与 reasoning 修复的样本有交集，最终变化样本并集为 237 条。

每行 repair manifest 以 canonical `sample_id`、prompt/provided-data hash 和 generation lineage 绑定，记录 old/new reasoning、answer、response SHA、修复方法及使用的 teacher cache 文件 SHA。候选 repair manifest 共 1,743 行，SHA256 为 `cd7dee86792c8190a5daceb99612c36a0396506f7dfa4da551a6f41cbb6c3d0b`。

## 管线修复

- DeepSeek teacher contract 升级为 v3（新 cache schema v4）：只接受 `finish_reason=stop`、非空 reasoning、严格且无额外字段的 `{answer,evidence_ids}`、纯文本 answer 和独立非空 evidence-ID 列表。返回的 ID 必须真实存在于当前 fact card，answer 不得内联 `ev-*`、`(E1)` 或当前 fact-card ID。新 cache 绑定 prompt、fact card、Minutes 输入、teacher contract、system prompt 和 provenance SHA，包含 payload digest 并以只读文件保存；旧 cache 不修改且不复用。失败响应最多重试 4 次，原始 reasoning/content 逐字保存到 rejected-attempt cache；耗尽后缓存 `status=rejected` 并持续 fail closed，不再把 raw content 回退为 answer。
- Compressed publisher 升级为 reasoning+answer 双审计合同：逐行绑定 canonical row/reasoning/answer SHA，拒绝 reasoning-only 旧审计、JSON-like/过短 answer、内联 evidence ID、schema/meta、Markdown、控制标签和跨 split 重复。任何语义失败都整体拒绝，不再删除失败行后发布；完整树 fsync 后封为只读，再使用 `RENAME_NOREPLACE` 发布。
- SFT Trainer 与 token auditor 现在共用 renderer：模板字符串移除且只移除一个 literal BOS，再由 TRL 添加一个 BOS；训练字符串 token IDs 必须与 `apply_chat_template(tokenize=True)` 完全一致。Completion 仍作为独立字符串，避免 DeepSeek assistant template 丢弃 `</think>` 前的 reasoning。
- 新增 clean-release manifest 验证：发布 CLI 必须同时验证且封存 deterministic token-validator 收据与 Qwen 收据；后者必须绑定 Qwen3.5-9B 权重健康证明、五项 rubric、Judge attempts/raw hash 以及每条 candidate/evidence hash。Trainer 启动时逐项复核 release manifest、split/repair/audit/validator 收据与通过状态，并精确加载 manifest 指定的 train/eval 文件；legacy glob 出现多个候选时 fail closed。训练时不重新运行 Qwen 全量审核。

## 数据与 token 门禁

`chk1_clean_sft_token_validation.json` 的全量结果为：

| 门禁 | 结果 |
|---|---:|
| train/eval/test | 1,354 / 199 / 190 |
| prompt hash 保持 | 1,743 / 1,743 |
| provided_data hash 保持 | 1,743 / 1,743 |
| JSON/evidence-ID/meta/control/Markdown 污染 | 0 |
| answer-in-reasoning 复写 | 0 |
| 跨 split exact/空白归一化重复 | 0 |
| 严格周期尾 | 0 |
| 单 BOS / 最终 EOS / completion mask 问题 | 0 |
| `max_length=4608` 截断 | 0 |

真实 DeepSeek tokenizer 的 token 分布：reasoning `517 / 996 / 1216 / 1600`，answer `19 / 114 / 187 / 279`，完整训练序列 `1418 / 2565 / 3435 / 4019`，依次为 min/p50/p95/max。验证收据 SHA256 为 `692b6b97f5dd8fa56bd563367d8ea86bde4be37a6c4890cf8d3fbcdc653b8f96`，tokenizer bundle 绑定 SHA256 为 `6c9b764ad58e4bb6a4daec4f942d5c295a7e805dae179cbc0745df8e3af048b2`。

## Qwen source-only 审核

审核结果为 `status=failed`：

| 项目 | 结果 |
|---|---:|
| 应审核的变化样本 | 237 |
| Judge 成功返回 | 235 |
| 通过 / 失败 | 1 / 234 |
| Judge 最终错误 | 2 |
| quote 校验后的 blocking violations | 611 |
| 无法通过 quote 校验、未计入惩罚的 violations | 228 |

611 条 blocking violation 全部位于 final answer，reasoning 为 0。类型分布是 factual major/minor=`490/111`，causal major/minor=`7/3`；numerical 和 target-leakage 均为 0。两个 Judge 错误都是连续 3 次返回无法解析的 JSON。

这批结果同时暴露了 Judge 校准问题：至少 117 条被标为 blocking 的 explanation 自身包含“与 evidence 一致”、“证据支持”、“数字正确”或“无事实错误”等肯定表述，但仍输出 factual violation。另一部分 answer 的确包含由数据无法直接支持的因果解释或趋势延伸。现有单 Judge 无法可靠地自动区分这两类情况；但按本计划的固定合同，所有通过 candidate-quote 子串校验的事实/因果 violation 都必须阻止发布，不能为了通过而二次过滤。

审核 summary SHA256 为 `780436e35db5690fc90cb68eedcc9c61653d67d8d2036dd68306d81907aee90e`；235 条 row audit 的 SHA256 为 `dec652ccbe2cef30c32134878e521869e3fa2c9096dd17efde8cd357c6f1baea`。

## Release 与训练配置

正式目标目录 `dataset/processed/retrain_v2/chk1_reasoning_compressed_flash_max_v2_clean_20260809` 不存在，与该 release 绑定的新 SFT 配置也未生成。这是语义门禁失败后的预期行为，而不是未完成的发布操作。

已构建的 candidate 仍保持 `quality_status=pending_semantic_audit`，目录和文件均为只读，不能被 Trainer 当作通过的 release 加载。旧 v1 release、teacher cache、历史配置、adapter 和 merged 模型均保持原样。

## 验证记录

- 最终整合回归（teacher/generation/publisher/repair/audit/validator/renderer/release/loader/workflow）：203 passed。
- Clean validator 包含真实 DeepSeek tokenizer 和全部 1,743 条 candidate 的 token/mask 检查。
- 相关 Python 文件 Ruff 和 `compileall` 检查通过。
- 生产 candidate 的 deterministic 收据已通过新发布门禁复核；已失败的 Qwen summary 被同一门禁以 `semantic audit did not pass` 明确拒绝。
- 本地 Qwen vLLM 审核服务已在生成失败收据后正常关闭，未留下训练或审核进程。

Clean 数据只有重新训练 chk1 后才会影响模型；本轮不会自动开始训练。

## 后续处理建议

下一轮需要先修改语义审核合同，用可校验的两阶段方式区分“Judge 自相矛盾”与真实的 unsupported/causal claim，然后对确认有问题的 answer 进行 source-only 改写并重跑 237 条审核。在新合同通过前，不应发布该 candidate 或用它重训 chk1。
