# chk1 本地数据重建执行方案

## 范围与目标

本任务只处理 chk1 的训练数据，不修改 chk2–chk4、训练配置、模型权重、共享 checkpoint 路径或评估流水线。其他 session 仅通过不可变 `handoff.json` 消费本次发布。

目标数据契约为：

```text
point-in-time 数据 → FOMC 风格分析
```

硬约束：

- 不调用 DeepSeek API，也不读取 DeepSeek API key。
- 旧 DeepSeek 逐样本 target 不进入 canonical chk1。
- 复用本地 meeting/topic 元数据、既有 meeting-level split、指标映射、ALFRED cache、reference-free student prompt 和仅来自 train split 的 Minutes 风格统计。
- 远程 `deepseek-v4-pro` 直接生成 analysis/answer；获取阶段不调用本地 verifier、critic 或 reward model。
- 保留全部原始和既有 processed 文件；新产物只写入 `dataset/processed/retrain_v2/chk1/<release_id>/`。

## 实现步骤

### 1. 冻结旧数据并建立 inventory

- 枚举 chk1 相关 raw、processed、student prompt、teacher prompt/response、Minutes、指标源和 ALFRED cache。
- 记录相对路径、字节数、JSONL 行数和 SHA256，不修改源文件。
- 将 `dataset/processed_llama/**/analysis_sft` 在 inventory 中标记为 `legacy_privileged`，只允许复现和拒绝检测使用。

### 2. 固定 canonical 样本与 split

- 以现有 train/eval/test 的 3,890/535/462 候选为起点。
- 主键固定为 `meeting_date + atomic_topic`，生成稳定 `sample_id`。
- meeting-level 时间 split 保持既有 102/13/13；禁止随机重分。
- 每个 atomic topic 建立唯一 `section_style_id`。同一主键出现输入或元数据冲突时 fail closed，不任意择一。

### 3. 构建 point-in-time fact card

- 优先复用仓库内 meeting/topic/source 映射、原始指标文件、ALFRED D−1 snapshots 和 provenance 逻辑。
- 宏观观测必须证明 `release_ts <= cutoff_ts` 且对应 vintage 在 cutoff 时存在；市场值最多到会议开始前最后一个交易日收盘。
- 移除决议后 `current_rate`，只保留会议前已知的上次目标区间。
- 无 release/vintage 证明的观测进入 exclusions，不进入 canonical fact card。
- 缺失数据只允许从官方 ALFRED/FRED 补齐；该步骤与生成式模型完全隔离。
- fact card 仅保留最新值、3/6/12 月或季度变化、同比/环比、趋势/拐点/极值和有限近期观测；每项带 evidence ID、单位、公式、来源 SHA256、release/vintage/cutoff。
- tokenizer 后 prompt 上限 4,096 tokens；超限、空表、编码异常和异常 topic 均 fail closed，禁止截断。

### 4. 提炼无事实泄漏的风格指南

- 只读取 train split Minutes；eval/test Minutes 禁止进入提炼输入。
- 按 section 统计措辞、句式、证据顺序、风险与不确定性表达。
- 删除数字、日期、人物、机构/事件实体、具体会议结论和政策行动。
- 旧 DeepSeek response 最多用于聚合结构统计，绝不作为逐样本示例或 target。
- 风格指南版本化并记录 SHA256；每个样本只引用 `section_style_id`。

### 5. DeepSeek A 获取

- 启动前检测两张 A30；如存在其他计算进程则安全退出/等待，不终止进程。
- 远程 `deepseek-v4-pro` thinking API 生成 analysis/answer；本步骤不使用 GPU。
- `message.reasoning_content` 映射为 analysis，JSON `message.content.answer` 映射为 answer。
- teacher 输入严格限定为 `fact_card + atomic_topic + section_style_guide + output_contract`。
- 同会议 Minutes、旧 response/reasoning、决策标签和决议后数据不得进入 teacher prompt。
- DeepSeek `content` 必须是只有 `answer/evidence_ids` 的 JSON；SFT response 唯一转换为 `analysis\n</think>\nanswer`，拒绝控制 token。

### 6. 输出合同与单次格式修复

- 基础输出合同检查 analysis/answer 非空、evidence IDs 非空且唯一、analysis/answer 边界、字符编码和控制 token。
- 不运行确定性 candidate-content verifier；不再按数字、因果词、8-token overlap 或 analysis/answer 句子重复自动拒绝。
- 不运行 chk0 critic 或任何 reward model。
- DeepSeek 输出合同失败时进行一次格式修复请求，仍失败即记录排除。禁止旧 target 或其他 LLM fallback。
- 最终 response 必须通过学生 tokenizer 的 `3072+1024<=4096` 预算；禁止截断。

### 7. 不可变发布与 handoff

正式目录：

```text
dataset/processed/retrain_v2/chk1/<release_id>/
  sft/{train,eval,test}.jsonl
  manifests/{train,eval,test}.jsonl
  audit/exclusions.jsonl
  audit/quality_report.json
  handoff.json
```

- SFT 仅暴露 `prompt`、`response`、`provided_data`。
- manifest 保存 sample/split/cutoff/evidence lineage、模型/tokenizer/style/prompt 哈希、生成配置、各文本哈希与排除原因。
- `handoff.json` 保存 release/schema/split counts、split/manifest SHA256、template/style/model hashes 和 quality status。
- 发布文件使用内容哈希与原子 finalize；完成后视为不可变。其他 session 不读取工作缓存，只读取 handoff 指向的 release。

## 测试与发布门禁

- 20 条 smoke：覆盖正常、空表、超长、缺 vintage、数字推导和外部事实。
- 200 条按 topic/time 分层 pilot；通过后才允许全量生成。
- teacher prompt 中 `reference_excerpt`、旧 response、决策标签出现次数为 0。
- split 交叉、重复主键、冲突 target、point-in-time 违规和静默截断均为 0。
- train 接受率至少 70%，每个有效 topic 至少 50%。
- 200 条人工审核：critical unsupported claim 为 0，平均风格评分至少 4/5。
- 网络防护测试：DeepSeek hostname 访问和 DeepSeek API key 读取立即失败。
- 按 sample ID 与输入/模型/模板哈希幂等复用；仅重建哈希变化的样本。

## 执行边界

- 本任务止于 chk1 数据 release 与 handoff，不训练 chk1。
- 不修改 chk2–chk4、reward、checkpoint、共享训练配置或整体环境。
- 本地 Qwen3.5-9B 与 chk0 可用；DeepSeek API 禁用。
- 官方 ALFRED/FRED 是唯一允许的缺失 point-in-time 数据补充源。
- 任何无法证明正确性的样本都进入隔离区，质量门禁不得降级。
