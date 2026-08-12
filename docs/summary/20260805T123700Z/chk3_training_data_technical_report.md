# chk3 Minutes-SFT 训练数据处理技术报告

生成时间：2026-08-05 12:37:00 UTC  
报告对象：`chk3_minutes_clean_v1_20260805`  
目标读者：训练流程维护者、数据审计者和后续实验复现者

## 技术总结

chk3 数据已经整理为一份独立、不可变、可审计的 Minutes-SFT release。最终数据严格实现：

```text
chk1 final analysis -> native reasoning -> formal FOMC Minutes paragraph
```

最终 release 包含 2,072 条样本：train 1,683、validation 199、test 190。训练文件每行只
包含 `prompt` 和 `response`，sample ID、split、来源哈希、DeepSeek identity、token 统计和
处理模式全部保存在独立 manifest 中，不进入模型输入。

处理过程中确认原 acquisition 的 2,072 条 target 虽然全部处于 accepted 状态，但输入中
仍有两类不适合直接训练的 transport 噪声：50 条 analysis 以 `{` 开头，其中 46 条是截断
JSON；另外原始 analysis 中有 225 条包含 `ev-` 内部 evidence ID。最终处理结果如下：

| 处理结果 | 样本数 | 占总样本比例 | 最终处理 |
|---|---:|---:|---|
| Analysis 原样保留 | 1,839 | 88.75% | 原 target 重新校验后复用 |
| 移除 `ev-` citation | 183 | 8.83% | 180 条复用 target，3 条重生成 |
| 完整 JSON envelope | 4 | 0.19% | 无损提取 `answer`；1 条复用 target，3 条重生成 |
| 截断 JSON envelope | 46 | 2.22% | 从 point-in-time fact card 重建 analysis，并重生成 target |
| **合计** | **2,072** | **100%** | 无删除、无截断 |

最终 2,020 条 target 在清洗后的 analysis 上重新通过当前 validator，52 条使用
`deepseek-v4-pro` 重生成。所有 2,072 条最终 target 的 returned model 和 fingerprint 完全
一致：

```text
model: deepseek-v4-pro
system_fingerprint: fp_9954b31ca7_prod0820_fp8_kvcache_20260402
```

独立全量重放结果为 2,072/2,072 通过；transport wrapper、analysis evidence citation、重复
ID、空字段和 response boundary 错误均为 0。最大完整训练序列为 3,029 tokens，低于
4,096 上限。

数据层面已经可训练，但当前 release 明确标记为 `dag_bindable=false`：当前 run 还没有
sealed chk2 parent，也没有与 chk4 `decision_grpo` 共同组成完整 chk2-derived release。不能
为了提前启动 chk3 而伪造这两个依赖。

## 最终产物与数据定义

### Release 位置

```text
dataset/processed/retrain_v2/chk3_minutes_clean_v1_20260805/
  handoff.json
  release_manifest.json
  chk3_minutes_sft.template.yaml
  audits/
    data_quality.json
  minutes_alignment/
    train.jsonl
    validation.jsonl
    test.jsonl
    manifests/
      train.jsonl
      validation.jsonl
      test.jsonl
```

### 样本 grain 和 split

- 一行代表一个 chk1 canonical analysis sample。
- sample universe 由 immutable chk1 canonical release 决定。
- chk1 的 `eval` 在训练 release 中重命名为 `validation`；sample ID 和顺序保持不变。
- sample ID 全局唯一；没有新增、删除或跨 split 移动。
- train+validation 共 1,882 条，供 Trainer 训练和评估；test 190 条作为独立 holdout。

| Split | 样本数 | Manifest 样本数 | 用途 |
|---|---:|---:|---|
| train | 1,683 | 1,683 | SFT 训练 |
| validation | 199 | 199 | 训练期 evaluation |
| test | 190 | 190 | 独立 holdout |
| **合计** | **2,072** | **2,072** |  |

### 模型输入和监督目标

最终训练行 schema 固定为：

```json
{
  "prompt": "<rewrite instruction plus JSON-escaped analysis>",
  "response": "<reasoning>\n</think>\n<formal Minutes paragraph>"
}
```

每条 user prompt 为：

```text
Rewrite the following analysis as formal FOMC Minutes prose:

{"analysis":"<clean chk1 final analysis>"}
```

这里的 JSON 只作为 user prompt 的结构化边界；`analysis` 内部不再包含 provider JSON
envelope。student 不接收 `provided_data`、fact card、原始指标、chk1 reasoning、原始
Minutes、vote 或 policy decision。

response 不存储开头 `<think>`，因为 DeepSeek tokenizer 的 generation prompt 会自动以：

```text
<think>
```

结尾。完整渲染结果为：

```text
<think>
reasoning
</think>
Minutes paragraph
```

最终 `response` 恰好包含一个 `</think>`，不使用 `<answer>` 或 `</answer>`。

### Analysis 和 target 处理模式

Manifest 中的 `analysis_mode` 含义：

| 值 | 含义 |
|---|---|
| `unchanged` | chk1 final answer 已是干净 prose，原样保留 |
| `evidence_citation_projection` | 仅移除 `ev-<hex>` 内部 citation，并清理空括号/空白 |
| `valid_transport_projection` | 输入是完整 JSON envelope，无损提取内部 `answer` |
| `deepseek_point_in_time_recovery` | 输入 JSON 已截断，使用原 chk1 point-in-time source 重建 |

Manifest 中的 `target_mode` 含义：

| 值 | 含义 |
|---|---|
| `acquisition_revalidated` | analysis 未变化，原 provider response 全量重新通过 validator |
| `acquisition_reprojected` | analysis 被确定性清理，原 provider response 在新输入上重新通过 |
| `deepseek_regenerated` | 原 response 在新输入上不能安全复用，重新请求 DeepSeek |

最终模式组合如下：

| Analysis mode / Target mode | Train | Validation | Test | 合计 |
|---|---:|---:|---:|---:|
| unchanged / acquisition_revalidated | 1,494 | 179 | 166 | 1,839 |
| evidence citation / acquisition_reprojected | 149 | 16 | 15 | 180 |
| evidence citation / DeepSeek regenerated | 1 | 0 | 2 | 3 |
| valid transport / acquisition_reprojected | 0 | 0 | 1 | 1 |
| valid transport / DeepSeek regenerated | 3 | 0 | 0 | 3 |
| point-in-time recovery / DeepSeek regenerated | 36 | 4 | 6 | 46 |
| **合计** | **1,683** | **199** | **190** | **2,072** |

## 原始 acquisition 与问题定位

### 上游绑定

原 chk3 acquisition 位于：

```text
output/data/retrain_v2/chk3/deepseek_v4_pro_v2
```

它绑定的 chk1 handoff 为：

```text
output/data/retrain_v2/chk1/canonical_releases/
  chk1_full_v7_automated_v2_20260804/handoff.json
```

关键绑定：

| 项目 | 值 |
|---|---|
| chk1 release | `chk1_full_v7_automated_v2_20260804` |
| chk1 handoff SHA256 | `7eb1ceb8e3ea41b6305e40f15d4d6492ced64d45de370118a2f52ddf121a5448` |
| chk3 acquisition generator SHA256 | `20234ee7942449e43bf9530c388b344cf3c3da6b4f99d86d23945810411d7187` |
| prompt contract canonical SHA256 | `b1edfd69eb3163ea254a02e5bf0acfd25cb0ea015aa06aa2fbc8088ad146e0fe` |
| acquisition accepted | 2,072 |
| acquisition failures/pending | 0 |

### 50 条 JSON transport 污染

原 prepared analysis 中有 50 条以 `{` 开头：train 39、eval 4、test 7。

- 4 条是可以完整 `json.loads` 的 envelope：train 3、test 1。
- 46 条是 invalid/truncated JSON：train 36、eval 4、test 6。
- 最严重的一条 analysis 只有单个字符 `{`。

截断 envelope 不能只保留“最后一个句号前”的残片：这会造成部分数字、日期和方向关系
永久丢失，也会把不完整 analysis 当作正确训练输入。因此最终方案没有继续复用这些残片。

### `ev-` evidence ID 污染

原始 2,072 条 analysis 中有 225 条包含 `ev-<hex>`：train 183、eval 20、test 22。
其中 42 条同时属于 JSON envelope，所以单独执行 citation projection 的是 183 条。

这些 ID 是 chk1 内部引用，不是经济数量。短 ID（例如 `ev-5591`）暴露了旧 validator 的
边界问题：原 `_EVIDENCE_ID_RE` 只移除至少 6 位 hex 的 ID，数值解析器会把 `5591` 错当
作正文经济数量。teacher 在 Minutes 中正确省略 ID 后，旧门禁会错误报告
`missing_numbers:5591`。

最终 release 在进入 chk3 validator 前统一移除任意长度的 `ev-[0-9a-f]+`；最终 analysis
中的 evidence ID 数量为 0。事实正文、数值、日期和不确定性表达保持不变。

## 处理方法

### 1. Fail-closed acquisition 加载

Materializer 首先要求：

- `summary.status=complete`；
- `total_accepted=2072`；
- `failures.jsonl` 为空；
- prepared、teacher response、SFT 和 manifest 四类文件逐行等长；
- sample ID、split、source index、prompt 和 response SHA256 一致；
- 全局 sample ID 唯一。

任一条件失败时停止，不发布部分 release。

### 2. 确定性 analysis 分类和投影

处理顺序为：

1. 非 JSON analysis：检查并移除 `ev-` citation。
2. 完整 JSON：解析顶层 `answer`，或 `content.answer` / JSON-string `content.answer`。
3. 无法完整解析的 JSON：标记为需要 point-in-time recovery。
4. 投影后强制非空、长度至少 40 字符、不以 `{`/`[` 开头、无模型控制标签。

有效 JSON 的投影只选择已经存在的完整字符串，不发明 prose。citation projection 只删除
内部 ID、空括号和由删除产生的空白/标点噪声。

### 3. 46 条截断 analysis 的 point-in-time recovery

每条 recovery 只读取该 sample 在 immutable chk1 preparation bundle 中的：

- `generator_prompt`；
- point-in-time `fact_card`；
- `atomic_topic`；
- section style guide。

不读取 same-meeting Minutes、chk1 旧 reasoning、chk2 输出、vote 或 policy decision。

Recovery 使用：

```text
model: deepseek-v4-pro
thinking: enabled
reasoning_effort: high
response_format: json_object
max_tokens: 4096
default concurrency: 8
credential env: DEEPSEEK_API_KEY
fallback model: none
```

Recovery prompt 在原 chk1 generator contract 上增加以下限制：

- reasoning_content 优先控制在 100–300 words，保证 structured content 不被截断；
- answer 只能使用 fact card 中逐字出现的数字表面值；
-禁止四舍五入、近似、rescale、差值、求和、比例或增长率计算；
- 可以少写次要数值，但不能新增 fact-card 外数值；
- answer 必须是一个简洁分析段落，并返回对应 evidence IDs。

### 4. Recovery grounding verifier

Recovery answer 通过以下门禁：

- finish reason 必须为 `stop` 或 `end_turn`；
- answer 非空、不是 transport wrapper、无控制标签；
- answer 长度至少 40 字符；
- provider evidence ID 不得指向 fact card 外部；
- 数字必须来自完整 immutable fact card；
- 不允许未经授权的原因、事件、人物或政策行为；
- final analysis 不超过 1,024 tokenizer tokens。

旧 chk1 verifier 主要索引 evidence `value` 字段，但日期和 metric label 也可能合法包含数值，
例如 meeting year `2013` 或 `10-Year Treasury`。Materializer 因而增加了一个严格表面值门禁：
只要该数字以独立、完全相同的表面形式出现在 fact card 任意字段，就允许；仍不允许：

```text
96.9476 -> 96.95
```

或根据原值新算：

```text
1.6 percent decline
```

这一区分修复了日期/期限的 false positive，同时继续拒绝近似和派生数量。

### 5. Target reuse 优先，必要时才重生成

对于每条 clean analysis，先把原 acquisition 的 provider raw response 重新构造为
`DeepSeekTeacherResponse`，然后使用当前 validator 和新的 student prompt 全量重放。

只有以下情况才重新请求 target：

- analysis 被 point-in-time recovery 重建；
- JSON/citation 投影改变了 numeric/date contract，原 target 不再通过；
- 原 reasoning 经当前 sanitizer 后不满足格式或 token 契约。

最终需要重生成 52 条：train 40、validation 4、test 8。其中 46 条对应 recovery analysis，
其余 6 条为 3 条 valid-transport projection 和 3 条 evidence-citation projection。

Target teacher 使用 inventory-driven prompt：

- 逐 occurrence 列出必须保留的 quantity；
- 列出必须保留的日期；
- reasoning 目标 100–350 words；
- answer 建议不超过 500 words；
- repair diagnostics 必须静默应用，不得在 reasoning 中复述错误码。

### 6. Provider identity 和缓存

所有新 API response 必须满足：

```text
returned_model == deepseek-v4-pro
system_fingerprint == fp_9954b31ca7_prod0820_fp8_kvcache_20260402
```

出现 model 或 fingerprint drift 时整批失败，不混用 provider identity。

Accepted cache 绑定：

- sample ID、split；
- analysis SHA256；
- prompt SHA256；
- teacher contract SHA256；
- system prompt SHA256；
- response ID、returned model、fingerprint；
- provider raw reasoning/content；
- validator 结果。

缓存是 immutable、content-addressed，并支持中断恢复。rejected response 也保留，不能覆盖
accepted cache，也不会进入 release。

工程过程中依次产生 v1–v4 cache schema，用于修复以下问题：

| 版本 | 发现的问题 | 修复 |
|---|---|---|
| v1 | 只按 teacher 返回的 evidence ID 子集核验，漏列 citation 会误拒绝 | 使用完整 fact-card universe |
| v2 | 旧 verifier 仍只认识 evidence value | 强化 source recovery 约束 |
| v3 | 日期年份和 metric 期限数字被误判 | 增加 fact card 全字段精确表面值检查 |
| v4 | 短 `ev-xxxx` 被当作数量 | 在 student analysis 中统一移除所有 `ev-` citation |

工作缓存目前保留 196 个互不重复的 provider response ID，包含工程迭代的 accepted/rejected
响应。最终 release 只选择 v4 当前绑定下的有效记录：46 个 source recovery response 和
52 个新 target response。v4 target cache 有 53 个 accepted entry，其中 1 个绑定旧 analysis
hash，因最终 citation projection 改变而没有进入 release。

最后一次 CLI summary 中的 `analysis_api_calls=0` 和 `target_api_calls=4` 只表示最后一次
resume invocation 的 cache miss 数量，不是整个工程期间的累计 API 请求数。

### 7. 原子 release 发布

只有在全部 2,072 条完成后才会创建 temporary staging directory。全部训练文件、manifest、
audit 和 config template 写完并校验后，通过单次 `os.replace` 发布到最终 release path。

如果目标 release 已存在且 handoff 不是 immutable/passed，脚本拒绝覆盖；如果已存在且合法，
返回现有 handoff。不会修改原 acquisition 或 chk1 canonical release。

## Validator 与质量门禁

### Response 格式

- response 恰好一个 `</think>`；
- reasoning 和 Minutes 均非空；
- 两部分不能包含 `<think>`、`<answer>`、`</answer>` 或其他控制标记；
- Minutes 必须恰好一个 paragraph；
- Minutes 不得出现 `ev-` evidence ID；
- tokenizer chat template 必须以 `<think>\n` 结束。

### 数量语义

analysis 和 Minutes 的数量使用 `Decimal` 规范化后做 Counter 双向比较。允许可证明精确等价
的单位/形式变换，例如：

```text
510 thousand <-> 510,000
1,256,000 <-> 1.256 million
0.30 percent <-> 30 basis points
5.25 percent <-> 5-1/4 percent
```

不使用 tolerance；不允许近似、四舍五入、数量级错误或根据输入重新计算新数字。数量
occurrence 次数也必须一致。

### 日期和月份

- 月份比较大小写敏感；
- `May` 作为月份，`may` 作为情态动词；
- 输入和 Minutes 的月份/日期集合必须一致；
- calendar year 作为日期检查，不作为经济数量 Counter；
- month-day 表达整体处理，避免 day 被重复计为数量。

### Attribution

Minutes 只有在 analysis 已包含相应类别时才能使用：

- staff；
- participants / members / officials / policymakers；
- Committee / FOMC / Board / Federal Reserve；
- meeting / discussion / deliberation；
- vote / decision / policy action。

### Reasoning 元话语

reasoning 只能讨论原文事实、关系、保真检查和正式措辞。禁止：

- JSON、API、schema、key/field、response format、output contract；
- validation error、prompt/instruction、teacher/student、tool；
- “we are asked”“final answer”等答题元话语；
- 完整复制 analysis；
- 多次起草最终 paragraph。

reasoning 最少 64 tokens、最多 2,400 tokens。

## Token 统计和双 A30 训练约束

最终 release 使用本地 `models/DeepSeek-R1-Distill-Llama-8B` tokenizer，并按真实 chat
template、generation prompt、response 和 EOS 计算。禁止截断。

### 全局 token 分布

| 部分 | Min | P50 | P95 | P99 | Max | Admission limit |
|---|---:|---:|---:|---:|---:|---:|
| Prompt | 327 | 429 | 515 | 563 | 615 | 3,072 |
| Completion | 115 | 829 | 1,880 | 2,303 | 2,522 | 无独立上限 |
| Reasoning | 80 | 696 | 1,709 | 2,123 | 2,376 | 2,400 |
| Full sequence | 454 | 1,268 | 2,350 | 2,757 | 3,029 | 4,096 |

### 按 split 的 min/max

| Split | Prompt | Completion | Reasoning | Total |
|---|---:|---:|---:|---:|
| train | 329–615 | 157–2,522 | 103–2,376 | 489–3,029 |
| validation | 327–576 | 215–2,467 | 154–2,289 | 562–2,946 |
| test | 335–593 | 115–2,344 | 80–2,248 | 454–2,753 |

旧 DAG 对 chk3 设置了独立 completion<=1,024。最终数据有 712 条超过该旧上限，占
34.36%，但全部低于 4,096 总长度。直接把 reasoning 截到 1,024 会破坏监督目标和
completion-only loss 边界，因此没有做截断或删除。

为保护现有 run，默认 `configs/retrain_v2/dag.yaml` 已恢复为 v9 固定 SHA256：

```text
a07b9e7ad274b91f13171c2abbcecb4181e9c13f5e1d5b247177353c999145de
```

后续新 run 应使用：

```text
configs/retrain_v2/dag_chk3_full_completion.yaml
```

该 DAG 的 chk3 admission 为 prompt<=3,072、completion 无独立上限、total<=4,096，仍然
fail closed。它的 SHA256 为：

```text
9364fe46ee776e7706843dfd3aeff4d5f0c9137de6ad46da4bfb041dff281e98
```

chk3 YAML 保留针对两张 24 GiB A30 的配置：

| 配置 | 值 | 目的 |
|---|---|---|
| dtype / precision | bfloat16 | A30 支持的稳定精度 |
| quantization | NF4 + double quant | 降低 8B parent 显存 |
| optimizer | paged AdamW 8-bit | 降低 optimizer state 显存 |
| per-device batch | 1 | 控制峰值显存 |
| gradient accumulation | 8 | 提升有效 batch |
| gradient checkpointing | true | 用计算换显存 |
| attention | SDPA | chk3 A30-tested path |
| max length | 4,096 | 覆盖全部样本且保留 headroom |
| packing | false | 保持 completion mask 和样本边界 |
| completion-only loss | true | 只监督 reasoning+Minutes completion |
| LoRA rank / alpha | 32 / 64 | 当前 QLoRA contract |
| launcher | DDP 2×A30 | 两卡同步 SFT |

本次只准备数据，没有启动 chk3 training。

## 独立验证结果

最终发布后进行了与 materializer 分离的第二次审计：

1. 重算 release manifest 中 8 个文件的 SHA256，全部一致。
2. 验证 handoff 指向的 release manifest 和 data-quality audit SHA256。
3. 重新加载 2,072 条 training rows 和 2,072 条 manifests。
4. 检查 `{prompt,response}` schema、sample ID uniqueness 和 per-row hashes。
5. 从 prompt JSON 中提取 analysis，检查无 `{`/`[` transport 和无 `ev-` citation。
6. 用本地 DeepSeek tokenizer 和当前 validator 重放全部 2,072 条 target。
7. 对 train+validation 1,882 条运行独立 SFT token-budget preflight。

结果：

| 门禁 | 结果 |
|---|---|
| Release file hashes | 8/8 passed |
| Row count | 2,072/2,072 |
| Unique sample IDs | 2,072/2,072 |
| Full target replay | 2,072/2,072 passed |
| Train+validation preflight | 1,882/1,882 passed |
| Test holdout replay | 190/190 passed |
| Transport-wrapped analysis | 0 |
| Analysis evidence citation | 0 |
| Invalid response boundary | 0 |
| Provider identity drift | 0 |

相关测试：

```text
ruff: passed
pytest: 164 passed
git diff --check: passed
```

测试覆盖 materializer helpers、chk3 target generator、targeted repair、DAG、token-budget
gate，以及默认 DAG 与 total-only alternate DAG 的兼容性。

本报告没有使用趋势图：这是一次固定 release 的离散质量审计，精确计数、token 分位数和
lineage 表比趋势图更直接，避免给静态 snapshot 制造时间序列含义。

## Release 文件与哈希

| 文件 | Rows | Bytes | SHA256 |
|---|---:|---:|---|
| `minutes_alignment/train.jsonl` | 1,683 | 7,362,781 | `bd60ccd9b6175b2ee069d80ad75e876523b1e9f26ef0f14fb16287428eb3efae` |
| `minutes_alignment/validation.jsonl` | 199 | 897,722 | `4844f68306db2a3f8bef55f5b3eb53524d9effa95d5062a3c15e45621ffc36fa` |
| `minutes_alignment/test.jsonl` | 190 | 852,024 | `d3ba4eee3ada2e9871ae776c0adc86fe72dbf295008820824857922d41f3c927` |
| `minutes_alignment/manifests/train.jsonl` | 1,683 | 2,050,745 | `9e46b68b20f22906304150c33d4af7b9edee129bdf95e527ba5b7e224b158028` |
| `minutes_alignment/manifests/validation.jsonl` | 199 | 243,069 | `655aba8e7c8519a34767f35b68423340a3d5ae2b88ae40e008e6b195a19130be` |
| `minutes_alignment/manifests/test.jsonl` | 190 | 231,016 | `09597cc87fdb4010cb1893a83b244dd0e44b7d99c21102e6e569cf7481a3345c` |
| `audits/data_quality.json` | — | 1,832 | `4d5c9bc4354bc75a5423615f48d57dd87558c0e7e10b6bb16a21a99a8712c1c4` |
| `chk3_minutes_sft.template.yaml` | — | 2,866 | `e0765cdf34b6c0896b2bb946815eefba5c45d94735e916b5296691739007a466` |

顶层完整性文件：

| 文件 | SHA256 |
|---|---|
| `handoff.json` | `5dcc069781f71c0eff494cce48c80f6b6e54f35a374710bed4c1ad246fbe54b8` |
| `release_manifest.json` | `3ff0b1ca409d0539993f4e9e3062c439c3dcd79923d89edc287a0605ac2fda8b` |

实现文件：

| 文件 | SHA256 |
|---|---|
| `jobs/generation/materialize_chk3_training_data.py` | `f703c1afa22bcae00a4342957234a10a4f29baf7435abb52b8d8384437a4bd24` |
| `jobs/generation/generate_chk3_sft_targets.py` | `20234ee7942449e43bf9530c388b344cf3c3da6b4f99d86d23945810411d7187` |
| `jobs/generation/repair_chk3_sft_targets.py` | `435eaf61490a2bcfedacff747b7e8ce05273aa1dd8e89177334b07a78b5ed100` |
| `configs/retrain_v2/chk3_minutes_sft.yaml` | `735333ace3a0dc23965e9ac3840ce835e3df988d36af00e21b3348d98d25b188` |

## 可复现命令

全部命令使用 `fomc_trainer`：

```bash
# 只分类和预检，不调用 API
conda run -n fomc_trainer python -m \
  jobs.generation.materialize_chk3_training_data --dry-run

# 实际恢复、重生成并发布；accepted cache 自动复用
conda run -n fomc_trainer python -m \
  jobs.generation.materialize_chk3_training_data --concurrency 8

# 相关回归测试
conda run -n fomc_trainer pytest -q \
  tests/test_materialize_chk3_training_data.py \
  tests/test_chk3_sft_target_generation.py \
  tests/test_repair_chk3_sft_targets.py \
  tests/test_retrain_v2_token_budget_gate.py \
  tests/test_retrain_v2_dag.py

# 静态检查
conda run -n fomc_trainer ruff check \
  jobs/generation/materialize_chk3_training_data.py \
  jobs/retrain_v2/dag.py \
  jobs/retrain_v2/token_budget_gate.py \
  tests/test_materialize_chk3_training_data.py \
  tests/test_retrain_v2_dag.py
```

Release 已存在时，脚本不会覆盖 immutable passed handoff。accepted/rejected API response 位于：

```text
output/data/retrain_v2/chk3/training_release_clean_v1/cache/
```

逐样本重建清单见同目录文档：

```text
docs/summary/20260805T123700Z/chk3_regenerated_sample_inventory.md
```

## 限制与稳健性说明

### 当前结论能证明什么

- 数据 schema、population、split、hash 和 token budget 是确定且可复现的。
- 最终 Minutes 对 analysis 的数量、日期和 attribution 契约通过当前 deterministic validator。
- 新增 DeepSeek 输出来自固定 model/fingerprint，没有 provider identity 混用。
- 没有通过截断、删除失败行或降低门禁来得到 2,072 条完整 release。

### 当前结论不能证明什么

- deterministic validator 不能代替人类对每段 Minutes 文风和经济叙事质量的完整评估。
- provider fingerprint 固定不等于模型服务永久可复现；缓存是复现最终 response 的依据。
- 本次没有运行 chk3 loss smoke、训练稳定性或下游 generation evaluation。
- standalone release 不是完整 `derived_release_manifest.json`，不能绕过 chk2 parent seal。

### Provenance 的位置

最终 per-row manifest 保存 target teacher response ID、model、fingerprint、usage 和所有 row hash。
46 条 source-analysis recovery 的 provider raw 和 response ID 保存在 materializer v4 accepted
cache；为避免修改已经发布并哈希的 release，本次没有事后向 release manifest 添加第二套
source-recovery provider 字段。对应 sample-to-response 映射已写入逐样本清单。其中
`chk1-analysis-2022-11-02-0c0691811be35f30` 在 source recovery 后又执行了确定性的 citation
projection，所以 cache 中投影前 analysis hash 与最终 manifest hash 不同；response ID、sample
ID 和投影规则均已保留，不属于 lineage 缺失。

另外，reasoning 元话语门禁是基于当前明确规则和 pattern 的 deterministic 检查；2,072/2,072
通过表示所有样本满足当前可执行合同，不等于已经由人工逐条确认 reasoning 在语义层面完全没有
任何近义元话语。若训练目标对隐藏 reasoning 的文风纯度要求高，应在启动正式训练前增加抽样或
独立语义审核；这不影响数量、日期、attribution、格式和 token-budget 的现有通过结论。

## 后续步骤

1. 完成并 seal chk2 merged artifact 和 tokenizer bundle。
2. 明确 chk3 lineage contract：按已确认设计，student input 来自 chk1 final analysis，而 parent
   weights 来自 chk2；不要在绑定阶段静默改成 chk2 rollout。
3. 生成或绑定 chk4 `decision_grpo`，构造完整 chk2-derived release，或明确拆分 chk3/chk4
   derived release contract。
4. 创建新 run 或导入 sealed chk1/chk2，使用
   `configs/retrain_v2/dag_chk3_full_completion.yaml`。
5. 将本 release 的 `minutes_alignment` 和 config template 绑定到 resolved config。
6. 运行 `verify_parent`、release validator 和 chk3 tokenizer-exact preflight。
7. 先做两卡短 smoke run，再启动完整 2×A30 DDP QLoRA SFT。

## 仍需明确的问题

1. 完整 derived release 是否继续强制同时包含 chk3 `minutes_alignment` 和 chk4
   `decision_grpo`，还是允许两个独立 downstream release？
2. `analysis_lineage` audit 是否需要新增明确规则：数据来源是 chk1 canonical answer，但模型
   parent 是 sealed chk2？
3. chk3 训练完成后采用哪套 generation evaluation：事实保真、Minutes style、人类偏好或
   与原始 Minutes 的语义指标？

这些问题不影响本 release 的数据质量，但会影响下一步 DAG 绑定和训练结果的解释。
