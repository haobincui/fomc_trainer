# chk0 与 chk1 checkpoint-200：33-row 冻结生成四组测试设计与结果

状态：正式 generation、strict/LT 评分与封存审计均已完成。  
创建时间：2026-08-10 UTC。  
比较边界：仅 `chk0 -> chk1 checkpoint-200`。  
主分析：`strict-final-answer-v2`。  
稳健性分析：`length-tolerant-open-tags-v1`。

> **总结果**：执行有效，但 chk0 与 chk1 cp200 在 strict 口径下均为 `0/33` valid，且都在 8,192 output tokens 触顶。四个 Holm family 均无校正后显著差异，因此预注册分类为 **inconclusive on this stress benchmark**。这不表示模型等效；它表示双方在这个 OOD Minutes 压力任务上都是 100% delivery failure，无法从本次 strict benchmark 证明训练的增量价值。

## 1. 测试问题与结论边界

本测试回答一个受限问题：在完全相同的 33 个冻结输入、参考文本、解码参数和逐行 seed 下，从本地 `DeepSeek-R1-Distill-Llama-8B` 基座 `chk0` 到 clean-v2、LR=`1e-6` 训练所得 checkpoint-200 的 merged `chk1`，输出行为发生了什么变化。

测试使用“冻结到会议前一日（D-1）的宏观金融证据 -> 指定 FOMC Minutes 段落正文”的共同压力基准。它同时检查：

- 模型能否结束生成并产生可评分的最终正文；
- 与官方 Minutes 参考段落的语义和表层相似性；
- 输出中的规则可覆盖事实、数字、时间、单位是否得到输入证据支持；
- 宏观方向和政策立场是否与输入证据一致。

本设计不能回答以下问题：

- 不能识别 `chk1 -> chk2`、`chk2 -> chk3` 或任一下游训练边界；
- 不能把 paired artifact difference 解释为随机试验意义上的训练因果效应；
- 不能把这个 all-evidence Minutes drafting 压力测试当作 chk1 的 in-domain atomic-analysis 验证；
- 不能仅凭 length-tolerant 原始 completion 得分，声称模型已经生成了可交付 final answer；
- 不能用本测试单独决定模型是否应提升到 chk2。它只为 `chk0 -> chk1` 相邻边界提供共同输入上的诊断证据。

## 2. 冻结样本、参考文本与无泄漏合同

### 2.1 资产身份

| 资产 | 冻结路径 | 行数 | SHA-256 / payload SHA-256 | 用途 |
|---|---|---:|---|---|
| prompts | `output/evaluation/main/checkpoint_generation/checkpoint_eval_11_v1/dataset/prompts.jsonl` | 33 | `9e19c23d8a06a5ae2626f707bfa681fcb1789562bf43a6732128157faa0553de` | 两模型的字节级共同输入 |
| references | `output/evaluation/main/checkpoint_generation/checkpoint_eval_11_v1/dataset/references.jsonl` | 33 | `80026c603962b00a506b4021535d25ff7cddbf4526618c7a729b2242fc123a9a` | 独立保存的官方 Minutes 段落 |
| test manifest | `output/evaluation/main/checkpoint_generation/checkpoint_eval_11_v1/dataset/test_manifest.json` | 1 | file `e73bb3a29b56fe8152799611fa9aeb67e6c30573ec2f8eda8ac9082878ee35a9`; payload `4be304b880e46721587ada9a7294dcd161c8529a7dccbdf602768cd35bb3986e` | 样本集合、顺序和构建收据 |
| generation config | `configs/main/checkpoint_generation_eval_11.json` | 1 | `0eb08f0a360927d4aab7a14ebb74cf939a7cd59c79aed474746af1bb800577ff` | 解码与运行合同 |
| semantic model manifest | `configs/main/checkpoint_eval_semantic_models.json` | 1 | file `639ca25cf4f5e695b0c6cfeeb47aba75d3e6dfeda1ad87510c23bb4f4a378bf7`; payload `37a6896c388f965a572d3a3e99c2668a913513f2ac0883a0660634c807130445` | 冻结 BERTScore 和 MPNet scorer |

正式执行必须复用上述 manifest payload SHA、33 个 `sample_id` 及其原始顺序。不得重采样、补样、按输出质量删行或为任一模型替换 prompt/reference。两模型每个 `sample_id` 的 prompt bytes、reference bytes、row seed 和 generation config 必须相同。

### 2.2 33-row 全量矩阵

样本不是抽样子集，而是以下 11 次会议与 3 个 section 的完整笛卡尔积，共 `11 x 3 = 33` 行：

| meeting date | `participants_views` | `economic_situation` | `financial_situation` |
|---|---:|---:|---:|
| 2025-03-19 | 1 | 1 | 1 |
| 2025-05-07 | 1 | 1 | 1 |
| 2025-06-18 | 1 | 1 | 1 |
| 2025-07-30 | 1 | 1 | 1 |
| 2025-09-17 | 1 | 1 | 1 |
| 2025-10-29 | 1 | 1 | 1 |
| 2025-12-10 | 1 | 1 | 1 |
| 2026-01-28 | 1 | 1 | 1 |
| 2026-03-18 | 1 | 1 | 1 |
| 2026-04-29 | 1 | 1 | 1 |
| 2026-06-17 | 1 | 1 | 1 |

三个 section 的目标分别为：

- `participants_views`：Participants' Views on Current Conditions and the Economic Outlook；
- `economic_situation`：Staff Review of the Economic Situation；
- `financial_situation`：Staff Review of the Financial Situation。

每个 meeting 输入包含 26 个 indicator blocks；逐行约 648–651 条结构化 evidence facts。证据冻结 cutoff 为目标会议前一个日历日 D-1。主要表同时报告全部 11 次会议/33 行，以及预先定义的 prospective-only 9 次会议/27 行；后者排除 2025-03-19 和 2025-05-07，不得根据结果重新定义。

### 2.3 reference 字段与泄漏风险

`prompts.jsonl` 的 33 行均满足 `response=""` 且 `reference_used_in_prompt=false`；33 个 `sample_id` 唯一。官方 Minutes reference 只存在于独立 `references.jsonl`，只用于评价，不得拼入 system prompt、user prompt、generation metadata 或重试提示。

冻结构建审计对规范化后的 prompt/reference 做了连续 20-token exact-span 扫描，33 行均未发现匹配。这排除了被该规则覆盖的长片段直接复制，但不证明不存在更短片段、改写、事件重合或训练语料层面的语义泄漏。尤其是：

- reference 是事后发布的官方 Minutes，而 prompt 是 point-in-time D-1 evidence；
- 该扫描不是训练数据成员推断，也不证明基座预训练语料从未包含对应 Minutes；
- prospective-only 子集是更严格的时间视角，但仍不应被描述成完全不可见的随机 holdout，除非另有逐条训练谱系证据。

正式结果必须同时附上 prompt/reference/sample ID/hash preflight 收据；任一 hash、行数、顺序或 join cardinality 不一致时，本次比较无效。

## 3. 两个模型的冻结身份

模型身份以 `docs/summary/20260810T124500Z/chk1_cp200_merge_for_chk2/evaluation_checkpoint_manifest.json` 和 exact-merge lineage 为准。

| 比较角色 | artifact / design ID | 冻结路径 | 模型目录 SHA-256 | 谱系 |
|---|---|---|---|---|
| chk0 | `eval-chk0-base` / `chk-0` | `/home/haobin_cui/research_files_space_2/fomc_trainer/models/DeepSeek-R1-Distill-Llama-8B` | `bfb086cb87e9616e60805fc6f6f95826294b1de70b158cfea2d3e213df38ca11` | root，无 parent |
| chk1 checkpoint-200 merged | `eval-chk1-clean-v2-lr1e6-cp200` / `chk-1` | `/home/haobin_cui/research_files_space_2/fomc_trainer/output/training/retrain_v2/chk1_clean_v2_lr1e6_selected_cp200_for_chk2_20260810/merged/chk1` | `0989b94792f8ab6377e2979aaedeb37010b9a2c5e05c77a459b4e5cda5b806e3` | verified parent=`eval-chk0-base` |

chk1 的 source adapter 必须是：

```text
/home/haobin_cui/research_files_space_2/fomc_trainer/output/training/retrain_v2/chk1_clean_v2_override_full3ep_lr1e6_maxlen7168_20260810/adapters/chk1/checkpoint-200
```

其 adapter directory SHA-256 为 `fa454f73071b3989bf7c9ea3aaabb5a599196396a3ecb369d8956b849bacdc2f`。exact-merge 证据文件为 `docs/summary/20260810T124500Z/chk1_cp200_merge_for_chk2/chk1_cp200_exact_merge_lineage.json`，file SHA-256 为 `a09010d8e79b5ad9df3329596c8308bc32d9afb9d5d85e4c51872d17acca49d5`，payload SHA-256 为 `09234146493643b64b93b8afb1db8cd31f5b86216112a3f7466d3475a139ea3d`。

本测试不得静默解析 `latest`、数值最大的 checkpoint、同目录其他 adapter 或已更新的 merged model。checkpoint-200 是在本次评价之外选定的固定对象；本评价本身不训练、不 merge、不改变权重。

## 4. 共同生成协议

### 4.1 system prompt

两模型使用完全相同的 system prompt：

```text
You are drafting one section of historical FOMC Minutes from a frozen D-1 macro-financial evidence packet. Use only the supplied evidence. Do not invent numerical values, dates, policy decisions, or participant views. Return the requested section body only.
```

### 4.2 解码参数

| 参数 | 固定值 |
|---|---|
| decoding | greedy |
| temperature | `0.0` |
| top_p | `1.0` |
| max_new_tokens | `8192` |
| max_model_len | `24576` |
| base seed | `20260729` |
| row seed | `sample-id-sha256-v1`，由同一 `sample_id` 确定 |
| input truncation | 禁止；一旦发生即 fatal |
| output task | requested section body only |

`max_new_tokens=8192` 是本次正式可比性合同。不得把 3072-token probe 与本次正式 run 混在同一结果表；如果另做 3072、4096 或其他上限的资源敏感性实验，必须使用不同 run ID，标为探索性分析，且不能替换主结果。

尽管 greedy decoding 在理想实现中不依赖随机 seed，仍保存 base seed 与逐行 seed，以约束后端、批次或未来实现的可重复性。两模型分别使用各自封存并校验哈希的 tokenizer，由同一 renderer 生成输入；本次 33 条输入还需验证两侧 chat template 与最终 rendered token IDs 逐条一致。推理精度、停止条件和 batch-independent row mapping 保持一致。生成顺序或 batch size 可以因硬件调整，但不得改变逐行输入、配置或输出上限。

每条 raw completion、finish reason、input/output token count、input truncation flag、validation/parse status、模型与配置 hash 都要保存在本次 run 的不可变工件中。不得只保存后处理 candidate。

### 4.3 中间落盘与恢复合同

正式执行采用 batch size `2`，并在每个内部 batch 完成后立即持久化。对 artifact `<id>`，可检查的中间工件位于：

```text
generations/.partial/<id>/generations.progress.v1.jsonl
generations/.partial/<id>/state.progress.v1.json
generations/.partial/<id>/.lock
```

progress JSONL 使用独立 envelope schema，不能被 scorer 当成正式 generation；每批 append 后执行 `flush + fsync`，随后原子替换带 self-hash 的 state。state 记录 committed bytes、partial SHA-256、row count、next prompt index、commit/resume count，并绑定 checkpoint、模型、tokenizer、prompt 顺序、config、system prompt、解码参数、batch size、base/sample seed 和推理环境。

恢复时只接受与冻结 prompt 顺序一致的已提交前缀。state 之后的未提交完整行或半行会被截去；已提交前缀的 hash、JSON、sample ID、顺序、输入 hash 或 seed 任一不一致均 fail closed。同一 artifact 的 batch size 或执行合同不得中途改变，并发 writer 由稳定 inode 上的 `flock` 拒绝。只有 `33/33` 提交后才写正式 `<id>.jsonl`，随后冻结 `generation_complete` state；最终 v2 manifest 单向绑定该 state 的文件哈希与自摘要，以及 partial 的路径、哈希、字节数、行数和执行合同哈希。manifest 写入后不再修改冻结 state；partial 与 state 均保留并由正式校验器复查。

## 5. 两种评分口径

### 5.1 先报告输出有效性审计

在四组质量指标之前，必须先报告：

- 每模型生成工件是否为 33/33；
- `finish_reason` 分布及 token-limit 行数；
- input truncation 行数；
- strict valid 行数；
- length-tolerant valid 行数；
- `<think>`、`</think>`、`<answer>`、`</answer>` 的出现与边界位置；
- length-tolerant extraction mode（raw 枚举为 `answer_tag`、`full_completion` 或 `empty`；展示时将 `answer_tag` 解释为 after-answer-open）计数；
- 空 candidate、parse failure、validation failure 与非有限 scorer 输出计数。

这些是解释相似性和事实指标的必要前提。不能在模型大量未结束时只报告 length-tolerant 内容分数。

### 5.2 主分析：`strict-final-answer-v2`

记原始 completion 为 `x`，strict 后处理提取的 candidate 为 `c_strict`。令 `U=1` 表示：

- upstream `valid_generation` 不为 false；
- 若存在 validation status，其值属于 `passed/valid/validated/ok/success`；
- 任一显式 parseability 字段不为 false。

令 `L=1` 表示 `finish_reason` 属于 `length/max_tokens/token_limit`。strict validity 定义为：

```text
V_strict = 1[c_strict 非空] * U * 1[L=0] * 1[input_truncation != true]
```

token-limit、输入截断、空 candidate、明确 parse/validation failure 都是 fatal。无效行不得从固定 33-row 分母删除；预先规定的有限惩罚为：

- BERTScore、MPNet cosine 和 ROUGE-L 记 0；
- trigram repetition 记 1；
- body-only format compliance 记 0；
- 规则可覆盖的事实 headline 指标按既定 worst-case 有限值记分；
- length 只作描述，不用惩罚值覆盖。

strict 是唯一主分析。所有“改善/退化”主结论必须以它为依据。

### 5.3 稳健性分析：`length-tolerant-open-tags-v1`

该口径直接处理 raw completion，不要求 `</think>`。对大小写不敏感的首个 `<answer>` opening tag：

```text
如果存在 <answer>：
    candidate = opening tag 之后的文本；可移除一个 terminal </answer>；再 trim
否则：
    candidate = 完整 raw completion；再 trim
```

若存在 `<answer>` 但其后为空，则该行无效，不能回退到完整 completion。`</think>` 是否存在、出现在哪里都不影响这一提取规则。有效性仅要求 candidate 非空且未发生 input truncation；token-limit 只作为审计 flag，不作为 fatal。

这一口径的核心限制是：当 raw completion 没有 `<answer>` 时，完整未结束 reasoning 也会被评分。因此它回答的是“未完成输出中是否仍有可测内容”，不是“最终 answer 是否合格”。报告必须把它标为 robustness，不能用其高分覆盖 strict failure，也不能与 final-answer-only 结果混称。

## 6. 四组指标的预注册定义

以下四组名称、成员和方向在看结果前固定。所有 token/claim eligibility 和 `NA` 必须逐行保留；不得为提高平均值而删去不利的有限值。

### 6.1 第一组：语义相似性

#### BERTScore P/R/F1

使用本地冻结的 `FacebookAI/roberta-large`，revision `722cf37b1afa9454edce342e7895e588b6ff1d59`，模型目录 SHA-256 `b970a47c99ab5d994a7bcd689ad86b48fb709076c5d309f6a39d5763478eab88`，第 17 层。关闭 IDF，不做 baseline rescaling。precision、recall、F1 全部保存，F1 为 headline。

#### independent MPNet cosine

使用 `sentence-transformers/all-mpnet-base-v2`，revision `e8c3b32edf5434bc2275fc9bab85f82640a19130`，模型目录 SHA-256 `1c8bfc2c3cb29e484b3ac3585c5166de44389c32cbfdc33d1f2b51634e37403f`。采用 attention-mask mean pooling 后 L2 normalize，报告 cosine similarity `[-1, 1]`。它与 reward 独立，不是训练时 Judge。

#### 长文本共同处理

candidate 与 reference 分别按句子边界切成 encoder 上限内的 chunk：总长度不超过 512 tokens，即内容预算 510 tokens；超过预算的单句按 token 无损切分。两侧 chunk 按 ordinal 使用 `zip_longest` 对齐，以 paired chunk 两侧 token 数较大者为权重；一侧缺失时该 pair 得 0。BERTScore P/R/F1 分别加权汇总；MPNet 亦用同一长文本对齐框架。任何一侧都不得静默截断。

解释限制：这些指标衡量与官方 Minutes reference 的表达相似性，不直接证明事实来自 prompt evidence；也可能惩罚事实正确但措辞不同的输出。

### 6.2 第二组：表层文本质量

共同 lexical tokenizer 使用 casefold 后的正则 token：英文字母词（允许一个内部 apostrophe）以及带可选正负号的整数/小数；标点不进入 token 序列。它独立于 RoBERTa/MPNet tokenizer。

#### ROUGE-L

若 candidate token 序列长度为 `n`，reference 为 `q`，最长公共子序列长度为 `L`：

```text
ROUGE-L F1 = 2L / (n + q)
```

任一侧为空时为 0。实现可以使用 exact bit-parallel LCS，但不得改变 estimand。

#### 长度

报告 generated lexical-token count `n` 和 `length_ratio = n/q`。长度是描述量，不预设越长或越短越好；解释时必须与终止、重复、事实密度联合阅读。

#### trigram 重复

当 `n >= 3`：

```text
trigram_repetition = 1 - unique_generated_trigrams / generated_trigram_positions
```

`n < 3` 时为 0。该指标越低越好；接近 1 表示大部分 trigram positions 是重复的。

#### body-only 格式合规

二值指标。正文合同禁止 reasoning/answer tags、evidence delimiters 和显式 section heading，并执行配置中声明的 required/forbidden patterns、literal tags/sections、字符边界和 known-tag balance。它是窄格式检查，不是总体写作质量 Judge。length-tolerant 提取可以忽略 `</think>`，但返回 reasoning tag 仍可独立违反 body-only 合同。

### 6.3 第三组：事实和数值一致性

#### numeric value accuracy

对生成的非时间数字 `y`，按其显示小数位数 `d(y)` 匹配 allowed evidence value `v`：

```text
ROUND_HALF_UP(v, d(y)) == y
```

不使用相对模糊容差，不做隐式单位换算。numeric value accuracy 是匹配数字数/生成的非时间数字数；没有非时间数字时为 `NA`。

#### evidence value coverage

分母是 distinct structured numeric evidence facts。一个 fact 只有在生成数字匹配其 value，且该 fact 声明单位时生成单位也匹配，才算 covered。该指标是 evidence fact-level recall，必须和 conditional numeric accuracy 联合解释；面对约 648–651 facts 的压力输入时，低 coverage 不自动等于 factual error。

#### unit accuracy

只有匹配至少一个已声明单位的 structured fact 的生成数字进入分母。生成单位必须属于该 value 的预期标准化单位集合；expected unit 存在而单位缺失时为错误。不对百分比、百分点、bp 等作隐式换算。

#### time accuracy

识别 ISO date/range、year range、year、quarter、month-year 和指定 relative period。规范化 case、空白和 dash 后，只有和 evidence allowed-time exact match 才正确；没有生成 time expression 时为 `NA`。

#### novel-number rate

分母包含所有生成数字，也包含 time expression 内数字；未匹配 allowed set 的比例为 novel-number rate，越低越好。evidence 中出现的日期数字会进入 allowed set，因此不会仅因其是年份而自动标 novel。

#### rule-covered unsupported claim rate

规则范围仅覆盖：非时间 numeric claims、time claims、direction claims、policy-stance claims 和显式 custom claim rules。unsupported rate 是规则判定不支持的 detected claims / 全部 rule-covered detected claims；没有 detected claim 时为 `NA`，越低越好。standalone unit check 不重复进入分母，因为单位冲突已由对应 numeric claim 处理。

该指标不是完整自然语言蕴含或通用 factuality 检测。规则没覆盖的错误不会被发现；无法从 evidence 得到 expected class 的 direction/stance claim 在该规则范围内记 unsupported。

### 6.4 第四组：方向一致性

#### inflation/growth/employment/unemployment direction coverage + consistency

四个核心 topic 为 inflation、growth、employment、unemployment，方向标准化为 `up/down/flat`。若没有显式 expected direction，则证据优先级依次为 short-run `derived_absolute_change`、YoY `derived_year_absolute_change`、其他 directional facts；数值派生事实用符号推断方向。选中证据 mixed/conflicting 时，该 topic 不进入 consistency eligibility。

direction consistency 的分母是生成文本中可由 evidence 判定 expected direction 的 topic-direction claims，分子是方向一致的 claims；没有 coverable claim 时为 `NA`。direction coverage 的分母是有明确 expected direction 的 topic 集合，分子是生成文本中被提及且带方向的对应 topics。报告总体 coverage/consistency，并保留四个 topic 的 eligibility 和错误审计，不能只给宏观平均值而隐藏某一 topic 的系统性反向。

#### policy stance coverage + consistency

stance 标准化为 `hawkish/dovish/neutral`。显式 evidence stance 优先；否则唯一 policy-rate direction 映射为：

```text
up -> hawkish
down -> dovish
flat -> neutral
```

stance consistency 是 coverable generated stance claims 中与 expected stance 一致的比例；没有 coverable claim 时为 `NA`。当 expected stance 可用时，row-level stance coverage 是是否产生至少一个 stance claim 的二值指标。

## 7. 聚合、配对统计与多重比较

### 7.1 两个固定分析子集

- all-11：11 meetings，33 rows；
- prospective-only：预先排除前两次会议，9 meetings，27 rows。

两个子集分别分析，不把同一行在两个子集中当作独立样本。所有结果必须给有限 row eligibility `n` 和 meeting-cluster eligibility；conditional metric 的 `NA` 不得改成 0。strict 预注册 fatal penalties 是例外，按第 5.2 节执行。

### 7.2 meeting-equal-weight 聚合

对任一 row-level metric `z`，先在每个 meeting 内对其 3 个 section 的有限值取平均，再对 eligible meetings 等权平均。不得把所有 claims 或所有 rows 直接 pooled 后计算一个全局比率。这样每次会议权重相同，不会因某 meeting 产生更多 claims 而支配结果。

### 7.3 唯一配对 contrast

唯一预注册 contrast 为：

```text
delta = chk1 checkpoint-200 - chk0
```

每个 `sample_id` 在两模型间严格 paired；先在 meeting 内对双方均为有限值的 section difference 取平均，再对 meeting differences 等权平均。不得报告 chk2、历史 chk1、其他 checkpoint 或不同 max token run 与本次结果的同表显著性比较。

对于 BERTScore、MPNet、ROUGE-L、format、accuracy、coverage 和 consistency，正 delta 方向通常更好；对于 trigram repetition、novel-number rate 和 unsupported-claim rate，负 delta 更好。length/length ratio 无统一优劣方向。结果表必须显式显示 `higher/lower/descriptive`，避免把正 delta 一律解释为改善。

### 7.4 不确定性与检验

- 使用 seed `20260729`，以 meeting mean 或 paired meeting difference 为 cluster，做 10,000 次有放回 bootstrap；
- 报告 2.5% 与 97.5% percentile 形成的 95% CI；至少需要 2 个 eligible meeting clusters；
- paired p-value 使用 two-sided exact sign-flip test；all-11 枚举 `2^11=2048` 个符号组合，prospective-only 枚举 `2^9=512` 个组合；实际 eligibility 更少时按其 cluster 数完整枚举；
- 在每个 scoring policy 与每个 subset 内，对所有可估计的预注册 metric contrasts 单独做 Holm family-wise correction；实际 family size `K` 必须由 scorer audit 写出，不能删去不显著项目后重算；
- 当前通用 scorer 的 Holm family 还保守地纳入 `valid_output`、generated character count 和 generated sentence count；因此实际 `K` 可能大于第 6 节四组表中展示的指标数。正式报告读取 `contrasts.holm_family_size` 与 audit，不事后缩小 family 或重算更有利的 p-value；
- bootstrap CI 不做 multiplicity adjustment，所以 CI 排除 0 与 Holm-adjusted p-value 大于 0.05 可以同时发生；
- `p > 0.05` 只表示没有足够证据拒绝零差异，不证明等效或 non-inferior。

不计算以 section 当独立 cluster 的伪重复显著性，不用 33 行 naive bootstrap 替代 meeting-cluster bootstrap，也不在看到结果后挑选 seed、会议或指标。

## 8. 正式执行与解释门槛

### 8.1 执行有效性硬门槛

满足以下条件，才可把 run 称为“完成的正式比较”：

- 第 2、3、4 节的所有输入、reference、manifest、模型、tokenizer 与 config hash 完全匹配；
- 两模型均有 33/33 唯一 generation records，无重复或跨模型 sample mismatch；
- 两模型逐行 prompt/reference/seed 完全相同；
- input truncation 为 0；
- 四组 scorer 均完成，所有值只能为有限数或按合同记录的 `NA`，不得出现 silent scorer failure；
- strict 与 length-tolerant 分开产出审计和统计结果；
- raw completion 与后处理 candidate 的 hash/lineage 可回溯。

任一硬门槛失败时，结果状态为 execution invalid，不允许通过排除坏行来修补。修复执行问题后必须用新 run ID 重跑受影响部分，并保留失败收据。

### 8.2 模型比较的预注册解释

本基准不设一个把不同量纲指标任意加权的总分，也不以单个相似性指标决定“通过”。结论使用以下受限分类：

预结果解释澄清（2026-08-10 13:34 UTC，generation 尚未落盘）：最终四分类只由 **strict all-11** 主分析 family 决定；prospective-only 是预注册 secondary/sensitivity，不得单独改变最终 label，避免在两个各自校正的 subset family 之间挑选结果。以下分类按互斥顺序执行，避免“至少一个不利指标”和“有利、不利并存”之间的文字重叠。先判断 execution validity；随后把有利与不利的 Holm 显著结果并存归为 inconclusive；只在没有有利显著结果时，才把不利显著结果归为 degradation。

- **adjacent-edge improvement evidence**：strict all-11 主分析中，至少一个预注册质量指标显示 Holm-adjusted 的有利 paired difference，且没有任何其他预注册质量指标显示 Holm-adjusted 的不利 paired difference；同时必须满足 chk1 strict-valid 行数不低于 chk0、chk1 token-limit 行数不高于 chk0，且双方 input truncation 均为 0。若显著性方向满足但任一 guard 不满足，则归为 inconclusive 并披露原因。
- **adjacent-edge degradation evidence**：strict 主分析中，至少一个预注册质量指标显示 Holm-adjusted 的不利 paired difference，且没有 Holm-adjusted 的有利 paired difference。报告必须指出受影响组、eligibility 和 effect size，不能只写 p-value。
- **inconclusive on this stress benchmark**：没有 Holm-adjusted 的有利或不利证据，或有利与不利指标并存。该分类不等于模型等效。
- **execution invalid**：未满足第 8.1 节硬门槛。

length-tolerant 结果只能解释 strict failure 内部仍存在的内容，不能单独把 degradation 改判为 improvement。无论分类为何，都不能直接推出 chk1 的 atomic-analysis 产品任务通过或失败。

## 9. 正式结果

### 9.1 结论摘要与执行审计

预注册最终分类：**inconclusive on this stress benchmark**。执行本身有效，但 strict all-11 中 chk0 与 chk1 cp200 都是 `0/33` valid；17 个有明确效用方向的四组质量指标均没有 Holm-adjusted 的有利或不利差异。这个结果不能解释为模型等效，更不能证明 chk1 已适合进入 chk2。

正式 run root：`output/evaluation/main/checkpoint_generation/retrain_v2_chk0_chk1_cp200_20260810_v1`。正式 run 于 `2026-08-10T14:05:26Z` 开始；chk1 generation 的起止时间为 `2026-08-10T15:17:52Z` 至 `2026-08-10T16:38:14Z`。生成使用 `llama_factory`（Python 3.10.14、PyTorch 2.6.0、Transformers 4.52.3、vLLM 0.8.5.post1、单张 A30 24 GB、`CUDA_VISIBLE_DEVICES=0`）；日志记录的 BF16 属于 operational、未被 generation manifest 单独封存的元数据。评分 launcher 使用 `fomc_trainer`、语义模型运行在 GPU0，这两项同样属于 operational 信息，因为评分环境没有独立的 sealed environment receipt；已封存的是 evaluator source SHA `da4b12f13345c72719f51a0b15ab4abd562409e424631a6c04528576a5047c8f`，以及 strict/LT run-spec SHA `e06f17824b08eab594ff2e510fa4e1f7d10256936b2d33b2231632688398eed4` / `fed38d6693f6b01dd785a33c2f9ec2fe4ec93a46955a256f84e4b08b1f365163`。生成封存完成时间为 chk0 `2026-08-10T15:13:41Z`、chk1 `2026-08-10T16:38:14Z`；strict/LT 日志最后更新时间分别为 `16:40:07Z` 与 `16:42:29Z`。当前 git HEAD `08dee23a285ce22ea0dff56353796f93499f57f4`/dirty worktree（301 paths）同样属于 operational、未封存元数据。

| 检查项 | 期望 | 实际 | 状态 |
|---|---:|---:|---|
| prompt rows / unique sample IDs | 33 / 33 | 33 / 33 | pass |
| reference rows / unique sample IDs | 33 / 33 | 33 / 33 | pass |
| chk0 / chk1 generation rows | 33 / 33 | 33 / 33 | pass |
| paired prompt/reference/seed mismatch | 0 | 0 | pass |
| input truncation rows（两模型） | 0 | 0 / 0 | pass |
| strict / LT scorer status | validated / validated | validated / validated | pass |
| non-finite non-NA / missing input | 0 / 0 | 0 / 0 | pass |
| exact merge mismatch | 0 | 0（291=224 adapted+67 unchanged；448 adapter tensors） | pass |

关键封存身份：

- chk0 generation SHA：`0b40f7ba9c8388b7dc480b941e0cbdfce948eede13aaca67daab9d50bf884467`；v2 manifest file/payload：`3e35af432f4628f2bfbaf5bcffc981233dcbe33d2957543dc38817bb8c3d1f85` / `8a89388af628fb67431f078e9e1d91a25c90dc5cb5112abf725e4a1318d26a4f`；partial/state：`2a253d4f1f1e77b6f88fa48813cf3dc790d993e36969928e98c87609c350d04d` / `9dfbff10bf1672e8c0ba8f5bf58b665cf334d346d5feda27eefae682d66b20b1`。
- chk1 generation SHA：`1c81f236fe61ea386cec6f6789fb35b1c993eb5266b6f442c81f7584c96d310d`；v2 manifest file/payload：`ee158386670d5b170cf5d09f9877a51c7ed266c5ff8200f3b2947d167014ec81` / `8e44ff6d6f8e1d58631b9425363d4a529d121839581a46e2ff002ec2cfc1dd4c`；partial/state：`75dacfe2b5168e1d798d7f5ed6f1fa60a15a497f97144ff45b5c0ac84bd34562` / `dd1d025a56cda8bc27fd4c1d43b9313e1269dc69641edb5315b004c1db7e9525`。
- strict audit/input-validation SHA：`4aaf140384e56128d52daa55aa75eb22c6a8c3411376931884e28132df003c10` / `e989a19017ab2805021abc17ac3b31107c887c8475572ee423641e2eb1ea6880`。
- LT audit/input-validation SHA：`00c0d750736595602bd94d9c163f5b4ccbb749d8eb0a5d2b229156889d4162d7` / `b26ec358dfa258a5fc44c764de41aa9da9cf22e37374c433302d2c93db8e6b19`。
- 四组汇总 file/payload SHA：`2c8f3d6b4d8f058f278762805f43f9bfaf3f9ae5b340879ca253b1db25063efc` / `7ca80ccefc2ce2ede1828d7881b28a30f7b4b80bb5077284a25f430ed15c9c6c`。

### 9.2 输出有效性、边界与提取审计

| 模型 | strict valid all / prospective | LT valid all / prospective | token-limit all / prospective | `full_completion` | `answer_tag` | input truncation |
|---|---:|---:|---:|---:|---:|---:|
| chk0 | 0/33 / 0/27 | 33/33 / 27/27 | 33 / 27 | 33 | 0 | 0 |
| chk1 cp200 | 0/33 / 0/27 | 33/33 / 27/27 | 33 / 27 | 33 | 0 | 0 |

| 模型 | `<think>` rows | `</think>` rows | `<answer>` rows | `</answer>` rows | empty candidate | parse/validation failure | finish reason | raw model output tokens |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| chk0 | 0 | 0 | 0 | 0 | 0 | 0 | `length=33` | 8,192 × 33 |
| chk1 cp200 | 0 | 0 | 0 | 0 | 0 | 0 | `length=33` | 8,192 × 33 |

两模型全部失败都来自 `non_normal_finish:length`，不是输入截断或显式 parser/validator error。strict 表中的 BERTScore、MPNet、ROUGE 等 0 值是预注册 fatal penalty；semantic backend 明确记录为 `inference_skipped=no_nonfatal_generation_rows`，不是实际测得的语义相似度。LT 因无 `<answer>` 而把 66 条 raw completion 全部标记为 `full_completion`；它们不是可交付 final answer。下表中的 generated token count 是 scorer 的文本 tokenizer 计数，不是 vLLM 的原始 8,192 model-token 计数。

### 9.3 strict primary：all 11 meetings / 33 rows

Holm family size：`K=22`。模型列为 marginal meeting-equal estimate；delta 仅基于共同 finite pairs，因此两模型 marginal 均值之差不一定逐位等于 delta。

| 组 | 指标 | 方向 | chk0 | chk1 cp200 | paired delta `chk1-chk0` | raw p | Holm p |
|---|---|---|---:|---:|---:|---:|---:|
| 语义相似性 | BERTScore precision | higher | 0.0000 [0.0000, 0.0000]; n=33/M=11 | 0.0000 [0.0000, 0.0000]; n=33/M=11 | +0.0000 [+0.0000, +0.0000]; n=33/M=11; dz=NA | 1 | 1 |
| 语义相似性 | BERTScore recall | higher | 0.0000 [0.0000, 0.0000]; n=33/M=11 | 0.0000 [0.0000, 0.0000]; n=33/M=11 | +0.0000 [+0.0000, +0.0000]; n=33/M=11; dz=NA | 1 | 1 |
| 语义相似性 | BERTScore F1 | higher | 0.0000 [0.0000, 0.0000]; n=33/M=11 | 0.0000 [0.0000, 0.0000]; n=33/M=11 | +0.0000 [+0.0000, +0.0000]; n=33/M=11; dz=NA | 1 | 1 |
| 语义相似性 | independent MPNet cosine | higher | 0.0000 [0.0000, 0.0000]; n=33/M=11 | 0.0000 [0.0000, 0.0000]; n=33/M=11 | +0.0000 [+0.0000, +0.0000]; n=33/M=11; dz=NA | 1 | 1 |
| 表层文本质量 | ROUGE-L F1 | higher | 0.0000 [0.0000, 0.0000]; n=33/M=11 | 0.0000 [0.0000, 0.0000]; n=33/M=11 | +0.0000 [+0.0000, +0.0000]; n=33/M=11; dz=NA | 1 | 1 |
| 表层文本质量 | generated token count | descriptive | 6,261.6 [6,019.4, 6,502.6]; n=33/M=11 | 6,194.7 [6,038.4, 6,372.0]; n=33/M=11 | -67.0 [-232.5, +96.4]; n=33/M=11; dz=-0.227 | 0.4609 | 1 |
| 表层文本质量 | length ratio | descriptive | 7.5872 [6.9494, 8.2844]; n=33/M=11 | 7.5869 [7.1439, 8.1048]; n=33/M=11 | -0.0002 [-0.2855, +0.3261]; n=33/M=11; dz=-0.000 | 1 | 1 |
| 表层文本质量 | trigram repetition | lower | 1.0000 [1.0000, 1.0000]; n=33/M=11 | 1.0000 [1.0000, 1.0000]; n=33/M=11 | +0.0000 [+0.0000, +0.0000]; n=33/M=11; dz=NA | 1 | 1 |
| 表层文本质量 | body-only format compliance | higher | 0.0000 [0.0000, 0.0000]; n=33/M=11 | 0.0000 [0.0000, 0.0000]; n=33/M=11 | +0.0000 [+0.0000, +0.0000]; n=33/M=11; dz=NA | 1 | 1 |
| 事实和数值一致性 | numeric value accuracy | higher | 0.0000 [0.0000, 0.0000]; n=33/M=11 | 0.0000 [0.0000, 0.0000]; n=33/M=11 | +0.0000 [+0.0000, +0.0000]; n=33/M=11; dz=NA | 1 | 1 |
| 事实和数值一致性 | evidence value coverage | higher | 0.0000 [0.0000, 0.0000]; n=33/M=11 | 0.0000 [0.0000, 0.0000]; n=33/M=11 | +0.0000 [+0.0000, +0.0000]; n=33/M=11; dz=NA | 1 | 1 |
| 事实和数值一致性 | unit accuracy | higher | 0.0000 [0.0000, 0.0000]; n=33/M=11 | 0.0000 [0.0000, 0.0000]; n=33/M=11 | +0.0000 [+0.0000, +0.0000]; n=33/M=11; dz=NA | 1 | 1 |
| 事实和数值一致性 | time accuracy | higher | 0.0000 [0.0000, 0.0000]; n=33/M=11 | 0.0000 [0.0000, 0.0000]; n=33/M=11 | +0.0000 [+0.0000, +0.0000]; n=33/M=11; dz=NA | 1 | 1 |
| 事实和数值一致性 | novel-number rate | lower | 1.0000 [1.0000, 1.0000]; n=33/M=11 | 1.0000 [1.0000, 1.0000]; n=33/M=11 | +0.0000 [+0.0000, +0.0000]; n=33/M=11; dz=NA | 1 | 1 |
| 事实和数值一致性 | rule-covered unsupported rate | lower | 1.0000 [1.0000, 1.0000]; n=33/M=11 | 1.0000 [1.0000, 1.0000]; n=33/M=11 | +0.0000 [+0.0000, +0.0000]; n=33/M=11; dz=NA | 1 | 1 |
| 方向一致性 | macro direction coverage | higher | 0.0000 [0.0000, 0.0000]; n=33/M=11 | 0.0000 [0.0000, 0.0000]; n=33/M=11 | +0.0000 [+0.0000, +0.0000]; n=33/M=11; dz=NA | 1 | 1 |
| 方向一致性 | macro direction consistency | higher | 0.0000 [0.0000, 0.0000]; n=33/M=11 | 0.0000 [0.0000, 0.0000]; n=33/M=11 | +0.0000 [+0.0000, +0.0000]; n=33/M=11; dz=NA | 1 | 1 |
| 方向一致性 | policy stance coverage | higher | 0.0000 [0.0000, 0.0000]; n=21/M=7 | 0.0000 [0.0000, 0.0000]; n=21/M=7 | +0.0000 [+0.0000, +0.0000]; n=21/M=7; dz=NA | 1 | 1 |
| 方向一致性 | policy stance consistency | higher | 0.0000 [0.0000, 0.0000]; n=21/M=7 | 0.0000 [0.0000, 0.0000]; n=21/M=7 | +0.0000 [+0.0000, +0.0000]; n=21/M=7; dz=NA | 1 | 1 |

### 9.4 strict secondary：prospective-only 9 meetings / 27 rows

Holm family size：`K=22`。模型列为 marginal meeting-equal estimate；delta 仅基于共同 finite pairs，因此两模型 marginal 均值之差不一定逐位等于 delta。

| 组 | 指标 | 方向 | chk0 | chk1 cp200 | paired delta `chk1-chk0` | raw p | Holm p |
|---|---|---|---:|---:|---:|---:|---:|
| 语义相似性 | BERTScore precision | higher | 0.0000 [0.0000, 0.0000]; n=27/M=9 | 0.0000 [0.0000, 0.0000]; n=27/M=9 | +0.0000 [+0.0000, +0.0000]; n=27/M=9; dz=NA | 1 | 1 |
| 语义相似性 | BERTScore recall | higher | 0.0000 [0.0000, 0.0000]; n=27/M=9 | 0.0000 [0.0000, 0.0000]; n=27/M=9 | +0.0000 [+0.0000, +0.0000]; n=27/M=9; dz=NA | 1 | 1 |
| 语义相似性 | BERTScore F1 | higher | 0.0000 [0.0000, 0.0000]; n=27/M=9 | 0.0000 [0.0000, 0.0000]; n=27/M=9 | +0.0000 [+0.0000, +0.0000]; n=27/M=9; dz=NA | 1 | 1 |
| 语义相似性 | independent MPNet cosine | higher | 0.0000 [0.0000, 0.0000]; n=27/M=9 | 0.0000 [0.0000, 0.0000]; n=27/M=9 | +0.0000 [+0.0000, +0.0000]; n=27/M=9; dz=NA | 1 | 1 |
| 表层文本质量 | ROUGE-L F1 | higher | 0.0000 [0.0000, 0.0000]; n=27/M=9 | 0.0000 [0.0000, 0.0000]; n=27/M=9 | +0.0000 [+0.0000, +0.0000]; n=27/M=9; dz=NA | 1 | 1 |
| 表层文本质量 | generated token count | descriptive | 6,267.8 [5,980.3, 6,543.7]; n=27/M=9 | 6,142.9 [5,986.0, 6,331.2]; n=27/M=9 | -124.9 [-301.1, +59.9]; n=27/M=9; dz=-0.423 | 0.2188 | 1 |
| 表层文本质量 | length ratio | descriptive | 7.7822 [7.0789, 8.5391]; n=27/M=9 | 7.6588 [7.1594, 8.2581]; n=27/M=9 | -0.1234 [-0.3947, +0.2365]; n=27/M=9; dz=-0.235 | 0.5117 | 1 |
| 表层文本质量 | trigram repetition | lower | 1.0000 [1.0000, 1.0000]; n=27/M=9 | 1.0000 [1.0000, 1.0000]; n=27/M=9 | +0.0000 [+0.0000, +0.0000]; n=27/M=9; dz=NA | 1 | 1 |
| 表层文本质量 | body-only format compliance | higher | 0.0000 [0.0000, 0.0000]; n=27/M=9 | 0.0000 [0.0000, 0.0000]; n=27/M=9 | +0.0000 [+0.0000, +0.0000]; n=27/M=9; dz=NA | 1 | 1 |
| 事实和数值一致性 | numeric value accuracy | higher | 0.0000 [0.0000, 0.0000]; n=27/M=9 | 0.0000 [0.0000, 0.0000]; n=27/M=9 | +0.0000 [+0.0000, +0.0000]; n=27/M=9; dz=NA | 1 | 1 |
| 事实和数值一致性 | evidence value coverage | higher | 0.0000 [0.0000, 0.0000]; n=27/M=9 | 0.0000 [0.0000, 0.0000]; n=27/M=9 | +0.0000 [+0.0000, +0.0000]; n=27/M=9; dz=NA | 1 | 1 |
| 事实和数值一致性 | unit accuracy | higher | 0.0000 [0.0000, 0.0000]; n=27/M=9 | 0.0000 [0.0000, 0.0000]; n=27/M=9 | +0.0000 [+0.0000, +0.0000]; n=27/M=9; dz=NA | 1 | 1 |
| 事实和数值一致性 | time accuracy | higher | 0.0000 [0.0000, 0.0000]; n=27/M=9 | 0.0000 [0.0000, 0.0000]; n=27/M=9 | +0.0000 [+0.0000, +0.0000]; n=27/M=9; dz=NA | 1 | 1 |
| 事实和数值一致性 | novel-number rate | lower | 1.0000 [1.0000, 1.0000]; n=27/M=9 | 1.0000 [1.0000, 1.0000]; n=27/M=9 | +0.0000 [+0.0000, +0.0000]; n=27/M=9; dz=NA | 1 | 1 |
| 事实和数值一致性 | rule-covered unsupported rate | lower | 1.0000 [1.0000, 1.0000]; n=27/M=9 | 1.0000 [1.0000, 1.0000]; n=27/M=9 | +0.0000 [+0.0000, +0.0000]; n=27/M=9; dz=NA | 1 | 1 |
| 方向一致性 | macro direction coverage | higher | 0.0000 [0.0000, 0.0000]; n=27/M=9 | 0.0000 [0.0000, 0.0000]; n=27/M=9 | +0.0000 [+0.0000, +0.0000]; n=27/M=9; dz=NA | 1 | 1 |
| 方向一致性 | macro direction consistency | higher | 0.0000 [0.0000, 0.0000]; n=27/M=9 | 0.0000 [0.0000, 0.0000]; n=27/M=9 | +0.0000 [+0.0000, +0.0000]; n=27/M=9; dz=NA | 1 | 1 |
| 方向一致性 | policy stance coverage | higher | 0.0000 [0.0000, 0.0000]; n=15/M=5 | 0.0000 [0.0000, 0.0000]; n=15/M=5 | +0.0000 [+0.0000, +0.0000]; n=15/M=5; dz=NA | 1 | 1 |
| 方向一致性 | policy stance consistency | higher | 0.0000 [0.0000, 0.0000]; n=15/M=5 | 0.0000 [0.0000, 0.0000]; n=15/M=5 | +0.0000 [+0.0000, +0.0000]; n=15/M=5; dz=NA | 1 | 1 |

### 9.5 length-tolerant robustness：all 11 meetings / 33 rows

该表评分未结束的 raw completion，只作诊断，不改变 strict 分类。

Holm family size：`K=20`。模型列为 marginal meeting-equal estimate；delta 仅基于共同 finite pairs，因此两模型 marginal 均值之差不一定逐位等于 delta。

| 组 | 指标 | 方向 | chk0 | chk1 cp200 | paired delta `chk1-chk0` | raw p | Holm p |
|---|---|---|---:|---:|---:|---:|---:|
| 语义相似性 | BERTScore precision | higher | 0.1404 [0.1325, 0.1489]; n=33/M=11 | 0.1388 [0.1307, 0.1476]; n=33/M=11 | -0.0016 [-0.0042, +0.0008]; n=33/M=11; dz=-0.354 | 0.2764 | 1 |
| 语义相似性 | BERTScore recall | higher | 0.1496 [0.1411, 0.1586]; n=33/M=11 | 0.1496 [0.1409, 0.1589]; n=33/M=11 | +0.0000 [-0.0021, +0.0022]; n=33/M=11; dz=0.003 | 0.9932 | 1 |
| 语义相似性 | BERTScore F1 | higher | 0.1448 [0.1366, 0.1535]; n=33/M=11 | 0.1439 [0.1356, 0.1529]; n=33/M=11 | -0.0009 [-0.0031, +0.0014]; n=33/M=11; dz=-0.210 | 0.5 | 1 |
| 语义相似性 | independent MPNet cosine | higher | 0.0425 [0.0359, 0.0491]; n=33/M=11 | 0.0484 [0.0397, 0.0565]; n=33/M=11 | +0.0060 [-0.0023, +0.0139]; n=33/M=11; dz=0.419 | 0.1924 | 1 |
| 表层文本质量 | ROUGE-L F1 | higher | 0.0268 [0.0248, 0.0288]; n=33/M=11 | 0.0245 [0.0221, 0.0270]; n=33/M=11 | -0.0022 [-0.0054, +0.0007]; n=33/M=11; dz=-0.414 | 0.1963 | 1 |
| 表层文本质量 | generated token count | descriptive | 6,261.6 [6,019.4, 6,502.6]; n=33/M=11 | 6,194.7 [6,038.4, 6,372.0]; n=33/M=11 | -67.0 [-232.5, +96.4]; n=33/M=11; dz=-0.227 | 0.4609 | 1 |
| 表层文本质量 | length ratio | descriptive | 7.5872 [6.9494, 8.2844]; n=33/M=11 | 7.5869 [7.1439, 8.1048]; n=33/M=11 | -0.0002 [-0.2855, +0.3261]; n=33/M=11; dz=-0.000 | 1 | 1 |
| 表层文本质量 | trigram repetition | lower | 0.9840 [0.9814, 0.9862]; n=33/M=11 | 0.9864 [0.9854, 0.9874]; n=33/M=11 | +0.0024 [+0.0005, +0.0045]; n=33/M=11; dz=0.665 | 0.06836 | 1 |
| 表层文本质量 | body-only format compliance | higher | 1.0000 [1.0000, 1.0000]; n=33/M=11 | 1.0000 [1.0000, 1.0000]; n=33/M=11 | +0.0000 [+0.0000, +0.0000]; n=33/M=11; dz=NA | 1 | 1 |
| 事实和数值一致性 | numeric value accuracy | higher | 0.9993 [0.9985, 1.0000]; n=11/M=7 | 0.9692 [0.9075, 1.0000]; n=10/M=8 | +0.0003 [+0.0000, +0.0010]; n=8/M=6; dz=0.408 | 1 | 1 |
| 事实和数值一致性 | evidence value coverage | higher | 0.0015 [0.0000, 0.0041]; n=33/M=11 | 0.0032 [0.0000, 0.0096]; n=33/M=11 | +0.0017 [-0.0038, +0.0089]; n=33/M=11; dz=0.151 | 1 | 1 |
| 事实和数值一致性 | unit accuracy | higher | 0.0428 [0.0001, 0.1157]; n=11/M=7 | 0.0312 [0.0000, 0.0938]; n=10/M=8 | -0.0019 [-0.0039, -0.0001]; n=8/M=6; dz=-0.699 | 0.25 | 1 |
| 事实和数值一致性 | time accuracy | higher | 1.0000 [1.0000, 1.0000]; n=3/M=2 | 1.0000 [1.0000, 1.0000]; n=4/M=3 | +0.0000 [+0.0000, +0.0000]; n=3/M=2; dz=NA | 1 | 1 |
| 事实和数值一致性 | novel-number rate | lower | 0.0007 [0.0000, 0.0015]; n=13/M=7 | 0.0274 [0.0000, 0.0822]; n=13/M=9 | -0.0003 [-0.0010, +0.0000]; n=10/M=6; dz=-0.408 | 1 | 1 |
| 事实和数值一致性 | rule-covered unsupported rate | lower | 0.8619 [0.6715, 0.9938]; n=13/M=7 | 0.7778 [0.5185, 0.9630]; n=13/M=9 | -0.0118 [-0.0386, +0.0030]; n=10/M=6; dz=-0.366 | 1 | 1 |
| 方向一致性 | macro direction coverage | higher | 0.0000 [0.0000, 0.0000]; n=33/M=11 | 0.0000 [0.0000, 0.0000]; n=33/M=11 | +0.0000 [+0.0000, +0.0000]; n=33/M=11; dz=NA | 1 | 1 |
| 方向一致性 | macro direction consistency | higher | NA [NA, NA]; n=0/M=0 | NA [NA, NA]; n=0/M=0 | NA [NA, NA]; n=0/M=0; dz=NA | NA | NA |
| 方向一致性 | policy stance coverage | higher | 0.0000 [0.0000, 0.0000]; n=21/M=7 | 0.0000 [0.0000, 0.0000]; n=21/M=7 | +0.0000 [+0.0000, +0.0000]; n=21/M=7; dz=NA | 1 | 1 |
| 方向一致性 | policy stance consistency | higher | NA [NA, NA]; n=0/M=0 | NA [NA, NA]; n=0/M=0 | NA [NA, NA]; n=0/M=0; dz=NA | NA | NA |

### 9.6 length-tolerant robustness：prospective-only 9 meetings / 27 rows

该表同样仅是 raw-completion robustness。

Holm family size：`K=20`。模型列为 marginal meeting-equal estimate；delta 仅基于共同 finite pairs，因此两模型 marginal 均值之差不一定逐位等于 delta。

| 组 | 指标 | 方向 | chk0 | chk1 cp200 | paired delta `chk1-chk0` | raw p | Holm p |
|---|---|---|---:|---:|---:|---:|---:|
| 语义相似性 | BERTScore precision | higher | 0.1396 [0.1307, 0.1491]; n=27/M=9 | 0.1387 [0.1289, 0.1492]; n=27/M=9 | -0.0008 [-0.0030, +0.0016]; n=27/M=9; dz=-0.224 | 0.5195 | 1 |
| 语义相似性 | BERTScore recall | higher | 0.1487 [0.1390, 0.1591]; n=27/M=9 | 0.1489 [0.1386, 0.1598]; n=27/M=9 | +0.0002 [-0.0021, +0.0028]; n=27/M=9; dz=0.052 | 0.9141 | 1 |
| 语义相似性 | BERTScore F1 | higher | 0.1439 [0.1347, 0.1538]; n=27/M=9 | 0.1435 [0.1336, 0.1542]; n=27/M=9 | -0.0004 [-0.0026, +0.0021]; n=27/M=9; dz=-0.093 | 0.7734 | 1 |
| 语义相似性 | independent MPNet cosine | higher | 0.0420 [0.0340, 0.0500]; n=27/M=9 | 0.0507 [0.0414, 0.0586]; n=27/M=9 | +0.0086 [+0.0011, +0.0164]; n=27/M=9; dz=0.701 | 0.05859 | 1 |
| 表层文本质量 | ROUGE-L F1 | higher | 0.0259 [0.0240, 0.0280]; n=27/M=9 | 0.0254 [0.0226, 0.0280]; n=27/M=9 | -0.0005 [-0.0032, +0.0019]; n=27/M=9; dz=-0.122 | 0.7266 | 1 |
| 表层文本质量 | generated token count | descriptive | 6,267.8 [5,980.3, 6,543.7]; n=27/M=9 | 6,142.9 [5,986.0, 6,331.2]; n=27/M=9 | -124.9 [-301.1, +59.9]; n=27/M=9; dz=-0.423 | 0.2188 | 1 |
| 表层文本质量 | length ratio | descriptive | 7.7822 [7.0789, 8.5391]; n=27/M=9 | 7.6588 [7.1594, 8.2581]; n=27/M=9 | -0.1234 [-0.3947, +0.2365]; n=27/M=9; dz=-0.235 | 0.5117 | 1 |
| 表层文本质量 | trigram repetition | lower | 0.9838 [0.9809, 0.9864]; n=27/M=9 | 0.9860 [0.9850, 0.9871]; n=27/M=9 | +0.0022 [+0.0001, +0.0045]; n=27/M=9; dz=0.588 | 0.1562 | 1 |
| 表层文本质量 | body-only format compliance | higher | 1.0000 [1.0000, 1.0000]; n=27/M=9 | 1.0000 [1.0000, 1.0000]; n=27/M=9 | +0.0000 [+0.0000, +0.0000]; n=27/M=9; dz=NA | 1 | 1 |
| 事实和数值一致性 | numeric value accuracy | higher | 0.9991 [0.9980, 1.0000]; n=7/M=5 | 0.9589 [0.8767, 1.0000]; n=7/M=6 | +0.0005 [+0.0000, +0.0015]; n=5/M=4; dz=0.500 | 1 | 1 |
| 事实和数值一致性 | evidence value coverage | higher | 0.0018 [0.0000, 0.0049]; n=27/M=9 | 0.0039 [0.0000, 0.0118]; n=27/M=9 | +0.0021 [-0.0046, +0.0109]; n=27/M=9; dz=0.170 | 1 | 1 |
| 事实和数值一致性 | unit accuracy | higher | 0.0598 [0.0000, 0.1618]; n=7/M=5 | 0.0417 [0.0000, 0.1250]; n=7/M=6 | -0.0027 [-0.0055, +0.0000]; n=5/M=4; dz=-0.861 | 0.5 | 1 |
| 事实和数值一致性 | time accuracy | higher | 1.0000 [1.0000, 1.0000]; n=3/M=2 | 1.0000 [1.0000, 1.0000]; n=4/M=3 | +0.0000 [+0.0000, +0.0000]; n=3/M=2; dz=NA | 1 | 1 |
| 事实和数值一致性 | novel-number rate | lower | 0.0009 [0.0000, 0.0020]; n=9/M=5 | 0.0352 [0.0000, 0.1057]; n=10/M=7 | -0.0005 [-0.0015, +0.0000]; n=7/M=4; dz=-0.500 | 1 | 1 |
| 事实和数值一致性 | rule-covered unsupported rate | lower | 0.8068 [0.5488, 0.9913]; n=9/M=5 | 0.7143 [0.4286, 0.9524]; n=10/M=7 | -0.0179 [-0.0580, +0.0044]; n=7/M=4; dz=-0.449 | 1 | 1 |
| 方向一致性 | macro direction coverage | higher | 0.0000 [0.0000, 0.0000]; n=27/M=9 | 0.0000 [0.0000, 0.0000]; n=27/M=9 | +0.0000 [+0.0000, +0.0000]; n=27/M=9; dz=NA | 1 | 1 |
| 方向一致性 | macro direction consistency | higher | NA [NA, NA]; n=0/M=0 | NA [NA, NA]; n=0/M=0 | NA [NA, NA]; n=0/M=0; dz=NA | NA | NA |
| 方向一致性 | policy stance coverage | higher | 0.0000 [0.0000, 0.0000]; n=15/M=5 | 0.0000 [0.0000, 0.0000]; n=15/M=5 | +0.0000 [+0.0000, +0.0000]; n=15/M=5; dz=NA | 1 | 1 |
| 方向一致性 | policy stance consistency | higher | NA [NA, NA]; n=0/M=0 | NA [NA, NA]; n=0/M=0 | NA [NA, NA]; n=0/M=0; dz=NA | NA | NA |

### 9.7 规则诊断与重复模式

all-11 冻结输入提供 54 个可判定方向 row-slots：growth-up 15、employment-up 6、unemployment-up/down/flat 为 15/9/9，inflation 0；policy stance 为 neutral 21、unavailable 12。prospective-9 有 48 个方向 slots：growth-up 15、employment-up 6、unemployment-up/down/flat 为 12/9/6；stance 为 neutral 15、unavailable 12。两模型 raw completion 在两个 subset 中均没有检测到 direction 或 policy-stance claim，因此 direction coverage 与 stance coverage 都为 0，consistency 因分母为 0 记为 `NA`。strict 中这些无效输出按预注册 worst-case 记 0。

| pooled LT raw-completion 诊断 | chk0 | chk1 cp200 |
|---|---:|---:|
| numeric claims | 6,118 | 8,237 |
| unit checks | 6,115 | 8,021 |
| time checks | 6 | 361 |
| number mentions / novel numbers | 6,136 / 3 | 9,320 / 216 |
| rule-covered / unsupported | 6,124 / 5,956 | 8,598 / 8,235 |
| unsupported: unit conflict | 5,953 | 8,019 |
| unsupported: number absent from evidence | 3 | 216 |
| direction / stance claims | 0 / 0 | 0 / 0 |

这些 pooled counts 只解释截断 raw completion 的错误构成，不能称为 final-answer claims，也不能替代 meeting-equal headline rate。尤其 LT 的 numeric accuracy marginal mean 受各行 eligibility 和 meeting 等权聚合影响，不能由 pooled unsupported 数量直接反推。

prospective-9 的 pooled raw-completion 诊断如下；两模型的 direction/stance mentioned 与 correct 仍全部为 0：

| prospective-9 pooled LT 诊断 | chk0 | chk1 cp200 |
|---|---:|---:|
| numeric / unit / time checks | 5,043 / 5,040 / 6 | 6,675 / 6,459 / 361 |
| number mentions / novel numbers | 5,061 / 3 | 7,758 / 216 |
| rule-covered / unsupported | 5,049 / 4,882 | 7,036 / 6,673 |
| unsupported: unit conflict | 4,879 | 6,457 |
| unsupported: number absent from evidence | 3 | 216 |

按单条 completion 内重复 occurrence 超额计数，最常见 trigram 为；括号内是至少出现一次该 trigram 的行数：

- chk0：`i need to` 8,048 次超额（33 rows）、`to use the` 2,834（9 rows）、`need to use` 2,776（10 rows）。
- chk1 cp200：`i need to` 6,003 次超额（33 rows）、`think about the` 3,399（10 rows）、`let me think` 3,105（10 rows）。

LT meeting-equal trigram repetition 为 chk0 `0.9840`、chk1 `0.9864`，paired delta `+0.0024`，95% CI `[+0.0005,+0.0045]`，raw `p=0.06836`、Holm `p=1.0`。unit accuracy 的 bootstrap CI 也排除 0，但 raw `p=0.25`、Holm `p=1.0`（仅 8 pairs/6 meetings）。CI 未做 multiplicity correction，不能据此宣称显著变化。

### 9.8 预注册解释

- execution validity gate：通过。两模型 33-row 矩阵完整、输入/seed 对齐、0 输入截断、strict/LT scorer 均 validated。
- strict all-11 favorable Holm-significant quality metrics：0；unfavorable：0。所有具效用方向的质量指标因双方同为无效输出而得到相同预注册惩罚；generated length 与 length ratio 是 descriptive，也不显著。
- 因此最终标签为 **inconclusive on this stress benchmark**。这不表示 chk0 与 chk1 等效；相反，它表明双方都是 100% delivery failure，均未能在 8,192 tokens 内完成压力任务，导致 strict benchmark 没有区分增量价值的能力。
- LT 中 BERTScore、MPNet、ROUGE-L、重复率、事实/数值与方向指标没有任何 Holm-adjusted 显著差异。LT 不能把 0/33 strict-valid 的失败改判为改善。
- 该 benchmark 是 raw D-1 all-evidence -> official Minutes；chk1 的训练产品合同是 atomic evidence -> analysis，存在严重 task mismatch。本结果只识别 `chk0 -> chk1 checkpoint-200`，不涉及 chk2/chk3/chk4，也不能单独决定是否进入 chk2。

## 10. 报告时必须保留的限制

1. 这是约 648–651 facts 的 all-evidence 输入到官方 Minutes 正文的端到端压力测试；chk1 的训练任务是较小 fact card/atomic topic 到 grounded FOMC analysis，输入规模、输出合同和目标文体均不完全一致。
2. 因 task mismatch，优异结果不能证明 chk1 的 atomic-analysis 质量，较差结果也不能单独否定 chk1。应另跑 task-aligned clean-v2 test 作为 in-domain 证据。
3. official Minutes reference 不是 teacher target；相似性只是一个 reference-based 视角。reference 未进入 prompt 不等于绝对不存在预训练/语义泄漏。
4. deterministic rules 只覆盖可解析数字、时间、方向、stance 和 custom rules，不是完整事实 Judge；未检出 unsupported 不等于全文事实无误。
5. 11 个 meeting clusters 的统计功效有限；Holm 后无显著差异不能证明模型等效。
6. strict 对 token-limit 施加预注册 worst-case penalty，主要反映“是否交付完整输出”；length-tolerant 可能把 raw reasoning 当正文，主要反映“未完成输出中是否含有内容”。二者估计对象不同。
7. checkpoint-200 的选择发生在本次测试之外。如果选择过程看过相同或相关基准，本结果可能存在 selection optimism；本报告不得把它描述为完全独立的 model-selection test。
8. 唯一可核验 parent-child lineage 是 `eval-chk0-base -> eval-chk1-clean-v2-lr1e6-cp200`。任何关于 chk2、chk3、chk4 或训练链整体的结论都超出本报告证据范围。
