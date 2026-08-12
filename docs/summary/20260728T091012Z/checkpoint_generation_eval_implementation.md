# Chapter 2 四模型共同测试评估：实施记录

可直接用于论文改写的英文方法、公式、完整结果和结论边界见
[`checkpoint_generation_eval_methods_formulas_results_en.md`](checkpoint_generation_eval_methods_formulas_results_en.md)。

## 科学结论边界

论文中的设计流程继续明确为：

```text
chk-0 → chk-1 → chk-2 → chk-3
```

其中 intended `chk-2` 是以 `chk-1` 为初始化继续进行 GRPO；intended
`chk-3` 再以 `chk-2` 为初始化。`chk-3` 的设计输入是 `chk-2` 生成的
分析，监督答案目标是对应的原始官方 FOMC Minutes excerpt；官方 Minutes
不是模型输入。该阶段用于强化 Minutes-style synthetic text generation。

存档工件取证得到的实际可恢复图则是：

```text
chk-0 → analysis SFT
chk-0 → archived analysis GRPO
analysis SFT → archived Minutes SFT adapter
```

因此，本次实验是四个经过 provenance 验证的存档工件在共同测试集上的
比较，不是 `chk-0→chk-1→chk-2→chk-3` 的阶段增量效应估计，也没有重新
训练任何模型。

还需要区分评价任务与 `chk-3` 的设计训练任务。共同测试使用相同的 raw
`D−1 evidence → Minutes` prompt，是一个 common-input、end-to-end stress
benchmark；而 `chk-3` 的设计任务是 `generated analysis → Minutes`。
这种 task/input mismatch 使四工件具有可比输入，但该实验单独不能证明
`chk-3` 的任务内 analysis-to-Minutes rewriting 能力，也不能识别
`chk-2→chk-3` 的训练增量。

## Checkpoint 取证与恢复

最终 checkpoint manifest：

```text
output/checkpoints/recovered/
  llama_sft_synthetic_20250526_2_cp1668_recovered_v1_20260729/
  checkpoint_manifest.json
```

- manifest 文件 SHA-256：
  `1265d8d0fc0c0940fa1c66c9c5655940b238a0235e30bf581f55f33a551b4d18`
- manifest payload SHA-256：
  `16d29ec20f283c34b1cf69e7b2d5459a5a228e86466e0c2d3ae620e39168acc7`
- `training_performed=false`

| 评价 alias | 设计角色 | 验证父工件 | model SHA-256 |
|---|---|---|---|
| `eval-base` | `chk-0` | 无 | `bfb086cb87e9616e60805fc6f6f95826294b1de70b158cfea2d3e213df38ca11` |
| `eval-analysis-sft` | `chk-1` | `eval-base` | `51311501381e997e5011f6030a9584211cedadd9a03416fae9e53b2308e819b1` |
| `eval-legacy-grpo-from-chk0` | `chk-2` 的存档角色 | `eval-base` | `ac77c7d8851468267445f51413c5fbf3032e547a636ea60da2344447a552a6fe` |
| `eval-minutes-sft-from-chk1` | `chk-3` 的存档角色 | `eval-analysis-sft` | `ac176afc59f73eccbded8e02b4eec7dd1de8b252c0b20352f5074cb9e02077ea` |

`eval-analysis-sft` 的 parent 另有独立、sealed 的权重级核验证据：

```text
docs/summary/20260728T091012Z/
  eval_analysis_sft_exact_merge_lineage_20260730.json
```

- evidence 文件 SHA-256：
  `dafc1b00f5f74c2ef22a2ba1b7c2e3b938165cce80d57641e98436277ad6d4f4`
- evidence payload SHA-256：
  `49554cf15a3fa7ab4035b99a36c8db7eaf2cb805d82e4ca7a2eccfa8672d9a3d`
- 291 个模型张量全部核验；
- 227 个非 LoRA 张量与 base 逐值完全相等；
- 64 个 q/v 投影张量精确满足 FP32 PEFT 合并表达式
  `to_bf16(base + 2 × B@A)`；
- 128 个 LoRA A/B 张量全部且仅被使用，0 mismatch。

该证据严格证明 archived analysis-SFT weights 是指定 base 与指定 adapter
的精确合并结果；它不独立证明训练数据内容。正式 evaluator 会绑定该证据
及其所引用的 checkpoint manifest，而不是仅信任 alias 名称。

manifest 中前三个可用 alias 的 archived model/tokenizer 路径依赖本机
sibling backup `../fomc_trainer_back/fomc_trainer`；这是一项明确的本机
复现依赖，而不是可移植的下载地址。依赖由 manifest 中的 model/tokenizer
content hashes 绑定；迁移到其他机器时必须恢复 hash-identical 内容并重新
验证 manifest，不能仅凭同名目录替换。恢复后的 Minutes 模型则保存在本
项目版本目录中，同样由 content hash 绑定。

旧的 `llama_sft_synthetic_20250526` merged 目录与 decision-GRPO cp1200
具有完全相同的模型 payload，已登记为
`legacy-minutes-merged-corrupted` 且 `usable_for_evaluation=false`。
被该目录污染的未完成 LOO run 已写入独立 invalidation record；原 run
未被修改。

## 共同测试集

版本目录：

```text
output/evaluation/main/checkpoint_generation/checkpoint_eval_11_v1
```

- 11 个会议，范围为 2025-03-19 至 2026-06-17；
- 每个会议 3 个 Minutes sections，共 33 rows；
- prospective-only robustness 排除前两个会议，共 9 个会议、27 rows；
- prompt 只包含截至会议前一日的历史 vintage 机械证据；
- 官方 Minutes HTML 与 reference 单独保存，只供 scorer 使用；
- 无二次 LLM 摘要、无 reference prompt leakage。

除 manifest 字段断言外，对 33 个 prompt/reference pair 的归一化连续
20-token 序列做了逐行交集扫描：命中行 `0/33`、总命中 `0`。

因此，该数据集评价 raw `D−1 evidence → Minutes` 的端到端生成，不等同
于 intended `chk-3` design 的 `generated analysis → Minutes` 训练输入
分布。共同输入消除了四工件之间的 prompt 差异，但不会消除这种任务错配。

关键冻结摘要：

| Artifact | SHA-256 / payload SHA-256 |
|---|---|
| 93-series ALFRED snapshot manifest | 由 `source_snapshots/snapshot_manifest.json` 内嵌 integrity 绑定 |
| 286-row D-1 ledger | `c6dd74fe33ac4923374c49a1f11d744cdcabd32bea0b9eef683d85f1c3e4b7f4` |
| ledger manifest payload | `d29bec6ea67a70f4692434c2be267889a466141a590c525b230c05de5920ee82` |
| 33 prompts | `9e19c23d8a06a5ae2626f707bfa681fcb1789562bf43a6732128157faa0553de` |
| 33 references | `80026c603962b00a506b4021535d25ff7cddbf4526618c7a729b2242fc123a9a` |

ALFRED 若因 series definition change 返回多个 CSV 组成的 ZIP，获取器会
保存原始 ZIP、成员清单和 transport hash，并只在 vintage columns 构成
无重复、无缺失的精确分区时归一化为标准 CSV。路径穿越、未知成员、重复
列、缺列和超限压缩包均 fail closed。

## 推理与评分

- 四模型使用 byte-identical prompts；
- greedy decoding：temperature `0.0`、top-p `1.0`；
- 每行记录 deterministic seed，base seed `20260729`；
- `max_new_tokens=8192`、`max_model_len=24576`；
- 四套 tokenizer 的实际最长 prompt 为 12,475 tokens，最大请求总量
  20,667，均通过无截断 preflight；
- token-limit、空输出、解析失败和任何未证明无截断的 row 都保留在分母。

评分包括：

1. BERTScore（frozen RoBERTa-large）与独立 MPNet cosine；
2. ROUGE-L、长度、trigram repetition、格式合规；
3. 数值、单位、时间、novel number 和 rule-covered unsupported claim；
4. inflation、growth、employment、unemployment 方向及基于
   Federal-Funds-Rate 证据的 policy stance。

语义 encoder 由 sealed
`configs/main/checkpoint_eval_semantic_models.json` 绑定（文件 SHA-256
`639ca25cf4f5e695b0c6cfeeb47aba75d3e6dfeda1ad87510c23bb4f4a378bf7`，
payload SHA-256
`37a6896c388f965a572d3a3e99c2668a913513f2ac0883a0660634c807130445`）。
正式模式要求 RoBERTa-large BERTScore 与独立 MPNet 两个 backend 同时
存在并核对本地目录 hash；绕过 runner 也不能生成缺少任一语义指标却标记
为 validated 的结果。

数值规则按生成文本显示的小数位做确定性的 half-up 舍入匹配，同时严格
区分 percent、percentage point 和 basis point，不做隐式单位换算。每行
seed 必须等于 `sample-id-sha256-v1` 从 base seed 与 `sample_id` 派生的
值。新启动的 generation leg 还会封存 Python、vLLM、Transformers、
PyTorch、CUDA、GPU 和平台环境；本次已在该字段加入前启动的 leg 会在
正式 audit 中明确标记为 legacy-compatible、environment not recorded，
而不会伪造事后 metadata。

长文本按 sentence boundary 分块，缺失侧按零分处理并按 token 数加权，
不静默截断。统计汇总以 meeting 为 cluster，使用 10,000 次 bootstrap、
seed `20260729`，并在验证过的实际 parent-child contrasts 内执行 Holm
校正。LLM judge 与 human rating 均未使用。

## 复现入口

```bash
run/eval_checkpoint_generation.sh prepare
run/eval_checkpoint_generation.sh generate-all
run/eval_checkpoint_generation.sh score
run/eval_checkpoint_generation.sh score-length-tolerant
```

也可在两张独立 GPU 上并行运行单个工件：

```bash
run/eval_checkpoint_generation.sh generate eval-base 1
run/eval_checkpoint_generation.sh generate eval-analysis-sft 0
```

入口只包含数据构建、checkpoint 推理和评分，不包含训练命令。

## `../synthetic_text` legacy 结果审计

对 sibling 目录
`../synthetic_text/synthetic_text/output/cos_synthetic_20250526` 的审计
恢复了两组各 140 rows 的旧结果。仅看 `answer` 段的历史均值：

| legacy 输出标签 | BERTScore F1 | Llama-3 cosine |
|---|---:|---:|
| generated | 0.837389 | 0.852572 |
| raw | 0.838478 | 0.859593 |

这些数值只是 legacy point estimates，不是本次四工件共同测试的结果，
也不会回填到新表；它们也不能复现 Chapter 2 旧汇总表中的
`Cos_answer=0.929/0.9031`，说明 workbook 与论文表格来自不同运行或评价
规格。旧文件的 split membership 与 checkpoint provenance
不满足当前 manifest 标准；BERTScore 与 cosine 的 encoder/version 也未
按当前 frozen evaluator manifest 绑定。旧 Llama-3 cosine 实现调用
tokenizer 时设置 `truncation=True`，却没有逐行 truncation flag 或长文本
分块审计，因而可能静默截断。旧 bootstrap 从行级结果中每次随机抽取 20
行，既未按 meeting cluster 重采样，也没有当前的 paired、Holm-adjusted
推断设计。基于这些差异，140-row legacy BERTScore/Llama-3 cosine 仅作为
审计记录保留，不能与新评估的 RoBERTa-large BERTScore、MPNet cosine 或
四模型 contrasts 混合解释。

## 当前正式结果

四工件推理与评分已完成。所有 generation JSONL 均为 33 行，文件 hash
与各自 sealed manifest 一致：

| 工件 | 全 11 会议信息有效行 | prospective 9 会议信息有效行 | 主要失败原因 |
|---|---:|---:|---|
| `eval-base` | 0/33 | 0/27 | 33 行均达到 8,192-token completion limit |
| `eval-analysis-sft` | 0/33 | 0/27 | 33 行均达到 completion limit |
| `eval-legacy-grpo-from-chk0` | 32/33 | 27/27 | 1 行达到 completion limit |
| `eval-minutes-sft-from-chk1` | 0/33 | 0/27 | 33 行均达到 completion limit |

以下单元格为 `all-11 / prospective-only` 的 meeting-equal-weight mean。
对于 fatal generation，语义、ROUGE、格式及可覆盖的事实 headline 指标是
预注册 worst-case penalty，不是对截断文本另行评分。

| 指标 | base | analysis SFT | archived GRPO from chk-0 | recovered Minutes SFT from chk-1 |
|---|---:|---:|---:|---:|
| valid output | 0 / 0 | 0 / 0 | 0.9697 / 1.0000 | 0 / 0 |
| BERTScore F1 | 0 / 0 | 0 / 0 | 0.5148 / 0.5294 | 0 / 0 |
| MPNet cosine | 0 / 0 | 0 / 0 | 0.4401 / 0.4562 | 0 / 0 |
| ROUGE-L F1 | 0 / 0 | 0 / 0 | 0.0999 / 0.1049 | 0 / 0 |
| length ratio（描述性） | 7.5872 / 7.7822 | 8.0275 / 8.1563 | 0.7383 / 0.4964 | 6.9199 / 7.5093 |
| trigram repetition | 1 / 1 | 1 / 1 | 0.0464 / 0.0168 | 1 / 1 |
| format compliance | 0 / 0 | 0 / 0 | 0.2424 / 0.2963 | 0 / 0 |
| numeric-value accuracy | 0 / 0 | 0 / 0 | 0.9251 / 0.9624 | 0 / 0 |
| evidence-value coverage | 0 / 0 | 0 / 0 | 0.0032 / 0.0031 | 0 / 0 |
| unit accuracy | 0 / 0 | 0 / 0 | 0.1654 / 0.1621 | 0 / 0 |
| time accuracy | 0 / 0 | 0 / 0 | 0.1333 / 0.1250 | 0 / 0 |
| novel-number rate | 1 / 1 | 1 / 1 | 0.0726 / 0.0369 | 1 / 1 |
| rule-covered unsupported rate | 1 / 1 | 1 / 1 | 0.8582 / 0.8490 | 1 / 1 |
| direction consistency | 0 / 0 | 0 / 0 | 0.3831 / 0.3571 | 0 / 0 |
| direction coverage | 0 / 0 | 0 / 0 | 0.6919 / 0.7716 | 0 / 0 |
| policy-stance consistency | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| policy-stance coverage | 0 / 0 | 0 / 0 | 0.6667 / 0.6667 | 0 / 0 |

archived GRPO 的 all-11 BERTScore F1 95% cluster-bootstrap interval 为
`[0.4701, 0.5534]`，MPNet cosine 为 `[0.3982, 0.4758]`，ROUGE-L F1
为 `[0.0899, 0.1077]`。prospective-only 对应区间分别为
`[0.4981, 0.5615]`、`[0.4237, 0.4854]` 和 `[0.1000, 0.1105]`。

32 个有效 GRPO outputs 中只有 8 个满足 body-only format contract，
其余主要包含 section heading；格式违规是 nonfatal，仍照常计算语义、
表层和事实指标。numeric-value accuracy 的较高数值只在规则覆盖的已提及
数字上成立，同时 evidence-value coverage 约为 0.3%、rule-covered
unsupported rate 约为 85.8%，不能据此声称广泛事实一致性。

`scores/contrasts.jsonl` 只包含三个 verified archived-parent edges。
每个 subset 的 66 个预注册 contrast-metric tests 作为一个 Holm family；
所有 adjusted p-value 均不低于 0.05。all-11 中 archived-GRPO-vs-base
的 valid-output 与 BERTScore F1 均为 `p_Holm=0.0644531`。三个全 fatal
工件的相同 penalty 不表示模型等价，也不证明某阶段无效。

本次结果只说明：在这个固定 raw `D−1 evidence → Minutes` prompt 与固定
decoding contract 下，archived GRPO branch 的完成可靠性明显优于另外三
个归档工件；其完成文本仍表现出低格式合规、极低 evidence coverage、高
unsupported rate 和弱方向一致性。由于 archived GRPO 来自 `chk-0`，
recovered Minutes adapter 来自 `chk-1`，不能把这些差异解释为 intended
`chk-1→chk-2` 或 `chk-2→chk-3` 增量；也不能用 Minutes branch 在这个
out-of-task benchmark 上的全 fatal 结果否定 intended
analysis-to-Minutes rewriting 能力。

正式评分封存信息：

| 文件 | 行数 | SHA-256 |
|---|---:|---|
| `scores/summary.jsonl` | 312 | `ab3f8282f0932e16734dc5d9036867362e241377dcb6e090a043c885c00f2e0c` |
| `scores/contrasts.jsonl` | 132 | `e234aadb73723b06a457224380d008d9b50f3a070968b9eb501d38612d9ee9a4` |
| `scores/row_scores.jsonl` | 132 | `4306d3aa297456fb33b264ae766d72cd8a92bc792f9c5e52fa8b4e25424bbffc` |
| `scores/claim_checks.jsonl` | 44,540 | `f9836b7927b727ea99feb49ab7e556ce9be130244e397baf499edefca8487693` |
| `scores/audit.json` | 1 object | `4919cbeb2ef6f450925c4a47fe6b1926cbfd6429bbb1213f9d75291c619a58be` |

- audit status：`validated`、`immutable=true`；
- audit payload SHA-256：
  `8dfe9754770cb93ad70337a90813927779e3c3662d7499bbf41b46b4c5f33ca8`；
- run-spec SHA-256：
  `73dce47dfb9ba4d3f2b268f5676a9ada111836688b3e1e23c86072e7bb2820ff`；
- 重复执行 `run/eval_checkpoint_generation.sh score` 成功走 immutable
  reuse 校验，没有再次进行 semantic encoder inference；
- RoBERTa 与 MPNet audit 均记录 512-token 总限制、510-token 内容预算、
  sentence-boundary chunking、`silent_truncation=false`。

旧的约 490/140 条不同 split 聚合未混入上述正式结果。

## 长度容忍稳健性结果

按后续指定的口径，另行运行了
`length-tolerant-open-tags-v1`。该口径不要求 `</think>`，只把首个
`<answer>` 起始标签作为 answer 分界；不存在 `<answer>` 时评分完整非空
raw completion。`finish_reason=length` 只保留为审计标记，不再判无效。
结果单独封存在 `scores_length_tolerant/`，没有替换上面的严格结果。

当前 132 条 completion 均没有 `<answer>` 起始标签，因而都走
`full_completion` 分支：

| 工件 | 全 11 会议有效行 | prospective 9 会议有效行 | completion-limit 行（全量 / prospective） |
|---|---:|---:|---:|
| `eval-base` | 33/33 | 27/27 | 33/33 / 27/27 |
| `eval-analysis-sft` | 33/33 | 27/27 | 33/33 / 27/27 |
| `eval-legacy-grpo-from-chk0` | 33/33 | 27/27 | 1/33 / 0/27 |
| `eval-minutes-sft-from-chk1` | 33/33 | 27/27 | 33/33 / 27/27 |

以下为 `all-11 / prospective-only` 的 meeting-equal-weight mean：

| 指标 | base | analysis SFT | archived GRPO from chk-0 | recovered Minutes SFT from chk-1 |
|---|---:|---:|---:|---:|
| BERTScore F1 | 0.1448 / 0.1439 | 0.1448 / 0.1429 | 0.6444 / 0.6626 | 0.1472 / 0.1444 |
| MPNet cosine | 0.0425 / 0.0420 | 0.0426 / 0.0378 | 0.4518 / 0.4626 | 0.0474 / 0.0442 |
| ROUGE-L F1 | 0.0268 / 0.0259 | 0.0355 / 0.0345 | 0.1163 / 0.1189 | 0.0328 / 0.0317 |
| length ratio（描述性） | 7.5872 / 7.7822 | 8.0275 / 8.1563 | 1.5655 / 1.3480 | 6.9199 / 7.5093 |
| trigram repetition | 0.9840 / 0.9838 | 0.9868 / 0.9874 | 0.1151 / 0.0945 | 0.9849 / 0.9892 |
| format compliance | 1.0000 / 1.0000 | 0.9697 / 0.9630 | 0.0000 / 0.0000 | 1.0000 / 1.0000 |
| evidence-value coverage | 0.0015 / 0.0018 | 0.0059 / 0.0015 | 0.0062 / 0.0055 | 0.0047 / 0.0035 |
| rule-covered unsupported rate | 0.8619 / 0.8068 | 0.5456 / 0.5733 | 0.8215 / 0.8153 | 0.4824 / 0.4341 |
| direction coverage | 0 / 0 | 0 / 0 | 0.8485 / 0.8889 | 0 / 0 |
| policy-stance coverage | 0 / 0 | 0 / 0 | 0.8095 / 0.7333 | 0 / 0 |

base、analysis SFT 和 Minutes SFT 虽然在本口径下计为有效，但其约
0.985 的 trigram repetition 和约 6.9--8.0 倍长度比说明这些文本主要是
达到 token limit 的重复 reasoning，不能把 33/33 解释成 33/33 高质量
final answers。完整的条件指标分母、方向一致性表、解释与工件 hash 见
`checkpoint_generation_eval_length_tolerant_results.md`。
