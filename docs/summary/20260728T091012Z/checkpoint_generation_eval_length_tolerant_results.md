# 四工件长度容忍评分结果

英文论文级方法、计算公式、置信区间、配对检验与结果解释见
[`checkpoint_generation_eval_methods_formulas_results_en.md`](checkpoint_generation_eval_methods_formulas_results_en.md)。

本文件报告同一份冻结、无泄漏 33-row test set 上的后验稳健性分析。它复用
已经封存的四份 generation，不重新训练也不重新生成模型输出，并写入独立的
`scores_length_tolerant/`；原来的严格 `scores/` 结果未被覆盖。

## 评分规则

本次使用 `length-tolerant-open-tags-v1`：

- `finish_reason=length` 只作为截断审计标记，不判为无效；
- 不要求 `</think>`；
- 若存在大小写不敏感的首个 `<answer>`，评分文本为该标签后的非空内容，
  并允许剥离末尾 `</answer>`；
- 若不存在 `<answer>`，评分文本为完整的非空 raw completion；
- `<think>` 仅作审计标记；
- 空候选或 `input_was_truncated=true` 仍然无效。

当前 132 条 raw completions 均不含 `<answer>` 起始标签，也不含返回文本内的
`<think>` 起始标签，因此全部使用 `full_completion` 分支。这里没有依赖
可能缺失的 `</think>`。

## 有效性与截断审计

| 实际工件 | 验证父模型 | 全 11 会议有效输出 | prospective 9 会议有效输出 | 达到 completion limit（全量 / prospective） | 输入截断 |
|---|---|---:|---:|---:|---:|
| `eval-base`（`chk-0` base） | — | 33/33 | 27/27 | 33/33 / 27/27 | 0 |
| `eval-analysis-sft`（`chk-1`） | `chk-0` | 33/33 | 27/27 | 33/33 / 27/27 | 0 |
| archived GRPO（`chk-2` 角色） | 实际为 `chk-0` | 33/33 | 27/27 | 1/33 / 0/27 | 0 |
| recovered Minutes SFT（`chk-3` 角色） | 实际为 `chk-1` | 33/33 | 27/27 | 33/33 / 27/27 | 0 |

设计上的训练链仍是
`chk-0 → chk-1 → chk-2 → chk-3`：`chk-2` 应由 `chk-1` 继续进行
GRPO，`chk-3` 应由 `chk-2` 继续进行 Minutes SFT，以生成的 analysis
为输入、原始 official Minutes 为监督目标，强化 synthetic-text generation。
上表后两项是现存归档分支，并不实现这两条设计边，所以不能把其差异写成
设计链的阶段增量效应。

## 语义与表层文本质量

下表每个单元格为 `all-11 / prospective-only-9` 的
meeting-equal-weight mean。

| 指标 | `chk-0` base | `chk-1` analysis SFT | archived GRPO from `chk-0` | recovered Minutes SFT from `chk-1` |
|---|---:|---:|---:|---:|
| BERTScore F1 ↑ | 0.1448 / 0.1439 | 0.1448 / 0.1429 | 0.6444 / 0.6626 | 0.1472 / 0.1444 |
| MPNet cosine ↑ | 0.0425 / 0.0420 | 0.0426 / 0.0378 | 0.4518 / 0.4626 | 0.0474 / 0.0442 |
| ROUGE-L F1 ↑ | 0.0268 / 0.0259 | 0.0355 / 0.0345 | 0.1163 / 0.1189 | 0.0328 / 0.0317 |
| 生成 token 数（描述性） | 6261.6 / 6267.8 | 6689.7 / 6720.7 | 1271.6 / 1132.3 | 5671.0 / 5974.2 |
| 长度比（描述性） | 7.5872 / 7.7822 | 8.0275 / 8.1563 | 1.5655 / 1.3480 | 6.9199 / 7.5093 |
| trigram repetition ↓ | 0.9840 / 0.9838 | 0.9868 / 0.9874 | 0.1151 / 0.0945 | 0.9849 / 0.9892 |
| 格式合规率 ↑ | 1.0000 / 1.0000 | 0.9697 / 0.9630 | 0.0000 / 0.0000 | 1.0000 / 1.0000 |

格式合规率只检查预先声明的 body-only 禁止项、标签平衡和显式格式规则，
不是整体写作质量分数。三个长循环分支中的 base、analysis SFT 和 Minutes
SFT 虽可通过这一窄格式检查，但其约 0.985 的 trigram repetition 和约
6.9--8.0 倍参考长度显示输出质量仍然很差。archived GRPO 的 raw completion
包含 reasoning/结构标记，因此在本次“整段 completion”口径下格式合规率为
0。

## 事实、数值与方向一致性

下表仍为 `all-11 / prospective-only-9` 均值。括号内为相应 subset 中
有资格进入该条件指标的 row 数；`NA` 表示没有可判定的生成声明，而不是零分。

| 指标 | `chk-0` base | `chk-1` analysis SFT | archived GRPO from `chk-0` | recovered Minutes SFT from `chk-1` |
|---|---:|---:|---:|---:|
| 数值准确率 ↑ | 0.9993 (11) / 0.9991 (7) | 0.9603 (11) / 0.9551 (8) | 0.9519 (29) / 0.9665 (23) | 0.9835 (15) / 0.9921 (11) |
| evidence-value coverage ↑ | 0.0015 (33) / 0.0018 (27) | 0.0059 (33) / 0.0015 (27) | 0.0062 (33) / 0.0055 (27) | 0.0047 (33) / 0.0035 (27) |
| 单位准确率 ↑ | 0.0428 (11) / 0.0598 (7) | 0.3569 (11) / 0.2867 (8) | 0.1458 (29) / 0.1262 (23) | 0.2706 (14) / 0.2746 (10) |
| 时间准确率 ↑ | 1.0000 (3) / 1.0000 (3) | 1.0000 (3) / 1.0000 (3) | 0.5379 (33) / 0.5796 (27) | 0.7500 (8) / 0.8333 (6) |
| novel-number rate ↓ | 0.0007 (13) / 0.0009 (9) | 0.0222 (12) / 0.0216 (9) | 0.0345 (33) / 0.0233 (27) | 0.0165 (15) / 0.0079 (11) |
| rule-covered unsupported rate ↓ | 0.8619 (13) / 0.8068 (9) | 0.5456 (12) / 0.5733 (9) | 0.8215 (33) / 0.8153 (27) | 0.4824 (15) / 0.4341 (11) |
| 方向一致性 ↑ | NA (0) / NA (0) | NA (0) / NA (0) | 0.3464 (29) / 0.3122 (25) | NA (0) / NA (0) |
| 方向覆盖率 ↑ | 0.0000 (33) / 0.0000 (27) | 0.0000 (33) / 0.0000 (27) | 0.8485 (33) / 0.8889 (27) | 0.0000 (33) / 0.0000 (27) |
| policy-stance 一致性 ↑ | NA (0) / NA (0) | NA (0) / NA (0) | 0.0000 (17) / 0.0000 (11) | NA (0) / NA (0) |
| policy-stance 覆盖率 ↑ | 0.0000 (21) / 0.0000 (15) | 0.0000 (21) / 0.0000 (15) | 0.8095 (21) / 0.7333 (15) | 0.0000 (21) / 0.0000 (15) |

数值准确率是“已经生成且可匹配的数字”的条件准确率。四个工件的
evidence-value coverage 只有 0.15%--0.62%，因此不能用接近 1 的条件数值
准确率声称充分覆盖输入事实。archived GRPO 虽有明显更高的语义分数、较低
重复率和非零方向覆盖，但 unsupported rate 为 0.8215、方向一致性为
0.3464，且 policy-stance 一致性为 0。

按每个 subset 内全部可用预设指标组成一个 Holm family 后，没有
`p_Holm < 0.05` 的 archived-parent contrast。all-11 中 archived GRPO
相对 base 的 BERTScore、MPNet、ROUGE-L 和 repetition contrast 均为
原始 `p=0.0009766`、`p_Holm=0.0566406`。这些分支差异也不识别设计上的
`chk-1→chk-2` 或 `chk-2→chk-3` 效应。

## 可复现工件

运行入口：

```bash
run/eval_checkpoint_generation.sh score-length-tolerant
```

| 文件 | 行数 | SHA-256 |
|---|---:|---|
| `scores_length_tolerant/summary.jsonl` | 320 | `437d6c7e6fb48a303d8bb1a60bb2c7f73bac8eb25de2e0960f85cd114f9bae6a` |
| `scores_length_tolerant/contrasts.jsonl` | 132 | `138c54170c58f1afe316d4ae3324e221c7d0d284fd6abf4342f73812d640482c` |
| `scores_length_tolerant/row_scores.jsonl` | 132 | `aa5852e20a12b9b08856e9fd2a820257d74389c0a67d405c2bf4b9a15cf3c6a5` |
| `scores_length_tolerant/claim_checks.jsonl` | 45,522 | `7ce972e6c19aaf1a36516af8e17202b761e71050a67850d534a108d7cc34caea` |
| `scores_length_tolerant/audit.json` | 1 object | `37fa80ea65fa5cf2020ca1f5391ea7323858a89c3919dcb2733c5ea33fca84ba` |

Audit payload SHA-256 为
`1c30f3462f3e0c7f6c903283e3e774b7f350c9d5f53247098849eb76439df6d7`，
run-spec SHA-256 为
`4bef2d18028c15a948c753542e625488edb7cb737b4e206134cdbeb84af5b9a9`。
重复执行同一命令已通过 immutable reuse 校验，没有再次进行 encoder
inference。原严格 `scores/audit.json` 的文件 SHA-256 仍为
`4919cbeb2ef6f450925c4a47fe6b1926cbfd6429bbb1213f9d75291c619a58be`。
