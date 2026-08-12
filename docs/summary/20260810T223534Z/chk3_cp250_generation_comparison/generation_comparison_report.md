# chk0 / chk1 / chk3 checkpoint-250 generation 对比

## 结论

chk3 checkpoint-250 **没有在这套冻结的 OOD Minutes 压力测试上消除退化**。chk0、chk1 和 chk3 都是 33/33 生成到 8,192-token 上限、0/33 正常停止、0/33 输出 `</think>`、0/33 strict 有效。三者均表现为高度周期性重复；chk3 的平均 word-trigram 重复率为 `0.988147`，略高于 chk1 的 `0.986368` 和 chk0 的 `0.984003`。

正式主分类是：

`stress_benchmark_inconclusive_all_artifacts_delivery_failure`

这表示三者在主测试上均发生 100% 交付失败，因此不能据此宣称 chk3 相比 chk1 语义改善、等价或产生了可识别的阶段增量。行为层面可以明确说：三者都发生了同类灾难性循环退化，cp250 没有恢复正常终止。

## 模型与精确 lineage

- chk0：`models/DeepSeek-R1-Distill-Llama-8B`
- chk1：clean-v2、LR `1e-6`、checkpoint-200 的精确 merge
- chk3：direct-from-chk1 SFT、checkpoint-250 的 evaluation-only 精确 merge
- chk3 adapter SHA-256：`396bdefef55c4521918a84df8e27abf8c82752eeadef3dcbb81188ffca332f45`
- chk3 merged directory fingerprint：`7a608cd4436323f7bce7b35893ad527177c89c5a2768c8fd6939f632b11007ac`
- 精确 merge 证明：291 个模型 tensor = 224 个适配 tensor + 67 个不变 tensor；448 个 LoRA tensor；0 mismatch。

## 测试合同与执行

- 冻结样本：11 个 meeting × 3 个 section = 33 条。
- 三个模型使用相同 prompt、样本顺序、逐样本 seed、reference 和 decoding 合同。
- `temperature=0`、`top_p=1`、`max_new_tokens=8192`、`max_model_len=24576`。
- chk3 生成仅使用物理 GPU0；manifest 记录 `CUDA_VISIBLE_DEVICES=0`、单张 NVIDIA A30。
- chk3 progress state 创建于 `2026-08-10T22:45:43Z`，生成于 `23:51:09Z` 完成；共 17 次持久化 commit、0 次恢复、无 OOM 或输入截断。
- 每批生成后 append + `fsync`，并原子更新 state；final JSONL、partial、frozen state 和 progress-bound manifest 的哈希链全部通过深验。
- strict (`strict-final-answer-v2`) 是主结果；length-tolerant (`length-tolerant-open-tags-v1`) 只分析未闭合 raw completion，不能称为有效答案。

## 生成交付与独立重复审计

| 模型 | Strict 有效 | `finish=length` | 8,192 cap | 正常 stop | `</think>` | Word-trigram 重复率 | Token-4gram 重复率 | 严格周期尾 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chk0 | 0/33 | 33/33 | 33/33 | 0/33 | 0/33 | 0.984003 | 0.983936 | 32/33 |
| chk1 cp200 | 0/33 | 33/33 | 33/33 | 0/33 | 0.986368 | 0.986530 | 33/33 |
| chk3 cp250 | 0/33 | 33/33 | 33/33 | 0/33 | 0.988147 | 0.987985 | 33/33 |

三组输入截断均为 0。chk3 相对 chk1 的 word-trigram 重复率均值增加 `+0.001779`；33 条中 23 条更高、10 条更低。该小差异不改变更重要的共同事实：三组均为 100% 长度上限退出、100% 缺少 reasoning boundary，并且近乎全部存在严格周期尾。

chk3 的代表性循环包括：

- `i need to` 总计出现 12,649 次，并成为 17/33 条输出的主导 trigram；chk0/chk1 分别为 8,081/6,036 次。
- 首条输出中一个 22-token 单元重复约 341 次。
- 第二条中一个 20-token 单元重复约 197 次。
- 第三条中一个 16-token 单元重复约 328 次。

## Length-tolerant raw-completion 诊断

下表只描述 raw completion，不能解读为 final-answer 质量：

| 模型 | BERTScore-F1 | MPNet cosine | ROUGE-L F1 | Evidence-value coverage | Direction coverage | Word-trigram repetition |
|---|---:|---:|---:|---:|---:|---:|
| chk0 | 0.144787 | 0.042456 | 0.026753 | 0.001538 | 0.000000 | 0.984003 |
| chk1 cp200 | 0.143931 | 0.048442 | 0.024522 | 0.003212 | 0.000000 | 0.986368 |
| chk3 cp250 | 0.142635 | 0.041712 | 0.026010 | 0.000000 | 0.000000 | 0.988147 |

chk3 raw text几乎没有可用的事实或方向覆盖。`numeric_value_accuracy` 等比例仅在极少数可匹配项上计算（chk0/chk1/chk3 eligible rows 分别只有 11/10/3），不能用其表面高值证明事实质量。

### 相邻 paired contrasts（all 11 meetings）

| Edge | BERTScore-F1 Δ | MPNet Δ | ROUGE-L Δ | Repetition Δ | Holm 结论 |
|---|---:|---:|---:|---:|---|
| chk0 → chk1 | -0.000856 | +0.005987 | -0.002231 | +0.002365 | 无显著指标 |
| chk1 → chk3 | -0.001296 | -0.006731 | +0.001488 | +0.001779 | 无显著指标 |

对 chk1→chk3：BERTScore-F1 的 95% CI 为 `[-0.003293, 0.000542]`；MPNet 为 `[-0.016853, 0.003901]`；ROUGE-L 为 `[-0.000218, 0.003166]`；重复率为 `[-0.001085, 0.004056]`。所有可计算的 Holm-adjusted p-value 均为 `1.0`；eligibility 不足的项目保持 N/A。每个 subset 的 LT family size 为 37；没有任何 Holm 显著结果。

方向与 policy-stance coverage 对三模型均为 0，因此 consistency 必须视为 N/A，而不是 0 分。

## 解释边界

该冻结测试是 raw D-1 evidence → official-Minutes-style section 的 OOD / end-to-end 压力测试。chk1 的训练任务是 atomic evidence → analysis；chk3 的训练任务是 analysis → Minutes-style paragraph。当前 prompt 同时改变了输入粒度和任务接口，因此：

- 结果足以否定“把 token 上限拉到 8,192 就会自然结束”。
- 结果足以说明 cp250 没有在这个压力测试上修复无限循环。
- 结果不能单独否定 cp250 在 task-aligned analysis → Minutes 输入上的能力；此前固定 3-case task-aligned probe 与本测试不是同一合同。
- 结果不能证明 chk3 与 chk1 等价，也不能把 LT 的 raw-text 小差异称为阶段增量。
- 这些结果不构成 canonical-DAG 晋升或下游训练授权。

## 封存证据

- chk3 final generation SHA-256：`bb6ecbb0e33da94b5bca851fefca6f3da0bf772a61abd26c1cc54bea15c999e5`
- chk3 generation manifest SHA-256：`44ebd996d2c237bf32719cbf2d4f600c2694e8b16825dca23fe46743414c2765`
- sealed summary SHA-256：`702fe1380bff17e2febcc9ef4e544fcc6c2ac9a5620796deae348aaad4b7b651`
- sealed summary payload SHA-256：`de36c737705b630fcb2245b890c4eec4992589012c34941dc4b0a647355836e8`
- strict audit SHA-256：`8d91ab6de16d15f1f10083ac261ecef55c2ad81a47ed6ea858b548f880dfbda9`
- LT audit SHA-256：`59fe8d84e2a29df687d1594e497b540ade61f0a71321f7cd9d85541345b68ea2`
- `chk3_cp250_exact_merge_lineage.json`
- `chk3_checkpoint_manifest.json`
- `three_leg_lineage_manifest.json`
- `generation_evaluation_summary.json`
- generation 和评分工件：`output/evaluation/main/checkpoint_generation/retrain_v2_chk0_chk1_chk3_cp250_20260810_v1/`

小 caveat：chk3 的 `2026-03-18::participants_views` 生成时记录为 8,192 tokens，但 decoded text 重新编码为 8,191；这是 decode→encode 非严格可逆造成的。长度退出判定使用生成时的原始计数，且该差异不影响周期和重复结论。
