# chk3 原生 analysis → Minutes N12 复测结果

## 结论

本次已改用 chk3 的原生合同：模型只接收封存 test split 中的 `analysis`，输出 native reasoning、唯一 `\n</think>\n`，随后输出单段 Minutes-style paragraph。chk0、chk1 cp200 与 chk3 cp250 使用相同 N12、prompt renderer、tokenizer、greedy 解码、seed、`max_new_tokens=3072`，并在物理 GPU0 上串行运行。

chk3 cp250 **不能判定为无退化**，不应据此晋升或用于下游训练：

- chk1：delivery `12/12`，在线 quality `12/12`，无触顶或周期尾。
- chk3 cp250：delivery `11/12`，在线 quality `10/12`；一条生成到 3,072 tokens 后触顶，无 EOS、无 `</think>`，并出现严格周期重复；另一条遗漏日期限定。
- chk0：在线 quality `8/12`；与 chk3 相同的贸易赤字样本也进入循环，说明 cp250 重新暴露了 chk1 已压住的 base attractor。

## 核心结果

| 模型 | Delivery | 在线 quality | EOS | Cap | 周期尾 | 数字保真 | 日期保真 | full 4-gram mean/max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| chk0 | 11/12 | 8/12 | 11/12 | 1/12 | 1/12 | 9/12 | 10/12 | 0.238157 / 0.910068 |
| chk1 cp200 | 12/12 | 12/12 | 12/12 | 0/12 | 0/12 | 12/12 | 12/12 | 0.113713 / 0.145614 |
| chk3 cp250 | 11/12 | 10/12 | 11/12 | 1/12 | 1/12 | 11/12 | 10/12 | 0.267695 / 0.911046 |

最关键失败样本为 `chk1-analysis-2024-06-12-b3c2fe77280d9f83`：

- chk1 在 698 generated tokens 正常 EOS，并给出唯一边界与完整 answer；
- chk3 cp250 生成满 3,072 tokens，boundary 为 0，full/tail token 4-gram repetition 为 `0.911046 / 0.947111`；
- 对应 teacher target 仅 843 completion tokens、健康闭合，因此不是 target 本身太长或含循环。

在 11 个双方均有 answer 的样本上，chk1→chk3 的 answer/reference ROUGE-L 平均增加 `0.056270`（8 胜、3 负）。这说明 cp250 在能正常交付时，表层改写更接近合成 reference；但该条件指标不能抵消 1/12 的灾难性交付失败。

## 完整合同 caveat

独立 CPU 审计还发现三模型的 reasoning 普遍包含系统提示禁止的任务元话语；因此按完整 prompt 合同，三模型均为 `0/12`。这与此前对 chk3 SFT reasoning 数据污染的审计一致，也说明在线结构门禁仍窄于实际训练合同。

N12 是确定性短/中/长分层诊断，不授权总体显著性或因果推断。Reference 是 DeepSeek 合成的 Minutes-style target，不是官方 Minutes；其中 1/12 为 analysis/reference identity 样本，non-identity N11 已单独报告。

## 工件

- 样本 manifest：`samples_n12.json`，SHA-256 `371d29601e343acf98cac6173663d842ead9acaa4a454991d512b3f00bf5ee77`
- Run root：`output/evaluation/main/chk3_native_analysis_to_minutes_n12_cp250_20260811_v1`
- 三模型 sealed comparison：`comparison.json`，SHA-256 `d89e2eafcabc763be15f0b5a405838f45615c394e1262546375bb4b819748baf`
- 完整生成：`chk0/generations.jsonl`、`chk1/generations.jsonl`、`chk3/generations.jsonl`
- 独立审计：`cpu_audit/native_analysis_to_minutes_report.md`
- 审计 summary payload SHA-256：`b78e00b097e0530b9d30fdd3b83dd9835c8be39724dd4554af668506d2824f69`

三份 run seal、comparison seal、模型权重身份、36/36 token-ID→text 解码、EOS、边界、输入/reference 绑定与指标均已独立复算通过。
