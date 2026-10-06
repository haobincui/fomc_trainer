# 主文本结果与结论口径修复记录（2026-09-04）

本次采用 **chk-2 checkpoint 50 的 all391 冻结实验**作为主文本比较的唯一实验来源。其逐行分数能够精确复现现有主表的六项差值、t 统计量和百分比，模型身份也与当前训练章节及 checkpoint 选择记录一致。旧结论的 cp318 历史实验没有被改写为新结果，其原始文件保留用于溯源。

## 两份实验不能合并

| 项目 | 当前主结果 | 旧结论引用的实验 |
|---|---|---|
| Minutes 模型 | paper chk-2 cp50，active adapter over chk-1 cp200 | 内部实验 CHK3 cp318，旧版论文曾映射为 chk-2 |
| 会议 | 124，2009-01-28–2025-01-29 | 128，1993–2008 |
| 输入与生成 | 391 个样本 × 10 次 × 3 模型 = 11,730 | 128 个 prompt × 10 次 × 3 模型 = 3,840 |
| 参考文本 | 发布集中的合成教师改写 | 独立历史实验的确定性合成参考 |
| 评分 | raw_best_effort：保留恢复文本，只有空文本置零 | reliability-adjusted：四道 gate 任一失败即将语义分数置零 |
| chk-2 − chk-1 MPNet | +0.002365079252 | +0.061446138611 |
| chk-2 − chk-1 BERTScore-F1 | +0.000833272783 | +0.057130403677 |
| 推断 | 冻结 B=1,000 hierarchical bootstrap、会议 sign-flip + 六项 Holm；正文 t 检验为补充重算 | 旧版 B=2,000 bootstrap 与不同的检验 family |
| 允许解释 | training-only release 内的重构相似度 | 另行标记的旧 checkpoint 历史诊断；不能作为 cp50 的外部结果 |

当前输出根目录：

`output/evaluation/retrain_v2/paper_chk2_text_similarity_all391_chk0_chk1_chk2cp50_vllm_t06_p09_k10_b1000_v1_20260901/`

这里直接使用根目录中的 `evaluation_manifest.json`、`complete_evaluation_manifest.json`、`score_manifest.json`、`samples.jsonl`、`row_scores.jsonl`、`model_summary.csv`、`pairwise_contrasts.csv` 和 `bootstrap_draws.jsonl`。该目录没有 `scores/` 子目录。哈希及复算结果另存于[溯源清单](../Chapter2Results/minutes_primary_results_provenance_20260904.json)。

旧结果根目录：

`output/evaluation/main/chk3_external_holdout_1993_2008_n128_k10_20260812_v1/six_metric_bootstrap_k10_corrected_v2/`

旧版 [RQ1 报告](../../results/2026-08-14_rq1result_chk0_chk1_chk3_cp318_k10_bootstrap.md)描述其模型、样本和四门槛评分；同级 `six_metric_bootstrap_v1` 已被取代。本次未修改旧实验、归档论文或旧报告。

## 数据分配与样本外边界

当前发布集位于：

`dataset/processed/retrain_v2/chk2_chk1_final_analysis_synthetic_minutes_flash_official_reference_v6_downstream128_recovery_v1_20260831/`

`release_manifest.json` 的 SHA-256 为 `cb022dcd379e069e3d5ad54d7c7fdbc9e2ee29f958c85d5ff45d066f7728c097`，明确标记 `training_only=true`、`evaluation_eligible=false`。305/42/44 行来自 98/13/13 场 train/validation/test 会议；只有 305 个 train 样本参与参数更新，不能把全部 391 行称为已用于梯度训练。

原始 post-2008 总体的 128 场缩至 124 场发生在发布筛选阶段。`audits/evidence_ledger.jsonl`、`audits/rejections.jsonl` 与 `minutes_alignment/manifests/*.jsonl` 的会议集合差确认，四场训练会议的全部候选行被拒：2010-09-21（12 行）、2012-06-20（10 行）、2016-04-27（14 行）、2016-11-02（13 行）。49 行中有 37 个 source、11 个 style 和 1 个 generation reject；不是评价输出缺失造成的样本删减。

数据章节仍保留另一个 **1993–2008 年的 128 场历史总体**，但将其用途统一为当前 cp50 Core8/LOO 诊断：128 场 ×（1 Full + 8 deletion + 8 neutral replacement）× 10 = 21,760 次生成。它既不是主表的 124 场，也不能沿用旧实验的“三模型主评价 3,840 次”标签。该历史面板已被重复评价；其在 analytical/Minutes 分支中的外部性质不能推广到使用了 109 场 pre-2009 会议训练的 decision 分支。

## 显著性解释

六项补充 paired t 检验均与原主表一致，原主表数值无需更换。chk-1 相对 chk-0 的两项 t 检验均 `p<0.001`。chk-2 相对 chk-1 的 MPNet 和 BERTScore-F1 的双侧 t 检验 p 分别为 0.005200320791、0.003970158055，应写在 **1% 水平**显著，不能沿用旧结论的 0.1% 水平。

与此同时，冻结 hierarchical bootstrap 的两项增量 95% 区间为 `[-0.000555839664, 0.005238743536]` 和 `[-0.000089703082, 0.001823157836]`，均跨零。冻结会议均值 sign-flip 的六项 Holm 校正 p 分别为 0.005239947601、0.004939950600。正文和结论分别说明 t/sign-flip 与包含重复生成变动的 hierarchical bootstrap，不将不同程序的区间和 p 值当作同一检验。最终解释为幅度有限、推断依赖程序的发布集内重构改善，不能证明样本外泛化、事实准确性或正式 Minutes 等价性。

## 修改与复核

- `chapter2.tex`：统一评价方法、结果开头、主表标题与表注、统计解释和结论；保留已复现的主表数值，补充 bootstrap 区间及适用范围。
- `sections/dataset_construction.tex`：增加主结果面板和 128→124 的来源说明；将旧外部主评价表改为历史 Core8/LOO 人口与输出表。
- `sections/intro.tex`：将贡献中的旧 reliability-adjusted/数值保真描述改为当前重构相似度评价。
- `sections/model_training.tex`：将旧 cp318 替换为有记录支持的 cp50；同步纠正为 NF4 QLoRA 训练、BF16 不量化主文本推理、active adapter。

可重跑的[核查 notebook](minutes_primary_results_reconciliation_20260904.ipynb)核查原始评分完整性、来源哈希、会议均值、配对 t 检验及冻结推断输出；[完整统计 CSV](../Chapter2Results/minutes_primary_meeting_paired_tests_20260904.csv)保留未四舍五入值。复算使用现有评分，不重新生成文本或重跑嵌入模型。LaTeX 检查限于静态结构和引用检查；当前工作区没有论文根文件及可用的 LaTeX 编译引擎，未声称完成 PDF 编译。
