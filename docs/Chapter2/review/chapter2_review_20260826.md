# Chapter 2 博士论文评审意见

**评审日期：** 2026-08-26

**评审对象：** Chapter 2 全章及其所有 section

**评审结论：** **Major Revision（大修）**

**当前状态：** 在核心证据版本统一、潜在信息泄漏排除、数据谱系补全、基线重算以及 LaTeX 阻断错误修复前，不建议将本章作为可提交或可答辩版本。

> 行号均对应 2026-08-26 工作区中的当前文件快照；后续修改后行号会自然移动。本意见把 `Chapter2/chapter2.tex` 与 `Chapter2/sections/*.tex` 视为论文正文。`Chapter2/results/` 中的报告仅用于交叉核对，不能替代在论文正文中完整报告方法和结果。

## 1. 评审范围与总体判断

本次逐节审阅覆盖了：

- `Chapter2/sections/intro.tex:1-93`：Introduction、Research Motivation、Research Questions、Contribution、Chapter Structure；
- `Chapter2/sections/literature_review.tex:1-30`：Synthetic Data、Structured Text Generation、Consistent Texts；
- `Chapter2/chapter2.tex:16-149`：Training Plan、Base Model、Llama 3.1 与 DeepSeek-R1-Distill 背景；
- `Chapter2/sections/dataset_construction.tex:1-1325`：语料、指标、teacher targets、Minutes rewriting、policy decision、data allocation、external evaluation；
- `Chapter2/sections/model_training.tex:1-408`：SFT、PPO/GRPO/DAPO、两条下游分支、PEFT/LoRA、训练配置；
- `Chapter2/chapter2.tex:150-503`：所有评价方法；
- `Chapter2/chapter2.tex:509-1582`：全部训练结果、文本重写结果、Core8 LOO、计量结果和决策结果；
- `Chapter2/chapter2.tex:1593-1679`：Conclusion 与 future research；
- 相关图表、交叉引用及 `Chapter2/results/` 中与正文结论对应的结果报告。

本章已经具备一个有潜力的研究框架：以共同的 analytical SFT checkpoint 为父模型，分出 Minutes-style rewriting 与 policy-decision 两条独立分支；数据按 meeting 进行时间切分；若干 prompt、reward 和审计规则写得比一般论文更透明；作者也主动报告了 GRPO 未通过 gate、LOO 不是 Shapley value、决策分支不消费 synthetic Minutes 等负面或限定性结果。

但目前存在一个先于所有语言润色的根本问题：**Methods、Results 与 Conclusion 显然没有冻结在同一版本的证据账本上。** 同一章中出现了不同 evaluation population、不同 sentiment model、不同金融市场 outcome、不同推断方法和不同决策测试集。再叠加潜在的 Minutes-derived topic-selection leakage、point-in-time 数据版本不明、训练数据运行时版本不一致，以及决策模型显著劣于朴素基线，当前核心结论无法被读者从正文独立复核。

## 2. 分节结论一览

| 部分 | 当前评价 | 最主要问题 |
|---|---|---|
| Introduction | 需重写研究边界 | 宣称 counterfactual scenarios、volatility-model enhancement、market impact 和 decision support，但实验没有直接实现这些目标 |
| Literature Review | 明显不足 | 仅约 30 行，未形成与研究设计相对应的文献缺口、构念定义和可检验假设 |
| Training Plan | 框架较清楚，但材料未同步 | shared-parent/two-branch 设计是优点；图、旧 checkpoint 名称与最终任务输入仍有残留冲突 |
| Base Model | 有事实性与技术性错误 | Base/Instruct 身份矛盾，架构公式及 tokenizer/template 描述不够准确 |
| Dataset Construction | 需要实质性重构/补证 | Minutes-derived topic mask 泄漏风险、实时 vintage 不明、row lineage 缺失、target 定义矛盾 |
| Model Training Details | 理论部分较好，复现性不足 | 最终 manifest 不完整；LoRA 参数量自相矛盾；真实 tokenizer、代码与硬件配置没有绑定 |
| Evaluation Methods | estimand 与指标定义未闭合 | semantic metric 实现冲突；相似度被过度解释；LOO 和计量识别不足；决策指标与基线不完整 |
| Training Results | 可作为诊断，不能支持稳定性结论 | checkpoint probes 极小、单一 seed、GRPO checkpoint 明确未通过 gates |
| Rewriting Results | 结果版本冲突 | 正文只报两个 t-test，结论却报告六指标 bootstrap/sign-flip 版本 |
| Core8 LOO Results | 仅能视为 exploratory sensitivity | reference 含被删 topic、主推断未完整展示、多重比较与 Monte Carlo 精度不足 |
| Econometric Results | 当前证据总体偏负面 | synthetic coefficients 未复现 official Minutes 长端系数，且正文与结论使用不同模型/outcome |
| Decision Results | 不支持预测能力主张 | 所有学习模型均未超过 always-Hold 基线；31-meeting post-hoc pooled 样本过小且时期/类别混杂 |
| Conclusion | 必须在结果冻结后重写 | 总结了正文未报告的 DistilBERT、FinBERT、Treasury、13-meeting test 等另一版本证据 |

## 3. 本章的主要优点

1. **最终的模型谱系概念清晰。** `chapter2.tex:19-55` 和 `model_training.tex:5-36` 将 analytical parent 与两个 downstream branches 分开，比把它们叙述成一个虚构的 end-to-end pipeline 更严谨。

2. **meeting-level chronological split 是正确方向。** `dataset_construction.tex:1123-1129` 明确禁止同一 meeting 跨 train/validation/test，避免了最直接的行级随机切分泄漏。

3. **统计单位意识较强。** `dataset_construction.tex:1116-1121` 主动区分 atomic rows、pairs、unique meetings、physical resamples 和 stochastic outputs；评价中先在 meeting 内平均重复生成，也是正确做法。

4. **prompt 与 reward 的透明度较好。** 数据构造部分展示了多类 prompt；`dataset_construction.tex:983-1105` 给出 reward 的 parser branches、公式和可核算示例。

5. **部分限制写得诚实。** 例如 LOO 不是完整 Shapley value（`chapter2.tex:263`），reward 不衡量 reasoning quality（`dataset_construction.tex:1103-1105`），base-model pretraining independence 不作保证（`dataset_construction.tex:1318-1322`），决策分支不消费 synthetic Minutes（`chapter2.tex:1596-1603`）。

6. **负结果没有完全被隐藏。** `chapter2.tex:819` 明确承认没有 GRPO checkpoint 通过全部 gate，决策结果也承认约四分之三的方向预测错误。这种透明度应保留，并提升到摘要性结论，而不是只放在局部限制中。

7. **大部分计数的内部算术是正确的。** 例如 `102+13+13=128`、`2,117-45=2,072`、`1,683+199+190=2,072`、`128×17×10=21,760`。当前问题主要不是简单加总，而是同一数字对应的 population、stage 与 provenance 没有统一。

## 4. 必须首先解决的 Critical Issues

### C1. Methods、Results 与 Conclusion 使用了不同版本的证据

这是全章最严重的问题，因为它使读者无法确定哪一套结果才是论文的正式结果。

**证据：**

- `chapter2.tex:893-969` 仅报告 MPNet cosine 与 BERTScore-F1 的 meeting-level paired t-test；`chapter2.tex:1603-1615` 却总结 exact numeric fidelity、三个 paired bootstrap confidence intervals、sign-flip p-values、structure/date/delivery failures 和 `random_decoding_regression` verdict。这些正式结果没有在当前 Results subsection 中完整呈现。
- `chapter2.tex:1623-1629` 总结所谓 “direct primary DistilBERT score-space diagnostic”、十个 gain intervals、correlation 和 level distance；正文没有对应方法、表或结果 subsection。
- `chapter2.tex:1148-1383` 的计量正文使用 Loughran–McDonald sentiment 与 Federal Funds futures；`chapter2.tex:1630-1638` 却总结 DistilBERT/FinBERT、Treasury outcome 和 `+0.005434` 的 current-sentiment coefficient difference。两者不是同一 estimand。
- `chapter2.tex:1390-1582` 只展示 31 个会议的 post-hoc pooled decision results；`chapter2.tex:1640-1652` 却总结 macro-F1、native strict-JSON delivery、replayed proxy reward 和已打开的 13-meeting held-out test，正文没有报告这些结果。
- `chapter2.tex:165` 宣称包含 meeting-level sentiment-score closeness，正文同样没有该结果。

`Chapter2/results/` 中确有若干 2026-08 月的较新报告，说明部分结论可能来自更新分析；但 supplementary report 的存在不能替代正文报告，也不能让读者自行猜测哪个版本生效。

**必须修改：**

1. 冻结一个唯一的 evidence release，给出 result ID、data manifest、model hash、tokenizer hash、generation manifest 和 analysis commit；
2. 以这一个 release 为依据，按相同顺序重写 Evaluation Methods、Empirical Results、Conclusion；
3. 删除或明确归档所有旧版本结果，不允许将旧表与新结论混合；
4. 每个结论句必须能反向定位到正文中的表、estimand、样本和不确定性区间。

### C2. 研究问题与实际实验没有对齐

Introduction 将研究描述为 scenario simulation、volatility-model enhancement、market impact preservation 和 policy decision support：见 `intro.tex:14,37,45,82,86`。但当前实验实际完成的是：

- 对历史 point-in-time indicator analyses 的 paragraph rewriting；
- 相对于 synthetic teacher target 的语义相似度；
- Full-vs-one-topic 删除的局部 sensitivity；
- 历史 sentiment–return association diagnostic；
- 一条不消费 synthetic Minutes 的独立 decision branch。

本章没有：

- 构造并评价明确的 counterfactual macro-financial scenarios；
- 将 synthetic text 输入 option-pricing 或 volatility model；
- 检验 option prices、implied volatility 或利率期权风险管理；
- 证明 synthetic text 保留真实 market impact；
- 证明对人类或机构有 actionable decision utility。

因此主问题 `intro.tex:45` 中 “market impact and policy-rate outcomes” 的提法超出证据；RQ2/RQ3 还要求复现 authentic FOMC deliberation/tone，但主要 reference 是 teacher-generated target，不是 official Minutes；RQ4 `intro.tex:56` 本身也不成句。

**建议采用更可实现的修改路径：** 将本章明确定位为 thesis 后续利率期权研究的 **methodological enabling chapter**，研究目标收窄为：

1. point-in-time evidence-to-analysis adaptation；
2. source-preserving Minutes-style rewriting；
3. target-relative prompt sensitivity；
4. exploratory historical association；
5. 独立、结果偏负面的 policy-action diagnostic。

若坚持 option/volatility 与 counterfactual contribution，则必须新增真实的 downstream experiment，而不能只改措辞。

### C3. Policy-decision 数据存在结构性的 Minutes-derived topic-selection leakage 风险

`dataset_construction.tex:74-76` 说明 indicator topics 是根据目标会议的 Minutes paragraphs 决定；`114-129` 再按 meeting-topic collapse；`411-414` 的 teacher 虽不看 Minutes 原文，但会看到由该过程选出的 `atomic_topic`；`725-747` 又把“该会议存在的 analyses”聚合成 decision brief。

这意味着，即使 prompt 中删除了目标 Minutes 文本、action 和 vote，**topic presence/missingness mask 本身仍可能来自会后发布的 Minutes，并编码该次会议讨论重点。** 目前平均每会约 `2,072/128=16.2` 个 topic，而不是固定 26-topic universe，因此这个风险不是纯理论上的。

`dataset_construction.tex:414` 的 “zero teacher-prompt reference leakage” 最多证明字段/文本层面没有显式 reference，不能证明 selection mechanism 没有泄漏。

**必须修改：**

- 使用固定 26-topic universe，或使用完全基于事前数据 availability 的预注册 topic inclusion rule；
- 对 fixed-topic、Minutes-derived topic mask、mask-only 三种输入做消融；
- 报告仅用 topic mask 预测 action/meeting identity 的 probe；
- 在泄漏排除前，不把 decision branch 称为严格的 target-neutral pre-meeting test。

### C4. `D-1` 规则不足以证明 point-in-time validity

`dataset_construction.tex:393-395` 声称 2,072 行均通过 point-in-time audit，但表中只看到 observation date、value 和笼统 availability basis，未说明 release timestamp、data vintage、timezone 与 revision policy（`461-474`）。GDP、PCE 等可修订序列若使用当前 FRED 历史值，就可能发生 revision leakage。

另外，pre-2009 的 109 个会议中有 25 个以 meeting-end date 匹配。两日会议的 `D-1` 可能已经是会议第一天，不能严格称为 pre-meeting information set。

**必须修改：**

- 对每个 series 提供 FRED/WRDS code、release calendar、vintage source、下载日期、timezone、频率、变换、季调、lookback 和 aggregation；
- 说明是否使用 ALFRED/实时 vintage；若没有，需重建 vintage 或将限制写入识别边界并做敏感性检验；
- 所有会议统一使用官方 start date；无法确认的 25 个会议应排除或单独做 robustness；
- 发布 row-level point-in-time audit，而不是只给汇总断言。

### C5. 数据谱系、监督目标与真实 runtime population 没有闭合

**(a) 行级谱系缺失。** `dataset_construction.tex:102-129` 明确承认没有 row-level manifest，因而无法连接：

`18,957 paragraphs → 6,266 filtered → 5,290 paragraphs / 5,397 raw-label rows → 4,887 controlled paragraphs / 5,327 assignments → 2,072 atomic analyses`。

补充语料 4,464 paragraphs 对应 44,116 assignments，平均约 9.88 labels/paragraph；主语料约 1.09。仅用定性说明无法排除 protocol drift 或 pipeline bug。

**(b) rewriting target 定义矛盾。** `dataset_construction.tex:527` 称 teacher reasoning 与 official Minutes paragraph 配对；`535-547,623-628,696-702` 则说明 target 是第二个 teacher 生成的 synthetic JSON answer，teacher 并未看到 official paragraph。`chapter2.tex:597` 又写成 synthetic rationale 后接 “a paragraph in the FOMC Minutes”。必须明确究竟是 official-text supervision 还是 teacher-style imitation。

**(c) canonical release 与实际 chk-1 runtime 不一致。** `dataset_construction.tex:513-514` 称 2,072 行用于 chk-1，但 `1131-1178` 显示 runtime 只有 1,743 行，真正训练仅 1,354 行；canonical training 1,683 与 runtime training 1,354 相差 329 行。更严重的是 semantic audit 未自动通过，却通过 undocumented override 继续使用（`1175-1178`）。chk-2 又使用完整 1,683 training pairs，其中可能包括未进入 chk-1 runtime 的行。

**必须修改：** 发布稳定 row ID、源文本 hash、meeting/section/paragraph/topic keys、每步输入/输出、rejection reason、classifier response、collapse 规则、最终 split 与 artifact hashes；披露 329 行差异和 override 的失败项目、阈值、批准规则与敏感性结果。

### C6. 128-meeting evaluation population 自相矛盾，直接影响是否存在训练泄漏

- `dataset_construction.tex:1279-1322` 将 external rewriting evaluation 定义为 1993-2008 的 128 个 regular meetings；
- `chapter2.tex:893-896` 却称相似度实验使用 128 个 post-2009 Minutes；
- `chapter2.tex:977` 又回到 1993-2008 的 128 个 external meetings。

如果 `chapter2.tex:894` 为真，post-2009 population 会与 task-specific training meetings 重叠，external generalization 主张失效；如果实际使用 1993-2008，则该句必须纠正。计量样本 2004-2015 也与 2009-2015 task-specific training period 部分重叠，只能称 mixed in-/out-of-sample historical diagnostic。

**必须修改：** 在正文给出 frozen meeting-ID appendix、split hash、generation manifest、每个 experiment 的 train/validation/test overlap matrix。

### C7. Policy-decision 模型没有击败最简单的基线

根据 `chapter2.tex:1470`，31 个 pooled meetings 包含 19 Hold、7 Cut、5 Hike。由正文表格可直接重算：

| 指标 | 最佳学习模型 | 永远预测 Hold/0 bp | 结论 |
|---|---:|---:|---|
| Direction accuracy | 24.84% | 61.29% | 学习模型显著更差 |
| Balanced accuracy | 32.73% | 33.33% | 未超过多数类基线；也未超过 uniform-random 的期望 33.33% |
| Meeting-level modal accuracy | 12.90% | 61.29% | 学习模型显著更差 |
| Exact-action accuracy | 最高约 19.35% | 61.29% | 学习模型显著更差 |

因此 `chapter2.tex:1574` 的 “strongest overall decision-quality performance” 最多只能表示“四个较弱 checkpoint 中的最高点估计”，不能表示存在 policy prediction utility。

此外，31-meeting pooling 是 post hoc；hike 全来自历史 panel，era 与 class 混杂；真正独立单位只有 31 个 meeting，Cut 只有 7、Hike 只有 5，50-bp strata 各只有一个 meeting。每会十次生成不能增加 meeting-level effective sample size。

**必须修改：**

- 加入 always-Hold、uniform random、empirical-frequency random、last-action/no-change、market-implied/Fed Funds futures 等基线；
- 分别报告 historical 与 post-cutoff frozen panels，不把它们合并成一个 temporal-generalization 结论；
- 报告 meeting-clustered interval/paired comparison、invalid/parse rate、macro-F1、exact action 和 delivery；
- 若将重复生成解释为预测分布，补充 Brier score、log score 与 calibration；
- 结论应明确写成“当前 decision branch 未显示超过朴素基线的预测能力”。

### C8. 当前计量证据更接近“未复现”，不能被表述为 market-impact validation

`chapter2.tex:1176-1252` 给出的系数为：

| Source | FF3 | FF6 | FF12 |
|---|---:|---:|---:|
| Official Minutes | -0.3490 | -0.7240 | -1.5981 |
| Model chk-2 | +0.0738 | +0.0022 | -0.3065 |

合成文本在 FF3、FF6 上反号，FF12 幅度也远小于 official Minutes；`chapter2.tex:1302-1346` 中差异区间在 FF3、FF6、FF12 均排除零，首先说明“不相同”，而不是“接近”。所谓 chk-2 相对 chk-0 的平均绝对距离优势仅 `0.6278-0.6223=0.0055`，未做直接检验，且主要由 FF12 驱动。不同 horizon 的原始 coefficient scale 也不同，不加标准化地等权平均缺乏清晰经济意义。

其他识别问题包括：

- `r=100 log(P_t/P_{t-1})` 的单位应是百分比回报，表格却称 coefficients/distances 为 basis points；
- 82 个有效 news-shock events 被置于约 2,300-2,600 个日度 observations 中，HC1 不能自然解决时间序列相关；
- 0.368 的固定参数来源与估计不确定性没有交代；
- `1162` 称十个 stochastic replicates，`1173` 又称五个；
- `RZ=Minutes-Statement` 对所有模型共享 official Statement，需有 Statement-only/null/shuffled-text baseline；
- synthetic Minutes 从未被市场观察，因此只能检验 historical association replication，不能验证真实 causal market impact。

**必须修改：** 将结论改为“未复现 official Minutes 的长端 coefficient pattern”；若目标是 equivalence，预先设定经济上可解释的 equivalence margin，并使用 TOST 或直接 paired distance-gain test；采用 HAC/event-cluster/block inference，完整交代 release window、timezone、contract roll、standardization 与 non-event encoding。

### C9. Model identity、LoRA 数量与训练 manifest 存在严重复现矛盾

**Model identity：** `chapter2.tex:70` 先正确说 DeepSeek checkpoint 由 Llama-3.1-8B Base 衍生，随后又说 Llama-3.1-Instruct-8B 是 backbone；`chapter2.tex:79` 又说从 LLaMA 3-8B-Instruct 开始。DeepSeek 官方模型卡将 `DeepSeek-R1-Distill-Llama-8B` 标为从 `Llama-3.1-8B-Base` 衍生，而不是 Instruct。应依据 [DeepSeek 官方仓库](https://github.com/deepseek-ai/DeepSeek-R1) 与实际 artifact config 统一描述。

**Tokenizer/chat template：** `model_training.tex:241-264` 报告的是一套泛化的 Llama 3 role-token 表，并写了不存在的 `<|end_id|>`；实际 released checkpoint 的 [官方 tokenizer config](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/blob/main/tokenizer_config.json) 使用 DeepSeek 风格的 `<｜User｜>`、`<｜Assistant｜>`、`<｜end▁of▁sentence｜>` 等 token，并在 generation prompt 中加入 `<think>`。`model_training.tex:268-280` 又说监督 target 只有 `</think>` closing marker，没有解释 opening marker 从何处注入。若实验使用自定义模板，必须报告模板全文、revision、special-token IDs、EOS/PAD、loss mask 和一条真实 tokenized example；否则 loss、终止行为和 checkpoint 均无法复现。

**LoRA parameter count：** `chapter2.tex:519` 声称 chk-1 使用 `r=32`，并作用于每层七个 projections；`chapter2.tex:574` 却报告只有 3,407,872 个 trainable parameters。按发布模型的 32 layers、hidden size 4096、intermediate size 14336 以及七 projection 计算，rank-32 LoRA 约为 83.9M 参数，不可能是 3.4M。更值得注意的是，3,407,872 **恰好对应全 32 层、rank-8、只作用于 `q_proj` 与 `v_proj`** 的参数量，强烈提示 `chapter2.tex:519` 可能误复制了另一训练阶段的配置。必须回查 adapter config、state-dict keys/shapes 和训练日志。真实维度可由 [官方发布 config](https://huggingface.co/deepseek-ai/DeepSeek-R1-Distill-Llama-8B/raw/main/config.json) 核验。

**Manifest：** `model_training.tex:363-368` 明确承认当前表不是完整 reproduction manifest，并将 optimizer、learning rate、epoch、batch、seed、sequence length、quantization 和 adapter settings 留空；但 `chapter2.tex:519` 又给出一套 chk-1 配置。两处必须合并到唯一、可验证配置，而不是分别叙述。

**必须修改：** 对每个 stage 给出 exact model/tokenizer revision、weight hash、chat template、special-token IDs、PEFT config、target modules/layers、trainable-parameter audit、optimizer/scheduler、precision/quantization、batch/accumulation、sequence length/truncation、seed、hardware/software、code commit、dataset manifest 与 checkpoint-selection manifest。

### C10. 当前源文件存在会阻断或破坏正式排版的错误

至少包括：

- `chapter2.tex:345`：错误嵌套 `\cite{\citet{loughran2011liability}}`；
- `chapter2.tex:519`：`$1\times10^{-6}` 的数学模式未闭合；
- `chapter2.tex:525-565`：表声明四列，但数据行只有三项，结果列错位/为空；
- `chapter2.tex:614-617`：表头缺 `\\`，四列声明与三列数据不一致；
- `chapter2.tex:579`：引用的 `.pdf` 文件不存在；当前目录只有 extensionless 文件和 `.png`；
- `chapter2.tex:735,814`：图路径与已设置的 `\graphicspath` 组合不正确；
- `chapter2.tex:812-816`：GRPO figure 没有 `\caption`，`\label` 不会正确绑定 figure counter；
- `chapter2.tex:660`：将 figure 写作 `Table~\ref{...}`；
- `dataset_construction.tex:401,514`：中文全角逗号 `，` 会破坏部分 pdfLaTeX 流程；
- `dataset_construction.tex:1273-1274`：`tab:ch2:chk3_training_composition` 与 `tab:ch2:chk3_bounded_resampling` 在项目中没有对应定义。

在源码可稳定编译、引用全部解析、表格列数正确、图片可加载之前，任何内容性修改都难以形成可交付版本。

## 5. 逐节详细意见

### 5.1 Chapter title

当前标题 `Synthetic Texts Generation with a Fine-tuned Large Language Model` 不自然。建议改为：

> **Synthetic Text Generation with a Fine-Tuned Large Language Model**

如果收窄贡献，可进一步用更准确的标题，例如：

> **Source-Grounded Synthetic Monetary-Policy Text Generation and Evaluation**

标题不应暗示完整的 counterfactual policy simulator，除非新增相应实验。

### 5.2 Introduction（`intro.tex:1-93`）

1. `intro.tex:4` 强调期货/期权近乎连续交易，但本章随后使用 release-day closes，并明确不识别 intraday response。该交易时段细节与实证设计关系有限；应缩短，或者真正加入 intraday event study。
2. `intro.tex:14` 的 “130 times since the 1990s” 需说明统计截止日、scheduled/unscheduled、target point/range 与零变动是否计入，并给出可复核来源。
3. `intro.tex:14,37` 声称 alternative scenarios，但数据和结果均为历史会议输入；没有给出 counterfactual design、plausibility constraints 或 scenario coverage。
4. `intro.tex:25-27` 对 two-branch 的描述较清楚，应成为全章主叙事；同时删除任何暗示 chk-2 输出进入 chk-3 的句子或图。
5. `intro.tex:45` 把 “market impact” 与 “policy-rate outcomes” 当成 synthetic text preservation；实际 market regression 只能是 association diagnostic，decision task 又是独立分支。
6. RQ2 的 “replicate structured deliberation” 与 RQ3 的 “authentic Minutes” 需要 official/human reference；当前 teacher target 无法直接回答。
7. `intro.tex:56` “How does model-based decision prediction on historical FOMC actions?” 语法和研究对象均不完整。
8. `intro.tex:76` 的 exact numeric preservation contribution 必须同时报告绝对水平；更新结果报告显示最佳 checkpoint 仅 590/1,280 通过，即 46.09%，不能只强调相对提升。
9. `intro.tex:82` 将 similarity、prompt sensitivity、econometric association 与 “realism” 捆绑过强。这些指标测量的是不同构念，应逐一定义，不宜组成未经验证的总体真实性概念。
10. `intro.tex:86` 的 “actionable insights” 与 “augment human decision-making” 没有用户研究、utility benchmark 或部署实验支持，应删除或显著降调。

建议在 Introduction 末尾给出明确的 hypothesis/estimand map：每个 RQ 对应输入、reference、population、metric、statistical unit 和可支持的结论边界。

### 5.3 Literature Review（`literature_review.tex:1-30`）

该部分对 PhD thesis chapter 明显过短，而且更像三段背景介绍，没有建立研究缺口。

需要补充至少以下六条文献链：

1. central-bank communication、FOMC Minutes/Statements、monetary-policy text 与资产价格；
2. 金融领域 LLM、FinBERT/BloombergGPT/finance benchmarks；
3. synthetic text/data 的构念效度、teacher bias、privacy 与 contamination；
4. controlled generation、faithfulness、factuality、numeric consistency 与 hallucination evaluation；
5. LLM monetary-policy decision/inference/forecasting 与 real-time information sets；
6. 文本经济含义的检验：event study、equivalence、external validity 和 human evaluation。

当前 bibliography 中似乎已有若干可利用但正文未讨论的条目，例如 `silva2025centralbank`、`tang2026mindshift`、`shah2025wcb`、`gossi2023finbert`、`kim2023analyzing`、`jung2016have`、`wu2023bloomberggpt`、`xie2024finben`。应核验文献内容后形成比较表，而不是简单增加 citation 数量。

还需明确定义并区分：style similarity、semantic alignment、source faithfulness、factual correctness、grounding、consistency、economic association、forecast accuracy、decision utility。`literature_review.tex:26-28` 目前把 coherence 与 faithfulness 合称 consistency，又暗示 fine-tuning 可减少 unsupported generation，但没有说明基于何种证据与评价设计。

文献综述最终应导出：现有研究做了什么、没有解决什么、本章的设计如何填补缺口、每项 contribution 相对哪篇文献新增了什么。当前版本没有形成这一逻辑链。

### 5.4 Training Plan（`chapter2.tex:16-67`）

shared-parent/two-branch 叙述是本章最清楚的结构之一，应保留。

但有三个问题：

1. paper-level 与 experimental checkpoint names 的映射增加认知负担。最好在所有 artifact 和正文中统一命名；若无法重跑，应给一个永久 mapping table，并确保后文不再出现旧名。
2. `model_training.tex:39-46` 说保留图仍显示旧 `chk-3/chk-4`，但当前 `stage1.png` 已显示 `chk-2/chk-3`，说明图和解释再次不同步。
3. `stage2.png` 的 decision panel 仍将 “Minutes” 画成输入，而最终任务声称输入是 target-neutral pre-meeting brief。这不是可由 caption 消除的 cosmetic issue，必须重画。

训练计划还应在一张表中绑定：task、parent、train population、target provenance、optimizer stage、selected checkpoint、selection data、final evaluation population，避免读者在多个 section 间拼接。

### 5.5 Base Model（`chapter2.tex:68-139`）

1. 统一 Base/Instruct 身份，参见 C9。章节应记录实际 artifact 的 repo/revision/hash，而不是泛指一个家族。
2. DeepSeek-R1-Distill-Llama-8B 已经经历 reasoning-response distillation/post-training，不是原始 pretrained foundation base。建议区分 `Llama-3.1-8B Base`（foundation backbone）、`DeepSeek-R1-Distill-Llama-8B`（released starting checkpoint）与本章 chk-1 至 chk-3（chapter-specific checkpoints）。
3. `chapter2.tex:79` 关于单张 24GB GPU 的可部署性取决于 inference/training、precision、quantization、context length、KV cache 与 batch size，不能作为无条件事实；该段还混用了 Llama 3 与 Llama 3.1 的型号和 context length。
4. `chapter2.tex:85` 对 MHA/MQA/GQA 的质量与内存比较过度简化，需要更严谨来源或删减为与本实验相关的配置说明。
5. `chapter2.tex:92-105` 的 ReLU/SwiGLU 公式和维度不严谨：转置、bias 与三维 `W` 定义均有问题；发布 config 还显示 MLP/attention bias 设置，不应写成通用含 bias 的实现公式。可改为标准逐元素形式 `SiLU(xW_g) \odot (xW_u)`，再说明 down projection。
6. “gate layers enhance long-context performance” 缺少直接支持。SwiGLU 的作用不应与 long context 简单画等号。
7. `chapter2.tex:115` 的 RoPE 描述过于简化；建议只说明实际配置与位置编码机制，不写教程式优劣判断。
8. `chapter2.tex:124-130` 将 few-shot CoT 错写成多轮对话；few-shot CoT 通常是在同一 prompt 中提供含中间推理的 demonstrations。全文 `Chain-of-Though` 应改为 `Chain-of-Thought`。
9. `chapter2.tex:139` 的 671B 是 DeepSeek-R1 的 total parameter count；MoE 计算规模还应区分 activated parameters per token，避免用 671B 直接比较运行成本。
10. `chapter2.tex:121-139` 的 prompt/CoT 教程较长、语言问题较多，却没有解释本研究为何要保存 teacher-generated reasoning prefix、如何审计其正确性。应压缩通识背景，增加与本实验直接相关的 model-card limitation、prompt convention 和 reasoning-target caveat。

### 5.6 Dataset Construction（`dataset_construction.tex:1-1325`）

除 C3-C6 外，还需处理以下问题：

1. **Classifier 不可复现。** `dataset_construction.tex:76-95` 只写 ChatGPT-4o，未给 exact snapshot、API date、prompt、temperature/top-p、seed、retry/schema；manual cleaning 未给 annotators、codebook、double coding、agreement 或抽样 precision/recall。
2. **Ontology 可能使用全样本设计。** 未说明 raw-label discovery 与人工 taxonomy 是否只在 102 个 training meetings 上完成。若 validation/test 语料参与 ontology construction，属于 test-informed preprocessing。
3. **Topic 概念混并。** PCE Price、Consumer Spending/PCE 并非同一 underlying indicator；Consumer Confidence 下含 retail sales、Home Prices 下含 rental vacancy、Corporate Bond Yields 下含 financial-conditions subindex、Bank Capital 下含 total assets。应称 broad topic grouping，并报告 alternative taxonomy robustness。
4. **Indicator 表不足以重建数据。** `193-395` 缺 series code、source owner、frequency、unit、seasonal adjustment、transform、missing/interpolation、outlier 与 meeting aggregation。
5. **Teacher 不可复现。** `411-415` 的 “DeepSeek-V4 Pro” 缺 provider、exact model ID/snapshot、API date、sampling settings、max tokens、seed、retry；`evidence_ids` 也不是逐 claim entailment audit。
6. **Meeting identity 只是字段删除，不是匿名。** observation dates、数值组合和 topic mask 可能唯一识别会议。应改称 identity field omitted，并做 meeting-identification probe。
7. **Current policy stance 不清。** 指标表含 federal funds target upper limit，但 `414` 又称 prompt 不含 current policy rate。action magnitude 相对于现有 stance 定义；应明确 lagged target range 是否进入模型，并做有无 current stance 的对照。
8. **Fidelity validator 未定义。** `634` 所称 prescribed checks 没有 normalization、容差、失败/重试数量。numeric multiset equality 与允许单位/数字形式转换可能冲突。
9. **第二 teacher stage 没有 attrition accounting。** 应报告候选、失败、retry、格式失败、numeric/date failure 和人工检查数；如零淘汰，也必须明示。
10. **Decision rationale teacher prompt 缺失。** `725-729` 描述了一个会看到 gold action 的 teacher，但正文没有展示该 prompt、output contract 和 validation gate。
11. **Core 与 supplementary decision briefs 不同域。** core 使用已生成 atomic analyses，supplement 使用带 `relative_changes` 的 structured evidence；应报告长度、schema、source classifier 以及 only-core/only-supplement ablation。
12. **Truncation strategy 缺失。** 多 topic brief 在 student context 下如何排序、去重、截断，以及哪些 topics 被系统删除，均未报告。
13. **Policy-action ground truth 不可审计。** `714-727` 没有说明 label 来自 Statement、implementation note 或本地数据库，也未说明 target point/range、range-width change、ZLB、emergency/unscheduled meeting、非常规政策、非标准幅度和 adjudication。
14. **109 vs 128 meetings 缺 attrition。** pre-2009 external population 有 128 meetings，但 decision supplement 只有 109；需逐会说明 19 个 exclusions。
15. **“Others” 隐藏一半 taxonomy。** 表只命名 13 topics，另外 13 个被聚合为 Others；应在正文或 appendix 展开完整 26 类。
16. **数据治理缺失。** 增加 Federal Reserve/FRED、WRDS 和 model-provider output 的许可/再分发说明；讨论仿 FOMC 权威文风的误用风险、teacher bias、ethics approval 是否不适用。

### 5.7 Model Training Details（`model_training.tex:1-408`）

理论部分中，SFT loss、PPO clipping、GRPO group-relative advantage 以及 DAPO token aggregation 的区分总体清楚，这是优点。但必须把“理论介绍”与“真实运行配置”更紧密地绑定。

1. `model_training.tex:241-264` 给出泛化的 LLaMA 3 special-token 表，但实际 DeepSeek artifact 可能修改 tokenizer/config/chat template；其中 `<|end_id|>` 也需核验。应报告 tokenizer revision 与实际 `apply_chat_template` 后 token IDs。
2. DeepSeek 官方 model card 对原始 distill model 的推荐使用包括避免 system prompt 等设置；本研究使用 system/user roles 并非天然错误，因为做了后训练，但应说明该偏离并报告 base-model prompt-template sensitivity。
3. `model_training.tex:268-280` 将 teacher-generated `reasoning_content` 作为监督 prefix。该文本不等于真实、可解释的模型推理；应称 rationale trace 或 teacher-generated rationale，并评价其 factuality/consistency。
4. `model_training.tex:298-309` 对 chk-2 target 的定义必须与 dataset section 统一：是 synthetic Minutes-style paragraph，而不是 official Minutes paragraph。
5. 标准 completion NLL 没有显式 numeric/date/claim-preservation loss；`296-309` 不应写成 optimization objective 本身保证 fidelity。更准确的说法是 target construction 与筛选规则旨在鼓励 preservation，而 hard gates 主要用于 checkpoint selection。
6. GRPO reward 只评 action 与 format，不评 rationale truth/economic coherence；因此 `chapter2.tex:680` 的 “reasoning capacity improving” 不成立，应改为 action-and-delivery proxy optimization。
7. Reward weights 缺少决策理论依据和敏感性分析。当前 strict wrong action 可因格式获得 0.05，而 fenced exact action 仅 0.2375；应解释这个 ordering 或重新设计。Training Details 至少应精确交叉引用 `dataset_construction.tex:983-1105` 的 reward 公式，并绑定 parser/version/hash。
8. `loss_type=dapo` 是 trainer/version-specific 实现；需报告 TRL/Transformers commit、`eta`、standard-deviation convention、reference-policy identity、old-policy update cadence、reward scaling 与 token masking，而不能只给抽象公式。
9. `chapter2.tex:22` 称 complete methods and realized configurations 已报告，最终 training table 却明示只是 verified subset（`model_training.tex:363-368`）。这不仅不够复现，而且是正文直接矛盾，参见 C9。

### 5.8 Evaluation Methods（`chapter2.tex:150-503`）

#### 5.8.1 Textual similarity

- `chapter2.tex:190` 说用 Llama hidden layers/tokenizer 形成 embedding，结果 `933` 却使用 `all-mpnet-base-v2`；必须统一实际实现。
- `185` 将 cosine 范围写成 0-1；原始 cosine 范围是 -1 到 1。
- target/candidate 的 `Y` 与 `\hat Y` 在不同公式中反转；点积只有 unit normalization 后才等于 cosine；`||Y||` 又同时像向量范数和 token 数。
- BERTScore 未在正文报告 model/revision/layer、IDF 与 baseline rescaling。较新的 result report 实际使用 `roberta-large` layer 17、无 IDF/无 rescaling，应将冻结配置写入正文。
- “reliability-adjusted” 必须给公式。当前较新结果在任一 hard gate 失败时把两个 semantic scores 置零，因此它衡量的是 delivery probability 与 conditional semantic quality 的混合物，不能称纯语义质量。
- synthetic teacher reference 不是 “high-quality” 的独立事实基准。Cosine/BERTScore 只能支持 target alignment，不能支持 factuality、hallucination-free 或 official Minutes equivalence。

建议至少加入 claim-level entailment/contradiction、entity-value-date binding、错误类型人工编码、双盲 human evaluation，以及 official-text/异构-teacher robustness。

#### 5.8.2 Statistical tests

- `228-238` 写单向假设，`963` 却使用 two-sided t-test；应预先确定方向、family 和主检验。
- Results 只给 mean difference/t-stat，没有模型绝对均值、SD、effect size 和 95% CI。
- 连续 meeting 可能有时间相关性；优先使用 meeting/year block bootstrap 或其他合适的 clustered inference。
- 若最终主分析是较新报告中的 hierarchical bootstrap + sign-flip + Holm，应删除旧 t-test 版本并完整展示六个指标的绝对水平与差值。

#### 5.8.3 Core8 LOO / “Sufficient Information”

这个标题过强。Full 与 mask output 都相对于包含完整 topic 的 synthetic reference 比较，所以 positive difference 部分来自设计本身；它只能衡量 target-relative sensitivity，不能证明信息“充分”、被正确理解或存在 causal importance。

需要加入 shuffled topic、wrong value、wrong sign、placebo topic、alternative neutral text 等负控。当前 Full-vs-one-topic 也不是 Shapley value；32 个 primary intervals 的 multiplicity 尚未调整，K adequacy 还标记 `increase_k_recommended`。主 hierarchical bootstrap CI 应展示在正文，而不是只展示 supplemental t-statistics。

#### 5.8.4 Econometric validation

应将标题从 validation/market impact 改为 exploratory historical association diagnostic，并按 C8 重新定义 outcome、unit、event window、error structure 与 equivalence target。

#### 5.8.5 Decision metrics

Methods 承诺 macro-F1、exact-action、delivery，但当前只完整定义 direction accuracy/recall/balanced accuracy。必须明确 invalid output denominator、JSON parsing、tie-breaking、modal rule、action magnitude、native delivery 与 replayed reward。基线应在方法阶段预注册，而不是结果出来后再选择。

### 5.9 Empirical Results: Training（`chapter2.tex:509-879`）

1. chk-1 checkpoint selection 仅有 8 个 probes；chk-2 仅 12 个 cases。它们可作 smoke test，不能提供 population-level reliability evidence。
2. chk-1 gate table `525-565` 列错位，无法看清实际 result；且正文说有八项条件，表中只列六个 row-level gate categories。
3. chk-2 cp318 因 12/12 通过而优于 cp250 的 10/12，存在小样本 selection optimism。应在独立冻结 selection set 上复核。
4. decision SFT cp38 被保留，是因为 cp39 在一个 “至少一个 sampled cut-direction correct output” gate 上失败；这种 gate 对随机 seed 非常敏感。
5. `chapter2.tex:809` 的 step-level SE 不是 training-seed uncertainty；所有模型只有一个 training seed，不能推断优化方法的稳定效果。
6. `chapter2.tex:819` 说没有 checkpoint 通过全部 gates，cp450 blind delivery 只有 0.30、阈值 0.75。因此 cp450 必须始终标作 failed-gate exploratory artifact，而非成功的 GRPO model。
7. “all 77 gates” 是明显 typo，应为七个 gates。
8. LoRA 数量矛盾和训练配置问题见 C9。

### 5.10 Empirical Results: Minutes rewriting（`chapter2.tex:893-969`）

该部分应整体替换为冻结后的 K=10 六指标分析。至少报告：

- 每个 model 的 structure/delivery、numeric fidelity、date fidelity、degeneration-free、MPNet、BERTScore 的绝对值和 95% CI；
- chk-2 minus chk-1 的 paired differences、bootstrap CI、raw 与 Holm-adjusted p-values；
- valid-only semantic scores 与 reliability-adjusted scores分开；
- failure taxonomy 与 examples；
- hard-gate operational verdict 与统计推断的区别。

必须正面呈现绝对弱点：较新结果报告中 paper-level chk-2 的 strict numeric fidelity 是 0.460938，即仅 590/1,280 generations 完全保留 numeric multiset，690 次失败。即使相对 chk-1 提升 5.625 percentage points，也不能写成已经实现高度 source preservation。

### 5.11 Empirical Results: Core8 LOO（`chapter2.tex:975-1141`）

1. 明确 evaluation population 是 1993-2008 还是其他时期，并绑定 manifest。
2. 正文说 K=10，但当前可见的独立 LOO supporting report 还包括 K=5 版本；必须唯一化并给 output hash。
3. 主要结果必须是 hierarchical bootstrap intervals；目前 table 重心仍是 t-stat/Holm stars。
4. 32 cells 中 16 个 CI above zero、16 个 cross zero 的整体图景比“CPI strongest”等排名更重要；应避免在 Monte Carlo precision 未充分时作确定性 ranking。
5. 不应将 positive sensitivity 解释为 factual grounding；增加上述负控后再判断。

### 5.12 Empirical Results: Econometric diagnostic（`chapter2.tex:1148-1383`）

除 C8 外，标题称 Treasury，但正文 outcome 是 Federal Funds futures；必须选择并统一。正文的主要发现应是：official Minutes 的较长 horizon coefficient 没有被 synthetic source 复现。若保留平均 absolute coefficient distance，应给尺度标准化、权重依据和直接差异检验；不能将最小描述性距离称为有效性验证。

### 5.13 Empirical Results: Decision prediction（`chapter2.tex:1390-1582`）

除 C7 外：

1. 分别冻结并展示各 panel；不要只说 “separately reported” 而不提供表或准确 appendix reference。
2. 报告每个 checkpoint 的 invalid/JSON failure、cap/EOS、direction、macro-F1、exact action、magnitude MAE、delivery 与 replayed reward，明确 denominator。
3. 解释 K=10 modal ties 如何处理。
4. historical 19-meeting panel 的选择规则、历史数据可能存在于 base-model pretraining 的 contamination 风险需要讨论。
5. 不能因某 checkpoint 在四个 learned artifacts 中“最好”而使用 winner language；结果的核心是所有模型都未超过简单基线，GRPO 也未实现 no-regression。

### 5.14 Conclusion（`chapter2.tex:1593-1679`）

`1596-1603` 对 two-branch boundary 的限定是准确的，应保留。但余下结论必须在 C1 的唯一 evidence release 冻结后重写。

推荐的结论顺序是：

1. 说明本章实际研究了什么，以及没有研究什么；
2. 报告 rewriting 的绝对性能与相对增量，同时突出 numeric fidelity 仍低；
3. 将 LOO 明确限定为 exploratory target-relative sensitivity；
4. 说明 econometric diagnostic 未建立 coefficient equivalence 或 causal market impact；
5. 明确 decision branch 未超过朴素基线、GRPO checkpoint 未通过 gates；
6. 将真正未完成的工作放到 future research，不要在 conclusion 中用未出现在 Results 的另一套数字补齐故事。

未来研究不能替代当前方法缺口。例如 multiple-testing adjustment、TOST、multiple seeds、reward ablation 与 chronological holdout 如果是当前主张成立的必要条件，就应在本章完成，不能全部推迟。

## 6. 思维逻辑与论证链专项审查

### 6.1 总体逻辑判断

本章不是“完全没有逻辑”，而是**同时使用了两条范围不同的论证链**，但没有把它们分开：

| 层次 | 实际叙事 |
|---|---|
| Introduction/RQ/Contribution 的宏大叙事 | 历史文本稀缺 → LLM 生成反事实 FOMC scenarios/sections → synthetic Minutes 具有真实性并保留 market impact/policy signals → 可用于 volatility modelling 与 decision support |
| 实际实验能够识别的窄叙事 | 历史 point-in-time evidence → teacher-generated analysis/rewriting target → student 对 synthetic target 的相对对齐；另做 LOO sensitivity、历史 association 和不消费 synthetic Minutes 的独立 action-classification branch |

真正的问题是，两条链之间的四个关键桥梁——**counterfactual validity、official/human authenticity、causal market impact、downstream decision utility**——均未由当前设计建立。因此后半章若按其谨慎措辞单独看，许多描述是合理的；但它们不能推出前半章设置的宏大结论。

最能反映真实 pipeline 的简化图是：

```text
历史证据 D
  ├─→ Teacher 1 生成 accepted analysis Z
  │       ├─→ Teacher 2 生成 rewriting target R
  │       │       └─→ chk-2 生成 R_hat，并与 R 比较
  │       └─→ Teacher 聚合 meeting brief B
  │               └─→ chk-3 预测 action A_hat
  └─→ 历史市场变量/政策标签用于后续诊断
```

这里有两个必须讲清楚的事实：

1. `chk-1 cp200` 是两条分支的共同**权重初始化**，但 chk-2 的训练输入明确是 teacher-generated accepted analysis，而不是 chk-1 inference output（`dataset_construction.tex:696-700`）；decision brief 也由 teacher-side construction 生成（`dataset_construction.tex:725-747`）。所以 shared parent 是参数谱系，不是已验证的运行时 `chk-1 output → downstream input` 数据流。
2. decision branch 不消费 chk-2 synthetic paragraph（`dataset_construction.tex:720-722`; `chapter2.tex:1596-1603`），因而其结果在逻辑上不能验证 synthetic Minutes 的 decision utility。

### 6.2 最关键的逻辑断裂清单

下表不是一般性的“还可以做更多 robustness”，而是逐项判断现有证据能否推出正文主张。

| # | 当前或隐含推断 | 缺失的逻辑前提 / 错误类型 | 当前最多可成立的结论 |
|---:|---|---|---|
| 1 | 历史 policy episodes 很少 → 生成更多文本即可缓解有效样本稀缺（`intro.tex:11-14`） | synthetic generations 是模型条件分布的输出，不是新的独立政策事件；K 次 decoding 也不是 K 个新的经济周期 | 可增加 stress-test inputs 或估计 decoder variability，不能自动增加真实世界 effective sample size |
| 2 | 历史 analysis-to-paragraph rewriting = counterfactual scenario simulation（`intro.tex:14,37,45`） | 没有 intervention-defined counterfactual inputs、可行性约束、out-of-support validation 或完整 meeting/document simulation；这是概念偷换 | 当前是 historical supplied-analysis rewriting，而非已验证的 counterfactual policy simulator |
| 3 | chk-1 是共同父模型 → 它在运行时提供 common analytical representation，且 shared-parent design 有效（`intro.tex:25,69-70`; `chapter2.tex:1598`） | 权重初始化不等于 inference data flow；也没有从 chk-0 直接初始化 chk-2/chk-3 的 matched ablation | 可以把 shared parent 作为工程设计描述，不能声称其增益已被识别 |
| 4 | teacher target 是 high-quality reference → 接近它等于接近 authentic/official Minutes（`chapter2.tex:163`; `dataset_construction.tex:535-547`） | teacher 同时生成监督与同构评价 reference，缺独立 gold standard；这是 closed-loop construct validation | 可以检验 teacher-target imitation/alignment，不能证明 official/human equivalence |
| 5 | MPNet/BERTScore 上升 → factual faithfulness、hallucination 减少或学到额外经济信息（`chapter2.tex:163-168,228-238,969`） | metric-to-construct 跳跃；主题相似文本仍可错配 entity-number-date-direction，两个 metric 也共享同一 output/reference | 只能说在冻结 reference 与 metric 下 semantic-alignment score 提高 |
| 6 | 删除 topic 后与 full reference 的相似度下降 → 该 topic 是“充分信息”、被模型正确使用且经济重要（`chapter2.tex:244-273,1130-1141`） | full reference 本身含被删 topic，正 delta 部分由设计机械诱导；敏感性不等于正确使用、充分性或因果重要性 | 只能说 output-to-target similarity 对该 prompt block 有局部敏感性 |
| 7 | prompt 不含 Minutes 文本/action/meeting ID → input 在信息上独立、无泄漏（`dataset_construction.tex:411-414,737-747`） | 形式字段删除不等于信息独立；Minutes-derived topic mask、日期与独特数值组合仍可识别 meeting/outcome | 最多称 same-meeting text/action fields omitted；在 mask/date probes 前不能称严格 reference-free/target-neutral |
| 8 | `observation_date ≤ D-1` → 数据在当时可知（`dataset_construction.tex:393-395,461-474`） | observation date ≠ release timestamp ≠ vintage timestamp；修订数据与两日会议 start/end date 破坏该蕴含 | 只有绑定 release/vintage/timezone 后才能声称 point-in-time valid |
| 9 | gold action 条件下生成的 teacher rationale → 独立的政策 reasoning ground truth（`dataset_construction.tex:725-729`） | rationale 知道结局后生成，是 outcome-conditioned rationalization；它不证明 blind reasoning 会导出该 action | 应称 action-conditioned explanatory supervision，不应作为 reasoning validity 证据 |
| 10 | synthetic sentiment 与历史 return 有关联 → synthetic text 真实并具有 market impact（`chapter2.tex:284-290,375`） | synthetic text 从未被市场观察；共同宏观输入、市场指标和 official Statement 可同时驱动 sentiment 与 return；相关不等于因果 | 只能称 historical association/coefficient-proximity diagnostic |
| 11 | 某 synthetic coefficient 显著或离 official 点估计最近 → 与 official effect 等价（`chapter2.tex:1165-1383`） | significance test 与 equivalence test 逻辑不同；非显著差异不证明等价，显著差异反而反对精确相等；最小距离也未必显著 | 未设 margin/TOST/direct gain test 时，只能报告描述性 coefficient 与 distance |
| 12 | cp38 在几个 learned states 中最好 → 模型具有 policy prediction ability（`chapter2.tex:1574`） | 组内排名不是有效性；模型没有超过 always-Hold/uniform-random 基线 | cp38 只是被比较 artifacts 中点估计最高者；当前证据支持“未显示超过朴素基线的预测力” |
| 13 | 31 meetings × 10 generations = 310 个决策样本（`chapter2.tex:1392-1394`） | Monte Carlo replicates 共享同一 meeting information；增加 K 只降低 decoder-distribution 估计误差，不增加独立事件数 | 独立外部单位仍是 31 meetings，其中 Hike=5、Cut=7 |
| 14 | GRPO reward/训练曲线改善 → reasoning capacity 改善或 GRPO 导致 format-decision trade-off（`chapter2.tex:680,809-819`） | reward 不评价 reasoning；只有一个 seed、无 ablation，且 cp450 未通过 gates | 只能描述这一条训练轨迹中两个 artifacts 的差异，不能归因于 GRPO 一般效应 |
| 15 | chk-2 相对 chk-1 显著提高 → rewriting 已达到实用或高 fidelity 水平（`chapter2.tex:969`） | relative improvement 不等于 absolute adequacy；缺少预设最低质量阈值，strict numeric pass 仍仅 46.09% | 可声称相对 target alignment 改善，同时必须说绝对 numeric fidelity 仍弱 |
| 16 | decision branch 的表现 → synthetic Minutes 保留 policy-outcome signal（`intro.tex:45,79,82`） | 两条 branch 没有该数据流，属于 non sequitur | decision task 只能作为独立 action-classification diagnostic |
| 17 | Conclusion 可以汇总 `Chapter2/results/` 的更新结果，即使 Results 正文仍是旧分析 | 论证链要求 Methods 定义、Results 展示、Conclusion 总结同一 evidence release | 未在正文出现的方法/结果不能作为本章已证明的结论 |

### 6.3 最严重的潜在泄漏逻辑

当前所谓 “reference-free” 或 “target-neutral” 的核心漏洞，可用以下潜在路径表示：

```text
实际会议讨论/政策结果
        ↓
会后发布的 Official Minutes M_h
        ↓
根据 M_h 选择本会 topic mask T_h
        ↓
D-1 数值 + T_h → atomic analyses/meeting brief B_h
        ↓
模型用 B_h 预测当次政策结果
```

即使 prompt 中完全删除 Minutes 原文、meeting ID 和 action 字段，`T_h` 仍可能是会后信息的载体。这条路径目前尚未被证明一定产生实质预测泄漏，但它足以使“信息独立”这一前提不成立，直到 fixed-topic、mask-only 与 meeting-identification probes 排除风险为止。正确表述应是“显式目标字段未进入 prompt”，而不是“没有 reference leakage”。

另一个容易被忽视的逻辑问题是，109 个 pre-2009 training meetings 与 19 个 historical external meetings 看起来来自旧 workbook 的 coverage/complement，而不是在完整 128-meeting roster 上事前冻结的统一 split。如果是否进入 workbook 与年份、政策 regime、action class 或数据完整度相关，external status 会带来 selection bias。应先重建完整 roster，再在 feature/label construction 前冻结 split。

此外，decision task 事实上混合了不同的信息契约：2009-2025 core 先从 Minutes-derived topics 生成 atomic analyses，1993-2008 supplement 使用另一套 structured topic evidence/prompt，external evaluation 又声称使用固定顺序的 Core8 blocks。source、era、topic coverage、current-policy-state availability 与 action class 因而可能纠缠。没有统一 schema 时，“模型根据同一种事前信息预测 action”这一 estimand 并不成立；应统一 fixed-topic contract，并报告 core-only、supplement-only 与 pooled-training ablation。

211 个 unique training meetings 被 class-dependent resampling 为 312 physical rows，本身可以是合理的训练处理，但它改变了模型学到的 class prior。若不做 prior correction/calibration，就不能把 stochastic generation frequency 当成自然政策概率；必须同时报告原始与重采样 class distribution，并用自然分布上的 proper scoring rules 评价。

### 6.4 Teacher pipeline 的循环性应如何准确描述

当前 teacher pipeline 并非对所有目标都“无效”。它对 **distillation fidelity** 是合理的：如果研究问题就是 student 能否模仿 teacher，则用 teacher target 评价有意义。逻辑错误发生在将这一闭环结果外推为 external validity：

```text
Teacher 生成 analysis/rewriting target
        ↓
Student 学习 teacher distribution
        ↓
Student 与同源 teacher reference 比较
        ↓
高分被解释为 authentic FOMC reasoning/Minutes realism   ← 此处跳跃
```

因此需要把四个构念永久拆开：

1. **Source faithfulness：** output 是否忠实于 supplied analysis；
2. **Teacher-target alignment：** output 是否接近 deterministic synthetic reference；
3. **External factuality：** claims 是否被独立数据支持；
4. **Institutional authenticity：** 是否达到 official/human Minutes 的专业质量。

现有 Cosine/BERTScore 主要覆盖第 2 项；numeric/date/structure gates 局部覆盖第 1 项；第 3、4 项尚没有足够独立证据。不能把四者统一命名为 “realism” 或 “hallucination validation”。

### 6.5 各 Research Question 的逻辑闭合状态

| Research Question | 当前是否闭合 | 原因 | 当前可支持的最窄回答 |
|---|---|---|---|
| Main RQ：生成 grounded FOMC-style sections，并保留 market impact/policy outcomes（`intro.tex:43-45`） | **否** | 实际为单 paragraph；无 counterfactual scenario、causal market impact 或 synthetic-Minutes-to-decision 数据流 | 当前只评价受控 rewriting、局部 sensitivity、历史 association 与独立 decision branch |
| RQ1：reason over indicators / extract policy insights（`intro.tex:50`） | **否** | chk-1 主要以 teacher rationale、loss、格式和 repetition gate 评价，没有独立 reasoning correctness/economic-coherence test；下游输入还绕过 chk-1 inference | 模型学习了 teacher-supervised evidence-to-analysis output contract，尚未证明可靠经济推理 |
| RQ2：replicate structured deliberation/trade-off language（`intro.tex:52`） | **否** | 单段相似度没有直接测量委员异质性、风险权衡或 deliberative discourse | 最多可检验单段 institutional-style target alignment |
| RQ3：match authentic Minutes tone/format/information（`intro.tex:54`） | **部分闭合** | 有 synthetic-target similarity 与部分 source gates，但 reference 不是 official/human gold，且 style/content 未拆开 | 可回答 teacher-target alignment 与局部 source preservation，不能回答 authentic equivalence |
| RQ4：historical policy-action prediction（`intro.tex:56`） | **描述性闭合，但答案偏负面** | 有 action results，但样本小、post-hoc pooling、无朴素基线于正文，且 retrospective/contamination 风险未排除 | cp38 在 learned states 中点估计最高，但未超过 always-Hold；不支持 decision utility |

Contribution 的逻辑状态也应相应调整：shared-parent two-branch 是一个**设计选择**，没有 direct-initialization ablation 就不是已证明有效的设计创新；multi-dimensional evaluation 是一个**评价框架**，不能自动证明 realism；independent decision branch 可以作为透明的负面诊断，但不能成为 synthetic paragraph 的 downstream validation。

### 6.6 一条能够成立的重写逻辑链

如果不新增大规模实验，建议把整章重构为以下论证：

1. **动机收窄。** 历史政策文本有限，促使研究受控的 conditional rewriting 与 diagnostic evaluation；不声称 synthetic outputs 是新的独立政策事件，也不声称完整模拟 counterfactual FOMC meetings。
2. **准确界定 artifact。** teacher-generated analysis、teacher rewriting target、model-generated paragraph、official Minutes、decision brief 分别命名，绝不混称 synthetic Minutes。
3. **准确界定 lineage。** chk-1 是 common parent checkpoint/parameter initialization；chk-2 与 chk-3 是独立 branches；不暗示 chk-1 inference output 被下游实际消费。
4. **设定可检验主问题。** 可改为：

   > Does task-specific supervised post-training improve source fidelity and frozen-target alignment in analysis-to-FOMC-Minutes-style paragraph rewriting relative to the immediate parent checkpoint on temporally external meetings?

5. **将次级问题与证据一一绑定。**

   - six rewriting metrics → source preservation 与 synthetic-target alignment；
   - Core8 interventions → target-relative prompt sensitivity；
   - sentiment regression → historical association/coefficient proximity；
   - independent decision branch → retrospective action quality 与 delivery；
   - 不用任何一项替代 independent factuality、official authenticity 或 causal impact。

6. **按结果真实回答。** chk-2 相对 parent 有若干 alignment/numeric 增益，但绝对 numeric fidelity 仍弱且有 hard-gate regression；LOO 结果混合；计量检验不建立 equivalence；decision branch 不超过朴素基线；GRPO 没有 no-regression improvement。
7. **贡献降到证据同尺度。** 可以贡献透明的 two-branch workflow、paired external artifact evaluation、target-relative intervention diagnostic、reward/delivery/action 分离报告及负结果审计；不应声称 realistic counterfactual simulator、official equivalence、market impact、policy decision support 或 GRPO reasoning improvement。

按这条链，整章会形成真正闭环：

```text
明确的受控任务
  → 与任务匹配的数据和训练更新
  → 冻结且可比较的模型/样本
  → 与构念匹配的指标和基线
  → 限定性的结果
  → 不越过证据边界的结论
```

### 6.7 若坚持原来的宏大主张，必须新增哪些逻辑桥梁

如果作者希望保留 “counterfactual scenarios—market impact—decision support” 的主叙事，仅改写措辞不够，至少需要：

1. 预先定义的 counterfactual intervention set、经济可行性约束与 out-of-distribution stress tests；
2. 独立专家/official reference 对 factuality、style 与 deliberation 的盲评；
3. 真正的 end-to-end `indicators → chk-1 output → chk-2 paragraph` 测试及误差传播分析；
4. raw indicators、chk-1 analysis、chk-2 synthetic paragraph 进入同一个 frozen decision model 的输入臂比较；
5. 与 option prices、implied volatility 或 volatility model 直接连接的 downstream experiment；
6. 若声称 market impact，使用真实 exposure/发布设计或其他可信因果识别，而不是把未发布 synthetic text 回归到历史 return；
7. direct-from-chk-0 与 shared-parent initialization 的 matched ablation；
8. prospective/knowledge-cutoff-safe meetings、统一信息契约与朴素/市场基线。

### 6.8 建议采用的推断用语

| 当前高风险用语 | 建议替换 |
|---|---|
| demonstrates faithfulness / detects hallucination | improves frozen synthetic-target alignment; source faithfulness is assessed only by specified gates |
| information sufficiency / indicator importance | target-relative sensitivity to a prompt block |
| validates market impact / economic impact | reports a historical sentiment–return association diagnostic |
| equivalent to / mirrors official Minutes | descriptively closer under the specified metric；等价须另做 margin-based test |
| predicts policy decisions / decision support | retrospective action classification；只有超过基线后才能讨论 predictive utility |
| GRPO improves reasoning | post-GRPO artifact differs from its SFT parent on the measured action/delivery proxies |
| reference-free / no leakage | same-meeting text and explicit target fields are omitted；结构性独立性需另行验证 |
| common analytical representation | common parent checkpoint/parameter initialization |

### 6.9 逻辑上已经写对、应当保留并上移的限定

以下限定本身逻辑严谨，建议把它们前移到 Introduction、RQ、Contribution 和 Evaluation Methods，而不是只留在结果末尾：

- `chapter2.tex:1118-1122`：LOO 不是 causal topic importance；
- `chapter2.tex:1133`：两种 intervention 的差异不能作因果解释；
- `chapter2.tex:1383` 后半：没有 direct distance-gain test，不能声称一般性 coefficient-proximity improvement；
- `chapter2.tex:1397-1399,1582`：pooled analysis 是 post-hoc descriptive，重复生成不是独立 meetings；
- `chapter2.tex:1596-1603`：decision branch 不验证 synthetic Minutes utility；
- `chapter2.tex:1613-1615`：synthetic targets 不建立 official/human equivalence；
- `chapter2.tex:1636-1638`：不支持 equivalence 或 causal market impact；
- `chapter2.tex:1648-1652`：单 seed、无 ablation，不能声称 causal GRPO effect。

**逻辑专项结论：** 当前章最稳妥的学术定位不是“已验证的 FOMC counterfactual simulation/decision-support system”，而是“一个带有明确失败记录的 teacher-supervised post-training 与 artifact-evaluation study”。只要以这个定位反向重写 Introduction、RQs、Contributions 和前半部分 Evaluation motivation，论证可以闭合；若坚持现有宏大定位，则必须补上 6.7 所列的新实验。

## 7. 可复现性与研究治理清单

正式提交前，建议建立一个 chapter-level release bundle，至少包括：

- 唯一的数据字典与 93 个 indicator 的 series metadata；
- 原始 paragraph 至 final row 的完整 lineage manifest；
- meeting IDs、meeting start/end dates、release dates、vintage timestamps 与 split hashes；
- classifier/teacher 的 exact model IDs、prompts、sampling、retry logs 与 raw outputs；
- official Minutes 与 synthetic target 的清晰 provenance；
- 每个 model checkpoint 的 weight hash、tokenizer hash、chat template 与 PEFT config；
- 每个 stage 的完整 training config、software/hardware、seed 和 code commit；
- checkpoint-selection data 与 final-test data 的隔离证明；
- generation-level outputs、row seeds、parsing results、hard-gate results；
- statistical-analysis scripts、主/次检验 family、bootstrap/sign-flip implementation；
- policy-action label source、coding rules、edge cases、adjudication 与 class counts；
- 数据许可、再分发限制、model-provider terms 与伦理/误用声明。

建议在正文增加一张“RQ-to-evidence”表：

| RQ | Population | Input | Reference/label | Primary metric | Statistical unit | 可支持的结论 |
|---|---|---|---|---|---|---|
| Rewriting | 冻结后唯一时期 | source analysis | synthetic teacher target | six frozen metrics | meeting | teacher-target alignment 与 source-preservation performance |
| Topic sensitivity | 相同 frozen population | Full vs intervention | full synthetic target | paired delta | meeting | target-relative local sensitivity |
| Historical association | 明确 event sample | sentiment source | futures/return outcome | coefficient/equivalence | event/time block | association replication，不是 causal impact |
| Decision | 独立 frozen meetings | pre-meeting brief | audited policy label | balanced accuracy + baselines | meeting | 相对基线的 policy-action performance |

## 8. 表达、结构与排版意见

### 8.1 章节结构

- 大量 `\subsection*` 后接 `\label`。starred headings 不递增计数器，也通常不进入目录，label 可能指向上一编号。建议统一使用编号 subsection，或显式处理 ToC 与 anchors。
- Base-model 架构和 prompting 的教程篇幅偏长，真正的数据 provenance、identification 与 reproducibility 反而不足。建议删减通识教材内容，把篇幅移到核心方法。
- 同一信息在 Training Plan、Dataset、Model Training、Training Results 多次重复，且每次略有差异。采用一张 canonical lineage/config table 后，其他位置只引用。

### 8.2 术语统一

统一以下术语和大小写：

- DeepSeek-R1-Distill-Llama-8B；
- Llama 3.1，不要在 LLaMA 3、Llama3、LLAMA3 之间切换；
- FOMC Minutes-style paragraph rewriting；
- `Model chk-0/1/2/3` 与 experimental artifact names；
- validation 与 evaluation；
- meeting、paragraph、assignment、atomic analysis、pair、generation 等统计单位。

避免把 synthetic teacher target 称为 authentic/official/high-quality reference，除非有独立 human/official validation。

### 8.3 代表性语言问题

全章需要专业英文编辑。代表性错误包括：

- `chapter2.tex:153`：stochastic decoding 不等于总是选择 highest-probability token；
- `chapter2.tex:160`：F1 不是 classification accuracy，而是 precision/recall harmonic mean；
- `chapter2.tex:179`：`ROGUE` 应为 ROUGE，`BOG` 应为 BOW；
- `chapter2.tex:522`：`to make sure the generation stability`、`Meanwhile, During` 与 comma splice；
- `chapter2.tex:597`：`Model chk-1checkpoint 200`；
- `chapter2.tex:600`：`summaried`；
- `chapter2.tex:680`：`policy decision making prediction task`；
- `chapter2.tex:741`：`passsed`；
- `chapter2.tex:981`：`A positive estimates`；
- `chapter2.tex:1150`：`TThis`、`a empirical model`；
- `dataset_construction.tex:20`：`128 meeting`；
- `dataset_construction.tex:527`：多处主谓、冠词和大小写错误。

这些不是穷尽清单。建议在实证内容冻结后进行一次完整 copy-edit，而不是逐个修补当前版本。

## 9. 建议的修订顺序与验收标准

### P0：决定章节是否可审计

1. 冻结唯一 evidence/data/model release；
2. 统一 Methods、Results、Conclusion；
3. 排除或量化 topic-selection leakage 与 real-time vintage leakage；
4. 补全 row lineage、policy labels 与 runtime override 说明；
5. 在相同 frozen population 上加入朴素决策基线；
6. 修复全部编译、图表和引用错误。

**P0 验收标准：** 每个正文数字能定位到唯一 artifact；任何 meeting 是否训练/选择/测试均可一眼判断；无 unresolved references、missing figures 或 malformed tables；结论不再出现正文没有的方法或结果。

### P1：决定核心实证主张是否成立

1. 重跑并完整报告 six-metric rewriting evaluation；
2. 报告绝对性能与 failure taxonomy，并加入 human/claim-level factuality validation；
3. 给 LOO 增加负控、多重比较控制与 Monte Carlo adequacy；
4. 重新设定 econometric estimand、单位、error structure、null baseline 与 equivalence test；
5. 分 panel 报告 decision results，增加 calibration 和 meeting-level uncertainty；
6. 至少对关键训练阶段使用多个 seeds，或把所有训练比较严格限定为 one-run artifact comparison。

**P1 验收标准：** rewriting 的构念效度、LOO 的解释边界、econometric association 和 decision utility 被分开；任何 “improvement”“faithfulness”“equivalence”“prediction” 都有相应直接检验。

### P2：提升为 PhD thesis 的论证与呈现标准

1. 重写 Introduction/RQs/contributions；
2. 系统扩充 Literature Review 并明确 gap；
3. 压缩通识架构教程，增加真实运行细节；
4. 重画训练流程图；
5. 完整英文润色和术语统一；
6. 增加 reproducibility、licensing、ethics 与 misuse discussion。

## 10. 最终评审结论

本章的研究设计有可取之处，尤其是共同父模型、两条独立下游分支、meeting-level 时间切分、显式 prompt/reward contract 以及对若干负结果的透明报告。这些内容构成了大修后形成扎实方法章节的基础。

但当前版本还不能支持其最强的研究主张。最关键的原因不是英语表达，而是：

1. 正文与结论来自不同结果版本；
2. decision input 可能通过 Minutes-derived topic selection 泄漏会后信息；
3. point-in-time vintage 与数据谱系不足以审计；
4. rewriting 主要测量 teacher-target alignment，不能直接证明 official-Minutes fidelity；
5. econometric pattern 没有复现 official Minutes 的主要长端系数；
6. decision model 未击败 always-Hold 基线；
7. 训练配置与 LoRA 参数量存在实质矛盾；
8. 当前源码仍有编译和引用阻断问题。

因此本次建议为 **Major Revision**，而且属于需要统一证据、补充审计并重做部分分析的实质性大修，不是单纯的文字润色。只有在 P0 全部完成、P1 的核心证据至少得到清晰而克制的闭环后，本章才适合进入正式 thesis submission。
