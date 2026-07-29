# Chapter 2 严格评审备忘录

## 1. 总体判断

**结论：以现稿状态看，本章暂不合格，不建议直接视为达到可提交/可答辩的 PhD thesis chapter 标准。**

原因不在于选题没有价值。相反，这一章的题目、数据来源和方法组合都很有潜力，也已经形成了一个看起来完整的经验框架。但按严格评审口径看，当前版本最致命的问题不是“写得不够漂亮”，而是**论证链没有闭合、实验识别不够干净、对照组不充分、可复现性信息不完整、结果解释多处超过证据本身**。这意味着读者很容易接受“这是一个值得继续推进的研究方向”，却很难被说服“这一章已经可靠地证明了作者声称的结论”。对 PhD thesis 而言，这个差距是实质性的。

## 2. 一类问题：会影响是否达标的核心硬伤

### 2.1 训练与评估主线没有讲清，模型谱系和比较对象不闭合

**为什么这是论文级硬伤：**  
本章最核心的主张是“经过 post-training 的推理型 LLM 可以生成 grounded 的 FOMC Minutes 风格文本，并在经济上有意义”。如果读者不能准确追踪“到底有哪些阶段、每个阶段产出什么 checkpoint、最终拿哪个 checkpoint 去做哪个实验、表里的 baseline 到底是谁”，那么所有结果都会失去可解释性。

**具体证据：**  
在 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:21) 到 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:34) 中，章节主线被定义为“两步训练计划”，即 Step 1 SFT + Step 2 GRPO，并暗示这两步之后就进入 downstream generation 与 simulation。  
但在 [dataset_construction.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/dataset_construction.tex:571) 到 [dataset_construction.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/dataset_construction.tex:667) 中，又明确加入了一个额外的 `downstream alignment stage`，把 analysis 重写为 Minutes 风格文本。  
同时，在 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:47) 到 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:48) 中，“baseline” 被写成 `LLaMA 3-8B-Instruct`；但在 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:89) 到 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:101) 中，又说最终选择的 base model 是 `DeepSeek-R1-Distill-Llama-8B`。到了结果区，比较对象又变成 `Minutes-aligned model`、`baseline model`、`domain-adapted baseline model` 等不同叫法。

**对结论造成的风险：**  
读者无法判断：
`Minutes-aligned model` 是否是在 `DeepSeek-R1-Distill-Llama-8B` 上先做 SFT+GRPO，再做 style alignment；  
还是在 `LLaMA 3-8B-Instruct` 上直接做了另一条训练链；  
还是 `baseline model` 指的是未做 alignment 但做过两步训练的模型。  
在这种情况下，文本相似度结果、decision prediction 结果、econometric validation 结果就不再是“有清晰归因的模型比较”，而更像“几个名字相近的 checkpoint 之间的模糊比较”。

**该怎么改：**  
必须在正文里引入一张**唯一的流水线总图**和一张**checkpoint 对照表**，至少明确：
`Backbone-0`：`LLaMA 3-8B-Instruct` 还是 `DeepSeek-R1-Distill-Llama-8B`；  
`Checkpoint-1`：分析型 SFT 后模型；  
`Checkpoint-2`：GRPO 后模型；  
`Checkpoint-3`：Minutes-style alignment 后模型；  
每个结果表到底比较哪两个 checkpoint。  
没有这一步，整章最核心的实验结构都还处于“不够可审稿”的状态。

### 2.2 样本划分与泄漏风险没有被控制，或至少没有被证明已控制

**为什么这是论文级硬伤：**  
这章的任务本质上是条件生成与风格重写。如果 train/test 切分不按 meeting 级别隔离，而是随机打散段落或样本，那么同一次会议的邻近内容、同一段讨论的不同重写版本，甚至同一 target 的强提示，都可能同时出现在训练集和测试集。这样得到的高相似度并不能证明模型学会了“泛化生成”，只能证明模型在高度相似的局部语境里“学会了重写或补全”。

**具体证据：**  
在 [dataset_construction.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/dataset_construction.tex:843) 到 [dataset_construction.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/dataset_construction.tex:865) 中，只给出了 80/10/10 的切分比例，没有说明这是随机切分、按时间切分、还是按 FOMC meeting 切分。  
在 [dataset_construction.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/dataset_construction.tex:524) 到 [dataset_construction.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/dataset_construction.tex:559) 中，teacher prompt 直接包含 `reference excerpt`。  
在 [dataset_construction.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/dataset_construction.tex:603) 到 [dataset_construction.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/dataset_construction.tex:659) 中，Minutes-style alignment 阶段又把 model-generated summaries 作为 input、官方 section text 作为 target，并用与 target 高度贴近的提示模板做蒸馏和筛选。

**对结论造成的风险：**  
如果切分不是按 meeting 隔离，那么：
训练集可能已经见过同一会议的其他 section、其他 paragraph、甚至高度相近的 reference excerpt；  
测试时的高 cosine/BERTScore 可能主要反映“语料内重写能力”，而不是“真正从指标泛化到 unseen meeting narrative 的能力”；  
decision prediction 中声称的 post-2009 表现也可能混入 meeting-level information leakage。  
这会直接削弱本章对“simulation capability”和“grounded generation”的主张。

**该怎么改：**  
必须改成**按 FOMC meeting 的严格切分**，最好再进一步改成**时间序列切分**。正文要明确声明：
同一次 meeting 的任何 paragraph、summary、section、distilled reasoning trace 都不能同时出现在 train 和 evaluation/test；  
teacher prompt 中的 `reference excerpt` 只能用于训练集构造，不能污染测试集构造；  
文本相似度结果应优先报告真正的 held-out meetings，而不是随机 sample-level holdout。  
如果现有实验做不到，就必须诚实把当前结果重新表述为“in-corpus validation”，而不是“泛化能力证明”。

### 2.3 “Sufficient Information” 的识别设计目前站不住

**为什么这是论文级硬伤：**  
这一部分承担的是“grounding / sufficiency” 证明任务。也就是说，它要回答的是：模型生成的文本是不是由输入指标驱动，而不是只是写得像。如果这一环识别设计不严谨，那么整章最重要的“不是在瞎编，而是在用数据说话”的论断就失去支点。

**具体证据：**  
在 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:346) 到 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:368) 中，正文先介绍了 Shapley Value 和 TokenShapley 的一般定义；  
但真正实施时，在 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:367) 到 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:389) 中只做了“完整 prompt 与去掉一个指标后的 prompt”的差值，这本质上是 leave-one-out masking，不是 Shapley，也不是 TokenShapley 的 expected marginal contribution。  
更严重的是，在 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:991) 到 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:1041) 中，第一组结果把“未遮蔽 prompt 生成的 synthetic Minutes”本身当作 target output。  
而在 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:1048) 到 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:1169) 中，第二组结果用 actual Minutes 作为 target，但大量 leave-one-out 后的 cosine similarity 反而高于 `Unmasked` 基线，例如 GDP、Labour Market 等行；表 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:1127) 到 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:1152) 甚至直接出现负向变化，如 PCE 为 `-3.3180%`。

**对结论造成的风险：**  
这里至少有两个识别问题。  
第一，用 synthetic output 当 target，相当于拿模型自己的 unmasked 输出去评价“去掉输入后有没有偏离自己”，这更像 self-consistency test，不是 grounding test。  
第二，当 masked 结果比 unmasked 更接近真实 Minutes 时，说明该指标要么噪声较大，要么 prompt 设计有问题，要么模型对该指标的使用方向并不稳定。此时再把“更高相似度”统一解释成“指标有正贡献”，在逻辑上说不通。

**该怎么改：**  
如果保留现有设计，必须把它重新命名为**leave-one-out masking similarity test**，不要再称为 Shapley 或 TokenShapley。  
同时要把两种 target 分开解释：
对 synthetic target，只能说是“internal reliance / self-consistency”；  
对 actual Minutes target，必须明确定义 contribution 的符号，并解释为什么某些指标被移除后反而更接近真实文本。  
如果作者确实想做“Shapley-style grounding”，就应使用 subset sampling 或 permutation-based approximation，并把 utility function、aggregation 方式、显著性检验写完整。

### 2.4 Econometric validation 目前不能支撑“market relevance / robustness” 这一层结论

**为什么这是论文级硬伤：**  
这是整章试图从“文本像不像”走向“经济上有没有意义”的关键一步。如果这一节成立，读者才会相信 synthetic text 不只是风格模仿，而是保留了某种真实的 policy signal。问题在于，当前展示出来的统计证据并不足以支撑这一强结论。

**具体证据：**  
在 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:1190) 到 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:1238) 中，作者把这一节表述为对 synthetic Minutes 的 econometric validation，并称其为 robustness check。  
但真正汇报的核心数字只有：
fine-tuned 模型的 bootstrap mean coefficient 为 `0.00271`，baseline 为 `0.00224`；  
对应 `t-statistic` 从 `0.6169` 上升到 `0.7152`。  
在常规经验研究口径下，这样的 t-statistic 远远达不到显著性阈值，更谈不上“统计上支持 market relevance”。

**对结论造成的风险：**  
当前结果最多说明“方向上更接近 benchmark coefficient”，或者“fine-tuned 比 baseline 略有改善”。  
它不能说明 synthetic text 已经恢复出足以解释金融市场反应的稳定信号，更不能说明 empirical relationship “persists” 或“robustly replicates” 了真实文本中的 market effect。  
如果这一点不收缩表述，本章会给人一种“结论大于证据”的明显印象。

**该怎么改：**  
这一节必须重写成**探索性验证**而不是强验证。  
至少要补：
完整回归表；  
bootstrap 设计细节；  
置信区间；  
真实文本回归与 synthetic 文本回归的并列表；  
“方向改善”与“统计显著”之间的严格区分。  
如果 t-statistic 仍然维持在 1 以下，正文就不应再使用“market relevance 已被验证”这类表述。

### 2.5 Decision prediction 证据不完整，无法支撑“优于传统方法/具有强决策模拟能力”的主张

**为什么这是论文级硬伤：**  
决策预测是整章最容易被导师或外审拿来追问的部分，因为它直接对应作者在引言里的重要 contribution claim。这个任务如果没有合适 baseline、没有处理类别不平衡、没有解释 evaluation protocol，就很容易变成“一个复杂模型对另一个复杂模型的封闭比较”。

**具体证据：**  
在 [intro.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/intro.tex:48) 中，研究问题明确问“model-based decision prediction compare with baseline approaches”；  
在 [intro.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/intro.tex:63) 到 [intro.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/intro.tex:64) 中，更明确承诺要与 `FedWatch` 等传统量化方法做 meaningful comparison。  
但在 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:1252) 到 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:1376) 中，结果只比较了 `Minutes-aligned model` 与 `Baseline model`，没有任何传统方法、期货隐含概率方法、简单 logit/ordered logit、甚至 majority class baseline。  
此外，表 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:1268) 到 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:1280) 的两个模型分母都不同，`142/180` 对 `120/178`，说明评估样本并非严格配对，或者存在 parse failure / exclusion，但正文没有解释。  
再者，本任务明显类别不平衡，`No Change` 样本占绝对多数，但只报告 accuracy。  
最后，在 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:1310) 中写“generated 500 synthetic Minutes and then drew 20 random samples (sampling interval: 2,000)”，这在数值上和实验逻辑上都不清楚。

**对结论造成的风险：**  
当前结果最多能说明“在作者定义的 LLM-vs-LLM 比较里，Minutes-aligned model 优于另一个 LLM checkpoint”。  
它不能说明：
对历史 FOMC action 的预测已经优于简单基线；  
文本生成确实给决策预测带来了超出期货市场信息的增量价值；  
作者完成了引言里承诺的“与传统方法比较”。

**该怎么改：**  
必须补入至少四类 baseline：
majority class baseline；  
上一期决策延续 baseline；  
基于联邦基金期货/隐含概率的市场 baseline；  
简单计量基线，如 multinomial logit 或 ordered logit。  
同时把指标从单一 accuracy 扩展到 balanced accuracy、macro-F1、confusion matrix，并解释为什么两个模型的有效样本数不同。  
没有这一步，这一节不能作为“强贡献”写进引言。

## 3. 二类问题：实验细节与可复现性缺口

### 3.1 弱监督标注没有人工验证，无法判断标签噪声有多大

**为什么这是问题：**  
这章的全部 prompt 构造都建立在 paragraph-to-indicator labeling 上。如果标签本身噪声很大，那么后续 SFT、GRPO、grounding test、decision prediction 的输入都可能被系统性污染。

**具体证据：**  
在 [dataset_construction.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/dataset_construction.tex:125) 到 [dataset_construction.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/dataset_construction.tex:166) 中，作者说明用 `ChatGPT-4o` 做 weakly supervised classification，再人工清洗 label ontology、再 relabel。  
但全文没有给出任何人工复核样本、precision/recall、inter-annotator agreement、错误类别统计，甚至没有说明“人工清洗”覆盖的是 ontology 还是 sample-level label correctness。

**风险：**  
读者无法判断 26 个 reference labels 到底是“高质量结构化标签”，还是“看起来有条理但误差未知的自动标签”。这会削弱整个数据构造部分的可信度。

**该怎么改：**  
补一个小规模人工验证实验即可，不需要很大。  
例如随机抽 200 个 paragraph，人工标注后报告一级标签准确率、多标签漏标率、最常见混淆对。  
哪怕指标不高，也比完全不报告更可审。

### 3.2 Teacher model 与 Judge model 的定义不够可复现

**为什么这是问题：**  
本章的 reasoning trace、reward、alignment 数据都依赖 teacher/judge。只要这些模型定义不清，别人就无法知道实验是在比较训练方法，还是在比较某个特定 teacher/judge 组合的产物。

**具体证据：**  
teacher model 在 [dataset_construction.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/dataset_construction.tex:524) 和 [dataset_construction.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/dataset_construction.tex:659) 中被写成 `DeepSeek-R1`，但没有给出版本、调用方式、部署方式、是否本地、是否 API、stop 条件、seed、max tokens。  
judge model 在 [model_training.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/model_training.tex:443) 到 [model_training.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/model_training.tex:521) 与 [model_training.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/model_training.tex:573) 中被描述为 “LLM-as-Judge reward model”，但没有明确说 judge 到底是哪一个模型。

**风险：**  
读者无法区分：
结果是来自训练本身；  
还是来自 teacher 质量；  
还是来自特定 judge 偏好。  
这对 thesis 来说是明显的 reproducibility gap。

**该怎么改：**  
补一张 “External Models Used” 表：模型名、版本、来源、用途、关键 decoding 设置、调用日期。  
如果模型版本后续会变，这张表尤其必要。

### 3.3 Reward 设计说明与实际展示并不完全一致

**为什么这是问题：**  
RL 部分的说服力建立在 reward 定义清晰、实现一致、结果可对应。现在这一块存在“描述的是一个 reward system，表里展示的是另一个 reward system”的迹象。

**具体证据：**  
在 [model_training.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/model_training.tex:407) 到 [model_training.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/model_training.tex:417) 中，reward 被定义为三部分：format、accuracy、reasoning。  
但在训练结果区 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:598) 到 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:614) 中，结果文字只强调 `accuracy reward` 与 `reasoning reward` 等权，几乎没有解释 `format reward` 在总 reward 中的角色。  
此外，在 [model_training.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/model_training.tex:443) 中说 accuracy reward 有 7 个维度，但在表 [model_training.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/model_training.tex:529) 到 [model_training.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/model_training.tex:562) 的实际 prompt 里，只显式展示了 3 个类别的 placeholder。

**风险：**  
读者会怀疑：
真正被实现的是不是 7 维 rubric；  
format reward 是否真的参与了总 reward；  
训练日志里“total reward”到底对应哪几个组成项。  
这会让 GRPO 部分显得像“说法多于实现细节”。

**该怎么改：**  
把 reward 定义、实际实现、训练日志三者统一起来。  
正文必须明确给出：
总 reward 的精确公式；  
每一项是否参与训练；  
每一项的权重；  
表格里的 prompt 是否是完整版本还是节选版本。  
如果 prompt 为节选版本，要标明“abridged”。

### 3.4 GRPO 与 QLoRA 的关键参数没有交代完整

**为什么这是问题：**  
对于 PhD thesis，方法可以不要求代码级复现，但不能只停在“我们用了 GRPO/QLoRA”。必须让读者知道你到底用什么配置跑起来的。

**具体证据：**  
在 [model_training.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/model_training.tex:392) 到 [model_training.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/model_training.tex:397) 中，只说明了 GRPO 的概念，没有给出 `group size G`、`beta`、`epsilon` 等实际数值。  
在 [model_training.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/model_training.tex:719) 到 [model_training.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/model_training.tex:736) 中，只说用了 PEFT、QLoRA、FP4，但没有 LoRA rank、alpha、dropout、target modules。  
在 [model_training.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/model_training.tex:740) 到 [model_training.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/model_training.tex:755) 中，超参数只给出 learning rate、batch size、epochs、temperature、top-p，没有 optimizer、scheduler、warmup、max sequence length、gradient clipping、seed、effective batch size、硬件细节。

**风险：**  
这一缺口会让“方法细节”看起来像方法综述，而不是实验记录。  
读者知道你用了哪些名词，却不知道你具体怎么训练。

**该怎么改：**  
补一张真正的实验配置总表。  
至少包含：
base model；  
optimizer；  
scheduler；  
LoRA/QLoRA 参数；  
sequence length；  
gradient accumulation；  
GRPO group size；  
KL coefficient；  
clip epsilon；  
judge model；  
seed；  
GPU/显存。  
这是 thesis 级实验说明的最低配置。

### 3.5 Minutes-style alignment 阶段的数据划分和训练结果叙述不完整

**为什么这是问题：**  
文本相似度结果主要是在为 alignment 阶段背书。如果 alignment 阶段本身的数据切分、训练设置、结果 checkpoint 没交代清楚，那么相似度结果就失去了依附对象。

**具体证据：**  
在 [dataset_construction.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/dataset_construction.tex:576) 到 [dataset_construction.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/dataset_construction.tex:667) 中，构造了 1,392 个 Q\&A pairs。  
但正文没有像 analysis-generation 阶段那样给出对应的正式 train/eval/test split。相关切分信息只残留在注释里。  
结果区 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:641) 到 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:675) 的训练结果段落也基本处于被注释掉的状态。

**风险：**  
读者知道有一个 alignment 阶段，也看到 alignment 的文本相似度结果，但不知道该阶段到底如何训练、如何选 checkpoint、如何切分数据。  
这会让最漂亮的一组结果反而最难被信任。

**该怎么改：**  
把 alignment 阶段从“注释里的实验”恢复成“正文里的实验”。  
至少补：
该阶段数据切分；  
训练超参数；  
loss curve 或 early stopping 依据；  
最终使用哪个 checkpoint；  
文本相似度表到底对应哪个 held-out split。

## 4. 三类问题：一致性与技术性错误

### 4.1 标签与交叉引用错误会直接削弱可信度

**证据：**  
在 [intro.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/intro.tex:76) 中，章节组织部分引用的是 `ch2:sec:application`；  
但实际标签写在 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:144) 为 `ch2:sec:applictaion`。  
这不是单纯拼写问题，而是会在编译后直接生成错误引用。

**影响：**  
这种错误会让外审第一时间怀疑整章是否经过系统检查，也会放大读者对其他实验细节错误的敏感度。

**修改方案：**  
统一所有 section label，并在全文跑一轮 `missing ref / duplicate label` 检查。  
本章当前已经暴露出“组织段承诺的标签”与“正文实际标签”不一致的问题，不应只修这一处。

### 4.2 样本数与分母多处不一致，说明实验口径没有完全锁定

**证据：**  
在 [dataset_construction.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/dataset_construction.tex:48) 中，`Total Minutes` 为 123；  
但在 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:1269) 中，post-2009 decision prediction 却使用了 128 个 Minutes。  
在 [dataset_construction.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/dataset_construction.tex:855) 中，`train_sft` 数量为 3,129；  
但在 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:575) 中，SFT 每个 epoch 覆盖 3,128 个 training samples。  
在 decision prediction 表 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:1269) 与 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:1277) 中，两个模型总体分母还分别是 180 和 178。

**影响：**  
这些不一致会让读者怀疑：
是不是有样本丢失没解释；  
是不是不同实验版本的数据混进了同一章；  
是不是表格和正文并非同一轮实验产物。  
对 thesis 来说，这会显著削弱结果可信度。

**修改方案：**  
对全章所有样本数建立一个统一口径表。  
每个数字都必须能回答两个问题：  
“这个数从哪来？”  
“为什么它和另一个看起来相近的数不一样？”

### 4.3 引言里的承诺没有在结果部分兑现

**证据：**  
在 [intro.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/intro.tex:48) 与 [intro.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/intro.tex:63) 到 [intro.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/sections/intro.tex:64) 中，作者明确承诺与 baseline approaches、传统量化方法、`FedWatch` 等比较。  
但在结果部分 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:1248) 到 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:1376) 中，没有完成这类比较。

**影响：**  
这会让本章呈现出典型的 “Introduction sells a bigger paper than Results actually deliver” 的问题。  
在 thesis 写作里，这是非常常见但也非常伤的毛病。

**修改方案：**  
二选一：
要么真的补上这些 baseline；  
要么收缩引言里的 contribution claim。  
不能继续保留当前这种“承诺了强比较，但正文没有做”的状态。

### 4.4 结果表与文字解释在方向上不完全一致

**证据：**  
在 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:1048) 中，作者已经承认 actual Minutes 版本里“indicator contribution can be interpreted via the change relative to the unmasked baseline”。  
但表 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:1061) 到 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:1088) 显示，不少指标在 masking 后的 cosine similarity 高于 `Unmasked`。  
进一步地，在 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:1114) 到 [chapter2.tex](/Users/haobincui/Documents/phd/PhdThesis/Chapter2/chapter2.tex:1169) 的相对变化表中，甚至直接出现明显负值，如 PCE 的 `-3.3180%`。

**影响：**  
这意味着“去掉指标后更接近真实文本”的情形并不是偶发，而是结构性存在。  
如果正文还继续把这一节解释成“指标整体上都有正向贡献”，那就是解释方向与结果本身脱节。

**修改方案：**  
必须单独讨论负贡献或反向贡献指标，不能只汇报显著性的星号。  
这一节需要从“证明指标有用”改为“分析哪些指标真正被模型有效利用，哪些指标引入噪声或被误用”。

## 5. 修改方案

### 5.1 必须修改

1. **重写章节主线。**  
加一张总流程图和一张 checkpoint 表，把 backbone 选择、analysis-generation SFT、GRPO、Minutes-style alignment、decision evaluation 串成一条唯一的数据流与模型流。

2. **改成严格的 meeting-level 或时间序列切分。**  
正文必须明确 train/eval/test 的切分单位，并补一段 leakage 说明，保证同一 meeting 的任何衍生样本不会跨集合出现。

3. **重写 grounding test。**  
如果保留现有做法，就统一改称 `leave-one-out masking`，并把 synthetic target 与 actual target 的解释完全分开；如果要继续用 “Shapley” 这个词，就必须做 subset-sampling 近似。

4. **为 decision prediction 增加真正的 baseline。**  
至少补 majority class、上一期决策延续、Fed funds futures/FedWatch、简单 logit/ordered logit。没有这些，不能再声称与传统方法形成 meaningful comparison。

5. **重写 econometric validation 的结论。**  
把“方向改善”与“统计显著”明确分开。若 t-statistic 仍然远低于常规阈值，就只能写成 exploratory evidence。

6. **补齐复现参数表。**  
把 teacher model、judge model、LoRA/QLoRA 配置、GRPO 参数、optimizer、scheduler、seed、硬件信息统一列成一张表。

### 5.2 建议修改

1. **压缩基础模型原理介绍。**  
`GQA / SwiGLU / RoPE / prompt engineering / CoT` 这些内容现在占用了不少篇幅，但并不是本章识别贡献所在。应把篇幅让给实验设计、对照组与结果边界。

2. **加一张实验总表。**  
每个实验只回答五件事：训练数据、测试数据、比较对象、评价指标、能支持的结论边界。这样可以显著提升整章的可审性。

3. **把注释里残留的旧实验口径清理掉。**  
当前正文与大量注释版本并存，很容易把不同轮次实验数字混在一起。正式提交前必须把“活跃口径”锁定。

### 5.3 可选优化

1. **增加人工评审小样本。**  
哪怕只对 30 到 50 个 generated sections 做人工评分，也能显著增强对 “institutional style” 与 “reasoning usefulness” 的可信度。

2. **增加 error analysis / failure cases。**  
建议专门列出几类失败样本，例如：
去掉某个指标后文本反而更接近真实 Minutes；  
decision prediction 在 `Cut` 类别上崩溃；  
synthetic text 在 econometric test 中没有恢复显著性。  
这会让整章更像 thesis，而不是只展示“表现较好的结果”。

3. **最后再做语言和 LaTeX 级清理。**  
本章当然也有语病、拼写、注释残留和交叉引用问题，但这些应放在逻辑和实验问题修完之后统一清理，不应本末倒置。

