# chk1 Clean-v2 Smoke 生成退化根因诊断

生成时间：2026-08-10 UTC

## 技术摘要

当前最符合全部证据的解释是一个**多因素触发的非线性解码稳定性问题**：teacher compression contract 把信息很少的样本也扩写成约 1,000-token、风格高度同质的 reasoning；completion-only SFT 的绝大多数监督因此都在奖励“继续写 reasoning”。当 rank-32、alpha-64、覆盖全部 attention 与 MLP 投影的 83,886,080 参数 LoRA 在 step 9–10 进入最高学习率区间时，模型的自由生成轨迹跨过稳定性阈值，开始偏离训练分布并以自己的重复文本继续条件化，最终形成自强化循环。

这不是单一坏样本、clean-v2 answer 修复、缺少 `</think>` 标签、double BOS、截断、adapter 损坏或 OOM。整体 train/eval loss 也无法发现它，因为 teacher-forced token loss 不评估模型在自身错误前缀上的恢复能力。

正式 chk1 训练应继续保持未启动状态。

## checkpoint-8 到 checkpoint-9 出现了明确的非线性退化

对 checkpoint-8 和 checkpoint-9 使用同一组 6 个 held-out 样本、相同 seed、相同采样/greedy 设置和相同 1,024-token 诊断上限。checkpoint-10 使用此前封存的 3,072-token fail-fast 结果。

| 模型 | 诊断范围 | 主要结果 |
|---|---:|---|
| chk0 | 8 cases，3,072 cap | 8/8 EOS；8/8 合同有效；0 catastrophic；最大全文重复率 0.1425 |
| checkpoint-8 | 8 cases，1,024 cap | 8/8 合同有效；无重复型 catastrophic；唯一失败为最长 greedy 的 cap-only，已经有 `</think>` 和 answer，尾部重复率仅 0.0236 |
| checkpoint-9 | 8 cases，1,024 cap | **4/8 catastrophic**；EOS 与合同有效率均为 62.5%；平均/最大全文重复率 0.4903/0.7845；周期尾比例 25% |
| checkpoint-10 | fail-fast case，3,072 cap | 3,072 tokens 触顶；无 EOS、无 `</think>`、无 answer；全文/尾部重复率 0.9231/0.9647 |

checkpoint-8 的 cap 不能与 chk0 的 3,072 上限直接比较，所以这里没有把 cap rate 当成同口径指标；重点是它已经生成 boundary 和 answer，且没有重复尾。checkpoint-9 则同时在 sampled 与 greedy、多种 topic 上出现实质循环。

同一条 unchanged、822-token Unemployment Rate test prompt 的轨迹尤其清楚：

| 模型 | 输出 | EOS/合同 | 全文重复率 | 尾部重复率 |
|---|---:|---|---:|---:|
| chk0 | 374 tokens | 是/有效 | 0.0405 | 0.0405 |
| checkpoint-8 | 535 tokens | 是/有效 | 0.0791 | 0.0825 |
| checkpoint-9 | 477 tokens | 是/有效 | 0.0994 | 0.0994 |
| checkpoint-10 | 3,072 tokens | 否/无效 | 0.9231 | 0.9647 |

把 checkpoint-10 输出重新按同一 tokenizer 截到前 1,024 tokens 后，全文/末 512 token 的 4-gram 重复率已经是 0.7689/0.9293。也就是说，较大的 3,072 上限只是让既有循环继续增长，不是循环的起因。

没有绘制 checkpoint 趋势线，因为 checkpoint-10 是 fail-fast 单案且 completion cap 与 checkpoint-8/9 不同；精确表格更不容易制造虚假的连续趋势。

## teacher “压缩”合同实际上强制制造长篇填充

`jobs/retrain_v2/compress_chk1_reasoning.py` 第 80–81 行要求 teacher 写 **5–10 段、通常 800–1,400 英文词**。这与同一 prompt 中“much shorter”“remove repeated evidence”的目标存在直接冲突，而且长度没有随 evidence 数量自适应。

全量 1,743 条 clean-v2 数据的统计为：

- reasoning 中位数 996 tokens、均值约 992.6 tokens；76.6% 不少于 900 tokens。
- 中位 target 为 8 段、35 句。
- evidence 数和 reasoning 长度的 Pearson 相关仅 0.118；evidence 数和 answer 长度的相关为 0.678。
- 335 条只有 3 个 evidence 的样本，reasoning 仍平均 956.8 tokens，即每条事实平均扩写 318.9 tokens。
- 57.3% 的训练 reasoning 含以 “At the same time” 开头的句子，19.2% 含 “On balance the evidence” 开头的句子，说明跨样本话术同质化明显。

失败 held-out 样本只有 3 个失业率观测，却有 1,013-token、8 段、41 句的 gold reasoning，answer 只有 68 tokens。chk0 在相同 prompt 上自然生成约 269 reasoning tokens 后就输出 `</think>`；teacher target 却继续扩写到 1,013 tokens。smoke 输出约在 token 174 开始加入输入没有的同比叙述，在 token 225 左右进入重复。

因此，模型在事实已经耗尽、基座本来会结束 reasoning 的位置，被 SFT 持续训练为继续展开相同材料。

## 监督权重几乎全部落在长 reasoning，而非结束行为

按 clean-v2 train repair manifest 的 1,354 条样本计数：

- reasoning tokens：1,338,719，占约 89.38%。
- answer tokens：156,321，占约 10.44%。
- 核心 `</think>` token：每样本 1 个，约占 0.0904%。
- EOS token：每样本 1 个，约占 0.0904%。

这里的比例只计算 reasoning、answer、核心 boundary token 和 EOS；若把 boundary 周围换行也算作结构 token，结构占比会略高，但不改变 reasoning 监督占绝对多数的结论。

这不意味着 boundary 没有被监督。真实 tokenizer/collator 验证已经确认 1,743/1,743 行都是单 BOS、单 boundary、最终 EOS，prompt 全 mask，reasoning、answer、boundary 和 EOS 全部进入 completion labels，且零截断。问题是序列级训练目标让“继续写”成为远比“结束”更常见的行为。

## 高容量 LoRA 和高学习率更新触发了阈值

本次 QLoRA 有 448 个 adapter tensors、83,886,080 个可训练参数，`r=32`、`alpha=64`，同时覆盖 `q/k/v/o/gate/up/down` 七类投影。它没有 KL 或其他保持 chk0 原行为的约束。

step 8–10 的日志为：

| Step | Logged LR | Loss | Grad norm | 生成状态 |
|---:|---:|---:|---:|---|
| 8 | 3.89e-5 | 1.6896 | 0.984 | checkpoint-8 基本稳定 |
| 9 | 4.44e-5 | 1.5798 | 0.707 | checkpoint-9 多样本开始循环 |
| 10 | 5.00e-5 | 1.6130 | 0.645 | checkpoint-10 短 prompt 也完全失控 |

checkpoint-9 scheduler state 已把下一次 optimizer update 的 LR 设为 `5e-5`，因此 checkpoint-9 到 checkpoint-10 的更新使用峰值学习率。正式约 170-step 配置在同一位置也会达到近似相同峰值，因此这不是仅由 smoke 总步数造成的假象。

adapter tensors 全部有限；checkpoint-8→9 与 checkpoint-9→10 的参数更新连续，后者的中位更新 norm 甚至更小。grad norm 也没有尖峰。这排除了普通的数值爆炸，更支持“小而有方向的更新跨过序列生成稳定性边界”。

当前证据仍不能把峰值 LR 和具体 step-10 batch 的方向完全分离。只有保持完全相同数据顺序、把 LR 降低后重放第 10 步，才能完成这个因果区分。

## fixed-prefix 证明模型没有忘记 `</think>` 标签

额外的 next-token probe 分别把 chk0、checkpoint-8/9/10 强制放在两个正确 reasoning 末端，再检查下一 token：

| 固定前缀 | chk0 `</think>` 概率/排名 | checkpoint-10 概率/排名 |
|---|---|---|
| chk0 自然 reasoning 末端（269 continuation tokens） | 0.999996 / rank 1 | 0.999987 / rank 1 |
| gold reasoning 末端（1,013 continuation tokens） | 0.999968 / rank 1 | 0.999272 / rank 1 |

adapter 确实随 step 单调压低了 gold 末端的 boundary logit：chk0 为 24.0，checkpoint-10 为 20.75，对应非-boundary 概率约放大 22 倍；但 `</think>` 在两个正确末端仍是绝对 top-1。

所以直接机制不是“模型完全不会输出 boundary”，也不是 boundary label/mask 丢失。模型在 teacher-forced 的正确轨迹上知道如何结束；它在自由生成的更早阶段走偏，开始以自己的 unsupported/repeated tokens 为下一步上下文，随后永远到不了训练分布里的正确末端。这是典型的 exposure error / sequence-level instability。

## clean-v2 修复与主题顺序是次要因素，不是单独根因

clean-v2 的 1,743 条中，1,542 条 answer 未改；只有 31/1,354 条 train reasoning 被做过局部 meta 清理，97.7% 的 train reasoning 没有变化。失败 test 样本本身 `changed=false`，旧、新 response SHA 完全相同。

step 9 的精确 16 条全局 batch 只有 2 条 changed row，且都只是从 answer 删除 evidence IDs；没有 reasoning repair。该 batch 的 target 4-gram 重复率均不灾难，均值 0.0774。step 10 的 target 4-gram 均值为 0.0595，反而略低于全 train 均值 0.0627。因此不能把 checkpoint-8→9 或 checkpoint-9→10 的突变解释成直接读入了一条无限循环 target。

不过，seed 42 的前 160 个训练样本里有 15 条 Unemployment Rate，占 9.38%，而全 train 只有 73/1,354=5.39%；随机抽到至少 15 条的超几何尾概率约为 1.93%。这可能加速首次 unemployment fail-fast 样本的局部退化。但 checkpoint-9 已同时在 Commodity Prices、Unemployment Rate 和 VIX 等不同 topic 上失败，所以 topic 集中只是放大器，不是全局解释。

## 已基本排除的原因

- **训练 targets 直接含同样循环：** 全 train strict periodic tail 为 0；失败循环短语在 SFT 数据中没有命中。
- **clean answer 修复直接污染 reasoning：** 失败样本 unchanged；step 9 没有 reasoning repair；循环发生在 `</think>` 之前。
- **double BOS、缺 EOS、boundary 格式或 completion mask 错误：** 真实 tokenizer/collator 全量验证通过。
- **训练截断：** 0/1,743 截断，最大完整序列 4,019，小于 `max_length=4,608`。
- **adapter/root checkpoint 不一致：** root adapter 与 checkpoint-10 的配置和权重逐字一致。
- **adapter 数值损坏：** 448 个 tensors 全部有限，更新尺度连续。
- **probe/tokenizer/EOS 解码错误：** chk0 在同一 prompt、seed、量化和 probe 路径下正常；fixed-prefix 也读到正确 special boundary token。
- **temperature 是唯一原因：** checkpoint-9 在 sampled 和 greedy 两种模式都出现循环。
- **Qwen semantic override 直接解释结构崩坏：** 该 audit 只审 237 条 changed rows，611 条 blocking violation 全位于 answer；它仍是 factuality 风险，但不能解释 unchanged 样本在 reasoning 阶段失去 boundary。

## 原因优先级

1. **高置信：长、与 evidence 数无关且高度模板化的 reasoning targets，造成脆弱的持续扩写先验。**
2. **高置信：`5e-5` 峰值附近、83.9M 参数全投影 LoRA 更新触发 free-running 稳定性阈值。**
3. **高置信机制：teacher forcing 没有训练模型从自己的错误前缀恢复，导致早期偏差变成自强化循环。**
4. **中等置信放大器：前 160 条的 topic 集中和缺少明确的推理长度上限。**
5. **较低置信：changed answers 的事实问题可能损害内容准确性，但不是本次无 boundary 循环的主要来源。**
6. **低概率：数据中直接循环、双 BOS、截断、adapter 损坏、加载错误、temperature 或 attention backend。**

## 最小可证伪实验

为了区分 target 设计和 optimizer 强度，下一轮应采用正交而不是一次改很多项：

1. **LR-only smoke：** 数据、顺序、seed、LoRA 完全不变，只把最大 LR 改为 `1e-5`；每 step 保存并对固定 8 cases 运行探针。
2. **target-only smoke：** LR 保持 `5e-5`，把 reasoning 改为 evidence 自适应长度；3 facts 建议 256–384 tokens，整体先硬限 512 tokens、1–4 段。
3. **联合 smoke：** `1e-5` + 短 target，用于确认两项一起能否稳定。
4. **topic-balanced sensitivity：** 保持配置不变，仅让前 160 条接近全 train topic 分布，判断 unemployment 局部集中贡献。
5. **逐 checkpoint 门禁：** 任何一步只要出现 cap、无 `</think>`、无 answer、严格周期尾或全文重复率至少 0.5，就立即停止；不能再只看 loss。

不要把缩短 `max_new_tokens` 或加入 repetition penalty 当作根因修复；它们最多截断或掩盖循环。

## 证据与方法限制

- checkpoint-8/9 的 1,024-token诊断和 checkpoint-10 的 3,072-token fail-fast 不是完全相同 cap，因此只比较共同的合同、EOS、重复行为和 checkpoint-10 的前 1,024-token重算，不把 cap rate做连续趋势。
- checkpoint-10 正式 probe 因第一条已灾难失败而停止，不能据此估计 checkpoint-10 的总体失败率；checkpoint-9 的完整 8-case 结果已经证明问题不局限于一条 prompt。
- 当前证据支持高置信机制诊断，但只有上述 LR/target 正交 A/B 才能给出各因素的独立因果贡献。
- 便携 HTML 报告未生成：当前环境没有 `node`/`npm`，无法执行报告插件的规范打包与浏览器验证；本 Markdown 是仓库内的主诊断文档。

原始工件已复制到本目录 `evidence/`；汇总数值见 `diagnostic_metrics.json`，文件完整性见 `evidence/SHA256SUMS`。
