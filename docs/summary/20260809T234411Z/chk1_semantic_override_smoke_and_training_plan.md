# chk1-only 语义门禁例外、退化 smoke 与正式重训计划

生成时间：2026-08-09T23:44:11Z

## 决策与边界

用户明确授权：本轮可绕过 clean-v2 数据的 Qwen source-only 语义门禁，仅用于重新训练 chk1；本轮不训练、替换或发布 chk2/chk3/chk4。

这个例外不会把数据标记为 `semantic_audit=passed`。候选数据继续保持 `pending_semantic_audit`，失败的 Qwen 审核收据也原样保留。例外训练产生的 adapter/merged model 必须标记为 `chk1-only semantic override`，禁止作为 retrain-v2 DAG 中 chk2、chk3 或 chk4 的已审核父模型。

## 固定输入

- chk0：`models/DeepSeek-R1-Distill-Llama-8B`
- clean-v2 candidate：`output/data/retrain_v2/chk1/chk1_reasoning_compressed_flash_max_v2_clean_20260809_candidate`
- candidate manifest SHA-256：`41a5111875b052a44daec4bde5d622e25e19429158819aec43e1ec85d1945700`
- deterministic token validation SHA-256：`12c9f9f56c3946a98e3ca4eb4770818c30489b71c5cc498bbedb24edafef187b`
- failed Qwen audit summary SHA-256：`780436e35db5690fc90cb68eedcc9c61653d67d8d2036dd68306d81907aee90e`
- split：train/eval/test=`1354/199/190`
- 已通过的确定性门禁：单 BOS、最终 EOS、completion mask、无截断、结构/污染检查、split/hash/order 绑定。
- 被明确绕过的唯一门禁：237 条变更样本的 Qwen source-only 语义审核。该审核结果为 completed=235、passed=1、failed=234、judge_errors=2、blocking violations=611。

## 实施步骤

1. 新增严格的 chk1-only override verifier。它同时绑定 candidate manifest、确定性验证收据、失败的 Qwen audit 与本次用户授权收据的路径和哈希；与正常 passed-release verifier 互斥；训练只能精确读取 manifest 指定的 train/eval 文件。
2. 创建用户授权收据，包含 `stage=chk1`、`allowed_operation=sft_training`、`downstream_stages_allowed=[]`、固定输入哈希和自摘要。任何输入变更都会使例外失效。
3. 创建独立 smoke 配置，从 chk0 开始在 GPU0+GPU1 上训练 10 optimizer steps；不恢复旧 checkpoint，不复用正式输出目录。保持正式超参数，只将 logging/eval/save 调整为足以观察 10-step 行为。
4. 训练 smoke 前，在相同的固定 task-aligned 样本集合上生成 chk0 baseline；smoke 后直接加载 LoRA adapter，在完全相同的 prompt、seed 和解码参数下重新生成。样本覆盖短/中/长 prompt 以及 changed/unchanged 行。
5. 退化门禁至少检查：OOM/非有限值、EOS/length、`</think>` 边界和非空 answer、完整与尾窗 token 4-gram 重复率、严格周期尾巴，以及 smoke 相对 chk0 的重复率变化。出现灾难性循环、周期尾巴、触顶高重复或明显恶化则停止，不启动正式训练。
6. smoke 通过后，创建另一全新正式 run，从 chk0 重新开始两轮 SFT；绝不从 smoke checkpoint 续训。使用两张 A30、每 10 step eval、每 step save，保留最新 3 个和所有 10 的倍数 checkpoint。
7. 启动正式训练后至少监控早期 optimizer steps，检查 OOM、NCCL、NaN/Inf、loss 与 eval 日志。所有输出继续保留 chk1-only override lineage，不进入 chk2。

## 通过条件

- smoke 正好完成 10 steps，训练/eval loss 有限，无 OOM、NCCL 或数据加载错误；
- smoke 输出没有严格周期尾巴或重复吸引子；不得出现 `max_new_tokens` 触顶且高重复；
- smoke 的灾难性重复样本数为 0，聚合重复率不比同样本 chk0 baseline 明显恶化；
- prompt、candidate、deterministic receipt、failed audit 和 authorization receipt 的哈希在训练运行收据中完整记录；
- 正式训练必须从 chk0 在空目录启动，不能恢复 smoke 或旧 chk1。

## 明确风险

绕过后只证明数据的确定性结构与 token 合同通过，并不证明 237 条修订样本的语义均被 Qwen 判定为无错误。因此本次结果适合做 chk1 训练与退化实验，但在重新完成可靠语义审核前，不得提升为 chk2 的训练父模型。
