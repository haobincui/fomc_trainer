# chk2 checkpoint-150 合并与 chk0/chk1/chk2 共同测试

## 当前结论

- 已人工停止 fresh chk2 GRPO run，停止点为 step 183；停止原因为用户请求触发的
  `KeyboardInterrupt`，不是 OOM。
- 已停止该 run 专属的 Qwen Judge 服务，两张卡随后均释放；评测启动后 GPU0
  被另一个独立 WGAN 任务占用。
- 根据训练窗口 reward、zero-reward、格式完整率和截断率的联合比较，固定选择
  `checkpoint-150`，没有改用更晚但窗口指标较差的 checkpoint。
- `checkpoint-150` 已非破坏式合并为新的独立完整模型；源 adapter 保持不变。
- chk1 与 chk2 均已通过 CPU 逐张量精确 LoRA 合并校验。

## 固定工件

| 角色 | 路径 | 目录 SHA-256 |
|---|---|---|
| chk0 | `models/DeepSeek-R1-Distill-Llama-8B` | `bfb086cb87e9616e60805fc6f6f95826294b1de70b158cfea2d3e213df38ca11` |
| chk1 | `output/training/retrain_v2/chk1_compressed_flash_max_v1_20260805/merged/chk1` | `70be17c6d879f70eb98ab54bcf2a29f2f4991b2d7ab73b5696cbb6a0139f83aa` |
| chk2 adapter | `output/training/retrain_v2/chk2_compressed_chk1_v1_reward_v3_long4096_fresh_20260807/adapters/chk2/checkpoint-150` | `2130a852906cc201aee192e0927a7ea43f52e1dca0c0fac28900e42f85ef0e81` |
| chk2 merged | `output/training/retrain_v2/chk2_compressed_chk1_v1_reward_v3_selected_cp150_20260809/merged/chk2` | `fe4a207a75d3883606e565e7001dbf66bda81563677dd061bc13fd5440b56c18` |

chk2 adapter 的 `adapter_model.safetensors` SHA-256 为
`98641caada0e7b6be8ade362b2ea681bf4d9ff593d706e97aae021c87ca8d977`。

## 合并验证

- chk1：291/291 model tensors 通过；224 个 LoRA 目标张量与 67 个未适配
  张量均符合精确合并表达式，0 mismatch。
- chk2：291/291 model tensors 通过；224 个 LoRA 目标张量精确等于
  `to_stored_dtype(base + 2 * B@A)`，67 个未适配张量与 chk1 逐值相等；
  448 个 adapter tensors 全部且仅被使用，0 mismatch。
- chk2 exact-evidence payload SHA-256：
  `5f09033f168a8e84416b1bb8d081bf5cacf34c354a94ef60371b9a56b3a85a1f`。

验证工件：

- `chk1_exact_merge_lineage.json`
- `chk2_cp150_exact_merge_lineage.json`
- `merge_checkpoint150.yaml`
- `chk1_merge_verification.yaml`
- `checkpoint_manifest.json`
- `lineage_manifest.json`

## 共同测试合同

复用 `docs/summary/20260728T091012Z` 所述冻结 Chapter 2 共同测试：

- 11 个会议、每会议 3 个 Minutes section、每模型 33 rows；
- prospective-only subset 为 9 个会议、每模型 27 rows；
- byte-identical frozen prompts 和 references；
- greedy decoding：temperature 0、top-p 1、sample-ID deterministic seed；
- `max_new_tokens=8192`、`max_model_len=24576`，禁止输入截断；
- strict-final-answer 与 length-tolerant-open-tags 两套评分；
- 同一 BERTScore RoBERTa-large、MPNet、ROUGE-L、长度、重复、格式、
  数值/单位/时间、unsupported、方向和 policy-stance 指标；
- meeting-cluster bootstrap 10,000 次，seed 20260729；
- 不使用 LLM judge 或人工评分。

冻结输入 SHA-256：

- prompts：`9e19c23d8a06a5ae2626f707bfa681fcb1789562bf43a6732128157faa0553de`
- references：`80026c603962b00a506b4021535d25ff7cddbf4526618c7a729b2242fc123a9a`

历史 runner/scorer 被严格封存为四个旧工件，且旧 exact-merge evidence 绑定旧
chk1，因此不能安全地原地替换。当前运行新增三工件入口：

- `run/eval_retrain_v2_checkpoint_generation.sh`
- `run/eval_retrain_v2_checkpoint_pipeline.sh`
- `jobs/eval/eval_retrain_v2_checkpoint_generation.py`

新入口仍使用 canonical generation 和原 row scorer；generic evaluator 新增显式
full/prospective subset 参数。现有 generation/evaluator 测试为 52/52 通过，
另有三工件双-subset smoke 通过。

## 解释边界

该共同测试的输入/目标是 raw `D-1 evidence -> official Minutes section`。
当前 chk1/chk2 的产品任务是 `D-1 evidence -> FOMC analysis`，因此这是可比的
end-to-end stress benchmark，但存在明确 task mismatch。最终结果可以比较三模型
在同一压力测试上的行为，不能单独当作当前 chk2 analysis 能力的任务内验证。

## 运行目录

`output/evaluation/main/checkpoint_generation/retrain_v2_chk0_chk1_chk2_cp150_20260809`

推理、严格评分和长度容忍评分由 tmux 流水线自动完成。结果完成后在本文件追加
validated 输出 hash、核心表格和结论。

### 2026-08-09 15:53 UTC 状态

- `eval_cp150_chk0_20260809`：chk0 为 6/33、0 generation error；
- `eval_cp150_pipeline_20260809`：等待 chk0 sealed manifest，随后自动执行余下阶段；
- GPU0 的共享显存尝试因无法建立 24,576-token KV cache 而在发布输出前终止；
  后续采用安全的空闲卡策略；
- 当前没有 chk2 training 或 Judge 进程。
