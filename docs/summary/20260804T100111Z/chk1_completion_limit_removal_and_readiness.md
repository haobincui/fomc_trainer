# chk1 completion 上限移除与训练就绪状态

更新时间：2026-08-04T10:59:50Z

本文件是既有执行方案
[`../20260803T150258Z/fomc_retraining_v2_plan.md`](../20260803T150258Z/fomc_retraining_v2_plan.md)
在 DeepSeek full-v7 数据完成后的增量执行记录。

## 结论

chk1 不再使用 `completion <= 1024` 作为训练准入条件。已经完成的 v7 generation
manifest 中仍保留 `3072/1024/4096` 采集期 observation，保证 API cache、输出与
generation code provenance 可重放；base release 会使用独立训练审计器重新计数：

```text
prompt <= 3072
completion: 无独立上限
prompt + completion <= 7168
truncation = false
```

7,168 是硬件验证后的物理总上下文上限，不是 target 长度上限。它覆盖正式发布数据中
最长的 6,855-token 样本；一个 A30 rank 上的真实 QLoRA r32 forward、backward 和
PagedAdamW8bit step 已通过。2026-08-04 后续将 chk1 launcher 扩展为两个相同模型
副本的 DDP，而不是跨卡切分模型。

## 数据进度

- full generation：`generation_full_v7`，状态 `complete`。
- generation 候选 2,117；接受 2,115；排除 2；retrieval rate `0.9990552668871044`。
- canonical admission 检出 43 条 reasoning 非空但 final analysis 为空，全部转为显式
  `empty_response_component` exclusion；没有修改 generation，也没有静默丢行。
- canonical 最终 split：train 1,683、eval 199、test 190，共 2,072 条；exclusion
  总数为 45，满足 `2072 + 45 = 2117`。
- generation handoff payload SHA256：
  `df8baa2c9e0633b4485029943afa312985e17bb687b504f08a1a3fbc741e4590`。
- generation provenance SHA256：
  `dd870c83332d19fab17530b0ea4c56dae6ab68f4a546f4a0df1c48b836ad7b80`。
- immutable generation code 中的 `prompt_projection.py` 已保持原始 SHA256：
  `57923d3689449e0bdae1d0a4ac1a1eeb178c4c2a1053ca6b19b35ff84f330e15`。

新训练合同对 2,072 条发布样本逐条使用本地 chk0 tokenizer 复算，全部通过；全量
最大 prompt/completion/total 分别为 `2735/4464/6855`。因此合格的长 completion
不需要重新生成，也不需要截断。

不可变 canonical release：
`output/data/retrain_v2/chk1/canonical_releases/chk1_full_v7_automated_v2_20260804/`，
handoff SHA256 为
`7eb1ceb8e3ea41b6305e40f15d4d6492ced64d45de370118a2f52ddf121a5448`。
自动质量 schema 为 `chk1-automated-data-quality-report-v2`，状态 `passed`。

最终 analysis base release：
`dataset/processed/retrain_v2/analysis_base_full_v7_automated_v3_20260804/`，artifact
SHA256 为 `b8128b394420f90d8e7cb2e0e422a7dc097b5d6c68adb30f73ded29f764c040f`；
schema、token budget、reference leakage、point-in-time、split integrity、target
consistency、encoding、teacher grounding 八项审计全部通过。

## GPU 与 DDP 配置验证

固定 chk1 配置为：

- 物理 GPU0+GPU1 双 rank DDP；每个 rank 各自持有完整 QLoRA 模型；
- NF4 double-quant QLoRA，bf16，LoRA `r=32/alpha=64`；
- q/k/v/o/gate/up/down 七类 target module 全部保留；
- batch 1，gradient accumulation 8，gradient checkpointing；
- Liger fused kernel；
- `flex_attention`；
- `max_length=7168`，禁止 runtime truncation。
- 每 rank batch 1、gradient accumulation 8，有效 batch 16。

实际压力测试结果：

```json
{
  "sequence_length": 7168,
  "attn_implementation": "flex_attention",
  "peak_allocated_gib": 14.379,
  "peak_reserved_gib": 14.619,
  "trainable_parameters": 83886080,
  "status": "passed"
}
```

对比测试确认 SDPA 的 8,192、7,168 和真实最长 train 6,855 都会在单张 A30 上
OOM；activation offload 与 reentrant checkpointing 没有解决临时张量峰值。
`flex_attention` 在不降低 LoRA rank、不删除 target module、不缩短样本的情况下通过。

## 已完成代码修复

- DAG 中 chk1 launcher 固定为 `ddp_2xa30`。
- chk1 stage 使用 `accelerate --num_processes 2` 启动双 rank
  `jobs.train.train_sft`，显式暴露物理 GPU0、GPU1。
- stage 在正式训练前自动执行 NCCL 检查和双 rank 7,168-token
  `flex_attention` QLoRA DDP smoke。
- token-budget gate 支持 `completion: null`，同时继续强制 prompt 与 total 上限。
- acquisition token observation 与 training admission contract 分离；训练逻辑位于
  `jobs/retrain_v2/sft_training_budget.py`，不改变 immutable generation code。
- canonical workflow 使用 `generation_full_v7` 和 `prepared_sparse_v3`，允许重放
  自洽的采集期 `passed=false` observation，并在发布边界排除空 response component。
- canonical/base release 的人工 cohort 与 approval 参数、文件和门禁已经删除；质量
  决策完全来自确定性自动 replay。
- base builder 已兼容当前 DeepSeek provider provenance，不再错误要求本地
  Qwen teacher、chk0 critic 或旧 verifier 字段；旧 sealed local-critic release 仍可走
  显式 legacy 分支。
- 运行手册已更新为 GPU0+GPU1 双卡 DDP、v7 路径与 7,168-token 训练合同。

## 当前剩余条件

数据与依赖已经满足 chk1 启动条件。`fomc_trainer` 已按训练 lock/freeze 清理旧
`open-r1` editable、vLLM、DeepSpeed、e2b、lighteval、xformers 等冲突包；Python
3.10.9、Torch 2.10.0+cu128 与完整 freeze、import、`pip check` 均通过。

正式 run `retrain_v2_full_v7_automated_v5_20260804` 已初始化，run manifest SHA256
为 `0c4c98f02cea011acf6eeceab451515d19ea29811a27fc0c9e837d78ee4c08ec`；chk1
immutable preflight 已返回 `status=ready`。

两张 A30 位于不同 NUMA 节点，拓扑为 `SYS`；direct P2P 会卡在 NCCL communicator
初始化。launcher 已固定 `NCCL_P2P_DISABLE=1` 与 `NCCL_IB_DISABLE=1`，共享内存
transport 的双卡 all-reduce 已通过，并增加 60 秒超时保护。两卡当前均空闲，各约
24,036 MiB free。

## 验证记录

- `run/retrain_v2/chk1_data.sh verify --mode full`：通过。
- canonical `chk1_full_v7_automated_v2_20260804`：发布并复验通过。
- base `analysis_base_full_v7_automated_v3_20260804`：构建并由 DAG 完整复验通过。
- 全量 2,072 行训练 token audit：全部通过。
- 单 rank（GPU1）7,168-token QLoRA smoke：通过。
- retrain-v2/chk1 联合回归：`494 passed`；核心 release/base/env/DAG 子集：
  `192 passed`。
- `fomc_trainer` CPU/import/freeze 检查：通过，`pip check` 无冲突。
- chk1 immutable parent/config/data/source/environment preflight：通过。
- 磁盘 resource gate：通过，空闲 450 GiB。
- 双 rank 7,168-token GPU smoke：通过；world size 2，两个 rank 的
  forward/backward/PagedAdamW8bit step 均完成，max peak allocated/reserved 为
  `14.692/14.932 GiB`。
