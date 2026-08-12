# FOMC v2 重训执行方案（2×A30 24GB）

> 2026-08-04 增量更新：chk1 full-v7 数据完成后，completion 训练上限与 GPU
> 拓扑已按实测修正。最新执行证据见
> [`../20260804T100111Z/chk1_completion_limit_removal_and_readiness.md`](../20260804T100111Z/chk1_completion_limit_removal_and_readiness.md)。

## 目标与模型血缘

正式模型采用分叉关系：

```text
chk0  DeepSeek-R1-Distill-Llama-8B
  └─ chk1  FOMC 风格分析 SFT
       └─ chk2  分析 GRPO
            ├─ chk3  数据+chk2分析 → Minutes 原文
            └─ chk4  chk2分析 → 利率方向及幅度，直接 GRPO
```

chk3 和 chk4 必须引用同一个 chk2 哈希；chk4 不经过 chk3，也不增加 decision SFT。

## 环境与硬件

- 硬件为 2×NVIDIA A30 24GB、SM80、bf16；两卡无 NVLink，P2P 可用但跨 NUMA。
- 不做全参数训练；各训练阶段统一采用 NF4 QLoRA、bf16 compute、double quant、gradient checkpointing 和 paged AdamW 8-bit。chk1 使用经 A30 满长验证的 `flex_attention` 与 Liger；chk2/chk3/chk4 保持 SDPA，chk3 启用 Liger；GRPO 关闭 Liger。
- LoRA 默认 `r=32`、`alpha=64`、dropout `0.05`，覆盖 attention 与 MLP projections。
- chk1 使用物理 GPU0+GPU1 双 rank DDP；chk2 使用 GPU0 的 Qwen3.5-9B judge 服务和 GPU1 的 policy；chk3/chk4 使用双卡 DDP。
- GPU 单步峰值必须小于 22GiB。正式 GPU 任务启动前等待其他项目释放显卡，不终止任何外部进程。
- CPU 完成 adapter merge；磁盘可用空间低于 150GB 时暂停，启动正式训练前至少保留 250GB。

固定使用训练与 judge 两个角色环境：

- `fomc_trainer`：Python 3.10.9、Torch 2.10.0+cu128、Transformers 5.5.4、TRL 1.2.0、PEFT 0.15.2、Accelerate 1.4.0、Datasets 4.8.4、bitsandbytes 0.48.2、liger-kernel 0.8.1，以及评估依赖。安装器按训练 lock 的依赖闭包清理旧 editable/legacy 包。
- `fomc_judge_v2`：Python 3.10.9、Torch 2.10.0+cu128、vLLM 0.19.1、Transformers 5.5.4、bitsandbytes 0.48.2。
- 不安装旧 README 中的 `flash-attn==2.5.6`、xformers、旧 lighteval/e2b 或不需要的 DeepSpeed。
- 安装后必须通过完整 distribution 集合/精确版本冻结检查、`pip check`、CUDA/bf16、4-bit 模型加载、NCCL all-reduce、单步 forward/backward 和 Qwen API smoke test，并生成精确 lock 与 `pip freeze`。额外包、缺包、版本漂移、URL/editable 或范围依赖一律拒绝。`bitsandbytes<0.48.1` 与当前 vLLM dynamic-BnB 路径不兼容，因此禁止回退到旧 README 版本。

## 数据重建

- 现有 `dataset/processed` 保留为 v1 审计材料，新发布写入 `dataset/processed/retrain_v2/<release_id>/`。数据改为两阶段不可变发布，避免“chk3/chk4 输入依赖尚未训练的 chk2”这一循环依赖。
- base release 只包含 chk1/chk2 的 `analysis_sft` 与 `analysis_grpo`；先完成并 seal chk2。
- derived release 再用该精确 chk2 哈希、temperature 0、`do_sample=false` 生成冻结 analysis，并发布 chk3/chk4 数据。绑定后禁止 rebind；chk3/chk4 必须引用同一 sealed chk2 SHA。
- 复用仓库现有 ALFRED D-1 获取、原始响应和 manifest 机制，扩展到 train/eval/test。
- 宏观数据必须满足 `release_ts <= cutoff_ts` 且使用当时 vintage；市场数据不超过会议开始前最后一个交易日收盘。
- 删除决议后 `current_rate`，输入只保留上次会议后已经生效的目标区间。
- chk1/chk2 主键为 `meeting_date + atomic_topic`；chk3 为 `meeting_date + minutes_paragraph_id`；chk4 为一会议一例。
- 保留 post-2009 的 102/13/13 个时间 split；chk4 仅在 train 中补充清洗后的 1993–2008 会议。
- analysis 训练角色按稳定 sample ID 固定为 70% SFT-only、20% GRPO-only、10% shared replay。
- 数据发布门禁：零 split 交叉、零同输入冲突目标、零空表、零 reference 泄漏、零 cutoff 违规、零 mojibake、零运行时截断。

旧本地数据复用审计已完成：复用 point-in-time `provided_data`、topic、meeting split 与 Minutes 风格资产，不复用存在 same-meeting reference 泄漏或缺少 provenance 的旧 response。监督 target 已使用 reference-free DeepSeek teacher 重建为不可变 `generation_full_v7`；生成阶段在 2,117 个候选中接受 2,115 条、明确排除 2 条。canonical admission 又排除 43 条空 final，正式发布 2,072 条（train/eval/test 为 1,683/199/190）。发布质量只由可重放的自动门禁决定。

当前执行绑定为 canonical `chk1_full_v7_automated_v2_20260804`、base
`analysis_base_full_v7_automated_v3_20260804` 和 run
`retrain_v2_full_v7_automated_v5_20260804`。chk1 静态 preflight、跨 NUMA NCCL
all-reduce 和双 rank 7,168-token 满长 smoke 均已通过，可以启动正式 SFT。

固定 token 上限：

- chk1：prompt 不超过 3072，completion 无独立训练上限，总长不超过 7168；不截断。GPU1 上 `flex_attention` + Liger QLoRA r32 的 7168 满长实测 peak reserved 14.619GiB。manifest 中的 1024/4096 是不可变采集期 observation，不再作为训练准入门禁。
- chk2：prompt 2560、completion 1024、4 generations、单次 generation batch 4；实测 rollout/backward 峰值约 18.72/16.95GiB。
- chk3：prompt 3072、target 1024、总长不超过 4096；与 chk1 使用同一已验证 SFT 显存档。
- chk4：prompt 2560、completion 512；双卡下每 rank 本地 rollout 4 条，实测 rollout/backward 峰值约 18.72/15.72GiB。
- 所有阶段使用 `token_limit_policy=error`，通过确定性摘要压缩输入而不是静默截断。

## 训练阶段

### chk0

冻结本地 DeepSeek-R1-Distill-Llama-8B、tokenizer 和 chat template，记录完整 SHA256。DeepSeek 模板已经提供 reasoning 起始标记，completion 不重复输出 `<think>`。

### chk1：FOMC 风格分析 SFT

- 从 train Minutes 提炼去数字、日期、实体的固定风格指南。
- teacher 只看到 point-in-time 数据、topic 和风格指南；每个数字和趋势判断必须关联 evidence ID。
- 监督 response 全部来自当前固定 teacher acquisition；旧 response 只用于审计比较，不进入训练 target。
- 物理 GPU0+GPU1 双 rank QLoRA DDP；每 rank batch 1、gradient accumulation 8、有效 batch 16；2 epochs、LR `5e-5`、cosine、5% warmup、weight decay `0.01`。

### chk2：分析 GRPO

- 父模型为固定哈希的 merged chk1。
- GPU0 运行 Qwen3.5-9B judge，默认 4-bit、`max_model_len=6144`、`max_num_seqs=4`、`gpu_memory_utilization=0.72`；GPU1 训练 policy，`use_vllm=false`。
- Reward：`0.60×Qwen rubric + 0.25×确定性证据支持度 + 0.10×输出契约 + 0.05×简洁性`。
- judge 使用 temperature 0、严格 JSON、最多四路并发、3 次重试；服务失败后抛出基础设施错误并中止/可恢复续训，绝不把 judge 故障伪装成零 reward 或静默跳过。
- judge 请求在任何并发/HTTP 之前对完整 batch 做本地 Qwen tokenizer 精确计数，并要求 `prompt_tokens + 512 <= 6144`；训练前另做保守候选预留预算。任一越界停止整批，禁止运行时截断。服务 health probe 必须同时验证模型目录指纹、served alias、`max_model_len=6144`，并逐 ID 比较本地与 `/tokenize` 的完整 token 序列。
- judge 只接收证据和候选文本，不接收精确 meeting date、Minutes 或实际决议，避免利用预训练记忆反推历史结果。
- LR `5e-7`、`beta=0.04`、4 generations、temperature `0.7`、top-p `0.9`、1 epoch。
- 固定 2560/1024/4、generation batch 4。原 4096/1024/batch 8、4096/batch 4 均真实 OOM，3072/batch 4 的 reserved 22.64GiB 也超过安全线，因此不再作为默认档。

### chk3：Minutes SFT

- 从 chk2 分叉；使用冻结 chk2、temperature 0 重新生成 analysis。
- prompt 包含 point-in-time 数据、chk2 analysis 和 section/topic；target 只含清洗后的官方 Minutes paragraph/excerpt。
- 使用 no-think 模板，仅输出 Minutes prose；excerpt-level 训练，推理后按官方顺序组装 section。
- 两卡 QLoRA DDP；2 epochs、LR `2e-5`、每卡 batch 1、累积 8。

### chk4：直接决策 GRPO

- 从 chk2 直接初始化；一会议一例。
- 输出严格 JSON：`{"direction":"cut|hold|hike","magnitude_bp":0}`；hold 为 0，cut/hike 为 25/50/75/100。
- Reward：`0.05×格式 + 0.45×方向正确 + 0.30×同方向幅度接近度 + 0.20×精确 signed-bp`。
- 方向平衡采样，非 hold 内按幅度逆频率采样。
- 两卡 QLoRA DDP；LR `1e-6`、`beta=0.04`、4 generations、最多 3 epochs；每 rank prompt 上限 2560、completion 上限 512，显式 `ddp_find_unused_parameters=false`。
- 正式训练前先用冻结 chk2 做 decision rollout pilot：截断率 <10%、合法最终 JSON ≥20%；不达标时先修 prompt/输出契约，不关闭 `mask_truncated_completions`。

## 验证与产物

- Qwen judge 先用 200 条候选校准，至少 50 条双人复核；pairwise agreement ≥75%、Spearman ≥0.60。
- chk0/chk1/chk2 比较格式率、数字支持率、unsupported claim 和盲评；chk2 相对 chk1 盲评胜率至少 60%，事实错误不得上升。
- chk3 比较事实一致性、幻觉率、section 覆盖、ROUGE-L/BERTScore 和盲评。
- chk4 报告 macro-F1、balanced accuracy、精确 bp accuracy、bp MAE、Brier score、混淆矩阵及 bootstrap CI，并与 majority、lag-1、date-only 和 logistic baseline 比较。
- 每阶段使用唯一 run ID，禁止覆盖；保留 best、final、trainer state、adapter、merged model、配置、日志和完整校验和。
- run 初始化时封存训练源码精确文件集/哈希、requirements、Python、完整 installed distribution 集合、Torch/CUDA 和固定 GPU 拓扑；每个父模型、judge、数据绑定、merge 与 seal 边界都复验，防止运行中代码或环境漂移。
- stage 必须严格执行 `train -> record-training -> merge -> record-merge -> seal`。training/merge receipt 对 LoRA safetensors、parent、trainer state、非零训练统计、merged tensor/index/分片位置及 adapter 不变性做内容级验证；仅有目录或伪造占位文件不能被 seal。
- merge 从 adapter 副本在 CPU 执行，禁止修改源 adapter。
- 每个正式 checkpoint 复制到第二独立存储并完成一次恢复验证。
