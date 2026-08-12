# retrain_v2：从 chk1 输入到 chk2 的执行手册

本文只描述 provenance-controlled retrain-v2 路径。旧的 `configs/main/`、
`run/generate_input/` 和默认 Python 环境不是本次重训入口。

模型血缘固定为：

```text
chk0  DeepSeek-R1-Distill-Llama-8B
  └─ chk1  analysis SFT
       └─ chk2  analysis GRPO
            ├─ chk3  Minutes SFT
            └─ chk4  decision GRPO
```

chk4 不经过 decision SFT，chk3/chk4 都从同一个 sealed chk2 分叉。

## 当前状态（2026-08-04 UTC）

DeepSeek full generation 已完成并复验：2,117 个候选中生成阶段接受 2,115 条，明确
排除 2 条无效 generator result。canonical admission 进一步排除了 43 条 reasoning
非空但 final analysis 为空的样本，最终可训练 split 为 train/eval/test
`1683/199/190`，共 2,072 条。

当前不可变 canonical release 是
`chk1_full_v7_automated_v2_20260804`，handoff SHA256 为
`7eb1ceb8e3ea41b6305e40f15d4d6492ced64d45de370118a2f52ddf121a5448`；
最终 analysis base 是 `analysis_base_full_v7_automated_v3_20260804`，artifact SHA256
为 `b8128b394420f90d8e7cb2e0e422a7dc097b5d6c68adb30f73ded29f764c040f`，完整验证已通过。
release 只依赖自动、可重放的质量门禁。较早的
`chk1_full_v7_automated_20260804` 是历史产物，包含空 final，不得训练。

正式 run `retrain_v2_full_v7_automated_v5_20260804` 已初始化；chk1 不带
`--execute` 的 immutable preflight 返回 `status=ready`。当前 GPU0 仍有外部 `py312`
compute process，因此双卡 GPU gate 会在模型加载前停止；待两卡空闲后执行第 6 节命令。

双卡 chk1 launcher 已配置完成。若 GPU0 或 GPU1 存在其他任务，GPU gate 会在任何
模型加载前停止；等待两张卡都空闲后重试同一命令，不要终止外部进程。

旧本地 `deepseek-reasoner` response 不能直接作为 retrain-v2 target：旧 prompt
包含 same-meeting reference、current rate/rate change，且没有固定 revision、
response ID 或 evidence ID。可复用的是 point-in-time 输入、split、topic 和风格
资产。详见
[`chk1_local_reuse_audit.md`](../../docs/summary/20260803T150258Z/chk1_local_reuse_audit.md)。

当前 chk1 teacher 路径是 `deepseek-v4-pro` thinking API：`reasoning_content` 蒸馏为
`analysis`，JSON `content.answer` 蒸馏为 `answer`。这一步不运行 verifier、critic
或 reward model；基础输出合同、采集期 token observation、缓存与 handoff 完整性仍然强制
执行。Qwen3.5-9B 不参与 chk1，只作为 chk2 GRPO 的 LLM reward/judge。远程请求
默认并发 8（`DEEPSEEK_CONCURRENCY`，上限 32）。凭证只能通过
`DEEPSEEK_API_KEY` 注入，禁止写入配置文件。

## 固定环境与 2×A30 拓扑

不要在 base/default Python 中安装或训练。入口固定使用：

- `fomc_trainer`：chk1 数据验证、release builder 和正式训练。
- `fomc_judge_v2`：Qwen3.5-9B vLLM judge。

只读环境检查：

```bash
run/check_retrain_v2_envs.sh --train --skip-gpu
run/check_retrain_v2_envs.sh --judge --skip-gpu
run/retrain_v2/resource_gate.sh
```

如果版本/freeze 检查失败，不要在环境里零散执行 `pip install`。用锁文件驱动的安装器
修复对应隔离环境；GPU0 仍被占用时先跳过安装后的 CUDA 检查，再执行上面的 CPU
合同检查：

```bash
run/setup_retrain_v2_envs.sh --train --skip-checks
run/setup_retrain_v2_envs.sh --judge --skip-checks
run/check_retrain_v2_envs.sh --train --skip-gpu
run/check_retrain_v2_envs.sh --judge --skip-gpu
```

安装器会固定 Python/pip/Torch 与 lock 中的精确版本，自动移除训练依赖闭包以外的旧
`open-r1` editable、vLLM、DeepSpeed、e2b、lighteval、xformers 等包，并重写精确
freeze；它不会删除 conda 环境。若已有同名环境的 Python 不是 3.10.9，安装器会停止。
依赖或 freeze 在正式 run 初始化后发生变化时，应创建新的 run ID。

正式硬件角色不可重映射：

- chk1：物理 GPU0+GPU1，双 rank DDP bf16 QLoRA；每个 rank 使用
  `flex_attention`、Liger、batch 1、gradient accumulation 8，有效 batch 16。
- 两张 A30 跨 NUMA 且拓扑为 `SYS`。本机 direct P2P 会卡在 NCCL communicator
  初始化，因此 launcher 固定 `NCCL_P2P_DISABLE=1`、`NCCL_IB_DISABLE=1`，使用已
  实测通过的共享内存 transport；不要删除或覆盖这两个设置。
- chk2：GPU0 运行 Qwen3.5-9B judge；GPU1 运行 policy QLoRA。
- GPU 门禁要求每张目标卡没有 compute process、至少 22,000 MiB free、利用率
  不高于 10%。
- stage 训练前要求至少 250 GiB 可用磁盘；CPU merge 前要求至少 150 GiB。

环境的 CPU/import/`pip check` 合同和单 rank 7,168-token QLoRA
forward/backward/optimizer smoke 已通过。chk1 launcher 会在两卡均空闲后先执行
NCCL 检查和双 rank 7,168-token DDP smoke，再进入正式训练。chk2 阶段改为 GPU0
运行 judge、GPU1 运行 policy。

## 1. 验证 prepared 输入

默认 prepared handoff：

```text
output/data/retrain_v2/chk1/prepared_sparse_v3/prepare_handoff.json
```

先执行无写入、无 teacher/model call 的 dry-run：

```bash
run/retrain_v2/chk1_data.sh dry-run
```

dry-run 必须报告 `population_count=2117`、full selection 2,117，并完成 preparation
文件 SHA、row SHA、split/topic/PIT lineage、当前模型/Tokenizer provenance 和安全投影
预算检查。它不会创建 generation 输出，也不会使用 GPU。

## 2. 生成 chk1 teacher targets

DeepSeek teacher acquisition 只使用 API，不调用本地模型或 GPU。
`DEEPSEEK_API_KEY` 已设置后，按以下顺序运行：

```bash
run/retrain_v2/chk1_data.sh generate --mode smoke
run/retrain_v2/chk1_data.sh verify --mode smoke

run/retrain_v2/chk1_data.sh generate --mode pilot
run/retrain_v2/chk1_data.sh verify --mode pilot

run/retrain_v2/chk1_data.sh generate --mode full
run/retrain_v2/chk1_data.sh verify --mode full
```

规模固定为 smoke 20、pilot 200、full 2,117。总接受率至少 70%，每个 topic
接受率至少 50%；prepared projection、teacher output contract 和采集期
`3072+1024<=4096` observation 都保留在不可变 manifest 中。采集期 `passed=false`
不再等同于训练排除：训练发布器会用 chk0 tokenizer 重新计算 `prompt<=3072`、
`completion` 无独立上限、`total<=7168`，且禁止截断。

如果某阶段因基础设施中断且已有 sample cache，使用同一命令追加 `--resume`：

```bash
run/retrain_v2/chk1_data.sh generate --mode full --resume
```

`--resume` 会复验已有 cache/handoff；它不是忽略哈希错误的强制开关。不要手工修改
generation 目录，也不要删除单个坏行后继续。

## 3. 发布 immutable canonical release

```bash
run/retrain_v2/chk1_data.sh publish \
  --release-id <CANONICAL_RELEASE_ID>
```

默认 handoff：

```text
output/data/retrain_v2/chk1/canonical_releases/<CANONICAL_RELEASE_ID>/handoff.json
```

发布器会重新执行完整 full-generation replay，包括 prepared 安全投影、真实 chk0
Tokenizer token audit、所有 sample-cache terminal 等价性、generation code/model/
Tokenizer provenance，以及 generation/prepare 全文件图。它自动排除缺失 reasoning
或 final analysis 的 terminal，并把原因写入 `audit/exclusions.jsonl`；人口总数必须等于
可训练样本与 exclusions 之和。release 使用原子 no-clobber 发布；同一输入重跑只做
幂等复验，不会覆盖不同输入。

## 4. 构建 analysis base release

canonical release 不能直接交给 run initializer；必须经过独立 base builder：

```bash
conda run --no-capture-output -n fomc_trainer \
  python -m jobs.retrain_v2.build_base_release \
  --repo-root "$PWD" \
  --canonical-handoff \
    output/data/retrain_v2/chk1/canonical_releases/<CANONICAL_RELEASE_ID>/handoff.json \
  --base-release-id <BASE_RELEASE_ID>
```

成功产物：

```text
dataset/processed/retrain_v2/<BASE_RELEASE_ID>/base_release_manifest.json
```

builder 独立重放 source generation/preparation、canonical admission、PIT lineage、
reference leakage、split、target consistency 和实际 chk0 Tokenizer chat/token
semantics；然后生成 `analysis_sft` 与 `analysis_grpo`。不要手工创建或修改 base
release。`--recover-stale-staging` 只用于已检查且确认属于同一 immutable 输入的
中断 staging。

## 5. 只读验收并初始化唯一 run

在代码、requirements、环境和 base release 都稳定后选一个从未使用的 run ID：

```bash
run/retrain_v2/pipeline.sh \
  --run-id <RUN_ID> \
  --base-release-id <BASE_RELEASE_ID>
```

不带 `--initialize` 时只执行 DAG plan 和完整 base-release validation，不创建 run。
通过后只初始化一次：

```bash
run/retrain_v2/pipeline.sh \
  --run-id <RUN_ID> \
  --base-release-id <BASE_RELEASE_ID> \
  --initialize
```

正式 manifest：

```text
output/training/retrain_v2/<RUN_ID>/run_manifest.json
```

初始化会固定源码、requirements、完整 installed distributions、Python/Torch/CUDA、
GPU 拓扑、chk0、judge 和 base release SHA。初始化后任何这些内容变化都会阻断该
run；源码变化后应使用新 run ID，不能修补旧 manifest。

本次已经初始化的固定组合是：

```text
RUN_ID=retrain_v2_full_v7_automated_v5_20260804
BASE_RELEASE_ID=analysis_base_full_v7_automated_v3_20260804
```

## 6. 训练、merge 并 seal chk1

先做 immutable parent/config/data 预检：

```bash
run/retrain_v2/stage.sh \
  --run-manifest output/training/retrain_v2/<RUN_ID>/run_manifest.json \
  --stage chk1
```

物理 GPU0 和 GPU1 都空闲时执行：

```bash
run/retrain_v2/stage.sh \
  --run-manifest output/training/retrain_v2/<RUN_ID>/run_manifest.json \
  --stage chk1 \
  --execute
```

`--execute` 在同一独占 stage lock 中完成或恢复：token/environment/GPU preflight、
双卡 NCCL 与满长 DDP smoke、双 rank SFT、training receipt、CPU merge、merge
attestation/receipt 和 seal。中断后重跑
同一命令；launcher 会根据 immutable recovery state 续做，不能删除 receipt 或覆盖
adapter/merged 目录。

## 7. 启动 judge 并训练到 chk2

chk1 必须已 sealed。正式长任务使用持久化 launcher；它在独立 tmux session 中启动
GPU0 judge，等待固定模型 endpoint healthy，再执行 GPU1 chk2，并把 judge、训练和
编排日志分别写入 run 的 `logs/`。训练结束并完成 merge/seal 后会自动停止 judge：

```bash
run/retrain_v2/start_chk2_background.sh \
  --run-manifest output/training/retrain_v2/<RUN_ID>/run_manifest.json
```

仅做交互式调试时，才在独立终端直接启动 judge：

```bash
run/retrain_v2/judge.sh
```

固定合同为 GPU0、`models/Qwen3.5-9B`、served alias `Qwen3.5-9B`、
`127.0.0.1:8000`、`max_model_len=8192`、Judge 输出上限 2048、
`max_num_seqs=4`、dynamic BnB、`gpu_memory_utilization=0.72`。chk2 每个优化步
保存 checkpoint；持久化 launcher 最多自动尝试 6 次，每次都会重新启动并验证 Judge，
随后从最新完整 checkpoint 恢复。不要用其他服务冒充同 alias。

judge healthy 后，可先在另一终端做 chk2 parent/config/data 预检：

```bash
run/retrain_v2/stage.sh \
  --run-manifest output/training/retrain_v2/<RUN_ID>/run_manifest.json \
  --stage chk2
```

再执行 policy（固定 GPU1）：

```bash
run/retrain_v2/stage.sh \
  --run-manifest output/training/retrain_v2/<RUN_ID>/run_manifest.json \
  --stage chk2 \
  --execute
```

chk2 launcher 会在训练前/后记录并离线复验 judge service identity、模型目录、
Tokenizer parity 和 health；judge 必须保持运行直到 post attestation 完成。随后同一
launcher 记录 training receipt、CPU merge、merge receipt 并 seal chk2。失败时重跑
同一 `--execute` 命令，不要更换 URL/model/port 或伪造 health 响应。

## 当前不要执行的操作

- 不要把 2,117 条 prepared 输入称为 2,117 条 SFT target。
- 不要复用旧 `deepseek-reasoner` response 作为 target。
- 不要在 base release 发布前初始化正式 run。
- 不要在 GPU0 或 GPU1 存在外部进程时启动 chk1；不要在 GPU0 外部进程存在时启动 judge。
- 不要终止、暂停或迁移不属于本任务的 GPU 进程。
- 不要使用 base Python、旧 README 的 `flash-attn==2.5.6` 环境或旧 main workflow
  启动本次 retrain-v2。

## CLI 与自检

实际帮助入口：

```bash
conda run -n fomc_trainer python -m jobs.retrain_v2.chk1.workflow --help
conda run -n fomc_trainer python -m jobs.retrain_v2.chk1.canonical_workflow --help
conda run -n fomc_trainer python -m jobs.retrain_v2.build_base_release --help
run/retrain_v2/pipeline.sh --help
run/retrain_v2/stage.sh --help
```

canonical workflow 的精确范围回归命令：

```bash
conda run -n fomc_trainer python -m pytest -q \
  tests/test_retrain_v2_chk1_canonical_workflow.py \
  tests/test_chk1_release.py \
  tests/test_chk1_generation_pipeline.py \
  tests/test_retrain_v2_chk1_workflow.py
```

```bash
CUDA_VISIBLE_DEVICES=1 conda run -n fomc_trainer python -m pytest -q \
  tests/test_retrain_v2_*.py tests/test_chk1_*.py \
  tests/test_optional_dependency_detection.py
```

以本次实际命令输出为准；不要沿用旧文档中的历史 pass 数。
