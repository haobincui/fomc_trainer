# chk4 Decision-SFT GPU1 启动记录

启动时间：2026-08-10T22:42:12Z

## 拓扑

- 仅使用物理 GPU1：`CUDA_VISIBLE_DEVICES=1`
- Accelerate 单进程：`--num_processes 1`
- per-device batch：1
- gradient accumulation：8
- effective optimizer batch：8
- GPU0 未被本训练使用

## 上游 parent

- source adapter：chk1 checkpoint-200
- chk4-scoped merged parent SHA-256：`6ad91dbb1571485df995c2443b97149badc3b8b6a6528aff82f6e2e8c824df1e`
- exact merge evidence file SHA-256：`c715e5694a3c95247dabb0b475ae3574b5822418f4c0f7680a1c6e14e5bde360`
- exact merge evidence payload SHA-256：`d5e74345d2bb6180f98f9c31873af94b88be7908d0953f09758657cfe224e051`
- parent stage receipt SHA-256：`6941511082a1099c3873cdc7e8a98adfebb96cff5c44b308195d16f7f5856b84`

## 运行位置

- tmux：`fomc_chk4_sft_gpu1`
- config：`configs/retrain_v2/chk4_decision_sft_from_chk1_cp200_core_v3_20260810.yaml`
- config SHA-256：`6e43b01a0ee438722a9075e78b27b4390c44514a81e65b6f0ef3b54590be9cef`
- adapter：`output/training/retrain_v2/chk4_from_chk1_cp200_sft_grpo_core_v3_20260810/adapters/chk4_sft`
- console log：`output/training/retrain_v2/chk4_from_chk1_cp200_sft_grpo_core_v3_20260810/logs/sft.log`
- loss log：`output/training/retrain_v2/chk4_from_chk1_cp200_sft_grpo_core_v3_20260810/adapters/chk4_sft/loss_history.jsonl`

## 首次门禁

- release/runtime/config/authorization preflight：passed
- fresh start：passed
- step 1–10：finite
- step 10 eval loss：`2.0737204551696777`
- step 10 checkpoint：完整保存
- retention：step 10 时保留 `checkpoint-8/9/10`
- GPU1 显存：约 9,993 MiB
- GPU0 显存：14 MiB
- OOM、CUDA fatal、NCCL fatal、Traceback：0

训练目标为 3 epochs、54 optimizer steps。训练仍在后台继续。
