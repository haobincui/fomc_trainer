#!/bin/bash
set -e

LINE="==============================================="

LOG_DIR=logs/train
mkdir -p "$LOG_DIR"
LOG_FILE=$LOG_DIR/grpo_$(date "+%Y%m%d").log


source activate fomc_trainer
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export TRANSFORMERS_VERBOSITY=debug

CONFIG=configs/grpo/grpo_analysis_meeting_level.yaml
# CONFIG=configs/grpo/grpo_20250521.yaml
# CONFIG=configs/grpo/grpo_decision_20250601.yaml
ACCELERATE_CONFIG=configs/accelerate/zero2.yaml

export OPEN_R1_JUDGE_URL=${OPEN_R1_JUDGE_URL:-http://localhost:11432/api/chat/}
export OPEN_R1_JUDGE_MODEL=${OPEN_R1_JUDGE_MODEL:-gemma3:12b}

echo " "
echo "${LINE}"
echo "Initializing **GRPO** training"
echo "Using config: [$CONFIG]"
echo "Using accelerate config: [$ACCELERATE_CONFIG]"
echo "Log file: [$LOG_FILE]"
echo "Using conda env: $(which python)"
echo "CUDA_VISIBLE_DEVICES: [$CUDA_VISIBLE_DEVICES]"
echo "Judge URL: [$OPEN_R1_JUDGE_URL]"
echo "Judge Model: [$OPEN_R1_JUDGE_MODEL]"
echo "${LINE}"

nohup accelerate launch --config_file "$ACCELERATE_CONFIG" scripts/train/train_grpo.py \
    --config "$CONFIG" > "$LOG_FILE" 2>&1 &

echo " "
echo "✅ Start training PID: [$!]"
echo "✅ To monitor logs: tail -f $LOG_FILE"
echo "${LINE}"



# accelerate launch --config_file configs/accelerate/zero3.yaml train_grpo.py \
#     --config configs/grpo/grpo_20250514.yaml
