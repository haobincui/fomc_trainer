#!/bin/bash



set -e


LINE="==============================================="


CURRENT_DATE=$(date +%Y%m%d_%H%M%S)

LOG_DIR=logs/train
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/sft_${CURRENT_DATE}.log"

source activate fomc_trainer
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# CONFIG=configs/sft/sft_20250515.yaml
# CONFIG=configs/sft/sft_synthetic_20250521.yaml
CONFIG=configs/sft/sft_decision_20250526.yaml
ACCELERATE_CONFIG=configs/accelerate/zero3.yaml


echo " "
echo "${LINE}"
echo "Initializing **SFT** training"
echo "Using config: [$CONFIG]"
echo "Using accelerate config: [$ACCELERATE_CONFIG]"
echo "Log file: [$LOG_FILE]"
echo "Using conda env: $(which python)"
echo "CUDA_VISIBLE_DEVICES: [$CUDA_VISIBLE_DEVICES]"
echo "${LINE}"

nohup accelerate launch --config_file "$ACCELERATE_CONFIG" train_sft.py \
    --config "$CONFIG" > "$LOG_FILE" 2>&1 &


echo " "
echo "✅ Start training PID: [$!]"
echo "✅ To monitor logs: tail -f $LOG_FILE"
echo "${LINE}"

# accelerate launch --config_file configs/accelerate/zero3.yaml train_grpo.py \
#     --config configs/grpo/grpo_20250514.yaml

# CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file configs/accelerate/zero3.yaml script_merge_model.py






