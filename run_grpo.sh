#!/bin/bash
set -e

LINE="==============================================="

LOG_DIR=logs/train
mkdir -p "$LOG_DIR"
LOG_FILE=$LOG_DIR/grpo_$(date "+%Y%m%d").log


source activate fomc_trainer
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export TRANSFORMERS_VERBOSITY=debug

CONFIG=configs/grpo/grpo_20250515.yaml
ACCELERATE_CONFIG=configs/accelerate/zero3.yaml

echo " "
echo "${LINE}"
echo "Initializing **GRPO** training"
echo "Using config: [$CONFIG]"
echo "Using accelerate config: [$ACCELERATE_CONFIG]"
echo "Log file: [$LOG_FILE]"
echo "Using conda env: $(which python)"
echo "CUDA_VISIBLE_DEVICES: [$CUDA_VISIBLE_DEVICES]"
echo "${LINE}"

nohup accelerate launch --config_file "$ACCELERATE_CONFIG" train_grpo.py \
    --config "$CONFIG" > "$LOG_FILE" 2>&1 &

echo " "
echo "✅ Start training PID: [$!]"
echo "✅ To monitor logs: tail -f $LOG_FILE"
echo "${LINE}"



# accelerate launch --config_file configs/accelerate/zero3.yaml train_grpo.py \
#     --config configs/grpo/grpo_20250514.yaml
