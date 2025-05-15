#!/bin/bash

echo "Starting the Pipeline to generate response"
source activate fomc_trainer
LINE="--------------------------------------------------------------------------------------"

LOG_FILE=logs/train_grpo_$(date "+%Y%m%d_%H%M%S").log


mkdir -p "./logs"

echo " "
echo "${LINE}"

export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export TRANSFORMERS_VERBOSITY=debug


accelerate launch --config_file configs/accelerate/zero3.yaml train_grpo.py \
    --config configs/grpo/grpo_20250514.yaml



# nohup CUDA_VISIBLE_DEVICES=1 accelerate launch --config_file configs/accelerate/zero3.yaml train_grpo.py \
#     --config configs/grpo/grpo_20250514.yaml > "$LOG_FILE" 2>&1 &

echo "Finished training:"
echo "tail -f $LOG_FILE"
