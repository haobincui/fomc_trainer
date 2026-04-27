#!/bin/bash

set -e


LINE="==============================================="


CURRENT_DATE=$(date +%Y%m%d_%H%M%S)

LOG_DIR=logs/eval
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/eval_mask_${CURRENT_DATE}.log"

source activate fomc_trainer
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


nohup python scripts/eval/eval_mask.py test3 \
    --input-folder output/validation/generation_stage2_synthetic/shapley/latest/ft_model/after_2009 \
    --output-file output/validation/generation_stage2_synthetic/shapley/latest/shapley_result/test3_shapley_result_with_diff.jsonl > "$LOG_FILE" 2>&1 &


echo " "
echo "✅ Start training PID: [$!]"
echo "✅ To monitor logs: tail -f $LOG_FILE"
echo "${LINE}"
