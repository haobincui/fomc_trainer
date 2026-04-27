#!/bin/bash

set -e


LINE="==============================================="


CURRENT_DATE=$(date +%Y%m%d_%H%M%S)

LOG_DIR=logs/generation
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/generation_${CURRENT_DATE}.log"

source activate fomc_trainer
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


nohup python scripts/generation/data_to_analysis_generation.py \
    --input-folder after_2009 \
    --model output/merged/llama_sft_synthetic_20250526 \
    --simulation-step 1 \
    --output-dir output/validation/generation_stage2_synthetic/data_to_analysis/latest/ft_model > "$LOG_FILE" 2>&1 &


echo " "
echo "✅ Start training PID: [$!]"
echo "✅ To monitor logs: tail -f $LOG_FILE"
echo "${LINE}"
