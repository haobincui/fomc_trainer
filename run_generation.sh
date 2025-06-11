#!/bin/bash



set -e


LINE="==============================================="


CURRENT_DATE=$(date +%Y%m%d_%H%M%S)

LOG_DIR=logs/generation
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/generation_${CURRENT_DATE}.log"

source activate fomc_trainer
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


nohup python script_generation.py  > "$LOG_FILE" 2>&1 &


echo " "
echo "✅ Start training PID: [$!]"
echo "✅ To monitor logs: tail -f $LOG_FILE"
echo "${LINE}"
