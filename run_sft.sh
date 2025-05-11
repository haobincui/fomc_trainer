#!/bin/bash

echo "Starting the Pipeline to generate response"
source activate fomc_trainer
CURRENT_DATE=$(date +%Y%m%d)
LINE="--------------------------------------------------------------------------------------"

LOG_FILE="./logs/train_${CURRENT_DATE}.log"

mkdir -p "./logs"

echo " "
echo "${LINE}"


# nohup cve-cli run \
# --input_file="./input/llm_data.xlsx" \
# --use_async=True \
#   > "$LOG_FILE" 2>&1 &


nohup accelerate launch --config_file configs/accelerate/zero3.yaml train_sft.py \
    --config configs/sft/sft_20250508.yaml > "$LOG_FILE" 2>&1 &

echo "Finished training:"
echo "tail -f $LOG_FILE"
