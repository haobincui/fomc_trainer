#!/bin/bash
set -e

source activate vllm_env
export CUDA_VISIBLE_DEVICES=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "Using conda env: $(which python)"

MODEL=${MODEL:-output/merged/llama_sft_20250522}
# MODEL=${MODEL:-models/Qwen3-14B-unsloth-bnb-4bit}
PORT=${PORT:-8000}
LOG_DIR=logs/vllm_service
mkdir -p "$LOG_DIR"
LOG_FILE=$LOG_DIR/vllm_$(date "+%Y%m%d_%H%M%S").log

LINE="==============================================="
echo " "
echo "${LINE}"
echo "Starting vLLM Service..."
echo "Model: $MODEL"
echo "Port: $PORT"
echo "Log: $LOG_FILE"
echo "${LINE}"

nohup python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 8192 \
    --tensor-parallel-size 1 \
    --max-num-seqs 1 \
    --port "$PORT" > "$LOG_FILE" 2>&1 &

echo " "
echo "${LINE}"
echo "✅ vLLM service started for model [$MODEL] on port [$PORT]"
echo "✅ PID: $!"
echo "✅ Logs: tail -f $LOG_FILE"
echo "${LINE}"


# python -m vllm.entrypoints.openai.api_server \
#     --model models/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit \
#     --dtype bfloat16 \
#     --gpu-memory-utilization 0.85 \
#     --max-model-len 16384 \
#     --tensor-parallel-size 1 \
#     --max-num-seqs 1 \
#     --port 8000

