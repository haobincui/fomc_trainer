

source activate vllm_env

export CUDA_VISIBLE_DEVICES=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -m vllm.entrypoints.openai.api_server \
    --model models/DeepSeek-R1-Distill-Qwen-14B-unsloth-bnb-4bit \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.85 \
    --max-model-len 16384 \
    --tensor-parallel-size 1 \
    --max-num-seqs 1 \
    --port 8000
LOG_FILE=logs/vllm_service/train_grpo_$(date "+%Y%m%d_%H%M%S").log

mkdir -p  "logs/vllm_service"


# echo " "
# echo "${LINE}"



# nohup python -m vllm.entrypoints.openai.api_server \
#     --model DeepSeek-AI/DeepSeek-R1-Distill-Llama-8B-unsloth-bnb-4bit \
#     --dtype auto \
#     --gpu-memory-utilization 0.95 \
#     --max-model-len 16384 \
#     --tensor-parallel-size 1 \
#     --port 8000 > "$LOG_FILE" 2>&1 &

echo "Started Vllm service, "
echo "tail -f $LOG_FILE"

