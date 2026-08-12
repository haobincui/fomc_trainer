#!/usr/bin/env bash
set -u

run_dir="/home/haobin_cui/research_files_space_2/fomc_trainer/output/training/retrain_v2/chk2_compressed_chk1_v1_reward_v3_long4096_fresh_20260807"
log="$run_dir/logs/chk2.log"
watch="$run_dir/logs/watchdog.log"

while true; do
  now=$(date -u +%Y-%m-%dT%H:%M:%SZ)
  if grep -Eq 'Training completed|train_runtime|Training finished' "$log" 2>/dev/null; then
    echo "$now COMPLETE sleeping600" >> "$watch"
    sleep 600
    echo "$(date -u +%Y-%m-%dT%H:%M:%SZ) EXIT" >> "$watch"
    break
  fi

  pid=$(ps -eo pid=,args= | grep '[p]ython -u -m jobs.train.train_grpo' | grep -F "$run_dir" | sed -n '1s/[[:space:]].*//p')
  if [ -z "$pid" ]; then
    echo "$now MISSING restarting" >> "$watch"
    CUDA_VISIBLE_DEVICES=1 nohup /home/haobin_cui/.conda/envs/fomc_trainer/bin/python -u -m jobs.train.train_grpo \
      --config /home/haobin_cui/research_files_space_2/fomc_trainer/configs/retrain_v2/chk2_analysis_grpo_compressed_chk1_v1_20260805.yaml \
      --output_dir "$run_dir/adapters/chk2" \
      --peft_merged_model_path "$run_dir/merged/chk2" >> "$log" 2>&1 &
    echo "$now STARTED pid=$! CUDA_VISIBLE_DEVICES=1" >> "$watch"
  else
    echo "$now RUNNING pid=$pid" >> "$watch"
  fi
  sleep 60
done
