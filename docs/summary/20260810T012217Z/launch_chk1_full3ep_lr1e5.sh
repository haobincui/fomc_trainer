#!/usr/bin/env bash

set -euo pipefail

repo_root="/home/haobin_cui/research_files_space_2/fomc_trainer"
config_path="configs/retrain_v2/chk1_analysis_sft_clean_v2_override_full3ep_lr1e5_maxlen7168_20260810.yaml"
expected_config_sha256="5f692bc2b80ed1c22ce547edb3b09c88339fcff19b40ebaf7386d37a0c1c67b3"
output_root="output/training/retrain_v2/chk1_clean_v2_override_full3ep_lr1e5_maxlen7168_20260810"

cd "${repo_root}"

observed_config_sha256="$(sha256sum "${config_path}" | awk '{print $1}')"
if [[ "${observed_config_sha256}" != "${expected_config_sha256}" ]]; then
  echo "ERROR: training config digest changed before launch" >&2
  exit 1
fi

if [[ -e "${output_root}" ]]; then
  echo "ERROR: fresh training output already exists: ${output_root}" >&2
  exit 1
fi

./run/retrain_v2/gpu_gate.sh 0 1

export CUDA_VISIBLE_DEVICES=0,1
export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

exec conda run --no-capture-output -n fomc_trainer \
  python -m accelerate.commands.launch \
  --num_processes 2 \
  --mixed_precision bf16 \
  --dynamo_backend no \
  -m jobs.train.train_sft \
  --config "${config_path}"
