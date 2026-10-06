#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 {smoke|generate|validate} EVALUATION_MANIFEST OUTPUT_ROOT" >&2
  exit 2
fi

mode="$1"
evaluation_manifest="$2"
output_root="$3"
case "${mode}" in
  smoke|generate|validate) ;;
  *) echo "mode must be smoke, generate, or validate" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vllm_python="${VLLM_PYTHON_BIN:-/home/haobin_cui/.conda/envs/vllm_env/bin/python}"
module="jobs.eval.paper_chk2_text_similarity_vllm"

[[ -x "${vllm_python}" ]] || { echo "missing vLLM Python: ${vllm_python}" >&2; exit 2; }
[[ -f "${evaluation_manifest}" && ! -L "${evaluation_manifest}" ]] || {
  echo "missing/unsafe evaluation manifest: ${evaluation_manifest}" >&2
  exit 2
}

mkdir -p "${output_root}"
output_root="$(cd "${output_root}" && pwd)"
evaluation_manifest="$(cd "$(dirname "${evaluation_manifest}")" && pwd)/$(basename "${evaluation_manifest}")"

export PYTHONPATH="${repo_root}/src:${repo_root}"
export PYTHONNOUSERSITE=1
export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
unset VLLM_DP_MASTER_IP VLLM_DP_MASTER_PORT MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK || true

cd "${repo_root}"

if [[ "${mode}" == "validate" ]]; then
  exec "${vllm_python}" -u -m "${module}" validate \
    --manifest "${evaluation_manifest}" \
    --output-root "${output_root}"
fi

worker_args=(
  "${mode}"
  --manifest "${evaluation_manifest}"
  --output-root "${output_root}"
  --resume
)
if [[ "${mode}" == "smoke" ]]; then
  worker_args+=(--smoke-samples 6 --smoke-replicates 2)
fi

gpu_stability_log="${output_root}/gpu-memory-stability.log"
for sample_index in 0 1 2 3 4 5 6; do
  if ! gpu_rows_raw="$(nvidia-smi --id=0,1 \
    --query-gpu=index,memory.free,memory.total \
    --format=csv,noheader,nounits)"; then
    echo "nvidia-smi failed during GPU stability gate" >&2
    exit 1
  fi
  mapfile -t gpu_rows <<<"${gpu_rows_raw}"
  [[ "${#gpu_rows[@]}" -eq 2 ]] || {
    echo "GPU stability gate expected exactly two GPU rows" >&2
    exit 1
  }
  seen_gpu0=0
  seen_gpu1=0
  for gpu_row in "${gpu_rows[@]}"; do
    IFS=',' read -r observed_gpu free_mib total_mib <<<"${gpu_row}"
    observed_gpu="${observed_gpu//[[:space:]]/}"
    free_mib="${free_mib//[[:space:]]/}"
    total_mib="${total_mib//[[:space:]]/}"
    [[ "${observed_gpu}" == "0" || "${observed_gpu}" == "1" ]] || {
      echo "unexpected GPU index during stability gate: ${observed_gpu}" >&2
      exit 1
    }
    if [[ "${observed_gpu}" == "0" ]]; then seen_gpu0=1; else seen_gpu1=1; fi
    [[ "${free_mib}" =~ ^[0-9]+$ && "${total_mib}" =~ ^[0-9]+$ ]] || {
      echo "non-numeric GPU memory observation" >&2
      exit 1
    }
    (( total_mib > 0 && free_mib > 2048 )) || {
      echo "GPU${observed_gpu} lacks the required 2048 MiB headroom" >&2
      exit 1
    }
    utilization_hundredths=$(( (free_mib - 2048) * 100 / total_mib ))
    (( utilization_hundredths >= 78 )) || {
      echo "GPU${observed_gpu} stable utilization budget is below 0.78" >&2
      exit 1
    }
    printf 'sample=%s gpu=%s free_mib=%s total_mib=%s utilization_floor=0.%02d\n' \
      "${sample_index}" "${observed_gpu}" "${free_mib}" "${total_mib}" \
      "${utilization_hundredths}" >>"${gpu_stability_log}"
  done
  (( seen_gpu0 == 1 && seen_gpu1 == 1 )) || {
    echo "GPU stability gate did not observe both physical GPUs" >&2
    exit 1
  }
  if [[ "${sample_index}" -lt 6 ]]; then sleep 5; fi
done

CUDA_VISIBLE_DEVICES=0 "${vllm_python}" -u -m "${module}" \
  "${worker_args[@]}" --shard-id 0 >"${output_root}/worker-0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "${vllm_python}" -u -m "${module}" \
  "${worker_args[@]}" --shard-id 1 >"${output_root}/worker-1.log" 2>&1 &
pid1=$!

status0=0
status1=0
wait "${pid0}" || status0=$?
wait "${pid1}" || status1=$?
if [[ "${status0}" -ne 0 || "${status1}" -ne 0 ]]; then
  echo "worker failure: shard0=${status0} shard1=${status1}" >&2
  exit 1
fi

validate_args=(
  validate
  --manifest "${evaluation_manifest}"
  --output-root "${output_root}"
)
if [[ "${mode}" == "smoke" ]]; then
  validate_args+=(--smoke --smoke-samples 6 --smoke-replicates 2)
fi
"${vllm_python}" -u -m "${module}" "${validate_args[@]}"
