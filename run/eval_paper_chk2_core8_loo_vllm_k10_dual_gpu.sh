#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 {smoke|generate|validate} EVALUATION_MANIFEST RUN_ROOT" >&2
  exit 2
fi

mode="$1"
evaluation_manifest="$2"
run_root="$3"
case "${mode}" in
  smoke|generate|validate) ;;
  *) echo "mode must be smoke, generate, or validate" >&2; exit 2 ;;
esac

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
vllm_python="${VLLM_PYTHON_BIN:-/home/haobin_cui/.conda/envs/vllm_env/bin/python}"
module="jobs.eval.eval_paper_chk2_core8_loo_vllm_k10_dual_dp1"

[[ -x "${vllm_python}" ]] || {
  echo "missing vLLM Python: ${vllm_python}" >&2
  exit 2
}
[[ -f "${evaluation_manifest}" && ! -L "${evaluation_manifest}" ]] || {
  echo "missing/unsafe evaluation manifest: ${evaluation_manifest}" >&2
  exit 2
}

mkdir -p "${run_root}"
run_root="$(cd "${run_root}" && pwd)"
evaluation_manifest="$(cd "$(dirname "${evaluation_manifest}")" && pwd)/$(basename "${evaluation_manifest}")"
mode_root="${run_root}/generation"
if [[ "${mode}" == "smoke" ]]; then
  mode_root="${run_root}/smoke"
fi
mkdir -p "${mode_root}"

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
    --output-root "${run_root}"
fi

worker_args=(
  worker
  --manifest "${evaluation_manifest}"
  --output-root "${run_root}"
  --resume
)
if [[ "${mode}" == "smoke" ]]; then
  worker_args+=(--smoke --smoke-meetings 2)
fi

# Seven observations five seconds apart provide a 30-second stability window.
# The workers independently recompute and bind the final vLLM memory fraction.
gpu_stability_log="${mode_root}/gpu-memory-stability.log"
stability_attempt=0
while true; do
  stability_attempt=$((stability_attempt + 1))
  stability_pass=1
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
      (( total_mib > 0 )) || {
        echo "GPU${observed_gpu} reported an invalid total memory" >&2
        exit 1
      }
      utilization_hundredths=0
      if (( free_mib > 2048 )); then
        utilization_hundredths=$(( (free_mib - 2048) * 100 / total_mib ))
      fi
      if (( utilization_hundredths < 78 )); then
        stability_pass=0
      fi
      printf 'attempt=%s sample=%s gpu=%s free_mib=%s total_mib=%s utilization_floor=0.%02d pass=%s\n' \
        "${stability_attempt}" "${sample_index}" "${observed_gpu}" \
        "${free_mib}" "${total_mib}" "${utilization_hundredths}" \
        "$(( utilization_hundredths >= 78 ))" >>"${gpu_stability_log}"
    done
    (( seen_gpu0 == 1 && seen_gpu1 == 1 )) || {
      echo "GPU stability gate did not observe both physical GPUs" >&2
      exit 1
    }
    if (( stability_pass == 0 )); then
      break
    fi
    if [[ "${sample_index}" -lt 6 ]]; then sleep 5; fi
  done
  if (( stability_pass == 1 )); then
    break
  fi
  printf 'attempt=%s status=waiting_for_both_gpus_minimum_0.78\n' \
    "${stability_attempt}" >>"${gpu_stability_log}"
  sleep 30
done

CUDA_VISIBLE_DEVICES=0 "${vllm_python}" -u -m "${module}" \
  "${worker_args[@]}" --shard-id 0 >"${mode_root}/worker-0.log" 2>&1 &
pid0=$!
CUDA_VISIBLE_DEVICES=1 "${vllm_python}" -u -m "${module}" \
  "${worker_args[@]}" --shard-id 1 >"${mode_root}/worker-1.log" 2>&1 &
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
  --output-root "${run_root}"
)
if [[ "${mode}" == "smoke" ]]; then
  validate_args+=(--smoke --smoke-meetings 2)
fi
"${vllm_python}" -u -m "${module}" "${validate_args[@]}"
