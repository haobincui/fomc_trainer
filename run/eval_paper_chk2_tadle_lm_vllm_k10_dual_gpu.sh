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
module="jobs.eval.eval_paper_chk2_tadle_lm_vllm_k10_dual_dp1"
common_module="jobs.eval.paper_chk2_tadle_lm_k10_common"
worker_targets=(chk0 chk1_paper_chk2_cp50)
allow_busy_gpu="${PAPER_CHK2_ALLOW_BUSY_GPU:-0}"

[[ "${allow_busy_gpu}" == "0" || "${allow_busy_gpu}" == "1" ]] || {
  echo "PAPER_CHK2_ALLOW_BUSY_GPU must be 0 or 1" >&2
  exit 2
}

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

if [[ "${mode}" != "validate" ]]; then
  available_kib="$(df -Pk "${run_root}" | awk 'NR==2 {print $4}')"
  [[ "${available_kib}" =~ ^[0-9]+$ ]] || {
    echo "could not determine free disk space for ${run_root}" >&2
    exit 1
  }
  minimum_kib=$((20 * 1024 * 1024))
  (( available_kib >= minimum_kib )) || {
    echo "generation requires at least 20 GiB free under ${run_root}" >&2
    exit 1
  }
fi

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

validate_target() {
  local worker_target="$1"
  local smoke_flag=()
  if [[ "${mode}" == "smoke" ]]; then
    smoke_flag+=(--smoke)
  fi
  local validation_pids=()
  for shard_id in 0 1; do
    "${vllm_python}" -u -m "${module}" validate \
      --manifest "${evaluation_manifest}" \
      --output-root "${run_root}" \
      --model-id "${worker_target}" \
      --shard-id "${shard_id}" \
      "${smoke_flag[@]}" \
      >"${mode_root}/${worker_target}-validate-${shard_id}.log" 2>&1 &
    validation_pids+=("$!")
  done
  local validation_status=0
  for validation_pid in "${validation_pids[@]}"; do
    wait "${validation_pid}" || validation_status=$?
  done
  (( validation_status == 0 )) || {
    echo "parallel shard validation failed for ${worker_target}" >&2
    exit 1
  }
}

if [[ "${mode}" == "validate" ]]; then
  for worker_target in "${worker_targets[@]}"; do
    validate_target "${worker_target}"
  done
  "${vllm_python}" -u -m "${common_module}" validate-all \
    --output-root "${run_root}" --resume
  exit 0
fi

wait_for_gpu_memory() {
  local worker_target="$1"
  local stability_log="${mode_root}/${worker_target}-gpu-memory-stability.log"
  local stability_attempt=0
  while true; do
    stability_attempt=$((stability_attempt + 1))
    local stability_pass=1
    for sample_index in 0 1 2 3 4 5 6; do
      local gpu_rows_raw
      if ! gpu_rows_raw="$(nvidia-smi --id=0,1 \
        --query-gpu=index,memory.free,memory.total,utilization.gpu \
        --format=csv,noheader,nounits)"; then
        echo "nvidia-smi failed during GPU stability gate" >&2
        exit 1
      fi
      mapfile -t gpu_rows <<<"${gpu_rows_raw}"
      [[ "${#gpu_rows[@]}" -eq 2 ]] || {
        echo "GPU stability gate expected exactly two GPU rows" >&2
        exit 1
      }
      local seen_gpu0=0
      local seen_gpu1=0
      for gpu_row in "${gpu_rows[@]}"; do
        IFS=',' read -r observed_gpu free_mib total_mib compute_percent <<<"${gpu_row}"
        observed_gpu="${observed_gpu//[[:space:]]/}"
        free_mib="${free_mib//[[:space:]]/}"
        total_mib="${total_mib//[[:space:]]/}"
        compute_percent="${compute_percent//[[:space:]]/}"
        [[ "${observed_gpu}" == "0" || "${observed_gpu}" == "1" ]] || {
          echo "unexpected GPU index: ${observed_gpu}" >&2
          exit 1
        }
        if [[ "${observed_gpu}" == "0" ]]; then seen_gpu0=1; else seen_gpu1=1; fi
        [[ "${free_mib}" =~ ^[0-9]+$ && "${total_mib}" =~ ^[0-9]+$ && "${compute_percent}" =~ ^[0-9]+$ ]] || {
          echo "non-numeric GPU memory observation" >&2
          exit 1
        }
        required_free_mib=22528
        if [[ "${allow_busy_gpu}" == "1" ]]; then
          required_free_mib=20480
        fi
        if (( free_mib < required_free_mib || (allow_busy_gpu == 0 && compute_percent > 10) )); then
          stability_pass=0
        fi
        printf 'attempt=%s sample=%s target=%s gpu=%s free_mib=%s total_mib=%s fixed_vllm_budget_mib=20480 required_free_mib=%s compute_percent=%s allow_busy_gpu=%s pass=%s\n' \
          "${stability_attempt}" "${sample_index}" "${worker_target}" "${observed_gpu}" \
          "${free_mib}" "${total_mib}" "${required_free_mib}" "${compute_percent}" \
          "${allow_busy_gpu}" \
          "$(( free_mib >= required_free_mib && (allow_busy_gpu == 1 || compute_percent <= 10) ))" >>"${stability_log}"
      done
      (( seen_gpu0 == 1 && seen_gpu1 == 1 )) || {
        echo "GPU stability gate did not observe both physical GPUs" >&2
        exit 1
      }
      if (( stability_pass == 0 )); then
        break
      fi
      if [[ "${allow_busy_gpu}" == "1" ]]; then
        break
      fi
      if [[ "${sample_index}" -lt 6 ]]; then sleep 5; fi
    done
    if (( stability_pass == 1 )); then
      return 0
    fi
    printf 'attempt=%s target=%s status=waiting_for_required_memory allow_busy_gpu=%s\n' \
      "${stability_attempt}" "${worker_target}" "${allow_busy_gpu}" >>"${stability_log}"
    sleep 30
  done
}

for worker_target in "${worker_targets[@]}"; do
  wait_for_gpu_memory "${worker_target}"
  worker_args=(
    worker
    --manifest "${evaluation_manifest}"
    --output-root "${run_root}"
    --model-id "${worker_target}"
    --resume
  )
  if [[ "${mode}" == "smoke" ]]; then
    worker_args+=(--smoke)
  fi

  CUDA_VISIBLE_DEVICES=0 "${vllm_python}" -u -m "${module}" \
    "${worker_args[@]}" --shard-id 0 >"${mode_root}/${worker_target}-worker-0.log" 2>&1 &
  pid0=$!
  CUDA_VISIBLE_DEVICES=1 "${vllm_python}" -u -m "${module}" \
    "${worker_args[@]}" --shard-id 1 >"${mode_root}/${worker_target}-worker-1.log" 2>&1 &
  pid1=$!

  status0=0
  status1=0
  wait "${pid0}" || status0=$?
  wait "${pid1}" || status1=$?
  if [[ "${status0}" -ne 0 || "${status1}" -ne 0 ]]; then
    echo "worker failure for ${worker_target}: shard0=${status0} shard1=${status1}" >&2
    exit 1
  fi
  validate_target "${worker_target}"
done

# The per-shard receipts above are necessary but not sufficient.  Seal the
# paired three-model Cartesian product so that formal workers can prove the
# 96-row smoke succeeded, and so formal generation has one canonical
# 20,160-row closure.  This must use the pinned vLLM runtime because deep token
# replay is deliberately bound to its transformers/tokenizers versions.
if [[ "${mode}" == "smoke" ]]; then
  "${vllm_python}" -u -m "${common_module}" validate-smoke \
    --output-root "${run_root}" --resume
else
  "${vllm_python}" -u -m "${common_module}" validate-all \
    --output-root "${run_root}" --resume
fi
