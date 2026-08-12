#!/usr/bin/env bash

set -euo pipefail

if (($# == 0)); then
  echo "Usage: $0 GPU_ID [GPU_ID ...]" >&2
  exit 2
fi

# Shared-GPU baselines measured immediately before the compressed-chk1 chk2
# launch on 2026-08-05.  GPU 0 had 6.67 GiB in a pre-existing workload; retain
# a narrow margin while refusing a launch if that workload grows materially.
declare -Ar BASELINE_MAX_USED_MIB=(
  [0]=7168
  [1]=7168
)
declare -Ar BASELINE_MIN_FREE_MIB=(
  [0]=17000
  [1]=17000
)
declare -A seen_gpu_ids=()

validate_integer() {
  local name="$1"
  local value="$2"
  if [[ ! "${value}" =~ ^[0-9]{1,6}$ ]]; then
    echo "ERROR: ${name} must be an integer." >&2
    exit 2
  fi
}

for gpu_id in "$@"; do
  if [[ ! "${gpu_id}" =~ ^[0-9]+$ ]]; then
    echo "ERROR: invalid GPU ID: ${gpu_id}" >&2
    exit 2
  fi
  if [[ -n "${seen_gpu_ids[${gpu_id}]:-}" ]]; then
    echo "ERROR: duplicate GPU ID: ${gpu_id}" >&2
    exit 2
  fi
  if [[ -z "${BASELINE_MAX_USED_MIB[${gpu_id}]:-}" ]]; then
    echo "ERROR: GPU ${gpu_id} has no approved shared-usage baseline." >&2
    exit 2
  fi
  seen_gpu_ids["${gpu_id}"]=1

  max_used_name="FOMC_RETRAIN_GPU${gpu_id}_MAX_PREEXISTING_USED_MIB"
  min_free_name="FOMC_RETRAIN_GPU${gpu_id}_MIN_FREE_MIB"
  max_used_mib="${!max_used_name:-${BASELINE_MAX_USED_MIB[${gpu_id}]}}"
  min_free_mib="${!min_free_name:-${BASELINE_MIN_FREE_MIB[${gpu_id}]}}"
  validate_integer "${max_used_name}" "${max_used_mib}"
  validate_integer "${min_free_name}" "${min_free_mib}"
  max_used_value=$((10#${max_used_mib}))
  min_free_value=$((10#${min_free_mib}))
  if ((max_used_value > BASELINE_MAX_USED_MIB[${gpu_id}])); then
    echo "ERROR: ${max_used_name} cannot weaken the ${BASELINE_MAX_USED_MIB[${gpu_id}]} MiB baseline." >&2
    exit 2
  fi
  if ((min_free_value < BASELINE_MIN_FREE_MIB[${gpu_id}])); then
    echo "ERROR: ${min_free_name} cannot weaken the ${BASELINE_MIN_FREE_MIB[${gpu_id}]} MiB baseline." >&2
    exit 2
  fi

  IFS=, read -r used_mib free_mib utilization < <(
    nvidia-smi --id="${gpu_id}" \
      --query-gpu=memory.used,memory.free,utilization.gpu \
      --format=csv,noheader,nounits
  )
  used_mib="${used_mib//[[:space:]]/}"
  free_mib="${free_mib//[[:space:]]/}"
  utilization="${utilization//[[:space:]]/}"
  if [[ ! "${used_mib}" =~ ^[0-9]{1,7}$ ]] \
    || [[ ! "${free_mib}" =~ ^[0-9]{1,7}$ ]] \
    || [[ ! "${utilization}" =~ ^[0-9]{1,3}$ ]]; then
    echo "ERROR: GPU ${gpu_id} returned invalid memory/utilization telemetry." >&2
    exit 2
  fi
  used_value=$((10#${used_mib}))
  free_value=$((10#${free_mib}))
  utilization_value=$((10#${utilization}))
  if ((utilization_value > 100)); then
    echo "ERROR: GPU ${gpu_id} returned invalid utilization: ${utilization_value}%." >&2
    exit 2
  fi
  if ((used_value > max_used_value)); then
    echo "ERROR: GPU ${gpu_id} already uses ${used_value} MiB; shared baseline allows at most ${max_used_value} MiB." >&2
    exit 2
  fi
  if ((free_value < min_free_value)); then
    echo "ERROR: GPU ${gpu_id} has ${free_value} MiB free; shared baseline requires ${min_free_value} MiB." >&2
    exit 2
  fi

  mapfile -t compute_pids < <(
    nvidia-smi --id="${gpu_id}" --query-compute-apps=pid \
      --format=csv,noheader,nounits 2>/dev/null | sed '/^[[:space:]]*$/d'
  )
  pid_summary="none"
  if ((${#compute_pids[@]})); then
    pid_summary="${compute_pids[*]}"
  fi
  echo "GPU ${gpu_id} shared gate passed: used=${used_value}MiB free=${free_value}MiB utilization=${utilization_value}% existing_pids=${pid_summary}"
done
