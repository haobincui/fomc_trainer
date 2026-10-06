#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${FOMC_TRAINER_PYTHON:-/home/haobin_cui/.conda/envs/fomc_trainer/bin/python}"
vllm_python="${VLLM_PYTHON_BIN:-/home/haobin_cui/.conda/envs/vllm_env/bin/python}"
output_root_raw="${PAPER_CHK2_TADLE_OUTPUT_ROOT:-${repo_root}/output/evaluation/retrain_v2/paper_chk2_cp50_tadle_form_lm_ff_futures_k10_v1_20260902}"
cpu_workers="${PAPER_CHK2_TADLE_CPU_WORKERS:-32}"
allow_busy_gpu="${PAPER_CHK2_ALLOW_BUSY_GPU:-0}"

[[ -x "${python_bin}" ]] || {
  echo "missing fomc_trainer Python: ${python_bin}" >&2
  exit 2
}
[[ -x "${vllm_python}" ]] || {
  echo "missing vLLM Python: ${vllm_python}" >&2
  exit 2
}
command -v setsid >/dev/null 2>&1 || {
  echo "missing required process-detachment utility: setsid" >&2
  exit 2
}
[[ "${cpu_workers}" =~ ^[1-9][0-9]*$ ]] || {
  echo "PAPER_CHK2_TADLE_CPU_WORKERS must be a positive integer" >&2
  exit 2
}
[[ "${allow_busy_gpu}" == "0" || "${allow_busy_gpu}" == "1" ]] || {
  echo "PAPER_CHK2_ALLOW_BUSY_GPU must be 0 or 1" >&2
  exit 2
}
logical_cpus="$(nproc)"
(( cpu_workers <= logical_cpus )) || {
  echo "requested ${cpu_workers} CPU workers but only ${logical_cpus} logical CPUs are available" >&2
  exit 2
}

mkdir -p "${output_root_raw}"
output_root="$(cd "${output_root_raw}" && pwd)"
log_path="${output_root}/background_all.log"
pid_path="${output_root}/background_all.pid"
launcher_lock="${output_root}/.background-launch.lock"

[[ ! -L "${pid_path}" && ! -L "${log_path}" && ! -L "${launcher_lock}" ]] || {
  echo "refusing symlink PID/log/lock path under ${output_root}" >&2
  exit 2
}

# Serialize the liveness check, spawn, and PID publication.  The child closes
# this descriptor before exec so a second launcher can inspect the published
# PID instead of waiting for the entire evaluation to finish.
exec 9>"${launcher_lock}"
flock -x 9

if [[ -f "${pid_path}" ]]; then
  existing_pid="$(tr -d '[:space:]' <"${pid_path}")"
  if [[ "${existing_pid}" =~ ^[0-9]+$ ]] && kill -0 "${existing_pid}" 2>/dev/null; then
    existing_cmdline="$(tr '\0' ' ' <"/proc/${existing_pid}/cmdline" 2>/dev/null || true)"
    if [[ "${existing_cmdline}" == *"jobs.eval.eval_paper_chk2_tadle_lm_vllm_k10"* \
      && "${existing_cmdline}" == *"${output_root}"* ]]; then
      echo "Already running paper chk-2 Tadle-form K=10 PID=${existing_pid}"
      echo "Log=${log_path}"
      exit 0
    fi
  fi
fi

cd "${repo_root}"

# A new session keeps the long-running evaluator alive when this launcher is
# itself invoked from an IDE/remote command runner that tears down descendants
# in its original process group after the shell exits.
nohup setsid env \
  FOMC_TRAINER_PYTHON="${python_bin}" \
  VLLM_PYTHON_BIN="${vllm_python}" \
  PYTHONPATH="${repo_root}/src:${repo_root}${PYTHONPATH:+:${PYTHONPATH}}" \
  PYTHONNOUSERSITE=1 \
  PYTHONUNBUFFERED=1 \
  TOKENIZERS_PARALLELISM=false \
  OMP_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  MKL_NUM_THREADS=1 \
  NUMEXPR_MAX_THREADS="${cpu_workers}" \
  PAPER_CHK2_ALLOW_BUSY_GPU="${allow_busy_gpu}" \
  "${python_bin}" -u -m jobs.eval.eval_paper_chk2_tadle_lm_vllm_k10 \
  all \
  --output-root "${output_root}" \
  --bootstrap-workers "${cpu_workers}" \
  --resume \
  9>&- >>"${log_path}" 2>&1 &
pid=$!

# Do not replace the discoverable PID with a short-lived lock loser or an
# immediate import/configuration failure.
sleep 2
if ! kill -0 "${pid}" 2>/dev/null; then
  echo "paper chk-2 Tadle-form K=10 failed during startup; inspect ${log_path}" >&2
  tail -n 20 "${log_path}" >&2 || true
  exit 1
fi
pid_tmp="${pid_path}.tmp.$$"
printf '%s\n' "${pid}" >"${pid_tmp}"
mv -T "${pid_tmp}" "${pid_path}"

echo "Started paper chk-2 Tadle-form K=10 PID=${pid}"
echo "Environment=fomc_trainer (generation workers use the pinned vllm_env runtime)"
echo "CPUWorkers=${cpu_workers}/${logical_cpus} (sentiment, diagnostics, and Bootstrap pools; BLAS pinned to 1 thread per worker)"
echo "AllowBusyGPU=${allow_busy_gpu} (vLLM remains capped at approximately 20 GiB per card)"
echo "Log=${log_path}"
