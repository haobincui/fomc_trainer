#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
MODE=${1:-}

if [[ -n "${MODE}" && "${MODE}" != "--worker" ]]; then
  echo "usage: $0" >&2
  echo "Resume with the same LOO_RUN_ID by setting LOO_RESUME=1." >&2
  exit 2
fi

EXPERIMENT_CONFIG=${LOO_EXPERIMENT_CONFIG:-"${REPO_ROOT}/configs/main/loo_experiment_legacy6.json"}
RUN_ID=${LOO_RUN_ID:-"legacy6-two-stage-$(date -u +%Y%m%dT%H%M%SZ)"}
WORKFLOW_BASE=${LOO_WORKFLOW_BASE:-"${REPO_ROOT}/output/evaluation/main/canonical_loo/legacy6_workflows"}
WORKFLOW_ROOT="${WORKFLOW_BASE}/${RUN_ID}"
LOG_FILE="${WORKFLOW_ROOT}/run.log"
PID_FILE="${WORKFLOW_ROOT}/run.pid"
STATUS_FILE="${WORKFLOW_ROOT}/workflow_status.jsonl"
GLOBAL_LOCK="${XDG_RUNTIME_DIR:-/tmp}/fomc-trainer-${UID}-physical-gpu1.lock"
PILOT_REUSE_ROOT="${REPO_ROOT}/output/evaluation/main/canonical_loo"
PILOT_REUSE_ANALYSIS_MANIFEST=${LOO_REUSE_PILOT_ANALYSIS_MANIFEST:-"${PILOT_REUSE_ROOT}/workflows/canonical-d1-20260729T123849Z/generations/pilot_eval_13/analysis/analysis_manifest.json"}
PILOT_REUSE_ANALYSIS_OUTPUT=${LOO_REUSE_PILOT_ANALYSIS_OUTPUT:-"${PILOT_REUSE_ROOT}/workflows/canonical-d1-20260729T123849Z/generations/pilot_eval_13/analysis/indicator_analysis.jsonl"}
PILOT_REUSE_PROJECTION_MANIFEST=${LOO_REUSE_PILOT_PROJECTION_MANIFEST:-"${PILOT_REUSE_ROOT}/pilot_eval_13/pilot-remaining20-recovered-v1-20260730T100713Z/analysis_projection/analysis_projection_manifest.json"}
PILOT_REUSE_PROJECTION_OUTPUT=${LOO_REUSE_PILOT_PROJECTION_OUTPUT:-"${PILOT_REUSE_ROOT}/pilot_eval_13/pilot-remaining20-recovered-v1-20260730T100713Z/analysis_projection/minutes_analysis.jsonl"}
LAUNCHER_STATUS_STARTED=0
LAUNCHER_SUCCEEDED=0

record_launcher_status() {
  local status=$1
  local exit_code=${2:-0}
  printf '{"timestamp_utc":"%s","stage":"launcher","status":"%s","exit_code":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${status}" \
    "${exit_code}" >>"${STATUS_FILE}"
}

cleanup_worker() {
  local exit_code=$?
  if [[ "${MODE}" == "--worker" && "${LAUNCHER_STATUS_STARTED}" == "1" \
      && "${LAUNCHER_SUCCEEDED}" != "1" ]]; then
    record_launcher_status failed "${exit_code}"
  fi
  if [[ "${MODE}" == "--worker" && -f "${PID_FILE}" ]]; then
    local recorded_pid
    recorded_pid=$(<"${PID_FILE}")
    if [[ "${recorded_pid}" == "$$" ]]; then
      rm -f "${PID_FILE}"
    fi
  fi
  return "${exit_code}"
}

if [[ "${MODE}" == "--worker" ]]; then
  mkdir -p "${WORKFLOW_ROOT}"
  record_launcher_status running
  LAUNCHER_STATUS_STARTED=1
  trap cleanup_worker EXIT
fi

if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "LOO_RUN_ID must contain only letters, digits, dot, underscore, or hyphen." >&2
  exit 2
fi
if [[ ! -f "${EXPERIMENT_CONFIG}" ]]; then
  echo "Scoped experiment config is missing: ${EXPERIMENT_CONFIG}" >&2
  exit 1
fi
if ! command -v flock >/dev/null 2>&1; then
  echo "flock is required." >&2
  exit 1
fi

source "${SCRIPT_DIR}/_canonical_gpu1.sh"

gpu1_is_idle() {
  local process_uuids
  if ! process_uuids=$(nvidia-smi \
      --query-compute-apps=gpu_uuid \
      --format=csv,noheader,nounits 2>/dev/null); then
    echo "Could not query physical GPU1 compute processes; refusing to launch." >&2
    return 1
  fi
  if grep -Fxq "${LOO_CANONICAL_PHYSICAL_GPU_UUID}" <<<"${process_uuids}"; then
    echo "Physical GPU1 already has a compute process." >&2
    return 1
  fi
}

if [[ "${MODE}" != "--worker" ]]; then
  gpu1_is_idle
  mkdir -p "${WORKFLOW_BASE}"
  if [[ "${LOO_RESUME:-0}" == "1" ]]; then
    if [[ ! -d "${WORKFLOW_ROOT}" ]]; then
      echo "LOO_RESUME=1 requires an existing workflow: ${WORKFLOW_ROOT}" >&2
      exit 1
    fi
  elif ! mkdir "${WORKFLOW_ROOT}"; then
    echo "Workflow exists; set LOO_RESUME=1 or choose a new LOO_RUN_ID." >&2
    exit 1
  fi
  if [[ -f "${PID_FILE}" ]]; then
    existing_pid=$(<"${PID_FILE}")
    if [[ "${existing_pid}" =~ ^[1-9][0-9]*$ ]] \
        && kill -0 "${existing_pid}" 2>/dev/null; then
      echo "Workflow already has a live process: ${existing_pid}" >&2
      exit 1
    fi
  fi
  nohup env \
    LOO_RUN_ID="${RUN_ID}" \
    LOO_WORKFLOW_BASE="${WORKFLOW_BASE}" \
    LOO_EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG}" \
    LOO_RESUME="${LOO_RESUME:-0}" \
    LOO_REUSE_PILOT_ANALYSIS_MANIFEST="${PILOT_REUSE_ANALYSIS_MANIFEST}" \
    LOO_REUSE_PILOT_ANALYSIS_OUTPUT="${PILOT_REUSE_ANALYSIS_OUTPUT}" \
    LOO_REUSE_PILOT_PROJECTION_MANIFEST="${PILOT_REUSE_PROJECTION_MANIFEST}" \
    LOO_REUSE_PILOT_PROJECTION_OUTPUT="${PILOT_REUSE_PROJECTION_OUTPUT}" \
    "${SCRIPT_PATH}" --worker >>"${LOG_FILE}" 2>&1 </dev/null &
  worker_pid=$!
  printf '%s\n' "${worker_pid}" >"${PID_FILE}"
  sleep 1
  if ! kill -0 "${worker_pid}" 2>/dev/null; then
    echo "Legacy-six workflow exited during startup." >&2
    tail -n 60 "${LOG_FILE}" >&2 || true
    exit 1
  fi
  echo "Started the legacy-six two-stage LOO workflow on physical GPU1."
  echo "run_id=${RUN_ID}"
  echo "pid=${worker_pid}"
  echo "log=${LOG_FILE}"
  echo "output=${WORKFLOW_ROOT}"
  exit 0
fi

mkdir -p "${WORKFLOW_BASE}"
exec 8>"${GLOBAL_LOCK}"
if ! flock -n 8; then
  echo "Another canonical workflow holds the physical-GPU1 lock." >&2
  exit 1
fi
gpu1_is_idle

cd "${REPO_ROOT}"
echo "started_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "run_id=${RUN_ID}"
echo "experiment_config=${EXPERIMENT_CONFIG}"
echo "physical_gpu_index=1"
echo "physical_gpu_uuid=${LOO_CANONICAL_PHYSICAL_GPU_UUID}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "training_performed=false"

LOO_FOREGROUND=1 \
LOO_SUPERVISOR_PID="$$" \
LOO_RESUME=1 \
LOO_RUN_ID="${RUN_ID}" \
LOO_WORKFLOW_BASE="${WORKFLOW_BASE}" \
LOO_EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG}" \
LOO_REUSE_PILOT_ANALYSIS_MANIFEST="${PILOT_REUSE_ANALYSIS_MANIFEST}" \
LOO_REUSE_PILOT_ANALYSIS_OUTPUT="${PILOT_REUSE_ANALYSIS_OUTPUT}" \
LOO_REUSE_PILOT_PROJECTION_MANIFEST="${PILOT_REUSE_PROJECTION_MANIFEST}" \
LOO_REUSE_PILOT_PROJECTION_OUTPUT="${PILOT_REUSE_PROJECTION_OUTPUT}" \
LOO_MINUTES_TOKEN_LIMIT_POLICY=error \
  "${SCRIPT_DIR}/generate_loo_end_to_end.sh" all

echo "completed_at_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "status=complete"
echo "result=${WORKFLOW_ROOT}/scoped_workflow_release_manifest.json"
record_launcher_status complete
LAUNCHER_SUCCEEDED=1
