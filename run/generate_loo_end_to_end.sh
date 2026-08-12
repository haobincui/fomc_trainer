#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-all}
INTERNAL_MODE=${2:-}

case "${MODE}" in
  all|pilot|formal)
    ;;
  *)
    echo "usage: $0 [all|pilot|formal]" >&2
    exit 2
    ;;
esac

if [[ -n "${LOO_INTERVENTION_INDICATORS:-}" && "${MODE}" != "pilot" ]]; then
  echo "Partial intervention shards currently require mode=pilot." >&2
  echo "Run a complete pilot release before starting formal_test_13." >&2
  exit 2
fi

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
source "${SCRIPT_DIR}/_canonical_gpu1.sh"
if [[ -n "${LOO_PYTHON:-}" ]]; then
  PYTHON_BIN=${LOO_PYTHON}
elif [[ -n "${HOME:-}" && -x "${HOME}/.conda/envs/llama_factory/bin/python" ]]; then
  PYTHON_BIN="${HOME}/.conda/envs/llama_factory/bin/python"
else
  PYTHON_BIN=python
fi
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
SOURCE_REGISTRY=${LOO_SOURCE_REGISTRY:-"${REPO_ROOT}/configs/main/loo_indicator_sources.json"}
INDICATOR_ROSTER=${LOO_INDICATOR_ROSTER:-"${REPO_ROOT}/configs/main/leave_one_out_roster.json"}
SECTION_ROSTER=${LOO_SECTION_ROSTER:-"${REPO_ROOT}/configs/main/loo_sections.json"}
GENERATION_CONFIG=${LOO_GENERATION_CONFIG:-"${REPO_ROOT}/configs/main/canonical_loo_generation.json"}
EXPERIMENT_CONFIG=${LOO_EXPERIMENT_CONFIG:-}
PILOT_POPULATION=${LOO_PILOT_POPULATION:-"${REPO_ROOT}/configs/main/loo_population_pilot_eval_13.json"}
FORMAL_POPULATION=${LOO_FORMAL_POPULATION:-"${REPO_ROOT}/configs/main/loo_population_formal_test_13.json"}
ANALYSIS_MODEL=${LOO_ANALYSIS_MODEL:-"${REPO_ROOT}/../fomc_trainer_back/fomc_trainer/output/merged/llama_grpo_20250515"}
ANALYSIS_TOKENIZER=${LOO_ANALYSIS_TOKENIZER:-"${REPO_ROOT}/models/DeepSeek-R1-Distill-Llama-8B"}
MINUTES_MODEL=${LOO_MINUTES_MODEL:-"${REPO_ROOT}/output/checkpoints/recovered/llama_sft_synthetic_20250526_2_cp1668_recovered_v1_20260729/model"}
MINUTES_TOKENIZER=${LOO_MINUTES_TOKENIZER:-"${MINUTES_MODEL}"}
FETCH_WORKERS=${LOO_FETCH_WORKERS:-1}
FETCH_RATE=${LOO_FETCH_RATE:-1}
DEFAULT_PILOT_REUSE_ROOT="${REPO_ROOT}/output/evaluation/main/canonical_loo"
PILOT_REUSE_ANALYSIS_MANIFEST=${LOO_REUSE_PILOT_ANALYSIS_MANIFEST:-"${DEFAULT_PILOT_REUSE_ROOT}/workflows/canonical-d1-20260729T123849Z/generations/pilot_eval_13/analysis/analysis_manifest.json"}
PILOT_REUSE_ANALYSIS_OUTPUT=${LOO_REUSE_PILOT_ANALYSIS_OUTPUT:-"${DEFAULT_PILOT_REUSE_ROOT}/workflows/canonical-d1-20260729T123849Z/generations/pilot_eval_13/analysis/indicator_analysis.jsonl"}
PILOT_REUSE_PROJECTION_MANIFEST=${LOO_REUSE_PILOT_PROJECTION_MANIFEST:-"${DEFAULT_PILOT_REUSE_ROOT}/pilot_eval_13/pilot-remaining20-recovered-v1-20260730T100713Z/analysis_projection/analysis_projection_manifest.json"}
PILOT_REUSE_PROJECTION_OUTPUT=${LOO_REUSE_PILOT_PROJECTION_OUTPUT:-"${DEFAULT_PILOT_REUSE_ROOT}/pilot_eval_13/pilot-remaining20-recovered-v1-20260730T100713Z/analysis_projection/minutes_analysis.jsonl"}
PILOT_REUSE_PROJECTION_OUTPUT_SHA256=${LOO_REUSE_PILOT_PROJECTION_OUTPUT_SHA256:-"9eef6db583ce4813234ea7953c2bd48768c7093cc99ee0c529a300326fef3cfb"}

require_file() {
  if [[ -z "$1" || ! -f "$1" ]]; then
    echo "Required file does not exist: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ -z "$1" || ! -d "$1" ]]; then
    echo "Required directory does not exist: $1" >&2
    exit 1
  fi
}

require_file "${SOURCE_REGISTRY}"
require_file "${INDICATOR_ROSTER}"
require_file "${SECTION_ROSTER}"
require_file "${GENERATION_CONFIG}"
require_file "${PILOT_POPULATION}"
require_file "${FORMAL_POPULATION}"
if [[ -n "${EXPERIMENT_CONFIG}" ]]; then
  require_file "${EXPERIMENT_CONFIG}"
  if [[ -n "${LOO_INTERVENTION_INDICATORS:-}" ]]; then
    echo "LOO_EXPERIMENT_CONFIG cannot be combined with LOO_INTERVENTION_INDICATORS." >&2
    exit 2
  fi
  "${PYTHON_BIN}" -m open_r1.validator.loo_experiment_scope \
    --experiment-config "${EXPERIMENT_CONFIG}" \
    --roster "${INDICATOR_ROSTER}" \
    --pilot-population "${PILOT_POPULATION}" \
    --formal-population "${FORMAL_POPULATION}" >/dev/null
  if [[ "${MODE}" == "pilot" || "${MODE}" == "all" ]]; then
    require_file "${PILOT_REUSE_ANALYSIS_MANIFEST}"
    require_file "${PILOT_REUSE_ANALYSIS_OUTPUT}"
    require_file "${PILOT_REUSE_PROJECTION_MANIFEST}"
    require_file "${PILOT_REUSE_PROJECTION_OUTPUT}"
    observed_projection_sha256=$(sha256sum \
      "${PILOT_REUSE_PROJECTION_OUTPUT}" | awk '{print $1}')
    if [[ "${observed_projection_sha256}" != "${PILOT_REUSE_PROJECTION_OUTPUT_SHA256}" ]]; then
      echo "Frozen pilot projection output SHA-256 mismatch." >&2
      exit 1
    fi
  fi
fi
require_dir "${ANALYSIS_MODEL}"
require_dir "${ANALYSIS_TOKENIZER}"
require_dir "${MINUTES_MODEL}"
require_dir "${MINUTES_TOKENIZER}"

case "${FETCH_WORKERS}" in
  1|2)
    ;;
  *)
    echo "LOO_FETCH_WORKERS must be 1 or 2." >&2
    exit 2
    ;;
esac

if ! command -v flock >/dev/null 2>&1; then
  echo "Required command is unavailable: flock" >&2
  exit 1
fi

"${PYTHON_BIN}" - "${FETCH_RATE}" <<'PY'
import sys
from decimal import Decimal, InvalidOperation

try:
    rate = Decimal(sys.argv[1])
except InvalidOperation as error:
    raise SystemExit("LOO_FETCH_RATE must be a number greater than 0 and at most 2") from error
if not rate.is_finite() or not Decimal("0") < rate <= Decimal("2"):
    raise SystemExit("LOO_FETCH_RATE must be a number greater than 0 and at most 2")
PY

if [[ "${MODE}" == "formal" ]]; then
  if [[ -n "${EXPERIMENT_CONFIG}" ]]; then
    require_file "${LOO_PILOT_SCOPED_RELEASE_MANIFEST:-}"
    require_file "${LOO_PILOT_RESULTS_MANIFEST:-}"
  else
    require_file "${LOO_PILOT_RELEASE_MANIFEST:-}"
  fi
fi

RUN_ID=${LOO_RUN_ID:-"canonical-d1-$(date -u +%Y%m%dT%H%M%SZ)"}
if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "LOO_RUN_ID must contain only letters, digits, dot, underscore, or hyphen." >&2
  exit 2
fi
WORKFLOW_BASE=${LOO_WORKFLOW_BASE:-"${REPO_ROOT}/output/evaluation/main/canonical_loo/workflows"}
WORKFLOW_ROOT="${WORKFLOW_BASE}/${RUN_ID}"
SNAPSHOT_DIR="${WORKFLOW_ROOT}/source_snapshots"
SNAPSHOT_MANIFEST="${SNAPSHOT_DIR}/snapshot_manifest.json"
SCOPED_WORKFLOW_MANIFEST="${WORKFLOW_ROOT}/scoped_workflow_release_manifest.json"

if [[ "${INTERNAL_MODE}" != "--worker" ]]; then
  "${PYTHON_BIN}" - <<'PY'
import os
import sys

try:
    import torch
    import transformers
    import vllm
    from generate_new_response import generate_new_response  # noqa: F401
    from vllm.platforms.cuda import device_id_to_physical_device_id
except Exception as error:
    raise SystemExit(
        "Generation preflight failed. Set LOO_PYTHON to a Python environment "
        f"with working torch, transformers, and vLLM: {type(error).__name__}: "
        f"{error}"
    ) from error
expected_index = os.environ.get("LOO_CANONICAL_PHYSICAL_GPU_INDEX")
expected_uuid = os.environ.get("LOO_CANONICAL_PHYSICAL_GPU_UUID")
visible_device = os.environ.get("CUDA_VISIBLE_DEVICES")
logical_uuid = (
    getattr(torch.cuda.get_device_properties(0), "uuid", None)
    if torch.cuda.is_available() and torch.cuda.device_count() == 1
    else None
)
logical_uuid = str(logical_uuid) if logical_uuid is not None else ""
if logical_uuid and not logical_uuid.startswith("GPU-"):
    logical_uuid = f"GPU-{logical_uuid}"
if (
    not torch.cuda.is_available()
    or torch.cuda.device_count() != 1
    or expected_index != "1"
    or not expected_uuid
    or visible_device != expected_index
    or logical_uuid.lower() != expected_uuid.lower()
    or device_id_to_physical_device_id(0) != int(expected_index)
):
    raise SystemExit(
        "Generation preflight requires physical GPU 1 to be the only visible "
        "CUDA device and its runtime UUID to match nvidia-smi"
    )
print(
    f"generation_python={sys.executable} torch={torch.__version__} "
    f"transformers={transformers.__version__} vllm={vllm.__version__} "
    f"visible_gpus={torch.cuda.device_count()} "
    f"physical_gpu_index={expected_index} physical_gpu_uuid={expected_uuid} "
    f"cuda_visible_devices={visible_device} logical_gpu_uuid={logical_uuid}"
)
PY
  if [[ "${LOO_OFFLINE:-0}" == "1" ]]; then
    require_file "${SNAPSHOT_MANIFEST}"
  elif [[ "${LOO_SKIP_NETWORK_PREFLIGHT:-0}" != "1" ]]; then
    "${PYTHON_BIN}" - <<'PY'
from urllib.request import Request, urlopen

url = (
    "https://alfred.stlouisfed.org/graph/alfredgraph.csv"
    "?id=GDP&cosd=2023-01-01&coed=2023-07-25"
    "&vintage_date=2023-07-25"
)
request = Request(url, headers={"User-Agent": "fomc-trainer-canonical-loo/1"})
with urlopen(request, timeout=60) as response:
    if response.status != 200:
        raise SystemExit(f"ALFRED preflight returned HTTP {response.status}")
    body = response.read()
text = body.decode("utf-8-sig")
header = text.splitlines()[0] if text else ""
if header != "observation_date,GDP_20230725":
    raise SystemExit(f"Unsafe ALFRED preflight response header: {header!r}")
PY
  fi
fi

if [[ "${INTERNAL_MODE}" != "--worker" ]]; then
  mkdir -p "${WORKFLOW_BASE}"
  if [[ "${LOO_RESUME:-0}" == "1" ]]; then
    if [[ ! -d "${WORKFLOW_ROOT}" ]]; then
      echo "LOO_RESUME=1 requires an existing workflow directory: ${WORKFLOW_ROOT}" >&2
      exit 1
    fi
  elif ! mkdir "${WORKFLOW_ROOT}"; then
    echo "Workflow already exists; set LOO_RESUME=1 or choose a new LOO_RUN_ID: ${WORKFLOW_ROOT}" >&2
    exit 1
  fi
  if [[ -f "${WORKFLOW_ROOT}/run.pid" ]]; then
    EXISTING_PID=$(<"${WORKFLOW_ROOT}/run.pid")
    if [[ "${EXISTING_PID}" =~ ^[1-9][0-9]*$ ]] && kill -0 "${EXISTING_PID}" 2>/dev/null; then
      if [[ "${LOO_FOREGROUND:-0}" != "1" \
          || "${LOO_SUPERVISOR_PID:-}" != "${EXISTING_PID}" \
          || "${PPID}" != "${EXISTING_PID}" ]]; then
        echo "Workflow already has a live worker (PID ${EXISTING_PID}): ${WORKFLOW_ROOT}" >&2
        exit 1
      fi
      echo "Validated foreground supervisor PID ${EXISTING_PID}."
    fi
  fi
  if [[ "${LOO_FOREGROUND:-0}" == "1" ]]; then
    exec "${SCRIPT_PATH}" "${MODE}" --worker "${RUN_ID}" "${WORKFLOW_ROOT}"
  fi
  nohup "${SCRIPT_PATH}" "${MODE}" --worker "${RUN_ID}" "${WORKFLOW_ROOT}" \
    >>"${WORKFLOW_ROOT}/run.log" 2>&1 </dev/null &
  WORKER_PID=$!
  printf '%s\n' "${WORKER_PID}" >"${WORKFLOW_ROOT}/run.pid"
  echo "Started D-1 canonical LOO workflow."
  echo "run_id=${RUN_ID}"
  echo "mode=${MODE}"
  echo "pid=${WORKER_PID}"
  echo "log=${WORKFLOW_ROOT}/run.log"
  echo "output=${WORKFLOW_ROOT}"
  exit 0
fi

if [[ $# -ne 4 ]]; then
  echo "Internal worker invocation is incomplete." >&2
  exit 2
fi

RUN_ID=$3
WORKFLOW_ROOT=$4
SNAPSHOT_DIR="${WORKFLOW_ROOT}/source_snapshots"
SNAPSHOT_MANIFEST="${SNAPSHOT_DIR}/snapshot_manifest.json"
SCOPED_WORKFLOW_MANIFEST="${WORKFLOW_ROOT}/scoped_workflow_release_manifest.json"
FROZEN_INPUT_DIR="${WORKFLOW_ROOT}/inputs"
FROZEN_REGISTRY="${FROZEN_INPUT_DIR}/loo_indicator_sources.json"
FROZEN_INDICATOR_ROSTER="${FROZEN_INPUT_DIR}/leave_one_out_roster.json"
FROZEN_SECTION_ROSTER="${FROZEN_INPUT_DIR}/loo_sections.json"
FROZEN_GENERATION_CONFIG="${FROZEN_INPUT_DIR}/canonical_loo_generation.json"
FROZEN_PILOT_POPULATION="${FROZEN_INPUT_DIR}/loo_population_pilot_eval_13.json"
FROZEN_FORMAL_POPULATION="${FROZEN_INPUT_DIR}/loo_population_formal_test_13.json"
FROZEN_PILOT_RELEASE="${FROZEN_INPUT_DIR}/pilot_release_manifest.json"
FROZEN_PILOT_RESULTS="${FROZEN_INPUT_DIR}/pilot_results_manifest.json"
FROZEN_EXPERIMENT_CONFIG="${FROZEN_INPUT_DIR}/loo_experiment_legacy6.json"
FROZEN_REUSE_DIR="${FROZEN_INPUT_DIR}/reused_pilot_analysis"
FROZEN_REUSE_ANALYSIS_MANIFEST="${FROZEN_REUSE_DIR}/analysis_manifest.json"
FROZEN_REUSE_ANALYSIS_OUTPUT="${FROZEN_REUSE_DIR}/indicator_analysis.jsonl"
FROZEN_REUSE_PROJECTION_MANIFEST="${FROZEN_REUSE_DIR}/analysis_projection_manifest.json"
FROZEN_REUSE_PROJECTION_OUTPUT="${FROZEN_REUSE_DIR}/minutes_analysis.jsonl"
STATUS_FILE="${WORKFLOW_ROOT}/workflow_status.jsonl"
CURRENT_STAGE=initializing

exec 9>"${WORKFLOW_ROOT}/.worker.lock"
if ! flock -n 9; then
  echo "Another worker already holds the workflow lock: ${WORKFLOW_ROOT}" >&2
  exit 1
fi

record_status() {
  local status=$1
  local exit_code=${2:-0}
  printf '{"timestamp_utc":"%s","stage":"%s","status":"%s","exit_code":%s}\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "${CURRENT_STAGE}" \
    "${status}" \
    "${exit_code}" >>"${STATUS_FILE}"
}

read_manifest_status() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(manifest["status"])
PY
}

on_exit() {
  local exit_code=$?
  if [[ ${exit_code} -ne 0 ]]; then
    record_status failed "${exit_code}"
  fi
  if [[ -f "${WORKFLOW_ROOT}/run.pid" ]]; then
    local recorded_pid
    recorded_pid=$(<"${WORKFLOW_ROOT}/run.pid")
    if [[ "${recorded_pid}" == "$$" ]]; then
      rm -f "${WORKFLOW_ROOT}/run.pid"
    fi
  fi
  return "${exit_code}"
}
trap on_exit EXIT

cd "${REPO_ROOT}"
mkdir -p "${FROZEN_INPUT_DIR}"

freeze_input() {
  local source_path=$1
  local frozen_path=$2
  local label=$3
  if [[ -f "${frozen_path}" ]]; then
    if ! cmp -s "${source_path}" "${frozen_path}"; then
      echo "Frozen workflow ${label} differs from the configured input." >&2
      exit 1
    fi
  elif [[ -e "${frozen_path}" ]]; then
    echo "Frozen workflow ${label} path is not a regular file: ${frozen_path}" >&2
    exit 1
  else
    install -m 0444 "${source_path}" "${frozen_path}"
  fi
}

freeze_input "${SOURCE_REGISTRY}" "${FROZEN_REGISTRY}" "source registry"
freeze_input "${INDICATOR_ROSTER}" "${FROZEN_INDICATOR_ROSTER}" "indicator roster"
freeze_input "${SECTION_ROSTER}" "${FROZEN_SECTION_ROSTER}" "section roster"
freeze_input "${GENERATION_CONFIG}" "${FROZEN_GENERATION_CONFIG}" "generation config"
freeze_input "${PILOT_POPULATION}" "${FROZEN_PILOT_POPULATION}" "pilot population"
freeze_input "${FORMAL_POPULATION}" "${FROZEN_FORMAL_POPULATION}" "formal population"
if [[ -n "${EXPERIMENT_CONFIG}" ]]; then
  freeze_input \
    "${EXPERIMENT_CONFIG}" \
    "${FROZEN_EXPERIMENT_CONFIG}" \
    "scoped experiment config"
  EXPERIMENT_CONFIG=${FROZEN_EXPERIMENT_CONFIG}
  if [[ "${MODE}" == "pilot" || "${MODE}" == "all" ]]; then
    mkdir -p "${FROZEN_REUSE_DIR}"
    freeze_input \
      "${PILOT_REUSE_ANALYSIS_MANIFEST}" \
      "${FROZEN_REUSE_ANALYSIS_MANIFEST}" \
      "reused pilot analysis manifest"
    freeze_input \
      "${PILOT_REUSE_ANALYSIS_OUTPUT}" \
      "${FROZEN_REUSE_ANALYSIS_OUTPUT}" \
      "reused pilot analysis output"
    freeze_input \
      "${PILOT_REUSE_PROJECTION_MANIFEST}" \
      "${FROZEN_REUSE_PROJECTION_MANIFEST}" \
      "reused pilot projection manifest"
    freeze_input \
      "${PILOT_REUSE_PROJECTION_OUTPUT}" \
      "${FROZEN_REUSE_PROJECTION_OUTPUT}" \
      "reused pilot projection output"
    PILOT_REUSE_ANALYSIS_MANIFEST=${FROZEN_REUSE_ANALYSIS_MANIFEST}
    PILOT_REUSE_ANALYSIS_OUTPUT=${FROZEN_REUSE_ANALYSIS_OUTPUT}
    PILOT_REUSE_PROJECTION_MANIFEST=${FROZEN_REUSE_PROJECTION_MANIFEST}
    PILOT_REUSE_PROJECTION_OUTPUT=${FROZEN_REUSE_PROJECTION_OUTPUT}
  fi
fi
if [[ "${MODE}" == "formal" ]]; then
  PILOT_PREREQUISITE=${LOO_PILOT_RELEASE_MANIFEST:-}
  if [[ -n "${EXPERIMENT_CONFIG}" ]]; then
    PILOT_PREREQUISITE=${LOO_PILOT_SCOPED_RELEASE_MANIFEST}
  fi
  freeze_input \
    "${PILOT_PREREQUISITE}" \
    "${FROZEN_PILOT_RELEASE}" \
    "pilot release prerequisite"
  if [[ -n "${EXPERIMENT_CONFIG}" ]]; then
    LOO_PILOT_SCOPED_RELEASE_MANIFEST=${FROZEN_PILOT_RELEASE}
    freeze_input \
      "${LOO_PILOT_RESULTS_MANIFEST}" \
      "${FROZEN_PILOT_RESULTS}" \
      "pilot results prerequisite"
    LOO_PILOT_RESULTS_MANIFEST=${FROZEN_PILOT_RESULTS}
  else
    LOO_PILOT_RELEASE_MANIFEST=${FROZEN_PILOT_RELEASE}
  fi
fi

SOURCE_REGISTRY=${FROZEN_REGISTRY}
INDICATOR_ROSTER=${FROZEN_INDICATOR_ROSTER}
SECTION_ROSTER=${FROZEN_SECTION_ROSTER}
GENERATION_CONFIG=${FROZEN_GENERATION_CONFIG}
PILOT_POPULATION=${FROZEN_PILOT_POPULATION}
FORMAL_POPULATION=${FROZEN_FORMAL_POPULATION}

CURRENT_STAGE=fetch_sources
record_status running
FETCH_ARGS=(
  --registry "${FROZEN_REGISTRY}"
  --output-dir "${SNAPSHOT_DIR}"
  --max-workers "${FETCH_WORKERS}"
  --requests-per-second "${FETCH_RATE}"
)
if [[ "${MODE}" == "all" ]]; then
  FETCH_ARGS+=(
    --population "${PILOT_POPULATION}"
    --population "${FORMAL_POPULATION}"
    --expected-vintage-count 26
  )
else
  if [[ "${MODE}" == "pilot" ]]; then
    FETCH_ARGS+=(--population "${PILOT_POPULATION}")
  else
    FETCH_ARGS+=(--population "${FORMAL_POPULATION}")
  fi
  FETCH_ARGS+=(--expected-vintage-count 13)
fi
if [[ -e "${SNAPSHOT_MANIFEST}" || "${LOO_RESUME:-0}" == "1" ]]; then
  FETCH_ARGS+=(--resume)
fi
if [[ "${LOO_OFFLINE:-0}" == "1" && ! -f "${SNAPSHOT_MANIFEST}" ]]; then
  echo "LOO_OFFLINE=1 requires a complete snapshot manifest." >&2
  exit 1
fi
"${PYTHON_BIN}" -m jobs.main.fetch_loo_source_snapshots "${FETCH_ARGS[@]}"
record_status complete

run_population() {
  local population_id=$1
  local population_file=$2
  local phase=$3
  local ledger_dir="${WORKFLOW_ROOT}/ledgers/${population_id}"
  local ledger_manifest="${ledger_dir}/ledger_manifest.json"
  local generation_root="${WORKFLOW_ROOT}/generations/${population_id}"
  local release_manifest="${generation_root}/generation_release_manifest.json"
  local shard_manifest="${generation_root}/intervention_shard_manifest.json"
  local scoped_release_manifest="${generation_root}/scoped_generation_release_manifest.json"

  CURRENT_STAGE="build_${phase}_ledger"
  record_status running
  if [[ ! -f "${ledger_manifest}" ]]; then
    if [[ -e "${ledger_dir}" ]]; then
      if [[ "${LOO_RESUME:-0}" != "1" ]]; then
        echo "Incomplete ledger directory exists: ${ledger_dir}" >&2
        exit 1
      fi
      mkdir -p "${WORKFLOW_ROOT}/failed_attempts/ledgers"
      mv "${ledger_dir}" \
        "${WORKFLOW_ROOT}/failed_attempts/ledgers/${population_id}-$(date -u +%Y%m%dT%H%M%SZ)"
    fi
    "${PYTHON_BIN}" -m jobs.main.build_loo_indicator_ledger \
      --registry "${FROZEN_REGISTRY}" \
      --snapshot-manifest "${SNAPSHOT_MANIFEST}" \
      --population "${population_file}" \
      --roster "${INDICATOR_ROSTER}" \
      --output-dir "${ledger_dir}"
  fi
  "${PYTHON_BIN}" -m jobs.main.validate_loo_indicator_ledger \
    --ledger-manifest "${ledger_manifest}" \
    --registry "${FROZEN_REGISTRY}" \
    --snapshot-manifest "${SNAPSHOT_MANIFEST}" \
    --population "${population_file}" \
    --roster "${INDICATOR_ROSTER}"
  record_status complete

  if [[ -n "${EXPERIMENT_CONFIG}" && "${phase}" == "pilot" ]]; then
    local smoke_root="${WORKFLOW_ROOT}/generations/smoke_pilot_longest"
    local smoke_gate="${smoke_root}/smoke_gate_manifest.json"
    CURRENT_STAGE=generate_smoke
    record_status running
    if [[ -f "${smoke_gate}" ]]; then
      "${PYTHON_BIN}" -m jobs.generation.finalize_loo_scoped_release \
        --validate-smoke-gate "${smoke_gate}"
    else
      local smoke_resume=0
      if [[ -e "${smoke_root}" ]]; then
        if [[ "${LOO_RESUME:-0}" != "1" ]]; then
          echo "Incomplete smoke directory exists: ${smoke_root}" >&2
          exit 1
        fi
        smoke_resume=1
      fi
      LOO_FOREGROUND=1 \
      LOO_RESUME="${smoke_resume}" \
      LOO_RUN_ID="${RUN_ID}-smoke" \
      LOO_RUN_ROOT="${smoke_root}" \
      LOO_SCOPED_RELEASE_KIND=smoke \
      LOO_EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG}" \
      LOO_REUSE_ANALYSIS_MANIFEST="${PILOT_REUSE_ANALYSIS_MANIFEST}" \
      LOO_REUSE_ANALYSIS_OUTPUT="${PILOT_REUSE_ANALYSIS_OUTPUT}" \
      LOO_REUSE_PROJECTION_MANIFEST="${PILOT_REUSE_PROJECTION_MANIFEST}" \
      LOO_REUSE_PROJECTION_OUTPUT="${PILOT_REUSE_PROJECTION_OUTPUT}" \
      LOO_REUSE_PROJECTION_OUTPUT_SHA256="${PILOT_REUSE_PROJECTION_OUTPUT_SHA256}" \
      LOO_INDICATOR_INPUT="${ledger_dir}/indicator_inputs.jsonl" \
      LOO_LEDGER_MANIFEST="${ledger_manifest}" \
      LOO_SNAPSHOT_MANIFEST="${SNAPSHOT_MANIFEST}" \
      LOO_SOURCE_REGISTRY="${FROZEN_REGISTRY}" \
      LOO_INDICATOR_ROSTER="${INDICATOR_ROSTER}" \
      LOO_SECTION_ROSTER="${SECTION_ROSTER}" \
      LOO_GENERATION_CONFIG="${GENERATION_CONFIG}" \
      LOO_PILOT_POPULATION="${PILOT_POPULATION}" \
      LOO_FORMAL_POPULATION="${FORMAL_POPULATION}" \
      LOO_ANALYSIS_MODEL="${ANALYSIS_MODEL}" \
      LOO_ANALYSIS_TOKENIZER="${ANALYSIS_TOKENIZER}" \
      LOO_MINUTES_MODEL="${MINUTES_MODEL}" \
      LOO_MINUTES_TOKENIZER="${MINUTES_TOKENIZER}" \
      LOO_MINUTES_TOKEN_LIMIT_POLICY=error \
      LOO_BATCH_SIZE="${LOO_BATCH_SIZE:-20}" \
        "${SCRIPT_DIR}/_run_canonical_loo_generation.sh" \
        "${population_id}" "${population_file}"
      require_file "${smoke_gate}"
      "${PYTHON_BIN}" -m jobs.generation.finalize_loo_scoped_release \
        --validate-smoke-gate "${smoke_gate}"
    fi
    record_status complete
  fi

  CURRENT_STAGE="generate_${phase}"
  record_status running
  if [[ -n "${EXPERIMENT_CONFIG}" ]]; then
    if [[ -f "${scoped_release_manifest}" ]]; then
      "${PYTHON_BIN}" -m jobs.generation.finalize_loo_scoped_release \
        --validate-release "${scoped_release_manifest}"
      record_status complete
      return
    fi
    local generation_resume=0
    if [[ -e "${generation_root}" ]]; then
      if [[ "${LOO_RESUME:-0}" != "1" ]]; then
        echo "Incomplete scoped generation directory exists: ${generation_root}" >&2
        exit 1
      fi
      generation_resume=1
    fi
    REUSE_ANALYSIS_ARGS=()
    if [[ "${phase}" == "pilot" ]]; then
      REUSE_ANALYSIS_ARGS=(
        LOO_REUSE_ANALYSIS_MANIFEST="${PILOT_REUSE_ANALYSIS_MANIFEST}"
        LOO_REUSE_ANALYSIS_OUTPUT="${PILOT_REUSE_ANALYSIS_OUTPUT}"
        LOO_REUSE_PROJECTION_MANIFEST="${PILOT_REUSE_PROJECTION_MANIFEST}"
        LOO_REUSE_PROJECTION_OUTPUT="${PILOT_REUSE_PROJECTION_OUTPUT}"
        LOO_REUSE_PROJECTION_OUTPUT_SHA256="${PILOT_REUSE_PROJECTION_OUTPUT_SHA256}"
      )
    fi
    PILOT_RELEASE_ARGS=()
    if [[ "${phase}" == "formal" ]]; then
      if [[ "${MODE}" == "all" ]]; then
        PILOT_SCOPED_RELEASE="${WORKFLOW_ROOT}/generations/pilot_eval_13/scoped_generation_release_manifest.json"
      else
        PILOT_SCOPED_RELEASE=${LOO_PILOT_SCOPED_RELEASE_MANIFEST}
      fi
      PILOT_RELEASE_ARGS=(
        LOO_PILOT_SCOPED_RELEASE_MANIFEST="${PILOT_SCOPED_RELEASE}"
      )
    fi
    env \
      LOO_FOREGROUND=1 \
      LOO_RESUME="${generation_resume}" \
      LOO_RUN_ID="${RUN_ID}-${phase}" \
      LOO_RUN_ROOT="${generation_root}" \
      LOO_SCOPED_RELEASE_KIND="${phase}" \
      LOO_EXPERIMENT_CONFIG="${EXPERIMENT_CONFIG}" \
      LOO_INDICATOR_INPUT="${ledger_dir}/indicator_inputs.jsonl" \
      LOO_LEDGER_MANIFEST="${ledger_manifest}" \
      LOO_SNAPSHOT_MANIFEST="${SNAPSHOT_MANIFEST}" \
      LOO_SOURCE_REGISTRY="${FROZEN_REGISTRY}" \
      LOO_INDICATOR_ROSTER="${INDICATOR_ROSTER}" \
      LOO_SECTION_ROSTER="${SECTION_ROSTER}" \
      LOO_GENERATION_CONFIG="${GENERATION_CONFIG}" \
      LOO_PILOT_POPULATION="${PILOT_POPULATION}" \
      LOO_FORMAL_POPULATION="${FORMAL_POPULATION}" \
      LOO_ANALYSIS_MODEL="${ANALYSIS_MODEL}" \
      LOO_ANALYSIS_TOKENIZER="${ANALYSIS_TOKENIZER}" \
      LOO_MINUTES_MODEL="${MINUTES_MODEL}" \
      LOO_MINUTES_TOKENIZER="${MINUTES_TOKENIZER}" \
      LOO_MINUTES_TOKEN_LIMIT_POLICY=error \
      LOO_BATCH_SIZE="${LOO_BATCH_SIZE:-20}" \
      "${REUSE_ANALYSIS_ARGS[@]}" \
      "${PILOT_RELEASE_ARGS[@]}" \
        "${SCRIPT_DIR}/_run_canonical_loo_generation.sh" \
        "${population_id}" "${population_file}"
    require_file "${scoped_release_manifest}"
    "${PYTHON_BIN}" -m jobs.generation.finalize_loo_scoped_release \
      --validate-release "${scoped_release_manifest}"
    record_status complete
    return
  fi

  if [[ ! -f "${release_manifest}" ]]; then
    if [[ -e "${generation_root}" ]]; then
      echo "Partial generation directory cannot be resumed safely: ${generation_root}" >&2
      exit 1
    fi
    LOO_FOREGROUND=1 \
    LOO_RUN_ID="${RUN_ID}-${phase}" \
    LOO_RUN_ROOT="${generation_root}" \
    LOO_INDICATOR_INPUT="${ledger_dir}/indicator_inputs.jsonl" \
    LOO_LEDGER_MANIFEST="${ledger_manifest}" \
    LOO_SNAPSHOT_MANIFEST="${SNAPSHOT_MANIFEST}" \
    LOO_SOURCE_REGISTRY="${FROZEN_REGISTRY}" \
    LOO_INDICATOR_ROSTER="${INDICATOR_ROSTER}" \
    LOO_SECTION_ROSTER="${SECTION_ROSTER}" \
    LOO_GENERATION_CONFIG="${GENERATION_CONFIG}" \
    LOO_ANALYSIS_MODEL="${ANALYSIS_MODEL}" \
    LOO_ANALYSIS_TOKENIZER="${ANALYSIS_TOKENIZER}" \
    LOO_MINUTES_MODEL="${MINUTES_MODEL}" \
    LOO_MINUTES_TOKENIZER="${MINUTES_TOKENIZER}" \
    LOO_BATCH_SIZE="${LOO_BATCH_SIZE:-20}" \
      "${SCRIPT_DIR}/_run_canonical_loo_generation.sh" \
      "${population_id}" "${population_file}"
  fi
  if [[ -n "${LOO_INTERVENTION_INDICATORS:-}" ]]; then
    require_file "${shard_manifest}"
    record_status "$(read_manifest_status "${shard_manifest}")"
  else
    require_file "${release_manifest}"
    record_status complete
  fi
}

if [[ "${MODE}" == "pilot" || "${MODE}" == "all" ]]; then
  run_population pilot_eval_13 "${PILOT_POPULATION}" pilot
fi

if [[ -n "${LOO_INTERVENTION_INDICATORS:-}" ]]; then
  SHARD_MANIFEST="${WORKFLOW_ROOT}/generations/pilot_eval_13/intervention_shard_manifest.json"
  SHARD_STATUS=$(read_manifest_status "${SHARD_MANIFEST}")
  CURRENT_STAGE=workflow
  record_status "${SHARD_STATUS}"
  echo "status=${SHARD_STATUS}"
  echo "standalone_canonical_release=false"
  echo "intervention_shard_manifest=${SHARD_MANIFEST}"
  echo "output=${WORKFLOW_ROOT}"
  exit 0
fi

if [[ "${MODE}" == "formal" || "${MODE}" == "all" ]]; then
  if [[ -n "${EXPERIMENT_CONFIG}" ]]; then
    if [[ "${MODE}" == "formal" ]]; then
      PILOT_RELEASE=${LOO_PILOT_SCOPED_RELEASE_MANIFEST}
    else
      PILOT_RELEASE="${WORKFLOW_ROOT}/generations/pilot_eval_13/scoped_generation_release_manifest.json"
    fi
    require_file "${PILOT_RELEASE}"
    "${PYTHON_BIN}" -m jobs.generation.finalize_loo_scoped_release \
      --validate-release "${PILOT_RELEASE}"
  elif [[ "${MODE}" == "formal" ]]; then
    CURRENT_STAGE=bind_pilot_prerequisite
    mkdir -p "${WORKFLOW_ROOT}/prerequisites"
    PILOT_RELEASE="${WORKFLOW_ROOT}/prerequisites/pilot_release_manifest.json"
    if [[ -f "${PILOT_RELEASE}" ]]; then
      cmp -s "${LOO_PILOT_RELEASE_MANIFEST}" "${PILOT_RELEASE}" || {
        echo "Frozen pilot prerequisite differs from requested manifest." >&2
        exit 1
      }
    else
      install -m 0444 "${LOO_PILOT_RELEASE_MANIFEST}" "${PILOT_RELEASE}"
    fi
  else
    PILOT_RELEASE="${WORKFLOW_ROOT}/generations/pilot_eval_13/generation_release_manifest.json"
  fi
  require_file "${PILOT_RELEASE}"
  run_population formal_test_13 "${FORMAL_POPULATION}" formal
fi

score_scoped_population() {
  local population_id=$1
  local population_file=$2
  local generation_root="${WORKFLOW_ROOT}/generations/${population_id}"
  local score_dir="${WORKFLOW_ROOT}/scoring/${population_id}"
  local report_dir="${score_dir}/report"
  local result_manifest="${report_dir}/${population_id}_loo_six_indicator_manifest.json"
  local reference_file="${score_dir}/${population_id}_actual_minutes_reference.jsonl"
  local reference_manifest="${score_dir}/${population_id}_actual_minutes_reference_manifest.json"
  local actual_minutes_source=${LOO_ACTUAL_MINUTES_SOURCE:-"${REPO_ROOT}/archive/data/dataset_20260421/raw_data/archive/synthetic_text_20250518.jsonl"}
  local actual_minutes_source_sha256=${LOO_ACTUAL_MINUTES_SOURCE_SHA256:-"0edc4c44ce13870933a461d3507cc24c40e792667e54bda97a0150b1ebc3ac06"}
  local embedding_model=${LOO_EMBEDDING_MODEL:-"${REPO_ROOT}/models/DeepSeek-R1-Distill-Llama-8B"}
  local embedding_model_sha256=${LOO_EMBEDDING_MODEL_SHA256:-"bfb086cb87e9616e60805fc6f6f95826294b1de70b158cfea2d3e213df38ca11"}
  local scoring_python=${LOO_SCORING_PYTHON:-"${REPO_ROOT}/.missing-scoring-python"}
  if [[ ! -x "${scoring_python}" ]]; then
    if [[ -n "${HOME:-}" && -x "${HOME}/.conda/envs/fomc_trainer/bin/python" ]]; then
      scoring_python="${HOME}/.conda/envs/fomc_trainer/bin/python"
    else
      scoring_python=${PYTHON_BIN}
    fi
  fi
  require_file "${actual_minutes_source}"
  require_dir "${embedding_model}"

  CURRENT_STAGE="score_${population_id}"
  record_status running
  if [[ -f "${result_manifest}" ]]; then
    PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" \
      "${scoring_python}" -m jobs.eval.summarize_canonical_loo \
        --scope-manifest "${EXPERIMENT_CONFIG}" \
        --population-id "${population_id}" \
        --validate-report-manifest "${result_manifest}"
    record_status complete
    return
  fi
  if [[ -e "${score_dir}" ]]; then
    if [[ "${LOO_RESUME:-0}" != "1" ]]; then
      echo "Incomplete scoped scoring directory exists: ${score_dir}" >&2
      exit 1
    fi
    mkdir -p "${WORKFLOW_ROOT}/failed_attempts/scoring"
    mv "${score_dir}" \
      "${WORKFLOW_ROOT}/failed_attempts/scoring/${population_id}-$(date -u +%Y%m%dT%H%M%SZ)"
  fi
  mkdir -p "${score_dir}"
  PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" \
    "${scoring_python}" -u -m jobs.eval.build_loo_actual_minutes_reference \
      --source-file "${actual_minutes_source}" \
      --population-manifest "${population_file}" \
      --scope-manifest "${EXPERIMENT_CONFIG}" \
      --population-id "${population_id}" \
      --expected-source-sha256 "${actual_minutes_source_sha256}" \
      --output-file "${reference_file}" \
      --manifest-file "${reference_manifest}"

  run_scoped_arm_score() {
    local arm=$1
    local regime=$2
    local strategy=$3
    PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" \
      "${scoring_python}" -u -m jobs.eval.eval_leave_one_out \
        --input-folder "${generation_root}/generations/${arm}" \
        --output-file "${score_dir}/${arm}_summary.jsonl" \
        --target-mode actual-minutes \
        --reference-file "${reference_file}" \
        --reference-manifest "${reference_manifest}" \
        --scope-manifest "${EXPERIMENT_CONFIG}" \
        --generation-release-manifest "${generation_root}/scoped_generation_release_manifest.json" \
        --population-id "${population_id}" \
        --regime "${regime}" \
        --intervention-strategy "${strategy}" \
        --embedding-model-path "${embedding_model}" \
        --embedding-model-sha256 "${embedding_model_sha256}" \
        --embedding-batch-size "${LOO_EMBEDDING_BATCH_SIZE:-1}" \
        --embedding-long-text-policy chunk-mean \
        --embedding-max-tokens "${LOO_EMBEDDING_MAX_TOKENS:-4096}" \
        --score-chunk-size "${LOO_SCORE_CHUNK_SIZE:-64}" \
        --bootstrap-samples 10000 \
        --bootstrap-seed 20260728
  }
  run_scoped_arm_score deletion_primary primary indicator_block_deletion
  run_scoped_arm_score neutral_primary primary indicator_block_neutral_replacement
  run_scoped_arm_score deletion_stochastic stochastic indicator_block_deletion
  run_scoped_arm_score neutral_stochastic stochastic indicator_block_neutral_replacement

  PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}" \
    "${scoring_python}" -u -m jobs.eval.summarize_canonical_loo \
      --scope-manifest "${EXPERIMENT_CONFIG}" \
      --population-id "${population_id}" \
      --deletion-primary-rows "${score_dir}/deletion_primary_summary.rows.jsonl" \
      --deletion-primary-audit "${score_dir}/deletion_primary_summary.audit.json" \
      --neutral-primary-rows "${score_dir}/neutral_primary_summary.rows.jsonl" \
      --neutral-primary-audit "${score_dir}/neutral_primary_summary.audit.json" \
      --deletion-stochastic-rows "${score_dir}/deletion_stochastic_summary.rows.jsonl" \
      --deletion-stochastic-audit "${score_dir}/deletion_stochastic_summary.audit.json" \
      --neutral-stochastic-rows "${score_dir}/neutral_stochastic_summary.rows.jsonl" \
      --neutral-stochastic-audit "${score_dir}/neutral_stochastic_summary.audit.json" \
      --output-dir "${report_dir}" \
      --bootstrap-draws 10000 \
      --bootstrap-seed 20260728
  require_file "${result_manifest}"
  record_status complete
}

if [[ -n "${EXPERIMENT_CONFIG}" ]]; then
  if [[ "${MODE}" == "pilot" || "${MODE}" == "all" ]]; then
    score_scoped_population pilot_eval_13 "${PILOT_POPULATION}"
  fi
  if [[ "${MODE}" == "formal" || "${MODE}" == "all" ]]; then
    score_scoped_population formal_test_13 "${FORMAL_POPULATION}"
  fi
fi

CURRENT_STAGE=seal_workflow
record_status running
if [[ -n "${EXPERIMENT_CONFIG}" ]]; then
  SCOPED_WORKFLOW_ARGS=(
    --run-id "${RUN_ID}"
    --mode "${MODE}"
    --workflow-root "${WORKFLOW_ROOT}"
    --experiment-config "${EXPERIMENT_CONFIG}"
    --source-registry "${FROZEN_REGISTRY}"
    --snapshot-manifest "${SNAPSHOT_MANIFEST}"
  )
  if [[ "${MODE}" == "pilot" || "${MODE}" == "all" ]]; then
    SCOPED_WORKFLOW_ARGS+=(
      --smoke-gate "${WORKFLOW_ROOT}/generations/smoke_pilot_longest/smoke_gate_manifest.json"
      --pilot-ledger "${WORKFLOW_ROOT}/ledgers/pilot_eval_13/ledger_manifest.json"
      --pilot-release "${WORKFLOW_ROOT}/generations/pilot_eval_13/scoped_generation_release_manifest.json"
      --pilot-results-manifest "${WORKFLOW_ROOT}/scoring/pilot_eval_13/report/pilot_eval_13_loo_six_indicator_manifest.json"
      --reused-pilot-analysis-manifest "${PILOT_REUSE_ANALYSIS_MANIFEST}"
      --reused-pilot-analysis-output "${PILOT_REUSE_ANALYSIS_OUTPUT}"
      --reused-pilot-projection-manifest "${PILOT_REUSE_PROJECTION_MANIFEST}"
      --reused-pilot-projection-output "${PILOT_REUSE_PROJECTION_OUTPUT}"
    )
  else
    SCOPED_WORKFLOW_ARGS+=(
      --pilot-release "${PILOT_RELEASE}"
      --pilot-results-manifest "${LOO_PILOT_RESULTS_MANIFEST}"
    )
  fi
  if [[ "${MODE}" == "formal" || "${MODE}" == "all" ]]; then
    SCOPED_WORKFLOW_ARGS+=(
      --formal-ledger "${WORKFLOW_ROOT}/ledgers/formal_test_13/ledger_manifest.json"
      --formal-release "${WORKFLOW_ROOT}/generations/formal_test_13/scoped_generation_release_manifest.json"
      --formal-results-manifest "${WORKFLOW_ROOT}/scoring/formal_test_13/report/formal_test_13_loo_six_indicator_manifest.json"
    )
  fi
  if [[ -f "${SCOPED_WORKFLOW_MANIFEST}" ]]; then
    "${PYTHON_BIN}" -m jobs.generation.finalize_loo_scoped_workflow \
      --validate-workflow "${SCOPED_WORKFLOW_MANIFEST}"
  else
    "${PYTHON_BIN}" -m jobs.generation.finalize_loo_scoped_workflow \
      "${SCOPED_WORKFLOW_ARGS[@]}" \
      --output "${SCOPED_WORKFLOW_MANIFEST}"
  fi
  record_status complete
  CURRENT_STAGE=workflow
  record_status complete
  echo "status=complete"
  echo "scoped_workflow_release_manifest=${WORKFLOW_ROOT}/scoped_workflow_release_manifest.json"
  echo "output=${WORKFLOW_ROOT}"
  exit 0
fi

WORKFLOW_ARTIFACTS=(
  --artifact "source_registry=${FROZEN_REGISTRY}"
  --artifact "snapshot_manifest=${SNAPSHOT_MANIFEST}"
)
if [[ "${MODE}" == "pilot" || "${MODE}" == "all" ]]; then
  WORKFLOW_ARTIFACTS+=(
    --artifact "pilot_ledger_manifest=${WORKFLOW_ROOT}/ledgers/pilot_eval_13/ledger_manifest.json"
    --artifact "pilot_release_manifest=${WORKFLOW_ROOT}/generations/pilot_eval_13/generation_release_manifest.json"
  )
fi
if [[ "${MODE}" == "formal" || "${MODE}" == "all" ]]; then
  if [[ "${MODE}" == "formal" ]]; then
    WORKFLOW_ARTIFACTS+=(--artifact "pilot_release_manifest=${PILOT_RELEASE}")
  fi
  WORKFLOW_ARTIFACTS+=(
    --artifact "formal_ledger_manifest=${WORKFLOW_ROOT}/ledgers/formal_test_13/ledger_manifest.json"
    --artifact "formal_release_manifest=${WORKFLOW_ROOT}/generations/formal_test_13/generation_release_manifest.json"
  )
fi
"${PYTHON_BIN}" -m jobs.main.finalize_loo_workflow \
  --run-id "${RUN_ID}" \
  --mode "${MODE}" \
  --workflow-root "${WORKFLOW_ROOT}" \
  "${WORKFLOW_ARTIFACTS[@]}" \
  --output "${WORKFLOW_ROOT}/workflow_release_manifest.json"
record_status complete
CURRENT_STAGE=workflow
record_status complete

echo "status=complete"
echo "workflow_release_manifest=${WORKFLOW_ROOT}/workflow_release_manifest.json"
echo "output=${WORKFLOW_ROOT}"
