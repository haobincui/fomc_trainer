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

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
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
PILOT_POPULATION=${LOO_PILOT_POPULATION:-"${REPO_ROOT}/configs/main/loo_population_pilot_eval_13.json"}
FORMAL_POPULATION=${LOO_FORMAL_POPULATION:-"${REPO_ROOT}/configs/main/loo_population_formal_test_13.json"}
ANALYSIS_MODEL=${LOO_ANALYSIS_MODEL:-"${REPO_ROOT}/../fomc_trainer_back/fomc_trainer/output/merged/llama_grpo_20250515"}
ANALYSIS_TOKENIZER=${LOO_ANALYSIS_TOKENIZER:-"${REPO_ROOT}/models/DeepSeek-R1-Distill-Llama-8B"}
MINUTES_MODEL=${LOO_MINUTES_MODEL:-"${REPO_ROOT}/../fomc_trainer_back/fomc_trainer/output/merged/llama_sft_synthetic_20250526"}
MINUTES_TOKENIZER=${LOO_MINUTES_TOKENIZER:-"${MINUTES_MODEL}"}
FETCH_WORKERS=${LOO_FETCH_WORKERS:-1}
FETCH_RATE=${LOO_FETCH_RATE:-1}

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
  require_file "${LOO_PILOT_RELEASE_MANIFEST:-}"
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

if [[ "${INTERNAL_MODE}" != "--worker" ]]; then
  "${PYTHON_BIN}" - <<'PY'
import sys

try:
    import torch
    import transformers
    import vllm
    from generate_new_response import generate_new_response  # noqa: F401
except Exception as error:
    raise SystemExit(
        "Generation preflight failed. Set LOO_PYTHON to a Python environment "
        f"with working torch, transformers, and vLLM: {type(error).__name__}: "
        f"{error}"
    ) from error
if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
    raise SystemExit("Generation preflight requires at least one visible CUDA GPU")
print(
    f"generation_python={sys.executable} torch={torch.__version__} "
    f"transformers={transformers.__version__} vllm={vllm.__version__} "
    f"visible_gpus={torch.cuda.device_count()}"
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
      echo "Workflow already has a live worker (PID ${EXISTING_PID}): ${WORKFLOW_ROOT}" >&2
      exit 1
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
FROZEN_INPUT_DIR="${WORKFLOW_ROOT}/inputs"
FROZEN_REGISTRY="${FROZEN_INPUT_DIR}/loo_indicator_sources.json"
FROZEN_INDICATOR_ROSTER="${FROZEN_INPUT_DIR}/leave_one_out_roster.json"
FROZEN_SECTION_ROSTER="${FROZEN_INPUT_DIR}/loo_sections.json"
FROZEN_GENERATION_CONFIG="${FROZEN_INPUT_DIR}/canonical_loo_generation.json"
FROZEN_PILOT_POPULATION="${FROZEN_INPUT_DIR}/loo_population_pilot_eval_13.json"
FROZEN_FORMAL_POPULATION="${FROZEN_INPUT_DIR}/loo_population_formal_test_13.json"
FROZEN_PILOT_RELEASE="${FROZEN_INPUT_DIR}/pilot_release_manifest.json"
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
if [[ "${MODE}" == "formal" ]]; then
  freeze_input \
    "${LOO_PILOT_RELEASE_MANIFEST}" \
    "${FROZEN_PILOT_RELEASE}" \
    "pilot release prerequisite"
  LOO_PILOT_RELEASE_MANIFEST=${FROZEN_PILOT_RELEASE}
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

  CURRENT_STAGE="build_${phase}_ledger"
  record_status running
  "${PYTHON_BIN}" -m jobs.main.build_loo_indicator_ledger \
    --registry "${FROZEN_REGISTRY}" \
    --snapshot-manifest "${SNAPSHOT_MANIFEST}" \
    --population "${population_file}" \
    --roster "${INDICATOR_ROSTER}" \
    --output-dir "${ledger_dir}"
  "${PYTHON_BIN}" -m jobs.main.validate_loo_indicator_ledger \
    --ledger-manifest "${ledger_manifest}" \
    --registry "${FROZEN_REGISTRY}" \
    --snapshot-manifest "${SNAPSHOT_MANIFEST}" \
    --population "${population_file}" \
    --roster "${INDICATOR_ROSTER}"
  record_status complete

  CURRENT_STAGE="generate_${phase}"
  record_status running
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
  require_file "${release_manifest}"
  record_status complete
}

if [[ "${MODE}" == "pilot" || "${MODE}" == "all" ]]; then
  run_population pilot_eval_13 "${PILOT_POPULATION}" pilot
fi

if [[ "${MODE}" == "formal" || "${MODE}" == "all" ]]; then
  if [[ "${MODE}" == "formal" ]]; then
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

CURRENT_STAGE=seal_workflow
record_status running
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
