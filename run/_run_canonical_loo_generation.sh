#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 POPULATION_ID POPULATION_JSON [--worker RUN_ID RUN_ROOT]" >&2
  exit 2
fi

POPULATION_ID=$1
POPULATION_JSON=$2
MODE=${3:-}

case "${POPULATION_ID}" in
  pilot_eval_13)
    PHASE=pilot
    ;;
  formal_test_13)
    PHASE=formal
    ;;
  *)
    echo "Unsupported frozen population: ${POPULATION_ID}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
SCRIPT_PATH="${SCRIPT_DIR}/$(basename "${BASH_SOURCE[0]}")"
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)
source "${SCRIPT_DIR}/_canonical_gpu1.sh"
export PYTHONPATH="${REPO_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"

ANALYSIS_MODEL=${LOO_ANALYSIS_MODEL:-"${REPO_ROOT}/../fomc_trainer_back/fomc_trainer/output/merged/llama_grpo_20250515"}
ANALYSIS_TOKENIZER=${LOO_ANALYSIS_TOKENIZER:-"${REPO_ROOT}/models/DeepSeek-R1-Distill-Llama-8B"}
MINUTES_MODEL=${LOO_MINUTES_MODEL:-"${REPO_ROOT}/output/checkpoints/recovered/llama_sft_synthetic_20250526_2_cp1668_recovered_v1_20260729/model"}
MINUTES_TOKENIZER=${LOO_MINUTES_TOKENIZER:-"${MINUTES_MODEL}"}
INDICATOR_ROSTER=${LOO_INDICATOR_ROSTER:-"${REPO_ROOT}/configs/main/leave_one_out_roster.json"}
SECTION_ROSTER=${LOO_SECTION_ROSTER:-"${REPO_ROOT}/configs/main/loo_sections.json"}
GENERATION_CONFIG=${LOO_GENERATION_CONFIG:-"${REPO_ROOT}/configs/main/canonical_loo_generation.json"}
EXPERIMENT_CONFIG=${LOO_EXPERIMENT_CONFIG:-}
PILOT_POPULATION_FILE=${LOO_PILOT_POPULATION:-"${REPO_ROOT}/configs/main/loo_population_pilot_eval_13.json"}
FORMAL_POPULATION_FILE=${LOO_FORMAL_POPULATION:-"${REPO_ROOT}/configs/main/loo_population_formal_test_13.json"}
INDICATOR_INPUT=${LOO_INDICATOR_INPUT:-"${REPO_ROOT}/dataset/processed/main/evaluation_inputs/canonical_loo/${POPULATION_ID}/indicator_inputs.jsonl"}
LEDGER_MANIFEST=${LOO_LEDGER_MANIFEST:-"${INDICATOR_INPUT%/*}/ledger_manifest.json"}
SNAPSHOT_MANIFEST=${LOO_SNAPSHOT_MANIFEST:-}
SOURCE_REGISTRY=${LOO_SOURCE_REGISTRY:-"${REPO_ROOT}/configs/main/loo_indicator_sources.json"}
if [[ -n "${LOO_PYTHON:-}" ]]; then
  PYTHON_BIN=${LOO_PYTHON}
elif [[ -n "${HOME:-}" && -x "${HOME}/.conda/envs/llama_factory/bin/python" ]]; then
  PYTHON_BIN="${HOME}/.conda/envs/llama_factory/bin/python"
else
  PYTHON_BIN=python
fi
BATCH_SIZE=${LOO_BATCH_SIZE:-20}
PARTIAL_INTERVENTION_SHARD=0
SCOPED_EXPERIMENT=0
SCOPED_RELEASE_KIND=${LOO_SCOPED_RELEASE_KIND:-${PHASE}}
INTERVENTION_INDICATORS=()
INTERVENTION_INDICATOR_ARGS=()
SAMPLE_SELECTION_ARGS=()
SMOKE_SAMPLE_IDS=()
EXPERIMENT_CONFIG_ARGS=()
EXPERIMENT_SPEC_ARGS=()

require_file() {
  if [[ -z "$1" || ! -f "$1" ]]; then
    echo "Required file does not exist: $1" >&2
    exit 1
  fi
}

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "Required directory does not exist: $1" >&2
    exit 1
  fi
}

require_file "${POPULATION_JSON}"
require_file "${INDICATOR_INPUT}"
require_file "${INDICATOR_ROSTER}"
require_file "${SECTION_ROSTER}"
require_file "${GENERATION_CONFIG}"
require_file "${LEDGER_MANIFEST}"
require_file "${SNAPSHOT_MANIFEST}"
require_file "${SOURCE_REGISTRY}"
require_dir "${ANALYSIS_MODEL}"
require_dir "${ANALYSIS_TOKENIZER}"
require_dir "${MINUTES_MODEL}"
require_dir "${MINUTES_TOKENIZER}"

if [[ -n "${EXPERIMENT_CONFIG}" ]]; then
  require_file "${EXPERIMENT_CONFIG}"
  require_file "${PILOT_POPULATION_FILE}"
  require_file "${FORMAL_POPULATION_FILE}"
  case "${SCOPED_RELEASE_KIND}" in
    smoke|pilot|formal)
      ;;
    *)
      echo "LOO_SCOPED_RELEASE_KIND must be smoke, pilot, or formal." >&2
      exit 2
      ;;
  esac
  if [[ "${SCOPED_RELEASE_KIND}" == "smoke" && "${POPULATION_ID}" != "pilot_eval_13" ]]; then
    echo "The scoped smoke gate must use pilot_eval_13." >&2
    exit 2
  fi
  if [[ "${SCOPED_RELEASE_KIND}" == "formal" && "${POPULATION_ID}" != "formal_test_13" ]]; then
    echo "A scoped formal release must use formal_test_13." >&2
    exit 2
  fi
  if [[ "${SCOPED_RELEASE_KIND}" == "pilot" && "${POPULATION_ID}" != "pilot_eval_13" ]]; then
    echo "A scoped pilot release must use pilot_eval_13." >&2
    exit 2
  fi
  if [[ -n "${LOO_INTERVENTION_INDICATORS:-}" ]]; then
    echo "LOO_EXPERIMENT_CONFIG cannot be combined with LOO_INTERVENTION_INDICATORS." >&2
    exit 2
  fi
  mapfile -t INTERVENTION_INDICATORS < <(
    "${PYTHON_BIN}" -m open_r1.validator.loo_experiment_scope \
      --experiment-config "${EXPERIMENT_CONFIG}" \
      --roster "${INDICATOR_ROSTER}" \
      --pilot-population "${PILOT_POPULATION_FILE}" \
      --formal-population "${FORMAL_POPULATION_FILE}" \
      --print-intervention-indicators
  )
  if [[ ${#INTERVENTION_INDICATORS[@]} -ne 6 ]]; then
    echo "Scoped legacy-six experiment did not resolve exactly six indicators." >&2
    exit 1
  fi
  SCOPED_EXPERIMENT=1
  PARTIAL_INTERVENTION_SHARD=1
  EXPERIMENT_CONFIG_ARGS=(--experiment-config "${EXPERIMENT_CONFIG}")
  EXPERIMENT_SPEC_ARGS=(--source "experiment_scope=${EXPERIMENT_CONFIG}")
  for indicator in "${INTERVENTION_INDICATORS[@]}"; do
    INTERVENTION_INDICATOR_ARGS+=(
      --include-intervention-indicator "${indicator}"
    )
  done
fi

if [[ "${SCOPED_EXPERIMENT}" == "0" && -n "${LOO_INTERVENTION_INDICATORS:-}" ]]; then
  mapfile -t INTERVENTION_INDICATORS < <(
    "${PYTHON_BIN}" - "${INDICATOR_ROSTER}" \
      "${LOO_INTERVENTION_INDICATORS}" <<'PY'
import json
import sys
from pathlib import Path

roster = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
baseline = str(roster.get("baseline_indicator") or "").strip()
indicators = [
    str(value).strip()
    for value in roster.get("indicators", [])
    if str(value).strip()
]
requested = [value.strip() for value in sys.argv[2].split(",")]
if not requested or any(not value for value in requested):
    raise SystemExit(
        "LOO_INTERVENTION_INDICATORS must be a comma-separated non-empty list"
    )
if len(requested) != len(set(requested)):
    raise SystemExit("LOO_INTERVENTION_INDICATORS contains duplicate values")
if baseline in requested:
    raise SystemExit(
        f"Do not include baseline {baseline!r}; it is generated automatically"
    )
unknown = sorted(set(requested) - set(indicators))
if unknown:
    raise SystemExit(
        f"LOO_INTERVENTION_INDICATORS contains values outside the roster: {unknown}"
    )
requested_set = set(requested)
ordered = [value for value in indicators if value in requested_set]
if ordered == indicators:
    raise SystemExit(
        "LOO_INTERVENTION_INDICATORS selects the full roster; unset it for a "
        "standalone canonical run"
    )
for value in ordered:
    print(value)
PY
  )
  if [[ ${#INTERVENTION_INDICATORS[@]} -eq 0 ]]; then
    echo "LOO_INTERVENTION_INDICATORS resolved to an empty shard." >&2
    exit 2
  fi
  PARTIAL_INTERVENTION_SHARD=1
  for indicator in "${INTERVENTION_INDICATORS[@]}"; do
    INTERVENTION_INDICATOR_ARGS+=(
      --include-intervention-indicator "${indicator}"
    )
  done
fi

if [[ -n "${LOO_MINUTES_TOKEN_LIMIT_POLICY:-}" ]]; then
  MINUTES_TOKEN_LIMIT_POLICY=${LOO_MINUTES_TOKEN_LIMIT_POLICY}
else
  MINUTES_TOKEN_LIMIT_POLICY=error
fi
case "${MINUTES_TOKEN_LIMIT_POLICY}" in
  error|exclude)
    ;;
  *)
    echo "LOO_MINUTES_TOKEN_LIMIT_POLICY must be 'error' or 'exclude'." >&2
    exit 2
    ;;
esac
if [[ "${SCOPED_EXPERIMENT}" == "1" && "${MINUTES_TOKEN_LIMIT_POLICY}" != "error" ]]; then
  echo "Scoped six-indicator releases require LOO_MINUTES_TOKEN_LIMIT_POLICY=error." >&2
  exit 2
fi
if [[ "${SCOPED_EXPERIMENT}" == "1" && "${SCOPED_RELEASE_KIND}" == "formal" ]]; then
  require_file "${LOO_PILOT_SCOPED_RELEASE_MANIFEST:-}"
fi

ANALYSIS_SETTINGS=$(
  "${PYTHON_BIN}" - "${GENERATION_CONFIG}" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1])
config = json.loads(config_path.read_text(encoding="utf-8"))
try:
    analysis = config["decoding"]["indicator_analysis"]
    max_new_tokens = analysis["max_new_tokens"]
    requested_max_output_tokens = analysis["requested_max_output_tokens"]
    max_model_len = analysis["max_model_len"]
    max_token_limit_errors = analysis["max_token_limit_errors"]
except (KeyError, TypeError) as error:
    raise SystemExit(
        "Frozen generation config lacks complete indicator_analysis decoding "
        f"settings: {error}"
    ) from error
values = (
    max_new_tokens,
    requested_max_output_tokens,
    max_model_len,
    max_token_limit_errors,
)
if any(isinstance(value, bool) or not isinstance(value, int) for value in values):
    raise SystemExit("Indicator-analysis decoding settings must be integers")
if (
    requested_max_output_tokens <= 0
    or requested_max_output_tokens > max_new_tokens
    or max_new_tokens <= 0
    or max_model_len <= max_new_tokens
):
    raise SystemExit("Invalid indicator-analysis token budget")
if not 0 <= max_token_limit_errors <= 2:
    raise SystemExit("max_token_limit_errors must be from 0 through 2")
print(*values)
PY
)
read -r ANALYSIS_MAX_NEW_TOKENS ANALYSIS_REQUESTED_MAX_OUTPUT_TOKENS \
  ANALYSIS_MAX_MODEL_LEN ANALYSIS_MAX_TOKEN_LIMIT_ERRORS \
  <<<"${ANALYSIS_SETTINGS}"

CANONICAL_SETTINGS=$(
  "${PYTHON_BIN}" - "${GENERATION_CONFIG}" <<'PY'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
try:
    execution = config["execution"]
    projection = config["analysis_projection"]
    primary = config["decoding"]["minutes_primary"]
    stochastic = config["decoding"]["minutes_stochastic_robustness"]
except (KeyError, TypeError) as error:
    raise SystemExit(
        f"Frozen generation config lacks canonical execution settings: {error}"
    ) from error

expected_execution = {
    "gpu_policy": "single-physical-gpu-by-nvidia-smi-index-v1",
    "physical_gpu_index": 1,
    "expected_visible_cuda_devices": 1,
    "tensor_parallel_size": 1,
}
expected_projection = {
    "policy": "deepseek-final-answer-after-think-v1",
    "source_field": "generated",
    "output_field": "minutes_analysis",
    "required_format": "deepseek_think_completion",
    "delimiter": "</think>",
    "required_delimiter_count": 1,
    "allow_plain_text_fallback": False,
    "require_nonempty_reasoning": True,
    "require_nonempty_final_answer": True,
    "truncation": "forbidden",
    "secondary_generation": "forbidden",
}
if any(execution.get(key) != value for key, value in expected_execution.items()):
    raise SystemExit("Frozen execution policy must pin physical GPU 1 with TP=1")
if any(projection.get(key) != value for key, value in expected_projection.items()):
    raise SystemExit("Frozen analysis projection policy is not canonical")

primary_budget = (primary.get("max_new_tokens"), primary.get("max_model_len"))
stochastic_budget = (
    stochastic.get("max_new_tokens"),
    stochastic.get("max_model_len"),
)
if primary_budget != (8192, 16384) or stochastic_budget != primary_budget:
    raise SystemExit(
        "Canonical Minutes primary/stochastic budgets must both be 8192/16384"
    )
print(
    primary_budget[0],
    primary_budget[1],
    projection["source_field"],
    projection["output_field"],
    execution["physical_gpu_index"],
    execution["expected_visible_cuda_devices"],
    execution["tensor_parallel_size"],
)
PY
)
read -r MINUTES_MAX_NEW_TOKENS MINUTES_MAX_MODEL_LEN \
  PROJECTION_SOURCE_FIELD PROJECTION_OUTPUT_FIELD \
  CONFIGURED_GPU_INDEX CONFIGURED_VISIBLE_GPU_COUNT CONFIGURED_TP_SIZE \
  <<<"${CANONICAL_SETTINGS}"

CHECKPOINT_SETTINGS=$(
  "${PYTHON_BIN}" - "${GENERATION_CONFIG}" "${REPO_ROOT}" <<'PY'
import json
import sys
from pathlib import Path

config_path = Path(sys.argv[1]).expanduser().resolve()
repo_root = Path(sys.argv[2]).expanduser().resolve()
config = json.loads(config_path.read_text(encoding="utf-8"))
try:
    artifacts = config["artifacts"]
    provenance = config["checkpoint_provenance"]
    values = {
        "model": artifacts["minutes_model"],
        "tokenizer": artifacts["minutes_tokenizer"],
        "manifest": provenance["minutes_checkpoint_manifest"],
        "artifact_id": provenance["minutes_artifact_id"],
        "manifest_sha256": provenance["minutes_manifest_sha256"],
        "manifest_payload_sha256": (
            provenance["minutes_manifest_payload_sha256"]
        ),
        "model_sha256": provenance["minutes_model_sha256"],
        "tokenizer_sha256": provenance["minutes_tokenizer_sha256"],
        "runtime_verification": provenance["runtime_verification"],
    }
except (KeyError, TypeError) as error:
    raise SystemExit(
        "Frozen generation config lacks complete checkpoint provenance "
        f"settings: {error}"
    ) from error

def resolve_from_repo(value: object, *, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise SystemExit(f"{label} must be a non-empty path")
    path = Path(text).expanduser()
    return (path if path.is_absolute() else repo_root / path).resolve()

def require_sha256(value: object, *, label: str) -> str:
    text = str(value or "").strip().lower()
    if len(text) != 64 or any(c not in "0123456789abcdef" for c in text):
        raise SystemExit(f"{label} must be a lowercase SHA-256 digest")
    return text

if values["artifact_id"] != "eval-minutes-sft-from-chk1":
    raise SystemExit(
        "Canonical Minutes artifact must be eval-minutes-sft-from-chk1"
    )
if values["runtime_verification"] != "required-before-generation":
    raise SystemExit("Runtime checkpoint verification must be required")

print(
    resolve_from_repo(values["model"], label="Minutes model"),
    resolve_from_repo(values["tokenizer"], label="Minutes tokenizer"),
    resolve_from_repo(values["manifest"], label="checkpoint manifest"),
    values["artifact_id"],
    require_sha256(values["manifest_sha256"], label="manifest SHA-256"),
    require_sha256(
        values["manifest_payload_sha256"],
        label="manifest payload SHA-256",
    ),
    require_sha256(values["model_sha256"], label="model SHA-256"),
    require_sha256(values["tokenizer_sha256"], label="tokenizer SHA-256"),
)
PY
)
read -r CONFIGURED_MINUTES_MODEL CONFIGURED_MINUTES_TOKENIZER \
  MINUTES_CHECKPOINT_MANIFEST MINUTES_CHECKPOINT_ARTIFACT_ID \
  MINUTES_CHECKPOINT_MANIFEST_SHA256 \
  MINUTES_CHECKPOINT_MANIFEST_PAYLOAD_SHA256 \
  MINUTES_CHECKPOINT_MODEL_SHA256 MINUTES_CHECKPOINT_TOKENIZER_SHA256 \
  <<<"${CHECKPOINT_SETTINGS}"

RESOLVED_MINUTES_MODEL=$(cd "${MINUTES_MODEL}" && pwd -P)
RESOLVED_MINUTES_TOKENIZER=$(cd "${MINUTES_TOKENIZER}" && pwd -P)
if [[ "${RESOLVED_MINUTES_MODEL}" != "${CONFIGURED_MINUTES_MODEL}" ]]; then
  echo "Runtime Minutes model differs from the frozen canonical config." >&2
  echo "runtime=${RESOLVED_MINUTES_MODEL}" >&2
  echo "configured=${CONFIGURED_MINUTES_MODEL}" >&2
  exit 1
fi
if [[ "${RESOLVED_MINUTES_TOKENIZER}" != "${CONFIGURED_MINUTES_TOKENIZER}" ]]; then
  echo "Runtime Minutes tokenizer differs from the frozen canonical config." >&2
  echo "runtime=${RESOLVED_MINUTES_TOKENIZER}" >&2
  echo "configured=${CONFIGURED_MINUTES_TOKENIZER}" >&2
  exit 1
fi
require_file "${MINUTES_CHECKPOINT_MANIFEST}"

if (
  [[ "${LOO_CANONICAL_PHYSICAL_GPU_INDEX}" != "${CONFIGURED_GPU_INDEX}" ]] ||
  [[ "${LOO_CANONICAL_VISIBLE_DEVICE_COUNT}" != "${CONFIGURED_VISIBLE_GPU_COUNT}" ]] ||
  [[ "${CONFIGURED_TP_SIZE}" != "1" ]]
); then
  echo "Resolved GPU pin differs from the frozen single-GPU execution policy." >&2
  exit 1
fi

if [[ "${MODE}" != "--worker" ]]; then
  "${PYTHON_BIN}" - <<'PY'
import os
import torch
from vllm.platforms.cuda import device_id_to_physical_device_id

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
    expected_index != "1"
    or not expected_uuid
    or visible_device != expected_index
    or torch.cuda.device_count() != 1
    or logical_uuid.lower() != expected_uuid.lower()
    or device_id_to_physical_device_id(0) != int(expected_index)
):
    raise SystemExit(
        "Canonical LOO GPU preflight failed: physical GPU 1 must be the "
        "only visible CUDA device and its runtime UUID must match nvidia-smi"
    )
print(
    "canonical_gpu_preflight=ok "
    f"physical_index={expected_index} uuid={expected_uuid} "
    f"cuda_visible_devices={visible_device} logical_uuid={logical_uuid} "
    f"logical_device_count={torch.cuda.device_count()}"
)
PY
fi

if [[ "${MODE}" != "--worker" ]]; then
  RUN_ID=${LOO_RUN_ID:-"${POPULATION_ID}-$(date -u +%Y%m%dT%H%M%SZ)"}
  if [[ ! "${RUN_ID}" =~ ^[A-Za-z0-9._-]+$ ]]; then
    echo "LOO_RUN_ID must contain only letters, digits, dot, underscore, or hyphen." >&2
    exit 2
  fi
  if [[ -n "${LOO_RUN_ROOT:-}" ]]; then
    RUN_ROOT=${LOO_RUN_ROOT}
  else
    OUTPUT_BASE=${LOO_OUTPUT_BASE:-"${REPO_ROOT}/output/evaluation/main/canonical_loo/${POPULATION_ID}"}
    RUN_ROOT="${OUTPUT_BASE}/${RUN_ID}"
  fi
  mkdir -p "$(dirname "${RUN_ROOT}")"
  if [[ "${LOO_RESUME:-0}" == "1" ]]; then
    if [[ ! -d "${RUN_ROOT}" ]]; then
      echo "LOO_RESUME=1 requires an existing run directory: ${RUN_ROOT}" >&2
      exit 1
    fi
  elif ! mkdir "${RUN_ROOT}"; then
    echo "Run directory already exists; choose a new LOO_RUN_ID: ${RUN_ROOT}" >&2
    exit 1
  fi

  if [[ "${LOO_FOREGROUND:-0}" == "1" ]]; then
    exec "${SCRIPT_PATH}" "${POPULATION_ID}" "${POPULATION_JSON}" --worker "${RUN_ID}" "${RUN_ROOT}"
  fi

  nohup "${SCRIPT_PATH}" "${POPULATION_ID}" "${POPULATION_JSON}" --worker "${RUN_ID}" "${RUN_ROOT}" \
    >"${RUN_ROOT}/run.log" 2>&1 </dev/null &
  WORKER_PID=$!
  printf '%s\n' "${WORKER_PID}" >"${RUN_ROOT}/run.pid"
  echo "Started canonical LOO generation."
  echo "run_id=${RUN_ID}"
  echo "pid=${WORKER_PID}"
  echo "log=${RUN_ROOT}/run.log"
  echo "output=${RUN_ROOT}"
  exit 0
fi

if [[ $# -ne 5 ]]; then
  echo "Internal worker invocation is incomplete." >&2
  exit 2
fi

RUN_ID=$4
RUN_ROOT=$5
RAW_ANALYSIS_DIR="${RUN_ROOT}/analysis_raw"
PROJECTION_DIR="${RUN_ROOT}/analysis_projection"
PROMPT_DIR="${RUN_ROOT}/prompts"
SPEC_FILE="${RUN_ROOT}/generation_spec.json"
PREPARED_RUN_ROOT=""
REUSED_PROJECTION=0
RESUMING_SAME_RUN=0

if [[ "${LOO_RESUME:-0}" == "1" && -z "${LOO_REUSE_ANALYSIS_MANIFEST:-}" \
    && -f "${RAW_ANALYSIS_DIR}/analysis_manifest.json" ]]; then
  LOO_REUSE_ANALYSIS_MANIFEST="${RAW_ANALYSIS_DIR}/analysis_manifest.json"
fi
if [[ "${LOO_RESUME:-0}" == "1" && -z "${LOO_REUSE_PREPARED_RUN_ROOT:-}" \
    && -f "${PROJECTION_DIR}/analysis_projection_manifest.json" \
    && -f "${PROJECTION_DIR}/minutes_analysis.jsonl" \
    && -f "${PROMPT_DIR}/prompt_manifest.json" \
    && -f "${SPEC_FILE}" ]]; then
  LOO_REUSE_PREPARED_RUN_ROOT=${RUN_ROOT}
  RESUMING_SAME_RUN=1
fi

if [[ -n "${LOO_REUSE_PREPARED_RUN_ROOT:-}" ]]; then
  require_dir "${LOO_REUSE_PREPARED_RUN_ROOT}"
  PREPARED_RUN_ROOT=$(
    cd "${LOO_REUSE_PREPARED_RUN_ROOT}"
    pwd
  )
  if [[ "${PREPARED_RUN_ROOT}" == "$(cd "${RUN_ROOT}" && pwd)" \
      && "${RESUMING_SAME_RUN}" != "1" ]]; then
    echo "Prepared inputs must come from a different, read-only run root." >&2
    exit 1
  fi
  PROJECTION_DIR="${PREPARED_RUN_ROOT}/analysis_projection"
  PROMPT_DIR="${PREPARED_RUN_ROOT}/prompts"
  SPEC_FILE="${PREPARED_RUN_ROOT}/generation_spec.json"
  require_file "${PROJECTION_DIR}/analysis_projection_manifest.json"
  require_file "${PROJECTION_DIR}/minutes_analysis.jsonl"
  require_file "${PROMPT_DIR}/prompt_manifest.json"
  require_file "${PROMPT_DIR}/run_intervention_roster.json"
  require_dir "${PROMPT_DIR}/exact_delete"
  require_dir "${PROMPT_DIR}/neutral"
  require_file "${SPEC_FILE}"
fi

if [[ "${LOO_RESUME:-0}" == "1" && -z "${PREPARED_RUN_ROOT}" ]]; then
  STALE_PREPARED=()
  [[ -e "${PROMPT_DIR}" ]] && STALE_PREPARED+=("${PROMPT_DIR}")
  [[ -e "${SPEC_FILE}" ]] && STALE_PREPARED+=("${SPEC_FILE}")
  if [[ "${PROJECTION_DIR}" == "${RUN_ROOT}/analysis_projection" \
      && -e "${PROJECTION_DIR}" ]]; then
    STALE_PREPARED+=("${PROJECTION_DIR}")
  fi
  if [[ ${#STALE_PREPARED[@]} -gt 0 ]]; then
    STALE_DIR="${RUN_ROOT}/failed_attempts/prepared-$(date -u +%Y%m%dT%H%M%SZ)"
    mkdir -p "${STALE_DIR}"
    for stale_path in "${STALE_PREPARED[@]}"; do
      mv "${stale_path}" "${STALE_DIR}/"
    done
  fi
fi

if [[ -n "${LOO_REUSE_PROJECTION_MANIFEST:-}" ]]; then
  if [[ -n "${PREPARED_RUN_ROOT}" ]]; then
    echo "Projection-only reuse cannot be combined with prepared-run reuse." >&2
    exit 2
  fi
  REUSED_PROJECTION_MANIFEST=$(
    "${PYTHON_BIN}" - \
      "${LOO_REUSE_PROJECTION_MANIFEST}" \
      "${LOO_REUSE_PROJECTION_OUTPUT_SHA256:-}" \
      "${LOO_REUSE_PROJECTION_OUTPUT:-}" <<'PY'
import json
import sys
from pathlib import Path

from open_r1.provenance import sha256_file

manifest_path = Path(sys.argv[1]).expanduser().resolve()
expected_output_sha = sys.argv[2].strip().lower()
override_output = sys.argv[3].strip()
if not manifest_path.is_file():
    raise SystemExit(f"Reusable projection manifest does not exist: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
if manifest.get("status") not in (None, "complete"):
    raise SystemExit("Reusable projection manifest is not complete")
if manifest.get("generation_performed") is not False:
    raise SystemExit("Reusable projection must declare generation_performed=false")
output_record = manifest.get("output")
if not isinstance(output_record, dict):
    raise SystemExit("Reusable projection manifest lacks output metadata")
if override_output:
    output_path = Path(override_output).expanduser().resolve()
    expected_local_output = (manifest_path.parent / "minutes_analysis.jsonl").resolve()
    if output_path != expected_local_output:
        raise SystemExit(
            "Reusable projection output override must be the frozen copy next "
            "to its manifest"
        )
else:
    output_path = Path(str(output_record.get("path") or "")).expanduser()
    if not output_path.is_absolute():
        output_path = manifest_path.parent / output_path
    output_path = output_path.resolve()
observed_sha = sha256_file(output_path)
if output_record.get("sha256") != observed_sha:
    raise SystemExit("Reusable projection output hash differs from its manifest")
if expected_output_sha and observed_sha != expected_output_sha:
    raise SystemExit(
        "Reusable projection output differs from the externally frozen SHA-256"
    )
inventory = manifest.get("inventory")
if not isinstance(inventory, dict) or (
    inventory.get("meeting_count"),
    inventory.get("indicator_count"),
    inventory.get("row_count"),
) != (13, 26, 338):
    raise SystemExit("Reusable projection does not cover the full pilot 13x26 matrix")
print(manifest_path)
PY
  )
  PROJECTION_DIR=$(dirname "${REUSED_PROJECTION_MANIFEST}")
  require_file "${PROJECTION_DIR}/minutes_analysis.jsonl"
  REUSED_PROJECTION=1
fi

if ! command -v flock >/dev/null 2>&1; then
  echo "Required command is unavailable: flock" >&2
  exit 1
fi
exec 9>"${RUN_ROOT}/.worker.lock"
if ! flock -n 9; then
  echo "Another worker already holds the generation lock: ${RUN_ROOT}" >&2
  exit 1
fi

cleanup_pid() {
  local exit_code=$?
  if [[ -f "${RUN_ROOT}/run.pid" ]]; then
    local recorded_pid
    recorded_pid=$(<"${RUN_ROOT}/run.pid")
    if [[ "${recorded_pid}" == "$$" ]]; then
      rm -f "${RUN_ROOT}/run.pid"
    fi
  fi
  return "${exit_code}"
}
trap cleanup_pid EXIT

cd "${REPO_ROOT}"

CHECKPOINT_RUNTIME_VERIFICATION=$(
  "${PYTHON_BIN}" -m jobs.main.checkpoint_provenance verify-artifact \
    --checkpoint-manifest "${MINUTES_CHECKPOINT_MANIFEST}" \
    --artifact-id "${MINUTES_CHECKPOINT_ARTIFACT_ID}" \
    --model "${MINUTES_MODEL}" \
    --tokenizer "${MINUTES_TOKENIZER}" \
    --expected-manifest-sha256 \
      "${MINUTES_CHECKPOINT_MANIFEST_SHA256}" \
    --expected-manifest-payload-sha256 \
      "${MINUTES_CHECKPOINT_MANIFEST_PAYLOAD_SHA256}" \
    --expected-model-sha256 "${MINUTES_CHECKPOINT_MODEL_SHA256}" \
    --expected-tokenizer-sha256 "${MINUTES_CHECKPOINT_TOKENIZER_SHA256}"
)

echo "run_id=${RUN_ID}"
echo "population=${POPULATION_ID}"
echo "generation_only=true"
echo "training_performed=false"
echo "indicator_input=${INDICATOR_INPUT}"
echo "ledger_manifest=${LEDGER_MANIFEST}"
echo "snapshot_manifest=${SNAPSHOT_MANIFEST}"
echo "source_registry=${SOURCE_REGISTRY}"
echo "canonical_physical_gpu_index=${LOO_CANONICAL_PHYSICAL_GPU_INDEX}"
echo "canonical_physical_gpu_uuid=${LOO_CANONICAL_PHYSICAL_GPU_UUID}"
echo "cuda_visible_devices=${CUDA_VISIBLE_DEVICES}"
echo "tensor_parallel_size=${CONFIGURED_TP_SIZE}"
echo "minutes_token_limit_policy=${MINUTES_TOKEN_LIMIT_POLICY}"
echo "minutes_checkpoint_artifact_id=${MINUTES_CHECKPOINT_ARTIFACT_ID}"
echo "minutes_checkpoint_manifest=${MINUTES_CHECKPOINT_MANIFEST}"
echo "minutes_checkpoint_manifest_sha256=${MINUTES_CHECKPOINT_MANIFEST_SHA256}"
echo "minutes_checkpoint_manifest_payload_sha256=${MINUTES_CHECKPOINT_MANIFEST_PAYLOAD_SHA256}"
echo "minutes_model_sha256=${MINUTES_CHECKPOINT_MODEL_SHA256}"
echo "minutes_tokenizer_sha256=${MINUTES_CHECKPOINT_TOKENIZER_SHA256}"
echo "minutes_checkpoint_runtime_verification=${CHECKPOINT_RUNTIME_VERIFICATION}"
if [[ "${PARTIAL_INTERVENTION_SHARD}" == "1" ]]; then
  if [[ "${SCOPED_EXPERIMENT}" == "1" ]]; then
    echo "intervention_scope=preregistered_six_indicator_experiment"
    echo "experiment_config=${EXPERIMENT_CONFIG}"
    echo "scoped_release_kind=${SCOPED_RELEASE_KIND}"
    echo "standalone_scoped_population_release=true"
    echo "standalone_full_roster_canonical_release=false"
  else
    echo "intervention_scope=partial_intervention_shard"
    echo "standalone_canonical_release=false"
  fi
  echo "intervention_indicators=$(IFS=,; echo "${INTERVENTION_INDICATORS[*]}")"
else
  echo "intervention_scope=full"
fi

REUSED_ANALYSIS_MODEL_ARGS=()
REUSED_ANALYSIS_TOKENIZER_ARGS=()
REUSED_ANALYSIS_SOURCE_ARGS=()
if [[ -n "${LOO_REUSE_ANALYSIS_MANIFEST:-}" ]]; then
  RAW_ANALYSIS_MANIFEST=$(
    "${PYTHON_BIN}" - "${LOO_REUSE_ANALYSIS_MANIFEST}" "${POPULATION_ID}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1]).expanduser().resolve()
if not manifest_path.is_file():
    raise SystemExit(f"Reusable analysis manifest does not exist: {manifest_path}")
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
population_id = manifest.get("inventory", {}).get("population_id")
if population_id != sys.argv[2]:
    raise SystemExit(
        "Reusable analysis population differs from this run: "
        f"{population_id!r} != {sys.argv[2]!r}"
    )
print(manifest_path)
PY
  )
  if [[ -n "${LOO_REUSE_ANALYSIS_OUTPUT:-}" ]]; then
    require_file "${LOO_REUSE_ANALYSIS_OUTPUT}"
    "${PYTHON_BIN}" - \
      "${RAW_ANALYSIS_MANIFEST}" \
      "${LOO_REUSE_ANALYSIS_OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

from open_r1.provenance import sha256_file

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
candidate = Path(sys.argv[2]).expanduser().resolve()
expected = manifest.get("output", {}).get("sha256")
if sha256_file(candidate) != expected:
    raise SystemExit("Reusable analysis output override differs from its manifest")
PY
    RAW_ANALYSIS_OUTPUT=$(cd "$(dirname "${LOO_REUSE_ANALYSIS_OUTPUT}")" && pwd)/$(basename "${LOO_REUSE_ANALYSIS_OUTPUT}")
  else
    RAW_ANALYSIS_OUTPUT=$(
      "${PYTHON_BIN}" - "${RAW_ANALYSIS_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
output_path = Path(str(manifest.get("output", {}).get("path") or ""))
if not output_path.is_absolute():
    output_path = manifest_path.parent / output_path
output_path = output_path.resolve()
if not output_path.is_file():
    raise SystemExit(f"Reusable analysis output does not exist: {output_path}")
print(output_path)
PY
    )
  fi
  while IFS=$'\t' read -r artifact_kind artifact_name artifact_path; do
    case "${artifact_kind}" in
      model)
        REUSED_ANALYSIS_MODEL_ARGS+=(
          --model "${artifact_name}=${artifact_path}"
        )
        ;;
      tokenizer)
        REUSED_ANALYSIS_TOKENIZER_ARGS+=(
          --tokenizer "${artifact_name}=${artifact_path}"
        )
        ;;
      source)
        REUSED_ANALYSIS_SOURCE_ARGS+=(
          --source "${artifact_name}=${artifact_path}"
        )
        ;;
      *)
        echo "Unsupported reusable analysis artifact kind: ${artifact_kind}" >&2
        exit 1
        ;;
    esac
  done < <(
    "${PYTHON_BIN}" - "${RAW_ANALYSIS_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
inputs = manifest.get("inputs", {})
declarations = (
    ("model", "reused_analysis_model", "model"),
    ("tokenizer", "reused_analysis_tokenizer", "tokenizer"),
    ("source", "reused_analysis_input_jsonl", "input_jsonl"),
    ("source", "reused_analysis_ledger_manifest", "ledger_manifest"),
    ("source", "reused_analysis_snapshot_manifest", "snapshot_manifest"),
    ("source", "reused_analysis_source_registry", "source_registry"),
    ("source", "reused_analysis_roster", "roster"),
    ("source", "reused_analysis_population", "population"),
)
for artifact_kind, artifact_name, input_name in declarations:
    record = inputs.get(input_name)
    if not isinstance(record, dict):
        raise SystemExit(
            f"Reusable analysis manifest lacks inputs.{input_name}"
        )
    path = Path(str(record.get("path") or "")).expanduser()
    if not path.is_absolute():
        path = manifest_path.parent / path
    path = path.resolve()
    if not path.exists():
        raise SystemExit(
            f"Reusable analysis input is unavailable: {input_name}={path}"
        )
    print(f"{artifact_kind}\t{artifact_name}\t{path}")
PY
  )
  echo "reusing_raw_analysis_manifest=${RAW_ANALYSIS_MANIFEST}"
  echo "reusing_raw_analysis_output=${RAW_ANALYSIS_OUTPUT}"
else
  "${PYTHON_BIN}" -m jobs.generation.canonical_indicator_analysis \
    --input "${INDICATOR_INPUT}" \
    --roster "${INDICATOR_ROSTER}" \
    --population "${POPULATION_JSON}" \
    --ledger-manifest "${LEDGER_MANIFEST}" \
    --snapshot-manifest "${SNAPSHOT_MANIFEST}" \
    --source-registry "${SOURCE_REGISTRY}" \
    --model "${ANALYSIS_MODEL}" \
    --tokenizer "${ANALYSIS_TOKENIZER}" \
    --output-dir "${RAW_ANALYSIS_DIR}" \
    --seed 20260728 \
    --batch-size "${BATCH_SIZE}" \
    --max-new-tokens "${ANALYSIS_MAX_NEW_TOKENS}" \
    --requested-max-output-tokens \
      "${ANALYSIS_REQUESTED_MAX_OUTPUT_TOKENS}" \
    --max-model-len "${ANALYSIS_MAX_MODEL_LEN}" \
    --max-token-limit-errors "${ANALYSIS_MAX_TOKEN_LIMIT_ERRORS}"
  RAW_ANALYSIS_MANIFEST="${RAW_ANALYSIS_DIR}/analysis_manifest.json"
  RAW_ANALYSIS_OUTPUT="${RAW_ANALYSIS_DIR}/indicator_analysis.jsonl"
fi

if [[ -n "${PREPARED_RUN_ROOT}" ]]; then
  "${PYTHON_BIN}" - \
    "${SPEC_FILE}" \
    "${PROJECTION_DIR}/analysis_projection_manifest.json" \
    "${PROMPT_DIR}/prompt_manifest.json" \
    "${RUN_ID}" \
    "${PHASE}" \
    "${POPULATION_ID}" <<'PY'
import json
import sys
from pathlib import Path

from open_r1.validator.loo_generation_spec import (
    load_and_validate_generation_spec,
)

spec_path = Path(sys.argv[1])
projection_path = Path(sys.argv[2])
prompt_path = Path(sys.argv[3])
run_id, phase, population_id = sys.argv[4:]
spec = load_and_validate_generation_spec(
    spec_path,
    verify_artifact_paths=False,
)
expected_spec = {
    "run_id": run_id,
    "phase": phase,
    "population_id": population_id,
}
mismatches = {
    key: {"expected": value, "observed": spec.get(key)}
    for key, value in expected_spec.items()
    if spec.get(key) != value
}
projection = json.loads(projection_path.read_text(encoding="utf-8"))
prompt = json.loads(prompt_path.read_text(encoding="utf-8"))
if mismatches:
    raise SystemExit(f"Prepared generation spec identity mismatch: {mismatches}")
if projection.get("status") != "complete":
    raise SystemExit("Prepared analysis projection is not complete")
if prompt.get("population_id") != population_id:
    raise SystemExit("Prepared prompt population differs from this run")
print(
    "prepared_generation_inputs=validated "
    f"run_id={run_id} population_id={population_id}"
)
PY
  echo "reusing_prepared_generation_inputs=${PREPARED_RUN_ROOT}"
  echo "prepared_generation_outputs_ignored=${PREPARED_RUN_ROOT}/generations"
else
  if [[ "${REUSED_PROJECTION}" == "1" ]]; then
    "${PYTHON_BIN}" - \
      "${REUSED_PROJECTION_MANIFEST}" \
      "${RAW_ANALYSIS_MANIFEST}" \
      "${RAW_ANALYSIS_OUTPUT}" <<'PY'
import json
import sys
from pathlib import Path

from open_r1.provenance import sha256_file

projection_path = Path(sys.argv[1])
analysis_manifest_path = Path(sys.argv[2]).resolve()
analysis_output_path = Path(sys.argv[3]).resolve()
projection = json.loads(projection_path.read_text(encoding="utf-8"))
expected_manifest_sha = projection.get("input_analysis_manifest", {}).get("sha256")
expected_input_sha = projection.get("input", {}).get("sha256")
if sha256_file(analysis_manifest_path) != expected_manifest_sha:
    raise SystemExit("Reusable projection binds a different analysis manifest")
if sha256_file(analysis_output_path) != expected_input_sha:
    raise SystemExit("Reusable projection binds a different analysis output")
print(
    "reused_projection=validated "
    f"manifest={projection_path} output_sha256={projection['output']['sha256']}"
)
PY
  else
    "${PYTHON_BIN}" -m jobs.generation.project_indicator_analysis \
      --input "${RAW_ANALYSIS_OUTPUT}" \
      --analysis-manifest "${RAW_ANALYSIS_MANIFEST}" \
      --tokenizer "${MINUTES_TOKENIZER}" \
      --output-dir "${PROJECTION_DIR}" \
      --source-field "${PROJECTION_SOURCE_FIELD}" \
      --output-field "${PROJECTION_OUTPUT_FIELD}"
  fi

  "${PYTHON_BIN}" -m jobs.generation.loo_prompt_builder \
    --analysis-blocks "${PROJECTION_DIR}/minutes_analysis.jsonl" \
    --section-roster "${SECTION_ROSTER}" \
    --indicator-roster "${INDICATOR_ROSTER}" \
    --population "${POPULATION_JSON}" \
    --tokenizer "${MINUTES_TOKENIZER}" \
    --output-dir "${PROMPT_DIR}" \
    --analysis-field "${PROJECTION_OUTPUT_FIELD}" \
    --minutes-max-new-tokens "${MINUTES_MAX_NEW_TOKENS}" \
    --minutes-max-model-len "${MINUTES_MAX_MODEL_LEN}" \
    "${EXPERIMENT_CONFIG_ARGS[@]}"

  "${PYTHON_BIN}" -m jobs.generation.build_loo_generation_spec \
    --run-id "${RUN_ID}" \
    --phase "${PHASE}" \
    --population-id "${POPULATION_ID}" \
    --output "${SPEC_FILE}" \
    --model "analysis_model=${ANALYSIS_MODEL}" \
    --model "minutes_model=${MINUTES_MODEL}" \
    "${REUSED_ANALYSIS_MODEL_ARGS[@]}" \
    --tokenizer "analysis_tokenizer=${ANALYSIS_TOKENIZER}" \
    --tokenizer "minutes_tokenizer=${MINUTES_TOKENIZER}" \
    "${REUSED_ANALYSIS_TOKENIZER_ARGS[@]}" \
    --source "indicator_input=${INDICATOR_INPUT}" \
    --source "ledger_manifest=${LEDGER_MANIFEST}" \
    --source "snapshot_manifest=${SNAPSHOT_MANIFEST}" \
    --source "source_registry=${SOURCE_REGISTRY}" \
    --source "population=${POPULATION_JSON}" \
    --source "indicator_roster=${INDICATOR_ROSTER}" \
    --source "section_roster=${SECTION_ROSTER}" \
    --source "analysis_output=${RAW_ANALYSIS_OUTPUT}" \
    --source "analysis_manifest=${RAW_ANALYSIS_MANIFEST}" \
    --source "analysis_projection_output=${PROJECTION_DIR}/minutes_analysis.jsonl" \
    --source "analysis_projection_manifest=${PROJECTION_DIR}/analysis_projection_manifest.json" \
    --source "prompt_manifest=${PROMPT_DIR}/prompt_manifest.json" \
    --source "exact_delete_prompts=${PROMPT_DIR}/exact_delete" \
    --source "neutral_prompts=${PROMPT_DIR}/neutral" \
    "${EXPERIMENT_SPEC_ARGS[@]}" \
    "${REUSED_ANALYSIS_SOURCE_ARGS[@]}" \
    --generation-config "${GENERATION_CONFIG}" \
    --replicate-seed 20260728 \
    --replicate-seed 20260729 \
    --replicate-seed 21260729 \
    --replicate-seed 22260729 \
    --replicate-seed 23260729 \
    --replicate-seed 24260729
fi

SPEC_SHA256=$(sha256sum "${SPEC_FILE}" | awk '{print $1}')

if [[ "${SCOPED_EXPERIMENT}" == "1" && "${SCOPED_RELEASE_KIND}" == "smoke" ]]; then
  mapfile -t SMOKE_SAMPLE_IDS < <(
    "${PYTHON_BIN}" - \
      "${PROMPT_DIR}" \
      "${INTERVENTION_INDICATORS[@]}" <<'PY'
import json
import sys
from pathlib import Path

prompt_dir = Path(sys.argv[1])
allowed = {"None", *sys.argv[2:]}
files = sorted((prompt_dir / "exact_delete").glob("*.jsonl")) + sorted(
    (prompt_dir / "neutral").glob("*.jsonl")
)
if not files:
    raise SystemExit("No prompt artifacts are available for smoke selection")
best = {}
for path in files:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if str(row.get("indicator") or "") not in allowed:
                continue
            section = str(row.get("section_name") or row.get("section_family") or "")
            sample_id = str(row.get("sample_id") or "")
            token_count = row.get("prompt_token_count_chat_template")
            if not section or not sample_id or not isinstance(token_count, int):
                raise SystemExit("Prompt row lacks smoke-selection metadata")
            candidate = (token_count, sample_id)
            if section not in best or candidate > best[section]:
                best[section] = candidate
if len(best) != 3:
    raise SystemExit(f"Expected three section families, found {sorted(best)}")
for section in sorted(best):
    print(best[section][1])
PY
  )
  if [[ ${#SMOKE_SAMPLE_IDS[@]} -ne 3 ]]; then
    echo "Smoke selection did not resolve three rows." >&2
    exit 1
  fi
  for sample_id in "${SMOKE_SAMPLE_IDS[@]}"; do
    SAMPLE_SELECTION_ARGS+=(--include-sample-id "${sample_id}")
  done
  printf 'smoke_sample_id=%s\n' "${SMOKE_SAMPLE_IDS[@]}"
fi

run_minutes_generation() {
  local arm=$1
  local regime=$2
  local strategy=$3
  local prompt_folder=$4
  local intervention_manifest=$5
  local simulation_step=$6
  local base_seed=$7
  local temperature=$8
  local top_p=$9

  "${PYTHON_BIN}" -m jobs.generation.mask_generation \
    --input-folder "${prompt_folder}" \
    --model "${MINUTES_MODEL}" \
    --tokenizer "${MINUTES_TOKENIZER}" \
    --simulation-step "${simulation_step}" \
    --output-dir "${RUN_ROOT}/generations/${arm}_${regime}" \
    --roster-file "${PROMPT_DIR}/run_intervention_roster.json" \
    --batch-size "${BATCH_SIZE}" \
    --seed "${base_seed}" \
    --temperature "${temperature}" \
    --top-p "${top_p}" \
    --max-new-tokens "${MINUTES_MAX_NEW_TOKENS}" \
    --max-model-len "${MINUTES_MAX_MODEL_LEN}" \
    --masking-strategy "${strategy}" \
    --seed-policy sample-id-sha256-v1 \
    --require-normal-finish \
    --token-limit-policy "${MINUTES_TOKEN_LIMIT_POLICY}" \
    --intervention-manifest "${intervention_manifest}" \
    --prompt-manifest "${PROMPT_DIR}/prompt_manifest.json" \
    --generation-spec "${SPEC_FILE}" \
    --generation-spec-sha256 "${SPEC_SHA256}" \
    "${EXPERIMENT_CONFIG_ARGS[@]}" \
    "${SAMPLE_SELECTION_ARGS[@]}" \
    "${INTERVENTION_INDICATOR_ARGS[@]}"
}

run_minutes_generation \
  deletion primary indicator_block_deletion \
  "${PROMPT_DIR}/exact_delete" \
  "${PROMPT_DIR}/intervention_manifest.json" \
  1 20260728 0.0 1.0

run_minutes_generation \
  neutral primary indicator_block_neutral_replacement \
  "${PROMPT_DIR}/neutral" \
  "${PROMPT_DIR}/neutral_intervention_manifest.json" \
  1 20260728 0.0 1.0

run_minutes_generation \
  deletion stochastic indicator_block_deletion \
  "${PROMPT_DIR}/exact_delete" \
  "${PROMPT_DIR}/intervention_manifest.json" \
  5 20260729 0.6 0.9

run_minutes_generation \
  neutral stochastic indicator_block_neutral_replacement \
  "${PROMPT_DIR}/neutral" \
  "${PROMPT_DIR}/neutral_intervention_manifest.json" \
  5 20260729 0.6 0.9

if [[ "${SCOPED_EXPERIMENT}" == "1" ]]; then
  SCOPED_FINALIZER_ARGS=(
    --run-root "${RUN_ROOT}"
    --generation-spec "${SPEC_FILE}"
    --analysis-manifest "${RAW_ANALYSIS_MANIFEST}"
    --projection-manifest "${PROJECTION_DIR}/analysis_projection_manifest.json"
    --prompt-manifest "${PROMPT_DIR}/prompt_manifest.json"
    --experiment-config "${EXPERIMENT_CONFIG}"
    --roster "${INDICATOR_ROSTER}"
    --release-kind "${SCOPED_RELEASE_KIND}"
  )
  if [[ "${SCOPED_RELEASE_KIND}" == "smoke" ]]; then
    SCOPED_MANIFEST="${RUN_ROOT}/smoke_gate_manifest.json"
    for sample_id in "${SMOKE_SAMPLE_IDS[@]}"; do
      SCOPED_FINALIZER_ARGS+=(--smoke-sample-id "${sample_id}")
    done
  else
    SCOPED_MANIFEST="${RUN_ROOT}/scoped_generation_release_manifest.json"
  fi
  if [[ "${SCOPED_RELEASE_KIND}" == "formal" ]]; then
    SCOPED_FINALIZER_ARGS+=(
      --pilot-release "${LOO_PILOT_SCOPED_RELEASE_MANIFEST}"
    )
  fi
  "${PYTHON_BIN}" -m jobs.generation.finalize_loo_scoped_release \
    "${SCOPED_FINALIZER_ARGS[@]}" \
    --output "${SCOPED_MANIFEST}"
  echo "status=complete"
  echo "standalone_scoped_population_release=$([[ "${SCOPED_RELEASE_KIND}" == "smoke" ]] && echo false || echo true)"
  echo "standalone_full_roster_canonical_release=false"
  echo "generation_spec_sha256=${SPEC_SHA256}"
  echo "scoped_manifest=${SCOPED_MANIFEST}"
  echo "output=${RUN_ROOT}"
  exit 0
fi

if [[ "${PARTIAL_INTERVENTION_SHARD}" == "1" ]]; then
  SHARD_FINALIZER_ARGS=()
  for indicator in "${INTERVENTION_INDICATORS[@]}"; do
    SHARD_FINALIZER_ARGS+=(--indicator "${indicator}")
  done
  SHARD_MANIFEST="${RUN_ROOT}/intervention_shard_manifest.json"
  "${PYTHON_BIN}" -m jobs.generation.finalize_loo_intervention_shard \
    --run-root "${RUN_ROOT}" \
    --generation-spec "${SPEC_FILE}" \
    --analysis-manifest "${RAW_ANALYSIS_MANIFEST}" \
    --projection-manifest \
      "${PROJECTION_DIR}/analysis_projection_manifest.json" \
    --prompt-manifest "${PROMPT_DIR}/prompt_manifest.json" \
    "${SHARD_FINALIZER_ARGS[@]}" \
    --output "${SHARD_MANIFEST}"
  SHARD_STATUS=$(
    "${PYTHON_BIN}" - "${SHARD_MANIFEST}" <<'PY'
import json
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
print(manifest["status"])
PY
  )
  echo "status=${SHARD_STATUS}"
  echo "standalone_canonical_release=false"
  echo "generation_spec_sha256=${SPEC_SHA256}"
  echo "intervention_shard_manifest=${SHARD_MANIFEST}"
  echo "output=${RUN_ROOT}"
  exit 0
fi

"${PYTHON_BIN}" -m jobs.generation.finalize_loo_generation \
  --run-root "${RUN_ROOT}" \
  --generation-spec "${SPEC_FILE}" \
  --generation-spec-sha256 "${SPEC_SHA256}" \
  --analysis-manifest "${RAW_ANALYSIS_MANIFEST}" \
  --projection-manifest "${PROJECTION_DIR}/analysis_projection_manifest.json" \
  --prompt-manifest "${PROMPT_DIR}/prompt_manifest.json" \
  --output "${RUN_ROOT}/generation_release_manifest.json"

echo "status=complete"
echo "generation_spec_sha256=${SPEC_SHA256}"
echo "generation_release_manifest=${RUN_ROOT}/generation_release_manifest.json"
echo "output=${RUN_ROOT}"
