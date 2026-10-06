#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 {prepare|smoke|formal|all|validate|status}" >&2
  exit 2
fi

MODE="$1"
case "${MODE}" in
  prepare|smoke|formal|all|validate|status) ;;
  *)
    echo "unsupported mode: ${MODE}" >&2
    exit 2
    ;;
esac

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FOMC_PYTHON_BIN="${FOMC_PYTHON_BIN:-/home/haobin_cui/.conda/envs/fomc_trainer/bin/python}"
VLLM_PYTHON_BIN="${VLLM_PYTHON_BIN:-/home/haobin_cui/.conda/envs/vllm_env/bin/python}"
PREPARER_MODULE="jobs.eval.prepare_chk3_core8_loo_vllm_k10_increment"
RUNNER_MODULE="jobs.eval.eval_chk3_core8_loo_vllm_k10_increment_dual_dp1"
ORCHESTRATOR_MODULE="jobs.eval.orchestrate_chk3_core8_loo_vllm_k10_increment_dual_dp1"

RUN_ROOT="${REPO_ROOT}/output/evaluation/main/chk3_cp318_core8_loo_vllm_k10_n128_1993_2008_20260824_v1"
PREPARATION_ROOT="${RUN_ROOT}/preparation_increment_k6_k10_v1"
COHORT="${PREPARATION_ROOT}/cohort_n2176_increment_k6_k10.v1.json"
POLICY="${PREPARATION_ROOT}/execution_policy.increment-k6-k10.v1.json"
LEDGER="${PREPARATION_ROOT}/prompt_token_ledger_n2176.increment-k6-k10.v1.jsonl"
PREPARATION_MANIFEST="${PREPARATION_ROOT}/preparation.increment-k6-k10.v1.json"
SMOKE_ROOT="${RUN_ROOT}/generation_smoke_increment_k6_k10_two_meetings_v1"
FORMAL_ROOT="${RUN_ROOT}/generation_formal_increment_k6_k10_n2176_v1"
SMOKE_MANIFEST="${SMOKE_ROOT}/chk3/manifest.json"
FORMAL_MANIFEST="${FORMAL_ROOT}/chk3/manifest.json"
PIPELINE_LOG="${RUN_ROOT}/pipeline_increment_k6_k10_dual_dp1_20260824.log"
PIPELINE_LOCK="/tmp/fomc_trainer_chk3_core8_loo_vllm_k10_increment_dual_dp1_pipeline.lock"
MAX_NUM_SEQS=16

# These values are populated only after the sealed preparation has been
# independently validated below.  They are intentionally not unbound
# placeholders baked into this launcher before the create-only preparation.
COHORT_SHA256=""
POLICY_SHA256=""
LEDGER_SHA256=""
PREPARATION_SHA256=""

GPU_WAIT_TIMEOUT_SECONDS="${GPU_WAIT_TIMEOUT_SECONDS:-172800}"
GPU_POLL_SECONDS="${GPU_POLL_SECONDS:-30}"
READY_TIMEOUT_SECONDS="${READY_TIMEOUT_SECONDS:-1800}"

mkdir -p "${RUN_ROOT}"
exec 9>"${PIPELINE_LOCK}"
if ! flock -n 9; then
  echo "another LOO K=6--10 incremental pipeline holds ${PIPELINE_LOCK}" >&2
  exit 2
fi
exec > >(tee -a "${PIPELINE_LOG}") 2>&1

timestamp() { date -u +%Y-%m-%dT%H:%M:%SZ; }
fail() { echo "[$(timestamp)] ERROR: $*" >&2; exit 1; }

export CUDA_DEVICE_ORDER=PCI_BUS_ID
unset CUDA_VISIBLE_DEVICES || true
unset VLLM_DP_MASTER_IP VLLM_DP_MASTER_PORT || true
unset VLLM_DP_RANK VLLM_DP_RANK_LOCAL VLLM_DP_SIZE || true
unset MASTER_ADDR MASTER_PORT WORLD_SIZE RANK LOCAL_RANK LOCAL_WORLD_SIZE || true
unset GROUP_RANK ROLE_RANK || true
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}"
export VLLM_USE_V1=1
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn

cd "${REPO_ROOT}"
[[ -x "${FOMC_PYTHON_BIN}" ]] || fail "missing FOMC Python"
[[ -x "${VLLM_PYTHON_BIN}" ]] || fail "missing vLLM Python"
command -v flock >/dev/null 2>&1 || fail "flock is required"
command -v jq >/dev/null 2>&1 || fail "jq is required"

validate_preparation() {
  local metadata
  metadata="$("${FOMC_PYTHON_BIN}" - \
    "${PREPARATION_ROOT}" \
    "${COHORT}" \
    "${POLICY}" \
    "${LEDGER}" \
    "${PREPARATION_MANIFEST}" <<'PY'
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

from jobs.eval import eval_chk3_core8_loo_vllm_k10_increment_dual_dp1 as runner
from jobs.eval import prepare_chk3_core8_loo_vllm_k10_increment as preparation
from open_r1.validator.loo_generation_spec import validate_manifest_integrity


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"JSON artifact is not an object: {path}")
    validate_manifest_integrity(value)
    return value


root, cohort_path, policy_path, ledger_path, preparation_path = (
    Path(value).expanduser() for value in sys.argv[1:]
)
expected_paths = {
    "cohort": (cohort_path, preparation.COHORT_FILENAME),
    "execution_policy": (policy_path, preparation.EXECUTION_POLICY_FILENAME),
    "token_ledger": (ledger_path, preparation.LEDGER_FILENAME),
}
if root.is_symlink() or not root.is_dir():
    raise RuntimeError(f"missing or unsafe preparation directory: {root}")
if preparation_path.name != preparation.PREPARATION_FILENAME:
    raise RuntimeError("launcher/preparer preparation filename drift")
for key, (path, expected_name) in expected_paths.items():
    if path.name != expected_name:
        raise RuntimeError(f"launcher/preparer {key} filename drift")
for path in (cohort_path, policy_path, ledger_path, preparation_path):
    if path.is_symlink() or not path.is_file() or path.parent.resolve() != root.resolve():
        raise RuntimeError(f"missing or unsafe preparation artifact: {path}")

preparation_manifest = load_json(preparation_path)
if (
    preparation_manifest.get("status") != "complete"
    or preparation_manifest.get("evaluation_id") != runner.EVALUATION_ID
):
    raise RuntimeError("preparation terminal identity/status drift")

bindings = {}
for key, (path, _expected_name) in expected_paths.items():
    binding = preparation_manifest.get(key)
    observed = {
        "path": str(path.resolve()),
        "sha256": sha256_file(path),
        "bytes": path.stat().st_size,
    }
    if not isinstance(binding, dict):
        raise RuntimeError(f"preparation lacks {key} binding")
    if any(binding.get(field) != observed[field] for field in observed):
        raise RuntimeError(f"preparation {key} binding drift")
    bindings[key] = observed

cohort = load_json(cohort_path)
policy = load_json(policy_path)
if cohort.get("execution_policy", {}).get("sha256") != bindings["execution_policy"]["sha256"]:
    raise RuntimeError("cohort execution-policy binding drift")
loaded_policy = runner.load_execution_policy(policy_path)
if loaded_policy.get("receipt_binding", {}).get("sha256") != bindings["execution_policy"]["sha256"]:
    raise RuntimeError("runner execution-policy validation drift")

cohort_sha256 = bindings["cohort"]["sha256"]
_cohort, ledger, sample_manifest, _bound_rows = runner.load_cohort(
    cohort_path, cohort_sha256
)
cases = runner.canonical_cases(
    samples=sample_manifest["samples"], ledger=ledger, smoke=False
)
shard_sizes = [len(runner.shard_cases(cases, shard)) for shard in (0, 1)]
if len(cases) != 10880 or shard_sizes != [5440, 5440]:
    raise RuntimeError(
        f"incremental case/shard coverage drift: cases={len(cases)} shards={shard_sizes}"
    )

print(
    json.dumps(
        {
            "cohort_sha256": cohort_sha256,
            "policy_sha256": bindings["execution_policy"]["sha256"],
            "ledger_sha256": bindings["token_ledger"]["sha256"],
            "preparation_sha256": sha256_file(preparation_path),
            "cases": len(cases),
            "shard_cases": shard_sizes,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
)
PY
  )" || fail "sealed incremental preparation validation failed"

  COHORT_SHA256="$(jq -er '.cohort_sha256' <<<"${metadata}")" || \
    fail "could not read validated cohort SHA-256"
  POLICY_SHA256="$(jq -er '.policy_sha256' <<<"${metadata}")" || \
    fail "could not read validated policy SHA-256"
  LEDGER_SHA256="$(jq -er '.ledger_sha256' <<<"${metadata}")" || \
    fail "could not read validated ledger SHA-256"
  PREPARATION_SHA256="$(jq -er '.preparation_sha256' <<<"${metadata}")" || \
    fail "could not read validated preparation SHA-256"
  echo "sealed_preparation_valid cases=10880 shard0=5440 shard1=5440 cohort_sha256=${COHORT_SHA256}"
}

prepare_or_validate() {
  if [[ ! -e "${PREPARATION_ROOT}" && ! -L "${PREPARATION_ROOT}" ]]; then
    "${FOMC_PYTHON_BIN}" -u -m "${PREPARER_MODULE}" \
      --output-dir "${PREPARATION_ROOT}"
  elif [[ -L "${PREPARATION_ROOT}" || ! -d "${PREPARATION_ROOT}" ]]; then
    fail "unsafe preparation root: ${PREPARATION_ROOT}"
  fi
  validate_preparation
}

run_model() {
  local root="$1"
  local scope="$2"
  local resume="$3"
  local args=()
  if [[ "${resume}" == "true" ]]; then
    args+=(--allow-resume)
  fi
  if [[ "${scope}" == "formal_merged_panel" ]]; then
    args+=(--formal-authorization "${SMOKE_MANIFEST}")
  fi
  "${FOMC_PYTHON_BIN}" -u -m "${ORCHESTRATOR_MODULE}" \
    --python-bin "${VLLM_PYTHON_BIN}" \
    --model-id chk3 \
    --cohort "${COHORT}" \
    --cohort-sha256 "${COHORT_SHA256}" \
    --model-root "${root}/chk3" \
    --scope "${scope}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --gpu-wait-timeout-seconds "${GPU_WAIT_TIMEOUT_SECONDS}" \
    --gpu-poll-seconds "${GPU_POLL_SECONDS}" \
    --ready-timeout-seconds "${READY_TIMEOUT_SECONDS}" \
    "${args[@]}"
}

validate_run() {
  local manifest="$1"
  local scope="$2"
  [[ -f "${manifest}" && ! -L "${manifest}" ]] || \
    fail "sealed ${scope} manifest is missing: ${manifest}"
  "${FOMC_PYTHON_BIN}" -u -m "${RUNNER_MODULE}" validate-run \
    --manifest "${manifest}" \
    --cohort "${COHORT}" \
    --cohort-sha256 "${COHORT_SHA256}" \
    --model-id chk3 \
    --scope "${scope}" \
    --max-num-seqs "${MAX_NUM_SEQS}"
}

run_smoke() {
  # Smoke is create-only.  A partial smoke cannot authorize formal work and
  # must fail closed rather than being silently resumed.
  run_model "${SMOKE_ROOT}" infrastructure_smoke false
  validate_run "${SMOKE_MANIFEST}" infrastructure_smoke
  jq -e '
    .coverage.cases == 170
    and .coverage.shard_cases == {"shard0":85,"shard1":85}
    and .coverage.input_truncation_cases == 0
    and .coverage.absolute_indexes_exact == true
    and .coverage.overlaps == 0
  ' "${SMOKE_MANIFEST}" >/dev/null || fail "LOO K=6--10 smoke gates failed"
}

run_formal() {
  validate_run "${SMOKE_MANIFEST}" infrastructure_smoke
  # Formal generation alone may resume durable, deeply validated shard WALs.
  run_model "${FORMAL_ROOT}" formal_merged_panel true
  validate_run "${FORMAL_MANIFEST}" formal_merged_panel
  jq -e '
    .coverage.cases == 10880
    and .coverage.shard_cases == {"shard0":5440,"shard1":5440}
    and .coverage.input_truncation_cases == 0
    and .coverage.absolute_indexes_exact == true
    and .coverage.overlaps == 0
  ' "${FORMAL_MANIFEST}" >/dev/null || fail "LOO K=6--10 formal coverage failed"
}

show_status() {
  echo "preparation_root=${PREPARATION_ROOT}"
  echo "cohort_sha256=${COHORT_SHA256}"
  echo "policy_sha256=${POLICY_SHA256}"
  echo "ledger_sha256=${LEDGER_SHA256}"
  echo "preparation_sha256=${PREPARATION_SHA256}"
  for root in "${SMOKE_ROOT}/chk3" "${FORMAL_ROOT}/chk3"; do
    echo "root=${root}"
    if [[ -f "${root}/manifest.json" ]]; then
      jq '{status,evaluation_scope,coverage,runtime}' "${root}/manifest.json"
    else
      for shard in 0 1; do
        local state="${root}/shard${shard}/state.progress.v1.json"
        if [[ -f "${state}" ]]; then
          jq '{status,shard_id,completed_cases,expected_cases,resume_count,updated_at_utc}' "${state}"
        else
          echo "shard${shard}: not_started"
        fi
      done
    fi
  done
}

echo "[$(timestamp)] pipeline_start mode=${MODE} target=incremental_k6_k10"
prepare_or_validate

case "${MODE}" in
  prepare) ;;
  smoke) run_smoke ;;
  formal) run_formal ;;
  all)
    run_smoke
    run_formal
    ;;
  validate)
    validate_run "${SMOKE_MANIFEST}" infrastructure_smoke
    validate_run "${FORMAL_MANIFEST}" formal_merged_panel
    ;;
  status) show_status ;;
esac

echo "[$(timestamp)] pipeline_complete mode=${MODE} target=incremental_k6_k10"

