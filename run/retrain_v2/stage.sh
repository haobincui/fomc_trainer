#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRAIN_ENV="${FOMC_RETRAIN_TRAIN_ENV:-fomc_trainer}"
DDP_GPUS="${FOMC_RETRAIN_DDP_GPUS:-0,1}"
POLICY_GPU_ID="${FOMC_RETRAIN_POLICY_GPU_ID:-1}"
RUN_MANIFEST=""
STAGE=""
EXECUTE=false

usage() {
  echo "Usage: $0 --run-manifest PATH --stage chk1|chk2|chk3|chk4 [--execute]" >&2
  echo "Without --execute this performs only the immutable preflight." >&2
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-manifest)
      RUN_MANIFEST="${2:-}"
      shift 2
      ;;
    --stage)
      STAGE="${2:-}"
      shift 2
      ;;
    --execute)
      EXECUTE=true
      shift
      ;;
    *)
      usage
      exit 2
      ;;
  esac
done

case "${STAGE}" in
  chk1|chk2|chk3|chk4) ;;
  *)
    usage
    exit 2
    ;;
esac

if [[ -z "${RUN_MANIFEST}" ]]; then
  usage
  exit 2
fi

# The measured A30 profiles and the immutable DAG assign physical roles, not
# merely a device count.  Environment overrides may select the same pinned
# values for orchestration convenience, but they must never remap the judge,
# policy, or DDP ranks to different cards.
if [[ "${DDP_GPUS}" != "0,1" ]]; then
  echo "ERROR: retrain-v2 DDP topology is pinned to physical GPUs 0,1." >&2
  exit 2
fi
if [[ "${POLICY_GPU_ID}" != "1" ]]; then
  echo "ERROR: retrain-v2 chk2 policy is pinned to physical GPU 1." >&2
  exit 2
fi
cd "${ROOT_DIR}"
RUN_MANIFEST="$(realpath "${RUN_MANIFEST}")"
TRAIN_PYTHON="$(conda run -n "${TRAIN_ENV}" python -c 'import sys; print(sys.executable)')"
if [[ -z "${TRAIN_PYTHON}" || "${TRAIN_PYTHON}" == *$'\n'* || ! -x "${TRAIN_PYTHON}" ]]; then
  echo "ERROR: unable to resolve the ${TRAIN_ENV} Python executable." >&2
  exit 2
fi

json_field() {
  local field="$1"
  "${TRAIN_PYTHON}" -c \
    'import json, sys
value = json.load(sys.stdin)
for part in sys.argv[1].split("."):
    value = value[part]
print(json.dumps(value) if isinstance(value, (dict, list, bool)) else value)' \
    "${field}"
}

if [[ "${EXECUTE}" == true ]]; then
  STAGE_LOCK_PATH="$("${TRAIN_PYTHON}" -m jobs.retrain_v2.stage_lock \
    --repo-root "${ROOT_DIR}" \
    --run-manifest "${RUN_MANIFEST}" \
    --stage "${STAGE}")"
  if [[ -z "${STAGE_LOCK_PATH}" || "${STAGE_LOCK_PATH}" == *$'\n'* ]]; then
    echo "ERROR: unable to resolve the canonical ${STAGE} lock path." >&2
    exit 2
  fi
  STAGE_LOCK_DIR="$(dirname "${STAGE_LOCK_PATH}")"
  if [[ -L "${STAGE_LOCK_DIR}" || (-e "${STAGE_LOCK_DIR}" && ! -d "${STAGE_LOCK_DIR}") ]]; then
    echo "ERROR: unsafe retrain-v2 stage lock directory: ${STAGE_LOCK_DIR}" >&2
    exit 2
  fi
  mkdir -p -m 700 "${STAGE_LOCK_DIR}"
  if [[ -L "${STAGE_LOCK_PATH}" || (-e "${STAGE_LOCK_PATH}" && ! -f "${STAGE_LOCK_PATH}") ]]; then
    echo "ERROR: unsafe retrain-v2 stage lock file: ${STAGE_LOCK_PATH}" >&2
    exit 2
  fi
  exec {STAGE_LOCK_FD}>>"${STAGE_LOCK_PATH}"
  if ! flock -n "${STAGE_LOCK_FD}"; then
    echo "ERROR: another launcher already owns the ${STAGE} stage lock." >&2
    exit 2
  fi
  export FOMC_RETRAIN_STAGE_LOCK_FD="${STAGE_LOCK_FD}"
  export FOMC_RETRAIN_STAGE_LOCK_PATH="${STAGE_LOCK_PATH}"
  export FOMC_RETRAIN_STAGE_LOCK_STAGE="${STAGE}"
  "${TRAIN_PYTHON}" -m jobs.retrain_v2.stage_lock \
    --repo-root "${ROOT_DIR}" \
    --run-manifest "${RUN_MANIFEST}" \
    --stage "${STAGE}" \
    --verify-inherited >/dev/null
fi

IMMUTABLE_CACHE_PROBE="$(
  "${TRAIN_PYTHON}" -m jobs.retrain_v2.preflight_cache \
    --repo-root "${ROOT_DIR}" \
    probe \
    --run-manifest "${RUN_MANIFEST}" \
    --stage "${STAGE}" \
    --scope immutable
)"
IMMUTABLE_CACHE_HIT="$(json_field 'hit' <<<"${IMMUTABLE_CACHE_PROBE}")"
IMMUTABLE_SNAPSHOT_SHA256="$(
  json_field 'snapshot_sha256' <<<"${IMMUTABLE_CACHE_PROBE}"
)"
if [[ "${IMMUTABLE_CACHE_HIT}" == "true" ]]; then
  echo "Immutable preflight cache hit for ${STAGE}: ${IMMUTABLE_SNAPSHOT_SHA256}"
else
  echo "Immutable inputs changed; running full parent/data/config verification."
  "${TRAIN_PYTHON}" -m jobs.retrain_v2.dag \
    --repo-root "${ROOT_DIR}" \
    verify-parent \
    --run-manifest "${RUN_MANIFEST}" \
    --stage "${STAGE}"
  "${TRAIN_PYTHON}" -m jobs.retrain_v2.preflight_cache \
    --repo-root "${ROOT_DIR}" \
    record \
    --run-manifest "${RUN_MANIFEST}" \
    --stage "${STAGE}" \
    --scope immutable \
    --snapshot-sha256 "${IMMUTABLE_SNAPSHOT_SHA256}" >/dev/null
fi

CONFIG_PATH="$("${TRAIN_PYTHON}" -m jobs.retrain_v2.dag \
  --repo-root "${ROOT_DIR}" \
  stage-config-path \
  --run-manifest "${RUN_MANIFEST}" \
  --stage "${STAGE}")"

if [[ "${EXECUTE}" == false ]]; then
  echo "Preflight passed. Re-run with --execute to train, merge, and seal ${STAGE}."
  exit 0
fi

LAUNCH_CACHE_PROBE="$(
  "${TRAIN_PYTHON}" -m jobs.retrain_v2.preflight_cache \
    --repo-root "${ROOT_DIR}" \
    probe \
    --run-manifest "${RUN_MANIFEST}" \
    --stage "${STAGE}" \
    --scope launch
)"
LAUNCH_CACHE_HIT="$(json_field 'hit' <<<"${LAUNCH_CACHE_PROBE}")"
LAUNCH_SNAPSHOT_SHA256="$(
  json_field 'snapshot_sha256' <<<"${LAUNCH_CACHE_PROBE}"
)"
if [[ "${LAUNCH_CACHE_HIT}" == "true" ]]; then
  RECOVERY_STATE="$(json_field 'payload.recovery_state' <<<"${LAUNCH_CACHE_PROBE}")"
  if [[ -z "${RECOVERY_STATE}" || "${RECOVERY_STATE}" == "null" ]]; then
    echo "ERROR: launch preflight cache is missing its recovery state." >&2
    exit 2
  fi
  echo "Launch preflight cache hit for ${STAGE}: ${LAUNCH_SNAPSHOT_SHA256}"
else
  RECOVERY_STATE="$(
    "${TRAIN_PYTHON}" -m jobs.retrain_v2.stage_recovery \
      --repo-root "${ROOT_DIR}" \
      --run-manifest "${RUN_MANIFEST}" \
      --stage "${STAGE}" \
      | "${TRAIN_PYTHON}" -c \
        'import json, sys; value = json.load(sys.stdin).get("state"); assert isinstance(value, str) and value; print(value)'
  )"
fi
echo "Recovery state for ${STAGE}: ${RECOVERY_STATE}"

if [[ "${RECOVERY_STATE}" == "sealed_complete" ]]; then
  echo "${STAGE} is already sealed and fully verified."
  exit 0
fi

RUN_ROOT="$(dirname "${RUN_MANIFEST}")"
JUDGE_PRE_ATTESTATION="${RUN_ROOT}/attestations/judge.pre.json"
JUDGE_POST_ATTESTATION="${RUN_ROOT}/attestations/judge.post.json"

verify_judge_pre() {
  "${TRAIN_PYTHON}" -m jobs.retrain_v2.judge_attestation \
    --repo-root "${ROOT_DIR}" \
    --run-manifest "${RUN_MANIFEST}" \
    --verify-pre >/dev/null
}

ensure_judge_pre_before_training() {
  if [[ -e "${JUDGE_POST_ATTESTATION}" || -L "${JUDGE_POST_ATTESTATION}" ]]; then
    echo "ERROR: chk2 post-judge attestation exists before a training receipt." >&2
    exit 2
  fi
  if [[ -e "${JUDGE_PRE_ATTESTATION}" || -L "${JUDGE_PRE_ATTESTATION}" ]]; then
    verify_judge_pre
  else
    "${TRAIN_PYTHON}" -m jobs.retrain_v2.judge_attestation \
      --repo-root "${ROOT_DIR}" \
      --run-manifest "${RUN_MANIFEST}" \
      --phase pre >/dev/null
  fi
}

ensure_judge_post_after_training() {
  verify_judge_pre
  if [[ -e "${JUDGE_POST_ATTESTATION}" || -L "${JUDGE_POST_ATTESTATION}" ]]; then
    "${TRAIN_PYTHON}" -m jobs.retrain_v2.judge_attestation \
      --repo-root "${ROOT_DIR}" \
      --run-manifest "${RUN_MANIFEST}" \
      --verify-pair >/dev/null
  else
    "${TRAIN_PYTHON}" -m jobs.retrain_v2.judge_attestation \
      --repo-root "${ROOT_DIR}" \
      --run-manifest "${RUN_MANIFEST}" \
      --phase post >/dev/null
    "${TRAIN_PYTHON}" -m jobs.retrain_v2.judge_attestation \
      --repo-root "${ROOT_DIR}" \
      --run-manifest "${RUN_MANIFEST}" \
      --verify-pair >/dev/null
  fi
}

run_training_preflight() {
  "${ROOT_DIR}/run/check_retrain_v2_envs.sh" --train --skip-gpu
  "${ROOT_DIR}/run/retrain_v2/resource_gate.sh"
  "${TRAIN_PYTHON}" -m jobs.retrain_v2.token_budget_gate \
    --repo-root "${ROOT_DIR}" \
    --run-manifest "${RUN_MANIFEST}" \
    --stage "${STAGE}"
  case "${STAGE}" in
    chk1|chk3|chk4)
      FOMC_RETRAIN_TRAIN_GPUS="${DDP_GPUS}" \
        "${ROOT_DIR}/run/check_retrain_v2_envs.sh" --train
      ;;
    chk2)
      "${ROOT_DIR}/run/check_retrain_v2_envs.sh" --judge --skip-gpu
      FOMC_RETRAIN_TRAIN_GPUS="${POLICY_GPU_ID}" \
        "${ROOT_DIR}/run/check_retrain_v2_envs.sh" --train --skip-nccl
      ;;
  esac
}

record_launch_preflight_if_needed() {
  if [[ "${LAUNCH_CACHE_HIT}" == "true" ]]; then
    return 0
  fi
  "${TRAIN_PYTHON}" -m jobs.retrain_v2.preflight_cache \
    --repo-root "${ROOT_DIR}" \
    record \
    --run-manifest "${RUN_MANIFEST}" \
    --stage "${STAGE}" \
    --scope launch \
    --snapshot-sha256 "${LAUNCH_SNAPSHOT_SHA256}" \
    --recovery-state "${RECOVERY_STATE}" >/dev/null
  echo "Recorded launch preflight cache for ${STAGE}: ${LAUNCH_SNAPSHOT_SHA256}"
}

run_stage_training() {
  export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
  # GPU0/GPU1 cross NUMA via SYS PCIe on this host. Direct P2P hangs during
  # NCCL initialization; shared-memory transport is the validated fallback.
  export NCCL_P2P_DISABLE=1
  export NCCL_IB_DISABLE=1
  case "${STAGE}" in
    chk1)
      IFS=, read -r -a ddp_gpu_ids <<< "${DDP_GPUS}"
      if ((${#ddp_gpu_ids[@]} != 2)); then
        echo "ERROR: FOMC_RETRAIN_DDP_GPUS must name exactly two GPUs." >&2
        exit 2
      fi
      "${ROOT_DIR}/run/retrain_v2/gpu_gate.sh" "${ddp_gpu_ids[@]}"
      export CUDA_VISIBLE_DEVICES="${DDP_GPUS}"
      timeout --signal=INT --kill-after=15s 600s \
        "${TRAIN_PYTHON}" -m torch.distributed.run \
        --standalone \
        --nproc_per_node 2 \
        --module jobs.retrain_v2.model_smoke \
        --model "${ROOT_DIR}/models/DeepSeek-R1-Distill-Llama-8B" \
        --sequence-length 7168 \
        --attn-implementation flex_attention \
        --distributed
      "${TRAIN_PYTHON}" -m accelerate.commands.launch \
        --num_processes 2 \
        --mixed_precision bf16 \
        --dynamo_backend no \
        -m jobs.train.train_sft \
        --config "${CONFIG_PATH}"
      ;;
    chk3)
      IFS=, read -r -a ddp_gpu_ids <<< "${DDP_GPUS}"
      if ((${#ddp_gpu_ids[@]} != 2)); then
        echo "ERROR: FOMC_RETRAIN_DDP_GPUS must name exactly two GPUs." >&2
        exit 2
      fi
      "${ROOT_DIR}/run/retrain_v2/gpu_gate.sh" "${ddp_gpu_ids[@]}"
      export CUDA_VISIBLE_DEVICES="${DDP_GPUS}"
      "${TRAIN_PYTHON}" -m accelerate.commands.launch \
        --num_processes 2 \
        --mixed_precision bf16 \
        --dynamo_backend no \
        -m jobs.train.train_sft \
        --config "${CONFIG_PATH}"
      ;;
    chk2)
      "${ROOT_DIR}/run/retrain_v2/gpu_gate.sh" "${POLICY_GPU_ID}"
      ensure_judge_pre_before_training
      export CUDA_VISIBLE_DEVICES="${POLICY_GPU_ID}"
      "${TRAIN_PYTHON}" -m jobs.train.train_grpo --config "${CONFIG_PATH}"
      ensure_judge_post_after_training
      ;;
    chk4)
      IFS=, read -r -a ddp_gpu_ids <<< "${DDP_GPUS}"
      if ((${#ddp_gpu_ids[@]} != 2)); then
        echo "ERROR: FOMC_RETRAIN_DDP_GPUS must name exactly two GPUs." >&2
        exit 2
      fi
      "${ROOT_DIR}/run/retrain_v2/gpu_gate.sh" "${ddp_gpu_ids[@]}"
      export CUDA_VISIBLE_DEVICES="${DDP_GPUS}"
      "${TRAIN_PYTHON}" -m accelerate.commands.launch \
        --num_processes 2 \
        --mixed_precision bf16 \
        --dynamo_backend no \
        -m jobs.train.train_grpo \
        --config "${CONFIG_PATH}"
      ;;
  esac
}

record_training_if_needed() {
  if [[ "${STAGE}" == "chk2" ]]; then
    ensure_judge_post_after_training
  fi
  "${TRAIN_PYTHON}" -m jobs.retrain_v2.execution_receipt \
    --repo-root "${ROOT_DIR}" \
    record-training \
    --run-manifest "${RUN_MANIFEST}" \
    --stage "${STAGE}"
}

merge_and_record_if_needed() {
  if [[ "${RECOVERY_STATE}" == "training_receipt_ready_to_merge" ]]; then
    "${ROOT_DIR}/run/check_retrain_v2_envs.sh" --train --skip-gpu
    "${ROOT_DIR}/run/retrain_v2/resource_gate.sh" --merge
  fi
  "${TRAIN_PYTHON}" -m jobs.retrain_v2.merge_adapter \
    --repo-root "${ROOT_DIR}" \
    --run-manifest "${RUN_MANIFEST}" \
    --stage "${STAGE}"
}

case "${RECOVERY_STATE}" in
  fresh_train|checkpoint_resume_ready)
    if [[ "${LAUNCH_CACHE_HIT}" != "true" ]]; then
      run_training_preflight
      record_launch_preflight_if_needed
    fi
    run_stage_training
    record_training_if_needed
    RECOVERY_STATE="training_receipt_ready_to_merge"
    merge_and_record_if_needed
    ;;
  training_complete_ready_to_record)
    record_training_if_needed
    RECOVERY_STATE="training_receipt_ready_to_merge"
    merge_and_record_if_needed
    ;;
  training_receipt_ready_to_merge|attested_merge_ready_to_record)
    merge_and_record_if_needed
    ;;
  merge_receipt_ready_to_seal)
    ;;
  *)
    echo "ERROR: unsupported recovery state: ${RECOVERY_STATE}" >&2
    exit 2
    ;;
esac

"${TRAIN_PYTHON}" -m jobs.retrain_v2.dag \
  --repo-root "${ROOT_DIR}" \
  seal-stage \
  --run-manifest "${RUN_MANIFEST}" \
  --stage "${STAGE}"
