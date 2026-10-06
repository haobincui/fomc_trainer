#!/usr/bin/env bash
set -euo pipefail

readonly TASK_REPO="/home/haobin_cui/research_files_space_2/fomc_trainer"
readonly TASK_PYTHON="/home/haobin_cui/.conda/envs/fomc_trainer/bin/python"
readonly TASK_ROOT="output/evaluation/retrain_v2/chk4_external_deterministic_core8_n19_n12_v4_20260824"
readonly TASK_MANIFEST_SHA="689706912c74f800a7e6ad02aa62a8b0616c19b874bd845c1adc74b7dc6d2d22"
readonly TASK_LOG="${TASK_REPO}/${TASK_ROOT}/pipeline.log"
readonly TASK_BLOCKING_PID="${WAIT_FOR_PID:-}"

cd "${TASK_REPO}"
exec > >(tee -a "${TASK_LOG}") 2>&1

export PYTHONPATH=src
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export CUBLAS_WORKSPACE_CONFIG=:4096:8

echo "[$(date -u +%FT%TZ)] external Core8 evaluation queued"

if [[ -n "${TASK_BLOCKING_PID}" ]]; then
    while kill -0 "${TASK_BLOCKING_PID}" 2>/dev/null; do
        if [[ "$(ps -o stat= -p "${TASK_BLOCKING_PID}" 2>/dev/null | tr -d '[:space:]' | cut -c1)" == "Z" ]]; then
            break
        fi
        sleep 30
    done
fi

while [[ "$(nvidia-smi --id=0 --query-compute-apps=used_memory --format=csv,noheader,nounits 2>/dev/null | awk '{sum += $1} END {print sum + 0}')" -gt 1024 ]]; do
    sleep 30
done

echo "[$(date -u +%FT%TZ)] GPU 0 available; starting 93 fresh generations"
CUDA_VISIBLE_DEVICES=0 "${TASK_PYTHON}" \
    -m jobs.retrain_v2.evaluate_chk4_external_core8_panels run \
    --output-root "${TASK_ROOT}" \
    --manifest-sha256 "${TASK_MANIFEST_SHA}"

echo "[$(date -u +%FT%TZ)] generation complete; scoring separate panels"
"${TASK_PYTHON}" \
    -m jobs.retrain_v2.evaluate_chk4_external_core8_panels score \
    --output-root "${TASK_ROOT}" \
    --manifest-sha256 "${TASK_MANIFEST_SHA}"

echo "[$(date -u +%FT%TZ)] external Core8 evaluation complete"
