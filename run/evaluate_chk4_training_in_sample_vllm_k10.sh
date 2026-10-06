#!/usr/bin/env bash
set -euo pipefail

repo_root=/home/haobin_cui/research_files_space_2/fomc_trainer
python_bin=/home/haobin_cui/.conda/envs/fomc_trainer/bin/python
module=jobs.retrain_v2.evaluate_chk4_training_in_sample_stochastic_vllm_k10
output_root=${CHK4_TRAINING_DIAGNOSTIC_ROOT:-${repo_root}/output/evaluation/retrain_v2/chk4_training_in_sample_stochastic_n211_vllm_t06_p09_k10_v1_20260826}
audit_statement=${CHK4_TRAINING_DIAGNOSTIC_AUDIT_STATEMENT:-}

export PYTHONPATH=${repo_root}/src:${repo_root}
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

exec 9>/tmp/fomc_trainer_chk4_training_in_sample_k10_pipeline.lock
flock -n 9 || {
    echo "training-meeting diagnostic pipeline lock is already held" >&2
    exit 1
}

cd "${repo_root}"

if [[ ! -f "${output_root}/evaluation_manifest.json" ]]; then
    "${python_bin}" -m "${module}" prepare --output-root "${output_root}"
fi

manifest=${output_root}/evaluation_manifest.json
manifest_sha=$(sha256sum "${manifest}" | awk '{print $1}')

"${python_bin}" -m "${module}" smoke \
    --manifest "${manifest}" \
    --manifest-sha256 "${manifest_sha}" \
    --resume

authorization=${output_root}/formal_generation_authorization.json
if [[ ! -f "${authorization}" ]]; then
    if [[ ${#audit_statement} -lt 20 ]]; then
        echo "Set CHK4_TRAINING_DIAGNOSTIC_AUDIT_STATEMENT to a substantive independent-audit statement before formal generation." >&2
        exit 2
    fi
    "${python_bin}" -m "${module}" authorize \
        --manifest "${manifest}" \
        --manifest-sha256 "${manifest_sha}" \
        --audit-statement "${audit_statement}"
fi

authorization_sha=$(sha256sum "${authorization}" | awk '{print $1}')

"${python_bin}" -m "${module}" run \
    --manifest "${manifest}" \
    --manifest-sha256 "${manifest_sha}" \
    --authorization "${authorization}" \
    --authorization-sha256 "${authorization_sha}" \
    --resume

if [[ ! -f "${output_root}/report/report_receipt.json" ]]; then
    "${python_bin}" -m "${module}" score \
        --manifest "${manifest}" \
        --manifest-sha256 "${manifest_sha}" \
        --authorization "${authorization}" \
        --authorization-sha256 "${authorization_sha}"
fi

"${python_bin}" -m "${module}" status --output-root "${output_root}"

