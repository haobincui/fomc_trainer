#!/usr/bin/env bash
set -euo pipefail

# Wait for the durable GPU0 generation suite, then validate, score, bootstrap,
# and render the portable report. Generation is owned by a separate tmux
# session so this follow-up can fail closed if that session exits without a
# sealed formal manifest.

bootstrap_repo_root="/home/haobin_cui/research_files_space_2/fomc_trainer"
bootstrap_python="/home/haobin_cui/.conda/envs/fomc_trainer/bin/python"
bootstrap_generation_session="chk3_bootstrap_n190_k5_20260811"
bootstrap_run_root="${bootstrap_repo_root}/output/evaluation/main/chk3_native_stochastic_bootstrap_cp318_n190_k5_20260811_v1"
bootstrap_generation_root="${bootstrap_run_root}/formal_generation_v1"
bootstrap_scoring_root="${bootstrap_run_root}/six_metric_bootstrap_v1"
bootstrap_report_root="${bootstrap_run_root}/report_v1"
bootstrap_sample_manifest="${bootstrap_repo_root}/docs/summary/20260811T162400Z/chk3_native_stochastic_bootstrap_cp318/samples_n190.json"
bootstrap_sample_sha256="314623a5e361e241da0f4602d993fc84919bb76606404b42b90b322e7e3a2ec9"
bootstrap_selection_manifest="${bootstrap_repo_root}/docs/summary/20260811T003000Z/chk3_native_analysis_to_minutes_cp250_eval/samples_n12.json"
bootstrap_selection_sha256="371d29601e343acf98cac6173663d842ead9acaa4a454991d512b3f00bf5ee77"
bootstrap_greedy_anchor="${bootstrap_repo_root}/output/evaluation/main/chk3_checkpoint_selection_cp318_20260811_v1/six_metrics/chk0_chk1_chk3_cp318_n12_cpu_v1.json"
bootstrap_greedy_anchor_sha256="acaea49f4865f080623bf42c2397abe4b18fc84411e7800463e08bbc0ae6b84a"
bootstrap_semantic_manifest="${bootstrap_repo_root}/configs/main/checkpoint_eval_semantic_models.json"

cd "${bootstrap_repo_root}"

while tmux has-session -t "${bootstrap_generation_session}" 2>/dev/null; do
  sleep 60
done

if [[ ! -f "${bootstrap_generation_root}/manifest.json" ]]; then
  echo "generation session ended without a sealed formal suite manifest" >&2
  exit 1
fi
if [[ -e "${bootstrap_scoring_root}" || -e "${bootstrap_report_root}" ]]; then
  echo "follow-up output already exists; refusing to overwrite" >&2
  exit 1
fi

bootstrap_suite_sha256="$({ sha256sum "${bootstrap_generation_root}/manifest.json"; } | awk '{print $1}')"

PYTHONPATH=src:. "${bootstrap_python}" -m jobs.eval.eval_chk3_stochastic_bootstrap_generation validate-suite \
  --manifest "${bootstrap_generation_root}/manifest.json" \
  --sample-manifest "${bootstrap_sample_manifest}" \
  --sample-manifest-sha256 "${bootstrap_sample_sha256}" \
  --scope formal_full_test

env \
  CUDA_DEVICE_ORDER=PCI_BUS_ID \
  CUDA_VISIBLE_DEVICES=0 \
  TOKENIZERS_PARALLELISM=false \
  PYTHONUNBUFFERED=1 \
  PYTHONPATH=src:. \
  "${bootstrap_python}" -u -m jobs.eval.score_chk3_stochastic_bootstrap \
  --generation-suite-manifest "${bootstrap_generation_root}/manifest.json" \
  --generation-suite-manifest-sha256 "${bootstrap_suite_sha256}" \
  --chk0-manifest "${bootstrap_generation_root}/chk0/manifest.json" \
  --chk1-manifest "${bootstrap_generation_root}/chk1/manifest.json" \
  --chk3-manifest "${bootstrap_generation_root}/chk3/manifest.json" \
  --sample-manifest "${bootstrap_sample_manifest}" \
  --sample-manifest-sha256 "${bootstrap_sample_sha256}" \
  --selection-sample-manifest "${bootstrap_selection_manifest}" \
  --selection-sample-manifest-sha256 "${bootstrap_selection_sha256}" \
  --greedy-anchor-scorecard "${bootstrap_greedy_anchor}" \
  --greedy-anchor-scorecard-sha256 "${bootstrap_greedy_anchor_sha256}" \
  --semantic-manifest "${bootstrap_semantic_manifest}" \
  --output-dir "${bootstrap_scoring_root}" \
  --semantic-device cuda:0 \
  --semantic-batch-size 8 \
  --gpu-wait-timeout-seconds 172800 \
  --gpu-poll-seconds 30 \
  --bootstrap-draws 2000 \
  --bootstrap-seed 20260812

bootstrap_score_sha256="$({ sha256sum "${bootstrap_scoring_root}/manifest.json"; } | awk '{print $1}')"

PYTHONPATH=src:. "${bootstrap_python}" -m jobs.eval.render_chk3_stochastic_bootstrap_report \
  --score-manifest "${bootstrap_scoring_root}/manifest.json" \
  --score-manifest-sha256 "${bootstrap_score_sha256}" \
  --output-dir "${bootstrap_report_root}" \
  --title "CHK0 / CHK1 / CHK3 cp318 随机生成 Bootstrap 评测"

echo "bootstrap follow-up complete"
echo "score_manifest=${bootstrap_scoring_root}/manifest.json"
echo "report_manifest=${bootstrap_report_root}/manifest.json"
echo "report_html=${bootstrap_report_root}/report.html"
