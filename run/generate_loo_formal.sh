#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO_ROOT=$(cd "${SCRIPT_DIR}/.." && pwd)

exec "${SCRIPT_DIR}/_run_canonical_loo_generation.sh" \
  formal_test_13 \
  "${REPO_ROOT}/configs/main/loo_population_formal_test_13.json" \
  "$@"
