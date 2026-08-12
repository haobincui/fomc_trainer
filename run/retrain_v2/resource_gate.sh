#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODE="stage"
if (($# > 1)); then
  echo "Usage: $0 [--merge]" >&2
  exit 2
fi
if (($# == 1)); then
  if [[ "$1" != "--merge" ]]; then
    echo "Usage: $0 [--merge]" >&2
    exit 2
  fi
  MODE="merge"
fi

if [[ "${MODE}" == "merge" ]]; then
  BASELINE_MIN_FREE_GIB=150
  MIN_FREE_GIB="${FOMC_RETRAIN_MIN_MERGE_DISK_GIB:-${BASELINE_MIN_FREE_GIB}}"
  OVERRIDE_NAME="FOMC_RETRAIN_MIN_MERGE_DISK_GIB"
else
  BASELINE_MIN_FREE_GIB=250
  MIN_FREE_GIB="${FOMC_RETRAIN_MIN_DISK_GIB:-${BASELINE_MIN_FREE_GIB}}"
  OVERRIDE_NAME="FOMC_RETRAIN_MIN_DISK_GIB"
fi

if [[ ! "${MIN_FREE_GIB}" =~ ^[0-9]{1,6}$ ]]; then
  echo "ERROR: ${OVERRIDE_NAME} must be an integer in [${BASELINE_MIN_FREE_GIB}, 999999]." >&2
  exit 2
fi
min_free_gib_value=$((10#${MIN_FREE_GIB}))
if ((min_free_gib_value < BASELINE_MIN_FREE_GIB)); then
  echo "ERROR: ${OVERRIDE_NAME} cannot weaken the ${BASELINE_MIN_FREE_GIB} GiB baseline." >&2
  exit 2
fi

available_kib="$(df -Pk "${ROOT_DIR}" | awk 'NR == 2 {print $4}')"
if [[ ! "${available_kib}" =~ ^[0-9]{1,18}$ ]]; then
  echo "ERROR: unable to determine free disk space for ${ROOT_DIR}." >&2
  exit 2
fi
available_kib_value=$((10#${available_kib}))
required_kib=$((min_free_gib_value * 1024 * 1024))
if ((available_kib_value < required_kib)); then
  available_gib=$((available_kib_value / 1024 / 1024))
  echo "ERROR: only ${available_gib} GiB is free; ${min_free_gib_value} GiB is required." >&2
  exit 2
fi

available_gib=$((available_kib_value / 1024 / 1024))
echo "Resource gate passed: ${available_gib} GiB free (minimum ${min_free_gib_value} GiB)."
