#!/usr/bin/env bash

# This file is sourced by canonical LOO launchers before any Python/CUDA import.
# Resolve and audit physical GPU 1 before exposing it numerically. The vLLM
# version used by this project parses CUDA_VISIBLE_DEVICES entries as integers,
# so UUID selectors are not compatible even though CUDA itself accepts them.
# CUDA_DEVICE_ORDER plus the launcher preflight independently verify that
# logical cuda:0 has the UUID reported by nvidia-smi for physical index 1.

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Canonical LOO requires nvidia-smi to pin physical GPU 1." >&2
  return 1 2>/dev/null || exit 1
fi

LOO_CANONICAL_PHYSICAL_GPU_INDEX=1
LOO_CANONICAL_PHYSICAL_GPU_UUID=$(
  nvidia-smi \
    --id="${LOO_CANONICAL_PHYSICAL_GPU_INDEX}" \
    --query-gpu=uuid \
    --format=csv,noheader,nounits
)
LOO_CANONICAL_PHYSICAL_GPU_UUID=${LOO_CANONICAL_PHYSICAL_GPU_UUID//[[:space:]]/}
if [[ ! "${LOO_CANONICAL_PHYSICAL_GPU_UUID}" =~ ^GPU-[A-Fa-f0-9-]+$ ]]; then
  echo "Could not resolve a unique UUID for physical GPU 1." >&2
  return 1 2>/dev/null || exit 1
fi

export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="${LOO_CANONICAL_PHYSICAL_GPU_INDEX}"
export LOO_CANONICAL_PHYSICAL_GPU_INDEX
export LOO_CANONICAL_PHYSICAL_GPU_UUID
export LOO_CANONICAL_VISIBLE_DEVICE_COUNT=1
