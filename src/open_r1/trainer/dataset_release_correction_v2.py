"""Independent runtime binding for the chk4 correction-v2 SFT release.

This module is additive so historical dataset-release source hashes remain
unchanged. It delegates the full data replay to the versioned publisher and
adds the exact system-prompt and selected-cp38 model boundary required by SFT.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jobs.retrain_v2.materialize_chk4_pre2009_correction_v2_release import (
    DATASET_ROLE,
    CorrectionV2ReleaseError,
    verify_runtime_release,
)
from open_r1.trainer.dataset_release import CHK4_STUDENT_SYSTEM_PROMPT


class CorrectionV2RuntimeBindingError(RuntimeError):
    """The correction-v2 runtime binding failed closed."""


def verify_correction_v2_runtime_binding(
    *,
    dataset_dir: str | Path,
    manifest_path: str | Path,
    expected_manifest_sha256: str,
    dataset_role: str,
    system_prompt: str | None,
    model_path: str | Path,
) -> dict[str, Any]:
    if dataset_role != DATASET_ROLE:
        raise CorrectionV2RuntimeBindingError("correction-v2 dataset role drift")
    if system_prompt != CHK4_STUDENT_SYSTEM_PROMPT:
        raise CorrectionV2RuntimeBindingError("correction-v2 system prompt drift")
    try:
        return verify_runtime_release(
            dataset_dir=Path(dataset_dir),
            manifest_path=Path(manifest_path),
            expected_manifest_sha256=expected_manifest_sha256,
            dataset_role=dataset_role,
            model_path=Path(model_path),
        )
    except (CorrectionV2ReleaseError, OSError, ValueError) as exc:
        raise CorrectionV2RuntimeBindingError(
            f"correction-v2 release replay failed: {exc}"
        ) from exc


__all__ = [
    "CorrectionV2RuntimeBindingError",
    "verify_correction_v2_runtime_binding",
]
