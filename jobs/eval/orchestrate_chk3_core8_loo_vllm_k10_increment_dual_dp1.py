"""Supervise the two independent GPU workers for the CHK3 LOO K=6--10 increment.

This is a new orchestration profile.  It reuses the already validated
dual-DP1 process, GPU, barrier, WAL, and resume machinery without changing
the frozen K=1 or K=5 artifacts.  Output artifacts remain separate, while the
profile reuses the canonical cross-pipeline physical-GPU lock namespace and
delegates every data- and seed-sensitive check to the incremental runner.
"""

from __future__ import annotations

import contextlib
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from jobs.eval import (
    eval_chk3_core8_loo_vllm_k10_increment_dual_dp1 as runner,
)
from jobs.eval import (
    orchestrate_chk3_beta_core8_merged_vllm_k5_dual_dp1_stochastic_schedule_v2 as shared,
)


RUNNER_MODULE = "jobs.eval.eval_chk3_core8_loo_vllm_k10_increment_dual_dp1"
EVALUATION_ID = runner.EVALUATION_ID
READY_SCHEMA = runner.READY_SCHEMA
GO_SCHEMA = runner.GO_SCHEMA
DONE_SCHEMA = runner.DONE_SCHEMA
TIMING_SCHEMA = runner.ORCHESTRATOR_TIMING_SCHEMA
MODEL_ORDER = runner.MODEL_ORDER
SCOPES = ("infrastructure_smoke", "formal_merged_panel")
MAX_NUM_SEQS = runner.ALLOWED_MAX_NUM_SEQS
GPU_LOCKS = tuple(
    Path(runner.GPU_LOCK_PATH_TEMPLATE.format(index=index))
    for index in range(2)
)


class LooK10IncrementOrchestrationError(shared.DualDp1OrchestrationError):
    """The K=6--10 dual-worker orchestration contract failed."""


class ExecutionPolicyError(LooK10IncrementOrchestrationError):
    """The K=6--10 execution policy could not be validated."""


def _runner_error_type() -> type[BaseException]:
    """Return the incremental runner's public contract error type."""

    error_type = getattr(runner, "Core8LooVllmK10IncrementError", None)
    if not isinstance(error_type, type) or not issubclass(error_type, BaseException):
        raise TypeError("incremental runner does not expose its contract error type")
    return error_type


def _load_policy(path: Path) -> dict[str, Any]:
    try:
        return runner.load_execution_policy(path)
    except _runner_error_type() as exc:
        raise ExecutionPolicyError(str(exc)) from exc


_POLICY_ADAPTER = SimpleNamespace(
    DEFAULT_RECEIPT=runner.DEFAULT_EXECUTION_POLICY,
    StochasticScheduleAmendmentError=ExecutionPolicyError,
    load_and_validate_receipt=_load_policy,
)


def _formal_authorization_evidence(
    *,
    formal_authorization: Path | None,
    cohort: Path,
    cohort_sha256: str,
    scope: str,
    max_num_seqs: int,
) -> dict[str, Any] | None:
    try:
        return runner._formal_authorization_for_scope(
            scope=scope,
            authorization_path=formal_authorization,
            cohort_path=cohort,
            cohort_sha256=cohort_sha256,
            max_num_seqs=max_num_seqs,
        )
    except _runner_error_type() as exc:
        raise LooK10IncrementOrchestrationError(str(exc)) from exc


def _orchestrator_replacements() -> dict[str, Any]:
    return {
        "__doc__": __doc__,
        "RUNNER_MODULE": RUNNER_MODULE,
        "EVALUATION_ID": EVALUATION_ID,
        "READY_SCHEMA": READY_SCHEMA,
        "GO_SCHEMA": GO_SCHEMA,
        "DONE_SCHEMA": DONE_SCHEMA,
        "TIMING_SCHEMA": TIMING_SCHEMA,
        "MODEL_ORDER": MODEL_ORDER,
        "SCOPES": SCOPES,
        "MAX_NUM_SEQS": MAX_NUM_SEQS,
        "GPU_LOCKS": GPU_LOCKS,
        "remediation": _POLICY_ADAPTER,
        "_formal_authorization_evidence": _formal_authorization_evidence,
    }


def configure_shared_orchestrator() -> dict[str, Any]:
    """Install this profile and return the values needed to restore it."""

    replacements = _orchestrator_replacements()
    previous = {name: getattr(shared, name) for name in replacements}
    for name, value in replacements.items():
        setattr(shared, name, value)
    return previous


def restore_shared_orchestrator(previous: dict[str, Any]) -> None:
    for name, value in previous.items():
        setattr(shared, name, value)


@contextlib.contextmanager
def configured_shared_orchestrator() -> Any:
    previous = configure_shared_orchestrator()
    try:
        yield shared
    finally:
        restore_shared_orchestrator(previous)


def run_model(**kwargs: Any) -> dict[str, Any]:
    with configured_shared_orchestrator():
        return shared.run_model(**kwargs)


def _worker_environment(**kwargs: Any) -> dict[str, str]:
    with configured_shared_orchestrator():
        return shared._worker_environment(**kwargs)


def _runner_command(**kwargs: Any) -> list[str]:
    with configured_shared_orchestrator():
        return shared._runner_command(**kwargs)


def _parser() -> Any:
    with configured_shared_orchestrator():
        return shared._parser()


def main(argv: Sequence[str] | None = None) -> int:
    with configured_shared_orchestrator():
        return shared.main(argv)


def __getattr__(name: str) -> Any:
    value = getattr(shared, name)
    if callable(value):

        def scoped(*args: Any, **kwargs: Any) -> Any:
            with configured_shared_orchestrator():
                return getattr(shared, name)(*args, **kwargs)

        return scoped
    return value


if __name__ == "__main__":
    raise SystemExit(main())
