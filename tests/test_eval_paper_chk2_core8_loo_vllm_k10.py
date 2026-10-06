from __future__ import annotations

from pathlib import Path
import json

import pytest

from jobs.eval import eval_paper_chk2_core8_loo_vllm_k10 as orchestrator


def _sealed_manifest(root: Path) -> Path:
    path = root / "preparation/evaluation_manifest.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}\n", encoding="utf-8")
    return path


def test_parser_exposes_frozen_phases() -> None:
    parser = orchestrator.build_parser()
    args = parser.parse_args(["all", "--resume", "--bootstrap-workers", "6"])
    assert args.phase == "all"
    assert args.resume is True
    assert args.bootstrap_workers == 6


def test_launcher_waits_for_gpu_budget_instead_of_exiting() -> None:
    text = orchestrator.VLLM_LAUNCHER.read_text(encoding="utf-8")
    assert "while true; do" in text
    assert "status=waiting_for_both_gpus_minimum_0.78" in text
    assert "if (( stability_pass == 1 )); then" in text
    assert "sleep 30" in text
    assert "stable utilization budget is below 0.78" not in text


def test_generate_delegates_to_dual_gpu_launcher(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _sealed_manifest(tmp_path)
    commands: list[list[str]] = []
    monkeypatch.setattr(orchestrator, "_run", lambda command: commands.append(list(command)))

    result = orchestrator.run_phase(
        "generate", output_root=tmp_path, resume=True, bootstrap_workers=6
    )

    assert result == {"status": "generation_complete", "rows": 21_760}
    assert commands == [
        [
            str(orchestrator.VLLM_LAUNCHER),
            "generate",
            str(manifest),
            str(tmp_path),
        ]
    ]


def test_bootstrap_freezes_draw_count_and_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _sealed_manifest(tmp_path)
    commands: list[list[str]] = []
    monkeypatch.setattr(orchestrator, "_run", lambda command: commands.append(list(command)))

    result = orchestrator.run_phase(
        "bootstrap", output_root=tmp_path, resume=True, bootstrap_workers=6
    )

    assert result == {"status": "bootstrapped", "draws": 10_000, "workers": 6}
    assert commands[0][-5:] == [
        "--run-root",
        str(tmp_path),
        "--workers",
        "6",
        "--resume",
    ]


def test_downstream_phase_requires_preparation(tmp_path: Path) -> None:
    with pytest.raises(orchestrator.Core8OrchestratorError, match="run prepare"):
        orchestrator.run_phase(
            "score", output_root=tmp_path, resume=False, bootstrap_workers=6
        )


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n", encoding="utf-8")


def test_final_manifest_is_create_once_then_verify_only(tmp_path: Path) -> None:
    _write_json(
        tmp_path / "preparation/evaluation_manifest.json",
        {
            "schema_version": "paper-chk2-core8-loo-preparation-manifest-v1",
            "status": "complete",
            "sealed": True,
            "immutable": True,
            "coverage": {"generation_rows": 21_760},
        },
    )
    _write_json(
        tmp_path / "generation/validation.json",
        {
            "schema_version": "paper-chk2-core8-loo-vllm-k10-validation-v1",
            "status": "complete",
            "mode": "formal",
            "model_id": "paper_chk2",
            "model_label": "paper-chk2-cp50-lora-over-chk1-cp200",
            "generation_rows": 21_760,
            "meetings": 128,
            "replicates": 10,
            "gates": {"tuple_closure": True, "cp50_lora": True},
        },
    )
    _write_json(
        tmp_path / "score/score_manifest.json",
        {
            "schema_version": "paper-chk2-core8-loo-score-manifest-v1",
            "status": "complete",
            "immutable": True,
            "paper_chk2_scope": True,
            "historical_generation_or_score_rows_reused": False,
        },
    )
    _write_json(
        tmp_path / "score/bootstrap_results.json",
        {
            "schema_version": "paper-chk2-core8-loo-statistics-v1",
            "status": "complete",
            "coverage": {
                "generation_rows": 21_760,
                "meeting_cells": 2_048,
                "primary_topic_arm_metric_cells": 32,
            },
            "bootstrap_contract": {"draws": 10_000},
        },
    )
    report_artifacts = {}
    for role, filename in {
        "canonical_artifact": "artifact.json",
        "portable_html": "report.html",
        "markdown_report": "report.md",
        "latex_table": "core8_loo_table.tex",
        "latex_results": "core8_loo_results.tex",
    }.items():
        path = tmp_path / "report" / filename
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"{role}\n", encoding="utf-8")
        report_artifacts[role] = orchestrator._binding(path)
    _write_json(
        tmp_path / "report/report_manifest.json",
        {
            "schema_version": "paper-chk2-core8-loo-report-manifest-v1",
            "status": "complete",
            "immutable": True,
            "created_at_utc": "2026-09-01T00:00:00Z",
            "artifacts": report_artifacts,
        },
    )

    created = orchestrator.seal_or_verify_final_manifest(tmp_path)
    verified = orchestrator.seal_or_verify_final_manifest(tmp_path)
    assert created == verified
    manifest = json.loads((tmp_path / "evaluation_manifest.json").read_text())
    assert manifest["schema_version"] == orchestrator.FINAL_MANIFEST_SCHEMA
    assert manifest["immutable"] is True
    assert manifest["coverage"]["generation_rows"] == 21_760
    assert manifest["interpretation"]["pristine_unseen_test_claim_allowed"] is False

    (tmp_path / "report/report.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(orchestrator.Core8OrchestratorError, match="binding drift"):
        orchestrator.seal_or_verify_final_manifest(tmp_path)
