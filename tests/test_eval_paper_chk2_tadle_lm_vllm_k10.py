from __future__ import annotations

from pathlib import Path

import pytest

from jobs.eval import eval_paper_chk2_tadle_lm_vllm_k10 as orchestrator


ROOT = Path(__file__).resolve().parents[1]


def test_cli_exposes_every_sealed_phase() -> None:
    expected = {
        "prepare",
        "smoke",
        "generate",
        "validate",
        "assemble",
        "sentiment",
        "estimate",
        "bootstrap",
        "report",
        "all",
    }
    observed = {
        orchestrator.parse_args([phase, "--bootstrap-workers", "1"]).phase
        for phase in expected
    }
    assert observed == expected


def test_cli_rejects_unknown_phase() -> None:
    with pytest.raises(SystemExit):
        orchestrator.parse_args(["legacy-cp318"])


def test_dual_gpu_launcher_is_non_destructive_and_waits_for_idle_capacity() -> None:
    text = (ROOT / "run/eval_paper_chk2_tadle_lm_vllm_k10_dual_gpu.sh").read_text(
        encoding="utf-8"
    )
    assert "worker_targets=(chk0 chk1_paper_chk2_cp50)" in text
    assert "CUDA_VISIBLE_DEVICES=0" in text
    assert "CUDA_VISIBLE_DEVICES=1" in text
    assert "free_mib < 22528" in text
    assert "fixed_vllm_budget_mib=20480" in text
    assert "compute_percent > 10" in text
    assert "validate-smoke" in text
    assert "validate-all" in text
    assert "kill " not in text
    assert "rm -" not in text


def test_background_launcher_uses_pinned_runtimes_and_resume() -> None:
    text = (ROOT / "run/eval_paper_chk2_tadle_lm_k10.sh").read_text(encoding="utf-8")
    assert "/home/haobin_cui/.conda/envs/fomc_trainer/bin/python" in text
    assert "jobs.eval.eval_paper_chk2_tadle_lm_vllm_k10" in text
    assert "PAPER_CHK2_TADLE_CPU_WORKERS:-32" in text
    assert '--bootstrap-workers "${cpu_workers}"' in text
    assert 'VLLM_PYTHON_BIN="${vllm_python}"' in text
    assert "flock -x 9" in text
    assert "nohup setsid env" in text
    assert "--resume" in text
    assert '>>"${log_path}"' in text


def test_terminal_binding_verifier_handles_declared_path_conventions(
    tmp_path: Path,
) -> None:
    path = tmp_path / "generation/generation_rows.jsonl"
    path.parent.mkdir(parents=True)
    path.write_text("{}\n", encoding="utf-8")
    relative = orchestrator._binding(path, relative_to=tmp_path)
    absolute = dict(relative)
    absolute["path"] = str(path.resolve())
    relative["rows"] = 1
    absolute["rows"] = 1

    orchestrator._verify_bound_file(
        tmp_path,
        relative,
        "generation/generation_rows.jsonl",
        rows=1,
    )
    orchestrator._verify_bound_file(
        tmp_path,
        absolute,
        "generation/generation_rows.jsonl",
        rows=1,
        absolute_record_path=True,
    )
    with pytest.raises(orchestrator.TadleOrchestratorError, match="path drift"):
        orchestrator._verify_bound_file(
            tmp_path,
            absolute,
            "generation/generation_rows.jsonl",
            rows=1,
        )

    link = tmp_path / "generation/linked_rows.jsonl"
    link.symlink_to(path)
    with pytest.raises(orchestrator.TadleOrchestratorError, match="unsafe"):
        orchestrator._binding(link)


def test_all_routes_token_replay_and_assembly_to_vllm_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, str]] = []
    monkeypatch.setattr(orchestrator, "PIPELINE_LOCK", tmp_path / "pipeline.lock")
    monkeypatch.setattr(
        orchestrator,
        "_phase_common",
        lambda phase, _root, *, resume: calls.append(("fomc-common", phase)),
    )
    monkeypatch.setattr(
        orchestrator,
        "_phase_vllm_common",
        lambda phase, _root, *, resume: calls.append(("vllm-common", phase)),
    )
    monkeypatch.setattr(
        orchestrator,
        "_phase_generation",
        lambda phase, _root: calls.append(("vllm-launcher", phase)),
    )
    monkeypatch.setattr(
        orchestrator,
        "_phase_statistics",
        lambda phase, _root, *, workers, resume: calls.append(("fomc-stats", phase)),
    )
    monkeypatch.setattr(
        orchestrator,
        "_phase_report",
        lambda _root, *, resume: calls.append(("fomc-report", "report")),
    )
    monkeypatch.setattr(
        orchestrator,
        "seal_or_verify_final_manifest",
        lambda _root: {"status": "complete"},
    )

    assert orchestrator.main(["all", "--output-root", str(tmp_path / "run")]) == 0
    assert calls == [
        ("fomc-common", "prepare"),
        ("vllm-launcher", "smoke"),
        ("vllm-launcher", "generate"),
        ("vllm-common", "assemble"),
        ("fomc-stats", "sentiment"),
        ("fomc-stats", "estimate"),
        ("fomc-stats", "bootstrap"),
        ("fomc-report", "report"),
    ]
