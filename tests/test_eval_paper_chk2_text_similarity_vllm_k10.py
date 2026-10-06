from __future__ import annotations

from pathlib import Path

from jobs.eval import eval_paper_chk2_text_similarity_vllm_k10 as subject


def _prepared_root(tmp_path: Path) -> Path:
    root = tmp_path / "evaluation"
    root.mkdir()
    (root / "evaluation_manifest.json").write_text("{}\n", encoding="utf-8")
    return root


def test_generate_dispatches_dual_gpu_then_consolidates_with_resume(
    tmp_path: Path, monkeypatch
) -> None:
    root = _prepared_root(tmp_path)
    commands: list[list[str]] = []
    monkeypatch.setattr(subject, "_run", lambda command, env=None: commands.append(list(command)))
    result = subject.run_phase(
        "generate",
        output_root=root,
        resume=True,
        semantic_device="cuda:0",
        semantic_batch_size=8,
    )
    assert result == {"status": "generation_complete", "rows": 11_730}
    assert commands[0][0] == str(subject.VLLM_LAUNCHER)
    assert commands[0][1] == "generate"
    assert commands[0][-1] == str(root / "generation")
    assert "consolidate" in commands[1]
    assert commands[1][-1] == "--resume"


def test_score_and_report_dispatch_fixed_artifact_paths(tmp_path: Path, monkeypatch) -> None:
    root = _prepared_root(tmp_path)
    commands: list[list[str]] = []
    monkeypatch.setattr(subject, "_run", lambda command, env=None: commands.append(list(command)))
    subject.run_phase(
        "score",
        output_root=root,
        resume=False,
        semantic_device="cuda:1",
        semantic_batch_size=12,
    )
    score = commands.pop()
    assert "jobs.eval.paper_chk2_text_similarity_scoring" in score
    assert score[score.index("--semantic-device") + 1] == "cuda:1"
    assert score[score.index("--semantic-batch-size") + 1] == "12"
    assert score[score.index("--generation-rows") + 1] == str(root / "generation_rows.jsonl")

    subject.run_phase(
        "report",
        output_root=root,
        resume=True,
        semantic_device="cuda:0",
        semantic_batch_size=8,
    )
    report = commands.pop()
    assert "jobs.eval.render_paper_chk2_text_similarity_report" in report
    assert report[report.index("--score-root") + 1] == str(root)
    assert report[report.index("--report-root") + 1] == str(root / "report")
    assert report[-1] == "--resume"


def test_bootstrap_dispatches_parallel_workers(tmp_path: Path, monkeypatch) -> None:
    root = _prepared_root(tmp_path)
    commands: list[list[str]] = []
    monkeypatch.setattr(subject, "_run", lambda command, env=None: commands.append(list(command)))
    result = subject.run_phase(
        "bootstrap",
        output_root=root,
        resume=True,
        semantic_device="cuda:0",
        semantic_batch_size=8,
        bootstrap_workers=48,
    )
    command = commands.pop()
    assert "jobs.eval.paper_chk2_text_similarity_bootstrap_parallel" in command
    assert command[command.index("--workers") + 1] == "48"
    assert command[-1] == "--resume"
    assert result == {"status": "bootstrapped", "draws": 1_000, "workers": 48}


def test_cli_exposes_prespecified_phases() -> None:
    parser = subject.build_parser()
    args = parser.parse_args(["all", "--resume"])
    assert args.phase == "all"
    assert args.resume is True
    assert args.semantic_device == "cuda:0"
    assert args.bootstrap_workers == 6
