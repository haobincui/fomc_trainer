from __future__ import annotations

import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_SCRIPT = REPO_ROOT / "run/build_paper_chk2_downstream_v6.sh"
TRANSITION_SCRIPT = REPO_ROOT / "run/auto_transition_paper_chk2_source_to_v6.sh"


def test_v6_shell_entrypoints_are_valid_and_do_not_embed_credentials() -> None:
    subprocess.run(
        ["bash", "-n", str(BUILD_SCRIPT), str(TRANSITION_SCRIPT)],
        check=True,
    )
    for script in (BUILD_SCRIPT, TRANSITION_SCRIPT):
        text = script.read_text(encoding="utf-8")
        assert "sk-" not in text
        assert "DEEPSEEK_API_KEY" in text


def test_direct_v6_runner_is_locked_and_uses_concurrency_128() -> None:
    text = BUILD_SCRIPT.read_text(encoding="utf-8")
    assert "flock -n 9" in text
    assert "generate_paper_chk2_downstream_v6" in text
    assert "--phase all --concurrency 128 --resume" in text
    assert "handoff_manifest.json" in text


def test_transition_waits_for_seal_boundary_then_builds_and_publishes() -> None:
    text = TRANSITION_SCRIPT.read_text(encoding="utf-8")
    receipt = text.index("source_admission_receipt.json")
    source_lock = text.index('flock -n 8')
    seal = text.index("seal_paper_chk2_source_handoff_v1")
    build = text.index("generate_paper_chk2_downstream_v6")
    publish = text.index("publish_paper_chk2_downstream_v6")
    assert receipt < source_lock < seal < build < publish
    assert "--phase all --concurrency 128 --resume" in text


def test_transition_has_root_scoped_liveness_and_failure_terminal() -> None:
    text = TRANSITION_SCRIPT.read_text(encoding="utf-8")
    assert 'exec 8>"$SOURCE_LOCK"' in text
    assert "pgrep" not in text
    assert "Transition FAILED UTC=%s Stage=%s Exit=%s" in text
    assert 'CURRENT_STAGE="run_v6_downstream"' in text
    assert 'CURRENT_STAGE="publish_release"' in text
    assert "unset DEEPSEEK_API_KEY" in text
