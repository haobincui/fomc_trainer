from __future__ import annotations

import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
RECOVERY_SCRIPT = REPO_ROOT / "run/recover_paper_chk2_downstream_v6.sh"


def _script_text() -> str:
    return RECOVERY_SCRIPT.read_text(encoding="utf-8")


def _assignment(text: str, name: str) -> str:
    match = re.search(rf'^{re.escape(name)}="([^"]+)"$', text, re.MULTILINE)
    assert match is not None, f"missing launcher assignment: {name}"
    return match.group(1)


def test_recovery_launcher_is_valid_and_requires_fomc_trainer() -> None:
    subprocess.run(["bash", "-n", str(RECOVERY_SCRIPT)], check=True)
    text = _script_text()

    assert "set -Eeuo pipefail" in text
    assert '[[ "${CONDA_DEFAULT_ENV:-}" != "fomc_trainer" ]]' in text
    assert "Activate the fomc_trainer conda environment" in text


def test_recovery_launcher_holds_both_locks_without_truncating_original_lock() -> None:
    text = _script_text()

    recovery_open = text.index('exec 9>"$RECOVERY_LOCK"')
    recovery_flock = text.index("flock -n 9")
    original_open = text.index('exec 8>>"$ORIGINAL_LOCK"')
    original_flock = text.index("flock -n 8")
    launch = text.index("nohup setsid")

    assert recovery_open < recovery_flock < original_open < original_flock < launch
    assert 'exec 8>"$ORIGINAL_LOCK"' not in text
    assert text.count("flock -n ") == 2
    assert 'ORIGINAL_LOCK="$ORIGINAL_ROOT/build.lock"' in text


def test_recovery_launcher_uses_an_independent_root_and_resume_contract() -> None:
    text = _script_text()
    original_root = _assignment(text, "ORIGINAL_ROOT")
    recovery_root = _assignment(text, "RECOVERY_ROOT")

    assert original_root != recovery_root
    assert "v6_downstream128_20260831" in original_root
    assert "v6_downstream128_recovery_v1_20260831" in recovery_root
    assert 'RECOVERY_LOG="${RECOVERY_ROOT}.log"' in text
    assert 'RECOVERY_LOCK="${RECOVERY_ROOT}.lock"' in text
    assert "jobs.generation.recover_paper_chk2_downstream_v6" in text
    assert '--original-root "$ORIGINAL_ROOT"' in text
    assert '--recovery-root "$RECOVERY_ROOT"' in text
    assert '--handoff-root "$HANDOFF_ROOT"' in text
    assert "--phase all" in text
    assert "--concurrency 128 --resume" in text


def test_recovery_launcher_never_logs_or_embeds_the_api_key() -> None:
    text = _script_text()

    assert "sk-" not in text
    assert 'read -r -s -p "DeepSeek API key (input is hidden): "' in text
    assert "export DEEPSEEK_API_KEY" in text
    assert "unset DEEPSEEK_API_KEY" in text
    assert text.index("export DEEPSEEK_API_KEY") < text.index("nohup setsid")
    assert text.index("nohup setsid") < text.index("unset DEEPSEEK_API_KEY")

    for line in text.splitlines():
        if any(token in line for token in ("printf ", "echo ", "RECOVERY_LOG")):
            assert "$DEEPSEEK_API_KEY" not in line
            assert "${DEEPSEEK_API_KEY}" not in line


def test_recovery_launcher_cannot_start_the_legacy_generator_directly() -> None:
    text = _script_text()

    assert "jobs.generation.generate_paper_chk2_downstream_v6" not in text
    assert "build_paper_chk2_downstream_v6.sh" not in text
    assert text.count("python -m") == 1
