"""Fail-closed launcher for the isolated checkpoint-200 DeepSeek chk2 candidate."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import yaml

from open_r1.provenance import fingerprint_artifact_path, sha256_file


SCHEMA_VERSION = "chk2-deepseek-candidate-launch-binding-v1"
RECEIPT_SCHEMA_VERSION = "chk2-deepseek-candidate-launch-receipt-v1"
EXPECTED_REWARD = "grounded_analysis_v3_deepseek_high"
EXPECTED_MODEL = "deepseek-v4-flash"
EXPECTED_URL = "https://api.deepseek.com"
EXPECTED_SESSION = "fomc_chk2_cp200_deepseek_high_totalsl_v1_20260810"
EXPECTED_RUN_ROOT = Path(
    "output/training/retrain_v2/"
    "chk2_clean_v2_cp200_deepseek_high_totalsl_v1_20260810"
)
EXPECTED_OUTPUT_DIR = EXPECTED_RUN_ROOT / "adapters/chk2"
MIN_GPU_FREE_MIB = 17 * 1024
MIN_DISK_FREE_BYTES = 250 * 1024**3


class PreflightError(RuntimeError):
    """Raised when a launch invariant is not satisfied."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _load_object(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise PreflightError(f"{label} is missing or unsafe: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise PreflightError(f"{label} is not valid JSON: {path}") from exc
    if not isinstance(value, dict):
        raise PreflightError(f"{label} must be a JSON object: {path}")
    return value


def _repo_path(repo_root: Path, declared: object, *, label: str) -> Path:
    text = str(declared or "").strip()
    relative = Path(text)
    if not text or relative.is_absolute() or ".." in relative.parts:
        raise PreflightError(f"{label} must be a safe repo-relative path")
    resolved = (repo_root / relative).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise PreflightError(f"{label} escapes the repository: {text}") from exc
    return resolved


def _validate_artifact(
    repo_root: Path, raw: object, *, label: str
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(raw, Mapping):
        raise PreflightError(f"{label} binding must be an object")
    path = _repo_path(repo_root, raw.get("path"), label=f"{label}.path")
    expected_sha = str(raw.get("sha256") or "").strip().lower()
    if len(expected_sha) != 64:
        raise PreflightError(f"{label}.sha256 is invalid")
    actual = fingerprint_artifact_path(path)
    if actual["sha256"] != expected_sha:
        raise PreflightError(
            f"{label} fingerprint mismatch: expected {expected_sha}, "
            f"found {actual['sha256']}"
        )
    return path, actual


def _validate_config(config_path: Path, repo_root: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise PreflightError(f"Training config is invalid YAML: {config_path}") from exc
    if not isinstance(payload, dict):
        raise PreflightError("Training config must be a mapping")
    exact = {
        "model_name_or_path": (
            "output/training/retrain_v2/"
            "chk1_clean_v2_lr1e6_selected_cp200_for_chk2_20260810/merged/chk1"
        ),
        "dataset_name": (
            "dataset/processed/retrain_v2/"
            "analysis_grpo_full_v7_totalsl_billions_v1_20260810"
        ),
        "output_dir": EXPECTED_OUTPUT_DIR.as_posix(),
        "reward_funcs": [EXPECTED_REWARD],
        "max_prompt_length": 2560,
        "max_completion_length": 4096,
        "num_generations": 4,
        "generation_batch_size": 4,
        "gradient_accumulation_steps": 8,
        "num_train_epochs": 1,
        "judge_url": EXPECTED_URL,
        "judge_model": EXPECTED_MODEL,
        "judge_max_completion_tokens": 16384,
        "judge_timeout": 420,
        "judge_max_retries": 2,
        "judge_backoff_seconds": 2.0,
        "judge_api_key_env": "DEEPSEEK_API_KEY",
        "save_steps": 1,
        "checkpoint_keep_last": 3,
        "checkpoint_keep_every_n_steps": 10,
        "overwrite_output_dir": False,
    }
    mismatches = {
        key: {"expected": expected, "actual": payload.get(key)}
        for key, expected in exact.items()
        if payload.get(key) != expected
    }
    if mismatches:
        raise PreflightError(f"Training config contract mismatch: {mismatches}")
    for key in ("model_name_or_path", "dataset_name"):
        _repo_path(repo_root, payload[key], label=f"config.{key}")
    if payload.get("save_total_limit", "missing") is not None:
        raise PreflightError("save_total_limit must be null")
    if payload.get("save_strategy") != "steps":
        raise PreflightError("save_strategy must be steps")
    return payload


def _gpu_free_mib() -> int:
    command = [
        "nvidia-smi",
        "--query-gpu=memory.free",
        "--format=csv,noheader,nounits",
        "-i",
        "1",
    ]
    try:
        result = subprocess.run(
            command, check=True, capture_output=True, text=True, timeout=15
        )
        value = int(result.stdout.strip())
    except (OSError, ValueError, subprocess.SubprocessError) as exc:
        raise PreflightError("Unable to read physical GPU1 free memory") from exc
    return value


def _session_exists(session: str) -> bool:
    result = subprocess.run(
        ["tmux", "has-session", "-t", f"={session}"],
        check=False,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return result.returncode == 0


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    serialized = json.dumps(
        payload, ensure_ascii=False, indent=2, sort_keys=True
    ).encode("utf-8") + b"\n"
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    descriptor = os.open(path, flags, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=False) as handle:
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
    finally:
        os.close(descriptor)


def preflight(
    *, repo_root: Path, binding_path: Path, require_fresh: bool = True
) -> dict[str, Any]:
    repo_root = repo_root.resolve()
    binding_path = binding_path.resolve()
    binding = _load_object(binding_path, label="launch binding")
    if binding.get("schema_version") != SCHEMA_VERSION:
        raise PreflightError("Unsupported launch binding schema")
    if binding.get("session_name") != EXPECTED_SESSION:
        raise PreflightError("Launch binding session is not the pinned candidate session")
    artifacts = binding.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise PreflightError("Launch binding artifacts are missing")
    checked: dict[str, Any] = {}
    paths: dict[str, Path] = {}
    for label in (
        "config",
        "source_checkpoint",
        "merged_chk1",
        "dataset_overlay",
        "overlay_manifest",
        "authorization",
        "merge_lineage",
        "reward_module",
        "reward_registry",
    ):
        path, fingerprint = _validate_artifact(repo_root, artifacts.get(label), label=label)
        paths[label] = path
        checked[label] = fingerprint

    config_payload = _validate_config(paths["config"], repo_root)
    if paths["source_checkpoint"].as_posix().endswith("lr1e5/checkpoint-200"):
        raise PreflightError("The lr=1e-5 checkpoint-200 is explicitly forbidden")
    expected_output = (repo_root / EXPECTED_OUTPUT_DIR).resolve()
    expected_run_root = (repo_root / EXPECTED_RUN_ROOT).resolve()
    if _repo_path(
        repo_root, config_payload["output_dir"], label="config.output_dir"
    ) != expected_output:
        raise PreflightError("Config output directory is not the pinned fresh run")
    if require_fresh and expected_output.exists():
        raise PreflightError(f"Fresh output directory already exists: {expected_output}")
    if _session_exists(EXPECTED_SESSION):
        raise PreflightError(f"tmux session already exists: {EXPECTED_SESSION}")
    if not os.environ.get("DEEPSEEK_API_KEY", "").strip():
        raise PreflightError("DEEPSEEK_API_KEY is missing from fomc_trainer")
    for name in ("OPENAI_LOG", "HTTPX_LOG_LEVEL"):
        if os.environ.get(name, "").strip().casefold() in {"debug", "trace"}:
            raise PreflightError(f"Unsafe {name} logging is enabled")
    if shutil.which("tmux") is None:
        raise PreflightError("tmux is required")
    if shutil.which("conda") is None:
        raise PreflightError("conda is required")

    gpu_free = _gpu_free_mib()
    if gpu_free < MIN_GPU_FREE_MIB:
        raise PreflightError(
            f"GPU1 has {gpu_free} MiB free; at least {MIN_GPU_FREE_MIB} MiB is required"
        )
    disk_free = shutil.disk_usage(repo_root).free
    if disk_free < MIN_DISK_FREE_BYTES:
        raise PreflightError(
            f"Disk has {disk_free} bytes free; at least {MIN_DISK_FREE_BYTES} required"
        )
    return {
        "binding": binding,
        "binding_sha256": sha256_file(binding_path),
        "checked_artifacts": checked,
        "config_path": paths["config"],
        "run_root": expected_run_root,
        "output_dir": expected_output,
        "gpu1_free_mib": gpu_free,
        "disk_free_bytes": disk_free,
    }


def launch(*, repo_root: Path, binding_path: Path, receipt_path: Path) -> dict[str, Any]:
    result = preflight(repo_root=repo_root, binding_path=binding_path)
    run_root = result["run_root"]
    log_dir = run_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=False, mode=0o700)
    training_log = log_dir / "train.log"
    conda = shutil.which("conda")
    assert conda is not None
    training_command = [
        conda,
        "run",
        "--no-capture-output",
        "-n",
        "fomc_trainer",
        "python",
        "-u",
        "-m",
        "jobs.train.train_grpo",
        "--config",
        str(result["config_path"]),
    ]
    shell_command = (
        f"cd {shlex.quote(str(repo_root.resolve()))} && "
        "export CUDA_VISIBLE_DEVICES=1 "
        "PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True "
        "TOKENIZERS_PARALLELISM=false PYTHONUNBUFFERED=1 "
        f"PYTHONPATH={shlex.quote(str(repo_root.resolve() / 'src'))}:. && "
        f"exec {shlex.join(training_command)} "
        f">>{shlex.quote(str(training_log))} 2>&1"
    )
    started_at = _utc_now()
    subprocess.run(
        ["tmux", "new-session", "-d", "-s", EXPECTED_SESSION, shell_command],
        check=True,
    )
    time.sleep(1.0)
    if not _session_exists(EXPECTED_SESSION):
        raise PreflightError(
            f"Training process exited during launch; inspect {training_log}"
        )
    receipt = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "status": "started",
        "started_at_utc": started_at,
        "session_name": EXPECTED_SESSION,
        "config_path": str(result["config_path"].relative_to(repo_root.resolve())),
        "binding_path": str(binding_path.resolve().relative_to(repo_root.resolve())),
        "binding_sha256": result["binding_sha256"],
        "run_root": str(run_root.relative_to(repo_root.resolve())),
        "training_log": str(training_log.relative_to(repo_root.resolve())),
        "physical_policy_gpu": 1,
        "cuda_visible_devices": "1",
        "world_size": 1,
        "local_judge_started": False,
        "resume_from_checkpoint": None,
        "gpu1_free_mib_at_launch": result["gpu1_free_mib"],
        "disk_free_bytes_at_launch": result["disk_free_bytes"],
        "command_sha256": __import__("hashlib").sha256(
            shell_command.encode("utf-8")
        ).hexdigest(),
        "api_key_source": "DEEPSEEK_API_KEY",
        "api_key_recorded": False,
    }
    try:
        _write_immutable_json(receipt_path.resolve(), receipt)
    except Exception:
        subprocess.run(
            ["tmux", "kill-session", "-t", f"={EXPECTED_SESSION}"], check=False
        )
        raise
    return receipt


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--binding", type=Path, required=True)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--execute", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.execute:
            if args.receipt is None:
                raise PreflightError("--receipt is required with --execute")
            result = launch(
                repo_root=args.repo_root,
                binding_path=args.binding,
                receipt_path=args.receipt,
            )
        else:
            checked = preflight(
                repo_root=args.repo_root,
                binding_path=args.binding,
                require_fresh=True,
            )
            result = {
                "status": "preflight_passed",
                "session_name": EXPECTED_SESSION,
                "binding_sha256": checked["binding_sha256"],
                "gpu1_free_mib": checked["gpu1_free_mib"],
                "disk_free_bytes": checked["disk_free_bytes"],
            }
    except (OSError, PreflightError, subprocess.SubprocessError, ValueError) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
