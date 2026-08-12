"""Exclusive pre/post attestations for the immutable retrain-v2 judge service.

The attestation deliberately records only stable service identity fields.  vLLM
0.19.1 does not expose a stable process start time: ``ModelCard.created`` is
constructed while serving each ``/v1/models`` response, so it is documented and
excluded from the pre/post identity comparison.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping

import requests

from jobs.retrain_v2.dag import verify_judge_artifact
from jobs.retrain_v2.judge_health import (
    _verify_loaded_model_root,
    check_judge,
)
from jobs.retrain_v2.stage_lock import (
    StageLockError,
    require_inherited_stage_lock,
)


SCHEMA_VERSION = 1
PHASES = ("pre", "post")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SERVER_START_TIME_NOTE = (
    "unavailable: vLLM 0.19.1 /v1/models has no stable server start time; "
    "ModelCard.created is generated for each response and is excluded"
)


class JudgeAttestationError(ValueError):
    """Raised when a judge attestation is unsafe, incomplete, or has drifted."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise JudgeAttestationError(message)


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _canonical_bytes(payload: Any) -> bytes:
    try:
        rendered = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise JudgeAttestationError(
            "Attestation payload is not canonical JSON"
        ) from exc
    return rendered.encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    except OSError as exc:
        raise JudgeAttestationError(f"Unable to hash attestation: {path}") from exc
    return digest.hexdigest()


def _lexical_absolute(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _require_repo_path(
    value: str | Path,
    *,
    repo_root: Path,
    label: str,
    require_exists: bool = True,
) -> Path:
    candidate = Path(value)
    lexical = _lexical_absolute(
        candidate if candidate.is_absolute() else repo_root / candidate
    )
    try:
        relative = lexical.relative_to(repo_root)
    except ValueError as exc:
        raise JudgeAttestationError(f"{label} escapes the repository") from exc
    current = repo_root
    for part in relative.parts:
        current /= part
        _require(not current.is_symlink(), f"{label} must not contain symlinks")
    if require_exists:
        _require(lexical.exists(), f"{label} does not exist")
    return lexical.resolve()


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    _require(not path.is_symlink(), f"{label} must not be a symlink")
    _require(path.is_file(), f"{label} is missing")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise JudgeAttestationError(f"Unable to parse {label}") from exc
    _require(isinstance(payload, dict), f"{label} must contain an object")
    return payload


def _manifest_identity(manifest_path: Path, *, repo_root: Path) -> dict[str, str]:
    payload = _load_json(manifest_path, label="run manifest")
    _require(payload.get("schema_version") == 2, "Unsupported run manifest schema")
    run_id = payload.get("run_id")
    _require(isinstance(run_id, str) and run_id != "", "Run manifest has no run_id")
    return {
        "run_id": run_id,
        "path": manifest_path.relative_to(repo_root).as_posix(),
    }


def _immutable_contract(record: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "served_model_name",
        "url",
        "timeout",
        "max_retries",
        "backoff_seconds",
        "tokenizer_path",
        "max_model_len",
        "max_completion_tokens",
        "candidate_reserve_tokens",
        "boundary_margin_tokens",
    )
    contract = {key: record.get(key) for key in keys}
    _require(
        all(value is not None for value in contract.values()),
        "Judge contract is incomplete",
    )
    return contract


def _attestation_context(
    run_manifest: str | Path, repo_root: str | Path
) -> dict[str, Any]:
    root = _lexical_absolute(Path(repo_root))
    _require(root.is_dir(), "Repository root is missing")
    _require(
        not root.is_symlink() and root == root.resolve(),
        "Repository root must be canonical",
    )
    manifest = _require_repo_path(run_manifest, repo_root=root, label="run manifest")
    try:
        verified = verify_judge_artifact(manifest, repo_root=root)
    except Exception as exc:  # noqa: BLE001 - normalize the public module boundary
        raise JudgeAttestationError(str(exc)) from exc
    contract = _immutable_contract(verified)
    artifact = verified.get("artifact")
    _require(isinstance(artifact, dict), "Judge artifact fingerprint is missing")
    artifact_path = _require_repo_path(
        str(artifact.get("path") or ""), repo_root=root, label="judge artifact"
    )
    _require(artifact_path.is_dir(), "Judge artifact is not a directory")
    tokenizer_path = _require_repo_path(
        str(contract["tokenizer_path"]),
        repo_root=root,
        label="judge tokenizer",
    )
    _require(
        tokenizer_path == artifact_path,
        "Judge tokenizer and loaded model root must be the same immutable artifact",
    )
    manifest_record = _manifest_identity(manifest, repo_root=root)
    projection = {"contract": contract, "artifact": artifact}
    return {
        "repo_root": root,
        "manifest_path": manifest,
        "manifest": manifest_record,
        "run_root": manifest.parent,
        "contract": contract,
        "artifact": dict(artifact),
        "artifact_path": artifact_path,
        "manifest_projection_sha256": _canonical_sha256(projection),
    }


def _authorization_headers() -> dict[str, str]:
    api_key = os.environ.get("OPEN_R1_JUDGE_API_KEY")
    return {"Authorization": f"Bearer {api_key}"} if api_key else {}


def fetch_service_identity(
    *,
    url: str,
    model: str,
    timeout: int,
    expected_model_root: Path,
) -> dict[str, Any]:
    """Read stable service identity fields from the real vLLM models endpoint."""

    base_url = url.split("/v1/chat/completions", 1)[0].rstrip("/")
    response = requests.get(
        f"{base_url}/v1/models",
        headers=_authorization_headers(),
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    _require(isinstance(payload, dict), "Judge models response must be an object")
    cards = [item for item in payload.get("data", []) if isinstance(item, dict)]
    matching = [item for item in cards if item.get("id") == model]
    _require(
        len(cards) == 1 and len(matching) == 1,
        "Judge service must expose exactly one matching model card",
    )
    card = matching[0]
    root = _verify_loaded_model_root(card.get("root"), expected=expected_model_root)
    _require(
        card.get("max_model_len") is not None,
        "Judge model card has no max_model_len",
    )
    stable_fields = {
        "endpoint": base_url,
        "id": card.get("id"),
        "object": card.get("object", "model"),
        "owned_by": card.get("owned_by", "vllm"),
        "root": str(root),
        "parent": card.get("parent"),
        "max_model_len": card.get("max_model_len"),
    }
    observed_max_model_len = stable_fields["max_model_len"]
    _require(
        isinstance(observed_max_model_len, int)
        and not isinstance(observed_max_model_len, bool)
        and observed_max_model_len > 0,
        "Judge model card max_model_len is invalid",
    )
    return {
        "stable_fields": stable_fields,
        "stable_fields_sha256": _canonical_sha256(stable_fields),
        "server_start_time": _SERVER_START_TIME_NOTE,
        "excluded_unstable_fields": ["created"],
    }


def _validate_health(
    health: Any, *, contract: Mapping[str, Any], artifact_path: Path
) -> dict[str, Any]:
    _require(isinstance(health, dict), "Judge health result must be an object")
    _require(health.get("status") == "ready", "Judge health is not ready")
    _require(health.get("tokenizer_parity") is True, "Token-ID parity was not verified")
    _require(health.get("weight_attested") is True, "Judge model root was not attested")
    _require(
        health.get("loaded_model_root") == str(artifact_path),
        "Judge health model root drifted",
    )
    for health_key, contract_key in (
        ("url", "url"),
        ("model", "served_model_name"),
        ("max_model_len", "max_model_len"),
        ("max_completion_tokens", "max_completion_tokens"),
    ):
        _require(
            health.get(health_key) == contract.get(contract_key),
            f"Judge health {health_key} drifted",
        )
    prompt_tokens = health.get("golden_prompt_tokens")
    _require(
        isinstance(prompt_tokens, int)
        and not isinstance(prompt_tokens, bool)
        and prompt_tokens > 0,
        "Judge health has no golden prompt token count",
    )
    return dict(health)


def _attestation_path(context: Mapping[str, Any], phase: str) -> Path:
    _require(phase in PHASES, "Judge attestation phase must be pre or post")
    return context["run_root"] / "attestations" / f"judge.{phase}.json"


def _validate_timestamp(value: Any) -> datetime:
    _require(
        isinstance(value, str) and value.endswith("Z"),
        "Attestation time must be UTC",
    )
    try:
        observed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise JudgeAttestationError("Attestation time is not ISO-8601") from exc
    _require(observed.tzinfo is not None, "Attestation time must include a timezone")
    return observed


def _write_exclusive_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(
        not path.parent.is_symlink(), "Attestation directory must not be a symlink"
    )
    _require(
        path.parent == path.parent.resolve(), "Attestation directory must be canonical"
    )
    _require(
        not path.exists() and not path.is_symlink(),
        "Judge attestation already exists",
    )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise JudgeAttestationError("Judge attestation already exists") from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _load_attestation(path: Path, *, expected_phase: str) -> dict[str, Any]:
    payload = _load_json(path, label=f"judge {expected_phase} attestation")
    expected_keys = {
        "schema_version",
        "attestation_type",
        "phase",
        "recorded_at_utc",
        "run_manifest",
        "immutable_judge_contract",
        "judge_manifest_projection_sha256",
        "model_artifact",
        "service_identity",
        "token_id_parity",
        "health_result",
        "pre_binding",
        "canonical_payload_sha256",
    }
    _require(set(payload) == expected_keys, "Judge attestation has unexpected fields")
    _require(
        payload.get("schema_version") == SCHEMA_VERSION,
        "Unsupported attestation schema",
    )
    _require(
        payload.get("attestation_type") == "judge_service",
        "Wrong attestation type",
    )
    _require(payload.get("phase") == expected_phase, "Wrong judge attestation phase")
    _validate_timestamp(payload.get("recorded_at_utc"))
    digest = payload.get("canonical_payload_sha256")
    _require(
        isinstance(digest, str) and _SHA256_RE.fullmatch(digest) is not None,
        "Attestation payload SHA is invalid",
    )
    canonical_payload = {
        key: value
        for key, value in payload.items()
        if key != "canonical_payload_sha256"
    }
    _require(
        _canonical_sha256(canonical_payload) == digest,
        "Attestation canonical payload hash mismatch",
    )
    identity = payload.get("service_identity")
    _require(isinstance(identity, dict), "Attestation service identity is missing")
    _require(
        set(identity)
        == {
            "stable_fields",
            "stable_fields_sha256",
            "server_start_time",
            "excluded_unstable_fields",
        },
        "Attestation service identity has unexpected fields",
    )
    stable = identity.get("stable_fields")
    _require(isinstance(stable, dict), "Attestation stable service fields are missing")
    _require(
        set(stable)
        == {
            "endpoint",
            "id",
            "object",
            "owned_by",
            "root",
            "parent",
            "max_model_len",
        },
        "Attestation stable service fields changed",
    )
    _require(
        identity.get("stable_fields_sha256") == _canonical_sha256(stable),
        "Service identity hash mismatch",
    )
    _require(
        identity.get("server_start_time") == _SERVER_START_TIME_NOTE,
        "Server start-time limitation record drifted",
    )
    _require(
        identity.get("excluded_unstable_fields") == ["created"],
        "Unstable service field exclusion drifted",
    )
    recorded_health = payload.get("health_result")
    _require(isinstance(recorded_health, dict), "Attestation health result is missing")
    token_parity = payload.get("token_id_parity")
    _require(
        isinstance(token_parity, dict)
        and token_parity
        == {
            "verified": True,
            "method": "exact local/server token-ID sequence via judge_health",
            "golden_prompt_tokens": recorded_health.get("golden_prompt_tokens"),
            "max_model_len_verified": True,
        },
        "Token-ID parity attestation is invalid",
    )
    projection_sha = payload.get("judge_manifest_projection_sha256")
    _require(
        isinstance(projection_sha, str)
        and _SHA256_RE.fullmatch(projection_sha) is not None,
        "Judge manifest projection SHA is invalid",
    )
    pre_binding = payload.get("pre_binding")
    if expected_phase == "pre":
        _require(pre_binding is None, "Pre attestation must not bind itself")
    else:
        _require(isinstance(pre_binding, dict), "Post pre binding is missing")
        _require(
            set(pre_binding) == {"path", "file_sha256", "canonical_payload_sha256"},
            "Post pre binding has unexpected fields",
        )
        for key in ("file_sha256", "canonical_payload_sha256"):
            value = pre_binding.get(key)
            _require(
                isinstance(value, str) and _SHA256_RE.fullmatch(value) is not None,
                f"Post pre binding {key} is invalid",
            )
    return payload


def _attestation_file_binding(
    path: Path, payload: Mapping[str, Any], *, repo_root: Path
) -> dict[str, str]:
    return {
        "path": path.relative_to(repo_root).as_posix(),
        "file_sha256": _sha256_file(path),
        "canonical_payload_sha256": str(payload["canonical_payload_sha256"]),
    }


def record_judge_attestation(
    run_manifest: str | Path,
    *,
    phase: str,
    repo_root: str | Path,
    recorded_at_utc: str | None = None,
    health_checker: Callable[..., dict[str, Any]] | None = None,
    identity_fetcher: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Record one immutable service observation; post is bound to verified pre."""

    context = _attestation_context(run_manifest, repo_root)
    try:
        require_inherited_stage_lock(
            context["manifest_path"], "chk2", context["repo_root"]
        )
    except StageLockError as exc:
        raise JudgeAttestationError(
            f"Judge attestation requires the inherited chk2 stage lock: {exc}"
        ) from exc
    path = _attestation_path(context, phase)
    _require_repo_path(
        path.parent,
        repo_root=context["repo_root"],
        label="attestation directory",
        require_exists=False,
    )
    _require(
        not path.exists() and not path.is_symlink(),
        "Judge attestation already exists",
    )

    pre_payload = None
    pre_path = _attestation_path(context, "pre")
    pre_record = None
    if phase == "pre":
        post_path = _attestation_path(context, "post")
        _require(
            not post_path.exists() and not post_path.is_symlink(),
            "Post attestation already exists",
        )
    elif phase == "post":
        pre_path = _require_repo_path(
            pre_path,
            repo_root=context["repo_root"],
            label="judge pre attestation",
        )
        pre_payload = _load_attestation(pre_path, expected_phase="pre")
        _validate_static_record(pre_payload, context=context, label="Pre attestation")
        _require(
            pre_payload["pre_binding"] is None,
            "Pre attestation must not bind itself",
        )
        pre_record = _attestation_file_binding(
            pre_path, pre_payload, repo_root=context["repo_root"]
        )
    else:
        raise JudgeAttestationError("Judge attestation phase must be pre or post")

    observation_time = recorded_at_utc or _utc_now()
    observed_timestamp = _validate_timestamp(observation_time)
    if pre_payload is not None:
        _require(
            observed_timestamp >= _validate_timestamp(pre_payload["recorded_at_utc"]),
            "Post attestation predates pre",
        )

    contract = context["contract"]
    identity_function = identity_fetcher or fetch_service_identity
    service_identity = identity_function(
        url=contract["url"],
        model=contract["served_model_name"],
        timeout=contract["timeout"],
        expected_model_root=context["artifact_path"],
    )
    _require(isinstance(service_identity, dict), "Service identity must be an object")
    _require(
        set(service_identity)
        == {
            "stable_fields",
            "stable_fields_sha256",
            "server_start_time",
            "excluded_unstable_fields",
        },
        "Service identity has unexpected fields",
    )
    stable_fields = service_identity.get("stable_fields")
    _require(isinstance(stable_fields, dict), "Stable service identity is missing")
    _require(
        set(stable_fields)
        == {
            "endpoint",
            "id",
            "object",
            "owned_by",
            "root",
            "parent",
            "max_model_len",
        },
        "Stable service identity fields changed",
    )
    _require(
        service_identity.get("stable_fields_sha256")
        == _canonical_sha256(stable_fields),
        "Stable service identity hash mismatch",
    )
    _require(
        service_identity.get("server_start_time") == _SERVER_START_TIME_NOTE
        and service_identity.get("excluded_unstable_fields") == ["created"],
        "Service identity must document the unavailable stable start time",
    )
    _require(
        stable_fields.get("id") == contract["served_model_name"]
        and stable_fields.get("root") == str(context["artifact_path"])
        and stable_fields.get("max_model_len") == contract["max_model_len"],
        "Stable service identity disagrees with immutable contract",
    )
    if pre_payload is not None:
        _require(
            stable_fields == pre_payload["service_identity"]["stable_fields"],
            "Post judge service identity differs from pre",
        )

    health_function = health_checker or check_judge
    health = health_function(
        url=contract["url"],
        model=contract["served_model_name"],
        timeout=contract["timeout"],
        expected_model_root=context["artifact_path"],
        tokenizer_path=context["artifact_path"],
        max_model_len=contract["max_model_len"],
        max_completion_tokens=contract["max_completion_tokens"],
    )
    health_result = _validate_health(
        health, contract=contract, artifact_path=context["artifact_path"]
    )
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "attestation_type": "judge_service",
        "phase": phase,
        "recorded_at_utc": observation_time,
        "run_manifest": context["manifest"],
        "immutable_judge_contract": contract,
        "judge_manifest_projection_sha256": context["manifest_projection_sha256"],
        "model_artifact": context["artifact"],
        "service_identity": service_identity,
        "token_id_parity": {
            "verified": True,
            "method": "exact local/server token-ID sequence via judge_health",
            "golden_prompt_tokens": health_result["golden_prompt_tokens"],
            "max_model_len_verified": True,
        },
        "health_result": health_result,
        "pre_binding": pre_record,
    }
    _validate_record_against_context(payload, context=context)
    payload["canonical_payload_sha256"] = _canonical_sha256(payload)
    _write_exclusive_atomic(path, payload)
    return {"status": "recorded", "path": str(path), **payload}


def _validate_record_against_context(
    payload: Mapping[str, Any], *, context: Mapping[str, Any]
) -> None:
    contract = context["contract"]
    stable = payload["service_identity"]["stable_fields"]
    expected_endpoint = contract["url"].split("/v1/chat/completions", 1)[0].rstrip("/")
    _require(
        stable.get("endpoint") == expected_endpoint, "Judge endpoint identity drifted"
    )
    _require(
        stable.get("id") == contract["served_model_name"],
        "Judge served-model identity drifted",
    )
    _require(
        stable.get("root") == str(context["artifact_path"]),
        "Judge model-root identity drifted",
    )
    _require(
        stable.get("max_model_len") == contract["max_model_len"],
        "Judge context identity drifted",
    )
    _validate_health(
        payload.get("health_result"),
        contract=contract,
        artifact_path=context["artifact_path"],
    )


def _validate_static_record(
    payload: Mapping[str, Any], *, context: Mapping[str, Any], label: str
) -> None:
    _require(
        payload["run_manifest"] == context["manifest"],
        f"{label} run manifest binding drifted",
    )
    _require(
        payload["immutable_judge_contract"] == context["contract"],
        f"{label} judge contract drifted",
    )
    _require(
        payload["model_artifact"] == context["artifact"],
        f"{label} judge artifact drifted",
    )
    _require(
        payload["judge_manifest_projection_sha256"]
        == context["manifest_projection_sha256"],
        f"{label} judge manifest projection drifted",
    )
    _validate_record_against_context(payload, context=context)


def verify_judge_attestations(
    run_manifest: str | Path,
    *,
    repo_root: str | Path,
    require_post: bool = True,
) -> dict[str, Any]:
    """Verify recorded provenance and pre/post binding without contacting the judge."""

    context = _attestation_context(run_manifest, repo_root)
    pre_path = _require_repo_path(
        _attestation_path(context, "pre"),
        repo_root=context["repo_root"],
        label="judge pre attestation",
    )
    pre = _load_attestation(pre_path, expected_phase="pre")
    _validate_static_record(pre, context=context, label="Pre attestation")
    _require(pre["pre_binding"] is None, "Pre attestation must not bind itself")
    if not require_post:
        pre_binding = _attestation_file_binding(
            pre_path, pre, repo_root=context["repo_root"]
        )
        return {
            "status": "verified",
            "run_id": context["manifest"]["run_id"],
            "pre_attestation_sha256": pre["canonical_payload_sha256"],
            "post_attestation_sha256": None,
            "pre_attestation": pre_binding,
            "post_attestation": None,
        }

    post_path = _require_repo_path(
        _attestation_path(context, "post"),
        repo_root=context["repo_root"],
        label="judge post attestation",
    )
    post = _load_attestation(post_path, expected_phase="post")
    _validate_static_record(post, context=context, label="Post attestation")
    _require(
        post["pre_binding"]
        == _attestation_file_binding(pre_path, pre, repo_root=context["repo_root"]),
        "Post attestation pre binding drifted",
    )
    _require(
        post["service_identity"]["stable_fields"]
        == pre["service_identity"]["stable_fields"],
        "Post judge service identity differs from pre",
    )
    pre_time = _validate_timestamp(pre["recorded_at_utc"])
    post_time = _validate_timestamp(post["recorded_at_utc"])
    _require(post_time >= pre_time, "Post attestation predates pre")
    pre_binding = _attestation_file_binding(
        pre_path, pre, repo_root=context["repo_root"]
    )
    post_binding = _attestation_file_binding(
        post_path, post, repo_root=context["repo_root"]
    )
    return {
        "status": "verified",
        "run_id": context["manifest"]["run_id"],
        "pre_attestation_sha256": pre["canonical_payload_sha256"],
        "post_attestation_sha256": post["canonical_payload_sha256"],
        "pre_attestation": pre_binding,
        "post_attestation": post_binding,
        "service_identity_sha256": post["service_identity"]["stable_fields_sha256"],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-manifest", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--phase", choices=PHASES)
    action.add_argument(
        "--verify-pre",
        action="store_true",
        help="Verify the immutable pre attestation without contacting the judge.",
    )
    action.add_argument(
        "--verify-pair",
        action="store_true",
        help="Verify the immutable pre/post pair without contacting the judge.",
    )
    args = parser.parse_args(argv)
    try:
        if args.verify_pre or args.verify_pair:
            result = verify_judge_attestations(
                args.run_manifest,
                repo_root=args.repo_root,
                require_post=bool(args.verify_pair),
            )
        else:
            result = record_judge_attestation(
                args.run_manifest,
                phase=args.phase,
                repo_root=args.repo_root,
            )
    except Exception as exc:  # noqa: BLE001 - CLI must fail closed as one JSON result
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
