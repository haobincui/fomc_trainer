"""Fail-closed readiness probe for the retrain-v2 Qwen judge service."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import requests
import yaml

from jobs.retrain_v2.dag import verify_judge_artifact
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v2 import (
    _judge_one,
    build_judge_request,
    get_judge_tokenizer,
    render_judge_prompt_token_ids,
)


def _verify_loaded_model_root(value: object, *, expected: Path) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError("served judge model has no loaded-weight root attestation")
    candidate = Path(value)
    if not candidate.is_absolute():
        raise RuntimeError("served judge loaded-weight root must be absolute")
    if not candidate.exists() or not candidate.is_dir():
        raise RuntimeError(f"served judge loaded-weight root is missing: {candidate}")
    if candidate.is_symlink():
        raise RuntimeError("served judge loaded-weight root must not be a symlink")
    resolved = candidate.resolve(strict=True)
    if candidate != resolved:
        raise RuntimeError("served judge loaded-weight root contains a symlink or non-canonical path")
    expected_resolved = expected.resolve(strict=True)
    if resolved != expected_resolved:
        raise RuntimeError(
            f"served judge root {resolved} disagrees with immutable artifact {expected_resolved}"
        )
    return resolved


def check_judge(
    *,
    url: str,
    model: str,
    timeout: int,
    expected_model_root: Path | None = None,
    tokenizer_path: Path | None = None,
    max_model_len: int = 8192,
    max_completion_tokens: int = 2048,
    tokenizer: object | None = None,
) -> dict:
    base_url = url.split("/v1/chat/completions", 1)[0].rstrip("/")
    headers: dict[str, str] = {}
    api_key = os.environ.get("OPEN_R1_JUDGE_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    response = requests.get(f"{base_url}/v1/models", headers=headers, timeout=timeout)
    response.raise_for_status()
    all_cards = [
        item for item in response.json().get("data", []) if isinstance(item, dict)
    ]
    cards = [item for item in all_cards if item.get("id") == model]
    if len(all_cards) != 1 or len(cards) != 1:
        served = {
            item.get("id") for item in all_cards
        }
        raise RuntimeError(
            f"judge model {model!r} must have exactly one model card; found {sorted(served)!r}"
        )
    loaded_root = None
    if expected_model_root is not None:
        loaded_root = _verify_loaded_model_root(
            cards[0].get("root"), expected=expected_model_root
        )
    if cards[0].get("max_model_len") != max_model_len:
        raise RuntimeError(
            f"judge max_model_len={cards[0].get('max_model_len')!r} disagrees "
            f"with immutable value {max_model_len}"
        )

    golden_body = build_judge_request(
        evidence="indicator_x=4.0 percent; direction=unchanged",
        candidate="Indicator X was 4.0 percent and was unchanged.",
        model=model,
        max_completion_tokens=max_completion_tokens,
    )
    local_tokenizer = tokenizer
    if local_tokenizer is None:
        if tokenizer_path is None:
            raise RuntimeError("judge health requires the immutable tokenizer path")
        local_tokenizer = get_judge_tokenizer(tokenizer_path)
    local_token_ids = render_judge_prompt_token_ids(golden_body, local_tokenizer)
    local_count = len(local_token_ids)
    tokenize_body = {
        "model": model,
        "messages": golden_body["messages"],
        "add_generation_prompt": True,
        "add_special_tokens": False,
        "chat_template_kwargs": golden_body["chat_template_kwargs"],
    }
    tokenized = requests.post(
        f"{base_url}/tokenize",
        headers=headers,
        json=tokenize_body,
        timeout=timeout,
    )
    tokenized.raise_for_status()
    tokenized_payload = tokenized.json()
    if not isinstance(tokenized_payload, dict):
        raise RuntimeError("judge /tokenize response must be an object")
    server_count = tokenized_payload.get("count")
    if isinstance(server_count, bool) or not isinstance(server_count, int):
        raise RuntimeError("judge /tokenize response has no integer count")
    server_token_ids = tokenized_payload.get("tokens")
    if not isinstance(server_token_ids, list) or any(
        isinstance(token_id, bool)
        or not isinstance(token_id, int)
        or token_id < 0
        for token_id in server_token_ids
    ):
        raise RuntimeError("judge /tokenize response has no valid token ID sequence")
    if server_count != len(server_token_ids):
        raise RuntimeError("judge /tokenize count disagrees with its token ID sequence")
    server_max_model_len = tokenized_payload.get("max_model_len")
    if server_max_model_len != max_model_len:
        raise RuntimeError(
            f"judge /tokenize max_model_len={server_max_model_len!r} disagrees "
            f"with immutable value {max_model_len}"
        )
    if server_count != local_count or server_token_ids != local_token_ids:
        raise RuntimeError(
            "judge tokenizer parity mismatch: local and server token ID sequences differ "
            f"(local_count={local_count}, server_count={server_count})"
        )

    evaluation, _raw, attempts = _judge_one(
        evidence="indicator_x=4.0 percent; direction=unchanged",
        candidate="Indicator X was 4.0 percent and was unchanged.",
        url=url,
        model=model,
        timeout=timeout,
        api_key=api_key,
        max_retries=1,
        backoff_seconds=0,
        max_completion_tokens=max_completion_tokens,
        body=golden_body,
    )
    return {
        "status": "ready",
        "url": url,
        "model": model,
        "attempts": attempts,
        "rubric_keys": sorted(evaluation),
        "weight_attested": loaded_root is not None,
        "loaded_model_root": str(loaded_root) if loaded_root is not None else None,
        "max_model_len": max_model_len,
        "max_completion_tokens": max_completion_tokens,
        "golden_prompt_tokens": local_count,
        "tokenizer_parity": True,
    }


def _judge_contract_from_config(path: Path) -> tuple[str, str, int, Path, int, int]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"unable to load resolved judge config: {path}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"resolved judge config must be a mapping: {path}")
    url = payload.get("judge_url")
    model = payload.get("judge_model")
    timeout = payload.get("judge_timeout", 60)
    tokenizer_path = payload.get("judge_tokenizer_path")
    max_model_len = payload.get("judge_max_model_len")
    max_completion_tokens = payload.get("judge_max_completion_tokens")
    if not isinstance(url, str) or not url:
        raise RuntimeError("resolved chk2 config has no judge_url")
    if not isinstance(model, str) or not model:
        raise RuntimeError("resolved chk2 config has no judge_model")
    if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
        raise RuntimeError("resolved chk2 judge_timeout must be a positive integer")
    if not isinstance(tokenizer_path, str) or not tokenizer_path:
        raise RuntimeError("resolved chk2 config has no judge_tokenizer_path")
    resolved_tokenizer_path = Path(tokenizer_path)
    if not resolved_tokenizer_path.is_absolute():
        resolved_tokenizer_path = (Path.cwd() / resolved_tokenizer_path).absolute()
    if (
        not resolved_tokenizer_path.is_dir()
        or resolved_tokenizer_path != resolved_tokenizer_path.resolve()
    ):
        raise RuntimeError("resolved judge tokenizer path is missing or non-canonical")
    for name, value in (
        ("judge_max_model_len", max_model_len),
        ("judge_max_completion_tokens", max_completion_tokens),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise RuntimeError(f"resolved chk2 {name} must be a positive integer")
    for env_name, expected in (
        ("OPEN_R1_JUDGE_URL", url),
        ("OPEN_R1_JUDGE_MODEL", model),
        ("FOMC_RETRAIN_JUDGE_MODEL_PATH", str(resolved_tokenizer_path)),
        ("FOMC_RETRAIN_JUDGE_MAX_MODEL_LEN", str(max_model_len)),
    ):
        override = os.environ.get(env_name)
        if override is not None and override != expected:
            raise RuntimeError(
                f"{env_name}={override!r} disagrees with immutable config value {expected!r}"
            )
    return (
        url,
        model,
        timeout,
        resolved_tokenizer_path,
        max_model_len,
        max_completion_tokens,
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.environ.get(
            "OPEN_R1_JUDGE_URL", "http://127.0.0.1:8000/v1/chat/completions"
        ),
    )
    parser.add_argument(
        "--model", default=os.environ.get("OPEN_R1_JUDGE_MODEL", "Qwen3.5-9B")
    )
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument(
        "--config",
        type=Path,
        help="Read the immutable judge URL/model/timeout from a resolved chk2 YAML.",
    )
    parser.add_argument(
        "--run-manifest",
        type=Path,
        help="Attest the served model root against this immutable run manifest.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    args = parser.parse_args()
    try:
        url, model, timeout, tokenizer_path, max_model_len, max_completion_tokens = (
            _judge_contract_from_config(args.config.resolve())
            if args.config is not None
            else (
                args.url,
                args.model,
                args.timeout,
                Path("models/Qwen3.5-9B").resolve(),
                8192,
                2048,
            )
        )
        expected_model_root = None
        if args.run_manifest is not None:
            attestation = verify_judge_artifact(
                args.run_manifest.resolve(), repo_root=args.repo_root.resolve()
            )
            expected_contract = (
                attestation["url"],
                attestation["served_model_name"],
                attestation["timeout"],
                Path(attestation["artifact"]["path"]),
                attestation["max_model_len"],
                attestation["max_completion_tokens"],
            )
            if (
                url,
                model,
                timeout,
                tokenizer_path,
                max_model_len,
                max_completion_tokens,
            ) != expected_contract:
                raise RuntimeError(
                    "resolved judge config disagrees with immutable run manifest contract"
                )
            expected_model_root = Path(attestation["artifact"]["path"])
        result = check_judge(
            url=url,
            model=model,
            timeout=timeout,
            expected_model_root=expected_model_root,
            tokenizer_path=tokenizer_path,
            max_model_len=max_model_len,
            max_completion_tokens=max_completion_tokens,
        )
    except Exception as exc:  # noqa: BLE001 - command must convert all failures to a gate
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
