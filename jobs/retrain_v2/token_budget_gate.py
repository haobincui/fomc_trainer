"""Fail-closed, tokenizer-exact token budget gate for retrain-v2 stages.

The CLI derives its limits from the DAG pinned by a run manifest and uses the
resolved stage config plus the sealed parent's tokenizer.  It intentionally
does not rely on a Trainer truncation setting: every train/validation JSONL row
must fit before a training process is launched.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Mapping

import yaml

from jobs.retrain_v2.dag import (
    DagValidationError,
    stage_config_path,
    verify_parent,
)
from open_r1.provenance import sha256_file
from open_r1.trainer.prompt_contract import compose_user_prompt
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v2 import (
    JudgeContextBudgetError,
    build_judge_request,
    count_judge_prompt_tokens,
    get_judge_tokenizer,
    validate_judge_evidence,
)
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v3 import (
    build_judge_request_v3,
)


STAGE_IDS = ("chk1", "chk2", "chk3", "chk4")
SFT_KINDS = {"analysis_sft", "minutes_sft"}
GRPO_KINDS = {"analysis_grpo", "decision_grpo"}


class TokenBudgetError(ValueError):
    """Raised when token-budget provenance or a dataset row is invalid."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise TokenBudgetError(message)


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise TokenBudgetError(f"Unable to parse {label}: {path}") from exc
    _require(isinstance(payload, dict), f"{label} must contain a JSON object: {path}")
    return payload


def _load_yaml_object(path: Path, *, label: str) -> dict[str, Any]:
    _require(path.is_file(), f"{label} does not exist: {path}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise TokenBudgetError(f"Unable to parse {label}: {path}") from exc
    _require(isinstance(payload, dict), f"{label} must contain a mapping: {path}")
    return payload


def _repo_path(repo_root: Path, value: str, *, label: str) -> Path:
    candidate = Path(value)
    resolved = candidate.resolve() if candidate.is_absolute() else (repo_root / candidate).resolve()
    try:
        resolved.relative_to(repo_root)
    except ValueError as exc:
        raise TokenBudgetError(f"{label} escapes the repository: {value}") from exc
    return resolved


def load_stage_contract(
    run_manifest: str | Path,
    *,
    stage_id: str,
    repo_root: str | Path,
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    """Load a budget and resolved config bound by the pinned run manifest."""

    _require(stage_id in STAGE_IDS, f"Unsupported training stage: {stage_id}")
    root = Path(repo_root).resolve()
    manifest_path = Path(run_manifest).resolve()
    manifest = _load_json_object(manifest_path, label="run manifest")
    _require(manifest.get("schema_version") == 2, "Unsupported run manifest schema")

    run_id = str(manifest.get("run_id") or "")
    expected_manifest = root / "output" / "training" / "retrain_v2" / run_id / "run_manifest.json"
    _require(
        manifest_path == expected_manifest.resolve(),
        f"Run manifest is not at its immutable run path: {expected_manifest}",
    )

    dag_record = manifest.get("dag")
    _require(isinstance(dag_record, dict), "Run manifest has no pinned DAG record")
    dag_path = _repo_path(root, str(dag_record.get("path") or ""), label="pinned DAG")
    expected_dag_sha = str(dag_record.get("sha256") or "")
    _require(
        len(expected_dag_sha) == 64 and sha256_file(dag_path) == expected_dag_sha,
        "Pinned DAG hash mismatch",
    )
    dag = _load_yaml_object(dag_path, label="pinned DAG")
    _require(dag.get("schema_version") == 2, "Unsupported pinned DAG schema")

    stage_definition = dag.get("stages", {}).get(stage_id)
    _require(isinstance(stage_definition, dict), f"Pinned DAG has no {stage_id} stage")
    stage_kind = stage_definition.get("kind")
    _require(
        stage_kind in SFT_KINDS | GRPO_KINDS,
        f"Unsupported stage kind for token audit: {stage_kind!r}",
    )

    budget = dag.get("token_budgets", {}).get(stage_id)
    _require(isinstance(budget, dict), f"Pinned DAG has no token budget for {stage_id}")
    _require(budget.get("overflow_policy") == "error", f"{stage_id} overflow policy must be error")
    for key in ("prompt", "completion"):
        value = budget.get(key)
        if stage_kind in SFT_KINDS and key == "completion" and value is None:
            continue
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"{stage_id} budget.{key} must be a positive integer",
        )
    if stage_kind in SFT_KINDS:
        total = budget.get("total")
        _require(
            isinstance(total, int) and not isinstance(total, bool) and total > 0,
            f"{stage_id} budget.total must be a positive integer",
        )
        if budget.get("completion") is not None:
            _require(
                budget["prompt"] + budget["completion"] == total,
                f"{stage_id} prompt+completion budgets must equal total",
            )

    stage_record = manifest.get("stages", {}).get(stage_id)
    _require(isinstance(stage_record, dict), f"Run manifest has no {stage_id} record")
    config_record = stage_record.get("resolved_config")
    _require(isinstance(config_record, dict), f"{stage_id} has no bound resolved config")
    config_path = _repo_path(
        root,
        str(config_record.get("path") or ""),
        label=f"{stage_id} resolved config",
    )
    _require(config_path.is_file(), f"Resolved config does not exist: {config_path}")
    _require(
        sha256_file(config_path) == config_record.get("sha256"),
        f"Resolved config hash mismatch for {stage_id}",
    )
    config = _load_yaml_object(config_path, label=f"{stage_id} resolved config")
    if stage_kind in SFT_KINDS:
        _require(
            config.get("max_length") == budget["total"],
            f"{stage_id} config max_length does not match its pinned total budget",
        )
    else:
        _require(
            config.get("max_completion_length") == budget["completion"],
            f"{stage_id} config max_completion_length does not match its pinned budget",
        )
        _require(
            config.get("chat_template_kwargs", {}).get("truncation") is False,
            f"{stage_id} config must explicitly disable prompt truncation",
        )
        if stage_id == "chk2":
            judge = manifest.get("judge")
            _require(isinstance(judge, dict), "Run manifest has no judge binding")
            for config_key, manifest_key in (
                ("judge_tokenizer_path", "tokenizer_path"),
                ("judge_max_model_len", "max_model_len"),
                ("judge_max_completion_tokens", "max_completion_tokens"),
                ("judge_candidate_reserve_tokens", "candidate_reserve_tokens"),
                ("judge_boundary_margin_tokens", "boundary_margin_tokens"),
            ):
                _require(
                    config.get(config_key) == judge.get(manifest_key),
                    f"chk2 {config_key} disagrees with immutable run manifest",
                )
    config["_stage_kind"] = stage_kind
    return dict(budget), config, config_path


def _find_split_files(dataset_path: Path) -> dict[str, Path]:
    patterns = {
        "train": ("*train.jsonl",),
        "validation": ("*eval.jsonl", "*validation.jsonl", "*val.jsonl"),
    }
    result: dict[str, Path] = {}
    _require(dataset_path.is_dir(), f"Dataset directory does not exist: {dataset_path}")
    for split, globs in patterns.items():
        matches = sorted(
            {
                candidate.resolve()
                for pattern in globs
                for candidate in dataset_path.glob(pattern)
                if candidate.is_file()
            }
        )
        _require(
            len(matches) == 1,
            f"{dataset_path} must contain exactly one {split} JSONL; found {len(matches)}",
        )
        result[split] = matches[0]
    return result


def _token_ids(value: Any) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, list) and value and isinstance(value[0], list):
        _require(len(value) == 1, "Tokenizer unexpectedly returned a batched result")
        value = value[0]
    _require(isinstance(value, list), "Tokenizer did not return a token ID list")
    return value


def _messages(system_prompt: str | None, user_prompt: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system_prompt is not None:
        _require(isinstance(system_prompt, str), "system_prompt must be a string or null")
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_prompt})
    return messages


def _sample_identifier(row: Mapping[str, Any], *, split: str, line_number: int) -> str:
    """Return only a non-content identifier suitable for the audit report."""

    for key in ("sample_id", "id"):
        value = row.get(key)
        if isinstance(value, (str, int)) and not isinstance(value, bool):
            normalized = str(value).strip()
            if normalized:
                return normalized
    return f"{split}:{line_number}"


def _count_grpo_row(
    row: Mapping[str, Any],
    *,
    tokenizer: Any,
    config: Mapping[str, Any],
) -> tuple[int, None, int]:
    prompt_column = str(config.get("dataset_prompt_column") or "prompt")
    _require(prompt_column in row, f"missing required column {prompt_column!r}")
    user_prompt = row[prompt_column]
    _require(isinstance(user_prompt, str) and user_prompt != "", f"{prompt_column!r} must be a non-empty string")
    try:
        user_prompt = compose_user_prompt(
            user_prompt, config.get("user_prompt_suffix")
        )
    except ValueError as exc:
        raise TokenBudgetError(str(exc)) from exc
    kwargs = dict(config.get("chat_template_kwargs") or {})
    tokenized = tokenizer.apply_chat_template(
        _messages(config.get("system_prompt"), user_prompt),
        tokenize=True,
        add_generation_prompt=True,
        return_dict=False,
        **kwargs,
    )
    prompt_tokens = len(_token_ids(tokenized))
    return prompt_tokens, None, prompt_tokens


def _count_sft_row(
    row: Mapping[str, Any],
    *,
    tokenizer: Any,
    config: Mapping[str, Any],
) -> tuple[int, int, int]:
    prompt_column = str(config.get("dataset_prompt_column") or "prompt")
    _require(prompt_column in row, f"missing required column {prompt_column!r}")
    _require("response" in row, "missing required column 'response'")
    user_prompt = row[prompt_column]
    response = row["response"]
    _require(isinstance(user_prompt, str) and user_prompt != "", f"{prompt_column!r} must be a non-empty string")
    _require(isinstance(response, str) and response != "", "'response' must be a non-empty string")

    prompt_text = tokenizer.apply_chat_template(
        _messages(config.get("system_prompt"), user_prompt),
        tokenize=False,
        add_generation_prompt=True,
    )
    _require(isinstance(prompt_text, str), "Chat template did not render a prompt string")
    eos_token = getattr(tokenizer, "eos_token", None)
    _require(isinstance(eos_token, str) and eos_token != "", "Tokenizer has no EOS token")
    completion_text = response if response.endswith(eos_token) else response + eos_token

    prompt_ids = _token_ids(tokenizer(text=prompt_text))
    full_ids = _token_ids(tokenizer(text=prompt_text + completion_text))
    _require(
        full_ids[: len(prompt_ids)] == prompt_ids,
        "Tokenized prompt is not a prefix of prompt+response; SFT completion mask would be unsafe",
    )
    prompt_tokens = len(prompt_ids)
    total_tokens = len(full_ids)
    return prompt_tokens, total_tokens - prompt_tokens, total_tokens


def audit_dataset(
    *,
    dataset_path: str | Path,
    tokenizer: Any,
    config: Mapping[str, Any],
    budget: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit train/validation JSONL rows and return non-content token records."""

    dataset = Path(dataset_path).resolve()
    split_files = _find_split_files(dataset)
    stage_kind = config.get("_stage_kind")
    _require(stage_kind in SFT_KINDS | GRPO_KINDS, f"Unsupported stage kind: {stage_kind!r}")
    records: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}

    for split, path in split_files.items():
        row_count = 0
        max_prompt = 0
        max_completion = 0
        max_total = 0
        try:
            handle = path.open("r", encoding="utf-8")
        except OSError as exc:
            raise TokenBudgetError(f"Unable to open {split} JSONL: {path}") from exc
        with handle:
            for line_number, raw_line in enumerate(handle, start=1):
                _require(raw_line.strip() != "", f"{path}:{line_number}: blank JSONL row")
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise TokenBudgetError(f"{path}:{line_number}: invalid JSON") from exc
                _require(isinstance(row, dict), f"{path}:{line_number}: row must be an object")
                try:
                    if stage_kind in GRPO_KINDS:
                        prompt_tokens, completion_tokens, total_tokens = _count_grpo_row(
                            row, tokenizer=tokenizer, config=config
                        )
                    else:
                        prompt_tokens, completion_tokens, total_tokens = _count_sft_row(
                            row, tokenizer=tokenizer, config=config
                        )
                except TokenBudgetError as exc:
                    raise TokenBudgetError(f"{path}:{line_number}: {exc}") from exc

                _require(
                    prompt_tokens <= budget["prompt"],
                    f"{path}:{line_number}: prompt uses {prompt_tokens} tokens; budget is {budget['prompt']}",
                )
                if completion_tokens is not None:
                    completion_budget = budget.get("completion")
                    if completion_budget is not None:
                        _require(
                            completion_tokens <= completion_budget,
                            f"{path}:{line_number}: completion uses {completion_tokens} tokens; budget is {completion_budget}",
                        )
                    _require(
                        total_tokens <= budget["total"],
                        f"{path}:{line_number}: total uses {total_tokens} tokens; budget is {budget['total']}",
                    )

                record = {
                    "sample_id": _sample_identifier(
                        row, split=split, line_number=line_number
                    ),
                    "split": split,
                    "line": line_number,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                }
                records.append(record)
                row_count += 1
                max_prompt = max(max_prompt, prompt_tokens)
                max_completion = max(max_completion, completion_tokens or 0)
                max_total = max(max_total, total_tokens)

        _require(row_count > 0, f"{path}: split contains no rows")
        summaries[split] = {
            "path": str(path),
            "rows": row_count,
            "max_prompt_tokens": max_prompt,
            "max_completion_tokens": max_completion if stage_kind in SFT_KINDS else None,
            "max_total_tokens": max_total,
        }

    return {
        "status": "passed",
        "stage_kind": stage_kind,
        "budget": dict(budget),
        "splits": summaries,
        "rows": records,
    }


def audit_chk2_judge_baseline(
    *,
    dataset_path: str | Path,
    tokenizer: Any,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Audit known evidence with an empty candidate plus a conservative reserve."""

    dataset = Path(dataset_path).resolve()
    split_files = _find_split_files(dataset)
    model = config.get("judge_model")
    _require(isinstance(model, str) and model, "chk2 config has no judge_model")
    # Direct unit callers and historical v2 receipts predate this discriminator;
    # absent metadata therefore resolves to the immutable v2 request contract.
    reward_funcs = config.get("reward_funcs", ["grounded_analysis_v2"])
    _require(
        reward_funcs in (["grounded_analysis_v2"], ["grounded_analysis_v3"]),
        "chk2 config has an unsupported grounded-analysis reward",
    )
    request_builder = (
        build_judge_request_v3
        if reward_funcs == ["grounded_analysis_v3"]
        else build_judge_request
    )
    numeric_contract: dict[str, int] = {}
    for key in (
        "judge_max_model_len",
        "judge_max_completion_tokens",
        "judge_candidate_reserve_tokens",
        "judge_boundary_margin_tokens",
    ):
        value = config.get(key)
        _require(
            isinstance(value, int) and not isinstance(value, bool) and value > 0,
            f"chk2 {key} must be a positive integer",
        )
        numeric_contract[key] = value

    records: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for split, path in split_files.items():
        max_baseline = 0
        max_reserved_total = 0
        row_count = 0
        with path.open("r", encoding="utf-8") as handle:
            for line_number, raw_line in enumerate(handle, start=1):
                _require(raw_line.strip() != "", f"{path}:{line_number}: blank JSONL row")
                try:
                    row = json.loads(raw_line)
                except json.JSONDecodeError as exc:
                    raise TokenBudgetError(
                        f"{path}:{line_number}: invalid JSON"
                    ) from exc
                _require(
                    isinstance(row, dict), f"{path}:{line_number}: row must be an object"
                )
                evidence = row.get("provided_data")
                _require(
                    isinstance(evidence, str) and evidence != "",
                    f"{path}:{line_number}: missing non-empty provided_data",
                )
                try:
                    evidence = validate_judge_evidence(evidence)
                except ValueError as exc:
                    raise TokenBudgetError(f"{path}:{line_number}: {exc}") from exc
                body = request_builder(
                    evidence=evidence,
                    candidate="",
                    model=model,
                    max_completion_tokens=numeric_contract[
                        "judge_max_completion_tokens"
                    ],
                )
                baseline = count_judge_prompt_tokens(body, tokenizer)
                reserved_total = (
                    baseline
                    + numeric_contract["judge_candidate_reserve_tokens"]
                    + numeric_contract["judge_max_completion_tokens"]
                    + numeric_contract["judge_boundary_margin_tokens"]
                )
                _require(
                    reserved_total <= numeric_contract["judge_max_model_len"],
                    f"{path}:{line_number}: judge baseline reserve uses "
                    f"{reserved_total} tokens; model limit is "
                    f"{numeric_contract['judge_max_model_len']}",
                )
                records.append(
                    {
                        "sample_id": _sample_identifier(
                            row, split=split, line_number=line_number
                        ),
                        "split": split,
                        "line": line_number,
                        "empty_candidate_prompt_tokens": baseline,
                        "conservative_reserved_total_tokens": reserved_total,
                    }
                )
                row_count += 1
                max_baseline = max(max_baseline, baseline)
                max_reserved_total = max(max_reserved_total, reserved_total)
        _require(row_count > 0, f"{path}: split contains no rows")
        summaries[split] = {
            "path": str(path),
            "rows": row_count,
            "max_empty_candidate_prompt_tokens": max_baseline,
            "max_conservative_reserved_total_tokens": max_reserved_total,
        }
    return {
        "status": "passed",
        "runtime_candidate_gate_required": True,
        "guarantees_unknown_candidate_fit": False,
        "contract": numeric_contract,
        "splits": summaries,
        "rows": records,
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def run_gate(
    run_manifest: str | Path,
    *,
    stage_id: str,
    repo_root: str | Path,
) -> dict[str, Any]:
    root = Path(repo_root).resolve()
    manifest_path = Path(run_manifest).resolve()

    # This re-hashes the sealed parent, bound dataset, and resolved config.  The
    # launcher also runs the same check before --execute is considered, but the
    # gate remains independently fail-closed when invoked directly.
    verify_parent(manifest_path, stage_id=stage_id, repo_root=root)
    budget, config, config_path = load_stage_contract(
        manifest_path, stage_id=stage_id, repo_root=root
    )
    _require(stage_config_path(manifest_path, stage_id=stage_id) == config_path, "Resolved config path mismatch")

    dataset_path = _repo_path(root, str(config.get("dataset_name") or ""), label="dataset")
    model_path = _repo_path(root, str(config.get("model_name_or_path") or ""), label="tokenizer model")
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        str(model_path),
        revision=config.get("model_revision"),
        trust_remote_code=bool(config.get("trust_remote_code", False)),
        local_files_only=True,
        use_fast=True,
    )
    if config.get("chat_template") is not None:
        tokenizer.chat_template = config["chat_template"]

    result = audit_dataset(
        dataset_path=dataset_path,
        tokenizer=tokenizer,
        config=config,
        budget=budget,
    )
    if stage_id == "chk2":
        judge_tokenizer_path = _repo_path(
            root,
            str(config.get("judge_tokenizer_path") or ""),
            label="judge tokenizer",
        )
        judge_tokenizer = get_judge_tokenizer(judge_tokenizer_path)
        result["judge_context_baseline"] = audit_chk2_judge_baseline(
            dataset_path=dataset_path,
            tokenizer=judge_tokenizer,
            config=config,
        )
    result.update(
        {
            "stage": stage_id,
            "run_manifest": str(manifest_path),
            "resolved_config": str(config_path),
            "tokenizer_path": str(model_path),
        }
    )
    report_path = manifest_path.parent / "preflight" / f"{stage_id}_token_budget.json"
    _write_json_atomic(report_path, result)
    result["report"] = str(report_path)
    return result


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGE_IDS, required=True)
    args = parser.parse_args(argv)
    try:
        result = run_gate(
            args.run_manifest,
            stage_id=args.stage,
            repo_root=args.repo_root,
        )
    except (
        DagValidationError,
        JudgeContextBudgetError,
        OSError,
        TokenBudgetError,
        ValueError,
    ) as exc:
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
