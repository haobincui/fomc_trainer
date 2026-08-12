"""Fail-closed pre-training rollout pilot for retrain-v2 chk4.

The pilot uses only the sealed chk2 parent and the already-bound decision
validation split.  It never writes prompts or completions to its attestation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import yaml

from jobs.retrain_v2.dag import sha256_file, verify_parent
from open_r1.trainer.rewards.reward_funcs.decision_reward_v2 import (
    parse_decision_json,
)


PILOT_PROMPTS = 8
GENERATIONS_PER_PROMPT = 4
PILOT_COMPLETIONS = PILOT_PROMPTS * GENERATIONS_PER_PROMPT
MAX_PROMPT_TOKENS = 2560
MAX_COMPLETION_TOKENS = 512
MIN_VALID_COMPLETIONS = 20
MAX_TRUNCATION_RATE_EXCLUSIVE = 0.10
PILOT_SEED = 42


class DecisionPilotError(RuntimeError):
    """Raised when the immutable chk4 pilot contract cannot be satisfied."""


@dataclass(frozen=True)
class PilotCompletion:
    text: str
    completion_tokens: int
    truncated: bool


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DecisionPilotError(message)


def _canonical_bytes(payload: Any) -> bytes:
    try:
        value = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise DecisionPilotError("pilot payload is not canonical JSON") from exc
    return value.encode("utf-8")


def _canonical_sha256(payload: Any) -> str:
    return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _write_exclusive(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _require(not path.parent.is_symlink(), "pilot attestation directory is a symlink")
    _require(
        not path.exists() and not path.is_symlink(),
        f"pilot attestation already exists: {path}",
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
            raise DecisionPilotError(
                f"pilot attestation already exists: {path}"
            ) from exc
        directory_fd = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        temporary.unlink(missing_ok=True)


def _stable_row_id(row: Mapping[str, Any], index: int) -> str:
    for key in ("sample_id", "id", "meeting_id", "meeting_date"):
        value = row.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return f"validation-line-{index + 1}"


def select_pilot_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Select eight deterministic, unique validation prompts without logging text."""

    candidates: list[tuple[str, str, dict[str, Any]]] = []
    seen: set[str] = set()
    for index, source in enumerate(rows):
        row = dict(source)
        prompt = row.get("prompt")
        _require(isinstance(prompt, str) and prompt.strip(), "pilot row has no prompt")
        stable_id = _stable_row_id(row, index)
        identity = _canonical_sha256({"id": stable_id, "prompt": prompt})
        _require(identity not in seen, "pilot validation rows contain a duplicate")
        seen.add(identity)
        candidates.append((_canonical_sha256({"seed": PILOT_SEED, "id": identity}), identity, row))
    _require(
        len(candidates) >= PILOT_PROMPTS,
        f"chk4 pilot requires at least {PILOT_PROMPTS} validation rows",
    )
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [
        {"row_id_sha256": identity, "row": row}
        for _order, identity, row in candidates[:PILOT_PROMPTS]
    ]


def evaluate_pilot(
    selected: Sequence[Mapping[str, Any]],
    *,
    generate: Callable[[str, int], tuple[int, Sequence[PilotCompletion]]],
) -> dict[str, Any]:
    """Run and evaluate the pinned 8x4 pilot using an injected generator."""

    _require(len(selected) == PILOT_PROMPTS, "pilot selection must contain eight rows")
    records: list[dict[str, Any]] = []
    for prompt_index, item in enumerate(selected):
        row = item.get("row")
        row_hash = item.get("row_id_sha256")
        _require(isinstance(row, Mapping), "pilot selection row is invalid")
        _require(
            isinstance(row_hash, str) and len(row_hash) == 64,
            "pilot row hash is invalid",
        )
        prompt = row.get("prompt")
        _require(isinstance(prompt, str) and prompt, "pilot row prompt is invalid")
        prompt_tokens, completions = generate(prompt, PILOT_SEED + prompt_index)
        _require(
            isinstance(prompt_tokens, int)
            and not isinstance(prompt_tokens, bool)
            and 0 < prompt_tokens <= MAX_PROMPT_TOKENS,
            f"pilot prompt exceeds the {MAX_PROMPT_TOKENS}-token contract",
        )
        _require(
            len(completions) == GENERATIONS_PER_PROMPT,
            "pilot generator must return exactly four completions per prompt",
        )
        for generation_index, completion in enumerate(completions):
            _require(
                isinstance(completion, PilotCompletion),
                "pilot generator returned an invalid completion record",
            )
            _require(
                0 <= completion.completion_tokens <= MAX_COMPLETION_TOKENS,
                "pilot completion token count is invalid",
            )
            records.append(
                {
                    "row_id_sha256": row_hash,
                    "prompt_index": prompt_index,
                    "generation_index": generation_index,
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion.completion_tokens,
                    "truncated": completion.truncated,
                    "format_valid": parse_decision_json(completion.text) is not None,
                }
            )

    _require(
        len(records) == PILOT_COMPLETIONS,
        f"pilot must produce exactly {PILOT_COMPLETIONS} completions",
    )
    truncated = sum(record["truncated"] for record in records)
    valid = sum(record["format_valid"] for record in records)
    truncation_rate = truncated / len(records)
    passed = (
        truncation_rate < MAX_TRUNCATION_RATE_EXCLUSIVE
        and valid >= MIN_VALID_COMPLETIONS
    )
    return {
        "status": "passed" if passed else "failed",
        "contract": {
            "prompts": PILOT_PROMPTS,
            "generations_per_prompt": GENERATIONS_PER_PROMPT,
            "completions": PILOT_COMPLETIONS,
            "max_prompt_tokens": MAX_PROMPT_TOKENS,
            "max_completion_tokens": MAX_COMPLETION_TOKENS,
            "minimum_valid_completions": MIN_VALID_COMPLETIONS,
            "maximum_truncation_rate_exclusive": MAX_TRUNCATION_RATE_EXCLUSIVE,
            "seed": PILOT_SEED,
        },
        "summary": {
            "valid_completions": valid,
            "truncated_completions": truncated,
            "truncation_rate": truncation_rate,
        },
        "records": records,
    }


def _load_config(path: Path) -> dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise DecisionPilotError(f"unable to load chk4 config: {path}") from exc
    _require(isinstance(payload, dict), "chk4 config must be a mapping")
    expected = {
        "dataset_prompt_column": "prompt",
        "dataset_test_split": "validation",
        "load_in_4bit": True,
        "max_completion_length": MAX_COMPLETION_TOKENS,
        "num_generations": GENERATIONS_PER_PROMPT,
        "temperature": 0.7,
        "top_p": 0.9,
        "mask_truncated_completions": True,
    }
    for key, value in expected.items():
        _require(payload.get(key) == value, f"chk4 pilot config drifted at {key}")
    _require(
        isinstance(payload.get("system_prompt"), str) and payload["system_prompt"],
        "chk4 config has no system prompt",
    )
    return payload


def _find_validation_file(dataset_path: Path) -> Path:
    matches: list[Path] = []
    for pattern in ("*eval.jsonl", "*validation.jsonl", "*val.jsonl"):
        matches.extend(dataset_path.glob(pattern))
    unique = sorted(set(matches))
    _require(len(unique) == 1, "chk4 pilot requires exactly one validation JSONL")
    _require(not unique[0].is_symlink(), "chk4 pilot validation file is a symlink")
    return unique[0]


def _load_validation_rows(dataset_path: Path) -> list[dict[str, Any]]:
    path = _find_validation_file(dataset_path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            _require(line.strip() != "", f"blank validation row at line {line_number}")
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise DecisionPilotError(
                    f"invalid validation JSON at line {line_number}"
                ) from exc
            _require(isinstance(row, dict), "pilot validation row must be an object")
            rows.append(row)
    return rows


def _build_transformers_generator(
    *, model_path: Path, system_prompt: str
) -> Callable[[str, int], tuple[int, Sequence[PilotCompletion]]]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    _require(torch.cuda.is_available(), "chk4 pilot requires CUDA")
    _require(torch.cuda.device_count() == 1, "chk4 pilot must see exactly one policy GPU")
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        local_files_only=True,
        trust_remote_code=True,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",
        quantization_config=quantization,
        device_map={"": 0},
    )
    model.eval()
    eos_value = model.generation_config.eos_token_id
    eos_ids = (
        {int(value) for value in eos_value}
        if isinstance(eos_value, (list, tuple))
        else {int(eos_value if eos_value is not None else tokenizer.eos_token_id)}
    )

    def generate(prompt: str, seed: int) -> tuple[int, Sequence[PilotCompletion]]:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
        rendered = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=True,
        )
        _require(isinstance(rendered, str), "chk4 chat template did not render text")
        encoded = tokenizer(
            rendered,
            return_tensors="pt",
            add_special_tokens=False,
        )
        prompt_tokens = int(encoded["input_ids"].shape[-1])
        _require(
            prompt_tokens <= MAX_PROMPT_TOKENS,
            f"chk4 pilot prompt uses {prompt_tokens} tokens",
        )
        inputs = {key: value.to(model.device) for key, value in encoded.items()}
        generator = torch.Generator(device=model.device).manual_seed(seed)
        with torch.inference_mode():
            sequences = model.generate(
                **inputs,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                num_return_sequences=GENERATIONS_PER_PROMPT,
                max_new_tokens=MAX_COMPLETION_TOKENS,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=sorted(eos_ids),
                use_cache=True,
                generator=generator,
            )
        generated: list[PilotCompletion] = []
        for sequence in sequences:
            token_ids = [int(value) for value in sequence[prompt_tokens:].tolist()]
            eos_index = next(
                (index for index, token_id in enumerate(token_ids) if token_id in eos_ids),
                None,
            )
            truncated = eos_index is None and len(token_ids) >= MAX_COMPLETION_TOKENS
            content_ids = token_ids if eos_index is None else token_ids[:eos_index]
            while content_ids and content_ids[-1] == tokenizer.pad_token_id:
                content_ids.pop()
            generated.append(
                PilotCompletion(
                    text=tokenizer.decode(content_ids, skip_special_tokens=False),
                    completion_tokens=len(content_ids),
                    truncated=truncated,
                )
            )
        return prompt_tokens, generated

    return generate


def run_pilot(
    *, run_manifest: Path, repo_root: Path, config_path: Path
) -> dict[str, Any]:
    root = repo_root.resolve()
    manifest_path = run_manifest.resolve()
    verification = verify_parent(
        manifest_path,
        stage_id="chk4",
        repo_root=root,
        allow_stage_outputs=True,
    )
    config = _load_config(config_path.resolve())
    expected_config = Path(verification["resolved_config"]).resolve()
    _require(config_path.resolve() == expected_config, "chk4 pilot config is not pinned")
    model_path = Path(verification["parent"]["path"]).resolve()
    dataset_path = Path(verification["dataset"]["path"]).resolve()
    selected = select_pilot_rows(_load_validation_rows(dataset_path))
    generator = _build_transformers_generator(
        model_path=model_path,
        system_prompt=config["system_prompt"],
    )
    result = evaluate_pilot(selected, generate=generator)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    run_id = manifest.get("run_id")
    _require(isinstance(run_id, str) and run_id, "run manifest has no run_id")
    evidence = {
        "schema_version": 1,
        "attestation_type": "chk4_decision_rollout_pilot",
        "run_id": run_id,
        "stage_id": "chk4",
        "run_manifest_sha256": sha256_file(manifest_path),
        "config_sha256": sha256_file(config_path.resolve()),
        "parent_sha256": verification["parent"]["sha256"],
        "dataset_sha256": verification["dataset"]["artifact"]["sha256"],
        **result,
    }
    payload = {
        "recorded_at_utc": _utc_now(),
        "evidence": evidence,
        "evidence_sha256": _canonical_sha256(evidence),
    }
    output = manifest_path.parent / "attestations" / "chk4.rollout_pilot.json"
    _write_exclusive(output, payload)
    if result["status"] != "passed":
        raise DecisionPilotError(
            "chk4 rollout pilot failed: require truncation_rate < 0.10 and "
            "at least 20 valid final JSON completions"
        )
    return {"status": "passed", "path": str(output), **payload}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--config", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = run_pilot(
            run_manifest=args.run_manifest,
            repo_root=args.repo_root,
            config_path=args.config,
        )
    except Exception as exc:  # noqa: BLE001 - fail-closed command boundary
        print(
            json.dumps(
                {"status": "blocked", "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
