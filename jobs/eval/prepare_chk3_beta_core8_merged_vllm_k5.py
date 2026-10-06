"""Freeze the exact token-input ledger for the merged Core8 vLLM K=5 run.

The existing N=2,048 harmonized sample manifest remains the source-of-truth
for row identity and source lineage.  Its historical K=10 generation-design
block is deliberately treated as transport metadata only.  This preparer
creates a new K=5 cohort and, critically, materializes the complete prompt
token IDs produced by the bound FOMC tokenizer.  The vLLM runner consumes
those IDs directly and never reapplies a chat template at generation time.
"""

from __future__ import annotations

import argparse
import copy
import json
import platform
from collections.abc import Mapping, Sequence
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

from jobs.eval import chk3_beta_core8_merged_contract as contract
from jobs.eval import eval_chk3_beta_core8_merged_stochastic_k10 as k10_profile
from jobs.eval import eval_chk3_stochastic_bootstrap_generation as core
from jobs.eval.prepare_chk3_external_holdout_smoke import (
    _binding,
    _write_new_json,
    _write_new_jsonl,
)
from open_r1.validator.loo_generation_spec import seal_manifest


ROOT = Path(__file__).resolve().parents[2]
EVALUATION_ID = "chk3-beta-core8-merged-1993-2025-n2048-vllm-k5-v1"
COHORT_SCHEMA = "chk3-beta-core8-merged-vllm-k5-cohort-v1"
LEDGER_ROW_SCHEMA = "chk3-beta-core8-merged-vllm-prompt-token-ledger-row-v1"
PREPARATION_SCHEMA = "chk3-beta-core8-merged-vllm-k5-preparation-v1"
REPLICATE_SEEDS = (
    20260811,
    21260811,
    22260811,
    23260811,
    24260811,
)
MODEL_ORDER = ("chk1", "chk3", "chk0")
ABSOLUTE_CHUNK_SIZE = 40
EXPECTED_PROMPTS = contract.EXPECTED_ROWS
EXPECTED_CASES_PER_MODEL = EXPECTED_PROMPTS * len(REPLICATE_SEEDS)
EXPECTED_TOTAL_CASES = EXPECTED_CASES_PER_MODEL * len(MODEL_ORDER)


class VllmK5PreparationError(RuntimeError):
    """The exact-token K=5 preparation contract was violated."""


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _sha256_text(value: str) -> str:
    return core._sha256_text(value)


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _token_id_sha256(token_ids: Sequence[int]) -> str:
    return _sha256_text(_canonical(list(token_ids)))


def build_token_ledger(
    *,
    sample_manifest: Mapping[str, Any],
    bound_rows: Mapping[str, Mapping[str, Any]],
    tokenizer: Any,
    system_prompt: str,
    user_prompt_suffix: str | None,
) -> list[dict[str, Any]]:
    """Materialize one exact chat-template token sequence per frozen prompt."""

    samples = sample_manifest.get("samples")
    if not isinstance(samples, list) or len(samples) != EXPECTED_PROMPTS:
        raise VllmK5PreparationError("source sample inventory is not N=2,048")
    rows: list[dict[str, Any]] = []
    for absolute_prompt_index, sample in enumerate(samples):
        if not isinstance(sample, Mapping):
            raise VllmK5PreparationError("source sample is not an object")
        sample_id = str(sample.get("sample_id") or "")
        source = bound_rows.get(sample_id)
        if not sample_id or not isinstance(source, Mapping):
            raise VllmK5PreparationError(
                f"source row is missing for sample {sample_id!r}"
            )
        try:
            messages = core.native_eval.native_probe._messages(
                source,
                system_prompt=system_prompt,
                user_prompt_suffix=user_prompt_suffix,
            )
            prompt_token_ids = core.native_eval.native_probe._prompt_ids(
                tokenizer, messages
            )
        except core.native_eval.native_probe.Chk3ProbeError as exc:
            raise VllmK5PreparationError(str(exc)) from exc
        if (
            not prompt_token_ids
            or any(
                not isinstance(value, int) or isinstance(value, bool) or value < 0
                for value in prompt_token_ids
            )
            or len(prompt_token_ids) != sample.get("prompt_token_count")
        ):
            raise VllmK5PreparationError(f"prompt-token contract drift for {sample_id}")
        source_prompt = str(source.get("prompt") or "")
        if _sha256_text(source_prompt) != sample.get("prompt_sha256"):
            raise VllmK5PreparationError(f"source prompt hash drift for {sample_id}")
        rows.append(
            {
                "schema_version": LEDGER_ROW_SCHEMA,
                "absolute_prompt_index": absolute_prompt_index,
                "source_sample_line_number": absolute_prompt_index + 1,
                "sample_id": sample_id,
                "meeting_id": sample.get("meeting_id"),
                "prompt_sha256": sample.get("prompt_sha256"),
                "messages_sha256": _sha256_text(_canonical(messages)),
                "prompt_token_count": len(prompt_token_ids),
                "prompt_token_ids_sha256": _token_id_sha256(prompt_token_ids),
                "prompt_token_ids": list(prompt_token_ids),
            }
        )
    return rows


def prepare(
    *,
    source_sample_manifest_path: Path,
    source_sample_manifest_sha256: str,
    pre_release_manifest: Path,
    post_release_manifest: Path,
    pre_release_sha256: str | None,
    post_release_sha256: str,
    output_dir: Path,
) -> dict[str, Any]:
    """Create an immutable K=5 cohort beside, not over, the K=10 artifacts."""

    unresolved_output = output_dir.expanduser()
    if unresolved_output.is_symlink():
        raise VllmK5PreparationError("refusing a symlink output directory")
    output_dir = unresolved_output.resolve()
    if output_dir.exists():
        raise VllmK5PreparationError(f"output already exists: {output_dir}")

    try:
        sources = k10_profile.configure_profile(
            pre_release_manifest=pre_release_manifest,
            post_release_manifest=post_release_manifest,
            pre_release_sha256=pre_release_sha256,
            post_release_sha256=post_release_sha256,
        )
        sample_manifest, observed_sample_sha = core.load_full_test_sample_manifest(
            source_sample_manifest_path, source_sample_manifest_sha256
        )
        bound_rows = core._load_bound_rows(sample_manifest)
    except (
        contract.MergedCore8ContractError,
        core.StochasticBootstrapGenerationError,
    ) as exc:
        raise VllmK5PreparationError(str(exc)) from exc

    prompt_contract = sample_manifest.get("prompt_contract")
    tokenizer_binding = sample_manifest.get("tokenizer")
    if not isinstance(prompt_contract, Mapping) or not isinstance(
        tokenizer_binding, Mapping
    ):
        raise VllmK5PreparationError("source prompt/tokenizer contract is missing")
    tokenizer_path = Path(str(tokenizer_binding.get("path"))).expanduser().resolve()
    config_path = (
        Path(str(prompt_contract.get("training_config"))).expanduser().resolve()
    )
    try:
        system_prompt, suffix, config_sha = (
            core.native_eval.native_probe._load_prompt_contract(config_path)
        )
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(tokenizer_path), local_files_only=True, trust_remote_code=True
        )
        observed_tokenizer = dict(
            core.native_eval.native_probe._tokenizer_fingerprint(tokenizer_path)
        )
        tokenizer_semantics = core.native_eval._tokenizer_semantics(tokenizer)
    except (core.native_eval.native_probe.Chk3ProbeError, OSError, ValueError) as exc:
        raise VllmK5PreparationError(str(exc)) from exc
    if config_sha != prompt_contract.get("training_config_sha256"):
        raise VllmK5PreparationError("training prompt contract changed")
    if observed_tokenizer != dict(tokenizer_binding):
        raise VllmK5PreparationError("FOMC tokenizer file inventory changed")

    ledger_rows = build_token_ledger(
        sample_manifest=sample_manifest,
        bound_rows=bound_rows,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        user_prompt_suffix=suffix,
    )
    ledger_path = output_dir / "prompt_token_ledger_n2048.v1.jsonl"
    _write_new_jsonl(ledger_path, ledger_rows)
    ledger_binding = _binding(ledger_path, rows=EXPECTED_PROMPTS)

    cohort_payload = {
        "schema_version": COHORT_SCHEMA,
        "status": "complete",
        "evaluation_id": EVALUATION_ID,
        "immutable": True,
        "task_contract_id": core.TASK_CONTRACT_ID,
        "research_scope": k10_profile.RESEARCH_SCOPE,
        "not_all_held_out": True,
        "source_sample_manifest": {
            "path": str(source_sample_manifest_path.expanduser().resolve()),
            "sha256": observed_sample_sha,
            "payload_sha256": sample_manifest["integrity"]["payload_sha256"],
            "transport_generation_design_ignored": True,
        },
        "harmonized_source_releases": copy.deepcopy(dict(sources.source_bindings)),
        "merged_core8_profile": copy.deepcopy(
            dict(sample_manifest["merged_core8_profile"])
        ),
        "prompt_contract": copy.deepcopy(dict(prompt_contract)),
        "fomc_tokenizer": {
            **copy.deepcopy(dict(tokenizer_binding)),
            "semantics": tokenizer_semantics,
            "role": "single_frozen_input_and_output_tokenizer_for_all_three_models",
        },
        "token_ledger": {
            **ledger_binding,
            "row_schema_version": LEDGER_ROW_SCHEMA,
            "full_prompt_token_ids_persisted": True,
            "vllm_runtime_chat_templating": False,
        },
        "generation_design": {
            "backend": "vllm-async-engine-v1-continuous-batching",
            "models": list(MODEL_ORDER),
            "prompts": EXPECTED_PROMPTS,
            "meetings": contract.EXPECTED_MEETINGS,
            "topics_per_meeting": len(contract.CORE_TOPICS),
            "replicate_seeds": list(REPLICATE_SEEDS),
            "replicates": len(REPLICATE_SEEDS),
            "rows_per_model": EXPECTED_CASES_PER_MODEL,
            "total_rows": EXPECTED_TOTAL_CASES,
            "canonical_case_order": "sample_manifest_order_then_replicate_id",
            "absolute_chunk_size": ABSOLUTE_CHUNK_SIZE,
            "absolute_chunk_count_per_model": (
                EXPECTED_CASES_PER_MODEL // ABSOLUTE_CHUNK_SIZE
            ),
            "partial_chunk_count_per_model": (
                EXPECTED_CASES_PER_MODEL % ABSOLUTE_CHUNK_SIZE
            ),
            "row_seed": "derive_row_seed(replicate_seed,sample_id)",
        },
        "preparation_runtime": {
            "python": platform.python_version(),
            "transformers": _package_version("transformers"),
            "tokenizers": _package_version("tokenizers"),
            "prompt_token_ids_materialized_once": True,
        },
        "implementation": {
            "preparer": _binding(Path(__file__).resolve()),
            "source_profile": _binding(Path(str(k10_profile.__file__)).resolve()),
            "data_contract": _binding(Path(str(contract.__file__)).resolve()),
        },
    }
    cohort = seal_manifest(cohort_payload)
    cohort_path = output_dir / "cohort_n2048_k5.v1.json"
    _write_new_json(cohort_path, cohort)
    preparation = seal_manifest(
        {
            "schema_version": PREPARATION_SCHEMA,
            "status": "complete",
            "evaluation_id": EVALUATION_ID,
            "cohort": {
                **_binding(cohort_path),
                "payload_sha256": cohort["integrity"]["payload_sha256"],
            },
            "token_ledger": ledger_binding,
            "coverage": {
                "meetings": contract.EXPECTED_MEETINGS,
                "prompts": EXPECTED_PROMPTS,
                "replicates": len(REPLICATE_SEEDS),
                "models": len(MODEL_ORDER),
                "rows_per_model": EXPECTED_CASES_PER_MODEL,
                "total_generation_rows": EXPECTED_TOTAL_CASES,
                "absolute_chunks_per_model": (
                    EXPECTED_CASES_PER_MODEL // ABSOLUTE_CHUNK_SIZE
                ),
            },
            "validation": {
                "deep_harmonized_source_validation": "passed",
                "exact_prompt_token_ids_persisted": True,
                "prompt_token_count_mismatches": 0,
                "runtime_chat_template_application": False,
            },
        }
    )
    _write_new_json(output_dir / "preparation.json", preparation)
    return preparation


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-sample-manifest", required=True, type=Path)
    parser.add_argument("--source-sample-manifest-sha256", required=True)
    parser.add_argument(
        "--pre-release", type=Path, default=contract.PRE_RELEASE_MANIFEST
    )
    parser.add_argument(
        "--post-release", type=Path, default=contract.POST_RELEASE_MANIFEST
    )
    parser.add_argument("--pre-release-sha256")
    parser.add_argument("--post-release-sha256", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        result = prepare(
            source_sample_manifest_path=args.source_sample_manifest,
            source_sample_manifest_sha256=args.source_sample_manifest_sha256,
            pre_release_manifest=args.pre_release,
            post_release_manifest=args.post_release,
            pre_release_sha256=args.pre_release_sha256,
            post_release_sha256=args.post_release_sha256,
            output_dir=args.output_dir,
        )
    except (VllmK5PreparationError, OSError, ValueError) as exc:
        print(
            _canonical(
                {
                    "status": "failed",
                    "error_type": type(exc).__name__,
                    "message": str(exc),
                }
            )
        )
        return 1
    print(_canonical(result))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ABSOLUTE_CHUNK_SIZE",
    "COHORT_SCHEMA",
    "EVALUATION_ID",
    "EXPECTED_CASES_PER_MODEL",
    "LEDGER_ROW_SCHEMA",
    "MODEL_ORDER",
    "REPLICATE_SEEDS",
    "VllmK5PreparationError",
    "build_token_ledger",
    "prepare",
]
