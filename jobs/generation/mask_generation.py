import argparse
import glob
import hashlib
import json
import logging
import os
from pathlib import Path

from generate_new_response import (
    ROW_SEED_POLICY_BATCH,
    ROW_SEED_POLICY_SAMPLE,
    generate_new_response,
)
from open_r1.generate import get_system_prompt
from open_r1.minutes_prompt import (
    MinutesPromptSpec,
    load_minutes_prompt_config,
    validate_minutes_prompt_spec,
)
from open_r1.provenance import (
    fingerprint_artifact_path,
    sha256_file,
    validate_sha256,
)
from open_r1.validator.intervention import (
    has_line_block_boundaries,
    match_indicator_marker,
    normalise_indicator_text,
    single_contiguous_deletion,
)
from open_r1.validator.loo_generation_spec import (
    derive_row_seed,
    load_and_validate_generation_spec,
    validate_generation_completion,
)
from open_r1.validator.neutral_intervention import (
    validate_neutral_intervention_manifest,
)


GENERATION_SCHEMA_VERSION = "loo-generation-v4"
INTERVENTION_SCHEMA_VERSION = "loo-intervention-v2"
ROSTER_SCHEMA_VERSION = "loo-intervention-roster-v1"
PROMPT_MANIFEST_SCHEMA_VERSIONS = frozenset(
    {"loo-prompt-manifest-v1", "loo-prompt-manifest-v2"}
)
DELETION_MASKING_STRATEGY = "indicator_block_deletion"
NEUTRAL_MASKING_STRATEGY = "indicator_block_neutral_replacement"
TOKEN_LIMIT_POLICY_ERROR = "error"
TOKEN_LIMIT_POLICY_EXCLUDE = "exclude"
TOKEN_LIMIT_POLICIES = frozenset(
    {TOKEN_LIMIT_POLICY_ERROR, TOKEN_LIMIT_POLICY_EXCLUDE}
)
TOKEN_LIMIT_EXCLUSION_POLICY = "record-and-exclude-token-limit-v1"


def _sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inspect_prompt_file(path: str | Path) -> tuple[int, str]:
    row_count = 0
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid prompt JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}")
            if not str(row.get("prompt") or "").strip():
                raise ValueError(f"Prompt is empty in {path}:{line_number}")
            row_count += 1
    if row_count == 0:
        raise ValueError(f"Prompt artifact is empty: {path}")
    return row_count, _sha256_file(path)


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_frozen_json(path: Path, payload: dict) -> None:
    serialised = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    )
    if path.is_file():
        existing = path.read_text(encoding="utf-8")
        if existing != serialised:
            raise ValueError(
                f"Refusing to overwrite incompatible frozen manifest {path}; "
                "use a new run-scoped output directory"
            )
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(f".{path.name}.tmp")
    temporary_path.write_text(serialised, encoding="utf-8")
    temporary_path.replace(path)


def _completion_error_record(
    row: dict,
    *,
    finish_reason: str,
    output_token_count: int,
    max_new_tokens: int,
    max_model_len: int,
) -> dict:
    return {
        "error_type": "token_limit_finish",
        "sample_id": str(row.get("sample_id") or ""),
        "meeting_date": str(row.get("meeting_date") or ""),
        "section_name": str(row.get("section_name") or ""),
        "generation_position": int(row["generation_position"]),
        "finish_reason": finish_reason,
        "output_token_count": output_token_count,
        "max_new_tokens": max_new_tokens,
        "max_model_len": max_model_len,
        "generated_sha256": str(row.get("generated_sha256") or ""),
    }


def apply_completion_policy_to_output(
    output_file: str | Path,
    *,
    max_new_tokens: int,
    max_model_len: int,
    token_limit_policy: str,
) -> dict:
    """Validate completion metadata and annotate explicit scoring exclusions.

    Every terminal raw row remains in the JSONL. Under the opt-in ``exclude``
    policy, only token-limit finishes are marked as ineligible for scoring;
    input truncation, invalid accounting, and unknown finish reasons still
    raise through ``validate_generation_completion``.
    """

    if token_limit_policy not in TOKEN_LIMIT_POLICIES:
        raise ValueError(
            f"Unsupported token_limit_policy={token_limit_policy!r}"
        )
    path = Path(output_file)
    if not path.is_file():
        raise FileNotFoundError(f"Generation output does not exist: {path}")

    rows: list[dict] = []
    changed = False
    exclusions: list[dict] = []
    passed_count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid generation output {path}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object in {path}:{line_number}"
                )
            try:
                prompt_token_count = int(row["prompt_token_count"])
                prompt_preflight_token_count = int(
                    row["prompt_preflight_token_count"]
                )
                output_token_count = int(row["output_token_count"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Generation output {path}:{line_number} lacks valid "
                    "completion token counts"
                ) from exc
            finish_reason = str(row.get("generation_finish_reason") or "")
            completion = validate_generation_completion(
                input_token_count=prompt_preflight_token_count,
                output_token_count=output_token_count,
                max_new_tokens=max_new_tokens,
                context_limit=max_model_len,
                finish_reason=finish_reason,
                input_was_truncated=row.get("input_was_truncated"),
                consumed_input_token_count=prompt_token_count,
                token_limit_policy=token_limit_policy,
            )
            status = str(completion["generation_validation_status"])
            if status == "excluded_token_limit_finish":
                error = _completion_error_record(
                    row,
                    finish_reason=finish_reason,
                    output_token_count=output_token_count,
                    max_new_tokens=max_new_tokens,
                    max_model_len=max_model_len,
                )
                exclusions.append(error)
            else:
                error = None
                passed_count += 1

            if row.get("generation_validation_status") != status:
                row["generation_validation_status"] = status
                changed = True
            if error is None:
                if "generation_validation_error" in row:
                    del row["generation_validation_error"]
                    changed = True
            elif row.get("generation_validation_error") != error:
                row["generation_validation_error"] = error
                changed = True
            rows.append(row)

    if not rows:
        return {
            "policy": (
                TOKEN_LIMIT_EXCLUSION_POLICY
                if token_limit_policy == TOKEN_LIMIT_POLICY_EXCLUDE
                else "strict-token-limit-error-v1"
            ),
            "token_limit_policy": token_limit_policy,
            "attempted_row_count": 0,
            "passed_row_count": 0,
            "excluded_row_count": 0,
            "exclusions": [],
        }
    if changed:
        temporary = path.with_name(f".{path.name}.completion.tmp")
        temporary.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in rows
            ),
            encoding="utf-8",
        )
        temporary.replace(path)

    return {
        "policy": (
            TOKEN_LIMIT_EXCLUSION_POLICY
            if token_limit_policy == TOKEN_LIMIT_POLICY_EXCLUDE
            else "strict-token-limit-error-v1"
        ),
        "token_limit_policy": token_limit_policy,
        "attempted_row_count": len(rows),
        "passed_row_count": passed_count,
        "excluded_row_count": len(exclusions),
        "exclusions": exclusions,
    }


def _read_json_object(path: str | Path, *, label: str) -> dict:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} does not exist: {source}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid {label} JSON {source}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {source}")
    return payload


def validate_prompt_manifest_binding(
    *,
    prompt_manifest_file: str | Path,
    input_dir: str | Path,
    masking_strategy: str,
) -> dict:
    """Bind a prompt folder to the immutable inventory emitted by its builder."""

    manifest_path = Path(prompt_manifest_file).expanduser().resolve()
    manifest = _read_json_object(
        manifest_path,
        label="LOO prompt manifest",
    )
    prompt_schema_version = manifest.get("schema_version")
    if prompt_schema_version not in PROMPT_MANIFEST_SCHEMA_VERSIONS:
        raise ValueError(
            "Unsupported LOO prompt manifest schema: "
            f"{manifest.get('schema_version')!r}"
        )
    if masking_strategy == DELETION_MASKING_STRATEGY:
        relative_folder = "exact_delete"
        accepted_arms = {"full", "delete"}
        intervention_key = "intervention_manifest"
    elif masking_strategy == NEUTRAL_MASKING_STRATEGY:
        relative_folder = "neutral"
        accepted_arms = {"full_export", "neutral"}
        intervention_key = "neutral_intervention_manifest"
    else:
        raise ValueError(f"Unsupported masking strategy: {masking_strategy!r}")

    resolved_input_dir = Path(input_dir).expanduser().resolve()
    expected_input_dir = (manifest_path.parent / relative_folder).resolve()
    if resolved_input_dir != expected_input_dir:
        raise ValueError(
            "Prompt folder is not the run-scoped folder declared by the prompt "
            f"manifest: expected={expected_input_dir}, observed={resolved_input_dir}"
        )

    raw_artifacts = manifest.get("artifacts")
    if not isinstance(raw_artifacts, list):
        raise ValueError("LOO prompt manifest has no artifact inventory")
    declared_by_name: dict[str, dict] = {}
    for entry in raw_artifacts:
        if not isinstance(entry, dict):
            raise ValueError("LOO prompt manifest artifact entries must be objects")
        relative_path = str(entry.get("relative_path") or "").strip()
        arm = str(entry.get("arm") or "").strip()
        if not relative_path.startswith(f"{relative_folder}/"):
            continue
        if arm not in accepted_arms:
            raise ValueError(
                f"Unexpected prompt arm {arm!r} for {relative_path}"
            )
        filename = Path(relative_path).name
        if filename in declared_by_name:
            raise ValueError(
                f"Duplicate prompt artifact filename in manifest: {filename}"
            )
        declared_by_name[filename] = entry

    observed_paths = {
        path.name: path.resolve()
        for path in sorted(resolved_input_dir.glob("*.jsonl"))
    }
    if set(observed_paths) != set(declared_by_name):
        raise ValueError(
            "Prompt directory does not match its frozen prompt manifest: "
            f"missing={sorted(set(declared_by_name) - set(observed_paths))}, "
            f"extra={sorted(set(observed_paths) - set(declared_by_name))}"
        )
    for filename, entry in declared_by_name.items():
        row_count, digest = _inspect_prompt_file(observed_paths[filename])
        if (
            entry.get("row_count") != row_count
            or entry.get("sha256") != digest
        ):
            raise ValueError(
                f"Prompt artifact no longer matches its manifest: {filename}"
            )

    raw_minutes_prompt = manifest.get("minutes_system_prompt")
    minutes_prompt_spec = (
        None
        if raw_minutes_prompt is None
        else validate_minutes_prompt_spec(raw_minutes_prompt)
    )
    if minutes_prompt_spec is not None:
        context_budget = manifest.get("context_budget")
        if not isinstance(context_budget, dict):
            raise ValueError("Prompt manifest lacks its Minutes context budget")
        expected_budget = {
            "system_prompt_version": minutes_prompt_spec.version,
            "system_prompt_sha256": minutes_prompt_spec.sha256,
            "requested_max_output_tokens": (
                minutes_prompt_spec.requested_max_output_tokens
            ),
            "max_new_tokens": minutes_prompt_spec.hard_max_new_tokens,
            "max_model_len": minutes_prompt_spec.max_model_len,
            "input_truncation": minutes_prompt_spec.input_truncation,
        }
        budget_mismatches = {
            key: {"expected": value, "observed": context_budget.get(key)}
            for key, value in expected_budget.items()
            if context_budget.get(key) != value
        }
        if budget_mismatches:
            raise ValueError(
                "Prompt manifest Minutes contract differs from its context "
                f"budget: {budget_mismatches}"
            )
        expected_row_metadata = {
            "generation_system_prompt_version": minutes_prompt_spec.version,
            "generation_system_prompt_sha256": minutes_prompt_spec.sha256,
            "generation_requested_max_output_tokens": (
                minutes_prompt_spec.requested_max_output_tokens
            ),
            "generation_hard_max_new_tokens": (
                minutes_prompt_spec.hard_max_new_tokens
            ),
            "generation_max_model_len": minutes_prompt_spec.max_model_len,
        }
        for path in observed_paths.values():
            for row in _read_prompt_rows(path).values():
                row_mismatches = {
                    key: {"expected": value, "observed": row.get(key)}
                    for key, value in expected_row_metadata.items()
                    if row.get(key) != value
                }
                if row_mismatches:
                    raise ValueError(
                        f"Prompt row Minutes contract mismatch in {path}: "
                        f"{row_mismatches}"
                    )

    for entry in manifest.get("source_artifacts", []):
        if not isinstance(entry, dict):
            raise ValueError("Prompt source-artifact entries must be objects")
        source_path = Path(str(entry.get("path") or "")).expanduser().resolve()
        if not source_path.is_file():
            raise FileNotFoundError(
                f"Frozen prompt source artifact is unavailable: {source_path}"
            )
        if sha256_file(source_path) != entry.get("sha256"):
            raise ValueError(
                f"Frozen prompt source artifact changed: {source_path}"
            )

    referenced_files: dict[str, dict] = {}
    for key in (
        "run_intervention_roster",
        "prompt_ledger",
        intervention_key,
    ):
        reference = manifest.get(key)
        if not isinstance(reference, dict):
            raise ValueError(f"Prompt manifest lacks {key}")
        relative_path = str(reference.get("relative_path") or "").strip()
        expected_hash = validate_sha256(
            reference.get("sha256"),
            label=f"prompt_manifest.{key}.sha256",
        )
        referenced_path = (manifest_path.parent / relative_path).resolve()
        if manifest_path.parent.resolve() not in referenced_path.parents:
            raise ValueError(
                f"Prompt manifest reference escapes its run directory: {key}"
            )
        if not referenced_path.is_file():
            raise FileNotFoundError(
                f"Prompt manifest reference is missing: {referenced_path}"
            )
        if sha256_file(referenced_path) != expected_hash:
            raise ValueError(
                f"Prompt manifest reference hash mismatch: {referenced_path}"
            )
        referenced_files[key] = {
            "path": str(referenced_path),
            "sha256": expected_hash,
        }

    return {
        "schema_version": prompt_schema_version,
        "path": str(manifest_path),
        "sha256": sha256_file(manifest_path),
        "population_id": manifest.get("population_id"),
        "population_context": manifest.get("population_context"),
        "artifact_count": len(declared_by_name),
        "input_folder_sha256": fingerprint_artifact_path(
            resolved_input_dir
        )["sha256"],
        "references": referenced_files,
        "minutes_system_prompt": (
            None
            if minutes_prompt_spec is None
            else minutes_prompt_spec.as_dict()
        ),
        "experiment_config": manifest.get("experiment_config"),
    }


def validate_generation_spec_binding(
    *,
    generation_spec_file: str | Path,
    expected_file_sha256: str | None,
    model_artifact_sha256: str,
    tokenizer_artifact_sha256: str,
    prompt_manifest_metadata: dict,
    input_dir: str | Path,
) -> dict:
    """Validate a frozen spec and ensure it contains the active artifacts."""

    spec_path = Path(generation_spec_file).expanduser().resolve()
    spec = load_and_validate_generation_spec(
        spec_path,
        verify_artifact_paths=False,
        expected_file_sha256=expected_file_sha256,
    )
    frozen = spec["frozen_artifacts"]
    frozen_models = frozen.get("models")
    frozen_tokenizers = frozen.get("tokenizers")
    minutes_model = (
        frozen_models.get("minutes_model")
        if isinstance(frozen_models, dict)
        else None
    )
    minutes_tokenizer = (
        frozen_tokenizers.get("minutes_tokenizer")
        if isinstance(frozen_tokenizers, dict)
        else None
    )
    if not isinstance(minutes_model, dict):
        raise ValueError(
            "Frozen generation spec lacks the named minutes_model artifact"
        )
    if not isinstance(minutes_tokenizer, dict):
        raise ValueError(
            "Frozen generation spec lacks the named minutes_tokenizer artifact"
        )
    source_hashes = {
        str(entry.get("sha256"))
        for entry in frozen["sources"].values()
        if isinstance(entry, dict)
    }
    prompt_folder_hash = fingerprint_artifact_path(input_dir)["sha256"]
    required_source_hashes = {
        str(prompt_manifest_metadata["sha256"]),
        str(prompt_folder_hash),
    }
    prompt_experiment_binding = prompt_manifest_metadata.get("experiment_config")
    if isinstance(prompt_experiment_binding, dict):
        required_source_hashes.add(
            validate_sha256(
                prompt_experiment_binding.get("sha256"),
                label="prompt experiment_config.sha256",
            )
        )
    if spec.get("population_id") != prompt_manifest_metadata.get(
        "population_id"
    ):
        raise ValueError(
            "Frozen generation spec population does not match the prompt "
            "manifest population"
        )
    if model_artifact_sha256 != minutes_model.get("sha256"):
        raise ValueError(
            "Active generation model does not match frozen_artifacts.models."
            "minutes_model"
        )
    if tokenizer_artifact_sha256 != minutes_tokenizer.get("sha256"):
        raise ValueError(
            "Active generation tokenizer does not match frozen_artifacts."
            "tokenizers.minutes_tokenizer"
        )
    if not required_source_hashes.issubset(source_hashes):
        raise ValueError(
            "Frozen generation spec does not contain the active prompt manifest "
            "and prompt directory"
        )
    prompt_minutes_spec_payload = prompt_manifest_metadata.get(
        "minutes_system_prompt"
    )
    if prompt_minutes_spec_payload is not None:
        prompt_minutes_spec = validate_minutes_prompt_spec(
            prompt_minutes_spec_payload
        )
        generation_minutes_spec = validate_minutes_prompt_spec(
            spec.get("generation_config", {}).get("minutes_system_prompt")
        )
        if generation_minutes_spec != prompt_minutes_spec:
            raise ValueError(
                "Generation spec Minutes prompt differs from the prompt manifest"
            )
    execution = spec.get("generation_config", {}).get("execution")
    execution_environment = None
    if isinstance(execution, dict):
        expected_execution = {
            "gpu_policy": "single-physical-gpu-by-nvidia-smi-index-v1",
            "physical_gpu_index": 1,
            "expected_visible_cuda_devices": 1,
            "tensor_parallel_size": 1,
        }
        mismatches = {
            key: {"expected": value, "observed": execution.get(key)}
            for key, value in expected_execution.items()
            if execution.get(key) != value
        }
        expected_uuid = os.environ.get("LOO_CANONICAL_PHYSICAL_GPU_UUID")
        execution_environment = {
            "physical_gpu_index": os.environ.get(
                "LOO_CANONICAL_PHYSICAL_GPU_INDEX"
            ),
            "physical_gpu_uuid": expected_uuid,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "declared_visible_device_count": os.environ.get(
                "LOO_CANONICAL_VISIBLE_DEVICE_COUNT"
            ),
            "tensor_parallel_size": 1,
        }
        if (
            mismatches
            or execution_environment["physical_gpu_index"] != "1"
            or not isinstance(expected_uuid, str)
            or not expected_uuid.startswith("GPU-")
            or execution_environment["cuda_visible_devices"]
            != execution_environment["physical_gpu_index"]
            or execution_environment["declared_visible_device_count"] != "1"
        ):
            raise ValueError(
                "Canonical generation must expose only physical GPU 1 by its "
                "numeric index while retaining its UUID as audit metadata "
                f"before model loading: config_mismatches={mismatches}, "
                f"environment={execution_environment}"
            )
    return {
        "path": str(spec_path),
        "file_sha256": sha256_file(spec_path),
        "payload_sha256": spec["integrity"]["payload_sha256"],
        "schema_version": spec["schema_version"],
        "run_id": spec["run_id"],
        "phase": spec["phase"],
        "population_id": spec["population_id"],
        "execution_environment": execution_environment,
    }


def _parse_prompt_artifact_identity(path: Path) -> tuple[str, str]:
    if "_masked_" not in path.stem:
        raise ValueError(
            f"Prompt filename {path.name!r} must use "
            "'<indicator>_masked_<context>.jsonl'"
        )
    indicator, context = path.stem.split("_masked_", 1)
    if not indicator.strip() or not context.strip():
        raise ValueError(f"Prompt filename has an incomplete identity: {path.name}")
    return indicator.strip(), context.strip()


def _normalise_identity_value(field: str, value: object) -> str:
    text = str(value or "").strip()
    return text[:10] if field == "meeting_date" else text


def _prompt_row_key(row: dict, *, path: Path, line_number: int) -> tuple[str, ...]:
    meeting_date = _normalise_identity_value("meeting_date", row.get("meeting_date"))
    section_name = _normalise_identity_value("section_name", row.get("section_name"))
    if not meeting_date or not section_name:
        raise ValueError(
            f"Canonical prompt row {path}:{line_number} requires non-empty "
            "meeting_date and section_name"
        )

    sample_id = _normalise_identity_value("sample_id", row.get("sample_id"))
    if sample_id:
        return "sample_id", sample_id
    source_index = row.get("source_index", row.get("index"))
    if source_index is None or not str(source_index).strip():
        raise ValueError(
            f"Canonical prompt row {path}:{line_number} requires sample_id or "
            "source_index/index"
        )
    return "legacy", meeting_date, section_name, str(source_index).strip()


def _read_prompt_rows(path: Path) -> dict[tuple[str, ...], dict]:
    rows: dict[tuple[str, ...], dict] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid prompt JSON in {path}:{line_number}: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"Expected a JSON object in {path}:{line_number}")
            prompt = str(row.get("prompt") or "")
            if not prompt.strip():
                raise ValueError(f"Prompt is empty in {path}:{line_number}")
            key = _prompt_row_key(row, path=path, line_number=line_number)
            if key in rows:
                raise ValueError(f"Duplicate prompt row key {key!r} in {path}")
            rows[key] = row
    if not rows:
        raise ValueError(f"Prompt artifact is empty: {path}")
    return rows


def _load_intervention_roster(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(f"LOO intervention roster does not exist: {path}")
    try:
        roster = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid LOO intervention roster {path}: {exc}") from exc
    if not isinstance(roster, dict):
        raise ValueError(f"LOO intervention roster must be a JSON object: {path}")
    if roster.get("schema_version") != ROSTER_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported intervention roster schema in {path}: "
            f"{roster.get('schema_version')!r}"
        )

    baseline_indicator = str(roster.get("baseline_indicator") or "").strip()
    indicators = [
        str(value).strip()
        for value in roster.get("indicators", [])
        if str(value).strip()
    ]
    contexts = [
        str(value).strip()
        for value in roster.get("contexts", [])
        if str(value).strip()
    ]
    if not baseline_indicator or not indicators or not contexts:
        raise ValueError(
            f"Intervention roster {path} requires baseline_indicator, indicators, "
            "and contexts"
        )
    if baseline_indicator in indicators:
        raise ValueError("baseline_indicator must not also appear in indicators")
    if len(indicators) != len(set(indicators)) or len(contexts) != len(set(contexts)):
        raise ValueError(f"Intervention roster contains duplicate values: {path}")

    marker_payload = roster.get("indicator_markers")
    if not isinstance(marker_payload, dict) or set(marker_payload) != set(indicators):
        raise ValueError(
            f"Intervention roster {path} requires indicator_markers for exactly "
            "the declared indicator universe"
        )
    for indicator in indicators:
        markers = marker_payload.get(indicator)
        if (
            not isinstance(markers, list)
            or not markers
            or any(not normalise_indicator_text(marker) for marker in markers)
        ):
            raise ValueError(
                f"Intervention roster {path} has invalid markers for {indicator!r}"
            )
    return roster


def build_intervention_manifest(
    *,
    input_dir: Path,
    input_prompt_files: list[Path],
    roster_file: Path,
) -> dict:
    """Validate the frozen intervention universe and exact prompt deletions."""

    roster = _load_intervention_roster(roster_file)
    baseline_indicator = str(roster["baseline_indicator"]).strip()
    indicators = [str(value).strip() for value in roster["indicators"]]
    contexts = [str(value).strip() for value in roster["contexts"]]
    indicator_markers = {
        indicator: [str(value).strip() for value in roster["indicator_markers"][indicator]]
        for indicator in indicators
    }
    expected_identities = {
        (indicator, context)
        for indicator in [baseline_indicator, *indicators]
        for context in contexts
    }

    paths_by_identity: dict[tuple[str, str], Path] = {}
    for path in input_prompt_files:
        identity = _parse_prompt_artifact_identity(path)
        if identity in paths_by_identity:
            raise ValueError(
                f"Duplicate prompt artifact identity {identity}: "
                f"{paths_by_identity[identity]} and {path}"
            )
        paths_by_identity[identity] = path
    observed_identities = set(paths_by_identity)
    if observed_identities != expected_identities:
        missing = sorted(expected_identities - observed_identities)
        extra = sorted(observed_identities - expected_identities)
        raise ValueError(
            "Prompt artifacts do not match the frozen intervention roster: "
            f"missing={missing[:10]}, extra={extra[:10]}"
        )

    rows_by_identity = {
        identity: _read_prompt_rows(path)
        for identity, path in paths_by_identity.items()
    }
    input_artifacts: list[dict] = []
    interventions: list[dict] = []
    for identity, path in sorted(paths_by_identity.items()):
        indicator, context = identity
        input_artifacts.append(
            {
                "relative_path": path.relative_to(input_dir).as_posix(),
                "indicator": indicator,
                "context": context,
                "source_row_count": len(rows_by_identity[identity]),
                "source_file_sha256": _sha256_file(path),
            }
        )

    for context in contexts:
        full_path = paths_by_identity[(baseline_indicator, context)]
        full_rows = rows_by_identity[(baseline_indicator, context)]
        full_keys = set(full_rows)
        for indicator in indicators:
            masked_path = paths_by_identity[(indicator, context)]
            masked_rows = rows_by_identity[(indicator, context)]
            masked_keys = set(masked_rows)
            if masked_keys != full_keys:
                missing = sorted(full_keys - masked_keys)
                extra = sorted(masked_keys - full_keys)
                raise ValueError(
                    f"Prompt row coverage mismatch for {indicator}/{context}: "
                    f"missing={missing[:5]}, extra={extra[:5]}"
                )
            for row_key in sorted(full_keys):
                full_row = full_rows[row_key]
                masked_row = masked_rows[row_key]
                for field in ("meeting_date", "section_name"):
                    full_value = _normalise_identity_value(field, full_row.get(field))
                    masked_value = _normalise_identity_value(field, masked_row.get(field))
                    if full_value != masked_value:
                        raise ValueError(
                            f"Prompt identity mismatch for {indicator}/{context}/"
                            f"{row_key!r}: {field} full={full_value!r}, "
                            f"masked={masked_value!r}"
                        )
                full_prompt = str(full_row["prompt"])
                masked_prompt = str(masked_row["prompt"])
                try:
                    deletion_start, deletion_end, removed_block = (
                        single_contiguous_deletion(full_prompt, masked_prompt)
                    )
                except ValueError as exc:
                    raise ValueError(
                        f"Invalid masking intervention for {indicator}/{context}/"
                        f"{row_key!r} ({full_path.name} -> {masked_path.name}): {exc}"
                    ) from exc
                if not has_line_block_boundaries(
                    full_prompt,
                    deletion_start,
                    deletion_end,
                ):
                    raise ValueError(
                        f"Invalid masking intervention for {indicator}/{context}/"
                        f"{row_key!r}: the removed text is not bounded as a "
                        "complete prompt line/block"
                    )
                matched_marker = match_indicator_marker(
                    removed_block,
                    indicator_markers[indicator],
                )
                if matched_marker is None:
                    raise ValueError(
                        f"Invalid masking intervention for {indicator}/{context}/"
                        f"{row_key!r}: the removed block's first semantic line "
                        "does not contain any declared indicator marker "
                        f"{indicator_markers[indicator]!r}"
                    )
                interventions.append(
                    {
                        "indicator": indicator,
                        "context": context,
                        "row_key": list(row_key),
                        "meeting_date": _normalise_identity_value(
                            "meeting_date",
                            full_row.get("meeting_date"),
                        ),
                        "section_name": _normalise_identity_value(
                            "section_name",
                            full_row.get("section_name"),
                        ),
                        "full_prompt_sha256": _sha256_text(full_prompt),
                        "masked_prompt_sha256": _sha256_text(masked_prompt),
                        "removed_block_sha256": _sha256_text(removed_block),
                        "matched_indicator_marker": matched_marker,
                        "deletion_start": deletion_start,
                        "deletion_end": deletion_end,
                        "removed_character_count": len(removed_block),
                        "line_block_boundaries_validated": True,
                    }
                )

    masked_hashes_by_sample: dict[
        tuple[str, tuple[str, ...]],
        dict[str, str],
    ] = {}
    deletion_spans_by_sample: dict[
        tuple[str, tuple[str, ...]],
        list[tuple[int, int, str]],
    ] = {}
    for entry in interventions:
        sample_key = (
            str(entry["context"]),
            tuple(str(value) for value in entry["row_key"]),
        )
        masked_hash = str(entry["masked_prompt_sha256"])
        previous_indicator = masked_hashes_by_sample.setdefault(
            sample_key,
            {},
        ).get(masked_hash)
        if previous_indicator is not None:
            raise ValueError(
                f"Duplicate masked prompt for {sample_key!r}: indicators "
                f"{previous_indicator!r} and {entry['indicator']!r} produce "
                "the same intervention"
            )
        masked_hashes_by_sample[sample_key][masked_hash] = str(entry["indicator"])
        deletion_spans_by_sample.setdefault(sample_key, []).append(
            (
                int(entry["deletion_start"]),
                int(entry["deletion_end"]),
                str(entry["indicator"]),
            )
        )

    for sample_key, spans in deletion_spans_by_sample.items():
        sorted_spans = sorted(spans)
        for previous, current in zip(
            sorted_spans,
            sorted_spans[1:],
            strict=False,
        ):
            if current[0] < previous[1]:
                raise ValueError(
                    f"Overlapping indicator-block deletions for {sample_key!r}: "
                    f"{previous[2]!r} span={previous[:2]} overlaps "
                    f"{current[2]!r} span={current[:2]}"
                )

    return {
        "schema_version": INTERVENTION_SCHEMA_VERSION,
        "roster_id": str(roster.get("roster_id") or roster_file.stem),
        "roster_file_sha256": _sha256_file(roster_file),
        "baseline_indicator": baseline_indicator,
        "indicators": indicators,
        "indicator_markers": indicator_markers,
        "contexts": contexts,
        "validation": "exact_single_contiguous_prompt_deletion",
        "indicator_identity_validation": (
            "declared_marker_in_removed_block_header"
        ),
        "block_boundary_validation": "complete_line_or_block_boundaries",
        "indicator_block_span_validation": (
            "pairwise_non_overlapping_per_sample"
        ),
        "distinct_masked_prompt_per_indicator": True,
        "input_artifacts": input_artifacts,
        "interventions": interventions,
    }


def validate_output_file(
    section_output_file: str,
    *,
    replicate_id: int,
    replicate_seed: int,
    expected_row_count: int,
    source_file_sha256: str,
    batch_size: int,
    model_path: str,
    model_artifact_sha256: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int,
    masking_strategy: str,
    indicator: str,
    evaluation_context: str,
    expected_prompt_rows: list[dict],
    seed_policy: str = ROW_SEED_POLICY_BATCH,
    max_model_len: int = 16384,
    require_normal_finish: bool = False,
    tokenizer_path: str | None = None,
    tokenizer_artifact_sha256: str | None = None,
    token_limit_policy: str = TOKEN_LIMIT_POLICY_ERROR,
    system_prompt_sha256: str | None = None,
    system_prompt_version: str | None = None,
    requested_max_output_tokens: int | None = None,
    experiment_config_sha256: str | None = None,
) -> bool:
    """Validate completeness, source identity, and resumable run metadata."""

    model_artifact_sha256 = validate_sha256(
        model_artifact_sha256,
        label="model_artifact_sha256",
    )
    if not os.path.exists(section_output_file):
        return False

    rows = []
    with open(section_output_file, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid existing generation output {section_output_file}:{line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise ValueError(
                    f"Expected a JSON object in {section_output_file}:{line_number}"
                )
            rows.append(row)
    if not rows:
        return False
    if len(rows) != expected_row_count:
        logging.warning(
            "Incomplete output %s: expected %s rows, found %s; it will be regenerated.",
            section_output_file,
            expected_row_count,
            len(rows),
        )
        return False
    if len(expected_prompt_rows) != expected_row_count:
        raise ValueError(
            f"Expected prompt inventory has {len(expected_prompt_rows)} rows, "
            f"but {expected_row_count} were declared for {section_output_file}"
        )

    try:
        generation_positions = {int(row["generation_position"]) for row in rows}
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Existing output {section_output_file} lacks valid generation_position "
            "metadata. Use a new run-scoped output directory."
        ) from exc
    expected_positions = set(range(expected_row_count))
    if generation_positions != expected_positions:
        logging.warning(
            "Incomplete output %s: generation positions do not cover 0..%s; "
            "it will be regenerated.",
            section_output_file,
            expected_row_count - 1,
        )
        return False

    for row in rows:
        generation_position = int(row["generation_position"])
        source_row = expected_prompt_rows[generation_position]
        expected_row_key = _prompt_row_key(
            source_row,
            path=Path(section_output_file),
            line_number=generation_position + 1,
        )
        observed_row_key = _prompt_row_key(
            row,
            path=Path(section_output_file),
            line_number=generation_position + 1,
        )
        expected_prompt = str(source_row.get("prompt") or "")
        observed_prompt = str(row.get("prompt") or "")
        expected_prompt_sha256 = _sha256_text(expected_prompt)
        row_binding_mismatches: dict[str, dict[str, object]] = {}
        if observed_row_key != expected_row_key:
            row_binding_mismatches["row_key"] = {
                "expected": list(expected_row_key),
                "observed": list(observed_row_key),
            }
        if observed_prompt != expected_prompt:
            row_binding_mismatches["prompt"] = {
                "expected_sha256": expected_prompt_sha256,
                "observed_sha256": _sha256_text(observed_prompt),
            }
        if row.get("source_prompt_sha256") != expected_prompt_sha256:
            row_binding_mismatches["source_prompt_sha256"] = {
                "expected": expected_prompt_sha256,
                "observed": row.get("source_prompt_sha256"),
            }
        for field in ("meeting_date", "section_name", "sample_id"):
            expected_identity = _normalise_identity_value(field, source_row.get(field))
            observed_identity = _normalise_identity_value(field, row.get(field))
            if expected_identity != observed_identity:
                row_binding_mismatches[field] = {
                    "expected": expected_identity,
                    "observed": observed_identity,
                }
        expected_source_index = source_row.get(
            "source_index",
            source_row.get("index", generation_position),
        )
        if str(row.get("source_index")) != str(expected_source_index):
            row_binding_mismatches["source_index"] = {
                "expected": expected_source_index,
                "observed": row.get("source_index"),
            }
        if row_binding_mismatches:
            raise ValueError(
                f"Existing output {section_output_file} is not row-bound to its "
                f"source prompt at generation_position={generation_position}: "
                f"{row_binding_mismatches}. Use a new run-scoped output directory."
            )

        if seed_policy == ROW_SEED_POLICY_SAMPLE:
            expected_seed = derive_row_seed(
                replicate_seed,
                source_row.get("sample_id"),
            )
        elif seed_policy == ROW_SEED_POLICY_BATCH:
            expected_seed = replicate_seed + (
                generation_position // batch_size
            )
        else:
            raise ValueError(f"Unsupported seed policy: {seed_policy!r}")
        expected = {
            "replicate_id": str(replicate_id),
            "generation_seed": expected_seed,
            "generation_seed_policy": seed_policy,
            "generation_model": model_path,
            "generation_model_sha256": model_artifact_sha256,
            "generation_tokenizer": tokenizer_path or model_path,
            "generation_batch_size": int(batch_size),
            "decoding_temperature": float(temperature),
            "decoding_top_p": float(top_p),
            "max_new_tokens": int(max_new_tokens),
            "max_model_len": int(max_model_len),
            "masking_strategy": masking_strategy,
            "indicator": indicator,
            "evaluation_context": evaluation_context,
            "source_file_sha256": source_file_sha256,
            "source_row_count": int(expected_row_count),
            "generation_system_prompt_sha256": (
                _sha256_text(get_system_prompt())
                if system_prompt_sha256 is None
                else validate_sha256(
                    system_prompt_sha256,
                    label="system_prompt_sha256",
                )
            ),
        }
        if system_prompt_version is not None:
            expected["generation_system_prompt_version"] = system_prompt_version
        if requested_max_output_tokens is not None:
            expected["generation_requested_max_output_tokens"] = int(
                requested_max_output_tokens
            )
        if experiment_config_sha256 is not None:
            expected["experiment_config_sha256"] = validate_sha256(
                experiment_config_sha256,
                label="experiment_config_sha256",
            )
        if tokenizer_artifact_sha256 is not None:
            expected["generation_tokenizer_sha256"] = validate_sha256(
                tokenizer_artifact_sha256,
                label="generation_tokenizer_sha256",
            )
        mismatches = {
            key: {"expected": value, "observed": row.get(key)}
            for key, value in expected.items()
            if row.get(key) != value
        }
        if mismatches:
            raise ValueError(
                f"Existing output {section_output_file} has incompatible or missing "
                f"generation metadata: {mismatches}. Use a new run-scoped output directory."
            )
        generated = str(row.get("generated") or "")
        if row.get("generated_sha256") != _sha256_text(generated):
            raise ValueError(
                f"Existing output {section_output_file} has an invalid "
                f"generated_sha256 at generation_position={generation_position}"
            )
        if require_normal_finish:
            try:
                prompt_token_count = int(row["prompt_token_count"])
                prompt_preflight_token_count = int(
                    row["prompt_preflight_token_count"]
                )
                output_token_count = int(row["output_token_count"])
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"Existing output {section_output_file} lacks canonical "
                    "token-count metadata"
                ) from exc
            if (
                prompt_token_count < 1
                or prompt_preflight_token_count < 1
                or output_token_count < 1
            ):
                raise ValueError(
                    f"Existing output {section_output_file} has non-positive "
                    "token counts"
                )
            if (
                requested_max_output_tokens is not None
                and output_token_count > requested_max_output_tokens
            ):
                raise ValueError(
                    f"Existing output {section_output_file} exceeds the requested "
                    "Minutes output limit at generation_position="
                    f"{generation_position}: output={output_token_count}, "
                    f"requested={requested_max_output_tokens}"
                )
            completion = validate_generation_completion(
                input_token_count=prompt_preflight_token_count,
                output_token_count=output_token_count,
                max_new_tokens=max_new_tokens,
                context_limit=max_model_len,
                finish_reason=str(
                    row.get("generation_finish_reason") or ""
                ),
                input_was_truncated=row.get("input_was_truncated"),
                consumed_input_token_count=prompt_token_count,
                token_limit_policy=token_limit_policy,
            )
            if token_limit_policy == TOKEN_LIMIT_POLICY_EXCLUDE:
                expected_status = completion["generation_validation_status"]
                if row.get("generation_validation_status") != expected_status:
                    raise ValueError(
                        f"Existing output {section_output_file} has an invalid "
                        "generation_validation_status at "
                        f"generation_position={generation_position}"
                    )
                expected_error = (
                    _completion_error_record(
                        row,
                        finish_reason=str(
                            row.get("generation_finish_reason") or ""
                        ),
                        output_token_count=output_token_count,
                        max_new_tokens=max_new_tokens,
                        max_model_len=max_model_len,
                    )
                    if expected_status == "excluded_token_limit_finish"
                    else None
                )
                if expected_error is None:
                    if "generation_validation_error" in row:
                        raise ValueError(
                            f"Passed output row in {section_output_file} "
                            "contains generation_validation_error"
                        )
                elif row.get("generation_validation_error") != expected_error:
                    raise ValueError(
                        f"Existing output {section_output_file} has invalid "
                        "token-limit exclusion metadata at "
                        f"generation_position={generation_position}"
                    )
    return True


def validate_all_outputs(expected_artifacts: list[dict]) -> None:
    """Fail unless every manifest artifact exists and matches its recorded hash."""

    problems: list[str] = []
    for artifact in expected_artifacts:
        path = Path(artifact["path"])
        if not path.is_file():
            problems.append(f"missing: {path}")
            continue
        observed_hash = _sha256_file(path)
        if observed_hash != artifact["output_sha256"]:
            problems.append(f"hash mismatch: {path}")
    if problems:
        raise RuntimeError(
            "Generated-output validation failed: " + "; ".join(problems[:10])
        )
    logging.info("✅ All generated artifacts are complete and hash-validated.")


def resolve_input_dir(input_folder: str) -> str:
    if os.path.isdir(input_folder):
        return input_folder
    return os.path.join("dataset/processed/main/evaluation_inputs/source_prompts/mask_indicator", input_folder)


def select_intervention_prompt_files(
    *,
    input_prompt_files: list[str],
    roster: dict,
    include_intervention_indicators: list[str] | None,
) -> tuple[list[str], dict]:
    """Select a generation shard without changing the frozen full baseline.

    The complete prompt directory and intervention manifest are validated
    before this selector is called.  A shard therefore changes only which
    intervention outputs are generated; the baseline prompt still contains
    every evidence block in the frozen roster.
    """

    baseline = str(roster["baseline_indicator"]).strip()
    full_indicators = [str(value).strip() for value in roster["indicators"]]
    contexts = [str(value).strip() for value in roster["contexts"]]

    if include_intervention_indicators is None:
        selected_indicators = list(full_indicators)
        mode = "full"
    else:
        requested = [str(value).strip() for value in include_intervention_indicators]
        if not requested or any(not value for value in requested):
            raise ValueError(
                "--include-intervention-indicator requires at least one "
                "non-empty indicator"
            )
        if len(requested) != len(set(requested)):
            raise ValueError("Intervention indicator selection contains duplicates")
        if baseline in requested:
            raise ValueError(
                f"Baseline indicator {baseline!r} is included automatically and "
                "must not be passed as an intervention indicator"
            )
        unknown = sorted(set(requested) - set(full_indicators))
        if unknown:
            raise ValueError(
                "Intervention indicator selection is outside the frozen roster: "
                f"{unknown}"
            )
        requested_set = set(requested)
        selected_indicators = [
            indicator
            for indicator in full_indicators
            if indicator in requested_set
        ]
        mode = (
            "full"
            if selected_indicators == full_indicators
            else "partial_intervention_shard"
        )

    selected_universe = {baseline, *selected_indicators}
    selected_files = [
        path
        for path in input_prompt_files
        if _parse_prompt_artifact_identity(Path(path))[0] in selected_universe
    ]
    observed_identities = {
        _parse_prompt_artifact_identity(Path(path))
        for path in selected_files
    }
    expected_identities = {
        (indicator, context)
        for indicator in [baseline, *selected_indicators]
        for context in contexts
    }
    if observed_identities != expected_identities:
        raise ValueError(
            "Selected prompt artifacts do not cover the requested intervention "
            f"shard: missing={sorted(expected_identities - observed_identities)}, "
            f"extra={sorted(observed_identities - expected_identities)}"
        )

    scope = {
        "schema_version": "loo-intervention-generation-scope-v1",
        "mode": mode,
        "baseline_indicator": baseline,
        "full_roster_indicators": full_indicators,
        "generated_intervention_indicators": selected_indicators,
        "omitted_intervention_indicators": [
            indicator
            for indicator in full_indicators
            if indicator not in set(selected_indicators)
        ],
        "baseline_contains_full_roster": True,
        "standalone_canonical_release_allowed": mode == "full",
    }
    return selected_files, scope


def run_mask_generation(
    *,
    input_folder: str,
    model_path: str,
    simulation_step: int,
    output_dir: str,
    roster_file: str,
    batch_size: int = 20,
    seed: int = 20260728,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_new_tokens: int = 8192,
    max_model_len: int = 16384,
    masking_strategy: str = DELETION_MASKING_STRATEGY,
    seed_policy: str = ROW_SEED_POLICY_SAMPLE,
    expected_model_sha256: str | None = None,
    tokenizer_path: str | None = None,
    expected_tokenizer_sha256: str | None = None,
    require_normal_finish: bool = False,
    intervention_manifest_file: str | None = None,
    prompt_manifest_file: str | None = None,
    generation_spec_file: str | None = None,
    expected_generation_spec_sha256: str | None = None,
    include_intervention_indicators: list[str] | None = None,
    include_sample_ids: list[str] | None = None,
    token_limit_policy: str = TOKEN_LIMIT_POLICY_ERROR,
    experiment_config_file: str | None = None,
) -> None:
    input_dir = resolve_input_dir(input_folder)
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"Masking prompt folder does not exist: {input_dir}")
    input_prompt_files = sorted(glob.glob(os.path.join(input_dir, "*.jsonl")))

    if not input_prompt_files:
        raise ValueError(f"No masking prompt JSONL files found in: {input_dir}")
    if simulation_step < 1:
        raise ValueError("simulation_step must be positive")
    if token_limit_policy not in TOKEN_LIMIT_POLICIES:
        raise ValueError(
            f"Unsupported token_limit_policy={token_limit_policy!r}"
        )
    if (
        token_limit_policy == TOKEN_LIMIT_POLICY_EXCLUDE
        and not require_normal_finish
    ):
        raise ValueError(
            "token_limit_policy='exclude' requires --require-normal-finish so "
            "all non-token-limit completion errors remain fatal"
        )

    minutes_prompt_spec: MinutesPromptSpec | None = None
    experiment_config_binding: dict | None = None
    scoped_experiment: dict | None = None
    if experiment_config_file is not None:
        (
            minutes_prompt_spec,
            experiment_config_binding,
            scoped_experiment,
        ) = load_minutes_prompt_config(experiment_config_file)
        if (
            max_new_tokens != minutes_prompt_spec.hard_max_new_tokens
            or max_model_len != minutes_prompt_spec.max_model_len
            or token_limit_policy != minutes_prompt_spec.token_limit_policy
            or not require_normal_finish
        ):
            raise ValueError(
                "Scoped Minutes generation requires its frozen hard/context "
                "limits, token_limit_policy='error', and --require-normal-finish"
            )
        decoding = scoped_experiment.get("decoding")
        if not isinstance(decoding, dict):
            raise ValueError("Scoped experiment lacks decoding settings")
        replicate_seeds = [
            seed + (index * 1_000_000) for index in range(simulation_step)
        ]
        matching_regimes = [
            name
            for name, settings in decoding.items()
            if isinstance(settings, dict)
            and settings.get("temperature") == temperature
            and settings.get("top_p") == top_p
            and settings.get("replicate_seeds") == replicate_seeds
        ]
        if len(matching_regimes) != 1:
            raise ValueError(
                "Runtime decoding does not match exactly one frozen scoped "
                f"regime: observed={{'temperature': {temperature}, 'top_p': "
                f"{top_p}, 'replicate_seeds': {replicate_seeds}}}"
            )
        decoding_regime = matching_regimes[0]
    else:
        decoding_regime = None

    model_artifact = fingerprint_artifact_path(model_path)
    if expected_model_sha256 is not None:
        expected_model_sha256 = validate_sha256(
            expected_model_sha256,
            label="expected_model_sha256",
        )
        if model_artifact["sha256"] != expected_model_sha256:
            raise ValueError(
                "Generation model fingerprint does not match "
                f"--model-sha256: expected {expected_model_sha256}, "
                f"observed {model_artifact['sha256']}"
            )
    resolved_tokenizer_path = tokenizer_path or model_path
    tokenizer_artifact = (
        model_artifact
        if Path(resolved_tokenizer_path).resolve() == Path(model_path).resolve()
        else fingerprint_artifact_path(resolved_tokenizer_path)
    )
    if expected_tokenizer_sha256 is not None:
        expected_tokenizer_sha256 = validate_sha256(
            expected_tokenizer_sha256,
            label="expected_tokenizer_sha256",
        )
        if tokenizer_artifact["sha256"] != expected_tokenizer_sha256:
            raise ValueError(
                "Generation tokenizer fingerprint does not match "
                f"--tokenizer-sha256: expected {expected_tokenizer_sha256}, "
                f"observed {tokenizer_artifact['sha256']}"
            )
    if seed_policy not in {
        ROW_SEED_POLICY_BATCH,
        ROW_SEED_POLICY_SAMPLE,
    }:
        raise ValueError(f"Unsupported seed policy: {seed_policy!r}")
    if masking_strategy not in {
        DELETION_MASKING_STRATEGY,
        NEUTRAL_MASKING_STRATEGY,
    }:
        raise ValueError(f"Unsupported masking strategy: {masking_strategy!r}")
    requested_sample_ids = [
        str(value).strip() for value in (include_sample_ids or [])
    ]
    if any(not value for value in requested_sample_ids):
        raise ValueError("--include-sample-id values must be non-empty")
    if len(requested_sample_ids) != len(set(requested_sample_ids)):
        raise ValueError("--include-sample-id values must be unique")

    prompt_manifest_metadata = None
    if prompt_manifest_file is not None:
        prompt_manifest_metadata = validate_prompt_manifest_binding(
            prompt_manifest_file=prompt_manifest_file,
            input_dir=input_dir,
            masking_strategy=masking_strategy,
        )
    elif masking_strategy == NEUTRAL_MASKING_STRATEGY:
        raise ValueError(
            "Neutral replacement generation requires --prompt-manifest"
        )
    if experiment_config_binding is not None:
        if prompt_manifest_metadata is None:
            raise ValueError(
                "--experiment-config requires --prompt-manifest so the exact "
                "Minutes prompt can be bound before model loading"
            )
        prompt_spec_payload = prompt_manifest_metadata.get(
            "minutes_system_prompt"
        )
        prompt_spec = validate_minutes_prompt_spec(prompt_spec_payload)
        if prompt_spec != minutes_prompt_spec:
            raise ValueError(
                "Prompt manifest Minutes contract differs from --experiment-config"
            )
        prompt_experiment_binding = prompt_manifest_metadata.get(
            "experiment_config"
        )
        if (
            not isinstance(prompt_experiment_binding, dict)
            or prompt_experiment_binding.get("sha256")
            != experiment_config_binding["sha256"]
            or Path(
                str(prompt_experiment_binding.get("path") or "")
            ).expanduser().resolve()
            != Path(experiment_config_binding["path"])
        ):
            raise ValueError(
                "Prompt manifest is not bound to the active scoped experiment "
                "configuration"
            )

    generation_spec_metadata = None
    if generation_spec_file is not None:
        if prompt_manifest_metadata is None:
            raise ValueError(
                "--generation-spec requires --prompt-manifest so source "
                "artifacts can be bound"
            )
        generation_spec_metadata = validate_generation_spec_binding(
            generation_spec_file=generation_spec_file,
            expected_file_sha256=expected_generation_spec_sha256,
            model_artifact_sha256=model_artifact["sha256"],
            tokenizer_artifact_sha256=tokenizer_artifact["sha256"],
            prompt_manifest_metadata=prompt_manifest_metadata,
            input_dir=input_dir,
        )
    elif expected_generation_spec_sha256 is not None:
        raise ValueError(
            "--generation-spec-sha256 requires --generation-spec"
        )

    if masking_strategy == DELETION_MASKING_STRATEGY:
        intervention_manifest = build_intervention_manifest(
            input_dir=Path(input_dir).resolve(),
            input_prompt_files=[
                Path(path).resolve() for path in input_prompt_files
            ],
            roster_file=Path(roster_file).resolve(),
        )
        if prompt_manifest_metadata is not None:
            source_reference = prompt_manifest_metadata["references"][
                "intervention_manifest"
            ]
            if (
                intervention_manifest_file is not None
                and Path(intervention_manifest_file).expanduser().resolve()
                != Path(source_reference["path"])
            ):
                raise ValueError(
                    "Explicit deletion intervention manifest differs from the "
                    "prompt manifest reference"
                )
    else:
        if prompt_manifest_metadata is None:
            raise AssertionError("Neutral prompt manifest was not validated")
        source_reference = prompt_manifest_metadata["references"][
            "neutral_intervention_manifest"
        ]
        resolved_intervention_manifest = (
            Path(intervention_manifest_file).expanduser().resolve()
            if intervention_manifest_file is not None
            else Path(source_reference["path"])
        )
        if resolved_intervention_manifest != Path(source_reference["path"]):
            raise ValueError(
                "Neutral intervention manifest differs from the prompt "
                "manifest reference"
            )
        from transformers import AutoTokenizer

        validation_tokenizer = AutoTokenizer.from_pretrained(
            resolved_tokenizer_path,
            local_files_only=True,
            trust_remote_code=True,
        )
        intervention_manifest = validate_neutral_intervention_manifest(
            manifest=resolved_intervention_manifest,
            input_dir=Path(input_dir).resolve(),
            roster_file=Path(roster_file).resolve(),
            tokenizer=validation_tokenizer,
            tokenizer_artifact_sha256=tokenizer_artifact["sha256"],
        )

    active_roster = _load_intervention_roster(Path(roster_file).resolve())
    generation_input_prompt_files, intervention_scope = (
        select_intervention_prompt_files(
            input_prompt_files=input_prompt_files,
            roster=active_roster,
            include_intervention_indicators=include_intervention_indicators,
        )
    )
    if scoped_experiment is not None:
        scope_mismatches: dict[str, object] = {}
        if scoped_experiment.get("baseline_indicator") != active_roster.get(
            "baseline_indicator"
        ):
            scope_mismatches["baseline_indicator"] = active_roster.get(
                "baseline_indicator"
            )
        if scoped_experiment.get("full_context_roster_id") != active_roster.get(
            "source_roster_id", active_roster.get("roster_id")
        ):
            scope_mismatches["full_context_roster_id"] = active_roster.get(
                "source_roster_id", active_roster.get("roster_id")
            )
        if scoped_experiment.get("full_context_indicators") != active_roster.get(
            "indicators"
        ):
            scope_mismatches["full_context_indicators"] = active_roster.get(
                "indicators"
            )
        if scoped_experiment.get("intervention_indicators") != intervention_scope.get(
            "generated_intervention_indicators"
        ):
            scope_mismatches["intervention_indicators"] = intervention_scope.get(
                "generated_intervention_indicators"
            )
        if prompt_manifest_metadata.get("population_id") not in scoped_experiment.get(
            "populations", {}
        ):
            scope_mismatches["population_id"] = prompt_manifest_metadata.get(
                "population_id"
            )
        if scope_mismatches:
            raise ValueError(
                "Runtime prompts differ from the scoped experiment: "
                f"{scope_mismatches}"
            )
    if intervention_scope["mode"] != "full":
        logging.info(
            "Generating partial intervention shard: %d/%d indicators; "
            "the full-roster baseline remains unchanged.",
            len(intervention_scope["generated_intervention_indicators"]),
            len(intervention_scope["full_roster_indicators"]),
        )
    intervention_manifest_path = Path(output_dir) / "intervention_manifest.json"
    _write_frozen_json(intervention_manifest_path, intervention_manifest)
    intervention_manifest_sha256 = _sha256_file(intervention_manifest_path)

    prompt_preflight: list[dict] = []
    common_full_sample_ids: list[str] | None = None
    requested_sample_set = set(requested_sample_ids)
    for input_prompt_file in generation_input_prompt_files:
        resolved_prompt_file = Path(input_prompt_file).resolve()
        full_rows = list(_read_prompt_rows(resolved_prompt_file).values())
        full_sample_ids = [str(row.get("sample_id") or "").strip() for row in full_rows]
        if any(not sample_id for sample_id in full_sample_ids):
            raise ValueError(
                f"Sample filtering requires sample_id on every prompt row: "
                f"{resolved_prompt_file}"
            )
        if len(full_sample_ids) != len(set(full_sample_ids)):
            raise ValueError(f"Duplicate sample_id in {resolved_prompt_file}")
        if common_full_sample_ids is None:
            common_full_sample_ids = full_sample_ids
        elif full_sample_ids != common_full_sample_ids:
            raise ValueError(
                "Selected prompt artifacts do not share one ordered sample-ID "
                f"universe: {resolved_prompt_file}"
            )
        missing_requested = requested_sample_set - set(full_sample_ids)
        if missing_requested:
            raise ValueError(
                f"Requested smoke sample IDs are absent from "
                f"{resolved_prompt_file}: {sorted(missing_requested)}"
            )
        selected_rows = (
            [row for row in full_rows if row["sample_id"] in requested_sample_set]
            if requested_sample_ids
            else full_rows
        )
        if requested_sample_ids:
            selected_by_id = {row["sample_id"]: row for row in selected_rows}
            selected_rows = [selected_by_id[value] for value in requested_sample_ids]
        prompt_preflight.append(
            {
                "path": str(resolved_prompt_file),
                "full_rows": full_rows,
                "selected_rows": selected_rows,
                "full_source_row_count": len(full_rows),
                "selected_source_row_count": len(selected_rows),
                "source_file_sha256": _sha256_file(resolved_prompt_file),
            }
        )
    if common_full_sample_ids is None:
        raise AssertionError("Prompt preflight unexpectedly found no artifacts")
    selected_sample_ids = (
        requested_sample_ids if requested_sample_ids else common_full_sample_ids
    )
    sample_selection = {
        "policy": (
            "explicit-sample-id-filter-v1"
            if requested_sample_ids
            else "full-population-v1"
        ),
        "mode": "filtered_smoke" if requested_sample_ids else "full_population",
        "requested_sample_ids": requested_sample_ids,
        "selected_sample_ids": selected_sample_ids,
        "selected_row_count_per_artifact": len(selected_sample_ids),
        "full_source_row_count_per_artifact": len(common_full_sample_ids),
        "population_release_allowed": not requested_sample_ids,
    }
    runtime_system_prompt = (
        get_system_prompt()
        if minutes_prompt_spec is None
        else minutes_prompt_spec.text
    )
    runtime_system_prompt_sha256 = _sha256_text(runtime_system_prompt)
    runtime_system_prompt_version = (
        None if minutes_prompt_spec is None else minutes_prompt_spec.version
    )
    runtime_requested_max_output_tokens = (
        None
        if minutes_prompt_spec is None
        else minutes_prompt_spec.requested_max_output_tokens
    )
    runtime_experiment_config_sha256 = (
        None
        if experiment_config_binding is None
        else experiment_config_binding["sha256"]
    )

    logging.info("🚀 Using model: %s", model_path)
    total_generated = 0
    total_skipped = 0
    input_artifacts: list[dict] = []
    expected_artifacts: list[dict] = []
    attempted_row_count = 0
    passed_row_count = 0
    completion_exclusions: list[dict] = []

    for file_idx, preflight in enumerate(prompt_preflight, 1):
        input_prompt_file = preflight["path"]
        relative_path = os.path.relpath(input_prompt_file, input_dir)
        output_prefix = os.path.join(output_dir, relative_path)
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
        logging.info(
            "📘 [%s/%s] Processing: %s",
            file_idx,
            len(generation_input_prompt_files),
            os.path.basename(input_prompt_file),
        )

        full_source_row_count = preflight["full_source_row_count"]
        source_row_count = preflight["selected_source_row_count"]
        source_file_sha256 = preflight["source_file_sha256"]
        source_prompt_rows = preflight["selected_rows"]
        source_stem = Path(input_prompt_file).stem
        if "_masked_" in source_stem:
            indicator, evaluation_context = source_stem.split("_masked_", 1)
        else:
            indicator, evaluation_context = source_stem, "unspecified"
        input_artifacts.append(
            {
                "path": str(Path(input_prompt_file).resolve()),
                "relative_path": relative_path,
                "indicator": indicator,
                "context": evaluation_context,
                "source_row_count": full_source_row_count,
                "full_source_row_count": full_source_row_count,
                "selected_source_row_count": source_row_count,
                "source_file_sha256": source_file_sha256,
            }
        )

        for i in range(simulation_step):
            replicate_seed = seed + (i * 1_000_000)
            section_output_file = output_prefix.replace(".jsonl", f"_{i}.jsonl")
            completion_audit = None
            if require_normal_finish and Path(section_output_file).is_file():
                completion_audit = apply_completion_policy_to_output(
                    section_output_file,
                    max_new_tokens=max_new_tokens,
                    max_model_len=max_model_len,
                    token_limit_policy=token_limit_policy,
                )
            output_is_valid = validate_output_file(
                section_output_file,
                replicate_id=i,
                replicate_seed=replicate_seed,
                expected_row_count=source_row_count,
                source_file_sha256=source_file_sha256,
                batch_size=batch_size,
                model_path=model_path,
                model_artifact_sha256=model_artifact["sha256"],
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens,
                masking_strategy=masking_strategy,
                indicator=indicator,
                evaluation_context=evaluation_context,
                expected_prompt_rows=source_prompt_rows,
                seed_policy=seed_policy,
                max_model_len=max_model_len,
                require_normal_finish=require_normal_finish,
                tokenizer_path=resolved_tokenizer_path,
                tokenizer_artifact_sha256=tokenizer_artifact["sha256"],
                token_limit_policy=token_limit_policy,
                system_prompt_sha256=runtime_system_prompt_sha256,
                system_prompt_version=runtime_system_prompt_version,
                requested_max_output_tokens=(
                    runtime_requested_max_output_tokens
                ),
                experiment_config_sha256=(
                    runtime_experiment_config_sha256
                ),
            )
            if output_is_valid:
                logging.info("   ⚙️ Skipping existing file: %s", os.path.basename(section_output_file))
                total_skipped += 1
            else:
                logging.info("   ├── Generating synthetic sample %s/%s → %s", i + 1, simulation_step, section_output_file)
                generated_rows = generate_new_response(
                    source_prompt_rows,
                    model_path,
                    section_output_file,
                    batch_size=batch_size,
                    seed=replicate_seed,
                    replicate_id=i,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    max_model_len=max_model_len,
                    tokenizer_path=resolved_tokenizer_path,
                    system_prompt=runtime_system_prompt,
                    seed_policy=seed_policy,
                    generation_metadata={
                        "indicator": indicator,
                        "evaluation_context": evaluation_context,
                        "masking_strategy": masking_strategy,
                        "generation_model_sha256": model_artifact["sha256"],
                        "generation_tokenizer_sha256": tokenizer_artifact[
                            "sha256"
                        ],
                        "source_file": str(Path(input_prompt_file).resolve()),
                        "source_file_sha256": source_file_sha256,
                        "source_row_count": source_row_count,
                        "generation_system_prompt_version": (
                            runtime_system_prompt_version
                        ),
                        "generation_system_prompt_sha256": (
                            runtime_system_prompt_sha256
                        ),
                        "generation_requested_max_output_tokens": (
                            runtime_requested_max_output_tokens
                        ),
                        "experiment_config_sha256": (
                            runtime_experiment_config_sha256
                        ),
                    },
                )
                if len(generated_rows) != source_row_count:
                    raise RuntimeError(
                        f"Generation for {input_prompt_file} replicate {i} produced "
                        f"{len(generated_rows)}/{source_row_count} rows. The partial "
                        f"output remains at {section_output_file} for diagnosis."
                    )
                if require_normal_finish:
                    completion_audit = apply_completion_policy_to_output(
                        section_output_file,
                        max_new_tokens=max_new_tokens,
                        max_model_len=max_model_len,
                        token_limit_policy=token_limit_policy,
                    )
                if not validate_output_file(
                    section_output_file,
                    replicate_id=i,
                    replicate_seed=replicate_seed,
                    expected_row_count=source_row_count,
                    source_file_sha256=source_file_sha256,
                    batch_size=batch_size,
                    model_path=model_path,
                    model_artifact_sha256=model_artifact["sha256"],
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=max_new_tokens,
                    masking_strategy=masking_strategy,
                    indicator=indicator,
                    evaluation_context=evaluation_context,
                    expected_prompt_rows=source_prompt_rows,
                    seed_policy=seed_policy,
                    max_model_len=max_model_len,
                    require_normal_finish=require_normal_finish,
                    tokenizer_path=resolved_tokenizer_path,
                    tokenizer_artifact_sha256=tokenizer_artifact["sha256"],
                    token_limit_policy=token_limit_policy,
                    system_prompt_sha256=runtime_system_prompt_sha256,
                    system_prompt_version=runtime_system_prompt_version,
                    requested_max_output_tokens=(
                        runtime_requested_max_output_tokens
                    ),
                    experiment_config_sha256=(
                        runtime_experiment_config_sha256
                    ),
                ):
                    raise RuntimeError(
                        f"Generated output failed completeness validation: "
                        f"{section_output_file}"
                    )
                total_generated += 1

            if completion_audit is None:
                completion_audit = {
                    "policy": "not-evaluated-v1",
                    "token_limit_policy": TOKEN_LIMIT_POLICY_ERROR,
                    "attempted_row_count": source_row_count,
                    "passed_row_count": source_row_count,
                    "excluded_row_count": 0,
                    "exclusions": [],
                }
            artifact_relative_path = os.path.relpath(
                section_output_file,
                output_dir,
            )
            artifact_exclusions = [
                {
                    **record,
                    "artifact_relative_path": artifact_relative_path,
                    "indicator": indicator,
                    "context": evaluation_context,
                    "replicate_id": str(i),
                }
                for record in completion_audit["exclusions"]
            ]
            attempted_row_count += int(
                completion_audit["attempted_row_count"]
            )
            passed_row_count += int(completion_audit["passed_row_count"])
            completion_exclusions.extend(artifact_exclusions)
            expected_artifacts.append(
                {
                    "path": str(Path(section_output_file).resolve()),
                    "relative_path": artifact_relative_path,
                    "indicator": indicator,
                    "context": evaluation_context,
                    "replicate_id": str(i),
                    "replicate_seed": replicate_seed,
                    "model_artifact_sha256": model_artifact["sha256"],
                    "source_file_sha256": source_file_sha256,
                    "source_row_count": source_row_count,
                    "full_source_row_count": full_source_row_count,
                    "selected_source_row_count": source_row_count,
                    "attempted_row_count": completion_audit[
                        "attempted_row_count"
                    ],
                    "scorable_row_count": completion_audit[
                        "passed_row_count"
                    ],
                    "excluded_row_count": completion_audit[
                        "excluded_row_count"
                    ],
                    "completion_exclusions": artifact_exclusions,
                    "output_sha256": _sha256_file(section_output_file),
                }
            )

    logging.info("✅ All done! Total generated: %s | Skipped: %s", total_generated, total_skipped)
    validate_all_outputs(expected_artifacts)
    final_model_artifact = fingerprint_artifact_path(model_path)
    if final_model_artifact["sha256"] != model_artifact["sha256"]:
        raise RuntimeError(
            "The frozen generation model changed while outputs were being "
            "generated; the run is invalid"
        )
    final_tokenizer_artifact = (
        final_model_artifact
        if Path(resolved_tokenizer_path).resolve() == Path(model_path).resolve()
        else fingerprint_artifact_path(resolved_tokenizer_path)
    )
    if final_tokenizer_artifact["sha256"] != tokenizer_artifact["sha256"]:
        raise RuntimeError(
            "The frozen generation tokenizer changed while outputs were "
            "being generated; the run is invalid"
        )
    excluded_row_count = len(completion_exclusions)
    manifest = {
        "schema_version": GENERATION_SCHEMA_VERSION,
        "status": (
            "complete_with_exclusions"
            if excluded_row_count
            else "complete"
        ),
        "generation_attempts_complete": True,
        "scorable_population_complete": excluded_row_count == 0,
        "input_folder": str(Path(input_dir).resolve()),
        "output_dir": str(Path(output_dir).resolve()),
        "model_path": model_path,
        "model_artifact": model_artifact,
        "model_artifact_after_generation": final_model_artifact,
        "tokenizer_path": resolved_tokenizer_path,
        "tokenizer_artifact": tokenizer_artifact,
        "tokenizer_artifact_after_generation": final_tokenizer_artifact,
        "simulation_step": simulation_step,
        "base_seed": seed,
        "replicate_seeds": [seed + (index * 1_000_000) for index in range(simulation_step)],
        "batch_size": batch_size,
        "temperature": temperature,
        "top_p": top_p,
        "max_new_tokens": max_new_tokens,
        "max_model_len": max_model_len,
        "seed_policy": seed_policy,
        "require_normal_finish": require_normal_finish,
        "completion_validation": {
            "policy": (
                TOKEN_LIMIT_EXCLUSION_POLICY
                if token_limit_policy == TOKEN_LIMIT_POLICY_EXCLUDE
                else "strict-token-limit-error-v1"
            ),
            "token_limit_policy": token_limit_policy,
            "attempted_row_count": attempted_row_count,
            "passed_row_count": passed_row_count,
            "excluded_row_count": excluded_row_count,
            "exclusions": completion_exclusions,
        },
        "system_prompt_sha256": runtime_system_prompt_sha256,
        "minutes_system_prompt": (
            None if minutes_prompt_spec is None else minutes_prompt_spec.as_dict()
        ),
        "experiment_config": experiment_config_binding,
        "decoding_regime": decoding_regime,
        "masking_strategy": masking_strategy,
        "intervention_scope": intervention_scope,
        "sample_selection": sample_selection,
        "intervention_manifest": {
            "relative_path": intervention_manifest_path.name,
            "sha256": intervention_manifest_sha256,
            "schema_version": intervention_manifest["schema_version"],
            "roster_id": intervention_manifest["roster_id"],
            "roster_file_sha256": intervention_manifest["roster_file_sha256"],
        },
        "prompt_manifest": prompt_manifest_metadata,
        "generation_spec": generation_spec_metadata,
        "execution_environment": (
            None
            if generation_spec_metadata is None
            else generation_spec_metadata.get("execution_environment")
        ),
        "input_files": [
            str(Path(path).resolve())
            for path in generation_input_prompt_files
        ],
        "input_artifacts": input_artifacts,
        "expected_artifacts": expected_artifacts,
    }
    manifest_path = Path(output_dir) / "generation_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest_path = manifest_path.with_name(f".{manifest_path.name}.tmp")
    temporary_manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_manifest_path.replace(manifest_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate masked prompt outputs for leave-one-out masking evaluation.")
    parser.add_argument("--input-folder", required=True, help="Input folder name under dataset/processed/main/evaluation_inputs/... or an absolute path.")
    parser.add_argument("--model", required=True, help="Model path used for generation.")
    parser.add_argument("--simulation-step", type=int, default=1)
    parser.add_argument("--output-dir", required=True, help="Destination directory for generated files.")
    parser.add_argument(
        "--roster-file",
        required=True,
        help=(
            "Frozen JSON roster declaring the complete baseline, indicator, and "
            "context universe."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=8192)
    parser.add_argument("--max-model-len", type=int, default=16384)
    parser.add_argument("--masking-strategy", default="indicator_block_deletion")
    parser.add_argument(
        "--seed-policy",
        choices=[ROW_SEED_POLICY_BATCH, ROW_SEED_POLICY_SAMPLE],
        default=ROW_SEED_POLICY_SAMPLE,
    )
    parser.add_argument(
        "--model-sha256",
        dest="expected_model_sha256",
        help="Expected immutable directory fingerprint for the generation model.",
    )
    parser.add_argument(
        "--tokenizer",
        dest="tokenizer_path",
        help="Optional external tokenizer path; defaults to --model.",
    )
    parser.add_argument(
        "--tokenizer-sha256",
        dest="expected_tokenizer_sha256",
        help="Expected immutable directory fingerprint for the tokenizer.",
    )
    parser.add_argument(
        "--require-normal-finish",
        action="store_true",
        help=(
            "Reject outputs without token counts or a normal non-length "
            "finish reason."
        ),
    )
    parser.add_argument(
        "--token-limit-policy",
        choices=sorted(TOKEN_LIMIT_POLICIES),
        default=TOKEN_LIMIT_POLICY_ERROR,
        help=(
            "Use 'error' for fail-closed canonical generation. Use 'exclude' "
            "to retain token-limit rows as audited raw records, mark them "
            "unscorable, and continue generating later artifacts."
        ),
    )
    parser.add_argument(
        "--intervention-manifest",
        dest="intervention_manifest_file",
        help=(
            "Run-scoped intervention manifest. Required for neutral prompts "
            "unless it is discoverable from --prompt-manifest."
        ),
    )
    parser.add_argument(
        "--prompt-manifest",
        dest="prompt_manifest_file",
        help="Frozen prompt-builder manifest used to bind input artifacts.",
    )
    parser.add_argument(
        "--generation-spec",
        dest="generation_spec_file",
        help="Frozen loo-generation-spec-v1 JSON for this experiment.",
    )
    parser.add_argument(
        "--generation-spec-sha256",
        dest="expected_generation_spec_sha256",
        help="Optional externally anchored SHA-256 of --generation-spec.",
    )
    parser.add_argument(
        "--include-intervention-indicator",
        dest="include_intervention_indicators",
        action="append",
        help=(
            "Generate only this intervention indicator while retaining the "
            "complete frozen baseline. Repeat for a validated partial shard. "
            "The baseline indicator is always included automatically."
        ),
    )
    parser.add_argument(
        "--include-sample-id",
        dest="include_sample_ids",
        action="append",
        help=(
            "Generate only this sample_id in every selected prompt artifact. "
            "Repeat for a non-inferential smoke run. Filtered output is never "
            "eligible for a population release."
        ),
    )
    parser.add_argument(
        "--experiment-config",
        dest="experiment_config_file",
        help=(
            "Scoped experiment JSON containing the exact Minutes prompt and "
            "decoding contract."
        ),
    )
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args()
    run_mask_generation(
        input_folder=args.input_folder,
        model_path=args.model,
        simulation_step=args.simulation_step,
        output_dir=args.output_dir,
        roster_file=args.roster_file,
        batch_size=args.batch_size,
        seed=args.seed,
        temperature=args.temperature,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        max_model_len=args.max_model_len,
        masking_strategy=args.masking_strategy,
        seed_policy=args.seed_policy,
        expected_model_sha256=args.expected_model_sha256,
        tokenizer_path=args.tokenizer_path,
        expected_tokenizer_sha256=args.expected_tokenizer_sha256,
        require_normal_finish=args.require_normal_finish,
        intervention_manifest_file=args.intervention_manifest_file,
        prompt_manifest_file=args.prompt_manifest_file,
        generation_spec_file=args.generation_spec_file,
        expected_generation_spec_sha256=(
            args.expected_generation_spec_sha256
        ),
        include_intervention_indicators=(
            args.include_intervention_indicators
        ),
        include_sample_ids=args.include_sample_ids,
        token_limit_policy=args.token_limit_policy,
        experiment_config_file=args.experiment_config_file,
    )
