"""Authorize and execute the user-requested chk1 cp200 -> chk2 parent merge.

The failed source semantic audit remains explicit in every receipt.  This script
does not alter the source adapter, old authorization, dataset, or prior models.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from jobs.main.checkpoint_provenance import write_immutable_json
from jobs.retrain_v2.merge_adapter import merge_from_config
from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
CONFIG_REL = Path(
    "configs/retrain_v2/"
    "chk1_clean_v2_lr1e6_cp200_merge_for_chk2_20260810.yaml"
)
CONFIG = ROOT / CONFIG_REL
SOURCE_CONFIG_REL = Path(
    "configs/retrain_v2/"
    "chk1_analysis_sft_clean_v2_override_full3ep_lr1e6_maxlen7168_20260810.yaml"
)
SOURCE_CONFIG = ROOT / SOURCE_CONFIG_REL
BASE_REL = Path("models/DeepSeek-R1-Distill-Llama-8B")
BASE = ROOT / BASE_REL
ADAPTER_REL = Path(
    "output/training/retrain_v2/"
    "chk1_clean_v2_override_full3ep_lr1e6_maxlen7168_20260810/"
    "adapters/chk1/checkpoint-200"
)
ADAPTER = ROOT / ADAPTER_REL
MERGED_REL = Path(
    "output/training/retrain_v2/"
    "chk1_clean_v2_lr1e6_selected_cp200_for_chk2_20260810/"
    "merged/chk1"
)
MERGED = ROOT / MERGED_REL
OLD_AUTH_REL = Path(
    "docs/summary/20260809T234411Z/chk1_semantic_override_authorization.json"
)
OLD_AUTH = ROOT / OLD_AUTH_REL
AUDIT_REL = Path(
    "output/data/retrain_v2/chk1/"
    "chk1_reasoning_compressed_flash_max_v2_clean_20260809_qwen_source_audit/"
    "summary.json"
)
AUDIT = ROOT / AUDIT_REL
CANDIDATE_MANIFEST_REL = Path(
    "output/data/retrain_v2/chk1/"
    "chk1_reasoning_compressed_flash_max_v2_clean_20260809_candidate/"
    "candidate_manifest.json"
)
CANDIDATE_MANIFEST = ROOT / CANDIDATE_MANIFEST_REL
VALIDATION_REL = Path(
    "docs/summary/20260809T213520Z/chk1_clean_sft_token_validation.json"
)
VALIDATION = ROOT / VALIDATION_REL
RUNTIME_REL = Path(
    "output/training/retrain_v2/"
    "chk1_clean_v2_override_full3ep_lr1e6_maxlen7168_20260810/"
    "adapters/chk1/resolved_runtime_config.json"
)
RUNTIME = ROOT / RUNTIME_REL
PROBE_REL = Path(
    "docs/summary/20260810T113500Z/"
    "chk1_lr1e6_cp200_vs_chk0_comparison.json"
)
PROBE = ROOT / PROBE_REL
SELECTION_REL = Path(
    "docs/summary/20260810T123000Z/chk1_checkpoint_selection/artifact.json"
)
SELECTION = ROOT / SELECTION_REL
AUTH = HERE / "chk1_cp200_to_chk2_authorization.json"
PROMOTION = HERE / "promotion_manifest.json"
MERGE_RESULT = HERE / "merge_result.json"


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _config_payload() -> dict[str, Any]:
    value = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    _require(isinstance(value, dict), "Merge config must be a mapping")
    _require(
        value.get("model_name_or_path") == str(BASE_REL),
        "Merge config base path mismatch",
    )
    _require(
        value.get("output_dir") == str(ADAPTER_REL),
        "Merge config adapter path mismatch",
    )
    _require(
        value.get("peft_merged_model_path") == str(MERGED_REL),
        "Merge config destination mismatch",
    )
    _require(value.get("semantic_audit_status") == "failed", "Audit risk hidden")
    return value


def _validate_sources() -> dict[str, Any]:
    config = _config_payload()
    for path in (
        SOURCE_CONFIG,
        BASE,
        ADAPTER,
        OLD_AUTH,
        AUDIT,
        CANDIDATE_MANIFEST,
        VALIDATION,
        RUNTIME,
        PROBE,
        SELECTION,
    ):
        _require(path.exists(), f"Required source is missing: {path}")
    _require(not MERGED.exists(), f"Merged destination already exists: {MERGED}")

    state = _read_json(ADAPTER / "trainer_state.json")
    _require(state.get("global_step") == 200, "Adapter is not checkpoint-200")
    adapter_config = _read_json(ADAPTER / "adapter_config.json")
    _require(adapter_config.get("peft_type") == "LORA", "Adapter is not LoRA")
    _require(
        adapter_config.get("base_model_name_or_path") == str(BASE_REL),
        "Adapter parent base mismatch",
    )
    _require(adapter_config.get("r") == 32, "Adapter rank mismatch")
    _require(adapter_config.get("lora_alpha") == 64, "Adapter alpha mismatch")

    old_auth = _read_json(OLD_AUTH)
    _require(old_auth.get("status") == "authorized", "Old chk1 authorization invalid")
    _require(
        old_auth.get("scope", {}).get("downstream_stages_allowed") == [],
        "Old authorization scope drifted",
    )

    audit = _read_json(AUDIT)
    _require(audit.get("status") == "failed", "Expected failed semantic audit")
    counts = audit.get("counts", {})
    _require(counts.get("failed") == 234, "Semantic audit failed-count drift")
    _require(counts.get("judge_errors") == 2, "Semantic audit error-count drift")
    _require(
        counts.get("blocking_violations") == 611,
        "Semantic audit blocker-count drift",
    )

    probe = _read_json(PROBE)
    _require(probe.get("status") == "passed", "Checkpoint-200 probe did not pass")
    candidate = probe.get("candidate", {})
    for key, expected in (
        ("finite_rate", 1.0),
        ("eos_rate", 1.0),
        ("contract_valid_rate", 1.0),
        ("cap_rate", 0.0),
        ("periodic_tail_rate", 0.0),
        ("catastrophic_cases", 0),
    ):
        _require(candidate.get(key) == expected, f"Probe gate drift: {key}")

    _require(
        sha256_file(SOURCE_CONFIG) == config["source_training_config_sha256"],
        "Source training config hash mismatch",
    )
    _require(
        sha256_file(PROBE) == config["source_probe_comparison_sha256"],
        "Probe comparison hash mismatch",
    )
    _require(
        sha256_file(SELECTION) == config["selection_artifact_sha256"],
        "Selection artifact hash mismatch",
    )
    return {
        "config": config,
        "state": state,
        "adapter_config": adapter_config,
        "old_auth": old_auth,
        "audit": audit,
        "probe": probe,
    }


def authorize() -> dict[str, Any]:
    sources = _validate_sources()
    adapter_fingerprint = fingerprint_artifact_path(ADAPTER)
    base_fingerprint = fingerprint_artifact_path(BASE)
    payload_without_sha = {
        "schema_version": "chk1-cp200-to-chk2-override-authorization-v1",
        "status": "authorized",
        "authorization_basis": "explicit_user_instruction",
        "authorized_at_utc": "2026-08-10T12:45:00Z",
        "authorized_by": "workspace_user",
        "user_instruction": "现在使用cp200帮我merge出新模型用于chk2的训练",
        "scope": {
            "source_stage": "chk1",
            "target_stage": "chk2",
            "allowed_operations": [
                "merge_checkpoint_200",
                "use_merged_checkpoint_200_as_chk2_parent",
            ],
            "downstream_stages_allowed": ["chk2"],
            "further_downstream_stages_allowed": [],
        },
        "bindings": {
            "base_model": base_fingerprint,
            "source_adapter": adapter_fingerprint,
            "adapter_weights_sha256": sha256_file(
                ADAPTER / "adapter_model.safetensors"
            ),
            "checkpoint_step": 200,
            "source_training_config": {
                "path": str(SOURCE_CONFIG_REL),
                "sha256": sha256_file(SOURCE_CONFIG),
            },
            "merge_config": {
                "path": str(CONFIG_REL),
                "sha256": sha256_file(CONFIG),
            },
            "merged_destination": str(MERGED_REL),
            "candidate_manifest": {
                "path": str(CANDIDATE_MANIFEST_REL),
                "sha256": sha256_file(CANDIDATE_MANIFEST),
            },
            "deterministic_validation_receipt": {
                "path": str(VALIDATION_REL),
                "sha256": sha256_file(VALIDATION),
            },
            "semantic_audit_summary": {
                "path": str(AUDIT_REL),
                "sha256": sha256_file(AUDIT),
                "status": sources["audit"]["status"],
                "counts": sources["audit"]["counts"],
            },
            "original_chk1_authorization": {
                "path": str(OLD_AUTH_REL),
                "sha256": sha256_file(OLD_AUTH),
                "downstream_stages_allowed": [],
            },
            "training_runtime_receipt": {
                "path": str(RUNTIME_REL),
                "sha256": sha256_file(RUNTIME),
            },
            "checkpoint_200_probe": {
                "path": str(PROBE_REL),
                "sha256": sha256_file(PROBE),
                "status": sources["probe"]["status"],
                "sample_manifest_sha256": sources["probe"][
                    "sample_manifest_sha256"
                ],
            },
            "checkpoint_selection_artifact": {
                "path": str(SELECTION_REL),
                "sha256": sha256_file(SELECTION),
            },
        },
        "risk_acknowledgements": [
            "semantic_audit_failed",
            "semantic_audit_contains_611_blocking_violations",
            "semantic_audit_contains_2_judge_errors",
            "override_is_limited_to_chk2_parent_use",
            "not_authorized_for_chk3_or_chk4",
            "canonical_sealed_chk1_release_is_absent",
        ],
    }
    payload = {
        **payload_without_sha,
        "authorization_sha256": _canonical_sha256(payload_without_sha),
    }
    file_sha = write_immutable_json(AUTH, payload)
    promotion = seal_manifest(
        {
            "schema_version": "chk1-checkpoint-promotion-manifest-v1",
            "status": "authorized_for_merge",
            "claim_scope": (
                "Promotes the existing nested checkpoint-200 for an explicitly "
                "authorized chk2-parent merge. It is not a retroactive canonical "
                "training receipt and does not claim the semantic audit passed."
            ),
            "source_stage": "chk1",
            "target_stage": "chk2",
            "checkpoint_step": 200,
            "authorization": {
                "path": str(AUTH.relative_to(ROOT)),
                "file_sha256": file_sha,
                "authorization_sha256": payload["authorization_sha256"],
            },
            "bindings": payload["bindings"],
            "training_receipt": {
                "status": "unavailable_not_applicable",
                "reason": (
                    "The selected source is a historical nested checkpoint, not "
                    "the canonical adapter root; retroactive receipt creation "
                    "would misstate the actual training path."
                ),
            },
            "semantic_audit": payload["bindings"]["semantic_audit_summary"],
            "risk_acknowledgements": payload["risk_acknowledgements"],
        }
    )
    promotion_file_sha = write_immutable_json(PROMOTION, promotion)
    return {
        "status": "authorized",
        "receipt": str(AUTH),
        "file_sha256": file_sha,
        "promotion_manifest": str(PROMOTION),
        "promotion_manifest_file_sha256": promotion_file_sha,
        "promotion_payload_sha256": promotion["integrity"]["payload_sha256"],
    }


def _load_authorization() -> tuple[dict[str, Any], str]:
    payload = _read_json(AUTH)
    recorded = payload.get("authorization_sha256")
    without_sha = dict(payload)
    without_sha.pop("authorization_sha256", None)
    _require(recorded == _canonical_sha256(without_sha), "Authorization hash mismatch")
    _require(payload.get("status") == "authorized", "Authorization is not active")
    scope = payload.get("scope", {})
    _require(scope.get("target_stage") == "chk2", "Authorization target drift")
    _require(
        scope.get("downstream_stages_allowed") == ["chk2"],
        "Authorization downstream scope drift",
    )
    _require(
        payload.get("bindings", {}).get("merge_config", {}).get("sha256")
        == sha256_file(CONFIG),
        "Authorization merge-config binding drift",
    )
    return payload, sha256_file(AUTH)


def _load_promotion() -> tuple[dict[str, Any], str]:
    payload = _read_json(PROMOTION)
    payload_sha = validate_manifest_integrity(payload)
    _require(
        payload.get("status") == "authorized_for_merge",
        "Promotion is not authorized for merge",
    )
    _require(payload.get("checkpoint_step") == 200, "Promotion step drift")
    _require(
        payload.get("semantic_audit", {}).get("status") == "failed",
        "Promotion hid semantic audit failure",
    )
    _require(
        payload.get("authorization", {}).get("file_sha256") == sha256_file(AUTH),
        "Promotion authorization binding drift",
    )
    return payload, payload_sha


def merge() -> dict[str, Any]:
    _validate_sources()
    authorization, authorization_file_sha = _load_authorization()
    promotion, promotion_payload_sha = _load_promotion()
    binding = {
        "schema_version": "chk1-cp200-to-chk2-merge-binding-v1",
        "source_stage": "chk1",
        "target_stage": "chk2",
        "source_checkpoint_step": 200,
        "base_model": authorization["bindings"]["base_model"],
        "source_adapter": authorization["bindings"]["source_adapter"],
        "merge_config": authorization["bindings"]["merge_config"],
        "authorization": {
            "path": str(AUTH.relative_to(ROOT)),
            "file_sha256": authorization_file_sha,
            "authorization_sha256": authorization["authorization_sha256"],
        },
        "promotion_manifest": {
            "path": str(PROMOTION.relative_to(ROOT)),
            "file_sha256": sha256_file(PROMOTION),
            "payload_sha256": promotion_payload_sha,
            "training_receipt_status": promotion["training_receipt"]["status"],
        },
        "semantic_audit": authorization["bindings"]["semantic_audit_summary"],
        "checkpoint_probe": authorization["bindings"]["checkpoint_200_probe"],
        "checkpoint_selection": authorization["bindings"][
            "checkpoint_selection_artifact"
        ],
        "scope_boundary": {
            "allowed": ["chk2_parent_model"],
            "not_authorized": ["chk3", "chk4"],
        },
    }
    result = merge_from_config(
        CONFIG,
        merge_attestation_binding=binding,
        repo_root=ROOT,
    )
    result_payload = {
        "schema_version": "chk1-cp200-to-chk2-merge-result-v1",
        "status": "merged",
        "authorization": binding["authorization"],
        "merge": result,
    }
    result_file_sha = write_immutable_json(MERGE_RESULT, result_payload)
    return {
        "status": "merged",
        "destination": str(MERGED),
        "merge_result": str(MERGE_RESULT),
        "merge_result_file_sha256": result_file_sha,
        "merged_artifact": result["merged_artifact"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("authorize", "merge"))
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = authorize() if args.action == "authorize" else merge()
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
