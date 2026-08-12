from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from jobs.retrain_v2 import merge_chk4_selected_sft_checkpoint as selected
from jobs.retrain_v2.merge_attestation import (
    SEMANTIC_BOUNDARY,
    SEMANTIC_METHOD,
    create_merge_attestation,
)
from jobs.retrain_v2.probe_chk4_decision_grpo_stratified import (
    replay_decision_dense_v3,
    summarize_stratified_results,
)
from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.validator.loo_generation_spec import seal_manifest


SOURCE_ROOT = Path(__file__).resolve().parents[1]


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _copy_config(source_relative: Path, destination: Path) -> dict[str, Any]:
    value = yaml.safe_load((SOURCE_ROOT / source_relative).read_text(encoding="utf-8"))
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(value, sort_keys=False), encoding="utf-8")
    return value


def _adapter_config(base: str, sft: dict[str, Any]) -> dict[str, Any]:
    return {
        "base_model_name_or_path": base,
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": sft["peft_r"],
        "lora_alpha": sft["peft_lora_alpha"],
        "lora_dropout": sft["peft_lora_dropout"],
        "bias": sft["peft_bias"],
        "target_modules": sft["peft_target_modules"],
        "use_dora": False,
        "use_rslora": False,
    }


def _make_checkpoint(
    output: Path, *, step: int, sft: dict[str, Any], base: str
) -> Path:
    checkpoint = output / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    _write_json(checkpoint / "adapter_config.json", _adapter_config(base, sft))
    (checkpoint / "adapter_model.safetensors").write_bytes(f"adapter-{step}".encode())
    _write_json(
        checkpoint / "trainer_state.json",
        {"global_step": step, "max_steps": 24, "epoch": step / 8},
    )
    for name in ("optimizer.pt", "scheduler.pt", "training_args.bin", "rng_state.pth"):
        (checkpoint / name).write_bytes(f"{name}-{step}".encode())
    return checkpoint


def _create_parent_lineage(
    root: Path,
    *,
    run_root: Path,
    base: Path,
    parent_config: Path,
    sft_config: Path,
    grpo_config: Path,
) -> None:
    source_root = root / "upstream_parent_lineage"
    source_parent = source_root / "parent_stage.json"
    source_authorization = source_root / "authorization.json"
    source_exact = source_root / "parent_exact_merge.json"
    source_attestation = base / "merge_attestation.json"
    _write_json(source_parent, {"status": "passed", "receipt_sha256": "b" * 64})
    _write_json(
        source_authorization,
        {"status": "authorized", "authorization_sha256": "c" * 64},
    )
    attestation_binding = {"scope": "test-upstream-parent"}
    create_merge_attestation(
        base,
        binding=attestation_binding,
        semantic_evidence={
            "method": SEMANTIC_METHOD,
            "merge_completed": True,
            "lora_modules_before": 1,
            "residual_lora_modules_after": 0,
            "residual_lora_state_keys_after": 0,
            "merged_parameter_tensors": 1,
            "merged_parameter_count": 1,
            "functional_equivalence_boundary": SEMANTIC_BOUNDARY,
        },
    )
    base_fingerprint = fingerprint_artifact_path(base)
    exact = seal_manifest(
        {
            "schema_version": "test-exact-v1",
            "subject_artifact_id": "test-parent",
            "conclusion": selected.EXACT_CONCLUSION,
            "sources": {"merged_model": base_fingerprint},
        }
    )
    _write_json(source_exact, exact)
    reused = {
        "merged_artifact": base_fingerprint,
        "source_parent_stage": {
            "path": str(source_parent.resolve()),
            "file_sha256": sha256_file(source_parent),
            "receipt_sha256": "b" * 64,
        },
        "source_authorization": {
            "path": str(source_authorization.resolve()),
            "file_sha256": sha256_file(source_authorization),
            "payload_sha256": "c" * 64,
        },
        "source_exact_merge_evidence": {
            "path": str(source_exact.resolve()),
            "file_sha256": sha256_file(source_exact),
            "payload_sha256": exact["integrity"]["payload_sha256"],
        },
        "source_merge_attestation": {
            "path": str(source_attestation.resolve()),
            "file_sha256": sha256_file(source_attestation),
            "binding_sha256": hashlib.sha256(
                json.dumps(
                    attestation_binding,
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
        },
        "source_branch_id": "test-upstream",
        "sources": {},
    }
    authorization_payload = {
        "schema_version": "chk1-cp200-to-chk4-authorization-v1",
        "status": "authorized",
        "authorization_basis": "explicit_user_instruction",
        "profile": selected.PROFILE_NAME,
        "branch_id": selected.BRANCH_ID,
        "bindings": {
            "configs": {
                "parent": {
                    "path": str(parent_config.resolve()),
                    "sha256": sha256_file(parent_config),
                },
                "sft": {
                    "path": str(sft_config.resolve()),
                    "sha256": sha256_file(sft_config),
                },
                "grpo": {
                    "path": str(grpo_config.resolve()),
                    "sha256": sha256_file(grpo_config),
                },
            },
            "run_root": str(run_root.resolve()),
            "release_manifest": {"sha256": selected.RELEASE_MANIFEST_SHA256},
            "reused_parent": reused,
        },
    }
    authorization = {
        **authorization_payload,
        "authorization_sha256": hashlib.sha256(
            json.dumps(
                authorization_payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }
    authorization_path = run_root / "receipts/chk1_cp200_to_chk4_authorization.json"
    _write_json(authorization_path, authorization)
    parent_stage_payload = {
        "schema_version": "chk4-reused-parent-stage-receipt-v1",
        "status": "passed",
        "profile": selected.PROFILE_NAME,
        "branch_id": selected.BRANCH_ID,
        "stage": "parent_reuse",
        "config": authorization["bindings"]["configs"]["parent"],
        "authorization": {
            "path": str(authorization_path.resolve()),
            "file_sha256": sha256_file(authorization_path),
            "payload_sha256": authorization["authorization_sha256"],
        },
        "reused_parent": reused,
        "merged_artifact": base_fingerprint,
    }
    parent_stage = {
        **parent_stage_payload,
        "receipt_sha256": hashlib.sha256(
            json.dumps(
                parent_stage_payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    }
    _write_json(run_root / "receipts/parent_stage.json", parent_stage)


def _pilot_artifacts(
    root: Path,
    *,
    grpo_config: Path,
    base: Path,
    checkpoint: Path,
    passed: bool = True,
) -> tuple[Path, str, Path, str]:
    samples = []
    for direction_index, (direction, magnitude) in enumerate(
        (("hold", 0), ("hike", 25), ("cut", 25))
    ):
        first_seed = 100 + direction_index * 5
        samples.append(
            {
                "sample_id": f"sample-{direction}",
                "split": "train",
                "direction": direction,
                "magnitude_bp": magnitude,
                "greedy_seed": first_seed,
                "sample_seeds": [first_seed + index for index in range(1, 5)],
            }
        )
    tokenizer_sha = "a" * 64
    manifest = {
        "schema_version": selected.PILOT_MANIFEST_SCHEMA,
        "release": {"manifest_sha256": selected.RELEASE_MANIFEST_SHA256},
        "dataset": {"role": "decision_grpo", "split": "train"},
        "prompt_contract": {
            "training_config": str(grpo_config),
            "training_config_sha256": sha256_file(grpo_config),
        },
        "tokenizer": {"bundle_sha256": tokenizer_sha},
        "selection": {"source_split": "train"},
        "samples": samples,
    }
    manifest_path = root / "pilot/sample_manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha = sha256_file(manifest_path)

    base_fingerprint = fingerprint_artifact_path(base)
    adapter_fingerprint = fingerprint_artifact_path(checkpoint)
    composition_payload = {
        "base_model_directory_sha256": base_fingerprint["sha256"],
        "adapter_directory_sha256": adapter_fingerprint["sha256"],
    }
    composition_sha = hashlib.sha256(
        json.dumps(
            composition_payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    effective = {
        "path": f"{base}::peft::{checkpoint}",
        "kind": "peft_composition",
        "sha256": composition_sha,
        "file_count": base_fingerprint["file_count"]
        + adapter_fingerprint["file_count"],
        "total_bytes": base_fingerprint["total_bytes"]
        + adapter_fingerprint["total_bytes"],
        "algorithm": "test-composition",
    }
    provenance = {
        "sample_manifest_sha256": manifest_sha,
        "release_manifest_sha256": selected.RELEASE_MANIFEST_SHA256,
        "training_config_sha256": sha256_file(grpo_config),
        "tokenizer_bundle_sha256": tokenizer_sha,
        "model": base_fingerprint,
        "adapter": adapter_fingerprint,
        "effective_model": effective,
        "model_source": {
            "mode": "base_model_plus_adapter",
            "base_model": {"directory": base_fingerprint, "files": {}},
            "adapter": {"directory": adapter_fingerprint, "files": {}},
            "composition": {
                "algorithm": "test-composition",
                "payload": composition_payload,
                "sha256": composition_sha,
            },
        },
    }
    rows = []
    for sample in samples:
        cases = [("greedy", 0, sample["greedy_seed"])] + [
            ("sampled", index, seed)
            for index, seed in enumerate(sample["sample_seeds"], start=1)
        ]
        for mode, index, seed in cases:
            decision_json = json.dumps(
                {
                    "direction": sample["direction"],
                    "magnitude_bp": sample["magnitude_bp"],
                },
                separators=(",", ":"),
            )
            answer = (
                f"```json\n{decision_json}\n```"
                if mode == "sampled" and index % 2 == 0
                else decision_json
            )
            completion = (
                "Evidence supports the selected policy stance.\n</think>\n" + answer
            )
            replay = replay_decision_dense_v3(
                completion,
                {
                    "direction": sample["direction"],
                    "magnitude_bp": sample["magnitude_bp"],
                },
                hit_eos=True,
                cap_reached=False,
            )
            rows.append(
                {
                    "schema_version": "chk4-decision-grpo-stratified-probe-v1",
                    "sample_id": sample["sample_id"],
                    "split": "train",
                    "target_direction": sample["direction"],
                    "target_magnitude_bp": sample["magnitude_bp"],
                    "generation_mode": mode,
                    "generation_index": index,
                    "seed": seed,
                    "completion": completion,
                    "completion_sha256": hashlib.sha256(
                        completion.encode()
                    ).hexdigest(),
                    "provenance": {
                        "sample_manifest_sha256": manifest_sha,
                        "release_manifest_sha256": selected.RELEASE_MANIFEST_SHA256,
                        "training_config_sha256": sha256_file(grpo_config),
                        "tokenizer_bundle_sha256": tokenizer_sha,
                        "model_sha256": base_fingerprint["sha256"],
                        "model_loading_mode": "base_model_plus_adapter",
                        "base_model_sha256": base_fingerprint["sha256"],
                        "adapter_sha256": adapter_fingerprint["sha256"],
                        "effective_model_sha256": composition_sha,
                    },
                    "decision_dense_v3_reward": replay["reward"],
                    "decision_dense_v3_nonzero": replay["nonzero"],
                    "response_format": replay["response_format"],
                    "strict_json": replay["strict_json"],
                    "fenced_json": replay["fenced_json"],
                    "decision_prediction": replay["prediction"],
                    "decision_direction_correct": replay["direction_correct"],
                    "decision_exact": replay["exact"],
                    "decision_rejection_reason": replay["rejection_reason"],
                    "decision_forced_zero_reason": replay["forced_zero_reason"],
                    "cap_reached": False,
                    "think_boundary_count": 1,
                    "hit_eos": True,
                    "strict_periodic_tail": False,
                }
            )
    results_path = root / "pilot/results.jsonl"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )
    summary = dict(summarize_stratified_results(rows, provenance=provenance))
    if not passed:
        summary["quality_status"] = "failed"
        summary["quality_gate"]["reasons"] = ["forced-test-failure"]
    summary["results"] = {
        "path": str(results_path.resolve()),
        "rows": 15,
        "sha256": sha256_file(results_path),
    }
    summary_path = root / "pilot/summary.json"
    _write_json(summary_path, summary)
    return manifest_path, manifest_sha, summary_path, sha256_file(summary_path)


def _fixture(tmp_path: Path, *, step: int = 10, passed: bool = True) -> dict[str, Any]:
    parent_path = tmp_path / selected.PARENT_CONFIG_RELATIVE
    sft_path = tmp_path / selected.SFT_CONFIG_RELATIVE
    grpo_path = tmp_path / selected.GRPO_CONFIG_RELATIVE
    _copy_config(selected.PARENT_CONFIG_RELATIVE, parent_path)
    sft = _copy_config(selected.SFT_CONFIG_RELATIVE, sft_path)
    _copy_config(selected.GRPO_CONFIG_RELATIVE, grpo_path)
    base = (tmp_path / sft["model_name_or_path"]).resolve()
    base.mkdir(parents=True)
    _write_json(base / "config.json", {"model_type": "llama"})
    (base / "model.safetensors").write_bytes(b"base-model")

    run_root = tmp_path / selected.RUN_ROOT_RELATIVE
    _create_parent_lineage(
        tmp_path,
        run_root=run_root,
        base=base,
        parent_config=parent_path,
        sft_config=sft_path,
        grpo_config=grpo_path,
    )
    output = run_root / "adapters/chk4_sft"
    output.mkdir(parents=True)
    _write_json(
        output / "adapter_config.json", _adapter_config(sft["model_name_or_path"], sft)
    )
    (output / "adapter_model.safetensors").write_bytes(b"root-adapter-do-not-use")
    _write_json(
        output / "resolved_runtime_config.json",
        {
            "chk4_standalone_branch": {
                "branch_id": selected.BRANCH_ID,
                "profile": selected.PROFILE_NAME,
                "stage": "decision_sft",
                "config": {"path": str(sft_path), "sha256": sha256_file(sft_path)},
            },
            "model": {"model_name_or_path": sft["model_name_or_path"]},
            "dataset": {
                "chk4_decision": {
                    "scope": {"role": "decision_sft"},
                    "release_manifest": {"sha256": selected.RELEASE_MANIFEST_SHA256},
                }
            },
            "training": {
                "output_dir": sft["output_dir"],
                "learning_rate": 1.0e-5,
                "max_steps": 24,
                "warmup_steps": 2,
                "gradient_accumulation_steps": 8,
                "per_device_train_batch_size": 1,
            },
        },
    )
    _write_json(
        output / "trainer_state.json",
        {"global_step": 24, "max_steps": 24, "epoch": 3.0},
    )
    _write_json(output / "train_results.json", {"train_loss": 0.125})
    checkpoint = _make_checkpoint(
        output, step=step, sft=sft, base=sft["model_name_or_path"]
    )
    manifest, manifest_sha, summary, summary_sha = _pilot_artifacts(
        tmp_path,
        grpo_config=grpo_path,
        base=base,
        checkpoint=checkpoint,
        passed=passed,
    )
    return {
        "repo_root": tmp_path,
        "checkpoint_step": step,
        "pilot_manifest": manifest,
        "pilot_manifest_sha256": manifest_sha,
        "pilot_summary": summary,
        "pilot_summary_sha256": summary_sha,
        "checkpoint": checkpoint,
        "base": base,
        "output": output,
    }


def test_plan_binds_exact_checkpoint_and_never_root_adapter(tmp_path: Path) -> None:
    values = _fixture(tmp_path, step=10)
    plan = selected.build_selection_plan(
        **{
            key: values[key]
            for key in (
                "repo_root",
                "checkpoint_step",
                "pilot_manifest",
                "pilot_manifest_sha256",
                "pilot_summary",
                "pilot_summary_sha256",
            )
        }
    )
    assert plan.checkpoint == values["checkpoint"]
    assert plan.checkpoint != values["output"]
    assert plan.destination == (
        tmp_path
        / selected.RUN_ROOT_RELATIVE
        / "selected_sft_checkpoints/checkpoint-10/merged/chk4_sft"
    )
    assert plan.public()["root_adapter_is_source"] is False


def test_checkpoint_24_is_still_an_explicit_checkpoint_directory(
    tmp_path: Path,
) -> None:
    values = _fixture(tmp_path, step=24)
    kwargs = {
        key: values[key]
        for key in (
            "repo_root",
            "checkpoint_step",
            "pilot_manifest",
            "pilot_manifest_sha256",
            "pilot_summary",
            "pilot_summary_sha256",
        )
    }
    plan = selected.build_selection_plan(**kwargs)
    assert plan.checkpoint.name == "checkpoint-24"
    assert plan.checkpoint.parent == values["output"]
    assert plan.checkpoint != values["output"]


def test_incomplete_training_and_failed_or_wrong_checkpoint_pilot_fail_closed(
    tmp_path: Path,
) -> None:
    incomplete = _fixture(tmp_path / "incomplete", step=10)
    state = incomplete["output"] / "trainer_state.json"
    _write_json(state, {"global_step": 20, "max_steps": 24, "epoch": 2.5})
    with pytest.raises(selected.SelectionMergeError, match="completed step"):
        selected.build_selection_plan(
            **{
                key: incomplete[key]
                for key in (
                    "repo_root",
                    "checkpoint_step",
                    "pilot_manifest",
                    "pilot_manifest_sha256",
                    "pilot_summary",
                    "pilot_summary_sha256",
                )
            }
        )

    failed = _fixture(tmp_path / "failed", step=10, passed=False)
    with pytest.raises(selected.SelectionMergeError, match="did not pass"):
        selected.build_selection_plan(
            **{
                key: failed[key]
                for key in (
                    "repo_root",
                    "checkpoint_step",
                    "pilot_manifest",
                    "pilot_manifest_sha256",
                    "pilot_summary",
                    "pilot_summary_sha256",
                )
            }
        )

    wrong = _fixture(tmp_path / "wrong", step=10)
    _make_checkpoint(
        wrong["output"],
        step=20,
        sft=yaml.safe_load(
            (wrong["repo_root"] / selected.SFT_CONFIG_RELATIVE).read_text()
        ),
        base=yaml.safe_load(
            (wrong["repo_root"] / selected.SFT_CONFIG_RELATIVE).read_text()
        )["model_name_or_path"],
    )
    wrong["checkpoint_step"] = 20
    with pytest.raises(selected.SelectionMergeError, match="pilot checkpoint adapter"):
        selected.build_selection_plan(
            **{
                key: wrong[key]
                for key in (
                    "repo_root",
                    "checkpoint_step",
                    "pilot_manifest",
                    "pilot_manifest_sha256",
                    "pilot_summary",
                    "pilot_summary_sha256",
                )
            }
        )


def test_parent_lineage_drift_and_unsupported_step_fail_closed(tmp_path: Path) -> None:
    values = _fixture(tmp_path, step=10)
    kwargs = {
        key: values[key]
        for key in (
            "repo_root",
            "checkpoint_step",
            "pilot_manifest",
            "pilot_manifest_sha256",
            "pilot_summary",
            "pilot_summary_sha256",
        )
    }
    kwargs["checkpoint_step"] = 11
    with pytest.raises(selected.SelectionMergeError, match="10, 20, or 24"):
        selected.build_selection_plan(**kwargs)

    kwargs["checkpoint_step"] = 10
    parent_stage = json.loads(
        (
            tmp_path / selected.RUN_ROOT_RELATIVE / "receipts/parent_stage.json"
        ).read_text()
    )
    exact_path = Path(
        parent_stage["reused_parent"]["source_exact_merge_evidence"]["path"]
    )
    with exact_path.open("a", encoding="utf-8") as handle:
        handle.write(" ")
    with pytest.raises(selected.SelectionMergeError, match="hash drift"):
        selected.build_selection_plan(**kwargs)


def test_execute_is_cpu_create_only_and_preserves_source_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path, step=10)
    kwargs = {
        key: values[key]
        for key in (
            "repo_root",
            "checkpoint_step",
            "pilot_manifest",
            "pilot_manifest_sha256",
            "pilot_summary",
            "pilot_summary_sha256",
        )
    }
    source_before = fingerprint_artifact_path(values["checkpoint"])
    calls: dict[str, Any] = {}

    def fake_merge(
        config_path: Path, *, repo_root: Path, merge_attestation_binding: Any
    ) -> dict[str, Any]:
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
        assert Path(config["output_dir"]).resolve() == values["checkpoint"]
        destination = Path(config["peft_merged_model_path"])
        destination.mkdir(parents=True)
        _write_json(destination / "config.json", {"model_type": "llama"})
        (destination / "model.safetensors").write_bytes(b"merged")
        _write_json(
            destination / "merge_attestation.json",
            {"binding": merge_attestation_binding},
        )
        calls["binding"] = merge_attestation_binding
        return {"status": "merged"}

    def fake_attestation(destination: Path, *, expected_binding: Any) -> dict[str, Any]:
        payload = json.loads((destination / "merge_attestation.json").read_text())
        assert payload["binding"] == expected_binding
        return payload

    def fake_exact(**exact_kwargs: Any) -> dict[str, Any]:
        assert exact_kwargs["adapter"] == values["checkpoint"]
        payload = {
            "schema_version": "test-exact-v1",
            "subject_artifact_id": "chk4-decision-sft-selected-checkpoint-10",
            "conclusion": selected.EXACT_CONCLUSION,
            "sources": {
                "base_model": fingerprint_artifact_path(exact_kwargs["base_model"]),
                "adapter": fingerprint_artifact_path(exact_kwargs["adapter"]),
                "merged_model": fingerprint_artifact_path(exact_kwargs["merged_model"]),
            },
        }
        return seal_manifest(payload)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    receipt = selected.execute_selection(
        **kwargs,
        merge_runner=fake_merge,
        exact_verifier=fake_exact,
        attestation_verifier=fake_attestation,
    )
    destination = Path(receipt["destination"])
    assert receipt["status"] == "passed"
    assert receipt["root_adapter_is_source"] is False
    assert calls["binding"]["source_checkpoint"] == source_before
    assert fingerprint_artifact_path(values["checkpoint"]) == source_before
    assert destination.is_dir()
    assert destination.stat().st_mode & 0o777 == 0o555
    assert Path(receipt["receipt"]).stat().st_mode & 0o777 == 0o400
    with pytest.raises(selected.SelectionMergeError, match="already exists"):
        selected.execute_selection(
            **kwargs,
            merge_runner=fake_merge,
            exact_verifier=fake_exact,
            attestation_verifier=fake_attestation,
        )


def test_execute_rejects_visible_gpu_before_any_merge(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    values = _fixture(tmp_path, step=10)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    called = False

    def merge(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal called
        called = True
        return {}

    with pytest.raises(selected.SelectionMergeError, match="CPU-only"):
        selected.execute_selection(
            **{
                key: values[key]
                for key in (
                    "repo_root",
                    "checkpoint_step",
                    "pilot_manifest",
                    "pilot_manifest_sha256",
                    "pilot_summary",
                    "pilot_summary_sha256",
                )
            },
            merge_runner=merge,
            exact_verifier=merge,
            attestation_verifier=merge,
        )
    assert called is False
