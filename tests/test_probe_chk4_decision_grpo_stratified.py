from __future__ import annotations

import hashlib
import json
import stat
from pathlib import Path
from typing import Any

import pytest
import yaml

from jobs.retrain_v2 import probe_chk4_decision_grpo_stratified as pilot


class FakeTokenizer:
    def apply_chat_template(
        self, messages: list[dict[str, str]], **_: Any
    ) -> list[int]:
        # The user prompt is deliberately the complete variable portion so the
        # fixture can assert lower-median token-length selection without loading
        # a real tokenizer.
        return list(range(1, len(messages[-1]["content"]) + 2))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _release_fixture(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path / "release"
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir(parents=True)
    tokenizer_file = tokenizer_dir / "tokenizer.json"
    tokenizer_file.write_text('{"fixture":true}\n', encoding="utf-8")

    specs = [
        ("hold", 0, [("hold-a", "hh"), ("hold-b", "hhhh"), ("hold-c", "hhhhhh")], 1),
        (
            "hike",
            25,
            [("hike-a", "hhh"), ("hike-b", "hhhhh"), ("hike-c", "hhhhhhh")],
            2,
        ),
        ("cut", 50, [("cut-a", "h"), ("cut-b", "hhhhhhhh"), ("cut-c", "hhhhhhhhh")], 3),
    ]
    unique: list[dict[str, Any]] = []
    physical: list[dict[str, Any]] = []
    for direction, magnitude, samples, repeats in specs:
        for sample_id, prompt in samples:
            prompt_sha = hashlib.sha256(prompt.encode()).hexdigest()
            unique.append(
                {
                    "sample_id": sample_id,
                    "split": "train",
                    "direction": direction,
                    "magnitude_bp": magnitude,
                    "prompt_sha256": prompt_sha,
                    "repeat_factor": repeats,
                }
            )
            physical.extend(
                {
                    "sample_id": sample_id,
                    "prompt": prompt,
                    "direction": direction,
                    "magnitude_bp": magnitude,
                }
                for _ in range(repeats)
            )
    train_path = root / pilot.TRAIN_RELATIVE
    unique_path = root / pilot.UNIQUE_TRAIN_RELATIVE
    _write_jsonl(train_path, physical)
    _write_jsonl(unique_path, unique)
    manifest = {
        "schema_version": pilot.base_probe.RELEASE_SCHEMA_VERSION,
        "release_id": "fixture-release",
        "immutable": True,
        "quality_status": "passed",
        "training_ready": True,
        "files": {
            pilot.TRAIN_RELATIVE: {
                "path": pilot.TRAIN_RELATIVE,
                "rows": len(physical),
                "bytes": train_path.stat().st_size,
                "sha256": _sha(train_path),
            },
            pilot.UNIQUE_TRAIN_RELATIVE: {
                "path": pilot.UNIQUE_TRAIN_RELATIVE,
                "rows": len(unique),
                "bytes": unique_path.stat().st_size,
                "sha256": _sha(unique_path),
            },
        },
        "sources": {
            "tokenizer": {
                "path": str(tokenizer_dir.resolve()),
                "files": {
                    "tokenizer.json": {
                        "bytes": tokenizer_file.stat().st_size,
                        "sha256": _sha(tokenizer_file),
                    }
                },
            }
        },
    }
    manifest_path = root / "release_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    manifest_sha = _sha(manifest_path)
    config = {
        "dataset_name": str((root / "decision_grpo").resolve()),
        "dataset_chk4_role": "decision_grpo",
        "dataset_chk4_release_manifest": str(manifest_path.resolve()),
        "dataset_chk4_release_manifest_sha256": manifest_sha,
        "dataset_prompt_column": "prompt",
        "dataset_train_split": "train",
        # It is permitted for training to name an eval split; the pilot never
        # selects it and the fixture intentionally does not provide it.
        "dataset_test_split": "validation",
        "system_prompt": "Decide from the supplied training evidence.",
        "max_prompt_length": 2560,
        "max_completion_length": 1024,
        "num_generations": 4,
        "temperature": 0.7,
        "top_p": 0.9,
        "use_vllm": False,
        "reward_funcs": ["decision_dense_v3"],
    }
    config_path = tmp_path / "grpo.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    def verifier(candidate_root: Path, *, expected_manifest_sha256: str):
        assert candidate_root == root.resolve()
        assert expected_manifest_sha256 == manifest_sha
        return json.loads(manifest_path.read_text(encoding="utf-8"))

    return {
        "root": root,
        "manifest": manifest,
        "manifest_path": manifest_path,
        "manifest_sha": manifest_sha,
        "config_path": config_path,
        "train_path": train_path,
        "verifier": verifier,
    }


def _pre2009_release_fixture(tmp_path: Path) -> dict[str, Any]:
    fixture = _release_fixture(tmp_path)
    old_root = fixture["root"]
    root = tmp_path / pilot.CHK4_PRE2009_AUGMENTED_RELEASE_ID
    old_root.rename(root)

    tokenizer = fixture["manifest"]["sources"]["tokenizer"]
    tokenizer_path = Path(tokenizer["path"])
    tokenizer_files = tokenizer["files"]
    manifest = dict(fixture["manifest"])
    manifest.update(
        {
            "schema_version": pilot.CHK4_PRE2009_AUGMENTED_RELEASE_SCHEMA,
            "release_id": pilot.CHK4_PRE2009_AUGMENTED_RELEASE_ID,
            "release_type": "train_only_augmentation",
            "parent_releases": {},
        }
    )
    manifest.pop("sources")
    manifest_path = root / "release_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest_sha = _sha(manifest_path)

    config = yaml.safe_load(fixture["config_path"].read_text(encoding="utf-8"))
    config.update(
        {
            "dataset_name": str((root / "decision_grpo").resolve()),
            "dataset_chk4_role": pilot.CHK4_PRE2009_GRPO_ROLE,
            "dataset_chk4_release_manifest": str(manifest_path.resolve()),
            "dataset_chk4_release_manifest_sha256": manifest_sha,
        }
    )
    fixture["config_path"].write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    calls: list[dict[str, Any]] = []

    def verifier(**kwargs: Any) -> dict[str, Any]:
        calls.append(kwargs)
        files = dict(tokenizer_files)
        return {
            "schema_version": "chk4-pre2009-augmented-runtime-binding-v1",
            "release_id": pilot.CHK4_PRE2009_AUGMENTED_RELEASE_ID,
            "dataset_role": pilot.CHK4_PRE2009_GRPO_ROLE,
            "physical_dataset_role": pilot.CORE_GRPO_ROLE,
            "release_manifest_path": str(manifest_path.resolve()),
            "release_manifest_sha256": manifest_sha,
            "split_files": {
                "train": (root / "decision_grpo/train.jsonl").resolve(),
                "validation": (root / "decision_grpo/validation.jsonl").resolve(),
            },
            "test_verified_but_not_loaded": True,
            "tokenizer_binding": {
                "model_path": str(tokenizer_path.resolve()),
                "files": files,
                "bundle_sha256": pilot.base_probe.sha256_text(
                    pilot.base_probe.canonical_json(files)
                ),
            },
        }

    fixture.update(
        {
            "root": root,
            "manifest": manifest,
            "manifest_path": manifest_path,
            "manifest_sha": manifest_sha,
            "train_path": root / pilot.TRAIN_RELATIVE,
            "tokenizer_path": tokenizer_path,
            "verifier": verifier,
            "verifier_calls": calls,
        }
    )
    return fixture


def _build(fixture: dict[str, Any]) -> dict[str, Any]:
    return dict(
        pilot.build_stratified_manifest(
            release_root=fixture["root"],
            release_manifest_sha256=fixture["manifest_sha"],
            tokenizer=FakeTokenizer(),
            training_config=fixture["config_path"],
            seed=100,
            release_verifier=fixture["verifier"],
            verification_model_path=fixture.get("tokenizer_path"),
        )
    )


def test_prepare_selects_train_only_lower_medians_and_binds_hashes(
    tmp_path: Path,
) -> None:
    fixture = _release_fixture(tmp_path)
    manifest = _build(fixture)

    assert manifest["schema_version"] == pilot.SAMPLE_SCHEMA_VERSION
    assert manifest["dataset"]["role"] == pilot.CORE_GRPO_ROLE
    assert manifest["dataset"]["split"] == "train"
    assert manifest["dataset"]["sha256"] == _sha(fixture["train_path"])
    assert manifest["release"]["manifest_sha256"] == fixture["manifest_sha"]
    assert manifest["prompt_contract"]["training_config_sha256"] == _sha(
        fixture["config_path"]
    )
    assert len(manifest["tokenizer"]["bundle_sha256"]) == 64
    assert [row["sample_id"] for row in manifest["samples"]] == [
        "hold-b",
        "hike-b",
        "cut-b",
    ]
    assert [row["split"] for row in manifest["samples"]] == ["train"] * 3
    assert [row["direction"] for row in manifest["samples"]] == list(pilot.DIRECTIONS)
    assert [row["magnitude_bp"] for row in manifest["samples"]] == [0, 25, 50]
    assert [len(row["physical_line_numbers"]) for row in manifest["samples"]] == [
        1,
        2,
        3,
    ]
    assert len(pilot._generation_cases(manifest)) == 15
    serialized = pilot.base_probe.canonical_json(manifest)
    assert '"prompt":' not in serialized
    assert '"response":' not in serialized
    assert '"split":"validation"' not in serialized
    assert '"split":"test"' not in serialized


def test_prepare_supports_pre2009_logical_role_and_runtime_verifier(
    tmp_path: Path,
) -> None:
    fixture = _pre2009_release_fixture(tmp_path)
    manifest = _build(fixture)

    assert manifest["release"]["schema_version"] == (
        pilot.CHK4_PRE2009_AUGMENTED_RELEASE_SCHEMA
    )
    assert manifest["dataset"]["role"] == pilot.CHK4_PRE2009_GRPO_ROLE
    assert manifest["dataset"]["path"] == str(fixture["train_path"].resolve())
    assert len(manifest["samples"]) == 3
    assert len(pilot._generation_cases(manifest)) == 15
    assert len(fixture["verifier_calls"]) == 1
    call = fixture["verifier_calls"][0]
    assert call["dataset_dir"] == (fixture["root"] / "decision_grpo").resolve()
    assert call["manifest_path"] == fixture["manifest_path"].resolve()
    assert call["expected_manifest_sha256"] == fixture["manifest_sha"]
    assert call["dataset_role"] == pilot.CHK4_PRE2009_GRPO_ROLE
    assert call["model_path"] == fixture["tokenizer_path"].resolve()


def test_pre2009_external_sha_and_config_role_fail_closed(tmp_path: Path) -> None:
    fixture = _pre2009_release_fixture(tmp_path)
    with pytest.raises(pilot.PilotError, match="SHA-256 mismatch"):
        pilot.build_stratified_manifest(
            release_root=fixture["root"],
            release_manifest_sha256="0" * 64,
            tokenizer=FakeTokenizer(),
            training_config=fixture["config_path"],
            release_verifier=fixture["verifier"],
        )
    assert fixture["verifier_calls"] == []

    config = yaml.safe_load(fixture["config_path"].read_text(encoding="utf-8"))
    config["dataset_chk4_role"] = pilot.CORE_GRPO_ROLE
    fixture["config_path"].write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )

    def core_verifier(_root: Path, *, expected_manifest_sha256: str) -> dict[str, Any]:
        assert expected_manifest_sha256 == fixture["manifest_sha"]
        return fixture["manifest"]

    with pytest.raises(pilot.PilotError, match="unsupported chk4 release schema"):
        pilot.build_stratified_manifest(
            release_root=fixture["root"],
            release_manifest_sha256=fixture["manifest_sha"],
            tokenizer=FakeTokenizer(),
            training_config=fixture["config_path"],
            release_verifier=core_verifier,
        )


def test_sample_manifest_rejects_pre2009_schema_role_drift(tmp_path: Path) -> None:
    fixture = _pre2009_release_fixture(tmp_path)
    manifest = _build(fixture)
    manifest["dataset"]["role"] = pilot.CORE_GRPO_ROLE
    path = tmp_path / "drifted-samples.json"
    path.write_text(json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8")
    with pytest.raises(pilot.PilotError, match="schema/logical role drift"):
        pilot._validate_sample_manifest(path, _sha(path))


def test_prepare_fails_closed_on_source_or_config_drift(tmp_path: Path) -> None:
    fixture = _release_fixture(tmp_path)
    _build(fixture)
    fixture["train_path"].write_text("{}\n", encoding="utf-8")
    with pytest.raises(pilot.PilotError, match="SHA-256 mismatch"):
        _build(fixture)

    fixture = _release_fixture(tmp_path / "second")
    config = yaml.safe_load(fixture["config_path"].read_text(encoding="utf-8"))
    config["temperature"] = 1.0
    fixture["config_path"].write_text(yaml.safe_dump(config), encoding="utf-8")
    with pytest.raises(pilot.PilotError, match="temperature must be"):
        _build(fixture)


def test_manifest_and_output_writes_are_create_only_and_read_only(
    tmp_path: Path,
) -> None:
    path = tmp_path / "bound" / "manifest.json"
    pilot._write_exclusive_readonly_json(path, {"ok": True})
    assert stat.S_IMODE(path.stat().st_mode) == 0o444
    with pytest.raises(pilot.PilotError, match="overwrite"):
        pilot._write_exclusive_readonly_json(path, {"ok": False})

    manifest_sha = _sha(path)
    with pytest.raises(pilot.PilotError, match="unsupported"):
        pilot._validate_sample_manifest(path, manifest_sha)


@pytest.mark.parametrize(
    ("completion", "target", "hit_eos", "cap", "reward", "response_format"),
    [
        (
            'Reasoning.\n</think>\n{"direction":"hold","magnitude_bp":0}',
            {"direction": "hold", "magnitude_bp": 0},
            True,
            False,
            1.0,
            "strict_json",
        ),
        (
            'Reasoning.\n</think>\n```json\n{"direction":"hold","magnitude_bp":0}\n```',
            {"direction": "hold", "magnitude_bp": 0},
            True,
            False,
            0.2375,
            "fenced_json",
        ),
        (
            'Reasoning.\n</think>\n```json\n{"direction":"cut","magnitude_bp":25}\n```',
            {"direction": "hold", "magnitude_bp": 0},
            True,
            False,
            0.0,
            "fenced_json",
        ),
        (
            'Reasoning.\n</think>\n{"direction":"hold","magnitude_bp":"0"}',
            {"direction": "hold", "magnitude_bp": 0},
            True,
            False,
            0.0,
            "invalid",
        ),
        (
            'Reasoning without boundary {"direction":"hold","magnitude_bp":0}',
            {"direction": "hold", "magnitude_bp": 0},
            True,
            False,
            0.0,
            "invalid",
        ),
        (
            'Reasoning.\n</think>\n{"direction":"hold","magnitude_bp":0}',
            {"direction": "hold", "magnitude_bp": 0},
            False,
            True,
            0.0,
            "strict_json",
        ),
    ],
)
def test_v3_replay_is_semantic_and_termination_fail_closed(
    completion: str,
    target: dict[str, Any],
    hit_eos: bool,
    cap: bool,
    reward: float,
    response_format: str,
) -> None:
    result = pilot.replay_decision_dense_v3(
        completion, target, hit_eos=hit_eos, cap_reached=cap
    )
    assert result["reward"] == reward
    assert result["response_format"] == response_format
    if not hit_eos or cap:
        assert result["forced_zero_reason"] == "truncated_or_unterminated"


def _result_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for direction_index, direction in enumerate(pilot.DIRECTIONS):
        for index in range(5):
            sampled = index > 0
            reward = [1.0, 0.0, 0.2375, 0.0, 1.0][index]
            fenced = index == 2
            rows.append(
                {
                    "sample_id": f"{direction}-median",
                    "split": "train",
                    "target_direction": direction,
                    "target_magnitude_bp": 0 if direction == "hold" else 25,
                    "generation_mode": "sampled" if sampled else "greedy",
                    "generation_index": index if sampled else 0,
                    "decision_dense_v3_reward": reward,
                    "hit_eos": True,
                    "cap_reached": False,
                    "think_boundary_count": 1,
                    "fenced_json": fenced,
                    "strict_json": not fenced and reward > 0,
                    "decision_direction_correct": reward > 0,
                    "decision_exact": reward == 1.0,
                    "strict_periodic_tail": False,
                }
            )
    return rows


def test_summary_reports_each_direction_and_quality_gate() -> None:
    rows = _result_rows()
    summary = pilot.summarize_stratified_results(rows, provenance={"hash": "x"})
    assert summary["quality_status"] == "passed"
    assert summary["cases"] == 15
    for direction in pilot.DIRECTIONS:
        metrics = summary["directions"][direction]
        assert metrics["all"]["cases"] == 5
        assert metrics["sampled"]["cases"] == 4
        assert metrics["sampled"]["nonzero_count"] == 2
        assert metrics["sampled"]["direction_correct_count"] == 2
        assert metrics["sampled"]["reward_std"] > 0
        assert metrics["all"]["cap_count"] == 0
        assert metrics["all"]["boundary_count"] == 5
        assert metrics["all"]["fence_count"] == 1
        assert metrics["all"]["bare_count"] == 2

    with pytest.raises(pilot.PilotError, match="exactly 15"):
        pilot.summarize_stratified_results(rows[:-1], provenance={})


def test_summary_fails_gate_for_zero_reward_direction_and_cap() -> None:
    rows = _result_rows()
    for row in rows:
        if row["target_direction"] == "cut" and row["generation_mode"] == "sampled":
            row["decision_dense_v3_reward"] = 0.0
            row["decision_direction_correct"] = False
        if row["target_direction"] == "hold":
            row["hit_eos"] = False
            row["cap_reached"] = True
            row["decision_dense_v3_reward"] = 0.0
    summary = pilot.summarize_stratified_results(rows, provenance={})
    assert summary["quality_status"] == "failed"
    reasons = summary["quality_gate"]["reasons"]
    assert "cut:sampled_nonzero_count_lt_1" in reasons
    assert "cut:sampled_direction_correct_count_lt_1" in reasons
    assert "cut:sampled_reward_zero_std" in reasons
    assert "overall_cap_rate_gt_0.25" in reasons


def _model_adapter_fixture(tmp_path: Path) -> tuple[Path, Path]:
    base = tmp_path / "base"
    adapter = tmp_path / "checkpoint-10"
    base.mkdir(parents=True)
    adapter.mkdir(parents=True)
    (base / "config.json").write_text(
        json.dumps({"architectures": ["FixtureForCausalLM"]}) + "\n",
        encoding="utf-8",
    )
    (base / "generation_config.json").write_text(
        json.dumps({"eos_token_id": 2}) + "\n",
        encoding="utf-8",
    )
    (base / "model.safetensors").write_bytes(b"base-weights")
    adapter_config = {
        "base_model_name_or_path": str(base.resolve()),
        "bias": "none",
        "corda_config": None,
        "eva_config": None,
        "exclude_modules": None,
        "inference_mode": True,
        "lora_alpha": 64,
        "lora_bias": False,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": 32,
        "target_modules": ["q_proj", "v_proj"],
        "task_type": "CAUSAL_LM",
        "trainable_token_indices": None,
    }
    (adapter / "adapter_config.json").write_text(
        json.dumps(adapter_config, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (adapter / "adapter_model.safetensors").write_bytes(b"adapter-weights")
    (adapter / "optimizer.pt").write_bytes(b"optimizer-state")
    return base, adapter


def test_run_parser_exposes_two_mutually_exclusive_model_modes() -> None:
    parser = pilot._build_parser()
    tail = [
        "--sample-manifest",
        "samples.json",
        "--sample-manifest-sha256",
        "a" * 64,
        "--output-dir",
        "probe-output",
        "--model-label",
        "cp10",
    ]

    merged = parser.parse_args(["run", "--model", "merged", *tail])
    assert merged.model == Path("merged")
    assert merged.base_model is None
    assert merged.adapter is None

    peft = parser.parse_args(
        ["run", "--base-model", "base", "--adapter", "checkpoint-10", *tail]
    )
    assert peft.model is None
    assert peft.base_model == Path("base")
    assert peft.adapter == Path("checkpoint-10")

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--model",
                "merged",
                "--base-model",
                "base",
                "--adapter",
                "checkpoint-10",
                *tail,
            ]
        )


def test_model_mode_validation_requires_complete_exclusive_pair() -> None:
    with pytest.raises(pilot.PilotError, match="provided together"):
        pilot._resolve_model_loading_mode(
            model_path=None,
            base_model_path=Path("base"),
            adapter_path=None,
        )
    with pytest.raises(pilot.PilotError, match="mutually exclusive"):
        pilot._resolve_model_loading_mode(
            model_path=Path("merged"),
            base_model_path=None,
            adapter_path=Path("adapter"),
        )


def test_merged_model_mode_keeps_legacy_directory_fingerprint(
    tmp_path: Path,
) -> None:
    base, _ = _model_adapter_fixture(tmp_path)

    source = pilot._prepare_model_source(
        model_path=base,
        base_model_path=None,
        adapter_path=None,
    )

    expected = pilot.fingerprint_artifact_path(base)
    assert source["mode"] == pilot.MERGED_MODEL_MODE
    assert source["model_fingerprint"] == expected
    assert source["adapter_fingerprint"] is None
    assert source["effective_model_fingerprint"] == expected
    assert source["result_provenance"] == {}


def test_peft_source_records_directory_and_critical_file_hashes(
    tmp_path: Path,
) -> None:
    base, adapter = _model_adapter_fixture(tmp_path)

    source = pilot._prepare_model_source(
        model_path=None,
        base_model_path=base,
        adapter_path=adapter,
    )

    assert source["mode"] == pilot.PEFT_ADAPTER_MODE
    provenance = source["provenance"]
    base_binding = provenance["base_model"]
    adapter_binding = provenance["adapter"]
    assert base_binding["directory"] == pilot.fingerprint_artifact_path(base)
    assert adapter_binding["directory"] == pilot.fingerprint_artifact_path(adapter)
    assert base_binding["files"]["model.safetensors"]["sha256"] == _sha(
        base / "model.safetensors"
    )
    assert adapter_binding["files"]["adapter_config.json"]["sha256"] == _sha(
        adapter / "adapter_config.json"
    )
    assert adapter_binding["files"]["adapter_model.safetensors"]["sha256"] == _sha(
        adapter / "adapter_model.safetensors"
    )
    composition = provenance["composition"]
    assert source["model_fingerprint"] == base_binding["directory"]
    assert source["adapter_fingerprint"] == adapter_binding["directory"]
    assert composition["sha256"] == source["effective_model_fingerprint"]["sha256"]
    assert (
        source["result_provenance"]["base_model_sha256"]
        == base_binding["directory"]["sha256"]
    )
    assert (
        source["result_provenance"]["adapter_sha256"]
        == adapter_binding["directory"]["sha256"]
    )

    old_composition_sha = composition["sha256"]
    (adapter / "optimizer.pt").write_bytes(b"changed-optimizer-state")
    changed = pilot._prepare_model_source(
        model_path=None,
        base_model_path=base,
        adapter_path=adapter,
    )
    assert changed["effective_model_fingerprint"]["sha256"] != old_composition_sha


def test_peft_source_fails_closed_on_wrong_base_or_symlink(tmp_path: Path) -> None:
    base, adapter = _model_adapter_fixture(tmp_path)
    other_base = tmp_path / "other-base"
    other_base.mkdir()
    (other_base / "config.json").write_text("{}\n", encoding="utf-8")
    (other_base / "model.safetensors").write_bytes(b"other")

    with pytest.raises(pilot.PilotError, match="different base model"):
        pilot._prepare_model_source(
            model_path=None,
            base_model_path=other_base,
            adapter_path=adapter,
        )

    (adapter / "unsafe-link").symlink_to(base / "config.json")
    with pytest.raises(pilot.PilotError, match="contains a symlink"):
        pilot._prepare_model_source(
            model_path=None,
            base_model_path=base,
            adapter_path=adapter,
        )


def test_local_peft_loader_is_offline_pinned_and_sanitized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import peft
    import peft.tuners.lora as lora_module

    calls: dict[str, Any] = {}

    class FakeLoraConfig:
        def __init__(self, **kwargs: Any) -> None:
            calls["config_kwargs"] = kwargs

    class FakeLoraLayer:
        pass

    class FakeTensor:
        is_meta = False

    class FakeWrapped:
        def modules(self):
            return iter([self, FakeLoraLayer()])

        def named_parameters(self):
            return iter([("base.weight", FakeTensor())])

        def named_buffers(self):
            return iter([("base.cache", FakeTensor())])

    def from_pretrained(model: object, path: str, **kwargs: Any) -> FakeWrapped:
        calls["model"] = model
        calls["path"] = path
        calls["load_kwargs"] = kwargs
        return FakeWrapped()

    monkeypatch.setattr(peft, "LoraConfig", FakeLoraConfig)
    monkeypatch.setattr(lora_module, "LoraLayer", FakeLoraLayer)
    monkeypatch.setattr(
        peft.PeftModel,
        "from_pretrained",
        staticmethod(from_pretrained),
    )
    adapter_config = {
        "base_model_name_or_path": "/base",
        "corda_config": None,
        "peft_type": "LORA",
        "r": 32,
        "target_modules": ["q_proj"],
        "task_type": "CAUSAL_LM",
    }
    base_model = object()
    adapter_path = tmp_path / "checkpoint-10"

    wrapped = pilot._attach_local_peft_adapter(
        base_model,
        adapter_path=adapter_path,
        adapter_config=adapter_config,
    )

    assert isinstance(wrapped, FakeWrapped)
    assert "corda_config" not in calls["config_kwargs"]
    assert calls["load_kwargs"]["config"].__class__ is FakeLoraConfig
    assert calls["load_kwargs"]["device_map"] == {"": 0}
    assert calls["load_kwargs"]["is_trainable"] is False
    assert calls["load_kwargs"]["local_files_only"] is True
