from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobs.retrain_v2 import probe_chk4_decision_sft_generation as probe


class FakeChatTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **_kwargs,
    ):
        assert tokenize is True
        assert add_generation_prompt is True
        words = " ".join(message["content"] for message in messages).split()
        return list(range(1, len(words) + 2))


class FakeDecoder:
    def decode(
        self,
        token_ids,
        *,
        skip_special_tokens: bool,
        clean_up_tokenization_spaces: bool,
    ) -> str:
        assert skip_special_tokens is False
        assert clean_up_tokenization_spaces is False
        values = {
            1: "careful reasoning ",
            2: "</think>",
            3: '{"direction":"hold","magnitude_bp":0}',
            99: "<eos>",
        }
        return "".join(values[value] for value in token_ids)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _release_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict, object]:
    release = tmp_path / "release"
    tokenizer = tmp_path / "tokenizer"
    tokenizer.mkdir()
    tokenizer_file = tokenizer / "tokenizer.json"
    tokenizer_file.write_text("{}\n", encoding="utf-8")

    prompt_lengths = [1, 5, 3, 7, 9]
    directions = ["hold", "hike", "cut", "hold", "hike"]
    magnitudes = [0, 25, 50, 0, 75]
    rows: list[dict] = []
    unique: list[dict] = []
    for index, (words, direction, magnitude) in enumerate(
        zip(prompt_lengths, directions, magnitudes, strict=True), start=1
    ):
        prompt = " ".join(f"prompt-{index}-{offset}" for offset in range(words))
        response = (
            f"reasoning-{index}\n</think>\n"
            + json.dumps(
                {"direction": direction, "magnitude_bp": magnitude},
                separators=(",", ":"),
            )
        )
        rows.append({"prompt": prompt, "response": response})
        unique.append(
            {
                "sample_id": f"sample-{index}",
                "split": "validation",
                "meeting_date": f"2020-01-{index:02d}",
                "direction": direction,
                "magnitude_bp": magnitude,
                "prompt_sha256": probe.sha256_text(prompt),
                "response_sha256": probe.sha256_text(response),
            }
        )

    validation_path = release / probe.VALIDATION_RELATIVE
    unique_path = release / probe.UNIQUE_VALIDATION_RELATIVE
    _write_jsonl(validation_path, rows)
    _write_jsonl(unique_path, unique)
    manifest = {
        "schema_version": probe.RELEASE_SCHEMA_VERSION,
        "release_id": "fixture-chk4-v1",
        "immutable": True,
        "quality_status": "passed",
        "training_ready": True,
        "files": {
            probe.VALIDATION_RELATIVE: {
                "path": probe.VALIDATION_RELATIVE,
                "rows": len(rows),
                "bytes": validation_path.stat().st_size,
                "sha256": probe.sha256_file(validation_path),
            },
            probe.UNIQUE_VALIDATION_RELATIVE: {
                "path": probe.UNIQUE_VALIDATION_RELATIVE,
                "rows": len(unique),
                "bytes": unique_path.stat().st_size,
                "sha256": probe.sha256_file(unique_path),
            },
        },
        "sources": {
            "tokenizer": {
                "path": str(tokenizer.resolve()),
                "files": {
                    "tokenizer.json": {
                        "bytes": tokenizer_file.stat().st_size,
                        "sha256": probe.sha256_file(tokenizer_file),
                    }
                },
            }
        },
    }
    manifest_path = release / "release_manifest.json"
    _write_json(manifest_path, manifest)
    manifest_sha = probe.sha256_file(manifest_path)
    config = tmp_path / "chk4_sft.yaml"
    config.write_text(
        "\n".join(
            (
                f"dataset_name: {release.resolve() / 'decision_sft'}",
                "dataset_chk4_role: decision_sft",
                f"dataset_chk4_release_manifest: {manifest_path.resolve()}",
                f"dataset_chk4_release_manifest_sha256: {manifest_sha}",
                "dataset_prompt_column: prompt",
                "dataset_test_split: validation",
                "system_prompt: fixed decision system prompt",
                "",
            )
        ),
        encoding="utf-8",
    )

    def verifier(root: Path, *, expected_manifest_sha256: str):
        assert root == release.resolve()
        assert expected_manifest_sha256 == manifest_sha
        return manifest

    return release, config, manifest, verifier


def _build_fixture_manifest(tmp_path: Path) -> tuple[dict, object]:
    release, config, _, verifier = _release_fixture(tmp_path)
    manifest_path = release / "release_manifest.json"
    artifact = probe.build_sample_manifest(
        release_root=release,
        release_manifest_sha256=probe.sha256_file(manifest_path),
        tokenizer=FakeChatTokenizer(),
        training_config=config,
        seed=100,
        release_verifier=verifier,
    )
    return dict(artifact), verifier


def test_sample_manifest_binds_all_validation_and_marks_fixed_quick_subset(
    tmp_path: Path,
) -> None:
    artifact, _ = _build_fixture_manifest(tmp_path)

    assert artifact["schema_version"] == probe.SAMPLE_SCHEMA_VERSION
    assert len(artifact["samples"]) == 5
    assert [row["sample_id"] for row in artifact["samples"]] == [
        "sample-1",
        "sample-2",
        "sample-3",
        "sample-4",
        "sample-5",
    ]
    quick = {
        row["quick_bucket"]: row["sample_id"]
        for row in artifact["samples"]
        if row["quick_bucket"] is not None
    }
    assert quick == {
        "short": "sample-1",
        "medium": "sample-2",
        "long": "sample-5",
    }
    assert artifact["samples"][0]["greedy_seed"] == 100
    assert artifact["samples"][0]["sample_seeds"] == [101, 102, 103, 104]
    assert artifact["samples"][1]["greedy_seed"] == 105
    serialized = json.dumps(artifact, ensure_ascii=False)
    assert "prompt-1-0" not in serialized
    assert "fixed decision system prompt" not in serialized
    assert "reasoning-1" not in serialized


def test_generation_cases_are_13_style_formal_or_three_row_quick(
    tmp_path: Path,
) -> None:
    artifact, _ = _build_fixture_manifest(tmp_path)

    formal = probe._generation_cases(artifact, probe_mode="formal")
    quick = probe._generation_cases(artifact, probe_mode="quick")

    assert len(formal) == 5 * 5
    assert len(quick) == 3 * 5
    assert sum(case["generation_mode"] == "greedy" for case in formal) == 5
    assert sum(case["generation_mode"] == "sampled" for case in formal) == 20
    assert {case["sample"]["quick_bucket"] for case in quick} == set(probe.BUCKETS)


def test_bound_row_reload_rejects_dataset_hash_drift(tmp_path: Path) -> None:
    artifact, verifier = _build_fixture_manifest(tmp_path)
    validation = Path(artifact["dataset"]["path"])
    validation.write_text(validation.read_text(encoding="utf-8") + "\n", encoding="utf-8")

    with pytest.raises(probe.ProbeError, match="SHA-256 mismatch"):
        probe._load_bound_rows(artifact, release_verifier=verifier)


def test_decode_removes_terminal_eos_and_preserves_native_boundary() -> None:
    completion = probe.decode_completion_preserving_boundary(
        FakeDecoder(), [1, 2, 3, 99], {99}
    )

    assert completion == 'careful reasoning </think>{"direction":"hold","magnitude_bp":0}'
    assert "<eos>" not in completion


def test_analyze_completion_accepts_exact_plain_decision_json_and_eos() -> None:
    metrics = probe.analyze_completion(
        text='careful reasoning\n</think>\n{"direction":"hold","magnitude_bp":0}',
        generated_token_ids=[*range(1, 40), 99],
        eos_token_ids=99,
        max_new_tokens=100,
    )

    assert metrics["status"] == "valid"
    assert metrics["think_boundary_count"] == 1
    assert metrics["plain_json"] is True
    assert metrics["exact_keys"] is True
    assert metrics["decision_domain_valid"] is True
    assert metrics["hit_eos"] is True
    assert metrics["cap_reached"] is False
    assert metrics["delivery_valid"] is True


@pytest.mark.parametrize(
    ("text", "expected_failure"),
    [
        (
            'reasoning</think>{"direction":"hold","magnitude_bp":0}</think>',
            "think_boundary_count_not_one",
        ),
        (
            'reasoning</think><answer>{"direction":"hold","magnitude_bp":0}</answer>',
            "answer_not_plain_single_json",
        ),
        (
            'reasoning</think>{"direction":"hold","magnitude_bp":0,"extra":1}',
            "decision_keys_not_exact",
        ),
        (
            'reasoning</think>{"direction":"hold","magnitude_bp":0} trailing',
            "answer_not_plain_single_json",
        ),
        (
            'reasoning</think>{"direction":"hold","magnitude_bp":25}',
            "decision_domain_invalid",
        ),
    ],
)
def test_analyze_completion_rejects_contract_drift(
    text: str, expected_failure: str
) -> None:
    metrics = probe.analyze_completion(
        text=text,
        generated_token_ids=[1, 2, 3, 99],
        eos_token_ids=99,
        max_new_tokens=100,
    )

    assert metrics["contract_valid"] is False
    assert expected_failure in metrics["failure_reasons"]


def test_analyze_completion_records_missing_eos_cap_and_repetition() -> None:
    repeated = list(range(16)) * 4
    metrics = probe.analyze_completion(
        text='reasoning</think>{"direction":"hold","magnitude_bp":0}',
        generated_token_ids=repeated,
        eos_token_ids=999,
        max_new_tokens=len(repeated),
    )

    assert metrics["hit_eos"] is False
    assert metrics["cap_reached"] is True
    assert metrics["strict_periodic_tail"] is True
    assert "missing_terminal_eos" in metrics["failure_reasons"]
    assert "completion_length_cap" in metrics["failure_reasons"]


def test_decision_reward_replay_matches_dense_v2_formula() -> None:
    reward = probe.replay_decision_dense_v2(
        'reasoning</think>{"direction":"hike","magnitude_bp":50}',
        {"direction": "hike", "magnitude_bp": 75},
    )

    assert reward == {
        "reward_name": "decision_dense_v2",
        "reward": 0.725,
        "direction_correct": True,
        "magnitude_score": 0.75,
        "exact": False,
    }
    assert probe.replay_decision_dense_v2(
        "not a structured completion",
        {"direction": "hold", "magnitude_bp": 0},
    )["reward"] == 0.0


def _summary_row(
    sample_id: str,
    generation_mode: str,
    generation_index: int,
    reward: float,
) -> dict:
    return {
        "sample_id": sample_id,
        "quick_bucket": None,
        "generation_mode": generation_mode,
        "generation_index": generation_index,
        "seed": generation_index,
        "prompt_token_count": 100,
        "completion_token_count": 20,
        "think_boundary_count": 1,
        "plain_json": True,
        "exact_keys": True,
        "decision_domain_valid": True,
        "contract_valid": True,
        "hit_eos": True,
        "cap_reached": False,
        "strict_periodic_tail": False,
        "delivery_valid": True,
        "full_token_4gram_repetition": 0.0,
        "tail_token_4gram_repetition": 0.0,
        "decision_dense_v2_reward": reward,
        "failure_reasons": [],
    }


def test_summary_reports_sampled_zero_reward_and_zero_std_group_rates() -> None:
    results: list[dict] = []
    for sample_id, rewards in (
        ("sample-a", [0.0, 0.0, 0.0, 0.0]),
        ("sample-b", [0.05, 0.5, 1.0, 0.5]),
    ):
        results.append(_summary_row(sample_id, "greedy", 0, 1.0))
        results.extend(
            _summary_row(sample_id, "sampled", index, reward)
            for index, reward in enumerate(rewards, start=1)
        )

    summary = probe.summarize_results(
        results, provenance={"release_manifest_sha256": "a" * 64}, probe_mode="formal"
    )
    groups = summary["decision_dense_v2_sampled_groups"]

    assert summary["cases"] == 10
    assert summary["validation_groups"] == 2
    assert groups["zero_reward_groups"] == 1
    assert groups["zero_reward_group_rate"] == 0.5
    assert groups["zero_std_groups"] == 1
    assert groups["zero_std_group_rate"] == 0.5


def test_single_gpu_visibility_is_explicit() -> None:
    assert probe._require_single_visible_gpu({"CUDA_VISIBLE_DEVICES": "1"}) == "1"
    with pytest.raises(probe.ProbeError, match="exactly one GPU"):
        probe._require_single_visible_gpu({})
    with pytest.raises(probe.ProbeError, match="exactly one GPU"):
        probe._require_single_visible_gpu({"CUDA_VISIBLE_DEVICES": "0,1"})
