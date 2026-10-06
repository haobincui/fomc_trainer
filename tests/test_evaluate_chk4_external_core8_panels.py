from __future__ import annotations

import copy
from collections import Counter

import pytest

from jobs.retrain_v2 import evaluate_chk4_external_core8_panels as protocol


class _FakeTokenizer:
    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        truncation=False,
        return_dict=False,
    ):
        assert tokenize is True
        assert add_generation_prompt is True
        assert truncation is False
        assert return_dict is False
        assert [message["role"] for message in messages] == ["system", "user"]
        return [1] * 100


def test_historical_n19_has_complete_core8_and_official_directive_labels() -> None:
    documents, sources = protocol._historical_documents()

    assert len(documents) == len(sources) == 19
    assert Counter(row["direction"] for row in documents) == {
        "cut": 4,
        "hold": 10,
        "hike": 5,
    }
    assert all(row["topic_order"] == list(protocol.CORE_TOPICS) for row in documents)
    assert all(len(row["topic_source_hashes"]) == 8 for row in documents)
    assert all(row["label_source"] == "official_minutes_directive" for row in documents)
    assert all(row["missing_record_fallback_used"] is False for row in documents)


def test_postcutoff_n12_has_complete_start_d1_core8() -> None:
    documents, sources = protocol._postcutoff_documents()

    assert len(documents) == len(sources) == 12
    assert Counter(row["direction"] for row in documents) == {"cut": 3, "hold": 9}
    assert all(row["topic_order"] == list(protocol.CORE_TOPICS) for row in documents)
    assert all(len(row["topic_source_hashes"]) == 8 for row in documents)
    assert all(row["missing_record_fallback_used"] is False for row in documents)
    assert all(
        row["evidence_cutoff"] < row["meeting_start_date"] <= row["meeting_end_date"]
        for row in documents
    )


def test_build_samples_uses_blind_common_input_contract(monkeypatch) -> None:
    monkeypatch.setattr(protocol, "_decision_exposure_dates", lambda: set())
    samples, sources = protocol._build_samples(_FakeTokenizer(), "frozen system")

    assert len(samples) == len(sources) == 31
    assert all(row["prompt_token_count"] == 100 for row in samples)
    assert all(
        row["input_contract"] == "deterministic_core8_source_analysis_concatenation_v1"
        for row in samples
    )
    for row in samples:
        assert row["prompt"].startswith(protocol.USER_PREFIX)
        for field in (
            "meeting_id",
            "meeting_start_date",
            "meeting_end_date",
            "decision_date",
        ):
            identity = str(row.get(field) or "")
            assert not identity or identity not in row["prompt"]


def test_postcutoff_balanced_accuracy_does_not_insert_absent_hike_zero() -> None:
    rows = [
        {
            "target_direction": "cut",
            "decision_prediction": {"direction": "cut"},
        },
        {
            "target_direction": "hold",
            "decision_prediction": {"direction": "hold"},
        },
    ]

    block = protocol.score_block(rows, supported_only=True)

    assert block["supported_classes"] == ["cut", "hold"]
    assert block["supported_class_balanced_accuracy"] == 1.0
    assert block["fixed_three_class_balanced_accuracy"] is None
    assert block["mechanical_zero_insertion_balanced_accuracy"] == pytest.approx(2 / 3)
    assert block["mechanical_zero_insertion_is_estimand"] is False
    assert block["primary_balanced_accuracy"] == 1.0


def test_historical_balanced_accuracy_requires_all_three_classes() -> None:
    rows = [
        {
            "target_direction": direction,
            "decision_prediction": {"direction": direction},
        }
        for direction in protocol.DIRECTIONS
    ]
    assert protocol.score_block(rows, supported_only=False)[
        "fixed_three_class_balanced_accuracy"
    ] == pytest.approx(1.0)

    with pytest.raises(
        protocol.ExternalCore8EvaluationError,
        match="requires all classes",
    ):
        protocol.score_block(rows[:2], supported_only=False)


def test_exact_two_sided_mcnemar() -> None:
    rows_a = [
        {"sample_id": str(index), "decision_direction_correct": value}
        for index, value in enumerate((True, False, False, False))
    ]
    rows_b = [
        {"sample_id": str(index), "decision_direction_correct": value}
        for index, value in enumerate((False, True, True, True))
    ]

    result = protocol.exact_mcnemar(rows_a, rows_b)

    assert result["a_only_correct"] == 1
    assert result["b_only_correct"] == 3
    assert result["discordant_pairs"] == 4
    assert result["p_value_raw"] == pytest.approx(0.625)


def test_holm_adjustment_preserves_order_and_handles_ties() -> None:
    comparisons = [
        {"name": "large", "p_value_raw": 0.04},
        {"name": "tie_a", "p_value_raw": 0.01},
        {"name": "tie_b", "p_value_raw": 0.01},
    ]

    adjusted = protocol.holm_adjust_pairwise(
        comparisons,
        panel="historical_n19",
    )

    assert [row["name"] for row in adjusted] == ["large", "tie_a", "tie_b"]
    assert [row["p_value_holm"] for row in adjusted] == pytest.approx(
        [0.04, 0.03, 0.03]
    )
    assert all(
        row["holm_family"] == "within_panel_three_pairwise_model_comparisons"
        and row["holm_family_panel"] == "historical_n19"
        and row["holm_family_size"] == 3
        for row in adjusted
    )


@pytest.mark.parametrize("field", protocol.MATERIAL_REPLAY_FIELDS)
def test_every_material_derived_field_fails_closed_on_corruption(field: str) -> None:
    expected = {
        name: {"field": name, "value": 1} for name in protocol.MATERIAL_REPLAY_FIELDS
    }
    expected.update({"schema_version": "expected", "sample_id": "sample"})
    stored = copy.deepcopy(expected)
    stored[field] = {"field": field, "value": 2}

    with pytest.raises(
        protocol.ExternalCore8EvaluationError,
        match=field,
    ):
        protocol._assert_exact_replay(stored, expected)


class _LengthTenTokenizer:
    eos_token_id = 2

    def __len__(self) -> int:
        return 10


class _ReplayMustNotRun:
    @staticmethod
    def _result_row(**kwargs):
        raise AssertionError("invalid raw IDs must fail before replay")


@pytest.mark.parametrize(
    "raw_ids",
    ([True], [10], ["1"], [1] * (protocol.MAX_NEW_TOKENS + 1)),
)
def test_raw_completion_ids_fail_closed_before_replay(raw_ids) -> None:
    with pytest.raises(protocol.ExternalCore8EvaluationError, match="raw completion"):
        protocol._replay_result_row(
            stored={"generated_token_ids_raw_padded": raw_ids},
            sample={},
            model={"sha256": "model"},
            contract={"batch_index": 0, "seed": 1},
            manifest_sha="manifest",
            tokenizer=_LengthTenTokenizer(),
            eos_ids={2},
            final_protocol=_ReplayMustNotRun(),
        )


def test_postcutoff_statement_contract_checks_date_action_magnitude_and_range() -> None:
    raw = b"""
    <html><body><p>September 17, 2025</p>
    <p>The Committee decided to lower the target range for the federal funds
    rate by 1/4 percentage point to 4 to 4-1/4 percent.</p></body></html>
    """
    source = {
        "meeting_id": "fomc-20250916-20250917",
        "url": "https://www.federalreserve.gov/newsevents/pressreleases/monetary20250917a.htm",
        "meeting_end_date": "2025-09-17",
        "direction": "cut",
        "magnitude_bp": 25,
        "resulting_target_range": "4.00--4.25",
    }
    result = protocol._validate_postcutoff_statement(
        raw,
        source,
        source["url"],
    )
    assert set(result.values()) >= {"passed"}


def test_historical_config_has_no_missing_record_fallback() -> None:
    config = protocol._read_json(
        protocol.HISTORICAL_CONFIG,
        label="historical N19 config",
    )
    assert config["label_policy"].endswith("no missing-record-to-zero fallback")
    assert all(
        row["missing_record_fallback_used"] is False for row in config["meetings"]
    )


def test_manifest_contract_binds_evaluator_parser_reward_and_source_builders() -> None:
    assert set(protocol.IMPLEMENTATION_FILES) == {
        "evaluator",
        "final_result_parser_and_scorer",
        "generation_contract",
        "reward_replay",
        "eos_normalization",
        "decision_reward",
        "core8_source_analysis_builder",
        "source_compression",
    }
    assert all(path.is_file() for path in protocol.IMPLEMENTATION_FILES.values())
    assert (
        protocol.IMPLEMENTATION_FILES["evaluator"].resolve()
        == protocol.Path(protocol.__file__).resolve()
    )


def test_tokenizer_bundle_is_byte_identical_across_models() -> None:
    binding = protocol._shared_tokenizer_bundle_binding()
    assert set(binding) == set(protocol.TOKENIZER_BUNDLE_FILES)
    assert all(record["bytes"] > 0 for record in binding.values())
    assert all(len(record["sha256"]) == 64 for record in binding.values())


def test_historical_source_handoff_and_ledger_manifests_are_directly_bound() -> None:
    binding = protocol._historical_source_lineage_bindings()
    assert binding["source_handoff"]["sha256"] == (
        "333196b8e4577b77e6b74f6745a75418b145bce9de2579c9f2dd5e14cd0946a7"
    )
    assert len(binding["ledger_manifests"]) == 10
    assert all(
        record["path"].endswith("ledger_manifest.json")
        for record in binding["ledger_manifests"].values()
    )


def test_post_run_receipt_binds_batch_results_contract_and_runtime(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(protocol, "EXPECTED_RESULT_ROWS", 1)
    root = tmp_path / "evaluation"
    run_root = root / "run"
    contract = {
        "model_label": protocol.MODELS[0]["label"],
        "panel": "historical_n19",
        "batch_index": 0,
        "sample_ids": ["sample"],
        "seed": 1,
    }
    batch_path = protocol._batch_path(run_root, contract)
    batch_path.parent.mkdir(parents=True)
    batch_path.write_text('{"sample_id":"sample"}\n', encoding="utf-8")
    (run_root / "batch_contract.json").write_text("{}\n", encoding="utf-8")
    (run_root / "runtime.json").write_text("{}\n", encoding="utf-8")
    (run_root / "results.jsonl").write_text(
        '{"sample_id":"sample"}\n', encoding="utf-8"
    )
    receipt = protocol._sealed_payload(
        protocol._run_receipt_payload(
            root=root,
            manifest_sha="manifest",
            contracts=[contract],
        )
    )
    protocol._write_exclusive_json(run_root / "run_receipt.json", receipt)

    validated = protocol._validate_run_receipt(
        root=root,
        manifest_sha="manifest",
        contracts=[contract],
    )
    assert validated["results"]["rows"] == 1

    batch_path.write_text('{"sample_id":"corrupted"}\n', encoding="utf-8")
    with pytest.raises(
        protocol.ExternalCore8EvaluationError,
        match="receipt batch drift",
    ):
        protocol._validate_run_receipt(
            root=root,
            manifest_sha="manifest",
            contracts=[contract],
        )
