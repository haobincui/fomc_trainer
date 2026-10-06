from __future__ import annotations

from pathlib import Path
from typing import Any
import importlib.util
import json

import pytest

from jobs.generation import generate_paper_chk2_downstream_v6 as v6
from jobs.generation import generate_paper_chk2_chk1_analysis_rewrite as v5


def _row(split: str = "train", index: int = 0) -> v5.PreparedRow:
    digest = v5.sha256_text(f"{split}:{index}")
    return v5.PreparedRow(
        sample_id=f"sample-{split}-{index}",
        split=split,
        source_split=split if split != "validation" else "eval",
        split_index=index,
        source_line_number=index + 1,
        generation_manifest_line_number=index + 1,
        meeting_date="2021-12-15",
        atomic_topic="inflation",
        section_style_id="economic_conditions",
        prompt="prompt",
        provided_data="Inflation rose.",
        source_analysis="Inflation rose.",
        prompt_sha256=digest,
        provided_data_sha256=digest,
        source_analysis_sha256=digest,
        candidate_response_sha256=digest,
        source_answer_sha256=digest,
        source_response_sha256=digest,
        generation_manifest_row_sha256=digest,
        source_row_sha256=digest,
    )


def _source_reject() -> dict[str, Any]:
    return {
        "complete": True,
        "machine_pass": False,
        "reasons": ["source_claim_not_supported"],
        "result": {"primary": {}},
        "provider": {},
        "contract_repair_used": False,
    }


def _source_pass() -> dict[str, Any]:
    return {
        "complete": True,
        "machine_pass": True,
        "reasons": [],
        "result": {"primary": {}},
        "provider": {},
        "contract_repair_used": False,
    }


def _v5_fixtures():
    path = Path(__file__).with_name("test_generate_paper_chk2_chk1_analysis_rewrite.py")
    spec = importlib.util.spec_from_file_location("v5_downstream_test_fixtures", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _build_pass_terminal(tmp_path: Path):
    fixtures = _v5_fixtures()
    row = fixtures._row()
    reference = fixtures._reference_bank()
    terminal = v6._process_downstream_terminal(
        row,
        output=tmp_path,
        tokenizer=fixtures.FakeTokenizer(),
        reference_bank=reference,
        official_reference_bank_sha256="a" * 64,
        backend=fixtures.StyleRepairBackend(),
        identity=v5.ProviderIdentityRegistry(),
        environment={v5.API_KEY_ENV: "not-persisted"},
        config=v5.ProviderConfig(),
        code_sha256="b" * 64,
        source_audit=_source_pass(),
    )
    assert terminal["terminal_status"] == v5.TERMINAL_PASS
    return fixtures, row, reference, terminal


class NoCallBackend:
    def generate(self, **kwargs):  # pragma: no cover - assertion is the test
        raise AssertionError(f"provider must not be called: {kwargs.get('role')}")


def _pass_terminal(sample_id: str) -> dict[str, Any]:
    replay = {
        "single_bos": True,
        "single_eos": True,
        "completion_only_prompt_masked": True,
        "completion_mask_covers_reasoning_boundary_answer_eos": True,
        "no_truncation": True,
        "prompt_tokens": 5,
        "completion_tokens": 7,
        "total_tokens": 12,
    }
    return {
        "sample_id": sample_id,
        "terminal_status": v5.TERMINAL_PASS,
        "training_pass": True,
        "generation": {
            "deterministic_validation": {"diagnostics": {"tokenizer_replay": replay}}
        },
    }


def test_cli_is_downstream_only_and_defaults_to_128() -> None:
    parser = v6.build_parser()
    args = parser.parse_args([])
    assert args.concurrency == 128
    assert args.phase == "all"
    with pytest.raises(SystemExit):
        parser.parse_args(["--phase", "source-audit"])
    assert v6.MAX_CONCURRENCY == 128


def test_run_binding_seals_implementation_and_handoff() -> None:
    first = v6._run_binding_sha("a" * 64, "b" * 64)
    assert first != v6._run_binding_sha("c" * 64, "b" * 64)
    assert first != v6._run_binding_sha("a" * 64, "d" * 64)


def test_nonprepare_requires_exact_concurrency_128(tmp_path: Path) -> None:
    with pytest.raises(v5.SyntheticRewriteError, match="requires concurrency=128"):
        v6.run_pipeline(
            object(),  # rejected before any handoff field is read
            output_root=tmp_path,
            tokenizer=object(),
            backend=NoCallBackend(),
            environment={},
            phase="all",
            concurrency=32,
        )


def test_immutable_writer_rejects_symlink(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}\n")
    link = tmp_path / "receipt.json"
    link.symlink_to(target)
    with pytest.raises(v5.SyntheticRewriteError, match="symlink"):
        v6._store_immutable(link, {"status": "complete"})


def test_source_reject_terminal_never_calls_provider(tmp_path: Path) -> None:
    row = _row()
    terminal = v6._process_downstream_terminal(
        row,
        output=tmp_path,
        tokenizer=object(),
        reference_bank={},
        official_reference_bank_sha256="a" * 64,
        backend=NoCallBackend(),
        identity=v5.ProviderIdentityRegistry(),
        environment={},
        config=v5.ProviderConfig(),
        code_sha256="b" * 64,
        source_audit=_source_reject(),
    )
    assert terminal["terminal_status"] == v5.TERMINAL_SOURCE_REJECT
    assert terminal["training_pass"] is False
    assert not any(
        (tmp_path / "cache" / role).exists() for role in v6.SOURCE_PROVIDER_ROLES
    )


def test_preflight_uses_admitted_rows_and_tops_up_same_split() -> None:
    prepared = {
        "train": tuple(_row("train", index) for index in range(4)),
        "validation": tuple(_row("validation", index) for index in range(4)),
        "test": tuple(_row("test", index) for index in range(3)),
    }
    source_results = {
        row.sample_id: {"machine_pass": True}
        for rows in prepared.values()
        for row in rows
    }
    rejected_once: set[str] = set()

    def worker(row: v5.PreparedRow) -> dict[str, Any]:
        if row.split not in rejected_once:
            rejected_once.add(row.split)
            return {
                "sample_id": row.sample_id,
                "terminal_status": v5.TERMINAL_STYLE_REJECT,
                "training_pass": False,
                "generation": {},
            }
        return _pass_terminal(row.sample_id)

    selected, terminals, attempts = v6._select_downstream_preflight(
        prepared, source_results, worker=worker, concurrency=128
    )
    assert len(selected) == 8
    assert {
        split: sum(row.split == split for row in selected) for split in v5.SPLITS
    } == {
        "train": 3,
        "validation": 3,
        "test": 2,
    }
    assert len(terminals) == 11
    assert sum(not attempt["selected"] for attempt in attempts) == 3


def test_resume_rejects_source_role_cache_namespace(tmp_path: Path) -> None:
    row = _row()
    source_root = tmp_path / "cache" / v5.ROLE_SOURCE_AUDIT_PRIMARY
    source_root.mkdir(parents=True)
    (source_root / f"{v5.sha256_text(row.sample_id)}.json").write_text("{}\n")
    with pytest.raises(
        v5.SyntheticRewriteError, match="source-role cache is forbidden"
    ):
        v6._load_downstream_terminal(
            tmp_path,
            row,
            code_sha256="a" * 64,
            official_reference_bank_sha256="b" * 64,
            identity=v5.ProviderIdentityRegistry(),
            reference_bank={},
            config=v5.ProviderConfig(),
            environment={},
            source_audit=_source_reject(),
            tokenizer=object(),
        )


def test_full_resume_rejects_provider_system_prompt_binding_tamper(
    tmp_path: Path,
) -> None:
    fixtures, row, reference, _terminal = _build_pass_terminal(tmp_path)
    cache_path = v5._cache_path(tmp_path, v5.ROLE_REWRITE_PRIMARY, row)
    cache = json.loads(cache_path.read_text())
    cache["binding"]["system_prompt_sha256"] = "0" * 64
    cache["binding_sha256"] = v5.sha256_text(v5.canonical_json(cache["binding"]))
    cache_path.write_text(v5.canonical_json(cache) + "\n")
    with pytest.raises(v5.SyntheticRewriteError, match="drift|binding mismatch"):
        v6._load_downstream_terminal(
            tmp_path,
            row,
            code_sha256="b" * 64,
            official_reference_bank_sha256="a" * 64,
            identity=v5.ProviderIdentityRegistry(),
            reference_bank=reference,
            config=v5.ProviderConfig(),
            environment={v5.API_KEY_ENV: "not-persisted"},
            source_audit=_source_pass(),
            tokenizer=fixtures.FakeTokenizer(),
        )
    assert fixtures is not None


def test_full_style_repair_terminal_resumes_without_provider_calls(
    tmp_path: Path,
) -> None:
    _fixtures, row, reference, terminal = _build_pass_terminal(tmp_path)
    assert terminal["generation"]["style_repair_used"] is True
    resumed = v6._load_downstream_terminal(
        tmp_path,
        row,
        code_sha256="b" * 64,
        official_reference_bank_sha256="a" * 64,
        identity=v5.ProviderIdentityRegistry(),
        reference_bank=reference,
        config=v5.ProviderConfig(),
        environment={v5.API_KEY_ENV: "not-persisted"},
        source_audit=_source_pass(),
        tokenizer=_fixtures.FakeTokenizer(),
    )
    assert resumed == terminal


def test_full_fidelity_repair_terminal_resumes_without_provider_calls(
    tmp_path: Path,
) -> None:
    fixtures = _v5_fixtures()

    class FidelityPassBackend:
        def __init__(self) -> None:
            self.count = 0

        def generate(self, *, role, user_prompt, **kwargs):
            del kwargs
            self.count += 1
            if role == v5.ROLE_REWRITE_PRIMARY:
                content, reasoning = {"answer": fixtures.REWRITE}, fixtures.REASONING
            elif role == v5.ROLE_VALIDATOR_A_PRIMARY:
                content, reasoning = (
                    fixtures._validator_a_payload(user_prompt, passed=False),
                    "Primary fidelity check.",
                )
            elif role == v5.ROLE_REWRITE_FIDELITY_REPAIR:
                content, reasoning = (
                    {"answer": fixtures.REWRITE},
                    fixtures.REASONING,
                )
            elif role == v5.ROLE_VALIDATOR_A_FIDELITY_REPAIR:
                content, reasoning = (
                    fixtures._validator_a_payload(user_prompt, passed=True),
                    "Repaired fidelity check.",
                )
            elif role == v5.ROLE_VALIDATOR_B_PRIMARY:
                content, reasoning = (
                    fixtures._validator_b_payload(
                        user_prompt, [8, 8, 8, 8, 8, 8], reported_pass=True
                    ),
                    "Official style check.",
                )
            else:  # pragma: no cover
                raise AssertionError(role)
            return fixtures._response(
                f"fidelity-{self.count}", content, reasoning=reasoning
            )

    row = fixtures._row()
    reference = fixtures._reference_bank()
    tokenizer = fixtures.FakeTokenizer()
    terminal = v6._process_downstream_terminal(
        row,
        output=tmp_path,
        tokenizer=tokenizer,
        reference_bank=reference,
        official_reference_bank_sha256="a" * 64,
        backend=FidelityPassBackend(),
        identity=v5.ProviderIdentityRegistry(),
        environment={v5.API_KEY_ENV: "not-persisted"},
        config=v5.ProviderConfig(),
        code_sha256="b" * 64,
        source_audit=_source_pass(),
    )
    assert terminal["terminal_status"] == v5.TERMINAL_PASS
    assert terminal["generation"]["fidelity_repair_used"] is True
    assert terminal["generation"]["style_repair_used"] is False
    resumed = v6._load_downstream_terminal(
        tmp_path,
        row,
        code_sha256="b" * 64,
        official_reference_bank_sha256="a" * 64,
        identity=v5.ProviderIdentityRegistry(),
        reference_bank=reference,
        config=v5.ProviderConfig(),
        environment={v5.API_KEY_ENV: "not-persisted"},
        source_audit=_source_pass(),
        tokenizer=tokenizer,
    )
    assert resumed == terminal


def test_full_resume_rejects_selected_candidate_terminal_tamper(tmp_path: Path) -> None:
    _fixtures, row, reference, _terminal = _build_pass_terminal(tmp_path)
    terminal_path = v5._terminal_path(tmp_path, row)
    payload = json.loads(terminal_path.read_text())
    payload["record"]["rewritten_minutes"] = "A different paragraph."
    payload["record"]["rewritten_minutes_sha256"] = v5.sha256_text(
        payload["record"]["rewritten_minutes"]
    )
    payload["record"]["sft_response"] = (
        payload["record"]["teacher_response_analysis"]
        + v5.BOUNDARY
        + payload["record"]["rewritten_minutes"]
    )
    payload["record"]["response_sha256"] = v5.sha256_text(
        payload["record"]["sft_response"]
    )
    payload["record_sha256"] = v5.sha256_text(v5.canonical_json(payload["record"]))
    terminal_path.write_text(v5.canonical_json(payload) + "\n")
    with pytest.raises(
        v5.SyntheticRewriteError, match="selected rewrite candidate drift"
    ):
        v6._load_downstream_terminal(
            tmp_path,
            row,
            code_sha256="b" * 64,
            official_reference_bank_sha256="a" * 64,
            identity=v5.ProviderIdentityRegistry(),
            reference_bank=reference,
            config=v5.ProviderConfig(),
            environment={v5.API_KEY_ENV: "not-persisted"},
            source_audit=_source_pass(),
            tokenizer=_fixtures.FakeTokenizer(),
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("phase", "generate"),
        ("status", "prepared"),
        ("configured_concurrency", 32),
        ("runner_sha256", "9" * 64),
    ],
)
def test_execution_receipt_header_rejects_semantic_tamper(
    field: str, value: Any
) -> None:
    implementation = {
        "artifacts": {"downstream_v6": {"sha256": "1" * 64}},
        "composite_sha256": "2" * 64,
    }
    receipt = {key: None for key in v6.EXECUTION_RECEIPT_FIELDS - {"receipt_sha256"}}
    receipt.update(
        {
            "schema_version": v6.EXECUTION_SCHEMA_VERSION,
            "phase": "all",
            "status": "complete",
            "configured_concurrency": 128,
            "maximum_concurrency": 128,
            "runner_sha256": "1" * 64,
        }
    )
    receipt[field] = value
    receipt["receipt_sha256"] = v5.sha256_text(v5.canonical_json(receipt))
    with pytest.raises(v5.SyntheticRewriteError, match="receipt header drift"):
        v6._validate_execution_receipt_header(
            receipt, phase="all", implementation=implementation
        )


def test_preflight_header_rejects_332_selection_tamper() -> None:
    split_by_id = {
        **{f"train-{index}": "train" for index in range(3)},
        **{f"validation-{index}": "validation" for index in range(3)},
        **{f"test-{index}": "test" for index in range(2)},
    }
    selected = list(split_by_id)
    preflight = {
        "schema_version": v6.SCHEMA_VERSION,
        "source_handoff_manifest_sha256": "a" * 64,
        "target_split_counts": v6.PREFLIGHT_SPLIT_COUNTS,
        "selected_ids": selected,
        "attempts": [],
        "passed": True,
    }
    v6._validate_preflight_header(
        preflight, handoff_sha256="a" * 64, split_by_id=split_by_id
    )
    preflight["selected_ids"] = selected[:-1]
    with pytest.raises(v5.SyntheticRewriteError, match="preflight contract drift"):
        v6._validate_preflight_header(
            preflight, handoff_sha256="a" * 64, split_by_id=split_by_id
        )
