from __future__ import annotations

import json
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Mapping

import pytest

import jobs.generation.prepare_chk4_supplement as prepare_supplement

from jobs.generation.generate_chk4_sft_targets import (
    DEFAULT_FFR_HISTORY,
    DEFAULT_WORKBOOK,
    MANIFEST_SCHEMA,
    TeacherResponse,
    canonical_json,
    sha256_text,
)
from jobs.generation.generate_chk4_supplement import (
    ProviderDriftError,
    _cache_key,
    _evidence_profile_binding,
    _expand_grounded_abbreviated_year_ranges,
    _expand_grounded_basis_points,
    _grounded_summary_number_atoms,
    _number_atoms,
    _summary_rows,
    audit_blind_predictions,
    combine_and_materialize,
    qualitative_reasoning_has_number_or_date,
    run_blind_predictions,
    run_summaries,
    run_teacher_targets,
)
from jobs.generation.prepare_chk4_supplement import (
    ADMISSION_PROFILE,
    DEFAULT_BASE_REGISTRY,
    DEFAULT_BASE_ROSTER,
    EXPECTED_ACTION_COUNTS,
    MIN_VALID_ATOMIC_TOPICS,
    POPULATION,
    compress_indicator_row,
    load_candidates,
    prepare_release,
)


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return list(range(max(1, len(text.split()))))

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return "\n".join(message["content"] for message in messages) + "\nassistant:"


class QueueBackend:
    def __init__(self, contents: list[str], fingerprints: list[str] | None = None):
        self.contents = list(contents)
        self.fingerprints = list(fingerprints or ["fp-fixed"] * len(contents))
        self.calls: list[dict[str, Any]] = []

    def generate(
        self,
        *,
        config: Any,
        system_prompt: str,
        user_prompt: str,
        environment: Mapping[str, str] | None,
    ) -> TeacherResponse:
        del environment
        index = len(self.calls)
        self.calls.append(
            {
                "model": config.model,
                "max_tokens": config.max_tokens,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
            }
        )
        return TeacherResponse(
            reasoning="native provenance",
            content=self.contents[index],
            response_id=f"resp-{index}",
            returned_model="deepseek-v4-pro",
            system_fingerprint=self.fingerprints[index],
            finish_reason="stop",
            created=1,
            usage={"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        )


def test_supplement_alfred_transport_uses_unmodified_standard_request(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: dict[str, Any] = {}

    class FakeResponse:
        status = 200
        headers = {"Content-Type": "application/csv"}

        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return b"observation_date,SERIES_19930101\n1992-12-01,1\n"

    def fake_urlopen(request: Any, *, timeout: float) -> FakeResponse:
        observed["url"] = request.full_url
        observed["timeout"] = timeout
        observed["user_agent"] = request.get_header("User-agent")
        return FakeResponse()

    monkeypatch.setattr(prepare_supplement, "urlopen", fake_urlopen)
    response = prepare_supplement._supplement_alfred_http_get(
        "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=SERIES",
        12.0,
    )

    assert observed == {
        "url": "https://alfred.stlouisfed.org/graph/alfredgraph.csv?id=SERIES",
        "timeout": 12.0,
        "user_agent": None,
    }
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/csv"
    assert response.body.startswith(b"observation_date")


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )


def _write_evidence_profile(root: Path, *, count: int) -> None:
    handoff_payload_sha256 = "b" * 64
    handoff = {
        "schema_version": "chk1-source-handoff-v1",
        "payload_sha256": handoff_payload_sha256,
    }
    handoff_path = root / "sources/materialized/source_handoff.json"
    handoff_path.parent.mkdir(parents=True, exist_ok=True)
    handoff_path.write_text(canonical_json(handoff) + "\n", encoding="utf-8")

    evidence_audit = {
        "schema_version": "chk4-decision-supplement-evidence-audit-v2",
        "status": "complete",
        "admission_profile": ADMISSION_PROFILE,
        "candidate_count": count,
        "admitted_count": count,
        "rejected_count": 0,
        "minimum_admitted": count,
        "minimum_valid_atomic_topics": MIN_VALID_ATOMIC_TOPICS,
        "source_handoff_payload_sha256": handoff_payload_sha256,
    }
    audit_path = root / "reports/evidence_audit.json"
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(canonical_json(evidence_audit) + "\n", encoding="utf-8")


def _minimal_summary_inputs(root: Path, count: int = 1) -> None:
    prepared = []
    admitted = []
    for index in range(count):
        sample_id = f"dec-{index:024x}"
        prompt = (
            "Summarize target-neutral evidence: "
            '{"topic_evidence":[{"category":"prices","latest_value":"2"},'
            '{"category":"employment_activity","latest_value":"1"},'
            '{"category":"financial_conditions","latest_value":"3"}]}'
        )
        prepared.append(
            {
                "sample_id": sample_id,
                "prompt": prompt,
                "input_sha256": sha256_text(prompt),
                "prompt_sha256": sha256_text(prompt),
            }
        )
        admitted.append(
            {
                "sample_id": sample_id,
                "meeting_date": (date(1993, 1, 1) + timedelta(days=index)).isoformat(),
                "source_ids": [f"source-{index}"],
                "admission_profile": ADMISSION_PROFILE,
                "category_coverage": [
                    "prices",
                    "employment_activity",
                    "financial_conditions",
                ],
                "gold": {"direction": "hold", "magnitude_bp": 0},
            }
        )
    _write_jsonl(root / "prepared/summary_requests.jsonl", prepared)
    _write_jsonl(root / "manifests/admitted.jsonl", admitted)
    _write_evidence_profile(root, count=count)


def test_real_supplement_labels_are_frozen_and_reconciled() -> None:
    candidates, audit = load_candidates(DEFAULT_WORKBOOK, DEFAULT_FFR_HISTORY)
    assert len(candidates) == 109
    assert candidates[0].meeting_date == "1993-03-23"
    assert candidates[-1].meeting_date == "2008-12-16"
    assert audit["supplement_action_counts"] == EXPECTED_ACTION_COUNTS
    assert audit["ffr_mismatches"] == 0


def test_preflight_freezes_separate_roster_registry_and_nine_populations(
    tmp_path: Path,
) -> None:
    result = prepare_release(
        output_root=tmp_path,
        workbook=DEFAULT_WORKBOOK,
        ffr_history=DEFAULT_FFR_HISTORY,
        base_roster=DEFAULT_BASE_ROSTER,
        base_registry=DEFAULT_BASE_REGISTRY,
    )
    assert result["status"] == "prepared"
    assert result["api_requests"] == 0
    plan = json.loads((tmp_path / "sources/plan/source_plan.json").read_text())
    assert [len(row["meeting_dates"]) for row in plan["populations"]] == [
        13,
        13,
        13,
        13,
        13,
        11,
        11,
        11,
        11,
    ]
    roster = json.loads(
        (tmp_path / "sources/contracts/leave_one_out_roster_pre2009.json").read_text()
    )
    registry = json.loads(
        (tmp_path / "sources/contracts/loo_indicator_sources_pre2009.json").read_text()
    )
    assert roster["contexts"] == ["1993_2008_pre_meeting_d1"]
    assert registry["roster_id"] == roster["roster_id"]
    assert registry["policy"]["runtime_fallback"] == "forbidden"


def test_deterministic_compression_selects_two_series_and_removes_dates() -> None:
    def series(code: str, count: int) -> dict[str, Any]:
        return {
            "series_id": code,
            "title": f"Series {code}",
            "frequency": "monthly",
            "units": "Percent",
            "observations": [
                {
                    "date": (date(1990, 1, 1) + timedelta(days=30 * index)).isoformat(),
                    "value": str(index),
                }
                for index in range(count)
            ],
        }

    result = compress_indicator_row(
        {
            "indicator": "Consumer-Price-Index-(CPI)",
            "source_payload": {
                "series": [series("B", 4), series("A", 5), series("C", 3)]
            },
        }
    )
    assert result is not None
    assert [row["series_code"] for row in result["series"]] == ["A", "B"]
    rendered = canonical_json(
        {key: value for key, value in result.items() if key != "provenance"}
    )
    assert "1990-" not in rendered
    assert result["category"] == "prices"


def test_summary_repair_cache_and_resume(tmp_path: Path) -> None:
    _minimal_summary_inputs(tmp_path)
    valid = json.dumps(
        {
            "meeting_decision_brief": (
                "Inflation pressures persisted while employment activity remained "
                "firm, financial market conditions were supportive, and the balance "
                "of risks remained uncertain."
            )
        }
    )
    backend = QueueBackend(["not-json", valid])
    summary = run_summaries(
        output_root=tmp_path,
        tokenizer_path=Path("unused"),
        concurrency=1,
        resume=False,
        backend=backend,
        tokenizer=FakeTokenizer(),
    )
    assert summary["accepted_count"] == 1
    assert len(backend.calls) == 2
    rows = [
        json.loads(line)
        for line in (tmp_path / "summaries/meeting_decision_briefs.jsonl")
        .read_text()
        .splitlines()
    ]
    assert len(rows) == 1

    no_call = QueueBackend([])
    resumed = run_summaries(
        output_root=tmp_path,
        tokenizer_path=Path("unused"),
        concurrency=1,
        resume=True,
        backend=no_call,
        tokenizer=FakeTokenizer(),
    )
    assert resumed["resumed_count"] == 1
    assert no_call.calls == []


def test_summary_has_one_bounded_second_repair(tmp_path: Path) -> None:
    _minimal_summary_inputs(tmp_path)
    valid = json.dumps(
        {
            "meeting_decision_brief": (
                "Inflation pressures persisted while employment activity remained "
                "firm, financial conditions were supportive, and risks remained "
                "uncertain."
            )
        }
    )
    backend = QueueBackend(["not-json", "still-not-json", valid])
    summary = run_summaries(
        output_root=tmp_path,
        tokenizer_path=Path("unused"),
        concurrency=1,
        resume=False,
        backend=backend,
        tokenizer=FakeTokenizer(),
    )
    assert summary["accepted_count"] == 1
    assert len(backend.calls) == 3
    accepted = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (tmp_path / "cache/summaries/accepted").rglob("*.json")
    ]
    assert accepted[0]["attempt"] == "repair_2"
    contract = json.loads(
        (tmp_path / "summaries/prompt_contract.grounding-v6.json").read_text(
            encoding="utf-8"
        )
    )
    assert contract["repair_attempts"] == 2


def test_summary_revalidates_compatible_v5_cache_without_provider_call(
    tmp_path: Path,
) -> None:
    _minimal_summary_inputs(tmp_path)
    valid = json.dumps(
        {
            "meeting_decision_brief": (
                "Inflation pressures persisted while employment activity remained "
                "firm, financial conditions were supportive, and risks remained "
                "uncertain."
            )
        }
    )
    run_summaries(
        output_root=tmp_path,
        tokenizer_path=Path("unused"),
        concurrency=1,
        resume=False,
        backend=QueueBackend([valid]),
        tokenizer=FakeTokenizer(),
    )
    contract_path = tmp_path / "summaries/prompt_contract.grounding-v6.json"
    current_contract = json.loads(contract_path.read_text(encoding="utf-8"))
    prior_contract = dict(current_contract)
    prior_contract.pop("contract_sha256")
    prior_contract["code_sha256"] = "f" * 64
    prior_contract["repair_attempts"] = 1
    prior_contract["contract_sha256"] = sha256_text(canonical_json(prior_contract))
    (tmp_path / "summaries/prompt_contract.grounding-v5.json").write_text(
        canonical_json(prior_contract) + "\n",
        encoding="utf-8",
    )

    current_path = next((tmp_path / "cache/summaries/accepted").rglob("*.json"))
    current_payload = json.loads(current_path.read_text(encoding="utf-8"))
    row = _summary_rows(tmp_path)[0]
    prior_key = _cache_key("summaries", row, prior_contract)
    prior_payload = dict(current_payload)
    prior_payload["cache_key"] = prior_key
    prior_payload["contract_sha256"] = prior_contract["contract_sha256"]
    prior_path = (
        tmp_path / "cache/summaries/accepted" / prior_key[:2] / f"{prior_key}.json"
    )
    prior_path.parent.mkdir(parents=True, exist_ok=True)
    prior_path.write_text(canonical_json(prior_payload) + "\n", encoding="utf-8")
    current_path.unlink()

    no_call = QueueBackend([])
    summary = run_summaries(
        output_root=tmp_path,
        tokenizer_path=Path("unused"),
        concurrency=1,
        resume=True,
        backend=no_call,
        tokenizer=FakeTokenizer(),
    )
    assert summary["revalidated_count"] == 1
    assert summary["api_requests"] == 0
    assert no_call.calls == []
    current_key = _cache_key("summaries", row, current_contract)
    promoted = json.loads(
        (
            tmp_path
            / "cache/summaries/accepted"
            / current_key[:2]
            / f"{current_key}.json"
        ).read_text(encoding="utf-8")
    )
    assert (
        promoted["revalidated_from"]["contract_sha256"]
        == prior_contract["contract_sha256"]
    )


def test_provider_fingerprint_drift_stops_stage(tmp_path: Path) -> None:
    _minimal_summary_inputs(tmp_path, count=2)
    valid = json.dumps(
        {
            "meeting_decision_brief": (
                "Inflation was 2 while employment activity was 1, financial market "
                "conditions were 3, and risks remained uncertain."
            )
        }
    )
    backend = QueueBackend([valid, valid], fingerprints=["fp-one", "fp-two"])
    with pytest.raises(ProviderDriftError):
        run_summaries(
            output_root=tmp_path,
            tokenizer_path=Path("unused"),
            concurrency=1,
            resume=False,
            backend=backend,
            tokenizer=FakeTokenizer(),
        )


def test_unsupported_summary_number_uses_only_repair_attempt(tmp_path: Path) -> None:
    _minimal_summary_inputs(tmp_path)
    unsupported = json.dumps(
        {
            "meeting_decision_brief": (
                "Inflation was 99 while employment activity was 1 and financial "
                "market conditions were 3."
            )
        }
    )
    repaired = json.dumps(
        {
            "meeting_decision_brief": (
                "Inflation pressures persisted while employment activity was "
                "firm, financial market conditions were supportive, and risks "
                "remained uncertain."
            )
        }
    )
    backend = QueueBackend([unsupported, repaired])
    run_summaries(
        output_root=tmp_path,
        tokenizer_path=Path("unused"),
        concurrency=1,
        resume=False,
        backend=backend,
        tokenizer=FakeTokenizer(),
    )
    assert len(backend.calls) == 2
    assert (
        "fully qualitative language with no digits"
        in backend.calls[1]["system_prompt"].lower()
    )
    rejected = list((tmp_path / "cache/summaries/rejected").rglob("*.json"))
    assert len(rejected) == 1
    assert "unsupported_numbers:99" in rejected[0].read_text()


def test_summary_number_grounding_accepts_explicit_unit_scaling(
    tmp_path: Path,
) -> None:
    sample_id = "dec-000000000000000000000000"
    payload = {
        "topic_evidence": [
            {
                "category": "prices",
                "topic": "Consumer-Price-Index-(CPI)",
                "series": [
                    {
                        "latest_value": "145.8",
                        "relative_changes": [],
                        "units": "Index 1982-1984=100",
                    }
                ],
            },
            {
                "category": "employment_activity",
                "topic": "Labour-Market",
                "series": [
                    {
                        "latest_value": "100000",
                        "relative_changes": [
                            {
                                "relative_horizon": "previous",
                                "value_change": "256",
                            },
                            {
                                "relative_horizon": "short",
                                "value_change": "-130",
                            },
                        ],
                        "units": "Thousands of Persons",
                    }
                ],
            },
            {
                "category": "financial_conditions",
                "topic": "Money-Supply",
                "series": [
                    {
                        "latest_value": "100",
                        "relative_changes": [],
                        "units": "Billions of Dollars",
                    }
                ],
            },
        ]
    }
    prompt = "Summarize target-neutral evidence: " + canonical_json(payload)
    _write_jsonl(
        tmp_path / "prepared/summary_requests.jsonl",
        [
            {
                "sample_id": sample_id,
                "prompt": prompt,
                "input_sha256": sha256_text(prompt),
                "prompt_sha256": sha256_text(prompt),
            }
        ],
    )
    _write_jsonl(
        tmp_path / "manifests/admitted.jsonl",
        [{"sample_id": sample_id, "admission_profile": ADMISSION_PROFILE}],
    )
    _write_evidence_profile(tmp_path, count=1)
    backend = QueueBackend(
        [
            json.dumps(
                {
                    "meeting_decision_brief": (
                        "The CPI was 145.8 (1982-84=100), payroll employment "
                        "added 256,000 persons before falling 130,000, and the "
                        "M2 money stock was 100.0 billion dollars. Inflation "
                        "risks may remain balanced."
                    )
                }
            )
        ]
    )
    summary = run_summaries(
        output_root=tmp_path,
        tokenizer_path=Path("unused"),
        concurrency=1,
        resume=False,
        backend=backend,
        tokenizer=FakeTokenizer(),
    )
    assert summary["accepted_count"] == 1
    assert len(backend.calls) == 1


def test_summary_number_grounding_is_occurrence_and_unit_scoped() -> None:
    supplied_range = '"units":"Index 1982-1984=100"'
    valid = _expand_grounded_abbreviated_year_ranges(
        supplied_range,
        "The index was 145.8 (1982-84=100).",
    )
    assert valid == "The index was 145.8 (1982-1984=100)."

    extra_hallucination = _expand_grounded_abbreviated_year_ranges(
        supplied_range,
        "The index used 1982-84=100 and rose 84 points.",
    )
    grounded = _number_atoms(supplied_range)
    assert _number_atoms(extra_hallucination) - grounded == {"84"}
    assert (
        "85"
        in _number_atoms(
            _expand_grounded_abbreviated_year_ranges(
                supplied_range,
                "The index used 1982-85=100.",
            )
        )
        - grounded
    )
    assert (
        _expand_grounded_abbreviated_year_ranges(
            '"start":1982,"end":1984',
            "The index used 1982-84=100.",
        )
        == "The index used 1982-84=100."
    )

    scaled_payload = canonical_json(
        {
            "topic_evidence": [
                {
                    "series": [
                        {
                            "units": "Thousands of Persons",
                            "latest_value": "100",
                            "relative_changes": [
                                {"value_change": "-130"},
                                {"value_change": "-9"},
                            ],
                        }
                    ]
                }
            ]
        }
    )
    scaled = _grounded_summary_number_atoms(scaled_payload)
    assert {"130000", "9000"} <= scaled
    assert "131000" not in scaled
    unscaled = _grounded_summary_number_atoms(
        scaled_payload.replace("Thousands of Persons", "Persons")
    )
    assert "130000" not in unscaled


def test_qualitative_date_guard_distinguishes_modal_may_from_month() -> None:
    assert not qualitative_reasoning_has_number_or_date(
        "Inflation risks may remain balanced."
    )
    assert qualitative_reasoning_has_number_or_date(
        "The observation was recorded in May."
    )


def test_summary_basis_point_conversion_is_occurrence_and_unit_scoped() -> None:
    source = canonical_json(
        {
            "topic_evidence": [
                {
                    "series": [
                        {
                            "units": "Percent",
                            "relative_changes": [
                                {"value_change": "-0.39"},
                                {"value_change": "-0.04"},
                            ],
                        }
                    ]
                }
            ]
        }
    )
    valid = _expand_grounded_basis_points(
        source,
        "Yields fell 39 basis points and the policy rate fell 4 bps.",
    )
    assert valid == (
        "Yields fell 0.39 percentage points and the policy rate fell "
        "0.04 percentage points."
    )
    assert "40 basis points" in _expand_grounded_basis_points(
        source,
        "Yields fell 40 basis points.",
    )
    assert "39 points" in _expand_grounded_basis_points(
        source,
        "Yields fell 39 points.",
    )
    wrong_units = source.replace('"Percent"', '"Index 2017=100"')
    assert "39 basis points" in _expand_grounded_basis_points(
        wrong_units,
        "The index fell 39 basis points.",
    )
    mixed = _expand_grounded_basis_points(
        source,
        "Yields fell 39 basis points, then rose another 39 points.",
    )
    grounded = _grounded_summary_number_atoms(source)
    assert _number_atoms(mixed) - grounded == {"39"}


def test_blind_completion_budget_uses_one_repair(tmp_path: Path) -> None:
    _minimal_summary_inputs(tmp_path)
    brief = "Inflation eased while employment activity and financial conditions were stable."
    _write_jsonl(
        tmp_path / "summaries/meeting_decision_briefs.jsonl",
        [
            {
                "sample_id": "dec-000000000000000000000000",
                "meeting_decision_brief": brief,
            }
        ],
    )
    too_long = json.dumps(
        {
            "reasoning": " ".join(["stable"] * 600),
            "direction": "hold",
            "magnitude_bp": 0,
        }
    )
    repaired = json.dumps(
        {
            "reasoning": "Inflation eased and activity was stable.",
            "direction": "hold",
            "magnitude_bp": 0,
        }
    )
    backend = QueueBackend([too_long, repaired])
    result = run_blind_predictions(
        output_root=tmp_path,
        tokenizer_path=Path("unused"),
        concurrency=1,
        resume=False,
        backend=backend,
        tokenizer=FakeTokenizer(),
    )
    assert result["accepted_count"] == 1
    assert len(backend.calls) == 2


def test_blind_prediction_never_sends_gold_and_audit_does_not_filter(
    tmp_path: Path,
) -> None:
    _minimal_summary_inputs(tmp_path)
    brief = "Inflation eased while employment activity and financial conditions were stable."
    _write_jsonl(
        tmp_path / "summaries/meeting_decision_briefs.jsonl",
        [
            {
                "sample_id": "dec-000000000000000000000000",
                "meeting_decision_brief": brief,
            }
        ],
    )
    backend = QueueBackend(
        [
            json.dumps(
                {
                    "reasoning": "Inflation eased and activity was stable.",
                    "direction": "hold",
                    "magnitude_bp": 0,
                }
            )
        ]
    )
    run_blind_predictions(
        output_root=tmp_path,
        tokenizer_path=Path("unused"),
        concurrency=1,
        resume=False,
        backend=backend,
        tokenizer=FakeTokenizer(),
    )
    sent = backend.calls[0]["user_prompt"].lower()
    assert "canonical decision" not in sent
    assert '"direction"' not in sent
    metrics = audit_blind_predictions(tmp_path)
    assert metrics["sample_count"] == 1
    assert metrics["selection_policy"] == "audit_only_never_filter_training_rows"


def test_hidden_gold_target_is_serialized_locally(tmp_path: Path) -> None:
    _minimal_summary_inputs(tmp_path)
    brief = "Inflation eased while employment activity and financial conditions were stable."
    _write_jsonl(
        tmp_path / "summaries/meeting_decision_briefs.jsonl",
        [
            {
                "sample_id": "dec-000000000000000000000000",
                "meeting_decision_brief": brief,
            }
        ],
    )
    backend = QueueBackend(
        [
            json.dumps(
                {
                    "reasoning": "Easing inflation and stable activity support maintaining the stance.",
                    "direction": "hold",
                    "magnitude_bp": 0,
                }
            )
        ]
    )
    result = run_teacher_targets(
        output_root=tmp_path,
        tokenizer_path=Path("unused"),
        concurrency=1,
        resume=False,
        backend=backend,
        tokenizer=FakeTokenizer(),
    )
    assert result["accepted_count"] == 1
    row = json.loads(
        (tmp_path / "teacher_targets/supplement/sft/train.jsonl").read_text().strip()
    )
    assert row["response"].count("</think>") == 1
    assert row["response"].endswith('</think>\n{"direction":"hold","magnitude_bp":0}')


def _teacher_release(
    root: Path, *, counts: dict[str, int], role: str, start: date
) -> None:
    cursor = 0
    directions = (("hold", 0), ("cut", 25), ("hike", 25))
    for split in ("train", "validation", "test"):
        manifests = []
        sft = []
        for index in range(counts.get(split, 0)):
            meeting = (start + timedelta(days=cursor)).isoformat()
            cursor += 1
            direction, magnitude = directions[index % len(directions)]
            sample_id = f"dec-{sha256_text(role + meeting)[:24]}"
            prompt = 'Make a decision:\n{"analysis":"neutral target-free analysis"}'
            gold_text = json.dumps(
                {"direction": direction, "magnitude_bp": magnitude},
                separators=(",", ":"),
            )
            response = "Grounded qualitative reasoning.\n</think>\n" + gold_text
            manifests.append(
                {
                    "schema_version": MANIFEST_SCHEMA,
                    "sample_id": sample_id,
                    "meeting_date": meeting,
                    "split": split,
                    "population": "decision_core_post2009_v1"
                    if role == "core"
                    else POPULATION,
                    "population_role": role,
                    "source_ids": [f"source-{sample_id}"],
                    "gold": {"direction": direction, "magnitude_bp": magnitude},
                    "input_sha256": sha256_text("neutral target-free analysis"),
                    "prompt_sha256": sha256_text(prompt),
                    "gold_sha256": sha256_text(gold_text),
                    "contract_sha256": f"contract-{role}",
                }
            )
            sft.append({"prompt": prompt, "response": response})
        _write_jsonl(root / "manifests" / f"{split}.jsonl", manifests)
        _write_jsonl(root / "sft" / f"{split}.jsonl", sft)
    (root / "summary.json").write_text(
        json.dumps({"status": "complete", "contract_sha256": f"contract-{role}"}),
        encoding="utf-8",
    )


def test_combined_release_rebalances_after_supplement_merge(tmp_path: Path) -> None:
    core = tmp_path / "core"
    output = tmp_path / "output"
    supplement = output / "teacher_targets/supplement"
    _teacher_release(
        core,
        counts={"train": 102, "validation": 13, "test": 13},
        role="core",
        start=date(2009, 1, 1),
    )
    _teacher_release(
        supplement,
        counts={"train": 109, "validation": 0, "test": 0},
        role="supplement",
        start=date(1993, 1, 1),
    )
    admitted = [
        {**row, "admission_profile": ADMISSION_PROFILE}
        for row in (
            json.loads(line)
            for line in (supplement / "manifests/train.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
    ]
    _write_jsonl(output / "manifests/admitted.jsonl", admitted)
    _write_evidence_profile(output, count=109)
    supplement_summary_path = supplement / "summary.json"
    supplement_summary = json.loads(supplement_summary_path.read_text(encoding="utf-8"))
    supplement_summary.update(_evidence_profile_binding(output))
    supplement_summary_path.write_text(
        canonical_json(supplement_summary) + "\n",
        encoding="utf-8",
    )
    result = combine_and_materialize(
        output_root=output,
        core_teacher_root=core,
        resume=False,
    )
    training = result["training_summary"]
    assert training["unique_counts"] == {"train": 211, "validation": 13, "test": 13}
    assert training["repeat_factors"]["hold"] == 1
    assert training["physical_counts"]["decision_sft"]["train"] >= 200
