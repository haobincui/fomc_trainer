from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from jobs.generation.generate_chk4_sft_targets import (
    MAX_TEACHER_TOKENS,
    Chk4TargetError,
    ModelDriftError,
    PreparedRow,
    ProviderIdentityGuard,
    TeacherResponse,
    build_prompt_contract,
    generate_one,
    load_and_validate_labels,
    load_core_split_map,
    map_rate_change,
    run,
    sha256_text,
    validate_response,
)
from jobs.generation.materialize_chk4_training_data import materialize
from jobs.generation.generate_chk4_meeting_briefs import (
    MeetingInput,
    build_contract as build_brief_contract,
    run as run_briefs,
)
from jobs.generation.repair_chk4_meeting_briefs import (
    _repair_contract,
    repair_one,
)
from jobs.generation.repair_chk4_sft_targets import (
    _repair_contract as _target_repair_contract,
    repair_one as repair_target_one,
)


class TinyTokenizer:
    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt):
        assert tokenize is False
        assert add_generation_prompt is True
        return "\n".join(message["content"] for message in messages) + "\n<think>\n"

    def encode(self, text, *, add_special_tokens=False):
        assert add_special_tokens is False
        return text.split()


class SequenceBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def generate(self, **kwargs):
        assert kwargs["config"].max_tokens == MAX_TEACHER_TOKENS
        self.requests.append(kwargs)
        return self.responses.pop(0)


class EchoGoldBackend:
    def __init__(self):
        self._lock = threading.Lock()
        self.count = 0

    def generate(self, **kwargs):
        gold = json.loads(kwargs["user_prompt"].splitlines()[-1])
        gold["reasoning"] = (
            "Inflation risks remained elevated while activity and labor "
            "conditions moderated, supporting a cautious policy stance."
        )
        with self._lock:
            self.count += 1
            response_id = f"resp-{self.count}"
        return response(
            content=json.dumps(gold, separators=(",", ":")),
            response_id=response_id,
        )


class NeutralBriefBackend:
    def __init__(self):
        self._lock = threading.Lock()
        self.count = 0

    def generate(self, **_kwargs):
        with self._lock:
            self.count += 1
            response_id = f"brief-{self.count}"
        return response(
            reasoning="Synthesize the supplied inflation, activity, labor, and financial evidence.",
            content=json.dumps(
                {
                    "meeting_decision_brief": (
                        "Inflation pressures were mixed, labor and real activity "
                        "showed moderation, financial conditions remained firm, "
                        "and the balance of risks was two-sided."
                    )
                }
            ),
            response_id=response_id,
        )


class BriefSequenceBackend:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    def generate(self, **kwargs):
        self.requests.append(kwargs)
        return self.responses.pop(0)


def response(
    *,
    reasoning="Inflation remained elevated, while employment growth moderated.",
    content=None,
    response_id="resp-1",
    model="deepseek-v4-pro",
    fingerprint="fp-stable",
):
    if content is None:
        content = json.dumps(
            {
                "reasoning": reasoning,
                "direction": "hold",
                "magnitude_bp": 0,
            },
            separators=(",", ":"),
        )
    return TeacherResponse(
        reasoning=reasoning,
        content=content,
        response_id=response_id,
        returned_model=model,
        system_fingerprint=fingerprint,
        finish_reason="stop",
        created=1,
        usage={"prompt_tokens": 20, "completion_tokens": 20, "total_tokens": 40},
    )


def row() -> PreparedRow:
    analysis = "Inflation remained elevated, while employment growth moderated."
    prompt = "Make one policy decision:\n" + json.dumps({"analysis": analysis})
    gold = '{"direction":"hold","magnitude_bp":0}'
    return PreparedRow(
        sample_id="dec-0123456789abcdef01234567",
        meeting_date="2020-01-01",
        split="train",
        population="decision_core_post2009_v1",
        analysis=analysis,
        prompt=prompt,
        teacher_prompt=prompt + "\n" + gold,
        direction="hold",
        magnitude_bp=0,
        source_ids=("source-1",),
        input_sha256=sha256_text(analysis),
        prompt_sha256=sha256_text(prompt),
        gold_sha256=sha256_text(gold),
    )


def test_local_label_mapping_is_deterministic():
    assert map_rate_change("No change") == ("hold", 0)
    assert map_rate_change("Raise by 75 basis points") == ("hike", 75)
    assert map_rate_change("Cut by 100 basis points") == ("cut", 100)
    with pytest.raises(Chk4TargetError, match="unsupported policy magnitude"):
        map_rate_change("Cut by 20 basis points")


def test_archived_labels_reconcile_to_ffr_history():
    root = Path(__file__).resolve().parents[1]
    labels, audit = load_and_validate_labels(
        root
        / "archive/code/process_decsion_output/summary_base_with_meeting_date.xlsx",
        root / "dataset/raw_data/input_data/ffr/ffr_hist.xlsx",
    )
    assert len(labels) == 237
    assert audit["workbook_rows"] == 241
    assert audit["duplicate_rows"] == 4
    assert audit["ffr_mismatches"] == 0


def test_response_maps_reasoning_to_locally_serialized_gold():
    target = validate_response(
        row=row(),
        response=response(),
        tokenizer=TinyTokenizer(),
    )
    assert target["completion"].endswith(
        '</think>\n{"direction":"hold","magnitude_bp":0}'
    )
    assert target["completion"].count("</think>") == 1


def test_only_one_content_repair_is_used(tmp_path):
    first = response(
        content=json.dumps(
            {
                "reasoning": "Inflation remained elevated.",
                "direction": "hike",
                "magnitude_bp": 25,
            }
        ),
        response_id="resp-bad",
    )
    repaired = response(response_id="resp-good")
    backend = SequenceBackend([first, repaired])
    script = Path(__file__).resolve().parents[1] / (
        "jobs/generation/generate_chk4_sft_targets.py"
    )
    contract = build_prompt_contract(sha256_text(script.read_text(encoding="utf-8")))
    result = generate_one(
        row(),
        output_root=tmp_path,
        tokenizer=TinyTokenizer(),
        backend=backend,
        guard=ProviderIdentityGuard(),
        contract=contract,
        environment={"DEEPSEEK_API_KEY": "not-used-by-mock"},
    )
    assert result["status"] == "accepted"
    assert result["attempt"] == "repair"
    assert len(backend.requests) == 2
    assert "repair_directive" in backend.requests[1]["user_prompt"]


def test_unsupported_number_is_rejected():
    with pytest.raises(Exception, match="unsupported_reasoning_numbers"):
        validate_response(
            row=row(),
            response=response(reasoning="Inflation was 9 percent and employment moderated."),
            tokenizer=TinyTokenizer(),
        )


def test_returned_model_drift_stops_generation(tmp_path):
    backend = SequenceBackend([response(model="some-fallback")])
    contract = build_prompt_contract("0" * 64)
    with pytest.raises(ModelDriftError, match="returned model"):
        generate_one(
            row(),
            output_root=tmp_path,
            tokenizer=TinyTokenizer(),
            backend=backend,
            guard=ProviderIdentityGuard(),
            contract=contract,
            environment={},
        )


def test_full_core_mock_generation_and_resume(tmp_path):
    root = Path(__file__).resolve().parents[1]
    core_root = root / (
        "output/data/retrain_v2/chk1/canonical_releases/"
        "chk1_full_v7_automated_v2_20260804/manifests"
    )
    core = load_core_split_map(core_root)
    briefs = tmp_path / "briefs"
    briefs.mkdir()
    for split in ("train", "validation", "test"):
        rows = [
            {
                "meeting_date": meeting,
                "meeting_decision_brief": (
                    "Inflation remained elevated, while employment growth moderated."
                ),
                "source_ids": [f"source-{index}"],
            }
            for index, meeting in enumerate(
                sorted(date for date, assigned in core.items() if assigned == split)
            )
        ]
        (briefs / f"{split}.jsonl").write_text(
            "".join(json.dumps(item) + "\n" for item in rows), encoding="utf-8"
        )
    output = tmp_path / "output"
    backend = EchoGoldBackend()
    common = {
        "briefs_root": briefs,
        "output_root": output,
        "workbook": root
        / "archive/code/process_decsion_output/summary_base_with_meeting_date.xlsx",
        "ffr_history": root / "dataset/raw_data/input_data/ffr/ffr_hist.xlsx",
        "core_manifest_root": core_root,
        "tokenizer_path": root / "models/DeepSeek-R1-Distill-Llama-8B",
        "concurrency": 4,
        "tokenizer": TinyTokenizer(),
        "environment": {},
    }
    first = run(
        **common,
        dry_run=False,
        resume=False,
        backend=backend,
    )
    assert first["status"] == "complete"
    assert first["accepted_count"] == 128
    assert backend.count == 128
    assert sum(1 for _ in (output / "sft/train.jsonl").open()) == 102
    assert sum(1 for _ in (output / "sft/validation.jsonl").open()) == 13
    assert sum(1 for _ in (output / "sft/test.jsonl").open()) == 13

    resumed = run(
        **common,
        dry_run=False,
        resume=True,
        backend=SequenceBackend([]),
    )
    assert resumed["status"] == "complete"
    assert resumed["resumed_count"] == 128

    training_root = tmp_path / "training_data"
    materialized = materialize(
        teacher_root=output,
        output_root=training_root,
    )
    assert materialized["status"] == "complete"
    assert materialized["unique_counts"] == {
        "train": 102,
        "validation": 13,
        "test": 13,
    }
    assert materialized["repeat_factors"]["hold"] == 1
    assert 1 <= materialized["repeat_factors"]["cut"] <= 4
    assert 1 <= materialized["repeat_factors"]["hike"] <= 4
    assert materialized["physical_counts"]["decision_sft"]["validation"] == 13
    assert materialized["physical_counts"]["decision_grpo"]["test"] == 13
    assert materialize(
        teacher_root=output,
        output_root=training_root,
        allow_existing=True,
    ) == materialized


def test_full_gold_blind_brief_generation_with_mock(tmp_path):
    root = Path(__file__).resolve().parents[1]
    backend = NeutralBriefBackend()
    output = tmp_path / "briefs"
    summary = run_briefs(
        chk1_root=root
        / (
            "output/data/retrain_v2/chk1/canonical_releases/"
            "chk1_full_v7_automated_v2_20260804"
        ),
        output_root=output,
        tokenizer_path=root / "models/DeepSeek-R1-Distill-Llama-8B",
        dry_run=False,
        resume=False,
        concurrency=4,
        backend=backend,
        environment={},
        tokenizer=TinyTokenizer(),
    )
    assert summary["status"] == "complete"
    assert summary["gold_blind"] is True
    assert summary["accepted_count"] == 128
    assert backend.count == 128
    assert sum(1 for _ in (output / "train.jsonl").open()) == 102
    assert sum(1 for _ in (output / "validation.jsonl").open()) == 13
    assert sum(1 for _ in (output / "test.jsonl").open()) == 13
    prepared_text = "".join(
        path.read_text(encoding="utf-8")
        for path in sorted((output / "prepared").glob("*.jsonl"))
    ).lower()
    assert "rate_change" not in prepared_text
    assert "current_rate" not in prepared_text
    assert "teacher-only canonical decision" not in prepared_text


def test_targeted_brief_repair_retries_only_failed_row_and_preserves_raw(tmp_path):
    analysis = "Inflation pressures were mixed and employment growth moderated."
    user_prompt = "Synthesize supplied analyses:\n" + json.dumps(
        {"atomic_analyses": [{"atomic_topic": "labor", "analysis": analysis}]}
    )
    meeting_row = MeetingInput(
        sample_id="brief-test-repair",
        split="train",
        meeting_date="2020-01-01",
        atomic=({"atomic_topic": "labor", "analysis": analysis},),
        source_ids=("source-1",),
        user_prompt=user_prompt,
        input_sha256=sha256_text(analysis),
        prompt_sha256=sha256_text(user_prompt),
        category_coverage=("employment_activity",),
    )
    root = Path(__file__).resolve().parents[1]
    brief_script = root / "jobs/generation/generate_chk4_meeting_briefs.py"
    repair_script = root / "jobs/generation/repair_chk4_meeting_briefs.py"
    original_contract = build_brief_contract(
        sha256_text(brief_script.read_text(encoding="utf-8"))
    )
    repair_contract = _repair_contract(
        sha256_text(repair_script.read_text(encoding="utf-8")),
        original_contract["contract_sha256"],
    )
    backend = BriefSequenceBackend(
        [
            response(content="", response_id="brief-repair-bad"),
            response(
                content=json.dumps(
                    {
                        "meeting_decision_brief": (
                            "Inflation pressures were mixed and employment "
                            "growth moderated."
                        )
                    }
                ),
                response_id="brief-repair-good",
            ),
        ]
    )
    result = repair_one(
        meeting_row,
        initial_errors=("invalid_json", "empty_brief"),
        output_root=tmp_path,
        tokenizer=TinyTokenizer(),
        backend=backend,
        guard=ProviderIdentityGuard(),
        original_contract=original_contract,
        repair_contract=repair_contract,
        environment={},
        max_attempts=2,
    )
    assert result.accepted is True
    assert result.requests == 2
    assert len(backend.requests) == 2
    assert result.payload["attempt"] == "targeted_repair_2"
    assert len(list((tmp_path / "cache/accepted").glob("*/*.json"))) == 1
    rejected = list(
        (tmp_path / "cache/targeted_repair_rejected").glob("*/*.json")
    )
    assert len(rejected) == 1
    rejected_payload = json.loads(rejected[0].read_text(encoding="utf-8"))
    assert rejected_payload["provider_raw"]["content"] == ""


def test_targeted_sft_repair_uses_original_validator_and_cache_key(tmp_path):
    root = Path(__file__).resolve().parents[1]
    target_script = root / "jobs/generation/generate_chk4_sft_targets.py"
    repair_script = root / "jobs/generation/repair_chk4_sft_targets.py"
    original_contract = build_prompt_contract(
        sha256_text(target_script.read_text(encoding="utf-8"))
    )
    repair_contract = _target_repair_contract(
        sha256_text(repair_script.read_text(encoding="utf-8")),
        original_contract["contract_sha256"],
        2,
    )
    backend = SequenceBackend(
        [
            response(
                reasoning="Inflation was 9 percent.",
                response_id="target-repair-bad",
            ),
            response(
                reasoning=(
                    "Inflation risks remained elevated while activity and labor "
                    "conditions showed moderation, supporting a steady stance."
                ),
                response_id="target-repair-good",
            ),
        ]
    )
    result = repair_target_one(
        row(),
        output_root=tmp_path,
        tokenizer=TinyTokenizer(),
        backend=backend,
        guard=ProviderIdentityGuard(),
        original_contract=original_contract,
        repair_contract=repair_contract,
        environment={},
        max_attempts=2,
    )
    assert result.accepted is True
    assert result.requests == 2
    assert result.payload["attempt"] == "targeted_repair_2"
    assert len(list((tmp_path / "cache/accepted").glob("*/*.json"))) == 1
    rejected = list(
        (tmp_path / "cache/targeted_repair_rejected").glob("*/*/*.json")
    )
    assert len(rejected) == 1
    rejected_payload = json.loads(rejected[0].read_text(encoding="utf-8"))
    assert rejected_payload["provider_raw"]["reasoning_content"] == (
        "Inflation was 9 percent."
    )
