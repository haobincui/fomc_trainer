import json
from concurrent.futures import ThreadPoolExecutor
from datetime import date, datetime
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from open_r1.configs import GRPOScriptArguments
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v2 import (
    JudgeContextBudgetError,
    JudgeInfrastructureError,
    _TOKENIZER_CACHE,
    build_judge_request,
    get_judge_tokenizer,
    _judge_one,
    validate_judge_context_batch,
    grounded_analysis_reward_v2,
    numeric_grounding,
    validate_judge_evidence,
)
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v3 import (
    _judge_one_v3,
    build_judge_request_v3,
    grounded_analysis_reward_v3,
    numeric_grounding_v3,
    reasoning_efficiency,
)
from open_r1.trainer.rewards.reward_funcs.decision_reward_v2 import (
    decision_dense_reward_v2,
    parse_decision_json,
)
from open_r1.trainer.rewards.reward_register import get_reward_funcs


def _completion(answer: str, reasoning: str = "Evidence supports the conclusion."):
    return [{"content": f"{reasoning}\n</think>\n{answer}"}]


class _FixedJudgeTokenizer:
    def __init__(self, count: int = 64):
        self.count = count
        self.messages = None
        self.kwargs = None

    def apply_chat_template(self, messages, **kwargs):
        self.messages = messages
        self.kwargs = kwargs
        return list(range(self.count))

    def encode(self, text, add_special_tokens=False):
        assert add_special_tokens is False
        return list(range(self.count))


class _SequenceJudgeTokenizer(_FixedJudgeTokenizer):
    def __init__(self, counts):
        self.counts = iter(counts)
        super().__init__(0)

    def apply_chat_template(self, messages, **kwargs):
        self.count = next(self.counts)
        return super().apply_chat_template(messages, **kwargs)


def _tokenizer_dir(tmp_path):
    path = tmp_path / "Qwen3.5-9B"
    path.mkdir(exist_ok=True)
    return path


def test_numeric_grounding_reports_unsupported_values():
    score, unsupported = numeric_grounding(
        "Unemployment rose from 4.0 percent to 5.5 percent.",
        "unemployment_rate=4.0",
    )
    assert score == 0.5
    assert unsupported == [5.5]


def test_numeric_grounding_handles_compact_deepseek_text_and_ignores_ids():
    evidence = json.dumps(
        {
            "evidence": [
                {
                    "evidence_id": "ev-5a50937c943c0fc1f905a1cb",
                    "observation_date": "2012-07-01",
                    "value": "8.3",
                },
                {
                    "evidence_id": "ev-f6a791fe82163a8cec5bc003",
                    "observation_date": "2012-09-01",
                    "value": "7.8",
                },
            ]
        }
    )
    candidate = (
        "Theunemploymentratefellfrom8.3percentinJuly2012to7.8percentinSeptember2012."
        "Evidence:ev-5a50937c943c0fc1f905a1cb,ev-f6a791fe82163a8cec5bc003."
    )
    assert numeric_grounding(candidate, evidence) == (1.0, [])


def test_numeric_grounding_still_rejects_compact_unsupported_reasoning_number():
    score, unsupported = numeric_grounding(
        "<think>Draftvaluewas9.0.</think><answer>Reportedvaluewas4.0.</answer>",
        '{"value":"4.0"}',
    )
    assert score == 0.5
    assert unsupported == [9.0]


@patch("open_r1.trainer.rewards.reward_funcs.analysis_reward_v2.requests.post")
def test_judge_request_is_deterministic_json_schema(mock_post):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": json.dumps(
                        {
                            "data_fidelity": 4,
                            "trend_reasoning": 4,
                            "policy_relevance": 4,
                            "uncertainty_calibration": 4,
                            "fomc_style": 4,
                            "unsupported_claims": [],
                        }
                    )
                }
            }
        ]
    }
    mock_post.return_value = response
    evaluation, _, attempts = _judge_one(
        evidence="x=1",
        candidate="x is 1",
        url="http://judge/v1/chat/completions",
        model="Qwen3.5-9B",
        timeout=2,
        api_key=None,
        max_retries=3,
        backoff_seconds=0,
    )
    assert evaluation["data_fidelity"] == 4
    assert attempts == 1
    body = mock_post.call_args.kwargs["json"]
    assert body["temperature"] == 0
    assert body["chat_template_kwargs"] == {"enable_thinking": False}
    assert body["max_completion_tokens"] == 2048
    assert "max_tokens" not in body
    assert body["response_format"]["type"] == "json_schema"
    claim_schema = body["response_format"]["json_schema"]["schema"]["properties"][
        "unsupported_claims"
    ]
    assert claim_schema["maxItems"] == 8
    assert claim_schema["items"]["maxLength"] == 240


def _valid_judge_payload():
    return {
        "data_fidelity": 4,
        "trend_reasoning": 4,
        "policy_relevance": 4,
        "uncertainty_calibration": 4,
        "fomc_style": 4,
        "unsupported_claims": [],
    }


def _valid_judge_payload_v3(*, violations=None):
    return {
        "data_fidelity": 4,
        "trend_reasoning": 4,
        "policy_relevance": 4,
        "uncertainty_calibration": 4,
        "fomc_style": 4,
        "violations": [] if violations is None else violations,
    }


def _violation(section, kind, severity, quote):
    return {
        "section": section,
        "kind": kind,
        "severity": severity,
        "candidate_quote": quote,
        "explanation": "Deterministic test violation.",
    }


@pytest.mark.parametrize(
    "raw",
    [
        "prefix " + json.dumps(_valid_judge_payload()),
        json.dumps(_valid_judge_payload()) + " suffix",
        json.dumps({**_valid_judge_payload(), "data_fidelity": 4.0}),
        json.dumps({**_valid_judge_payload(), "data_fidelity": True}),
        json.dumps({**_valid_judge_payload(), "extra": 1}),
        json.dumps(
            {key: value for key, value in _valid_judge_payload().items() if key != "fomc_style"}
        ),
        json.dumps({**_valid_judge_payload(), "unsupported_claims": ["x"] * 9}),
        json.dumps({**_valid_judge_payload(), "unsupported_claims": ["x" * 241]}),
    ],
    ids=[
        "prefix",
        "suffix",
        "float",
        "bool",
        "extra",
        "missing",
        "too_many_claims",
        "claim_too_long",
    ],
)
@patch("open_r1.trainer.rewards.reward_funcs.analysis_reward_v2.requests.post")
def test_judge_response_contract_rejects_non_strict_payloads(mock_post, raw):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"choices": [{"message": {"content": raw}}]}
    mock_post.return_value = response
    with pytest.raises(JudgeInfrastructureError):
        _judge_one(
            evidence="x=1",
            candidate="x is 1",
            url="http://judge/v1/chat/completions",
            model="Qwen3.5-9B",
            timeout=2,
            api_key=None,
            max_retries=1,
            backoff_seconds=0,
        )


@patch("open_r1.trainer.rewards.reward_funcs.analysis_reward_v2.requests.post")
def test_judge_failure_is_not_converted_to_zero_reward(mock_post):
    mock_post.side_effect = OSError("offline")
    with pytest.raises(JudgeInfrastructureError):
        _judge_one(
            evidence="x=1",
            candidate="x is 1",
            url="http://judge/v1/chat/completions",
            model="Qwen3.5-9B",
            timeout=2,
            api_key=None,
            max_retries=2,
            backoff_seconds=0,
        )


@patch("open_r1.trainer.rewards.reward_funcs.analysis_reward_v2._judge_one")
def test_grounded_reward_reaches_one_for_fully_supported_output(mock_judge, tmp_path):
    mock_judge.return_value = (
        {
            "data_fidelity": 4.0,
            "trend_reasoning": 4.0,
            "policy_relevance": 4.0,
            "uncertainty_calibration": 4.0,
            "fomc_style": 4.0,
            "unsupported_claims": [],
        },
        "{}",
        1,
    )
    reward = grounded_analysis_reward_v2(
        [_completion("The reported value was 4.0 percent.")],
        ["reported_value=4.0"],
        meeting_date=["2020-01-01"],
        tokenizer_path=str(_tokenizer_dir(tmp_path)),
        judge_tokenizer=_FixedJudgeTokenizer(),
    )
    assert reward == [1.0]
    assert mock_judge.call_args.kwargs["evidence"] == "reported_value=4.0"
    assert mock_judge.call_args.kwargs["candidate"] == (
        "<think>\nEvidence supports the conclusion.\n</think>\n"
        "<answer>\nThe reported value was 4.0 percent.\n</answer>"
    )
    assert "2020-01-01" not in mock_judge.call_args.kwargs["evidence"]


@patch("open_r1.trainer.rewards.reward_funcs.analysis_reward_v2._judge_one")
def test_complete_think_and_answer_are_both_scored(mock_judge, tmp_path):
    mock_judge.return_value = (_valid_judge_payload(), "{}", 1)
    reward = grounded_analysis_reward_v2(
        [_completion("The reported value was 4.0 percent.", reasoning="Draft value was 9.0.")],
        ["reported_value=4.0"],
        tokenizer_path=str(_tokenizer_dir(tmp_path)),
        judge_tokenizer=_FixedJudgeTokenizer(),
    )
    candidate = mock_judge.call_args.kwargs["candidate"]
    assert candidate == (
        "<think>\nDraft value was 9.0.\n</think>\n"
        "<answer>\nThe reported value was 4.0 percent.\n</answer>"
    )
    assert reward == [0.25]


@pytest.mark.parametrize(
    "evidence",
    [
        '{"meeting_date":"2020-01-01","indicator":4.0}',
        '{"nested":{"reference_answer":"hold"}}',
        '{"meetingDate":"2020-01-01"}',
        "gold_label = cut",
        "current vote: hike",
        "target meeting minutes: unavailable",
        "current_actual_decision=hold",
        '{"sample_id":"chk1-analysis-2020-01-29-deadbeef","indicator":4.0}',
        '{"canonical_key":"2020-01-29::inflation","indicator":4.0}',
        '{"cutoff_ts":"2020-01-28T23:59:59Z","indicator":4.0}',
        '{"availability_upper_bound_ts":"2020-01-28T05:00:00Z","indicator":4.0}',
        '{"source_id":"canonical-loo:2020-01-29:inflation","indicator":4.0}',
    ],
)
def test_judge_evidence_rejects_target_reference_markers(evidence):
    with pytest.raises(ValueError, match="forbidden target/reference"):
        validate_judge_evidence(evidence)


def test_judge_evidence_allows_observation_dates_and_prior_policy_facts():
    evidence = (
        "observation_date=2019-12-31; prior_policy_rate=1.75; "
        "previous decision was a 25 basis point cut"
    )
    assert validate_judge_evidence(evidence) == evidence


@patch("open_r1.trainer.rewards.reward_funcs.analysis_reward_v2._judge_one")
def test_v2_retry_contract_ignores_environment_drift(mock_judge, monkeypatch, tmp_path):
    mock_judge.return_value = (_valid_judge_payload(), "{}", 1)
    monkeypatch.setenv("OPEN_R1_JUDGE_MAX_RETRIES", "5")
    monkeypatch.setenv("OPEN_R1_JUDGE_BACKOFF_SECONDS", "9")
    grounded_analysis_reward_v2(
        [_completion("The reported value was 4.0 percent.")],
        ["reported_value=4.0"],
        max_retries=3,
        backoff_seconds=1.0,
        tokenizer_path=str(_tokenizer_dir(tmp_path)),
        judge_tokenizer=_FixedJudgeTokenizer(),
    )
    assert mock_judge.call_args.kwargs["max_retries"] == 3
    assert mock_judge.call_args.kwargs["backoff_seconds"] == 1.0


@pytest.mark.parametrize(
    ("max_retries", "backoff_seconds"),
    [(0, 1.0), (6, 1.0), (True, 1.0), (3, -0.1), (3, 10.1), (3, float("inf"))],
)
def test_v2_retry_contract_rejects_out_of_range_values(
    max_retries, backoff_seconds, tmp_path
):
    with pytest.raises(ValueError, match="judge (max_retries|backoff_seconds)"):
        grounded_analysis_reward_v2(
            [_completion("The reported value was 4.0 percent.")],
            ["reported_value=4.0"],
            max_retries=max_retries,
            backoff_seconds=backoff_seconds,
            tokenizer_path=str(_tokenizer_dir(tmp_path)),
            judge_tokenizer=_FixedJudgeTokenizer(),
        )


def test_canonical_body_preserves_escaping_unicode_and_render_contract():
    body = build_judge_request(
        evidence='quote="x"\\line\n货币',
        candidate="café\n路径",
        model="Qwen3.5-9B",
    )
    payload = json.loads(body["messages"][1]["content"])
    assert payload == {
        "evidence": 'quote="x"\\line\n货币',
        "candidate": "café\n路径",
    }
    tokenizer = _FixedJudgeTokenizer(123)
    assert validate_judge_context_batch(
        [body], tokenizer=tokenizer, max_model_len=8192, max_completion_tokens=2048
    ) == [123]
    assert tokenizer.kwargs == {
        "tokenize": True,
        "add_generation_prompt": True,
        "return_dict": False,
        "enable_thinking": False,
    }


@pytest.mark.parametrize(("prompt_tokens", "passes"), [(6144, True), (6145, False)])
def test_exact_judge_context_boundary(prompt_tokens, passes):
    body = build_judge_request(evidence="x", candidate="y", model="Qwen3.5-9B")
    if passes:
        assert validate_judge_context_batch(
            [body],
            tokenizer=_FixedJudgeTokenizer(prompt_tokens),
            max_model_len=8192,
            max_completion_tokens=2048,
        ) == [prompt_tokens]
    else:
        with pytest.raises(JudgeContextBudgetError, match="model limit is 8192"):
            validate_judge_context_batch(
                [body],
                tokenizer=_FixedJudgeTokenizer(prompt_tokens),
                max_model_len=8192,
                max_completion_tokens=2048,
            )


@patch("open_r1.trainer.rewards.reward_funcs.analysis_reward_v2._judge_one")
def test_entire_batch_context_is_checked_before_zero_http(mock_judge, tmp_path):
    with pytest.raises(JudgeContextBudgetError, match="request 1"):
        grounded_analysis_reward_v2(
            [_completion("first"), _completion("second")],
            ["indicator=1", "indicator=2"],
            tokenizer_path=str(_tokenizer_dir(tmp_path)),
            judge_tokenizer=_SequenceJudgeTokenizer([6144, 6145]),
        )
    mock_judge.assert_not_called()


def test_process_tokenizer_singleton_loads_once(monkeypatch, tmp_path):
    tokenizer_path = _tokenizer_dir(tmp_path)
    loaded = []
    sentinel = _FixedJudgeTokenizer()
    _TOKENIZER_CACHE.clear()

    def fake_load(path):
        loaded.append(path)
        return sentinel

    monkeypatch.setattr(
        "open_r1.trainer.rewards.reward_funcs.analysis_reward_v2._load_tokenizer_uncached",
        fake_load,
    )
    with ThreadPoolExecutor(max_workers=4) as executor:
        observed = list(executor.map(lambda _: get_judge_tokenizer(tokenizer_path), range(8)))
    assert observed == [sentinel] * 8
    assert loaded == [tokenizer_path.resolve()]


def test_empty_reward_batch_does_not_load_tokenizer(monkeypatch):
    load = Mock(side_effect=AssertionError("must not load"))
    monkeypatch.setattr(
        "open_r1.trainer.rewards.reward_funcs.analysis_reward_v2.get_judge_tokenizer",
        load,
    )
    assert grounded_analysis_reward_v2([], []) == []
    load.assert_not_called()


@patch("open_r1.trainer.rewards.reward_funcs.analysis_reward_v2._judge_one")
def test_reward_log_records_count_without_request_text(mock_judge, tmp_path):
    mock_judge.return_value = (_valid_judge_payload(), "{}", 1)
    reward_log = tmp_path / "reward.jsonl"
    grounded_analysis_reward_v2(
        [_completion("SENSITIVE_CANDIDATE")],
        ["safe_indicator=SENSITIVE_EVIDENCE"],
        tokenizer_path=str(_tokenizer_dir(tmp_path)),
        judge_tokenizer=_FixedJudgeTokenizer(321),
        save_path=str(reward_log),
    )
    record = json.loads(reward_log.read_text(encoding="utf-8"))
    assert record["judge_prompt_tokens"] == 321
    serialized = json.dumps(record)
    assert "SENSITIVE_CANDIDATE" not in serialized
    assert "SENSITIVE_EVIDENCE" not in serialized


@patch("open_r1.trainer.rewards.reward_funcs.analysis_reward_v2._judge_one")
@pytest.mark.parametrize(
    ("meeting_date", "expected"),
    [
        (date(2020, 1, 29), "2020-01-29"),
        (datetime(2020, 1, 29, 0, 0), "2020-01-29"),
    ],
)
def test_reward_log_normalizes_arrow_date_metadata(
    mock_judge, tmp_path, meeting_date, expected
):
    mock_judge.return_value = (_valid_judge_payload(), "{}", 1)
    reward_log = tmp_path / "reward.jsonl"
    grounded_analysis_reward_v2(
        [_completion("Candidate")],
        ["safe_indicator=Evidence"],
        meeting_date=[meeting_date],
        tokenizer_path=str(_tokenizer_dir(tmp_path)),
        judge_tokenizer=_FixedJudgeTokenizer(321),
        save_path=str(reward_log),
    )
    assert json.loads(reward_log.read_text(encoding="utf-8"))["meeting_date"] == expected


def test_decision_parser_and_exact_reward():
    answer = json.dumps({"direction": "cut", "magnitude_bp": 50})
    assert parse_decision_json(_completion(answer)[0]["content"]) == ("cut", 50)
    assert decision_dense_reward_v2(
        [_completion(answer)], direction=["cut"], magnitude_bp=[50]
    ) == [1.0]


def test_decision_wrong_magnitude_receives_dense_partial_credit():
    answer = json.dumps({"direction": "hike", "magnitude_bp": 25})
    reward = decision_dense_reward_v2(
        [_completion(answer)], direction=["hike"], magnitude_bp=[75]
    )[0]
    assert reward == pytest.approx(0.65)


def test_invalid_decision_json_receives_zero():
    assert decision_dense_reward_v2(
        [_completion('{"direction":"hold","magnitude_bp":25}')],
        direction=["hold"],
        magnitude_bp=[0],
    ) == [0.0]


def test_invalid_non_hold_target_magnitude_fails_closed():
    with pytest.raises(ValueError, match="invalid decision target"):
        decision_dense_reward_v2(
            [_completion('{"direction":"cut","magnitude_bp":25}')],
            direction=["cut"],
            magnitude_bp=[0],
        )


def test_decision_reward_records_are_persisted(tmp_path):
    save_path = tmp_path / "reward.jsonl"
    valid_answer = json.dumps({"direction": "hold", "magnitude_bp": 0})
    rewards = decision_dense_reward_v2(
        [_completion(valid_answer), _completion("not-json")],
        direction=["hold", "cut"],
        magnitude_bp=[0, 25],
        meeting_date=["2020-01-01", "2020-02-01"],
        save_path=str(save_path),
    )
    assert rewards == [1.0, 0.0]
    records = [json.loads(line) for line in save_path.read_text().splitlines()]
    assert records[0]["prediction"] == {"direction": "hold", "magnitude_bp": 0}
    assert records[0]["exact"] is True
    assert records[1]["prediction"] is None
    assert records[1]["format_valid"] is False


def test_registry_injects_decision_reward_save_path(tmp_path):
    args = GRPOScriptArguments(
        dataset_name="dummy",
        reward_funcs=["decision_dense_v2"],
        save_reward=True,
    )
    reward_func = get_reward_funcs(
        args,
        SimpleNamespace(output_dir=str(tmp_path)),
    )[0]
    answer = json.dumps({"direction": "cut", "magnitude_bp": 25})
    assert reward_func(
        completions=[_completion(answer)],
        direction=["cut"],
        magnitude_bp=[25],
    ) == [1.0]
    assert (tmp_path / "reward.jsonl").is_file()


def test_v2_rewards_are_registered():
    args = GRPOScriptArguments(
        dataset_name="dummy",
        reward_funcs=["grounded_analysis_v2", "decision_dense_v2"],
    )
    funcs = get_reward_funcs(args)
    assert [func.__name__ for func in funcs] == [
        "grounded_analysis_reward_v2",
        "decision_dense_reward_v2",
    ]


def test_registry_pins_v2_retry_contract_in_partial():
    args = GRPOScriptArguments(
        dataset_name="dummy",
        reward_funcs=["grounded_analysis_v2"],
        judge_max_retries=3,
        judge_backoff_seconds=1.0,
    )
    reward_func = get_reward_funcs(args)[0]
    assert reward_func.keywords["max_retries"] == 3
    assert reward_func.keywords["backoff_seconds"] == 1.0


def test_v3_numeric_grounding_handles_compact_quarter_dates_and_basis_points():
    evidence = json.dumps(
        {
            "atomic_topic": "PolicyRate",
            "evidence": [
                {
                    "metric": "Policy Rate",
                    "series_id": "RATE",
                    "units": "Percent",
                    "observation_date": "2019-01-01",
                    "value": "2.0",
                },
                {
                    "metric": "Policy Rate",
                    "series_id": "RATE",
                    "units": "Percent",
                    "observation_date": "2019-03-01",
                    "value": "2.25",
                },
            ],
        }
    )
    candidate = "InQ12019,thepolicyraterose25basispointsfrom2.0percentto2.25percent."
    assert numeric_grounding_v3(candidate, evidence) == (1.0, [])


def test_v3_numeric_grounding_rejects_unsupported_answer_value():
    score, unsupported = numeric_grounding_v3(
        "Thefinalvaluewas9.0percent.",
        json.dumps({"evidence": [{"metric": "Rate", "value": "4.0"}]}),
    )
    assert score == 0.0
    assert unsupported == [9.0]


def test_v3_reasoning_efficiency_uses_token_length_and_repetition():
    efficient, count, repetition = reasoning_efficiency(
        "concise", _FixedJudgeTokenizer(64)
    )
    assert efficient == 1.0
    assert count == 64
    assert repetition == 0.0
    long_score, long_count, _ = reasoning_efficiency(
        "long", _FixedJudgeTokenizer(1536)
    )
    assert long_count == 1536
    assert long_score == 0.0


def test_v3_judge_request_declares_plain_text_answer_contract():
    body = build_judge_request_v3(
        evidence='{"value":4.0}',
        candidate="<think>x</think><answer>y</answer>",
        model="Qwen3.5-9B",
    )
    system = body["messages"][0]["content"]
    assert "plain text, not JSON" in system
    schema = body["response_format"]["json_schema"]["schema"]
    assert schema["required"][-1] == "violations"
    violation = schema["properties"]["violations"]["items"]
    assert violation["additionalProperties"] is False
    assert set(violation["properties"]["section"]["enum"]) == {
        "think",
        "answer",
    }


@patch("open_r1.trainer.rewards.reward_funcs.analysis_reward_v3.requests.post")
def test_v3_judge_response_contract_is_strict(mock_post):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [{"message": {"content": json.dumps(_valid_judge_payload_v3())}}]
    }
    mock_post.return_value = response
    evaluation, _, attempts = _judge_one_v3(
        evidence="x=1",
        candidate="x is 1",
        url="http://judge/v1/chat/completions",
        model="Qwen3.5-9B",
        timeout=2,
        api_key=None,
        max_retries=1,
        backoff_seconds=0,
    )
    assert evaluation["violations"] == []
    assert attempts == 1


@patch("open_r1.trainer.rewards.reward_funcs.analysis_reward_v3.requests.post")
def test_v3_judge_accepts_complete_json_code_fence(mock_post):
    response = Mock()
    response.raise_for_status.return_value = None
    raw = "```json\n" + json.dumps(_valid_judge_payload_v3()) + "\n```"
    response.json.return_value = {"choices": [{"message": {"content": raw}}]}
    mock_post.return_value = response

    evaluation, returned_raw, attempts = _judge_one_v3(
        evidence="x=1",
        candidate="x is 1",
        url="http://judge/v1/chat/completions",
        model="Qwen3.5-9B",
        timeout=2,
        api_key=None,
        max_retries=1,
        backoff_seconds=0,
    )

    assert evaluation == _valid_judge_payload_v3()
    assert returned_raw == raw
    assert attempts == 1


@patch("open_r1.trainer.rewards.reward_funcs.analysis_reward_v3.requests.post")
def test_v3_judge_uses_compact_request_after_malformed_output(mock_post):
    malformed = Mock()
    malformed.raise_for_status.return_value = None
    malformed.json.return_value = {
        "choices": [{"message": {"content": '{"data_fidelity":4'}}]
    }
    valid = Mock()
    valid.raise_for_status.return_value = None
    valid.json.return_value = {
        "choices": [{"message": {"content": json.dumps(_valid_judge_payload_v3())}}]
    }
    mock_post.side_effect = [malformed, valid]

    evaluation, _, attempts = _judge_one_v3(
        evidence="x=1",
        candidate="x is 1",
        url="http://judge/v1/chat/completions",
        model="Qwen3.5-9B",
        timeout=2,
        api_key=None,
        max_retries=2,
        backoff_seconds=0,
        max_completion_tokens=4096,
    )

    retry_body = mock_post.call_args_list[1].kwargs["json"]
    assert evaluation == _valid_judge_payload_v3()
    assert attempts == 2
    assert retry_body["max_completion_tokens"] == 1024
    assert "at most two highest-severity violations" in retry_body["messages"][0]["content"]


@patch("open_r1.trainer.rewards.reward_funcs.analysis_reward_v3.requests.post")
def test_v3_judge_still_rejects_prose_around_json(mock_post):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": "prefix " + json.dumps(_valid_judge_payload_v3())
                }
            }
        ]
    }
    mock_post.return_value = response

    with pytest.raises(JudgeInfrastructureError):
        _judge_one_v3(
            evidence="x=1",
            candidate="x is 1",
            url="http://judge/v1/chat/completions",
            model="Qwen3.5-9B",
            timeout=2,
            api_key=None,
            max_retries=1,
            backoff_seconds=0,
        )


@patch("open_r1.trainer.rewards.reward_funcs.analysis_reward_v3._judge_one_v3")
def test_v3_think_meta_numbers_do_not_cap_supported_answer(mock_judge, tmp_path):
    mock_judge.return_value = (_valid_judge_payload_v3(), "{}", 1)
    reward = grounded_analysis_reward_v3(
        [_completion("The reported value was 4.0 percent.", reasoning="Plan 3 schema fields.")],
        [json.dumps({"evidence": [{"metric": "Rate", "value": "4.0"}]})],
        tokenizer_path=str(_tokenizer_dir(tmp_path)),
        judge_tokenizer=_FixedJudgeTokenizer(64),
    )
    assert reward == [1.0]
    candidate = mock_judge.call_args.kwargs["candidate"]
    assert "<think>\nPlan 3 schema fields.\n</think>" in candidate
    assert "<answer>\nThe reported value was 4.0 percent.\n</answer>" in candidate


@patch("open_r1.trainer.rewards.reward_funcs.analysis_reward_v3._judge_one_v3")
def test_v3_optional_opening_think_tag_uses_all_trailing_text_as_answer(
    mock_judge, tmp_path
):
    mock_judge.return_value = (_valid_judge_payload_v3(), "{}", 1)
    trailing_answer = "First answer line.\nSecond answer line."
    reward_log = tmp_path / "reward.jsonl"
    reward = grounded_analysis_reward_v3(
        [[{"content": f"<think>Grounded reasoning.</think>{trailing_answer}"}]],
        [json.dumps({"evidence": [{"metric": "Rate", "value": "4.0"}]})],
        save_path=str(reward_log),
        tokenizer_path=str(_tokenizer_dir(tmp_path)),
        judge_tokenizer=_FixedJudgeTokenizer(64),
    )

    assert reward == [1.0]
    candidate = mock_judge.call_args.kwargs["candidate"]
    assert candidate == (
        "<think>\nGrounded reasoning.\n</think>\n"
        f"<answer>\n{trailing_answer}\n</answer>"
    )
    contract = json.loads(reward_log.read_text(encoding="utf-8"))["contract"]
    assert contract["parsed_format"] == "deepseek_think_completion"
    assert contract["closing_think_count"] == 1
    assert contract["answer_chars_after_boundary"] == len(trailing_answer)
    assert contract["completion_chars"] == len(
        f"<think>Grounded reasoning.</think>{trailing_answer}"
    )
    assert contract["plain_answer_fallback"] is False


@patch("open_r1.trainer.rewards.reward_funcs.analysis_reward_v3._judge_one_v3")
def test_v3_plain_text_completion_without_boundary_fails_closed_before_judge(
    mock_judge, tmp_path
):
    reward_log = tmp_path / "reward.jsonl"
    plain_answer = "The supported value was 4.0 percent."
    reward = grounded_analysis_reward_v3(
        [[{"content": plain_answer}]],
        [json.dumps({"evidence": [{"metric": "Rate", "value": "4.0"}]})],
        save_path=str(reward_log),
        tokenizer_path=str(_tokenizer_dir(tmp_path)),
        judge_tokenizer=_FixedJudgeTokenizer(64),
    )

    assert reward == [0.0]
    mock_judge.assert_not_called()
    record = json.loads(reward_log.read_text(encoding="utf-8"))
    assert record["components"]["structure_score"] == 0.0
    assert record["contract"]["plain_answer_fallback"] is False
    assert record["contract"]["accepted_for_judge"] is False
    assert record["contract"]["rejection_reason"] == "missing_think_boundary"
    assert record["contract"]["answer_nonempty"] is False
    assert record["contract"]["reasoning_nonempty"] is True
    assert record["judge_attempts"] == 0
    assert record["judge_prompt_tokens"] == 0
    assert record["judge_response_sha256"] is None
    assert record["reward_before_penalty"] == 0.0


@patch("open_r1.trainer.rewards.reward_funcs.analysis_reward_v3._judge_one_v3")
def test_v3_boundary_without_trailing_answer_fails_closed_before_judge(
    mock_judge, tmp_path
):
    reward_log = tmp_path / "reward.jsonl"
    reward = grounded_analysis_reward_v3(
        [[{"content": "Grounded reasoning.\n</think>\n"}]],
        [json.dumps({"evidence": [{"metric": "Rate", "value": "4.0"}]})],
        save_path=str(reward_log),
        tokenizer_path=str(_tokenizer_dir(tmp_path)),
        judge_tokenizer=_FixedJudgeTokenizer(64),
    )

    assert reward == [0.0]
    mock_judge.assert_not_called()
    contract = json.loads(reward_log.read_text(encoding="utf-8"))["contract"]
    assert contract["closing_think_count"] == 1
    assert contract["answer_chars_after_boundary"] == 0
    assert contract["accepted_for_judge"] is False
    assert contract["rejection_reason"] == "empty_answer_after_boundary"


@patch("open_r1.trainer.rewards.reward_funcs.analysis_reward_v3._judge_one_v3")
def test_v3_answer_major_error_penalty_exceeds_think_minor_error(
    mock_judge, tmp_path
):
    mock_judge.side_effect = [
        (
            _valid_judge_payload_v3(
                violations=[_violation("think", "factual", "minor", "Draft claim")]
            ),
            "{}",
            1,
        ),
        (
            _valid_judge_payload_v3(
                violations=[_violation("answer", "factual", "major", "Major claim")]
            ),
            "{}",
            1,
        ),
    ]
    rewards = grounded_analysis_reward_v3(
        [
            _completion("Supported answer.", reasoning="Draft claim."),
            _completion("Major claim.", reasoning="Supported thought."),
        ],
        [json.dumps({"evidence": [{"value": "4.0"}]})] * 2,
        tokenizer_path=str(_tokenizer_dir(tmp_path)),
        judge_tokenizer=_FixedJudgeTokenizer(64),
    )
    assert rewards[0] == pytest.approx(0.98)
    assert rewards[1] == pytest.approx(0.80)


@patch("open_r1.trainer.rewards.reward_funcs.analysis_reward_v3._judge_one_v3")
def test_v3_format_violation_never_adds_factual_penalty(mock_judge, tmp_path):
    mock_judge.return_value = (
        _valid_judge_payload_v3(
            violations=[_violation("think", "format", "major", "JSON planning")]
        ),
        "{}",
        1,
    )
    reward = grounded_analysis_reward_v3(
        [_completion("Supported answer.", reasoning="JSON planning.")],
        [json.dumps({"evidence": [{"value": "4.0"}]})],
        tokenizer_path=str(_tokenizer_dir(tmp_path)),
        judge_tokenizer=_FixedJudgeTokenizer(64),
    )
    assert reward == [1.0]


@patch("open_r1.trainer.rewards.reward_funcs.analysis_reward_v3._judge_one_v3")
def test_v3_unmatched_quote_is_logged_but_not_penalized(mock_judge, tmp_path):
    mock_judge.return_value = (
        _valid_judge_payload_v3(
            violations=[_violation("answer", "factual", "major", "Absent quote")]
        ),
        "{}",
        1,
    )
    reward_log = tmp_path / "reward.jsonl"
    reward = grounded_analysis_reward_v3(
        [_completion("Supported answer.")],
        [json.dumps({"evidence": [{"value": "4.0"}]})],
        save_path=str(reward_log),
        tokenizer_path=str(_tokenizer_dir(tmp_path)),
        judge_tokenizer=_FixedJudgeTokenizer(64),
    )
    assert reward == [1.0]
    record = json.loads(reward_log.read_text())
    assert record["validated_violations"] == []
    assert record["invalid_violations"][0]["invalid_reason"] == "quote_not_found"


@patch("open_r1.trainer.rewards.reward_funcs.analysis_reward_v3._judge_one_v3")
def test_v3_empty_answer_and_quote_validated_target_leakage_are_zero(
    mock_judge, tmp_path
):
    mock_judge.return_value = (
        _valid_judge_payload_v3(
            violations=[
                _violation(
                    "answer",
                    "target_leakage",
                    "major",
                    "The target meeting voted to cut.",
                )
            ]
        ),
        "{}",
        1,
    )
    rewards = grounded_analysis_reward_v3(
        [
            [{"content": ""}],
            _completion("The target meeting voted to cut."),
        ],
        [json.dumps({"evidence": [{"value": "4.0"}]})] * 2,
        tokenizer_path=str(_tokenizer_dir(tmp_path)),
        judge_tokenizer=_FixedJudgeTokenizer(64),
    )
    assert rewards == [0.0, 0.0]
    assert mock_judge.call_count == 1


@patch("open_r1.trainer.rewards.reward_funcs.analysis_reward_v3._judge_one_v3")
def test_v3_reward_log_excludes_raw_candidate_and_evidence(mock_judge, tmp_path):
    mock_judge.return_value = (_valid_judge_payload_v3(), "{}", 1)
    reward_log = tmp_path / "reward.jsonl"
    grounded_analysis_reward_v3(
        [_completion("SENSITIVE_CANDIDATE")],
        [json.dumps({"evidence": [{"metric": "SENSITIVE_EVIDENCE", "value": 4.0}]})],
        save_path=str(reward_log),
        tokenizer_path=str(_tokenizer_dir(tmp_path)),
        judge_tokenizer=_FixedJudgeTokenizer(64),
    )
    serialized = reward_log.read_text()
    assert "SENSITIVE_CANDIDATE" not in serialized
    assert "SENSITIVE_EVIDENCE" not in serialized


def test_v3_reward_is_registered_without_changing_v2():
    args = GRPOScriptArguments(
        dataset_name="dummy",
        reward_funcs=["grounded_analysis_v2", "grounded_analysis_v3"],
    )
    funcs = get_reward_funcs(args)
    assert [func.__name__ for func in funcs] == [
        "grounded_analysis_reward_v2",
        "grounded_analysis_reward_v3",
    ]
