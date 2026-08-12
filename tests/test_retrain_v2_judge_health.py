from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from jobs.retrain_v2.judge_health import (
    _judge_contract_from_config,
    _verify_loaded_model_root,
    check_judge,
)


def _valid_evaluation():
    return {
        "data_fidelity": 4,
        "trend_reasoning": 4,
        "policy_relevance": 4,
        "uncertainty_calibration": 4,
        "fomc_style": 4,
        "unsupported_claims": [],
    }


class _FakeTokenizer:
    def __init__(self, count=37):
        self.count = count

    def apply_chat_template(self, _messages, **_kwargs):
        return list(range(self.count))


def _tokenize_response(count=37, *, tokens=None, max_model_len=8192):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "count": count,
        "tokens": list(range(count)) if tokens is None else tokens,
        "max_model_len": max_model_len,
    }
    return response


@patch("jobs.retrain_v2.judge_health._judge_one")
@patch("jobs.retrain_v2.judge_health.requests.post")
@patch("jobs.retrain_v2.judge_health.requests.get")
def test_judge_health_checks_served_model_and_strict_request(
    mock_get, mock_post, mock_judge
):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": [{"id": "Qwen3.5-9B", "max_model_len": 8192}]
    }
    mock_get.return_value = response
    mock_post.return_value = _tokenize_response()
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

    result = check_judge(
        url="http://127.0.0.1:8000/v1/chat/completions",
        model="Qwen3.5-9B",
        timeout=5,
        tokenizer=_FakeTokenizer(),
    )

    assert result["status"] == "ready"
    assert result["tokenizer_parity"] is True
    assert mock_judge.call_args.kwargs["max_retries"] == 1
    assert mock_post.call_args.args[0].endswith("/tokenize")
    assert mock_post.call_args.kwargs["json"]["chat_template_kwargs"] == {
        "enable_thinking": False
    }


@patch("jobs.retrain_v2.judge_health.requests.get")
def test_judge_health_fails_when_wrong_model_is_served(mock_get):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {"data": [{"id": "wrong-model"}]}
    mock_get.return_value = response

    with pytest.raises(RuntimeError, match="exactly one model card"):
        check_judge(
            url="http://127.0.0.1:8000/v1/chat/completions",
            model="Qwen3.5-9B",
            timeout=5,
        )


def test_loaded_weight_root_attestation_accepts_exact_canonical_path(tmp_path: Path):
    model = tmp_path / "Qwen3.5-9B"
    model.mkdir()
    assert _verify_loaded_model_root(str(model), expected=model) == model.resolve()


def test_loaded_weight_root_attestation_rejects_wrong_missing_or_symlink(tmp_path: Path):
    expected = tmp_path / "expected"
    expected.mkdir()
    wrong = tmp_path / "wrong"
    wrong.mkdir()
    with pytest.raises(RuntimeError, match="disagrees"):
        _verify_loaded_model_root(str(wrong), expected=expected)
    with pytest.raises(RuntimeError, match="missing"):
        _verify_loaded_model_root(str(tmp_path / "missing"), expected=expected)
    link = tmp_path / "link"
    link.symlink_to(expected, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink"):
        _verify_loaded_model_root(str(link), expected=expected)


@patch("jobs.retrain_v2.judge_health._judge_one")
@patch("jobs.retrain_v2.judge_health.requests.post")
@patch("jobs.retrain_v2.judge_health.requests.get")
def test_check_judge_attests_model_card_root(
    mock_get, mock_post, mock_judge, tmp_path: Path
):
    model = tmp_path / "Qwen3.5-9B"
    model.mkdir()
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": [
            {
                "id": "Qwen3.5-9B",
                "root": str(model),
                "max_model_len": 8192,
            }
        ]
    }
    mock_get.return_value = response
    mock_post.return_value = _tokenize_response()
    mock_judge.return_value = (_valid_evaluation(), "{}", 1)
    result = check_judge(
        url="http://127.0.0.1:8000/v1/chat/completions",
        model="Qwen3.5-9B",
        timeout=5,
        expected_model_root=model,
        tokenizer=_FakeTokenizer(),
    )
    assert result["weight_attested"] is True
    assert result["loaded_model_root"] == str(model.resolve())


def test_judge_contract_comes_from_immutable_config(tmp_path: Path, monkeypatch):
    tokenizer = tmp_path / "Qwen3.5-9B"
    tokenizer.mkdir()
    config = tmp_path / "chk2.yaml"
    config.write_text(
        "judge_url: http://127.0.0.1:8000/v1/chat/completions\n"
        "judge_model: Qwen3.5-9B\n"
        "judge_timeout: 180\n"
        f"judge_tokenizer_path: {tokenizer}\n"
        "judge_max_model_len: 8192\n"
        "judge_max_completion_tokens: 2048\n",
        encoding="utf-8",
    )
    monkeypatch.delenv("OPEN_R1_JUDGE_URL", raising=False)
    monkeypatch.delenv("OPEN_R1_JUDGE_MODEL", raising=False)
    assert _judge_contract_from_config(config) == (
        "http://127.0.0.1:8000/v1/chat/completions",
        "Qwen3.5-9B",
        180,
        tokenizer,
        8192,
        2048,
    )


def test_judge_contract_rejects_runtime_endpoint_drift(tmp_path: Path, monkeypatch):
    tokenizer = tmp_path / "Qwen3.5-9B"
    tokenizer.mkdir()
    config = tmp_path / "chk2.yaml"
    config.write_text(
        "judge_url: http://127.0.0.1:8000/v1/chat/completions\n"
        "judge_model: Qwen3.5-9B\n"
        "judge_timeout: 180\n"
        f"judge_tokenizer_path: {tokenizer}\n"
        "judge_max_model_len: 8192\n"
        "judge_max_completion_tokens: 2048\n",
        encoding="utf-8",
    )
    monkeypatch.setenv(
        "OPEN_R1_JUDGE_URL", "http://127.0.0.1:18080/v1/chat/completions"
    )
    with pytest.raises(RuntimeError, match="disagrees with immutable config"):
        _judge_contract_from_config(config)


@pytest.mark.parametrize("served_len", [None, 8191, 8193])
@patch("jobs.retrain_v2.judge_health.requests.get")
def test_judge_health_rejects_model_len_drift(mock_get, served_len):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": [{"id": "Qwen3.5-9B", "max_model_len": served_len}]
    }
    mock_get.return_value = response
    with pytest.raises(RuntimeError, match="max_model_len"):
        check_judge(
            url="http://127.0.0.1:8000/v1/chat/completions",
            model="Qwen3.5-9B",
            timeout=5,
            tokenizer=_FakeTokenizer(),
        )


@patch("jobs.retrain_v2.judge_health.requests.post")
@patch("jobs.retrain_v2.judge_health.requests.get")
def test_judge_health_rejects_local_server_tokenizer_mismatch(mock_get, mock_post):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": [{"id": "Qwen3.5-9B", "max_model_len": 8192}]
    }
    mock_get.return_value = response
    mock_post.return_value = _tokenize_response(38)
    with pytest.raises(RuntimeError, match="tokenizer parity mismatch"):
        check_judge(
            url="http://127.0.0.1:8000/v1/chat/completions",
            model="Qwen3.5-9B",
            timeout=5,
            tokenizer=_FakeTokenizer(37),
        )


@patch("jobs.retrain_v2.judge_health.requests.post")
@patch("jobs.retrain_v2.judge_health.requests.get")
def test_judge_health_rejects_same_count_but_different_token_ids(mock_get, mock_post):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": [{"id": "Qwen3.5-9B", "max_model_len": 8192}]
    }
    mock_get.return_value = response
    mock_post.return_value = _tokenize_response(
        37, tokens=[*range(36), 999]
    )
    with pytest.raises(RuntimeError, match="tokenizer parity mismatch"):
        check_judge(
            url="http://127.0.0.1:8000/v1/chat/completions",
            model="Qwen3.5-9B",
            timeout=5,
            tokenizer=_FakeTokenizer(37),
        )


@pytest.mark.parametrize("tokenize_len", [None, 8191, 8193])
@patch("jobs.retrain_v2.judge_health.requests.post")
@patch("jobs.retrain_v2.judge_health.requests.get")
def test_judge_health_rejects_tokenize_model_len_drift(
    mock_get, mock_post, tokenize_len
):
    response = Mock()
    response.raise_for_status.return_value = None
    response.json.return_value = {
        "data": [{"id": "Qwen3.5-9B", "max_model_len": 8192}]
    }
    mock_get.return_value = response
    mock_post.return_value = _tokenize_response(max_model_len=tokenize_len)
    with pytest.raises(RuntimeError, match="/tokenize max_model_len"):
        check_judge(
            url="http://127.0.0.1:8000/v1/chat/completions",
            model="Qwen3.5-9B",
            timeout=5,
            tokenizer=_FakeTokenizer(),
        )
