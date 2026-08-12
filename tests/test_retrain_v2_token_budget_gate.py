from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobs.retrain_v2.token_budget_gate import (
    TokenBudgetError,
    audit_chk2_judge_baseline,
    audit_dataset,
    load_stage_contract,
    main as token_gate_main,
)
from open_r1.provenance import sha256_file
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v2 import (
    JudgeContextBudgetError,
)


class _WhitespaceTokenizer:
    eos_token = " <eos>"
    chat_template = "fake"

    @staticmethod
    def _ids(text: str) -> list[int]:
        return list(range(len(text.split())))

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        return_dict=False,
        **_kwargs,
    ):
        text = " ".join(f"{item['role']} {item['content']}" for item in messages)
        if add_generation_prompt:
            text += " assistant "
        return self._ids(text) if tokenize else text

    def __call__(self, *, text: str):
        return {"input_ids": self._ids(text)}


class _FixedJudgeTokenizer:
    def __init__(self, count: int):
        self.count = count

    def apply_chat_template(self, _messages, **_kwargs):
        return list(range(self.count))


def _write_splits(path: Path, row: dict) -> None:
    path.mkdir(parents=True)
    for name in ("train.jsonl", "eval.jsonl"):
        (path / name).write_text(json.dumps(row) + "\n", encoding="utf-8")


def test_grpo_gate_counts_system_and_user_chat_prompt(tmp_path: Path) -> None:
    dataset = tmp_path / "grpo"
    _write_splits(dataset, {"prompt": "one two three"})
    result = audit_dataset(
        dataset_path=dataset,
        tokenizer=_WhitespaceTokenizer(),
        config={
            "_stage_kind": "analysis_grpo",
            "dataset_prompt_column": "prompt",
            "system_prompt": "system words",
            "chat_template_kwargs": {"truncation": False},
        },
        budget={"prompt": 8, "completion": 4, "overflow_policy": "error"},
    )
    assert result["splits"]["train"]["max_prompt_tokens"] == 8
    assert result["rows"][0]["completion_tokens"] is None


def test_grpo_gate_counts_runtime_user_prompt_suffix(tmp_path: Path) -> None:
    dataset = tmp_path / "grpo"
    _write_splits(dataset, {"prompt": "one two three"})
    result = audit_dataset(
        dataset_path=dataset,
        tokenizer=_WhitespaceTokenizer(),
        config={
            "_stage_kind": "analysis_grpo",
            "dataset_prompt_column": "prompt",
            "system_prompt": "system words",
            "user_prompt_suffix": "close reasoning now",
            "chat_template_kwargs": {"truncation": False},
        },
        budget={"prompt": 11, "completion": 4, "overflow_policy": "error"},
    )
    assert result["splits"]["train"]["max_prompt_tokens"] == 11


def test_grpo_gate_rejects_prompt_overflow(tmp_path: Path) -> None:
    dataset = tmp_path / "grpo"
    _write_splits(dataset, {"prompt": "one two three"})
    with pytest.raises(TokenBudgetError, match="prompt uses 8 tokens; budget is 7"):
        audit_dataset(
            dataset_path=dataset,
            tokenizer=_WhitespaceTokenizer(),
            config={"_stage_kind": "decision_grpo", "system_prompt": "system words"},
            budget={"prompt": 7, "completion": 4, "overflow_policy": "error"},
        )


def test_sft_gate_records_prompt_completion_and_total(tmp_path: Path) -> None:
    dataset = tmp_path / "sft"
    _write_splits(
        dataset,
        {
            "sample_id": "safe-sample-1",
            "prompt": "SENSITIVE_PROMPT one",
            "response": "SENSITIVE_RESPONSE four",
        },
    )
    result = audit_dataset(
        dataset_path=dataset,
        tokenizer=_WhitespaceTokenizer(),
        config={"_stage_kind": "analysis_sft", "system_prompt": "sys"},
        budget={"prompt": 6, "completion": 3, "total": 9, "overflow_policy": "error"},
    )
    first = result["rows"][0]
    assert first == {
        "sample_id": "safe-sample-1",
        "split": "train",
        "line": 1,
        "prompt_tokens": 6,
        "completion_tokens": 3,
        "total_tokens": 9,
    }
    serialized = json.dumps(result)
    assert "SENSITIVE_PROMPT" not in serialized
    assert "SENSITIVE_RESPONSE" not in serialized


def test_sft_gate_accepts_completion_without_independent_ceiling(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "sft"
    _write_splits(
        dataset,
        {"prompt": "one", "response": " ".join(["token"] * 20)},
    )
    result = audit_dataset(
        dataset_path=dataset,
        tokenizer=_WhitespaceTokenizer(),
        config={"_stage_kind": "analysis_sft", "system_prompt": "sys"},
        budget={
            "prompt": 6,
            "completion": None,
            "total": 30,
            "overflow_policy": "error",
        },
    )

    assert result["rows"][0]["completion_tokens"] == 21
    assert result["rows"][0]["total_tokens"] <= 30


@pytest.mark.parametrize(
    ("row", "message"),
    [
        ({"response": "answer"}, "missing required column 'prompt'"),
        ({"prompt": "question"}, "missing required column 'response'"),
    ],
)
def test_sft_gate_rejects_missing_columns(tmp_path: Path, row: dict, message: str) -> None:
    dataset = tmp_path / "sft"
    _write_splits(dataset, row)
    with pytest.raises(TokenBudgetError, match=message):
        audit_dataset(
            dataset_path=dataset,
            tokenizer=_WhitespaceTokenizer(),
            config={"_stage_kind": "minutes_sft"},
            budget={"prompt": 8, "completion": 8, "total": 16, "overflow_policy": "error"},
        )


def test_gate_requires_both_exact_split_files(tmp_path: Path) -> None:
    dataset = tmp_path / "missing-validation"
    dataset.mkdir()
    (dataset / "train.jsonl").write_text('{"prompt":"x"}\n', encoding="utf-8")
    with pytest.raises(TokenBudgetError, match="exactly one validation JSONL"):
        audit_dataset(
            dataset_path=dataset,
            tokenizer=_WhitespaceTokenizer(),
            config={"_stage_kind": "analysis_grpo"},
            budget={"prompt": 8, "completion": 8, "overflow_policy": "error"},
        )


def test_contract_reads_budget_from_manifest_pinned_dag(tmp_path: Path) -> None:
    root = tmp_path
    run_root = root / "output/training/retrain_v2/run_a"
    config_path = run_root / "resolved_configs/chk2.yaml"
    config_path.parent.mkdir(parents=True)
    config_path.write_text(
        "dataset_name: somewhere\n"
        "max_completion_length: 1024\n"
        "chat_template_kwargs: {truncation: false}\n"
        "judge_tokenizer_path: models/Qwen3.5-9B\n"
        "judge_max_model_len: 8192\n"
        "judge_max_completion_tokens: 2048\n"
        "judge_candidate_reserve_tokens: 1536\n"
        "judge_boundary_margin_tokens: 32\n",
        encoding="utf-8",
    )
    dag_path = root / "configs/retrain_v2/dag.yaml"
    dag_path.parent.mkdir(parents=True)
    dag_path.write_text(
        "schema_version: 2\n"
        "token_budgets:\n"
        "  chk2: {prompt: 2560, completion: 1024, overflow_policy: error}\n"
        "stages:\n"
        "  chk2: {kind: analysis_grpo}\n",
        encoding="utf-8",
    )
    manifest_path = run_root / "run_manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "run_id": "run_a",
                "dag": {"path": str(dag_path), "sha256": sha256_file(dag_path)},
                "judge": {
                    "tokenizer_path": "models/Qwen3.5-9B",
                    "max_model_len": 8192,
                    "max_completion_tokens": 2048,
                    "candidate_reserve_tokens": 1536,
                    "boundary_margin_tokens": 32,
                },
                "stages": {
                    "chk2": {
                        "resolved_config": {
                            "path": str(config_path),
                            "sha256": sha256_file(config_path),
                        }
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    budget, config, observed_config_path = load_stage_contract(
        manifest_path, stage_id="chk2", repo_root=root
    )
    assert budget["prompt"] == 2560
    assert config["_stage_kind"] == "analysis_grpo"
    assert observed_config_path == config_path


def _judge_baseline_config(**overrides):
    config = {
        "judge_model": "Qwen3.5-9B",
        "judge_max_model_len": 8192,
        "judge_max_completion_tokens": 2048,
        "judge_candidate_reserve_tokens": 1536,
        "judge_boundary_margin_tokens": 32,
    }
    config.update(overrides)
    return config


def test_chk2_judge_baseline_reserves_candidate_output_and_margin(tmp_path: Path):
    dataset = tmp_path / "chk2"
    _write_splits(
        dataset,
        {"sample_id": "safe", "prompt": "unused", "provided_data": "indicator=4"},
    )
    result = audit_chk2_judge_baseline(
        dataset_path=dataset,
        tokenizer=_FixedJudgeTokenizer(4576),
        config=_judge_baseline_config(),
    )
    assert result["runtime_candidate_gate_required"] is True
    assert result["guarantees_unknown_candidate_fit"] is False
    assert result["rows"][0]["conservative_reserved_total_tokens"] == 8192
    assert "indicator=4" not in json.dumps(result)


def test_chk2_judge_baseline_overflow_fails_closed(tmp_path: Path):
    dataset = tmp_path / "chk2"
    _write_splits(
        dataset,
        {"sample_id": "overflow", "prompt": "unused", "provided_data": "safe=1"},
    )
    with pytest.raises(TokenBudgetError, match="baseline reserve uses 8193"):
        audit_chk2_judge_baseline(
            dataset_path=dataset,
            tokenizer=_FixedJudgeTokenizer(4577),
            config=_judge_baseline_config(),
        )


@pytest.mark.parametrize(
    "row",
    [
        {"prompt": "x"},
        {"prompt": "x", "provided_data": ""},
        {"prompt": "x", "provided_data": "meeting_date=2020-01-01"},
    ],
)
def test_chk2_judge_baseline_requires_safe_provided_data(tmp_path: Path, row):
    dataset = tmp_path / "chk2"
    _write_splits(dataset, row)
    with pytest.raises(TokenBudgetError, match="provided_data|forbidden"):
        audit_chk2_judge_baseline(
            dataset_path=dataset,
            tokenizer=_FixedJudgeTokenizer(10),
            config=_judge_baseline_config(),
        )


def test_stage_launcher_runs_gate_only_after_execute_guard() -> None:
    script = Path("run/retrain_v2/stage.sh").read_text(encoding="utf-8")
    guard = script.index('if [[ "${EXECUTE}" == false ]]')
    gate = script.index("jobs.retrain_v2.token_budget_gate")
    gpu_gate = script.index("run/retrain_v2/gpu_gate.sh")
    training = script.index("jobs.train.train_sft")
    assert script.count("jobs.retrain_v2.token_budget_gate") == 1
    assert guard < gate < gpu_gate < training


def test_gate_loads_only_tokenizer_not_model_weights() -> None:
    source = Path("jobs/retrain_v2/token_budget_gate.py").read_text(encoding="utf-8")
    assert "AutoTokenizer" in source
    assert "AutoModel" not in source
    assert "local_files_only=True" in source


def test_gate_cli_serializes_context_budget_error_as_blocked(monkeypatch, capsys):
    monkeypatch.setattr(
        "jobs.retrain_v2.token_budget_gate.run_gate",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            JudgeContextBudgetError("synthetic overflow")
        ),
    )
    return_code = token_gate_main(
        ["--run-manifest", "unused.json", "--stage", "chk2"]
    )
    assert return_code == 2
    assert json.loads(capsys.readouterr().out) == {
        "status": "blocked",
        "error": "synthetic overflow",
    }
