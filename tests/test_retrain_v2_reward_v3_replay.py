import json

import pytest

from jobs.retrain_v2.replay_chk2_reward_v3 import (
    ReplayError,
    evidence_signature,
    extract_evidence_from_rendered_prompt,
    summarize_replay,
)


def _record(reward):
    return {
        "type": "grounded_analysis_v3",
        "reward": reward,
        "judge_attempts": 1,
        "invalid_violations": [],
        "penalties": {
            "major_answer_error": 0,
            "minor_answer_error": 0,
            "major_think_error": 0,
            "minor_think_error": 0,
            "unbounded": 0.0,
        },
    }


def test_extract_evidence_from_compact_rendered_prompt():
    evidence = {"atomic_topic": "Inflation", "evidence": [{"value": "2.0"}]}
    prompt = "system<User>Analyze" + json.dumps(evidence, separators=(",", ":")) + "<Assistant>"
    extracted = extract_evidence_from_rendered_prompt(prompt)
    assert json.loads(extracted) == evidence


def test_extract_evidence_requires_json_object():
    with pytest.raises(ReplayError, match="no JSON fact card"):
        extract_evidence_from_rendered_prompt("no fact card")


def test_evidence_signature_survives_tokenizer_space_loss():
    sealed = json.dumps(
        {
            "atomic_topic": "Unemployment Rate",
            "evidence": [
                {"evidence_id": "ev-abc", "metric": "Unemployment Rate"},
                {"evidence_id": "ev-def", "metric": "Payroll Employment"},
            ],
        }
    )
    decoded = sealed.replace("Unemployment Rate", "UnemploymentRate").replace(
        "Payroll Employment", "PayrollEmployment"
    )
    assert evidence_signature(decoded) == evidence_signature(sealed) == (
        "ev-abc",
        "ev-def",
    )


def test_replay_summary_applies_distribution_gates():
    groups = [
        [_record(0.10), _record(0.25), _record(0.50), _record(0.75)],
        [_record(0.20), _record(0.35), _record(0.60), _record(0.90)],
    ]
    summary = summarize_replay(groups)
    assert summary["reward"]["finite"] is True
    assert summary["reward"]["bounded_0_1"] is True
    assert summary["zero_variance_group_fraction"] == 0.0
    assert summary["nonfactual_penalty_violations"] == 0
    assert summary["passed"] is True


def test_replay_summary_rejects_concentrated_zero_variance_rewards():
    summary = summarize_replay(
        [[_record(0.25), _record(0.25), _record(0.25), _record(0.25)]]
    )
    assert summary["reward"]["max_point_mass_fraction"] == 1.0
    assert summary["zero_variance_group_fraction"] == 1.0
    assert summary["passed"] is False
