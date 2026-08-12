from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT / "configs/main/checkpoint_generation_eval_11.json"
)
POPULATION = (
    ROOT / "configs/main/loo_population_checkpoint_eval_11.json"
)
SEMANTIC_MODELS = (
    ROOT / "configs/main/checkpoint_eval_semantic_models.json"
)


def test_common_checkpoint_config_freezes_clean_matrix() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    population = json.loads(POPULATION.read_text(encoding="utf-8"))
    dates = config["population"]["meeting_dates"]
    prospective = config["population"]["prospective_only_meeting_dates"]

    assert config["training_performed"] is False
    assert len(dates) == 11
    assert dates == sorted(set(dates))
    assert len(prospective) == 9
    assert set(prospective) < set(dates)
    assert config["population"]["population_id"] == "checkpoint_eval_11"
    assert population["population_id"] == "checkpoint_eval_11"
    assert population["meeting_dates"] == dates
    assert len(config["sections"]) == 3
    assert len({section["section_id"] for section in config["sections"]}) == 3


def test_common_checkpoint_config_forbids_reference_and_training() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    assert config["information_cutoff"] == {
        "policy": "previous-calendar-day-v1",
        "meeting_day_data": "forbidden",
        "reference_in_prompt": False,
        "secondary_llm_summary": False,
    }
    generation = config["generation"]
    assert generation["temperature"] == 0.0
    assert generation["top_p"] == 1.0
    assert generation["input_truncation"] == "forbidden"
    assert generation["token_limit_finish"] == "invalid"
    assert generation["max_model_len"] == 24576
    assert generation["max_new_tokens"] == 8192
    assert config["evaluation"]["llm_judge"] is False
    assert config["evaluation"]["human_rating"] is False


def test_semantic_encoders_are_revision_and_content_pinned() -> None:
    manifest = json.loads(SEMANTIC_MODELS.read_text(encoding="utf-8"))
    bertscore = manifest["models"]["bertscore"]
    embedding = manifest["models"]["embedding_cosine"]

    assert len(bertscore["resolved_revision"]) == 40
    assert len(embedding["resolved_revision"]) == 40
    assert len(bertscore["directory_sha256"]) == 64
    assert len(embedding["directory_sha256"]) == 64
    assert bertscore["num_layers"] == 17
    assert embedding["independent_from_training_reward"] is True
    assert manifest["network_at_scoring_time"] is False
