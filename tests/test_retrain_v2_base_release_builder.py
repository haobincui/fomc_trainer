from __future__ import annotations

import hashlib
import json
import shutil
import stat
from datetime import date, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from jobs.retrain_v2 import build_base_release as builder_module
from jobs.retrain_v2.build_base_release import (
    BaseReleaseError,
    build_base_release,
    stable_train_role,
)
from jobs.retrain_v2.chk1.contracts import (
    HANDOFF_SCHEMA_VERSION,
    MANIFEST_SCHEMA_VERSION,
    RELEASE_SCHEMA_VERSION,
    canonical_json,
    sha256_text,
)
from jobs.retrain_v2.chk1.release import QUALITY_SCHEMA_VERSION
from jobs.retrain_v2.chk1.prompt_projection import (
    build_chat_token_counter,
    build_sft_token_auditor,
    project_preteacher_inputs,
)
from jobs.retrain_v2.dag import DagValidationError, validate_base_release
from jobs.main.checkpoint_provenance import (
    fingerprint_model_payload,
    fingerprint_tokenizer_payload,
)
from open_r1.provenance import sha256_file


REPO_ROOT = Path(__file__).resolve().parents[1]
DIGESTS = {
    "teacher": "1" * 64,
    "critic": "2" * 64,
    "student_tokenizer": "3" * 64,
    "generator_tokenizer": "4" * 64,
    "style": "5" * 64,
    "generation": "6" * 64,
    "prompt": "7" * 64,
    "cache": "8" * 64,
    "evidence": "9" * 64,
    "raw": "a" * 64,
    "request": "b" * 64,
    "snapshot": "c" * 64,
    "registry": "d" * 64,
    "source": "e" * 64,
}


class TinyTokenizer:
    eos_token = "<eos>"

    @staticmethod
    def _render(messages: list[dict[str, str]]) -> str:
        return "".join(
            f"<{message['role']}>\n{message['content']}\n" for message in messages
        ) + "<assistant>\n"

    def apply_chat_template(
        self,
        messages: list[dict[str, str]],
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        truncation: bool | None = None,
        return_dict: bool | None = None,
    ):
        assert add_generation_prompt is True
        if tokenize:
            assert truncation is False
            assert return_dict is False
        rendered = self._render(messages)
        return list(rendered.encode("utf-8")) if tokenize else rendered

    def __call__(self, *, text: str):
        return {"input_ids": list(text.encode("utf-8"))}


class OverflowTokenizer(TinyTokenizer):
    def apply_chat_template(self, *args, tokenize: bool, **kwargs):
        rendered = super().apply_chat_template(*args, tokenize=tokenize, **kwargs)
        # Preserve the renderer/TRL token-ID parity contract while forcing the
        # complete sample over the admission budget.
        return [0] * 5000 if tokenize else rendered

    def __call__(self, *, text: str):
        return {"input_ids": [0] * 5000}


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )


def _sample_ids_for_all_roles() -> list[str]:
    selected: dict[str, str] = {}
    index = 0
    while set(selected) != {"sft_only", "grpo_only", "shared"}:
        sample_id = f"2001-01-02::synthetic-train-{index}"
        selected.setdefault(stable_train_role(sample_id), sample_id)
        index += 1
    return [selected[role] for role in ("sft_only", "grpo_only", "shared")]


def _pair(
    sample_id: str,
    split: str,
    meeting_date: str,
    index: int,
    *,
    tokenizer,
    teacher_sha: str,
    critic_sha: str,
    student_tokenizer_sha: str,
):
    cutoff = f"{meeting_date}T00:00:00Z"
    evidence_id = f"evidence-{index}"
    evidence = {
        "evidence_id": evidence_id,
        "source_id": f"source-{index}",
        "source_sha256": DIGESTS["source"],
        "cutoff_ts": cutoff,
        "observation_date": "2000-12-31",
        "release_ts": "2000-12-31T00:00:00Z",
        "availability_basis": "actual_release_ts",
        "value": index + 1,
    }
    lineage = {
        "evidence_id": evidence_id,
        "source_id": evidence["source_id"],
        "source_sha256": DIGESTS["source"],
        "cutoff_ts": cutoff,
        "evidence_sha256": DIGESTS["evidence"],
        "raw_sha256": DIGESTS["raw"],
        "request_id": DIGESTS["request"],
        "snapshot_manifest_payload_sha256": DIGESTS["snapshot"],
        "registry_sha256": DIGESTS["registry"],
    }
    fact = {
        "schema_version": "chk1-point-in-time-fact-card-v1",
        "sample_id": sample_id,
        "meeting_date": meeting_date,
        "atomic_topic": f"topic-{index}",
        "cutoff_ts": cutoff,
        "canonical_key": {
            "meeting_date": meeting_date,
            "atomic_topic": f"topic-{index}",
        },
        "evidence": [evidence],
        "evidence_lineage": [lineage],
    }
    fact["fact_card_sha256"] = sha256_text(canonical_json(fact))
    prepared_fact = {
        key: fact[key]
        for key in (
            "schema_version",
            "sample_id",
            "canonical_key",
            "meeting_date",
            "atomic_topic",
            "cutoff_ts",
            "evidence",
        )
    }
    style_guide = {
        "section_style_id": f"style-{index}",
        "voice": "neutral synthetic analysis",
    }
    projected = project_preteacher_inputs(
        fact_card=prepared_fact,
        evidence_lineage=[lineage],
        atomic_topic=f"topic-{index}",
        style_guide=style_guide,
        student_chat_token_counter=build_chat_token_counter(tokenizer),
        generator_prompt_token_counter=lambda _prompt: 0,
    )
    prompt = projected.student_prompt
    reasoning = f"Evidence {evidence_id} supports synthetic trend {index}."
    final = f"Synthetic policy analysis {index} is grounded in {evidence_id}."
    response = f"{reasoning}\n</think>\n{final}"
    provided_data = projected.provided_data
    sft = {"prompt": prompt, "response": response, "provided_data": provided_data}
    prepared_payload = {
        "sample_id": sample_id,
        "split": split,
        "meeting_date": meeting_date,
        "atomic_topic": f"topic-{index}",
        "fact_card": prepared_fact,
        "evidence_lineage": [lineage],
        "generator_fact_card_sha256": sha256_text(canonical_json(prepared_fact)),
    }
    prepared = {
        **prepared_payload,
        "row_sha256": sha256_text(canonical_json(prepared_payload)),
    }
    sft_token_budget = dict(build_sft_token_auditor(tokenizer)(prompt, response))
    assert (
        sft_token_budget["prompt_tokens"]
        == projected.attestation["student_projection"]["prompt_tokens"]
    )
    manifest = {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "sample_id": sample_id,
        "meeting_date": meeting_date,
        "atomic_topic": f"topic-{index}",
        "section_style_id": f"style-{index}",
        "split": split,
        "cutoff_ts": cutoff,
        "evidence_lineage": [lineage],
        "style_guide_sha256": DIGESTS["style"],
        "teacher_model_sha256": teacher_sha,
        "critic_model_sha256": critic_sha,
        "tokenizer_sha256": student_tokenizer_sha,
        "generation": {
            "cache_key": DIGESTS["cache"],
            "generation_provenance_sha256": DIGESTS["generation"],
            "prompt_template_sha256": DIGESTS["prompt"],
            "generator_tokenizer_sha256": DIGESTS["generator_tokenizer"],
            "student_tokenizer_sha256": student_tokenizer_sha,
            "preparation_binding_sha256": "f" * 64,
            "prepared_row_sha256": sha256_text(canonical_json(prepared)),
            "model_input_projection": dict(projected.attestation),
            "sft_token_budget": sft_token_budget,
        },
        "prompt_sha256": sha256_text(prompt),
        "reasoning_sha256": sha256_text(reasoning),
        "final_analysis_sha256": sha256_text(final),
        "response_sha256": sha256_text(response),
        "provided_data_sha256": sha256_text(provided_data),
        "verifier": {"passed": True, "error_codes": []},
        "critic": {
            "grounded": True,
            "unsupported_claims": [],
            "style_score": 5,
            "reasoning_consistency": True,
        },
        "input_truncated": False,
    }
    return sft, manifest, prepared


def _seal(path: Path) -> None:
    for candidate in sorted(
        path.rglob("*"), key=lambda item: len(item.parts), reverse=True
    ):
        candidate.chmod(0o444 if candidate.is_file() else 0o555)
    path.chmod(0o555)


def _canonical_source(root: Path, *, with_exclusion: bool = False) -> Path:
    source = root / "synthetic/canonical/source_release"
    chk0 = root / "models/DeepSeek-R1-Distill-Llama-8B"
    teacher = root / "models/Qwen3.5-9B"
    teacher_sha = fingerprint_model_payload(teacher)["sha256"]
    critic_sha = fingerprint_model_payload(chk0)["sha256"]
    student_tokenizer_sha = fingerprint_tokenizer_payload(chk0)["sha256"]
    tokenizer = builder_module._load_tokenizer(chk0)
    first_meeting = date(2001, 1, 2)
    train_samples = [
        (
            f"{(first_meeting + timedelta(days=index)).isoformat()}::synthetic-train-{index}",
            (first_meeting + timedelta(days=index)).isoformat(),
        )
        for index in range(198)
    ]
    definitions = {
        "train": train_samples,
        "eval": [("synthetic-eval", "2002-02-02")],
        "test": [("synthetic-test", "2003-03-03")],
    }
    sft_rows: dict[str, list[dict]] = {}
    manifest_rows: dict[str, list[dict]] = {}
    prepared_rows: list[dict] = []
    ordinal = 0
    for split, samples in definitions.items():
        sft_rows[split] = []
        manifest_rows[split] = []
        for sample_id, meeting in samples:
            sft, manifest, prepared = _pair(
                sample_id,
                split,
                meeting,
                ordinal,
                tokenizer=tokenizer,
                teacher_sha=teacher_sha,
                critic_sha=critic_sha,
                student_tokenizer_sha=student_tokenizer_sha,
            )
            ordinal += 1
            sft_rows[split].append(sft)
            manifest_rows[split].append(manifest)
            prepared_rows.append(prepared)
        _write_jsonl(source / f"sft/{split}.jsonl", sft_rows[split])
        _write_jsonl(source / f"manifests/{split}.jsonl", manifest_rows[split])
    if with_exclusion:
        excluded_sft, excluded_manifest, excluded_prepared = _pair(
            "2001-12-31::synthetic-excluded",
            "train",
            "2001-12-31",
            ordinal,
            tokenizer=tokenizer,
            teacher_sha=teacher_sha,
            critic_sha=critic_sha,
            student_tokenizer_sha=student_tokenizer_sha,
        )
        del excluded_sft, excluded_manifest
        prepared_rows.append(excluded_prepared)
    exclusions = (
        [
            {
                "schema_version": "chk1-generation-exclusion-v1",
                "sample_id": "2001-12-31::synthetic-excluded",
                "split": "train",
                "meeting_date": "2001-12-31",
                "atomic_topic": excluded_prepared["atomic_topic"],
                "stage": "critic",
                "reason_code": "critic_rejected",
                "error_type": "SyntheticRejection",
                "error_codes": ["synthetic"],
                "prepared_row_sha256": sha256_text(
                    canonical_json(excluded_prepared)
                ),
            }
        ]
        if with_exclusion
        else []
    )
    _write_jsonl(source / "audit/exclusions.jsonl", exclusions)
    _write_jsonl(source / "synthetic/prepared.jsonl", prepared_rows)
    accepted_counts = {split: len(sft_rows[split]) for split in definitions}
    excluded_counts = {"train": 1} if with_exclusion else {}
    candidate_counts = {
        split: accepted_counts[split] + int(excluded_counts.get(split, 0))
        for split in definitions
    }
    population_count = sum(candidate_counts.values())
    topic_rates = {
        manifest["atomic_topic"]: 1.0
        for split in manifest_rows.values()
        for manifest in split
    }
    quality = {
        "schema_version": QUALITY_SCHEMA_VERSION,
        "status": "passed",
        "gates": {
            "train_acceptance": True,
            "per_topic_acceptance": True,
            "zero_tolerance_exclusions": True,
        },
        "population_binding_sha256": "f" * 64,
        "split_binding_sha256": "0" * 64,
        "topic_binding_sha256": "1" * 64,
        "population_count": population_count,
        "split_counts": accepted_counts,
        "candidate_counts": candidate_counts,
        "meeting_counts": candidate_counts,
        "accepted_meeting_counts": accepted_counts,
        "meeting_binding_modes": {
            split: "exact_membership" for split in definitions
        },
        "excluded_counts": excluded_counts,
        "exclusion_reasons": {"critic_rejected": 1} if with_exclusion else {},
        "train_acceptance_rate": (
            accepted_counts["train"] / candidate_counts["train"]
        ),
        "required_train_acceptance_rate": 0.7,
        "per_topic_acceptance_rates": {
            **topic_rates,
            **({"topic-0": 0.5} if with_exclusion else {}),
        },
        "topics_below_threshold": {},
        "zero_tolerance_exclusions": {},
        "manifest_provenance": {
            "generation_provenance_sha256": DIGESTS["generation"],
            "prompt_template_sha256": DIGESTS["prompt"],
            "generator_tokenizer_sha256": DIGESTS["generator_tokenizer"],
            "style_guide_sha256": DIGESTS["style"],
            "teacher_model_sha256": teacher_sha,
            "critic_model_sha256": critic_sha,
            "tokenizer_sha256": student_tokenizer_sha,
        },
        "invariants": {
            "canonical_key_unique": True,
            "sample_id_unique": True,
            "prompt_hash_unique": True,
            "response_hash_unique": True,
            "meeting_splits_disjoint": True,
            "expected_population_covered_once": True,
            "expected_topic_roster_exact": True,
            "input_truncation_count": 0,
            "teacher_prompt_leakage_count": 0,
        },
    }
    _write_json(source / "audit/quality_report.json", quality)
    split_files = {}
    manifest_files = {}
    for split in ("train", "eval", "test"):
        sft_path = source / f"sft/{split}.jsonl"
        manifest_path = source / f"manifests/{split}.jsonl"
        split_files[split] = {
            "path": f"sft/{split}.jsonl",
            "rows": len(sft_rows[split]),
            "sha256": sha256_file(sft_path),
        }
        manifest_files[split] = {
            "path": f"manifests/{split}.jsonl",
            "rows": len(manifest_rows[split]),
            "sha256": sha256_file(manifest_path),
        }
    handoff = {
        "schema_version": HANDOFF_SCHEMA_VERSION,
        "release_schema_version": RELEASE_SCHEMA_VERSION,
        "release_id": "synthetic_source",
        "generated_at_utc": "2026-08-03T00:00:00Z",
        "immutable": True,
        "quality_status": "passed",
        "population_binding_sha256": quality["population_binding_sha256"],
        "split_binding_sha256": quality["split_binding_sha256"],
        "topic_binding_sha256": quality["topic_binding_sha256"],
        "population_count": population_count,
        "split_counts": {split: len(sft_rows[split]) for split in definitions},
        "split_files": split_files,
        "manifest_files": manifest_files,
        "prompt_template_sha256": DIGESTS["prompt"],
        "generation_provenance_sha256": DIGESTS["generation"],
        "generator_tokenizer_sha256": DIGESTS["generator_tokenizer"],
        "style_guide_sha256": DIGESTS["style"],
        "teacher_model_sha256": teacher_sha,
        "critic_model_sha256": critic_sha,
        "tokenizer_sha256": student_tokenizer_sha,
        "quality_report": {
            "path": "audit/quality_report.json",
            "sha256": sha256_file(source / "audit/quality_report.json"),
        },
    }
    _write_json(source / "handoff.json", handoff)
    _seal(source)
    return source / "handoff.json"


def _tiny_repo(root: Path) -> None:
    shutil.copytree(REPO_ROOT / "configs/retrain_v2", root / "configs/retrain_v2")
    registry = root / "src/open_r1/trainer/rewards/reward_register.py"
    registry.parent.mkdir(parents=True)
    registry.write_text(
        "def get_reward_funcs():\n"
        "    REWARD_FUNCS_REGISTRY = {\n"
        "        'grounded_analysis_v2': object(),\n"
        "        'decision_dense_v2': object(),\n"
        "    }\n",
        encoding="utf-8",
    )
    chk0 = root / "models/DeepSeek-R1-Distill-Llama-8B"
    chk0.mkdir(parents=True)
    (chk0 / "config.json").write_text(
        '{"model_type":"llama","vocab_size":2}\n', encoding="utf-8"
    )
    (chk0 / "model.safetensors").write_bytes(b"synthetic-chk0")
    backend = Tokenizer(WordLevel({"[UNK]": 0, "[EOS]": 1}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.save(str(chk0 / "tokenizer.json"))
    _write_json(
        chk0 / "tokenizer_config.json",
        {
            "tokenizer_class": "PreTrainedTokenizerFast",
            "unk_token": "[UNK]",
            "eos_token": "[EOS]",
            "chat_template": (
                "{% for message in messages %}<{{ message['role'] }}>"
                "{{ message['content'] }}{% endfor %}"
                "{% if add_generation_prompt %}<assistant>{% endif %}"
            ),
        },
    )
    judge = root / "models/Qwen3.5-9B"
    judge.mkdir(parents=True)
    (judge / "config.json").write_text('{"model_type":"qwen3_5"}\n')
    (judge / "model.safetensors").write_bytes(b"synthetic-judge")


@pytest.fixture(autouse=True)
def _synthetic_canonical_replay(monkeypatch: pytest.MonkeyPatch):
    """Keep builder unit tests small; canonical_workflow has its own replay tests."""

    def replay(*, repo_root: Path, source_root: Path, handoff: dict):
        del repo_root, handoff
        sft_rows = {
            split: tuple(
                json.loads(line)
                for line in (source_root / f"sft/{split}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            for split in ("train", "eval", "test")
        }
        manifest_rows = {
            split: tuple(
                json.loads(line)
                for line in (source_root / f"manifests/{split}.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            )
            for split in ("train", "eval", "test")
        }
        exclusions = tuple(
            json.loads(line)
            for line in (source_root / "audit/exclusions.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        population = tuple(
            json.loads(line)
            for line in (source_root / "synthetic/prepared.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        generation_code = {
            "schema_version": "chk1-generation-code-bundle-v1",
            "files": [
                {
                    "path": "jobs/retrain_v2/chk1/generation_pipeline.py",
                    "bytes": 1,
                    "sha256": "a" * 64,
                }
            ],
        }
        generation_code["payload_sha256"] = sha256_text(
            canonical_json(generation_code)
        )
        verified = SimpleNamespace(
            sft_rows=sft_rows,
            manifest_rows=manifest_rows,
            exclusions=exclusions,
            population=population,
            generation_handoff={
                "generation_provenance": {"generation_code": generation_code}
            },
        )
        replay_audit = {
            "generation_handoff_sha256": "1" * 64,
            "prepare_handoff_sha256": "2" * 64,
            "generation_bundle_payload_sha256": "3" * 64,
            "preparation_bundle_payload_sha256": "4" * 64,
            "generation_code_payload_sha256": generation_code["payload_sha256"],
            "preparation_binding_sha256": "5" * 64,
            "selected_sample_ids_sha256": "6" * 64,
        }
        return verified, replay_audit

    monkeypatch.setattr(builder_module, "_replay_canonical_source", replay)


def test_builder_publishes_atomic_dag_valid_release(tmp_path: Path) -> None:
    _tiny_repo(tmp_path)
    handoff = _canonical_source(tmp_path)

    release = build_base_release(
        repo_root=tmp_path,
        canonical_handoff=handoff,
        base_release_id="base_synthetic",
        generated_at_utc="2026-08-03T01:00:00Z",
    )

    validated = validate_base_release(
        repo_root=tmp_path,
        base_release_id="base_synthetic",
    )
    assert validated["status"] == "valid"
    manifest = json.loads((release / "base_release_manifest.json").read_text())
    tokenizer_file_sha = sha256_file(
        tmp_path / "models/DeepSeek-R1-Distill-Llama-8B/tokenizer.json"
    )
    assert manifest["provenance"]["tokenizer_sha256"] != tokenizer_file_sha
    assert (
        manifest["provenance"]["tokenizer_runtime"]["mode"]
        == "auto_tokenizer_local_bundle"
    )
    assert (
        manifest["provenance"]["source_student_tokenizer_payload_sha256"]
        == fingerprint_tokenizer_payload(
            tmp_path / "models/DeepSeek-R1-Distill-Llama-8B"
        )["sha256"]
    )
    assert tokenizer_file_sha != manifest["provenance"]["tokenizer_sha256"]
    assert manifest["provenance"]["teacher_model"] == "local Qwen3.5-9B"
    assert manifest["provenance"]["critic_model"].startswith("chk0 DeepSeek")
    prompt_contract = json.loads(
        (release / manifest["provenance"]["artifacts"]["prompt"]["path"])
        .read_text(encoding="utf-8")
    )
    assert set(prompt_contract["training_system_prompts"]) == {
        "analysis_sft",
        "analysis_grpo",
    }
    assert prompt_contract["training_system_prompt_sha256"] == {
        stage: sha256_text(prompt)
        for stage, prompt in prompt_contract["training_system_prompts"].items()
    }

    train_sft = [json.loads(line) for line in (release / "analysis_sft/train.jsonl").read_text().splitlines()]
    train_grpo = [json.loads(line) for line in (release / "analysis_grpo/train.jsonl").read_text().splitlines()]
    source_train = [
        json.loads(line)
        for line in (handoff.parent / "manifests/train.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    expected_roles = [stable_train_role(row["sample_id"]) for row in source_train]
    assert len(train_sft) == sum(
        role in {"sft_only", "shared"} for role in expected_roles
    )
    assert len(train_grpo) == sum(
        role in {"grpo_only", "shared"} for role in expected_roles
    )
    assert len([json.loads(line) for line in (release / "analysis_sft/eval.jsonl").read_text().splitlines()]) == 1
    assert len([json.loads(line) for line in (release / "analysis_grpo/eval.jsonl").read_text().splitlines()]) == 1
    assert set(train_grpo[0]) == {"prompt", "provided_data", "sample_id", "meeting_date"}
    assert "meeting_date" not in json.loads(train_grpo[0]["provided_data"])
    assert "sample_id" not in json.loads(train_grpo[0]["provided_data"])
    safe_evidence = json.loads(train_grpo[0]["provided_data"])["evidence"][0]
    assert not {
        "source_id",
        "cutoff_ts",
        "availability_upper_bound_ts",
    } & set(safe_evidence)
    assert train_grpo[0]["meeting_date"] not in train_grpo[0]["provided_data"]
    assert train_grpo[0]["prompt"].count(train_grpo[0]["provided_data"]) == 1
    assert train_grpo[0]["prompt"].endswith(train_grpo[0]["provided_data"])
    target_audit = json.loads(
        (release / "audits/target_consistency.json").read_text(encoding="utf-8")
    )
    assert target_audit["details"]["grpo_projection_count"] == sum(
        len(
            (release / f"analysis_grpo/{split}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        )
        for split in ("train", "eval", "test")
    )
    assert len(target_audit["details"]["grpo_projection_contract_sha256"]) == 64
    assert release.stat().st_mode & (stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH) == 0
    assert not list(release.parent.glob(".base_synthetic.build-*"))

    assert (
        build_base_release(
            repo_root=tmp_path,
            canonical_handoff=handoff,
            base_release_id="base_synthetic",
        )
        == release
    )
    split_audit = json.loads(
        (release / "audits/split_integrity.json").read_text(encoding="utf-8")
    )
    assignment = split_audit["details"]["train_assignment"]
    assert "not exact quotas" in assignment["semantics"]
    assert assignment["actual_counts_by_atomic_topic"]
    assert assignment["actual_counts_by_meeting_year"]


@pytest.mark.parametrize(
    "handoff_schema",
    ["chk1-generation-handoff-v2", "chk1-prepare-handoff-v1"],
)
def test_builder_rejects_generation_or_prepared_handoff(
    tmp_path: Path, handoff_schema: str
) -> None:
    _tiny_repo(tmp_path)
    generation = tmp_path / "synthetic/generation"
    generation.mkdir(parents=True)
    _write_json(
        generation / "handoff.json",
        {"schema_version": handoff_schema},
    )
    _seal(generation)
    with pytest.raises(BaseReleaseError, match="not a canonical chk1 release"):
        build_base_release(
            repo_root=tmp_path,
            canonical_handoff=generation / "handoff.json",
            base_release_id="rejected_generation",
            test_tokenizer=TinyTokenizer(),
        )


def test_builder_rejects_writable_handoff(tmp_path: Path) -> None:
    _tiny_repo(tmp_path)
    canonical = _canonical_source(tmp_path)
    canonical.chmod(0o644)
    with pytest.raises(BaseReleaseError, match="source release is writable"):
        build_base_release(
            repo_root=tmp_path,
            canonical_handoff=canonical,
            base_release_id="rejected_writable",
            test_tokenizer=TinyTokenizer(),
        )


def test_token_overflow_is_fail_closed_without_truncation(tmp_path: Path) -> None:
    _tiny_repo(tmp_path)
    handoff = _canonical_source(tmp_path)
    with pytest.raises(BaseReleaseError, match="exceeds .*training token budget"):
        build_base_release(
            repo_root=tmp_path,
            canonical_handoff=handoff,
            base_release_id="overflow",
            test_tokenizer=OverflowTokenizer(),
        )
    assert not (tmp_path / "dataset/processed/retrain_v2/overflow").exists()
    assert not list(
        (tmp_path / "dataset/processed/retrain_v2").glob(".overflow.build-*")
    )


def test_builder_fails_closed_before_publication_on_lineage_tamper(
    tmp_path: Path,
) -> None:
    _tiny_repo(tmp_path)
    handoff = _canonical_source(tmp_path)
    source = handoff.parent
    for path in source.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    source.chmod(0o700)
    manifest_path = source / "manifests/train.jsonl"
    rows = [json.loads(line) for line in manifest_path.read_text().splitlines()]
    rows[0]["evidence_lineage"][0]["source_sha256"] = "0" * 64
    _write_jsonl(manifest_path, rows)
    handoff_payload = json.loads(handoff.read_text())
    handoff_payload["manifest_files"]["train"]["sha256"] = sha256_file(manifest_path)
    _write_json(handoff, handoff_payload)
    _seal(source)

    with pytest.raises(BaseReleaseError, match="lineage"):
        build_base_release(
            repo_root=tmp_path,
            canonical_handoff=handoff,
            base_release_id="tampered",
            test_tokenizer=TinyTokenizer(),
        )
    assert not (tmp_path / "dataset/processed/retrain_v2/tampered").exists()
    assert not list(
        (tmp_path / "dataset/processed/retrain_v2").glob(".tampered.build-*")
    )


def test_valid_exclusion_is_bound_into_population_arithmetic(tmp_path: Path) -> None:
    _tiny_repo(tmp_path)
    handoff = _canonical_source(tmp_path, with_exclusion=True)
    release = build_base_release(
        repo_root=tmp_path,
        canonical_handoff=handoff,
        base_release_id="with_exclusion",
    )
    assert (
        validate_base_release(
            repo_root=tmp_path, base_release_id="with_exclusion"
        )["status"]
        == "valid"
    )
    split_audit = json.loads(
        (release / "audits/split_integrity.json").read_text(encoding="utf-8")
    )
    assert split_audit["checked_rows"] == 201
    assert split_audit["details"]["accepted_counts"] == {
        "train": 198,
        "eval": 1,
        "test": 1,
    }
    assert split_audit["details"]["excluded_counts"] == {
        "train": 1,
        "eval": 0,
        "test": 0,
    }


@pytest.mark.parametrize("field", ["eos_token", "chat_template"])
def test_validate_base_release_rejects_tokenizer_config_or_template_drift(
    tmp_path: Path, field: str
) -> None:
    _tiny_repo(tmp_path)
    handoff = _canonical_source(tmp_path)
    build_base_release(
        repo_root=tmp_path,
        canonical_handoff=handoff,
        base_release_id="tokenizer_drift",
    )
    config_path = (
        tmp_path
        / "models/DeepSeek-R1-Distill-Llama-8B/tokenizer_config.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config[field] = "[CHANGED]" if field == "eos_token" else "{{ 'changed' }}"
    _write_json(config_path, config)
    with pytest.raises(DagValidationError, match="tokenizer (bundle|file manifest)"):
        validate_base_release(
            repo_root=tmp_path, base_release_id="tokenizer_drift"
        )


def test_source_tokenizer_drift_during_audit_cleans_staging(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tiny_repo(tmp_path)
    handoff = _canonical_source(tmp_path)
    original = builder_module._audit_tokens

    def mutate_after_audit(tokenizer, outputs, **kwargs):
        result = original(tokenizer, outputs, **kwargs)
        config_path = (
            tmp_path
            / "models/DeepSeek-R1-Distill-Llama-8B/tokenizer_config.json"
        )
        config_path.write_text(
            config_path.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        return result

    monkeypatch.setattr(builder_module, "_audit_tokens", mutate_after_audit)
    with pytest.raises(BaseReleaseError, match="changed during token audit"):
        build_base_release(
            repo_root=tmp_path,
            canonical_handoff=handoff,
            base_release_id="mid_build_drift",
        )
    assert not (tmp_path / "dataset/processed/retrain_v2/mid_build_drift").exists()
    assert not list(
        (tmp_path / "dataset/processed/retrain_v2").glob(
            ".mid_build_drift.build-*"
        )
    )


def test_injected_tokenizer_is_honestly_non_releasable(tmp_path: Path) -> None:
    _tiny_repo(tmp_path)
    handoff = _canonical_source(tmp_path)
    release = build_base_release(
        repo_root=tmp_path,
        canonical_handoff=handoff,
        base_release_id="injected",
        test_tokenizer=builder_module._load_tokenizer(
            tmp_path / "models/DeepSeek-R1-Distill-Llama-8B"
        ),
    )
    manifest = json.loads(
        (release / "base_release_manifest.json").read_text(encoding="utf-8")
    )
    assert (
        manifest["provenance"]["tokenizer_runtime"]["mode"]
        == "test_injected_non_releasable"
    )
    with pytest.raises(
        DagValidationError, match="production local tokenizer bundle runtime"
    ):
        validate_base_release(repo_root=tmp_path, base_release_id="injected")


def test_stale_staging_requires_explicit_recovery(tmp_path: Path) -> None:
    _tiny_repo(tmp_path)
    handoff = _canonical_source(tmp_path)
    parent = tmp_path / "dataset/processed/retrain_v2"
    stale = parent / ".stale_case.build-interrupted"
    stale.mkdir(parents=True)
    (stale / "partial.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(BaseReleaseError, match="stale staging exists"):
        build_base_release(
            repo_root=tmp_path,
            canonical_handoff=handoff,
            base_release_id="stale_case",
        )
    release = build_base_release(
        repo_root=tmp_path,
        canonical_handoff=handoff,
        base_release_id="stale_case",
        recover_stale_staging=True,
    )
    assert release.is_dir()
    assert not stale.exists()


def test_retry_revalidates_release_after_parent_fsync_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _tiny_repo(tmp_path)
    handoff = _canonical_source(tmp_path)
    parent = tmp_path / "dataset/processed/retrain_v2"
    destination = parent / "fsync_recovery"
    real_fsync = builder_module.os.fsync
    failed = False

    def fail_after_rename(descriptor: int) -> None:
        nonlocal failed
        try:
            target = Path(f"/proc/self/fd/{descriptor}").resolve()
        except OSError:
            target = Path("/missing")
        if not failed and target == parent and destination.exists():
            failed = True
            raise OSError("synthetic parent fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(builder_module.os, "fsync", fail_after_rename)
    with pytest.raises(OSError, match="synthetic parent fsync failure"):
        build_base_release(
            repo_root=tmp_path,
            canonical_handoff=handoff,
            base_release_id="fsync_recovery",
        )
    assert destination.is_dir()
    monkeypatch.setattr(builder_module.os, "fsync", real_fsync)
    assert (
        build_base_release(
            repo_root=tmp_path,
            canonical_handoff=handoff,
            base_release_id="fsync_recovery",
        )
        == destination
    )
    assert (
        validate_base_release(
            repo_root=tmp_path, base_release_id="fsync_recovery"
        )["status"]
        == "valid"
    )


def test_stable_role_is_full_sha256_modulo_100() -> None:
    sample_id = "fixed-sample"
    bucket = int(hashlib.sha256(sample_id.encode()).hexdigest(), 16) % 100
    expected = "sft_only" if bucket < 70 else "grpo_only" if bucket < 90 else "shared"
    assert stable_train_role(sample_id) == expected
