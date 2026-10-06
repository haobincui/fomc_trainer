from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from jobs.generation import publish_paper_chk2_downstream_v6_recovery as publisher


class FakeTokenizer:
    bos_token = "<BOS>"
    eos_token = "<EOS>"
    bos_token_id = 1
    eos_token_id = 2

    def _encode(self, text: str) -> list[int]:
        ids = [self.bos_token_id]
        if text.startswith(self.bos_token):
            text = text[len(self.bos_token) :]
        has_eos = text.endswith(self.eos_token)
        if has_eos:
            text = text[: -len(self.eos_token)]
        ids.extend(10 + ord(character) for character in text)
        if has_eos:
            ids.append(self.eos_token_id)
        return ids

    def __call__(self, *, text: str):
        return {"input_ids": self._encode(text)}

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        truncation: bool = False,
        return_dict: bool = False,
    ):
        assert add_generation_prompt is True
        assert truncation is False
        rendered = self.bos_token
        for message in messages:
            rendered += f"[{message['role']}]\n{message['content']}\n"
        rendered += "[assistant]\n<think>\n"
        return self._encode(rendered) if tokenize else rendered


def _json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(publisher.canonical_json(row) + "\n" for row in rows),
        encoding="utf-8",
    )


def _signed(value: dict, field: str) -> dict:
    result = dict(value)
    result[field] = publisher.sha256_text(publisher.canonical_json(result))
    return result


def _source_row(sample_id: str, split: str, index: int):
    return SimpleNamespace(
        sample_id=sample_id,
        split=split,
        source_line_number=index,
        source_analysis=f"Source {sample_id}.",
    )


def _terminal(sample_id: str, split: str, index: int, *, passed: bool) -> dict:
    prompt = publisher.v5.render_user_prompt(f"Source {sample_id}.")
    response = "Check the facts.\n</think>\nEconomic activity increased moderately."
    status = publisher.v5.TERMINAL_PASS if passed else "STYLE_QUALITY_REJECT"
    return {
        "sample_id": sample_id,
        "split": split,
        "source_index": index,
        "meeting_date": "2020-01-01",
        "atomic_topic": "activity",
        "section_style_id": "style-1",
        "terminal_status": status,
        "rejection_stage": None if passed else "validator_b",
        "rejection_reasons": [] if passed else ["style_low_score"],
        "student_prompt": prompt if passed else None,
        "sft_response": response if passed else None,
        "source_analysis_sha256": "1" * 64,
        "provided_data_sha256": "2" * 64,
        "prompt_sha256": publisher.sha256_text(prompt) if passed else None,
        "response_sha256": publisher.sha256_text(response) if passed else None,
        "source_audit": {"machine_pass": True},
        "validator_a": {"machine_pass": True} if passed else None,
        "validator_b": {"machine_pass": passed},
        "repair_history": [],
        "lineage": {"training_only": True},
    }


def test_prompt_contract_is_pinned_to_the_approved_training_prompts() -> None:
    contract = publisher._student_prompt_contract()
    assert contract["system_prompt_sha256"] == publisher.EXPECTED_SYSTEM_PROMPT_SHA256
    assert (
        contract["user_prompt_template_sha256"]
        == publisher.EXPECTED_USER_PROMPT_TEMPLATE_SHA256
    )
    assert contract["response_boundary"] == "</think>"
    assert contract["opening_think_supplied_by_chat_template"] is True


def test_parent_binding_recomputes_directory_and_authorization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}\n", encoding="utf-8")
    observed = publisher._directory_fingerprint(model)
    monkeypatch.setattr(publisher, "EXPECTED_PARENT_MODEL_SHA256", observed["sha256"])
    monkeypatch.setattr(publisher, "REPO_ROOT", tmp_path)

    authorization = _signed(
        {
            "schema_version": "chk1-cp200-to-chk2-override-authorization-v1",
            "status": "authorized",
            "scope": {
                "target_stage": "chk2",
                "downstream_stages_allowed": ["chk2"],
                "further_downstream_stages_allowed": [],
            },
        },
        "authorization_sha256",
    )
    auth_path = tmp_path / "authorization.json"
    _json(auth_path, authorization)
    monkeypatch.setattr(
        publisher,
        "EXPECTED_PARENT_AUTHORIZATION_SHA256",
        authorization["authorization_sha256"],
    )
    monkeypatch.setattr(
        publisher,
        "EXPECTED_PARENT_AUTHORIZATION_FILE_SHA256",
        publisher.sha256_file(auth_path),
    )
    checkpoint_payload = {
        "schema_version": "chk1-cp200-merged-checkpoint-manifest-v1",
        "status": "ready_for_chk2_parent_under_explicit_override",
        "model_fingerprint": observed,
        "authorization": {
            "authorization_sha256": authorization["authorization_sha256"],
            "file_sha256": publisher.sha256_file(auth_path),
        },
        "scope": {"allowed_stage": "chk2", "further_downstream_stages_allowed": []},
    }
    checkpoint = {
        **checkpoint_payload,
        "integrity": {
            "payload_sha256": publisher.sha256_text(
                publisher.canonical_json(checkpoint_payload)
            )
        },
    }
    checkpoint_path = tmp_path / "checkpoint.json"
    _json(checkpoint_path, checkpoint)
    monkeypatch.setattr(
        publisher,
        "EXPECTED_PARENT_CHECKPOINT_MANIFEST_FILE_SHA256",
        publisher.sha256_file(checkpoint_path),
    )

    binding, sources = publisher._validate_parent_binding(
        model_root=model,
        authorization_path=auth_path,
        checkpoint_manifest_path=checkpoint_path,
    )
    assert binding["model_sha256"] == observed["sha256"]
    assert binding["allowed_stage"] == "chk2"
    assert set(sources) == {
        "parent_checkpoint_manifest.json",
        "parent_authorization.json",
    }


def test_projection_preserves_handoff_order_and_excludes_compatibility_reject(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer = FakeTokenizer()
    rows = {
        "train": (_source_row("pass-b", "train", 4), _source_row("pass-a", "train", 2)),
        "validation": (),
        "test": (_source_row("compat-reject", "test", 8),),
    }
    terminals = {
        "pass-b": _terminal("pass-b", "train", 4, passed=True),
        "pass-a": _terminal("pass-a", "train", 2, passed=True),
        "compat-reject": _terminal("compat-reject", "test", 8, passed=False),
    }
    expected_tokenizer = []
    for sample_id in ("pass-b", "pass-a"):
        record = terminals[sample_id]
        expected_tokenizer.append(
            {
                "sample_id": sample_id,
                "split": "train",
                **publisher.v5._tokenizer_replay(
                    row=next(
                        row for row in rows["train"] if row.sample_id == sample_id
                    ),
                    response=record["sft_response"],
                    tokenizer=tokenizer,
                ),
            }
        )
    audit = tmp_path / "tokenizer.jsonl"
    _jsonl(audit, expected_tokenizer)
    receipt = {
        "status": "complete",
        "schema_version": publisher.recover.RECEIPT_SCHEMA,
        "final_terminal_count": 3,
        "final_missing_ids": [],
        "source_handoff_manifest_sha256": "a" * 64,
        "compatibility_ids": ["compat-reject"],
    }
    verified = SimpleNamespace(
        receipt=receipt,
        state=SimpleNamespace(terminals=terminals),
        tokenizer_audit_path=audit,
    )
    monkeypatch.setattr(
        publisher.recover, "load_and_verify_recovery", lambda *args, **kwargs: verified
    )
    monkeypatch.setattr(publisher, "EXPECTED_SOURCE_ROWS", 3)
    monkeypatch.setattr(
        publisher,
        "EXPECTED_PASS_COUNTS",
        {"train": 2, "validation": 0, "test": 0},
    )
    monkeypatch.setattr(publisher, "EXPECTED_COMPATIBILITY_COUNT", 1)
    monkeypatch.setattr(
        publisher,
        "EXPECTED_COMPATIBILITY_ID_DIGEST",
        publisher.sha256_text(publisher.canonical_json(["compat-reject"])),
    )
    source = SimpleNamespace(
        signed_manifest_sha256="a" * 64,
        handoff=SimpleNamespace(prepared=rows),
    )

    projected = publisher._verify_and_project_recovery(
        recovery_root=tmp_path,
        original_root=tmp_path / "original",
        source=source,
        tokenizer=tokenizer,
    )
    assert [row["prompt"] for row in projected["pass_rows"]["train"]] == [
        terminals["pass-b"]["student_prompt"],
        terminals["pass-a"]["student_prompt"],
    ]
    assert projected["compatibility_ids"] == ["compat-reject"]


def test_projection_fails_closed_if_compatibility_row_becomes_pass(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    terminal = _terminal("compat", "train", 0, passed=True)
    receipt = {
        "status": "complete",
        "schema_version": publisher.recover.RECEIPT_SCHEMA,
        "final_terminal_count": 1,
        "final_missing_ids": [],
        "source_handoff_manifest_sha256": "a" * 64,
        "compatibility_ids": ["compat"],
    }
    monkeypatch.setattr(
        publisher.recover,
        "load_and_verify_recovery",
        lambda *args, **kwargs: SimpleNamespace(
            receipt=receipt,
            state=SimpleNamespace(terminals={"compat": terminal}),
            tokenizer_audit_path=tmp_path / "unused.jsonl",
        ),
    )
    monkeypatch.setattr(publisher, "EXPECTED_SOURCE_ROWS", 1)
    monkeypatch.setattr(publisher, "EXPECTED_COMPATIBILITY_COUNT", 1)
    monkeypatch.setattr(
        publisher,
        "EXPECTED_COMPATIBILITY_ID_DIGEST",
        publisher.sha256_text(publisher.canonical_json(["compat"])),
    )
    source = SimpleNamespace(
        signed_manifest_sha256="a" * 64,
        handoff=SimpleNamespace(prepared={"train": (), "validation": (), "test": ()}),
    )
    with pytest.raises(publisher.PublicationError, match="admitted"):
        publisher._verify_and_project_recovery(
            recovery_root=tmp_path,
            original_root=tmp_path,
            source=source,
            tokenizer=FakeTokenizer(),
        )


def test_existing_release_is_verified_without_touching_acquisition(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    release = tmp_path / "release"
    release.mkdir()
    sentinel = {"status": "complete"}
    monkeypatch.setattr(publisher, "verify_release", lambda *args, **kwargs: sentinel)
    monkeypatch.setattr(
        publisher.legacy_publisher,
        "_verify_source_handoff",
        lambda **kwargs: (_ for _ in ()).throw(AssertionError("inputs touched")),
    )
    assert publisher.publish_release(release_root=release) is sentinel


def test_atomic_publish_never_replaces_existing_release(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "new").write_text("new", encoding="utf-8")
    release = tmp_path / "release"
    release.mkdir()
    (release / "old").write_text("old", encoding="utf-8")
    with pytest.raises(publisher.PublicationError, match="already exists"):
        publisher._atomic_publish(staging, release)
    assert (release / "old").read_text(encoding="utf-8") == "old"
    assert staging.is_dir()


def test_cli_exposes_recovery_original_parent_and_verify_only() -> None:
    args = publisher._parse_args(
        [
            "--recovery-root",
            "/tmp/recovery",
            "--original-root",
            "/tmp/original",
            "--parent-model-root",
            "/tmp/parent",
            "--verify-only",
        ]
    )
    assert args.recovery_root == Path("/tmp/recovery")
    assert args.original_root == Path("/tmp/original")
    assert args.parent_model_root == Path("/tmp/parent")
    assert args.verify_only is True
