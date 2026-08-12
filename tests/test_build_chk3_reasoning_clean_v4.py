from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from jobs.generation.build_chk3_reasoning_clean_v4 import (
    BOUNDARY,
    RELEASE_SCHEMA_VERSION,
    LocalReasoningCleanError,
    clean_reasoning,
    contamination_hits,
    run,
)
from jobs.generation.generate_chk3_sft_targets import (
    STUDENT_SYSTEM_PROMPT,
    canonical_json,
    render_user_prompt,
    sha256_file,
    sha256_text,
)


class FakeTokenizer:
    bos_token = "<s>"
    eos_token = "</s>"
    bos_token_id = 1
    eos_token_id = 2

    @staticmethod
    def _raw_ids(text: str) -> list[int]:
        pieces = __import__("re").findall(r"<s>|</s>|[^\s<]+|<", text)
        return [
            1
            if piece == "<s>"
            else 2
            if piece == "</s>"
            else 10 + sum(ord(char) for char in piece)
            for piece in pieces
        ]

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        ids = self._raw_ids(text)
        return ([self.bos_token_id] + ids) if add_special_tokens else ids

    def __call__(self, *, text: str):
        return {"input_ids": [self.bos_token_id] + self._raw_ids(text)}

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
        assert return_dict is False
        rendered = (
            self.bos_token
            + "SYS:"
            + str(messages[0]["content"])
            + "\nUSER:"
            + str(messages[1]["content"])
            + "\nASSISTANT:<think>\n"
        )
        return self._raw_ids(rendered) if tokenize else rendered


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(canonical_json(row) + "\n" for row in rows))


def _source_release(tmp_path: Path, *, reasoning: str) -> tuple[Path, str, str]:
    root = tmp_path / "source-v3"
    sample_id = "chk1-analysis-test"
    analysis = (
        "Employment increased 2 percent in May while output remained stable, "
        "and uncertainty about the next observation persisted."
    )
    minutes = (
        "Employment increased 2 percent in May, while output remained stable "
        "and uncertainty about the next observation persisted."
    )
    prompt = render_user_prompt(analysis)
    response = reasoning + BOUNDARY + minutes
    row = {"prompt": prompt, "response": response}
    manifest_row = {
        "schema_version": "chk3-minutes-training-release-v1",
        "sample_id": sample_id,
        "split": "train",
        "source_split": "train",
        "source_index": 0,
        "analysis_mode": "unchanged",
        "target_mode": "prior",
        "source_analysis_sha256": sha256_text(analysis),
        "analysis_sha256": sha256_text(analysis),
        "prompt_sha256": sha256_text(prompt),
        "response_sha256": sha256_text(response),
        "reasoning_sha256": sha256_text(reasoning),
        "minutes_sha256": sha256_text(minutes),
        "source_response_sha256": "a" * 64,
        "prompt_tokens": 1,
        "completion_tokens": 1,
        "total_tokens": 2,
        "reasoning_tokens": len(reasoning),
        "teacher": {"returned_model": "local-fixture"},
    }
    data_path = root / "minutes_alignment/train.jsonl"
    row_manifest_path = root / "minutes_alignment/manifests/train.jsonl"
    _write_jsonl(data_path, [row])
    _write_jsonl(row_manifest_path, [manifest_row])
    _write_json(root / "audits/data_quality.json", {"status": "passed"})
    config = {
        "dataset_name": "dataset/processed/retrain_v2/source-v3/minutes_alignment",
        "system_prompt": STUDENT_SYSTEM_PROMPT,
    }
    config_path = root / "chk3_minutes_sft.template.yaml"
    config_path.write_text(yaml.safe_dump(config, sort_keys=False))
    files: dict[str, dict] = {}
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        record = {
            "path": relative,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        if path.suffix == ".jsonl":
            record["rows"] = 1
        files[relative] = record
    release_manifest = {
        "schema_version": "chk3-minutes-training-release-v1",
        "release_id": "source-v3",
        "immutable": True,
        "quality_status": "passed",
        "files": files,
    }
    _write_json(root / "release_manifest.json", release_manifest)
    _write_json(
        root / "handoff.json",
        {
            "schema_version": "chk3-minutes-training-release-v1",
            "release_id": "source-v3",
            "immutable": True,
            "quality_status": "passed",
            "release_manifest": {
                "path": "release_manifest.json",
                "sha256": sha256_file(root / "release_manifest.json"),
            },
            "data_quality_audit": {
                "path": "audits/data_quality.json",
                "sha256": sha256_file(root / "audits/data_quality.json"),
            },
        },
    )
    return root, prompt, minutes


def test_cleanup_deletes_meta_citations_headings_bullets_and_duplicates() -> None:
    analysis = "Employment increased 2 percent in May."
    minutes = "Employment increased 2 percent in May."
    reasoning = """Plan:
- The analysis states that payroll employment increased 2 percent in May.
- Check the silent fidelity ledger and ensure every quantity is preserved.
- Payroll employment increased 2 percent in May (ev-acde1234).
Payroll employment increased 2 percent in May.
Possible draft: "Employment increased 2 percent in May."
Remove ev-citations; the output should be one formal paragraph."""

    result = clean_reasoning(reasoning, analysis=analysis, minutes=minutes)

    assert result.reasoning == "payroll employment increased 2 percent in May."
    assert {
        "heading",
        "bullet_marker",
        "citation_clause",
        "quoted_draft",
        "citation_process_clause",
    }.issubset(result.matched_rules)
    assert contamination_hits(
        result.reasoning, analysis=analysis, minutes=minutes
    ) == []


def test_cleanup_removes_exact_final_draft_and_exact_duplicate_sentences() -> None:
    analysis = "Activity declined in June and uncertainty remained elevated."
    minutes = "Activity declined in June, and uncertainty remained elevated."
    reasoning = (
        "Activity moved lower during June. Activity moved lower during June. "
        + minutes
        + " The comparison remained economically meaningful."
    )

    result = clean_reasoning(reasoning, analysis=analysis, minutes=minutes)

    assert result.reasoning == (
        "Activity moved lower during June. The comparison remained economically meaningful."
    )
    assert "quoted_final_minutes" in result.matched_rules
    assert "duplicate_sentence" in result.matched_rules


def test_cleanup_removes_reviewer_chatter_and_unmatched_quotes() -> None:
    result = clean_reasoning(
        (
            '"Employment growth slowed while uncertainty remained elevated. '
            "I'll produce a paragraph in FOMC Minutes style. This seems fine. "
            "No extra information, no rounding, and the modal verb may is not a month."
        ),
        analysis="Employment growth and uncertainty were discussed.",
        minutes="Employment gains moderated, and uncertainty remained elevated.",
    )

    assert result.reasoning == "Employment growth slowed while uncertainty remained elevated."
    assert "unmatched_quote" in result.matched_rules
    assert contamination_hits(
        result.reasoning,
        analysis="Employment growth and uncertainty were discussed.",
        minutes="Employment gains moderated, and uncertainty remained elevated.",
    ) == []


def test_undersized_cleanup_writes_needs_regeneration_and_never_publishes(
    tmp_path: Path,
) -> None:
    source, prompt, minutes = _source_release(
        tmp_path,
        reasoning=(
            "Plan: Check every quantity. Remove ev-citations. "
            "Employment rose in May."
        ),
    )
    release_parent = tmp_path / "releases"
    work_root = tmp_path / "work"

    summary = run(
        source_release=source,
        tokenizer_path=tmp_path / "unused-tokenizer",
        work_root=work_root,
        release_parent=release_parent,
        release_id="clean-v4",
        dry_run=False,
        tokenizer=FakeTokenizer(),
        expected_split_counts={"train": 1},
    )

    assert summary["status"] == "needs_regeneration"
    assert summary["needs_regeneration_rows"] == 1
    assert not (release_parent / "clean-v4").exists()
    need = json.loads((work_root / "needs_regeneration.jsonl").read_text())
    assert need["status"] == "needs_regeneration"
    assert need["prompt_sha256"] == sha256_text(prompt)
    assert need["final_answer_sha256"] == sha256_text(minutes)
    assert need["minutes_sha256"] == need["final_answer_sha256"]
    assert need["cleaned_reasoning_tokens"] < 64
    assert "reasoning_tokens:" in need["failure_reason"]


def test_clean_release_preserves_prompt_and_minutes_bytes_and_is_immutable(
    tmp_path: Path,
) -> None:
    source, prompt, minutes = _source_release(
        tmp_path,
        reasoning=(
            "The analysis states that employment increased 2 percent in May while "
            "output remained stable and uncertainty about the next observation persisted. "
            "Employment growth moved upward during May, whereas output conditions were "
            "unchanged and the outlook continued to be characterized by uncertainty. "
            "Payroll gains were broad enough to establish the direction of the movement, "
            "while the stable production reading supplied a contrasting condition. The "
            "temporal relationship placed both developments in May, and the uncertainty "
            "statement qualified how confidently later observations could extend the pattern. "
            "The economic sequence therefore combined a measured labor-market increase, an "
            "unchanged production condition, and an explicitly uncertain forward comparison. "
            "Check all quantities and remove ev-citations."
        ),
    )
    release_parent = tmp_path / "releases"
    work_root = tmp_path / "work"

    summary = run(
        source_release=source,
        tokenizer_path=tmp_path / "unused-tokenizer",
        work_root=work_root,
        release_parent=release_parent,
        release_id="clean-v4",
        dry_run=False,
        tokenizer=FakeTokenizer(),
        expected_split_counts={"train": 1},
    )

    assert summary["status"] == "complete"
    destination = release_parent / "clean-v4"
    published = json.loads((destination / "minutes_alignment/train.jsonl").read_text())
    assert published["prompt"] == prompt
    reasoning, final_minutes = published["response"].split(BOUNDARY, 1)
    assert final_minutes == minutes
    assert contamination_hits(
        reasoning,
        analysis=json.loads(prompt[len("Rewrite the following analysis as formal FOMC Minutes prose.\n\n") :])["analysis"],
        minutes=minutes,
    ) == []
    manifest = json.loads((destination / "release_manifest.json").read_text())
    assert manifest["schema_version"] == RELEASE_SCHEMA_VERSION
    assert manifest["immutable"] is True
    assert manifest["source"]["network_requests"] == 0
    assert destination.stat().st_mode & 0o222 == 0
    assert (destination / "minutes_alignment/train.jsonl").stat().st_mode & 0o222 == 0


def test_source_hash_mismatch_fails_before_any_output(tmp_path: Path) -> None:
    source, _, _ = _source_release(
        tmp_path,
        reasoning=(
            "Employment increased during May while output remained stable. "
            "Uncertainty around the next observation remained elevated."
        ),
    )
    with (source / "minutes_alignment/train.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")

    with pytest.raises(LocalReasoningCleanError, match="source file hash mismatch"):
        run(
            source_release=source,
            tokenizer_path=tmp_path / "unused-tokenizer",
            work_root=tmp_path / "work",
            release_parent=tmp_path / "releases",
            release_id="clean-v4",
            tokenizer=FakeTokenizer(),
            expected_split_counts={"train": 1},
        )

    assert not (tmp_path / "work").exists()
    assert not (tmp_path / "releases/clean-v4").exists()


def test_existing_destination_is_never_overwritten(tmp_path: Path) -> None:
    source, _, _ = _source_release(
        tmp_path,
        reasoning=(
            "Employment increased during May while output remained stable. "
            "Uncertainty around the next observation remained elevated and the economic "
            "comparison across the observations remained material. Payroll gains established "
            "the direction of the labor-market movement, while stable production supplied a "
            "contrasting economic condition for the period. The temporal relationship placed "
            "both developments in May, and uncertainty qualified the extent to which later "
            "observations could extend the pattern. The sequence combined a measured increase, "
            "an unchanged production condition, and a forward comparison that remained uncertain."
        ),
    )
    destination = tmp_path / "releases/clean-v4"
    destination.mkdir(parents=True)
    marker = destination / "owner.txt"
    marker.write_text("do not replace")

    with pytest.raises(LocalReasoningCleanError, match="already exists"):
        run(
            source_release=source,
            tokenizer_path=tmp_path / "unused-tokenizer",
            work_root=tmp_path / "work",
            release_parent=tmp_path / "releases",
            release_id="clean-v4",
            tokenizer=FakeTokenizer(),
            expected_split_counts={"train": 1},
        )

    assert marker.read_text() == "do not replace"
