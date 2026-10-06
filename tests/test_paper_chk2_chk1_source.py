from __future__ import annotations

import json
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pytest

from jobs.generation.paper_chk2_chk1_source import (
    EXPECTED_MEETING_COUNTS,
    EXPECTED_SECTION_STYLE_IDS,
    EXPECTED_SPLIT_COUNTS,
    SourcePreparationError,
    build_prepare_manifest,
    build_train_only_style_bank,
    extract_final_answer,
    load_chk1_source_rows,
    sha256_file,
    sha256_text,
    verify_prepare_manifest,
    verify_style_bank,
)


@pytest.fixture(scope="module")
def exact_source_dataset():
    return load_chk1_source_rows()


@pytest.fixture(scope="module")
def exact_style_bank(exact_source_dataset):
    return build_train_only_style_bank(exact_source_dataset)


def test_extract_final_answer_requires_one_literal_boundary() -> None:
    assert (
        extract_final_answer("reasoning\n</think>\nFinal analysis.")
        == "Final analysis."
    )
    for invalid in (
        "reasoning only",
        "reasoning</think>Final analysis.",
        "reasoning\n</think>\nA\n</think>\nB",
        "\n</think>\nFinal analysis.",
        "reasoning\n</think>\n",
        "reasoning\n</think>\n answer with edge whitespace ",
    ):
        with pytest.raises(SourcePreparationError):
            extract_final_answer(invalid)


def test_exact_chk1_source_population_and_split_isolation(exact_source_dataset) -> None:
    dataset = exact_source_dataset
    assert len(dataset.rows) == 1743
    assert dataset.split_counts == EXPECTED_SPLIT_COUNTS
    assert dataset.meeting_counts == EXPECTED_MEETING_COUNTS
    assert dataset.sample_id_digest == (
        "e3ec34a76fa720488e9aee50ad39b6bd53114fa6ddcc5220078ffc7139d4b8db"
    )
    assert dataset.rows_digest == (
        "d03294231a3bde29cf87512f6473a79590704f429a1e06b9bf6c13b1bbce13bd"
    )

    assert [row.split for row in dataset.rows[:1354]] == ["train"] * 1354
    assert [row.split for row in dataset.rows[1354:1553]] == ["validation"] * 199
    assert [row.split for row in dataset.rows[1553:]] == ["test"] * 190
    assert [row.split_index for row in dataset.rows_for_split("validation")] == list(
        range(1, 200)
    )
    assert len({row.sample_id for row in dataset.rows}) == 1743
    assert {row.section_style_id for row in dataset.rows} == set(
        EXPECTED_SECTION_STYLE_IDS
    )

    meeting_sets = {
        split: {row.meeting_date for row in dataset.rows_for_split(split)}
        for split in EXPECTED_SPLIT_COUNTS
    }
    assert meeting_sets["train"].isdisjoint(meeting_sets["validation"])
    assert meeting_sets["train"].isdisjoint(meeting_sets["test"])
    assert meeting_sets["validation"].isdisjoint(meeting_sets["test"])


def test_candidate_answer_is_new_source_while_generation_answer_is_chain_anchor(
    exact_source_dataset,
) -> None:
    # Changed rows intentionally have a cleaned candidate final answer.  The
    # loader must expose that answer while retaining the generation answer only
    # as the first hop of the immutable lineage chain.
    changed = [
        row
        for row in exact_source_dataset.rows
        if row.source_analysis_sha256 != row.source_answer_sha256
    ]
    assert len(changed) == 201
    assert all(
        sha256_text(row.source_analysis) == row.source_analysis_sha256
        for row in changed
    )
    assert all(row.source_row_sha256 for row in exact_source_dataset.rows)


def test_train_only_style_bank_is_balanced_deterministic_and_hash_bound(
    exact_source_dataset,
    exact_style_bank,
) -> None:
    first = exact_style_bank
    second = build_train_only_style_bank(exact_source_dataset)
    assert first == second
    assert first.style_bank_sha256 == (
        "87125ba4b7346812ad33bf915c527c9c83a8acc8bda6be70050fddf64cbf6f5d"
    )
    assert first.train_meeting_count == 102
    assert Counter(row.section_style_id for row in first.exemplars) == {
        EXPECTED_SECTION_STYLE_IDS[0]: 3,
        EXPECTED_SECTION_STYLE_IDS[1]: 3,
    }
    assert all(row.selection_mode == "direct_style" for row in first.exemplars)
    assert all(20 <= row.word_count <= 400 for row in first.exemplars)
    assert len({row.text_sha256 for row in first.exemplars}) == 6

    train_meetings = {
        row.meeting_date for row in exact_source_dataset.rows if row.split == "train"
    }
    held_out = {
        row.meeting_date for row in exact_source_dataset.rows if row.split != "train"
    }
    assert all(row.meeting_date in train_meetings for row in first.exemplars)
    assert all(row.meeting_date not in held_out for row in first.exemplars)
    verify_style_bank(first, exact_source_dataset)


def test_style_bank_verifier_rejects_a_held_out_exemplar(
    exact_source_dataset,
    exact_style_bank,
) -> None:
    held_out_date = next(
        row.meeting_date
        for row in exact_source_dataset.rows
        if row.split == "validation"
    )
    bad_exemplar = replace(exact_style_bank.exemplars[0], meeting_date=held_out_date)
    bad_bank = replace(
        exact_style_bank,
        exemplars=(bad_exemplar, *exact_style_bank.exemplars[1:]),
    )
    with pytest.raises(SourcePreparationError):
        verify_style_bank(bad_bank, exact_source_dataset)


def test_prepare_manifest_is_reproducible_and_verifiable(
    exact_source_dataset,
    exact_style_bank,
) -> None:
    manifest = build_prepare_manifest(exact_source_dataset, exact_style_bank)
    assert manifest["prepare_manifest_sha256"] == (
        "33ae9cce58ecca2e8e4c1e2452060496603822fc584357bd76deb8608bbdb447"
    )
    assert manifest["upstream_chk1_candidate_repair_provenance"] == {
        "candidate_response_changed_rows": 237,
        "candidate_final_answer_changed_rows": 201,
        "source_rows": 1743,
    }
    assert manifest["invariants"] == {
        "source_analysis_is_exact_chk1_final_answer": True,
        "source_analysis_was_repaired": False,
        "source_analysis_was_repaired_scope": "paper_chk2_pipeline_only",
        "source_dataset_contains_upstream_chk1_repairs": True,
        "c8_used_for_training": False,
        "style_bank_train_only": True,
        "held_out_official_minutes_used": False,
    }
    verify_prepare_manifest(manifest, exact_source_dataset, exact_style_bank)
    tampered = dict(manifest)
    tampered["sample_id_digest"] = "0" * 64
    with pytest.raises(SourcePreparationError):
        verify_prepare_manifest(tampered, exact_source_dataset, exact_style_bank)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _make_small_source_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, str]]:
    candidate_root = tmp_path / "candidate"
    generation_root = tmp_path / "generation"
    styles = (
        EXPECTED_SECTION_STYLE_IDS[0],
        EXPECTED_SECTION_STYLE_IDS[1],
        EXPECTED_SECTION_STYLE_IDS[0],
    )
    all_sidecars: list[dict] = []

    for index, source_split in enumerate(("train", "eval", "test"), 1):
        prompt = f"prompt-{source_split}"
        provided = json.dumps({"evidence": source_split}, separators=(",", ":"))
        original_answer = f"Original source answer {source_split}."
        candidate_answer = f"Candidate final answer {source_split}."
        candidate_response = f"reasoning-{source_split}\n</think>\n{candidate_answer}"
        source_response_sha = sha256_text(f"source-response-{source_split}")
        sample_id = f"sample-{source_split}"
        generation = {
            "sample_id": sample_id,
            "split": source_split,
            "meeting_date": f"202{index}-01-01",
            "atomic_topic": f"topic-{source_split}",
            "section_style_id": styles[index - 1],
            "prompt_sha256": sha256_text(prompt),
            "provided_data_sha256": sha256_text(provided),
            "final_analysis_sha256": sha256_text(original_answer),
            "response_sha256": sha256_text(f"generation-response-{source_split}"),
        }
        sidecar = {
            "sample_id": sample_id,
            "split": source_split,
            "source_line_number": 1,
            "generation_manifest_line_number": 1,
            "prompt_sha256": sha256_text(prompt),
            "provided_data_sha256": sha256_text(provided),
            "candidate_answer_sha256": sha256_text(candidate_answer),
            "candidate_response_sha256": sha256_text(candidate_response),
            "new_response_sha256": sha256_text(candidate_response),
            "source_answer_sha256": sha256_text(original_answer),
            "source_response_sha256": source_response_sha,
            "old_response_sha256": source_response_sha,
        }
        _write_jsonl(
            candidate_root / "analysis_sft" / f"{source_split}.jsonl",
            [
                {
                    "prompt": prompt,
                    "provided_data": provided,
                    "response": candidate_response,
                }
            ],
        )
        _write_jsonl(
            generation_root / "manifests" / f"{source_split}.jsonl", [generation]
        )
        all_sidecars.append(sidecar)
    _write_jsonl(candidate_root / "audits/repair_manifest.jsonl", all_sidecars)

    paths = {
        "candidate/analysis_sft/train.jsonl": candidate_root
        / "analysis_sft/train.jsonl",
        "candidate/analysis_sft/eval.jsonl": candidate_root / "analysis_sft/eval.jsonl",
        "candidate/analysis_sft/test.jsonl": candidate_root / "analysis_sft/test.jsonl",
        "candidate/audits/repair_manifest.jsonl": candidate_root
        / "audits/repair_manifest.jsonl",
        "generation/manifests/train.jsonl": generation_root / "manifests/train.jsonl",
        "generation/manifests/eval.jsonl": generation_root / "manifests/eval.jsonl",
        "generation/manifests/test.jsonl": generation_root / "manifests/test.jsonl",
    }
    return (
        candidate_root,
        generation_root,
        {logical_name: sha256_file(path) for logical_name, path in paths.items()},
    )


def test_loader_rejects_broken_generation_to_sidecar_answer_hash_chain(
    tmp_path: Path,
) -> None:
    candidate_root, generation_root, bindings = _make_small_source_fixture(tmp_path)
    generation_path = generation_root / "manifests/train.jsonl"
    generation = json.loads(generation_path.read_text(encoding="utf-8"))
    generation["final_analysis_sha256"] = "f" * 64
    _write_jsonl(generation_path, [generation])
    bindings["generation/manifests/train.jsonl"] = sha256_file(generation_path)

    with pytest.raises(SourcePreparationError, match="generation final analysis"):
        load_chk1_source_rows(
            candidate_root,
            generation_root,
            expected_split_counts={"train": 1, "validation": 1, "test": 1},
            expected_meeting_counts={"train": 1, "validation": 1, "test": 1},
            expected_file_sha256=bindings,
        )
