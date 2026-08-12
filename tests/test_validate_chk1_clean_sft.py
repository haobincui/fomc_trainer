from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobs.retrain_v2 import repair_chk1_sft_release as repair
from jobs.retrain_v2 import validate_chk1_clean_sft as validator


REPO_ROOT = Path(__file__).resolve().parents[1]
LOCAL_SOURCE_RELEASE = (
    REPO_ROOT / "dataset/processed/retrain_v2/"
    "chk1_reasoning_compressed_flash_max_v1_20260805"
)
LOCAL_CANDIDATE = (
    REPO_ROOT / "output/data/retrain_v2/chk1/"
    "chk1_reasoning_compressed_flash_max_v2_clean_20260809_candidate"
)
LOCAL_TOKENIZER = REPO_ROOT / "models/DeepSeek-R1-Distill-Llama-8B"


class FakeTokenizer:
    bos_token = "<bos>"
    bos_token_id = 1
    eos_token = "<eos>"
    eos_token_id = 2

    def __init__(self, *, leading_bos: int = 1) -> None:
        self.leading_bos = leading_bos
        self._vocabulary: dict[str, int] = {}

    def _ordinary_id(self, token: str) -> int:
        if token not in self._vocabulary:
            self._vocabulary[token] = len(self._vocabulary) + 10
        return self._vocabulary[token]

    def _encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        ids: list[int] = []
        while text.startswith(self.bos_token):
            ids.append(self.bos_token_id)
            text = text[len(self.bos_token) :]
        if add_special_tokens and not ids:
            ids.append(self.bos_token_id)
        text = text.replace(self.eos_token, f" {self.eos_token} ")
        for token in text.split():
            ids.append(
                self.eos_token_id
                if token == self.eos_token
                else self._ordinary_id(token)
            )
        return ids

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        **_kwargs,
    ):
        assert add_generation_prompt is True
        rendered = self.bos_token * self.leading_bos
        rendered += " ".join(
            f"{message['role']}:{message['content']}" for message in messages
        )
        rendered += " assistant: "
        return (
            self._encode(rendered, add_special_tokens=False) if tokenize else rendered
        )

    def __call__(self, *, text: str):
        return {"input_ids": self._encode(text, add_special_tokens=True)}

    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        return self._encode(text, add_special_tokens=add_special_tokens)


def _reasoning(label: str, count: int = 512) -> str:
    return " ".join(f"{label}-reason-{index}" for index in range(count))


def _answer(label: str, count: int = 16) -> str:
    return " ".join(f"{label}-answer-{index}" for index in range(count)) + "."


def _canonical_json(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(_canonical_json(row) + "\n" for row in rows), encoding="utf-8"
    )


def _make_releases(
    tmp_path: Path,
    *,
    mutate=None,
) -> tuple[Path, Path, Path, dict[str, int]]:
    source = tmp_path / "source"
    clean = tmp_path / "clean"
    tokenizer_path = tmp_path / "tokenizer"
    tokenizer_path.mkdir()
    (tokenizer_path / "tokenizer.json").write_text("{}\n", encoding="utf-8")
    expected = {"train": 1, "eval": 1, "test": 1}
    manifest_rows: list[dict] = []
    for split_index, split in enumerate(repair.SPLITS):
        label = f"{split}-{split_index}"
        row = {
            "prompt": f"point-in-time prompt {label}",
            "provided_data": _canonical_json({"split": split, "value": split_index}),
            "response": _reasoning(label) + repair.BOUNDARY + _answer(label),
        }
        source_row = dict(row)
        clean_row = dict(row)
        if mutate is not None:
            mutate(split, clean_row)
        _write_jsonl(source / "analysis_sft" / f"{split}.jsonl", [source_row])
        _write_jsonl(clean / "analysis_sft" / f"{split}.jsonl", [clean_row])
        manifest_rows.append(
            {
                "sample_id": f"sample-{split}",
                "split": split,
                "source_line_number": 1,
                "prompt_sha256": repair.sha256_text(clean_row["prompt"]),
                "provided_data_sha256": repair.sha256_text(clean_row["provided_data"]),
                "new_response_sha256": repair.sha256_text(clean_row["response"]),
            }
        )
    _write_jsonl(clean / "audits/repair_manifest.jsonl", manifest_rows)
    return source, clean, tokenizer_path, expected


def test_validator_passes_and_never_copies_training_text(tmp_path: Path) -> None:
    source, clean, tokenizer_path, expected = _make_releases(tmp_path)
    report = validator.validate_clean_sft(
        source_release=source,
        clean_release=clean,
        tokenizer_path=tokenizer_path,
        max_length=2000,
        expected_split_counts=expected,
        tokenizer=FakeTokenizer(),
    )

    assert report["status"] == "passed"
    assert report["counts"]["observed_rows"] == 3
    assert report["counts"]["prompt_hashes_unchanged"] == 3
    assert report["counts"]["provided_data_hashes_unchanged"] == 3
    assert report["counts"]["truncated_rows"] == 0
    assert report["issues"] == []
    assert all(report["quality_gates"].values())
    serialized = json.dumps(report, ensure_ascii=False)
    assert "train-0-reason-0" not in serialized
    assert "point-in-time prompt train-0" not in serialized


def test_validator_reports_hash_drift_contamination_and_cross_split_duplicate(
    tmp_path: Path,
) -> None:
    train_prompt = "point-in-time prompt train-0"
    secret = "NEVER-COPY-THIS-CONTAMINATED-TEXT"

    def mutate(split: str, row: dict[str, str]) -> None:
        if split == "eval":
            row["prompt"] = train_prompt
        if split == "train":
            reasoning, answer = row["response"].split(repair.BOUNDARY)
            answer = answer + f" {secret} ev-ab12"
            row["response"] = reasoning + repair.BOUNDARY + answer

    source, clean, tokenizer_path, expected = _make_releases(tmp_path, mutate=mutate)
    report = validator.validate_clean_sft(
        source_release=source,
        clean_release=clean,
        tokenizer_path=tokenizer_path,
        max_length=2000,
        expected_split_counts=expected,
        tokenizer=FakeTokenizer(),
    )

    assert report["status"] == "failed"
    assert report["issue_counts"]["source_prompt_hash_mismatch"] == 1
    assert report["issue_counts"]["evidence_id_contamination"] == 1
    assert report["issue_counts"]["cross_split_duplicate"] >= 2
    serialized = json.dumps(report, ensure_ascii=False)
    assert secret not in serialized
    duplicate = next(
        issue for issue in report["issues"] if issue["code"] == "cross_split_duplicate"
    )
    assert {location["split"] for location in duplicate["locations"]} == {
        "train",
        "eval",
    }


def test_validator_fails_closed_on_renderer_double_bos_and_overflow(
    tmp_path: Path,
) -> None:
    source, clean, tokenizer_path, expected = _make_releases(tmp_path)
    broken_renderer = validator.validate_clean_sft(
        source_release=source,
        clean_release=clean,
        tokenizer_path=tokenizer_path,
        max_length=2000,
        expected_split_counts=expected,
        tokenizer=FakeTokenizer(leading_bos=2),
    )
    overflow = validator.validate_clean_sft(
        source_release=source,
        clean_release=clean,
        tokenizer_path=tokenizer_path,
        max_length=100,
        expected_split_counts=expected,
        tokenizer=FakeTokenizer(),
    )

    assert broken_renderer["status"] == "failed"
    assert broken_renderer["issue_counts"]["sft_tokenization_contract"] == 3
    assert overflow["status"] == "failed"
    assert overflow["issue_counts"]["total_length_overflow"] == 3
    assert overflow["counts"]["truncated_rows"] == 3


def test_validator_detects_strict_periodic_response_tail(tmp_path: Path) -> None:
    repeated_unit = [f"repeat-{index}" for index in range(16)]

    def mutate(split: str, row: dict[str, str]) -> None:
        if split == "test":
            reasoning, _answer_text = row["response"].split(repair.BOUNDARY)
            periodic_answer = " ".join(repeated_unit * 3)
            row["response"] = reasoning + repair.BOUNDARY + periodic_answer

    source, clean, tokenizer_path, expected = _make_releases(tmp_path, mutate=mutate)
    report = validator.validate_clean_sft(
        source_release=source,
        clean_release=clean,
        tokenizer_path=tokenizer_path,
        max_length=2000,
        expected_split_counts=expected,
        tokenizer=FakeTokenizer(),
    )

    assert report["status"] == "failed"
    assert report["issue_counts"]["strict_periodic_tail"] == 1


def test_cli_writes_failed_receipt_and_returns_nonzero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source, clean, tokenizer_path, _expected = _make_releases(tmp_path)
    output = tmp_path / "receipts" / "validation.json"
    monkeypatch.setattr(validator, "_load_tokenizer", lambda _path: FakeTokenizer())

    return_code = validator.main(
        [
            "--source-release",
            str(source),
            "--clean-release",
            str(clean),
            "--model-tokenizer",
            str(tokenizer_path),
            "--max-length",
            "100",
            "--output",
            str(output),
        ]
    )

    receipt = json.loads(output.read_text(encoding="utf-8"))
    assert return_code == 1
    assert receipt["status"] == "failed"
    assert receipt["schema_version"] == validator.SCHEMA_VERSION
    assert receipt["validation_sha256"]


@pytest.mark.skipif(
    not LOCAL_SOURCE_RELEASE.exists()
    or not LOCAL_CANDIDATE.exists()
    or not LOCAL_TOKENIZER.exists(),
    reason="local sealed release, clean candidate, or DeepSeek tokenizer unavailable",
)
def test_local_clean_candidate_passes_full_real_tokenizer_gate() -> None:
    pytest.importorskip("transformers")
    report = validator.validate_clean_sft(
        source_release=LOCAL_SOURCE_RELEASE,
        clean_release=LOCAL_CANDIDATE,
        tokenizer_path=LOCAL_TOKENIZER,
        max_length=4608,
    )

    assert report["status"] == "passed", report["issue_counts"]
    assert report["counts"]["observed_rows"] == 1743
    assert report["counts"]["truncated_rows"] == 0
    assert report["token_statistics"]["max_total_tokens"] <= 4608
    assert (
        report["token_statistics"]["distributions"]["total"]["max"]
        == report["token_statistics"]["max_total_tokens"]
    )
    assert all(report["quality_gates"].values())
