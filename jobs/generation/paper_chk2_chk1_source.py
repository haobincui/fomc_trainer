"""Fail-closed source preparation for the paper chk-2 Minutes branch.

This module deliberately stops before any model request.  It binds the exact
chk-1 SFT candidate to its repair sidecar and the original generation
manifests, extracts only the final answer after the literal ``</think>``
boundary, and builds a small train-only official-Minutes style bank.

The candidate rows do not contain sample identifiers.  Their line identities
are therefore recovered from the repair sidecar; every subsequent join is by
``sample_id`` and is protected by the two-hop content-hash chain.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Mapping, Sequence

from jobs.retrain_v2.chk1.style_guide import stable_section_style_id


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CANDIDATE_ROOT = (
    REPO_ROOT / "output/data/retrain_v2/chk1/"
    "chk1_reasoning_compressed_flash_max_v2_clean_20260809_candidate"
)
DEFAULT_GENERATION_ROOT = REPO_ROOT / "output/data/retrain_v2/chk1/generation_full_v7"
DEFAULT_OFFICIAL_TRAIN_CORPUS = (
    REPO_ROOT / "dataset/processed/train/minutes_alignment/train_manifest.jsonl"
)

SOURCE_SCHEMA_VERSION = "paper-chk2-chk1-final-analysis-source-v1"
STYLE_BANK_SCHEMA_VERSION = "paper-chk2-train-only-minutes-style-bank-v1"
PREPARE_MANIFEST_SCHEMA_VERSION = "paper-chk2-chk1-source-prepare-v1"
THINK_BOUNDARY = "\n</think>\n"
SOURCE_SPLITS = ("train", "eval", "test")
SPLIT_MAP = {"train": "train", "eval": "validation", "test": "test"}
EXPECTED_SPLIT_COUNTS = {"train": 1354, "validation": 199, "test": 190}
EXPECTED_MEETING_COUNTS = {"train": 102, "validation": 13, "test": 13}
EXPECTED_SECTION_STYLE_IDS = (
    "section-style-v1-5d20dc44912c64a3fd59",
    "section-style-v1-7e6c09cbafe5336e7c73",
)

# These pins make the default loader mean the exact chk-1 cp200 training
# candidate, not merely any directory with a compatible shape.
EXPECTED_SOURCE_FILE_SHA256 = {
    "candidate/analysis_sft/train.jsonl": "f3f4b37ae225c4d4b8971212da9faaf0467cda80c0a301ffa4dc8bd9c6a32ddc",
    "candidate/analysis_sft/eval.jsonl": "e1622d22d5b14023128d4d3f4e3f377d40d8d5eb5cfeb1d5e3877af671af6eb4",
    "candidate/analysis_sft/test.jsonl": "60462438d7e72b39cd45a30e8964fb5b04c7f00d1b2749bc3db96459598300b1",
    "candidate/audits/repair_manifest.jsonl": "cd7dee86792c8190a5daceb99612c36a0396506f7dfa4da551a6f41cbb6c3d0b",
    "generation/manifests/train.jsonl": "53c04c2e4513a70c17a26efa74acdfe7c07bdcab5237bdb0e33a091105d06231",
    "generation/manifests/eval.jsonl": "b03aeb7a4e02b8df0cc8433a1c2d3ec7276e8c327647d24f7e73ea308ecf031e",
    "generation/manifests/test.jsonl": "f3c283753c1d5e8a9d6d5f22d4603d3de588ac42a14d12894f84983982f5bb04",
}
EXPECTED_OFFICIAL_TRAIN_CORPUS_SHA256 = (
    "623e09f9c7dac2ecc8657c6eab92574df4ac42da122ddcaadb7fa47bc90eb338"
)
EXPECTED_STYLE_BANK_SHA256 = (
    "87125ba4b7346812ad33bf915c527c9c83a8acc8bda6be70050fddf64cbf6f5d"
)

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_WORD_RE = re.compile(r"[A-Za-z0-9]+(?:[\-'’][A-Za-z0-9]+)*")
_LIST_LINE_RE = re.compile(r"^\s*(?:[-*•]|\(?\d+[.)]|[A-Z][.)])\s+")
_HEADING_RE = re.compile(
    r"^\s*(?:minutes of the federal open market committee|"
    r"staff review|participants['’]? views|committee policy action)\s*$",
    re.IGNORECASE,
)
_POLICY_RE = re.compile(
    r"\b(?:by unanimous vote|policy directive|target range for the federal funds|"
    r"committee (?:voted|approved|decided|authorized|directed|adopted)|"
    r"board (?:voted|approved|authorized)|ratified (?:these|the) transactions)\b",
    re.IGNORECASE,
)
_POLICY_SUBJECT_RE = re.compile(
    r"\b(?:monetary policy|policy action|policy decision|policy stance|"
    r"federal funds rate|open market operations|asset purchase program|"
    r"policy accommodation|discount rate)\b",
    re.IGNORECASE,
)
_ADMIN_RE = re.compile(
    r"\b(?:meeting (?:convened|adjourned)|attendance|secretary(?:'s)? note|"
    r"minutes of the previous meeting|elected as|appointment of|agenda item)\b",
    re.IGNORECASE,
)


class SourcePreparationError(ValueError):
    """Raised when source lineage or train-only isolation fails closed."""


@dataclass(frozen=True, slots=True)
class ArtifactBinding:
    logical_name: str
    path: str
    bytes: int
    rows: int
    sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class PreparedSourceRow:
    sample_id: str
    split: str
    source_split: str
    split_index: int
    source_line_number: int
    generation_manifest_line_number: int
    meeting_date: str
    atomic_topic: str
    section_style_id: str
    prompt: str
    provided_data: str
    source_analysis: str
    prompt_sha256: str
    provided_data_sha256: str
    source_analysis_sha256: str
    candidate_response_sha256: str
    source_answer_sha256: str
    source_response_sha256: str
    generation_manifest_row_sha256: str
    source_row_sha256: str

    def to_dict(self, *, include_text: bool = True) -> dict[str, Any]:
        result = asdict(self)
        if not include_text:
            result.pop("prompt")
            result.pop("provided_data")
            result.pop("source_analysis")
        return result


@dataclass(frozen=True, slots=True)
class PreparedSourceDataset:
    schema_version: str
    rows: tuple[PreparedSourceRow, ...]
    artifacts: tuple[ArtifactBinding, ...]
    split_counts: Mapping[str, int]
    meeting_counts: Mapping[str, int]
    sample_id_digest: str
    rows_digest: str

    def rows_for_split(self, split: str) -> tuple[PreparedSourceRow, ...]:
        return tuple(row for row in self.rows if row.split == split)

    def binding_map(self) -> dict[str, dict[str, Any]]:
        return {binding.logical_name: binding.to_dict() for binding in self.artifacts}


@dataclass(frozen=True, slots=True)
class StyleExemplar:
    exemplar_id: str
    section_style_id: str
    source_section_style_id: str
    meeting_date: str
    source_sample_id: str
    source_row_index: int
    source_corpus_line_number: int
    selection_mode: str
    selection_rank_sha256: str
    word_count: int
    text: str
    text_sha256: str
    source_row_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class StyleBank:
    schema_version: str
    source: ArtifactBinding
    train_meeting_count: int
    target_section_style_ids: tuple[str, ...]
    exemplars_per_style: int
    exemplars: tuple[StyleExemplar, ...]
    style_bank_sha256: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "source": self.source.to_dict(),
            "train_meeting_count": self.train_meeting_count,
            "target_section_style_ids": list(self.target_section_style_ids),
            "exemplars_per_style": self.exemplars_per_style,
            "exemplars": [row.to_dict() for row in self.exemplars],
            "style_bank_sha256": self.style_bank_sha256,
        }


def _canonical_json_bytes(value: Any) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    except (TypeError, ValueError) as exc:
        raise SourcePreparationError(
            f"value is not canonical finite JSON: {exc}"
        ) from exc


def sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json_bytes(value)).hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _portable_path(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def _required_text(row: Mapping[str, Any], field: str, *, label: str) -> str:
    value = row.get(field)
    if not isinstance(value, str) or not value.strip():
        raise SourcePreparationError(f"{label}.{field} must be non-empty text")
    return value


def _required_sha(row: Mapping[str, Any], field: str, *, label: str) -> str:
    value = _required_text(row, field, label=label)
    if not _SHA256_RE.fullmatch(value):
        raise SourcePreparationError(f"{label}.{field} is not a lowercase SHA-256")
    return value


def _read_jsonl(path: Path, *, label: str) -> list[tuple[int, dict[str, Any]]]:
    if not path.is_file():
        raise SourcePreparationError(f"missing {label}: {path}")
    rows: list[tuple[int, dict[str, Any]]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                raise SourcePreparationError(
                    f"blank line in {label} at line {line_number}"
                )
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SourcePreparationError(
                    f"invalid JSON in {label} at line {line_number}: {exc}"
                ) from exc
            if not isinstance(row, dict):
                raise SourcePreparationError(
                    f"{label} line {line_number} must be a JSON object"
                )
            rows.append((line_number, row))
    return rows


def _bind_jsonl(path: Path, logical_name: str, rows: int) -> ArtifactBinding:
    return ArtifactBinding(
        logical_name=logical_name,
        path=_portable_path(path),
        bytes=path.stat().st_size,
        rows=rows,
        sha256=sha256_file(path),
    )


def extract_final_answer(response: str) -> str:
    """Return the exact final answer after one literal native-think boundary."""

    if not isinstance(response, str):
        raise SourcePreparationError("response must be text")
    if response.count(THINK_BOUNDARY) != 1 or response.count("</think>") != 1:
        raise SourcePreparationError(
            "response must contain exactly one literal '\\n</think>\\n' boundary"
        )
    reasoning, answer = response.split(THINK_BOUNDARY, 1)
    if not reasoning.strip():
        raise SourcePreparationError("response reasoning before </think> is empty")
    if not answer.strip() or answer != answer.strip():
        raise SourcePreparationError(
            "final answer must be non-empty with no edge whitespace"
        )
    return answer


def _validate_date(value: str, *, label: str) -> str:
    try:
        date.fromisoformat(value)
    except ValueError as exc:
        raise SourcePreparationError(
            f"{label} is not an ISO meeting date: {value!r}"
        ) from exc
    return value


def _assert_hash(actual: str, expected: str, *, label: str) -> None:
    if actual != expected:
        raise SourcePreparationError(
            f"{label} hash mismatch: expected {expected}, observed {actual}"
        )


def _source_row_digest_payload(values: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: values[key]
        for key in (
            "sample_id",
            "split",
            "source_split",
            "split_index",
            "source_line_number",
            "generation_manifest_line_number",
            "meeting_date",
            "atomic_topic",
            "section_style_id",
            "prompt_sha256",
            "provided_data_sha256",
            "source_analysis_sha256",
            "candidate_response_sha256",
            "source_answer_sha256",
            "source_response_sha256",
            "generation_manifest_row_sha256",
        )
    }


def load_chk1_source_rows(
    candidate_root: str | Path = DEFAULT_CANDIDATE_ROOT,
    generation_root: str | Path = DEFAULT_GENERATION_ROOT,
    *,
    expected_split_counts: Mapping[str, int] = EXPECTED_SPLIT_COUNTS,
    expected_meeting_counts: Mapping[str, int] = EXPECTED_MEETING_COUNTS,
    expected_file_sha256: Mapping[str, str] = EXPECTED_SOURCE_FILE_SHA256,
) -> PreparedSourceDataset:
    """Load and verify the exact chk-1 final-answer source population."""

    if set(expected_split_counts) != set(EXPECTED_SPLIT_COUNTS):
        raise SourcePreparationError(
            "expected_split_counts must define the three final splits"
        )
    if set(expected_meeting_counts) != set(EXPECTED_MEETING_COUNTS):
        raise SourcePreparationError(
            "expected_meeting_counts must define the three final splits"
        )
    candidate_root = Path(candidate_root)
    generation_root = Path(generation_root)
    artifacts: list[ArtifactBinding] = []

    repair_path = candidate_root / "audits/repair_manifest.jsonl"
    repair_rows = _read_jsonl(repair_path, label="repair manifest")
    repair_binding = _bind_jsonl(
        repair_path, "candidate/audits/repair_manifest.jsonl", len(repair_rows)
    )
    artifacts.append(repair_binding)

    sidecar_by_location: dict[tuple[str, int], dict[str, Any]] = {}
    seen_sidecar_ids: set[str] = set()
    for manifest_line, row in repair_rows:
        label = f"repair manifest line {manifest_line}"
        sample_id = _required_text(row, "sample_id", label=label)
        source_split = _required_text(row, "split", label=label)
        source_line = row.get("source_line_number")
        if source_split not in SOURCE_SPLITS:
            raise SourcePreparationError(f"{label} has invalid split {source_split!r}")
        if (
            isinstance(source_line, bool)
            or not isinstance(source_line, int)
            or source_line < 1
        ):
            raise SourcePreparationError(f"{label}.source_line_number must be positive")
        if sample_id in seen_sidecar_ids:
            raise SourcePreparationError(
                f"duplicate repair-manifest sample_id: {sample_id}"
            )
        location = (source_split, source_line)
        if location in sidecar_by_location:
            raise SourcePreparationError(
                f"duplicate repair-manifest source location: {location}"
            )
        seen_sidecar_ids.add(sample_id)
        sidecar_by_location[location] = row

    generation_by_id: dict[str, tuple[int, dict[str, Any]]] = {}
    for source_split in SOURCE_SPLITS:
        path = generation_root / "manifests" / f"{source_split}.jsonl"
        raw_rows = _read_jsonl(path, label=f"generation {source_split} manifest")
        artifacts.append(
            _bind_jsonl(
                path, f"generation/manifests/{source_split}.jsonl", len(raw_rows)
            )
        )
        for line_number, row in raw_rows:
            label = f"generation {source_split} line {line_number}"
            sample_id = _required_text(row, "sample_id", label=label)
            if row.get("split") != source_split:
                raise SourcePreparationError(f"{label} split disagrees with its file")
            if sample_id in generation_by_id:
                raise SourcePreparationError(
                    f"duplicate generation sample_id: {sample_id}"
                )
            generation_by_id[sample_id] = (line_number, row)

    prepared: list[PreparedSourceRow] = []
    selected_ids: set[str] = set()
    for source_split in SOURCE_SPLITS:
        path = candidate_root / "analysis_sft" / f"{source_split}.jsonl"
        candidate_rows = _read_jsonl(path, label=f"candidate {source_split}")
        artifacts.append(
            _bind_jsonl(
                path,
                f"candidate/analysis_sft/{source_split}.jsonl",
                len(candidate_rows),
            )
        )
        mapped_split = SPLIT_MAP[source_split]
        expected_count = expected_split_counts.get(mapped_split)
        if expected_count is None or len(candidate_rows) != expected_count:
            raise SourcePreparationError(
                f"candidate {source_split} count mismatch: expected {expected_count}, "
                f"observed {len(candidate_rows)}"
            )

        for line_number, candidate in candidate_rows:
            label = f"candidate {source_split} line {line_number}"
            if set(candidate) != {"prompt", "provided_data", "response"}:
                raise SourcePreparationError(
                    f"{label} must have exact prompt/provided_data/response schema"
                )
            sidecar = sidecar_by_location.get((source_split, line_number))
            if sidecar is None:
                raise SourcePreparationError(f"{label} has no repair-sidecar identity")
            sample_id = _required_text(sidecar, "sample_id", label=label)
            if sample_id in selected_ids:
                raise SourcePreparationError(
                    f"duplicate selected sample_id: {sample_id}"
                )
            selected_ids.add(sample_id)
            generation_pair = generation_by_id.get(sample_id)
            if generation_pair is None:
                raise SourcePreparationError(
                    f"{label} sample_id is absent from generation manifest"
                )
            generation_line, generation = generation_pair
            if generation.get("split") != source_split:
                raise SourcePreparationError(f"{label} generation split mismatch")
            declared_generation_line = sidecar.get("generation_manifest_line_number")
            if declared_generation_line != generation_line:
                raise SourcePreparationError(
                    f"{label} generation-manifest line binding mismatch"
                )

            prompt = _required_text(candidate, "prompt", label=label)
            provided_data = _required_text(candidate, "provided_data", label=label)
            response = _required_text(candidate, "response", label=label)
            try:
                provided_object = json.loads(provided_data)
            except json.JSONDecodeError as exc:
                raise SourcePreparationError(
                    f"{label}.provided_data is invalid JSON"
                ) from exc
            if not isinstance(provided_object, dict):
                raise SourcePreparationError(
                    f"{label}.provided_data must encode an object"
                )
            source_analysis = extract_final_answer(response)

            prompt_sha = sha256_text(prompt)
            provided_sha = sha256_text(provided_data)
            candidate_response_sha = sha256_text(response)
            candidate_answer_sha = sha256_text(source_analysis)
            for owner, row in (("sidecar", sidecar), ("generation", generation)):
                _assert_hash(
                    prompt_sha,
                    _required_sha(row, "prompt_sha256", label=f"{label} {owner}"),
                    label=f"{label} {owner} prompt",
                )
                _assert_hash(
                    provided_sha,
                    _required_sha(
                        row, "provided_data_sha256", label=f"{label} {owner}"
                    ),
                    label=f"{label} {owner} provided_data",
                )
            _assert_hash(
                candidate_answer_sha,
                _required_sha(sidecar, "candidate_answer_sha256", label=label),
                label=f"{label} candidate answer",
            )
            _assert_hash(
                candidate_response_sha,
                _required_sha(sidecar, "candidate_response_sha256", label=label),
                label=f"{label} candidate response",
            )
            _assert_hash(
                candidate_response_sha,
                _required_sha(sidecar, "new_response_sha256", label=label),
                label=f"{label} new response",
            )
            source_answer_sha = _required_sha(
                sidecar, "source_answer_sha256", label=label
            )
            _assert_hash(
                source_answer_sha,
                _required_sha(generation, "final_analysis_sha256", label=label),
                label=f"{label} generation final analysis",
            )
            source_response_sha = _required_sha(
                sidecar, "source_response_sha256", label=label
            )
            _assert_hash(
                source_response_sha,
                _required_sha(sidecar, "old_response_sha256", label=label),
                label=f"{label} sidecar old response",
            )

            meeting_date = _validate_date(
                _required_text(generation, "meeting_date", label=label),
                label=f"{label}.meeting_date",
            )
            atomic_topic = _required_text(generation, "atomic_topic", label=label)
            section_style_id = _required_text(
                generation, "section_style_id", label=label
            )
            generation_row_sha = sha256_json(generation)
            values: dict[str, Any] = {
                "sample_id": sample_id,
                "split": mapped_split,
                "source_split": source_split,
                "split_index": line_number,
                "source_line_number": line_number,
                "generation_manifest_line_number": generation_line,
                "meeting_date": meeting_date,
                "atomic_topic": atomic_topic,
                "section_style_id": section_style_id,
                "prompt": prompt,
                "provided_data": provided_data,
                "source_analysis": source_analysis,
                "prompt_sha256": prompt_sha,
                "provided_data_sha256": provided_sha,
                "source_analysis_sha256": candidate_answer_sha,
                "candidate_response_sha256": candidate_response_sha,
                "source_answer_sha256": source_answer_sha,
                "source_response_sha256": source_response_sha,
                "generation_manifest_row_sha256": generation_row_sha,
            }
            values["source_row_sha256"] = sha256_json(
                _source_row_digest_payload(values)
            )
            prepared.append(PreparedSourceRow(**values))

    if selected_ids != seen_sidecar_ids:
        missing = sorted(seen_sidecar_ids - selected_ids)
        extra = sorted(selected_ids - seen_sidecar_ids)
        raise SourcePreparationError(
            f"candidate/sidecar identity partition mismatch: missing={missing[:3]}, extra={extra[:3]}"
        )
    if len(prepared) != sum(expected_split_counts.values()):
        raise SourcePreparationError(
            "prepared source row total disagrees with split contract"
        )

    meeting_sets = {
        split: {row.meeting_date for row in prepared if row.split == split}
        for split in EXPECTED_SPLIT_COUNTS
    }
    for split, meetings in meeting_sets.items():
        expected = expected_meeting_counts.get(split)
        if expected is None or len(meetings) != expected:
            raise SourcePreparationError(
                f"{split} meeting count mismatch: expected {expected}, observed {len(meetings)}"
            )
    split_names = tuple(meeting_sets)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            overlap = meeting_sets[left] & meeting_sets[right]
            if overlap:
                raise SourcePreparationError(
                    f"meeting split leakage between {left} and {right}: {sorted(overlap)}"
                )
    observed_styles = tuple(sorted({row.section_style_id for row in prepared}))
    if observed_styles != EXPECTED_SECTION_STYLE_IDS:
        raise SourcePreparationError(
            f"unexpected section-style population: {observed_styles!r}"
        )

    binding_map = {binding.logical_name: binding for binding in artifacts}
    if set(expected_file_sha256) != set(binding_map):
        raise SourcePreparationError(
            "expected_file_sha256 keys must exactly match the seven bound source artifacts"
        )
    for logical_name, expected_sha in expected_file_sha256.items():
        if not _SHA256_RE.fullmatch(expected_sha):
            raise SourcePreparationError(f"invalid expected hash for {logical_name}")
        _assert_hash(binding_map[logical_name].sha256, expected_sha, label=logical_name)

    split_counts = {
        split: sum(row.split == split for row in prepared)
        for split in EXPECTED_SPLIT_COUNTS
    }
    meeting_counts = {
        split: len(meeting_sets[split]) for split in EXPECTED_SPLIT_COUNTS
    }
    sample_id_digest = sha256_json([row.sample_id for row in prepared])
    rows_digest = sha256_json([row.source_row_sha256 for row in prepared])
    return PreparedSourceDataset(
        schema_version=SOURCE_SCHEMA_VERSION,
        rows=tuple(prepared),
        artifacts=tuple(sorted(artifacts, key=lambda item: item.logical_name)),
        split_counts=split_counts,
        meeting_counts=meeting_counts,
        sample_id_digest=sample_id_digest,
        rows_digest=rows_digest,
    )


def _normalize_exemplar(text: str) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip()


def _eligible_exemplar(raw_text: str, normalized: str) -> bool:
    word_count = len(_WORD_RE.findall(normalized))
    if word_count < 20 or word_count > 400:
        return False
    if "\n\n" in raw_text or _HEADING_RE.fullmatch(normalized):
        return False
    if any(_LIST_LINE_RE.match(line) for line in raw_text.splitlines() if line.strip()):
        return False
    if (
        _POLICY_RE.search(normalized)
        or _POLICY_SUBJECT_RE.search(normalized)
        or _ADMIN_RE.search(normalized)
    ):
        return False
    return True


def _style_bank_digest_payload(bank: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in bank.items() if key != "style_bank_sha256"}


def build_train_only_style_bank(
    dataset: PreparedSourceDataset | Sequence[PreparedSourceRow],
    official_corpus_path: str | Path = DEFAULT_OFFICIAL_TRAIN_CORPUS,
    *,
    exemplars_per_style: int = 3,
    expected_source_sha256: str = EXPECTED_OFFICIAL_TRAIN_CORPUS_SHA256,
) -> StyleBank:
    """Select deterministic official-Minutes exemplars from train meetings only."""

    rows = (
        dataset.rows if isinstance(dataset, PreparedSourceDataset) else tuple(dataset)
    )
    if isinstance(exemplars_per_style, bool) or exemplars_per_style < 1:
        raise SourcePreparationError("exemplars_per_style must be a positive integer")
    train_meetings = {row.meeting_date for row in rows if row.split == "train"}
    held_out_meetings = {row.meeting_date for row in rows if row.split != "train"}
    if not train_meetings or train_meetings & held_out_meetings:
        raise SourcePreparationError(
            "source rows do not provide isolated train meetings"
        )
    target_style_ids = tuple(sorted({row.section_style_id for row in rows}))
    if target_style_ids != EXPECTED_SECTION_STYLE_IDS:
        raise SourcePreparationError("style bank requires the two chk-1 section styles")

    corpus_path = Path(official_corpus_path)
    corpus_rows = _read_jsonl(corpus_path, label="official train Minutes corpus")
    source_binding = _bind_jsonl(
        corpus_path, "official_train_minutes_corpus", len(corpus_rows)
    )
    if not _SHA256_RE.fullmatch(expected_source_sha256):
        raise SourcePreparationError("expected official corpus hash is invalid")
    _assert_hash(
        source_binding.sha256,
        expected_source_sha256,
        label="official train Minutes corpus",
    )

    candidates: list[dict[str, Any]] = []
    seen_source_ids: set[str] = set()
    for line_number, row in corpus_rows:
        label = f"official train Minutes corpus line {line_number}"
        if row.get("split") != "train":
            raise SourcePreparationError(f"{label} is not labelled train")
        meeting_date = _validate_date(
            _required_text(row, "meeting_date", label=label),
            label=f"{label}.meeting_date",
        )
        if meeting_date in held_out_meetings or meeting_date not in train_meetings:
            raise SourcePreparationError(
                f"{label} is outside the frozen chk-1 train meetings: {meeting_date}"
            )
        source_sample_id = _required_text(row, "sample_id", label=label)
        if source_sample_id in seen_source_ids:
            raise SourcePreparationError(
                f"duplicate official source sample_id: {source_sample_id}"
            )
        seen_source_ids.add(source_sample_id)
        quality_flags = row.get("quality_flags", [])
        if not isinstance(quality_flags, list) or not all(
            isinstance(flag, str) for flag in quality_flags
        ):
            raise SourcePreparationError(f"{label}.quality_flags must be a string list")
        if quality_flags:
            continue
        section_name = _required_text(row, "section_name", label=label)
        raw_text = _required_text(row, "reference_excerpt", label=label)
        source_row_index = row.get("source_row_index")
        if isinstance(source_row_index, bool) or not isinstance(source_row_index, int):
            raise SourcePreparationError(f"{label}.source_row_index must be an integer")
        text = _normalize_exemplar(raw_text)
        if not _eligible_exemplar(raw_text, text):
            continue
        text_sha = sha256_text(text)
        source_style = stable_section_style_id(section_name)
        source_row_payload = {
            "sample_id": source_sample_id,
            "meeting_date": meeting_date,
            "source_row_index": source_row_index,
            "section_style_id": source_style,
            "raw_text_sha256": sha256_text(raw_text),
            "normalized_text_sha256": text_sha,
        }
        candidates.append(
            {
                "meeting_date": meeting_date,
                "source_sample_id": source_sample_id,
                "source_row_index": source_row_index,
                "source_corpus_line_number": line_number,
                "source_section_style_id": source_style,
                "word_count": len(_WORD_RE.findall(text)),
                "text": text,
                "text_sha256": text_sha,
                "source_row_sha256": sha256_json(source_row_payload),
            }
        )

    selected: list[StyleExemplar] = []
    used_text_hashes: set[str] = set()
    for target_style_id in target_style_ids:
        direct = [
            item
            for item in candidates
            if item["source_section_style_id"] == target_style_id
        ]
        general = [
            item
            for item in candidates
            if item["source_section_style_id"] != target_style_id
        ]
        chosen: list[tuple[dict[str, Any], str, str]] = []
        chosen_meetings: set[str] = set()
        for pool, mode in ((direct, "direct_style"), (general, "general_pool")):
            ranked = sorted(
                (
                    (
                        sha256_json(
                            {
                                "target_section_style_id": target_style_id,
                                "source_row_sha256": item["source_row_sha256"],
                            }
                        ),
                        item,
                    )
                    for item in pool
                    if item["text_sha256"] not in used_text_hashes
                ),
                key=lambda pair: pair[0],
            )
            # Prefer distinct meetings, then deterministically allow another
            # paragraph from a train meeting if the pool is unusually sparse.
            for distinct_only in (True, False):
                for rank_sha, item in ranked:
                    if len(chosen) >= exemplars_per_style:
                        break
                    if any(
                        existing[0]["text_sha256"] == item["text_sha256"]
                        for existing in chosen
                    ):
                        continue
                    if distinct_only and item["meeting_date"] in chosen_meetings:
                        continue
                    chosen.append((item, mode, rank_sha))
                    chosen_meetings.add(item["meeting_date"])
                if len(chosen) >= exemplars_per_style:
                    break
            if len(chosen) >= exemplars_per_style:
                break
        if len(chosen) != exemplars_per_style:
            raise SourcePreparationError(
                f"insufficient eligible train-only Minutes exemplars for {target_style_id}"
            )
        for position, (item, mode, rank_sha) in enumerate(chosen, 1):
            used_text_hashes.add(item["text_sha256"])
            exemplar_id = (
                f"minutes-style-{target_style_id.removeprefix('section-style-v1-')}-"
                f"{position}-{rank_sha[:12]}"
            )
            selected.append(
                StyleExemplar(
                    exemplar_id=exemplar_id,
                    section_style_id=target_style_id,
                    source_section_style_id=item["source_section_style_id"],
                    meeting_date=item["meeting_date"],
                    source_sample_id=item["source_sample_id"],
                    source_row_index=item["source_row_index"],
                    source_corpus_line_number=item["source_corpus_line_number"],
                    selection_mode=mode,
                    selection_rank_sha256=rank_sha,
                    word_count=item["word_count"],
                    text=item["text"],
                    text_sha256=item["text_sha256"],
                    source_row_sha256=item["source_row_sha256"],
                )
            )

    payload: dict[str, Any] = {
        "schema_version": STYLE_BANK_SCHEMA_VERSION,
        "source": source_binding.to_dict(),
        "train_meeting_count": len(train_meetings),
        "target_section_style_ids": list(target_style_ids),
        "exemplars_per_style": exemplars_per_style,
        "exemplars": [item.to_dict() for item in selected],
    }
    bank_sha = sha256_json(payload)
    bank = StyleBank(
        schema_version=STYLE_BANK_SCHEMA_VERSION,
        source=source_binding,
        train_meeting_count=len(train_meetings),
        target_section_style_ids=target_style_ids,
        exemplars_per_style=exemplars_per_style,
        exemplars=tuple(selected),
        style_bank_sha256=bank_sha,
    )
    verify_style_bank(bank, rows)
    return bank


def verify_style_bank(
    bank: StyleBank,
    source_rows: PreparedSourceDataset | Sequence[PreparedSourceRow],
) -> None:
    """Recompute embedded hashes and prove that every exemplar is train-only."""

    rows = (
        source_rows.rows
        if isinstance(source_rows, PreparedSourceDataset)
        else tuple(source_rows)
    )
    train_meetings = {row.meeting_date for row in rows if row.split == "train"}
    held_out_meetings = {row.meeting_date for row in rows if row.split != "train"}
    payload = bank.to_dict()
    observed_sha = payload.pop("style_bank_sha256")
    _assert_hash(sha256_json(payload), observed_sha, label="style bank")
    if bank.schema_version != STYLE_BANK_SCHEMA_VERSION:
        raise SourcePreparationError("style bank schema version mismatch")
    if tuple(sorted(bank.target_section_style_ids)) != EXPECTED_SECTION_STYLE_IDS:
        raise SourcePreparationError("style bank target section styles mismatch")
    expected_total = len(bank.target_section_style_ids) * bank.exemplars_per_style
    if len(bank.exemplars) != expected_total:
        raise SourcePreparationError("style bank exemplar count mismatch")
    counts = {style_id: 0 for style_id in bank.target_section_style_ids}
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    for exemplar in bank.exemplars:
        if exemplar.exemplar_id in seen_ids or exemplar.text_sha256 in seen_text:
            raise SourcePreparationError(
                "style bank has duplicate exemplar identity or text"
            )
        seen_ids.add(exemplar.exemplar_id)
        seen_text.add(exemplar.text_sha256)
        if exemplar.section_style_id not in counts:
            raise SourcePreparationError(
                "style bank contains an unexpected target style"
            )
        counts[exemplar.section_style_id] += 1
        if (
            exemplar.meeting_date not in train_meetings
            or exemplar.meeting_date in held_out_meetings
        ):
            raise SourcePreparationError(
                "style bank contains a held-out/non-train meeting"
            )
        _assert_hash(
            sha256_text(exemplar.text), exemplar.text_sha256, label="exemplar text"
        )
        if len(_WORD_RE.findall(exemplar.text)) != exemplar.word_count:
            raise SourcePreparationError("style exemplar word count mismatch")
        if not 20 <= exemplar.word_count <= 400:
            raise SourcePreparationError(
                "style exemplar violates the 20-400 word contract"
            )
        if (
            _POLICY_RE.search(exemplar.text)
            or _POLICY_SUBJECT_RE.search(exemplar.text)
            or _ADMIN_RE.search(exemplar.text)
        ):
            raise SourcePreparationError(
                "style exemplar contains policy/vote/admin content"
            )
    if any(count != bank.exemplars_per_style for count in counts.values()):
        raise SourcePreparationError("style bank is not balanced across target styles")


def build_prepare_manifest(
    dataset: PreparedSourceDataset,
    style_bank: StyleBank,
) -> dict[str, Any]:
    """Return the deterministic source-admission manifest used by acquisition."""

    verify_style_bank(style_bank, dataset)
    response_changed_rows = sum(
        row.candidate_response_sha256 != row.source_response_sha256
        for row in dataset.rows
    )
    answer_changed_rows = sum(
        row.source_analysis_sha256 != row.source_answer_sha256 for row in dataset.rows
    )
    manifest: dict[str, Any] = {
        "schema_version": PREPARE_MANIFEST_SCHEMA_VERSION,
        "source_dataset_schema_version": dataset.schema_version,
        "split_counts": dict(dataset.split_counts),
        "meeting_counts": dict(dataset.meeting_counts),
        "sample_id_digest": dataset.sample_id_digest,
        "source_rows_digest": dataset.rows_digest,
        "source_artifacts": dataset.binding_map(),
        "style_bank_sha256": style_bank.style_bank_sha256,
        "style_bank_source": style_bank.source.to_dict(),
        "style_bank_exemplar_count": len(style_bank.exemplars),
        "upstream_chk1_candidate_repair_provenance": {
            "candidate_response_changed_rows": response_changed_rows,
            "candidate_final_answer_changed_rows": answer_changed_rows,
            "source_rows": len(dataset.rows),
        },
        "invariants": {
            "source_analysis_is_exact_chk1_final_answer": True,
            "source_analysis_was_repaired": False,
            "source_analysis_was_repaired_scope": "paper_chk2_pipeline_only",
            "source_dataset_contains_upstream_chk1_repairs": bool(
                response_changed_rows or answer_changed_rows
            ),
            "c8_used_for_training": False,
            "style_bank_train_only": True,
            "held_out_official_minutes_used": False,
        },
    }
    manifest["prepare_manifest_sha256"] = sha256_json(manifest)
    return manifest


def verify_prepare_manifest(
    manifest: Mapping[str, Any],
    dataset: PreparedSourceDataset,
    style_bank: StyleBank,
) -> None:
    expected = build_prepare_manifest(dataset, style_bank)
    if dict(manifest) != expected:
        raise SourcePreparationError(
            "prepare manifest does not match its bound sources"
        )


__all__ = [
    "ArtifactBinding",
    "DEFAULT_CANDIDATE_ROOT",
    "DEFAULT_GENERATION_ROOT",
    "DEFAULT_OFFICIAL_TRAIN_CORPUS",
    "EXPECTED_MEETING_COUNTS",
    "EXPECTED_OFFICIAL_TRAIN_CORPUS_SHA256",
    "EXPECTED_SECTION_STYLE_IDS",
    "EXPECTED_SOURCE_FILE_SHA256",
    "EXPECTED_STYLE_BANK_SHA256",
    "EXPECTED_SPLIT_COUNTS",
    "PREPARE_MANIFEST_SCHEMA_VERSION",
    "PreparedSourceDataset",
    "PreparedSourceRow",
    "SOURCE_SCHEMA_VERSION",
    "STYLE_BANK_SCHEMA_VERSION",
    "SourcePreparationError",
    "StyleBank",
    "StyleExemplar",
    "THINK_BOUNDARY",
    "build_prepare_manifest",
    "build_train_only_style_bank",
    "extract_final_answer",
    "load_chk1_source_rows",
    "sha256_file",
    "sha256_json",
    "sha256_text",
    "verify_prepare_manifest",
    "verify_style_bank",
]
