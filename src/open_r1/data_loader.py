import glob
import logging
import os
from collections.abc import Mapping
from pathlib import Path

from datasets import load_dataset, DatasetDict


logger = logging.getLogger(__name__)


def _find_split_file(data_path: str, patterns: list[str], split_name: str) -> str:
    matches: list[str] = []
    for pattern in patterns:
        matches.extend(glob.glob(os.path.join(data_path, pattern)))

    unique_matches = sorted(set(matches))
    if not unique_matches:
        joined_patterns = ", ".join(patterns)
        raise FileNotFoundError(
            f"No file matching [{joined_patterns}] found for split '{split_name}' in {data_path}"
        )

    if len(unique_matches) > 1:
        raise ValueError(
            f"Multiple files match split '{split_name}' in {data_path}: "
            + ", ".join(unique_matches)
        )

    return unique_matches[0]


def _resolve_explicit_split_file(
    data_path: str | os.PathLike[str],
    value: str | os.PathLike[str],
    *,
    split_name: str,
) -> str:
    """Resolve one explicitly bound split without accepting globs or escapes."""

    root = Path(data_path).expanduser()
    try:
        root = root.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(f"Dataset directory is missing: {root}") from exc
    if not root.is_dir():
        raise NotADirectoryError(f"Dataset path is not a directory: {root}")

    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = root / candidate
    if candidate.is_symlink():
        raise ValueError(
            f"Explicit file for split '{split_name}' must not be a symlink: {candidate}"
        )
    try:
        resolved = candidate.resolve(strict=True)
    except OSError as exc:
        raise FileNotFoundError(
            f"Explicit file for split '{split_name}' is missing: {candidate}"
        ) from exc
    if not resolved.is_file():
        raise FileNotFoundError(
            f"Explicit path for split '{split_name}' is not a file: {resolved}"
        )
    try:
        resolved.relative_to(root)
    except ValueError as exc:
        raise ValueError(
            f"Explicit file for split '{split_name}' escapes dataset directory: {resolved}"
        ) from exc
    return str(resolved)


def load_train_eval_datasets(
    data_path: str | os.PathLike[str],
    *,
    split_files: Mapping[str, str | os.PathLike[str]] | None = None,
) -> DatasetDict:
    """
    Load train and eval JSONL files into a HuggingFace DatasetDict.

    Args:
        data_path: Directory containing the dataset split files.
        split_files: Optional exact ``train`` and ``validation`` paths supplied
            by an already-verified release manifest. No glob discovery occurs
            when these bindings are present.

    Returns:
        A DatasetDict with "train" and "validation" splits.
    """
    if split_files is None:
        train_file = _find_split_file(str(data_path), ["*train.jsonl"], "train")
        eval_file = _find_split_file(
            str(data_path),
            ["*eval.jsonl", "*validation.jsonl", "*val.jsonl"],
            "validation",
        )
    else:
        if set(split_files) != {"train", "validation"}:
            raise ValueError(
                "Explicit split_files must contain exactly train and validation"
            )
        train_file = _resolve_explicit_split_file(
            data_path, split_files["train"], split_name="train"
        )
        eval_file = _resolve_explicit_split_file(
            data_path, split_files["validation"], split_name="validation"
        )

    logger.info("✅ Found train file: %s", train_file)
    logger.info("✅ Found validation file: %s", eval_file)
    dataset = load_dataset("json", data_files={
        "train": train_file,
        "validation": eval_file
    })

    logger.info("✅ Loaded dataset: %s", dataset)
    logger.info(
        "Train size: %s, Eval size: %s",
        len(dataset["train"]),
        len(dataset["validation"]),
    )
    return dataset
