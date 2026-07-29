import glob
import logging
import os
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
        logger.warning(
            "Multiple files found for split '%s' in %s; using the first deterministic match: %s",
            split_name,
            data_path,
            unique_matches[0],
        )

    return unique_matches[0]


def load_train_eval_datasets(data_path) -> DatasetDict:
    """
    Load train and eval JSONL files into a HuggingFace DatasetDict.

    Args:
        train_path: Path to train.jsonl file
        eval_path: Path to eval.jsonl file

    Returns:
        A DatasetDict with "train" and "validation" splits.
    """
    train_file = _find_split_file(data_path, ["*train.jsonl"], "train")
    eval_file = _find_split_file(
        data_path,
        ["*eval.jsonl", "*validation.jsonl", "*val.jsonl"],
        "validation",
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
