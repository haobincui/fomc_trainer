import glob
import os
from datasets import load_dataset, DatasetDict

def load_train_eval_datasets(data_path) -> DatasetDict:
    """
    Load train and eval JSONL files into a HuggingFace DatasetDict.

    Args:
        train_path: Path to train.jsonl file
        eval_path: Path to eval.jsonl file

    Returns:
        A DatasetDict with "train" and "validation" splits.
    """
    train_files = glob.glob(os.path.join(data_path, "*train.jsonl"))
    eval_files = glob.glob(os.path.join(data_path, "*eval.jsonl"))

    if not train_files:
        raise FileNotFoundError(f"No file ending with 'train.jsonl' found in {data_path}")
    if not eval_files:
        raise FileNotFoundError(f"No file ending with 'val.jsonl' found in {data_path}")

    train_file = train_files[0]
    eval_file = eval_files[0]

    print(f"✅ Found train file: {train_file}")
    print(f"✅ Found val file: {eval_file}")
    dataset = load_dataset("json", data_files={
        "train": train_file,
        "validation": eval_file
    })

    print(f"✅ Loaded dataset: {dataset}")
    print(f"Train size: {len(dataset['train'])}, Eval size: {len(dataset['validation'])}")
    return dataset
