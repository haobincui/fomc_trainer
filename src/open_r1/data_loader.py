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
    train_path = data_path.replace(".jsonl", "_train.jsonl")
    eval_path = data_path.replace(".jsonl", "_eval.jsonl")
    dataset = load_dataset("json", data_files={
        "train": train_path,
        "eval": eval_path
    })

    print(f"✅ Loaded dataset: {dataset}")
    print(f"Train size: {len(dataset['train'])}, Eval size: {len(dataset['validation'])}")
    return dataset
