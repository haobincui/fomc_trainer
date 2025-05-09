import json
import random

def convert_response_jsonl_split(input_jsonl: str, train_jsonl: str, eval_jsonl: str, split_ratio: float = 0.9, seed: int = 42):
    """
    Convert a JSONL dataset of {"prompt", "reasoning", "response"} into:
    - train set with <think>reasoning</think><answer>response</answer>
    - eval set with the same format
    Saves to separate .jsonl files.

    Args:
        input_jsonl: Path to original jsonl file
        train_jsonl: Path to output train file
        eval_jsonl: Path to output eval file
        split_ratio: Proportion of data for training
        seed: Random seed for reproducibility
    """
    _think_start = "<think>"
    _think_end = "</think>"
    _answer_start = "<answer>"
    _answer_end = "</answer>"

    # Load all lines
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]
        print(f"✅ Loaded {len(lines)} entries from {input_jsonl}")

    # Shuffle and split
    random.seed(seed)
    random.shuffle(lines)
    split_idx = int(len(lines) * split_ratio)
    train_lines = lines[:split_idx]
    eval_lines = lines[split_idx:]

    def format_response(entry):
        try:
            return {
                "prompt": entry["prompt"],
                "response": f"{_think_start}{entry['reasoning']}{_think_end}{_answer_start}{entry['response']}{_answer_end}",
                "provided_data": entry["provided_data"],
            }
        except KeyError as e:
            print(f"⚠️ Missing key {entry['index']} in entry: {e}")
            return None

    # Write train
    with open(train_jsonl, 'w', encoding="utf-8") as f_train:
        for entry in train_lines:
            f_train.write(json.dumps(format_response(entry), ensure_ascii=False) + '\n')

    # Write eval
    with open(eval_jsonl, 'w', encoding="utf-8") as f_eval:
        for entry in eval_lines:
            f_eval.write(json.dumps(format_response(entry), ensure_ascii=False) + '\n')

    print(f"✅ Conversion complete: {len(train_lines)} train, {len(eval_lines)} eval examples saved.")

if __name__ == '__main__':
    input_jsonl = "./../dataset/raw_data/merged_response.jsonl"
    output_jsonl = "./../dataset/training_data/fomc_qa/fomc_qa.jsonl"
    convert_response_jsonl_split(input_jsonl, output_jsonl.replace("fomc_qa.jsonl", "fomc_qa_train.jsonl"), output_jsonl.replace("fomc_qa.jsonl", "fomc_qa_eval.jsonl"))


