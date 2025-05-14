import json
import random

def convert_response_jsonl_split(
    input_jsonl: str,
    train_jsonl: str,
    val_jsonl: str,
    test_jsonl: str,
    seed: int = 42
):
    """
    Split dataset:
    - total train = 80% → sft = 80% of train, grpo = 20% of train + 10% sft
    - eval = 10%
    - test = 10%
    """
    _think_start = "<think>"
    _think_end = "</think>"
    _answer_start = "<answer>"
    _answer_end = "</answer>"

    # Load data
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]
        print(f"✅ Loaded {len(lines)} entries from {input_jsonl}")

    random.seed(seed)
    random.shuffle(lines)
    n = len(lines)

    # First split: 80% train, 10% eval, 10% test
    train_size = int(n * 0.8)
    eval_size = int(n * 0.1)

    train_lines = lines[:train_size]
    eval_lines = lines[train_size:train_size + eval_size]
    test_lines = lines[train_size + eval_size:]

    # Split train into sft and grpo
    sft_size = int(train_size * 0.8)  # 80% of train
    grpo_base_size = train_size - sft_size  # 20% of train
    grpo_extra_size = int(sft_size * 0.1)  # extra 10% of sft

    sft_lines = train_lines[:sft_size]
    grpo_lines = train_lines[sft_size:] + random.sample(sft_lines, grpo_extra_size)

    def format_response(entry):
        try:
            return {
                "prompt": entry["prompt"],
                "response": f"{_think_start}{entry['reasoning']}{_think_end}{_answer_start}{entry['response']}{_answer_end}",
                "provided_data": entry["provided_data"],
            }
        except KeyError as e:
            print(f"⚠️ Missing key in entry {entry.get('index', 'unknown')}: {e}")
            return None

    def write_jsonl(path, dataset):
        with open(path, 'w', encoding="utf-8") as f_out:
            for entry in dataset:
                formatted = format_response(entry)
                if formatted:
                    f_out.write(json.dumps(formatted, ensure_ascii=False) + '\n')

    # Write
    sft_jsonl = train_jsonl.replace("fomc_qa_train.jsonl", "fomc_qa_sft_train.jsonl")
    grpo_jsonl = train_jsonl.replace("fomc_qa_train.jsonl", "fomc_qa_grpo_train.jsonl")
    write_jsonl(sft_jsonl, sft_lines)
    write_jsonl(grpo_jsonl, grpo_lines)
    write_jsonl(val_jsonl, eval_lines)
    write_jsonl(test_jsonl, test_lines)

    print(f"✅ Dataset split complete:")
    print(f"SFT: {len(sft_lines)}")
    print(f"GRPO: {len(grpo_lines)}")
    print(f"Eval: {len(eval_lines)}")
    print(f"Test: {len(test_lines)}")

if __name__ == '__main__':
    input_jsonl = "./../dataset/raw_data/merged_response.jsonl"
    output_jsonl = "./../dataset/training_data/fomc_qa/fomc_qa.jsonl"
    convert_response_jsonl_split(
        input_jsonl,
        output_jsonl.replace("fomc_qa.jsonl", "fomc_qa_train.jsonl"),
        output_jsonl.replace("fomc_qa.jsonl", "fomc_qa_eval.jsonl"),
        output_jsonl.replace("fomc_qa.jsonl", "fomc_qa_test.jsonl")
    )
