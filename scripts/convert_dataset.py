import json
import random

_think_start = "<think>"
_think_end = "</think>"
_answer_start = "<answer>"
_answer_end = "</answer>"

def format_response(entry):
    try:
        result_dict = {}
        if "reasoning" not in entry:
            entry["reasoning"] = ""
        if "provided_data" not in entry:
            entry["provided_data"] = ""
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



def load_and_split(input_jsonl: str, seed: int = 42):
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]
    print(f"✅ Loaded {len(lines)} entries from {input_jsonl}")

    random.seed(seed)
    random.shuffle(lines)
    n = len(lines)

    train_size = int(n * 0.8)
    eval_size = int(n * 0.1)

    train_lines = lines[:train_size]
    eval_lines = lines[train_size:train_size + eval_size]
    test_lines = lines[train_size + eval_size:]

    return train_lines, eval_lines, test_lines



def convert_response_jsonl_grpo_split(
    input_jsonl: str,
    train_jsonl_base: str,
    val_jsonl: str,
    test_jsonl: str,
    seed: int = 42
):
    train_lines, eval_lines, test_lines = load_and_split(input_jsonl, seed)

    sft_size = int(len(train_lines) * 0.8)
    grpo_base_size = len(train_lines) - sft_size
    grpo_extra_size = int(sft_size * 0.1)

    sft_lines = train_lines[:sft_size]
    grpo_lines = train_lines[sft_size:] + random.sample(sft_lines, grpo_extra_size)

    sft_jsonl = train_jsonl_base.replace("fomc_qa_train.jsonl", "fomc_qa_sft_train.jsonl")
    grpo_jsonl = train_jsonl_base.replace("fomc_qa_train.jsonl", "fomc_qa_grpo_train.jsonl")

    write_jsonl(sft_jsonl, sft_lines)
    write_jsonl(grpo_jsonl, grpo_lines)
    write_jsonl(val_jsonl, eval_lines)
    write_jsonl(test_jsonl, test_lines)

    print(f"✅ Dataset split complete:")
    print(f"SFT: {len(sft_lines)}")
    print(f"GRPO: {len(grpo_lines)}")
    print(f"Eval: {len(eval_lines)}")
    print(f"Test: {len(test_lines)}")



def convert_response_jsonl(
        input_jsonl: str,
        train_jsonl: str,
        val_jsonl: str,
        test_jsonl: str,
        seed: int = 42
):
    train_lines, eval_lines, test_lines = load_and_split(input_jsonl, seed)

    write_jsonl(train_jsonl, train_lines)
    write_jsonl(val_jsonl, eval_lines)
    write_jsonl(test_jsonl, test_lines)

    print(f"✅ Dataset split complete:")
    print(f"Train: {len(train_lines)}")
    print(f"Eval: {len(eval_lines)}")
    print(f"Test: {len(test_lines)}")



if __name__ == '__main__':
    # input_jsonl = "./../dataset/raw_data/merged_response.jsonl"
    # input_jsonl = "./../dataset/raw_data/synthetic_text_20250520.jsonl"
    # output_jsonl = "./../dataset/training_data/synthetic_text/synthetic_text_20250520.jsonl"
    input_jsonl = "./../dataset/raw_data/response_decision_20250513.jsonl"
    output_jsonl = "./../dataset/training_data/decision_making/response_decision_20250520.jsonl"

    input_file = "./../dataset/raw_data/decision_making_20250521.jsonl"
    output_file = "./../dataset/training_data/decision_making/decision_making_20250521.jsonl"
    convert_response_jsonl(
        input_jsonl,
        output_jsonl.replace(".jsonl", "_train.jsonl"),
        output_jsonl.replace(".jsonl", "_eval.jsonl"),
        output_jsonl.replace(".jsonl", "_test.jsonl")
    )
