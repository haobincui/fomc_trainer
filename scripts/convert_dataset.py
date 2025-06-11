import json
import random
from collections.abc import Callable

import pandas as pd

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


def write_jsonl(path, dataset, format: Callable = format_response):
    with open(path, 'w', encoding="utf-8") as f_out:
        for entry in dataset:
            formatted = format(entry)
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
        format: Callable = format_response,
        seed: int = 42
):
    train_lines, eval_lines, test_lines = load_and_split(input_jsonl, seed)

    write_jsonl(train_jsonl, train_lines, format)
    write_jsonl(val_jsonl, eval_lines, format)
    write_jsonl(test_jsonl, test_lines, format)

    print(f"✅ Dataset split complete:")
    print(f"Train: {len(train_lines)}")
    print(f"Eval: {len(eval_lines)}")
    print(f"Test: {len(test_lines)}")


def format_grpo_decision(entry):
    try:
        return {
            "index": entry["index"],
            "prompt": entry["prompt"],
            "response": f"{_think_start}{entry['reasoning']}{_think_end}{_answer_start}{entry['response']}{_answer_end}",
            "rate_change": entry['rate_change'],
            "current_rate": entry['current_rate'],
        }
    except KeyError as e:
        print(f"⚠️ Missing key in entry {entry.get('index', 'unknown')}: {e}")
        return {
            "index": entry["index"],
            "prompt": entry["prompt"],
            "response": "",
            "rate_change": entry['rate_change'],
            "current_rate": entry['current_rate'],
        }


def convert_grpo_jsonl(
        input_jsonl: str,
        train_jsonl: str,
        eval_jsonl: str,
        format: Callable = format_response,
        seed: int = 42
):
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]
    print(f"✅ Loaded {len(lines)} entries from {input_jsonl}")

    random.seed(seed)
    random.shuffle(lines)
    n = len(lines)

    train_size = int(n * 0.8)

    train_lines = lines[:train_size]
    eval_lines = lines[train_size:]


    write_jsonl(train_jsonl, train_lines, format)
    write_jsonl(eval_jsonl, eval_lines, format)

    print(f"✅ Dataset split complete:")
    print(f"Train: {len(train_lines)}")
    print(f"Eval: {len(eval_lines)}")

def convert_decision_grpo_jsonl(
        input_jsonl: str,
        train_jsonl: str,
        eval_jsonl: str,
        format: Callable = format_response,
        seed: int = 42):
    with open(input_jsonl, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f]
    print(f"✅ Loaded {len(lines)} entries from {input_jsonl}")
    total = len(lines)
    random.seed(seed)
    random.shuffle(lines)

    # 找出所有 "No change" 项的原始索引
    no_change_items = [(i, item) for i, item in enumerate(lines) if item.get("rate_change") == "No change"]
    print(f"🔍 Found {len(no_change_items)} 'No change' entries.")

    remove_ratio = 0.4
    train_ratio = 0.8

    if len(no_change_items) >= total / 2:
        # 随机选出需要删除的条目
        indices_to_remove = random.sample(no_change_items, int(len(no_change_items) * remove_ratio))
        remove_indices_set = set(i for i, _ in indices_to_remove)

        # 构造保留数据和被删除的数据
        lines_to_keep = [item for i, item in enumerate(lines) if i not in remove_indices_set]
        lines_removed = [item for _, item in indices_to_remove]

        print(f"✅ Removed {len(lines_removed)} 'No change' entries.")
    else:
        lines_to_keep = lines
        lines_removed = []
        print(f"⚠️ Only found {len(no_change_items)} 'No change' entries; nothing removed.")

    # Split
    n = len(lines_to_keep)
    train_size = int(n * train_ratio)
    train_lines = lines_to_keep[:train_size]
    eval_lines = lines_to_keep[train_size:] + lines_removed

    pd.DataFrame(train_lines).to_excel(train_jsonl.replace(".jsonl", ".xlsx"), index=False)
    pd.DataFrame(eval_lines).to_excel(eval_jsonl.replace(".jsonl", ".xlsx"), index=False)

    write_jsonl(train_jsonl, train_lines, format)
    write_jsonl(eval_jsonl, eval_lines, format)

    print(f"✅ Dataset split complete:")
    print(f"Train: {len(train_lines)}")
    print(f"Eval: {len(eval_lines)}")









if __name__ == '__main__':
    # input_jsonl = "./../dataset/raw_data/merged_response.jsonl"
    # input_jsonl = "./../dataset/raw_data/synthetic_text_20250520.jsonl"
    # output_jsonl = "./../dataset/training_data/synthetic_text/synthetic_text_20250520.jsonl"
    # input_jsonl = "./../dataset/raw_data/response_decision_20250513.jsonl"
    # output_jsonl = "./../dataset/training_data/decision_making/response_decision_20250520.jsonl"
    #
    # input_file = "./../dataset/raw_data/decision_making_20250521.jsonl"
    # output_file = "./../dataset/training_data/decision_making/decision_making_20250521.jsonl"
    #
    # input_jsonl = "./../dataset/raw_data/synthetic_text_20250520_reason_filted.jsonl"
    # output_jsonl = "./../dataset/training_data/synthetic_text/synthetic_text_20250520_reason_filted.jsonl"
    # input_jsonl = "./../dataset/raw_data/decision_making_20250526_reason.jsonl"
    # output_jsonl = "./../dataset/training_data/decision_grpo/decision_making_20250526_reason.jsonl"
    # convert_response_jsonl(
    #     input_jsonl,
    #     output_jsonl.replace(".jsonl", "_grpo_train.jsonl"),
    #     output_jsonl.replace(".jsonl", "_grpo_eval.jsonl"),
    #     output_jsonl.replace(".jsonl", "_grpo_test.jsonl"),
    #     format_grpo_decision
    # )
    # input_jsonl = "./../dataset/raw_data/decision_making_20250521.jsonl"
    # output_jsonl = "./../dataset/training_data/decision_making/decision_making_20250521.jsonl"
    # input_jsonl = "./../dataset/raw_data/decision_prompt_summary_20250512.jsonl"
    # output_jsonl = "./../dataset/training_data/decision_grpo/decision_prompt_summary_20250512.jsonl"

    # convert_grpo_jsonl(
    #     input_jsonl,
    #     output_jsonl.replace(".jsonl", "_grpo_train.jsonl"),
    #     output_jsonl.replace(".jsonl", "_grpo_eval.jsonl"),
    #     format_grpo_decision
    # )

    input_jsonl = "./../dataset/raw_data/decision_making_merged_20250529.jsonl"
    output_jsonl = "./../dataset/training_data/decision_grpo/decision_grpo_20250531.jsonl"

    convert_decision_grpo_jsonl(input_jsonl, output_jsonl.replace(".jsonl", "_grpo_train.jsonl"),
                                output_jsonl.replace(".jsonl", "_grpo_eval.jsonl"),
                               format_grpo_decision)




