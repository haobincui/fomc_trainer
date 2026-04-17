from log_config import *
import json
import os
import glob
import re
import pandas as pd
from pathlib import Path

from utils import bootstrap_stats, save_output


def _parse_final_vote(text):
    """Extracts content inside the last \\boxed{} after </think>."""
    if not isinstance(text, str):
        return ""
    match = re.search(r"\\boxed\{(.*?)\}", text.split("</think>")[-1], re.DOTALL)
    return match.group(1).strip() if match else ""


_rate_change_map = json.loads(Path("dataset/raw_data/rate_change_map.json").read_text(encoding="utf-8"))


def eval_decision(input_file: str, output_file = None) -> dict:
    """result dict: {"overall": xxx, "raise": xxx}"""
    if input_file.endswith(".xlsx"):
        df = pd.read_excel(input_file)
    elif input_file.endswith(".jsonl"):
        df = pd.read_json(input_file, lines=True)
    else:
        raise ValueError(f"Unsupported file name {input_file}")
    
    results = []
    i = 0
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        # target = row.get("target", "")
        # if not target:
        meeting_date = row["meeting_date"]
        if isinstance(meeting_date, pd.Timestamp):
            meeting_date = meeting_date.strftime("%Y-%m-%d")
        else:
            meeting_date = str(meeting_date).strip()

        target = _rate_change_map.get(meeting_date)
        if target is None:
            print(f"⚠️ Warning: meeting_date '{meeting_date}' not found in _rate_change_map.")
            target = {"rate_change": None, "current_rate": None}
        generated = row.get("generated", "")
        target_vote = target['rate_change']
        generated_vote = _parse_final_vote(generated)
        match_result = 1 if target_vote == generated_vote else 0

    
        row_dict.update({
            "target_vote": target_vote,
            "generated_vote": generated_vote,
            "match_result": match_result
        })
        results.append(row_dict)
        print(f"✅ Finished Index {idx} | Match: {match_result}")
        i += 1

    result_df = pd.DataFrame(results)

    if output_file:
        save_output(result_df, output_file)
    

    # 3) 按 rate_change 分组统计 
    results = []

    group_key = "target_vote"
    if group_key not in result_df.columns:
        raise ValueError(f"Missing column '{group_key}' in input.")

    result = {}

    # Overall accuracy
    overall_acc = result_df["match_result"].sum() / len(result_df)
    result["overall"] = overall_acc

    # Group-level accuracy
    for rate_value, sub_df in result_df.groupby(group_key, dropna=False):
        cur_acc = sub_df["match_result"].sum() / len(sub_df)
        result[rate_value] = cur_acc

    print("Accuracy:", result)
    return result






if __name__ == '__main__':
    # output_file = "output/validation/generation_stage2_decision/decision_making_grpo_20250528.xlsx"
    # input_file = "output/validation/generation_stage2_decision/decision_making_grpo_20250528_train_generated.xlsx"
    # input_file = "output/validation/generation_stage2_decision/decision_making_20250529_train_generated.xlsx"
    model_name_map = {
        "output/merged/llama_sft_synthetic_20250526": "ft",
        'models/DeepSeek-R1-Distill-Llama-8B': "base"
    }
    model_path = 'output/merged/llama_sft_synthetic_20250526'
    # model_path = 'models/DeepSeek-R1-Distill-Llama-8B'
    accs = []
    output_path = f"output/valiation/generation_stage2_synthetic/synthetic_for_decision/20250602/{model_name_map[model_path]}_model"

    minutes_files = glob.glob(output_path + f"/minutes/*.jsonl")
    print(f"Total Minutes {len(minutes_files)}")
    for minutes_file in minutes_files:
        output_file = minutes_file.replace("/minutes/", "/vote/").replace(".jsonl", ".xlsx")

        acc = eval_decision(minutes_file, output_file)
        accs.append(acc)
    
    result_file =output_path + "/acc_bootstrap_result.jsonl"

    acc_df = pd.DataFrame(accs)

    bootstrap_result = []
    for changes in acc_df.columns:
        acc = acc_df[changes].dropna().astype(float).tolist()
        stats = bootstrap_stats(acc)
        stats['target_vote'] = changes
        bootstrap_result.append(stats)
    
    save_output(bootstrap_result, result_file)
    print(f"Result: {bootstrap_result}")
    



    
