import re
import pandas as pd
from pathlib import Path


def _parse_final_vote(text):
    """Extracts content inside the last \\boxed{} after </think>."""
    if not isinstance(text, str):
        return ""
    match = re.search(r"\\boxed\{(.*?)\}", text.split("</think>")[-1], re.DOTALL)
    return match.group(1).strip() if match else ""


def run_eval_decision(input_file: str):
    if input_file.endswith(".xlsx"):
        df = pd.read_excel(input_file)
    elif input_file.endswith(".jsonl"):
        df = pd.read_json(input_file, lines=True)
    else:
        raise ValueError(f"Unsupported file name {input_file}")
    
    results = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        # target = row.get("target", "")
        # if not target:
        target = row['rate_change']
        generated = row.get("generated", "")
        target_vote = _parse_final_vote(target)
        generated_vote = _parse_final_vote(generated)
        match_result = 1 if target_vote == generated_vote else 0

    
        row_dict.update({
            "target_vote": target_vote,
            "generated_vote": generated_vote,
            "match_result": match_result
        })
        results.append(row_dict)
        print(f"✅ Finished Index {idx} | Match: {match_result}")

    result_df = pd.DataFrame(results)

    output_path = Path(input_file).with_stem(Path(input_file).stem + "_result")
    result_df.to_excel(output_path.with_suffix(".xlsx"), index=False)

    total = len(result_df)
    correct = result_df['match_result'].sum()
    accuracy = correct / total * 100

    print(f"\n🎯 Total samples: {total}")
    print(f"✅ Correct predictions: {correct}")
    print(f"📊 Accuracy: {accuracy:.2f}%")

    return accuracy, output_path



if __name__ == '__main__':
    # output_file = "output/validation/generation_stage2_decision/decision_making_grpo_20250528.xlsx"
    # input_file = "output/validation/generation_stage2_decision/decision_making_grpo_20250528_train_generated.xlsx"
    input_file = "output/validation/generation_stage2_decision/decision_making_20250529_train_generated.xlsx"
    run_eval_decision(input_file)
