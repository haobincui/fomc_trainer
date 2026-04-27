import argparse
import pandas as pd

from open_r1.utils.fomc import load_rate_change_map, normalize_meeting_date, parse_boxed_vote
from utils import save_output


def eval_decision(
    input_file: str,
    output_file: str | None = None,
    rate_change_map_path: str = "dataset/processed/main/input_sources/rate_change_map.json",
) -> dict:
    if input_file.endswith(".xlsx"):
        df = pd.read_excel(input_file)
    elif input_file.endswith(".jsonl"):
        df = pd.read_json(input_file, lines=True)
    else:
        raise ValueError(f"Unsupported file name {input_file}")

    rate_change_map = load_rate_change_map(rate_change_map_path)
    results = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        meeting_date = normalize_meeting_date(row.get("meeting_date"))
        target = rate_change_map.get(meeting_date, {"rate_change": None, "current_rate": None})
        generated_vote = parse_boxed_vote(row.get("generated", ""))
        target_vote = target["rate_change"]
        match_result = 1 if target_vote == generated_vote else 0

        row_dict.update(
            {
                "target_vote": target_vote,
                "generated_vote": generated_vote,
                "match_result": match_result,
            }
        )
        results.append(row_dict)
        print(f"✅ Finished Index {idx} | Match: {match_result}")

    result_df = pd.DataFrame(results)
    if output_file:
        save_output(result_df, output_file)

    result = {"overall": result_df["match_result"].sum() / len(result_df)}
    for rate_value, sub_df in result_df.groupby("target_vote", dropna=False):
        result[rate_value] = sub_df["match_result"].sum() / len(sub_df)

    print("Accuracy:", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate decision predictions against the archived rate-change map.")
    parser.add_argument("--input", required=True, help="Input .jsonl or .xlsx file containing generated decisions.")
    parser.add_argument("--output", help="Optional output file (.jsonl/.xlsx/.csv) for per-row evaluation results.")
    parser.add_argument(
        "--rate-change-map",
        default="dataset/processed/main/input_sources/rate_change_map.json",
        help="Path to the JSON file mapping meeting_date to target rate changes.",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    eval_decision(args.input, args.output, args.rate_change_map)
