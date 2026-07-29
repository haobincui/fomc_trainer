import argparse
import json
import logging
import os
import random

import pandas as pd

from create_prompt.create_prompt import create_decision_prompt
from generate_new_response import generate_new_response
from open_r1.utils.fomc import parse_boxed_vote
from open_r1.validator.cos.cos_calc import cosine_similarity_calc
from open_r1.validator.cos.embedding_model import get_cached_embedding_model
from utils import save_output


def calc_cos(output_file: str, model_path: str) -> None:
    df = pd.read_excel(output_file.replace(".xlsx", "_generated.xlsx"))
    embed_model = get_cached_embedding_model(model_path)
    df["cos"] = df.apply(lambda x: cosine_similarity_calc(x["target"], x["generated"], embed_model), axis=1)
    df.to_excel(output_file, index=False)
    logging.info("✅ Finished cosine similarity calculation for %s", output_file)


def bootstrap_cos(output_file: str, sample_size: int = 10) -> None:
    df = pd.read_excel(output_file.replace(".xlsx", "_generated.xlsx"))
    cos_list = df["cos"].tolist()
    cos_result = []
    for _ in range(sample_size):
        random_indexes = random.sample(range(len(cos_list)), min(20, len(cos_list)))
        bootstrap_values = [cos_list[j] for j in random_indexes]
        cos_result.append(sum(bootstrap_values) / len(bootstrap_values))

    pd.DataFrame({"cos": cos_result}).to_excel(output_file.replace(".xlsx", "_bootstrap_cos.xlsx"), index=False)
    logging.info("✅ Finished bootstrap cosine summary for %s", output_file)


def run_eval_decision(input_file: str | pd.DataFrame) -> float:
    if isinstance(input_file, pd.DataFrame):
        df = input_file
    elif input_file.endswith(".xlsx"):
        df = pd.read_excel(input_file)
    elif input_file.endswith(".jsonl"):
        df = pd.read_json(input_file, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {input_file}")

    results = []
    for _, row in df.iterrows():
        row_dict = row.to_dict()
        target_vote = row["rate_change"]
        generated_vote = parse_boxed_vote(row.get("generated", ""))
        match_result = 1 if target_vote == generated_vote else 0
        row_dict.update(
            {
                "target_vote": target_vote,
                "generated_vote": generated_vote,
                "match_result": match_result,
            }
        )
        results.append(row_dict)

    result_df = pd.DataFrame(results)
    if not isinstance(input_file, pd.DataFrame):
        output_path = Path(input_file).with_stem(Path(input_file).stem + "_result")
        result_df.to_excel(output_path.with_suffix(".xlsx"), index=False)

    total = len(result_df)
    correct = result_df["match_result"].sum()
    accuracy = correct / total * 100 if total else 0.0

    logging.info("🎯 Total samples: %s", total)
    logging.info("✅ Correct predictions: %s", correct)
    logging.info("📊 Accuracy: %.2f%%", accuracy)

    return accuracy


def run_stage1_generation(input_prompt_file: str, output_file: str, model_path: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    generate_new_response(input_prompt_file, model_path, output_file)
    logging.info("✅ Finished stage 1 generation")


def run_stage2_synthetic_generation(input_prompt_file: str, output_file: str, model_path: str) -> None:
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    generate_new_response(input_prompt_file, model_path, output_file)
    logging.info("✅ Finished stage 2 synthetic generation")


def assemble_synthetic_data(section_file: str, output_file: str | None = None) -> list[dict]:
    def _parse_section_detail(detail: str) -> str:
        text = detail.split("<answer>")[-1]
        text = text.split("</answer>")[0]
        return text.replace("<answer>", "").replace("</answer>", "").strip()

    meeting_dict: dict[str, dict] = {}
    with open(section_file, "r", encoding="utf-8") as fin:
        for line in fin:
            line_dict = json.loads(line)
            meeting_date = line_dict["meeting_date"]
            if meeting_date not in meeting_dict:
                meeting_dict[meeting_date] = {
                    "sections": {},
                    "rate_change": line_dict["rate_change"],
                    "target_rate": line_dict["current_rate"],
                }
            meeting_dict[meeting_date]["sections"][line_dict["section_name"]] = line_dict["generated"]

    results: list[dict] = []
    for meeting_date, content in meeting_dict.items():
        minutes = [f"FOMC Minutes for {meeting_date}", ""]
        for section_name, section_detail in content["sections"].items():
            minutes.append(f"{section_name}\n{_parse_section_detail(section_detail)}")
        results.append(
            {
                "meeting_date": meeting_date,
                "minutes": "\n\n".join(minutes).strip(),
                "rate_change": content["rate_change"],
                "target_rate": content["target_rate"],
            }
        )

    if output_file:
        save_output(results, output_file)
        logging.info("✅ Saved assembled minutes to %s", output_file)
    return results


def minutes_to_decision_prompt(
    minutes_file: str | list[dict] | pd.DataFrame,
    output_file: str | None = None,
    template_id: int | None = None,
    seed: int | None = None,
) -> list[dict]:
    if isinstance(minutes_file, list):
        df = pd.DataFrame(minutes_file)
    elif isinstance(minutes_file, pd.DataFrame):
        df = minutes_file.copy()
    elif minutes_file.endswith(".jsonl"):
        df = pd.read_json(minutes_file, lines=True)
    elif minutes_file.endswith(".xlsx"):
        df = pd.read_excel(minutes_file)
    else:
        raise ValueError("Unsupported file format. Please provide a .jsonl or .xlsx file.")

    prompt_records = []
    for row in df.to_dict(orient="records"):
        prompt_payload = create_decision_prompt(
            current_analysis=row["minutes"],
            current_rate=row["target_rate"],
            meeting_date=row["meeting_date"],
            template_id=template_id,
            seed=seed,
        )
        row["prompt"] = prompt_payload["prompt"]
        row["prompt_template_id"] = prompt_payload["prompt_template_id"]
        prompt_records.append(row)

    if output_file:
        save_output(prompt_records, output_file)
        logging.info("✅ Saved decision prompts to %s", output_file)
    return prompt_records


def run_stage2_synthetic_full(
    *,
    model_path: str,
    input_prompt_file: str,
    output_dir: str,
    start_index: int,
    end_index: int,
    generation_batch_size: int = 20,
    decision_batch_size: int = 1,
    template_id: int | None = None,
    seed: int | None = None,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    total = max(end_index - start_index, 0)
    for offset in range(total):
        run_index = start_index + offset
        section_output_file = os.path.join(output_dir, "section_file", f"synthetic_text_{run_index}.jsonl")
        minutes_output_file = os.path.join(output_dir, "minutes", f"synthetic_text_{run_index}.jsonl")
        prompt_output_file = os.path.join(output_dir, "prompt", f"synthetic_text_{run_index}.jsonl")
        decision_output_file = os.path.join(output_dir, "decision", f"synthetic_text_{run_index}.jsonl")

        logging.info("Using model %s", model_path)
        logging.info("Generating synthetic sample %s/%s", offset + 1, total)

        generate_new_response(input_prompt_file, model_path, section_output_file, batch_size=generation_batch_size)
        synthetic_minutes = assemble_synthetic_data(section_output_file, minutes_output_file)
        synthetic_minutes_with_prompt = minutes_to_decision_prompt(
            synthetic_minutes,
            output_file=prompt_output_file,
            template_id=template_id,
            seed=seed,
        )
        generate_new_response(
            synthetic_minutes_with_prompt,
            model_path,
            decision_output_file,
            batch_size=decision_batch_size,
        )
        logging.info("✅ Finished synthetic decision run %s", run_index)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Synthetic generation utilities for the Chapter 2 pipeline.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    stage1_parser = subparsers.add_parser("stage1", help="Generate stage-1 analysis outputs.")
    stage1_parser.add_argument("--input", required=True)
    stage1_parser.add_argument("--model", required=True)
    stage1_parser.add_argument("--output", required=True)

    stage2_parser = subparsers.add_parser("stage2-synthetic", help="Generate stage-2 synthetic section outputs.")
    stage2_parser.add_argument("--input", required=True)
    stage2_parser.add_argument("--model", required=True)
    stage2_parser.add_argument("--output", required=True)

    assemble_parser = subparsers.add_parser("assemble", help="Assemble section-level JSONL into meeting-level minutes.")
    assemble_parser.add_argument("--input", required=True)
    assemble_parser.add_argument("--output")

    prompt_parser = subparsers.add_parser("minutes-to-decision", help="Convert minutes into decision prompts.")
    prompt_parser.add_argument("--input", required=True)
    prompt_parser.add_argument("--output")
    prompt_parser.add_argument("--template-id", type=int)
    prompt_parser.add_argument("--seed", type=int)

    full_parser = subparsers.add_parser("stage2-full", help="Run section generation, minutes assembly, and decision generation.")
    full_parser.add_argument("--input", required=True)
    full_parser.add_argument("--model", required=True)
    full_parser.add_argument("--output-dir", required=True)
    full_parser.add_argument("--start-index", type=int, default=0)
    full_parser.add_argument("--end-index", type=int, required=True)
    full_parser.add_argument("--generation-batch-size", type=int, default=20)
    full_parser.add_argument("--decision-batch-size", type=int, default=1)
    full_parser.add_argument("--template-id", type=int)
    full_parser.add_argument("--seed", type=int)

    decision_parser = subparsers.add_parser("eval-decision", help="Evaluate a decision-generation file.")
    decision_parser.add_argument("--input", required=True)

    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args()

    if args.command == "stage1":
        run_stage1_generation(args.input, args.output, args.model)
    elif args.command == "stage2-synthetic":
        run_stage2_synthetic_generation(args.input, args.output, args.model)
    elif args.command == "assemble":
        assemble_synthetic_data(args.input, args.output)
    elif args.command == "minutes-to-decision":
        minutes_to_decision_prompt(args.input, args.output, template_id=args.template_id, seed=args.seed)
    elif args.command == "stage2-full":
        run_stage2_synthetic_full(
            model_path=args.model,
            input_prompt_file=args.input,
            output_dir=args.output_dir,
            start_index=args.start_index,
            end_index=args.end_index,
            generation_batch_size=args.generation_batch_size,
            decision_batch_size=args.decision_batch_size,
            template_id=args.template_id,
            seed=args.seed,
        )
    elif args.command == "eval-decision":
        run_eval_decision(args.input)


if __name__ == "__main__":
    main()
