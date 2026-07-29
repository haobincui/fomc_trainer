import argparse
import glob
import logging
import os

from generate_new_response import generate_new_response


def validate_output_file(section_output_file: str) -> bool:
    return os.path.exists(section_output_file)


def validate_all_outputs(input_prompt_files: list[str], simulation_step: int, input_dir: str, output_dir: str) -> None:
    logging.info("🔍 Starting global validation of generated outputs...")
    missing_files = []
    for input_file in input_prompt_files:
        relative_path = os.path.relpath(input_file, input_dir)
        output_prefix = os.path.join(output_dir, relative_path)
        for i in range(simulation_step):
            section_output_file = output_prefix.replace(".jsonl", f"_{i}.jsonl")
            if not os.path.exists(section_output_file):
                missing_files.append(section_output_file)

    if not missing_files:
        logging.info("✅ All input prompt files have been successfully processed.")
        return

    logging.warning("⚠️ Detected %s missing output files:", len(missing_files))
    for missing_file in missing_files:
        logging.warning("   └── Missing: %s", missing_file)


def resolve_input_dir(input_folder: str) -> str:
    if os.path.isdir(input_folder):
        return input_folder
    return os.path.join("dataset/processed/main/evaluation_inputs/source_prompts/mask_indicator", input_folder)


def run_mask_generation(
    *,
    input_folder: str,
    model_path: str,
    simulation_step: int,
    output_dir: str,
    batch_size: int = 20,
) -> None:
    input_dir = resolve_input_dir(input_folder)
    input_prompt_files = sorted(glob.glob(os.path.join(input_dir, "*.jsonl")))

    if not input_prompt_files:
        logging.info("⚠️ No .jsonl files found in: %s", input_dir)
        return

    logging.info("🚀 Using model: %s", model_path)
    total_generated = 0
    total_skipped = 0

    for file_idx, input_prompt_file in enumerate(input_prompt_files, 1):
        relative_path = os.path.relpath(input_prompt_file, input_dir)
        output_prefix = os.path.join(output_dir, relative_path)
        os.makedirs(os.path.dirname(output_prefix), exist_ok=True)
        logging.info("📘 [%s/%s] Processing: %s", file_idx, len(input_prompt_files), os.path.basename(input_prompt_file))

        for i in range(simulation_step):
            section_output_file = output_prefix.replace(".jsonl", f"_{i}.jsonl")
            if validate_output_file(section_output_file):
                logging.info("   ⚙️ Skipping existing file: %s", os.path.basename(section_output_file))
                total_skipped += 1
                continue

            logging.info("   ├── Generating synthetic sample %s/%s → %s", i + 1, simulation_step, section_output_file)
            generate_new_response(input_prompt_file, model_path, section_output_file, batch_size=batch_size)
            total_generated += 1

    logging.info("✅ All done! Total generated: %s | Skipped: %s", total_generated, total_skipped)
    validate_all_outputs(input_prompt_files, simulation_step, input_dir, output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate masked prompt outputs for leave-one-out masking evaluation.")
    parser.add_argument("--input-folder", required=True, help="Input folder name under dataset/processed/main/evaluation_inputs/... or an absolute path.")
    parser.add_argument("--model", required=True, help="Model path used for generation.")
    parser.add_argument("--simulation-step", type=int, default=5)
    parser.add_argument("--output-dir", required=True, help="Destination directory for generated files.")
    parser.add_argument("--batch-size", type=int, default=20)
    return parser


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    args = build_parser().parse_args()
    run_mask_generation(
        input_folder=args.input_folder,
        model_path=args.model,
        simulation_step=args.simulation_step,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
    )
