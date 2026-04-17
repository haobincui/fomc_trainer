from log_config import *
import os
import glob
import logging
from generate_new_response import generate_new_response


def validate_output_file(section_output_file: str) -> bool:
    """
    Check whether a single output file already exists.
    """
    return os.path.exists(section_output_file)


def validate_all_outputs(input_prompt_files, simulation_step, output_base_dir):
    """
    Perform a global validation after generation.

    Check whether all expected output files have been successfully generated.

    Parameters
    ----------
    input_prompt_files : list[str]
        List of input prompt .jsonl files.
    simulation_step : int
        Number of expected simulation files for each prompt.
    output_base_dir : str
        Root directory of generated outputs (e.g. 'output/valiation/.../ft_model/')

    Returns
    -------
    None
    """
    logging.info("\n🔍 Starting global validation of generated outputs...")
    missing_files = []

    for input_file in input_prompt_files:
        # 构造对应的输出前缀路径
        output_prefix = input_file.replace(
            "dataset/raw_data/mask_indicator_prompt/",
            output_base_dir
        )

        for i in range(simulation_step):
            section_output_file = output_prefix.replace(".jsonl", f"_{i}.jsonl")
            if not os.path.exists(section_output_file):
                missing_files.append(section_output_file)

    if not missing_files:
        logging.info("✅ All input prompt files have been successfully processed — no missing outputs.")
    else:
        logging.warning(f"⚠️ Detected {len(missing_files)} missing output files:")
        for mf in missing_files:
            logging.warning(f"   └── Missing: {mf}")
        logging.warning("⚠️ Please re-run generation for the missing files above.")


def run_mask_generation(input_prompt_folder, model_path, simulation_step):
    """
    Generate synthetic data for masked prompts.

    Each output file (.jsonl) contains synthetic sections, meeting_dates,
    and rate_changes.
    """
    input_prompt_files = sorted(glob.glob(
        os.path.join("dataset/raw_data/mask_indicator_prompt", input_prompt_folder, "*.jsonl")
    ))

    if not input_prompt_files:
        logging.info(f"⚠️ No .jsonl files found in: {input_prompt_folder}")
        return

    output_base_dir = "output/valiation/generation_stage2_synthetic/mask_prompt/20250602/ft_model/"
    logging.info(f"🚀 Using model: {model_path}")
    total_generated = 0
    total_skipped = 0

    for file_idx, input_prompt_file in enumerate(input_prompt_files, 1):
        try:
            output_path = input_prompt_file.replace(
                "dataset/raw_data/mask_indicator_prompt/",
                output_base_dir
            )

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            logging.info(f"\n📘 [{file_idx}/{len(input_prompt_files)}] Processing: {os.path.basename(input_prompt_file)}")

            for i in range(simulation_step):
                section_output_file = output_path.replace(".jsonl", f"_{i}.jsonl")

                # if the output file exit, pass
                if validate_output_file(section_output_file):
                    logging.info(f"   ⚙️ Skipping existing file: {os.path.basename(section_output_file)}")
                    total_skipped += 1
                    continue

                logging.info(f"   ├── Generating synthetic sample {i+1}/{simulation_step} → {section_output_file}")
                generate_new_response(input_prompt_file, model_path, section_output_file, batch_size=20)
                logging.info(f"   └── ✅ Saved: {section_output_file}")
                total_generated += 1

            logging.info(f"🎉 Completed file: {os.path.basename(input_prompt_file)} "
                         f"(Generated: {total_generated}, Skipped: {total_skipped})")

        except Exception as e:
            logging.error(f"❌ Error processing {input_prompt_file}: {e}")

    logging.info("=" * 80)
    logging.info(f"✅ All done! Total generated: {total_generated} | Skipped: {total_skipped}")
    logging.info("=" * 80)

    # global validation for all output files
    validate_all_outputs(input_prompt_files, simulation_step, output_base_dir)


if __name__ == '__main__':
    input_prompt_folder = "after_2009"
    model_path = 'output/merged/llama_sft_synthetic_20250526'
    simulation_step = 5  

    run_mask_generation(
        input_prompt_folder=input_prompt_folder,
        model_path=model_path,
        simulation_step=simulation_step
    )
