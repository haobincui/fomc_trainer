from log_config import *
import os
import glob
import logging

from generate_new_response import generate_new_response




def run_data_to_analysis_generation(input_prompt_folder, model_path, simulation_step):
    """Generate synthetic data for using mask prompt.

    each output file (.jsonl and .xlsx) contains sections, meeting_dates, and rate_changes.
    generation {total} synthetic file samples.

    saved in output dir.
    """
    # input_prompt_file = "dataset/raw_data/synthetic_text_20250520.jsonl"
    # input_prompt_folder = "dataset/raw_data/mask_indicator_prompt/after_2009"
    input_prompt_files = sorted(glob.glob(os.path.join("dataset/raw_data/data_to_analysis_prompt_no_ref/" + input_prompt_folder, "*.jsonl")))
    if not input_prompt_files:
        logging.info(f"⚠️ No .jsonl files found in: {input_prompt_folder}")
        return

    logging.info(f"using model {model_path}")

    total_generated = 0
    
    for file_idx, input_prompt_file in enumerate(input_prompt_files, 1):
        try:
            # input_prompt_folder = "dataset/raw_data/mask_indicator_prompt/after_2009/Bank-Capital_masked_after_2009.jsonl"
            # output_path = f"output/valiation/generation_stage2_synthetic/mask_prompt/20250602/ft_model/after_2009/Bank-Capital_masked_after_2009.jsonl"

            output_path = input_prompt_file.replace(
                    "dataset/raw_data/data_to_analysis_prompt_no_ref/",
                    "output/valiation/generation_stage2_synthetic/data_to_analysis/20250602/ft_model/"
                )

            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            logging.info(f"\n📘 [{file_idx}/{len(input_prompt_files)}] Processing: {os.path.basename(input_prompt_file)}")


            for i in range(simulation_step):
                section_output_file = output_path.replace(".jsonl", f"_{i}.jsonl")
                logging.info(f"   ├── Generating synthetic sample {i+1}/{simulation_step} → {section_output_file}")
                generate_new_response(input_prompt_file, model_path, section_output_file, batch_size=20)
                logging.info(f"   └── ✅ Saved: {section_output_file}")
                total_generated += 1

            logging.info(f"🎉 Completed file: {os.path.basename(input_prompt_file)} ({simulation_step} generated)")
        except Exception as e:
            logging.info(f"❌ Error processing {input_prompt_file}: {e}")



    logging.info("=" * 80)
    logging.info(f"✅ All done! Total generated files: {total_generated}")
    logging.info("=" * 80)


if __name__ == '__main__':
    input_prompt_folder = "after_2009"
    model_path = 'output/merged/llama_sft_synthetic_20250526'
    simulation_step = 1

    run_data_to_analysis_generation(input_prompt_folder=input_prompt_folder,
                        model_path=model_path,
                        simulation_step=simulation_step)
    

    
