
import torch
from transformers import AutoModelForCausalLM
from peft import PeftModel



# from src.open_r1.utils.model_utils import load_generation_model

model_path = "/home/haobin_cui/fomc_trainer/models/DeepSeek-R1-Distill-Llama-8B"

adapter_path = "/home/haobin_cui/fomc_trainer/output/adapters/llama_sft_20250521_2_copy"

# base_model = AutoModelForCausalLM.from_pretrained(
#     model_path, 
#     torch_dtype=torch.bfloat16, 
#     device_map="cuda:1",
# )
# model = PeftModel.from_pretrained(base_model, adapter_path)
# merged_model = model.merge_and_unload()
# merged_model.save_pretrained("output/merged/llama_sft_20250522")

# from open_r1.utils.model_utils import load_generation_model
from vllm import LLM, SamplingParams

LLM(
            model=model_path,
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )

model_path = "output/merged/llama_sft_20250522"
from vllm import LLM, SamplingParams

model = LLM(model=model_path,
            dtype="bfloat16",
            max_model_len=8192,
            gpu_memory_utilization=0.9,
            trust_remote_code=True,
        )

# model = load_generation_model(model_path, 0.9, 0.9, 4096)




# model = load_generation_model(model_path,  0.6, 0.6, 4096, adapter_path)


# get_model()



# from trl import ModelConfig, TrlParser


# from open_r1.configs import LoraArguments, SFTConfig, SFTScriptArguments
# from open_r1.trainer.sft_trainer import SftTrainer
# from open_r1.utils.model_utils import get_model

# def main(script_args, training_args, model_args, peft_args):

#     # Log summary
#     print("✅ Config parsed successfully.")
#     print("📦 Model:", model_args.model_name_or_path)
#     print("📂 Output directory:", training_args.output_dir)

#     model = get_model(model_args, training_args)
#     training_args.output_dir
#     model = PeftModel.from_pretrained(model, training_args.output_dir)
    


#     # # Start training
#     # sft_trainer.logger.info("*** 🚀 Start SFT training ***")
#     # sft_trainer.start_train()
#     # sft_trainer.logger.info(f"*** 🎉 Training finished successfully, saved in {training_args.output_dir} ***")

#     # # export
#     # sft_trainer.export_model()
#     # sft_trainer.logger.info(f"*** ✅ Model merged and exported to {peft_args.peft_merged_model_path}***")




# if __name__ == '__main__':
#     parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig, LoraArguments))
#     script_args, training_args, model_args, peft_args= parser.parse_args_and_config()
#     main(script_args, training_args, model_args, peft_args)