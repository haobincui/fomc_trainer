
from trl import ModelConfig, ScriptArguments, SFTTrainer, TrlParser, get_peft_config, setup_chat_format

from open_r1.configs import SFTConfig
from open_r1.trainer.sft_trainer import SftTrainer
from open_r1.trainer.trainer import Trainer

def main(config_file: str):
    # Parse config
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config(config_file)

    # Log summary
    print("✅ Config parsed successfully.")
    print("📦 Model:", model_args.model_name_or_path)
    print("📂 Output directory:", training_args.output_dir)

    # Initialize trainer
    sft_trainer = SftTrainer(script_args, training_args, model_args)
    sft_trainer.load_trainer()

    # Start training
    sft_trainer.logger.info("*** Start training ***")
    sft_trainer.start_train()



if __name__ == '__main__':
    pass

    # main(script_args, training_args, model_args)