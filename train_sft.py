
from trl import ModelConfig, TrlParser


from open_r1.configs import LoraArguments, SFTConfig, SFTScriptArguments
from open_r1.trainer.sft_trainer import SftTrainer

def main(script_args, training_args, model_args, peft_args):
    # Parse config
    # parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig))
    # script_args, training_args, model_args = parser.parse_args_and_config(config_file)

    # Log summary
    print("✅ Config parsed successfully.")
    print("📦 Model:", model_args.model_name_or_path)
    print("📂 Output directory:", training_args.output_dir)

    # Initialize trainer
    sft_trainer = SftTrainer(script_args, training_args, model_args, peft_args)
    sft_trainer.load_trainer()

    # Start training
    sft_trainer.logger.info("*** Start training ***")
    sft_trainer.start_train()



if __name__ == '__main__':
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig, LoraArguments))
    script_args, training_args, model_args, peft_args= parser.parse_args_and_config()
    main(script_args, training_args, model_args, peft_args)
    # fire.Fire(main)

    
