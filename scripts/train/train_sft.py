
from trl import ModelConfig, TrlParser


from open_r1.configs import LoraArguments, SFTConfig, SFTScriptArguments
from open_r1.trainer.sft_trainer import SftTrainer

def main(script_args, training_args, model_args, peft_args):

    # Log summary
    print("✅ Config parsed successfully.")
    print("📦 Model:", model_args.model_name_or_path)
    print("📂 Output directory:", training_args.output_dir)

    # Initialize trainer
    sft_trainer = SftTrainer(
        script_args=script_args,
        training_args=training_args,
        model_args=model_args,
        peft_args=peft_args
    )

    # Start training
    sft_trainer.logger.info("*** 🚀 Start SFT training ***")
    sft_trainer.start_train()
    sft_trainer.logger.info(f"*** 🎉 Training finished successfully, saved in {training_args.output_dir} ***")

    # export
    # sft_trainer.export_model()
    # sft_trainer.logger.info(f"*** ✅ Model merged and exported to {peft_args.peft_merged_model_path}***")




if __name__ == '__main__':
    parser = TrlParser((SFTScriptArguments, SFTConfig, ModelConfig, LoraArguments))
    script_args, training_args, model_args, peft_args= parser.parse_args_and_config()
    main(script_args, training_args, model_args, peft_args)


    
