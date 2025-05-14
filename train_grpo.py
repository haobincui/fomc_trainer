from trl import ModelConfig, TrlParser

from open_r1.configs import LoraArguments, GRPOScriptArguments, GRPOConfig
from open_r1.trainer.grpo_trainer import GrpoTrainer


def main(script_args, training_args, model_args, peft_args):

    # Log summary
    print("✅ Config parsed successfully.")
    print("📦 Model:", model_args.model_name_or_path)
    print("📂 Output directory:", training_args.output_dir)

    # Initialize trainer
    grpo_trainer = GrpoTrainer(
        script_args=script_args,
        training_args=training_args,
        model_args=model_args,
        peft_args=peft_args
    )

    # Start training
    grpo_trainer.logger.info("*** 🚀 Start SFT training ***")
    grpo_trainer.start_train()
    grpo_trainer.logger.info("*** 🎉 Training finished successfully ***")


if __name__ == '__main__':
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig, LoraArguments))
    script_args, training_args, model_args, peft_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args, peft_args)



