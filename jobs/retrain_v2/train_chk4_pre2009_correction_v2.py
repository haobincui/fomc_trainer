"""Train chk4 correction-v2 with its independent immutable schedule runtime."""

from __future__ import annotations

from dataclasses import dataclass

from trl import SFTTrainer, TrlParser

from open_r1.configs import LoraArguments, ModelConfig, SFTConfig, SFTScriptArguments
from open_r1.trainer.dataset_release_correction_v2 import (
    verify_correction_v2_runtime_binding,
)
from open_r1.trainer.fixed_schedule_sampler_correction_v2 import (
    SAMPLER_TYPE,
    ManifestFixedScheduleCorrectionV2SamplerMixin,
    validate_dataset_order,
    validate_runtime_contract,
)
from open_r1.trainer.sft_prompt_renderer import render_sft_prompt
from open_r1.trainer.sft_trainer import SftTrainer


class ManifestFixedScheduleCorrectionV2SFTTrainer(
    ManifestFixedScheduleCorrectionV2SamplerMixin, SFTTrainer
):
    """TRL trainer whose sampler is the sealed correction-v2 row order."""


@dataclass
class CorrectionV2SFTConfig(SFTConfig):
    """Additive config gate for the independent correction-v2 sampler."""

    def __post_init__(self) -> None:
        requested_sampler = self.train_sampler
        if requested_sampler != SAMPLER_TYPE:
            raise ValueError(f"correction-v2 requires train_sampler={SAMPLER_TYPE}")
        # Historical SFTConfig is receipt-bound and intentionally unchanged.
        # Let it validate every shared field through its existing default path,
        # then restore the independent sampler value used only by this entrypoint.
        self.train_sampler = "default"
        try:
            super().__post_init__()
        finally:
            self.train_sampler = requested_sampler


class CorrectionV2SftTrainer(SftTrainer):
    """Existing training stack with additive v2 release/sampler dispatch."""

    def _verify_chk4_release_binding(self, binding: dict[str, object]):
        cached = getattr(self, "_chk4_release_binding", None)
        if cached is None:
            cached = verify_correction_v2_runtime_binding(
                dataset_dir=self.script_args.dataset_name,
                manifest_path=binding["dataset_chk4_release_manifest"],
                expected_manifest_sha256=str(
                    binding["dataset_chk4_release_manifest_sha256"]
                ),
                dataset_role=str(binding["dataset_chk4_role"]),
                system_prompt=self.training_args.system_prompt,
                model_path=self.model_args.model_name_or_path,
            )
            self._chk4_release_binding = cached
        return cached

    def load_trainer(self):
        if self.training_args.train_sampler != SAMPLER_TYPE:
            raise ValueError(f"correction-v2 requires train_sampler={SAMPLER_TYPE}")
        binding_mode, binding = self._dataset_binding()
        if binding_mode != "chk4_decision_release":
            raise ValueError("correction-v2 requires a chk4 release binding")
        verified = self._verify_chk4_release_binding(binding)
        sampler_receipt = validate_runtime_contract(
            training_args=self.training_args,
            release_binding=verified,
        )

        raw_dataset = self.dataset
        raw_train = raw_dataset[self.script_args.dataset_train_split]
        # This must happen before TRL removes metadata columns.
        validate_dataset_order(raw_train)

        def convert_chat(example):
            if isinstance(example["prompt"], list):
                return {"prompt": render_sft_prompt(self.tokenizer, example["prompt"])}
            return example

        dataset = raw_dataset.map(convert_chat).rename_column("response", "completion")
        trainer = ManifestFixedScheduleCorrectionV2SFTTrainer(
            model=self.model,
            args=self.training_args,
            train_dataset=dataset[self.script_args.dataset_train_split],
            eval_dataset=(
                dataset[self.script_args.dataset_test_split]
                if self.training_args.eval_strategy != "no"
                else None
            ),
            processing_class=self.tokenizer,
            peft_config=self.peft_config,
            callbacks=self.load_callbacks(),
        )
        trainer._manifest_fixed_schedule_expected_rows = sampler_receipt["schedule"][
            "rows"
        ]
        return trainer

    def _build_runtime_config(self) -> dict:
        runtime = super()._build_runtime_config()
        binding_mode, binding = self._dataset_binding()
        if binding_mode != "chk4_decision_release":
            raise ValueError("correction-v2 runtime lost its dataset binding")
        verified = self._verify_chk4_release_binding(binding)
        runtime["training_sampler"] = validate_runtime_contract(
            training_args=self.training_args,
            release_binding=verified,
        )
        runtime["schema_version"] = "chk4-correction-v2-resolved-runtime-v1"
        return runtime


def main(
    script_args: SFTScriptArguments,
    training_args: CorrectionV2SFTConfig,
    model_args: ModelConfig,
    peft_args: LoraArguments,
) -> None:
    trainer = CorrectionV2SftTrainer(
        script_args=script_args,
        training_args=training_args,
        model_args=model_args,
        peft_args=peft_args,
    )
    trainer.logger.info("*** Start chk4 correction-v2 SFT ***")
    trainer.start_train()
    trainer.logger.info("*** chk4 correction-v2 SFT completed successfully ***")


if __name__ == "__main__":
    parser = TrlParser(
        (SFTScriptArguments, CorrectionV2SFTConfig, ModelConfig, LoraArguments)
    )
    parsed = parser.parse_args_and_config()
    main(*parsed)
