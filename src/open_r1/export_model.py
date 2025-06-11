




def export_model(model_args: ModelConfig, training_args: SFTConfig | GRPOConfig):
        """Get the model"""
        torch_dtype = (
            model_args.torch_dtype
            if model_args.torch_dtype in ["auto", None]
            else getattr(torch, model_args.torch_dtype)  # type: ignore
        )
        quantization_config = get_quantization_config(model_args)

        model_kwargs = dict(
            revision=model_args.model_revision,
            trust_remote_code=model_args.trust_remote_code,
            attn_implementation=model_args.attn_implementation,
            torch_dtype=torch_dtype,
            use_cache=False if training_args.gradient_checkpointing else True,
            # device_map=get_kbit_device_map() if quantization_config is not None else None,
            device_map=None,
        )
        if quantization_config:
            model_kwargs["quantization_config"]
        model = AutoModelForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            **model_kwargs,
        )
        return model


