"""Model-only QLoRA forward/backward smoke test without loading training data."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path


DEFAULT_SEQUENCE_LENGTH = 4096
DEFAULT_USE_LIGER = True


def run_smoke(
    model_path: Path,
    sequence_length: int,
    use_liger: bool,
    activation_offloading: bool = False,
    checkpoint_use_reentrant: bool = False,
    attn_implementation: str = "sdpa",
    distributed: bool = False,
) -> dict:
    import torch
    import torch.distributed as dist
    from bitsandbytes.optim import PagedAdamW8bit
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the model smoke test")
    if distributed:
        world_size = int(os.environ.get("WORLD_SIZE", "0"))
        local_rank = int(os.environ.get("LOCAL_RANK", "-1"))
        if world_size != 2 or torch.cuda.device_count() != 2 or local_rank not in (0, 1):
            raise RuntimeError(
                "Distributed smoke requires torchrun with exactly two visible GPUs"
            )
        if not dist.is_initialized():
            dist.init_process_group(backend="nccl")
    else:
        world_size = 1
        local_rank = 0
        if torch.cuda.device_count() != 1:
            raise RuntimeError("Expose exactly one idle GPU for the model smoke test")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if sequence_length <= 0:
        raise ValueError("sequence_length must be positive")

    quantization = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_storage=torch.bfloat16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        dtype=torch.bfloat16,
        attn_implementation=attn_implementation,
        quantization_config=quantization,
        device_map={"": local_rank},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={
            "use_reentrant": checkpoint_use_reentrant
        },
    )
    model = get_peft_model(
        model,
        LoraConfig(
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
            ],
        ),
    )
    if use_liger:
        # Match the SFT Trainer integration.  In particular, Llama's default
        # Liger patch uses fused linear cross-entropy so the full
        # [sequence, vocabulary] logits tensor is never materialized.
        from transformers.integrations.liger import apply_liger_kernel

        apply_liger_kernel(model, None)
    model.train()
    if distributed:
        from torch.nn.parallel import DistributedDataParallel

        training_model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=False,
        )
    else:
        training_model = model
    optimizer = PagedAdamW8bit(
        (
            parameter
            for parameter in training_model.parameters()
            if parameter.requires_grad
        ),
        lr=5e-5,
    )

    vocab_size = int(model.config.vocab_size)
    generator = torch.Generator(device=device).manual_seed(42 + local_rank)
    input_ids = torch.randint(
        0,
        vocab_size,
        (1, sequence_length),
        generator=generator,
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    torch.cuda.reset_peak_memory_stats()
    if activation_offloading:
        from trl.models.activation_offloading import get_act_offloading_ctx_manager

        activation_context = get_act_offloading_ctx_manager(model=training_model)
    else:
        from contextlib import nullcontext

        activation_context = nullcontext()
    with activation_context:
        output = training_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=input_ids,
            use_cache=False,
        )
        if not torch.isfinite(output.loss):
            raise RuntimeError("non-finite smoke-test loss")
        output.loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    peak_bytes = torch.cuda.max_memory_allocated()
    peak_reserved_bytes = torch.cuda.max_memory_reserved()
    peak_gib = peak_bytes / 1024**3
    peak_reserved_gib = peak_reserved_bytes / 1024**3
    if peak_reserved_gib >= 22:
        raise RuntimeError(
            f"peak reserved memory {peak_reserved_gib:.2f} GiB exceeds the 22 GiB gate"
        )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    local_result = {
        "rank": local_rank,
        "status": "passed",
        "model": str(model_path),
        "sequence_length": sequence_length,
        "use_liger_kernel": use_liger,
        "activation_offloading": activation_offloading,
        "checkpoint_use_reentrant": checkpoint_use_reentrant,
        "attn_implementation": attn_implementation,
        "loss": float(output.loss.detach().cpu()),
        "trainable_parameters": trainable,
        "optimizer": "PagedAdamW8bit",
        "peak_allocated_gib": round(peak_gib, 3),
        "peak_reserved_gib": round(peak_reserved_gib, 3),
    }
    if not distributed:
        return local_result

    rank_results: list[dict | None] = [None for _ in range(world_size)]
    dist.all_gather_object(rank_results, local_result)
    checked_results = [item for item in rank_results if item is not None]
    if len(checked_results) != world_size:
        raise RuntimeError("DDP smoke did not collect both rank results")
    return {
        "status": "passed",
        "distributed": "ddp",
        "world_size": world_size,
        "sequence_length": sequence_length,
        "attn_implementation": attn_implementation,
        "use_liger_kernel": use_liger,
        "optimizer": "PagedAdamW8bit",
        "max_peak_allocated_gib": max(
            item["peak_allocated_gib"] for item in checked_results
        ),
        "max_peak_reserved_gib": max(
            item["peak_reserved_gib"] for item in checked_results
        ),
        "ranks": checked_results,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/DeepSeek-R1-Distill-Llama-8B"),
    )
    parser.add_argument("--sequence-length", type=int, default=DEFAULT_SEQUENCE_LENGTH)
    parser.add_argument(
        "--attn-implementation",
        default="sdpa",
        choices=("sdpa", "flex_attention", "flash_attention_2"),
    )
    parser.add_argument(
        "--use-liger",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_USE_LIGER,
        help="Exercise the SFT Liger path (default: enabled; use --no-use-liger to disable).",
    )
    parser.add_argument(
        "--activation-offloading",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Exercise TRL activation offloading (default: disabled).",
    )
    parser.add_argument(
        "--checkpoint-use-reentrant",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use reentrant gradient checkpointing (default: disabled).",
    )
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Run a two-rank NCCL DDP smoke under torchrun.",
    )
    args = parser.parse_args()
    try:
        result = run_smoke(
            args.model.resolve(),
            args.sequence_length,
            args.use_liger,
            args.activation_offloading,
            args.checkpoint_use_reentrant,
            args.attn_implementation,
            args.distributed,
        )
    except Exception as exc:  # noqa: BLE001 - CLI converts failures into a gate
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    if not args.distributed or int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    try:
        import torch.distributed as dist

        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
    except Exception:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
