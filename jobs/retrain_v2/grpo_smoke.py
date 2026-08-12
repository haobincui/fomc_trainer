"""Data-free GRPO memory gate for one 24 GiB policy GPU.

The rollout phase performs the real long-prompt prefill for the configured
generation batch and keeps a conservative allocation for the completion KV
cache.  The training phase then reproduces TRL's completion-only log-prob
forward/backward for one policy micro-batch.  No repository dataset is read.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


GIB = 1024**3
DEFAULT_PROMPT_LENGTH = 2560
DEFAULT_COMPLETION_LENGTH = 1024
DEFAULT_GENERATION_BATCH_SIZE = 4


def _load_policy(model_path: Path):
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

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
        attn_implementation="sdpa",
        quantization_config=quantization,
        device_map={"": 0},
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(
        model,
        use_gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
    )
    return get_peft_model(
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


def _memory() -> dict[str, float]:
    import torch

    return {
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / GIB, 3),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / GIB, 3),
    }


def _rollout_gate(
    model,
    *,
    prompt_length: int,
    completion_length: int,
    generation_batch_size: int,
) -> dict[str, float]:
    import torch

    config = model.config
    num_layers = int(config.num_hidden_layers)
    num_kv_heads = int(config.num_key_value_heads)
    head_dim = int(getattr(config, "head_dim", config.hidden_size // config.num_attention_heads))
    generator = torch.Generator(device="cuda").manual_seed(4201)
    input_ids = torch.randint(
        0,
        int(config.vocab_size),
        (generation_batch_size, prompt_length),
        generator=generator,
        device="cuda",
    )
    attention_mask = torch.ones_like(input_ids)
    model.eval()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        output = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            logits_to_keep=1,
        )
        # Dynamic generation caches grow one token at a time.  Hold the
        # equivalent remaining K/V bytes while the real prompt cache is live,
        # without spending minutes decoding random tokens.
        completion_kv = torch.empty(
            (
                num_layers,
                2,
                generation_batch_size,
                num_kv_heads,
                completion_length,
                head_dim,
            ),
            dtype=torch.bfloat16,
            device="cuda",
        )
        completion_kv.zero_()
        torch.cuda.synchronize()
        result = _memory()
        # Keep both allocations alive until after peak accounting.
        if output.past_key_values is None or completion_kv.numel() == 0:
            raise RuntimeError("rollout cache smoke did not materialize K/V state")
    del output, completion_kv, input_ids, attention_mask
    torch.cuda.empty_cache()
    return result


def _training_gate(model, *, prompt_length: int, completion_length: int) -> dict[str, float]:
    import torch
    from bitsandbytes.optim import PagedAdamW8bit
    from trl.trainer.utils import selective_log_softmax

    total_length = prompt_length + completion_length
    generator = torch.Generator(device="cuda").manual_seed(4202)
    input_ids = torch.randint(
        0,
        int(model.config.vocab_size),
        (1, total_length),
        generator=generator,
        device="cuda",
    )
    attention_mask = torch.ones_like(input_ids)
    model.config.use_cache = False
    model.train()
    optimizer = PagedAdamW8bit(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=5e-7,
    )
    torch.cuda.reset_peak_memory_stats()
    output = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        use_cache=False,
        logits_to_keep=completion_length + 1,
    )
    logits = output.logits[:, :-1, :]
    logits = logits[:, -completion_length:, :]
    targets = input_ids[:, -completion_length:]
    logps = selective_log_softmax(logits, targets)
    loss = -logps.mean()
    if not torch.isfinite(loss):
        raise RuntimeError("non-finite GRPO smoke loss")
    loss.backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.synchronize()
    result = {"loss": float(loss.detach().cpu()), **_memory()}
    del output, logits, logps, loss, input_ids, attention_mask, targets
    torch.cuda.empty_cache()
    return result


def run_smoke(
    model_path: Path,
    *,
    prompt_length: int,
    completion_length: int,
    generation_batch_size: int,
    memory_gate_gib: float,
) -> dict:
    import torch

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one idle GPU for the GRPO smoke test")
    if min(prompt_length, completion_length, generation_batch_size) <= 0:
        raise ValueError("all GRPO smoke dimensions must be positive")
    model = _load_policy(model_path)
    rollout = _rollout_gate(
        model,
        prompt_length=prompt_length,
        completion_length=completion_length,
        generation_batch_size=generation_batch_size,
    )
    if rollout["peak_reserved_gib"] >= memory_gate_gib:
        raise RuntimeError(
            f"rollout peak reserved {rollout['peak_reserved_gib']:.2f} GiB "
            f"exceeds the {memory_gate_gib:g} GiB gate"
        )
    training = _training_gate(
        model,
        prompt_length=prompt_length,
        completion_length=completion_length,
    )
    if training["peak_reserved_gib"] >= memory_gate_gib:
        raise RuntimeError(
            f"training peak reserved {training['peak_reserved_gib']:.2f} GiB "
            f"exceeds the {memory_gate_gib:g} GiB gate"
        )
    return {
        "status": "passed",
        "model": str(model_path),
        "prompt_length": prompt_length,
        "completion_length": completion_length,
        "generation_batch_size": generation_batch_size,
        "memory_gate_gib": memory_gate_gib,
        "rollout": rollout,
        "training": training,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/DeepSeek-R1-Distill-Llama-8B"),
    )
    parser.add_argument("--prompt-length", type=int, default=DEFAULT_PROMPT_LENGTH)
    parser.add_argument(
        "--completion-length", type=int, default=DEFAULT_COMPLETION_LENGTH
    )
    parser.add_argument(
        "--generation-batch-size",
        type=int,
        default=DEFAULT_GENERATION_BATCH_SIZE,
    )
    parser.add_argument("--memory-gate-gib", type=float, default=22.0)
    args = parser.parse_args()
    try:
        result = run_smoke(
            args.model.resolve(),
            prompt_length=args.prompt_length,
            completion_length=args.completion_length,
            generation_batch_size=args.generation_batch_size,
            memory_gate_gib=args.memory_gate_gib,
        )
    except Exception as exc:  # noqa: BLE001 - CLI converts failures into a gate
        print(json.dumps({"status": "blocked", "error": str(exc)}, ensure_ascii=False))
        return 2
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
