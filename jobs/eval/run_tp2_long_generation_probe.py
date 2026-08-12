"""Run a single-model, two-GPU long-generation diagnostic with vLLM.

This is deliberately separate from the immutable checkpoint-generation
evaluator.  It records a new diagnostic artifact and never mutates an existing
generation manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace(
        "+00:00", "Z"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def trigram_repetition(text: str, *, tail_words: int | None = None) -> float:
    words = re.findall(r"\w+|[^\w\s]", text.lower(), flags=re.UNICODE)
    if tail_words is not None:
        words = words[-tail_words:]
    grams = [tuple(words[index : index + 3]) for index in range(len(words) - 2)]
    if not grams:
        return 0.0
    return 1.0 - len(set(grams)) / len(grams)


def gpu_snapshot() -> list[dict[str, Any]]:
    command = [
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,memory.used,memory.free,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    rows = []
    for line in completed.stdout.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 6:
            raise ValueError(f"Unexpected nvidia-smi row: {line!r}")
        rows.append(
            {
                "index": int(fields[0]),
                "name": fields[1],
                "memory_total_mib": int(fields[2]),
                "memory_used_mib": int(fields[3]),
                "memory_free_mib": int(fields[4]),
                "utilization_gpu_pct": int(fields[5]),
            }
        )
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--prompts", required=True, type=Path)
    parser.add_argument("--evaluation-config", required=True, type=Path)
    parser.add_argument("--sample-id", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--tensor-parallel-size", type=int, default=2)
    parser.add_argument("--max-model-len", type=int, default=65_536)
    parser.add_argument("--max-new-tokens", type=int, default=52_000)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20_260_729)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    model = args.model.expanduser().resolve()
    prompts = args.prompts.expanduser().resolve()
    evaluation_config = args.evaluation_config.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()

    if args.tensor_parallel_size != 2:
        raise ValueError("This diagnostic is intentionally fixed to TP=2")
    visible_devices = [
        value.strip()
        for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if value.strip()
    ]
    if visible_devices != ["0", "1"]:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES must be exactly '0,1' for this diagnostic; "
            f"observed={visible_devices!r}"
        )
    if not model.is_dir():
        raise FileNotFoundError(f"Model directory is missing: {model}")
    if not prompts.is_file() or not evaluation_config.is_file():
        raise FileNotFoundError("Prompt/config input is missing")
    if not 0.0 < args.gpu_memory_utilization < 1.0:
        raise ValueError("gpu-memory-utilization must be in (0, 1)")

    rows = read_jsonl(prompts)
    selected = [row for row in rows if row.get("sample_id") == args.sample_id]
    if len(selected) != 1:
        raise ValueError(
            f"Expected exactly one row for sample_id={args.sample_id!r}, got {len(selected)}"
        )
    row = selected[0]
    config = json.loads(evaluation_config.read_text(encoding="utf-8"))
    system_prompt = str(config["generation"]["system_prompt"])
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": str(row["prompt"])},
    ]

    output_dir.mkdir(parents=True, exist_ok=False)
    launch = {
        "schema_version": "tp2-long-generation-probe-v1",
        "status": "initializing",
        "created_at": utc_now(),
        "python": sys.executable,
        "python_version": platform.python_version(),
        "model": str(model),
        "model_config_sha256": sha256_file(model / "config.json"),
        "prompts": str(prompts),
        "prompts_sha256": sha256_file(prompts),
        "evaluation_config": str(evaluation_config),
        "evaluation_config_sha256": sha256_file(evaluation_config),
        "sample_id": args.sample_id,
        "source_prompt_sha256": text_sha256(str(row["prompt"])),
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_model_len": args.max_model_len,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "repetition_penalty": args.repetition_penalty,
        "seed": args.seed,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "cuda_visible_devices": visible_devices,
        "gpu_before": gpu_snapshot(),
    }
    (output_dir / "launch.json").write_text(
        json.dumps(launch, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(canonical_json({"event": "launch", **launch}), flush=True)

    started = time.monotonic()
    try:
        import torch
        import transformers
        import vllm
        from vllm import LLM, SamplingParams

        if vllm.__version__ != "0.8.5.post1":
            raise RuntimeError(f"Expected vLLM 0.8.5.post1, got {vllm.__version__}")
        engine = LLM(
            model=str(model),
            tokenizer=str(model),
            dtype="bfloat16",
            max_model_len=args.max_model_len,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=args.tensor_parallel_size,
            trust_remote_code=True,
            enable_prefix_caching=True,
            enable_chunked_prefill=True,
            max_num_seqs=1,
            max_num_batched_tokens=8192,
            disable_custom_all_reduce=True,
        )
        tokenizer = engine.get_tokenizer()
        prompt_token_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
        )
        prompt_token_count = len(prompt_token_ids)
        requested_total = prompt_token_count + args.max_new_tokens
        if requested_total > args.max_model_len:
            raise ValueError(
                "Preflight context overflow: "
                f"prompt={prompt_token_count}, output={args.max_new_tokens}, "
                f"model_len={args.max_model_len}"
            )
        preflight = {
            "event": "preflight_passed",
            "at": utc_now(),
            "prompt_token_count": prompt_token_count,
            "requested_total_tokens": requested_total,
            "context_headroom_tokens": args.max_model_len - requested_total,
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "vllm": vllm.__version__,
            "gpu_loaded": gpu_snapshot(),
        }
        (output_dir / "preflight.json").write_text(
            json.dumps(preflight, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(canonical_json(preflight), flush=True)

        sampling = SamplingParams(
            temperature=args.temperature,
            top_p=args.top_p,
            repetition_penalty=args.repetition_penalty,
            max_tokens=args.max_new_tokens,
            seed=args.seed,
        )
        outputs = engine.chat([messages], sampling_params=sampling, use_tqdm=True)
        if len(outputs) != 1 or len(outputs[0].outputs) != 1:
            raise RuntimeError("vLLM returned an unexpected output count")
        request_output = outputs[0]
        completion = request_output.outputs[0]
        generated = completion.text
        output_token_ids = list(completion.token_ids)
        observed_prompt_ids = list(request_output.prompt_token_ids)
        if len(observed_prompt_ids) != prompt_token_count:
            raise RuntimeError(
                "vLLM prompt-token count differs from preflight: "
                f"observed={len(observed_prompt_ids)}, preflight={prompt_token_count}"
            )

        answer = generated.split("</think>", 1)[1].strip() if "</think>" in generated else ""
        elapsed = time.monotonic() - started
        result = {
            "schema_version": "tp2-long-generation-result-v1",
            "status": "completed",
            "completed_at": utc_now(),
            "sample_id": args.sample_id,
            "prompt_token_count": prompt_token_count,
            "output_token_count": len(output_token_ids),
            "max_new_tokens": args.max_new_tokens,
            "max_model_len": args.max_model_len,
            "finish_reason": completion.finish_reason,
            "stop_reason": completion.stop_reason,
            "input_was_truncated": False,
            "has_think_boundary": "</think>" in generated,
            "has_nonempty_answer": bool(answer),
            "full_trigram_repetition": trigram_repetition(generated),
            "tail_2048_word_trigram_repetition": trigram_repetition(
                generated, tail_words=2048
            ),
            "generated_sha256": text_sha256(generated),
            "elapsed_seconds": elapsed,
            "output_tokens_per_second": len(output_token_ids) / elapsed,
            "gpu_after": gpu_snapshot(),
            "generated": generated,
        }
        (output_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        summary = {key: value for key, value in result.items() if key != "generated"}
        summary["generated_prefix"] = generated[:500]
        summary["generated_suffix"] = generated[-1000:]
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(canonical_json({"event": "completed", **summary}), flush=True)
        return 0
    except BaseException as error:
        failure = {
            "schema_version": "tp2-long-generation-failure-v1",
            "status": "failed",
            "failed_at": utc_now(),
            "sample_id": args.sample_id,
            "error_type": type(error).__name__,
            "error": str(error),
            "elapsed_seconds": time.monotonic() - started,
            "gpu_at_failure": gpu_snapshot(),
        }
        (output_dir / "failure.json").write_text(
            json.dumps(failure, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(canonical_json({"event": "failed", **failure}), flush=True)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
