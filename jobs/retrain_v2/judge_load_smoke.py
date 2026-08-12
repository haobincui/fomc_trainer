"""Four-concurrent, synthetic long-context load smoke for the v2 judge.

This command never reads training data.  It builds deterministic synthetic
indicator rows in memory, sizes the prompt with the local Qwen tokenizer, and
reuses the production judge request path (strict JSON schema and thinking off).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import threading
import time
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

from transformers import AutoTokenizer

from open_r1.trainer.rewards.reward_funcs.analysis_reward_v2 import (
    _JUDGE_SYSTEM_PROMPT,
    _judge_one,
)


DEFAULT_URL = "http://127.0.0.1:8000/v1/chat/completions"
DEFAULT_MODEL = "Qwen3.5-9B"
DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "Qwen3.5-9B"


def build_synthetic_evidence(row_count: int, *, request_id: int) -> str:
    """Return deterministic fake evidence with no real-world meeting content."""

    if row_count < 1:
        raise ValueError("row_count must be positive")
    rows = [
        "SYNTHETIC_LOAD_SMOKE=1; all values below are invented and are not real data."
    ]
    trends = ("rising", "falling", "unchanged", "mixed")
    for index in range(row_count):
        value = 50.0 + ((request_id * 17 + index * 13) % 401) / 10.0
        change = ((request_id * 11 + index * 7) % 61 - 30) / 10.0
        rows.append(
            f"record_id=SYN-{request_id:02d}-{index:05d}; "
            f"indicator=synthetic_series_{index % 37:02d}; period=t-{index % 48:02d}; "
            f"level={value:.1f} synthetic_units; change={change:+.1f} synthetic_units; "
            f"trend={trends[(request_id + index) % len(trends)]}; "
            "provenance=generated_in_memory_for_capacity_test"
        )
    return "\n".join(rows)


def synthetic_candidate(request_id: int) -> str:
    return (
        f"Synthetic request {request_id} contains a mixture of rising, falling, "
        "and unchanged invented series. The evidence is suitable only for a "
        "capacity test, so no real-world policy conclusion should be inferred."
    )


def count_chat_tokens(tokenizer: Any, evidence: str, candidate: str) -> int:
    user_payload = json.dumps(
        {"evidence": evidence, "candidate": candidate},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    messages = [
        {"role": "system", "content": _JUDGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_payload},
    ]
    tokens = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    # Transformers 5 may return a BatchEncoding even without
    # ``return_dict=True``; older releases return the input-id list directly.
    if isinstance(tokens, Mapping):
        tokens = tokens.get("input_ids")
        if tokens is None:
            raise RuntimeError("chat template result has no input_ids")
    if hasattr(tokens, "shape") and len(tokens.shape) == 2:
        return int(tokens.shape[-1])
    if tokens and isinstance(tokens[0], (list, tuple)):
        return len(tokens[0])
    return len(tokens)


def size_synthetic_prompt(
    tokenizer: Any,
    *,
    request_id: int,
    target_prompt_tokens: int,
) -> tuple[str, str, int, int]:
    """Find the largest whole-row prompt no longer than the requested target."""

    if target_prompt_tokens < 256:
        raise ValueError("target_prompt_tokens must be at least 256")
    candidate = synthetic_candidate(request_id)
    low, high = 1, 64
    while True:
        evidence = build_synthetic_evidence(high, request_id=request_id)
        if count_chat_tokens(tokenizer, evidence, candidate) >= target_prompt_tokens:
            break
        low, high = high, high * 2
        if high > 100_000:
            raise RuntimeError("could not size synthetic prompt")

    best = (build_synthetic_evidence(low, request_id=request_id), low)
    while low <= high:
        middle = (low + high) // 2
        evidence = build_synthetic_evidence(middle, request_id=request_id)
        count = count_chat_tokens(tokenizer, evidence, candidate)
        if count <= target_prompt_tokens:
            best = (evidence, middle)
            low = middle + 1
        else:
            high = middle - 1
    evidence, rows = best
    return evidence, candidate, count_chat_tokens(tokenizer, evidence, candidate), rows


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    rank = max(0, math.ceil(percentile * len(ordered)) - 1)
    return ordered[rank]


def read_gpu_memory_mib(gpu_index: int) -> int:
    completed = subprocess.run(
        [
            "nvidia-smi",
            f"--id={gpu_index}",
            "--query-gpu=memory.used",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=5,
    )
    return int(completed.stdout.strip())


def run_load_smoke(
    *,
    url: str,
    model: str,
    tokenizer: Any,
    concurrency: int,
    target_prompt_tokens: int,
    timeout: int,
    judge_fn: Callable[..., tuple[dict[str, Any], str, int]] = _judge_one,
    gpu_index: int | None = None,
    max_gpu_memory_mib: int = 22_000,
    memory_poll_interval: float = 0.2,
    memory_reader: Callable[[int], int] = read_gpu_memory_mib,
) -> dict[str, Any]:
    if concurrency < 1:
        raise ValueError("concurrency must be positive")

    prepared = []
    for request_id in range(concurrency):
        evidence, candidate, prompt_tokens, rows = size_synthetic_prompt(
            tokenizer,
            request_id=request_id,
            target_prompt_tokens=target_prompt_tokens,
        )
        prepared.append((request_id, evidence, candidate, prompt_tokens, rows))

    def invoke(item: tuple[int, str, str, int, int]) -> dict[str, Any]:
        request_id, evidence, candidate, prompt_tokens, rows = item
        started = time.perf_counter()
        try:
            evaluation, _raw, attempts = judge_fn(
                evidence=evidence,
                candidate=candidate,
                url=url,
                model=model,
                timeout=timeout,
                api_key=os.environ.get("OPEN_R1_JUDGE_API_KEY"),
                max_retries=1,
                backoff_seconds=0,
            )
            return {
                "request_id": request_id,
                "status": "passed",
                "latency_seconds": time.perf_counter() - started,
                "prompt_tokens": prompt_tokens,
                "synthetic_rows": rows,
                "attempts": attempts,
                "rubric_keys": sorted(evaluation),
            }
        except Exception as exc:  # noqa: BLE001 - aggregate every concurrent result
            return {
                "request_id": request_id,
                "status": "failed",
                "latency_seconds": time.perf_counter() - started,
                "prompt_tokens": prompt_tokens,
                "synthetic_rows": rows,
                "error": f"{type(exc).__name__}: {exc}",
            }

    memory_samples: list[int] = []
    stop_sampling = threading.Event()

    def sample_memory() -> None:
        while not stop_sampling.is_set():
            memory_samples.append(memory_reader(gpu_index))  # type: ignore[arg-type]
            stop_sampling.wait(memory_poll_interval)

    sampler = None
    if gpu_index is not None:
        sampler = threading.Thread(target=sample_memory, name="judge-gpu-sampler", daemon=True)
        sampler.start()

    wall_started = time.perf_counter()
    try:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [executor.submit(invoke, item) for item in prepared]
            results = [future.result() for future in as_completed(futures)]
    finally:
        stop_sampling.set()
        if sampler is not None:
            sampler.join(timeout=max(1.0, memory_poll_interval * 2))
    wall_seconds = time.perf_counter() - wall_started
    results.sort(key=lambda result: result["request_id"])

    passed = sum(result["status"] == "passed" for result in results)
    latencies = [float(result["latency_seconds"]) for result in results]
    output = {
        "status": "passed" if passed == concurrency else "failed",
        "input_kind": "in_memory_synthetic_only",
        "strict_json_schema": True,
        "enable_thinking": False,
        "url": url,
        "model": model,
        "concurrency": concurrency,
        "passed": passed,
        "success_rate": passed / concurrency,
        "target_prompt_tokens": target_prompt_tokens,
        "min_prompt_tokens": min(item[3] for item in prepared),
        "max_prompt_tokens": max(item[3] for item in prepared),
        "wall_seconds": wall_seconds,
        "latency_seconds": {
            "min": min(latencies),
            "p50": statistics.median(latencies),
            "p95": _percentile(latencies, 0.95),
            "max": max(latencies),
        },
        "requests": results,
    }
    if gpu_index is not None:
        if not memory_samples:
            raise RuntimeError(f"no GPU memory samples collected for GPU {gpu_index}")
        peak = max(memory_samples)
        memory_gate = peak < max_gpu_memory_mib
        output.update(
            {
                "gpu_index": gpu_index,
                "gpu_memory_samples": len(memory_samples),
                "peak_gpu_memory_mib": peak,
                "max_gpu_memory_mib_exclusive": max_gpu_memory_mib,
                "gpu_memory_gate_passed": memory_gate,
            }
        )
        if not memory_gate:
            output["status"] = "failed"
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default=os.environ.get("OPEN_R1_JUDGE_URL", DEFAULT_URL))
    parser.add_argument("--model", default=os.environ.get("OPEN_R1_JUDGE_MODEL", DEFAULT_MODEL))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path(os.environ.get("FOMC_RETRAIN_JUDGE_MODEL_PATH", DEFAULT_MODEL_PATH)),
    )
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--target-prompt-tokens", type=int, default=5000)
    parser.add_argument("--minimum-prompt-tokens", type=int, default=4800)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=int(os.environ.get("FOMC_RETRAIN_JUDGE_GPU_ID", "0")),
        help="Physical GPU index to sample with nvidia-smi.",
    )
    parser.add_argument("--max-gpu-memory-mib", type=int, default=22_000)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
        trust_remote_code=True,
    )
    result = run_load_smoke(
        url=args.url,
        model=args.model,
        tokenizer=tokenizer,
        concurrency=args.concurrency,
        target_prompt_tokens=args.target_prompt_tokens,
        timeout=args.timeout,
        gpu_index=args.gpu_index,
        max_gpu_memory_mib=args.max_gpu_memory_mib,
    )
    result["minimum_prompt_tokens_required"] = args.minimum_prompt_tokens
    prompt_gate = result["min_prompt_tokens"] >= args.minimum_prompt_tokens
    result["prompt_length_gate_passed"] = prompt_gate
    if not prompt_gate:
        result["status"] = "failed"
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if result["status"] == "passed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
