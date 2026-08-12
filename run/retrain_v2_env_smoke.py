#!/usr/bin/env python3
"""Small, model-free runtime checks for the retrain-v2 conda environments."""

from __future__ import annotations

import argparse
import os
import re
import sys
from importlib.metadata import PackageNotFoundError, distributions, version
from pathlib import Path
from typing import Iterable

from packaging.utils import canonicalize_name
from packaging.version import Version


EXPECTED_PYTHON = (3, 10, 9)
EXPECTED_CUDA = "12.8"
MIN_GPU_MEMORY_BYTES = 23 * 1024**3
REPO_ROOT = Path(__file__).resolve().parents[1]
ROLE_FREEZE_PATHS = {
    "train": REPO_ROOT / "requirements" / "retrain_v2_train.freeze.txt",
    "judge": REPO_ROOT / "requirements" / "retrain_v2_judge.freeze.txt",
}
_EXACT_PIN_PATTERN = re.compile(
    r"^(?P<name>[A-Za-z0-9](?:[A-Za-z0-9._-]*[A-Za-z0-9])?)"
    r"==(?P<version>[^\s;@]+)$"
)

TRAIN_VERSIONS = {
    "accelerate": "1.4.0",
    "bert-score": "0.3.13",
    "bitsandbytes": "0.48.2",
    "datasets": "4.8.4",
    "liger-kernel": "0.8.1",
    "numpy": "2.2.6",
    "peft": "0.15.2",
    "scikit-learn": "1.6.1",
    "torch": "2.10.0",
    "transformers": "4.57.6",
    "trl": "1.2.0",
}

JUDGE_VERSIONS = {
    "bitsandbytes": "0.48.2",
    "numpy": "2.2.6",
    "torch": "2.10.0",
    "torchaudio": "2.10.0",
    "torchvision": "0.25.0",
    "transformers": "5.5.4",
    "vllm": "0.19.1",
}

TRAIN_FORBIDDEN_DISTRIBUTIONS = (
    "deepspeed",
    "e2b",
    "e2b-code-interpreter",
    "flash-attn",
    "lighteval",
    "vllm",
    "xformers",
)


def distribution_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def parse_exact_freeze(path: Path) -> dict[str, str]:
    """Parse a fail-closed, exact ``name==version`` environment freeze."""
    expected: dict[str, str] = {}
    for line_number, raw_line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = _EXACT_PIN_PATTERN.fullmatch(line)
        if match is None:
            raise RuntimeError(f"Invalid freeze entry at line {line_number}")

        name = canonicalize_name(match.group("name"))
        package_version = match.group("version")
        try:
            parsed_version = Version(package_version)
        except ValueError as error:
            raise RuntimeError(
                f"Invalid version for package {name} at line {line_number}"
            ) from error
        if str(parsed_version) != package_version:
            raise RuntimeError(
                f"Non-canonical version for package {name} at line {line_number}"
            )
        if name in expected:
            raise RuntimeError(f"Duplicate package in freeze: {name}")
        expected[name] = package_version

    if not expected:
        raise RuntimeError("Freeze contains no package pins")
    return expected


def collect_distribution_versions(
    installed_distributions: Iterable[object] | None = None,
) -> dict[str, str]:
    """Collect canonical distribution names and exact metadata versions."""
    installed: dict[str, str] = {}
    duplicates: dict[str, set[str]] = {}
    candidates = distributions() if installed_distributions is None else installed_distributions
    for distribution in candidates:
        raw_name = distribution.metadata.get("Name")
        raw_version = distribution.version
        if not raw_name or not raw_version:
            raise RuntimeError("Installed distribution has missing name or version metadata")
        name = canonicalize_name(raw_name)
        if name in installed:
            duplicates.setdefault(name, {installed[name]}).add(raw_version)
            continue
        installed[name] = raw_version
    if duplicates:
        differences = [
            f"{name}: expected one installed version, found {', '.join(sorted(versions))}"
            for name, versions in sorted(duplicates.items())
        ]
        raise RuntimeError("Installed package mismatch:\n  " + "\n  ".join(differences))
    return installed


def check_freeze_contract(role: str) -> None:
    """Require the active environment to exactly equal the role freeze."""
    expected = parse_exact_freeze(ROLE_FREEZE_PATHS[role])
    installed = collect_distribution_versions()
    differences: list[str] = []
    for name in sorted(expected.keys() | installed.keys()):
        expected_version = expected.get(name)
        installed_version = installed.get(name)
        if expected_version == installed_version:
            continue
        differences.append(
            f"{name}: expected {expected_version or 'absent'}, "
            f"found {installed_version or 'missing'}"
        )
    if differences:
        raise RuntimeError(
            "Environment freeze mismatch:\n  " + "\n  ".join(differences)
        )


def check_supplied_freeze_path(role: str, supplied_path: Path) -> None:
    """Reject callers that try to substitute a role's repository freeze."""
    if supplied_path.resolve() != ROLE_FREEZE_PATHS[role].resolve():
        raise RuntimeError(f"Unexpected freeze path for role {role}")


def check_versions(role: str) -> None:
    if sys.version_info[:3] != EXPECTED_PYTHON:
        actual = ".".join(str(item) for item in sys.version_info[:3])
        expected = ".".join(str(item) for item in EXPECTED_PYTHON)
        raise RuntimeError(f"Python mismatch: expected {expected}, found {actual}")

    check_freeze_contract(role)

    expected_versions = TRAIN_VERSIONS if role == "train" else JUDGE_VERSIONS
    mismatches: list[str] = []
    for package, expected in expected_versions.items():
        actual = distribution_version(package)
        if actual is None or Version(actual).base_version != Version(expected).base_version:
            mismatches.append(f"{package}: expected {expected}, found {actual or 'missing'}")
    if mismatches:
        raise RuntimeError("Package version mismatch:\n  " + "\n  ".join(mismatches))

    if role == "train":
        forbidden = [
            package
            for package in TRAIN_FORBIDDEN_DISTRIBUTIONS
            if distribution_version(package) is not None
        ]
        if forbidden:
            raise RuntimeError(
                "Judge/legacy packages must not be installed in the training environment: "
                + ", ".join(forbidden)
            )


def check_cuda_and_bitsandbytes(*, minimum_gpu_count: int) -> None:
    import torch
    from bitsandbytes.nn import Linear4bit

    if torch.version.cuda != EXPECTED_CUDA:
        raise RuntimeError(
            f"PyTorch CUDA mismatch: expected {EXPECTED_CUDA}, found {torch.version.cuda}"
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch")
    if torch.cuda.device_count() < minimum_gpu_count:
        raise RuntimeError(
            f"At least {minimum_gpu_count} visible GPU(s) are required; "
            f"found {torch.cuda.device_count()}"
        )
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("The visible GPUs do not support bfloat16")

    for index in range(minimum_gpu_count):
        properties = torch.cuda.get_device_properties(index)
        if properties.total_memory < MIN_GPU_MEMORY_BYTES:
            gib = properties.total_memory / 1024**3
            raise RuntimeError(
                f"GPU {index} has {gib:.1f} GiB; retrain-v2 requires a 24 GiB-class GPU"
            )
        if properties.major < 8:
            raise RuntimeError(
                f"GPU {index} compute capability {properties.major}.{properties.minor} "
                "does not satisfy the A30-class requirement"
            )

    device = torch.device("cuda:0")
    layer = Linear4bit(
        16,
        8,
        bias=False,
        compute_dtype=torch.bfloat16,
        compress_statistics=True,
        quant_type="nf4",
    ).to(device)
    sample = torch.randn(2, 16, device=device, dtype=torch.bfloat16)
    output = layer(sample)
    if output.shape != (2, 8) or not torch.isfinite(output).all():
        raise RuntimeError("bitsandbytes NF4 CUDA smoke test returned invalid output")
    del output, sample, layer
    torch.cuda.empty_cache()


def check_training_imports() -> None:
    import liger_kernel
    import open_r1
    from transformers import AutoTokenizer
    from jobs.train import train_grpo, train_sft  # noqa: F401
    from open_r1.configs import GRPOConfig, SFTConfig  # noqa: F401

    module_path = Path(open_r1.__file__).resolve()
    if REPO_ROOT not in module_path.parents:
        raise RuntimeError(
            f"open_r1 resolved outside this repository: {module_path}"
        )
    if not Path(liger_kernel.__file__).resolve().is_file():
        raise RuntimeError("liger-kernel did not resolve to an importable package")

    # Transformers 5.5.x rewrites this ByteLevel tokenizer as a Metaspace
    # tokenizer under the unified Llama class, silently deleting every English
    # space on encode/decode.  This is a training-integrity gate, not a cosmetic
    # assertion: fused prompts and completions make the policy and Judge observe
    # different text.
    tokenizer_path = REPO_ROOT / "models" / "DeepSeek-R1-Distill-Llama-8B"
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
        trust_remote_code=False,
        use_fast=True,
    )
    probe = "The unemployment rate was 7.8 percent in September 2012."
    decoded = tokenizer.decode(tokenizer.encode(probe, add_special_tokens=False))
    if decoded != probe:
        raise RuntimeError(
            "DeepSeek tokenizer round-trip changed whitespace: "
            f"expected {probe!r}, found {decoded!r}"
        )


def check_judge_imports() -> None:
    import vllm
    from vllm.model_executor.models import ModelRegistry

    if vllm.__version__ != JUDGE_VERSIONS["vllm"]:
        raise RuntimeError(f"Unexpected vLLM import version: {vllm.__version__}")
    architecture = "Qwen3_5ForConditionalGeneration"
    if architecture not in ModelRegistry.get_supported_archs():
        raise RuntimeError(f"vLLM does not register {architecture}")


def run_nccl_worker() -> None:
    import torch
    import torch.distributed as dist

    if not dist.is_nccl_available():
        raise RuntimeError("This PyTorch build does not provide NCCL")
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", device_id=torch.device(f"cuda:{local_rank}")
    )
    rank = dist.get_rank()
    value = torch.tensor(float(rank + 1), device=local_rank)
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    expected = dist.get_world_size() * (dist.get_world_size() + 1) / 2
    if value.item() != expected:
        raise RuntimeError(
            f"NCCL all-reduce mismatch: expected {expected}, found {value.item()}"
        )
    dist.barrier()
    if rank == 0:
        print("NCCL two-GPU all-reduce: OK")
    dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("train", "judge"), required=True)
    parser.add_argument(
        "freeze_path",
        type=Path,
        help="Repository-pinned freeze file supplied by the environment checker.",
    )
    parser.add_argument("--nccl-worker", action="store_true")
    parser.add_argument(
        "--skip-cuda",
        action="store_true",
        help="Check package/import contracts without allocating a GPU.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    check_supplied_freeze_path(args.role, args.freeze_path)
    check_versions(args.role)
    if args.nccl_worker:
        run_nccl_worker()
        return

    if not args.skip_cuda:
        # The isolated judge is intentionally a one-GPU service.  Policy SFT
        # and non-judge GRPO stages use both A30s, while chk2 uses one policy
        # GPU plus this separate one-GPU judge process.
        minimum_gpu_count = 2 if args.role == "train" else 1
        check_cuda_and_bitsandbytes(minimum_gpu_count=minimum_gpu_count)
    if args.role == "train":
        check_training_imports()
    else:
        check_judge_imports()
    print(f"retrain-v2 {args.role} environment smoke: OK")


if __name__ == "__main__":
    main()
