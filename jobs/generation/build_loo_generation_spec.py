"""Build an immutable generation-only LOO experiment specification.

This command fingerprints model, tokenizer, and source artifacts.  It never
loads model weights and never trains, merges, or writes to any artifact path.
The only file it creates is the requested generation specification.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from open_r1.validator.loo_generation_spec import (
    build_generation_spec,
    write_frozen_generation_spec,
)


def parse_named_path(value: str) -> tuple[str, Path]:
    """Parse a CLI ``NAME=PATH`` artifact declaration."""

    if "=" not in value:
        raise argparse.ArgumentTypeError(
            f"Expected NAME=PATH, received {value!r}"
        )
    raw_name, raw_path = value.split("=", 1)
    name = raw_name.strip()
    path_text = raw_path.strip()
    if not name or not path_text:
        raise argparse.ArgumentTypeError(
            f"Both NAME and PATH are required in {value!r}"
        )
    return name, Path(path_text)


def named_paths_to_mapping(
    declarations: Sequence[tuple[str, Path]],
    *,
    option_name: str,
) -> dict[str, Path]:
    """Convert parsed declarations to a mapping, rejecting duplicate names."""

    paths: dict[str, Path] = {}
    for name, path in declarations:
        if name in paths:
            raise ValueError(
                f"Duplicate {option_name} name {name!r}; artifact names must be "
                "unique within each category"
            )
        paths[name] = path
    return paths


def non_negative_seed(value: str) -> int:
    try:
        seed = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"Replicate seed must be an integer, received {value!r}"
        ) from exc
    if seed < 0:
        raise argparse.ArgumentTypeError("Replicate seed must be non-negative")
    return seed


def load_generation_config(path: str | Path) -> dict:
    """Load the required JSON object containing immutable generation settings."""

    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Generation config JSON does not exist: {config_path}"
        )
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Invalid generation config JSON {config_path}: {exc}"
        ) from exc
    if not isinstance(config, dict) or not config:
        raise ValueError(
            f"Generation config must contain a non-empty JSON object: {config_path}"
        )
    return config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fingerprint frozen artifacts and write a generation-only "
            "loo-generation-spec-v1 manifest."
        )
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--phase", required=True)
    parser.add_argument("--population-id", required=True)
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination JSON path for the frozen generation specification.",
    )
    parser.add_argument(
        "--model",
        action="append",
        type=parse_named_path,
        required=True,
        metavar="NAME=PATH",
        help="Frozen model artifact; repeat for additional model roles.",
    )
    parser.add_argument(
        "--tokenizer",
        action="append",
        type=parse_named_path,
        required=True,
        metavar="NAME=PATH",
        help="Frozen tokenizer artifact; repeat for additional tokenizer roles.",
    )
    parser.add_argument(
        "--source",
        action="append",
        type=parse_named_path,
        required=True,
        metavar="NAME=PATH",
        help="Frozen source artifact; repeat for additional source roles.",
    )
    parser.add_argument(
        "--generation-config",
        required=True,
        type=Path,
        help="JSON object containing decoder, context, and other generation settings.",
    )
    parser.add_argument(
        "--replicate-seed",
        action="append",
        type=non_negative_seed,
        required=True,
        metavar="INT",
        help="Frozen replicate seed; repeat for stochastic replicates.",
    )
    return parser


def run(args: argparse.Namespace) -> tuple[Path, str, str]:
    """Build and write the spec, returning path, file hash, and payload hash."""

    models = named_paths_to_mapping(args.model, option_name="--model")
    tokenizers = named_paths_to_mapping(
        args.tokenizer,
        option_name="--tokenizer",
    )
    sources = named_paths_to_mapping(args.source, option_name="--source")
    generation_config = load_generation_config(args.generation_config)

    spec = build_generation_spec(
        run_id=args.run_id,
        phase=args.phase,
        population_id=args.population_id,
        models=models,
        tokenizers=tokenizers,
        sources=sources,
        generation_config=generation_config,
        replicate_seeds=args.replicate_seed,
    )
    output_path = args.output.expanduser().resolve()
    file_sha256 = write_frozen_generation_spec(output_path, spec)
    payload_sha256 = str(spec["integrity"]["payload_sha256"])
    return output_path, file_sha256, payload_sha256


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output_path, file_sha256, payload_sha256 = run(args)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))
    print(f"output={output_path}")
    print(f"file_sha256={file_sha256}")
    print(f"payload_sha256={payload_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
