"""CLI for constructing a canonical D-1 LOO indicator-input ledger."""

from __future__ import annotations

import argparse
from pathlib import Path

from open_r1.validator.loo_ledger import build_loo_indicator_ledger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build an immutable 11- or 13-meeting × 26-indicator ledger from a "
            "sealed historical-vintage snapshot manifest and raw CSV evidence."
        )
    )
    parser.add_argument(
        "--registry",
        required=True,
        help="loo-indicator-source-registry-v1 JSON",
    )
    parser.add_argument(
        "--snapshot-manifest",
        required=True,
        help="Sealed loo-source-snapshot-manifest-v1 JSON",
    )
    parser.add_argument(
        "--population",
        required=True,
        help="Frozen 11- or 13-meeting population JSON",
    )
    parser.add_argument(
        "--roster",
        required=True,
        help="Frozen 26-indicator roster JSON",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Destination for indicator_inputs.jsonl and audit artifacts",
    )
    return parser


def run(args: argparse.Namespace) -> tuple[Path, dict]:
    output_dir = Path(args.output_dir).expanduser().resolve()
    manifest = build_loo_indicator_ledger(
        registry_file=args.registry,
        snapshot_manifest_file=args.snapshot_manifest,
        population_file=args.population,
        roster_file=args.roster,
        output_dir=output_dir,
    )
    return output_dir / "ledger_manifest.json", manifest


def main() -> None:
    manifest_path, manifest = run(build_parser().parse_args())
    print(f"ledger_manifest={manifest_path}")
    print(f"population_id={manifest['population_id']}")
    print(
        "row_count="
        f"{manifest['outputs']['indicator_inputs']['row_count']}"
    )
    print(
        "payload_sha256="
        f"{manifest['integrity']['payload_sha256']}"
    )


if __name__ == "__main__":
    main()
