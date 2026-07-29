"""CLI for replay-validating a canonical D-1 LOO indicator ledger."""

from __future__ import annotations

import argparse
import json

from open_r1.validator.loo_ledger import validate_loo_indicator_ledger


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Rebuild a canonical LOO ledger from raw CSV evidence and verify "
            "all source, payload, output, and manifest hashes."
        )
    )
    parser.add_argument("--ledger-manifest", required=True)
    parser.add_argument("--registry", required=True)
    parser.add_argument("--snapshot-manifest", required=True)
    parser.add_argument("--population", required=True)
    parser.add_argument("--roster", required=True)
    parser.add_argument(
        "--expected-manifest-payload-sha256",
        help="Optional externally frozen ledger-manifest payload digest",
    )
    return parser


def run(args: argparse.Namespace) -> dict:
    return validate_loo_indicator_ledger(
        ledger_manifest_file=args.ledger_manifest,
        registry_file=args.registry,
        snapshot_manifest_file=args.snapshot_manifest,
        population_file=args.population,
        roster_file=args.roster,
        expected_manifest_payload_sha256=(
            args.expected_manifest_payload_sha256
        ),
    )


def main() -> None:
    result = run(build_parser().parse_args())
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
