from __future__ import annotations

import argparse


def sync_processed_datasets(config_path: str | None = None, *, profile: str = "compat") -> dict:
    return {
        "status": "deprecated",
        "message": "Datasets are written directly to dataset/processed/train; sync is no longer required.",
        "profile": profile,
        "config_path": config_path,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Deprecated no-op for direct-to-train dataset builds.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--profile", choices=["strict", "compat", "both"], default="compat")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = sync_processed_datasets(args.config, profile=args.profile)
    print(summary["message"])


if __name__ == "__main__":
    main()
