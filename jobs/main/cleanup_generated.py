from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

from open_r1.utils.main_pipeline import load_json

DEFAULT_INVENTORY = ROOT / "metadata" / "main" / "legacy_artifact_inventory.json"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Remove generated or deprecated artifacts listed in metadata/main/legacy_artifact_inventory.json.")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--execute", action="store_true", help="Actually remove the listed paths.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    inventory = load_json(args.inventory)
    remove_paths = [ROOT / path for path in inventory["remove_paths"]]

    print("## Legacy Artifact Cleanup")
    for path in remove_paths:
        status = "present" if path.exists() else "missing"
        print(f"- {path}: {status}")
        if args.execute and path.exists():
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
    if args.execute:
        print("✅ Cleanup finished")
    else:
        print("Dry run only. Re-run with --execute to remove the listed paths.")


if __name__ == "__main__":
    main()
