"""Report the fail-closed recovery state of one locked retrain-v2 stage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from jobs.retrain_v2.execution_receipt import STAGE_IDS, stage_recovery_state


class _BlockedArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        print(
            json.dumps(
                {"status": "blocked", "error": message},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        self.exit(2)


def build_parser() -> argparse.ArgumentParser:
    parser = _BlockedArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--run-manifest", type=Path, required=True)
    parser.add_argument("--stage", choices=STAGE_IDS, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = stage_recovery_state(
            args.run_manifest, args.stage, args.repo_root
        )
    except Exception as exc:  # noqa: BLE001 - strict fail-closed CLI boundary
        print(
            json.dumps(
                {"status": "blocked", "error": str(exc)},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        )
        return 2
    print(
        json.dumps(
            {"status": "ready", **result},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
