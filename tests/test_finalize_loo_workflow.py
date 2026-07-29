import tempfile
import unittest
import json
from pathlib import Path

from jobs.main.finalize_loo_workflow import (
    build_workflow_release,
    write_immutable,
)
from open_r1.provenance import sha256_file
from open_r1.validator.loo_generation_spec import (
    seal_manifest,
    validate_manifest_integrity,
)


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _pilot_artifacts(root: Path) -> list[tuple[str, Path]]:
    registry = root / "source_registry.json"
    _write_json(
        registry,
        {
            "schema_version": "loo-indicator-source-registry-v1",
            "policy": {
                "network_mode": "keyless",
                "access_interface": "alfred-graph-csv-v1",
            },
        },
    )
    snapshot = root / "snapshot_manifest.json"
    _write_json(
        snapshot,
        seal_manifest(
            {
                "schema_version": "loo-source-snapshot-manifest-v1",
                "status": "complete",
                "source_interface": "alfred-graph-csv-v1",
                "registry": {"sha256": sha256_file(registry)},
                "vintage_count": 13,
                "meetings": [
                    {"meeting_date": f"2024-01-{index + 1:02d}"} for index in range(13)
                ],
            }
        ),
    )
    ledger = root / "pilot_ledger_manifest.json"
    _write_json(
        ledger,
        seal_manifest(
            {
                "schema_version": "loo-indicator-ledger-manifest-v1",
                "status": "complete",
                "population_id": "pilot_eval_13",
                "inputs": {
                    "registry": {"sha256": sha256_file(registry)},
                    "snapshot_manifest": {"sha256": sha256_file(snapshot)},
                },
            }
        ),
    )
    release = root / "pilot_release_manifest.json"
    _write_json(
        release,
        seal_manifest(
            {
                "schema_version": "canonical-loo-generation-release-v1",
                "status": "complete",
                "generation_only": True,
                "training_performed": False,
                "phase": "pilot",
                "population_id": "pilot_eval_13",
                "analysis": {
                    "ledger_provenance": {
                        "source_registry_sha256": sha256_file(registry),
                        "snapshot_manifest_sha256": sha256_file(snapshot),
                        "ledger_manifest_sha256": sha256_file(ledger),
                    }
                },
            }
        ),
    )
    return [
        ("source_registry", registry),
        ("snapshot_manifest", snapshot),
        ("pilot_ledger_manifest", ledger),
        ("pilot_release_manifest", release),
    ]


class TestFinalizeLooWorkflow(unittest.TestCase):
    def test_seals_complete_pilot_inventory(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifacts = _pilot_artifacts(root)

            release = build_workflow_release(
                run_id="pilot-test",
                mode="pilot",
                workflow_root=root,
                artifacts=artifacts,
            )

            validate_manifest_integrity(release)
            self.assertEqual(release["status"], "complete")
            self.assertFalse(release["authentication_required"])
            self.assertEqual(
                set(release["artifacts"]),
                {name for name, _ in artifacts},
            )

    def test_rejects_missing_or_external_artifacts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            external = root.parent / "external-ledger.json"
            external.write_text("{}", encoding="utf-8")
            self.addCleanup(external.unlink, missing_ok=True)

            with self.assertRaisesRegex(ValueError, "outside workflow_root"):
                build_workflow_release(
                    run_id="bad",
                    mode="pilot",
                    workflow_root=root,
                    artifacts=[("source_registry", external)],
                )

    def test_rejects_duplicate_and_swapped_artifact_contracts(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            artifacts = _pilot_artifacts(root)
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                build_workflow_release(
                    run_id="duplicate",
                    mode="pilot",
                    workflow_root=root,
                    artifacts=[*artifacts, artifacts[-1]],
                )

            release_path = dict(artifacts)["pilot_release_manifest"]
            invalid = seal_manifest(
                {
                    "schema_version": "canonical-loo-generation-release-v1",
                    "status": "complete",
                    "generation_only": True,
                    "training_performed": False,
                    "phase": "formal",
                    "population_id": "formal_test_13",
                    "analysis": {
                        "ledger_provenance": {
                            "source_registry_sha256": sha256_file(
                                dict(artifacts)["source_registry"]
                            )
                        }
                    },
                }
            )
            _write_json(release_path, invalid)
            with self.assertRaisesRegex(ValueError, "population_id"):
                build_workflow_release(
                    run_id="swapped",
                    mode="pilot",
                    workflow_root=root,
                    artifacts=artifacts,
                )

    def test_immutable_write_is_idempotent_but_not_overwritable(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            payload = {"status": "complete"}
            output = root / "release.json"
            first = write_immutable(output, payload)
            second = write_immutable(output, payload)
            self.assertEqual(first, second)
            with self.assertRaisesRegex(ValueError, "incompatible"):
                write_immutable(output, {"status": "changed"})


if __name__ == "__main__":
    unittest.main()
