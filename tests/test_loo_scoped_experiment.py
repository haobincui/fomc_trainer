import fcntl
import json
import os
import stat
import subprocess
import tempfile
import unittest
from unittest.mock import patch
from pathlib import Path

from jobs.generation.finalize_loo_scoped_release import (
    build_scoped_release_manifest,
    load_and_validate_smoke_gate,
)
from jobs.generation.finalize_loo_scoped_workflow import (
    load_and_validate_scoped_workflow,
)
from open_r1.provenance import fingerprint_artifact_path, sha256_file
from open_r1.validator.loo_experiment_scope import (
    ExperimentScopeError,
    load_and_validate_experiment_scope,
)
from open_r1.validator.loo_generation_spec import seal_manifest


REPO_ROOT = Path(__file__).resolve().parents[1]
SCOPE_FILE = REPO_ROOT / "configs/main/loo_experiment_legacy6.json"
ROSTER_FILE = REPO_ROOT / "configs/main/leave_one_out_roster.json"
PILOT_FILE = REPO_ROOT / "configs/main/loo_population_pilot_eval_13.json"
FORMAL_FILE = REPO_ROOT / "configs/main/loo_population_formal_test_13.json"


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")


class TestLooScopedExperiment(unittest.TestCase):
    @staticmethod
    def _write_fake_nvidia(fake_bin: Path, *, process_query: str = "") -> None:
        fake_nvidia = fake_bin / "nvidia-smi"
        fake_nvidia.write_text(
            f"""#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "--id=1" ]]; then
  printf '%s\n' 'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
  exit 0
fi
printf '%b' '{process_query}'
""",
            encoding="utf-8",
        )
        fake_nvidia.chmod(fake_nvidia.stat().st_mode | stat.S_IXUSR)

    def test_frozen_scope_binds_full_26_and_exact_legacy_six(self):
        scope = load_and_validate_experiment_scope(
            SCOPE_FILE,
            roster_file=ROSTER_FILE,
            population_files={
                "pilot_eval_13": PILOT_FILE,
                "formal_test_13": FORMAL_FILE,
            },
        )
        self.assertEqual(len(scope["full_context_indicators"]), 26)
        self.assertEqual(len(scope["intervention_indicators"]), 6)
        self.assertNotIn("yield_curve", scope["intervention_indicators"])
        self.assertEqual(
            scope["minutes_system_prompt"]["requested_max_output_tokens"],
            4096,
        )

    def test_scope_rejects_a_seventh_intervention(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            scope = json.loads(SCOPE_FILE.read_text(encoding="utf-8"))
            scope["intervention_indicators"].append("Housing-Starts")
            path = Path(temporary_directory) / "scope.json"
            _write_json(path, scope)
            with self.assertRaisesRegex(
                ExperimentScopeError, "exact preregistered legacy-six"
            ):
                load_and_validate_experiment_scope(path)

    def test_shell_entrypoints_are_syntactically_valid_and_scope_aware(self):
        scripts = [
            REPO_ROOT / "run/_run_canonical_loo_generation.sh",
            REPO_ROOT / "run/generate_loo_end_to_end.sh",
            REPO_ROOT / "run/run_loo_legacy6_two_stage_gpu1.sh",
        ]
        subprocess.run(
            ["bash", "-n", *(str(path) for path in scripts)],
            check=True,
            cwd=REPO_ROOT,
        )
        top = scripts[-1].read_text(encoding="utf-8")
        self.assertIn('LOO_MINUTES_TOKEN_LIMIT_POLICY=error', top)
        self.assertIn('generate_loo_end_to_end.sh" all', top)
        self.assertIn("gpu1_is_idle", top)
        self.assertIn('LOO_RUN_ID="${RUN_ID}"', top)
        self.assertIn('LOO_WORKFLOW_BASE="${WORKFLOW_BASE}"', top)
        self.assertIn('LOO_SUPERVISOR_PID="$$"', top)
        end_to_end = scripts[1].read_text(encoding="utf-8")
        self.assertIn("generate_smoke", end_to_end)
        self.assertIn("LOO_REUSE_PROJECTION_MANIFEST", end_to_end)
        self.assertIn("score_scoped_population", end_to_end)
        self.assertIn('"${PPID}" != "${EXISTING_PID}"', end_to_end)

    def _run_launcher_with_fake_gpu_query(self, *, query_exit: int, busy: bool):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            fake_nvidia = fake_bin / "nvidia-smi"
            process_output = (
                "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee\\n" if busy else ""
            )
            fake_nvidia.write_text(
                f"""#!/usr/bin/env bash
set -euo pipefail
if [[ "$1" == "--id=1" ]]; then
  printf '%s\\n' 'GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee'
  exit 0
fi
printf '%b' '{process_output}'
exit {query_exit}
""",
                encoding="utf-8",
            )
            fake_nvidia.chmod(fake_nvidia.stat().st_mode | stat.S_IXUSR)
            environment = os.environ.copy()
            environment["PATH"] = f"{fake_bin}:{environment['PATH']}"
            environment["LOO_WORKFLOW_BASE"] = str(root / "workflows")
            environment["LOO_RUN_ID"] = "gpu-preflight-fixture"
            result = subprocess.run(
                [str(REPO_ROOT / "run/run_loo_legacy6_two_stage_gpu1.sh")],
                cwd=REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertFalse((root / "workflows/gpu-preflight-fixture").exists())
            return result

    def test_top_launcher_rejects_busy_gpu_without_starting(self):
        result = self._run_launcher_with_fake_gpu_query(query_exit=0, busy=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("already has a compute process", result.stderr)

    def test_top_launcher_fails_closed_when_gpu_query_fails(self):
        result = self._run_launcher_with_fake_gpu_query(query_exit=9, busy=False)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("refusing to launch", result.stderr)

    def test_launcher_worker_records_global_lock_failure(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            self._write_fake_nvidia(fake_bin)
            runtime = root / "runtime"
            runtime.mkdir()
            lock_path = runtime / f"fomc-trainer-{os.getuid()}-physical-gpu1.lock"
            with lock_path.open("w", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
                environment = os.environ.copy()
                environment.update(
                    {
                        "PATH": f"{fake_bin}:{environment['PATH']}",
                        "XDG_RUNTIME_DIR": str(runtime),
                        "LOO_WORKFLOW_BASE": str(root / "workflows"),
                        "LOO_RUN_ID": "locked-worker-fixture",
                    }
                )
                result = subprocess.run(
                    [
                        str(REPO_ROOT / "run/run_loo_legacy6_two_stage_gpu1.sh"),
                        "--worker",
                    ],
                    cwd=REPO_ROOT,
                    env=environment,
                    capture_output=True,
                    text=True,
                )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("holds the physical-GPU1 lock", result.stderr)
            status_path = root / (
                "workflows/locked-worker-fixture/workflow_status.jsonl"
            )
            statuses = [
                json.loads(line)
                for line in status_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                [(row["stage"], row["status"]) for row in statuses],
                [("launcher", "running"), ("launcher", "failed")],
            )

    def test_end_to_end_rejects_untrusted_live_supervisor_pid(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            self._write_fake_nvidia(fake_bin)
            fake_python = fake_bin / "python"
            fake_python.write_text(
                "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8"
            )
            fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
            model_dir = root / "model"
            model_dir.mkdir()
            workflow_root = root / "workflows/live-supervisor-fixture"
            workflow_root.mkdir(parents=True)
            (workflow_root / "run.pid").write_text(
                f"{os.getpid()}\n", encoding="utf-8"
            )
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "LOO_PYTHON": str(fake_python),
                    "LOO_SKIP_NETWORK_PREFLIGHT": "1",
                    "LOO_RESUME": "1",
                    "LOO_WORKFLOW_BASE": str(root / "workflows"),
                    "LOO_RUN_ID": "live-supervisor-fixture",
                    "LOO_ANALYSIS_MODEL": str(model_dir),
                    "LOO_ANALYSIS_TOKENIZER": str(model_dir),
                    "LOO_MINUTES_MODEL": str(model_dir),
                    "LOO_MINUTES_TOKENIZER": str(model_dir),
                }
            )
            result = subprocess.run(
                [str(REPO_ROOT / "run/generate_loo_end_to_end.sh"), "all"],
                cwd=REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already has a live worker", result.stderr)

    def test_failed_pilot_release_gate_never_enters_formal(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fake_bin = root / "bin"
            fake_bin.mkdir()
            self._write_fake_nvidia(fake_bin)
            command_log = root / "python-commands.log"
            fake_python = fake_bin / "python"
            fake_python.write_text(
                f"""#!/usr/bin/env bash
printf '%s\\n' "$*" >>'{command_log}'
if [[ "$*" == *"--validate-release"*"pilot_eval_13"* ]]; then
  exit 42
fi
exit 0
""",
                encoding="utf-8",
            )
            fake_python.chmod(fake_python.stat().st_mode | stat.S_IXUSR)
            model_dir = root / "model"
            model_dir.mkdir()
            workflow_root = root / "workflows/gate-failure-fixture"
            for relative in (
                "source_snapshots/snapshot_manifest.json",
                "ledgers/pilot_eval_13/ledger_manifest.json",
                "generations/smoke_pilot_longest/smoke_gate_manifest.json",
                "generations/pilot_eval_13/scoped_generation_release_manifest.json",
            ):
                _write_json(workflow_root / relative, {"status": "fixture"})
            environment = os.environ.copy()
            environment.update(
                {
                    "PATH": f"{fake_bin}:{environment['PATH']}",
                    "LOO_PYTHON": str(fake_python),
                    "LOO_RESUME": "1",
                    "LOO_WORKFLOW_BASE": str(root / "workflows"),
                    "LOO_RUN_ID": "gate-failure-fixture",
                    "LOO_EXPERIMENT_CONFIG": str(SCOPE_FILE),
                    "LOO_ANALYSIS_MODEL": str(model_dir),
                    "LOO_ANALYSIS_TOKENIZER": str(model_dir),
                    "LOO_MINUTES_MODEL": str(model_dir),
                    "LOO_MINUTES_TOKENIZER": str(model_dir),
                }
            )
            result = subprocess.run(
                [
                    str(REPO_ROOT / "run/generate_loo_end_to_end.sh"),
                    "all",
                    "--worker",
                    "gate-failure-fixture",
                    str(workflow_root),
                ],
                cwd=REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 42, result.stderr)
            self.assertFalse(
                (workflow_root / "ledgers/formal_test_13").exists(),
                "Formal must not start after a failed Pilot release gate",
            )

    def test_completed_workflow_resume_rejects_frozen_artifact_tamper(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory).resolve()
            artifact_names = {
                "experiment_config",
                "source_registry",
                "snapshot_manifest",
                "pilot_release",
                "smoke_gate",
                "pilot_ledger",
                "pilot_results_manifest",
                "reused_pilot_analysis_manifest",
                "reused_pilot_analysis_output",
                "reused_pilot_projection_manifest",
                "reused_pilot_projection_output",
                "formal_ledger",
                "formal_release",
                "formal_results_manifest",
            }
            artifact_paths = {}
            for name in artifact_names:
                path = root / "artifacts" / f"{name}.json"
                _write_json(path, {"name": name})
                artifact_paths[name] = path
            protocol = {
                "schema_version": "loo-scoring-protocol-v1",
                "embedding": {"artifact_sha256": "e" * 64},
            }
            release_base = {
                "minutes_model_sha256": "1" * 64,
                "minutes_tokenizer_sha256": "2" * 64,
                "analysis_model_sha256": "3" * 64,
                "analysis_tokenizer_sha256": "4" * 64,
            }
            workflow = seal_manifest(
                {
                    "schema_version": "loo-scoped-workflow-release-v1",
                    "status": "complete",
                    "run_id": "workflow-resume-fixture",
                    "mode": "all",
                    "experiment_id": "legacy6-full26-v1",
                    "experiment_config_sha256": sha256_file(
                        artifact_paths["experiment_config"]
                    ),
                    **release_base,
                    "embedding_model_sha256": "e" * 64,
                    "scoring_protocol_signature": protocol,
                    "artifacts": {
                        name: fingerprint_artifact_path(path)
                        for name, path in artifact_paths.items()
                    },
                }
            )
            workflow_path = root / "scoped_workflow_release_manifest.json"
            _write_json(workflow_path, workflow)
            pilot_release = {
                **release_base,
                "release_kind": "pilot",
                "population_id": "pilot_eval_13",
            }
            formal_release = {
                **release_base,
                "release_kind": "formal",
                "population_id": "formal_test_13",
            }
            patches = (
                patch(
                    "jobs.generation.finalize_loo_scoped_workflow."
                    "load_and_validate_experiment_scope",
                    return_value={"experiment_id": "legacy6-full26-v1"},
                ),
                patch(
                    "jobs.generation.finalize_loo_scoped_workflow._validate_ledger",
                    return_value={},
                ),
                patch(
                    "jobs.generation.finalize_loo_scoped_workflow."
                    "load_and_validate_smoke_gate",
                    return_value={"status": "passed"},
                ),
                patch(
                    "jobs.generation.finalize_loo_scoped_workflow."
                    "load_and_validate_scoped_release",
                    side_effect=[pilot_release, formal_release],
                ),
                patch(
                    "jobs.generation.finalize_loo_scoped_workflow."
                    "_validate_results_manifest",
                    side_effect=[protocol, protocol],
                ),
            )
            with patches[0], patches[1], patches[2], patches[3], patches[4]:
                validated = load_and_validate_scoped_workflow(workflow_path)
            self.assertEqual(validated["status"], "complete")

            artifact_paths["reused_pilot_analysis_output"].write_text(
                '{"tampered":true}\n', encoding="utf-8"
            )
            with self.assertRaisesRegex(ValueError, "Workflow artifact changed"):
                load_and_validate_scoped_workflow(workflow_path)

    def _build_smoke_fixture(self, root: Path) -> tuple[dict, dict]:
        scope_path = root / "scope.json"
        roster_path = root / "roster.json"
        scope_path.write_bytes(SCOPE_FILE.read_bytes())
        roster_path.write_bytes(ROSTER_FILE.read_bytes())
        scope = json.loads(scope_path.read_text(encoding="utf-8"))
        selected = scope["intervention_indicators"]
        baseline = scope["baseline_indicator"]
        model_sha = "1" * 64
        tokenizer_sha = "2" * 64
        spec = root / "generation_spec.json"
        prompt = root / "prompt_manifest.json"
        analysis = root / "analysis.json"
        projection = root / "projection.json"
        _write_json(
            spec,
            {
                "run_id": "fixture-smoke",
                "phase": "pilot",
                "population_id": "pilot_eval_13",
                "generation_config": {
                    "minutes_system_prompt": scope["minutes_system_prompt"]
                },
                "frozen_artifacts": {
                    "sources": {
                        "experiment_scope": {
                            "sha256": sha256_file(scope_path)
                        }
                    },
                    "models": {"minutes_model": {"sha256": model_sha}},
                    "tokenizers": {
                        "minutes_tokenizer": {"sha256": tokenizer_sha}
                    },
                },
            },
        )
        _write_json(
            prompt,
            {
                "experiment_config": {"sha256": sha256_file(scope_path)},
                "intervention_manifest": {"sha256": "a" * 64},
                "neutral_intervention_manifest": {"sha256": "b" * 64},
                "prompt_template": {"sha256": "c" * 64},
                "context_budget": {
                    "system_prompt_sha256": scope["minutes_system_prompt"][
                        "sha256"
                    ],
                    "requested_max_output_tokens": 4096,
                    "max_new_tokens": 8192,
                    "max_model_len": 16384,
                    "input_truncation": "forbidden",
                },
            },
        )
        analysis_output = root / "indicator_analysis.jsonl"
        analysis_output.write_text("{}\n", encoding="utf-8")
        _write_json(
            analysis,
            {
                "schema_version": "indicator-analysis-generation-v2",
                "status": "complete",
                "generation_only": True,
                "training_performed": False,
                "inventory": {
                    "meeting_count": 13,
                    "indicator_count": 26,
                    "row_count": 338,
                },
                "inputs": {
                    "model": {"sha256": "3" * 64},
                    "tokenizer": {"sha256": "4" * 64},
                },
                "output": {"sha256": sha256_file(analysis_output)},
            },
        )
        _write_json(
            projection,
            {
                "schema_version": "indicator-analysis-projection-v1",
                "status": "complete",
                "generation_performed": False,
                "summarization_performed": False,
                "truncation_performed": False,
                "input": {"sha256": sha256_file(analysis_output)},
                "input_analysis_manifest": {
                    "sha256": sha256_file(analysis),
                    "schema_version": "indicator-analysis-generation-v2",
                },
                "output": {"sha256": "5" * 64},
            },
        )
        samples = [
            f"2022-01-26::{section}" for section in scope["section_names"]
        ]
        universe = [baseline, *selected]
        run_settings = {
            "deletion_primary": (
                "indicator_block_deletion",
                "primary",
                1,
            ),
            "neutral_primary": (
                "indicator_block_neutral_replacement",
                "primary",
                1,
            ),
            "deletion_stochastic": (
                "indicator_block_deletion",
                "stochastic",
                5,
            ),
            "neutral_stochastic": (
                "indicator_block_neutral_replacement",
                "stochastic",
                5,
            ),
        }
        manifests = {}
        runtime_hashes = {}
        for run_name, (strategy, regime, replicates) in run_settings.items():
            output_dir = root / "generations" / run_name
            runtime_intervention = output_dir / "intervention_manifest.json"
            _write_json(runtime_intervention, {"strategy": strategy})
            runtime_sha = sha256_file(runtime_intervention)
            runtime_hashes[strategy] = runtime_sha
            artifacts = []
            for indicator in universe:
                for replicate_id in range(replicates):
                    output = output_dir / f"{indicator}_{replicate_id}.jsonl"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    rows = []
                    for sample in samples:
                        section = sample.split("::", 1)[1]
                        generated = (
                            f"baseline {sample} {replicate_id}"
                            if indicator == baseline
                            else f"{run_name} {indicator} {sample} {replicate_id}"
                        )
                        rows.append(
                            {
                                "sample_id": sample,
                                "meeting_date": "2022-01-26",
                                "section_name": section,
                                "generated": generated,
                                "generated_sha256": __import__("hashlib")
                                .sha256(generated.encode())
                                .hexdigest(),
                                "prompt_sha256": (
                                    f"baseline-{sample}"
                                    if indicator == baseline
                                    else f"{run_name}-{indicator}-{sample}"
                                ),
                                "generation_seed": 100 + replicate_id,
                                "generation_seed_policy": "sample-id-sha256-v1",
                                "generation_finish_reason": "stop",
                                "output_token_count": 30,
                                "input_was_truncated": False,
                                "generation_validation_status": "passed",
                                "generation_system_prompt_sha256": scope[
                                    "minutes_system_prompt"
                                ]["sha256"],
                            }
                        )
                    output.write_text(
                        "".join(json.dumps(row) + "\n" for row in rows),
                        encoding="utf-8",
                    )
                    artifacts.append(
                        {
                            "path": str(output),
                            "indicator": indicator,
                            "replicate_id": str(replicate_id),
                            "output_sha256": sha256_file(output),
                            "attempted_row_count": 3,
                            "scorable_row_count": 3,
                            "excluded_row_count": 0,
                        }
                    )
            decoding = scope["decoding"][regime]
            manifest = {
                "schema_version": "loo-generation-v4",
                "status": "complete",
                "generation_attempts_complete": True,
                "scorable_population_complete": True,
                "masking_strategy": strategy,
                "simulation_step": replicates,
                "temperature": decoding["temperature"],
                "top_p": decoding["top_p"],
                "replicate_seeds": decoding["replicate_seeds"],
                "max_new_tokens": 8192,
                "max_model_len": 16384,
                "seed_policy": "sample-id-sha256-v1",
                "system_prompt_sha256": scope["minutes_system_prompt"]["sha256"],
                "minutes_system_prompt": scope["minutes_system_prompt"],
                "experiment_config": {"sha256": sha256_file(scope_path)},
                "completion_validation": {
                    "token_limit_policy": "error",
                    "excluded_row_count": 0,
                },
                "generation_spec": {"file_sha256": sha256_file(spec)},
                "prompt_manifest": {"sha256": sha256_file(prompt)},
                "intervention_manifest": {
                    "sha256": runtime_sha
                },
                "model_artifact": {"sha256": model_sha},
                "model_artifact_after_generation": {"sha256": model_sha},
                "tokenizer_artifact": {"sha256": tokenizer_sha},
                "tokenizer_artifact_after_generation": {
                    "sha256": tokenizer_sha
                },
                "sample_selection": {
                    "policy": "explicit-sample-id-filter-v1",
                    "mode": "filtered_smoke",
                    "requested_sample_ids": samples,
                    "selected_sample_ids": samples,
                    "selected_row_count_per_artifact": 3,
                    "full_source_row_count_per_artifact": 39,
                    "population_release_allowed": False,
                },
                "intervention_scope": {
                    "baseline_indicator": baseline,
                    "full_roster_indicators": scope["full_context_indicators"],
                    "generated_intervention_indicators": selected,
                    "baseline_contains_full_roster": True,
                },
                "expected_artifacts": artifacts,
            }
            manifest_path = output_dir / "generation_manifest.json"
            _write_json(manifest_path, manifest)
            manifests[run_name] = manifest_path
        fixture = {
            "run_root": root,
            "generation_spec_file": spec,
            "prompt_manifest_file": prompt,
            "analysis_manifest_file": analysis,
            "projection_manifest_file": projection,
            "experiment_config_file": scope_path,
            "roster_file": roster_path,
            "release_kind": "smoke",
            "smoke_sample_ids": samples,
        }
        intervention_validation = {
            "deletion_manifest_sha256": "a" * 64,
            "deletion_proof_count": 1014,
            "neutral_manifest_sha256": "b" * 64,
            "neutral_proof_count": 1014,
            "deletion_runtime_manifest_sha256": runtime_hashes[
                "indicator_block_deletion"
            ],
            "neutral_runtime_manifest_sha256": runtime_hashes[
                "indicator_block_neutral_replacement"
            ],
            "prompt_template_sha256": "c" * 64,
            "exact_deletion_validated": True,
            "neutral_replacement_validated": True,
        }
        return fixture, manifests, intervention_validation

    def test_smoke_gate_validates_252_attempt_matrix(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture, _, intervention_validation = self._build_smoke_fixture(root)
            with patch(
                "jobs.generation.finalize_loo_scoped_release."
                "_validate_prompt_interventions",
                return_value=intervention_validation,
            ):
                gate = build_scoped_release_manifest(**fixture)
            self.assertEqual(gate["status"], "passed")
            self.assertEqual(gate["matrix"]["total_generation_attempts"], 252)
            path = root / "smoke.json"
            _write_json(path, gate)
            with patch(
                "jobs.generation.finalize_loo_scoped_release."
                "_validate_prompt_interventions",
                return_value=intervention_validation,
            ):
                self.assertEqual(
                    load_and_validate_smoke_gate(path)["status"], "passed"
                )

    def test_smoke_gate_rejects_output_above_requested_limit(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture, manifests, intervention_validation = self._build_smoke_fixture(root)
            manifest_path = manifests["deletion_primary"]
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            artifact = manifest["expected_artifacts"][0]
            output = Path(artifact["path"])
            rows = [json.loads(line) for line in output.read_text().splitlines()]
            rows[0]["output_token_count"] = 4097
            output.write_text(
                "".join(json.dumps(row) + "\n" for row in rows),
                encoding="utf-8",
            )
            artifact["output_sha256"] = sha256_file(output)
            _write_json(manifest_path, manifest)
            with patch(
                "jobs.generation.finalize_loo_scoped_release."
                "_validate_prompt_interventions",
                return_value=intervention_validation,
            ), self.assertRaisesRegex(ValueError, "4096-token limit"):
                build_scoped_release_manifest(**fixture)

    def test_smoke_resume_revalidates_and_rejects_raw_output_tamper(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            fixture, manifests, intervention_validation = self._build_smoke_fixture(root)
            with patch(
                "jobs.generation.finalize_loo_scoped_release."
                "_validate_prompt_interventions",
                return_value=intervention_validation,
            ):
                gate = build_scoped_release_manifest(**fixture)
            gate_path = root / "smoke.json"
            _write_json(gate_path, gate)
            deletion_manifest = json.loads(
                manifests["deletion_primary"].read_text(encoding="utf-8")
            )
            output_path = Path(deletion_manifest["expected_artifacts"][0]["path"])
            output_path.write_text(
                output_path.read_text(encoding="utf-8") + "{}\n",
                encoding="utf-8",
            )
            with patch(
                "jobs.generation.finalize_loo_scoped_release."
                "_validate_prompt_interventions",
                return_value=intervention_validation,
            ), self.assertRaisesRegex(ValueError, "output artifact hash mismatch"):
                load_and_validate_smoke_gate(gate_path)


if __name__ == "__main__":
    unittest.main()
