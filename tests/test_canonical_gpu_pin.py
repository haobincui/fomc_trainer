import os
import re
import stat
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GPU_HELPER = REPO_ROOT / "run/_canonical_gpu1.sh"
DIRECT_LAUNCHER = REPO_ROOT / "run/_run_canonical_loo_generation.sh"
END_TO_END_LAUNCHER = REPO_ROOT / "run/generate_loo_end_to_end.sh"
PILOT_LAUNCHER = REPO_ROOT / "run/generate_loo_pilot.sh"
FORMAL_LAUNCHER = REPO_ROOT / "run/generate_loo_formal.sh"
CANONICAL_SCRIPTS = (
    GPU_HELPER,
    DIRECT_LAUNCHER,
    END_TO_END_LAUNCHER,
    PILOT_LAUNCHER,
    FORMAL_LAUNCHER,
)


class TestCanonicalGpuPin(unittest.TestCase):
    def test_canonical_shell_scripts_have_valid_bash_syntax(self):
        subprocess.run(
            ["bash", "-n", *(str(path) for path in CANONICAL_SCRIPTS)],
            check=True,
            cwd=REPO_ROOT,
        )

    def test_helper_hard_pins_physical_index_one_numerically(self):
        source = GPU_HELPER.read_text(encoding="utf-8")

        self.assertRegex(
            source,
            r"(?m)^LOO_CANONICAL_PHYSICAL_GPU_INDEX=1$",
        )
        self.assertIn(
            '--id="${LOO_CANONICAL_PHYSICAL_GPU_INDEX}"',
            source,
        )
        self.assertIn("--query-gpu=uuid", source)
        self.assertIn(
            'export CUDA_VISIBLE_DEVICES="${LOO_CANONICAL_PHYSICAL_GPU_INDEX}"',
            source,
        )
        self.assertNotIn("${CUDA_VISIBLE_DEVICES:-1}", source)
        self.assertNotRegex(
            source,
            r"LOO_CANONICAL_PHYSICAL_GPU_INDEX=\$\{[^}]+:-",
        )

    def test_helper_overwrites_caller_visibility_with_gpu_one_index(self):
        expected_uuid = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
        with tempfile.TemporaryDirectory() as temporary_directory:
            fake_bin = Path(temporary_directory)
            fake_nvidia_smi = fake_bin / "nvidia-smi"
            fake_nvidia_smi.write_text(
                f"""#!/usr/bin/env bash
set -euo pipefail
[[ "$#" -eq 3 ]]
[[ "$1" == "--id=1" ]]
[[ "$2" == "--query-gpu=uuid" ]]
[[ "$3" == "--format=csv,noheader,nounits" ]]
printf '  {expected_uuid}  \\n'
""",
                encoding="utf-8",
            )
            fake_nvidia_smi.chmod(
                fake_nvidia_smi.stat().st_mode | stat.S_IXUSR
            )

            environment = os.environ.copy()
            environment["PATH"] = (
                f"{fake_bin}{os.pathsep}{environment.get('PATH', '')}"
            )
            environment["CUDA_VISIBLE_DEVICES"] = "0,1"
            environment["CUDA_DEVICE_ORDER"] = "FASTEST_FIRST"
            result = subprocess.run(
                [
                    "bash",
                    "-c",
                    """
set -euo pipefail
source "$1"
printf '%s\\n' \
  "$LOO_CANONICAL_PHYSICAL_GPU_INDEX" \
  "$LOO_CANONICAL_PHYSICAL_GPU_UUID" \
  "$CUDA_VISIBLE_DEVICES" \
  "$CUDA_DEVICE_ORDER" \
  "$LOO_CANONICAL_VISIBLE_DEVICE_COUNT"
""",
                    "bash",
                    str(GPU_HELPER),
                ],
                check=True,
                cwd=REPO_ROOT,
                env=environment,
                capture_output=True,
                text=True,
            )

        self.assertEqual(
            result.stdout.splitlines(),
            [
                "1",
                expected_uuid,
                "1",
                "PCI_BUS_ID",
                "1",
            ],
        )

    def test_gpu_helper_is_sourced_before_any_python_invocation(self):
        source_command = 'source "${SCRIPT_DIR}/_canonical_gpu1.sh"'
        python_invocation = re.compile(
            r'(?m)^[ \t]*"\$\{PYTHON_BIN\}"(?:[ \t]|$)'
        )

        for launcher in (DIRECT_LAUNCHER, END_TO_END_LAUNCHER):
            with self.subTest(launcher=launcher.name):
                source = launcher.read_text(encoding="utf-8")
                self.assertEqual(source.count(source_command), 1)
                source_position = source.index(source_command)
                first_python = python_invocation.search(source)
                self.assertIsNotNone(
                    first_python,
                    f"No PYTHON_BIN invocation found in {launcher}",
                )
                assert first_python is not None
                self.assertLess(source_position, first_python.start())

    def test_all_official_launchers_route_through_a_pinned_entrypoint(self):
        source_command = 'source "${SCRIPT_DIR}/_canonical_gpu1.sh"'
        self.assertIn(
            source_command,
            DIRECT_LAUNCHER.read_text(encoding="utf-8"),
        )
        self.assertIn(
            source_command,
            END_TO_END_LAUNCHER.read_text(encoding="utf-8"),
        )

        for wrapper in (PILOT_LAUNCHER, FORMAL_LAUNCHER):
            with self.subTest(wrapper=wrapper.name):
                source = wrapper.read_text(encoding="utf-8")
                self.assertIn(
                    'exec "${SCRIPT_DIR}/_run_canonical_loo_generation.sh"',
                    source,
                )


if __name__ == "__main__":
    unittest.main()
