#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TRAIN_ENV="${FOMC_RETRAIN_TRAIN_ENV:-fomc_trainer}"
JUDGE_ENV="${FOMC_RETRAIN_JUDGE_ENV:-fomc_judge_v2}"
PYTHON_VERSION="3.10.9"
PIP_VERSION="25.1"
SETUPTOOLS_VERSION="78.1.1"
WHEEL_VERSION="0.45.1"
PYTORCH_INDEX="https://download.pytorch.org/whl/cu128"
TRAIN_LOCK="${ROOT_DIR}/requirements/retrain_v2_train.lock"
JUDGE_LOCK="${ROOT_DIR}/requirements/retrain_v2_judge.lock"
TRAIN_FREEZE="${ROOT_DIR}/requirements/retrain_v2_train.freeze.txt"
JUDGE_FREEZE="${ROOT_DIR}/requirements/retrain_v2_judge.freeze.txt"

usage() {
  printf '%s\n' \
    "Usage: $0 [--train | --judge] [--skip-checks]" \
    "" \
    "With no role flag, both isolated environments are created or updated." \
    "Existing conda environments are never removed." \
    "" \
    "Environment overrides:" \
    "  FOMC_RETRAIN_TRAIN_ENV   Training environment name (default: fomc_trainer)" \
    "  FOMC_RETRAIN_JUDGE_ENV   Judge environment name (default: fomc_judge_v2)" \
    "  FOMC_RETRAIN_TRAIN_GPUS  Policy GPU IDs used by checks (default: 0,1)" \
    "  FOMC_RETRAIN_JUDGE_GPUS  Judge GPU ID used by checks (default: 0)"
}

check_prerequisites() {
  if ! command -v conda >/dev/null 2>&1; then
    echo "ERROR: conda is not available in PATH." >&2
    exit 1
  fi
  if ! command -v sha256sum >/dev/null 2>&1; then
    echo "ERROR: sha256sum is required." >&2
    exit 1
  fi
  if [[ ! -f "${TRAIN_LOCK}" || ! -f "${JUDGE_LOCK}" || ! -f "${TRAIN_FREEZE}" || ! -f "${JUDGE_FREEZE}" ]]; then
    echo "ERROR: retrain-v2 lock or freeze files are missing." >&2
    exit 1
  fi
}

environment_exists() {
  local env_name="$1"
  conda run -n "${env_name}" python -c "import sys; raise SystemExit(0)" \
    >/dev/null 2>&1
}

ensure_environment() {
  local env_name="$1"
  if environment_exists "${env_name}"; then
    echo "Reusing existing conda environment: ${env_name}"
  else
    echo "Creating conda environment: ${env_name}"
    conda create --yes --name "${env_name}" "python=${PYTHON_VERSION}"
  fi

  local actual_python
  actual_python="$(conda run -n "${env_name}" python -c 'import platform; print(platform.python_version())')"
  if [[ "${actual_python}" != "${PYTHON_VERSION}" ]]; then
    echo "ERROR: '${env_name}' uses Python ${actual_python}; expected ${PYTHON_VERSION}." >&2
    echo "Choose a new environment name; this script will not remove an existing environment." >&2
    exit 1
  fi

  conda run --no-capture-output -n "${env_name}" \
    python -m pip install --disable-pip-version-check --upgrade \
    "pip==${PIP_VERSION}" \
    "setuptools==${SETUPTOOLS_VERSION}" \
    "wheel==${WHEEL_VERSION}"
}

prune_train_environment() {
  local -a extra_packages=()
  mapfile -t extra_packages < <(
    conda run --no-capture-output -n "${TRAIN_ENV}" python - "${TRAIN_LOCK}" <<'PY'
import sys
from importlib.metadata import PackageNotFoundError, distribution, distributions
from pathlib import Path

from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name

lock_path = Path(sys.argv[1])
roots = {"pip", "setuptools", "wheel"}
for raw_line in lock_path.read_text(encoding="utf-8").splitlines():
    line = raw_line.strip()
    if not line or line.startswith(("#", "--")):
        continue
    try:
        roots.add(canonicalize_name(Requirement(line).name))
    except InvalidRequirement as exc:
        raise SystemExit(f"Invalid training lock entry: {line}") from exc

required = set(roots)
pending = list(sorted(roots))
while pending:
    name = pending.pop()
    try:
        metadata = distribution(name)
    except PackageNotFoundError as exc:
        raise SystemExit(f"Locked training package is missing: {name}") from exc
    for raw_requirement in metadata.requires or ():
        requirement = Requirement(raw_requirement)
        if requirement.marker is not None and not requirement.marker.evaluate({"extra": ""}):
            continue
        dependency = canonicalize_name(requirement.name)
        if dependency not in required:
            required.add(dependency)
            pending.append(dependency)

installed = {
    canonicalize_name(item.metadata["Name"])
    for item in distributions()
    if item.metadata.get("Name")
}
for name in sorted(installed - required):
    print(name)
PY
  )
  if ((${#extra_packages[@]})); then
    echo "Removing packages outside the retrain-v2 training dependency closure:"
    printf '  %s\n' "${extra_packages[@]}"
    conda run --no-capture-output -n "${TRAIN_ENV}" \
      python -m pip uninstall --yes "${extra_packages[@]}"
  fi
}

install_train_environment() {
  ensure_environment "${TRAIN_ENV}"
  echo "Training lock SHA256: $(sha256sum "${TRAIN_LOCK}" | awk '{print $1}')"
  echo "Training freeze SHA256: $(sha256sum "${TRAIN_FREEZE}" | awk '{print $1}')"
  conda run --no-capture-output -n "${TRAIN_ENV}" \
    python -m pip install --disable-pip-version-check --upgrade \
    --index-url "${PYTORCH_INDEX}" \
    "torch==2.10.0"
  conda run --no-capture-output -n "${TRAIN_ENV}" \
    python -m pip install --disable-pip-version-check --upgrade \
    --extra-index-url "${PYTORCH_INDEX}" \
    -r "${TRAIN_FREEZE}"
  prune_train_environment

  # setup.py declares the legacy monolithic dependency graph.  Register the
  # repository root for jobs/, then link only the open_r1 package itself.  Do
  # not add the whole src/ directory to sys.path: it contains a legacy
  # open_r1.egg-info whose metadata would make pip check enforce the obsolete
  # DeepSpeed/W&B dependency set.
  conda run --no-capture-output -n "${TRAIN_ENV}" python -c \
    'import site, sys; from pathlib import Path; root=Path(sys.argv[1]).resolve(); package=root/"src"/"open_r1"; site_dir=Path(site.getsitepackages()[0]); pth=site_dir/"fomc_retrain_v2_source.pth"; pth.write_text(f"{root}\n", encoding="utf-8"); link=site_dir/"open_r1"; expected=package.resolve(); current=link.resolve() if link.is_symlink() else None; (link.symlink_to(package, target_is_directory=True) if not link.exists() and not link.is_symlink() else None); current=link.resolve() if link.is_symlink() else current; assert current == expected, f"Refusing unexpected open_r1 path: {link}"; print(f"Registered repository source: {link} -> {expected}")' \
    "${ROOT_DIR}"
  echo "Exact training freeze installed from: ${TRAIN_FREEZE}"
}

install_judge_environment() {
  ensure_environment "${JUDGE_ENV}"
  echo "Judge lock SHA256: $(sha256sum "${JUDGE_LOCK}" | awk '{print $1}')"
  echo "Judge freeze SHA256: $(sha256sum "${JUDGE_FREEZE}" | awk '{print $1}')"
  conda run --no-capture-output -n "${JUDGE_ENV}" \
    python -m pip install --disable-pip-version-check --upgrade \
    --index-url "${PYTORCH_INDEX}" \
    "torch==2.10.0" \
    "torchvision==0.25.0" \
    "torchaudio==2.10.0"
  conda run --no-capture-output -n "${JUDGE_ENV}" \
    python -m pip install --disable-pip-version-check --upgrade \
    --extra-index-url "${PYTORCH_INDEX}" \
    -r "${JUDGE_FREEZE}"
  echo "Exact judge freeze installed from: ${JUDGE_FREEZE}"
}

main() {
  local install_train=0
  local install_judge=0
  local skip_checks=0
  local role_selected=0

  while (($#)); do
    case "$1" in
      --train)
        install_train=1
        role_selected=1
        ;;
      --judge)
        install_judge=1
        role_selected=1
        ;;
      --skip-checks)
        skip_checks=1
        ;;
      -h|--help)
        usage
        exit 0
        ;;
      *)
        echo "ERROR: unknown argument '$1'." >&2
        usage >&2
        exit 2
        ;;
    esac
    shift
  done

  if ((role_selected == 0)); then
    install_train=1
    install_judge=1
  fi

  check_prerequisites
  if ((install_train)); then
    install_train_environment
  fi
  if ((install_judge)); then
    install_judge_environment
  fi

  if ((skip_checks == 0)); then
    local check_args=()
    if ((install_train)); then
      check_args+=(--train)
    fi
    if ((install_judge)); then
      check_args+=(--judge)
    fi
    "${ROOT_DIR}/run/check_retrain_v2_envs.sh" "${check_args[@]}"
  fi

  echo "Requested retrain-v2 environments are ready."
}

main "$@"
