"""Validation for the preregistered six-indicator LOO experiment scope.

The scoped experiment is deliberately different from an ordinary partial
generation shard.  A scoped release is complete for its preregistered six
interventions while every baseline prompt still contains the frozen 26-block
context.  It must never be represented as a complete 26-intervention release.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from open_r1.provenance import sha256_text


SCOPE_SCHEMA_VERSION = "loo-scoped-experiment-v1"
LEGACY6_EXPERIMENT_ID = "legacy6-full26-v1"
LEGACY6_INDICATORS = (
    "Consumer-Price-Index-(CPI)",
    "Equity-Market-Indices",
    "Federal-Funds-Rate",
    "GDP-Growth",
    "Treasury-Yields",
    "Unemployment-Rate",
)
EXPECTED_SECTIONS = (
    "Participants' Views on Current Conditions and the Economic Outlook",
    "Staff Review of the Economic Situation",
    "Staff Review of the Financial Situation",
)
EXPECTED_POPULATIONS = {
    "pilot_eval_13": {"phase": "pilot", "split_label": "eval"},
    "formal_test_13": {"phase": "formal", "split_label": "test"},
}
EXPECTED_PROMPT_VERSION = "canonical-minutes-section-concise-v1"
EXPECTED_REQUESTED_MAX_OUTPUT_TOKENS = 4096
EXPECTED_HARD_MAX_NEW_TOKENS = 8192
EXPECTED_MAX_MODEL_LEN = 16384
EXPECTED_PRIMARY_SEEDS = [20260728]
EXPECTED_STOCHASTIC_SEEDS = [
    20260729,
    21260729,
    22260729,
    23260729,
    24260729,
]


class ExperimentScopeError(ValueError):
    """Raised when a scoped experiment declaration is unsafe or inconsistent."""


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExperimentScopeError(f"Invalid {label} JSON {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ExperimentScopeError(f"{label} must be a JSON object: {path}")
    return payload


def _require_string(value: object, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ExperimentScopeError(f"{label} must be a non-empty string")
    return text


def _require_exact_mapping_values(
    observed: Mapping[str, Any],
    expected: Mapping[str, Any],
    *,
    label: str,
) -> None:
    mismatches = {
        key: {"expected": value, "observed": observed.get(key)}
        for key, value in expected.items()
        if observed.get(key) != value
    }
    if mismatches:
        raise ExperimentScopeError(f"{label} mismatch: {mismatches}")


def _validate_scope_payload(scope: Mapping[str, Any]) -> None:
    if scope.get("schema_version") != SCOPE_SCHEMA_VERSION:
        raise ExperimentScopeError(
            f"Unsupported scoped-experiment schema: {scope.get('schema_version')!r}"
        )
    if scope.get("experiment_id") != LEGACY6_EXPERIMENT_ID:
        raise ExperimentScopeError(
            "The six-indicator runner only accepts the frozen "
            f"{LEGACY6_EXPERIMENT_ID!r} experiment"
        )

    baseline = _require_string(
        scope.get("baseline_indicator"), label="baseline_indicator"
    )
    full_roster = scope.get("full_context_indicators")
    selected = scope.get("intervention_indicators")
    if not isinstance(full_roster, list) or len(full_roster) != 26:
        raise ExperimentScopeError(
            "full_context_indicators must contain exactly 26 ordered indicators"
        )
    if any(not isinstance(item, str) or not item.strip() for item in full_roster):
        raise ExperimentScopeError("full_context_indicators contains an invalid value")
    if len(full_roster) != len(set(full_roster)):
        raise ExperimentScopeError("full_context_indicators contains duplicates")
    if baseline in full_roster:
        raise ExperimentScopeError(
            "The full baseline label must not be an intervention indicator"
        )
    if selected != list(LEGACY6_INDICATORS):
        raise ExperimentScopeError(
            "intervention_indicators must be the exact preregistered legacy-six "
            f"sequence: {list(LEGACY6_INDICATORS)}"
        )
    if not set(LEGACY6_INDICATORS).issubset(full_roster):
        raise ExperimentScopeError(
            "Every preregistered intervention must occur in the full context roster"
        )

    sections = scope.get("section_names")
    if sections != list(EXPECTED_SECTIONS):
        raise ExperimentScopeError(
            "section_names must be the exact ordered three-section protocol"
        )

    populations = scope.get("populations")
    if not isinstance(populations, Mapping) or set(populations) != set(
        EXPECTED_POPULATIONS
    ):
        raise ExperimentScopeError(
            "populations must contain exactly pilot_eval_13 and formal_test_13"
        )
    for population_id, expected in EXPECTED_POPULATIONS.items():
        record = populations.get(population_id)
        if not isinstance(record, Mapping):
            raise ExperimentScopeError(
                f"populations.{population_id} must be an object"
            )
        _require_exact_mapping_values(
            record, expected, label=f"populations.{population_id}"
        )
        dates = record.get("meeting_dates")
        if (
            not isinstance(dates, list)
            or len(dates) != 13
            or len(set(dates)) != 13
            or any(not isinstance(value, str) or not value for value in dates)
        ):
            raise ExperimentScopeError(
                f"populations.{population_id}.meeting_dates must contain 13 "
                "unique dates"
            )

    prompt = scope.get("minutes_system_prompt")
    if not isinstance(prompt, Mapping):
        raise ExperimentScopeError("minutes_system_prompt must be an object")
    text = _require_string(prompt.get("text"), label="minutes_system_prompt.text")
    expected_prompt = {
        "version": EXPECTED_PROMPT_VERSION,
        "sha256": sha256_text(text),
        "requested_max_output_tokens": EXPECTED_REQUESTED_MAX_OUTPUT_TOKENS,
        "hard_max_new_tokens": EXPECTED_HARD_MAX_NEW_TOKENS,
        "max_model_len": EXPECTED_MAX_MODEL_LEN,
        "input_truncation": "forbidden",
        "token_limit_policy": "error",
    }
    _require_exact_mapping_values(
        prompt, expected_prompt, label="minutes_system_prompt"
    )

    decoding = scope.get("decoding")
    if not isinstance(decoding, Mapping):
        raise ExperimentScopeError("decoding must be an object")
    expected_decoding = {
        "primary": {
            "temperature": 0.0,
            "top_p": 1.0,
            "replicate_seeds": EXPECTED_PRIMARY_SEEDS,
        },
        "stochastic": {
            "temperature": 0.6,
            "top_p": 0.9,
            "replicate_seeds": EXPECTED_STOCHASTIC_SEEDS,
        },
    }
    if decoding != expected_decoding:
        raise ExperimentScopeError(
            f"decoding differs from the frozen protocol: {decoding!r}"
        )

    release_policy = scope.get("release_policy")
    expected_policy = {
        "smoke": "non-inferential-engineering-validation-only",
        "pilot": "exploratory-protocol-validation",
        "formal": "six-indicator-final-inference",
        "complete_matrix_required": True,
        "generation_exclusions_allowed": False,
        "legacy_or_remaining20_pooling": "forbidden",
    }
    if release_policy != expected_policy:
        raise ExperimentScopeError(
            "release_policy differs from the frozen scoped-release protocol"
        )
    _require_string(scope.get("claim_boundary"), label="claim_boundary")


def load_and_validate_experiment_scope(
    scope_file: str | Path,
    *,
    roster_file: str | Path | None = None,
    population_files: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Load and validate a six-indicator scope and optional source bindings."""

    scope_path = Path(scope_file).expanduser().resolve()
    scope = _read_json_object(scope_path, label="scoped experiment")
    _validate_scope_payload(scope)

    if roster_file is not None:
        roster = _read_json_object(
            Path(roster_file).expanduser().resolve(), label="indicator roster"
        )
        expected_roster = {
            "baseline_indicator": scope["baseline_indicator"],
            "roster_id": scope["full_context_roster_id"],
            "indicators": scope["full_context_indicators"],
        }
        _require_exact_mapping_values(
            roster, expected_roster, label="indicator roster binding"
        )

    if population_files is not None:
        if set(population_files) != set(EXPECTED_POPULATIONS):
            raise ExperimentScopeError(
                "population_files must bind pilot_eval_13 and formal_test_13"
            )
        for population_id, raw_path in population_files.items():
            population = _read_json_object(
                Path(raw_path).expanduser().resolve(),
                label=f"{population_id} population",
            )
            expected = {
                "population_id": population_id,
                **dict(EXPECTED_POPULATIONS[population_id]),
                "meeting_dates": scope["populations"][population_id][
                    "meeting_dates"
                ],
            }
            _require_exact_mapping_values(
                population, expected, label=f"{population_id} binding"
            )
    return dict(scope)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate the frozen six-indicator scoped experiment."
    )
    parser.add_argument("--experiment-config", required=True)
    parser.add_argument("--roster")
    parser.add_argument("--pilot-population")
    parser.add_argument("--formal-population")
    parser.add_argument(
        "--print-intervention-indicators", action="store_true"
    )
    args = parser.parse_args()
    population_files = None
    if args.pilot_population or args.formal_population:
        if not args.pilot_population or not args.formal_population:
            parser.error(
                "--pilot-population and --formal-population must be provided together"
            )
        population_files = {
            "pilot_eval_13": args.pilot_population,
            "formal_test_13": args.formal_population,
        }
    scope = load_and_validate_experiment_scope(
        args.experiment_config,
        roster_file=args.roster,
        population_files=population_files,
    )
    if args.print_intervention_indicators:
        for indicator in scope["intervention_indicators"]:
            print(indicator)
    else:
        print(scope["experiment_id"])


if __name__ == "__main__":
    main()
