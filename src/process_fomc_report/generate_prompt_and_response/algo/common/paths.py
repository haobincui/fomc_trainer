from __future__ import annotations

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[5]
PROCESS_ROOT = REPO_ROOT / "src" / "process_fomc_report"
MODULE_ROOT = PROCESS_ROOT / "generate_prompt_and_response"
ALGO_ROOT = MODULE_ROOT / "algo"
RAW_DATA_ROOT = REPO_ROOT / "dataset" / "raw_data"
PROCESSED_ROOT = REPO_ROOT / "dataset" / "processed"
INPUT_ROOT = RAW_DATA_ROOT / "input_data"
OUTPUT_ROOT = PROCESSED_ROOT / "pipeline"
