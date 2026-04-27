from __future__ import annotations

import argparse
from collections import Counter

from process_fomc_report.generate_prompt_and_response.algo.common.config import (
    get_response_template,
    load_pipeline_config,
    resolve_path,
)
from process_fomc_report.generate_prompt_and_response.algo.common.io_utils import load_jsonl, write_json, write_jsonl
from process_fomc_report.generate_prompt_and_response.algo.common.llm_client import get_response, load_llm_config
from process_fomc_report.generate_prompt_and_response.algo.common.response_templates import format_response_text


def log(message: str) -> None:
    print(f"[analysis_teacher_responses] {message}", flush=True)


def _existing_by_sample_id(output_path):
    if not output_path.exists():
        return {}
    return {row["sample_id"]: row for row in load_jsonl(output_path)}


def _persist_split_state(output_root, split: str, output_rows_by_id: dict[str, dict], failure_rows: list[dict]) -> None:
    ordered = sorted(output_rows_by_id.values(), key=lambda item: (item["split"], item["sample_id"]))
    write_jsonl(output_root / f"{split}.jsonl", ordered)
    failed_path = output_root / f"{split}_failed.jsonl"
    if failure_rows:
        write_jsonl(failed_path, failure_rows)
    else:
        failed_path.unlink(missing_ok=True)


def generate_teacher_responses(
    config_path: str | None = None,
    *,
    scope: str = "after_2009",
    teacher_source: str | None = None,
    force_refresh: bool = False,
) -> dict:
    config = load_pipeline_config(config_path)
    analysis_cfg = config["analysis"]
    teacher_cfg = dict(config["teacher"])
    response_template = get_response_template(config)
    if teacher_source is not None:
        teacher_cfg["source"] = teacher_source
    if scope != "after_2009" and teacher_cfg["source"] == "archived_master":
        raise RuntimeError("archived_master teacher responses are only available for scope=after_2009.")
    prompt_root = resolve_path(analysis_cfg["teacher_prompt_root"]) / scope
    output_root = resolve_path(analysis_cfg["teacher_response_root"]) / scope
    master_rows = {
        row["sample_id"]: row
        for row in load_jsonl(resolve_path(analysis_cfg["master_path"]))
    }
    failures: dict[str, list[dict]] = {"train": [], "eval": [], "test": []}
    summary: dict[str, dict] = {}

    log(
        f"scope={scope} source={teacher_cfg['source']} force_refresh={force_refresh} "
        f"resume={teacher_cfg.get('resume', True)}"
    )
    llm_config = load_llm_config(teacher_cfg) if teacher_cfg["source"] == "llm" else None

    for split in ("train", "eval", "test"):
        prompt_rows = load_jsonl(prompt_root / f"{split}.jsonl")
        existing_candidates = (
            _existing_by_sample_id(output_root / f"{split}.jsonl")
            if teacher_cfg.get("resume", True) and not force_refresh
            else {}
        )
        validated_existing = {
            row["sample_id"]: existing_row
            for row in prompt_rows
            if (
                (existing_row := existing_candidates.get(row["sample_id"])) is not None
                and existing_row.get("prompt_hash") == row["prompt_hash"]
                and existing_row.get("response_template") == response_template
            )
        }
        output_rows_by_id = dict(validated_existing)
        _persist_split_state(output_root, split, output_rows_by_id, failures[split])
        status_counter: Counter[str] = Counter(row.get("status", "existing") for row in output_rows_by_id.values())
        cached_rows = len(validated_existing)
        materialized_rows = 0
        log(
            f"split={split} prompts={len(prompt_rows)} cached_candidates={len(existing_candidates)} "
            f"validated_cache_hits={cached_rows} "
            f"output={output_root / f'{split}.jsonl'}"
        )
        for index, row in enumerate(prompt_rows, start=1):
            sample_id = row["sample_id"]
            existing_row = validated_existing.get(sample_id)
            if existing_row is not None:
                log(
                    f"split={split} progress={index}/{len(prompt_rows)} "
                    f"sample_id={sample_id} status=cached"
                )
                continue
            if teacher_cfg["source"] == "archived_master":
                log(
                    f"split={split} progress={index}/{len(prompt_rows)} "
                    f"sample_id={sample_id} action=archived_master"
                )
                archived = master_rows[row["sample_id"]]
                result = {
                    "sample_id": row["sample_id"],
                    "split": split,
                    "prompt_hash": row["prompt_hash"],
                    "response": format_response_text(
                        archived.get("response", ""),
                        response_template=response_template,
                    ),
                    "reasoning": "",
                    "teacher_model": "archived_master_seed",
                    "status": "archived_master",
                    "response_template": response_template,
                }
            else:
                try:
                    log(
                        f"split={split} progress={index}/{len(prompt_rows)} "
                        f"sample_id={sample_id} action=llm_request"
                    )
                    response, reasoning = get_response(row["prompt"], llm_config)
                    result = {
                        "sample_id": row["sample_id"],
                        "split": split,
                        "prompt_hash": row["prompt_hash"],
                        "response": format_response_text(
                            response,
                            response_template=response_template,
                            reasoning_text=reasoning,
                        ),
                        "reasoning": reasoning,
                        "teacher_model": llm_config.model_name,
                        "status": "success" if response.strip() else "empty_response",
                        "response_template": response_template,
                    }
                except Exception as exc:  # noqa: BLE001
                    failures[split].append(
                        {
                            "sample_id": row["sample_id"],
                            "split": split,
                            "prompt_hash": row["prompt_hash"],
                            "error": str(exc),
                        }
                    )
                    status_counter["failed"] += 1
                    _persist_split_state(output_root, split, output_rows_by_id, failures[split])
                    log(
                        f"split={split} progress={index}/{len(prompt_rows)} "
                        f"sample_id={sample_id} status=failed failures={len(failures[split])} error={exc}"
                    )
                    continue
            output_rows_by_id[sample_id] = result
            status_counter[result["status"]] += 1
            materialized_rows += 1
            _persist_split_state(output_root, split, output_rows_by_id, failures[split])
            log(
                f"split={split} progress={index}/{len(prompt_rows)} "
                f"sample_id={sample_id} status={result['status']} "
                f"cached={cached_rows} generated={materialized_rows} failures={len(failures[split])}"
            )

        ordered = sorted(output_rows_by_id.values(), key=lambda item: (item["split"], item["sample_id"]))
        summary[split] = {
            "rows": len(ordered),
            "status_counts": dict(status_counter),
            "failures": len(failures[split]),
        }
        log(
            f"split={split} completed rows={len(ordered)} cached={cached_rows} "
            f"generated={materialized_rows} failures={len(failures[split])}"
        )

    write_json(resolve_path(config["pipeline"]["audit_root"]) / "analysis_sft" / f"teacher_responses_{scope}.json", summary)
    log(f"saved summary audit for scope={scope}")
    return {"output_root": str(output_root), "summary": summary}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate teacher responses for analysis prompts.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--scope", choices=["after_2009", "before_2009"], default="after_2009")
    parser.add_argument("--teacher-source", choices=["llm", "archived_master"], default=None)
    parser.add_argument("--force-refresh", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    result = generate_teacher_responses(
        args.config,
        scope=args.scope,
        teacher_source=args.teacher_source,
        force_refresh=args.force_refresh,
    )
    print(f"Saved teacher responses to {result['output_root']}")


if __name__ == "__main__":
    main()
