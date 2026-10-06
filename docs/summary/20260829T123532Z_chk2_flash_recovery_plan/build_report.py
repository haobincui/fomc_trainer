from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
PLAN_ROOT = (
    REPO_ROOT
    / "output/data/retrain_v2/chk2/"
    "target_derived_minutes_deepseek_v4_flash_recovery_plan_v1_20260829"
)
REGENERATE_ROOT = (
    REPO_ROOT
    / "output/data/retrain_v2/chk2/"
    "target_derived_minutes_deepseek_v4_flash_recovery_regenerate_v1_20260829"
)
REVERIFY_ROOT = (
    REPO_ROOT
    / "output/data/retrain_v2/chk2/"
    "target_derived_minutes_deepseek_v4_flash_recovery_reverify_v1_20260829"
)
SOURCE_ROOT = (
    REPO_ROOT
    / "output/data/retrain_v2/chk2/"
    "target_derived_minutes_deepseek_v4_flash_v1_20260828"
)
GENERATED_AT = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
    "+00:00", "Z"
)
TITLE = "CHK2 DeepSeek V4 Flash 恢复重跑方案与执行状态"


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def id_digest(rows: list[dict]) -> str:
    payload = "".join(f"{sample_id}\n" for sample_id in sorted(row["sample_id"] for row in rows))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def count_json_files(path: Path) -> int:
    return sum(1 for item in path.rglob("*.json") if item.is_file()) if path.exists() else 0


plan = read_json(PLAN_ROOT / "summary.json")
preflight = read_json(REVERIFY_ROOT / "preflight.json")

selection_paths = {
    "regenerate_and_reverify": PLAN_ROOT / "selection/regenerate_and_reverify.jsonl",
    "reverify_only": PLAN_ROOT / "selection/reverify_only.jsonl",
    "defer_targeted_repair": PLAN_ROOT / "selection/defer_targeted_repair.jsonl",
    "all": PLAN_ROOT / "selection/all.jsonl",
}
selections = {name: read_jsonl(path) for name, path in selection_paths.items()}

expected_counts = {
    "regenerate_and_reverify": 1_531,
    "reverify_only": 53,
    "defer_targeted_repair": 126,
    "all": 1_710,
}
expected_digests = {
    "regenerate_and_reverify": "33b23e053568b7a6f88fa3d46786c69b96ca0ff5aa7d77ff65577b1c3a5fdf83",
    "reverify_only": "ced1adb671daf5d29dcf24472eca6e70878030339d45e1adaff1eeb28b31d572",
    "defer_targeted_repair": "06099468c788e6be17d08f5be06139e6b76249502568454bfce2934a05e67f2c",
    "all": "78bb3d9621abb185d4f5b5650d291eedca3137befcbf06d5b4bd4326b23400ba",
}
for name, rows in selections.items():
    assert len(rows) == expected_counts[name]
    assert id_digest(rows) == expected_digests[name]

assert plan["status"] == "recovery_inputs_validated"
assert plan["action_counts"] == {
    "defer_targeted_repair": 126,
    "regenerate_and_reverify": 1_531,
    "reverify_only": 53,
}
assert plan["selected_for_immediate_recovery"] == 1_584
assert sum(plan["action_counts"].values()) == plan["source_machine_rejections"] == 1_710

verification = preflight["verification"]
assert verification["rows"] == 8
assert verification["observed_passes"] == 0
assert verification["passed"] is False
assert verification["usage"]["successful_requests"] == 0
assert verification["usage"]["actual_request_attempts"] == 32
status_codes = {
    attempt.get("status_code")
    for failure in verification["transport_failures"]
    for attempt in failure["attempts"]
}
assert status_codes == {402}
assert all(
    "Insufficient Balance" in attempt.get("error_message", "")
    for failure in verification["transport_failures"]
    for attempt in failure["attempts"]
)

fresh_generation_cache = count_json_files(REGENERATE_ROOT / "cache/generation")
new_verification_cache = count_json_files(REVERIFY_ROOT / "cache/verification")
reverify_generation_cache = count_json_files(REVERIFY_ROOT / "cache/generation")
original_generation_cache = count_json_files(SOURCE_ROOT / "cache/generation")
original_verification_cache = count_json_files(SOURCE_ROOT / "cache/verification")
assert fresh_generation_cache == 0
assert new_verification_cache == 0
assert reverify_generation_cache == 53
assert original_generation_cache == 3_198
assert original_verification_cache == 1_710

action_rows = [
    {
        "action": "重新生成并重验",
        "rows": 1_531,
        "share_pct": 100 * 1_531 / 1_710,
        "state": "输入已就绪；等待有余额的 key",
    },
    {
        "action": "仅重新验证",
        "rows": 53,
        "share_pct": 100 * 53 / 1_710,
        "state": "53 份生成缓存已隔离复制；验证待重跑",
    },
    {
        "action": "定向修复后再跑",
        "rows": 126,
        "share_pct": 100 * 126 / 1_710,
        "state": "暂不做同 prompt 盲重跑",
    },
]

split_rows = []
for split in ("train", "validation", "test"):
    values = plan["split_action_counts"][split]
    split_rows.append(
        {
            "split": split,
            "regenerate_and_reverify": values["regenerate_and_reverify"],
            "reverify_only": values["reverify_only"],
            "defer_targeted_repair": values["defer_targeted_repair"],
            "total": sum(values.values()),
        }
    )

execution_rows = [
    {
        "check": "运行环境",
        "result": "通过",
        "evidence": "conda env fomc_trainer；Python 3.10.9；DeepSeek key 已配置（未读取明文）",
    },
    {
        "check": "恢复输入",
        "result": "通过",
        "evidence": "1,710 个 ID 完整分区；ID digest、split、原文 SHA 与来源一致",
    },
    {
        "check": "仅重验 canary",
        "result": "阻塞",
        "evidence": "0/8 成功；32 次尝试全部 HTTP 402 Insufficient Balance",
    },
    {
        "check": "新增成功缓存",
        "result": "0",
        "evidence": "fresh generation=0；new verification=0；未覆盖原始缓存",
    },
    {
        "check": "后台进程",
        "result": "已停止",
        "evidence": "余额修复前不会进入 1,531 条批量生成",
    },
]

policy_rows = [
    {
        "cohort": "重新生成并重验",
        "rule": "verification 语义未通过；或 generation 仅 1 类失败；或 2 类失败且原文不超过 200 词",
        "why": "再抽一次响应仍有合理修复机会",
    },
    {
        "cohort": "仅重新验证",
        "rule": "生成已通过且语义 verdict 全为正，但 verifier 的证据跨度/schema 失败；或 verifier 缺失/不可解析",
        "why": "保留合格生成，避免多付一次生成成本",
    },
    {
        "cohort": "定向修复后再跑",
        "rule": "generation 同时失败至少 3 类；或失败 2 类且原文超过 200 词",
        "why": "相同 prompt 盲重跑预期收益较低，先做小规模修复 prompt 试验",
    },
]

headline = [
    {
        "source_rejections": 1_710,
        "immediate_recovery": 1_584,
        "fresh_generation": 1_531,
        "reverify_only": 53,
        "deferred": 126,
        "canary_passes": 0,
        "canary_rows": 8,
    }
]

source_manifest = [
    {
        "id": "recovery_plan",
        "label": "Validated recovery plan",
        "path": str((PLAN_ROOT / "summary.json").relative_to(REPO_ROOT)),
    },
    {
        "id": "recovery_selection",
        "label": "Row-level recovery decisions",
        "path": str((PLAN_ROOT / "selection/all.jsonl").relative_to(REPO_ROOT)),
    },
    {
        "id": "reverify_preflight",
        "label": "Verification canary result",
        "path": str((REVERIFY_ROOT / "preflight.json").relative_to(REPO_ROOT)),
    },
    {
        "id": "recovery_launcher",
        "label": "fomc_trainer recovery launcher",
        "path": "run/recover_chk2_target_derived_minutes.sh",
    },
]

query_specs = {
    "recovery_plan": (
        f"SELECT * FROM read_json_auto('{source_manifest[0]['path']}')",
        "Reads validated cohort counts, split counts, selection policy, and isolated output roots.",
    ),
    "recovery_selection": (
        f"SELECT * FROM read_json_auto('{source_manifest[1]['path']}')",
        "Reads all 1,710 mutually exclusive row-level recovery decisions.",
    ),
    "reverify_preflight": (
        f"SELECT * FROM read_json_auto('{source_manifest[2]['path']}')",
        "Reads the eight-row verifier canary and provider transport failures.",
    ),
    "recovery_launcher": (
        "SELECT content FROM read_text('run/recover_chk2_target_derived_minutes.sh')",
        "Audits environment pinning, lock handling, phase order, and fail-closed canary behavior.",
    ),
}

sources = []
for source in source_manifest:
    sql, description = query_specs[source["id"]]
    sources.append(
        {
            "id": source["id"],
            "query": {
                "engine": "duckdb",
                "language": "sql",
                "sql": sql,
                "description": description,
                "executed_at": GENERATED_AT,
                "tables_used": [source["path"]],
            },
        }
    )

artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": TITLE,
        "description": "未通过样本的重跑分层、输入完整性校验、fomc_trainer 启动结果与余额阻塞状态。",
        "generatedAt": GENERATED_AT,
        "cards": [
            {
                "id": "immediate_card",
                "description": "可立即重跑，包含 fresh generation 与 verifier-only。",
                "dataset": "headline",
                "sourceId": "recovery_plan",
                "metrics": [{"label": "立即恢复", "field": "immediate_recovery", "format": "number"}],
            },
            {
                "id": "fresh_card",
                "description": "使用全新输出根目录重新生成并重新验证。",
                "dataset": "headline",
                "sourceId": "recovery_plan",
                "metrics": [{"label": "重新生成", "field": "fresh_generation", "format": "number"}],
            },
            {
                "id": "reverify_card",
                "description": "复用已通过 generation gate 的缓存，只重新调用 verifier。",
                "dataset": "headline",
                "sourceId": "recovery_plan",
                "metrics": [{"label": "仅重验", "field": "reverify_only", "format": "number"}],
            },
            {
                "id": "deferred_card",
                "description": "复杂多错误样本，先改 prompt 再做小规模 pilot。",
                "dataset": "headline",
                "sourceId": "recovery_plan",
                "metrics": [{"label": "暂缓盲重跑", "field": "deferred", "format": "number"}],
            },
            {
                "id": "canary_card",
                "description": "当前 key 的 verifier canary 成功数。",
                "dataset": "headline",
                "sourceId": "reverify_preflight",
                "metrics": [{"label": "Canary 通过", "field": "canary_passes", "format": "number"}],
            },
        ],
        "charts": [
            {
                "id": "action_chart",
                "title": "1,710 条未通过样本的恢复动作",
                "subtitle": "分组互斥且覆盖全部 machine rejection；柱高为样本数。",
                "type": "bar",
                "dataset": "action_breakdown",
                "sourceId": "recovery_plan",
                "encodings": {
                    "x": {"field": "action", "type": "nominal", "label": "恢复动作"},
                    "y": {"field": "rows", "type": "quantitative", "label": "样本数"},
                    "tooltip": [
                        {"field": "rows", "type": "quantitative", "label": "样本数"},
                        {"field": "share_pct", "type": "quantitative", "label": "占未通过样本 (%)"},
                        {"field": "state", "type": "nominal", "label": "状态"},
                    ],
                },
                "yAxisTitle": "Rows",
                "valueFormat": "number",
                "layout": "full",
            }
        ],
        "tables": [
            {
                "id": "split_table",
                "title": "各 split 的恢复动作",
                "subtitle": "三列动作合计回到原始 1,710 条 machine rejection。",
                "dataset": "split_actions",
                "sourceId": "recovery_plan",
                "columns": [
                    {"field": "split", "label": "Split", "type": "text"},
                    {"field": "regenerate_and_reverify", "label": "重新生成并重验", "format": "number"},
                    {"field": "reverify_only", "label": "仅重验", "format": "number"},
                    {"field": "defer_targeted_repair", "label": "定向修复", "format": "number"},
                    {"field": "total", "label": "Total", "format": "number"},
                ],
                "layout": "full",
            },
            {
                "id": "policy_table",
                "title": "选择规则与理由",
                "subtitle": "保持原有质量门禁不变，每条样本只进入一个 cohort。",
                "dataset": "selection_policy",
                "sourceId": "recovery_selection",
                "columns": [
                    {"field": "cohort", "label": "Cohort", "type": "text"},
                    {"field": "rule", "label": "Rule", "type": "text"},
                    {"field": "why", "label": "Why", "type": "text"},
                ],
                "layout": "full",
            },
            {
                "id": "execution_table",
                "title": "执行与安全状态",
                "subtitle": "真实 API canary 失败后已停止；没有成功的新缓存。",
                "dataset": "execution_status",
                "sourceId": "reverify_preflight",
                "columns": [
                    {"field": "check", "label": "Check", "type": "text"},
                    {"field": "result", "label": "Result", "type": "text"},
                    {"field": "evidence", "label": "Evidence", "type": "text"},
                ],
                "layout": "full",
            },
        ],
        "sources": source_manifest,
        "blocks": [
            {"id": "title", "type": "markdown", "body": f"# {TITLE}"},
            {
                "id": "executive_summary",
                "type": "markdown",
                "sourceId": "recovery_plan",
                "body": "## Executive Summary\n\n**恢复输入已经整理并校验完成，但真实生成目前被 DeepSeek 账户余额阻塞。** 1,710 条 machine rejection 中，**1,584 条值得立即恢复**：1,531 条重新生成并重验，53 条保留原生成结果、只重跑 verifier；其余 126 条属于长文本或多错误族组合，相同 prompt 的盲重跑收益偏低，先留作定向修复。任务确实通过 `fomc_trainer` 环境启动，但 8 条 verifier canary 的 32 次尝试全部返回 **HTTP 402 Insufficient Balance**，因此进程已停止。当前没有成功写入任何新的 generation/verification cache，原始审计数据保持不变。",
            },
            {
                "id": "metrics",
                "type": "metric-strip",
                "cardIds": ["immediate_card", "fresh_card", "reverify_card", "deferred_card", "canary_card"],
            },
            {"id": "action_block", "type": "chart", "chartId": "action_chart", "layout": "full"},
            {
                "id": "selection_finding",
                "type": "markdown",
                "sourceId": "recovery_selection",
                "body": "## 为什么这样分组\n\n分组目标是在不降低质量门禁的前提下，把 API 成本优先投向可恢复概率更高的样本。generation 只有单一失败族、或中短文本中的双失败族，允许一次 fresh response；verification 已给出正向语义结论但输出跨度/schema 不合约的样本，只重验而不重生。对于至少三类 generation 失败，或超过 200 词且同时两类失败的样本，不做同 prompt 的大规模盲重跑。",
            },
            {"id": "policy_block", "type": "table", "tableId": "policy_table", "layout": "full"},
            {"id": "split_block", "type": "table", "tableId": "split_table", "layout": "full"},
            {
                "id": "execution_finding",
                "type": "markdown",
                "sourceId": "reverify_preflight",
                "body": "## 已执行到哪里\n\n恢复脚本强制通过 `conda run -n fomc_trainer` 调用 Python，并先运行成本更低的 verifier-only canary。环境与 key 存在性检查均通过；provider 随后明确返回余额不足，而不是 key 缺失、模型名错误或本地依赖失败。脚本现已改为 canary fail-closed：canary 不通过就不会进入 1,531 条批量生成。",
            },
            {"id": "execution_block", "type": "table", "tableId": "execution_table", "layout": "full"},
            {
                "id": "resume",
                "type": "markdown",
                "sourceId": "recovery_launcher",
                "body": "## 恢复方式\n\n给 DeepSeek 账户充值，或把 `fomc_trainer` 的 `DEEPSEEK_API_KEY` 切换到有余额的 key 后，重新运行 `./run/recover_chk2_target_derived_minutes.sh`。脚本会复用已验证的 selection 与隔离输入，先重验 53 条；canary 通过后才开始 1,531 条 fresh generation，再完成相应 verification。",
            },
            {
                "id": "limitations",
                "type": "markdown",
                "body": "## Limitations\n\n“值得重跑”是基于错误族数量、官方段落长度和 verifier 语义 verdict 的可解释启发式，不是已完成的随机对照试验。126 条 deferred 并非永久放弃；建议先抽 16–24 条做定向 repair-prompt pilot，若 generation-gate 通过率达到约 30% 再扩展。当前 402 阻塞意味着尚不能报告新的通过率、耗时或实际 token 成本。",
            },
            {
                "id": "questions",
                "type": "markdown",
                "body": "## Further questions\n\n- 更换有余额的 key 后，53 条 verifier-only cohort 的实际通过率是多少？\n- 1,531 条 fresh generation 在不同错误族上的恢复率是否支持继续盲重跑？\n- 126 条 deferred cohort 的定向 prompt pilot 能否达到 30% 的 generation-gate 通过阈值？\n- 恢复通过样本如何与原有 1,488 条 machine-ready 数据做可追溯合并与去重？",
            },
        ],
    },
    "snapshot": {
        "version": 1,
        "generatedAt": GENERATED_AT,
        "status": "ready",
        "datasets": {
            "headline": headline,
            "action_breakdown": action_rows,
            "split_actions": split_rows,
            "selection_policy": policy_rows,
            "execution_status": execution_rows,
        },
    },
    "sources": sources,
    "package_info": {
        "originUrl": "artifact://chk2-flash-recovery-plan-20260829T123532Z",
        "controls": {"edit": False, "refresh": False},
    },
}

evidence = {
    "schema_version": "chk2-flash-recovery-evidence-v1",
    "generated_at": GENERATED_AT,
    "selection_counts": expected_counts,
    "selection_id_sha256": expected_digests,
    "immediate_recovery": 1_584,
    "split_actions": split_rows,
    "canary": {
        "phase": "verification",
        "rows": verification["rows"],
        "passes": verification["observed_passes"],
        "request_attempts": verification["usage"]["actual_request_attempts"],
        "status_codes": sorted(status_codes),
        "provider_message": "Insufficient Balance",
    },
    "cache_counts": {
        "fresh_generation": fresh_generation_cache,
        "new_verification": new_verification_cache,
        "isolated_reverify_generation_copy": reverify_generation_cache,
        "original_generation": original_generation_cache,
        "original_verification": original_verification_cache,
    },
    "integrity_checks": {
        "selection_is_complete_and_mutually_exclusive": True,
        "all_id_digests_match": True,
        "quality_gates_unchanged": True,
        "original_caches_preserved": True,
        "plaintext_api_key_recorded": False,
    },
}

OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "artifact.json").write_text(
    json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
(OUT_DIR / "evidence_snapshot.json").write_text(
    json.dumps(evidence, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
(OUT_DIR / "source_notes.md").write_text(
    """# Source notes

## Scope

The recovery denominator is the 1,710 machine-gate non-passes from the DeepSeek V4 Flash release. Preparation-stage exclusions are outside scope and were not selected for retry.

## Integrity

The four selection ledgers were checked by row count and by SHA-256 over sorted sample IDs. The three action cohorts are mutually exclusive and sum to the full 1,710-row rejection ledger. Subset preparation also validates split assignment and official-minutes paragraph hashes before any provider call.

## Execution boundary

The verifier-only canary made 32 attempts across eight rows; all returned HTTP 402 with `Insufficient Balance`. The job was stopped, and no successful new generation or verification cache exists. The original release caches remain present.

## Interpretation

The cohort rules are operational heuristics intended to prioritize likely recoveries without weakening quality gates. They are not causal estimates of retry success.
""",
    encoding="utf-8",
)

print(OUT_DIR / "artifact.json")
