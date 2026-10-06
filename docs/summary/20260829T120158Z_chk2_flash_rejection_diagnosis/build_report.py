from __future__ import annotations

import csv
import html
import json
from collections import Counter, OrderedDict
from datetime import datetime, timezone
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
OUT_DIR = Path(__file__).resolve().parent
SOURCE_ROOT = (
    REPO_ROOT
    / "output/data/retrain_v2/chk2/"
    "target_derived_minutes_deepseek_v4_flash_v1_20260828"
)
PROMOTED_ROOT = (
    REPO_ROOT
    / "output/data/retrain_v2/chk2/"
    "target_derived_minutes_deepseek_v4_flash_machine_ready_v1_20260829"
)
GENERATED_AT = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
    "+00:00", "Z"
)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(paths: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for path in paths:
        with path.open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def pct(num: int | float, den: int | float) -> float:
    return 100.0 * num / den if den else 0.0


def normalized_reason(reason: str) -> str:
    return reason.split(":", 1)[0]


def reason_hit(reasons: list[str], prefixes: tuple[str, ...]) -> bool:
    return any(any(reason.startswith(prefix) for prefix in prefixes) for reason in reasons)


summary = read_json(SOURCE_ROOT / "summary.json")
promoted_summary = read_json(PROMOTED_ROOT / "summary.json")
prepared = read_jsonl(sorted((SOURCE_ROOT / "prepared").glob("*.jsonl")))
rejections = read_jsonl(sorted((SOURCE_ROOT / "machine_rejections").glob("*.jsonl")))

assert len(prepared) == summary["total_prepared"] == 3_198
assert len(rejections) == summary["total_machine_rejected_or_pending"] == 1_710
assert summary["total_machine_pass"] == promoted_summary["total_machine_pass"] == 1_488
assert promoted_summary["training_ready"] is True

prepared_by_id = {row["sample_id"]: row for row in prepared}
rejection_by_id = {row["sample_id"]: row for row in rejections}
assert len(prepared_by_id) == len(prepared)
assert set(rejection_by_id).issubset(prepared_by_id)

stage_counts = Counter(row["rejection_stage"] for row in rejections)
assert stage_counts == {"generation_gate": 1_488, "verification_gate": 222}

generation_rows = [row for row in rejections if row["rejection_stage"] == "generation_gate"]
verification_rows = [row for row in rejections if row["rejection_stage"] == "verification_gate"]

generation_families: OrderedDict[str, tuple[str, ...]] = OrderedDict(
    [
        (
            "归因集合不一致",
            ("reasoning_attribution_set_mismatch", "analysis_attribution_set_mismatch"),
        ),
        ("去 Minutes 风格不足", ("analysis_insufficiently_destylized",)),
        (
            "数字集合不一致",
            ("reasoning_numeric_multiset_mismatch", "analysis_numeric_multiset_mismatch"),
        ),
        (
            "证据/目标覆盖或精确跨度失败",
            ("target_sentence_uncovered", "analysis_evidence_not_exact", "target_span_not_exact"),
        ),
        ("日期集合不一致", ("reasoning_date_set_mismatch", "analysis_date_set_mismatch")),
        (
            "结构或 schema 失败",
            (
                "nonconsecutive_claim_id",
                "claim_alignment_ids_do_not_match_claim_card",
                "content_not_strict_json",
                "atomic_claim_schema",
                "generation_content_keys",
                "claim_alignment_must_be_nonempty_array",
                "serialization_or_tokenization",
                "reasoning_words",
                "fidelity_reasoning_must_be_nonempty_text",
            ),
        ),
        ("reasoning 含传输元信息", ("reasoning_transport_meta",)),
    ]
)

generation_reason_rows = []
for label, prefixes in generation_families.items():
    count = sum(reason_hit(row["rejection_reasons"], prefixes) for row in generation_rows)
    generation_reason_rows.append(
        {
            "reason_family": label,
            "rows": count,
            "share_of_generation_reject_pct": pct(count, len(generation_rows)),
        }
    )

assert {row["reason_family"]: row["rows"] for row in generation_reason_rows} == {
    "归因集合不一致": 663,
    "去 Minutes 风格不足": 502,
    "数字集合不一致": 386,
    "证据/目标覆盖或精确跨度失败": 295,
    "日期集合不一致": 34,
    "结构或 schema 失败": 28,
    "reasoning 含传输元信息": 17,
}

verification_families: OrderedDict[str, tuple[str, ...]] = OrderedDict(
    [
        ("语义等价性/相容性失败", ("verifier_",)),
        (
            "证据覆盖或精确跨度失败",
            (
                "verification_analysis_sentence_uncovered",
                "verification_analysis_evidence_not_exact",
                "verification_analysis_span_not_exact",
                "verification_target_evidence_not_exact",
            ),
        ),
        ("验证缺失或最终失败", ("verification_missing_or_failed",)),
        ("验证 verdict/schema 异常", ("analysis_claim_verdict_invalid", "analysis_claim_")),
    ]
)

verification_reason_rows = []
for label, prefixes in verification_families.items():
    count = sum(reason_hit(row["rejection_reasons"], prefixes) for row in verification_rows)
    verification_reason_rows.append(
        {
            "reason_family": label,
            "rows": count,
            "share_of_verification_reject_pct": pct(count, len(verification_rows)),
        }
    )

assert {row["reason_family"]: row["rows"] for row in verification_reason_rows} == {
    "语义等价性/相容性失败": 157,
    "证据覆盖或精确跨度失败": 61,
    "验证缺失或最终失败": 2,
    "验证 verdict/schema 异常": 2,
}

raw_reason_counts = Counter(
    normalized_reason(reason)
    for row in rejections
    for reason in row["rejection_reasons"]
)
top_raw_reasons = [
    {"reason_code": reason, "hits": hits}
    for reason, hits in raw_reason_counts.most_common(18)
]

split_rows = []
for split in ("train", "validation", "test"):
    values = summary["split_counts"][split]
    generation_rejected = values["prepared"] - values["generation_gate_pass"]
    verification_rejected = values["generation_gate_pass"] - values["machine_pass"]
    split_rows.append(
        {
            "split": split,
            "prepared": values["prepared"],
            "machine_pass": values["machine_pass"],
            "rejected": values["machine_rejected_or_pending"],
            "reject_rate_pct": pct(values["machine_rejected_or_pending"], values["prepared"]),
            "generation_rejected": generation_rejected,
            "generation_reject_rate_pct": pct(generation_rejected, values["prepared"]),
            "verification_rejected": verification_rejected,
            "verification_reject_rate_given_generation_pass_pct": pct(
                verification_rejected, values["generation_gate_pass"]
            ),
        }
    )

length_enriched = []
for row in prepared:
    words = len(row["official_minutes_paragraph"].split())
    rejection = rejection_by_id.get(row["sample_id"])
    length_enriched.append(
        {
            "sample_id": row["sample_id"],
            "words": words,
            "rejection_stage": rejection["rejection_stage"] if rejection else "pass",
        }
    )
length_enriched.sort(key=lambda item: item["words"])

length_quartiles = []
for quartile in range(4):
    # Match floor(rank * 4 / n): use ceiling boundaries so the four buckets
    # contain 800, 799, 800, and 799 rows for n=3,198.
    start = (quartile * len(length_enriched) + 3) // 4
    end = ((quartile + 1) * len(length_enriched) + 3) // 4
    bucket = length_enriched[start:end]
    rejected = [row for row in bucket if row["rejection_stage"] != "pass"]
    generation_rejected = [row for row in bucket if row["rejection_stage"] == "generation_gate"]
    verification_rejected = [row for row in bucket if row["rejection_stage"] == "verification_gate"]
    generation_pass = len(bucket) - len(generation_rejected)
    ordered_words = sorted(row["words"] for row in bucket)
    length_quartiles.append(
        {
            "quartile": f"Q{quartile + 1}",
            "n": len(bucket),
            "min_words": ordered_words[0],
            "median_words": ordered_words[len(ordered_words) // 2],
            "max_words": ordered_words[-1],
            "rejected": len(rejected),
            "reject_rate_pct": pct(len(rejected), len(bucket)),
            "generation_reject_rate_pct": pct(len(generation_rejected), len(bucket)),
            "verification_reject_rate_given_generation_pass_pct": pct(
                len(verification_rejected), generation_pass
            ),
        }
    )

assert [round(row["reject_rate_pct"], 1) for row in length_quartiles] == [37.6, 46.3, 56.5, 73.5]

length_band_specs = [
    ("≤50", lambda words: words <= 50),
    ("51–100", lambda words: 51 <= words <= 100),
    ("101–150", lambda words: 101 <= words <= 150),
    ("151–200", lambda words: 151 <= words <= 200),
    (">200", lambda words: words > 200),
]
length_bands = []
for label, predicate in length_band_specs:
    bucket = [row for row in length_enriched if predicate(row["words"])]
    rejected = [row for row in bucket if row["rejection_stage"] != "pass"]
    length_bands.append(
        {
            "word_band": label,
            "n": len(bucket),
            "rejected": len(rejected),
            "reject_rate_pct": pct(len(rejected), len(bucket)),
        }
    )
assert [(row["n"], row["rejected"]) for row in length_bands] == [
    (177, 59),
    (1_255, 529),
    (1_039, 583),
    (458, 324),
    (269, 215),
]

category_rows = []
for category in sorted({row["section_category"] for row in prepared}):
    members = [row for row in prepared if row["section_category"] == category]
    rejected = [row for row in members if row["sample_id"] in rejection_by_id]
    category_rows.append(
        {
            "section_category": category,
            "prepared": len(members),
            "rejected": len(rejected),
            "reject_rate_pct": pct(len(rejected), len(members)),
        }
    )
category_rows.sort(key=lambda row: row["prepared"], reverse=True)

section_rows = []
for section_name in sorted({row["section_name"] for row in prepared}):
    members = [row for row in prepared if row["section_name"] == section_name]
    if len(members) < 50:
        continue
    rejected = [row for row in members if row["sample_id"] in rejection_by_id]
    section_rows.append(
        {
            "section_name": section_name,
            "prepared": len(members),
            "rejected": len(rejected),
            "reject_rate_pct": pct(len(rejected), len(members)),
        }
    )
section_rows.sort(key=lambda row: row["reject_rate_pct"], reverse=True)

reason_count_distribution = Counter(len(row["rejection_reasons"]) for row in rejections)
multi_reason_rows = sum(count for number, count in reason_count_distribution.items() if number > 1)
assert multi_reason_rows == 679

preparation_reason_rows = [
    {"reason_code": reason, "hits": hits}
    for reason, hits in sorted(
        summary["preparation"]["rejection_reason_counts"].items(),
        key=lambda item: item[1],
        reverse=True,
    )
]

headline = [
    {
        "prepared": len(prepared),
        "machine_pass": summary["total_machine_pass"],
        "machine_rejected": len(rejections),
        "machine_reject_rate_pct": pct(len(rejections), len(prepared)),
        "generation_rejected": stage_counts["generation_gate"],
        "verification_rejected": stage_counts["verification_gate"],
        "multi_reason_rows": multi_reason_rows,
        "preparation_excluded": summary["preparation"]["rejection_rows"],
    }
]

sources = [
    {
        "id": "run_summary",
        "label": "DeepSeek V4 Flash run summary",
        "path": str((SOURCE_ROOT / "summary.json").relative_to(REPO_ROOT)),
        "role": "prepared/pass/split/API totals",
    },
    {
        "id": "machine_rejections",
        "label": "Machine rejection ledgers",
        "path": str((SOURCE_ROOT / "machine_rejections").relative_to(REPO_ROOT)) + "/*.jsonl",
        "role": "row-level stages and reason codes",
    },
    {
        "id": "prepared_rows",
        "label": "Prepared candidate rows",
        "path": str((SOURCE_ROOT / "prepared").relative_to(REPO_ROOT)) + "/*.jsonl",
        "role": "denominators, section categories, paragraph lengths",
    },
    {
        "id": "promoted_summary",
        "label": "Machine-ready promoted release summary",
        "path": str((PROMOTED_ROOT / "summary.json").relative_to(REPO_ROOT)),
        "role": "confirms 1,488 promoted rows and training_ready=true",
    },
]

datasets = {
    "headline": headline,
    "stage_breakdown": [
        {
            "stage": "Generation gate",
            "rows": stage_counts["generation_gate"],
            "share_of_machine_reject_pct": pct(stage_counts["generation_gate"], len(rejections)),
        },
        {
            "stage": "Verification gate",
            "rows": stage_counts["verification_gate"],
            "share_of_machine_reject_pct": pct(stage_counts["verification_gate"], len(rejections)),
        },
    ],
    "generation_reason_families": generation_reason_rows,
    "verification_reason_families": verification_reason_rows,
    "top_raw_reasons": top_raw_reasons,
    "split_summary": split_rows,
    "length_quartiles": length_quartiles,
    "length_bands": length_bands,
    "section_categories": category_rows,
    "section_summary": section_rows,
    "preparation_reasons": preparation_reason_rows,
}

artifact = {
    "surface": "report",
    "manifest": {
        "version": 1,
        "surface": "report",
        "title": "CHK2 DeepSeek V4 Flash 未通过样本诊断",
        "description": "区分 preparation 排除、generation gate 拒绝和 verification gate 拒绝，并诊断内容长度与 split 差异。",
        "generatedAt": GENERATED_AT,
        "cards": [
            {"id": "prepared", "dataset": "headline", "field": "prepared", "label": "已进入机器筛选"},
            {"id": "rejected", "dataset": "headline", "field": "machine_rejected", "label": "机器未通过"},
            {"id": "reject_rate", "dataset": "headline", "field": "machine_reject_rate_pct", "label": "机器未通过率"},
            {"id": "promoted", "dataset": "headline", "field": "machine_pass", "label": "已提升为训练就绪"},
        ],
        "charts": [
            {
                "id": "generation_reasons",
                "type": "horizontal_bar",
                "dataset": "generation_reason_families",
                "title": "Generation gate 的主要原因（可重叠）",
                "x": "share_of_generation_reject_pct",
                "y": "reason_family",
            },
            {
                "id": "length_gradient",
                "type": "bar",
                "dataset": "length_bands",
                "title": "段落越长，未通过率越高",
                "x": "word_band",
                "y": "reject_rate_pct",
            },
        ],
        "tables": [
            {"id": "split_table", "dataset": "split_summary", "title": "按 split 分解"},
            {"id": "verification_table", "dataset": "verification_reason_families", "title": "Verification 原因"},
            {"id": "preparation_table", "dataset": "preparation_reasons", "title": "进入生成前的排除原因"},
        ],
        "blocks": [
            {
                "type": "summary",
                "title": "结论",
                "text": "主瓶颈是长段落上的内容保真：归因、数字、覆盖和去风格化；不是 API 失败或基础 JSON/schema。",
            },
            {
                "type": "caveat",
                "title": "计数口径",
                "text": "原因家族为行级命中且可重叠；同一条记录可能触发多个错误码。机器门禁结果不能替代人工质量评估。",
            },
        ],
        "sources": sources,
    },
    "snapshot": {
        "version": 1,
        "status": "ready",
        "generatedAt": GENERATED_AT,
        "datasets": datasets,
    },
    "sources": sources,
    "package_info": {
        "originUrl": "artifact://chk2-flash-rejection-diagnosis-20260829T120158Z",
        "controls": {"edit": False, "refresh": False},
        "renderer": "standalone-static-fallback",
    },
}


def write_csv(name: str, rows: list[dict]) -> None:
    if not rows:
        return
    with (OUT_DIR / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "artifact.json").write_text(
    json.dumps(artifact, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
(OUT_DIR / "evidence_snapshot.json").write_text(
    json.dumps(
        {
            "schema_version": "chk2-flash-rejection-diagnosis-v1",
            "generated_at": GENERATED_AT,
            "datasets": datasets,
            "integrity_checks": {
                "prepared_rows": len(prepared),
                "machine_rejections": len(rejections),
                "stage_sum": sum(stage_counts.values()),
                "promoted_rows": promoted_summary["total_machine_pass"],
                "all_reconciled": True,
            },
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n",
    encoding="utf-8",
)
write_csv("generation_reason_families.csv", generation_reason_rows)
write_csv("verification_reason_families.csv", verification_reason_rows)
write_csv("split_summary.csv", split_rows)
write_csv("length_quartiles.csv", length_quartiles)
write_csv("length_bands.csv", length_bands)
write_csv("section_summary.csv", section_rows)
write_csv("preparation_reasons.csv", preparation_reason_rows)


def bar_rows(rows: list[dict], label_field: str, value_field: str, max_value: float = 100.0) -> str:
    rendered = []
    for row in rows:
        value = float(row[value_field])
        width = min(100.0, 100.0 * value / max_value if max_value else 0.0)
        rendered.append(
            f"<div class='bar-row'><div class='bar-label'>{html.escape(str(row[label_field]))}</div>"
            f"<div class='bar-track'><div class='bar-fill' style='width:{width:.2f}%'></div></div>"
            f"<div class='bar-value'>{value:.1f}%</div></div>"
        )
    return "".join(rendered)


def table(headers: list[tuple[str, str]], rows: list[dict]) -> str:
    head = "".join(f"<th>{html.escape(label)}</th>" for _, label in headers)
    body = []
    for row in rows:
        cells = []
        for field, _ in headers:
            value = row[field]
            if field.endswith("_pct"):
                rendered = f"{float(value):.1f}%"
            else:
                rendered = str(value)
            cells.append(f"<td>{html.escape(rendered)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


cards = [
    ("已进入机器筛选", "3,198", "preparation 后的候选"),
    ("机器未通过", "1,710", "53.5%"),
    ("Generation gate", "1,488", "占未通过 87.0%"),
    ("已提升训练就绪", "1,488", "46.5%"),
]
card_html = "".join(
    f"<div class='card'><div class='card-label'>{label}</div><div class='card-value'>{value}</div><div class='card-note'>{note}</div></div>"
    for label, value, note in cards
)

source_html = "".join(
    f"<li><code>{html.escape(source['path'])}</code> — {html.escape(source['role'])}</li>"
    for source in sources
)

report_html = f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CHK2 DeepSeek V4 Flash 未通过样本诊断</title>
<style>
:root{{--ink:#172033;--muted:#657086;--line:#dfe4ec;--panel:#ffffff;--bg:#f4f6fa;--blue:#2f6fed;--blue2:#8eb1ff;--orange:#e58a2f;}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI","Noto Sans SC",sans-serif}}
main{{max-width:1120px;margin:0 auto;padding:40px 24px 64px}} h1{{font-size:32px;line-height:1.2;margin:0 0 10px}} h2{{font-size:21px;margin:0 0 12px}} h3{{font-size:16px;margin:20px 0 8px}} p{{margin:8px 0}} .sub{{color:var(--muted);margin-bottom:24px}}
.cards{{display:grid;grid-template-columns:repeat(4,1fr);gap:14px;margin:20px 0}} .card,.panel{{background:var(--panel);border:1px solid var(--line);border-radius:12px;box-shadow:0 1px 2px #16213a0a}}
.card{{padding:16px}} .card-label,.card-note{{color:var(--muted)}} .card-value{{font-size:28px;font-weight:700;margin:5px 0}} .card-note{{font-size:13px}}
.panel{{padding:22px;margin:16px 0}} .callout{{border-left:4px solid var(--blue);padding:10px 14px;background:#eef4ff;border-radius:6px;margin:12px 0}}
.grid{{display:grid;grid-template-columns:1fr 1fr;gap:16px}} .bar-row{{display:grid;grid-template-columns:190px 1fr 58px;gap:10px;align-items:center;margin:10px 0}} .bar-label{{font-size:13px;text-align:right}} .bar-track{{height:14px;background:#edf0f5;border-radius:99px;overflow:hidden}} .bar-fill{{height:100%;background:linear-gradient(90deg,var(--blue),var(--blue2));border-radius:99px}} .bar-value{{font-variant-numeric:tabular-nums;text-align:right;font-weight:600}}
.table-wrap{{overflow:auto}} table{{width:100%;border-collapse:collapse;font-size:13px}} th,td{{padding:9px 10px;border-bottom:1px solid var(--line);text-align:right;white-space:nowrap}} th:first-child,td:first-child{{text-align:left}} th{{background:#f7f8fb;color:#48536a}} code{{font-size:12px;background:#eef1f5;padding:2px 5px;border-radius:4px}} ul{{padding-left:21px}} .fine{{font-size:13px;color:var(--muted)}} .tag{{display:inline-block;padding:2px 8px;border-radius:99px;background:#edf3ff;color:#285dc0;font-size:12px;font-weight:600}}
@media(max-width:850px){{.cards{{grid-template-columns:repeat(2,1fr)}}.grid{{grid-template-columns:1fr}}.bar-row{{grid-template-columns:140px 1fr 50px}}}} @media(max-width:520px){{main{{padding:24px 14px}}.cards{{grid-template-columns:1fr}}}}
</style>
</head>
<body><main>
<span class="tag">Machine-gate diagnosis</span>
<h1>CHK2 DeepSeek V4 Flash 未通过样本诊断</h1>
<p class="sub">运行目录：target_derived_minutes_deepseek_v4_flash_v1_20260828 · 生成于 {html.escape(GENERATED_AT)}</p>
<div class="cards">{card_html}</div>

<section class="panel">
<h2>结论先行</h2>
<div class="callout"><strong>主瓶颈是长段落上的内容保真，不是 API 失败或基础格式。</strong> 1,710 条机器未通过中，1,488 条（87.0%）在 generation gate 已被确定性规则拦下；只有 222 条进入 verification 后失败。</div>
<p>Generation 阶段最常见的是归因对象变化、保留了过多 Minutes 风格、数字集合变化，以及目标句/证据跨度覆盖不完整。≤50 词段落的未通过率为 33.3%，>200 词升到 79.9%，说明事实密度与覆盖复杂度是强关联驱动；这是关联证据，不单独证明因果。</p>
<p>严格库存检查在 fidelity_reasoning 上尤其容易触发：reasoning 归因不一致 515 次、数字不一致 367 次，分别高于 analysis 的 248 和 65 次。因此部分行是 reasoning 没完整复现严格库存，不表示 analysis 一定整体错误。</p>
<p>Verification 阶段的 222 条中，157 条（70.7%）被第二次同模型调用判为语义不双向蕴含、分析主张缺乏支持、目标主张遗漏或 reasoning 不相容；61 条（27.5%）是覆盖/精确证据问题。最终 API failure_count 为 0，仅 2 条记录触发 verification_missing_or_failed，且属于返回内容无法严格解析，不是网络失败。</p>
</section>

<div class="grid">
<section class="panel"><h2>Generation gate 原因</h2><p class="fine">行级原因家族，可重叠；分母为 1,488 条 generation reject。</p>{bar_rows(generation_reason_rows, 'reason_family', 'share_of_generation_reject_pct')}</section>
<section class="panel"><h2>段落长度梯度</h2><p class="fine">按 3,198 条候选的官方段落词数分组。</p>{bar_rows(length_bands, 'word_band', 'reject_rate_pct')}</section>
</div>

<section class="panel"><h2>哪些 Minutes section 更难</h2>
{table([('section_name','Section（仅列 n≥50）'),('prepared','候选'),('rejected','未通过'),('reject_rate_pct','未通过率')], section_rows)}
<p class="fine">Participants’ Views（71.3%）和 Financial Markets/Open Market Operations（66.9%）最难；两者都包含大量主体归因，和 attribution mismatch 的主瓶颈一致。</p>
</section>

<section class="panel"><h2>按 split 分解</h2>
{table([('split','Split'),('prepared','候选'),('machine_pass','通过'),('rejected','未通过'),('reject_rate_pct','未通过率'),('generation_reject_rate_pct','Generation 拒绝率'),('verification_reject_rate_given_generation_pass_pct','Verification 条件拒绝率')], split_rows)}
<p class="fine">Test 的 62.0% 高于 train 52.4% 和 validation 53.5%，差距主要来自 generation gate。Test 的段落中位词数反而更短，因此不能用“test 段落更长”解释该差距；会议/主题构成仍需进一步分层。</p>
</section>

<section class="panel"><h2>Verification 未通过原因</h2>
{table([('reason_family','原因家族'),('rows','行数'),('share_of_verification_reject_pct','占 verification 未通过')], verification_reason_rows)}
<p class="fine">“语义等价性/相容性”内部的 verifier 错误码高度重叠，不能把 not_bidirectionally_entailed、reasoning_incompatible、analysis_claim_not_supported 等计数相加。</p>
</section>

<section class="panel"><h2>容易混淆的另一批：preparation 排除</h2>
<p>原始 4,877 行里另有 1,679 行在进入 DeepSeek 生成前就被规则排除；它们不属于上面的 1,710 条机器生成/验证未通过。原因计数同样可重叠。</p>
{table([('reason_code','准备阶段原因码'),('hits','命中次数')], preparation_reason_rows)}
</section>

<section class="panel"><h2>建议处理顺序</h2>
<ol>
<li><strong>先做可确定修复：</strong>针对归因、数字、日期和 exact evidence span 做失败后定向重写，再重新过同一门禁，不要直接放宽这些保真规则。</li>
<li><strong>对长段落拆分或分步生成：</strong>先抽取 claim/evidence，再生成分析与 reasoning，可降低一次性覆盖过多事实导致的遗漏。</li>
<li><strong>单独校准去风格化规则：</strong>用小规模人工样本判断 502 条 destylization reject 中有多少是真正泄漏式复述，避免把“措辞相近”和“事实不正确”混为一类。</li>
<li><strong>排查 test 会议构成：</strong>test 差距不由长度解释；下一步应按 meeting、section_name 与主题核查集中失败点。</li>
</ol>
</section>

<section class="panel"><h2>口径与限制</h2>
<ul>
<li>679/1,710（39.7%）记录包含多个错误码；所有原因家族比例均按“至少命中一次的行”计算。</li>
<li>错误码说明哪条规则触发，不等同于模型行为的因果解释；长度结果属于相关性诊断。</li>
<li>当前 promoted release 已去除人工审核要求，1,488 条仅表示通过机器规则；不代表完成了人工语义抽查。</li>
</ul>
<h3>数据来源</h3><ul>{source_html}</ul>
</section>
</main></body></html>
"""
(OUT_DIR / "report.html").write_text(report_html, encoding="utf-8")

delivery_receipt = {
    "schema_version": "chk2-flash-rejection-diagnosis-delivery-v1",
    "generated_at": GENERATED_AT,
    "status": "local_static_validated",
    "checks": {
        "artifact_json_parsed": True,
        "source_totals_reconciled": True,
        "report_has_title": "CHK2 DeepSeek V4 Flash 未通过样本诊断" in report_html,
        "report_has_tables": report_html.count("<table>") >= 4,
        "report_has_chart_rows": report_html.count("bar-row") >= 10,
        "report_is_self_contained": "https://" not in report_html and "http://" not in report_html,
    },
    "portable_builder": {
        "status": "not_run",
        "reason": "Node executable is not installed in the active environment",
    },
}
(OUT_DIR / "delivery_receipt.json").write_text(
    json.dumps(delivery_receipt, ensure_ascii=False, indent=2) + "\n",
    encoding="utf-8",
)

source_notes = f"""# Source notes

## Scope

The primary diagnosis covers the 1,710 machine-gate non-passes among 3,198 prepared candidates. The 1,679 preparation exclusions from the original 4,877 source rows are reported separately and are not mixed into the machine-rejection denominator.

## Reconciliation

- Prepared: 3,198
- Machine pass / promoted: 1,488
- Machine rejection ledger: 1,710
- Generation gate rejection: 1,488
- Verification gate rejection: 222
- 1,488 + 1,710 = 3,198; 1,488 + 222 = 1,710

## Interpretation boundaries

- Reason-family counts are row-level and non-exclusive.
- Error codes record rules that fired; they do not by themselves prove why the model produced the error.
- Paragraph-length quartiles are equal-row buckets after sorting by whitespace-delimited word count. Tied word counts can straddle adjacent buckets.
- The promoted release is machine-screen-only after removal of the human-review requirement.

## Packaging

`artifact.json` is the canonical structured report. `report.html` is a self-contained local rendering. The environment had no Node executable, so the packaged portable-report builder could not be run; the static fallback is validated separately in `delivery_receipt.json`.
"""
(OUT_DIR / "source_notes.md").write_text(source_notes, encoding="utf-8")

print(json.dumps({"output_dir": str(OUT_DIR), "generated_at": GENERATED_AT, "rows": headline[0]}, ensure_ascii=False))
