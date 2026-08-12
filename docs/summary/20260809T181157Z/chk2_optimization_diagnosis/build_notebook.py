from __future__ import annotations

from pathlib import Path

import nbformat as nbf


OUTPUT_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = OUTPUT_DIR / "chk2_optimization_diagnosis.ipynb"


def markdown(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


cells = [
    markdown(
        r"""
# chk2 终止失败与评测合同优化诊断

## tl;dr

- 旧 33-row 共测不是 chk2 的主任务评测：其每条 prompt 中位 **12,402 tokens / 650 facts**，而 chk2 训练 prompt 中位约 **1,249 tokens / 9 facts**；任务也从 `atomic evidence → analysis` 变成了 `all evidence → Minutes section`。
- 三模型共 **99/99** 输出都生成满 8,192 tokens，**0/99** 出现 `</think>`；其中 **93/99** 尾部存在可严格复现的固定周期。因此失败发生在隐式 reasoning 阶段，并非 answer 后 EOS 失效。
- checkpoint-150 在训练域内最近 80 个 completion 中，**78/80** 有 `</think>` 和非空 answer，截断仅 **2/80**，rolling-20 reward 在 step 150 达到此前最高值。因此现有共测不能证明 cp150 退化。
- 优先动作是新增 task-aligned evaluator 和修正解码配置，而不是继续提高 8,192-token 上限。下一轮 SFT 前还应修复双 BOS、统一 reasoning 长度合同，并清理 **201/1,743（11.5%）** 不符合纯文本合同的 final answers。
"""
    ),
    markdown(
        r"""
## Context & Methods

本 notebook 回答两个问题：第一，为什么冻结的 chk0/chk1/chk2 共测全部达到 8,192-token 上限；第二，下一轮最小、最有信息量的优化顺序是什么。分析只读取已经冻结的训练数据、训练 completion、模型 tokenizer/config 和共测工件，不重跑模型，也不修改 checkpoint。

### Key Assumptions

- `</think>` 是 reasoning 与 answer 的唯一边界；chat template 已在生成 prompt 末尾预填 `<think>\n`，所以 raw completion 不应重复 `<think>`。
- “达到 token 上限”按 `finish_reason=length` 且 `output_token_count=max_new_tokens` 定义。
- “严格周期尾部”要求最后至少 128 个 tokenizer tokens 与某个前置固定周期逐 token 完全一致；这是保守定义，未检出的近周期漂移不算入 93 条。
- 旧 length-tolerant 分数仅用于故障诊断，因为无边界时 evaluator 会把整段未完成 reasoning 当作 candidate，而不是 final answer。
"""
    ),
    markdown("## Data"),
    code(
        r"""
from pathlib import Path
import json
import re
import statistics

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sys
import transformers
import yaml
from IPython.display import display
from transformers import AutoTokenizer

REPO_ROOT = Path("/home/haobin_cui/research_files_space_2/fomc_trainer")
OUT_DIR = REPO_ROOT / "docs/summary/20260809T181157Z/chk2_optimization_diagnosis"

MODEL_DIR = REPO_ROOT / "models/DeepSeek-R1-Distill-Llama-8B"
SFT_DIR = REPO_ROOT / "dataset/processed/retrain_v2/chk1_reasoning_compressed_flash_max_v1_20260805/analysis_sft"
GRPO_DIR = REPO_ROOT / "dataset/processed/retrain_v2/analysis_base_full_v7_automated_v3_20260804/analysis_grpo"
TRAIN_COMPLETIONS = REPO_ROOT / "output/training/retrain_v2/chk2_compressed_chk1_v1_reward_v3_long4096_fresh_20260807/adapters/chk2/completions"
EVAL_ROOT = REPO_ROOT / "output/evaluation/main/checkpoint_generation/retrain_v2_chk0_chk1_chk2_cp150_20260809"
EVAL_GENERATIONS = EVAL_ROOT / "generations"
EVAL_REFERENCES = REPO_ROOT / "output/evaluation/main/checkpoint_generation/checkpoint_eval_11_v1/dataset/references.jsonl"

SFT_CONFIG = REPO_ROOT / "configs/retrain_v2/chk1_analysis_sft_compressed_flash_max_v1_20260805.yaml"
GRPO_CONFIG = REPO_ROOT / "configs/retrain_v2/chk2_analysis_grpo_compressed_chk1_v1_20260805.yaml"
EVAL_CONFIG = REPO_ROOT / "configs/main/checkpoint_generation_eval_11.json"

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, local_files_only=True, use_fast=True)
expected_python = REPO_ROOT.parent.parent / ".conda/envs/fomc_trainer/bin/python"
probe_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": "environment probe"}],
    tokenize=True,
    add_generation_prompt=True,
)
assert Path(sys.executable).resolve() == expected_python.resolve(), (sys.executable, expected_python)
assert transformers.__version__ == "4.57.6", transformers.__version__
assert tokenizer.is_fast and tokenizer.__class__.__name__ == "LlamaTokenizerFast"
assert tokenizer.add_bos_token is True
assert isinstance(probe_ids, list) and probe_ids and all(isinstance(item, int) for item in probe_ids)
print({
    "python": sys.executable,
    "transformers": transformers.__version__,
    "tokenizer_class": tokenizer.__class__.__name__,
    "eos_token_id": tokenizer.eos_token_id,
    "bos_token_id": tokenizer.bos_token_id,
    "pad_token_id": tokenizer.pad_token_id,
})
"""
    ),
    code(
        r"""
def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def quantile_summary(values) -> dict:
    values = list(values)
    return {
        "n": len(values),
        "min": float(np.min(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
    }


def token_count(text: str) -> int:
    return len(tokenizer.encode(text, add_special_tokens=False))


def lexical_4gram_repetition(text: str) -> float:
    tokens = re.findall(r"\w+|[^\w\s]", text.lower())
    grams = [tuple(tokens[i : i + 4]) for i in range(max(0, len(tokens) - 3))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def exact_periodic_suffix(token_ids: list[int], min_tail: int = 128, max_period: int = 256):
    n_tokens = len(token_ids)
    for period in range(1, min(max_period, n_tokens - min_tail) + 1):
        if all(token_ids[i] == token_ids[i - period] for i in range(n_tokens - min_tail, n_tokens)):
            start = n_tokens - min_tail - period
            while start > 0 and token_ids[start - 1] == token_ids[start - 1 + period]:
                start -= 1
            return {
                "period_tokens": period,
                "start_token": start,
                "cycles": (n_tokens - start) / period,
                "suffix_share": (n_tokens - start) / n_tokens,
            }
    return None


def compose_user_prompt(prompt: str, suffix: str | None) -> str:
    if not suffix:
        return prompt
    return f"{prompt.rstrip()}\n\n{suffix.strip()}"
"""
    ),
    code(
        r"""
sft_config = yaml.safe_load(SFT_CONFIG.read_text(encoding="utf-8"))
grpo_config = yaml.safe_load(GRPO_CONFIG.read_text(encoding="utf-8"))
eval_config = json.loads(EVAL_CONFIG.read_text(encoding="utf-8"))

sft_rows = []
for split in ("train", "eval", "test"):
    for row in read_jsonl(SFT_DIR / f"{split}.jsonl"):
        row = dict(row)
        row["split"] = split
        sft_rows.append(row)

grpo_rows = []
for split in ("train", "eval", "test"):
    for row in read_jsonl(GRPO_DIR / f"{split}.jsonl"):
        row = dict(row)
        row["split"] = split
        grpo_rows.append(row)

generation_files = sorted(EVAL_GENERATIONS.glob("*.jsonl"))
generation_rows = {path.stem: read_jsonl(path) for path in generation_files}
reference_rows = read_jsonl(EVAL_REFERENCES)

print({
    "sft_rows": len(sft_rows),
    "grpo_rows": len(grpo_rows),
    "generation_rows": {key: len(value) for key, value in generation_rows.items()},
    "reference_rows": len(reference_rows),
})
"""
    ),
    markdown("## Results"),
    markdown(
        r"""
### 1. 旧共测把输入规模和输出任务同时改变了

下面使用相同 tokenizer 比较实际渲染后的 prompt。SFT/GRPO 每行只分析一个 atomic topic；旧共测把全部指标一次性拼进约 650 条事实，并要求直接撰写完整 Minutes section。
"""
    ),
    code(
        r"""
sft_prompt_tokens = []
sft_reasoning_tokens = []
sft_answer_tokens = []
sft_completion_tokens = []
sft_fact_counts = []
sft_series_counts = []
sft_answer_paragraphs = []
sft_full_train_tokens = []

for row in sft_rows:
    messages = [
        {"role": "system", "content": sft_config["system_prompt"]},
        {"role": "user", "content": row["prompt"]},
    ]
    rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    sft_prompt_tokens.append(len(tokenizer(rendered_prompt, add_special_tokens=False)["input_ids"]))
    reasoning, answer = row["response"].split("</think>", 1)
    sft_reasoning_tokens.append(token_count(reasoning))
    sft_answer_tokens.append(token_count(answer))
    sft_completion_tokens.append(token_count(row["response"]))
    fact_card = json.loads(row["provided_data"])
    sft_fact_counts.append(len(fact_card["evidence"]))
    sft_series_counts.append(len({fact["series_id"] for fact in fact_card["evidence"]}))
    sft_answer_paragraphs.append(len([p for p in answer.strip().split("\n\n") if p.strip()]))
    completion_with_eos = row["response"] + tokenizer.eos_token
    # This reproduces the current TRL plain prompt-completion path, including its second BOS.
    sft_full_train_tokens.append(len(tokenizer(rendered_prompt + completion_with_eos)["input_ids"]))

grpo_prompt_tokens = []
grpo_fact_counts = []
grpo_series_counts = []
for row in grpo_rows:
    user_content = compose_user_prompt(row["prompt"], grpo_config.get("user_prompt_suffix"))
    messages = [
        {"role": "system", "content": grpo_config["system_prompt"]},
        {"role": "user", "content": user_content},
    ]
    grpo_prompt_tokens.append(len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)))
    fact_card = json.loads(row["provided_data"])
    grpo_fact_counts.append(len(fact_card["evidence"]))
    grpo_series_counts.append(len({fact["series_id"] for fact in fact_card["evidence"]}))

eval_prompt_tokens = [row["prompt_token_count"] for row in next(iter(generation_rows.values()))]
eval_fact_counts = [len(row["evidence_facts"]) for row in next(iter(generation_rows.values()))]
eval_series_counts = [len({fact["series_id"] for fact in row["evidence_facts"]}) for row in next(iter(generation_rows.values()))]
eval_topic_counts = [len({fact["indicator"] for fact in row["evidence_facts"]}) for row in next(iter(generation_rows.values()))]
reference_tokens = [token_count(row["reference"]) for row in reference_rows]
reference_paragraphs = [len([p for p in row["reference"].split("\n\n") if p.strip()]) for row in reference_rows]

prompt_contract = pd.DataFrame([
    {
        "corpus": "chk1 SFT",
        "task": "atomic evidence → concise analysis",
        "rows": len(sft_rows),
        "prompt_p50": np.median(sft_prompt_tokens),
        "prompt_max": np.max(sft_prompt_tokens),
        "facts_p50": np.median(sft_fact_counts),
        "facts_max": np.max(sft_fact_counts),
        "series_p50": np.median(sft_series_counts),
        "target_tokens_p50": np.median(sft_answer_tokens),
        "target_paragraphs_p50": np.median(sft_answer_paragraphs),
    },
    {
        "corpus": "chk2 GRPO",
        "task": "atomic evidence → concise analysis",
        "rows": len(grpo_rows),
        "prompt_p50": np.median(grpo_prompt_tokens),
        "prompt_max": np.max(grpo_prompt_tokens),
        "facts_p50": np.median(grpo_fact_counts),
        "facts_max": np.max(grpo_fact_counts),
        "series_p50": np.median(grpo_series_counts),
        "target_tokens_p50": np.nan,
        "target_paragraphs_p50": np.nan,
    },
    {
        "corpus": "旧 33-row stress",
        "task": "all evidence → Minutes section",
        "rows": 33,
        "prompt_p50": np.median(eval_prompt_tokens),
        "prompt_max": np.max(eval_prompt_tokens),
        "facts_p50": np.median(eval_fact_counts),
        "facts_max": np.max(eval_fact_counts),
        "series_p50": np.median(eval_series_counts),
        "target_tokens_p50": np.median(reference_tokens),
        "target_paragraphs_p50": np.median(reference_paragraphs),
    },
])
display(prompt_contract.round(1))
print("eval topics per row:", quantile_summary(eval_topic_counts))
print("SFT full train token max / configured max:", max(sft_full_train_tokens), sft_config["max_length"])
"""
    ),
    code(
        r"""
fig, ax = plt.subplots(figsize=(9.5, 4.8))
x = np.arange(len(prompt_contract))
width = 0.35
ax.bar(x - width / 2, prompt_contract["prompt_p50"], width, label="Median", color="#326B9A")
ax.bar(x + width / 2, prompt_contract["prompt_max"], width, label="Maximum", color="#D7A23A")
ax.axhline(grpo_config["max_prompt_length"], color="#333333", linestyle="--", linewidth=1.3, label="chk2 cap: 2,560")
ax.set_xticks(x, prompt_contract["corpus"])
ax.set_ylabel("Rendered prompt tokens")
ax.set_title("Rendered prompt length by corpus")
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        r"""
### 2. 直接故障是 reasoning 循环，而不是 EOS 丢失

raw completion 已排除 chat template 预填的 `<think>`。因此关键判断是是否生成了 `</think>`：旧共测中三模型均为 0，说明 answer 从未开始。严格周期检测进一步确认绝大多数尾部已经进入短循环。
"""
    ),
    code(
        r"""
stress_records = []
period_records = []
model_labels = {
    "eval-chk0-base": "chk0",
    "eval-chk1-compressed-sft": "chk1",
    "eval-chk2-reward-v3-cp150": "chk2 cp150",
}

for artifact, rows in generation_rows.items():
    periods = []
    for row in rows:
        token_ids = tokenizer.encode(row["generated"], add_special_tokens=False)
        detected = exact_periodic_suffix(token_ids)
        if detected:
            periods.append(detected)
            period_records.append({"model": model_labels[artifact], **detected})
    stress_records.append({
        "model": model_labels[artifact],
        "rows": len(rows),
        "length_finish_rate": np.mean([row["generation_finish_reason"] == "length" for row in rows]),
        "boundary_rate": np.mean(["</think>" in row["generated"] for row in rows]),
        "explicit_answer_rate": np.mean(["</think>" in row["generated"] and bool(row["generated"].split("</think>", 1)[1].strip()) for row in rows]),
        "strict_periodic_suffix_rate": len(periods) / len(rows),
        "period_tokens_p50": np.median([item["period_tokens"] for item in periods]),
        "period_start_p50": np.median([item["start_token"] for item in periods]),
        "period_suffix_share_p50": np.median([item["suffix_share"] for item in periods]),
        "lexical_4gram_repetition_p50": np.median([lexical_4gram_repetition(row["generated"]) for row in rows]),
    })

stress_summary = pd.DataFrame(stress_records)
display(stress_summary.round(4))
print("strict periodic rows:", len(period_records), "/ 99")
"""
    ),
    code(
        r"""
cp150_frames = []
for step in range(131, 151):
    frame = pd.read_parquet(TRAIN_COMPLETIONS / f"completions_{step:05d}.parquet")
    cp150_frames.append(frame)
cp150 = pd.concat(cp150_frames, ignore_index=True)
cp150["completion_tokens"] = cp150["completion"].map(token_count)
cp150["has_boundary"] = cp150["completion"].str.contains("</think>", regex=False)
cp150["has_answer"] = cp150["completion"].map(
    lambda text: "</think>" in text and bool(text.split("</think>", 1)[1].strip())
)
cp150["hit_4096"] = cp150["completion_tokens"] >= 4096
cp150["reasoning_tokens"] = cp150["completion"].map(
    lambda text: token_count(text.split("</think>", 1)[0]) if "</think>" in text else np.nan
)
cp150["answer_tokens"] = cp150["completion"].map(
    lambda text: token_count(text.split("</think>", 1)[1]) if "</think>" in text else np.nan
)

health_comparison = pd.DataFrame([
    {"scope": "chk2 in-domain, steps 131–150", "metric": "Reached answer boundary", "rate": cp150["has_boundary"].mean()},
    {"scope": "chk2 in-domain, steps 131–150", "metric": "Hit token limit", "rate": cp150["hit_4096"].mean()},
    {"scope": "chk2 old Minutes stress", "metric": "Reached answer boundary", "rate": stress_summary.loc[stress_summary.model == "chk2 cp150", "boundary_rate"].iloc[0]},
    {"scope": "chk2 old Minutes stress", "metric": "Hit token limit", "rate": stress_summary.loc[stress_summary.model == "chk2 cp150", "length_finish_rate"].iloc[0]},
])
display(health_comparison)
print({
    "in_domain_rows": len(cp150),
    "boundary_rows": int(cp150.has_boundary.sum()),
    "nonempty_answer_rows": int(cp150.has_answer.sum()),
    "clipped_rows": int(cp150.hit_4096.sum()),
    "completion_tokens": quantile_summary(cp150.completion_tokens),
    "reasoning_tokens_valid": quantile_summary(cp150.reasoning_tokens.dropna()),
    "answer_tokens_valid": quantile_summary(cp150.answer_tokens.dropna()),
    "reward": quantile_summary(cp150.grounded_analysis_reward_v3),
})
"""
    ),
    code(
        r"""
fig, ax = plt.subplots(figsize=(9.5, 4.8))
pivot = health_comparison.pivot(index="metric", columns="scope", values="rate") * 100
pivot.plot(kind="bar", ax=ax, color=["#326B9A", "#D7A23A"], width=0.72)
ax.set_ylabel("Rows (%)")
ax.set_xlabel("")
ax.set_title("chk2 termination health: training domain vs old stress test")
ax.set_ylim(0, 105)
ax.legend(frameon=False, loc="upper center")
ax.grid(axis="y", alpha=0.2)
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        r"""
### 3. cp150 的训练域信号仍然是当前最强窗口

为避免用旧 Minutes stress test 错杀 checkpoint，这里复算 step 1–150 的 completion health。rolling-20 reward 在 step 150 达到此前最高值；边界率和截断率没有同步恶化。
"""
    ),
    code(
        r"""
training_records = []
for path in sorted(TRAIN_COMPLETIONS.glob("completions_*.parquet")):
    step = int(path.stem.split("_")[-1])
    if step > 150:
        continue
    frame = pd.read_parquet(path)
    for row in frame.itertuples(index=False):
        length = token_count(row.completion)
        training_records.append({
            "step": step,
            "reward": float(row.grounded_analysis_reward_v3),
            "has_boundary": "</think>" in row.completion,
            "hit_4096": length >= 4096,
            "completion_tokens": length,
        })

training_health = pd.DataFrame(training_records)
by_step = training_health.groupby("step").agg(
    reward=("reward", "mean"),
    boundary_rate=("has_boundary", "mean"),
    clipped_rate=("hit_4096", "mean"),
    completion_tokens=("completion_tokens", "mean"),
)
by_step["reward_rolling20"] = by_step.reward.rolling(20).mean()
by_step["boundary_rolling20"] = by_step.boundary_rate.rolling(20).mean()
by_step["clipped_rolling20"] = by_step.clipped_rate.rolling(20).mean()

best_reward_end = int(by_step.reward_rolling20.idxmax())
print({
    "best_rolling20_end_step": best_reward_end,
    "best_rolling20_reward": float(by_step.loc[best_reward_end, "reward_rolling20"]),
    "step150_boundary_rate_rolling20": float(by_step.loc[150, "boundary_rolling20"]),
    "step150_clipped_rate_rolling20": float(by_step.loc[150, "clipped_rolling20"]),
})

fig, ax = plt.subplots(figsize=(9.5, 4.8))
ax.plot(by_step.index, by_step.reward_rolling20, color="#326B9A", linewidth=2, label="Reward, rolling 20 steps")
ax.axvline(150, color="#D7A23A", linestyle="--", linewidth=1.5, label="Selected checkpoint 150")
ax.set_xlabel("Training step")
ax.set_ylabel("Mean reward")
ax.set_title("chk2 in-domain reward through checkpoint 150")
ax.grid(alpha=0.2)
ax.legend(frameon=False)
plt.tight_layout()
plt.show()
"""
    ),
    markdown(
        r"""
### 4. 训练合同内部仍有三处需要修复

第一，SFT reasoning 的最短 target 已超过 chk2 prompt 所写的 512-token 上限。第二，少量 final answer 保留 JSON/evidence IDs，和 chk2 的纯文本合同冲突。第三，当前 SFT 数据路径会在训练序列开头产生两个 BOS，而推理只有一个。
"""
    ),
    code(
        r"""
identifier_pattern = re.compile(r"(?i)(?:\bev-[0-9a-z]+\b|\bevidence_ids?\b)")
quality_rows = []
for row in sft_rows:
    answer = row["response"].split("</think>", 1)[1].strip()
    json_like = answer.startswith("{") or answer.startswith("[") or '"analysis"' in answer[:300] or '"answer"' in answer[:300]
    has_identifier = bool(identifier_pattern.search(answer))
    quality_rows.append({
        "split": row["split"],
        "json_like": json_like,
        "has_evidence_identifier": has_identifier,
        "contract_violation": json_like or has_identifier,
        "malformed_short_answer": len(answer) < 10,
    })
quality = pd.DataFrame(quality_rows)

contract_quality = pd.DataFrame([
    {"check": "JSON-like answer", "rows": int(quality.json_like.sum()), "rate": quality.json_like.mean()},
    {"check": "Evidence identifier", "rows": int(quality.has_evidence_identifier.sum()), "rate": quality.has_evidence_identifier.mean()},
    {"check": "Union contract violation", "rows": int(quality.contract_violation.sum()), "rate": quality.contract_violation.mean()},
    {"check": "Malformed <10 chars", "rows": int(quality.malformed_short_answer.sum()), "rate": quality.malformed_short_answer.mean()},
])
display(contract_quality.assign(rate_pct=contract_quality.rate * 100).round(3))
display(quality.groupby("split").agg(rows=("split", "size"), violations=("contract_violation", "sum"), malformed=("malformed_short_answer", "sum")))

reasoning_contract = pd.DataFrame([
    {"threshold": "≤512", "sft_target_share": np.mean(np.array(sft_reasoning_tokens) <= 512), "cp150_valid_share": np.mean(cp150.reasoning_tokens.dropna() <= 512)},
    {"threshold": "≤768", "sft_target_share": np.mean(np.array(sft_reasoning_tokens) <= 768), "cp150_valid_share": np.mean(cp150.reasoning_tokens.dropna() <= 768)},
    {"threshold": "≤1024", "sft_target_share": np.mean(np.array(sft_reasoning_tokens) <= 1024), "cp150_valid_share": np.mean(cp150.reasoning_tokens.dropna() <= 1024)},
    {"threshold": "≤1536", "sft_target_share": np.mean(np.array(sft_reasoning_tokens) <= 1536), "cp150_valid_share": np.mean(cp150.reasoning_tokens.dropna() <= 1536)},
])
display(reasoning_contract.round(4))
print("SFT reasoning:", quantile_summary(sft_reasoning_tokens))
print("SFT full completion:", quantile_summary(sft_completion_tokens))
"""
    ),
    code(
        r"""
fig, ax = plt.subplots(figsize=(9.5, 4.8))
plot_quality = contract_quality.iloc[:3].copy()
ax.bar(plot_quality["check"], plot_quality["rate"] * 100, color=["#D7A23A", "#326B9A", "#C65D45"])
for idx, row in plot_quality.iterrows():
    ax.text(idx, row.rate * 100 + 0.25, f"{row.rows} rows", ha="center", va="bottom", fontsize=9)
ax.set_ylabel("SFT rows (%)")
ax.set_title("chk1 final-answer output-contract violations")
ax.set_ylim(0, 13)
ax.grid(axis="y", alpha=0.2)
plt.tight_layout()
plt.show()
"""
    ),
    code(
        r"""
sample = sft_rows[0]
messages = [
    {"role": "system", "content": sft_config["system_prompt"]},
    {"role": "user", "content": sample["prompt"]},
]
rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
current_prompt_ids = tokenizer(rendered_prompt)["input_ids"]
current_train_ids = tokenizer(rendered_prompt + sample["response"] + tokenizer.eos_token)["input_ids"]
inference_prompt_ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
stripped_prompt = rendered_prompt.removeprefix(tokenizer.bos_token)
fixed_prompt_ids = tokenizer(stripped_prompt)["input_ids"]
fixed_train_ids = tokenizer(stripped_prompt + sample["response"] + tokenizer.eos_token)["input_ids"]

assert current_prompt_ids[:2] == [tokenizer.bos_token_id, tokenizer.bos_token_id]
assert fixed_prompt_ids == inference_prompt_ids

all_training_eos = []
all_training_lengths = []
for row in sft_rows:
    messages = [
        {"role": "system", "content": sft_config["system_prompt"]},
        {"role": "user", "content": row["prompt"]},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    ids = tokenizer(prompt + row["response"] + tokenizer.eos_token)["input_ids"]
    all_training_eos.append(ids[-1] == tokenizer.eos_token_id)
    all_training_lengths.append(len(ids))

token_path_checks = {
    "current_training_prefix_ids": current_prompt_ids[:4],
    "inference_prefix_ids": inference_prompt_ids[:4],
    "fixed_training_prefix_ids": fixed_train_ids[:4],
    "all_1743_end_with_eos": all(all_training_eos),
    "max_full_training_tokens": max(all_training_lengths),
    "configured_max_length": sft_config["max_length"],
    "eos_would_be_truncated": max(all_training_lengths) > sft_config["max_length"],
}
token_path_checks
"""
    ),
    markdown(
        r"""
### 5. length-tolerant 的轻微语义增益不能用于晋级

由于 0/99 有 answer boundary，length-tolerant evaluator 实际比较的是 8,192-token 未完成 reasoning 与 Minutes reference。它显示 chk2 相对 chk1 的 BERTScore/MPNet 只有很小变化，且 Holm 校正后均不显著；这些数字只能说明 fallback scorer 的行为，不能代表 final answer 质量。
"""
    ),
    code(
        r"""
lt_contrasts = read_jsonl(EVAL_ROOT / "scores_length_tolerant/contrasts.jsonl")
selected_metrics = {"bertscore_f1", "mpnet_cosine", "rouge_l_f1", "evidence_value_coverage", "repetition_rate"}
lt_chk2_vs_chk1 = pd.DataFrame([
    {
        "metric": row["metric"],
        "chk1": row["mean_baseline"],
        "chk2": row["mean_candidate"],
        "paired_difference": row["paired_mean_difference"],
        "ci_lower": row["ci_lower"],
        "ci_upper": row["ci_upper"],
        "holm_p": row["p_value_holm"],
    }
    for row in lt_contrasts
    if row["evaluation_subset"] == "all_11_meetings"
    and row["baseline_artifact_id"] == "eval-chk1-compressed-sft"
    and row["candidate_artifact_id"] == "eval-chk2-reward-v3-cp150"
    and row["metric"] in selected_metrics
]).sort_values("metric")
display(lt_chk2_vs_chk1.round(6))
"""
    ),
    markdown(
        r"""
## Takeaways

1. **保留 cp150，不基于旧共测降级。** step 150 是截至该点最强 rolling-20 reward 窗口，而且训练域终止健康；旧共测应重新命名为 long-context raw-evidence→Minutes stress benchmark。
2. **先建 task-aligned 190-row 主评测。** 复用 `analysis_grpo/test.jsonl`，使用和 chk2 训练完全相同的 atomic analysis prompt、无输入截断、answer-only factual metrics 和 meeting-cluster paired bootstrap。
3. **先做解码 A/B，再锁正式配置。** 本地模型 README 推荐 `temperature=0.6`（0.5–0.7）以避免 endless repetition，并建议把指令放在 user prompt。用 12 条 held-out prompts 对比 greedy 与 `0.6/0.95`，以及 repetition penalty `1.0/1.05`；正式评测采用胜出的固定配置和逐样本 seed。不要把 `</think>` 设为 stop。
4. **统一 reasoning 合同。** 当前 SFT target 中位 996 tokens、最短 517，而 chk2 suffix 写“最多 512”。建议统一为“目标 768–1,024，硬上限 1,536”，并把 completion cap 收紧至 2,048（诊断可先用 2,304 覆盖最长 SFT target）。
5. **清理并版本化 chk1 数据。** 201 条 output-contract violation 应通过本地确定性解析/去 ID 修复；仅对无法恢复的单条 `{` 回到已有缓存/原 target，避免重新生成全部数据。
6. **修复训练/推理 token parity。** 去掉当前 SFT 的双 BOS，并加入单 BOS、最终 EOS、`</think>` 后 answer、train/inference token parity 测试。
7. **若最终目标是 Minutes，走 chk2→chk3。** 不要让 chk1/chk2 直接承担 650-fact Minutes 写作；先按 atomic topic 生成 analysis，再由 chk3 聚合/改写 Minutes。
"""
    ),
    code(
        r"""
assert len(period_records) == 93
assert [int((stress_summary.set_index("model").loc[label, "strict_periodic_suffix_rate"] * 33).round()) for label in ("chk0", "chk1", "chk2 cp150")] == [31, 30, 32]
assert min(sft_reasoning_tokens) == 517
assert float(np.median(sft_reasoning_tokens)) == 996.0
assert max(sft_reasoning_tokens) == 1600
assert max(sft_completion_tokens) == 1784
assert max(all_training_lengths) == 4181
assert int(cp150.hit_4096.sum()) == 2
assert current_prompt_ids[:2] == [tokenizer.bos_token_id, tokenizer.bos_token_id]
assert fixed_prompt_ids == inference_prompt_ids

summary_payload = {
    "schema_version": "chk2-optimization-diagnosis-v1",
    "stress_test": {
        "rows": 99,
        "length_finished": int(sum(row["generation_finish_reason"] == "length" for rows in generation_rows.values() for row in rows)),
        "answer_boundaries": int(sum("</think>" in row["generated"] for rows in generation_rows.values() for row in rows)),
        "strict_periodic_suffixes": len(period_records),
        "prompt_tokens_p50": float(np.median(eval_prompt_tokens)),
        "facts_p50": float(np.median(eval_fact_counts)),
    },
    "training_domain_cp150_window": {
        "rows": len(cp150),
        "answer_boundaries": int(cp150.has_boundary.sum()),
        "nonempty_answers": int(cp150.has_answer.sum()),
        "clipped": int(cp150.hit_4096.sum()),
        "reward_mean": float(cp150.grounded_analysis_reward_v3.mean()),
        "reward_median": float(cp150.grounded_analysis_reward_v3.median()),
        "best_rolling20_end_step": best_reward_end,
    },
    "sft_contract": {
        "rows": len(sft_rows),
        "reasoning_tokens_p50": float(np.median(sft_reasoning_tokens)),
        "reasoning_tokens_p95": float(np.quantile(sft_reasoning_tokens, 0.95)),
        "reasoning_tokens_max": int(max(sft_reasoning_tokens)),
        "completion_tokens_p50": float(np.median(sft_completion_tokens)),
        "completion_tokens_p95": float(np.quantile(sft_completion_tokens, 0.95)),
        "completion_tokens_max": int(max(sft_completion_tokens)),
        "plain_text_contract_violations": int(quality.contract_violation.sum()),
        "json_like_answers": int(quality.json_like.sum()),
        "evidence_identifier_answers": int(quality.has_evidence_identifier.sum()),
        "malformed_short_answers": int(quality.malformed_short_answer.sum()),
        "all_sequences_end_with_eos": bool(all(all_training_eos)),
        "max_full_training_tokens": int(max(all_training_lengths)),
        "double_bos_reproduced": current_train_ids[:2] == [tokenizer.bos_token_id, tokenizer.bos_token_id],
    },
    "evaluation_contract": {
        "old_task": "all evidence -> Minutes section",
        "intended_chk2_task": "atomic evidence -> concise analysis",
        "old_temperature": eval_config["generation"]["temperature"],
        "old_top_p": eval_config["generation"]["top_p"],
        "old_max_new_tokens": eval_config["generation"]["max_new_tokens"],
        "chk2_training_temperature": grpo_config["temperature"],
        "chk2_training_top_p": grpo_config["top_p"],
        "chk2_prompt_cap": grpo_config["max_prompt_length"],
    },
}

(OUT_DIR / "diagnostic_summary.json").write_text(json.dumps(summary_payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
prompt_contract.to_csv(OUT_DIR / "prompt_contract_comparison.csv", index=False)
stress_summary.to_csv(OUT_DIR / "stress_loop_summary.csv", index=False)
contract_quality.to_csv(OUT_DIR / "sft_contract_quality.csv", index=False)
by_step.reset_index().to_csv(OUT_DIR / "chk2_training_health_through_step150.csv", index=False)
lt_chk2_vs_chk1.to_csv(OUT_DIR / "length_tolerant_chk2_vs_chk1.csv", index=False)

summary_payload
"""
    ),
]


notebook = nbf.v4.new_notebook(
    cells=cells,
    metadata={
        "kernelspec": {
            "display_name": "Python (fomc_trainer)",
            "language": "python",
            "name": "fomc_trainer",
        },
        "language_info": {"name": "python", "version": "3.10"},
    },
)

nbf.write(notebook, NOTEBOOK_PATH)
print(NOTEBOOK_PATH)
