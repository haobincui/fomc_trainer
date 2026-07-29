# Main Input Data Contract

这份文档描述的是 `dataset/processed/train/` 下四条任务数据链路的目标数据设计。

- 面向对象是最终训练集，而不是中间产物
- 如果当前实际数据与本文不一致，以本文作为后续重构目标
- 带 reasoning 的 `response` 默认使用 Gemini/Gemma thought-channel 格式

标准格式：

```text
<|channel>thought
[reasoning]
<channel|>[final answer]
```

## analysis_sft

### Task

`analysis_sft` 是 section-level analysis SFT 数据。

训练目标：
- 让模型学会根据会议时间、指标数据、参考材料写出结构化经济分析
- target response 必须包含 `reasoning + analysis`

### Prompt Contract

`prompt` 应包含：
- FOMC meeting date
- topic / section context
- 可用于分析的指标与表格信息
- 明确的写作任务，要求生成最终经济分析正文

`provided_data` 应保留支撑分析所需的原始指标和上下文材料。

### Target Response Contract

`response` 必须包含两部分：
- `reasoning`: 模型内部分析过程，使用 Gemini thought-channel
- `analysis`: 最终经济分析正文

`analysis` 是最终输出，不是标签、摘要说明或元注释。

### Required Fields

- `prompt`
- `response`
- `provided_data`

### Output

目标输出目录：

```text
dataset/processed/train/analysis_sft
```

最小样例：

```json
{
  "prompt": "You are preparing briefing notes for the FOMC meeting on 2020-06-10. Analyze recent developments in labor markets and inflation using the provided material.",
  "response": "<|channel>thought\nI should first identify the major labor-market changes, then connect them to inflation pressure, and finally write a concise briefing note.\n<channel|>Labor market conditions deteriorated sharply ahead of the June 10, 2020 meeting, with payroll losses concentrated in contact-intensive sectors and unemployment rising rapidly. Inflation pressures remained subdued as weak demand and falling energy prices offset isolated supply disruptions.",
  "provided_data": "Indicators, tables, and reference excerpts used to support the analysis."
}
```

## analysis_grpo

### Task

`analysis_grpo` 是 section-level analysis GRPO 数据。

训练目标：
- 在 rollout + reward 机制下继续优化经济分析质量
- 提高分析的准确性、推理质量、结构合规性和贴合参考答案的能力

`analysis_grpo` 不是以 supervised target 为唯一目标的阶段。

### Prompt Contract

`prompt` 应与 `analysis_sft` 保持同类分析任务口径：
- meeting date
- topic / section context
- indicator data
- analysis generation instruction

`provided_data` 应继续保留 judge / reward 可使用的原始信息。

### Target Response Contract

`response` 可以保留为参考答案，用于：
- online reward
- answer quality judging
- reasoning quality judging

但 `analysis_grpo` 的核心训练目标是基于模型 rollout 和 reward 优化，而不是单纯拟合这条 `response`。

### Required Fields

- `prompt`
- `response`
- `provided_data`

### Output

目标输出目录：

```text
dataset/processed/train/analysis_grpo
```

最小样例：

```json
{
  "prompt": "You are preparing briefing notes for the FOMC meeting on 2017-05-03. Analyze recent developments in personal consumption expenditures using the provided material.",
  "response": "<|channel>thought\nI should compare recent consumption growth with inflation and labor-market support, then produce a concise policy-style analysis.\n<channel|>Personal consumption expenditures continued to expand at a moderate pace ahead of the May 3, 2017 meeting, supported by firm labor income and household balance sheets, though monthly readings remained uneven across categories.",
  "provided_data": "Indicators, tables, and reference excerpts used for reward and evaluation."
}
```

## minutes_alignment

### Task

`minutes_alignment` 是 rewrite SFT 数据。

训练目标：
- 让模型学会围绕已有分析材料进行重写
- target response 必须包含 `reasoning + reference excerpt`

这里的 `reference` 指原始参考摘录 / `reference excerpt`，不是最终 rewrite 段落。

### Prompt Contract

`prompt` 应包含：
- meeting date
- target section name
- 待处理的原始分析文本
- 明确要求模型围绕给定参考材料完成改写相关任务

`provided_data` 可为空，也可保留额外上下文。

### Target Response Contract

`response` 必须包含两部分：
- `reasoning`: 模型如何理解原始分析、目标 section 和改写约束
- `reference excerpt`: 原始参考摘录本身

本文档把 `minutes_alignment` 的目标定义为 `reasoning + reference excerpt`，用于后续统一数据设计。

### Intermediate Build Flow

`minutes_alignment` 的中间链路应为：

```text
analysis_sft teacher response
-> rewrite prompt
-> teacher model
-> teacher rewrite reasoning + rewritten paragraph
-> final target = teacher rewrite reasoning + reference excerpt
```

其中：
- rewrite teacher response 需要先落盘到 `dataset/processed/pipeline/minutes_alignment/teacher_responses/`
- 最终训练集 `response` 仍然是 `reasoning + reference excerpt`
- teacher model 生成的 rewritten paragraph 只作为中间产物和 manifest 字段保留，不直接进入训练 jsonl

### Required Fields

- `prompt`
- `response`
- `provided_data`

### Output

目标输出目录：

```text
dataset/processed/train/minutes_alignment
```

最小样例：

```json
{
  "prompt": "You are drafting material for the FOMC minutes. Use the raw analysis below for the discussion of inflation developments and align the rewrite to the target section.",
  "response": "<|channel>thought\nI should identify the key inflation signals in the raw analysis, preserve the factual content, and surface the relevant source excerpt that anchors the rewrite.\n<channel|>Core inflation readings remained elevated over the intermeeting period, while energy-related price declines provided only partial offset in headline measures.",
  "provided_data": ""
}
```

## decision

### Task

`decision` 覆盖：
- `decision_sft`
- `decision_grpo`

训练目标：
- 让模型在给定政策背景、当前利率和分析材料时形成政策决策
- 训练数据本身必须包含 `target rate change`

### Prompt Contract

`prompt` 应包含：
- meeting date
- current target rate / current policy stance
- analysis text
- candidate policy options
- 明确的 FOMC decision task

### Target Response Contract

`response` 应表达最终政策判断。

对于带 reasoning 的样本，使用 Gemini thought-channel：
- `reasoning`: 模型如何权衡通胀、就业、金融条件和政策选项
- final answer: 最终政策决策说明

此外，canonical training sample 必须显式包含：
- `target rate change`
- repo field name: `rate_change`

这个字段属于训练样本本身，不只是 manifest 附带信息。

### Required Fields

- `prompt`
- `response`
- `provided_data`
- `rate_change`

### Output

目标输出目录：

```text
dataset/processed/train/decision_sft
dataset/processed/train/decision_grpo
```

最小样例：

```json
{
  "prompt": "You are a voting member of the FOMC. The current target range is 5.25% to 5.50%. Review the analysis and choose the appropriate policy action.",
  "response": "<|channel>thought\nI should weigh inflation persistence against labor-market cooling and financial conditions, then map that assessment to one of the allowed policy actions.\n<channel|>Given still-elevated inflation and resilient activity, the appropriate policy choice is to leave the target range unchanged at this meeting while maintaining a restrictive stance.",
  "provided_data": "",
  "rate_change": "No change"
}
```

## Summary

- `analysis_sft = reasoning + analysis`
- `analysis_grpo = rollout + reward optimization with reference response retained`
- `minutes_alignment = reasoning + reference excerpt`
- `decision = policy decision data with target rate change (repo field: rate_change)`


## generation process

## raw data

1. labeled with indicator: dataset/raw_data/labeled_text/merged_labeled_after_2009.xlsx
[line_id,section_name,raw_text, label, label_type, explanation, reason, response, file_name, date, new_label, relabel]
2. prompt templates: src/process_fomc_report/generate_prompt_and_response/templates

analysis_sft: 
    teacher: [src/process_fomc_report/generate_prompt_and_response/templates/analysis_teacher.md]
    student: [src/process_fomc_report/generate_prompt_and_response/templates/analysis_student.md]

analysis_grpo:
    [src/process_fomc_report/generate_prompt_and_response/templates/analysis_grpo.md]

minutes_alignment:
    [src/process_fomc_report/generate_prompt_and_response/templates/minutes_rewrite.md]

decision:
    sft: [src/process_fomc_report/generate_prompt_and_response/templates/decision_sft.md]
    grpo: [src/process_fomc_report/generate_prompt_and_response/templates/decision_grpo.md]

3. indicators:
   [dataset/raw_data/input_data]
   loader: [src/process_fomc_report/generate_prompt_and_response/algo/common/indicators.py]


## process

### analysis_sft (section level):
labeled indicator xlsx[relabel] + [section_name] + [line_id] + [raw_text] + indicator -(teacher template)-> teacher prompt -> teacher model -> get reasoning + analysis (teacher response)

- teacher response jsonl: prompt + teacher response(reasoning + analysis)

labeled indicator xlsx[relabel] + [section_name] + [line_id] + indicator -(student template) -> student prompt

- analysis_sft train jsonl: student prompt + teacher response

### analysis_grpo (section level):
- analysis_grpo train jsonl: student prompt + teacher response

### minutes_alignment (section level):
teacher response -(rewrite template)-> rewrite prompt -> teacher model -> get reasoning (teacher rewrite reasoning) + response 

- minutes_alignment train jsonl: prompt (teacher response + rewrite template) + response (teacher rewrite reasoning + [raw_text])

### decision (minutes level):

 [raw_text] + [meeting_date] -> [raw_minutes] + [rate_change] -decision template-> decision prompt

 

