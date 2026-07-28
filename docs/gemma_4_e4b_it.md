# Gemma 4 E4B IT 在本项目中的使用说明

本文档说明当前仓库中 `Gemma 4 E4B IT` 的定位、配置和使用方式。它参考
`docs/report/report.tex` 的学术化组织方式：先说明研究任务和模型定位，再介绍
base model，最后说明训练链路、checkpoint 关系和使用边界。本文是当前实现说明，
不是论文 LaTeX 章节替换稿，也不改写 `docs/report/report.tex` 中已经归档的历史实验叙述。

## 1. 背景与模型定位

本项目的核心任务是基于 FOMC Minutes 与宏观金融指标，训练和评估能够生成结构化
经济金融分析、FOMC 风格文本以及政策决策判断的语言模型。当前实现中使用的本地
backbone 是 `models/gemma-4-E4B-it`，正式名称应写作 `Gemma 4 E4B IT`。

`Gemma 4 E4B IT` 是 Google DeepMind Gemma 4 开放模型系列中的 instruction-tuned
版本。它被用作当前配置层面的 reasoning-capable backbone：模型先通过 SFT 适配
宏观金融分析任务，再通过后续 SFT/GRPO 分支服务于分析增强、Minutes 风格改写和
决策预测。为避免命名混淆，本文档统一使用 `Gemma`，不要将其写作或理解为
Google 的 `Gemini` API 模型。

## 2. 模型架构与能力

根据本地 `models/gemma-4-E4B-it/config.json` 和官方模型卡，当前模型具有以下
与项目相关的技术特征：

| 项目 | 说明 |
| --- | --- |
| 模型路径 | `models/gemma-4-E4B-it` |
| 架构 | `Gemma4ForConditionalGeneration` |
| 模型类型 | Gemma 4 dense E4B instruction-tuned model |
| 参数规模 | 约 4.5B effective parameters，约 8B total parameters with embeddings |
| 文本层数 | 42 layers |
| hidden size | 2560 |
| context length | 131072 tokens，即约 128K tokens |
| sliding window | 512 tokens |
| vocabulary size | 262144 |
| dtype | `bfloat16` |
| 支持模态 | 官方模型支持 text、image、audio、video 输入，并生成文本输出 |

E4B 中的 `E` 表示 effective parameters。该设计使用 Per-Layer Embeddings 提高小模型
部署效率，因此 effective parameter count 小于包含 embedding 后的 total parameter count。
Gemma 4 的注意力结构结合 sliding attention 与 full attention，用于在较低内存开销下
支持长上下文处理。

虽然模型具备多模态能力，本项目当前主要把它作为文本推理和文本生成模型使用：
输入由 FOMC 会议日期、目标章节、宏观金融指标表格和任务说明组成，输出为经济金融
分析、Minutes 风格文本或政策动作判断。

## 3. 与 `report.tex` 的关系

`docs/report/report.tex` 是论文报告的历史叙述文件，其中 `The Base Model` 章节把
Backbone-0 记录为 `DeepSeek-R1-Distill-Llama-8B`。这反映的是已归档实验和报告中的
模型 lineage，不应被本文档自动覆盖。

本文档记录的是当前仓库配置层面的 Gemma 使用情况。也就是说：

- `report.tex` 中关于 DeepSeek backbone 的表述仍属于历史论文叙述。
- 本文档中的 `Gemma 4 E4B IT` 说明当前 `configs/main/` 下的实现状态。
- 如果后续要把论文正文从 DeepSeek 迁移到 Gemma，需要单独更新实验结果、checkpoint
  mapping、评估表格和结论边界，不能只替换模型名称。

## 4. 项目中的使用位置

当前仓库中与 `Gemma 4 E4B IT` 直接相关的入口主要包括：

| 文件 | 用途 |
| --- | --- |
| `configs/main/analysis_sft.yaml` | 将 `model_name_or_path` 设为 `models/gemma-4-E4B-it`，作为分析 SFT 的初始模型。 |
| `configs/main/judge_chk2.yaml` | 将 `model_path` 和 `model_name` 设为 `models/gemma-4-E4B-it`，用于本地 vLLM/OpenAI-compatible 服务配置。 |
| `model_test.py` | 使用 `AutoProcessor` 和 `AutoModelForCausalLM` 加载 `models/gemma-4-E4B-it`，并演示 `enable_thinking=True` 的生成流程。 |
| `docs/report/report.tex` | 提供论文式章节组织方式和历史实验上下文；本文档参考其结构，但不修改该文件。 |

`configs/main/analysis_sft.yaml` 中的关键配置包括：

- `torch_dtype: "bfloat16"`
- `attn_implementation: "sdpa"`
- `top_p: 0.9`
- `temperature: 0.6`
- `max_prompt_length: 16384`
- `max_completion_length: 8192`
- `per_device_train_batch_size: 2`
- `per_device_eval_batch_size: 4`
- `peft_merged_model_path: output/training/main/merged/analysis_sft`

`configs/main/judge_chk2.yaml` 中的服务侧配置包括：

- `host: "127.0.0.1"`
- `port: 8000`
- `gpu_ids: "0"`
- `tp_size: 1`
- `gpu_memory_utilization: 0.95`
- `max_model_len: 12288`
- `model_path: "models/gemma-4-E4B-it"`
- `model_name: "models/gemma-4-E4B-it"`

## 5. Thinking 输出格式

Gemma 4 使用原生 chat template 和 thought-channel 输出格式。当前配置通过 system prompt
中的 `<|think|>` 开启 thinking 行为，并明确要求：

```text
Use the model's native thought-channel output format and do not emit legacy XML wrapper tags.
```

因此，本项目当前实现不应再依赖旧式 `<think>...</think>` 或 `<answer>...</answer>`
XML wrapper 作为 Gemma 输出的主格式。Gemma 4 的 reasoning 输出由原生 thought-channel
承载，典型结构为：

```text
<|channel>thought
[internal reasoning]
<channel|>
[final answer]
```

在 `model_test.py` 中，推理流程通过 `processor.apply_chat_template(..., enable_thinking=True)`
开启 thinking，并用 `processor.parse_response(response)` 解析模型输出。训练和评估代码
应优先使用 tokenizer/processor 提供的原生解析能力，避免用手写字符串规则混合旧格式
和新格式。

## 6. 训练与适配设置

参考 `report.tex` 中 checkpoint mapping 的写法，当前 Gemma 配置可以理解为以下实现链路：

| Checkpoint | Parent | 主要用途 | 输出位置 |
| --- | --- | --- | --- |
| Backbone-0 | - | 当前实现的 reasoning-capable backbone | `models/gemma-4-E4B-it` |
| Analysis SFT | Backbone-0 | 将 Gemma 适配到宏观金融指标分析任务 | `output/training/main/merged/analysis_sft` |
| Analysis GRPO | Analysis SFT | 通过 reward 优化分析质量和推理一致性 | `output/training/main/merged/analysis_grpo` |
| Minutes Alignment SFT | Analysis SFT | 将分析内容改写为 FOMC Minutes 风格文本 | `output/training/main/merged/minutes_alignment_sft` |
| Decision SFT | Analysis SFT | 为政策动作判断任务建立监督适配分支 | `output/training/main/merged/decision_sft` |
| Decision GRPO | Decision SFT | 用 rate accuracy 和 rate format rewards 强化政策动作输出 | `output/training/main/merged/decision_grpo` |

当前 SFT 配置采用 PEFT/LoRA 风格的参数高效适配，核心设置为：

- `peft_r: 8`
- `peft_lora_alpha: 16`
- `peft_lora_dropout: 0.05`
- `peft_target_modules: q_proj, v_proj`
- `learning_rate: 1.0e-05`
- `lr_scheduler_type: cosine_with_min_lr`
- `warmup_ratio: 0.1`
- `num_train_epochs: 3`

GRPO 分支在当前配置中用于 reward-guided optimization。分析 GRPO 侧使用 format、answer
和 reasoning rewards；决策 GRPO 侧使用 rate accuracy 和 rate format rewards。由于 RL
训练对 reward 设计和 checkpoint selection 更敏感，相关结果必须与具体 checkpoint 和
评估输入绑定解读。

## 7. 使用边界与注意事项

`Gemma 4 E4B IT` 的长上下文和 thinking 能力适合本项目中的结构化政策文本生成任务，
但模型输出本身不等于经济事实或政策结论。尤其是在 FOMC 场景下，生成文本可能出现
以下风险：

- 对输入指标的解释不足或过度解释。
- 生成与历史政策语境不完全一致的表述。
- 在长上下文中遗漏局部指标或错误关联不同表格。
- 在决策预测任务中给出格式正确但经济依据不足的政策动作。

因此，所有金融和政策相关结论都应通过项目中的 held-out evaluation、leave-one-out
masking、text similarity、sentiment-return validation 和 decision baseline comparison
等实验进行验证。和 `report.tex` 的结论边界一致，随机 meeting-level holdout 的结果
应解释为 held-out meeting validation，而不是严格的 chronological forecasting evidence。

## 8. References

- Google AI for Developers: Gemma 4 model overview  
  <https://ai.google.dev/gemma/docs/core>
- Google AI for Developers: Gemma 4 model card  
  <https://ai.google.dev/gemma/docs/core/model_card_4>
- Hugging Face: `google/gemma-4-E4B-it`  
  <https://huggingface.co/google/gemma-4-E4B-it>
- 本项目历史报告：`docs/report/report.tex`
- 本项目当前模型配置：`configs/main/analysis_sft.yaml`、`configs/main/judge_chk2.yaml`
