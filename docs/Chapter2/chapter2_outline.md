# Chapter 2 Outline — Synthetic Texts Generation with a Fine-tuned Large Language Model

- **Introduction** 
  - **Research Motivation** — Explains why the scarcity of FOMC text data drives the need for synthetic generation.
  - **Research Question** — States the formal research question the chapter addresses.
  - **Contribution** — Summarises the methodological and empirical contributions of the chapter.

<br>

- **Literature Review** — Positions the chapter against related work on synthetic data and large language models.
  - **Related Works on Synthetic Data** — Reviews prior approaches to synthetic text and tabular data generation.
  - **Large Language Models (LLMs) for Structured Text Generation** — Surveys LLM techniques relevant to structured financial-text generation.
  - **Consistent Texts** — Reviews methods for keeping generated text coherent with the underlying conditioning signals.

<br>

- **2-Step Training Plan** — Outlines the overall supervised-fine-tuning plus reinforcement-learning training pipeline used in the chapter.

<br>

- **Dataset Construction** — Describes how the FOMC-plus-indicator training dataset is built.
  - **FOMC Minutes and Indicators** — Introduces the raw FOMC minutes and macro indicators used as the base data.
  - **From FOMC Minutes to Analysis** — Pipeline that transforms minutes into analysis-style question–answer records.
  - **Economic and Financial Market Indicators** — The set of economic and market indicators joined to each document.
  - **Constructing Reasoning Process** — Procedure for constructing chain-of-thought reasoning traces for training.
  - **Synthetic Text Generation** — Scales up the generation of synthetic training samples.
  - **Dataset Allocation** — Specifies the train / validation / test split and how each partition is used.

<br>

- **The Base Model** — Discusses the choice of backbone language models for fine-tuning.
  - **Base Model: LLaMA 3-8B-Instruct** — Justifies LLaMA 3-8B-Instruct as the base model for supervised fine-tuning.
  - **From Chat Model to Reasoning Model: DeepSeek-R1-Distill-Llama-8B** — Motivates moving to a reasoning-oriented backbone for the RL stage.

<br>

- **Model Training Details** — Specifies the training recipe underlying the 2-step plan.
  - **Model Input** — Defines the input schema, prompt templates, and tokenisation used during training.
  - **Incorporate Reasoning Process** — Describes how reasoning traces are folded into the training signal.
  - **Supervised Fine-Tuning** — Details the SFT objective, data handling, and optimisation flow.
  - **Reinforcement Learning** — Introduces the RL stage built on Group Relative Policy Optimisation (GRPO).
  - **Reward Function Design** — Specifies the structure of the reward used during RL training.
    - **Format Reward** — Rewards outputs that follow the required structural format.
    - **Accuracy Reward** — Rewards correct final answers and policy decisions.
    - **Reasoning Reward** — Rewards the quality of the generated reasoning trace.
  - **Technique for Parameter-Efficient Fine-Tuning (PEFT)** — Describes the LoRA-style PEFT configuration used to fine-tune the backbone efficiently.
  - **Training Hyperparameters** — Lists the hyperparameters used for both SFT and RL stages.

<br>

- **Model Evaluation Methods** — Defines the evaluation framework used to judge synthetic-text quality.
  - **Evaluation Methods** — Umbrella subsection describing the overall evaluation protocol.
    - **Textual Similarity** — N-gram and embedding-based similarity metrics for comparing synthetic and reference texts.
    - **Sufficient Information** — Tests whether synthetic text preserves decision-relevant information from the source.
    - **Statistical Significance using Synthetic Texts** — Runs statistical tests on outputs derived from synthetic text.
    - **Decision-making Accuracy** — Measures how well downstream monetary-policy decisions can be predicted from synthetic text.

<br>

- **Empirical Results** — Reports the training and evaluation results of the two-step pipeline.
  - **Model Training Results** — Convergence behaviour and training diagnostics.
    - **Step 1: Supervised Fine-Tuning (SFT) Results** — Outcomes of the supervised fine-tuning stage.
    - **Step 2: Group Relative Policy Optimization (GRPO) Results** — Outcomes of the GRPO-based reinforcement-learning stage.
    - **Training Results Summary** — Consolidated summary of the two training stages.
  - **Model Evaluation Results** — Reports results under the evaluation framework defined above.
    - **Textual Similarity Performance** — Results on the textual-similarity metrics.
    - **Sufficient Information Results** — Results on the information-sufficiency tests.
    - **Empirical Models Test Results** — Results of econometric tests that use the synthetic texts as inputs.
    - **Decision Prediction Performance** — Results on downstream policy-decision prediction.

<br>

- **Conclusion** 
