# Chapter 2 Revision Notes

This file tracks all CH2 revisions with one-to-one mapping to TeX annotations.

## CH2-001
- ID: CH2-001
- File: Chapter2/sections/intro.tex
- Location: Chapter2/sections/intro.tex:3
- Issue Type: Wording
- Original: "Government announcements have been shown to exert significant influence on interest rate markets (\cite{vergote2012}; \cite{rosa2013financial}; \cite{jubinski2013fomc}). Among these, the Federal Open Market Committee (FOMC) plays a particularly critical role in shaping both the U.S. and global economic outlook throu..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "Government announcements materially influence interest-rate markets (\cite{vergote2012}; \cite{rosa2013financial}; \cite{jubinski2013fomc}). Within this communication channel, the Federal Open Market Committee (FOMC) is central because policy-rate decisions and accompanying narratives are rapidly incorporated into T..."

## CH2-002
- ID: CH2-002
- File: Chapter2/sections/intro.tex
- Location: Chapter2/sections/intro.tex:28
- Issue Type: Wording
- Original: "The motivation for this research arises from the critical importance of accurately forecasting monetary policy decisions and the well-recognized limitations of traditional econometric models in capturing qualitative and contextual signals. While quantitative forecasting techniques based on structured economic indica..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "This research is motivated by two facts. First, monetary-policy forecasting remains important for asset pricing and risk management. Second, conventional quantitative models do not fully exploit textual policy signals from minutes, statements, and staff discussions. As a result, they may miss informative cues about ..."

## CH2-003
- ID: CH2-003
- File: Chapter2/sections/intro.tex
- Location: Chapter2/sections/intro.tex:41
- Issue Type: Wording
- Original: "This study seeks to explore the potential of large language models (LLMs) as simulation tools for monetary policy communication and decision support. The primary research question guiding this investigation is: \textit{Can a fine-tuned reasoning-capable LLM generate FOMC-Minutes-style sections that are grounded in m..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "This chapter investigates whether large language models can be used as simulation tools for monetary-policy communication and decision support. The main research question is: \textit{Can a fine-tuned, reasoning-capable LLM generate FOMC-Minutes-style sections grounded in macroeconomic and financial indicators, and d..."

## CH2-004
- ID: CH2-004
- File: Chapter2/sections/intro.tex
- Location: Chapter2/sections/intro.tex:62
- Issue Type: Wording
- Original: "By systematically addressing these questions, this study aims to contribute to the emerging body of literature on the application of artificial intelligence and machine learning in macro-financial policy analysis and central banking decision-making."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "Addressing these questions contributes to the literature on AI-enabled macro-financial analysis and central-banking decision support. \subsection{Contribution}"

## CH2-005
- ID: CH2-005
- File: Chapter2/sections/intro.tex
- Location: Chapter2/sections/intro.tex:69
- Issue Type: Wording
- Original: "This study makes several important contributions to the literature at the intersection of central banking, financial economics, and artificial intelligence: \begin{itemize} \item[] \textbf{Methodological Innovation:} The research proposes a two-step post-training pipeline (SFT + GRPO) to adapt an LLM toward data-gro..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "This chapter contributes to the literature at the intersection of central banking, financial economics, and artificial intelligence in six ways: \begin{itemize} \item[] \textbf{Methodological framework:} It proposes a two-step post-training pipeline (SFT + GRPO) for data-grounded macro-financial reasoning, followed ..."

## CH2-006
- ID: CH2-006
- File: Chapter2/sections/intro.tex
- Location: Chapter2/sections/intro.tex:100
- Issue Type: Consistency
- Original: "Our chapter is organized as follows. Section~\ref{ch2:sec:lr} reviews related work on synthetic data and large language models. Section~\ref{ch2:sec:2-step-training} outlines the two-step post-training framework. Section~\ref{ch2:sec:the dataset} describes dataset construction and prompt templates. Section~\ref{ch2:..."
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: "The chapter is organized as follows. Section~\ref{ch2:sec:lr} reviews related work. Section~\ref{ch2:sec:2-step-training} introduces the two-step training design. Section~\ref{ch2:sec:the dataset} details dataset construction and prompts. Section~\ref{ch2:sec:base_model} presents the base model, while Sections~\ref{..."

## CH2-007
- ID: CH2-007
- File: Chapter2/sections/literature_review.tex
- Location: Chapter2/sections/literature_review.tex:5
- Issue Type: Wording
- Original: "Chapter~\ref{ch:review} reviews synthetic data generation and textual signals in finance (Section~\ref{ch0:sec:application:distribution} and Section~\ref{ch0:sec:application:text}). In this chapter, we focus on a specific form of synthetic text: structured monetary-policy communication in the style of the FOMC Minut..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "Chapter~\ref{ch:review} surveys synthetic data generation and text-based signals in finance (Section~\ref{ch0:sec:application:distribution} and Section~\ref{ch0:sec:application:text}). Building on that foundation, this chapter focuses on a narrower target: synthetic monetary-policy communication in the style of FOMC..."

## CH2-008
- ID: CH2-008
- File: Chapter2/sections/literature_review.tex
- Location: Chapter2/sections/literature_review.tex:20
- Issue Type: Wording
- Original: "Transformer-based LLMs are the dominant architecture for modern text generation and representation learning. We refer to Chapter~\ref{ch:review} (Section~\ref{sec: review: deep learning in finance: nlp}) for a detailed discussion of the Transformer, GPT/BERT-style model families, and fine-tuning mechanisms. For our ..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "Transformer-based LLMs are the dominant architecture for text generation and representation learning. Chapter~\ref{ch:review} (Section~\ref{sec: review: deep learning in finance: nlp}) reviews Transformer foundations, GPT/BERT model families, and fine-tuning strategies. In this chapter, we treat the LLM as a conditi..."

## CH2-009
- ID: CH2-009
- File: Chapter2/sections/literature_review.tex
- Location: Chapter2/sections/literature_review.tex:27
- Issue Type: Wording
- Original: "Large language models enable fluent, human-like text generation, but ensuring \textit{consistency} remains challenging, especially for long-form and structured documents. In this chapter, consistency has two closely related meanings: (i) internal coherence across paragraphs and sections, and (ii) \textit{faithfulnes..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "LLMs can generate fluent text, but consistency remains difficult in long, structured documents. In this chapter, consistency has two dimensions: (i) internal coherence across paragraphs and sections, and (ii) faithfulness to conditioning inputs (macroeconomic and financial indicators). A common failure mode is hallu..."

## CH2-010
- ID: CH2-010
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:7
- Issue Type: Wording
- Original: "In this section, we demonstrate the process of constructing the training dataset. The raw textual data are derived from the Federal Open Market Committee (FOMC) meeting minutes, complemented by corresponding macroeconomic and financial market indicators. These data are then systematically processed and transformed i..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "This section describes the construction of the training corpus. The raw text comes from Federal Open Market Committee (FOMC) Minutes and is paired with macroeconomic and financial indicators. We then process these sources into structured samples for model training and evaluation. \subsection{FOMC Minutes and Indicat..."

## CH2-011
- ID: CH2-011
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:16
- Issue Type: Wording
- Original: "The Federal Open Market Committee (FOMC) meetings have been recorded since the committee's formation under the Banking Act of 1935 (\cite{todd2016corollary}). Initially, from 1936 to 1967, the minutes were released only once a year in these early years. Between 1967 and 1992, the minutes were named as ``the Minutes ..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "FOMC meetings have been documented since the Committee's establishment under the Banking Act of 1935 (\cite{todd2016corollary}). Publication practices evolved over time: early records were less frequent and less standardized, while post-2009 Minutes adopted a relatively stable section structure. Because section-leve..."

## CH2-012
- ID: CH2-012
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:27
- Issue Type: Logic
- Original: "First, we downloaded FOMC Minutes spanning January 2009 to January 2025 from the Federal Reserve website and treated them as our \textbf{main text corpus}. The post-2009 Minutes follow a largely stable section structure; we therefore parsed the major sections (e.g., staff reviews, staff outlook, participants' views,..."
- Revision Rationale: Resolved narrative inconsistency and aligned the text with tables, sample definitions, or evaluation logic without changing empirical results.
- Revised Version: "We downloaded Minutes from January 2009 to January 2025 from the Federal Reserve website as the \textbf{main text corpus}. Given the stable section structure after 2009, we parsed major sections (staff reviews, staff outlook, participants' views, and policy actions) and obtained 486 section-level samples. We also co..."

## CH2-013
- ID: CH2-013
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:34
- Issue Type: Wording
- Original: "Notably, the term ``word'' in this context does not strictly refer to whole words. Instead, LLMs operate on ``sub-words'' or ``tokens'', which are components of words. For instance, ``Negative'' might be tokenized as ``Neg'' and ``agtive'' to reduce the size of the dictionary, which is known as token dictionary or t..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "In this chapter, ``word'' does not necessarily mean a full lexical word. LLMs operate on tokens (often sub-word units). For example, ``Negative'' may be split into smaller units such as ``Neg'' and ``ative''. This tokenization is controlled by the model-specific tokenizer. Larger models often use larger vocabularies..."

## CH2-014
- ID: CH2-014
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:62
- Issue Type: Wording
- Original: "As the maximum length for the major sections are less than 4096 tokens (except the ``Committee Policy Action''), we conduct a cutoff on the text and set the maximum length of the ``Assistant Prompt'' as 4096 to reduce the memory usage."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "Because most major sections are shorter than 4,096 tokens (except ``Committee Policy Action''), we cap the assistant output length at 4,096 tokens to control memory usage. % Furthermore, the dataset size may not be big enough to conduct fine-tuning. Therefore, we downloaded extra texts as supplementary training mate..."

## CH2-015
- ID: CH2-015
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:129
- Issue Type: Wording
- Original: "To construct the list of target indicators, we first asked the LLM to label all paragraphs without providing a predefined reference list. The model assigned labels based on its pre-trained knowledge, these are referred to as raw labels. However, several issues were observed with these raw labels, requiring further c..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "To construct candidate indicator labels, we first asked the LLM to label paragraphs without a predefined reference list. These model-generated tags are treated as \textit{raw labels}. We then identified several quality issues that required cleaning and standardization. First, some of the raw labels were too general ..."

## CH2-016
- ID: CH2-016
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:141
- Issue Type: Wording
- Original: "Finally, some labels represented subcomponents of broader measures. For example, ``Total Assets of the Federal Reserve'' and ``Total Liabilities of the Federal Reserve'' are both subset of the ``Federal Reserve Balance Sheet''."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "Finally, some labels represented subcomponents of broader constructs. For example, ``Total Assets of the Federal Reserve'' and ``Total Liabilities of the Federal Reserve'' were consolidated under ``Federal Reserve Balance Sheet''. Therefore, we manually cleaned the \textit{raw labels} and created a list of \textit{r..."

## CH2-017
- ID: CH2-017
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:157
- Issue Type: Wording
- Original: "Next, we instructed the LLM to assign the label ``Non-Core'' to paragraphs that contain no analysis, and ``Other'' when the model was unable to identify a relevant indicator. Since some paragraphs refer to multiple indicators, the model was allowed to extract all applicable indicators and separate them using commas ..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "Next, we introduced two fallback tags: ``Non-Core'' for paragraphs without analytical content, and ``Other'' when no relevant indicator could be mapped. Because some paragraphs refer to multiple indicators, the model was allowed to assign multiple labels separated by commas. This stage produced 5,290 labeled analyti..."

## CH2-018
- ID: CH2-018
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:163
- Issue Type: Wording
- Original: "Subsequently, we manually reviewed the \textit{raw labels}, standardizing and clustering them into 26 new categories to form our final list of \textit{reference labels}. We then asked the model to relabel all 6,266 paragraphs using these \textit{reference labels}, assigning ``Other'' if no relevant indicator could b..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "We then manually standardized the raw labels into 26 \textit{reference-label} categories and relabeled all 6,266 filtered paragraphs using this controlled taxonomy. ``Other'' was assigned when no indicator matched, and ``Non-Core'' when the paragraph contained no analytical content."

## CH2-019
- ID: CH2-019
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:170
- Issue Type: Logic
- Original: "Finally, we extracted 4,889 analytical paragraphs with a total of 5,781 corresponding standardized labels for our \textbf{main dataset}. We applied the same cleaning process to our \textbf{supplementary dataset}, resulting in 4,464 analytical paragraphs associated with 44,178 standardized labels. Table~\ref{tab:ch2:..."
- Revision Rationale: Resolved narrative inconsistency and aligned the text with tables, sample definitions, or evaluation logic without changing empirical results.
- Revised Version: "The final \textbf{main dataset} contains 4,889 analytical paragraphs with 5,781 standardized labels. Applying the same procedure to the \textbf{supplementary dataset} yields 4,464 analytical paragraphs with 44,178 labels. Table~\ref{tab:ch2:indicator_count} reports the label distribution. The supplementary set has m..."

## CH2-020
- ID: CH2-020
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:233
- Issue Type: Wording
- Original: "After processing the FOMC Minutes, we extracted relevant numerical data to construct the input prompts for our model. Based on our reference labels, we obtained both financial data from WRDS and economic data from the FRED Economic Database."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "After processing the Minutes, we extracted numerical series to build model prompts. Guided by the reference-label taxonomy, we collected financial series from WRDS and macroeconomic series from the FRED database."

## CH2-021
- ID: CH2-021
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:239
- Issue Type: Consistency
- Original: "For economic analysis, we collected U.S. macroeconomic data, including GDP growth, Consumer Price Indices (CPIs), the federal funds rate, unemployment rates, export and import figures, government bond yields, and other key indicators. For financial analysis, we included data such as equity market indices, commodity ..."
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: "For macroeconomic coverage, we include GDP growth, Consumer Price Index measures, Federal Funds Rate series, unemployment measures, trade indicators, and Treasury yields. For financial coverage, we include equity indices, commodity prices, volatility indices, Federal Reserve balance-sheet measures, overnight-rate se..."

## CH2-022
- ID: CH2-022
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:245
- Issue Type: Wording
- Original: "Our dataset covers a broad range of economic and financial dimensions relevant to monetary policy decision-making. In total, we gathered 102 indicators across 26 categories. Table~\ref{tab:ch2:extracted_data} lists all the indicators along with their corresponding data sources used in the analysis."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "Overall, the dataset spans 102 indicators in 26 categories that are relevant for monetary-policy analysis. Table~\ref{tab:ch2:extracted_data} lists the indicators used in this chapter. \begin{center} \begin{singlespace} \begin{scriptsize} \begin{longtable}{p{0.33\textwidth}|p{0.66\textwidth}} \toprule \textbf{Catego..."

## CH2-023
- ID: CH2-023
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:472
- Issue Type: Logic
- Original: "Our next step is to establish a logical link between the analysis found in the FOMC Minutes and the corresponding numerical data. The FOMC Minutes provide high-level summaries intended to support policy decisions, often lacking detailed and explicit reasoning processes. As a result, large language models (LLMs) cann..."
- Revision Rationale: Resolved narrative inconsistency and aligned the text with tables, sample definitions, or evaluation logic without changing empirical results.
- Revised Version: "The next step is to link numerical indicators to the analytical narratives in FOMC Minutes. Minutes typically provide high-level conclusions rather than explicit reasoning chains; therefore, direct mapping from raw data to final narrative is under-specified. We address this by distilling intermediate reasoning traces."

## CH2-024
- ID: CH2-024
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:477
- Issue Type: Wording
- Original: "Model distillation (\cite{hinton2015distilling}) involves training a smaller, less powerful model using data generated by a larger, more capable model. Rather than exposing the smaller model to a wide range of general data, distillation provides focused, high-quality data that encapsulates expert-level reasoning. Fo..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "Model distillation (\cite{hinton2015distilling}) trains a smaller model using outputs from a stronger teacher model. Instead of broad general-domain supervision, distillation supplies focused, task-relevant traces that encode higher-quality reasoning patterns."

## CH2-025
- ID: CH2-025
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:483
- Issue Type: Wording
- Original: "More specifically, we applied the ``DeepSeek-R1'' as our teacher model to generate the reasoning process. Table~\ref{tab:ch2:prompt_for_data_distillation} demonstrates the prompt we used for data distillation. We used the \texttt{meeting date}, \texttt{section name}, \texttt{topic} of the indicators, \texttt{table} ..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "In implementation, we use DeepSeek-R1 as the teacher model for reasoning-trace generation. As shown in Table~\ref{tab:ch2:prompt_for_data_distillation}, each prompt includes the meeting date, target section style, indicator topic, indicator tables, and a reference excerpt from FOMC Minutes. The excerpt anchors style..."

## CH2-026
- ID: CH2-026
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:496
- Issue Type: Wording
- Original: "You are an economist preparing briefing notes for the Federal Open Market Committee (FOMC) meeting on \texttt{**\{meeting\_date\}**}. Your task is to analyze recent trends in \texttt{**\{topic\}**} and related indicators, writing in the tone and style of the \texttt{**\{section\_style\}**} section of the FOMC minute..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "You are an economist preparing briefing notes for the Federal Open Market Committee (FOMC) meeting on \texttt{**\{meeting\_date\}**}. Your task is to analyze recent developments in \texttt{**\{topic\}**} and related indicators, using the tone and structure of the \texttt{**\{section\_style\}**} section of the FOMC M..."

## CH2-027
- ID: CH2-027
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:525
- Issue Type: Wording
- Original: "Finally, we collected 4,889 messages of ``Question'', ``Reasoning'', and ``Answer'' pairs as our dataset for fine-tuning."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "Finally, this procedure produced 4,889 ``Question--Reasoning--Answer'' training samples for fine-tuning. \subsection{Synthetic Text Generation}"

## CH2-028
- ID: CH2-028
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:535
- Issue Type: Wording
- Original: "To assess the simulation capabilities of large language models (LLMs) in structured text, we conducted additional fine-tuning, explicitly instructing the model to produce analyses aligned with the style and structure of actual FOMC Minutes."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "To evaluate structured-text simulation, we add a downstream alignment stage that trains the model to rewrite analysis into authentic FOMC-Minutes style. % %%%%%%%%%% % Plan: % Data -> analysis -> summary -> FOMC sections % individual indicator data -> individual analysis -> combined analysis -> summary -> FOMC section"

## CH2-029
- ID: CH2-029
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:549
- Issue Type: Logic
- Original: "To achieve this, we designed a pipeline that accepts individual indicator data as input and produces structured text aligned with sections from actual FOMC Minutes as output. The process comprises three main steps: analysis generation, summary creation, and section rewriting. First, the model analyzes individual eco..."
- Revision Rationale: Resolved narrative inconsistency and aligned the text with tables, sample definitions, or evaluation logic without changing empirical results.
- Revised Version: "The pipeline accepts indicator data and outputs section-aligned Minutes text in three steps: (i) indicator-level analysis generation, (ii) consolidation into a concise analytical summary, and (iii) section-specific rewriting into institutional FOMC style."

## CH2-030
- ID: CH2-030
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:555
- Issue Type: Logic
- Original: "Since the model had already demonstrated robust proficiency in economic and financial analysis from last stage, the primary objective at this stage was to teach it to generate structured text precisely aligned with specific formatting guidelines."
- Revision Rationale: Resolved narrative inconsistency and aligned the text with tables, sample definitions, or evaluation logic without changing empirical results.
- Revised Version: "Because the model already demonstrates baseline analytical capability from the prior stage, the objective here is style and structure alignment rather than new domain learning."

## CH2-031
- ID: CH2-031
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:562
- Issue Type: Consistency
- Original: "To facilitate this, we reorganized the FOMC Minutes data according to their respective section titles, using the model-generated summaries as inputs and the actual text from corresponding sections of the Minutes as target outputs. The model was then instructed to rewrite the provided analytical summaries to match th..."
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: "Accordingly, we reorganize Minutes data by section title, use model-generated summaries as inputs, and use the corresponding official section text as targets. The model is trained to rewrite summaries into section-consistent FOMC language. Table~\ref{tab:ch2:fomc_summary_prompt} shows the prompt template. \begin{tab..."

## CH2-032
- ID: CH2-032
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:616
- Issue Type: Logic
- Original: "After that, we need to construct the reasoning process ($c_i$) in the model's target response. As we have the target answer ($a_i$), here we applied a standard model distillation pipeline. Firstly, we use the prompt in Table~\ref{tab:ch2:fomc_summary_prompt} in prompting a powerful reasoning model (we used DeepSeek-..."
- Revision Rationale: Resolved narrative inconsistency and aligned the text with tables, sample definitions, or evaluation logic without changing empirical results.
- Revised Version: "We then construct the reasoning component ($c_i$) for each target response. Given target answer text ($a_i$), we query a stronger reasoning model (DeepSeek-R1) with the template in Table~\ref{tab:ch2:fomc_summary_prompt}. For each prompt, we generate five candidates with high-sampling settings (temperature = 0.6, to..."

## CH2-033
- ID: CH2-033
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:652
- Issue Type: Logic
- Original: "The dataset is relative small for model fine-tuning. Therefore, we used the Minutes before 2009 to boost our data. However, these Minutes have no clear section titles. We need create the section name for each paragraph and then construct the section. We used the section names from Table~\ref{tab:ch2:section_name_dis..."
- Revision Rationale: Resolved narrative inconsistency and aligned the text with tables, sample definitions, or evaluation logic without changing empirical results.
- Revised Version: "The post-2009 section-aligned sample is relatively small for fine-tuning. To expand coverage, we incorporate pre-2009 Minutes, which do not provide stable section headers. We therefore infer section labels paragraph by paragraph using the section taxonomy in Table~\ref{tab:ch2:section_name_distribution} and the prom..."

## CH2-034
- ID: CH2-034
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:665
- Issue Type: Wording
- Original: "You are a policy editor at the Federal Reserve tasked with organizing the content of the **FOMC Minutes**. \\"
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "You are a policy editor at the Federal Reserve responsible for organizing **FOMC Minutes** content. \\ Given a paragraph from the FOMC Minutes, assign it to the **single most appropriate section title** from the official list below: \\ **Available Section Titles**: \\ \texttt{**\{Section Names\}**} --- \\ ~\\ \#\#\#..."

## CH2-035
- ID: CH2-035
- File: Chapter2/sections/dataset_construction.tex
- Location: Chapter2/sections/dataset_construction.tex:802
- Issue Type: Logic
- Original: "Finally, we divided the entire dataset into three parts: Training data, Evaluation data, and Test data. The full dataset comprises 4,889 question-answer (Q\&A) pairs, covering the FOMC Minutes from Jan 2009 to Jan 2025. This dataset was initially divided into 80\% for training, 10\% for evaluation, and 10\% for test..."
- Revision Rationale: Resolved narrative inconsistency and aligned the text with tables, sample definitions, or evaluation logic without changing empirical results.
- Revised Version: "Finally, we split the dataset into training, evaluation, and test subsets. The full set contains 4,889 question--answer samples (January 2009 to January 2025), partitioned as 80\% training, 10\% evaluation, and 10\% test. Within training, 80\% is used for SFT and 20\% for GRPO. Because GRPO is more computationally e..."

## CH2-036
- ID: CH2-036
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:7
- Issue Type: Wording
- Original: "To enhance the model’s capabilities in economic and financial analysis, we employ a two-step post-training approach that combines Supervised Fine-Tuning (SFT) with Reinforcement Learning (RL) via Group Relative Policy Optimization (GRPO)."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "To improve macro-financial analytical performance, we use a two-step post-training pipeline: Supervised Fine-Tuning (SFT) followed by Reinforcement Learning (RL) with Group Relative Policy Optimization (GRPO)."

## CH2-037
- ID: CH2-037
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:12
- Issue Type: Consistency
- Original: "The initial stage, SFT, leverages a high-quality Question-Answer (Q\&A) dataset to train the model to generate outputs that closely align with reference answers. Given that the model is pre-trained on general-purpose textual data, SFT serves to guide the model in structuring its analysis based on economic and financ..."
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: "In Step 1, SFT uses high-quality question--answer data to align model outputs with reference analyses. Because the base model is pre-trained on general-domain corpora, SFT provides domain adaptation toward structured economic and financial reasoning. SFT also offers stable optimization through a well-defined supervi..."

## CH2-038
- ID: CH2-038
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:20
- Issue Type: Consistency
- Original: "In Step 2, we apply GRPO to further fine-tune the model via reinforcement learning. GRPO is an on-policy RL method that allows the model to optimize against reward signals without requiring token-level reference answers. Unlike SFT, RL supports flexible reward structures (e.g., LLM-as-Judge scoring; \cite{zheng2024j..."
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: "In Step 2, GRPO further refines reasoning behavior through reward optimization. Unlike SFT, RL does not require token-level references and can directly optimize policy behavior under task-specific reward signals (e.g., LLM-as-Judge scores; \cite{zheng2024judging}). We use GRPO (\cite{shao2024deepseekmath}) to reduce..."

## CH2-039
- ID: CH2-039
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:26
- Issue Type: Logic
- Original: "Following this training, the model is expected to demonstrate strong capabilities in conducting detailed economic and financial analysis, forecasting, and making monetary policy recommendations based on structured input data."
- Revision Rationale: Resolved narrative inconsistency and aligned the text with tables, sample definitions, or evaluation logic without changing empirical results.
- Revised Version: "After this two-step training, the model is expected to produce stronger data-grounded macro-financial analysis and more coherent policy-relevant reasoning from structured inputs. % To evaluate its effectiveness, we applied the model to two downstream tasks: (1) synthetic FOMC Minutes generation and (2) Federal Funds..."

## CH2-040
- ID: CH2-040
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:39
- Issue Type: Logic
- Original: "In summary, our overall training framework consists of two steps. \textit{Step 1} applies \textbf{Supervised Fine-Tuning (SFT)} to enhance domain-specific understanding and structured analysis generation. \textit{Step 2} applies \textbf{Group Relative Policy Optimization (GRPO)} to strengthen reasoning behavior thro..."
- Revision Rationale: Resolved narrative inconsistency and aligned the text with tables, sample definitions, or evaluation logic without changing empirical results.
- Revised Version: "In summary, Step 1 (SFT) performs domain adaptation for structured analysis, and Step 2 (GRPO) improves reasoning quality through reward-guided optimization. Figure~\ref{fig:ch2:training_process} summarizes the workflow. \begin{figure} \centering \includegraphics[width=\textwidth]{Chapter2/Chapter2Figs/training_proc..."

## CH2-041
- ID: CH2-041
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:69
- Issue Type: Wording
- Original: "To start with, we reformat our dataset into the specific templates for our SFT and GRPO."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "We first reformat the dataset into task-specific templates for SFT and GRPO."

## CH2-042
- ID: CH2-042
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:74
- Issue Type: Grammar
- Original: "For SFT, each input consists of three parts (denoted as $input^{SFT}_i = (q_i, o^{SFT}_i)$, $o^{SFT}_i = (c_i, a_i)$): Question ($q_i$), Reasoning ($c_i$), and Answer ($a_i$). Firstly, the reasoning process and the finally answer will be combined as the model's target output. Then, the input will be reformatted into..."
- Revision Rationale: Corrected grammar, tense, article usage, and sentence structure to meet formal academic writing standards.
- Revised Version: "For SFT, each sample is represented as $input_i^{SFT}=(q_i,o_i^{SFT})$, where $o_i^{SFT}=(c_i,a_i)$ contains a reasoning trace ($c_i$) and a final answer ($a_i$). The model predicts $o_i^*=(c_i^*,a_i^*)$ from question $q_i$, and training minimizes supervised loss $\mathcal{L}^{SFT}$ between predicted and target outp..."

## CH2-043
- ID: CH2-043
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:80
- Issue Type: Wording
- Original: "For RL training, we only use Questions ($input^{RL}_i = q_i$) as our input. The model will generate the output ($o^{RL}_i = (c_i, a_i)$) by optimizing a given reward function ($\mathcal{R}$)"
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "For RL training, we use only the question as input ($input_i^{RL}=q_i$). The model generates $o_i^{RL}=(c_i,a_i)$ and is updated to maximize the reward function $\mathcal{R}$."

## CH2-044
- ID: CH2-044
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:87
- Issue Type: Consistency
- Original: "Moreover, each LLM has its own required format for input texts, as the tokenizer encodes input text into a sequence of tokens. Special labels are used specifically to distinguish the sections: the ``\texttt{System Prompt}'', ``\texttt{User Prompt}'', and ``\texttt{Assistant Response}''. Table. \ref{tab:ch2:input_tem..."
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: "Each LLM requires a model-specific message format because tokenizers encode role boundaries differently. We therefore use explicit role tags for system, user, and assistant messages. Table~\ref{tab:ch2:input_template_for_llama3} shows the LLaMA 3 template, where ``\prompttag{start\_header\_id}'' and ``\prompttag{end..."

## CH2-045
- ID: CH2-045
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:118
- Issue Type: Grammar
- Original: "Compared with traditional LLMs, fine-tuning Reasoning Large Language Models (RLLMs) requires explicitly incorporating the reasoning process into the target answers. Recent literature commonly uses special markers, such as ``\texttt{<think>}'' and ``\texttt{</think>}'', to denote the intermediate reasoning trajectory..."
- Revision Rationale: Corrected grammar, tense, article usage, and sentence structure to meet formal academic writing standards.
- Revised Version: "Relative to standard chat fine-tuning, reasoning-oriented training requires explicit supervision of intermediate reasoning and final answers. We use ``\texttt{<think>}...\texttt{</think>}'' for reasoning traces and ``\texttt{<answer>}...\texttt{</answer>}'' for final outputs. The target format is: \begin{equation*} ..."

## CH2-046
- ID: CH2-046
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:130
- Issue Type: Wording
- Original: "Additionally, to reinforce adherence to this structured format, we employed a dedicated \textbf{System Prompt}, illustrated in Table~\ref{tab:ch2:system_prompt}. The \textbf{System Prompt} typically contains instructions or guidelines that establish a clear framework for guiding the mol to generate relevant and stru..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "To improve format compliance, we add a dedicated \textbf{system prompt} (Table~\ref{tab:ch2:system_prompt}) that enforces output tags and response structure. \begin{table}[h] \centering \footnotesize \begin{tabular}{p{0.98\textwidth}} \toprule \textbf{System Prompt:} \\ \midrule You are a helpful AI Assistant, desig..."

## CH2-047
- ID: CH2-047
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:179
- Issue Type: Grammar
- Original: "In this section, we explain the input for the Supervised Fine-Tuning (SFT). The SFT process aims at enhancing model's ability in financial analysis. The model will learn to generate a logical reasoning process and draw a similar conclusion as the required output. 4 main steps involves in fine-tuning. Firstly, we nee..."
- Revision Rationale: Corrected grammar, tense, article usage, and sentence structure to meet formal academic writing standards.
- Revised Version: "This subsection describes SFT inputs and optimization flow. SFT is used to improve domain-specific analytical generation by supervising both reasoning traces and final answers. The workflow includes four steps: preparing Q\&A training data, defining system and user prompts, applying parameter-efficient fine-tuning, ..."

## CH2-048
- ID: CH2-048
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:193
- Issue Type: Wording
- Original: "For example, we want to conduct an analysis on the current inflation rate. We use Personal Consumption Expenditures (PCE) Index as the key indicator. We construct the input prompt ($q_i$) as (Table \ref{tab:ch2:sft_user_prompt_example}):"
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "As an example, consider inflation analysis using the Personal Consumption Expenditures (PCE) indicator set. The corresponding input prompt $q_i$ is shown in Table~\ref{tab:ch2:sft_user_prompt_example}. \begin{table}[h] \centering \footnotesize \begin{tabular}{p{0.98\textwidth}} \toprule \textbf{User Prompt ($q_i$):}..."

## CH2-049
- ID: CH2-049
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:244
- Issue Type: Grammar
- Original: "The target output consists of the Reasoning process ($c_i$) and Answer ($a_i$), we firstly wrapped the Reasoning process using \textbf{\texttt{\textless think\textgreater ... \textless /think\textgreater}} and the Answer using \textbf{\texttt{\textless answer\textgreater ... \textless /answer\textgreater}}. Then we ..."
- Revision Rationale: Corrected grammar, tense, article usage, and sentence structure to meet formal academic writing standards.
- Revised Version: "The target output combines reasoning ($c_i$) and answer ($a_i$): reasoning is wrapped in \texttt{<think>...</think>} and the final answer in \texttt{<answer>...</answer>}. The combined target $o_i=(c_i,a_i)$ is illustrated in Table~\ref{tab:ch2:sft_target_output_example}. \begin{table}[h] \centering \footnotesize \b..."

## CH2-050
- ID: CH2-050
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:281
- Issue Type: Grammar
- Original: "The refinement of the model through retraining employs stochastic gradient descent (SGD) to iteratively adjust the model parameters ($\theta$) based on the \textit{loss function}. In partial fine-tuning (Partial FT), the update process selectively modifies certain layers while leaving others unchanged by zeroing out..."
- Revision Rationale: Corrected grammar, tense, article usage, and sentence structure to meet formal academic writing standards.
- Revised Version: "Model refinement is performed with stochastic gradient descent (SGD), updating parameters $\theta$ according to the training loss. Under partial fine-tuning, only selected layers are trainable, while frozen layers keep pre-trained knowledge intact."

## CH2-051
- ID: CH2-051
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:286
- Issue Type: Grammar
- Original: "During SFT training, the \textit{loss} is calculated as the discrepancy between the model's generated responses ($\bm{o^*} \in R^{m\times n}$) and the target responses ($\bm{o}^{SFT} \in R^{m \times n}$). This discrepancy is quantified by the multi-class Cross Entropy Loss ($\mathcal{L}^{CE}$):"
- Revision Rationale: Corrected grammar, tense, article usage, and sentence structure to meet formal academic writing standards.
- Revised Version: "During SFT, the objective is the discrepancy between generated responses ($\bm{o^*}\in\mathbb{R}^{m\times n}$) and targets ($\bm{o}^{SFT}\in\mathbb{R}^{m\times n}$), measured by multi-class cross-entropy loss $\mathcal{L}^{CE}$: \begin{equation*} \mathcal{L}^{CE} = -\frac{1}{n}\sum_{i=1}^{n} o^{SFT}_{k, i}\log o^*_i..."

## CH2-052
- ID: CH2-052
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:298
- Issue Type: Grammar
- Original: "where $o_{k,i} = [o_1, o_2, \cdots, o_k, \cdots, o_m]$ is a unit vector with $o_k = 1$ (also known as one-hot encoded vector), which represents the target token, $o^*_i(\cdot)$ denotes the probabilities for the predicted token given the input tokens $\bm{x}$ and previously generated words $\bm{y}$, which are always ..."
- Revision Rationale: Corrected grammar, tense, article usage, and sentence structure to meet formal academic writing standards.
- Revised Version: "where $o_{k,i}$ is a one-hot vector identifying the target token at position $i$, and $o_i^*(\cdot)$ is the model-assigned probability for each candidate token given inputs $\bm{x}$ and previously generated tokens $\bm{y}_{<i}$. The sequence length is $n$. Cross-entropy sums negative log probabilities assigned to ta..."

## CH2-053
- ID: CH2-053
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:310
- Issue Type: Wording
- Original: "Reinforcement Learning (RL) requires the model to update parameters ($\theta$) by optimizing feedback. Different from the Supervised Fine-Tuning, the target of the Reinforcement Learning is to maximize the feedback by changing the way of producing the output. As the same analysis can be constructed in different ways..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "Reinforcement learning (RL) updates parameters $\theta$ by maximizing feedback signals rather than matching fixed token-level targets. This is useful in analytical tasks where multiple valid response paths exist. Recent reasoning models (e.g., ChatGPT o1, DeepSeek-R1, Fin-R1, Fino1) illustrate the value of RL-style ..."

## CH2-054
- ID: CH2-054
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:317
- Issue Type: Grammar
- Original: "GRPO is an on-policy reinforcement learning algorithm derived from the Proximal Policy Optimization algorithm (PPO, \cite{schulman2017proximal}). PPO is a policy-gradient-based reinforcement learning technique designed for stable optimization. It requires the model to maximize a reward signal ($A_t$), provided by a ..."
- Revision Rationale: Corrected grammar, tense, article usage, and sentence structure to meet formal academic writing standards.
- Revised Version: "GRPO is an on-policy RL algorithm derived from Proximal Policy Optimization (PPO; \cite{schulman2017proximal}). In this framework, the policy $\pi_\theta$ is optimized to maximize a reward-driven objective $\mathcal{J}(\theta)$. Unlike SFT, RL training does not require reference targets; the model is scored by a rew..."

## CH2-055
- ID: CH2-055
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:323
- Issue Type: Grammar
- Original: "Additionally, PPO employs a clipping function ($CLIP(\cdot)$) to scale the reward in the objective function, enhancing training stability. The model parameters ($\theta$) are updated using stochastic gradient descent (SGD) until the objective function is maximized. The objective function ($\mathcal{J}^{\text{PPO}}$)..."
- Revision Rationale: Corrected grammar, tense, article usage, and sentence structure to meet formal academic writing standards.
- Revised Version: "PPO applies a clipping operator $CLIP(\cdot)$ for stable policy updates. Parameters $\theta$ are optimized with SGD to maximize the PPO objective: \begin{footnotesize} \begin{equation} \begin{split} \mathcal{J}^{\text{PPO}}(\theta) = &\mathbb{E}_{q\sim P(Q), o\sim\pi_{\theta_{\text{old}}}(O|q)}\frac{1}{|o|}\sum_{t=1..."

## CH2-056
- ID: CH2-056
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:389
- Issue Type: Grammar
- Original: "Here, $G$ is the size of the sampled outputs for each iteration $t$. $r_{i,t}$ is reward for output $i$ at iteration $t$. Instead of training a \textit{value model} in calculating the expected rewards, the GRPO uses the sum of the normalized rewards from a sample outputs of size $G$ as the advantage function $\hat{A..."
- Revision Rationale: Corrected grammar, tense, article usage, and sentence structure to meet formal academic writing standards.
- Revised Version: "Here, $G$ denotes the number of sampled outputs per iteration and $r_{i,t}$ is the reward for output $i$ at step $t$. GRPO avoids a separately trained value model by computing relative advantages from group-normalized rewards. A KL penalty is included directly in the objective to constrain policy drift from the refe..."

## CH2-057
- ID: CH2-057
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:394
- Issue Type: Logic
- Original: "Finally, to implement the GRPO, we use the same \textit{User Prompt ($q_i$)} (see Table~\ref{tab:ch2:sft_user_prompt_example}) as the SFT stage as our input. Then, we adopt the idea of ``LLM as Judge'' by using another powerful model as our reward model to score the output from our policy model."
- Revision Rationale: Resolved narrative inconsistency and aligned the text with tables, sample definitions, or evaluation logic without changing empirical results.
- Revised Version: "In implementation, GRPO uses the same user prompt template as SFT (Table~\ref{tab:ch2:sft_user_prompt_example}). Rewards are assigned with an LLM-as-Judge setup, where a stronger evaluator model scores policy outputs. \subsection{Reward Function Design}\label{ch2:sec:mode_training:reward_function} %"

## CH2-058
- ID: CH2-058
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:404
- Issue Type: Logic
- Original: "Inspired by existing literature, we adopt a two-aspect reward framework comprising a format reward and an accuracy reward. For each stage of GRPO training, we design multiple reward functions to guide the model’s behavior in alignment with specific task objectives."
- Revision Rationale: Resolved narrative inconsistency and aligned the text with tables, sample definitions, or evaluation logic without changing empirical results.
- Revised Version: "Following prior work, we use a multi-component reward design. The core components are format compliance, analytical quality, and reasoning coherence."

## CH2-059
- ID: CH2-059
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:409
- Issue Type: Logic
- Original: "In the RL training stage, the focus is on enhancing the model’s analytical ability. Unlike mathematical reasoning tasks, economic and financial analyses do not always yield a single ``correct'' answer. To address this, we employ the ``LLM-as-Judge framework'' (\cite{zheng2024judging}) to evaluate both the reasonin..."
- Revision Rationale: Resolved narrative inconsistency and aligned the text with tables, sample definitions, or evaluation logic without changing empirical results.
- Revised Version: "RL training targets analytical quality. Because macro-financial analysis does not have a unique ground-truth answer, we evaluate outputs with an LLM-as-Judge framework (\cite{zheng2024judging}). We use three reward components: \textit{Format Reward} $\mathcal{R}_{fmt}(o_i)$, \textit{Accuracy Reward} $\mathcal{R}_{ac..."

## CH2-060
- ID: CH2-060
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:421
- Issue Type: Grammar
- Original: "To ensure the reasoning process, we require the model to generate the output in the specific format. The reasoning process should be wrapped with \textbf{\texttt{\textless think\textgreater ... \textless /think\textgreater}} and the answer should be wrapped with \textbf{\texttt{\textless answer\textgreater ... \text..."
- Revision Rationale: Corrected grammar, tense, article usage, and sentence structure to meet formal academic writing standards.
- Revised Version: "To enforce output structure, reasoning must appear in \texttt{<think>...</think>} and the final response in \texttt{<answer>...</answer>}. The format reward is binary: \begin{equation}\label{eq:ch2:format_reward} \mathcal{R}_{\text{fmt}}(o_i) = \begin{cases} 1, & \text{if the format is correct} \\ 0, & \text{if the ..."

## CH2-061
- ID: CH2-061
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:440
- Issue Type: Grammar
- Original: "Unlike previous studies that trained their models on publicly available datasets, such as FinQA (\cite{chen2021finqa}), our training target is to generate comprehensive and logical economic and financial analyses. Since there is no fixed or predefined correct answer for such analyses, we adopt the "LLM as Judge" fra..."
- Revision Rationale: Corrected grammar, tense, article usage, and sentence structure to meet formal academic writing standards.
- Revised Version: "Our objective differs from tasks with fixed answers (e.g., FinQA; \cite{chen2021finqa}). We target comprehensive macro-financial analysis, where multiple valid responses may exist. We therefore use an LLM-as-Judge reward model (\cite{zheng2023judging}) and score outputs with a structured rubric. Because judge-based ..."

## CH2-062
- ID: CH2-062
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:518
- Issue Type: Wording
- Original: "Then, each criterion is scored on a scale from 1 to 5, where 1 indicates "Very Poor" and 5 indicates "Excellent." The \textit{total score}, obtained by summing the individual criterion scores, is then used as the accuracy reward. We employ the prompt shown in Table~\ref{tab:ch2:accuracy_reward_prompt} to format the ..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "Each criterion is scored from 1 to 5 (1 = very poor, 5 = excellent). The summed score is normalized by 35 to produce the final accuracy reward $\mathcal{R}_{\text{acc}}(o_i)\in[0,1]$. Table~\ref{tab:ch2:accuracy_reward_prompt} shows the evaluator prompt. \begin{table}[htbp] \centering \footnotesize \begin{tabular}{p..."

## CH2-063
- ID: CH2-063
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:570
- Issue Type: Wording
- Original: "Furthermore, to ensure that the model generates logically coherent responses, we introduced a reasoning reward ($\mathcal{R}_{\text{reason}}(o_i)$) to evaluate the quality of the reasoning trajectory. Similar to our \textit{Accuracy Reward}, we adopted the ``LLM-as-Judge'' framework, using a powerful \textit{reward ..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "To enforce logical consistency between reasoning and final answers, we add a reasoning reward $\mathcal{R}_{\text{reason}}(o_i)$. Using the same LLM-as-Judge framework, the evaluator assigns a score from 1 to 5 based on reasoning--answer consistency (Table~\ref{tab:ch2:reasoin_reward_prompt}). We normalize by 5 so t..."

## CH2-064
- ID: CH2-064
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:585
- Issue Type: Wording
- Original: "You are an economic and financial policy expert serving as an internal evaluator at the Federal Reserve. Your task is to assess the **internal logical consistency** between a language model’s reasoning and its final answer. \\"
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "You are an economic and financial policy expert serving as an internal Federal Reserve evaluator. Your task is to assess the **logical consistency** between a model's reasoning and its final answer. \\ ~\\ You will be provided with: \\ - **\textless Model Reasoning\textgreater**: The model’s internal reasoning pro..."

## CH2-065
- ID: CH2-065
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:721
- Issue Type: Wording
- Original: "Fine-tuning models to enhance their performance on specific task is a common practice in the NLP. When conducting Full-parameter Fine-Tuning (FT), the entire base model will be updated, leading to enhanced capabilities in generating task-specific text. However, this comes at a significant computational cost. Full-pa..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "Fine-tuning for specialized tasks is standard in NLP. Under full-parameter fine-tuning, all base-model weights are updated, which can improve task performance but is computationally expensive. For an 8B-parameter model, full fine-tuning can require roughly 120 GB of GPU memory. Partial fine-tuning reduces this requi..."

## CH2-066
- ID: CH2-066
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:727
- Issue Type: Wording
- Original: "To further optimize memory usage without compromising performance, Parameter-Efficient Fine-Tuning (PEFT) and Quantization techniques, such as LoRA (Low-Rank Adaptation, \cite{hu2021lora}), Quantized Low-Rank Adaptation (QLoRA, \cite{dettmers2024qlora}), and Gated Low-Rank Evolving (GaLoRE, \cite{zhao2024galore}), a..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "To reduce memory costs while maintaining performance, we use parameter-efficient fine-tuning (PEFT) with quantization, including LoRA (\cite{hu2021lora}), QLoRA (\cite{dettmers2024qlora}), and related methods such as GaLoRE (\cite{zhao2024galore}). PEFT updates only a small subset of parameters or adds lightweight t..."

## CH2-067
- ID: CH2-067
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:733
- Issue Type: Consistency
- Original: "LoRA's mechanism involves introducing additional layers, referred to as Adapters, with a much smaller rank (typically 1 or 2) compared to the pre-trained weight matrices (such as 2048 in the LLaMA3 model). These Adapters are specifically designed for the task at hand and undergo training during the fine-tuning proce..."
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: "LoRA introduces low-rank adapter matrices with far fewer trainable parameters than the original weight matrices. During fine-tuning, only adapter parameters are updated; base weights remain frozen. This design enables efficient domain adaptation on limited hardware. In this chapter, we use QLoRA, which combines LoRA..."

## CH2-068
- ID: CH2-068
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:742
- Issue Type: Consistency
- Original: "Following the literature in model training and model fine-tuning (see \cite{gpt2018}, \cite{meta2024llama3}, and \cite{yang2025qwen3} for model training; \cite{howard2018universal} and \cite{ding2023parameter} for model fine-tuning; and \cite{li2020rethinking}, \cite{vm2024fine} for empirical results), the general t..."
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: "Following the training and fine-tuning literature (\cite{gpt2018}; \cite{meta2024llama3}; \cite{yang2025qwen3}; \cite{howard2018universal}; \cite{ding2023parameter}; \cite{li2020rethinking}; \cite{vm2024fine}), we set the learning rate to $3.0\times10^{-6}$ and batch size to 2 for both SFT and GRPO. SFT runs for 3 e..."

## CH2-069
- ID: CH2-069
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:747
- Issue Type: Wording
- Original: "The choice of a batch size of 2 was driven by the constraints of GPU memory capacity. While a larger batch size could have improved training efficiency, it would have required significantly more memory."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "The batch size of 2 is hardware-constrained. Larger batches could improve throughput but would exceed available GPU memory."

## CH2-070
- ID: CH2-070
- File: Chapter2/sections/model_training.tex
- Location: Chapter2/sections/model_training.tex:752
- Issue Type: Grammar
- Original: "To prevent overfitting, enhance the model's generalization capability, and strengthen the association between prompts and desired responses, the SFT dataset was cycled through three times (i.e., 3 training epochs). In contrast, only 1 epoch was used in the GRPO stage, as the primary objective at this phase was not t..."
- Revision Rationale: Corrected grammar, tense, article usage, and sentence structure to meet formal academic writing standards.
- Revised Version: "We use 3 SFT epochs to strengthen prompt--response alignment while monitoring overfitting risk. GRPO is limited to 1 epoch because its objective is behavioral refinement through rewards rather than memorization of supervised examples."

## CH2-071
- ID: CH2-071
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:117
- Issue Type: Wording
- Original: "Finally, the model incorporates Rotary Positional Embeddings (RoPE, \cite{su2024roformer}), which encodes both the absolute and relative position information of tokens using a rotation matrix, unlike the original transformer that employed a sinusoidal function to encode only absolute position."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "Finally, the model incorporates Rotary Positional Embeddings (RoPE, \cite{su2024roformer}), which encode both absolute and relative token-position information through rotational transformations, unlike the original sinusoidal design that encodes absolute position only."

## CH2-072
- ID: CH2-072
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:122
- Issue Type: Wording
- Original: "Overall, the modifications, including the Grouped-Query Attention (GQA) and the Swish Gated Linear Unit (SwiGLU), alongside the Rotary Positional Embeddings (RoPE), constitute significant advancements in the model. These enhancements not only optimize the model's performance but also broaden its applicability across..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "Overall, the combined use of GQA, SwiGLU, and RoPE improves efficiency and long-context performance, making the LLaMA architecture a strong base for domain-specific fine-tuning. \subsection{From Chat Model to Reasoning Model: DeepSeek-R1-Distill-Llama-8B}"

## CH2-073
- ID: CH2-073
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:132
- Issue Type: Consistency
- Original: "Prompt engineering plays a pivotal role in eliciting the reliable response from language models. There are three ways in prompting the model, zero-shot standard prompting, zero-shot Chain-of-Though prompting (zero-shot CoT, \cite{kojima2022large}), and few-shot Chain-of-Though prompting (few-shot CoT, \cite{wei2022c..."
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: "Prompt design is important for eliciting reliable responses. A standard taxonomy includes: (i) zero-shot direct prompting, (ii) zero-shot Chain-of-Thought (CoT; \cite{kojima2022large}), and (iii) few-shot CoT (\cite{wei2022chain}). Direct prompting can work for simple tasks, but complex analytical tasks usually bene..."

## CH2-074
- ID: CH2-074
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:141
- Issue Type: Wording
- Original: "However, creating logical and coherent CoT prompts is inherently challenging due to requirements for multi-step reasoning, consistency across intermediate steps, and domain-specific expertise. A well-constructed CoT must not only segment the problem into intermediate steps but also ensure that each step is logically..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "Constructing consistently high-quality CoT prompts is difficult because it requires domain expertise and stable multi-step logic. Reasoning-oriented LLMs, such as ChatGPT-o1 (\cite{jaech2024openai}) and DeepSeek-R1 (\cite{shao2024deepseekmath}), mitigate this by training directly on reasoning traces. This design mot..."

## CH2-075
- ID: CH2-075
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:281
- Issue Type: Wording
- Original: "To start with, the textual similarity focuses on comparing the similarity between the target output and the generated output using mathematical distance, such as the Levenshtein Similarity Ratio (\cite{levenshtein1966binary}) and Cosine Similarity (\cite{tanguy2016natural}). The Levenshtein Similarity Ratio replies ..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "Textual similarity measures how closely the generated output matches the reference section. Classical string-based measures (e.g., Levenshtein distance and ROUGE; \cite{levenshtein1966binary}; \cite{lin2004rouge}) are informative but emphasize surface overlap. Because this chapter evaluates semantic alignment, we fo..."

## CH2-076
- ID: CH2-076
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:315
- Issue Type: Grammar
- Original: "Compared with the ``one hot key'' used in ``World-of-Bag'', the text embedding used in LLM has two significantly advantages. First of all, the text embedding utilizes the ``tokenizer'' converting the input text into a small sized vector rather than a large permutation matrix. Then, the ``hidden layer'' inside the LL..."
- Revision Rationale: Corrected grammar, tense, article usage, and sentence structure to meet formal academic writing standards.
- Revised Version: "Compared with one-hot bag-of-words representations, LLM embeddings offer dense, contextual vectors that preserve semantic and positional information. Text is tokenized into embeddings $\mathbf{h}_i\in\mathbb{R}^{m\times 1}$ and then contextualized through transformer layers. Hidden size $m$ varies across models (e.g..."

## CH2-077
- ID: CH2-077
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:364
- Issue Type: Grammar
- Original: "where $Y$ and $\hat{Y}$ are the corresponding text embedding for target and generated texts. $\| \cdot \|$ denoted the Euclidean norm, which is $\| Y \|_2 = \sqrt{y_1^2 + y_2^2 + ,..., + y_N^2}$"
- Revision Rationale: Corrected grammar, tense, article usage, and sentence structure to meet formal academic writing standards.
- Revised Version: "where $Y$ and $\hat{Y}$ are sentence embeddings for the reference and generated texts, and $\|\cdot\|_2$ denotes the Euclidean norm (e.g., $\|Y\|_2=\sqrt{y_1^2+y_2^2+\cdots+y_N^2}$). % According to the distribution theory, Finally, drawing from distributional semantics theory (\cite{firth1957linguistics}, \cite{sahl..."

## CH2-078
- ID: CH2-078
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:441
- Issue Type: Consistency
- Original: "Overall, we propose the following hypothesis for Cosine Similarity and BertScore:"
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: "Overall, we propose the following hypotheses for cosine similarity and BERTScore: \begin{equation*} \begin{split} &Cos_{FT} - Cos_{Base} > 0 \\ &BERTScore^{Recall}_{FT} - BERTScore^{Recall}_{Base} > 0 \\ &BERTScore^{Precision}_{FT} - BERTScore^{Precision}_{Base} > 0\\ &BERTScore^{F1}_{FT} - BERTScore^{F1}_{Base} > 0..."

## CH2-079
- ID: CH2-079
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:455
- Issue Type: Wording
- Original: "where the subscripts $FT$ and $Base$ refer to the fine-tuned and base models,This hypothesis asserts that the similarity between the target text and the response generated by the fine-tuned model will be higher than that between the target text and the response generated by the base model. A statistically significan..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "where subscripts $FT$ and $Base$ denote the fine-tuned and baseline models. The hypothesis is that fine-tuning increases semantic alignment with the target text; statistically significant positive differences support this claim. %%%%% end of text similarity \subsubsection*{Sufficient Information} % Shapley Value"

## CH2-080
- ID: CH2-080
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:467
- Issue Type: Wording
- Original: "Apart from the similarity, we expect the generated output to contain sufficient information and analysis directly derived from the input prompts rather than relying on the model's imagination."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "Beyond textual similarity, generated outputs should contain sufficient information grounded in the provided indicators, rather than unsupported model priors. % To evaluate this, our second metric assesses the contribution of the key prompts to the generated output. Specifically, we employ the Shapley Value (\cite{sh..."

## CH2-081
- ID: CH2-081
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:517
- Issue Type: Logic
- Original: "In this test, we evaluate the model’s simulation capacity by conducting empirical analyses on the synthetic text it generates. The current literature has confirmed the impact of the release of FOMC Minutes on the financial market, for instance, \cite{rosa2013financial} found that the release of the Minutes signifi..."
- Revision Rationale: Resolved narrative inconsistency and aligned the text with tables, sample definitions, or evaluation logic without changing empirical results.
- Revised Version: "This test evaluates simulation quality through downstream empirical models. Prior studies show that FOMC communication affects financial markets (\cite{rosa2013financial}; \cite{rosa2011high}; \cite{huang2021economic}; \cite{tadle2022fomc}). We therefore test whether sentiment extracted from synthetic Minutes retain..."

## CH2-082
- ID: CH2-082
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:526
- Issue Type: Logic
- Original: "To assess this, we adopt the empirical strategy from \cite{tadle2022fomc}, which examines how FOMC Minutes influence market expectations regarding monetary policy. Specifically, first of all, we apply the main regression model (Eq.\ref{eq:ch2:sentiment_reg})from their study, which investigates the relationship betwe..."
- Revision Rationale: Resolved narrative inconsistency and aligned the text with tables, sample definitions, or evaluation logic without changing empirical results.
- Revised Version: "We follow the empirical design in \citet{tadle2022fomc} and estimate Eq.~\ref{eq:ch2:sentiment_reg}, which links communication sentiment to interest-rate and exchange-rate movements. In our setting, sentiment is extracted from synthetic Minutes. \begin{equation}\label{eq:ch2:sentiment_reg} p^f_t = \lambda p^f_{t-1} ..."

## CH2-083
- ID: CH2-083
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:537
- Issue Type: Consistency
- Original: "where $p^f_t$ is the price for the Federal Fund Rate Future contract $f$ at time $t$. $VIX_t$ is the log change of the Volatility Index (VIX), $Z_t^M$ and $Z_t^S$ are the sentiment scores extracted from the Minutes and Statements, $Y_t$ is the monthly time effect, and $\phi_t$ is the error term."
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: "where $p_t^f$ is the price of Federal Funds futures contract $f$ at time $t$, $VIX_t$ is the log change in VIX, $Z_t^M$ and $Z_t^S$ are sentiment scores from Minutes and statements, $Y_t$ captures month effects, and $\phi_t$ is the error term. % where pft is the asset price in period t of the daily futures contract ..."

## CH2-084
- ID: CH2-084
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:675
- Issue Type: Consistency
- Original: "In this section, we will attempt to generate the FOMC Minutes using a Large Language Model (LLM). Fine-tuning large pre-trained models is a highly effective method for adapting general-purpose LLMs to specific content tasks. This technique enables the model to conform to a particular style, affecting not only the ou..."
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: "This section reports training and downstream evaluation results for FOMC-Minutes generation. We assess both optimization dynamics (loss/reward trajectories) and task performance under text-based and economics-based criteria. \subsection{Model Training Results} % \subsubsection*{Stage 1 Supervised Fine-Tuning Results..."

## CH2-085
- ID: CH2-085
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:687
- Issue Type: Consistency
- Original: "In this section, we will try to fine-tune the base model (\textbf{LLAMA3B-8}). The number of total trainable parameters is 3,407,872 out of 8,051,232,768 total parameters with one adopter (only 0.0423\% trainable rate)."
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: ""

## CH2-086
- ID: CH2-086
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:695
- Issue Type: Grammar
- Original: "Firstly, we conduct Supervised Fine-Tuning on the model to enhance the instruction-following ability. We trained the model with 3 epochs and 2,346 steps. Each epoch will go though all 3,128 training dataset and the gradient will be update every 4 generations, which makes it 2,346 steps in total. Additionally, the mo..."
- Revision Rationale: Corrected grammar, tense, article usage, and sentence structure to meet formal academic writing standards.
- Revised Version: "SFT is trained for 3 epochs (2,346 optimization steps). Each epoch covers 3,128 training samples, with gradient accumulation every 4 generations. Evaluation loss is logged every 20 steps. Figure~\ref{fig:ch2:stage1_sft_loss} reports training and evaluation loss. \begin{figure}[htbp] \begin{center} \includegraphics[w..."

## CH2-087
- ID: CH2-087
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:712
- Issue Type: Consistency
- Original: "Initial experiment starts with 3 epochs, as shown in Figure \ref{fig:ch2:stage1_sft_loss}. During the period, the training loss decreased from 2.1504 (at Step = 20) to 0.6913 (at Step = 2340), Simultaneously, the evaluation loss converged from 2.1193 (at Step 20) to 0.7097 after 1,560 steps and remained stable endin..."
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: "As shown in Figure~\ref{fig:ch2:stage1_sft_loss}, training loss declines from 2.1504 (step 20) to 0.6913 (step 2340). Evaluation loss decreases from 2.1193 to 0.7097 by step 1,560 and ends at 0.6984. After roughly step 1,340, training loss continues to fall while evaluation loss plateaus, which is consistent with mi..."

## CH2-088
- ID: CH2-088
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:754
- Issue Type: Consistency
- Original: "In summary, with only 0.042\% of parameters left trainable, the \textbf{LLAMA-3-8B} model demonstrated measurable improvements following the two-step training process. Supervised Fine-Tuning (SFT) reduced the training loss to 0.69 over three epochs, while the validation loss stabilized around 0.70, which indicates s..."
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: "In summary, with only 0.042\% trainable parameters, the two-step process yields measurable gains. SFT improves supervised fit, and GRPO improves reward-aligned reasoning stability. Together, these stages improve both reasoning consistency and final-answer quality. % \subsubsection*{Stage 2 Results} % In this section..."

## CH2-089
- ID: CH2-089
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:979
- Issue Type: Wording
- Original: "Noticeably, the results from all evaluation metrics confirmed the improvements of the fine-tuning. The percentage changes in all metrics of the ``answer'' part is significantly higher than the other parts. For instance, the $Cos$ difference for the ``answer'' is 2.872\% while only 0.7575\% for the ``full'' response...."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: ""

## CH2-090
- ID: CH2-090
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:639
- Issue Type: Consistency
- Original: "\caption{Daily Federal Fund Rate Changes (in BPS) from Jan 1, 2000 to May 9, 2025}"
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: "\caption{Daily Federal Funds Rate Changes (in bps), January 1, 2000 to May 9, 2025} \label{tab:ch2:ffr_changes} \end{minipage} \hfill \begin{minipage}[t]{0.48\textwidth} \centering \begin{tabular}{l} \toprule Target Vote Choices \\ \midrule Raise by 100 basis points \\ Raise by 75 basis points \\ Raise by 50 basis p..."

## CH2-091
- ID: CH2-091
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:1310
- Issue Type: Logic
- Original: "As a further step, we evaluate the \textit{simulation capability} of our fine-tuned model. Specifically, we adopt the sentiment–return regression framework (Eq.~\ref{eq:ch2:sentiment_reg}) originally proposed by \citet{tadle2022fomc} to examine whether the synthetic Minutes generated by the LLM can explain variati..."
- Revision Rationale: Resolved narrative inconsistency and aligned the text with tables, sample definitions, or evaluation logic without changing empirical results.
- Revised Version: "As a further validation step, we evaluate the \textit{simulation capability} of the fine-tuned model using the sentiment--return regression in Eq.~\ref{eq:ch2:sentiment_reg} (\citet{tadle2022fomc}). We test whether sentiment extracted from synthetic Minutes explains variation in financial-market variables (federal f..."

## CH2-092
- ID: CH2-092
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:1356
- Issue Type: Logic
- Original: "The bootstrap mean coefficient of $Z_{t}^{m}$ reached at 0.00271 for the Fine-tuned model and 0.00224 for the base model. Compared with the target coefficient (0.00505), there is a 20.97\% improvement. As for the t-statistics, there is a 15.94\% improvement from 0.61689 (base model) to 0.71521 (fine tuned model). Th..."
- Revision Rationale: Resolved narrative inconsistency and aligned the text with tables, sample definitions, or evaluation logic without changing empirical results.
- Revised Version: "The bootstrap mean coefficient for $Z_t^m$ is 0.00271 for the fine-tuned model and 0.00224 for the baseline model. Relative to the benchmark coefficient (0.00505), the fine-tuned model reduces the estimation gap by 20.97\%. The corresponding $t$-statistic increases from 0.6169 to 0.7152, a 15.94\% improvement. % dec..."

## CH2-093
- ID: CH2-093
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:1413
- Issue Type: Logic
- Original: "As shown in Table~\ref{tab:ch2:decision_acc}, the prediction accuracy of the model for FOMC Minutes before and after 2009 is generally comparable. Specifically, the fine-tuned model achieves an overall accuracy of 78.13\% for Minutes prior to 2009 and 80.77\% for those after 2009, while the base model yields 66.41\%..."
- Revision Rationale: Resolved narrative inconsistency and aligned the text with tables, sample definitions, or evaluation logic without changing empirical results.
- Revised Version: "Table~\ref{tab:ch2:decision_acc} shows broadly comparable performance across periods. For the Minutes-aligned model, overall accuracy is 78.13\% in the post-2009 sample and 80.77\% in the pre-2009 sample. For the baseline model, the corresponding values are 66.41\% and 70.21\%. Across all observations, the Minutes-a..."

## CH2-094
- ID: CH2-094
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:1423
- Issue Type: Logic
- Original: "We then proceed to evaluate the performance of synthetic sections generated by our \textit{Fine-Tuned Model}. Utilizing the same prompt template as in the previous evaluation, we replaced the original FOMC Minutes with synthetic ones and instructed the model to select the most appropriate monetary policy decision fr..."
- Revision Rationale: Resolved narrative inconsistency and aligned the text with tables, sample definitions, or evaluation logic without changing empirical results.
- Revised Version: "We next evaluate decisions inferred from synthetic sections. Using the same template, we replace original Minutes text with synthetic text and ask each model to select one policy action from Table~\ref{tab:ch2:vote_choice}. % Furthermore, we examine whether the models are capable of predicting monetary policy decisi..."

## CH2-095
- ID: CH2-095
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:1430
- Issue Type: Logic
- Original: "For this experiment, we simulated 500 synthetic FOMC Minutes using both the base and fine-tuned models, and subsequently draw 20 random samples with a sampling interval of 2,000. Finally, the results of prediction accuracy using synthetic FOMC Minutes is reported in Table~\ref{tab:ch2:decision_acc_synthetic}."
- Revision Rationale: Resolved narrative inconsistency and aligned the text with tables, sample definitions, or evaluation logic without changing empirical results.
- Revised Version: "For this experiment, we generated 500 synthetic Minutes with each model and then drew 20 random samples (sampling interval: 2,000). Decision-prediction accuracy based on synthetic Minutes is reported in Table~\ref{tab:ch2:decision_acc_synthetic}. % and Table~\ref{tab:ch2:decision_acc_indicators}, respectively. % TOD..."

## CH2-096
- ID: CH2-096
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:1511
- Issue Type: Wording
- Original: "In conclusion, our research introduces a novel methodology for generating structured synthetic text data. More specifically, we have established that a fine-tuned Large Language Model (LLM) can effectively produce simulations of the Federal Open Market Committee (FOMC) Minutes. Empirical evidence demonstrates that t..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "In conclusion, this chapter presents a framework for generating structured synthetic policy text with a fine-tuned LLM. The empirical results show that the Minutes-aligned model outperforms the baseline in textual fidelity, input grounding, and decision-relevant downstream tasks. These findings support the use of LL..."

## CH2-097
- ID: CH2-097
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:1520
- Issue Type: Consistency
- Original: "Additionally, the use of standardized prompts presents an efficient pathway for generating results, eliminating the necessity for users to guide the model through multiple steps to achieve desired outcomes. Therefore, an upcoming focus will be to rigorously evaluate the generative capacity of such standardized promp..."
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: "Future work should further validate standardized prompt templates, improve robustness under distribution shifts, and extend attribution methods beyond leave-one-out masking. These directions can strengthen reliability and expand the practical value of synthetic policy communication in financial applications."

## CH2-098
- ID: CH2-098
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:67
- Issue Type: Consistency
- Original: "\subsection{Base Model: LLAMA3-8B-Instruct} To generate consistent texts using the LLM, we choose to use \textbf{LLAMA3-8B-Instruct}, the most powerful open-sourced LLM developed by MetaAI (\cite{meta2024llama3}), as our base model. The LLAMA family is released with three main sizes, ``LLAMA-8B'', ``LLAMA-70B'', and..."
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: "\subsection{Base Model: LLaMA 3-8B-Instruct} For the baseline, we use \textbf{LLaMA 3-8B-Instruct} (\cite{meta2024llama3}). Within the LLaMA 3 family (8B, 70B, and 405B), the 8B model is the smallest and can be deployed on a single 24 GB GPU. The Instruct variant is optimized for dialogue-style instruction following..."

## CH2-099
- ID: CH2-099
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:76
- Issue Type: Consistency
- Original: "Notably, compared with the traditional transformer architecture (\cite{vaswani-etal2018}), the model has incorporated modifications to enhance training stability and generation quality."
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: "Compared with the original transformer architecture (\cite{vaswani-etal2018}), LLaMA 3 includes several modifications for stability and generation quality. % grouped-query attention (GQA)"

## CH2-100
- ID: CH2-100
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:83
- Issue Type: Grammar
- Original: "First of all, the model adopted Grouped-Query Attention (GQA, \cite{ainslie2023gqa}) in place of Multi-Query Attention (MQA, \cite{vaswani-etal2018}) and Multi-Head Attention (MHA, \cite{vaswani-etal2018}). While MHA can produce high-quality outputs, it demands significant memory capacity. Conversely, MQA significan..."
- Revision Rationale: Corrected grammar, tense, article usage, and sentence structure to meet formal academic writing standards.
- Revised Version: "First, the model adopts Grouped-Query Attention (GQA; \cite{ainslie2023gqa}). Relative to standard multi-head attention, GQA reduces memory costs while preserving much of the generation quality, which is useful for long-context processing. % SwiGLU"

## CH2-101
- ID: CH2-101
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:89
- Issue Type: Grammar
- Original: "Additionally, the Swish Gated Linear Unit (SwiGLU, Eq. \eqref{eq:ch2:swiglu}, \cite{shazeer2020glu}) replaces the Rectified Linear Unit (ReLU, Eq. \eqref{eq:ch2:relu}, \cite{glorot2011deep}) in the activation function. Extra gate layers are introduced in the model to enhance performance, particularly in managing lon..."
- Revision Rationale: Corrected grammar, tense, article usage, and sentence structure to meet formal academic writing standards.
- Revised Version: "Second, the model uses Swish Gated Linear Units (SwiGLU; Eq.~\eqref{eq:ch2:swiglu}, \cite{shazeer2020glu}) rather than ReLU (Eq.~\eqref{eq:ch2:relu}, \cite{glorot2011deep}), improving nonlinear expressiveness through gated activations. \begin{equation}\label{eq:ch2:relu} ReLU (\mathbf {x} )=\max(0,\mathbf {x} '\math..."

## CH2-102
- ID: CH2-102
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:107
- Issue Type: Grammar
- Original: "where $\mathbf{x} \in \mathbb{R}^{n \times m}$ is the output from previous layer, $\mathbf{W_1} \in \mathbb{R}^{k \times m \times n}$, $\mathbf{W_2} \in \mathbb{R}^{k \times m \times n}$, $\mathbf{b} \in \mathbb{R}^{n}$, and $\mathbf{c} \in \mathbb{R}^{n}$ are trainable parameters, with $\mathbf{W_1}$ and $\mathbf{W..."
- Revision Rationale: Corrected grammar, tense, article usage, and sentence structure to meet formal academic writing standards.
- Revised Version: "where $\mathbf{x}\in\mathbb{R}^{n\times m}$ is the previous-layer output; $\mathbf{W_1}$, $\mathbf{W_2}$, $\mathbf{b}$, and $\mathbf{c}$ are trainable parameters; and $\mathbf{W_2}$ with $\mathbf{c}$ parameterizes the gating branch. % \cite{su2024roformer} % rotary positional embeddings (RoPE)"

## CH2-103
- ID: CH2-103
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:244
- Issue Type: Wording
- Original: "In this section, we evaluate the performance of our fine-tuned Large Language Model (LLM). Given that an LLM is inherently a probabilistic model, which selects the next token based on the highest probability, a critical issue encountered is the phenomenon known as hallucination. Hallucination refers to instances whe..."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "In this section, we evaluate the fine-tuned LLM with a focus on reliability. Because LLMs are probabilistic generators, hallucination is a key risk. Following \citet{huang2025survey}, we distinguish factuality hallucination (conflict with external facts) from faithfulness hallucination (misalignment with provided in..."

## CH2-104
- ID: CH2-104
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:249
- Issue Type: Consistency
- Original: "On the other hand, evaluating the LLM's performance involves several aspects and different methods. Generally, there are two kinds of metrics, reference-based evaluation and reference-free evaluation. Firstly, the reference-based evaluation compared the model's outputs with the target outputs given the same prompts...."
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: "LLM evaluation can be broadly classified as reference-based or reference-free. Reference-based metrics compare generated outputs with target texts under the same prompts (e.g., BLEU, BERTScore; \cite{papineni2002bleu}; \cite{zhang2019bertscore}). Reference-free methods evaluate outputs without ground-truth text, usi..."

## CH2-105
- ID: CH2-105
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:254
- Issue Type: Wording
- Original: "For our research, we have already collected high-quality reference answers. Therefore, reference-based method will be applied in our evaluation."
- Revision Rationale: Rewrote for concise academic tone, clearer argument flow, and improved readability while preserving the original meaning.
- Revised Version: "Because high-quality reference answers are available in this study, we primarily use reference-based evaluation. More specifically, we will evaluate the fine-tuned model across four key dimensions: 1) the \textbf{textual similarity} of generated outputs, 2) the \textbf{sufficiency of information} contained in the ou..."

## CH2-106
- ID: CH2-106
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:262
- Issue Type: Logic
- Original: "The first two tests focus on detecting the faithfulness hallucination of the model. Both the textual similarity and the prompts contribution can provide quantitative measures of the overlap of key facts between the generated texts and the given texts. In contrast, the third and fourth dimensions focus on the model�..."
- Revision Rationale: Resolved narrative inconsistency and aligned the text with tables, sample definitions, or evaluation logic without changing empirical results.
- Revised Version: "The first two dimensions target faithfulness hallucination by measuring semantic similarity and prompt-grounded contribution. The third and fourth dimensions evaluate simulation utility, namely whether generated text preserves market-relevant signals and supports accurate Federal Reserve decision prediction. % \subs..."

## CH2-107
- ID: CH2-107
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:147
- Issue Type: Consistency
- Original: "Therefore, we selected the \textit{DeepSeek-R1 Distilled} variant of \textit{LLAMA3-8B} (known as \textbf{DeepSeek-R1-Distill-Llama-8B}) as our base model. This model was further fine-tuned on a high-quality reasoning dataset distilled from DeepSeek-R1, a powerful 671-billion-parameter reasoning model. Consistent wi..."
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: "Therefore, we selected the \textit{DeepSeek-R1 Distilled} variant of \textit{LLaMA 3-8B} (known as \textbf{DeepSeek-R1-Distill-Llama-8B}) as our base model. This model was further fine-tuned on a high-quality reasoning dataset distilled from DeepSeek-R1, a powerful 671-billion-parameter reasoning model. Consistent w..."

## CH2-108
- ID: CH2-108
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:690
- Issue Type: Consistency
- Original: "Step 1 fine-tunes the base model (\textbf{LLaMA3-8B}). We train 3,407,872 parameters out of 8,051,232,768 total parameters (trainable ratio: 0.0423\%)."
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: "Step 1 fine-tunes the base model (\textbf{LLaMA 3-8B}). We train 3,407,872 parameters out of 8,051,232,768 total parameters (trainable ratio: 0.0423\%)."

## CH2-109
- ID: CH2-109
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:396
- Issue Type: Consistency
- Original: "On the other hand, in addition to Cosine Similarity (a sentence-level metric), we also employ \textbf{BERTScore} (\cite{zhang2019bertscore}), which measures \textit{token-level} semantic similarity using contextual embeddings and cosine similarity. Let $Y=\{y_i\}$ denote the set of token embeddings in the generated ..."
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: "On the other hand, in addition to Cosine Similarity (a sentence-level metric), we also employ \textbf{BERTScore} (\cite{zhang2019bertscore}), which measures \textit{token-level} semantic similarity using contextual embeddings and cosine similarity. Let $Y=\{y_i\}$ denote the set of token embeddings in the generated ..."

## CH2-110
- ID: CH2-110
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:982
- Issue Type: Consistency
- Original: "All reported metrics improve after fine-tuning, with the largest gains in the ``answer'' segment. For example, the cosine-similarity gain is 2.872\% for ``answer'' versus 0.7575\% for ``full'' text. The gain in $BertScore_{answer}^{F1}$ is 1.5207\%, roughly twice the $BertScore_{full}^{F1}$ gain (0.7673\%). These di..."
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: "All reported metrics improve after fine-tuning, with the largest gains in the ``answer'' segment. For example, the cosine-similarity gain is 2.872\% for ``answer'' versus 0.7575\% for ``full'' text. The gain in $BERTScore_{answer}^{F1}$ is 1.5207\%, roughly twice the $BERTScore_{full}^{F1}$ gain (0.7673\%). These di..."

## CH2-111
- ID: CH2-111
- File: Chapter2/chapter2.tex
- Location: Chapter2/chapter2.tex:1042
- Issue Type: Consistency
- Original: "$BertScore_{full}^{Precision}$&0.8647&0.8575&0.8441\\ &$({4224.71}^{***})$&$({4297.97}^{***})$&$({25.10}^{***})$\\ $BertScore_{think}^{Precision}$&0.8653&0.8574&0.917\\ &$({4266.65}^{***})$&$({4402.03}^{***})$&$({28.16}^{***})$\\ $BertScore_{answer}^{Precision}$&0.8579&0.8441&\textbf{1.6387}\\ &$({4049.85}^{***})$&$..."
- Revision Rationale: Standardized chapter-wide terminology and notation to keep method labels, metric names, and policy-rate wording internally consistent.
- Revised Version: "$BERTScore_{full}^{Precision}$&0.8647&0.8575&0.8441\\ &$({4224.71}^{***})$&$({4297.97}^{***})$&$({25.10}^{***})$\\ $BERTScore_{think}^{Precision}$&0.8653&0.8574&0.917\\ &$({4266.65}^{***})$&$({4402.03}^{***})$&$({28.16}^{***})$\\ $BERTScore_{answer}^{Precision}$&0.8579&0.8441&\textbf{1.6387}\\ &$({4049.85}^{***})$&$..."

Total entries: 111
