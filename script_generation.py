import json
import logging
import os
import random
import re
from pathlib import Path

import pandas as pd

from create_prompt.prompt_template import DecisionPrompt
from generate_new_response import generate_new_response
from open_r1.validator.cos.cos_calc import cosine_similarity_calc
from open_r1.validator.cos.embedding_model import EmbeddingModel
from utils import save_output


def calc_cos(input_prompt_file: str, output_file: str, model_path: str, sample_size: int = 10):
    df = pd.read_excel(output_file.replace(".xlsx", "_generated.xlsx"))

    embed_model = EmbeddingModel(model_path)
    df['cos'] = df.apply(lambda x: cosine_similarity_calc(x['target'], x['generated'], embed_model), axis=1)
    df.to_excel(output_file, index=False)
    print(f"✅ Finished COS for all prompts.")


def bootstrap_cos(input_prompt_file: str, output_file: str, model_path: str, sample_size: int = 10):

    df = pd.read_excel(output_file.replace(".xlsx", "_generated.xlsx"))
    cos_list = df['cos'].tolist()
    cos_result = []
    for i in range(sample_size):

        random_indexs = random.sample(range(len(cos_list)), 20)
        bootstrap_cos = []
        for j in random_indexs:
            bootstrap_cos.append(cos_list[j])
        bootstrap_cos = sum(bootstrap_cos) / len(bootstrap_cos)
        cos_result.append(bootstrap_cos)

    pd.DataFrame({
        "cos": cos_result
    }).to_excel(output_file.replace(".xlsx", "_bootstrap_cos.xlsx"), index=False)
    logging.info(f"✅ Finished Bootstrop Cos.")





def _parse_final_vote(text):
    """Extracts content inside the last \\boxed{} after </think>."""
    if not isinstance(text, str):
        return ""
    match = re.search(r"\\boxed\{(.*?)\}", text.split("</think>")[-1], re.DOTALL)
    return match.group(1).strip() if match else ""


def run_eval_decision(input_file: str | pd.DataFrame) -> float:
    if isinstance(input_file, pd.DataFrame):
        df = input_file
    elif input_file.endswith(".xlsx"):
        df = pd.read_excel(input_file)
    elif input_file.endswith(".jsonl"):
        df = pd.read_json(input_file, lines=True)
    else:
        raise ValueError(f"Unsupported file format: {input_file}")

    results = []
    for idx, row in df.iterrows():
        row_dict = row.to_dict()
        target_vote = row['rate_change']
        generated = row.get("generated", "")
        generated_vote = _parse_final_vote(generated)
        match_result = 1 if target_vote == generated_vote else 0

        row_dict.update({
            "target_vote": target_vote,
            "generated_vote": generated_vote,
            "match_result": match_result
        })
        results.append(row_dict)

    result_df = pd.DataFrame(results)

    output_path = Path(input_file).with_stem(Path(input_file).stem + "_result")
    result_df.to_excel(output_path.with_suffix(".xlsx"), index=False)

    total = len(result_df)
    correct = result_df['match_result'].sum()
    accuracy = correct / total * 100

    print(f"\n🎯 Total samples: {total}")
    print(f"✅ Correct predictions: {correct}")
    print(f"📊 Accuracy: {accuracy:.2f}%")

    return accuracy



def bootstrap_acc(input_excels: list[str], output_file: str) -> dict:
    """
    Compute accuracy statistics by category from FOMC decision prediction results.

    Args:
        input_excels (list[str]): List of Excel file paths ending with `_train_result.xlsx` or `_eval_result.xlsx`.
        output_file (str): Path to the output Excel file.

    Returns:
        dict: Nested dictionary with model-tag as keys and accuracy values by category.
    """
    full_result = {}

    # 使用 ExcelWriter 管理多个 sheet
    with pd.ExcelWriter(output_file, engine='openpyxl', mode='w') as writer:
        for input_excel in input_excels:
            df = pd.read_excel(input_excel)

            if input_excel.endswith('_train_result.xlsx'):
                tag = 'train'
            elif input_excel.endswith('_eval_result.xlsx'):
                tag = 'eval'
            else:
                raise ValueError(f"Unsupported file name format: {input_excel}")

            model_name = input_excel.split('/')[-2]
            sheet_name = f"{model_name}-{tag}"
            acc_result = {}

            # Filter valid entries (non-empty generated_vote)
            df_valid = df[df['generated_vote'].astype(str).str.strip() != ""]

            # Total accuracy
            total_count = len(df_valid)
            correct_total = df_valid['match_result'].sum()
            acc_result['total'] = correct_total / total_count if total_count > 0 else None

            # Category-level accuracy
            for category in ['Cut', 'No', 'Raise']:
                df_cat = df_valid[df_valid['target_vote'].str.startswith(category)]
                total_cat = len(df_cat)
                correct_cat = df_cat['match_result'].sum()
                acc_result[category.lower()] = correct_cat / total_cat if total_cat > 0 else None

            # 存入总字典
            full_result[sheet_name] = acc_result

            # 写入当前 sheet
            pd.DataFrame([acc_result]).to_excel(writer, sheet_name=sheet_name, index=False)
            print(f"✅ Finished {tag} for model {model_name}, saved in sheet '{sheet_name}'")

    return full_result



def run_stage1_generation():
    #%% stage 1
    input_prompt_file = "dataset/training_data/fomc_qa/fomc_qa_test.jsonl"
    output_file = "output/validation/cos_20250523_2/stage1_cos_raw.xlsx"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # model_path = 'output/merged/llama_sft_20250522'
    model_path = 'models/DeepSeek-R1-Distill-Llama-8B'
    generate_new_response(input_prompt_file, output_file, model_path)
    print("finshed stage 1 generation")


def run_stage2_synthetic_generation():

    input_prompt_file = "dataset/training_data/synthetic_text/synthetic_text_20250520_reason_filted_test.jsonl"
    output_file = "output/valiation/generation_stage2_synthetic/synthetic_text_raw_20250526.xlsx"
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    model_path = 'output/merged/llama_sft_synthetic_20250526'
    # model_path = 'models/DeepSeek-R1-Distill-Llama-8B'
    
    generate_new_response(input_prompt_file, output_file, model_path)
    print("finished stage 2 synthetic generation")



def assemble_synthetic_data(section_file: str, output_file = None) -> list[dict]:
    """Combine section-level JSONL data into meeting-level minutes.

    Each input line must contain:
    meeting_date, section_name, rate_change, section_detail

    Output fields:
    - meeting_date
    - minutes (synthetic full text)
    - rate_change
    """

    def _parse_section_detail(detail: str):
        text = detail.split("<answer>")[-1]
        text = text.split("</answer>")[0]
        text = text.replace("<answer>", "").replace("</answer>", "").strip()
        return text

    def _combine_sections(section_name, section_detail):
        return f"{section_name}\n{_parse_section_detail(section_detail)}\n\n"

    meeting_dict = {}

    # 读取 section 级别数据

    with open(section_file, 'r', encoding='utf-8') as fin:
        for line in fin:
            line_dict = json.loads(line)
            date = line_dict['meeting_date']
            section = line_dict['section_name']
            detail = line_dict['generated']
            rate = line_dict['rate_change']
            target_rate = line_dict['target_rate']

            if date not in meeting_dict:
                meeting_dict[date] = {
                    'sections': {},
                    'rate_change': rate,
                    'target_rate': target_rate
                }
            meeting_dict[date]['sections'][section] = detail

    print(f"📊 Total meetings found: {len(meeting_dict)}")
    results  = []
    for meeting_date, content in meeting_dict.items():
        minutes = f"FOMC Minutes for {meeting_date}\n\n"
        for section_name, section_detail in content['sections'].items():
            minutes += _combine_sections(section_name, section_detail)

        result_dict = {
            'meeting_date': meeting_date,
            'minutes': minutes.strip(),
            'rate_change': content['rate_change'],
            'target_rate': content['target_rate'],
        }
        results.append(result_dict)

    if output_file:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print(f"📂 Saving assembled data to {output_file}")
        save_output(results, output_file)
        print(f"🎉 Finished assembling synthetic data， saved in {output_file}.")
    return  results


def minutes_to_decision_prompt(minutes_file: str | list[dict] | pd.DataFrame, output_file = None) -> list[dict]:
    prompt_template = DecisionPrompt()

    def _create_prompt(row):
        meeting_date = row['meeting_date']
        analysis = row['minutes']
        target_rate = row['target_rate']

        # Create the prompt using the template
        prompt = prompt_template.reformat_prompt(
            current_analysis=analysis,
            current_rate=target_rate,
            meeting_date=meeting_date
        )
        return prompt

    if isinstance(minutes_file, list):
        df = pd.DataFrame(minutes_file)
    elif isinstance(minutes_file, pd.DataFrame):
        df = minutes_file
    elif minutes_file.endswith(".jsonl"):
        df = pd.read_json(minutes_file, lines=True)
    elif minutes_file.endswith(".xlsx"):
        df = pd.read_excel(minutes_file)
    else:
        raise ValueError("Unsupported file format. Please provide a .jsonl or .xlsx file.")

    df['prompt'] = df.apply(_create_prompt, axis=1)
    if output_file:
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        print(f"📂 Saving decision prompts to {output_file}")
        save_output(df, output_file)
        print(f"🎉 Finished creating decision prompts， saved in {output_file}.")
    return df.to_dict(orient="records")






def run_stage2_synthetic_full():
    """Generate synthetic data for stage 2 to explore the decision-making accuracy.
    each output file (.jsonl and .xlsx) contains sections, meeting_dates, and rate_changes.
    generation {total} synthetic file samples.
    saved in output dir.
    """
    input_prompt_file = "dataset/raw_data/synthetic_text_20250520.jsonl"
    model_name_map = {
        "output/merged/llama_sft_synthetic_20250526": "ft",
        'models/DeepSeek-R1-Distill-Llama-8B': "base"
    }
    model_path = 'output/merged/llama_sft_synthetic_20250526'
    # model_path = 'models/DeepSeek-R1-Distill-Llama-8B'

    total = 100
    for i in range(total):
    
        output_file = f"output/valiation/generation_stage2_synthetic/synthetic_for_decision/{model_name_map[model_path]}_model/synthetic_text_20250601_{i}.jsonl"

        os.makedirs(os.path.dirname(output_file), exist_ok=True)


        print(f"using model {model_path}")
        print(f"Output file: {output_file}")
        print(f"Generating synthetic data for {i+1}/{total}...")
        # generate_new_response(input_prompt_file, output_file, model_path)
        synthetic_minutes = assemble_synthetic_data(output_file)
        synthetic_minutes_with_prompt = minutes_to_decision_prompt(synthetic_minutes)
        output_file = f"output/valiation/generation_stage2_synthetic/synthetic_for_decision/{model_name_map[model_path]}_model/results/synthetic_text_20250601_{i}.xlsx"
        generate_new_response(synthetic_minutes_with_prompt, model_path)


        print(f"saved in {output_file.replace('.xlsx', '_merged.jsonl')}")
        print(f"Finished {i}")
    print(f"🎉 Finished stage 2 synthetic full generation, Total {total}.")





if __name__ == '__main__':
    #%% stage 1
    # run_stage1_generation()


    #%% stage 2 -synthetic
    # run_stage2_synthetic_generation()

    # %%% synthetic data for stage 2 input
    run_stage2_synthetic_full()
    


    #%%% stage 2 decision-making 20250531

    # input_prompt_file = "dataset/training_data/decision_grpo/decision_grpo_20250531_grpo_eval.jsonl"
    # output_file = "output/validation/generation_stage2_decision/eval_result/decision_grpo_stage2_raw_20250531_eval.xlsx"

    # # input_prompt_file = "dataset/training_data/decision_grpo/decision_grpo_20250529_grpo_eval.jsonl"
    # # output_file = "output/validation/generation_stage2_decision/eval_result/decision_grpo_raw_20250529_eval.xlsx"
    # os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # model_path_1 = 'output/merged/llama_grpo_decision_cp1100_20250530'
    # model_path_2 = 'models/DeepSeek-R1-Distill-Llama-8B'
    # model_path_3 = "output/merged/llama_sft_20250522"
    # model_paths = [model_path_1, model_path_3]

    # version_date = '20250531'
    # # input_prompt_file_map = {
    # #     "eval": "dataset/training_data/decision_grpo/decision_grpo_20250531_grpo_eval.jsonl",
    # #     "train":  "dataset/training_data/decision_grpo/decision_grpo_20250531_grpo_train.jsonl"
    # # }
    # # sample_size = 50

    # for model_path in model_paths:
    #     model_name = os.path.basename(model_path.rstrip("/"))  # get last part of the path
    #     base_output_dir = f"output/validation/generation_stage2_decision/eval_result/{version_date}/{model_name}"
    #     os.makedirs(base_output_dir, exist_ok=True)
    #     for i in range(1000):
    #         for key, input_path in input_prompt_file_map.items():
    #             output_file_xlsx = os.path.join(base_output_dir, f"decision_grpo_{key}_{i}.xlsx")
    #             output_file_jsonl = output_file_xlsx.replace(".xlsx", ".jsonl")

    #             # === Step 1: generate response from model
    #             generate_new_response(input_path, output_file_xlsx, model_path, sample_size=sample_size)

    #             # === Step 2: convert to .xlsx if needed
    #             jsonl_to_xlsx(output_file_jsonl, output_file_xlsx)

    #             # === Step 3: evaluate
    #             run_eval_decision(output_file_xlsx)

    #             # time.sleep(5)

    #             print(f"✅ Finished {key} for {model_name}")
    #         output_acc_file = os.path.join(base_output_dir,f"summary_{i}.xlsx")
    #         bootstrap_acc(glob.glob(base_output_dir + "*_result.xlsx"), output_acc_file)

    # model_path_1 = 'output/merged/llama_grpo_decision_cp1100_20250530'
    # model_path_2 = 'models/DeepSeek-R1-Distill-Llama-8B'
    # model_path_3 = "output/merged/llama_sft_20250522"
    # model_paths = [model_path_2, model_path_3]

    # version_date = '20250531'

    # for model_path in model_paths:
    #     model_name = os.path.basename(model_path.rstrip("/"))  # get last part of the path
    #     base_output_dir = f"output/validation/generation_stage2_decision/eval_result/{version_date}/{model_name}"
    #     for key in ["eval", "train"]:
    #         output_file_xlsx = os.path.join(base_output_dir, f"decision_grpo_{key}.xlsx")
        

    #         run_eval_decision(output_file_xlsx)
    # print("finished all")

        
        





    

        







    


















