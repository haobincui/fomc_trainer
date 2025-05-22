import json
import logging
import random

from open_r1.generate import generate_response
from validator.cos.cos_calc import cosine_similarity_calc
from validator.cos.embedding_model import EmbeddingModel


import pandas as pd

def validate_cos_stage1(input_prompt_file:str, output_file: str, model_path: str, sample_size: int = 10):
    lines = open(input_prompt_file, 'r').readlines()
    total = len(lines)
    logging.info(f"✅ Total prompts: {total}")
    # 统计信息初始化
    n = 0
    s = 0
    f = 0
    failed_index = []
    output_index = []
    output_target = []
    output_generated = []
    for line in lines:
        item = json.loads(line)
        prompt = item['prompt']
        index = item['index']
        target = item['response']
        try:
            generated = generate_response(prompt, model_path, max_new_tokens=8192)
            s += 1
            n += 1
            output_index.append(index)
            output_target.append(target)
            output_generated.append(generated)
            logging.info(f"✅ No. {index} : Suc {s}, Fail {f}: index {index}.")
        except Exception as e:
            logging.info(f"❌ No. {index} : Error: [{e}]")
            f += 1
            n += 1
            failed_index.append(index)
            continue
    pd.DataFrame({
        "index": output_index,
        "target": output_target,
        "generated": output_generated
    }).to_excel(output_file.replace(".xlsx", "_generated.xlsx"), index=False)

    pd.DataFrame({"Failed": failed_index}).to_csv(output_file.replace(".xlsx", "_failed.csv"), index=False)
    logging.info(f"✅ Finished processing all prompts.")
    logging.info(f"✅ Total: {total}, Suc: {s}, Fail: {f}")

    # calc cos
    df = pd.read_excel(output_file.replace(".xlsx", "_generated.xlsx"))

    embed_model = EmbeddingModel(model_path)
    df['cos'] = df.apply(lambda x: cosine_similarity_calc(x['target'], x['generated'], embed_model), axis=1)
    df.to_excel(output_file, index=False)
    logging.info(f"✅ Finished COS for all prompts.")

    # bootstrap cos

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
    logging.info(f"✅ Finished all.")

if __name__ == '__main__':
    input_prompt_file = "dataset/training_data/fomc_qa/fomc_qa_test.jsonl"
    output_file = "output/validation/stage1_cos.xlsx"

    model_path = './models/llama3-8b'
    validate_cos_stage1(input_prompt_file, output_file, model_path)
















