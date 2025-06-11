import json
import random

import pandas as pd

from open_r1.generate import generate_responses
from utils import save_output


def generate_new_response(input_prompt_file: str | list[dict] | pd.DataFrame, model_path: str, output_file: str = None, sample_size=None) -> list[dict]:

    if isinstance(input_prompt_file, pd.DataFrame):
        lines = input_prompt_file.to_dict(orient='records')
    elif isinstance(input_prompt_file, list):
        lines = input_prompt_file
    elif isinstance(input_prompt_file, str):
        lines = open(input_prompt_file, 'r').readlines()
    else:
        raise ValueError("input_prompt_file must be a DataFrame, list of dicts, or a file path string.")

    total = len(lines)
    print(f"✅ Total prompts available: {total}")

    if sample_size:
        lines = random.sample(lines, sample_size)
        print(f"🎯 Sampled {sample_size} prompts.")

    # 输出收集
    output_index = []
    output_target = []
    output_generated = []
    failed_index = []

    batch_size = 20
    index = -1
    s = 0  # 成功数量
    f = 0  # 失败数量
    n = 0  # 总尝试数

    batch_prompts = []
    batch_targets = []
    batch_indices = []

    for line in lines:
        index += 1
        item = json.loads(line)
        prompt = item["prompt"]
        target = item.get("response", "")

        batch_prompts.append(prompt)
        batch_targets.append(target)
        batch_indices.append(index)

        if len(batch_prompts) == batch_size:
            try:
                batch_outputs = generate_responses(batch_prompts, model_path, max_new_tokens=8192)
                for i, generated in enumerate(batch_outputs):
                    output_index.append(batch_indices[i])
                    output_target.append(batch_targets[i])
                    output_generated.append(generated)
                    print(f"✅ No. {batch_indices[i]} : Suc {s + 1}, Fail {f}")
                    s += 1
                    n += 1
            except Exception as e:
                print(f"❌ Batch starting at index {batch_indices[0]} failed: {e}")
                failed_index.extend(batch_indices)
                f += len(batch_prompts)
                n += len(batch_prompts)
            finally:
                batch_prompts = []
                batch_targets = []
                batch_indices = []

    # 处理最后一个不满 batch 的剩余
    if batch_prompts:
        try:
            batch_outputs = generate_responses(batch_prompts, model_path, max_new_tokens=8192)
            for i, generated in enumerate(batch_outputs):
                output_index.append(batch_indices[i])
                output_target.append(batch_targets[i])
                output_generated.append(generated)
                print(f"✅ No. {batch_indices[i]} : Suc {s + 1}, Fail {f}")
                s += 1
                n += 1
        except Exception as e:
            print(f"❌ Final batch starting at index {batch_indices[0]} failed: {e}")
            failed_index.extend(batch_indices)
            f += len(batch_prompts)
            n += len(batch_prompts)

    # 写出结果到 output_file
    output_dicts = []
    for idx, tgt, gen in zip(output_index, output_target, output_generated):
        try:
            base_data = json.loads(lines[idx])
        except (IndexError, json.JSONDecodeError) as e:
            print(f"⚠️ Skipping index {idx} due to error: {e}")
            continue

        # Overwrite or add the new fields
        base_data.update({
            "index": idx,
            "target": tgt,
            "generated": gen
        })
        output_dicts.append(base_data)

    if output_file:
        save_output(output_dicts, output_file)
        print(f"✅ Output saved to {output_file}")
    print(f"🎯 Finished. Total: {n}, Success: {s}, Failed: {f}")
    if failed_index:
        print(f"❗ Failed indices: {failed_index}")
        pd.DataFrame({"Failed": failed_index}).to_csv(output_file.replace(".xlsx", "_failed.csv"), index=False)

    print(f"✅ Finished processing all prompts.")
    print(f"✅ Total: {total}, Suc: {s}, Fail: {f}")
    return output_dicts

