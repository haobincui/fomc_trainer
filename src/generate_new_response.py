import json
import os
import random
from pathlib import Path
from typing import Callable

import pandas as pd

from open_r1.generate import generate_responses, get_system_prompt
from open_r1.provenance import sha256_text
from open_r1.validator.loo_generation_spec import derive_row_seed
from utils import save_output


ROW_SEED_POLICY_BATCH = "batch-seed-v1"
ROW_SEED_POLICY_SAMPLE = "sample-id-sha256-v1"


def _normalise_generation_result(result: object) -> dict:
    if isinstance(result, str):
        return {
            "text": result,
            "finish_reason": None,
            "stop_reason": None,
            "prompt_token_count": None,
            "prompt_preflight_token_count": None,
            "output_token_count": None,
            "input_was_truncated": None,
        }
    if isinstance(result, dict):
        return {
            "text": str(result.get("text") or ""),
            "finish_reason": result.get("finish_reason"),
            "stop_reason": result.get("stop_reason"),
            "prompt_token_count": result.get("prompt_token_count"),
            "prompt_preflight_token_count": result.get("prompt_preflight_token_count"),
            "output_token_count": result.get("output_token_count"),
            "input_was_truncated": result.get("input_was_truncated"),
        }
    return {
        "text": "",
        "finish_reason": "invalid_result_type",
        "stop_reason": None,
        "prompt_token_count": None,
        "prompt_preflight_token_count": None,
        "output_token_count": None,
        "input_was_truncated": None,
    }


def generate_new_response(
    input_prompt_file: str | list[dict] | pd.DataFrame,
    model_path: str,
    output_file: str = None,
    batch_size: int = None,
    sample_size: int = None,
    *,
    seed: int | None = None,
    replicate_id: str | int | None = None,
    temperature: float = 0.7,
    top_p: float = 0.9,
    max_new_tokens: int = 8192,
    max_model_len: int = 16384,
    tokenizer_path: str | None = None,
    system_prompt: str | None = None,
    seed_policy: str = ROW_SEED_POLICY_BATCH,
    generation_metadata: dict | None = None,
    fail_closed: bool = False,
    progress_callback: Callable[[list[dict], list[int]], None] | None = None,
) -> list[dict]:
    if seed_policy not in {ROW_SEED_POLICY_BATCH, ROW_SEED_POLICY_SAMPLE}:
        raise ValueError(
            f"Unsupported seed_policy {seed_policy!r}; expected "
            f"{ROW_SEED_POLICY_BATCH!r} or {ROW_SEED_POLICY_SAMPLE!r}"
        )
    if seed_policy == ROW_SEED_POLICY_SAMPLE and seed is None:
        raise ValueError("sample-id-sha256-v1 requires a non-null base seed")
    effective_system_prompt = (
        get_system_prompt() if system_prompt is None else system_prompt
    )
    if (
        not isinstance(effective_system_prompt, str)
        or not effective_system_prompt.strip()
    ):
        raise ValueError("system_prompt must be a non-empty string")

    if output_file is not None:
        output_suffixes = {".jsonl", ".xlsx", ".csv"}
        second_suffix = Path(model_path).suffix.lower()
        third_suffix = Path(output_file).suffix.lower()
        if second_suffix in output_suffixes and third_suffix not in output_suffixes:
            model_path, output_file = output_file, model_path

    if isinstance(input_prompt_file, pd.DataFrame):
        lines = input_prompt_file.to_dict(orient="records")
    elif isinstance(input_prompt_file, list):
        lines = input_prompt_file
    elif isinstance(input_prompt_file, str):
        with open(input_prompt_file, "r", encoding="utf-8") as f:
            lines = [json.loads(line) for line in f]
    else:
        raise ValueError(
            "input_prompt_file must be a DataFrame, list of dicts, or a file path string."
        )

    total = len(lines)
    print(f"✅ Total prompts available: {total}")

    if sample_size:
        lines = random.Random(seed).sample(lines, sample_size)
        print(f"🎯 Sampled {sample_size} prompts.")

    # 输出收集
    output_index = []
    output_target = []
    output_generated = []
    output_generation_seeds = []
    output_generation_metadata = []
    output_dicts = []
    failed_index = []

    if not batch_size:
        batch_size = 20
    index = -1
    s = 0  # 成功数量
    f = 0  # 失败数量
    n = 0  # 总尝试数

    batch_prompts = []
    batch_targets = []
    batch_indices = []
    batch_number = 0

    def materialize_batch(start: int, expected_indices: list[int]) -> None:
        """Materialize and optionally persist one completed inference batch."""

        batch_rows = []
        for idx, tgt, gen, generation_seed, result_metadata in zip(
            output_index[start:],
            output_target[start:],
            output_generated[start:],
            output_generation_seeds[start:],
            output_generation_metadata[start:],
            strict=True,
        ):
            base_data = dict(lines[idx])
            if generation_metadata:
                base_data.update(generation_metadata)

            source_index = base_data.get("source_index", base_data.get("index", idx))
            base_data.update(
                {
                    "index": base_data.get("index", idx),
                    "source_index": source_index,
                    "generation_position": base_data.get("generation_position", idx),
                    "source_prompt_sha256": sha256_text(
                        str(base_data.get("prompt") or "")
                    ),
                    "target": tgt,
                    "generated": gen,
                    "generated_sha256": sha256_text(gen),
                    "replicate_id": (
                        None if replicate_id is None else str(replicate_id)
                    ),
                    "generation_seed": generation_seed,
                    "generation_seed_policy": seed_policy,
                    "generation_model": model_path,
                    "generation_tokenizer": tokenizer_path or model_path,
                    "generation_system_prompt_sha256": sha256_text(
                        effective_system_prompt
                    ),
                    "generation_batch_size": int(batch_size),
                    "decoding_temperature": float(temperature),
                    "decoding_top_p": float(top_p),
                    "max_new_tokens": int(max_new_tokens),
                    "max_model_len": int(max_model_len),
                    "generation_finish_reason": result_metadata.get(
                        "finish_reason"
                    ),
                    "generation_stop_reason": result_metadata.get("stop_reason"),
                    "prompt_token_count": result_metadata.get(
                        "prompt_token_count"
                    ),
                    "prompt_preflight_token_count": result_metadata.get(
                        "prompt_preflight_token_count"
                    ),
                    "output_token_count": result_metadata.get(
                        "output_token_count"
                    ),
                    "input_was_truncated": result_metadata.get(
                        "input_was_truncated"
                    ),
                }
            )
            batch_rows.append(base_data)

        output_dicts.extend(batch_rows)
        if progress_callback is not None:
            progress_callback(batch_rows, list(expected_indices))

    for item in lines:
        index += 1
        prompt = item["prompt"]
        target = item.get("response", "")

        batch_prompts.append(prompt)
        batch_targets.append(target)
        batch_indices.append(index)

        if len(batch_prompts) == batch_size:
            output_start = len(output_index)
            completed_batch_indices = list(batch_indices)
            batch_seed = None if seed is None else seed + batch_number
            row_seeds = (
                [
                    derive_row_seed(seed, lines[row_index].get("sample_id"))
                    for row_index in batch_indices
                ]
                if seed_policy == ROW_SEED_POLICY_SAMPLE
                else None
            )
            try:
                batch_outputs = generate_responses(
                    batch_prompts,
                    model_path,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    seed=None if row_seeds is not None else batch_seed,
                    row_seeds=row_seeds,
                    return_metadata=True,
                    max_model_len=max_model_len,
                    tokenizer_path=tokenizer_path,
                    system_prompt=effective_system_prompt,
                )
                for i, result in enumerate(batch_outputs):
                    metadata = _normalise_generation_result(result)
                    generated = metadata["text"]
                    if (
                        not isinstance(generated, str)
                        or not generated.strip()
                        or generated.strip() == "Failed"
                    ):
                        failed_index.append(batch_indices[i])
                        f += 1
                        n += 1
                        print(f"❌ No. {batch_indices[i]} : Suc {s}, Fail {f}")
                        continue
                    output_index.append(batch_indices[i])
                    output_target.append(batch_targets[i])
                    output_generated.append(generated)
                    output_generation_seeds.append(
                        row_seeds[i] if row_seeds is not None else batch_seed
                    )
                    output_generation_metadata.append(metadata)
                    print(f"✅ No. {batch_indices[i]} : Suc {s + 1}, Fail {f}")
                    s += 1
                    n += 1
                materialize_batch(output_start, completed_batch_indices)
            except Exception as e:
                if fail_closed:
                    raise RuntimeError(
                        f"Generation batch starting at index {batch_indices[0]} failed"
                    ) from e
                print(f"❌ Batch starting at index {batch_indices[0]} failed: {e}")
                failed_index.extend(batch_indices)
                f += len(batch_prompts)
                n += len(batch_prompts)
            finally:
                batch_prompts = []
                batch_targets = []
                batch_indices = []
                batch_number += 1

    # 处理最后一个不满 batch 的剩余
    if batch_prompts:
        output_start = len(output_index)
        completed_batch_indices = list(batch_indices)
        batch_seed = None if seed is None else seed + batch_number
        row_seeds = (
            [
                derive_row_seed(seed, lines[row_index].get("sample_id"))
                for row_index in batch_indices
            ]
            if seed_policy == ROW_SEED_POLICY_SAMPLE
            else None
        )
        try:
            batch_outputs = generate_responses(
                batch_prompts,
                model_path,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                seed=None if row_seeds is not None else batch_seed,
                row_seeds=row_seeds,
                return_metadata=True,
                max_model_len=max_model_len,
                tokenizer_path=tokenizer_path,
                system_prompt=effective_system_prompt,
            )
            for i, result in enumerate(batch_outputs):
                metadata = _normalise_generation_result(result)
                generated = metadata["text"]
                if (
                    not isinstance(generated, str)
                    or not generated.strip()
                    or generated.strip() == "Failed"
                ):
                    failed_index.append(batch_indices[i])
                    f += 1
                    n += 1
                    print(f"❌ No. {batch_indices[i]} : Suc {s}, Fail {f}")
                    continue
                output_index.append(batch_indices[i])
                output_target.append(batch_targets[i])
                output_generated.append(generated)
                output_generation_seeds.append(
                    row_seeds[i] if row_seeds is not None else batch_seed
                )
                output_generation_metadata.append(metadata)
                print(f"✅ No. {batch_indices[i]} : Suc {s + 1}, Fail {f}")
                s += 1
                n += 1
            materialize_batch(output_start, completed_batch_indices)
        except Exception as e:
            if fail_closed:
                raise RuntimeError(
                    f"Final generation batch starting at index {batch_indices[0]} failed"
                ) from e
            print(f"❌ Final batch starting at index {batch_indices[0]} failed: {e}")
            failed_index.extend(batch_indices)
            f += len(batch_prompts)
            n += len(batch_prompts)

    if output_file:
        output_parent = os.path.dirname(output_file)
        if output_parent:
            os.makedirs(output_parent, exist_ok=True)
        save_output(output_dicts, output_file)
        print(f"✅ Output saved to {output_file}")
    print(f"🎯 Finished. Total: {n}, Success: {s}, Failed: {f}")
    if failed_index and output_file:
        print(f"❗ Failed indices: {failed_index}")
        pd.DataFrame({"Failed": failed_index}).to_csv(
            output_file.split(".")[0] + "_failed.csv", index=False
        )

    print("✅ Finished processing all prompts.")
    print(f"✅ Total: {total}, Suc: {s}, Fail: {f}")
    return output_dicts
