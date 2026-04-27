import os
from functools import partial, update_wrapper
from typing import Callable

from open_r1.trainer.rewards.reward_funcs.online_reward import (
    answer_reward,
    combined_reward,
    reasoning_reward,
)
from open_r1.trainer.rewards.reward_funcs.reward_funcs import (
    accuracy_reward,
    binary_code_reward,
    code_reward,
    format_reward,
    get_code_format_reward,
    get_cosine_scaled_reward,
    get_repetition_penalty_reward,
    ioi_code_reward,
    len_reward,
    rate_accuracy_reward,
    rate_format_reward,
    reasoning_steps_reward,
    tag_count_reward,
)


def _online_reward_kwargs(script_args, training_args=None) -> dict:
    reward_kwargs = {
        "url": getattr(script_args, "judge_url", None),
        "model": getattr(script_args, "judge_model", None),
        "timeout": getattr(script_args, "judge_timeout", None),
        "verbose": getattr(script_args, "judge_verbose", None),
        "sleep_seconds": getattr(script_args, "judge_sleep_seconds", None),
        "api_key_env": getattr(script_args, "judge_api_key_env", None),
    }

    if training_args is not None and getattr(script_args, "save_reward", False):
        reward_kwargs["save_path"] = os.path.join(training_args.output_dir, "reward.jsonl")

    return reward_kwargs


def get_reward_funcs(script_args, training_args=None) -> list[Callable]:
    online_reward_kwargs = _online_reward_kwargs(script_args, training_args)
    REWARD_FUNCS_REGISTRY = {
        "rate_accuracy": rate_accuracy_reward,
        "rate_format": rate_format_reward,
        "online": update_wrapper(partial(combined_reward, **online_reward_kwargs), combined_reward),
        "reasoning": update_wrapper(partial(reasoning_reward, **online_reward_kwargs), reasoning_reward),
        "answer": update_wrapper(partial(answer_reward, **online_reward_kwargs), answer_reward),
        "accuracy": accuracy_reward,
        "format": format_reward,
        "reasoning_steps": reasoning_steps_reward,
        "cosine": get_cosine_scaled_reward(
            min_value_wrong=script_args.cosine_min_value_wrong,
            max_value_wrong=script_args.cosine_max_value_wrong,
            min_value_correct=script_args.cosine_min_value_correct,
            max_value_correct=script_args.cosine_max_value_correct,
            max_len=script_args.cosine_max_len,
        ),
        "repetition_penalty": get_repetition_penalty_reward(
            ngram_size=script_args.repetition_n_grams,
            max_penalty=script_args.repetition_max_penalty,
        ),
        "length": len_reward,
        "code": update_wrapper(
            partial(
                code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(
                    script_args, "enforce_same_language", False
                ),
            ),
            code_reward,
        ),
        "binary_code": update_wrapper(
            partial(
                binary_code_reward,
                num_parallel=script_args.parallel_code_exec_per_proc,
                provider_type=script_args.code_provider,
                enforce_same_language=getattr(
                    script_args, "enforce_same_language", False
                ),
            ),
            binary_code_reward,
        ),
        "ioi_code": update_wrapper(
            partial(
                ioi_code_reward,
                test_batch_size=script_args.code_eval_test_batch_size,
                provider_type=getattr(script_args, "ioi_provider", "piston"),
            ),
            ioi_code_reward,
        ),
        "code_format": get_code_format_reward(language=script_args.code_language),
        "tag_count": tag_count_reward,
    }
    reward_funcs = [REWARD_FUNCS_REGISTRY[func] for func in script_args.reward_funcs]

    return reward_funcs
