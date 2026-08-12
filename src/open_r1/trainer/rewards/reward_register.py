import os
from functools import partial, update_wrapper
from typing import Callable

from open_r1.trainer.rewards.reward_funcs.online_reward import (
    answer_reward,
    combined_reward,
    reasoning_reward,
)
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v2 import (
    grounded_analysis_reward_v2,
)
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v3 import (
    grounded_analysis_reward_v3,
)
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v3_deepseek_high import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MAX_OUTPUT_TOKENS,
    DEEPSEEK_MODEL,
    grounded_analysis_reward_v3_deepseek_high,
)
from open_r1.trainer.rewards.reward_funcs.analysis_reward_v3_deepseek_low import (
    DEEPSEEK_BASE_URL as DEEPSEEK_LOW_BASE_URL,
    DEEPSEEK_MAX_OUTPUT_TOKENS as DEEPSEEK_LOW_MAX_OUTPUT_TOKENS,
    DEEPSEEK_MODEL as DEEPSEEK_LOW_MODEL,
    DEEPSEEK_TIMEOUT600_SECONDS as DEEPSEEK_LOW_TIMEOUT600_SECONDS,
    grounded_analysis_reward_v3_deepseek_low,
    grounded_analysis_reward_v3_deepseek_low_timeout600,
)
from open_r1.trainer.rewards.reward_funcs.decision_reward_v2 import (
    decision_dense_reward_v2,
)
from open_r1.trainer.rewards.reward_funcs.decision_reward_v3 import (
    decision_dense_reward_v3,
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
        "tokenizer_path": getattr(script_args, "judge_tokenizer_path", None),
        "max_model_len": getattr(script_args, "judge_max_model_len", None),
        "max_completion_tokens": getattr(
            script_args, "judge_max_completion_tokens", None
        ),
        "timeout": getattr(script_args, "judge_timeout", None),
        "max_retries": getattr(script_args, "judge_max_retries", 3),
        "backoff_seconds": getattr(script_args, "judge_backoff_seconds", 1.0),
        "verbose": getattr(script_args, "judge_verbose", None),
        "sleep_seconds": getattr(script_args, "judge_sleep_seconds", None),
        "api_key_env": getattr(script_args, "judge_api_key_env", None),
    }

    if training_args is not None and getattr(script_args, "save_reward", False):
        reward_kwargs["save_path"] = os.path.join(training_args.output_dir, "reward.jsonl")

    return reward_kwargs


def get_reward_funcs(script_args, training_args=None) -> list[Callable]:
    online_reward_kwargs = _online_reward_kwargs(script_args, training_args)
    deepseek_high_reward_kwargs = {
        **online_reward_kwargs,
        "url": DEEPSEEK_BASE_URL,
        "model": DEEPSEEK_MODEL,
        "max_completion_tokens": DEEPSEEK_MAX_OUTPUT_TOKENS,
        "timeout": 420,
        "max_retries": 2,
        "backoff_seconds": 2.0,
        "api_key_env": "DEEPSEEK_API_KEY",
    }
    deepseek_low_reward_kwargs = {
        **online_reward_kwargs,
        "url": DEEPSEEK_LOW_BASE_URL,
        "model": DEEPSEEK_LOW_MODEL,
        "max_completion_tokens": DEEPSEEK_LOW_MAX_OUTPUT_TOKENS,
        "timeout": 420,
        "max_retries": 2,
        "backoff_seconds": 2.0,
        "api_key_env": "DEEPSEEK_API_KEY",
    }
    deepseek_low_timeout600_reward_kwargs = {
        **deepseek_low_reward_kwargs,
        "timeout": DEEPSEEK_LOW_TIMEOUT600_SECONDS,
    }
    persisted_reward_kwargs = (
        {"save_path": online_reward_kwargs["save_path"]}
        if "save_path" in online_reward_kwargs
        else {}
    )
    REWARD_FUNCS_REGISTRY = {
        "grounded_analysis_v2": update_wrapper(
            partial(grounded_analysis_reward_v2, **online_reward_kwargs),
            grounded_analysis_reward_v2,
        ),
        "grounded_analysis_v3": update_wrapper(
            partial(grounded_analysis_reward_v3, **online_reward_kwargs),
            grounded_analysis_reward_v3,
        ),
        "grounded_analysis_v3_deepseek_high": update_wrapper(
            partial(
                grounded_analysis_reward_v3_deepseek_high,
                **deepseek_high_reward_kwargs,
            ),
            grounded_analysis_reward_v3_deepseek_high,
        ),
        "grounded_analysis_v3_deepseek_low": update_wrapper(
            partial(
                grounded_analysis_reward_v3_deepseek_low,
                **deepseek_low_reward_kwargs,
            ),
            grounded_analysis_reward_v3_deepseek_low,
        ),
        "grounded_analysis_v3_deepseek_low_timeout600": update_wrapper(
            partial(
                grounded_analysis_reward_v3_deepseek_low_timeout600,
                **deepseek_low_timeout600_reward_kwargs,
            ),
            grounded_analysis_reward_v3_deepseek_low_timeout600,
        ),
        "decision_dense_v2": update_wrapper(
            partial(decision_dense_reward_v2, **persisted_reward_kwargs),
            decision_dense_reward_v2,
        ),
        "decision_dense_v3": update_wrapper(
            partial(decision_dense_reward_v3, **persisted_reward_kwargs),
            decision_dense_reward_v3,
        ),
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
