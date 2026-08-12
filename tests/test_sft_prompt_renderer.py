from __future__ import annotations

import json
from pathlib import Path

import pytest

from jobs.retrain_v2.chk1.prompt_projection import ANALYSIS_SFT_SYSTEM_PROMPT
from jobs.retrain_v2.sft_training_budget import build_training_sft_token_auditor
from open_r1.trainer.sft_prompt_renderer import (
    SftPromptRenderError,
    render_sft_prompt,
    tokenize_sft_text,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
TOKENIZER_PATH = REPO_ROOT / "models" / "DeepSeek-R1-Distill-Llama-8B"
SFT_RELEASE = (
    REPO_ROOT
    / "dataset"
    / "processed"
    / "retrain_v2"
    / "chk1_reasoning_compressed_flash_max_v1_20260805"
    / "analysis_sft"
)


class _BosCharacterTokenizer:
    bos_token = "<bos>"
    bos_token_id = 1
    eos_token = "<eos>"
    eos_token_id = 2

    def __init__(self, *, template_offset: int = 0, leading_bos: int = 1) -> None:
        self.template_offset = template_offset
        self.leading_bos = leading_bos

    def _encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        ids: list[int] = []
        while text.startswith(self.bos_token):
            ids.append(self.bos_token_id)
            text = text[len(self.bos_token) :]
        if add_special_tokens and not ids:
            ids.append(self.bos_token_id)
        if text.endswith(self.eos_token):
            text = text[: -len(self.eos_token)]
            suffix = [self.eos_token_id]
        else:
            suffix = []
        return ids + [ord(character) + 10 for character in text] + suffix

    def apply_chat_template(
        self, messages, *, tokenize: bool, add_generation_prompt: bool, **_kwargs
    ):
        assert add_generation_prompt is True
        rendered = self.bos_token * self.leading_bos + "".join(
            f"<{item['role']}>{item['content']}" for item in messages
        ) + "<assistant>"
        if not tokenize:
            return rendered
        ids = self._encode(rendered, add_special_tokens=False)
        if self.template_offset:
            ids[-1] += self.template_offset
        return ids

    def __call__(self, *, text: str):
        return {"input_ids": self._encode(text, add_special_tokens=True)}


def _messages() -> list[dict[str, str]]:
    return [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "prompt"},
    ]


def test_renderer_removes_literal_bos_and_preserves_canonical_ids() -> None:
    tokenizer = _BosCharacterTokenizer()

    rendered = render_sft_prompt(tokenizer, _messages())
    trl_ids = tokenize_sft_text(tokenizer, rendered)
    canonical_ids = tokenizer.apply_chat_template(
        _messages(), tokenize=True, add_generation_prompt=True
    )

    assert not rendered.startswith(tokenizer.bos_token)
    assert trl_ids == canonical_ids
    assert trl_ids[0] == tokenizer.bos_token_id
    assert trl_ids.count(tokenizer.bos_token_id) == 1


def test_renderer_rejects_multiple_template_bos_tokens() -> None:
    with pytest.raises(SftPromptRenderError, match="more than one leading BOS"):
        render_sft_prompt(_BosCharacterTokenizer(leading_bos=2), _messages())


def test_renderer_rejects_token_parity_mismatch() -> None:
    with pytest.raises(SftPromptRenderError, match="does not match"):
        render_sft_prompt(_BosCharacterTokenizer(template_offset=1), _messages())


def test_renderer_preserves_exact_parity_for_tokenizer_without_bos() -> None:
    class NoBosTokenizer:
        bos_token = None
        bos_token_id = None

        def apply_chat_template(
            self,
            messages,
            *,
            tokenize: bool,
            add_generation_prompt: bool,
            **_kwargs,
        ):
            assert add_generation_prompt is True
            rendered = "".join(item["content"] for item in messages) + "assistant"
            return [ord(character) for character in rendered] if tokenize else rendered

        def __call__(self, *, text: str):
            return {"input_ids": [ord(character) for character in text]}

    tokenizer = NoBosTokenizer()
    rendered = render_sft_prompt(tokenizer, _messages())

    assert tokenize_sft_text(tokenizer, rendered) == tokenizer.apply_chat_template(
        _messages(), tokenize=True, add_generation_prompt=True
    )


def test_training_auditor_uses_the_single_bos_renderer() -> None:
    tokenizer = _BosCharacterTokenizer()
    audit = build_training_sft_token_auditor(tokenizer)(
        "prompt", "reasoning\n</think>\nanswer"
    )

    assert audit["schema_version"] == "chk1-sft-training-token-budget-v3"
    assert audit["total_tokens"] == (
        audit["prompt_tokens"] + audit["completion_tokens"]
    )
    assert audit["passed"] is True


@pytest.mark.skipif(
    not TOKENIZER_PATH.exists() or not SFT_RELEASE.exists(),
    reason="local DeepSeek tokenizer or sealed chk1 release is unavailable",
)
def test_local_deepseek_release_has_single_bos_eos_prefix_and_no_4608_overflow() -> None:
    transformers = pytest.importorskip("transformers")
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        TOKENIZER_PATH,
        local_files_only=True,
    )
    max_total_tokens = 0
    row_count = 0

    for split in ("train", "eval", "test"):
        with (SFT_RELEASE / f"{split}.jsonl").open(encoding="utf-8") as handle:
            for raw_line in handle:
                row = json.loads(raw_line)
                rendered = render_sft_prompt(
                    tokenizer,
                    [
                        {
                            "role": "system",
                            "content": ANALYSIS_SFT_SYSTEM_PROMPT,
                        },
                        {"role": "user", "content": row["prompt"]},
                    ],
                )
                completion = row["response"]
                if not completion.endswith(tokenizer.eos_token):
                    completion += tokenizer.eos_token
                prompt_ids = tokenize_sft_text(tokenizer, rendered)
                full_ids = tokenize_sft_text(tokenizer, rendered + completion)

                assert full_ids[: len(prompt_ids)] == prompt_ids
                assert full_ids[0] == tokenizer.bos_token_id
                assert full_ids.count(tokenizer.bos_token_id) == 1
                assert full_ids[-1] == tokenizer.eos_token_id
                assert "</think>" in tokenizer.decode(
                    full_ids[len(prompt_ids) :], skip_special_tokens=False
                )

                completion_mask = [0] * len(prompt_ids) + [1] * (
                    len(full_ids) - len(prompt_ids)
                )
                assert all(value == 0 for value in completion_mask[: len(prompt_ids)])
                assert all(value == 1 for value in completion_mask[len(prompt_ids) :])
                assert completion_mask[-1] == 1
                assert len(full_ids) <= 4608
                max_total_tokens = max(max_total_tokens, len(full_ids))
                row_count += 1

    assert row_count == 1743
    assert max_total_tokens <= 4608
