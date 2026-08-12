from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest

from jobs.retrain_v2.compress_chk1_reasoning import (
    BOUNDARY,
    MODEL,
    PROMPT_TEMPLATE_VERSION,
    CompressionError,
    _compress_one,
    build_response_input,
    compose_output_row,
    extract_response_output,
    parse_compressed_content,
    parse_source_row,
    request_contract,
)


def _row() -> dict[str, str]:
    provided = '{"growth":"slower","inflation":"stable"}'
    prompt = f"Analyze point-in-time evidence.\n\n{provided}"
    return {
        "prompt": prompt,
        "provided_data": provided,
        "response": "A long but grounded reasoning draft." + BOUNDARY + "Activity moderated.",
    }


def test_prompt_recombines_source_reasoning_and_fixed_answer() -> None:
    item = parse_source_row(_row(), split="train", line_number=1)
    response_input = build_response_input(item)
    payload = json.loads(response_input)
    assert payload == {
        "schema_version": PROMPT_TEMPLATE_VERSION,
        "source_prompt": item.prompt,
        "original_reasoning": item.original_reasoning,
        "fixed_final_answer": item.fixed_final_answer,
    }
    assert "DEEPSEEK_API_KEY" not in response_input


def test_request_contract_uses_flash_responses_api_max_effort() -> None:
    contract = request_contract(model=MODEL, max_output_tokens=32768)
    assert contract["model"] == "deepseek-v4-flash"
    assert contract["api"] == "responses"
    assert contract["reasoning"] == {"effort": "max"}
    assert contract["max_output_tokens"] == 32768
    assert contract["text"]["format"]["type"] == "json_schema"
    assert contract["target_source"] == "response.output_text.compressed_reasoning"
    assert contract["ignored_source"] == "response.output[type=reasoning]"


def test_responses_output_keeps_visible_and_hidden_reasoning_separate() -> None:
    visible_json = json.dumps(
        {"status": "ok", "compressed_reasoning": "Visible compact reasoning."}
    )
    response = {
        "output_text": visible_json,
        "output": [
            {
                "type": "reasoning",
                "content": [{"type": "reasoning_text", "text": "private max-effort reasoning"}],
            },
            {
                "type": "message",
                "content": [{"type": "output_text", "text": visible_json}],
            },
        ],
    }
    visible, hidden = extract_response_output(response)
    assert visible == visible_json
    assert hidden == "private max-effort reasoning"
    assert hidden not in visible


def test_compression_call_uses_max_effort_and_never_persists_hidden_text(tmp_path) -> None:
    item = parse_source_row(_row(), split="train", line_number=1)
    visible_json = json.dumps(
        {"status": "ok", "compressed_reasoning": "one two three four"}
    )
    hidden_text = "private max-effort reasoning must not be persisted"

    class FakeResponses:
        def __init__(self) -> None:
            self.calls = []

        async def create(self, **kwargs):
            self.calls.append(kwargs)
            return SimpleNamespace(
                status="completed",
                incomplete_details=None,
                output_text=visible_json,
                output=[
                    SimpleNamespace(
                        type="reasoning",
                        content=[SimpleNamespace(type="reasoning_text", text=hidden_text)],
                    ),
                    SimpleNamespace(
                        type="message",
                        content=[SimpleNamespace(type="output_text", text=visible_json)],
                    ),
                ],
                usage=SimpleNamespace(
                    input_tokens=100,
                    output_tokens=20,
                    output_tokens_details=SimpleNamespace(reasoning_tokens=16),
                    total_tokens=120,
                ),
                model=MODEL,
                id="response-test",
            )

    fake_responses = FakeResponses()
    client = SimpleNamespace(responses=fake_responses)
    result = asyncio.run(
        _compress_one(
            item,
            client=client,
            semaphore=asyncio.Semaphore(1),
            output=tmp_path,
            model=MODEL,
            max_output_tokens=32768,
            min_reasoning_tokens=3,
            max_reasoning_tokens=6,
            retries=0,
            contract_sha256="a" * 64,
            token_counter=lambda value: len(value.split()),
            resume=False,
        )
    )
    call = fake_responses.calls[0]
    assert call["reasoning"] == {"effort": "max"}
    assert call["max_output_tokens"] == 32768
    assert call["text"]["format"]["type"] == "json_schema"
    assert result["provider_reasoning_tokens"] == 16
    cache_text = next((tmp_path / "cache").rglob("*.json")).read_text()
    assert hidden_text not in cache_text
    assert result["compressed_reasoning"] == "one two three four"


def test_parser_accepts_compact_json_and_reuses_answer_exactly() -> None:
    item = parse_source_row(_row(), split="train", line_number=1)
    raw = json.dumps({"status": "ok", "compressed_reasoning": "Grounded compact reasoning."})
    reasoning, tokens = parse_compressed_content(
        raw,
        token_counter=lambda value: len(value.split()),
        min_reasoning_tokens=2,
        max_reasoning_tokens=8,
    )
    output = compose_output_row(item, reasoning)
    assert tokens == 3
    assert output["response"] == "Grounded compact reasoning." + BOUNDARY + "Activity moderated."


@pytest.mark.parametrize(
    "payload",
    [
        {"status": "unsupported_answer", "compressed_reasoning": ""},
        {"status": "ok", "compressed_reasoning": "too short"},
        {"status": "ok", "compressed_reasoning": "one two three four five six seven"},
        {"status": "ok", "compressed_reasoning": "valid words </think> invalid"},
        {"status": "ok", "compressed_reasoning": "valid words", "answer": "drift"},
    ],
)
def test_parser_fails_closed(payload: dict[str, str]) -> None:
    with pytest.raises(CompressionError):
        parse_compressed_content(
            json.dumps(payload),
            token_counter=lambda value: len(value.split()),
            min_reasoning_tokens=3,
            max_reasoning_tokens=6,
        )


def test_source_requires_exact_native_boundary_and_embedded_evidence() -> None:
    row = _row()
    row["response"] = "reasoning without a boundary"
    with pytest.raises(CompressionError, match="boundary"):
        parse_source_row(row, split="train", line_number=1)

    row = _row()
    row["prompt"] = "evidence was accidentally omitted"
    with pytest.raises(CompressionError, match="provided_data exactly once"):
        parse_source_row(row, split="train", line_number=1)
