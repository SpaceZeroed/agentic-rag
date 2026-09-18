import json
from collections.abc import AsyncIterator

import httpx
import pytest

from agentic_rag.llm.base import LLMError
from agentic_rag.llm.tool_client import CompatibleToolLLM, parse_tool_turn
from agentic_rag.tools.models import tool_definitions

pytestmark = pytest.mark.anyio


def payload(*, finish: str = "tool_calls") -> dict[str, object]:
    return {
        "model": "tool-test",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": None,
                    "tool_calls": [
                        {
                            "id": "a",
                            "type": "function",
                            "function": {
                                "name": "calculate",
                                "arguments": '{"expression":"6/8*100"}',
                            },
                        }
                    ],
                    "reasoning": "private reason",
                },
                "finish_reason": finish,
            }
        ],
        "usage": {"prompt_tokens": 40, "completion_tokens": 12, "cost": 0.01},
    }


async def test_tool_wire_and_conversation_linkage() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer secret"
        requests.append(json.loads(request.content))
        result = (
            payload()
            if len(requests) == 1
            else {
                "model": "tool-test",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "75 [T1]", "tool_calls": None},
                        "finish_reason": "stop",
                    }
                ],
            }
        )
        return httpx.Response(200, json=result)

    async with httpx.AsyncClient(
        base_url="https://test/v1/",
        headers={"Authorization": "Bearer secret"},
        transport=httpx.MockTransport(handler),
    ) as client:
        llm = CompatibleToolLLM(client, "tool-test", reasoning_enabled=False)
        messages: tuple[dict[str, object], ...] = ({"role": "user", "content": "Compute"},)
        turn = await llm.complete(messages, tool_definitions(), max_tokens=100)
        assert turn.completion.reasoning_characters == len("private reason")
        assert turn.completion.provider_cost == 0.01
        assert "private reason" not in str(turn.assistant_message())
        messages += (
            turn.assistant_message(),
            {"role": "tool", "tool_call_id": "a", "content": '{"value":"75"}'},
        )
        final = await llm.complete(messages, tool_definitions(), max_tokens=100)
        assert final.completion.text == "75 [T1]"
    assert requests[0]["tool_choice"] == "auto"
    assert requests[0]["parallel_tool_calls"] is False
    assert requests[0]["reasoning"] == {"enabled": False}
    assert requests[1]["messages"] == list(messages)


@pytest.mark.parametrize("finish", ["stop", "length", "content_filter", "unknown"])
async def test_invalid_finish_fails_without_retries(finish: str) -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(200, json=payload(finish=finish))

    async with httpx.AsyncClient(
        base_url="https://test/", transport=httpx.MockTransport(handler)
    ) as client:
        with pytest.raises(LLMError):
            await CompatibleToolLLM(client, "test").complete(
                ({"role": "user", "content": "x"},), tool_definitions(), max_tokens=20
            )
    assert calls == 1


@pytest.mark.parametrize(
    ("body", "status"), [(b"private invalid JSON", 200), (b"private driver text", 500)]
)
async def test_malformed_and_http_errors_are_safe(body: bytes, status: int) -> None:
    async with httpx.AsyncClient(
        base_url="https://test/",
        transport=httpx.MockTransport(lambda request: httpx.Response(status, content=body)),
    ) as client:
        with pytest.raises(LLMError) as exc:
            await CompatibleToolLLM(client, "test").complete(
                ({"role": "user", "content": "x"},), [], max_tokens=20
            )
        assert "private" not in str(exc.value)


async def test_response_bound_closes_stream() -> None:
    class Stream(httpx.AsyncByteStream):
        closed = False

        async def __aiter__(self) -> AsyncIterator[bytes]:
            yield b"x" * (1024 * 1024 + 1)

        async def aclose(self) -> None:
            self.closed = True

    stream = Stream()
    async with httpx.AsyncClient(
        base_url="https://test/",
        transport=httpx.MockTransport(lambda request: httpx.Response(200, stream=stream)),
    ) as client:
        with pytest.raises(LLMError):
            await CompatibleToolLLM(client, "test").complete(
                ({"role": "user", "content": "x"},), [], max_tokens=20
            )
    assert stream.closed


def test_duplicate_call_ids_rejected() -> None:
    data = {
        "model": "test",
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "tool_calls": [
                        {"id": "same", "function": {"name": "calculate", "arguments": "{}"}}
                    ]
                    * 2,
                },
                "finish_reason": "tool_calls",
            }
        ],
    }
    with pytest.raises(LLMError):
        parse_tool_turn(data)


async def test_correction_request_disables_tools_on_wire() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        data = json.loads(request.content)
        assert data["tool_choice"] == "none"
        assert "tools" not in data and "parallel_tool_calls" not in data
        return httpx.Response(
            200,
            json={
                "model": "test",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Corrected [T1]"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    async with httpx.AsyncClient(
        base_url="https://test/", transport=httpx.MockTransport(handler)
    ) as client:
        result = await CompatibleToolLLM(client, "test").complete(
            ({"role": "user", "content": "x"},), [], max_tokens=20
        )
    assert result.completion.text == "Corrected [T1]" and not result.calls
