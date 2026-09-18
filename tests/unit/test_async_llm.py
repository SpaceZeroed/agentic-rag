import json
from collections.abc import AsyncIterator
from contextlib import aclosing

import anyio
import httpx
import pytest

from agentic_rag.llm.async_client import AsyncCompatibleLLM, TextDelta
from agentic_rag.llm.base import Completion, LLMError, Message

pytestmark = pytest.mark.anyio
MESSAGES = (Message("user", "private question"),)


def frame(content: str | None = None, finish: str | None = None) -> bytes:
    return (
        "data: "
        + json.dumps(
            {
                "model": "test",
                "choices": [
                    {
                        "index": 0,
                        "delta": {"content": content},
                        "finish_reason": finish,
                    }
                ],
            }
        )
        + "\r\n\r\n"
    ).encode()


class ByteStream(httpx.AsyncByteStream):
    def __init__(self, data: bytes, *, wait: bool = False) -> None:
        self.data = data
        self.wait = wait
        self.closed = False
        self.waiting = anyio.Event()

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for i in range(0, len(self.data), 7):
            yield self.data[i : i + 7]
        if self.wait:
            self.waiting.set()
            await anyio.sleep_forever()

    async def aclose(self) -> None:
        self.closed = True


@pytest.mark.parametrize("repeat_finish", [False, True])
async def test_stream_wire_contract_unicode_usage_and_cleanup(repeat_finish: bool) -> None:
    usage = (
        b'data: {"model":"test","choices":[],"usage":'
        b'{"prompt_tokens":10,"completion_tokens":3,"cost":0.01}}\n\n'
    )
    stream = ByteStream(
        b": keepalive\n\n"
        + frame("Привет ")
        + frame("[C1]")
        + frame(finish="stop")
        + (frame(finish="stop") if repeat_finish else b"")
        + usage
        + b"data: [DONE]\n\n"
    )

    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret"
        payload = json.loads(request.content)
        assert payload["stream"] is True
        assert payload["reasoning"] == {"enabled": False}
        assert payload["stream_options"] == {"include_usage": True}
        return httpx.Response(200, headers={"content-type": "text/event-stream"}, stream=stream)

    async with httpx.AsyncClient(
        base_url="https://test/v1/",
        headers={"Authorization": "Bearer secret"},
        transport=httpx.MockTransport(handler),
    ) as client:
        llm = AsyncCompatibleLLM(client, "test", reasoning_enabled=False)
        items = [item async for item in llm.stream(MESSAGES, max_tokens=20)]
    assert items[:2] == [TextDelta("Привет "), TextDelta("[C1]")]
    assert items[-1] == Completion("Привет [C1]", "test", "stop", 10, 3, 0.01, 0)
    assert stream.closed


@pytest.mark.parametrize(
    "data",
    [
        frame("draft"),
        frame("draft") + b"data: [DONE]\n\n",
        frame("draft", "length") + b"data: [DONE]\n\n",
        frame("draft", "stop") + frame("extra") + b"data: [DONE]\n\n",
        frame("draft", "stop") + frame("extra", "stop") + b"data: [DONE]\n\n",
        frame("draft", "stop") + frame(finish="length") + b"data: [DONE]\n\n",
        frame("draft", "length") + frame(finish="stop") + b"data: [DONE]\n\n",
        frame("draft", "stop") + frame(finish="stop"),
        b"data: invalid-json\n\n",
        b'data: {"error":"secret backend details"}\n\n',
        b"data: " + b"x" * 140000,
    ],
)
async def test_invalid_or_incomplete_stream_fails_closed(data: bytes) -> None:
    stream = ByteStream(data)
    async with httpx.AsyncClient(
        base_url="https://test/v1/",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=stream
            )
        ),
    ) as client:
        with pytest.raises(LLMError) as error:
            _ = [
                item
                async for item in AsyncCompatibleLLM(client, "test").stream(MESSAGES, max_tokens=20)
            ]
        assert "secret" not in str(error.value)
    assert stream.closed


@pytest.mark.parametrize(
    "delta",
    [
        {"reasoning": "extra"},
        {"reasoning_content": "extra"},
        {"tool_calls": [{}]},
        {"refusal": "extra"},
    ],
)
async def test_repeated_finish_rejects_new_payload(delta: dict[str, object]) -> None:
    terminal = {
        "model": "test",
        "choices": [{"index": 0, "delta": delta, "finish_reason": "stop"}],
    }
    await test_invalid_or_incomplete_stream_fails_closed(
        frame("draft", "stop")
        + ("data: " + json.dumps(terminal) + "\n\n").encode()
        + b"data: [DONE]\n\n"
    )


async def test_stream_cancellation_closes_upstream() -> None:
    stream = ByteStream(frame("draft"), wait=True)
    async with httpx.AsyncClient(
        base_url="https://test/v1/",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=stream
            )
        ),
    ) as client:

        async def consume() -> None:
            async with aclosing(
                AsyncCompatibleLLM(client, "test").stream(MESSAGES, max_tokens=20)
            ) as iterator:
                async for _ in iterator:
                    pass

        async with anyio.create_task_group() as group:
            group.start_soon(consume)
            await stream.waiting.wait()
            group.cancel_scope.cancel()
        assert stream.closed


@pytest.mark.parametrize("status", [200, 401, 500])
async def test_async_completion_contract(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert json.loads(request.content)["stream"] is False
        return httpx.Response(
            status,
            json={
                "model": "test",
                "choices": [
                    {
                        "message": {"role": "assistant", "content": "Answer [C1]"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    async with httpx.AsyncClient(
        base_url="https://test/v1/", transport=httpx.MockTransport(handler)
    ) as client:
        llm = AsyncCompatibleLLM(client, "test")
        if status == 200:
            assert (await llm.complete(MESSAGES, max_tokens=20)).text == "Answer [C1]"
        else:
            with pytest.raises(LLMError):
                await llm.complete(MESSAGES, max_tokens=20)


@pytest.mark.parametrize("streaming", [False, True])
async def test_transport_timeout_has_distinct_safe_error(streaming: bool) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("private upstream detail", request=request)

    async with httpx.AsyncClient(
        base_url="https://test/v1/", transport=httpx.MockTransport(handler)
    ) as client:
        llm = AsyncCompatibleLLM(client, "test")
        with pytest.raises(TimeoutError, match="^LLM timeout$"):
            if streaming:
                _ = [item async for item in llm.stream(MESSAGES, max_tokens=10)]
            else:
                await llm.complete(MESSAGES, max_tokens=10)


async def test_reasoning_is_counted_but_never_streamed() -> None:
    reasoning = (
        b'data: {"model":"test","choices":[{"index":0,"delta":'
        b'{"reasoning_content":"private reasoning"},"finish_reason":null}]}\n\n'
    )
    stream = ByteStream(reasoning + frame("Answer [C1]", "stop") + b"data: [DONE]\n\n")
    async with httpx.AsyncClient(
        base_url="https://test/v1/",
        transport=httpx.MockTransport(
            lambda request: httpx.Response(
                200, headers={"content-type": "text/event-stream"}, stream=stream
            )
        ),
    ) as client:
        items = [
            item
            async for item in AsyncCompatibleLLM(client, "test").stream(MESSAGES, max_tokens=20)
        ]
    assert items[0] == TextDelta("Answer [C1]")
    completion = items[-1]
    assert isinstance(completion, Completion)
    assert completion.reasoning_characters == len("private reasoning")
    assert "private reasoning" not in repr(items)
