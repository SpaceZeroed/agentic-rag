import json

import httpx
import pytest

from agentic_rag.llm.base import LLMError, Message
from agentic_rag.llm.compatible import CompatibleLLM

MESSAGES = (Message("system", "instructions"), Message("user", "Вопрос"))


def response_body() -> dict[str, object]:
    return {
        "model": "local-model",
        "choices": [
            {"message": {"role": "assistant", "content": "Ответ [C1]"}, "finish_reason": "stop"}
        ],
        "usage": {"prompt_tokens": 24, "completion_tokens": 5},
    }


def test_compatible_wire_contract() -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "http://local.test/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer private-key"
        payload = json.loads(request.content)
        assert payload == {
            "model": "local-model",
            "messages": [
                {"role": "system", "content": "instructions"},
                {"role": "user", "content": "Вопрос"},
            ],
            "max_tokens": 32,
            "temperature": 0,
            "stream": False,
        }
        return httpx.Response(200, json=response_body())

    with httpx.Client(
        base_url="http://local.test/v1/",
        transport=httpx.MockTransport(handle),
        headers={"Authorization": "Bearer private-key"},
    ) as client:
        result = CompatibleLLM(client, "local-model").complete(MESSAGES, max_tokens=32)
    assert result.text == "Ответ [C1]"
    assert (result.prompt_tokens, result.completion_tokens) == (24, 5)


@pytest.mark.parametrize(
    "kind",
    [
        "timeout",
        "connect",
        "401",
        "429",
        "500",
        "redirect",
        "json",
        "empty",
        "null",
        "length",
        "usage",
        "tool_calls",
    ],
)
def test_http_and_response_errors_are_safe(kind: str) -> None:
    def handle(request: httpx.Request) -> httpx.Response:
        if kind == "timeout":
            raise httpx.ReadTimeout("private-prompt", request=request)
        if kind == "connect":
            raise httpx.ConnectError("private-prompt", request=request)
        if kind.isdigit():
            return httpx.Response(int(kind), text="private-prompt")
        if kind == "redirect":
            return httpx.Response(307, headers={"location": "http://other.test"})
        if kind == "json":
            return httpx.Response(200, text="private-prompt")
        body = response_body()
        if kind == "empty":
            body["choices"] = []
        elif kind == "usage":
            body["usage"] = {"prompt_tokens": -1, "completion_tokens": 5}
        else:
            body["choices"] = [
                {
                    "message": {
                        "role": "assistant",
                        "content": None if kind == "null" else "partial",
                    },
                    "finish_reason": kind,
                }
            ]
        return httpx.Response(200, json=body)

    with (
        httpx.Client(
            base_url="http://local.test/v1/", transport=httpx.MockTransport(handle)
        ) as client,
        pytest.raises(LLMError) as caught,
    ):
        CompatibleLLM(client, "local-model").complete(MESSAGES, max_tokens=32)
    assert "private-prompt" not in str(caught.value)


def test_usage_can_be_absent() -> None:
    body = response_body()
    del body["usage"]
    with httpx.Client(
        base_url="http://local.test/v1/",
        transport=httpx.MockTransport(lambda r: httpx.Response(200, json=body)),
    ) as c:
        assert CompatibleLLM(c, "model").complete(MESSAGES, max_tokens=1).prompt_tokens is None
