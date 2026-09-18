"""Explicit fixture commands for offline demos; not a learned tool selector."""

import json

import anyio

from agentic_rag.llm.base import Completion
from agentic_rag.llm.tool_client import FunctionCall, ToolCall, ToolTurn


class FakeToolLLM:
    async def complete(
        self,
        messages: tuple[dict[str, object], ...],
        tools: list[dict[str, object]],
        *,
        max_tokens: int,
    ) -> ToolTurn:
        await anyio.lowlevel.checkpoint()
        last = messages[-1]
        if last["role"] == "tool":
            observation = json.loads(str(last["content"]))
            if observation["status"] == "error":
                text = "INSUFFICIENT_EVIDENCE"
            elif observation["name"] == "search_documents":
                passages = observation["data"]["passages"]
                text = (
                    f"FAKE fixture excerpt: {passages[0]['source']['text']} [{passages[0]['id']}]"
                    if passages
                    else "INSUFFICIENT_EVIDENCE"
                )
            else:
                data = json.dumps(observation["data"], ensure_ascii=False)
                text = f"FAKE fixture: {data} [{observation['reference']}]"
            return ToolTurn(Completion(text, "fake-tools", "stop"))
        query = str(last["content"])
        if query.startswith("/direct"):
            return ToolTurn(
                Completion("FAKE direct fixture; no evidence verified.", "fake-tools", "stop")
            )
        if query.startswith("/calc "):
            name, arguments = "calculate", {"expression": query[6:]}
        elif query.startswith("/catalog"):
            name, arguments = "document_catalog", {}
        else:
            name, arguments = "search_documents", {"query": query.removeprefix("/search ")}
        call = ToolCall(
            id="fake-call-1", function=FunctionCall(name=name, arguments=json.dumps(arguments))
        )
        return ToolTurn(Completion("", "fake-tools", "tool_calls"), (call,))
