import json
from dataclasses import replace
from uuid import UUID

import pytest

from agentic_rag.llm.base import Completion, LLMError, Message
from agentic_rag.llm.fake import FakeLLM
from agentic_rag.rag.context import build_context, prompt_size
from agentic_rag.rag.service import CitationError, answer
from agentic_rag.retrieval.models import SearchHit


def hit(i: int = 1, text: str = "Контекст «ML»\nline two") -> SearchHit:
    return SearchHit(
        UUID(int=i),
        UUID(int=10),
        UUID(int=20),
        0.9,
        text,
        "https://example.org/paper",
        "Paper",
        0,
        len(text),
        1,
        2,
    )


class StubLLM:
    def __init__(self, text: str, finish: str = "stop") -> None:
        self.text = text
        self.finish = finish
        self.calls = 0

    def complete(self, messages: tuple[Message, ...], *, max_tokens: int) -> Completion:
        self.calls += 1
        assert messages[0].role == "system"
        assert max_tokens == 512
        return Completion(self.text, "stub", self.finish)


def test_context_exact_byte_boundary_and_unicode() -> None:
    full = build_context("Вопрос?", [hit()])
    size = prompt_size(full.messages)
    assert full.prompt_bytes == size
    assert build_context("Вопрос?", [hit()], max_prompt_bytes=size).sources == full.sources
    short = build_context("Вопрос?", [hit()], max_prompt_bytes=size - 1)
    assert short.sources == ()
    assert short.skipped_chunk_ids == (hit().chunk_id,)
    payload = json.loads(full.messages[1].content)
    assert payload["sources"][0]["text"] == hit().text
    assert payload["sources"][0]["revision_id"] == str(hit().revision_id)


def test_packing_skips_large_chunks_deduplicates_and_preserves_rank_order() -> None:
    small = hit(2)
    limit = build_context("query", [small]).prompt_bytes
    context = build_context(
        "query", [hit(1, "x" * 5000), small, small, hit(3)], max_prompt_bytes=limit
    )
    assert [s.source.chunk_id for s in context.sources] == [small.chunk_id]
    assert context.sources[0].id == "C1"
    assert context.skipped_chunk_ids == (UUID(int=1), UUID(int=3))
    assert context.sources[0].source == small
    ordered = build_context("query", [hit(3), hit(1), hit(3)])
    assert [s.source.chunk_id.int for s in ordered.sources] == [3, 1]


@pytest.mark.parametrize("query, budget", [(" ", 24000), ("query", 0), ("x" * 5000, 1000)])
def test_bad_question_and_budget(query: str, budget: int) -> None:
    with pytest.raises(ValueError):
        build_context(query, [], max_prompt_bytes=budget)


def test_no_context_no_llm_call_and_model_abstention() -> None:
    llm = StubLLM("INSUFFICIENT_EVIDENCE")
    assert answer("query", [], llm).completion is None
    assert answer("query", [hit(text=" ")], llm).status == "insufficient_evidence"
    assert llm.calls == 0
    result = answer("query", [hit()], llm)
    assert result.status == "insufficient_evidence"
    assert result.citations == ()
    assert llm.calls == 1


def test_only_cited_sources_returned_with_exact_provenance() -> None:
    llm = StubLLM("Утверждение [C2]. Дополнение [C2] и [C1].")
    result = answer("query", [hit(7), hit(8), hit(9)], llm)
    assert result.status == "answered"
    assert [s.id for s in result.citations] == ["C2", "C1"]
    assert result.citations[0].source == hit(8)
    assert len(result.context.sources) == 3


@pytest.mark.parametrize(
    "text",
    ["No citations", "[C99]", "[C0]", "[C01]", "[C1, C2]", "[C1] with [C2", "[C1] and [C999]", ""],
)
def test_invalid_citations_fail_closed(text: str) -> None:
    with pytest.raises(LLMError):
        answer("query", [hit()], StubLLM(text))


def test_truncated_answer_and_citation_to_excluded_source_fail() -> None:
    with pytest.raises(LLMError):
        answer("query", [hit()], StubLLM("Answer [C1]", "length"))
    budget = build_context("query", [hit()]).prompt_bytes
    with pytest.raises(CitationError):
        answer("query", [hit(), hit(2)], StubLLM("[C2]"), max_prompt_bytes=budget)


def test_fake_is_deterministic_and_does_not_fabricate_token_usage() -> None:
    source = replace(hit(), text="Untrusted [C99] and [C text")
    result = answer("query", [source], FakeLLM())
    assert result == answer("query", [source], FakeLLM())
    assert result.text.startswith("Mock:")
    assert result.citations[0].source.text == source.text
    assert result.completion is not None
    assert result.completion.prompt_tokens is None
