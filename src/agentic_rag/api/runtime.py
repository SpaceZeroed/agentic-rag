"""Long-lived resources and a synchronous backend, independent of HTTP handlers."""

import logging
from collections.abc import AsyncIterator, Callable
from contextlib import AsyncExitStack, ExitStack, asynccontextmanager
from functools import partial
from typing import Protocol, cast

import anyio
import httpx
from qdrant_client import QdrantClient
from sqlalchemy import select

from agentic_rag.agents.fake import FakeToolLLM
from agentic_rag.agents.models import AgentLimits
from agentic_rag.agents.service import Agent
from agentic_rag.api.models import DocumentRequest, DocumentResponse, QueryRequest
from agentic_rag.core.config import Settings
from agentic_rag.embeddings.e5 import E5EmbeddingProvider
from agentic_rag.ingestion.models import ChunkingConfig
from agentic_rag.ingestion.parsing import parse_bytes
from agentic_rag.ingestion.service import prepare_parsed
from agentic_rag.llm.async_client import AsyncCompatibleLLM, AsyncFakeLLM, AsyncLLM
from agentic_rag.llm.tool_client import CompatibleToolLLM, ToolLLM
from agentic_rag.observability.llm import TracedLLM
from agentic_rag.observability.tracing import span
from agentic_rag.rag.context import Context, build_context
from agentic_rag.reranking.base import RerankingProvider
from agentic_rag.reranking.cross_encoder import CrossEncoderProvider
from agentic_rag.reranking.service import rerank
from agentic_rag.retrieval.hybrid import HybridRetriever
from agentic_rag.retrieval.models import SearchFilter, SearchHit
from agentic_rag.retrieval.qdrant import QdrantVectorIndex
from agentic_rag.retrieval.service import DenseRetriever
from agentic_rag.retrieval.sparse import SparseRetriever
from agentic_rag.storage.catalog import document_catalog
from agentic_rag.storage.database import create_database_engine
from agentic_rag.storage.models import DocumentRow
from agentic_rag.storage.repository import PostgresDocumentRepository
from agentic_rag.storage.retrieval import PostgresIndexCatalog
from agentic_rag.tools.models import CatalogInput, CatalogOutput, SearchInput

logger = logging.getLogger(__name__)


class Backend(Protocol):
    def context(self, request: QueryRequest) -> Context: ...
    def ingest(self, request: DocumentRequest) -> DocumentResponse: ...
    def health(self) -> dict[str, str]: ...
    def catalog_documents(self, arguments: CatalogInput) -> CatalogOutput: ...


class PostgresBackend:
    def __init__(self, settings: Settings, resources: ExitStack) -> None:
        if settings.database_url is None:
            raise ValueError("RAG_DATABASE_URL is required")
        self.settings = settings
        self.engine = create_database_engine(settings.database_url.get_secret_value())
        resources.callback(self.engine.dispose)
        self.repository = PostgresDocumentRepository(self.engine)
        self.catalog = PostgresIndexCatalog(self.engine)
        self.sparse = SparseRetriever(self.catalog)
        self.dense: DenseRetriever | None = None
        self.qdrant: QdrantClient | None = None
        self.reranker: RerankingProvider | None = None
        if settings.api_retrieval_mode != "bm25":
            embeddings = E5EmbeddingProvider(
                settings.data_dir / "models",
                model_id=settings.embedding_model,
                revision=settings.embedding_revision,
                batch_size=settings.embedding_batch_size,
                threads=settings.embedding_threads,
                local_files_only=settings.model_local_files_only,
            )
            self.qdrant = QdrantClient(
                url=settings.qdrant_url,
                api_key=settings.qdrant_api_key.get_secret_value()
                if settings.qdrant_api_key
                else None,
                timeout=30,
            )
            resources.callback(self.qdrant.close)
            index = QdrantVectorIndex(self.qdrant, embeddings.spec, settings.collection_prefix)
            self.dense = DenseRetriever(embeddings, index, self.catalog)
        if settings.api_rerank:
            self.reranker = CrossEncoderProvider(
                settings.data_dir / "models",
                model_id=settings.reranking_model,
                revision=settings.reranking_revision,
                batch_size=settings.reranking_batch_size,
                threads=settings.reranking_threads,
                local_files_only=settings.model_local_files_only,
            )

    def catalog_documents(self, arguments: CatalogInput) -> CatalogOutput:
        return document_catalog(self.engine, arguments)

    def context(self, request: QueryRequest) -> Context:
        settings = self.settings
        if (self.reranker or settings.api_retrieval_mode == "hybrid") and (
            request.k > settings.api_candidate_k
        ):
            raise ValueError("k exceeds configured candidate depth")
        filters = SearchFilter(tuple(request.document_ids), request.source_uri, request.media_type)
        k = settings.api_candidate_k if self.reranker else request.k
        collection = None
        hits: list[SearchHit]
        with span("retrieval") as current:
            current.set(mode=settings.api_retrieval_mode)
            if self.dense is None:
                hits = self.sparse.search(request.query, k=k, filters=filters)
            else:
                collection = self.dense.index.collection
                if settings.api_retrieval_mode == "hybrid":
                    hits = HybridRetriever(self.dense, candidate_k=settings.api_candidate_k).search(
                        request.query, k=k, filters=filters
                    )
                else:
                    hits = self.dense.search(request.query, k=k, filters=filters)
            current.set(candidate_count=len(hits))
        if self.reranker and hits:
            with span("reranking") as current:
                current.set(candidate_count=len(hits))
                hits = list(
                    rerank(
                        request.query, hits, self.reranker, self.catalog, collection, k=request.k
                    )
                )
                current.set(source_count=len(hits))
        return build_context(request.query, hits, max_prompt_bytes=settings.llm_max_prompt_bytes)

    def ingest(self, request: DocumentRequest) -> DocumentResponse:
        document = prepare_parsed(
            parse_bytes(request.content.encode("utf-8"), filename=request.filename),
            ChunkingConfig(request.max_chars, request.overlap),
            source_uri=request.source_uri,
        )
        saved = self.repository.save(document)
        if self.dense is None:
            return DocumentResponse(document=saved, index_status="not_required")
        try:
            self.dense.sync(SearchFilter(document_ids=(saved.document_id,)))
        except Exception as exc:
            logger.warning("document_index_failed", extra={"fields": {"type": type(exc).__name__}})
            # PostgreSQL has committed. An identical retry repairs the index.
            return DocumentResponse(
                document=saved, index_status="failed", error="index_unavailable"
            )
        return DocumentResponse(document=saved, index_status="ready")

    def health(self) -> dict[str, str]:
        status: dict[str, str] = {}
        try:
            with self.engine.connect() as connection:
                connection.execute(select(DocumentRow.id).limit(1))
            status["postgres"] = "ok"
        except Exception:
            status["postgres"] = "unavailable"
        if self.qdrant:
            try:
                self.qdrant.get_collections()
                status["qdrant"] = "ok"
            except Exception:
                status["qdrant"] = "unavailable"
        else:
            status["qdrant"] = "not_required"
        # Do not spend tokens or infer provider readiness from a local health check.
        status["llm"] = "fake" if self.settings.api_llm_provider == "fake" else "not_probed"
        return status


class Runtime:
    def __init__(
        self,
        settings: Settings,
        backend: Backend,
        llm: AsyncLLM,
        tool_llm: ToolLLM | None = None,
    ) -> None:
        self.settings = settings
        self.backend = backend
        self.llm = TracedLLM(llm)
        # Bound in-flight work; serialize shared CPU models and index mutations.
        self.requests = anyio.CapacityLimiter(settings.api_max_concurrent_requests)
        self.workers = anyio.CapacityLimiter(1)
        self.agent = (
            Agent(
                tool_llm,
                RuntimeTools(self),
                AgentLimits(
                    max_model_calls=settings.agent_max_model_calls,
                    max_tool_calls=settings.agent_max_tool_calls,
                    max_prompt_bytes=settings.agent_max_prompt_bytes,
                    max_observation_bytes=settings.agent_max_observation_bytes,
                    max_tokens=settings.llm_max_tokens,
                ),
                provider=settings.api_llm_provider,
            )
            if tool_llm is not None
            else None
        )

    async def run_sync[T](self, call: Callable[[], T]) -> T:
        # LangGraph cancels nodes with Task.cancel(), which can bypass a cancel
        # scope in that task. A child owned by an AnyIO task group is cancelled
        # through its scope; group exit waits for its shielded thread operation.
        result: T | None = None
        error: Exception | None = None

        async def work() -> None:
            nonlocal result, error
            try:
                result = await anyio.to_thread.run_sync(call, limiter=self.workers)
            except Exception as exc:
                error = exc

        async with anyio.create_task_group() as group:
            group.start_soon(work)
        await anyio.lowlevel.checkpoint()
        if error is not None:
            raise error
        return cast(T, result)


class RuntimeTools:
    def __init__(self, runtime: Runtime) -> None:
        self.runtime = runtime

    async def search(self, arguments: SearchInput) -> tuple[SearchHit, ...]:
        context = await self.runtime.run_sync(
            partial(
                self.runtime.backend.context,
                QueryRequest(query=arguments.query, k=arguments.k),
            )
        )
        return tuple(c.source for c in context.sources)

    async def catalog(self, arguments: CatalogInput) -> CatalogOutput:
        return await self.runtime.run_sync(
            partial(self.runtime.backend.catalog_documents, arguments)
        )


@asynccontextmanager
async def open_runtime(settings: Settings) -> AsyncIterator[Runtime]:
    if settings.api_llm_provider == "compatible" and not (
        settings.llm_model and settings.llm_model.strip()
    ):
        raise ValueError("RAG_LLM_MODEL is required for compatible generation")
    resources = ExitStack()
    try:
        backend = await anyio.to_thread.run_sync(lambda: PostgresBackend(settings, resources))
        async with AsyncExitStack() as stack:
            llm: AsyncLLM = AsyncFakeLLM()
            tool_llm: ToolLLM = FakeToolLLM()
            if settings.api_llm_provider == "compatible":
                headers = {}
                if settings.llm_api_key:
                    headers["Authorization"] = f"Bearer {settings.llm_api_key.get_secret_value()}"
                client = await stack.enter_async_context(
                    httpx.AsyncClient(
                        base_url=settings.llm_base_url.rstrip("/") + "/",
                        headers=headers,
                        timeout=settings.llm_timeout_seconds,
                        follow_redirects=False,
                        trust_env=False,
                    )
                )
                llm = AsyncCompatibleLLM(
                    client,
                    settings.llm_model or "",
                    reasoning_enabled=settings.llm_reasoning_enabled,
                )
                tool_llm = CompatibleToolLLM(
                    client,
                    settings.llm_model or "",
                    reasoning_enabled=settings.llm_reasoning_enabled,
                )
            yield Runtime(settings, backend, llm, tool_llm)
    finally:
        with anyio.CancelScope(shield=True):
            await anyio.to_thread.run_sync(resources.close)
