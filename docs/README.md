# Документация проекта

## С чего начать

1. [Паспорт проекта](project_brief.md) — цели, рамки и принципы.
2. [Архитектура](architecture.md) — актуальная схема, границы и риски.
3. [Дорожная карта](roadmap.md) — статус этапов и критерии завершения.
4. [Журнал результатов](results.md) — подтверждённые измерения и их пределы.

## Руководства по компонентам

| Область | Документ |
| --- | --- |
| Загрузка и хранение | [ingestion.md](ingestion.md) |
| Поиск | [dense_retrieval.md](dense_retrieval.md), [sparse_hybrid.md](sparse_hybrid.md), [reranking.md](reranking.md) |
| RAG | [basic_rag.md](basic_rag.md), [rag_evaluation.md](rag_evaluation.md) |
| HTTP API | [api.md](api.md) |
| Агент и MCP | [agent.md](agent.md), [mcp.md](mcp.md), [agent_evaluation.md](agent_evaluation.md) |
| Наблюдаемость | [observability.md](observability.md) |
| Архитектурные решения | [adr/](adr/) |
| Подготовка к интервью | [interview_notes.md](interview_notes.md) |

Документы о развёртывании, CI и streaming-бенчмарке находятся отдельно в
[deployment/](../deployment/), а машиночитаемые результаты — в `benchmarks/`.
Рабочие стенограммы, точки передачи контекста и заготовки сообщений коммитов намеренно
не хранятся в документации проекта.
