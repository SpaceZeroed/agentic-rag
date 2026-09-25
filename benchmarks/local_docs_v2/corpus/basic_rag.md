# Стадия 5 — Basic RAG

Стадия 5 реализует `query → retrieval → optional reranking → context → LLM → answer + citations`.
Это последовательный Python workflow. Поиск и reranking сохраняют свои контракты;
CLI — composition root, `rag/` не импортирует PostgreSQL, Qdrant или конкретный LLM.
HTTP API, агенты, streaming и оценка качества ответов остаются следующими стадиями.

## Контекст и его бюджет

`build_context()` принимает уже гидратированные канонические `SearchHit` в порядке
ранжирования. Присваивает включённым чанкам локальные метки C1, C2, …, удаляет повторы
по chunk ID, сохраняет целые чанки. Слишком большой чанк пропускается; следующий
может поместиться. Порядок оставшихся сохраняется. Пустой текст пропускается.
Overlap разных чанков не удаляется: объединение потребовало бы отдельного отображения
новых границ на источники. Список skipped_chunk_ids объясняет бюджетные пропуски.

Prompt состоит из system-инструкции и user JSON с вопросом, источниками, каноническим
текстом и метаданными. JSON сохраняет границы полей при кавычках/переносах в документах,
но не гарантирует защиту от prompt injection. System-инструкция объявляет источники
недоверенными данными и запрещает исполнять содержащиеся в них команды.

По умолчанию лимит **24000 UTF-8 байт сериализованного массива сообщений**, включая
инструкции, вопрос и метаданные, но без HTTP-конверта. Это инженерный лимит размера
prompt, не оценка токенов и не обещание вместимости в конкретную модель. Если уже
вопрос с инструкциями превышает бюджет, ошибка возникает до генерации. Если не
поместился ни один чанк, возвращается `insufficient_evidence` без вызова LLM.

Для конкретной LLM нужно считать `tokenizer.apply_chat_template(...)` тем же tokenizer,
revision и template, что использует сервер, и проверять:

`input_tokens + reserved_output_tokens <= model_context_window`.

Число символов / 4 не универсально: русский текст, код и специальные токены меняют
соотношение. Здесь `RAG_LLM_MAX_TOKENS=512` передаётся серверу как лимит генерации,
отдельно от лимита байт. Автоматической токенизации или обрезки серверного prompt нет.
Если сервер отверг длину, клиент возвращает ошибку. Перед подключением реальной модели
требуется согласовать эти ограничения; модель и её tokenizer ещё не выбраны.

## Ответ и цитаты

Модель пишет текст с `[C1]`, `[C2]`. Проверка требует хотя бы одну корректную метку;
отклоняет неизвестные ID и некорректные конструкции, начинающиеся на `[C`.
Поддерживается только точный синтаксис `[Cчисло]`, без `[C1, C2]`. Повторные цитаты
разрешены, список citations содержит уникальные ссылки в порядке первого появления.
Только процитированные источники включаются в citations; весь переданный контекст
остаётся в context для проверки результата. URL из свободного текста ответа не
считаются доверенными цитатами и автоматически не проверяются.

Каждая citation содержит полный исходный SearchHit: chunk/revision/document UUID,
source_uri, title, text, score, start/end_char и start/end_line. Для reranked hits
сохраняются retrieval_rank/retrieval_score. Смещения относятся к нормализованному
каноническому документу: start_char включительно, end_char исключительно; строки —
1-based. Чанк не обрезается, поэтому его координаты остаются исходными.

Результат — снимок источников на момент последней гидратации поиска/reranking. Во
время генерации текущая ревизия может смениться. Ответ сохраняет конкретную старую
revision_id и текст, не подменяет их новой версией и не обещает current на момент
доставки. Retained revision можно прочитать командой `show --revision`.

Проверка citation ID **не доказывает entailment** и не проверяет каждое утверждение.
Ошибочная LLM может процитировать существующий, но нерелевантный фрагмент. Это предмет
стадии 6; здесь не заявляется измеренное качество или groundedness.

При недостаточных данных модель должна вернуть ровно `INSUFFICIENT_EVIDENCE`.
Приложение преобразует это в структурированный отказ с пустыми citations. Наличие
retrieval hits само по себе не означает достаточных доказательств: score не является
калиброванной уверенностью. Отсутствие цитат в обычном ответе — ошибка, не тихий отказ.

## Интерфейс LLM

`LLM.complete(messages, max_tokens=...) -> Completion`: типизированные сообщения,
текст, model, finish_reason и необязательные prompt_tokens/completion_tokens.
Бизнес-логика зависит от Protocol, реализации — от него и transport library.

- `FakeLLM` понимает JSON нашего RAG prompt и детерминированно показывает первый
  фрагмент с префиксом Mock и цитатой. Он не отвечает на вопрос и не моделирует
  токенный лимит/инференс. `[C` внутри исходного фрагмента отображается как `［C`,
  чтобы содержимое документа не создало дополнительные метки. В provenance текст
  остаётся неизменным. Token usage отсутствует, а не выдумывается.
- `CompatibleLLM` вызывает `POST <base_url>/chat/completions`: model, messages,
  temperature=0, max_tokens, stream=false. API key необязателен. HTTPX уже присутствовал
  транзитивно через Qdrant; теперь это явная зависимость 0.28.x, без новых пакетов.
  CLI владеет клиентом и закрывает его, timeout по умолчанию 60 секунд для HTTPX
  операций (не полный deadline всей RAG-цепочки). Редиректы и environment proxy
  отключены. Автоматических повторов и fallback на fake нет.
- Внешний JSON валидируется Pydantic: один choice, assistant text, корректный usage
  при наличии. Принимается только finish_reason=stop; length/tool_calls и пустой
  ответ отвергаются. Это узкое подмножество протокола, без tools/reasoning/streaming.
  Temperature=0 не гарантирует побитовой воспроизводимости реального сервера.
- Ошибки транспорта, статуса/формата ответа и цитат дают `generation_failed`, exit 1,
  без текста prompt, токена доступа или тела ответа в сообщении приложения.

Совместимый HTTP endpoint и необходимость chat template описаны в
[документации vLLM](https://docs.vllm.ai/en/latest/serving/online_serving/openai_compatible_server/).
MockTransport тестирует протокол без сети. Проверка с реальным inference server пока
не выполнена; совместимость конкретной версии/модели проверяется при подключении.

## Запуск

Из корня проекта, после настройки `.env` и запуска PostgreSQL:

```bash
uv sync --locked --extra embeddings
uv run --locked --extra embeddings agentic-rag db-upgrade
uv run --locked --extra embeddings agentic-rag ingest benchmarks/dense_v1/corpus/paged_attention.md --max-chars 400 --overlap 60
uv run --locked --extra embeddings agentic-rag ask 'PagedAttention' --mode bm25 --llm fake
```

BM25 + fake не требует E5, Qdrant или GPU. Extra выше сохраняет уже установленный
CPU runtime проекта; для чистой BM25-установки его можно не устанавливать.
Для уже проиндексированного benchmark с collection prefix rag_dense_v1:

```bash
RAG_MODEL_LOCAL_FILES_ONLY=true uv run --locked --extra embeddings agentic-rag ask \
  'Как PagedAttention управляет KV cache?' --mode hybrid \
  --collection-prefix rag_dense_v1 --rerank --rerank-k 20 --k 5 --llm fake
```

Для новых документов сначала `agentic-rag index` с тем же collection prefix.
Остальные параметры совпадают с search: --mode, --k, --candidate-k, --rerank,
--rerank-k, --approximate и фильтры --document-id/--source-uri/--media-type.

Для существующего локального Chat Completions сервера:

```bash
RAG_LLM_BASE_URL=http://127.0.0.1:8000/v1 RAG_LLM_MODEL=your-served-model \
  uv run --locked --extra embeddings agentic-rag ask 'PagedAttention' \
  --mode bm25 --llm compatible
```

JSON stdout содержит mode, reranked, llm_provider и answer. В answer: status, text,
citations, context и completion. Это диагностический CLI, поэтому он печатает и
сам контекст; конфиденциальные документы потребуют соответствующего обращения с stdout.
Ошибки конфигурации/ввода дают exit 2, ошибки LLM — exit 1, ответ/отказ — exit 0.

## Проверка и вопросы

```bash
uv run --locked --extra embeddings ruff check .
uv run --locked --extra embeddings ruff format --check .
uv run --locked --extra embeddings mypy
uv run --locked --extra embeddings pytest
```

Тесты без внешних сервисов проверяют упаковку контекста, Unicode, границы бюджета,
provenance, abstention, fake, HTTP wire contract и сбои. PostgreSQL/Qdrant тесты
отдельно проверяют CLI с текущими версиями, фильтрами и reranking. См. results.md.

Вопросы для повторения (разбор завершён 2026-09-17; уточнения в interview_notes.md):

1. Почему валидный `[C1]` ещё не доказывает groundedness ответа?
2. Что именно нужно считать и резервировать для токенного бюджета конкретной LLM?
3. Каковы последствия пропуска большого чанка по сравнению с его обрезкой?
4. Почему snapshot citation должен содержать revision_id, даже если поиск выбирает current?
5. Какие ошибки обнаруживает fake/MockTransport, а какие требуют реальной модели?
