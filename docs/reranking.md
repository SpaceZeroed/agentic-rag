# Стадия 4 — переранжирование cross-encoder

Сначала dense + BM25 формируют кандидатов через RRF. Затем cross-encoder совместно
обрабатывает query и каждый passage и сортирует их по одному logit релевантности.
Только после этого выбираем top-k. Это дополнительная работа на каждом запросе:
в отличие от bi-encoder, результат для passage нельзя заранее вычислить независимо
от query. Рост candidate recall полезен, но больший пул означает больше инференса.

```mermaid
flowchart LR
  Q[Query] --> H[Dense + BM25 / RRF]
  H --> C[Candidate pool]
  C --> X[Pair tokenizer / cross-encoder / logits]
  X --> R[Sort and recheck current+ready]
  R --> T[Top-k with source coordinates]
```

## Выбор для Ryzen 5 3500U

4 физических ядра / 8 потоков, около 6.7 GiB RAM в WSL2, CPU без GPU.
Выбрана `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`, commit
`1427fd652930e4ba29e8149678df786c240d8825`: multilingual MiniLM с головой на один
logit, 12 слоёв, hidden size 384, примерно 0.1B параметров.
[Первичная карточка модели](https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1).
Обучение на переводах mMARCO не гарантирует cross-language RU→EN качество.

Альтернативы: [English MiniLM-L6](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L6-v2)
компактнее, но не соответствует многоязычной задаче;
[BGE v2 m3](https://bge-model.com/bge/bge_reranker_v2.html) многоязычная, но 568M
параметров существенно увеличивают требования. ONNX/int8 потенциально полезны на
CPU, но требуют отдельного сравнения качества и latency; в этой стадии не реализованы.

Исходная конфигурация зафиксирована до измерения: FP32, CPU, 4 потока, batch 4,
512 токенов на пару, 20 кандидатов и top-5. PyTorch set_num_threads действует на
весь процесс: используйте одинаковое число потоков для E5 и reranker. Скорость не
следует выводить только из размера модели или чужих GPU-бенчмарков.

## Контракты и ограничения

- `RerankingProvider.score(query, passages)` возвращает конечные logits в порядке
  входа. Provider заменяем; текущий адаптер рассчитан на single-logit sequence
  classification, а не произвольную модель Hugging Face.
- Tokenizer получает две последовательности без E5 prefixes. Special tokens и
  padding добавляет tokenizer; attention mask передаётся модели. `eval()` отключает
  dropout, `inference_mode()` исключает autograd. Никакого mean pooling или cosine.
- Ограничение 512 включает обе последовательности и special tokens. Truncation
  отключён: длинная пара вызывает явную ошибку. Чанк с 512 E5-токенами может не
  поместиться вместе с запросом. Sliding windows и passage-only truncation — другие
  политики, требующие собственной оценки и фиксации в spec.
- `score` в reranked output — raw logit, не вероятность. Сохраняются
  `retrieval_score`, `retrieval_rank`, IDs, text и координаты. При одинаковом logit
  используется UUID tie break. Scores разных методов не складываются.
- После scoring каталог повторно проверяет current (+ready для vector modes).
  Удалённые из видимости кандидаты пропускаются до top-k; при нехватке выдача короче.
  Проверка не является блокировкой на всё время доставки ответа.
- Пустой пул не вызывает provider.score. Ошибки модели не маскируются fallback.
- Search поддерживает reranking dense, BM25 и hybrid. Для BM25 не нужен E5/Qdrant.
  Без `--rerank` модель не загружается. Новых зависимостей и миграций нет:
  используется существующий extra `embeddings` (Torch/Transformers/SentencePiece).

## Запуск

Из корня проекта, после запуска PostgreSQL/Qdrant и индексации корпуса:

```bash
uv sync --locked --extra embeddings
export RAG_DATABASE_URL=postgresql+psycopg://rag:rag_local@127.0.0.1:55432/rag
uv run --locked --extra embeddings agentic-rag search \
  'Как QLoRA сокращает расход памяти?' --mode hybrid --rerank \
  --candidate-k 20 --rerank-k 20 --k 5 --collection-prefix rag_dense_v1
```

`--candidate-k` — глубина каждой hybrid ветки и максимум после RRF;
`--rerank-k` — сколько кандидатов передать cross-encoder;
`--k` — итоговая выдача. Нужно k ≤ rerank-k ≤ candidate-k для hybrid.
Модель скачивается в `data/models`; после скачивания можно включить
`RAG_MODEL_LOCAL_FILES_ONLY=true`. Настройки `RAG_RERANKING_MODEL`,
`RAG_RERANKING_REVISION`, `RAG_RERANKING_BATCH_SIZE`, `RAG_RERANKING_THREADS`.

```bash
RAG_MODEL_LOCAL_FILES_ONLY=true uv run --locked --extra embeddings agentic-rag evaluate \
  benchmarks/dense_v1/dataset.json --rerank --rerank-k 20 \
  --collection-prefix rag_dense_v1 --output data/artifacts/reranking_repeat.json
uv run --locked --extra embeddings pytest
uv run --locked --extra embeddings pytest \
  --postgres-url=postgresql+psycopg://rag:rag_local@127.0.0.1:55432/rag \
  --qdrant-url=http://127.0.0.1:6333 \
  --model-cache="$PWD/data/models" --reranker-cache="$PWD/data/models"
uv run --locked --extra embeddings ruff check .
uv run --locked --extra embeddings mypy
```

Evaluation `--rerank` сравнивает hybrid baseline и reranked на одном пуле каждого
вопроса. `--rerank-k` задаёт также глубину hybrid; `--compare` остаётся отдельным
сравнением dense/BM25/hybrid. Evaluation сохраняет/активирует ревизии корпуса и
синхронизирует индекс, как в стадиях 2–3. Используйте новый output для повторов.
Записываются candidate IDs/recall, baseline metrics, final hits, языковые metrics,
retrieval time, reranking time (включая повторную hydration), общая latency и spec.
Загрузка модели исключена, первый inference включён, warmup отсутствует.

Набор прежний: 10 документов, 20 чанков, 24 EN/RU вопроса, authored development
set без held-out split и с неполными qrels. При глубине 20 мы просматриваем весь
корпус — это не доказательство эффективности на большом индексе.
См. [реальные измерения](results.md).

## Что нужно знать для интервью

1. Bi-encoder позволяет precompute и быстрый поиск, cross-encoder даёт query-aware
   взаимодействие токенов ценой прохода на каждую пару.
2. Recall candidate pool — верхняя граница достижимого recall после сортировки.
3. Logits годятся для ранжирования одного запроса; sigmoid не делает их автоматически
   калиброванными вероятностями и не меняет математический порядок.
4. Batch, padding, длины пар и число потоков влияют на CPU latency и память.
5. Truncation меняет доступное evidence и должна быть явной частью эксперимента.
6. При 10x нагрузке нужны измерение очередей, ограничение concurrency/кандидатов,
   возможно batching/ONNX/int8; async само по себе не ускорит CPU-инференс.

Вопросы для повторения (разобраны с пользователем 2026-09-17; уточнения сохранены
в [interview notes](interview_notes.md)):

1. Почему нельзя заранее закешировать cross-encoder score каждого чанка?
2. Что означает candidate Recall@20=1 при итоговом Recall@5 меньше 1?
3. Почему sigmoid над logits обычно не улучшает MRR?
4. Как обрезание только passage может исказить evaluation по evidence spans?
5. Как определить, оправдывает ли прирост MRR дополнительную CPU latency?
