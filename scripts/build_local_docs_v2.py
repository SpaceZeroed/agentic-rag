"""Freeze an auditable candidate QA set from repository documentation."""

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "benchmarks/local_docs_v2"

# (document key, question, exact excerpt). Labels are assistant-authored drafts.
CASES = [
    (
        "0001-modular-monolith",
        "Где находится устанавливаемый пакет?",
        "пакет `agentic_rag` в `src/`",
    ),
    (
        "0001-modular-monolith",
        "От чего не зависит доменная логика?",
        "логика не зависит от HTTP-фреймворка и конкретных SDK.",
    ),
    ("0002-foundation-tooling", "Какая минимальная версия Python принята?", "Python 3.12+"),
    ("0002-foundation-tooling", "Что не фиксирует lockfile?", "но не ОС, драйверы, оборудование,"),
    ("0003-document-and-index-ownership", "Где хранятся текст и чанки?", "PostgreSQL."),
    (
        "0003-document-and-index-ownership",
        "Из какого хранилища восстанавливается Qdrant?",
        "восстановлен из PostgreSQL.",
    ),
    (
        "0004-versioned-ingestion",
        "Можно ли проверить ссылку после обновления документа?",
        "Ссылку можно проверить после смены текущей версии.",
    ),
    (
        "0004-versioned-ingestion",
        "Перезаписывает ли повторная загрузка доказательства незаметно?",
        "повторная загрузка не перезаписывает доказательства незаметно.",
    ),
    (
        "0005-dense-retrieval",
        "Какой провайдер используется для dense-поиска?",
        "CPU-совместимый E5-провайдер",
    ),
    ("0005-dense-retrieval", "Что требует смена спецификации индекса?", "пересобранного индекса."),
    ("0006-bm25-and-rrf", "Каким методом объединяются BM25 и dense-ранги?", "через RRF."),
    ("0006-bm25-and-rrf", "Что объединяет RRF?", "не калиброванные scores;"),
    (
        "0007-cross-encoder-reranking",
        "По чему cross-encoder сортирует кандидатов?",
        "по его logit.",
    ),
    (
        "0007-cross-encoder-reranking",
        "Почему сохраняется candidate recall?",
        "не маскирует потерю на предыдущем этапе.",
    ),
    ("0008-basic-rag", "Когда отклоняются неверные цитаты?", "отклоняются до выдачи ответа."),
    (
        "0008-basic-rag",
        "Что проверяет детерминированная подмена?",
        "контракт, а не фактическую полезность текста.",
    ),
    (
        "0009-rag-evaluation",
        "Какие компоненты качества оцениваются отдельно?",
        "retrieval, покрытие контекста,",
    ),
    ("0009-rag-evaluation", "К чему привязывается семантическая рубрика?", "к точному отчёту."),
    ("0010-http-api", "Что предоставляет FastAPI?", "загрузку, запрос, health и SSE"),
    (
        "0010-http-api",
        "Ускоряет ли async def CPU-вычисления сам по себе?",
        "`async def` сам по себе не ускоряет CPU-вычисления.",
    ),
    (
        "0011-bounded-agent",
        "Какие инструменты доступны агенту?",
        "поиском, каталогом и калькулятором.",
    ),
    (
        "0011-bounded-agent",
        "Гарантируют ли лимиты своевременный полезный отказ?",
        "не гарантируют своевременный полезный отказ.",
    ),
    ("0012-mcp-calculator", "Какой числовой тип использует MCP-калькулятор?", "`Decimal`"),
    ("0012-mcp-calculator", "Нужен ли сетевой сервис для проверки MCP?", "без сетевого сервиса."),
    (
        "0013-offline-agent-tool-review",
        "По каким данным оцениваются сценарии агента?",
        "по сохранённым неизменяемым трассам",
    ),
    (
        "0013-offline-agent-tool-review",
        "Равен ли статус выполнения семантической правильности?",
        "семантической правильности;",
    ),
    ("0014-explicit-observability", "Как включается экспорт в Langfuse?", "включается настройкой"),
    (
        "0014-explicit-observability",
        "Что связывает логи и спаны запроса?",
        "коррелированные JSON-логи и OpenTelemetry-спаны",
    ),
    (
        "ingestion",
        "Какой статус даёт повторная загрузка того же файла?",
        "статус будет `unchanged`",
    ),
    ("ingestion", "Какие форматы показывает схема загрузки?", "Markdown / TXT, UTF-8"),
    (
        "sparse_hybrid",
        "Как BM25 учитывает длину текста?",
        "с учётом их частоты, редкости и длины текста.",
    ),
    (
        "sparse_hybrid",
        "Что делает токенизатор с русскими буквами?",
        "Русские буквы сохраняются, но не переводятся.",
    ),
    (
        "reranking",
        "Какая модель выбрана для reranking?",
        "`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`",
    ),
    ("reranking", "Сколько кандидатов в исходной конфигурации reranker?", "20 кандидатов и top-5."),
    (
        "basic_rag",
        "Что происходит с чанком, который не входит в бюджет контекста?",
        "Слишком большой чанк пропускается",
    ),
    (
        "basic_rag",
        "В чём выражен лимит размера prompt по умолчанию?",
        "24000 UTF-8 байт сериализованного массива сообщений",
    ),
    (
        "api",
        "Какая архитектура обработчиков API выбрана?",
        "async HTTP-граница с вынесением существующих блокирующих операций в поток.",
    ),
    (
        "api",
        "Когда запускаются миграции API?",
        "Миграции запускаются явно, не при импорте или старте API.",
    ),
    (
        "observability",
        "Чем трасса отличается от метрики?",
        "Трасса — связанное дерево интервалов работы одного запроса.",
    ),
    (
        "observability",
        "Используется ли глобальный OTEL TracerProvider для Langfuse?",
        "не глобальный provider других библиотек.",
    ),
]
ABSENT = [
    "Какой точный процент ошибок был у production-трафика в августе 2026?",
    "Какой пароль у production PostgreSQL?",
    "Сколько пользователей ежедневно обращаются к публичному API?",
    "Какая цена Qdrant Cloud установлена для этого проекта?",
    "Какой номер GPU используется для обучения E5?",
    "В каком регионе размещён production-кластер Kubernetes?",
    "Какая фамилия у независимого человеческого рецензента набора?",
    "Какой SLA гарантирован внешним LLM-провайдером?",
    "Сколько запросов в секунду выдерживает production API?",
    "Каков секретный токен Langfuse?",
]
DISTRACTORS = (
    "architecture",
    "dense_retrieval",
    "rag_evaluation",
    "agent",
    "mcp",
    "project_brief",
    "roadmap",
    "results",
    "interview_notes",
    "agent_evaluation",
)


def main() -> None:
    if TARGET.exists():
        raise FileExistsError(TARGET)
    corpus_dir = TARGET / "corpus"
    corpus_dir.mkdir(parents=True)
    keys = {case[0] for case in CASES} | set(DISTRACTORS)
    documents = []
    contents = {}
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    for key in sorted(keys):
        path = ROOT / ("docs/adr" if key.startswith("00") else "docs") / f"{key}.md"
        destination = corpus_dir / f"{key}.md"
        shutil.copyfile(path, destination)
        contents[key] = destination.read_text(encoding="utf-8")
        documents.append(
            {
                "key": key,
                "path": f"corpus/{key}.md",
                "source_uri": (
                    f"https://github.com/SpaceZeroed/agentic-rag/blob/{commit}/"
                    f"{path.relative_to(ROOT).as_posix()}"
                ),
                "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(),
            }
        )
    questions = []
    for number, (key, query, evidence) in enumerate(CASES, 1):
        if evidence not in contents[key]:
            raise ValueError(f"Excerpt missing: {key}: {evidence}")
        questions.append(
            {
                "id": f"answer_{number:02}",
                "language": "ru",
                "query": query,
                "answerable": True,
                "reference_answer": evidence,
                "evidence": [{"document": key, "text": evidence}],
            }
        )
    for number, query in enumerate(ABSENT, 1):
        questions.append(
            {
                "id": f"absent_{number:02}",
                "language": "ru",
                "query": query,
                "answerable": False,
                "reference_answer": "В корпусе нет подтверждения; следует отказаться от ответа.",
                "evidence": [],
            }
        )
    english = [
        (
            "Which store is the source of truth for document text?",
            "0003-document-and-index-ownership",
            "PostgreSQL.",
        ),
        ("Which ranking fusion method combines dense and BM25?", "0006-bm25-and-rrf", "через RRF."),
        ("Which numeric type is used by the MCP calculator?", "0012-mcp-calculator", "`Decimal`"),
        (
            "What is the status of ingesting an unchanged file?",
            "ingestion",
            "статус будет `unchanged`",
        ),
        (
            "Does the default prompt budget count tokens?",
            "basic_rag",
            "24000 UTF-8 байт сериализованного массива сообщений",
        ),
    ]
    for number, (query, key, excerpt) in enumerate(english, 1):
        questions.append(
            {
                "id": f"english_{number:02}",
                "language": "en",
                "query": query,
                "answerable": True,
                "reference_answer": excerpt,
                "evidence": [{"document": key, "text": excerpt}],
            }
        )
    multi = [
        (
            "Где хранится канонический текст и каким провайдером строятся dense-векторы?",
            [
                ("0003-document-and-index-ownership", "PostgreSQL."),
                ("0005-dense-retrieval", "CPU-совместимый E5-провайдер"),
            ],
        ),
        (
            "Каким методом объединяются BM25 и dense, и как затем сортируются кандидаты reranker?",
            [
                ("0006-bm25-and-rrf", "через RRF."),
                ("0007-cross-encoder-reranking", "по его logit."),
            ],
        ),
        (
            "Когда проверяются ссылки ответа и что дополнительно оценивает RAG-рубрика?",
            [
                ("0008-basic-rag", "отклоняются до выдачи ответа."),
                ("0009-rag-evaluation", "поддержку утверждений и отказ."),
            ],
        ),
        (
            "Что предоставляет HTTP API и гарантируют ли лимиты агента своевременный отказ?",
            [
                ("0010-http-api", "загрузку, запрос, health и SSE"),
                ("0011-bounded-agent", "не гарантируют своевременный полезный отказ."),
            ],
        ),
        (
            "Каков статус повторной загрузки и проверяема ли старая ссылка?",
            [
                ("ingestion", "статус будет `unchanged`"),
                ("0004-versioned-ingestion", "Ссылку можно проверить после смены текущей версии."),
            ],
        ),
    ]
    for number, (query, pairs) in enumerate(multi, 1):
        questions.append(
            {
                "id": f"multi_{number:02}",
                "language": "ru",
                "query": query,
                "answerable": True,
                "reference_answer": " ".join(text for _, text in pairs),
                "evidence": [{"document": key, "text": excerpt} for key, excerpt in pairs],
            }
        )
    data = {
        "schema_version": 1,
        "name": "local_docs_v2_candidate",
        "label_policy": (
            "Assistant-authored draft questions and evidence on frozen repository documentation. "
            "Unanswerability is author judgment. Human audit pending; "
            "not independent held-out quality evidence."
        ),
        "max_chars": 1200,
        "overlap": 200,
        "documents": documents,
        "questions": questions,
    }
    (TARGET / "dataset.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Created {len(documents)} documents and {len(questions)} questions")


if __name__ == "__main__":
    main()
