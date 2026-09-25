"""Evaluate the project's BM25 implementation on BEIR SciFact document retrieval."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import defaultdict
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from agentic_rag.evaluation.metrics import retrieval_metrics
from agentic_rag.ingestion.models import Chunk, ChunkingConfig, PreparedDocument
from agentic_rag.retrieval.sparse import TOKENIZER_VERSION, BM25Config, BM25Index

URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"


def evaluate(archive: Path) -> dict[str, object]:
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    with zipfile.ZipFile(archive) as bundle:
        corpus = [json.loads(line) for line in bundle.read("scifact/corpus.jsonl").splitlines()]
        queries = {
            row["_id"]: row["text"]
            for line in bundle.read("scifact/queries.jsonl").splitlines()
            if (row := json.loads(line))
        }
        rows = csv.DictReader(
            bundle.read("scifact/qrels/test.tsv").decode("utf-8").splitlines(),
            delimiter="\t",
        )
        qrels: dict[str, dict[str, int]] = defaultdict(dict)
        for row in rows:
            qrels[row["query-id"]][row["corpus-id"]] = int(row["score"])

    config = ChunkingConfig(max_chars=100_000, overlap=0)
    documents = []
    original_ids = {}
    for row in corpus:
        identifier = str(row["_id"])
        title = row.get("title", "") or ""
        content = f"{title} {row['text']}".strip()
        document_id = uuid5(NAMESPACE_URL, f"beir:scifact:{identifier}")
        revision_id = uuid5(document_id, digest)
        chunk_id = uuid5(revision_id, "document")
        original_ids[chunk_id] = identifier
        chunk = Chunk(chunk_id, 0, content, 0, len(content), 1, 1)
        documents.append(
            PreparedDocument(
                document_id,
                revision_id,
                f"https://beir.ai/scifact/{identifier}",
                title,
                "text/plain",
                "",
                "",
                content,
                "beir",
                "one-document-one-chunk",
                config,
                (chunk,),
            )
        )
    if len(original_ids) != len(corpus):
        raise ValueError("Duplicate corpus IDs")
    index = BM25Index(documents, BM25Config())
    cases = []
    for query_id in sorted(qrels):
        ranked = [original_ids[item.chunk_id] for item in index.query(queries[query_id], 10)]
        grades = qrels[query_id]
        if not set(grades).issubset(set(row["_id"] for row in corpus)):
            raise ValueError(f"Qrels refer to absent documents: {query_id}")
        cases.append(
            {
                "query_id": query_id,
                "query": queries[query_id],
                "qrels": grades,
                "ranked": ranked,
                "metrics": {str(k): retrieval_metrics(ranked, grades, k) for k in (1, 3, 5, 10)},
            }
        )
    summary = {
        metric: sum(case["metrics"][str(k)][metric] for case in cases) / len(cases)
        for k in (1, 3, 5, 10)
        for metric in (f"recall@{k}", f"precision@{k}", f"mrr@{k}", f"ndcg@{k}")
    }
    return {
        "dataset": "BEIR SciFact test",
        "source": URL,
        "archive_sha256": digest,
        "corpus_documents": len(corpus),
        "evaluated_queries": len(cases),
        "unit": "whole document (title plus text), no application chunking",
        "retriever": "agentic_rag.retrieval.sparse.BM25Index",
        "tokenizer": TOKENIZER_VERSION,
        "bm25": {"k1": 1.2, "b": 0.75},
        "summary": summary,
        "cases": cases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(args.archive)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
