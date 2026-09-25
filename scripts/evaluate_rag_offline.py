"""Run frozen-dataset BM25/context metrics without external services."""

import argparse
import json
from pathlib import Path

from agentic_rag.evaluation.rag_runner import run_offline_bm25_evaluation


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Output already exists; choose a new report path")
    report = run_offline_bm25_evaluation(args.dataset, k=args.k)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as output:
        json.dump(report, output, ensure_ascii=False, indent=2)
        output.write("\n")
    print(
        json.dumps(
            {"output": str(args.output), "metrics": report["metrics"], "errors": report["errors"]}
        )
    )


if __name__ == "__main__":
    main()
