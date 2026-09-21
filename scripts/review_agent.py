"""Create unscored tool-use reviews or summarize existing hash-bound reviews offline."""

import argparse
from pathlib import Path

from agentic_rag.evaluation.agent_review import make_template, summarize_reviews, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    template = commands.add_parser("template")
    template.add_argument("--run-dir", type=Path, required=True)
    template.add_argument("--output", type=Path, required=True)
    summarize = commands.add_parser("summarize")
    summarize.add_argument("--run-dir", type=Path, nargs="+", required=True)
    summarize.add_argument("--tool-review", type=Path, nargs="+", required=True)
    summarize.add_argument("--quality-review", type=Path, nargs="+", required=True)
    summarize.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = (
            make_template(args.run_dir).model_dump()
            if args.command == "template"
            else summarize_reviews(args.run_dir, args.tool_review, args.quality_review)
        )
        write_json(args.output, result)
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f"Review failed: {exc}\n")
    print(f"Saved {args.output}; no model calls")


if __name__ == "__main__":
    main()
