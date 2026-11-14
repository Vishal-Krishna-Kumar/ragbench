"""Command line: compare retrievers on a dataset.

python -m ragbench.cli data/sample
python -m ragbench.cli data/sample --format json
python -m ragbench.cli data/sample --retrievers bm25 hybrid_rrf --format csv
python -m ragbench.cli data/sample --json results.json
python -m ragbench.cli data/sample --misses 1          # list queries each retriever gets wrong at rank 1
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys

from .data import load_dataset
from .evaluate import evaluate_retriever, missed_queries
from .retrievers import REGISTRY, make_retriever

COLUMNS = ["hit@1", "hit@3", "recall@5", "mrr", "ndcg@5", "ndcg@10"]
FORMATS = ("markdown", "table", "json", "csv")


def _summary(res: dict) -> dict:
    """One result without the per-query detail."""
    return {k: res[k] for k in ("retriever", "num_queries", "metrics", "latency_ms")}


def render_markdown(results: list[dict], dataset: str, num_docs: int) -> str:
    n = results[0]["num_queries"] if results else 0
    lines = [
        f"Dataset: {dataset}  ({num_docs} docs, {n} judged queries)",
        "",
        "| retriever | " + " | ".join(COLUMNS) + " | mean latency (ms) |",
        "|---|" + "---|" * (len(COLUMNS) + 1),
    ]
    for res in results:
        cells = [f"{res['metrics'][c]:.3f}" for c in COLUMNS]
        lines.append(f"| {res['retriever']} | " + " | ".join(cells) + f" | {res['latency_ms']['mean']:.2f} |")
    return "\n".join(lines)


def render_table(results: list[dict], dataset: str, num_docs: int) -> str:
    """Aligned plain-text table for reading in a terminal."""
    n = results[0]["num_queries"] if results else 0
    header = ["retriever", *COLUMNS, "mean latency (ms)"]
    rows = [
        [
            res["retriever"],
            *(f"{res['metrics'][c]:.3f}" for c in COLUMNS),
            f"{res['latency_ms']['mean']:.2f}",
        ]
        for res in results
    ]
    widths = [max(len(line[i]) for line in [header, *rows]) for i in range(len(header))]

    def align(cells: list[str]) -> str:
        rest = [cell.rjust(width) for cell, width in zip(cells[1:], widths[1:], strict=True)]
        return "  ".join([cells[0].ljust(widths[0]), *rest])

    rule = "  ".join("-" * width for width in widths)
    title = f"Dataset: {dataset}  ({num_docs} docs, {n} judged queries)"
    return "\n".join([title, "", align(header), rule, *(align(row) for row in rows)])


def render_json(results: list[dict], dataset: str, num_docs: int) -> str:
    payload = {"dataset": dataset, "num_docs": num_docs, "results": [_summary(r) for r in results]}
    return json.dumps(payload, indent=2)


def render_csv(results: list[dict], dataset: str, num_docs: int) -> str:
    buf = io.StringIO()
    writer = csv.writer(buf, lineterminator="\n")
    writer.writerow(["retriever", "num_queries", *COLUMNS, "latency_mean_ms", "latency_p95_ms"])
    for res in results:
        writer.writerow(
            [
                res["retriever"],
                res["num_queries"],
                *(res["metrics"][c] for c in COLUMNS),
                res["latency_ms"]["mean"],
                res["latency_ms"]["p95"],
            ]
        )
    return buf.getvalue().rstrip("\n")


RENDERERS = {
    "markdown": render_markdown,
    "table": render_table,
    "json": render_json,
    "csv": render_csv,
}
MISS_CUTOFFS = (1, 3, 5, 10)


def render_misses(results: list[dict], queries: dict[str, str], k: int) -> str:
    """Markdown section listing, per retriever, the queries with no relevant hit in the top k."""
    lines = [f"Queries with no relevant document in the top {k}:"]
    for res in results:
        missed = missed_queries(res, queries, k)
        lines.append("")
        lines.append(f"{res['retriever']}: {len(missed)} of {res['num_queries']}")
        for row in missed:
            lines.append(f"- [{row['query_id']}] {row['query']}  ->  retrieved {', '.join(row['top'])}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Benchmark retrievers for RAG.")
    ap.add_argument("dataset", help="directory with corpus.jsonl, queries.jsonl, qrels.jsonl")
    ap.add_argument("--retrievers", nargs="+", default=list(REGISTRY), choices=list(REGISTRY))
    ap.add_argument(
        "--format",
        choices=FORMATS,
        default="markdown",
        help="how to print the summary: markdown (default), table (aligned for the terminal), json or csv",
    )
    ap.add_argument(
        "--misses",
        type=int,
        choices=MISS_CUTOFFS,
        metavar="K",
        help="also list the queries with no relevant document in the top K (1, 3, 5 or 10)",
    )
    ap.add_argument("--json", help="also write full results (incl. per-query) to this file")
    args = ap.parse_args(argv)

    ds = load_dataset(args.dataset)
    results = []
    for name in args.retrievers:
        r = make_retriever(name).fit(ds.doc_ids, ds.docs)
        results.append(evaluate_retriever(r, ds.queries, ds.qrels))

    if args.misses and args.format == "json":
        payload = json.loads(render_json(results, args.dataset, len(ds.doc_ids)))
        for summary, res in zip(payload["results"], results, strict=True):
            summary[f"misses@{args.misses}"] = missed_queries(res, ds.queries, args.misses)
        print(json.dumps(payload, indent=2))
    else:
        print(RENDERERS[args.format](results, args.dataset, len(ds.doc_ids)))
        if args.misses and args.format == "markdown":
            print()
            print(render_misses(results, ds.queries, args.misses))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump(results, fh, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
