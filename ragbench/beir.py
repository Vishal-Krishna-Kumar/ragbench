"""Download a public BEIR dataset and convert it to the ragbench layout.

python -m ragbench.beir scifact --out data/scifact
python -m ragbench.cli data/scifact

BEIR ships ``corpus.jsonl`` / ``queries.jsonl`` with an ``_id`` key and the relevance
judgements as ``qrels/<split>.tsv``. ragbench uses ``id`` keys and a ``qrels.jsonl`` file,
so the conversion is a rename plus a filter to the queries that are judged in the split.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

BEIR_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def convert_beir(src: str | Path, out: str | Path, split: str = "test") -> dict[str, int]:
    """Convert an extracted BEIR dataset directory into ragbench's three JSONL files.

    Only queries with at least one positive judgement in ``split`` are kept, and judgements
    with a score of zero are dropped. Returns the number of documents, queries and judgements.
    """
    src, out = Path(src), Path(out)
    qrels_path = src / "qrels" / f"{split}.tsv"
    if not qrels_path.exists():
        available = sorted(p.stem for p in (src / "qrels").glob("*.tsv")) if (src / "qrels").exists() else []
        raise FileNotFoundError(f"no qrels for split '{split}' in {src} (available: {available})")

    qrels: list[dict] = []
    with qrels_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            score = int(row["score"])
            if score > 0:
                qrels.append({"query_id": row["query-id"], "doc_id": row["corpus-id"], "relevance": score})

    judged = {q["query_id"] for q in qrels}
    corpus = [
        {"id": d["_id"], "title": d.get("title", ""), "text": d.get("text", "")}
        for d in _read_jsonl(src / "corpus.jsonl")
    ]
    queries = [
        {"id": q["_id"], "text": q["text"]} for q in _read_jsonl(src / "queries.jsonl") if q["_id"] in judged
    ]

    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(out / "corpus.jsonl", corpus)
    _write_jsonl(out / "queries.jsonl", queries)
    _write_jsonl(out / "qrels.jsonl", qrels)
    return {"docs": len(corpus), "queries": len(queries), "qrels": len(qrels)}


def download_beir(name: str, cache: str | Path) -> Path:
    """Download and extract ``<name>.zip`` from the BEIR mirror; returns the extracted directory."""
    cache = Path(cache)
    cache.mkdir(parents=True, exist_ok=True)
    extracted = cache / name
    if not (extracted / "corpus.jsonl").exists():
        archive = cache / f"{name}.zip"
        if not archive.exists():
            urllib.request.urlretrieve(BEIR_URL.format(name=name), archive)
        with zipfile.ZipFile(archive) as zf:
            zf.extractall(cache)
    return extracted


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fetch a BEIR dataset and convert it for ragbench.")
    ap.add_argument("name", help="BEIR dataset name, e.g. scifact, nfcorpus, fiqa")
    ap.add_argument("--out", help="output directory (default: data/<name>)")
    ap.add_argument("--split", default="test", help="qrels split to use (default: test)")
    ap.add_argument("--cache", default=".beir_cache", help="where to keep the downloaded archive")
    args = ap.parse_args(argv)

    out = Path(args.out or f"data/{args.name}")
    counts = convert_beir(download_beir(args.name, args.cache), out, args.split)
    print(
        f"{args.name} ({args.split}): {counts['docs']} docs, {counts['queries']} queries, "
        f"{counts['qrels']} judgements -> {out}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
