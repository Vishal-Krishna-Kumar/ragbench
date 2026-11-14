import csv
import io
import json

import pytest

from ragbench.cli import COLUMNS, main
from ragbench.retrievers import REGISTRY

SAMPLE = "data/sample"


def test_default_output_is_a_markdown_table(capsys):
    assert main([SAMPLE]) == 0
    out = capsys.readouterr().out
    assert out.startswith("Dataset: data/sample")
    assert "| retriever | " + " | ".join(COLUMNS) in out
    for name in REGISTRY:
        assert f"| {name} |" in out


def test_json_format_is_parseable_and_has_no_per_query_detail(capsys):
    assert main([SAMPLE, "--format", "json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["dataset"] == SAMPLE and payload["num_docs"] > 0
    assert [r["retriever"] for r in payload["results"]] == list(REGISTRY)
    for res in payload["results"]:
        assert "per_query" not in res
        assert set(COLUMNS) <= set(res["metrics"])
        assert {"mean", "p95"} <= set(res["latency_ms"])


def test_csv_format_has_one_row_per_retriever(capsys):
    assert main([SAMPLE, "--retrievers", "bm25", "tfidf", "--format", "csv"]) == 0
    rows = list(csv.DictReader(io.StringIO(capsys.readouterr().out)))
    assert [r["retriever"] for r in rows] == ["bm25", "tfidf"]
    assert all(0.0 <= float(r["mrr"]) <= 1.0 for r in rows)
    assert "latency_p95_ms" in rows[0]


def test_json_file_keeps_per_query_detail(tmp_path, capsys):
    target = tmp_path / "results.json"
    assert main([SAMPLE, "--retrievers", "bm25", "--json", str(target)]) == 0
    capsys.readouterr()
    full = json.loads(target.read_text(encoding="utf-8"))
    assert full[0]["retriever"] == "bm25" and len(full[0]["per_query"]) == full[0]["num_queries"]


def test_unknown_format_is_rejected():
    with pytest.raises(SystemExit) as exc:
        main([SAMPLE, "--format", "yaml"])
    assert exc.value.code == 2
