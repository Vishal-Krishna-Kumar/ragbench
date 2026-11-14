import json

import pytest

from ragbench.beir import convert_beir
from ragbench.data import load_dataset


def _make_beir_dir(root):
    (root / "qrels").mkdir(parents=True)
    corpus = [
        {"_id": "d1", "title": "BM25", "text": "lexical ranking function"},
        {"_id": "d2", "title": "", "text": "dense embeddings for semantic search"},
    ]
    queries = [
        {"_id": "q1", "text": "lexical ranking"},
        {"_id": "q2", "text": "semantic search"},
        {"_id": "q3", "text": "only judged in the train split"},
    ]
    (root / "corpus.jsonl").write_text("\n".join(json.dumps(r) for r in corpus), encoding="utf-8")
    (root / "queries.jsonl").write_text("\n".join(json.dumps(r) for r in queries), encoding="utf-8")
    (root / "qrels" / "test.tsv").write_text(
        "query-id\tcorpus-id\tscore\nq1\td1\t2\nq2\td2\t1\nq2\td1\t0\n", encoding="utf-8"
    )
    (root / "qrels" / "train.tsv").write_text("query-id\tcorpus-id\tscore\nq3\td1\t1\n", encoding="utf-8")
    return root


def test_convert_beir_writes_a_dataset_ragbench_can_load(tmp_path):
    counts = convert_beir(_make_beir_dir(tmp_path / "beir"), tmp_path / "out")
    assert counts == {"docs": 2, "queries": 2, "qrels": 2}  # the zero-score judgement is dropped

    ds = load_dataset(tmp_path / "out")
    assert ds.doc_ids == ["d1", "d2"]
    assert set(ds.queries) == {"q1", "q2"}  # q3 is not judged in the test split
    assert ds.qrels == {"q1": {"d1": 2}, "q2": {"d2": 1}}


def test_convert_beir_can_use_another_split(tmp_path):
    counts = convert_beir(_make_beir_dir(tmp_path / "beir"), tmp_path / "out", split="train")
    assert counts["queries"] == 1
    assert set(load_dataset(tmp_path / "out").queries) == {"q3"}


def test_convert_beir_names_the_available_splits_when_one_is_missing(tmp_path):
    with pytest.raises(FileNotFoundError, match=r"available: \['test', 'train'\]"):
        convert_beir(_make_beir_dir(tmp_path / "beir"), tmp_path / "out", split="dev")
