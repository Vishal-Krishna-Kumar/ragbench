# Contributing to ragbench

Thanks for taking a look. ragbench is small on purpose: three retrievers, standard IR metrics,
a CLI and a FastAPI service. Contributions that keep it small and measurable are very welcome.

## Set up

```bash
git clone https://github.com/Vishal-Krishna-Kumar/ragbench.git
cd ragbench
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
```

## Before you open a pull request

```bash
ruff check .                         # lint
ruff format --check .                # formatting (run `ruff format .` to fix)
mypy ragbench                        # type check
pytest -q                            # unit and API tests
python -m ragbench.cli data/sample   # the benchmark table should still print
```

CI runs the same commands on Python 3.10, 3.12 and 3.13, then builds the Docker image and
smoke-tests the container.

## What makes a good change

- **One idea per pull request.** Small PRs get reviewed quickly.
- **A test for every behaviour change.** Metrics and retrievers are pure functions, so tests are cheap.
- **Numbers you actually measured.** If a change affects quality or latency, paste the before and after
  table from `python -m ragbench.cli data/sample` (or your own dataset) into the PR description.
- **No new heavy dependencies in the core package.** The retrievers are NumPy-only by design. Optional
  integrations belong behind an extra in `pyproject.toml`.

## Adding a retriever

1. Implement a class in `ragbench/retrievers.py` with `fit(doc_ids, docs)` and `search(query, k)`.
   `search` returns a list of `(doc_id, score)` pairs, best first.
2. Give it a `name` attribute and register it in `REGISTRY`.
3. Add it to the parametrised tests in `tests/test_retrievers.py`.

It will then show up in the CLI and the API without further changes.

## Reporting bugs and asking for features

Use the issue templates. For bugs, the smallest dataset or request that reproduces the problem
helps the most. Issues labelled `good first issue` are a good place to start.
