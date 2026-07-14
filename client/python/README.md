# neurastore-client

Python client for [NeuraStore](../../README.md), a unified vector +
structured-filter database engine.

## Install

From the repo (not yet published to PyPI):

```bash
cd client/python
pip install -e .
```

## Quickstart

```python
from neurastore_client import NeuraStoreClient

client = NeuraStoreClient("http://localhost:8080")

client.insert(1, [0.1, 0.2, 0.3], metadata={"category": "docs"})
client.insert(2, [0.9, 0.8, 0.7], metadata={"category": "code"})
client.build_index()

for hit in client.search([0.1, 0.2, 0.3], k=5):
    print(hit.id, hit.distance)

for hit in client.search_filtered([0.1, 0.2, 0.3], field="category", value="docs", k=5):
    print(hit.id, hit.distance)
```

Or as a context manager, which closes the connection pool when done:

```python
with NeuraStoreClient("http://localhost:8080") as client:
    client.insert(1, [0.1, 0.2, 0.3])
```

## Bulk loading

```python
client.insert_batch([
    {"id": 1, "vector": [0.1, 0.2], "metadata": {"category": "docs"}},
    {"id": 2, "vector": [0.3, 0.4], "metadata": {"category": "code"}},
])
```

There's also a `binary=True` option using a compact binary wire format
instead of JSON. It's real and correctness-tested, but its performance
advantage over plain JSON was investigated and **not confirmed** on the
hardware it was developed against — see the main repo's README for that
whole story. Don't assume it's faster without measuring your own setup;
default (`binary=False`) is the well-tested, unsurprising choice.

## Error handling

```python
from neurastore_client import NotFoundError, BadRequestError

try:
    record = client.get(999)
except NotFoundError:
    print("no such record")
```

Exception hierarchy:
- `NeuraStoreError` — base class, catch this for "anything went wrong"
- `ConnectionError` — server unreachable (not Python's built-in of the
  same name — see the package docstring)
- `NotFoundError` — record doesn't exist (HTTP 404)
- `BadRequestError` — malformed request, or querying before
  `build_index()` has been called (HTTP 400)
- `ServerError` — something broke server-side (HTTP 500)

## Vectors are plain lists, not numpy arrays

This client only depends on `requests` — no numpy required. Pass any
sequence of numbers (`list`, `tuple`, or a numpy array via `.tolist()`).

## CLI

Installing the package also installs a `neurastore` command:

```bash
export NEURASTORE_URL=http://localhost:8080   # or pass --url every time

neurastore health
neurastore insert --id 1 --vector 0.1,0.2,0.3 --metadata category=docs
neurastore insert --id 2 --vector 0.9,0.8,0.7 --metadata category=code
neurastore build-index
neurastore search --vector 0.1,0.2,0.3 --k 5
neurastore search-filtered --vector 0.1,0.2,0.3 --field category --value docs
neurastore stats
neurastore get --id 1
neurastore delete --id 1
```

Bulk load from a JSON file (a list of `{id, vector, metadata}` objects,
or `-` for stdin):

```bash
neurastore insert-batch --file records.json
neurastore insert-batch --file records.json --binary   # binary wire format
```

Add `--json` before the subcommand for machine-readable output:

```bash
neurastore --json stats
neurastore --json search --vector 0.1,0.2,0.3
```

Errors print a clear message to stderr and exit with status 1, instead
of a Python traceback — safe to script against
(`neurastore get --id 1 || echo "not found"`).

## Running tests

```bash
pip install -e ".[test]"
pytest
```

Unit tests (`test_client.py`) mock the HTTP layer and need no running
server. `test_integration.py` needs a real NeuraStore server running
(`cargo run --release --bin server -- ./data 8080` from the repo root)
and is skipped automatically if one isn't reachable.
