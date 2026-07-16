# NeuraStore

A unified vector + structured-filter database engine, written in Rust
— hybrid similarity + filter search without the overfetch tax most
systems pay for that combination.

## Try it in a couple of minutes

```bash
docker compose up --build
curl http://localhost:8080/health
```
(First build compiles from scratch and takes a few minutes — one-time
cost. Subsequent starts are fast.)

```bash
pip install -e client/python
```
```python
from neurastore_client import NeuraStoreClient
client = NeuraStoreClient("http://localhost:8080")
client.insert(1, [0.1, 0.2, 0.3], metadata={"category": "docs"})
client.build_index()
results = client.search([0.1, 0.2, 0.3], k=5)
```

## Why this exists

Most vector databases pay a real latency penalty the moment a
similarity search gets a structured filter attached — either they
overfetch an unfiltered result set and discard most of it, or they
maintain the filter and the vector index as two systems that can drift
out of sync. Measured directly: pgvector pays about a 2.6x tax on
filtered queries versus unfiltered ones.

NeuraStore pushes the filter directly into the HNSW graph traversal
instead. Three real, fairly-measured wins — full methodology, every
number reproducible, in [`COMPARISON.md`](./COMPARISON.md):

| Metric | pgvector | Milvus | NeuraStore |
|---|---|---|---|
| Recall@10 | 0.984 | 0.988 | 0.983 |
| Insert (vec/sec) | 1,633 | 2,545 | 15,649–17,927 |
| Filtered-query latency tax | 2.6x | ~1.1x | 1.13–1.32x |

**This isn't trying to replace Pinecone or Milvus.** It's a narrow,
proven bet on one specific problem. The honest gaps — one vector type
(dense float32 only), one index type (HNSW only), single-node only, no
hybrid/full-text search — are in [`PORTFOLIO.md`](./PORTFOLIO.md) and
`COMPARISON.md`, stated plainly, not hidden.

## Where the full story lives

This README is deliberately kept lean — the front door, not the whole
house. The complete phase-by-phase build log — every architecture
decision, every benchmark, every bug found and fixed along the way,
left visible rather than cleaned up afterward — lives in
[`HISTORY.md`](./HISTORY.md). That's where the real engineering
narrative is: how the storage engine was built and tested, how the
HNSW index was built from scratch and proven correct, the full
insert-throughput investigation, every phase's decisions and tradeoffs.

**If you want the short version first, read [`PORTFOLIO.md`](./PORTFOLIO.md)
instead** — it's built specifically to be that entry point.
`COMPARISON.md` has the exact benchmark methodology and raw numbers.
`HISTORY.md` has everything else, in full.
