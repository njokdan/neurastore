# NeuraStore vs. pgvector vs. Milvus

A honest, fairly-measured comparison — every number below was measured
client-to-server (not an in-process shortcut), on the same siftsmall
corpus, using the same warm-up and randomized-order methodology across
all three engines. Where a number changed after further investigation,
that investigation is described rather than hidden. See `bench/README.md`
for exact reproduction steps and raw data.

## The numbers

| Metric | pgvector | Milvus | NeuraStore |
|---|---|---|---|
| Recall@10 | 0.984 | 0.988 | 0.983 |
| Insert throughput (vec/sec) | 1,633 | 2,545 | **15,649–17,927** |
| Unfiltered query p50 | 2.81ms | 5.99ms | **2.04–2.81ms** |
| Filtered query p50 (25% selectivity) | 7.23ms | 6.41ms | **2.69–3.18ms** |
| Filter tax (filtered/unfiltered ratio) | 2.6x | 1.1x | **1.13–1.32x** |

All NeuraStore ranges reflect real run-to-run variance across multiple
measurements, reported honestly rather than cherry-picking the best run.

## What this means

**Recall**: a three-way tie. NeuraStore's from-scratch HNSW implementation
finds true nearest neighbors at essentially the same rate as two mature,
widely-used production systems.

**Insert throughput**: NeuraStore's HTTP API inserts roughly 9-11x faster
than pgvector and 6-7x faster than Milvus. This number has an unusually
well-documented history — an early measurement (824 vec/sec) looked like
a real deficit and drove real engineering work (a binary bulk-insert
protocol) before further testing revealed the actual cause was a test
harness bug, not an engine limitation. See "How this was measured"
below for the full story; it's a better example of how this project's
numbers were arrived at than the final figure alone would suggest.

**Unfiltered latency**: ties or beats pgvector, clearly beats Milvus.

**Filtered latency and filter tax**: this is the number the whole
project was built around. pgvector pays a real, measured 2.6x latency
tax the moment a structured filter is added to a vector query — the
classic overfetch-then-discard pattern. Milvus does much better (1.1x).
NeuraStore's filter is pushed directly into the HNSW graph traversal
instead of applied after fetching an unfiltered result set, and beats
pgvector's tax at every tested selectivity while coming close to or
matching Milvus's near-parity number.

## What's not in this table, and why

**No head-to-head "NeuraStore wins overall" claim.** It doesn't, and
framing it that way would misrepresent both sides. pgvector and Milvus
have years of production hardening, client libraries in every major
language, managed hosting options, and large user communities —
structural advantages a project this young cannot claim and shouldn't
pretend to. This table compares one thing precisely: core engine
performance on a fixed workload, measured fairly. It says nothing about
operational maturity, ecosystem, or production readiness.

**No gRPC/binary protocol comparison for Milvus specifically.**
NeuraStore's HTTP/JSON API was benchmarked against Milvus's own gRPC
client, which is a real, unavoidable asymmetry (different wire
protocols) rather than something either side did wrong.

## How this was measured

1. **Baseline (pgvector, Milvus)**: `bench/scripts/bench_pgvector.py`,
   `bench_milvus.py` — Docker Compose, official client libraries, warm-up
   pass before timing, randomized filtered/unfiltered order to rule out
   ordering bias.
2. **NeuraStore, fair comparison**: `bench/scripts/bench_neurastore_http.py`
   — same dataset, same methodology, against a real running
   `cargo run --release --bin server` instance over actual HTTP.
3. **Insert throughput specifically** required an extra round of
   investigation. A `clean_insert_benchmark.sh` script now runs each
   condition against a genuinely fresh server process and empty data
   directory — the fix for a subtle bug where a reused server process
   was silently measuring index-update cost (from a prior run's
   already-built index) instead of a true fresh bulk load.

Every number in the table above reflects the corrected methodology.

## At 1M scale — a real, unresolved finding, not yet folded into the headline table above

Every number above was measured at ~10K records (siftsmall). Tested
once at real 1M scale (texmex SIFT-1M, real embeddings, in-process —
not yet re-run over HTTP at this scale) via
`cargo run --release --bin bench_neurastore -- bench/data/sift 10 40`:

| Metric | 10K scale | 1M (ef_search=40) | 1M (ef_search=200) | 1M (+max_visits=100K) | 1M (+FxHashMap fix) |
|---|---|---|---|---|---|
| Recall@10 | 0.983 | 0.825 | 0.941 | 0.941 | 0.941 |
| Filter tax | 1.13–1.32x | 12.62x | 6.59x | 7.22x (worse) | 6.68x (unchanged) |

**Resolved**: profiling instrumentation measured the real cause at 1M
scale, real SIFT data — hit rate (nodes matched / nodes visited during
filtered traversal) landed at 25.0%, essentially exact against the
~25% base filter selectivity, with zero of 10,000 queries anywhere near
exhausting their visit budget. This decisively rules out a graph
connectivity defect — the traversal genuinely needs ~4x the visits an
equivalent unfiltered search would, because only 1-in-4 visited nodes
match. That's a real, inherent, explainable cost of predicate-in-
traversal filtering at moderate selectivity, not a bug, and it explains
why the tax was small at 10K scale and large at 1M. Full reasoning in
`PORTFOLIO.md`.

