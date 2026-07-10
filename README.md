# NeuraStore

A unified storage/query engine for AI-native workloads — hybrid vector +
structured-filter queries on data that's still being written, in one
engine, without a separate ETL/reindex pipeline.

## Headline claim (what we're proving)

Sub-millisecond hybrid vector + structured-filter queries on live-writing
data, benchmarked head-to-head against pgvector and Milvus.

## Status: Phase 1 — Storage Engine Core (complete)

Phase 0 proved durability (WAL + memtable + crash recovery). Phase 1
adds the actual LSM behavior on top:

- **SSTable** (`src/sstable.rs`) — immutable, sorted, on-disk table.
  Physically splits each record into a **row-oriented metadata blob**
  and a **columnar vector blob** (contiguous f32 arrays), linked by an
  index sorted by record id. Written via write-to-temp-then-rename, so
  a crash mid-write never leaves a partial file for a reader to trip
  over. 6 tests cover roundtrip correctness, tombstone survival, sort
  order, missing keys, corrupted magic bytes, and empty vectors/metadata.
- **Engine** (`src/engine.rs`) — now manages a directory (`wal.log` +
  `NNNNNN.sst` files) instead of a single WAL file. Memtable flushes to
  a new SSTable once it crosses a threshold (or on explicit `flush()`);
  reads check memtable first, then SSTables newest-to-oldest, stopping
  at the first match (a tombstone match means "definitely deleted," not
  "keep looking" — this is what makes deletes survive across a flush).
  9 tests cover flush mechanics, multi-SSTable reads, shadowing
  (newer write/delete overriding an older flushed value), restart
  recovery post-flush, and — the Phase 1 checkpoint's explicit bar —
  **100,000 records inserted, flushed across 10+ SSTables, and read
  back correctly** (including at SSTable boundaries).

Run `cargo test --release` — 22 tests total (Phase 0 + Phase 1),
~70s including the 100K-record scale test.

Run the demo twice to see SSTable-based recovery (not just WAL replay):

```bash
cargo run -- ./data   # inserts 3 records, flushes to 000001.sst
cargo run -- ./data   # "restart" -- recovers from the SSTable, wal.log is empty
```

## Status: Phase 2 — Static Vector Index (complete)

A from-scratch HNSW (Hierarchical Navigable Small World) implementation,
built on top of the Phase 1 storage engine:

- **HNSW core** (`src/hnsw.rs`) — layered graph, greedy best-first search,
  squared-L2 distance (matches the `vector_l2_ops`/`L2` metric used in
  the pgvector/Milvus baseline, so results are directly comparable). 7
  tests, including one that documents a real, non-obvious finding (see
  below) rather than hiding it.
- **VectorIndex** (`src/vector_index.rs`) — maps the engine's `RecordId`
  space to HNSW's internal dense node-id space. 3 tests.
- **Engine integration** — `Engine::build_index()` builds a fresh index
  from a snapshot of all live records; `Engine::search_knn()` queries it.
  Explicit build step, not automatic-on-write — Phase 2 is a *static*
  index (build once, then query); making it update incrementally as
  writes happen is Phase 3's job.
- **`bin/bench_neurastore`** — reads the same `.fvecs`/`.ivecs` SIFT
  dataset files and computes the same metrics (insert throughput, build
  time, latency percentiles, recall@k) as `bench/scripts/bench_pgvector.py`
  and `bench_milvus.py`, so all three engines' numbers land in the same
  table. Run with:
  ```bash
  python bench/scripts/prepare_dataset.py --mode siftsmall
  cargo run --release --bin bench_neurastore -- bench/data/siftsmall 10 40
  ```

### A real finding worth knowing before you tune parameters

While testing, a self-lookup query (searching with a vector's own value)
sometimes failed to find itself even with `ef_search` far larger than
the whole corpus. Root cause: HNSW's random per-node level assignment
needs *enough points per cluster* that at least one lands on an upper
graph layer — otherwise that whole cluster can become an island with no
path back to the global entry point, and no query-time search budget
fixes that; it's a construction-time density requirement, not a search
bug. `hnsw::tests::sparse_clusters_can_strand_a_whole_cluster_from_the_entry_point`
reproduces and documents this deliberately, since it's exactly the kind
of failure mode Phase 3 needs to keep in mind once the corpus grows
incrementally instead of being built all at once.

## Status: Phase 3 — Incremental/Streaming HNSW (complete)

This is the phase flagged from the start as the real wedge — not "can
HNSW do approximate search" (Phase 2 answered that), but "can the index
grow continuously, safely, alongside real read/write traffic, without a
full rebuild." Three concrete gaps closed:

- **Incremental insert without rebuild** (`src/vector_index.rs`,
  `src/engine.rs`) — `VectorIndex::insert()` adds one record to an
  *existing* index; `VectorIndex::build()` is now literally implemented
  as a loop of `insert()` calls (no separate "batch algorithm" to drift
  out of sync). `Engine::put`/`put_batch`/`delete` propagate into the
  index automatically once `build_index()` has been called once —
  calling it a second time is optional (useful for periodic
  tombstone-compaction, not required for correctness).
- **Soft-delete tombstones** — HNSW has no cheap way to physically
  remove a graph node (its neighbors reference it; repairing that is
  expensive). Handled the same way production systems like Milvus do:
  mark-and-filter now (`VectorIndex::delete`), reclaim space via a future
  periodic rebuild later — a documented, known gap, not a silent one.
  Search over-fetches from the graph to compensate for tombstoned
  results getting filtered out, so `k` results still come back even with
  deletes present (heuristic, not a hard guarantee under heavy deletion
  — see `vector_index.rs`).
- **Real concurrency, proven with real OS threads** — the vector index
  is wrapped in `Arc<RwLock<VectorIndex>>`; `Engine::index_handle()`
  hands out a cloneable, independently-lockable reference. Multiple
  reader threads can hold concurrent read locks for `search`
  simultaneously; a writer briefly takes a write lock to insert or
  delete. Coarse-grained (a writer blocks all readers for one graph
  insert's duration), not lock-free — a deliberate, documented tradeoff,
  not an oversight. `engine::tests::concurrent_reads_and_writes_are_
  actually_thread_safe` spawns 4 concurrent reader threads and 1 writer
  thread with `std::thread::scope`, not just types that happen to compile
  — run 5x back-to-back with no flakiness before considering it solid.

11 new tests (46 total), including a regression guard proving `build()`
and an equivalent manual insert loop produce bit-identical results, and
`bin/bench_neurastore` now runs a real-data check: builds from 80% of
siftsmall, streams the remaining 20% in one record at a time via
`Engine::put` (post-build, no second rebuild), and re-measures
recall/latency to confirm neither degraded.

### A seed-sensitivity note carried over from Phase 2, now more visible

While building the incremental-growth benchmark, a synthetic smoke-test
dataset (used only because this sandboxed dev environment can't download
the real SIFT corpus) triggered the same cluster-stranding phenomenon
documented in Phase 2 — recall varied from 0.38 to 0.87 purely based on
RNG seed, on artificially perfectly-disjoint synthetic clusters. This is
*not* a Phase 3 regression: the same default seed (42) already achieves
0.983 recall reliably on the real siftsmall corpus (Phase 2's numbers),
which doesn't have this pathological structure. Included here because
it's exactly the kind of thing worth writing down rather than quietly
re-rolling a seed until a demo looks clean.

### Confirmed on real data (siftsmall)

Two full runs on actual hardware (not the sandbox), both agreeing:

| | Pre-growth (80% corpus) | Post-growth (100%, no rebuild) |
|---|---|---|
| Recall@10 | 0.785 (expected — 20% of true neighbors not yet inserted) | **0.983** (matches Phase 2's full-batch number exactly) |

Recall reaching the same 0.983 via incremental streaming as via one big
batch build is the real proof: growing the index continuously doesn't
produce a degraded approximation of a fresh build — it reaches the same
quality.

One honest tail-latency observation from the streaming runs: per-record
incremental insert latency was healthy in aggregate (p50 ~1.5-1.8ms, p99
~4-6ms) but had large outliers (max 94-115ms across two runs, ~20-30x
the p99). Two plausible, non-exclusive causes, not yet root-caused:
HNSW's random level assignment occasionally gives a node a much higher
graph level than usual (costs proportionally more to connect), and/or
per-record fsync jitter in the streaming path (`Engine::put()`, unlike
`put_batch()`, still does one fsync per call). Not a blocker for Phase 3
— aggregate percentiles are healthy and the core recall claim is
proven — but worth profiling before any future latency-SLA work.

The multi-threaded concurrency test
(`concurrent_reads_and_writes_are_actually_thread_safe`) was also run 5x
back-to-back on real hardware with no flakiness.

## Roadmap

| Phase | Focus | Status |
|---|---|---|
| 0 | Setup & baseline — WAL, memtable, crash recovery + pgvector/Milvus baseline numbers | ✅ complete — see `bench/README.md` for full results |
| 1 | Storage engine core — LSM flush to SSTables, hybrid row/columnar layout | ✅ complete — 22 tests passing, 100K-record scale test passing |
| 2 | Static vector index — HNSW on a fixed corpus, recall/latency vs. baselines | ✅ complete — recall@10 0.983 (competitive: pgvector 0.984, Milvus 0.988). Insert throughput 11,355 vec/sec — **beats both baselines** (pgvector 1,633, Milvus 2,545) after fixing a per-write fsync bottleneck via batched WAL writes. See `bench/README.md`. |
| 3 | **Incremental/streaming HNSW** — inserts into the graph without full rebuild, concurrent reads during writes. *The core wedge.* | ✅ complete and confirmed on real hardware — recall 0.785→0.983 across incremental growth (matches Phase 2's full-batch number), concurrency test passed 5x on real hardware. See README section above. |
| 4 | Query fusion — push structured filters into HNSW traversal instead of overfetch-then-filter. **Target: match pgvector's ~2.8ms unfiltered p50 while keeping Milvus's ~1.1x (near-zero) filter tax instead of pgvector's ~2.6x.** | ⬜ |
| 5 | Interface & hardening — gRPC/HTTP API, load testing, benchmark report | ⬜ |

## Deliberately out of scope for now

Distribution/sharding, full SQL, multi-key transactions, replication.
Single-node, strongly-consistent, correctness-first — distribution gets
bolted on later once the engine underneath is trustworthy.

## Non-goals

This is not trying to beat every database on every axis. The bet is
narrow and specific: one engine for AI-native workloads that avoids the
sync-lag and overfetch problems of bolting a vector index onto a
general-purpose store.
