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

## Roadmap

| Phase | Focus | Status |
|---|---|---|
| 0 | Setup & baseline — WAL, memtable, crash recovery + pgvector/Milvus baseline numbers | ✅ complete — see `bench/README.md` for full results |
| 1 | Storage engine core — LSM flush to SSTables, hybrid row/columnar layout | ✅ complete — 22 tests passing, 100K-record scale test passing |
| 2 | Static vector index — HNSW on a fixed corpus, recall/latency vs. baselines | ✅ complete — recall@10 0.983 (competitive: pgvector 0.984, Milvus 0.988). Insert throughput 11,355 vec/sec — **beats both baselines** (pgvector 1,633, Milvus 2,545) after fixing a per-write fsync bottleneck via batched WAL writes. See `bench/README.md`. |
| 3 | **Incremental/streaming HNSW** — inserts into the graph without full rebuild, concurrent reads during writes. *The core wedge.* | ⬜ |
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
