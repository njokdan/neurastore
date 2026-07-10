# NeuraStore

A unified storage/query engine for AI-native workloads — hybrid vector +
structured-filter queries on data that's still being written, in one
engine, without a separate ETL/reindex pipeline.

## Headline claim (what we're proving)

Sub-millisecond hybrid vector + structured-filter queries on live-writing
data, benchmarked head-to-head against pgvector and Milvus.

## Status: Phase 0 — Setup & Baseline

This phase proves the durability foundation everything else is built on:

- **WAL** (`src/wal.rs`) — length-prefixed, CRC32-checksummed frames.
  Every write is fsync'd before it's acknowledged. Replay stops cleanly
  at a torn tail frame (the expected shape of a mid-write crash) rather
  than erroring the whole log.
- **MemTable** (`src/memtable.rs`) — in-memory sorted (`BTreeMap`) write
  buffer. Last-writer-wins by sequence number, so a replay can never
  regress a newer write with a stale one. Deletes are tombstone writes,
  LSM-style.
- **Engine** (`src/engine.rs`) — ties the two together: `put`/`delete`
  write WAL-first, then memtable; `open` replays the WAL to reconstruct
  state after a crash or restart.

Run `cargo test` — 10 tests cover the write path, delete/tombstone
semantics, out-of-order write handling, and two crash scenarios
(corrupted frame, torn tail write).

Run the demo twice against the same path to see recovery firsthand:

```bash
cargo run -- ./data/neurastore.wal   # inserts seed records
cargo run -- ./data/neurastore.wal   # "restart" — recovers them from the WAL
```

## Roadmap

| Phase | Focus | Status |
|---|---|---|
| 0 | Setup & baseline — WAL, memtable, crash recovery + pgvector/Milvus baseline numbers | ✅ complete — see `bench/README.md` for full results |
| 1 | Storage engine core — LSM flush to SSTables, hybrid row/columnar layout | ⬜ |
| 2 | Static vector index — HNSW on a fixed corpus, recall/latency vs. baselines | ⬜ |
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
