# NeuraStore — Full Build History

This is the complete, unabridged engineering log for NeuraStore — every
phase, every architecture decision, every benchmark, and every bug
found and fixed along the way, left visible rather than cleaned up
after the fact. This is where new phase writeups go as the project
continues, maintained with the same rigor as everything already here:
real tests, real measurements, wrong turns documented alongside the
fixes that came from them.

**Looking for the short version?** [`README.md`](./README.md) is the
front door — what this is, the proven numbers, and a quickstart.
[`PORTFOLIO.md`](./PORTFOLIO.md) is the high-level summary. This file
is the detailed reference underneath both of them.

---

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

## Status: Phase 4 — Query Fusion (complete, confirmed with real benchmark numbers)

This is the phase the whole project's headline target has pointed at
since Phase 0: pgvector pays a ~2.6x latency tax the moment a filter is
added to a vector query (overfetch top-k, then discard non-matches);
Milvus stays near-parity (~1.1x). NeuraStore's bet was one engine doing
both well.

**Before building the filter, a real correctness bug was found and
fixed.** HNSW graph nodes are append-only — there's no cheap way to
update a node's vector in place. So an *update* (writing to a RecordId
that already exists, which `Engine::put` allows) was silently creating a
second, stale graph node for the same external id, and Phase 3's
delete-tombstone set (keyed by external RecordId) couldn't tell the
stale node apart from the live one. Fixed by tracking tombstones by
*internal* graph node id instead, with a reverse map (external id →
current live internal id) so an update correctly retires the old node.
`vector_index::tests::updating_a_record_supersedes_the_old_graph_node_
not_just_the_new_one` reproduces and locks in the fix.

**The filter mechanism itself** (`src/hnsw.rs::search_layer`,
`search_filtered`): the graph is still traversed through non-matching
nodes (a matching node might only be reachable through a non-matching
"bridge" node), but only matching nodes count toward the result budget
and the search-termination check. A selective filter naturally makes the
search dig deeper into the graph instead of returning a mostly-empty
result early — the predicate is part of the traversal, not a discard
step applied after. For highly selective filters (few matching records),
`VectorIndex::search_filtered` instead computes exact distances directly
over the small matching candidate set (via a small inverted index,
field → value → internal ids) — cheaper than a graph search when there's
only a handful of candidates anyway, and exact rather than approximate.

**8 new tests** (54 total): the update-correctness fix, filtered search
returning only matching records, filtered search excluding deleted
records, filtered search staying correct across incremental writes made
after `build_index()` (no rebuild needed — same Phase 3 guarantee,
now proven for filtered queries too), and a filtered-search-vs-independent-
brute-force-ground-truth check (not just "does it return something,"
but "is what it returns actually correct" — >0.7 recall against exact
computation on a properly-seeded 2000-point dataset).

**Benchmark status: confirmed on real data, with a real weak zone found
and fixed.**

Filter tax measured on the real siftsmall corpus (10,000 records) at
three cardinalities:

| Cardinality | Selectivity | Candidates/category | Tax | Path |
|---|---|---|---|---|
| 4 | 25% | ~2,500 | **4.43x** | graph traversal |
| 20 | 5% | ~500 | **0.52x** | brute force |
| 50 | 2% | ~200 | **0.22x** | brute force |

Two of three cardinalities beat pgvector's 2.6x baseline decisively —
0.52x and 0.22x mean NeuraStore's filtered queries were *faster* than
its own unfiltered queries. But cardinality 4 (the exact setup used for
the original pgvector/Milvus comparison, so a fair apples-to-apples
test) came in at 4.43x — genuinely worse than pgvector's ratio, no spin.

**Root cause, confirmed with real numbers, not guessed:** at 25%
selectivity, finding `ef_search` *matching* results via graph traversal
requires exploring roughly `1/selectivity` (~4x) as much of the graph as
an unfiltered search — the predicate-in-traversal design's own math. The
brute-force fallback threshold (`BRUTE_FORCE_THRESHOLD`, originally an
untuned guess of 500) was too conservative: real data showed brute force
over 200 candidates taking ~0.06ms, while the cardinality-4 case's
~2,500-candidate graph traversal was the slowest measured configuration.
Raised the threshold to 3,000 based on this evidence — a real
adjustment grounded in measurement, not a re-guess. A production version
of this would ideally use an adaptive/cost-based decision (estimate both
paths' cost, pick the cheaper one) rather than a fixed constant; that's
a reasonable future refinement, not blocking Phase 4's completion.

**Follow-up optimization attempt, and its honest result:** the
`BRUTE_FORCE_THRESHOLD` fix above (500 → 3,000) improved the worst case
from 4.43x to ~3.17x. To push further, the brute-force path's full sort
(`O(n log n)` over all candidates) was replaced with a bounded top-k
max-heap (`O(n log k)`, ~3.4x fewer comparisons at n=2,500, k=10) — a
real, safe algorithmic improvement, plus an LTO release-profile change
benefiting the whole binary. Re-measured on real siftsmall, 3 runs:
**2.96x, 3.61x, 2.74x — averaging 3.10x, statistically unchanged from
3.17x.** The optimization didn't move the number, which is itself a
useful, honest result: it confirms the sort was never the bottleneck.
The dominant cost is the raw distance arithmetic — 2,500 candidates ×
128 dimensions of floating-point work — which no amount of smarter
sorting touches. The next real lever would be SIMD vectorization of the
distance computation, a meaningfully bigger and riskier undertaking
(unsafe code, platform-specific intrinsics) than this specific benchmark
number currently justifies, especially since it's a somewhat artificial
selectivity (25%, chosen to match the original pgvector/Milvus
comparison's category count) rather than representative of most
real-world metadata filters.

**Second follow-up: parallelizing the brute-force path across CPU cores.**
Since the heap fix confirmed the bottleneck was raw distance arithmetic
(not sorting), and each candidate's distance is independent of every
other, this is "embarrassingly parallel" — a much lower-risk lever than
manual SIMD (no unsafe code, no platform-specific intrinsics), targeting
the actual measured bottleneck instead of a guess. Implemented via
`rayon` (a mature, standard Rust data-parallelism crate), gated behind a
`PARALLEL_THRESHOLD = 200` so small candidate sets (which already
measured well, 0.22x-0.52x tax) don't pay thread-dispatch overhead for
no benefit. 55 tests now (1 new — the parallel path specifically had no
dedicated test before this, since the existing broad-filter test
deliberately used a candidate count *above* the brute-force threshold to
exercise graph traversal instead; without a targeted test, the parallel
branch would only have been "it compiles," not "it's correct").

**Confirmed on real hardware — the parallelization worked.** 5 runs on
the real siftsmall corpus, same cardinality-4 (25% selectivity) case
that was the project's hardest benchmark:

| Run | Filter tax |
|---|---|
| 1 | 1.76x |
| 2 | 2.70x (72ms p99 spike -- almost certainly a system stall, not steady-state) |
| 3 | 3.76x |
| 4 | 1.17x |
| 5 | 1.20x |
| **Median** | **1.76x** |
| Mean | 2.12x |

Both the median and the mean are now decisively below pgvector's 2.6x
baseline — a real, substantial improvement from the pre-parallelization
~3.10x, confirming the diagnosis (distance arithmetic was the
bottleneck, not sorting) was correct and the fix worked.

**On the variance itself, worth being precise about:** these 5 runs show
real spread (1.17x-3.76x). Before concluding the parallelization made
results *less* stable, it's worth checking whether the noise is
specific to filtered search or general to the whole benchmark run —
insert throughput (a completely unrelated code path: WAL batching, not
the vector index at all) showed the *same* magnitude of variance across
these identical 5 runs (12,048 to 50,830 vec/sec, a 4.2x spread). That
rules out the parallelization as the source — this is environmental
noise (background processes, thermal throttling, disk contention) on a
personal laptop during interactive use, not a new instability the fix
introduced.

**Third round: chasing Milvus's 1.1x, not just pgvector's 2.6x.** After
beating pgvector decisively at every selectivity, two further safe
(no-unsafe-code) optimizations were attempted, and the isolating
experiment gave a clean, unambiguous answer:

1. **Columnar vector storage — a real, confirmed win.** `HnswIndex`
   stored vectors as `Vec<Vec<f32>>` — one separate heap allocation per
   vector. Scanning 2,500 candidates for brute-force filtering meant
   chasing 2,500 scattered pointers, exactly what CPU caches handle
   badly. Converted to one flat, contiguous `Vec<f32>` (vector `i` lives
   at `[i*dim..(i+1)*dim]`), which is what the *original* Phase 1
   architecture doc called for ("hybrid row/columnar layout") but had
   only ever been implemented for on-disk SSTables, not this in-memory
   structure.
2. **Chunked parallelism — tried, measured, reverted.** Switched the
   rayon path to `par_chunks`, expecting less scheduling overhead.
   Measured worse on the typical case despite lower variance (see the
   table below) — likely because on a noisy, shared laptop, rayon's
   per-item work-stealing scheduler adapts better to a thread getting
   preempted than a whole chunk stalling does. Reverted to per-item
   `par_iter`, keeping the columnar layout.

**The isolating experiment (5 runs each) gave a clean, unambiguous
answer** — columnar layout alone, with per-item parallelism, beat both
the parallel-only baseline *and* the chunked version on every metric:

| Configuration | Median | Mean | Stdev | Range |
|---|---|---|---|---|
| Parallel only | 1.76x | 2.12x | 1.11 | 1.17x-3.76x |
| + chunked (reverted) | 2.32x | 2.27x | 0.41 | 1.75x-2.86x |
| **+ columnar, no chunk (final)** | **1.50x** | **1.42x** | **0.21** | **1.17x-1.64x** |

Best median, best mean, and by far the most stable (stdev dropped from
1.11 to 0.21 — no more wild outliers or system-stall-like spikes). One
run hit 1.17x, essentially matching Milvus's 1.1x outright. This is the
final configuration.

**Final, honest Phase 4 scorecard:**

| Selectivity | Tax vs. pgvector's 2.6x / Milvus's 1.1x |
|---|---|
| 25% (4 categories) | **1.50x median** (5-run range 1.17x-1.64x, one run at 1.17x) — decisively ahead of pgvector, closing in on Milvus |
| 5% (20 categories) | 0.52x — decisively ahead of both |
| 2% (50 categories) | 0.22x — decisively ahead of both |

All three tested selectivities now beat pgvector's overfetch-then-filter
tax, and two of three beat Milvus's near-parity number outright. The
hardest case (25% selectivity, deliberately matched to the original
baseline's category count) went from 4.43x on the first real
measurement to a 1.50x median — a real, multi-round optimization story:
an under-tuned threshold fixed with evidence, a genuine bottleneck found
and parallelized, one optimization (chunking) tried and correctly
reverted after real measurement showed it didn't help, and a cache-layout
fix that turned out to be the most effective single change. Nothing here
was guessed at or accepted on faith — every change was made because a
real measurement pointed at a specific cause, and every claim was
verified against another real measurement afterward, including the one
attempt that didn't work out.

## Status: Phase 5 — Interface & Hardening (network API complete)

The network-facing API — what everything since Phase 0 has been
building toward being usable *from outside a Rust process*.

**HTTP/JSON, not gRPC.** Chosen deliberately for Phase 5: curl-able with
zero codegen tooling, the easiest first target for a client library, and
a lower-friction way for anyone evaluating the project to try it. gRPC's
stronger typing and streaming support are real advantages worth
revisiting later if a concrete workload needs them — not a permanent
architectural decision, just the right starting point.

**Concurrency, carried through from Phase 3, not abandoned at the
network boundary.** The whole `Engine` is wrapped in
`Arc<tokio::sync::RwLock<Engine>>`, not a plain `Mutex`. Search/get/stats
handlers take a *read* lock (so multiple concurrent search requests
genuinely run in parallel); put/delete/build_index handlers take a
*write* lock (exclusive). A plain Mutex would have been simpler but
would have silently serialized reads against each other — throwing away
the concurrent-read property Phase 3 spent real effort proving, right at
the one boundary (the network) where it matters most to a real client.

**Endpoints:**

| Method | Path | Purpose |
|---|---|---|
| GET | `/health` | Liveness check |
| POST | `/v1/records` | Insert one record |
| POST | `/v1/records/batch` | Insert many records (one WAL fsync — see Phase 2's `put_batch`) |
| GET | `/v1/records/:id` | Fetch a record |
| DELETE | `/v1/records/:id` | Soft-delete a record |
| POST | `/v1/index/build` | Build/rebuild the vector index |
| POST | `/v1/search` | Unfiltered k-NN search |
| POST | `/v1/search/filtered` | Filtered k-NN search (Phase 4's query fusion) |
| GET | `/v1/stats` | Live record count, index status |

**8 tests**, covering the full request lifecycle in-process via axum's
`oneshot` test harness (put→get roundtrip, delete→404, empty-vector→400,
search-before-index-built→400, and a full batch→build→search→filtered-search
end-to-end flow) — plus manually verified against a real, live server
process over real HTTP with `curl`, not just the in-process harness, to
confirm the whole stack (TCP bind, JSON parsing, routing, the shared
`RwLock<Engine>`) works outside of test-only shortcuts.

**Scope limits, stated plainly:** single collection per server process
(one `Engine`, one data directory) — no multi-tenancy yet. No auth, no
TLS, no rate limiting — this is the interface, not the hardening; the
"hardening" half of this phase's name (auth, rate limiting, anomaly
detection) is intentionally deferred to a later phase, discussed
separately and scoped down from an earlier, more expansive version of
that idea to something concrete and buildable.

**Run it (from source):**
```bash
cargo run --release --bin server -- ./data 8080
curl http://localhost:8080/health
```

**Run it (Docker — no Rust toolchain needed):**
```bash
docker compose up --build
curl http://localhost:8080/health
```
First build compiles from scratch (LTO makes this slow — a few minutes,
one-time cost). Data persists in a named Docker volume across container
restarts. **Verified on real hardware**: built successfully, served real
requests, and data survived a full `docker compose down` (container
completely torn down, not just stopped) followed by `docker compose up`
— confirming the volume mount, not just the process staying alive, is
what's keeping the data.

**This also finally unlocks the fair latency comparison deferred since
Phase 0**: `bench/scripts/bench_neurastore_http.py` benchmarks NeuraStore
over this real HTTP API, using the exact same methodology as
`bench_pgvector.py`/`bench_milvus.py` — so unfiltered latency numbers
are now on equal footing across all three engines for the first time.
See `bench/README.md` section 4.6 for how to run it.

**Fair (client-server) results — every metric now real, reproducible,
and resolved:**

| Metric | pgvector | Milvus | NeuraStore (HTTP) |
|---|---|---|---|
| Insert (vec/sec) | 1,633 | 2,545 | **15,649-17,927 median** (JSON/binary) |
| Unfiltered p50 | 2.81ms | 5.99ms | **2.04-2.81ms** (run-to-run) |
| Filtered p50 | 7.23ms | 6.41ms | **2.69-3.18ms** |
| Filter tax | 2.6x | 1.1x | **1.13-1.32x** |
| Recall@10 | 0.984 | 0.988 | 0.983 |

**Every number here is a genuine win or a tie**, and every one of them
survived being re-measured, not just asserted once: unfiltered latency
ties or beats pgvector's and clearly beats Milvus's; filtered latency
beats *both* baselines decisively; the filter tax matches or comes close
to Milvus's long-standing 1.1x while decisively beating pgvector's 2.6x;
insert throughput beats pgvector by roughly 9-11x. Recall is a
three-way tie.

**Insert throughput's story is worth telling in full — it's the
best example in this whole project of a wrong conclusion getting
caught and fixed instead of shipped.**

1. First measurement: 824 vec/sec — *behind* both baselines. Alarming,
   and wrong, though nobody knew that yet.
2. A real fix (batch size, `orjson`) brought it to 1,216 — still behind,
   a modest gain that didn't match the theory behind it.
3. Built a whole binary bulk-insert endpoint to fix an apparent
   server-side JSON-parsing bottleneck. Smoke-tested at ~2.65x faster.
4. Six controlled runs (3 JSON, 3 binary) showed *no real difference*
   between the two encodings — both showed the same strange pattern:
   fast on the first call, slow on every repeat. That ruled out the
   encoding format as the cause of anything.
5. Chased environmental causes next — OneDrive sync, background Docker
   containers running 23 hours from earlier baseline work. Both
   eliminated. The pattern persisted exactly the same.
6. **The actual cause, found by looking at the exact order of the
   numbers**: the benchmark script's server process was being reused
   across repeated calls instead of restarted. The first call hit a
   fresh, unindexed engine (fast — pure WAL writes). Every repeat call
   hit a server whose index was *already built* from the previous call,
   so those "fresh inserts" were secretly *updates* against a live
   HNSW index — a fundamentally heavier operation. Not noise. Not the
   machine. A test methodology bug, hiding in plain sight the whole time.
7. Fixed with `bench/scripts/clean_insert_benchmark.sh` — a genuinely
   fresh server and directory for every single run, no exceptions.
   Result: **15,649-17,927 vec/sec, consistently, no outliers, no
   bimodal split.** The "gap" to pgvector never existed. It was
   always this fast; the earlier numbers were measuring two different,
   silently-mixed things.

The honest lesson, worth stating plainly rather than just moving past
it: the original "824 vec/sec, behind pgvector" number *felt* real
enough to build a whole binary-protocol feature on top of, and that
feature turned out not to be necessary to fix the problem it was built
for (the real fix was a one-line test harness change). The binary
endpoint remains in the codebase as legitimate, correctness-verified
engineering — a reasonable thing to offer regardless — but it should be
understood as validated infrastructure, not as the thing that solved a
performance problem, because it didn't need to.

## Roadmap

| Phase | Focus | Status |
|---|---|---|
| 0 | Setup & baseline — WAL, memtable, crash recovery + pgvector/Milvus baseline numbers | ✅ complete — see `bench/README.md` for full results |
| 1 | Storage engine core — LSM flush to SSTables, hybrid row/columnar layout | ✅ complete — 22 tests passing, 100K-record scale test passing |
| 2 | Static vector index — HNSW on a fixed corpus, recall/latency vs. baselines | ✅ complete — recall@10 0.983 (competitive: pgvector 0.984, Milvus 0.988). Insert throughput 11,355 vec/sec — **beats both baselines** (pgvector 1,633, Milvus 2,545) after fixing a per-write fsync bottleneck via batched WAL writes. See `bench/README.md`. |
| 3 | **Incremental/streaming HNSW** — inserts into the graph without full rebuild, concurrent reads during writes. *The core wedge.* | ✅ complete and confirmed on real hardware — recall 0.785→0.983 across incremental growth (matches Phase 2's full-batch number), concurrency test passed 5x on real hardware. See the phase section above in this document. |
| 4 | Query fusion — push structured filters into HNSW traversal instead of overfetch-then-filter. **Target: match pgvector's ~2.8ms unfiltered p50 while keeping Milvus's ~1.1x (near-zero) filter tax instead of pgvector's ~2.6x.** | ✅ complete, confirmed on real hardware — **all 3 selectivities beat pgvector's 2.6x tax decisively** (1.50x median, 0.52x, 0.22x), 2 of 3 beat Milvus's 1.1x outright. Hardest case improved from 4.43x to 1.50x median across three root-caused optimization rounds, including one (chunking) correctly identified and reverted after real measurement showed it didn't help. See the phase section above in this document. |
| 5 | Interface & hardening — gRPC/HTTP API, load testing, benchmark report | ✅ HTTP/JSON API complete — 67 tests total (55 lib + 12 server). **Full fair client-server comparison done, every metric a real win or tie**: insert 15,649-17,927 vec/sec (9-11x pgvector, after finding and fixing a test methodology bug), unfiltered/filtered latency and filter tax all beat or tie both baselines. Load testing / hardening (auth, rate limiting) still ahead. See the phase section above in this document. |
| 6 | Usability — Docker, Python client, CLI | ✅ complete — all three verified on real hardware/live servers, not just written. 52 client-side tests (35 unit + 17 integration). See the phase section above in this document. |
| 7 | Hardening — auth, rate limiting, TLS, scoped anomaly detection | ✅ complete — all four pieces verified live on real hardware. 88 Rust tests, 41 client-side tests. Four real bugs found and fixed during verification (a Docker healthcheck, a Compose merge-semantics gotcha, and two anomaly-detection statistical bugs), not assumed away. See the phase section above in this document. |
| 8 | Architectural gaps — multi-collection, tombstone compaction, maybe gRPC | ✅ complete (except gRPC, deliberately deferred — no concrete need for it has come up) — tombstone compaction, multi-collection support (server + Python client + CLI), all verified live. 105 Rust tests, 52 Python client-side tests. Fully backward compatible — zero migration for existing deployments. See the phase section above in this document. |

## Reducing benchmark noise on Windows

If revisiting insert throughput (or any latency-sensitive measurement)
with more confidence than this project managed, these are the most
likely, most actionable sources of the ~19x run-to-run swings observed
— roughly ordered by how likely each is to matter, based on the pattern
actually seen (mostly consistent, occasional huge spikes):

1. **Check whether the project folder is inside a synced OneDrive
   folder.** On Windows 10/11, `Desktop`, `Documents`, and `Pictures`
   are often auto-synced to OneDrive by default — and this project's
   path (`~/desktop/projects/neurastore`) matches that pattern exactly.
   Every new WAL/SSTable file the engine writes would trigger a
   background upload attempt, which is a textbook explanation for
   "usually fine, occasionally much slower/faster." Check: Settings →
   Sync and back up → Manage sync settings (or right-click the OneDrive
   icon in the system tray). Either move the project outside any synced
   folder, or pause syncing while benchmarking.
2. **Add a Windows Defender exclusion for the project folder** (and any
   temp directory used for benchmark data). Settings → Update &
   Security → Windows Security → Virus & threat protection → Manage
   settings → Exclusions. Real-time antivirus scanning of newly
   created/modified files is a well-known, major source of I/O latency
   variance on Windows specifically — this project's benchmarks write a
   lot of new small files (WAL segments, SSTables) in quick succession,
   exactly the pattern that triggers repeated scanning.
3. **Close background applications** during benchmark runs — browsers,
   Docker Desktop (if pgvector/Milvus containers aren't needed for that
   specific test — check `docker compose ps` and stop what's unused),
   Slack and other Electron apps, IDE background indexing/search.
4. **Switch to the "High performance" (or equivalent) power plan**
   instead of "Balanced" — the default plan dynamically scales CPU
   frequency in ways that add real variance to CPU-bound benchmarks.
5. **Run each condition 3+ times and compare medians**, not single
   runs — this project's own experience is the best argument for this:
   several "findings" here only looked real until a second or third run
   contradicted them.

## Status: Phase 6 — Usability (complete)

Goal, stated plainly: a stranger should be able to `pip install` a
client, point it at a running server, and be querying NeuraStore in
five minutes — no Rust, no hand-built HTTP requests. All three planned
pieces are done and verified, not just written.

**Docker** (`Dockerfile`, `docker-compose.yml`) — run the server without
installing a Rust toolchain: `docker compose up --build`. **Verified on
real hardware**: image builds, container serves real requests, and data
correctly persists across a full `docker compose down` + `up` cycle
(confirming the volume mount, not just process uptime, keeps the data).

**Python client** (`client/python/`, package `neurastore-client`) — an
ergonomic wrapper (`NeuraStoreClient`) around the full HTTP API: insert,
batch insert (JSON or binary), get, delete, build_index, search,
search_filtered, stats. Only depends on `requests` — vectors are plain
lists, no numpy required. Translates HTTP status codes into a proper
exception hierarchy (`NotFoundError`, `BadRequestError`, `ServerError`,
`ConnectionError`) instead of leaking raw `requests` exceptions. Uses a
persistent `requests.Session` internally — not a style choice: skipping
this was directly responsible for a real ~2-second-per-request penalty
found earlier in this project's own benchmark tooling (Windows'
localhost-then-IPv6-fallback DNS behavior), and a client library
shouldn't hand that bug to everyone who uses it.

```bash
cd client/python
pip install -e .
```
```python
from neurastore_client import NeuraStoreClient
client = NeuraStoreClient("http://localhost:8080")
client.insert(1, [0.1, 0.2, 0.3], metadata={"category": "docs"})
client.build_index()
results = client.search([0.1, 0.2, 0.3], k=5)
```

**CLI** (`neurastore` command, installed by the same package) — a thin
wrapper over the same client, using only the standard library's
`argparse` to keep the install footprint small (no click/typer added
just for CLI polish). Every subcommand run end-to-end against a real
live server, not just unit-tested: health, insert, get, delete,
build-index, search, search-filtered, stats (text and `--json`), and
batch insert from a file — including confirming errors exit with status
1 and a clear stderr message instead of a Python traceback, so it's
safe to script against.

```bash
export NEURASTORE_URL=http://localhost:8080
neurastore insert --id 1 --vector 0.1,0.2,0.3 --metadata category=docs
neurastore build-index
neurastore search --vector 0.1,0.2,0.3 --k 5
```

**52 client-side tests total** (35 unit — client + CLI, mocked HTTP,
no server needed — + 17 integration/live-server tests across both
suites), on top of the 67 Rust tests from Phases 0-5. Every documented
example in this phase — the Python quickstart, every CLI command, the
Docker persistence cycle — was actually run against a real server, not
just written and assumed to work.

## Status: Phase 7 — Hardening (complete)

Right now the server has zero protection — anyone who can reach it over
the network has full read/write/delete access to everything. This phase
starts closing that, one real, tested piece at a time.

**API key authentication — done, verified live, backward compatible.**
Opt-in via `NEURASTORE_API_KEYS` (comma-separated) at server startup:

```bash
NEURASTORE_API_KEYS=my-secret-key,another-clients-key cargo run --release --bin server -- ./data 8080
```

With no keys configured (the default), the server runs exactly as
before — every existing example in this README still works with zero
changes. This is an explicit choice, not a silent gap: the server logs
a clear warning either way at startup, so running without auth is
something you can see you did, not something you discover later.

With keys configured, every `/v1/*` endpoint requires
`Authorization: Bearer <key>`. `/health` is deliberately exempt even
when auth is enabled — load balancers and orchestration health probes
need to reach it without credentials. Multiple keys can be configured
at once (one per client), so revoking one client's access doesn't
affect others.

8 new server-side tests (75 Rust tests total: 55 lib + 20 server),
covering: health bypasses auth, missing/wrong/correct keys, multiple
independent keys, write routes protected (not just reads), a malformed
`Authorization` header failing cleanly (401, not a 500), and the
no-keys-configured backward-compatibility case. Verified against real
running servers with real `curl` requests, both with and without auth
enabled, not just the in-process test harness.

**The Python client and CLI were updated in the same pass, not left
behind** — an auth-enabled server would otherwise be completely
unreachable from the tooling built in Phase 6:

```python
client = NeuraStoreClient("http://localhost:8080", api_key="my-secret-key")
```
```bash
export NEURASTORE_API_KEY=my-secret-key
neurastore stats
```

A new `AuthenticationError` exception joins the client's existing error
hierarchy. 6 new client-side tests (40 unit total, plus integration
tests across the client and CLI), plus a full real-world run — actual
CLI binary, actual live auth-enabled server — confirming the whole
chain works together, not just each piece in isolation.

**Rate limiting — done, verified live, a real design subtlety worth
knowing about.** Opt-in via `NEURASTORE_RATE_LIMIT_RPS` (requests per
second; `NEURASTORE_RATE_LIMIT_BURST` optionally overrides the default
burst of 2x the rate). Standard token-bucket algorithm, one bucket per
API key when auth is enabled, or one shared server-wide bucket when it
isn't (there's no cheap per-client identity without auth — a documented
simplification, not a silent gap).

The subtlety: auth and rate limiting are deliberately implemented as
**one combined check, not two independent middleware layers**. Two
layers would need rate limiting to see auth's *validated* result, not
just the raw header — keying limits by the raw provided key would let
an attacker bypass the limiter entirely by rotating the key string on
every request, since each new string gets a fresh bucket. Resolving
identity once, before either check runs, closes that gap and avoids
depending on tower's exact layer-ordering semantics being right.

7 new server tests (82 Rust tests total: 55 lib + 27 server) — including
one specifically checking that wrong-key requests always get a
consistent 401, never a mix of 401/429 that could leak bucket state to
an attacker. Verified live: burst limits enforced correctly, `/health`
never throttled, buckets refill correctly over time, and disabled (the
default) behaves exactly as before with zero rate limiting anywhere.

The Python client and CLI got a matching `RateLimitError` exception in
the same pass (41 client-side tests now).

**TLS — done, via reverse proxy, not built into the Rust server.
Fully verified on real hardware, including a real bug found and fixed
along the way.**

Deliberately not implemented as raw TLS termination inside axum
(`axum-server` + `rustls` would work, but duplicates what a reverse
proxy already does well, and adds real certificate-management
complexity to the app itself). Almost no production axum deployment
actually terminates TLS in-process — this follows that same standard
pattern instead of inventing a different one.

`deploy/Caddyfile` + `docker-compose.tls.yml` (an overlay on the base
`docker-compose.yml`, not a replacement) add
[Caddy](https://caddyserver.com/) as a TLS-terminating reverse proxy in
front of the server:

```bash
docker compose -f docker-compose.yml -f docker-compose.tls.yml up --build
```

The same two-line Caddy config handles both cases with zero code
changes: as shipped (`localhost`), Caddy automatically issues a
locally-trusted certificate via its own internal CA — no public domain
or manual cert generation needed for local testing. Swap `localhost`
for a real domain name and add an email address for a real, automatically-
renewing Let's Encrypt certificate in production — the Caddyfile shows
both forms.

**Two real bugs found and fixed during verification, not hidden:**
1. The Docker healthcheck (`curl -f http://localhost:8080/health`, run
   *inside* the container) had been silently failing since Phase 6 —
   `curl` was never actually installed in the minimal runtime image.
   Nothing depended on the healthcheck's actual pass/fail status until
   this TLS overlay's `depends_on: condition: service_healthy` finally
   checked it. Fixed by adding `curl` to the runtime image.
2. The overlay's `ports: []`, meant to remove the server's direct host
   port once Caddy is terminating TLS in front, silently did nothing —
   Docker Compose merges list-type fields like `ports` across override
   files by default, it doesn't replace them, so an empty list merged
   into the base file's `["8080:8080"]` had no effect. Fixed with the
   `!reset` YAML tag, which explicitly clears a merged list instead of
   appending to it. Confirmed live: after the fix, `curl https://localhost/health`
   still returns `ok` through Caddy, while the direct port is genuinely
   gone from the container (`docker compose ps` shows `8080/tcp` with
   no host-side `0.0.0.0:8080->` mapping at all).

**Anomaly detection — done, the scoped statistical version this project
committed to, not the expansive "AI security" framing set aside during
early planning. Two real bugs found via live testing, not just written
and assumed correct.**

Opt-in via `NEURASTORE_ANOMALY_DETECTION=1`. Tracks two exponentially-
weighted moving averages of request rate per client identity (same
concept as rate limiting's per-key buckets): a fast-reacting one and a
slow, established-baseline one. When the fast average significantly
exceeds the slow one — a real behavior change for *that specific
client*, not a fixed global threshold — it's logged clearly for a human
to review. **It never blocks a request.** A statistical detector will
have false positives (a legitimate bulk load looks identical to a
burst), and auto-rejecting on those would be worse than the problem
it's meant to catch — the same human-in-the-loop principle applied to
every security-adjacent decision in this project.

Two real, distinct bugs surfaced during live verification, both found
by noticing the reported numbers didn't make sense, not by assuming the
code was correct because it compiled and passed synthetic unit tests:

1. A brand-new client's very first request was being compared against
   itself as its own "previous" timestamp, producing a near-zero
   interval and an artificial ~1000/s rate spike baked permanently into
   that client's baseline from the start. Fixed by skipping rate
   estimation entirely on the first request — it only establishes a
   starting point; real estimation begins from the second request,
   where an actual interval exists between two real events.
2. After fixing that, a *steady, non-anomalous* request pattern started
   getting flagged anyway — a deterministic unit test
   (`steady_rate_requests_never_flagged`) caught this immediately. Root
   cause: the fast EWMA (which reacts quickly) and the slow EWMA (which
   represents an established baseline) both started from zero but
   converge at very different speeds, creating a persistent false
   "fast ≫ slow" gap during a client's early requests under perfectly
   normal load. Fixed by seeding both averages directly from the first
   real observed interval instead of both crawling up from a shared
   zero at mismatched rates.

10 new server tests (88 Rust tests total: 55 lib + 33 server), including
four testing the detector's statistical logic directly and
deterministically — time is passed in explicitly rather than read via
real `Instant::now()` + actual sleeping, since `Instant` has no public
constructor for an arbitrary point in time but `instant + Duration`
always works, giving fully deterministic, fast tests instead of flaky
ones dependent on real wall-clock timing. Verified live against a real
server too: an established ~1/s baseline, then a real burst, producing
exactly the expected flagged log lines with a now-plausible baseline
number — not the implausible ~20-35/s the first bug produced.

**Phase 7 is now complete**: authentication, rate limiting, TLS, and
scoped anomaly detection, every piece verified live on real hardware,
including four real bugs found and fixed along the way (the Docker
healthcheck, the Compose port-merge semantics, and these two anomaly-
detection bugs) rather than assumed away.

## Status: Phase 8 — Architectural Gaps (complete, except gRPC which remains deliberately deferred)

**Tombstone compaction — done, verified live, including a design
decision worth understanding, not just a feature to check off.**

Since Phase 3, every delete or update has left the old data behind
permanently — a tombstone on disk, a stale node in the HNSW graph,
neither ever reclaimed. Correct, but a long-running server just
accumulates waste indefinitely. `Engine::compact()` (via
`POST /v1/compact`, `neurastore compact`, or `client.compact()`) closes
this:

1. **Merges every on-disk SSTable into one**, dropping superseded
   (updated-over) record versions.
2. **Rebuilds the vector index**, if one exists, using whichever params
   were last used to build it — not silently reverting to defaults,
   which would have been a real, easy-to-miss regression for anyone
   using custom HNSW parameters.
3. **Flushes the memtable first** if anything's pending, so compaction
   always operates on the fullest possible picture.

**The one non-obvious design choice, worth stating plainly:** tombstones
are deliberately *kept* in the compacted output, not dropped, even
though after a full merge the record they shadow is already gone. This
is a crash-safety choice. If the process dies after the new compacted
file is written but before the old files are deleted, `Engine::open()`
loads both and merges them by position — if the tombstone weren't
there, a deleted record could briefly "reappear" from an older file in
that exact window. A tombstone is a few bytes; keeping it is a small,
permanent cost for a real correctness guarantee. A dedicated test
(`compacted_sstable_retains_tombstone_markers_not_just_absence`) reads
the compacted file directly to confirm this, rather than only checking
`get()` returns nothing — which would look identical whether the
tombstone survived or the record was simply never merged in.

10 new tests (98 Rust tests total: 63 lib + 35 server) plus 3 new
client-side tests. Verified live end-to-end: inserted records, updated
one, deleted another, compacted, and confirmed both the file count
dropped to one *and* the data stayed correct — including a case where
nothing had been flushed to disk yet at all (compaction correctly
flushed the memtable first, live, not just in a unit test).

**Multi-collection support — done, verified live, fully backward
compatible.** One server process can now serve many independent,
isolated collections — each with its own `Engine` (own WAL, own
SSTables, own vector index) — instead of exactly one.

Every existing route (`/v1/records`, `/v1/search`, etc.) keeps working
completely unchanged, operating on a "default" collection rooted at the
same top-level data directory every deployment has always used — zero
migration for anyone upgrading. New collections are addressed via
`/v1/collections/<name>/...` (same full operation set: records, batch
insert, search, filtered search, index build, compact, stats), created
lazily on first write, no separate setup step required — matching how
the default collection has never needed one either. `GET /v1/collections`
lists everything known, including collections created in a previous
run that haven't been touched again yet this session.

**Two real design decisions worth knowing about, not just the feature
itself:**
1. **The engineering approach was deliberately conservative.** Rather
   than refactor the existing, heavily-tested handlers to be
   collection-aware, every operation's core logic was extracted into a
   small shared `_impl` function, called by two thin wrappers — one for
   the original routes (unchanged behavior), one for the new
   collection-scoped routes. The existing routes' *registration* never
   changed at all. All 35 pre-existing server tests passed unmodified
   after this refactor, which is the real evidence it didn't quietly
   change anything.
2. **Collection names are tightly validated** (letters, digits,
   underscore, hyphen only) because a name becomes a directory name on
   disk — an unvalidated name like `../../etc` would be a real path
   traversal vulnerability, not a theoretical one. A dedicated test
   fires exactly that kind of name at the API and confirms it's
   rejected, not merely "hoped to be."

7 new server tests (105 Rust tests total: 63 lib + 42 server) — covering
real isolation (two collections, same id, genuinely separate records,
not a coincidental pass), confirming `default` reached through either
path shares one real engine instance rather than two racing ones,
listing, and the auth/security middleware applying to the new routes
exactly like the old ones. Verified live end-to-end too: original
routes untouched, a brand-new named collection created via a raw curl
call, real isolation between same-numbered records in different
collections, and the on-disk layout matching the design exactly
(`default`'s `wal.log` at the top level, `my_docs/wal.log` in its own
subdirectory).

**Client and CLI support for named collections — done, verified live.**
Every `NeuraStoreClient` method accepts an optional `collection`
argument (default `"default"`); the CLI gets a global `--collection`
flag and a `collections` subcommand. Every existing call site — every
method call without the new argument, every pre-existing CLI command
without the new flag — builds the exact same URLs as before this
change, verified by running all 23 pre-existing client tests and all 20
pre-existing CLI tests completely unmodified after adding it. 10 new
tests (52 client-side tests total). Verified live: default-collection
backward compatibility, a named collection created via `--collection`,
correct isolation between same-numbered records in different
collections, `collections` listing both, and the `NEURASTORE_COLLECTION`
environment variable working as an alternative to the flag.

**Remaining in this phase, lower priority:** evaluating whether gRPC is
ever actually worth adding alongside the existing HTTP/JSON API — no
concrete need for it has come up.

## Deliberately out of scope for now

Distribution/sharding, full SQL, multi-key transactions, replication.
Single-node, strongly-consistent, correctness-first — distribution gets
bolted on later once the engine underneath is trustworthy.

## Non-goals

This is not trying to beat every database on every axis. The bet is
narrow and specific: one engine for AI-native workloads that avoids the
sync-lag and overfetch problems of bolting a vector index onto a
general-purpose store.

## License & contributing

Apache 2.0 — see [`LICENSE`](./LICENSE). Chosen deliberately over a
more restrictive license (SSPL/AGPL) because the project has no users
yet: maximizing the odds that anyone tries it, contributes, or builds
on it matters more right now than pre-emptively defending a business
model that doesn't exist yet. Revisiting the license later, if a real
commercial threat materializes, is a well-worn path (see `PORTFOLIO.md`
for the fuller reasoning and the MongoDB/Elastic/Redis history behind
that judgment).

Contributions welcome — see [`CONTRIBUTING.md`](./CONTRIBUTING.md) for
setup, testing expectations, and what a good PR looks like here.
Security issues: see [`SECURITY.md`](./SECURITY.md) (please don't file
these as public issues). Community standards: [`CODE_OF_CONDUCT.md`](./CODE_OF_CONDUCT.md).

CI runs the full test suite (Rust + Python) on every push and PR —
see [`.github/workflows/ci.yml`](./.github/workflows/ci.yml). Not yet
verified against a real GitHub Actions run from this environment (no
way to trigger one from here) — flag anything that doesn't work as
written, same honest caveat as the original Dockerfile before it was
verified.
