# NeuraStore Benchmark Harness (Phase 0 baselines)

Establishes pgvector and Milvus numbers *before* NeuraStore's own engine
is benchmarked, so "competitive" has a concrete target instead of being
a vibe. Run this on your actual dev machine, not a sandboxed container —
it needs Docker, real disk I/O, and enough RAM for two vector databases
running side by side.

**⚠️ Untested in this environment.** I wrote and reviewed this harness
carefully, but couldn't run it end-to-end here — this container's network
allowlist doesn't include Docker registries or the texmex dataset mirror.
Treat it as a strong first draft: run it, and if something breaks (a
pymilvus API mismatch is the most likely culprit — the API shifts
between Milvus versions), paste me the error and I'll fix it fast.

## 1. Prerequisites

- Docker + Docker Compose
- Python 3.10+ (uses `list[int]` type hints)
- ~2GB free disk for `siftsmall`, ~4GB free RAM recommended for Milvus's etcd+MinIO+standalone trio

```bash
cd bench
python -m venv venv
source venv/bin/activate      # Windows Git Bash: source venv/Scripts/activate
pip install -r requirements.txt
```

## 2. Bring up both databases

```bash
docker compose up -d
docker compose ps             # wait until milvus-standalone shows "healthy" -- can take ~30-60s
```

## 3. Prepare a dataset

Start with the synthetic path — it's instant and proves the harness works
end-to-end before you wait on a download:

```bash
python scripts/prepare_dataset.py --mode synthetic --n 100000 --dim 128
```

Once that runs clean, switch to the real benchmark corpus for numbers
that are comparable to published ANN benchmarks elsewhere:

```bash
python scripts/prepare_dataset.py --mode siftsmall   # 10K vectors, ~5MB, quick
# or, for the full comparison later:
python scripts/prepare_dataset.py --mode sift1m      # 1M vectors, ~500MB, slow
```

## 4. Run the benchmarks

```bash
python scripts/bench_pgvector.py --k 10
python scripts/bench_milvus.py --k 10
```

Each prints:
- Insert throughput (vectors/sec)
- HNSW index build time
- Unfiltered query latency (mean/p50/p95/p99/max) + recall@k
- Filtered (`WHERE category = X`) query latency — this is the
  overfetch-then-filter number NeuraStore's Phase 4 query fusion is
  trying to beat

## 5. What to send back

Paste me the console output from both scripts (or a screenshot). That
becomes the target line in NeuraStore's README roadmap table, and the
number Phase 2/4's own benchmarks get compared against.

## Phase 0 Baseline Results (finalized)

Measured on siftsmall (10,000 base vectors, dim=128, 100 queries — the
real texmex SIFT corpus), 3 runs each, benchmark order randomized per
run to rule out ordering bias, 20-query warm-up before each timed loop.
Numbers below are the 3-run average; see `git log` on this file for the
raw per-run output this was derived from.

| Metric | pgvector (HNSW) | Milvus (HNSW, standalone) |
|---|---|---|
| Insert throughput | ~1,633 vec/sec | ~2,545 vec/sec |
| Unfiltered query p50 | ~2.81ms | ~5.99ms |
| Unfiltered recall@10 | ~0.984 | ~0.988 |
| Filtered (`WHERE category=X`) query p50 | ~7.23ms | ~6.41ms |
| **Filter tax (filtered/unfiltered p50 ratio)** | **~2.6x** | **~1.1x** |

### What this means for NeuraStore's target

pgvector is faster unfiltered, but pays a real, reproducible ~2.6x
latency tax the moment a structured predicate is added — this is the
overfetch-then-filter pattern: it runs the ANN search first, then
filters the results, rather than pushing the predicate into the graph
traversal. Milvus is slower unfiltered at this scale (more architectural
overhead — gRPC round-trip, distributed-systems machinery that doesn't
pay off on only 10K vectors) but keeps filtering nearly free (~1.1x).

**Neither system offers both.** That gap — fast unfiltered search *and*
near-zero filter tax, in the same engine — is NeuraStore's headline
target for Phase 4 (query fusion): match pgvector's unfiltered speed
while keeping Milvus's near-parity filtered performance.

### A dataset lesson learned along the way

An earlier attempt to get a larger query sample by regenerating the
dataset in `--mode synthetic` (uniform random Gaussian vectors) caused
recall to collapse to ~0.68–0.74 for *both* systems, identically — not
a bug in either database, but the curse of dimensionality: random
Gaussian vectors in 128-D have nearly-equidistant pairwise distances,
so HNSW's graph navigation has no real gradient to walk. Lesson for
NeuraStore's own future benchmarking: synthetic uniform random vectors
are fine for testing harness *mechanics*, but not valid for benchmarking
*recall* — always use real embeddings (or at least clustered synthetic
data) for that. siftsmall's 100 real queries, run 3x with randomized
order and warm-up, was judged sufficient and more trustworthy than a
larger but structurally invalid synthetic sample.



- `bench_pgvector.py` uses `vector_l2_ops` (Euclidean distance) — matches
  the metric texmex ground truth is computed with. If you swap datasets,
  double check the metric matches.
- Filtered recall isn't computed (only latency) because `ground_truth.npy`
  is unfiltered nearest-neighbor truth — computing filtered ground truth
  would need a filtered brute-force pass. Worth adding once you're
  benchmarking NeuraStore's own filtered fusion in Phase 4.
- HNSW parameters (`m=16`, `ef_construction=64`, `ef_search=40`) are
  reasonable defaults, not tuned — if numbers look off vs. published
  benchmarks, tuning these is the first thing to check.
