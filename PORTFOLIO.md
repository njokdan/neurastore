# NeuraStore — Portfolio Summary

A unified vector + structured-filter storage engine, built from a WAL
up through a hardened, multi-collection network service — written in
Rust, benchmarked against pgvector and Milvus, and documented with
every wrong turn left visible alongside every win.

This document is the high-level story. For depth: [`README.md`](./README.md)
has the full phase-by-phase build log, [`COMPARISON.md`](./COMPARISON.md)
has the fair head-to-head numbers, and [`bench/README.md`](./bench/README.md)
has exact reproduction steps for every benchmark cited here.

## What it is, in one paragraph

NeuraStore stores vectors and structured metadata together and answers
hybrid similarity + filter queries without paying the cost most systems
pay for that combination — either overfetching an unfiltered result set
and discarding most of it, or maintaining two separate indexes that can
drift out of sync. It's a single-node engine: its own write-ahead log,
its own LSM-style storage layer, its own from-scratch HNSW vector
index, a real HTTP API, a Python client and CLI, and enough hardening
(auth, rate limiting, TLS, anomaly detection, multi-collection support)
to actually be run somewhere, not just demoed once.

## The numbers, stated plainly

- **105 Rust tests** (63 library, 42 server), **52 Python client-side
  tests** — all passing, all re-verified after every architectural
  change that touched them, not just written once and assumed correct.
- **46 commits**, each one a real, working state — the git history
  itself is a readable record of the build, including the commits that
  say "fix" and mean it.
- **9 build phases**, from raw WAL/memtable durability through
  multi-collection support, each one verified live against a real
  running server before being marked done, not just unit-tested in
  isolation.
- **Four real bugs found during Phase 7 verification alone** — a
  Docker healthcheck that had been silently failing since Phase 6, a
  Compose list-merge semantics misunderstanding, and two separate
  statistical bugs in the anomaly detector, each one caught by a test
  or a live measurement that didn't look right, then root-caused rather
  than patched around.

## The three real, fairly-measured wins — see `COMPARISON.md` for the full numbers

- **Insert throughput**: 15,649–17,927 vec/sec over real HTTP, roughly
  9–11x pgvector's 1,633 — a number that only became trustworthy after
  finding and fixing a test methodology bug (a reused server process
  silently measuring index-update cost instead of a fresh bulk load).
- **Filtered-query latency tax**: 1.13–1.32x, beating pgvector's 2.6x
  decisively and matching or approaching Milvus's long-standing ~1.1x —
  this was the actual design thesis of the whole project, proven with
  real, repeated measurement rather than asserted once.
- **Recall@10**: tied with both pgvector (0.984) and Milvus (0.988) at
  0.983, from an HNSW implementation built from scratch, not wrapped
  from an existing library.

Three wins. Not "beats every vector database at everything" — three
specific, real, reproducible numbers, each with its own measurement
story documented in full.

## The engineering story that matters more than any single number

The most representative moment in this whole project wasn't a benchmark
win — it was the insert-throughput investigation in Phase 5. An early
measurement (824 vec/sec) looked like a real deficit against pgvector.
That measurement drove real engineering: batch-size tuning, a faster
JSON library, and eventually a whole binary wire protocol built to
close what looked like a structural gap. Then a more rigorous test —
six controlled runs instead of one — showed the binary protocol wasn't
actually faster than plain JSON. The real cause, found by looking
carefully at the *order* the numbers came in rather than trusting an
average, was a one-line test harness bug: a server process being reused
across repeated benchmark calls, silently measuring index-update cost
from a prior run instead of a genuine fresh bulk load. The real number,
once the bug was fixed, was 9–11x faster than pgvector — the "gap"
had never existed.

That sequence — wrong number, real response, more rigorous re-test,
root cause found, number corrected, all of it left in the historical
record rather than quietly cleaned up — is the actual substance of this
project. The same pattern repeats in Phase 4 (a chunked-parallelism
optimization tried and correctly reverted after real measurement showed
it didn't help), Phase 7 (two separate statistical bugs in the anomaly
detector, each caught by a test that failed for the right reason), and
the TLS verification pass (a Docker healthcheck silently broken since
Phase 6, only surfaced once something finally depended on it).

## Honest competitive positioning

Full comparison against Pinecone, Milvus, and pgvector — including
where NeuraStore is currently outclassed, not just where it wins — is
worth reading in full rather than summarizing away. Short version:

- **1 vector data type** (dense float32) vs. 3–4 each for the others
  (binary, sparse, half-precision).
- **1 index type** (HNSW) vs. multiple each — no IVF, no quantization,
  no disk-based indexing, no GPU acceleration.
- **2 search modes** (unfiltered and single-field-equality-filtered
  k-NN) vs. 5–8+ each, including hybrid dense+sparse search, full-text
  search, reranking, and — for pgvector specifically — full SQL.
- **Single-node only** — no clustering, sharding, or replication.
  Milvus's distributed architecture handles 10B+ vectors across
  multiple nodes; nothing in NeuraStore's current design can do that.
- **Days of real-world hardening** vs. years of production traffic at
  the companies behind the alternatives.

Reaching feature and scale parity with Pinecone or Milvus isn't a
realistic goal for a project like this — that's genuinely years of
funded team effort, most concretely in distributed architecture and
hybrid search. The honest target is excellence in a deliberately
narrow niche (single-node hybrid vector + filter search), not a
head-to-head replacement for systems built for a different scale
entirely.

## What's deliberately not built, and why

- **gRPC** — a real, understood tradeoff (binary encoding, generated
  clients, native streaming) set aside because nothing concrete has
  ever needed it; HTTP/JSON's accessibility was a deliberate Phase 5
  choice, and duplicating every endpoint in a second protocol is a
  standing maintenance cost with no current justification.
- **Hosting, a managed service, community infrastructure** — these
  aren't engineering problems more code solves; they need capital,
  time, and real adoption that a codebase can't manufacture on its own.
- **Autonomous security/moderation features** — an earlier, more
  expansive "AI-driven security" idea was deliberately scoped down to
  bounded, statistical, advisory-only anomaly detection that flags for
  a human, never blocks on its own — the same human-in-the-loop
  principle applied everywhere security-adjacent decisions came up.

## Where to look for more

| Question | Where |
|---|---|
| How does each phase's architecture actually work? | [`README.md`](./README.md) |
| What are the exact, reproducible benchmark numbers? | [`COMPARISON.md`](./COMPARISON.md), [`bench/README.md`](./bench/README.md) |
| How do I run it? | `README.md`'s Phase 6 section (Docker, `cargo run`) |
| How do I use it from Python? | [`client/python/README.md`](./client/python/README.md) |
| How do I deploy it with TLS? | `README.md`'s Phase 7 TLS section, [`deploy/Caddyfile`](./deploy/Caddyfile) |
