# Contributing to NeuraStore

Thanks for considering it. This project has been built with a specific
discipline worth carrying forward: every claim is backed by a real,
reproducible test or measurement, and every wrong turn stays visible in
the history instead of being quietly cleaned up. See `PORTFOLIO.md` for
what that's looked like in practice.

## Getting set up

**Rust side** (the engine and server):
```bash
git clone https://github.com/njokdan/neurastore
cd neurastore
cargo build --release
cargo test --release
```
This runs the full suite: library tests (storage engine, WAL, HNSW
index, query fusion) and server tests (HTTP API, auth, rate limiting,
anomaly detection, multi-collection support). All of it should pass
before and after any change you make.

**Python client and CLI:**
```bash
cd client/python
pip install -e ".[test]"
pytest tests/test_client.py tests/test_cli.py -v
```
These are unit tests with the HTTP layer mocked — no running server
needed. `tests/test_integration.py` needs a real server
(`cargo run --release --bin server -- /tmp/some_dir 8081` in another
terminal) and skips itself automatically if none is reachable.

**Try it end to end:**
```bash
cargo run --release --bin server -- ./data 8080
# in another terminal
curl http://localhost:8080/health
```
Or via Docker: `docker compose up --build` (see `README.md`).

## What a good contribution looks like here

This project's history is unusually explicit about testing rigor, and
new contributions are expected to match it, not water it down:

- **Every new capability needs a real test**, not just a plausible one.
  Several bugs in this project's own history were only caught because a
  test asserted something specific and precise (e.g., reading a
  compacted file's raw bytes to confirm a tombstone survived, not just
  checking that `get()` returned nothing — those look identical unless
  you check the right thing).
- **Time-dependent logic gets deterministic tests**, not real sleeping.
  The anomaly detector's tests pass explicit `Instant` values (a fixed
  base time plus `Duration` offsets) rather than calling `sleep()` and
  hoping the timing works out — see `src/bin/server.rs`'s anomaly
  detection tests for the pattern.
- **Backward compatibility gets proven, not assumed.** When
  multi-collection support was added, the proof it didn't break
  anything was that all 35 pre-existing server tests passed completely
  unmodified afterward — not a claim, a re-run.
- **If a benchmark or performance claim doesn't hold up under a second,
  more rigorous look, say so and fix the record.** The project's own
  `HISTORY.md` documents a case where an early "insert throughput is
  behind pgvector" finding drove real engineering work, then turned out
  to be a test methodology bug once tested more rigorously — and that
  whole sequence is left visible rather than quietly corrected.

## Before opening a PR

1. `cargo test --release` passes, full output included in the PR
   description if it's not obviously green from CI.
2. `cargo build --release` has no new warnings.
3. Python changes: `pytest` passes for both `test_client.py` and
   `test_cli.py`.
4. If you touched a public API (an HTTP endpoint, a client method, a
   CLI command), update `HISTORY.md` with the detailed writeup (what
   changed, why, how it was tested) in the same PR, update `README.md`'s
   front-matter numbers table if the change affects the headline
   benchmark results, and update `client/python/README.md` for
   client-facing changes — this project treats undocumented behavior as
   unfinished behavior.
5. If you're fixing a bug, add the test that would have caught it
   *first*, confirm it fails against the old code, then fix it.

## Reporting a security issue

Please don't open a public issue for a security vulnerability — see
`SECURITY.md`.

## Code of conduct

See `CODE_OF_CONDUCT.md`. Short version: be respectful, assume good
faith, and keep disagreements about the work, not the person.
