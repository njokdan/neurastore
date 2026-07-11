"""
Benchmarks NeuraStore over its real HTTP API -- client-to-server, the
same way bench_pgvector.py and bench_milvus.py measure their targets.

This is the benchmark the project deliberately deferred: every prior
NeuraStore number (bin/bench_neurastore) was an in-process Rust function
call, which is not a fair comparison to pgvector/Milvus paying a real
SQL/gRPC round-trip. Now that Phase 5 has a real network API, this
script closes that gap -- same dataset, same warm-up + randomized-order
methodology, same percentile reporting, but over an actual HTTP
connection like the other two.

Uses a single requests.Session() for the whole run, not a fresh
requests.post() per call. This matters a lot on Windows specifically:
without connection reuse, each request pays a full new TCP handshake,
and resolving "localhost" on Windows often tries IPv6 (::1) first,
times out, then falls back to IPv4 -- a well-documented ~1-2 SECOND tax
per request. A persistent Session (and defaulting to 127.0.0.1 instead
of localhost) avoids both problems. Without this fix, per-query latency
here was measured at ~2000ms flat -- not a real engine number, a
Windows networking artifact.

Prerequisites:
    1. python bench/scripts/prepare_dataset.py --mode siftsmall
    2. cargo run --release --bin server -- /tmp/neurastore_http_bench 8080
       (running in a separate terminal, left up for the duration of this script)

Usage:
    python bench/scripts/bench_neurastore_http.py --k 10
"""
import argparse
import random
import time
import struct
from pathlib import Path

import numpy as np
import orjson
import requests
from bench_common import LatencyTracker, recall_at_k, warm_up

DATA_DIR = Path(__file__).parent.parent / "data"

JSON_HEADERS = {"Content-Type": "application/json"}
BINARY_HEADERS = {"Content-Type": "application/octet-stream"}
BINARY_MAGIC = b"NSBB"


def wait_for_server(session: requests.Session, base_url: str, timeout_s: float = 10.0):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = session.get(f"{base_url}/health", timeout=1)
            if r.status_code == 200:
                return
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(0.2)
    raise RuntimeError(
        f"NeuraStore server not responding at {base_url}/health after {timeout_s}s -- "
        "is `cargo run --release --bin server -- <dir> <port>` running?"
    )


def encode_binary_batch(ids: np.ndarray, vectors: np.ndarray, categories: np.ndarray) -> bytes:
    """Matches src/bin/server.rs's `parse_binary_batch` format exactly:
    magic + record_count + dim, then per record: id (u64 LE) + raw f32 LE
    vector bytes (no text encoding at all) + metadata_len (u32 LE) +
    UTF-8 JSON metadata. Vectors go out via numpy's .tobytes() -- no
    per-float text conversion, no .tolist(), no JSON encoding of the
    actual vector data, which is the whole point of this endpoint (see
    server.rs's parse_binary_batch docs for why the JSON path couldn't
    fully close this gap even after speeding up the client side alone).
    """
    n, dim = vectors.shape
    # Explicit little-endian float32, matching what the server expects --
    # explicit rather than assuming the platform's native byte order,
    # even though x86/ARM are LE in practice.
    vectors_le = vectors.astype("<f4")
    parts = [BINARY_MAGIC, struct.pack("<II", n, dim)]
    for i in range(n):
        meta_bytes = orjson.dumps({"category": str(categories[i])})
        parts.append(struct.pack("<Q", int(ids[i])))
        parts.append(vectors_le[i].tobytes())
        parts.append(struct.pack("<I", len(meta_bytes)))
        parts.append(meta_bytes)
    return b"".join(parts)


def insert_data_binary(session: requests.Session, base_url: str, base: np.ndarray, categories: np.ndarray, batch_size: int = 1000) -> float:
    start = time.perf_counter()
    n = base.shape[0]
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        ids = np.arange(i, end, dtype=np.uint64)
        body = encode_binary_batch(ids, base[i:end], categories[i:end])
        r = session.post(f"{base_url}/v1/records/batch/binary", data=body, headers=BINARY_HEADERS, timeout=30)
        r.raise_for_status()
    elapsed = time.perf_counter() - start
    return n / elapsed


def insert_data(session: requests.Session, base_url: str, base: np.ndarray, categories: np.ndarray, batch_size: int = 1000) -> float:
    start = time.perf_counter()
    n = base.shape[0]
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        records = [
            {"id": j, "vector": base[j], "metadata": {"category": str(categories[j])}}
            for j in range(i, end)
        ]
        # orjson with OPT_SERIALIZE_NUMPY serializes the numpy arrays
        # directly -- skips both the slower stdlib json module `requests`
        # uses internally for `json=...`, and the `.tolist()` conversion
        # step (itself real overhead at this scale: turning a numpy
        # float32 array into a Python list of floats isn't free either).
        # Sent as raw bytes via `data=`, not `json=`, since `json=` would
        # re-serialize with stdlib json regardless of this pre-encoding.
        body = orjson.dumps({"records": records}, option=orjson.OPT_SERIALIZE_NUMPY)
        r = session.post(f"{base_url}/v1/records/batch", data=body, headers=JSON_HEADERS, timeout=30)
        r.raise_for_status()
    elapsed = time.perf_counter() - start
    return n / elapsed


def build_index(session: requests.Session, base_url: str) -> float:
    start = time.perf_counter()
    r = session.post(f"{base_url}/v1/index/build", timeout=120)
    r.raise_for_status()
    return time.perf_counter() - start


def bench_unfiltered(session: requests.Session, base_url: str, queries: np.ndarray, ground_truth: np.ndarray, k: int, ef_search: int):
    tracker = LatencyTracker()
    recalls = []

    def run(i):
        session.post(f"{base_url}/v1/search", json={"vector": queries[i].tolist(), "k": k, "ef_search": ef_search}, timeout=10)

    warm_up(run, len(queries))

    for i, q in enumerate(queries):
        with tracker.timed():
            r = session.post(f"{base_url}/v1/search", json={"vector": q.tolist(), "k": k, "ef_search": ef_search}, timeout=10)
        r.raise_for_status()
        ids = [item["id"] for item in r.json()["results"]]
        recalls.append(recall_at_k(ids, ground_truth[i].tolist(), k))

    tracker.print_summary("NeuraStore (HTTP) unfiltered query latency")
    print(f"NeuraStore (HTTP) unfiltered recall@{k}: {np.mean(recalls):.3f}")
    return tracker


def bench_filtered(session: requests.Session, base_url: str, queries: np.ndarray, categories: list[str], k: int, ef_search: int):
    tracker = LatencyTracker()

    def run(i):
        cat = categories[i % len(categories)]
        session.post(
            f"{base_url}/v1/search/filtered",
            json={"vector": queries[i].tolist(), "k": k, "ef_search": ef_search, "field": "category", "value": cat},
            timeout=10,
        )

    warm_up(run, len(queries))

    for i, q in enumerate(queries):
        cat = categories[i % len(categories)]
        with tracker.timed():
            r = session.post(
                f"{base_url}/v1/search/filtered",
                json={"vector": q.tolist(), "k": k, "ef_search": ef_search, "field": "category", "value": cat},
                timeout=10,
            )
        r.raise_for_status()

    tracker.print_summary("NeuraStore (HTTP) filtered (category=X) query latency")
    return tracker


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1", help="Use 127.0.0.1, not localhost -- avoids a slow IPv6-then-fallback DNS resolution on Windows.")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--ef-search", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=1000, help="records per HTTP batch insert (1000 measured best; larger batches didn't help further -- see bench/README.md)")
    parser.add_argument("--binary", action="store_true", help="use the binary bulk-insert endpoint instead of JSON -- see bench/README.md's Phase 5 section for why this exists")
    args = parser.parse_args()
    base_url = f"http://{args.host}:{args.port}"

    # One Session for the whole run -- reuses a single keep-alive TCP
    # connection instead of paying a fresh handshake (and on Windows,
    # potentially a slow IPv6-fallback DNS lookup) on every single request.
    session = requests.Session()

    print(f"Checking NeuraStore server at {base_url}...")
    wait_for_server(session, base_url)
    print("Server is up.")

    base = np.load(DATA_DIR / "base.npy")
    queries = np.load(DATA_DIR / "queries.npy")
    categories = np.load(DATA_DIR / "categories.npy", allow_pickle=True)
    ground_truth = np.load(DATA_DIR / "ground_truth.npy")

    print(f"Dataset: {base.shape[0]} base vectors, dim={base.shape[1]}, {queries.shape[0]} queries")
    if queries.shape[0] != 100:
        print(f"NOTE: real siftsmall ships exactly 100 queries -- {queries.shape[0]} suggests bench/data")
        print("      currently holds a different/regenerated dataset, not the original siftsmall corpus.")
        print("      Re-run `python prepare_dataset.py --mode siftsmall` if that's not intentional.")

    print(f"Inserting (via HTTP, batched, {'binary' if args.binary else 'JSON'})...")
    if args.binary:
        throughput = insert_data_binary(session, base_url, base, categories, batch_size=args.batch_size)
    else:
        throughput = insert_data(session, base_url, base, categories, batch_size=args.batch_size)
    print(f"NeuraStore (HTTP) insert throughput: {throughput:.1f} vectors/sec")
    print("(Note: this includes real HTTP/JSON overhead per batch, unlike bin/bench_neurastore's")
    print(" in-process number -- expect it to be lower than the 11,355 vec/sec Phase 2 figure.)")

    print("Building index (via HTTP)...")
    build_secs = build_index(session, base_url)
    print(f"NeuraStore (HTTP) index build time: {build_secs:.2f}s")

    order = ["unfiltered", "filtered"]
    random.shuffle(order)
    print(f"Running benchmarks in order: {order} (randomized to avoid ordering bias)")
    unique_categories = list(dict.fromkeys(categories.tolist()))
    for kind in order:
        if kind == "unfiltered":
            bench_unfiltered(session, base_url, queries, ground_truth, args.k, args.ef_search)
        else:
            bench_filtered(session, base_url, queries, unique_categories, args.k, args.ef_search)


if __name__ == "__main__":
    main()
