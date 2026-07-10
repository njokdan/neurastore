"""
Benchmarks pgvector on the dataset produced by prepare_dataset.py.

Measures three things NeuraStore's Phase 2/4 numbers will be judged
against:
  1. Insert throughput (vectors/sec)
  2. Unfiltered ANN query latency + recall@10 vs. ground truth
  3. Filtered (hybrid) query latency -- WHERE category = X ORDER BY <->
     -- this is the overfetch-then-filter pattern NeuraStore's fusion
     is trying to beat. Recall isn't computed for the filtered case
     since ground_truth.npy is unfiltered; latency is the number that
     matters here.

Usage:
    python bench_pgvector.py --host localhost --k 10
"""
import argparse
import random
import time

import numpy as np
import psycopg2
import psycopg2.extras
from bench_common import LatencyTracker, recall_at_k, warm_up
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def connect(host: str, port: int):
    return psycopg2.connect(
        host=host, port=port, user="postgres", password="postgres", dbname="neurabench"
    )


def setup_table(conn, dim: int):
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute("DROP TABLE IF EXISTS items;")
        cur.execute(f"""
            CREATE TABLE items (
                id BIGINT PRIMARY KEY,
                embedding vector({dim}),
                category TEXT
            );
        """)
    conn.commit()


def insert_data(conn, base: np.ndarray, categories: np.ndarray, batch_size: int = 1000) -> float:
    start = time.perf_counter()
    with conn.cursor() as cur:
        rows = [
            (i, base[i].tolist(), str(categories[i]))
            for i in range(base.shape[0])
        ]
        psycopg2.extras.execute_values(
            cur,
            "INSERT INTO items (id, embedding, category) VALUES %s",
            rows,
            template="(%s, %s, %s)",
            page_size=batch_size,
        )
    conn.commit()
    elapsed = time.perf_counter() - start
    return base.shape[0] / elapsed


def build_index(conn, m: int = 16, ef_construction: int = 64):
    start = time.perf_counter()
    with conn.cursor() as cur:
        cur.execute(
            f"CREATE INDEX ON items USING hnsw (embedding vector_l2_ops) "
            f"WITH (m = {m}, ef_construction = {ef_construction});"
        )
    conn.commit()
    return time.perf_counter() - start


def bench_unfiltered(conn, queries: np.ndarray, ground_truth: np.ndarray, k: int, ef_search: int = 40):
    tracker = LatencyTracker()
    recalls = []
    with conn.cursor() as cur:
        cur.execute(f"SET hnsw.ef_search = {ef_search};")

        def run(i):
            cur.execute(
                "SELECT id FROM items ORDER BY embedding <-> %s::vector LIMIT %s;",
                (queries[i].tolist(), k),
            )
            cur.fetchall()

        warm_up(run, len(queries))

        for i, q in enumerate(queries):
            with tracker.timed():
                cur.execute(
                    "SELECT id FROM items ORDER BY embedding <-> %s::vector LIMIT %s;",
                    (q.tolist(), k),
                )
                results = [r[0] for r in cur.fetchall()]
            recalls.append(recall_at_k(results, ground_truth[i].tolist(), k))
    tracker.print_summary("pgvector unfiltered query latency")
    print(f"pgvector unfiltered recall@{k}: {np.mean(recalls):.3f}")


def bench_filtered(conn, queries: np.ndarray, categories: list[str], k: int, ef_search: int = 40):
    tracker = LatencyTracker()
    with conn.cursor() as cur:
        cur.execute(f"SET hnsw.ef_search = {ef_search};")

        def run(i):
            cat = categories[i % len(categories)]
            cur.execute(
                "SELECT id FROM items WHERE category = %s "
                "ORDER BY embedding <-> %s::vector LIMIT %s;",
                (cat, queries[i].tolist(), k),
            )
            cur.fetchall()

        warm_up(run, len(queries))

        for i, q in enumerate(queries):
            cat = categories[i % len(categories)]
            with tracker.timed():
                cur.execute(
                    "SELECT id FROM items WHERE category = %s "
                    "ORDER BY embedding <-> %s::vector LIMIT %s;",
                    (cat, q.tolist(), k),
                )
                cur.fetchall()
    tracker.print_summary(f"pgvector filtered (category=X) query latency")
    print("(Filtered recall not computed -- ground truth is unfiltered. "
          "Latency is the number that matters for the overfetch-then-filter comparison.)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=5432)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    base = np.load(DATA_DIR / "base.npy")
    queries = np.load(DATA_DIR / "queries.npy")
    categories = np.load(DATA_DIR / "categories.npy", allow_pickle=True)
    ground_truth = np.load(DATA_DIR / "ground_truth.npy")
    dim = base.shape[1]

    print(f"Dataset: {base.shape[0]} base vectors, dim={dim}, {queries.shape[0]} queries")

    conn = connect(args.host, args.port)
    setup_table(conn, dim)

    print("Inserting...")
    throughput = insert_data(conn, base, categories)
    print(f"pgvector insert throughput: {throughput:.1f} vectors/sec")

    print("Building HNSW index...")
    build_secs = build_index(conn)
    print(f"pgvector HNSW build time: {build_secs:.2f}s")

    order = ["unfiltered", "filtered"]
    random.shuffle(order)
    print(f"Running benchmarks in order: {order} (randomized to avoid ordering bias)")
    for kind in order:
        if kind == "unfiltered":
            bench_unfiltered(conn, queries, ground_truth, args.k)
        else:
            bench_filtered(conn, queries, list(dict.fromkeys(categories.tolist())), args.k)

    conn.close()


if __name__ == "__main__":
    main()
