"""
Benchmarks Milvus (standalone) on the same dataset as bench_pgvector.py,
so the two are directly comparable.

Usage:
    python bench_milvus.py --host localhost --k 10
"""
import argparse
import random
from pathlib import Path

import numpy as np
from bench_common import LatencyTracker, recall_at_k, warm_up
from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)

DATA_DIR = Path(__file__).parent.parent / "data"
COLLECTION_NAME = "neurabench_items"


def setup_collection(dim: int) -> Collection:
    if utility.has_collection(COLLECTION_NAME):
        utility.drop_collection(COLLECTION_NAME)

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=False),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
        FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=32),
    ]
    schema = CollectionSchema(fields, description="NeuraStore benchmark baseline")
    collection = Collection(COLLECTION_NAME, schema)
    return collection


def insert_data(collection: Collection, base: np.ndarray, categories: np.ndarray, batch_size: int = 5000) -> float:
    import time

    start = time.perf_counter()
    n = base.shape[0]
    for i in range(0, n, batch_size):
        end = min(i + batch_size, n)
        collection.insert([
            list(range(i, end)),
            base[i:end].tolist(),
            [str(c) for c in categories[i:end]],
        ])
    collection.flush()
    elapsed = time.perf_counter() - start
    return n / elapsed


def build_index(collection: Collection, m: int = 16, ef_construction: int = 64) -> float:
    import time

    start = time.perf_counter()
    collection.create_index(
        field_name="embedding",
        index_params={
            "index_type": "HNSW",
            "metric_type": "L2",
            "params": {"M": m, "efConstruction": ef_construction},
        },
    )
    collection.load()
    return time.perf_counter() - start


def bench_unfiltered(collection: Collection, queries: np.ndarray, ground_truth: np.ndarray, k: int, ef: int = 40):
    tracker = LatencyTracker()
    recalls = []

    def run(i):
        collection.search(
            data=[queries[i].tolist()],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"ef": ef}},
            limit=k,
        )

    warm_up(run, len(queries))

    for i, q in enumerate(queries):
        with tracker.timed():
            results = collection.search(
                data=[q.tolist()],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"ef": ef}},
                limit=k,
            )
        ids = [hit.id for hit in results[0]]
        recalls.append(recall_at_k(ids, ground_truth[i].tolist(), k))
    tracker.print_summary("Milvus unfiltered query latency")
    print(f"Milvus unfiltered recall@{k}: {np.mean(recalls):.3f}")


def bench_filtered(collection: Collection, queries: np.ndarray, categories: list[str], k: int, ef: int = 40):
    tracker = LatencyTracker()

    def run(i):
        cat = categories[i % len(categories)]
        collection.search(
            data=[queries[i].tolist()],
            anns_field="embedding",
            param={"metric_type": "L2", "params": {"ef": ef}},
            limit=k,
            expr=f'category == "{cat}"',
        )

    warm_up(run, len(queries))

    for i, q in enumerate(queries):
        cat = categories[i % len(categories)]
        with tracker.timed():
            collection.search(
                data=[q.tolist()],
                anns_field="embedding",
                param={"metric_type": "L2", "params": {"ef": ef}},
                limit=k,
                expr=f'category == "{cat}"',
            )
    tracker.print_summary("Milvus filtered (category=X) query latency")
    print("(Filtered recall not computed -- ground truth is unfiltered. "
          "Latency is the number that matters for the overfetch-then-filter comparison.)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", default="19530")
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    connections.connect(host=args.host, port=args.port)

    base = np.load(DATA_DIR / "base.npy")
    queries = np.load(DATA_DIR / "queries.npy")
    categories = np.load(DATA_DIR / "categories.npy", allow_pickle=True)
    ground_truth = np.load(DATA_DIR / "ground_truth.npy")
    dim = base.shape[1]

    print(f"Dataset: {base.shape[0]} base vectors, dim={dim}, {queries.shape[0]} queries")

    collection = setup_collection(dim)

    print("Inserting...")
    throughput = insert_data(collection, base, categories)
    print(f"Milvus insert throughput: {throughput:.1f} vectors/sec")

    print("Building HNSW index...")
    build_secs = build_index(collection)
    print(f"Milvus HNSW build + load time: {build_secs:.2f}s")

    order = ["unfiltered", "filtered"]
    random.shuffle(order)
    print(f"Running benchmarks in order: {order} (randomized to avoid ordering bias)")
    for kind in order:
        if kind == "unfiltered":
            bench_unfiltered(collection, queries, ground_truth, args.k)
        else:
            bench_filtered(collection, queries, list(dict.fromkeys(categories.tolist())), args.k)


if __name__ == "__main__":
    main()
