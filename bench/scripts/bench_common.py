"""Shared helpers for the pgvector and Milvus benchmark scripts, so both
report numbers the same way and are directly comparable."""
import time
from contextlib import contextmanager
from typing import Callable

import numpy as np


def recall_at_k(retrieved: list[int], ground_truth: list[int], k: int) -> float:
    """Fraction of the top-k ground truth neighbors that appear in the
    retrieved set. Standard ANN benchmark metric."""
    gt_set = set(ground_truth[:k])
    ret_set = set(retrieved[:k])
    if not gt_set:
        return 0.0
    return len(gt_set & ret_set) / len(gt_set)


def warm_up(run_query: Callable[[int], None], n_queries: int, n_warmup: int = 20):
    """Run `n_warmup` untimed queries before a benchmark's timed loop
    starts. Without this, the first measured queries absorb cold-start
    costs (connection/index page cache not yet warm, JIT-ish lazy init
    in the client library, etc.) that have nothing to do with the
    engine's steady-state performance -- and whichever benchmark runs
    *first* in a script unfairly eats that cost while a later benchmark
    doesn't. `run_query(i)` should execute query index `i % n_queries`
    and discard the result; timing is the caller's job, not this
    function's -- this only exists to prime caches beforehand.
    """
    for i in range(n_warmup):
        run_query(i % n_queries)


class LatencyTracker:
    """Collects per-query latencies and reports percentiles -- avoid
    reporting only a mean, since tail latency is usually what actually
    matters for a database."""

    def __init__(self):
        self.samples_ms: list[float] = []

    @contextmanager
    def timed(self):
        start = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.samples_ms.append(elapsed_ms)

    def summary(self) -> dict:
        arr = np.array(self.samples_ms)
        return {
            "count": len(arr),
            "mean_ms": float(np.mean(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(np.max(arr)),
        }

    def print_summary(self, label: str):
        s = self.summary()
        print(
            f"{label}: n={s['count']} "
            f"mean={s['mean_ms']:.3f}ms p50={s['p50_ms']:.3f}ms "
            f"p95={s['p95_ms']:.3f}ms p99={s['p99_ms']:.3f}ms max={s['max_ms']:.3f}ms"
        )
    """Collects per-query latencies and reports percentiles -- avoid
    reporting only a mean, since tail latency is usually what actually
    matters for a database."""

    def __init__(self):
        self.samples_ms: list[float] = []

    @contextmanager
    def timed(self):
        start = time.perf_counter()
        yield
        elapsed_ms = (time.perf_counter() - start) * 1000
        self.samples_ms.append(elapsed_ms)

    def summary(self) -> dict:
        arr = np.array(self.samples_ms)
        return {
            "count": len(arr),
            "mean_ms": float(np.mean(arr)),
            "p50_ms": float(np.percentile(arr, 50)),
            "p95_ms": float(np.percentile(arr, 95)),
            "p99_ms": float(np.percentile(arr, 99)),
            "max_ms": float(np.max(arr)),
        }

    def print_summary(self, label: str):
        s = self.summary()
        print(
            f"{label}: n={s['count']} "
            f"mean={s['mean_ms']:.3f}ms p50={s['p50_ms']:.3f}ms "
            f"p95={s['p95_ms']:.3f}ms p99={s['p99_ms']:.3f}ms max={s['max_ms']:.3f}ms"
        )
