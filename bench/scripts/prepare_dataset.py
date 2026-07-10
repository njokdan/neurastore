"""
Prepares a vector dataset for benchmarking pgvector / Milvus against
NeuraStore's eventual index.

Two modes:

  --mode synthetic (default, no download, instant)
      Generates random float32 vectors + a metadata "category" column
      (so filtered queries can be tested too) + brute-force ground truth.
      Good for a fast smoke test of the harness itself.

  --mode siftsmall / --mode sift1m
      Downloads the standard texmex SIFT benchmark corpus
      (http://corpus-texmex.irisa.fr/) -- the same dataset pgvector,
      Milvus, and most ANN benchmarks use, so results are comparable
      to published numbers elsewhere. siftsmall = 10K base vectors
      (~5MB, quick). sift1m = 1M base vectors (~500MB, slow download).

Usage:
    python prepare_dataset.py --mode synthetic --n 100000 --dim 128
    python prepare_dataset.py --mode siftsmall
    python prepare_dataset.py --mode sift1m
"""
import argparse
import os
import struct
import tarfile
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent.parent / "data"

SIFT_URLS = {
    "siftsmall": "ftp://ftp.irisa.fr/local/texmex/corpus/siftsmall.tar.gz",
    "sift1m": "ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz",
}


def read_fvecs(path: Path) -> np.ndarray:
    """Read a .fvecs file (texmex format: each vector prefixed by its
    int32 dimension)."""
    data = np.fromfile(path, dtype=np.int32)
    dim = data[0]
    return data.view(np.float32).reshape(-1, dim + 1)[:, 1:].copy()


def read_ivecs(path: Path) -> np.ndarray:
    """Read an .ivecs file (same framing, int32 payload) -- used for
    ground-truth neighbor indices."""
    data = np.fromfile(path, dtype=np.int32)
    dim = data[0]
    return data.reshape(-1, dim + 1)[:, 1:].copy()


def brute_force_ground_truth(base: np.ndarray, queries: np.ndarray, k: int = 100) -> np.ndarray:
    """Exact nearest neighbors via full distance computation. Used as
    ground truth for recall@k when the dataset doesn't ship its own
    (e.g. the synthetic path). O(n_queries * n_base) -- fine for
    smoke-test sizes, not for sift1m (which ships its own ground truth)."""
    gt = np.zeros((queries.shape[0], k), dtype=np.int64)
    for i, q in enumerate(queries):
        dists = np.linalg.norm(base - q, axis=1)
        gt[i] = np.argsort(dists)[:k]
    return gt


def prepare_synthetic(n: int, dim: int, n_queries: int, seed: int = 42):
    rng = np.random.default_rng(seed)
    base = rng.standard_normal((n, dim), dtype=np.float32)
    queries = rng.standard_normal((n_queries, dim), dtype=np.float32)
    categories = rng.choice(["docs", "code", "chat", "logs"], size=n)

    print(f"Computing brute-force ground truth for {n_queries} queries over {n} base vectors...")
    gt = brute_force_ground_truth(base, queries, k=100)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    np.save(DATA_DIR / "base.npy", base)
    np.save(DATA_DIR / "queries.npy", queries)
    np.save(DATA_DIR / "categories.npy", categories)
    np.save(DATA_DIR / "ground_truth.npy", gt)
    print(f"Wrote synthetic dataset to {DATA_DIR} (base={base.shape}, queries={queries.shape})")


def prepare_sift(mode: str):
    url = SIFT_URLS[mode]
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    archive_path = DATA_DIR / f"{mode}.tar.gz"

    if not archive_path.exists():
        print(f"Downloading {url} -- this may take a while for sift1m...")
        urllib.request.urlretrieve(url, archive_path)
    else:
        print(f"Found existing archive at {archive_path}, skipping download.")

    print("Extracting...")
    with tarfile.open(archive_path) as tar:
        tar.extractall(DATA_DIR)

    prefix = "siftsmall" if mode == "siftsmall" else "sift"
    extracted = DATA_DIR / prefix
    base = read_fvecs(extracted / f"{prefix}_base.fvecs")
    queries = read_fvecs(extracted / f"{prefix}_query.fvecs")
    gt = read_ivecs(extracted / f"{prefix}_groundtruth.ivecs")

    # texmex has no metadata -- synthesize a category column so filtered
    # (hybrid) queries can still be benchmarked on this dataset.
    rng = np.random.default_rng(42)
    categories = rng.choice(["docs", "code", "chat", "logs"], size=base.shape[0])

    np.save(DATA_DIR / "base.npy", base)
    np.save(DATA_DIR / "queries.npy", queries)
    np.save(DATA_DIR / "categories.npy", categories)
    np.save(DATA_DIR / "ground_truth.npy", gt)
    print(f"Wrote {mode} dataset to {DATA_DIR} (base={base.shape}, queries={queries.shape})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["synthetic", "siftsmall", "sift1m"], default="synthetic")
    parser.add_argument("--n", type=int, default=100_000, help="synthetic mode: number of base vectors")
    parser.add_argument("--dim", type=int, default=128, help="synthetic mode: vector dimension")
    parser.add_argument("--n-queries", type=int, default=200, help="synthetic mode: number of query vectors")
    args = parser.parse_args()

    if args.mode == "synthetic":
        prepare_synthetic(args.n, args.dim, args.n_queries)
    else:
        prepare_sift(args.mode)


if __name__ == "__main__":
    main()
