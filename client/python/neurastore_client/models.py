"""Data models returned by the NeuraStore client.

Plain dataclasses, not a heavier ORM-style abstraction -- these are
simple, transparent wrappers around the server's JSON responses, not a
place to hide business logic.
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class Record:
    """A single stored record, as returned by `get()`."""

    id: int
    vector: List[float]
    metadata: Dict[str, str] = field(default_factory=dict)

    @classmethod
    def _from_json(cls, data: dict) -> "Record":
        return cls(id=data["id"], vector=data["vector"], metadata=data.get("metadata", {}))


@dataclass
class SearchResult:
    """One hit from a `search()` or `search_filtered()` call. `distance`
    is squared Euclidean (L2) distance -- matches the metric used
    throughout NeuraStore's own benchmarking against pgvector/Milvus.
    Smaller is closer; take the square root yourself if you need true
    Euclidean distance rather than squared."""

    id: int
    distance: float

    @classmethod
    def _from_json(cls, data: dict) -> "SearchResult":
        return cls(id=data["id"], distance=data["distance"])


@dataclass
class Stats:
    """Server-side collection statistics, as returned by `stats()`."""

    live_records: int
    memtable_records: int
    sstable_count: int
    index_built: bool
    index_len: Optional[int]

    @classmethod
    def _from_json(cls, data: dict) -> "Stats":
        return cls(
            live_records=data["live_records"],
            memtable_records=data["memtable_records"],
            sstable_count=data["sstable_count"],
            index_built=data["index_built"],
            index_len=data.get("index_len"),
        )
