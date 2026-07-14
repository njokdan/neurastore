"""NeuraStore Python client.

    from neurastore_client import NeuraStoreClient

    client = NeuraStoreClient("http://localhost:8080")
    client.insert(1, [0.1, 0.2, 0.3], metadata={"category": "docs"})
    client.build_index()
    for hit in client.search([0.1, 0.2, 0.3], k=5):
        print(hit.id, hit.distance)

Note: `neurastore_client.ConnectionError` is this package's own
exception type (see exceptions.py), not Python's built-in
`ConnectionError` -- a deliberate choice (it clearly names what went
wrong, and NeuraStore's other exceptions follow the same
`NeuraStoreError` hierarchy), but worth knowing if you're catching
exceptions by name.
"""
from .client import NeuraStoreClient
from .exceptions import (
    AuthenticationError,
    BadRequestError,
    ConnectionError,
    NeuraStoreError,
    NotFoundError,
    RateLimitError,
    ServerError,
)
from .models import Record, SearchResult, Stats

__version__ = "0.1.0"

__all__ = [
    "NeuraStoreClient",
    "NeuraStoreError",
    "ConnectionError",
    "NotFoundError",
    "AuthenticationError",
    "RateLimitError",
    "BadRequestError",
    "ServerError",
    "Record",
    "SearchResult",
    "Stats",
]
