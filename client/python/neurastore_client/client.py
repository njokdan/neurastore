"""The main NeuraStore client.

Design choices worth knowing about:

- Only `requests` is a hard dependency -- vectors are plain lists of
  floats, not numpy arrays. Most people reaching for a database client
  don't want numpy forced on them; if you already have numpy arrays,
  pass `.tolist()` or an array directly (this client accepts anything
  iterable of numbers).
- Uses a single `requests.Session` internally, not a fresh connection
  per call. This isn't a stylistic choice -- NeuraStore's own benchmark
  tooling hit a real, measured ~2-second-per-request penalty on Windows
  from *not* doing this (repeated fresh connections can trigger a slow
  IPv6-then-fallback DNS resolution per call). A client library that
  didn't reuse connections would hand that same bug to everyone who
  used it.
- HTTP errors are translated into NeuraStore-specific exceptions
  (see exceptions.py), not left as raw `requests.HTTPError` -- so
  callers can `except neurastore_client.NotFoundError` instead of
  needing to know NeuraStore's status code conventions.
"""
import struct
from typing import Dict, List, Optional, Sequence

import requests

from .exceptions import (
    AuthenticationError,
    BadRequestError,
    ConnectionError,
    NotFoundError,
    RateLimitError,
    ServerError,
)
from .models import Record, SearchResult, Stats

_BINARY_MAGIC = b"NSBB"


class NeuraStoreClient:
    """Client for a running NeuraStore server.

    Example:
        >>> client = NeuraStoreClient("http://localhost:8080")
        >>> client.insert(1, [0.1, 0.2, 0.3], metadata={"category": "docs"})
        >>> client.build_index()
        >>> results = client.search([0.1, 0.2, 0.3], k=5)

    Or as a context manager, which closes the underlying connection
    pool when done:
        >>> with NeuraStoreClient("http://localhost:8080") as client:
        ...     client.insert(1, [0.1, 0.2, 0.3])
    """

    def __init__(
        self,
        base_url: str = "http://localhost:8080",
        timeout: float = 30.0,
        api_key: Optional[str] = None,
    ):
        """`api_key`, if given, is sent as `Authorization: Bearer <api_key>`
        on every request. Only needed if the server was started with
        `NEURASTORE_API_KEYS` set -- a server with no keys configured
        (the default) ignores this header entirely, so passing `api_key`
        against an unprotected server is harmless, not an error."""
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._session = requests.Session()
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

    def __enter__(self) -> "NeuraStoreClient":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        self._session.close()

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _collection_path(self, collection: str, suffix: str) -> str:
        """Builds the right path for a given collection. "default" uses
        the original, pre-multi-collection routes verbatim (e.g.
        `/v1/records`) -- not `/v1/collections/default/records`, even
        though the server treats both identically -- specifically so
        every existing call site, and every test asserting on exact
        URLs, keeps working completely unchanged. Any other collection
        name uses the new `/v1/collections/<name>/...` routes."""
        if collection == "default":
            return f"/v1{suffix}"
        return f"/v1/collections/{collection}{suffix}"

    def _handle_response(self, response: requests.Response) -> requests.Response:
        if response.status_code == 404:
            raise NotFoundError(_error_message(response))
        if response.status_code == 401:
            raise AuthenticationError(_error_message(response))
        if response.status_code == 429:
            raise RateLimitError(_error_message(response))
        if response.status_code == 400:
            raise BadRequestError(_error_message(response))
        if response.status_code >= 500:
            raise ServerError(_error_message(response))
        response.raise_for_status()
        return response

    def _request(self, method: str, path: str, **kwargs) -> requests.Response:
        try:
            response = self._session.request(method, self._url(path), timeout=self.timeout, **kwargs)
        except requests.exceptions.ConnectionError as e:
            raise ConnectionError(
                f"could not reach NeuraStore server at {self.base_url} -- is it running?"
            ) from e
        except requests.exceptions.Timeout as e:
            raise ConnectionError(f"request to {self.base_url} timed out after {self.timeout}s") from e
        return self._handle_response(response)

    # -- Health --------------------------------------------------------

    def health(self) -> bool:
        """Returns True if the server is reachable and healthy. Does not
        raise on a connection failure -- that's the point of this method,
        for callers who want to check without handling an exception."""
        try:
            response = self._session.get(self._url("/health"), timeout=self.timeout)
            return response.status_code == 200
        except requests.exceptions.RequestException:
            return False

    # -- Writes ----------------------------------------------------------

    def insert(
        self,
        id: int,
        vector: Sequence[float],
        metadata: Optional[Dict[str, str]] = None,
        collection: str = "default",
    ) -> None:
        """Insert or update a single record. Re-using an existing `id` is
        an update, not an error -- the old vector/metadata is replaced.
        `collection` addresses a named collection (created lazily on
        first write); omit it to use the default collection, exactly as
        before multi-collection support existed."""
        body = {"id": id, "vector": list(vector), "metadata": metadata or {}}
        self._request("POST", self._collection_path(collection, "/records"), json=body)

    def insert_batch(
        self,
        records: Sequence[Dict],
        binary: bool = False,
        collection: str = "default",
    ) -> None:
        """Insert or update many records in one request -- one WAL fsync
        for the whole batch server-side, much faster than the equivalent
        number of individual `insert()` calls for bulk loads.

        `records` is a sequence of dicts, each with `id`, `vector`, and
        optionally `metadata`:
            client.insert_batch([
                {"id": 1, "vector": [0.1, 0.2], "metadata": {"category": "docs"}},
                {"id": 2, "vector": [0.3, 0.4]},
            ])

        `binary=True` uses a compact binary wire format instead of JSON
        for the vector data -- correctness-verified to produce identical
        results to the JSON path, but its performance advantage over
        JSON was investigated and found NOT to be reliably confirmed on
        the hardware this client was developed against (see the main
        NeuraStore repo's README for that story). Left available since
        it's real, tested infrastructure -- just don't assume it's
        faster without measuring on your own setup.

        `collection` addresses a named collection; omit it for the
        default collection.
        """
        if not records:
            raise BadRequestError("records must not be empty")

        if binary:
            body = _encode_binary_batch(records)
            self._request(
                "POST",
                self._collection_path(collection, "/records/batch/binary"),
                data=body,
                headers={"Content-Type": "application/octet-stream"},
            )
        else:
            body = {"records": [
                {"id": r["id"], "vector": list(r["vector"]), "metadata": r.get("metadata", {})}
                for r in records
            ]}
            self._request("POST", self._collection_path(collection, "/records/batch"), json=body)

    def delete(self, id: int, collection: str = "default") -> None:
        """Soft-delete a record. A no-op (does not raise) if the id
        doesn't exist -- matches the server's own delete semantics."""
        self._request("DELETE", self._collection_path(collection, f"/records/{id}"))

    def build_index(self, collection: str = "default") -> None:
        """Build (or rebuild) the vector index from all currently live
        records. Required before `search()`/`search_filtered()` will
        work -- calling it again later is optional, not required for
        correctness (writes made after the first `build_index()` call
        are kept in sync automatically), but can help reclaim space from
        accumulated soft-deletes."""
        self._request("POST", self._collection_path(collection, "/index/build"))

    def compact(self, collection: str = "default") -> None:
        """Reclaims space accumulated from deletes and updates -- merges
        on-disk storage into a single file (dropping superseded record
        versions) and, if an index has been built, rebuilds it too
        (dropping stale/tombstoned graph nodes). Safe to call anytime;
        a no-op if there's nothing to compact. Worth calling periodically
        on a long-running server with a meaningful delete/update rate --
        without it, deleted and superseded data accumulates indefinitely."""
        self._request("POST", self._collection_path(collection, "/compact"))

    # -- Reads -----------------------------------------------------------

    def get(self, id: int, collection: str = "default") -> Record:
        """Fetch a single record by id. Raises NotFoundError if it
        doesn't exist or was deleted."""
        response = self._request("GET", self._collection_path(collection, f"/records/{id}"))
        return Record._from_json(response.json())

    def search(
        self,
        vector: Sequence[float],
        k: int = 10,
        ef_search: int = 40,
        collection: str = "default",
    ) -> List[SearchResult]:
        """Approximate k-nearest-neighbor search. `ef_search` trades
        recall for latency -- higher explores more of the graph for
        better accuracy at the cost of speed. Raises BadRequestError if
        `build_index()` hasn't been called yet."""
        body = {"vector": list(vector), "k": k, "ef_search": ef_search}
        response = self._request("POST", self._collection_path(collection, "/search"), json=body)
        return [SearchResult._from_json(r) for r in response.json()["results"]]

    def search_filtered(
        self,
        vector: Sequence[float],
        field: str,
        value: str,
        k: int = 10,
        ef_search: int = 40,
        collection: str = "default",
    ) -> List[SearchResult]:
        """k-NN search restricted to records where `metadata[field] ==
        value`. The predicate is pushed into the search itself (or
        answered by exact brute-force computation for highly selective
        filters), not applied by discarding an unfiltered result set
        after fetching it -- see the main NeuraStore repo's README for
        why that distinction is the whole point of this method existing."""
        body = {"vector": list(vector), "k": k, "ef_search": ef_search, "field": field, "value": value}
        response = self._request("POST", self._collection_path(collection, "/search/filtered"), json=body)
        return [SearchResult._from_json(r) for r in response.json()["results"]]

    def stats(self, collection: str = "default") -> Stats:
        """Current collection statistics -- live record count, whether
        the index has been built, etc."""
        response = self._request("GET", self._collection_path(collection, "/stats"))
        return Stats._from_json(response.json())

    def list_collections(self) -> List[str]:
        """Lists every known collection, including ones created in a
        previous server run that haven't been touched again yet this
        session. Always includes "default"."""
        response = self._request("GET", "/v1/collections")
        return response.json()["collections"]


def _error_message(response: requests.Response) -> str:
    try:
        return response.json().get("error", response.text)
    except ValueError:
        return response.text or f"HTTP {response.status_code}"


def _encode_binary_batch(records: Sequence[Dict]) -> bytes:
    """Matches the server's `parse_binary_batch` format exactly (see
    src/bin/server.rs): magic + record_count + dim, then per record:
    id (u64 LE) + raw f32 LE vector bytes + metadata_len (u32 LE) +
    UTF-8 JSON metadata. Uses the stdlib `struct` module directly --
    deliberately no numpy dependency, to keep this client's install
    footprint small."""
    import json

    dim = len(records[0]["vector"])
    parts = [_BINARY_MAGIC, struct.pack("<II", len(records), dim)]
    for r in records:
        vector = r["vector"]
        if len(vector) != dim:
            raise BadRequestError(
                f"record {r['id']} has vector length {len(vector)}, expected {dim} "
                "(all records in a batch must share the same dimension)"
            )
        metadata_bytes = json.dumps(r.get("metadata", {})).encode("utf-8")
        parts.append(struct.pack("<Q", int(r["id"])))
        parts.append(struct.pack(f"<{dim}f", *vector))
        parts.append(struct.pack("<I", len(metadata_bytes)))
        parts.append(metadata_bytes)
    return b"".join(parts)
