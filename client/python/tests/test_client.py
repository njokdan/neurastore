"""Unit tests for NeuraStoreClient -- mock the HTTP layer entirely, so
these run without a real NeuraStore server. See test_integration.py for
tests against a real, live server."""
import struct

import pytest
import responses

from neurastore_client import (
    AuthenticationError,
    BadRequestError,
    ConnectionError,
    NeuraStoreClient,
    NotFoundError,
    ServerError,
)

BASE_URL = "http://localhost:8080"


@pytest.fixture
def client():
    with NeuraStoreClient(BASE_URL) as c:
        yield c


@responses.activate
def test_health_true_when_server_responds_ok(client):
    responses.add(responses.GET, f"{BASE_URL}/health", body="ok", status=200)
    assert client.health() is True


@responses.activate
def test_health_false_when_server_unreachable(client):
    responses.add(
        responses.GET, f"{BASE_URL}/health", body=__import__("requests").exceptions.ConnectionError()
    )
    assert client.health() is False


@responses.activate
def test_insert_sends_correct_body(client):
    responses.add(responses.POST, f"{BASE_URL}/v1/records", status=201)
    client.insert(1, [1.0, 2.0, 3.0], metadata={"category": "docs"})

    assert len(responses.calls) == 1
    sent = responses.calls[0].request
    import json

    body = json.loads(sent.body)
    assert body == {"id": 1, "vector": [1.0, 2.0, 3.0], "metadata": {"category": "docs"}}


@responses.activate
def test_insert_defaults_metadata_to_empty_dict(client):
    responses.add(responses.POST, f"{BASE_URL}/v1/records", status=201)
    client.insert(1, [1.0])
    import json

    body = json.loads(responses.calls[0].request.body)
    assert body["metadata"] == {}


@responses.activate
def test_get_returns_record(client):
    responses.add(
        responses.GET,
        f"{BASE_URL}/v1/records/1",
        json={"id": 1, "vector": [1.0, 2.0], "metadata": {"category": "docs"}},
        status=200,
    )
    record = client.get(1)
    assert record.id == 1
    assert record.vector == [1.0, 2.0]
    assert record.metadata == {"category": "docs"}


@responses.activate
def test_get_raises_not_found_error_on_404(client):
    responses.add(
        responses.GET, f"{BASE_URL}/v1/records/999", json={"error": "record not found"}, status=404
    )
    with pytest.raises(NotFoundError, match="record not found"):
        client.get(999)


@responses.activate
def test_delete_sends_delete_request(client):
    responses.add(responses.DELETE, f"{BASE_URL}/v1/records/1", status=204)
    client.delete(1)  # should not raise
    assert responses.calls[0].request.method == "DELETE"


@responses.activate
def test_build_index_posts_to_correct_endpoint(client):
    responses.add(responses.POST, f"{BASE_URL}/v1/index/build", status=200)
    client.build_index()
    assert responses.calls[0].request.url == f"{BASE_URL}/v1/index/build"


@responses.activate
def test_compact_posts_to_correct_endpoint(client):
    responses.add(responses.POST, f"{BASE_URL}/v1/compact", status=200)
    client.compact()
    assert responses.calls[0].request.url == f"{BASE_URL}/v1/compact"


@responses.activate
def test_search_returns_parsed_results(client):
    responses.add(
        responses.POST,
        f"{BASE_URL}/v1/search",
        json={"results": [{"id": 1, "distance": 0.0}, {"id": 3, "distance": 0.02}]},
        status=200,
    )
    results = client.search([0.0, 0.0], k=2, ef_search=20)
    assert len(results) == 2
    assert results[0].id == 1
    assert results[0].distance == 0.0
    assert results[1].id == 3


@responses.activate
def test_search_before_index_built_raises_bad_request(client):
    responses.add(
        responses.POST,
        f"{BASE_URL}/v1/search",
        json={"error": "index not built yet -- call POST /v1/index/build first"},
        status=400,
    )
    with pytest.raises(BadRequestError, match="index not built"):
        client.search([0.0, 0.0])


@responses.activate
def test_search_filtered_sends_field_and_value(client):
    responses.add(responses.POST, f"{BASE_URL}/v1/search/filtered", json={"results": []}, status=200)
    client.search_filtered([0.0, 0.0], field="category", value="docs", k=5)
    import json

    body = json.loads(responses.calls[0].request.body)
    assert body["field"] == "category"
    assert body["value"] == "docs"


@responses.activate
def test_stats_returns_parsed_stats(client):
    responses.add(
        responses.GET,
        f"{BASE_URL}/v1/stats",
        json={
            "live_records": 100,
            "memtable_records": 10,
            "sstable_count": 2,
            "index_built": True,
            "index_len": 100,
        },
        status=200,
    )
    stats = client.stats()
    assert stats.live_records == 100
    assert stats.index_built is True
    assert stats.index_len == 100


@responses.activate
def test_server_error_raises_server_error(client):
    responses.add(
        responses.POST, f"{BASE_URL}/v1/records", json={"error": "internal failure"}, status=500
    )
    with pytest.raises(ServerError, match="internal failure"):
        client.insert(1, [1.0])


@responses.activate
def test_client_sends_authorization_header_when_api_key_given():
    with NeuraStoreClient(BASE_URL, api_key="my-secret-key") as authed_client:
        responses.add(responses.GET, f"{BASE_URL}/v1/stats", json={
            "live_records": 0, "memtable_records": 0, "sstable_count": 0,
            "index_built": False, "index_len": None,
        }, status=200)
        authed_client.stats()
        assert responses.calls[0].request.headers["Authorization"] == "Bearer my-secret-key"


@responses.activate
def test_client_sends_no_authorization_header_without_api_key(client):
    responses.add(responses.GET, f"{BASE_URL}/v1/stats", json={
        "live_records": 0, "memtable_records": 0, "sstable_count": 0,
        "index_built": False, "index_len": None,
    }, status=200)
    client.stats()
    assert "Authorization" not in responses.calls[0].request.headers


@responses.activate
def test_missing_or_wrong_api_key_raises_authentication_error(client):
    responses.add(
        responses.GET,
        f"{BASE_URL}/v1/stats",
        json={"error": "missing or invalid API key -- pass one via 'Authorization: Bearer <key>'"},
        status=401,
    )
    with pytest.raises(AuthenticationError, match="API key"):
        client.stats()


@responses.activate
def test_rate_limit_exceeded_raises_rate_limit_error(client):
    from neurastore_client import RateLimitError
    responses.add(
        responses.GET,
        f"{BASE_URL}/v1/stats",
        json={"error": "rate limit exceeded -- slow down and try again shortly"},
        status=429,
    )
    with pytest.raises(RateLimitError, match="rate limit"):
        client.stats()


def test_connection_error_when_server_unreachable():
    # No responses.activate -- nothing intercepts this, so it hits a
    # real (nonexistent) connection and should surface as our
    # ConnectionError, not requests' raw exception type.
    client = NeuraStoreClient("http://localhost:1")  # port 1 -- nothing listens there
    with pytest.raises(ConnectionError):
        client.insert(1, [1.0])


@responses.activate
def test_insert_batch_empty_raises_bad_request(client):
    with pytest.raises(BadRequestError):
        client.insert_batch([])


@responses.activate
def test_insert_batch_json_sends_all_records(client):
    responses.add(responses.POST, f"{BASE_URL}/v1/records/batch", status=201)
    client.insert_batch([
        {"id": 1, "vector": [1.0, 2.0], "metadata": {"category": "docs"}},
        {"id": 2, "vector": [3.0, 4.0]},
    ])
    import json

    body = json.loads(responses.calls[0].request.body)
    assert len(body["records"]) == 2
    assert body["records"][1]["metadata"] == {}  # defaulted


@responses.activate
def test_insert_batch_binary_uses_octet_stream_and_correct_format(client):
    responses.add(responses.POST, f"{BASE_URL}/v1/records/batch/binary", status=201)
    client.insert_batch(
        [{"id": 1, "vector": [1.0, 2.0, 3.0], "metadata": {"category": "docs"}}],
        binary=True,
    )

    sent = responses.calls[0].request
    assert sent.headers["Content-Type"] == "application/octet-stream"

    body = sent.body
    assert body[0:4] == b"NSBB"
    count, dim = struct.unpack("<II", body[4:12])
    assert count == 1
    assert dim == 3
    record_id = struct.unpack("<Q", body[12:20])[0]
    assert record_id == 1
    vector = struct.unpack("<3f", body[20:32])
    assert vector == (1.0, 2.0, 3.0)


@responses.activate
def test_insert_batch_binary_rejects_mismatched_dimension(client):
    with pytest.raises(BadRequestError, match="dimension|length"):
        client.insert_batch(
            [
                {"id": 1, "vector": [1.0, 2.0, 3.0]},
                {"id": 2, "vector": [1.0, 2.0]},  # wrong dim
            ],
            binary=True,
        )


@responses.activate
def test_insert_with_named_collection_uses_collection_scoped_url(client):
    responses.add(responses.POST, f"{BASE_URL}/v1/collections/my_docs/records", status=201)
    client.insert(1, [1.0], collection="my_docs")
    assert responses.calls[0].request.url == f"{BASE_URL}/v1/collections/my_docs/records"


@responses.activate
def test_get_with_named_collection_uses_collection_scoped_url(client):
    responses.add(
        responses.GET,
        f"{BASE_URL}/v1/collections/my_docs/records/1",
        json={"id": 1, "vector": [1.0], "metadata": {}},
        status=200,
    )
    client.get(1, collection="my_docs")
    assert responses.calls[0].request.url == f"{BASE_URL}/v1/collections/my_docs/records/1"


@responses.activate
def test_default_collection_uses_original_unprefixed_urls(client):
    # Not /v1/collections/default/records -- the original /v1/records,
    # verbatim, so every pre-multi-collection call site (and every test
    # in this file) keeps working with zero changes.
    responses.add(responses.POST, f"{BASE_URL}/v1/records", status=201)
    client.insert(1, [1.0])  # collection not passed -- defaults to "default"
    assert responses.calls[0].request.url == f"{BASE_URL}/v1/records"


@responses.activate
def test_search_and_delete_and_compact_respect_named_collection(client):
    responses.add(responses.POST, f"{BASE_URL}/v1/collections/my_docs/search", json={"results": []}, status=200)
    client.search([1.0], collection="my_docs")
    assert responses.calls[0].request.url == f"{BASE_URL}/v1/collections/my_docs/search"

    responses.add(responses.DELETE, f"{BASE_URL}/v1/collections/my_docs/records/1", status=204)
    client.delete(1, collection="my_docs")
    assert responses.calls[1].request.url == f"{BASE_URL}/v1/collections/my_docs/records/1"

    responses.add(responses.POST, f"{BASE_URL}/v1/collections/my_docs/compact", status=200)
    client.compact(collection="my_docs")
    assert responses.calls[2].request.url == f"{BASE_URL}/v1/collections/my_docs/compact"


@responses.activate
def test_list_collections_returns_names(client):
    responses.add(responses.GET, f"{BASE_URL}/v1/collections", json={"collections": ["default", "my_docs"]}, status=200)
    names = client.list_collections()
    assert names == ["default", "my_docs"]
