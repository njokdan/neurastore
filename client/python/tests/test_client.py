"""Unit tests for NeuraStoreClient -- mock the HTTP layer entirely, so
these run without a real NeuraStore server. See test_integration.py for
tests against a real, live server."""
import struct

import pytest
import responses

from neurastore_client import (
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
