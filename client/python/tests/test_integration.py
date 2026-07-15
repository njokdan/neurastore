"""Integration tests against a real, running NeuraStore server -- the
counterpart to test_client.py's mocked unit tests. Skipped automatically
if no server is reachable, so `pytest` still passes in CI/environments
without one running.

To run these for real:
    # from the repo root
    cargo run --release --bin server -- /tmp/neurastore_client_test 8081
    # in another terminal
    cd client/python && pytest tests/test_integration.py
"""
import pytest

from neurastore_client import NeuraStoreClient, NotFoundError

BASE_URL = "http://127.0.0.1:8081"


def _server_available() -> bool:
    try:
        return NeuraStoreClient(BASE_URL, timeout=1.0).health()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _server_available(),
    reason=f"no NeuraStore server reachable at {BASE_URL} -- see module docstring to start one",
)


@pytest.fixture
def client():
    with NeuraStoreClient(BASE_URL) as c:
        yield c


def test_health_check(client):
    assert client.health() is True


def test_insert_get_delete_roundtrip(client):
    client.insert(9001, [1.0, 2.0, 3.0], metadata={"category": "integration-test"})
    record = client.get(9001)
    assert record.vector == [1.0, 2.0, 3.0]
    assert record.metadata["category"] == "integration-test"

    client.delete(9001)
    with pytest.raises(NotFoundError):
        client.get(9001)


def test_batch_insert_build_index_and_search(client):
    client.insert_batch([
        {"id": 9101, "vector": [0.0, 0.0], "metadata": {"category": "docs"}},
        {"id": 9102, "vector": [10.0, 10.0], "metadata": {"category": "code"}},
        {"id": 9103, "vector": [0.1, 0.1], "metadata": {"category": "docs"}},
    ])
    client.build_index()

    results = client.search([0.0, 0.0], k=2, ef_search=20)
    ids = {r.id for r in results}
    assert 9101 in ids
    assert 9103 in ids
    assert 9102 not in ids

    filtered = client.search_filtered([0.0, 0.0], field="category", value="docs", k=5, ef_search=20)
    filtered_ids = {r.id for r in filtered}
    assert 9101 in filtered_ids
    assert 9103 in filtered_ids
    assert 9102 not in filtered_ids


def test_binary_batch_insert_matches_json_insert(client):
    client.insert_batch(
        [{"id": 9201, "vector": [5.0, 5.0], "metadata": {"category": "binary-test"}}],
        binary=True,
    )
    record = client.get(9201)
    assert record.vector == [5.0, 5.0]
    assert record.metadata["category"] == "binary-test"


def test_stats_reflects_inserted_data(client):
    client.insert(9301, [1.0])
    stats = client.stats()
    assert stats.live_records >= 1


def test_compact_reclaims_space_and_keeps_data_correct(client):
    client.insert(9401, [1.0, 1.0], metadata={"category": "docs"})
    client.insert(9401, [2.0, 2.0], metadata={"category": "docs"})  # update, same id
    client.delete(9401)
    client.insert(9402, [3.0, 3.0])
    client.compact()  # should not raise, data must still be correct after

    with pytest.raises(NotFoundError):
        client.get(9401)
    assert client.get(9402).vector == [3.0, 3.0]
