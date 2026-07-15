"""Unit tests for the CLI -- mock the HTTP layer, same approach as
test_client.py. See test_integration.py for the real end-to-end
CLI smoke test against a live server."""
import json

import pytest
import responses

from neurastore_client.cli import build_parser, run

BASE_URL = "http://localhost:8080"


def run_cli(argv, capsys):
    parser = build_parser()
    args = parser.parse_args(["--url", BASE_URL] + argv)
    exit_code = run(args)
    captured = capsys.readouterr()
    return exit_code, captured.out, captured.err


@responses.activate
def test_health_ok(capsys):
    responses.add(responses.GET, f"{BASE_URL}/health", body="ok", status=200)
    code, out, _ = run_cli(["health"], capsys)
    assert code == 0
    assert "ok" in out


@responses.activate
def test_health_unreachable_returns_nonzero(capsys):
    responses.add(
        responses.GET, f"{BASE_URL}/health", body=__import__("requests").exceptions.ConnectionError()
    )
    code, out, _ = run_cli(["health"], capsys)
    assert code == 1
    assert "unreachable" in out


@responses.activate
def test_insert_parses_vector_and_metadata(capsys):
    responses.add(responses.POST, f"{BASE_URL}/v1/records", status=201)
    code, out, _ = run_cli(
        ["insert", "--id", "1", "--vector", "0.1,0.2,0.3", "--metadata", "category=docs"], capsys
    )
    assert code == 0
    assert "inserted id=1" in out

    sent_body = json.loads(responses.calls[0].request.body)
    assert sent_body["vector"] == [0.1, 0.2, 0.3]
    assert sent_body["metadata"] == {"category": "docs"}


@responses.activate
def test_insert_supports_multiple_metadata_pairs(capsys):
    responses.add(responses.POST, f"{BASE_URL}/v1/records", status=201)
    run_cli(
        ["insert", "--id", "1", "--vector", "1.0", "--metadata", "a=1", "--metadata", "b=2"], capsys
    )
    sent_body = json.loads(responses.calls[0].request.body)
    assert sent_body["metadata"] == {"a": "1", "b": "2"}


def test_insert_rejects_malformed_vector(capsys):
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--url", BASE_URL, "insert", "--id", "1", "--vector", "not,numbers"])


@responses.activate
def test_get_prints_record(capsys):
    responses.add(
        responses.GET,
        f"{BASE_URL}/v1/records/1",
        json={"id": 1, "vector": [1.0, 2.0], "metadata": {"category": "docs"}},
        status=200,
    )
    code, out, _ = run_cli(["get", "--id", "1"], capsys)
    assert code == 0
    assert "id: 1" in out
    assert "1.0" in out


@responses.activate
def test_get_missing_record_prints_error_and_returns_nonzero(capsys):
    responses.add(
        responses.GET, f"{BASE_URL}/v1/records/999", json={"error": "record not found"}, status=404
    )
    code, out, err = run_cli(["get", "--id", "999"], capsys)
    assert code == 1
    assert "record not found" in err


@responses.activate
def test_get_json_output(capsys):
    responses.add(
        responses.GET,
        f"{BASE_URL}/v1/records/1",
        json={"id": 1, "vector": [1.0], "metadata": {}},
        status=200,
    )
    parser = build_parser()
    args = parser.parse_args(["--url", BASE_URL, "--json", "get", "--id", "1"])
    code = run(args)
    out = capsys.readouterr().out
    assert code == 0
    parsed = json.loads(out)
    assert parsed["id"] == 1


@responses.activate
def test_delete(capsys):
    responses.add(responses.DELETE, f"{BASE_URL}/v1/records/1", status=204)
    code, out, _ = run_cli(["delete", "--id", "1"], capsys)
    assert code == 0
    assert "deleted id=1" in out


@responses.activate
def test_build_index(capsys):
    responses.add(responses.POST, f"{BASE_URL}/v1/index/build", status=200)
    code, out, _ = run_cli(["build-index"], capsys)
    assert code == 0
    assert "index built" in out


@responses.activate
def test_compact(capsys):
    responses.add(responses.POST, f"{BASE_URL}/v1/compact", status=200)
    code, out, _ = run_cli(["compact"], capsys)
    assert code == 0
    assert "compacted" in out


@responses.activate
def test_search_prints_table(capsys):
    responses.add(
        responses.POST,
        f"{BASE_URL}/v1/search",
        json={"results": [{"id": 1, "distance": 0.0}, {"id": 2, "distance": 1.5}]},
        status=200,
    )
    code, out, _ = run_cli(["search", "--vector", "0.0,0.0", "--k", "2"], capsys)
    assert code == 0
    assert "1" in out and "2" in out


@responses.activate
def test_search_before_index_built_prints_error(capsys):
    responses.add(
        responses.POST,
        f"{BASE_URL}/v1/search",
        json={"error": "index not built yet -- call POST /v1/index/build first"},
        status=400,
    )
    code, out, err = run_cli(["search", "--vector", "0.0,0.0"], capsys)
    assert code == 1
    assert "index not built" in err


@responses.activate
def test_search_filtered_sends_field_and_value(capsys):
    responses.add(responses.POST, f"{BASE_URL}/v1/search/filtered", json={"results": []}, status=200)
    code, out, _ = run_cli(
        ["search-filtered", "--vector", "0.0,0.0", "--field", "category", "--value", "docs"], capsys
    )
    assert code == 0
    sent_body = json.loads(responses.calls[0].request.body)
    assert sent_body["field"] == "category"
    assert sent_body["value"] == "docs"


@responses.activate
def test_stats(capsys):
    responses.add(
        responses.GET,
        f"{BASE_URL}/v1/stats",
        json={
            "live_records": 5,
            "memtable_records": 5,
            "sstable_count": 0,
            "index_built": False,
            "index_len": None,
        },
        status=200,
    )
    code, out, _ = run_cli(["stats"], capsys)
    assert code == 0
    assert "live_records:     5" in out


@responses.activate
def test_api_key_flag_sends_authorization_header(capsys):
    responses.add(
        responses.GET,
        f"{BASE_URL}/v1/stats",
        json={"live_records": 0, "memtable_records": 0, "sstable_count": 0, "index_built": False, "index_len": None},
        status=200,
    )
    parser = build_parser()
    args = parser.parse_args(["--url", BASE_URL, "--api-key", "my-secret-key", "stats"])
    code = run(args)
    assert code == 0
    assert responses.calls[0].request.headers["Authorization"] == "Bearer my-secret-key"


@responses.activate
def test_collection_flag_uses_collection_scoped_url(capsys):
    responses.add(responses.POST, f"{BASE_URL}/v1/collections/my_docs/records", status=201)
    code, out, _ = run_cli(["--collection", "my_docs", "insert", "--id", "1", "--vector", "1.0"], capsys)
    assert code == 0
    assert responses.calls[0].request.url == f"{BASE_URL}/v1/collections/my_docs/records"


@responses.activate
def test_no_collection_flag_uses_default_unprefixed_url(capsys):
    responses.add(responses.POST, f"{BASE_URL}/v1/records", status=201)
    code, out, _ = run_cli(["insert", "--id", "1", "--vector", "1.0"], capsys)
    assert code == 0
    assert responses.calls[0].request.url == f"{BASE_URL}/v1/records"


@responses.activate
def test_collections_subcommand_lists_names(capsys):
    responses.add(responses.GET, f"{BASE_URL}/v1/collections", json={"collections": ["default", "my_docs"]}, status=200)
    code, out, _ = run_cli(["collections"], capsys)
    assert code == 0
    assert "default" in out
    assert "my_docs" in out


@responses.activate
def test_collections_subcommand_json_output(capsys):
    responses.add(responses.GET, f"{BASE_URL}/v1/collections", json={"collections": ["default"]}, status=200)
    parser = build_parser()
    args = parser.parse_args(["--url", BASE_URL, "--json", "collections"])
    code = run(args)
    out = capsys.readouterr().out
    assert code == 0
    assert json.loads(out) == ["default"]


@responses.activate
def test_missing_api_key_against_protected_server_prints_clear_error(capsys):
    responses.add(
        responses.GET,
        f"{BASE_URL}/v1/stats",
        json={"error": "missing or invalid API key -- pass one via 'Authorization: Bearer <key>'"},
        status=401,
    )
    code, out, err = run_cli(["stats"], capsys)
    assert code == 1
    assert "API key" in err


@responses.activate
def test_insert_batch_from_file(capsys, tmp_path):
    responses.add(responses.POST, f"{BASE_URL}/v1/records/batch", status=201)
    batch_file = tmp_path / "records.json"
    batch_file.write_text(json.dumps([
        {"id": 1, "vector": [1.0, 2.0], "metadata": {"category": "docs"}},
        {"id": 2, "vector": [3.0, 4.0]},
    ]))

    code, out, _ = run_cli(["insert-batch", "--file", str(batch_file)], capsys)
    assert code == 0
    assert "inserted 2 records" in out


def test_insert_batch_missing_file_returns_error(capsys):
    code, out, err = run_cli(["insert-batch", "--file", "/nonexistent/path.json"], capsys)
    assert code == 1
    assert "error" in err.lower()


def test_no_command_shows_usage_and_exits_nonzero():
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])
