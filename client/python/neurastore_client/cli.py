"""Command-line interface for NeuraStore, built on the same
NeuraStoreClient used programmatically -- this is a thin wrapper, not a
second implementation of the HTTP logic.

Uses only the standard library's argparse, deliberately -- the client
itself only depends on `requests`, and the CLI shouldn't add a heavier
dependency (like click or typer) just for a somewhat nicer UX. Keeping
the whole package's install footprint small matters more than that.

Usage:
    neurastore health
    neurastore insert --id 1 --vector 0.1,0.2,0.3 --metadata category=docs
    neurastore insert-batch --file records.json
    neurastore get --id 1
    neurastore delete --id 1
    neurastore build-index
    neurastore search --vector 0.1,0.2,0.3 --k 5
    neurastore search-filtered --vector 0.1,0.2,0.3 --field category --value docs
    neurastore stats

Set NEURASTORE_URL to avoid passing --url every time.
"""
import argparse
import json
import os
import sys
from typing import Dict, List

from .client import NeuraStoreClient
from .exceptions import NeuraStoreError

DEFAULT_URL = "http://localhost:8080"


def _parse_vector(s: str) -> List[float]:
    try:
        return [float(x) for x in s.split(",")]
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"invalid vector {s!r} -- expected comma-separated numbers, e.g. 0.1,0.2,0.3"
        )


def _parse_metadata(pairs: List[str]) -> Dict[str, str]:
    metadata = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise argparse.ArgumentTypeError(f"invalid metadata {pair!r} -- expected key=value")
        key, _, value = pair.partition("=")
        metadata[key] = value
    return metadata


def _print_results(results, as_json: bool) -> None:
    if as_json:
        print(json.dumps([{"id": r.id, "distance": r.distance} for r in results], indent=2))
        return
    if not results:
        print("(no results)")
        return
    print(f"{'id':<12}{'distance':>12}")
    for r in results:
        print(f"{r.id:<12}{r.distance:>12.6f}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="neurastore", description="NeuraStore command-line client")
    parser.add_argument(
        "--url",
        default=os.environ.get("NEURASTORE_URL", DEFAULT_URL),
        help=f"server URL (default: {DEFAULT_URL}, or $NEURASTORE_URL)",
    )
    parser.add_argument("--json", action="store_true", help="output machine-readable JSON instead of text")

    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("health", help="check whether the server is reachable")

    p = sub.add_parser("insert", help="insert or update a single record")
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--vector", type=_parse_vector, required=True, help="comma-separated floats, e.g. 0.1,0.2,0.3")
    p.add_argument("--metadata", action="append", metavar="KEY=VALUE", help="repeatable, e.g. --metadata category=docs")

    p = sub.add_parser("insert-batch", help="insert many records from a JSON file")
    p.add_argument(
        "--file",
        required=True,
        help="path to a JSON file containing a list of {id, vector, metadata} objects, or - for stdin",
    )
    p.add_argument("--binary", action="store_true", help="use the binary wire format instead of JSON")

    p = sub.add_parser("get", help="fetch a single record by id")
    p.add_argument("--id", type=int, required=True)

    p = sub.add_parser("delete", help="soft-delete a record by id")
    p.add_argument("--id", type=int, required=True)

    sub.add_parser("build-index", help="build (or rebuild) the vector index")

    p = sub.add_parser("search", help="unfiltered k-NN search")
    p.add_argument("--vector", type=_parse_vector, required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--ef-search", type=int, default=40)

    p = sub.add_parser("search-filtered", help="k-NN search restricted to metadata[field] == value")
    p.add_argument("--vector", type=_parse_vector, required=True)
    p.add_argument("--field", required=True)
    p.add_argument("--value", required=True)
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--ef-search", type=int, default=40)

    sub.add_parser("stats", help="show collection statistics")

    return parser


def _load_batch_file(path: str) -> List[Dict]:
    text = sys.stdin.read() if path == "-" else open(path).read()
    records = json.loads(text)
    if not isinstance(records, list):
        raise ValueError("batch file must contain a JSON array of records")
    return records


def run(args: argparse.Namespace) -> int:
    client = NeuraStoreClient(args.url)
    try:
        if args.command == "health":
            ok = client.health()
            print(json.dumps({"healthy": ok}) if args.json else ("ok" if ok else "unreachable"))
            return 0 if ok else 1

        if args.command == "insert":
            client.insert(args.id, args.vector, metadata=_parse_metadata(args.metadata))
            print(f"inserted id={args.id}")
            return 0

        if args.command == "insert-batch":
            records = _load_batch_file(args.file)
            client.insert_batch(records, binary=args.binary)
            print(f"inserted {len(records)} records")
            return 0

        if args.command == "get":
            record = client.get(args.id)
            if args.json:
                print(json.dumps({"id": record.id, "vector": record.vector, "metadata": record.metadata}, indent=2))
            else:
                print(f"id: {record.id}")
                print(f"vector: {record.vector}")
                print(f"metadata: {record.metadata}")
            return 0

        if args.command == "delete":
            client.delete(args.id)
            print(f"deleted id={args.id}")
            return 0

        if args.command == "build-index":
            client.build_index()
            print("index built")
            return 0

        if args.command == "search":
            results = client.search(args.vector, k=args.k, ef_search=args.ef_search)
            _print_results(results, args.json)
            return 0

        if args.command == "search-filtered":
            results = client.search_filtered(
                args.vector, field=args.field, value=args.value, k=args.k, ef_search=args.ef_search
            )
            _print_results(results, args.json)
            return 0

        if args.command == "stats":
            stats = client.stats()
            if args.json:
                print(json.dumps(stats.__dict__, indent=2))
            else:
                print(f"live_records:     {stats.live_records}")
                print(f"memtable_records: {stats.memtable_records}")
                print(f"sstable_count:    {stats.sstable_count}")
                print(f"index_built:      {stats.index_built}")
                print(f"index_len:        {stats.index_len}")
            return 0

        raise AssertionError(f"unhandled command: {args.command}")  # argparse should prevent this
    except NeuraStoreError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    except (ValueError, OSError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        client.close()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    sys.exit(run(args))


if __name__ == "__main__":
    main()
