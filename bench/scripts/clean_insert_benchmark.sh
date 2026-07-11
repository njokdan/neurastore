#!/bin/bash
# Runs a clean, methodology-safe comparison of JSON vs binary insert
# throughput: a genuinely fresh server (new process, new empty data
# directory) for EVERY single run, no exceptions -- this is exactly the
# guarantee that was silently broken before (server process reused
# across repeated benchmark calls within the same mode), which is what
# produced the "first call fast, every repeat call slow" pattern: repeat
# calls were quietly paying HNSW index-update cost from the previous
# call's already-built index, not measuring a fresh bulk load at all.
#
# Usage: run this from the neurastore/ repo root.
#   bash bench/scripts/clean_insert_benchmark.sh
#
# Requires the venv activated (numpy, requests, orjson available) and
# the server binary already built (cargo build --release --bin server).

set -e

PORT=8081
RUNS=3
SERVER_BIN="./target/release/server"
SCRIPT_DIR="bench/scripts"

if [ ! -f "$SERVER_BIN" ]; then
    echo "Server binary not found at $SERVER_BIN -- run 'cargo build --release --bin server' first."
    exit 1
fi

run_one() {
    local mode_flag="$1"
    local label="$2"
    local data_dir
    data_dir="/tmp/ns_clean_$(date +%s%N)"

    echo "--- $label: starting fresh server at $data_dir ---"
    "$SERVER_BIN" "$data_dir" "$PORT" > /tmp/ns_clean_server_log.txt 2>&1 &
    local server_pid=$!

    # Wait for the server to actually be up, not a fixed sleep --
    # more reliable than guessing how long startup takes.
    local waited=0
    until curl -s "http://127.0.0.1:$PORT/health" > /dev/null 2>&1; do
        sleep 0.2
        waited=$((waited + 1))
        if [ "$waited" -gt 50 ]; then
            echo "Server never came up -- check /tmp/ns_clean_server_log.txt"
            kill "$server_pid" 2>/dev/null || true
            return 1
        fi
    done

    (cd "$SCRIPT_DIR" && python bench_neurastore_http.py --k 10 --ef-search 40 --port "$PORT" $mode_flag 2>&1 | grep -E "throughput|recall")

    kill "$server_pid" 2>/dev/null || true
    wait "$server_pid" 2>/dev/null || true
    rm -rf "$data_dir"
    echo ""
}

echo "=== JSON path: $RUNS clean runs, fresh server + fresh directory every time ==="
for i in $(seq 1 $RUNS); do
    run_one "" "JSON run $i"
done

echo "=== Binary path: $RUNS clean runs, fresh server + fresh directory every time ==="
for i in $(seq 1 $RUNS); do
    run_one "--binary" "Binary run $i"
done

echo "Done. Compare the throughput lines above -- every single one of these"
echo "was measured against a genuinely fresh, never-before-written-to server,"
echo "so this comparison is finally apples-to-apples."
