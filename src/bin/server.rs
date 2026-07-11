//! NeuraStore's network-facing API (Phase 5).
//!
//! HTTP/JSON, not gRPC -- chosen for accessibility (curl-able, no
//! codegen toolchain needed to try it, easy first client library
//! target) over gRPC's stronger typing and streaming support. Worth
//! revisiting if/when a real client workload needs gRPC's advantages;
//! not a permanent architectural commitment.
//!
//! Concurrency model: the whole `Engine` is wrapped in
//! `Arc<tokio::sync::RwLock<Engine>>`, not a plain `Mutex`. This matters
//! for staying consistent with Phase 3's actual claim (concurrent reads,
//! serialized writes) at the network layer too: search/get/stats
//! handlers take a *read* lock, so multiple search requests can proceed
//! genuinely in parallel; put/delete/build_index handlers take a
//! *write* lock, exclusive with everything else. A plain Mutex would
//! have serialized all requests, including reads against each other --
//! technically simpler, but would have silently thrown away the
//! concurrency property this project spent real effort proving.
//!
//! Scope for Phase 5: single collection per server process (one Engine,
//! one data directory). Multi-collection/multi-tenant support is not
//! implemented -- a real gap, not hidden, and a reasonable place to draw
//! the line for a first network API.

use axum::{
    body::Bytes,
    extract::{Path, State},
    http::StatusCode,
    response::{IntoResponse, Json},
    routing::{delete, get, post},
    Router,
};
use neurastore::record::RecordId;
use neurastore::Engine;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::env;
use std::sync::Arc;
use tokio::sync::RwLock;

type SharedEngine = Arc<RwLock<Engine>>;

// ---------------------------------------------------------------------
// Request/response types
// ---------------------------------------------------------------------

#[derive(Deserialize)]
struct PutRequest {
    id: RecordId,
    vector: Vec<f32>,
    #[serde(default)]
    metadata: HashMap<String, String>,
}

#[derive(Deserialize)]
struct PutBatchRequest {
    records: Vec<PutRequest>,
}

#[derive(Serialize)]
struct RecordResponse {
    id: RecordId,
    vector: Vec<f32>,
    metadata: HashMap<String, String>,
}

#[derive(Deserialize)]
struct SearchRequest {
    vector: Vec<f32>,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default = "default_ef_search")]
    ef_search: usize,
}

#[derive(Deserialize)]
struct FilteredSearchRequest {
    vector: Vec<f32>,
    #[serde(default = "default_k")]
    k: usize,
    #[serde(default = "default_ef_search")]
    ef_search: usize,
    field: String,
    value: String,
}

fn default_k() -> usize {
    10
}
fn default_ef_search() -> usize {
    40
}

#[derive(Serialize)]
struct SearchResultItem {
    id: RecordId,
    distance: f32,
}

#[derive(Serialize)]
struct SearchResponse {
    results: Vec<SearchResultItem>,
}

#[derive(Serialize)]
struct StatsResponse {
    live_records: usize,
    memtable_records: usize,
    sstable_count: usize,
    index_built: bool,
    index_len: Option<usize>,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

/// Uniform error type for handlers -- converts to a JSON error body with
/// an appropriate status code, so every failure path (bad input, engine
/// error, not found) looks the same to a client instead of leaking
/// ad-hoc error shapes per endpoint.
enum ApiError {
    NotFound,
    BadRequest(String),
    Internal(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match self {
            ApiError::NotFound => (StatusCode::NOT_FOUND, "record not found".to_string()),
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
        };
        (status, Json(ErrorResponse { error: message })).into_response()
    }
}

// ---------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------

async fn health() -> &'static str {
    "ok"
}

async fn put_record(
    State(engine): State<SharedEngine>,
    Json(req): Json<PutRequest>,
) -> Result<StatusCode, ApiError> {
    if req.vector.is_empty() {
        return Err(ApiError::BadRequest("vector must not be empty".to_string()));
    }
    let mut engine = engine.write().await;
    engine
        .put(req.id, req.vector, req.metadata)
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    Ok(StatusCode::CREATED)
}

async fn put_batch(
    State(engine): State<SharedEngine>,
    Json(req): Json<PutBatchRequest>,
) -> Result<StatusCode, ApiError> {
    if req.records.is_empty() {
        return Err(ApiError::BadRequest("records must not be empty".to_string()));
    }
    for r in &req.records {
        if r.vector.is_empty() {
            return Err(ApiError::BadRequest(format!("record {} has an empty vector", r.id)));
        }
    }
    let entries: Vec<(RecordId, Vec<f32>, HashMap<String, String>)> =
        req.records.into_iter().map(|r| (r.id, r.vector, r.metadata)).collect();
    let mut engine = engine.write().await;
    engine.put_batch(entries).map_err(|e| ApiError::Internal(e.to_string()))?;
    Ok(StatusCode::CREATED)
}

const BINARY_BATCH_MAGIC: &[u8; 4] = b"NSBB";

#[derive(thiserror::Error, Debug)]
enum BinaryParseError {
    #[error("payload too short to contain a valid header")]
    TooShort,
    #[error("bad magic bytes -- expected NSBB, this isn't a NeuraStore binary batch")]
    BadMagic,
    #[error("payload truncated -- expected more bytes than were present (record {0})")]
    Truncated(usize),
    #[error("metadata for record {0} is not valid UTF-8 JSON: {1}")]
    BadMetadata(usize, String),
}

/// Parses the binary bulk-insert format -- the fix for the JSON tax
/// found in real benchmarking (bench/README.md's Phase 5 section):
/// standard JSON forces every float to be encoded as decimal text on
/// the way in and parsed back out of text on the way out, a real,
/// measured cost on both the client (fixed by switching to orjson) and
/// the server (this endpoint's actual point -- orjson alone only closed
/// part of the gap because the server was still paying serde_json's
/// text-to-float parsing cost regardless of how fast the client
/// produced that text).
///
/// Format:
///   [magic: 4 bytes "NSBB"]
///   [record_count: u32 LE][dim: u32 LE]
///   For each record:
///     [id: u64 LE]
///     [vector: dim * f32 LE, raw bytes -- no text encoding at all]
///     [metadata_len: u32 LE][metadata_len bytes: UTF-8 JSON object]
///
/// Metadata stays as small JSON strings deliberately -- it's a handful
/// of short key-value pairs, negligible in size next to the vector
/// data, and not worth a fully custom binary format for.
fn parse_binary_batch(bytes: &[u8]) -> Result<Vec<(RecordId, Vec<f32>, HashMap<String, String>)>, BinaryParseError> {
    if bytes.len() < 12 {
        return Err(BinaryParseError::TooShort);
    }
    if &bytes[0..4] != BINARY_BATCH_MAGIC {
        return Err(BinaryParseError::BadMagic);
    }
    let record_count = u32::from_le_bytes(bytes[4..8].try_into().unwrap()) as usize;
    let dim = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;

    let mut records = Vec::with_capacity(record_count);
    let mut cursor = 12usize;

    for i in 0..record_count {
        if cursor + 8 > bytes.len() {
            return Err(BinaryParseError::Truncated(i));
        }
        let id = u64::from_le_bytes(bytes[cursor..cursor + 8].try_into().unwrap());
        cursor += 8;

        let vec_bytes = dim * 4;
        if cursor + vec_bytes > bytes.len() {
            return Err(BinaryParseError::Truncated(i));
        }
        let vector: Vec<f32> = bytes[cursor..cursor + vec_bytes]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        cursor += vec_bytes;

        if cursor + 4 > bytes.len() {
            return Err(BinaryParseError::Truncated(i));
        }
        let meta_len = u32::from_le_bytes(bytes[cursor..cursor + 4].try_into().unwrap()) as usize;
        cursor += 4;

        if cursor + meta_len > bytes.len() {
            return Err(BinaryParseError::Truncated(i));
        }
        let metadata: HashMap<String, String> = if meta_len == 0 {
            HashMap::new()
        } else {
            serde_json::from_slice(&bytes[cursor..cursor + meta_len])
                .map_err(|e| BinaryParseError::BadMetadata(i, e.to_string()))?
        };
        cursor += meta_len;

        records.push((id, vector, metadata));
    }

    Ok(records)
}

async fn put_batch_binary(State(engine): State<SharedEngine>, body: Bytes) -> Result<StatusCode, ApiError> {
    let entries = parse_binary_batch(&body).map_err(|e| ApiError::BadRequest(e.to_string()))?;
    if entries.is_empty() {
        return Err(ApiError::BadRequest("records must not be empty".to_string()));
    }
    for (id, vector, _) in &entries {
        if vector.is_empty() {
            return Err(ApiError::BadRequest(format!("record {id} has an empty vector")));
        }
    }
    let mut engine = engine.write().await;
    engine.put_batch(entries).map_err(|e| ApiError::Internal(e.to_string()))?;
    Ok(StatusCode::CREATED)
}

async fn get_record(
    State(engine): State<SharedEngine>,
    Path(id): Path<RecordId>,
) -> Result<Json<RecordResponse>, ApiError> {
    let engine = engine.read().await;
    match engine.get(id) {
        Some(record) => Ok(Json(RecordResponse {
            id: record.id,
            vector: record.vector,
            metadata: record.metadata,
        })),
        None => Err(ApiError::NotFound),
    }
}

async fn delete_record(
    State(engine): State<SharedEngine>,
    Path(id): Path<RecordId>,
) -> Result<StatusCode, ApiError> {
    let mut engine = engine.write().await;
    engine.delete(id).map_err(|e| ApiError::Internal(e.to_string()))?;
    Ok(StatusCode::NO_CONTENT)
}

async fn build_index(State(engine): State<SharedEngine>) -> Result<StatusCode, ApiError> {
    let mut engine = engine.write().await;
    engine.build_index();
    Ok(StatusCode::OK)
}

async fn search(
    State(engine): State<SharedEngine>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    if req.vector.is_empty() {
        return Err(ApiError::BadRequest("vector must not be empty".to_string()));
    }
    let engine = engine.read().await;
    let results = engine
        .search_knn(&req.vector, req.k, req.ef_search)
        .ok_or_else(|| ApiError::BadRequest("index not built yet -- call POST /v1/index/build first".to_string()))?;
    Ok(Json(SearchResponse {
        results: results.into_iter().map(|(id, distance)| SearchResultItem { id, distance }).collect(),
    }))
}

async fn search_filtered(
    State(engine): State<SharedEngine>,
    Json(req): Json<FilteredSearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    if req.vector.is_empty() {
        return Err(ApiError::BadRequest("vector must not be empty".to_string()));
    }
    let engine = engine.read().await;
    let results = engine
        .search_knn_filtered(&req.vector, req.k, req.ef_search, &req.field, &req.value)
        .ok_or_else(|| ApiError::BadRequest("index not built yet -- call POST /v1/index/build first".to_string()))?;
    Ok(Json(SearchResponse {
        results: results.into_iter().map(|(id, distance)| SearchResultItem { id, distance }).collect(),
    }))
}

async fn stats(State(engine): State<SharedEngine>) -> Json<StatsResponse> {
    let engine = engine.read().await;
    Json(StatsResponse {
        live_records: engine.len(),
        memtable_records: engine.memtable_len(),
        sstable_count: engine.sstable_count(),
        index_built: engine.has_index(),
        index_len: engine.index_len(),
    })
}

fn build_router(engine: SharedEngine) -> Router {
    Router::new()
        .route("/health", get(health))
        .route("/v1/records", post(put_record))
        .route("/v1/records/batch", post(put_batch))
        .route("/v1/records/batch/binary", post(put_batch_binary))
        .route("/v1/records/:id", get(get_record))
        .route("/v1/records/:id", delete(delete_record))
        .route("/v1/index/build", post(build_index))
        .route("/v1/search", post(search))
        .route("/v1/search/filtered", post(search_filtered))
        .route("/v1/stats", get(stats))
        .with_state(engine)
        // axum applies a 2MB default request body limit. A batch of
        // 1,000 records at dim=128 (JSON floats, ~1.5-2KB/record) sits
        // right at that ceiling -- real-world batch inserts of
        // higher-dimensional vectors or larger batches would hit this
        // routinely. Raised to 50MB, generous enough for large batches
        // at typical embedding dimensions (up to ~1536 for common
        // OpenAI-style embeddings) without removing the safety limit
        // entirely (unbounded body size is a real DoS vector -- worth
        // keeping in mind once Phase 6's hardening work happens).
        .layer(axum::extract::DefaultBodyLimit::max(50 * 1024 * 1024))
}

#[tokio::main]
async fn main() {
    let args: Vec<String> = env::args().collect();
    let data_dir = args.get(1).cloned().unwrap_or_else(|| "./data".to_string());
    let port: u16 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(8080);

    let engine = Engine::open(&data_dir).expect("failed to open engine");
    println!("NeuraStore server -- data dir: {data_dir}");
    let shared: SharedEngine = Arc::new(RwLock::new(engine));

    let app = build_router(shared);
    let addr = format!("0.0.0.0:{port}");
    let listener = tokio::net::TcpListener::bind(&addr).await.expect("failed to bind");
    println!("Listening on http://{addr}");
    println!("Try: curl http://localhost:{port}/health");

    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .expect("server error");
}

async fn shutdown_signal() {
    tokio::signal::ctrl_c().await.expect("failed to listen for ctrl-c");
    println!("\nShutting down...");
}

#[cfg(test)]
mod tests {
    use super::*;
    use axum::body::Body;
    use axum::http::Request;
    use tower::ServiceExt; // for `oneshot`

    fn test_engine() -> SharedEngine {
        let dir = tempfile::tempdir().unwrap();
        // Leak the tempdir path deliberately -- tests need the directory
        // to outlive the Engine for the test's duration; the OS cleans
        // up temp dirs eventually regardless, and leaking here is
        // simpler than threading a TempDir guard through every test.
        let path = dir.into_path();
        let engine = Engine::open(&path).unwrap();
        Arc::new(RwLock::new(engine))
    }

    async fn body_json(response: axum::response::Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn health_check_returns_ok() {
        let app = build_router(test_engine());
        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn put_then_get_roundtrip() {
        let app = build_router(test_engine());

        let put_body = serde_json::json!({
            "id": 1,
            "vector": [1.0, 2.0, 3.0],
            "metadata": {"category": "docs"}
        });
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/records")
                    .header("content-type", "application/json")
                    .body(Body::from(put_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CREATED);

        let response = app
            .oneshot(Request::builder().uri("/v1/records/1").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert_eq!(json["id"], 1);
        assert_eq!(json["vector"], serde_json::json!([1.0, 2.0, 3.0]));
        assert_eq!(json["metadata"]["category"], "docs");
    }

    #[tokio::test]
    async fn get_nonexistent_record_returns_404() {
        let app = build_router(test_engine());
        let response = app
            .oneshot(Request::builder().uri("/v1/records/999").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn put_with_empty_vector_returns_400() {
        let app = build_router(test_engine());
        let put_body = serde_json::json!({"id": 1, "vector": []});
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/records")
                    .header("content-type", "application/json")
                    .body(Body::from(put_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn delete_then_get_returns_404() {
        let app = build_router(test_engine());

        let put_body = serde_json::json!({"id": 5, "vector": [1.0, 1.0]});
        app.clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/records")
                    .header("content-type", "application/json")
                    .body(Body::from(put_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        let response = app
            .clone()
            .oneshot(Request::builder().method("DELETE").uri("/v1/records/5").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NO_CONTENT);

        let response = app
            .oneshot(Request::builder().uri("/v1/records/5").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn search_without_index_returns_400() {
        let app = build_router(test_engine());
        let search_body = serde_json::json!({"vector": [1.0, 2.0], "k": 5});
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/search")
                    .header("content-type", "application/json")
                    .body(Body::from(search_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn put_batch_build_index_and_search_end_to_end() {
        let app = build_router(test_engine());

        let batch_body = serde_json::json!({
            "records": [
                {"id": 1, "vector": [0.0, 0.0], "metadata": {"category": "docs"}},
                {"id": 2, "vector": [10.0, 10.0], "metadata": {"category": "code"}},
                {"id": 3, "vector": [0.1, 0.1], "metadata": {"category": "docs"}}
            ]
        });
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/records/batch")
                    .header("content-type", "application/json")
                    .body(Body::from(batch_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CREATED);

        let response = app
            .clone()
            .oneshot(Request::builder().method("POST").uri("/v1/index/build").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let search_body = serde_json::json!({"vector": [0.0, 0.0], "k": 2, "ef_search": 20});
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/search")
                    .header("content-type", "application/json")
                    .body(Body::from(search_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        let ids: Vec<i64> = json["results"].as_array().unwrap().iter().map(|r| r["id"].as_i64().unwrap()).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
        assert!(!ids.contains(&2));

        // Filtered search too, end to end.
        let filtered_body = serde_json::json!({
            "vector": [0.0, 0.0], "k": 5, "ef_search": 20, "field": "category", "value": "docs"
        });
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/search/filtered")
                    .header("content-type", "application/json")
                    .body(Body::from(filtered_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        let ids: Vec<i64> = json["results"].as_array().unwrap().iter().map(|r| r["id"].as_i64().unwrap()).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
        assert!(!ids.contains(&2), "id 2 is 'code' category, should be excluded by the filter");
    }

    fn encode_binary_batch(records: &[(u64, Vec<f32>, &str)]) -> Vec<u8> {
        let dim = records.first().map(|(_, v, _)| v.len()).unwrap_or(0);
        let mut buf = Vec::new();
        buf.extend_from_slice(b"NSBB");
        buf.extend_from_slice(&(records.len() as u32).to_le_bytes());
        buf.extend_from_slice(&(dim as u32).to_le_bytes());
        for (id, vector, metadata_json) in records {
            buf.extend_from_slice(&id.to_le_bytes());
            for f in vector {
                buf.extend_from_slice(&f.to_le_bytes());
            }
            let meta_bytes = metadata_json.as_bytes();
            buf.extend_from_slice(&(meta_bytes.len() as u32).to_le_bytes());
            buf.extend_from_slice(meta_bytes);
        }
        buf
    }

    #[tokio::test]
    async fn binary_batch_insert_then_get_roundtrip() {
        let app = build_router(test_engine());
        let body = encode_binary_batch(&[
            (1, vec![1.0, 2.0, 3.0], r#"{"category":"docs"}"#),
            (2, vec![4.0, 5.0, 6.0], r#"{"category":"code"}"#),
        ]);

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/records/batch/binary")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::CREATED);

        let response = app
            .oneshot(Request::builder().uri("/v1/records/1").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert_eq!(json["vector"], serde_json::json!([1.0, 2.0, 3.0]));
        assert_eq!(json["metadata"]["category"], "docs");
    }

    #[tokio::test]
    async fn binary_batch_insert_matches_json_batch_insert_for_search() {
        // The real correctness bar: results from data loaded via the
        // binary endpoint must be indistinguishable from data loaded via
        // the JSON endpoint -- same engine, same VectorIndex underneath,
        // this only changes how bytes cross the wire.
        let app = build_router(test_engine());
        let body = encode_binary_batch(&[
            (1, vec![0.0, 0.0], r#"{"category":"docs"}"#),
            (2, vec![10.0, 10.0], r#"{"category":"code"}"#),
            (3, vec![0.1, 0.1], r#"{"category":"docs"}"#),
        ]);
        app.clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/records/batch/binary")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        app.clone()
            .oneshot(Request::builder().method("POST").uri("/v1/index/build").body(Body::empty()).unwrap())
            .await
            .unwrap();

        let search_body = serde_json::json!({"vector": [0.0, 0.0], "k": 2, "ef_search": 20});
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/search")
                    .header("content-type", "application/json")
                    .body(Body::from(search_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        let ids: Vec<i64> = json["results"].as_array().unwrap().iter().map(|r| r["id"].as_i64().unwrap()).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
        assert!(!ids.contains(&2));
    }

    #[tokio::test]
    async fn binary_batch_insert_rejects_bad_magic() {
        let app = build_router(test_engine());
        let mut body = encode_binary_batch(&[(1, vec![1.0], r#"{}"#)]);
        body[0] = b'X'; // corrupt the magic bytes
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/records/batch/binary")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn binary_batch_insert_rejects_truncated_payload() {
        let app = build_router(test_engine());
        let mut body = encode_binary_batch(&[(1, vec![1.0, 2.0, 3.0], r#"{"category":"docs"}"#)]);
        body.truncate(body.len() - 5); // chop off the tail
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/records/batch/binary")
                    .header("content-type", "application/octet-stream")
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn stats_reflects_engine_state() {
        let app = build_router(test_engine());
        let put_body = serde_json::json!({"id": 1, "vector": [1.0]});
        app.clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/records")
                    .header("content-type", "application/json")
                    .body(Body::from(put_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        let response = app.oneshot(Request::builder().uri("/v1/stats").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert_eq!(json["live_records"], 1);
        assert_eq!(json["index_built"], false);
    }
}
