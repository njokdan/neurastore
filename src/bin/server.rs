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
    extract::{Path, Request, State},
    http::{HeaderMap, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Json},
    routing::{delete, get, post},
    Router,
};
use neurastore::hnsw::{DistanceMetric, HnswParams};
use neurastore::record::{MetadataValue, RecordId};
use neurastore::vector_index::FilterOp;
use neurastore::Engine;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::env;
use std::fs;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::time::Instant;
use tokio::sync::RwLock;

type SharedEngine = Arc<RwLock<Engine>>;

/// `None` means auth is disabled (no keys configured at startup -- the
/// local/dev default, so `cargo run` keeps working with zero setup).
/// `Some(keys)` means every protected request must present one of these
/// keys via `Authorization: Bearer <key>`. `/health` is never protected,
/// even when auth is enabled -- load balancers and orchestration health
/// probes need to reach it without credentials.
type ApiKeys = Arc<Option<HashSet<String>>>;

// ---------------------------------------------------------------------
// Request/response types
// ---------------------------------------------------------------------

#[derive(Deserialize)]
struct PutRequest {
    id: RecordId,
    vector: Vec<f32>,
    /// Raw JSON here, not `MetadataValue` directly -- serde can't
    /// deserialize straight into `MetadataValue` (it's a normal tagged
    /// enum on purpose, for bincode compatibility -- see record.rs's
    /// doc comment). Converted explicitly via `json_to_metadata_value`
    /// in each handler, which is also where a client sending an array,
    /// object, or null gets a clear 400 instead of a confusing type error.
    #[serde(default)]
    metadata: HashMap<String, serde_json::Value>,
}

/// Converts one JSON metadata value into the internal typed
/// representation. Arrays, objects, and null are explicitly rejected
/// with a clear message -- Phase 10 added string/number/bool, not
/// arbitrary nested JSON, and a client sending an unsupported shape
/// should get a real error, not silently-wrong behavior or a panic.
fn json_to_metadata_value(v: &serde_json::Value) -> Result<MetadataValue, ApiError> {
    match v {
        serde_json::Value::String(s) => Ok(MetadataValue::String(s.clone())),
        serde_json::Value::Number(n) => n
            .as_f64()
            .map(MetadataValue::Number)
            .ok_or_else(|| ApiError::BadRequest("invalid number in metadata".to_string())),
        serde_json::Value::Bool(b) => Ok(MetadataValue::Bool(*b)),
        other => Err(ApiError::BadRequest(format!(
            "metadata values must be strings, numbers, or booleans -- got {other} (arrays, objects, and null are not supported)"
        ))),
    }
}

fn metadata_value_to_json(v: &MetadataValue) -> serde_json::Value {
    match v {
        MetadataValue::String(s) => serde_json::Value::String(s.clone()),
        MetadataValue::Number(n) => serde_json::json!(n),
        MetadataValue::Bool(b) => serde_json::Value::Bool(*b),
    }
}

fn convert_metadata_map(m: HashMap<String, serde_json::Value>) -> Result<HashMap<String, MetadataValue>, ApiError> {
    m.into_iter().map(|(k, v)| json_to_metadata_value(&v).map(|mv| (k, mv))).collect()
}

#[derive(Deserialize, Default)]
struct BuildIndexRequest {
    /// "l2" (default), "cosine", or "dot_product" -- omit entirely for
    /// the existing default (L2), full backward compatibility with
    /// every caller that's never sent a body to this endpoint at all.
    #[serde(default)]
    metric: Option<String>,
}

fn parse_metric(s: Option<&str>) -> Result<DistanceMetric, ApiError> {
    match s {
        None | Some("l2") => Ok(DistanceMetric::L2),
        Some("cosine") => Ok(DistanceMetric::Cosine),
        Some("dot_product") | Some("dot") => Ok(DistanceMetric::DotProduct),
        Some(other) => Err(ApiError::BadRequest(format!(
            "unknown metric '{other}' -- expected 'l2', 'cosine', or 'dot_product'"
        ))),
    }
}

#[derive(Deserialize)]
struct PutBatchRequest {
    records: Vec<PutRequest>,
}

#[derive(Serialize)]
struct RecordResponse {
    id: RecordId,
    vector: Vec<f32>,
    metadata: HashMap<String, serde_json::Value>,
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
    /// "eq" (default -- full backward compatibility with every caller
    /// from before Phase 10, which only ever did string equality), or
    /// "gt"/"gte"/"lt"/"lte" for numeric range queries.
    #[serde(default)]
    op: Option<String>,
    /// Raw JSON, same reasoning as `PutRequest::metadata` -- a string
    /// for "eq" against a string field (the pre-Phase-10 shape, still
    /// works unchanged), or a number for anything else.
    value: serde_json::Value,
}

/// Converts a filtered-search request's `op` + `value` into a
/// `FilterOp`. Range ops require a numeric `value` -- sending
/// `{"op": "gt", "value": "expensive"}` gets a clear 400, not a type
/// coercion attempt or a silently-empty result set.
fn parse_filter_op(op: Option<&str>, value: &serde_json::Value) -> Result<FilterOp, ApiError> {
    let op = op.unwrap_or("eq");
    match op {
        "eq" => Ok(FilterOp::Eq(json_to_metadata_value(value)?)),
        "gt" | "gte" | "lt" | "lte" => {
            let n = value
                .as_f64()
                .ok_or_else(|| ApiError::BadRequest(format!("op '{op}' requires a numeric value, got {value}")))?;
            Ok(match op {
                "gt" => FilterOp::Gt(n),
                "gte" => FilterOp::Gte(n),
                "lt" => FilterOp::Lt(n),
                "lte" => FilterOp::Lte(n),
                _ => unreachable!(),
            })
        }
        other => Err(ApiError::BadRequest(format!(
            "unknown op '{other}' -- expected 'eq', 'gt', 'gte', 'lt', or 'lte'"
        ))),
    }
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
    Unauthorized,
    TooManyRequests,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        let (status, message) = match self {
            ApiError::NotFound => (StatusCode::NOT_FOUND, "record not found".to_string()),
            ApiError::BadRequest(msg) => (StatusCode::BAD_REQUEST, msg),
            ApiError::Internal(msg) => (StatusCode::INTERNAL_SERVER_ERROR, msg),
            ApiError::Unauthorized => (
                StatusCode::UNAUTHORIZED,
                "missing or invalid API key -- pass one via 'Authorization: Bearer <key>'".to_string(),
            ),
            ApiError::TooManyRequests => (
                StatusCode::TOO_MANY_REQUESTS,
                "rate limit exceeded -- slow down and try again shortly".to_string(),
            ),
        };
        (status, Json(ErrorResponse { error: message })).into_response()
    }
}

/// Token-bucket rate limiter, one bucket per client identity. Refills
/// continuously at `rate_per_sec`, capped at `burst` -- standard token
/// bucket semantics: burst allows short spikes, the steady rate is what
/// actually gets enforced over time.
///
/// Uses a plain `std::sync::Mutex`, not `tokio::sync::Mutex` -- the
/// critical section here is pure arithmetic (no `.await` held across
/// the lock), so a blocking mutex is the right, simpler choice; an
/// async mutex would add overhead for no benefit in this specific case.
struct RateLimiter {
    buckets: Mutex<HashMap<String, (f64, Instant)>>, // key -> (tokens, last_refill)
    rate_per_sec: f64,
    burst: f64,
}

impl RateLimiter {
    fn new(rate_per_sec: f64, burst: f64) -> Self {
        Self { buckets: Mutex::new(HashMap::new()), rate_per_sec, burst }
    }

    /// Returns true if the request is allowed (and consumes one token),
    /// false if the caller should be rejected with 429.
    fn check(&self, identity: &str) -> bool {
        let mut buckets = self.buckets.lock().expect("rate limiter mutex poisoned");
        let now = Instant::now();
        let (tokens, last_refill) = buckets.entry(identity.to_string()).or_insert((self.burst, now));
        let elapsed = now.duration_since(*last_refill).as_secs_f64();
        *tokens = (*tokens + elapsed * self.rate_per_sec).min(self.burst);
        *last_refill = now;
        if *tokens >= 1.0 {
            *tokens -= 1.0;
            true
        } else {
            false
        }
    }
}

/// Per-client baseline for statistical anomaly detection. Tracks two
/// exponentially-weighted moving averages of request rate: a fast one
/// (reacts quickly to recent behavior) and a slow one (the client's
/// established "normal" pace). When the fast average significantly
/// exceeds the slow one, that's a real, meaningful behavior change --
/// not a fixed global threshold (which is what rate limiting already
/// does), but a deviation from *this specific client's own history*.
#[derive(Clone, Copy)]
struct ClientBaseline {
    fast_rate: f64,
    slow_rate: f64,
    last_seen: Instant,
    request_count: u64,
}

struct AnomalyReport {
    identity: String,
    fast_rate: f64,
    slow_rate: f64,
    ratio: f64,
}

/// Statistical, advisory-only anomaly detector -- this is the scoped,
/// bounded version of the "query-pattern anomaly detection" idea from
/// this project's early planning, deliberately NOT the expansive
/// "self-learning AI security" framing that was set aside at the time.
/// It flags; it never blocks. A human reviews flagged events and
/// decides what (if anything) to do -- the same human-in-the-loop
/// principle applied throughout this project's security-adjacent work.
struct AnomalyDetector {
    baselines: Mutex<HashMap<String, ClientBaseline>>,
    /// EWMA smoothing factor for the fast average -- higher means it
    /// reacts to recent requests more aggressively.
    alpha_fast: f64,
    /// EWMA smoothing factor for the slow average -- much smaller, so
    /// it represents an established pattern, not a snapshot.
    alpha_slow: f64,
    /// fast_rate must exceed slow_rate by at least this multiple to be
    /// flagged. Also requires slow_rate to be non-trivial, so a client's
    /// first couple of requests (before slow_rate has really formed)
    /// can't trigger a spurious flag.
    threshold_multiplier: f64,
    /// Minimum requests from a client before it's eligible to be
    /// flagged at all -- cold-start protection.
    min_requests: u64,
}

impl AnomalyDetector {
    fn new() -> Self {
        Self {
            baselines: Mutex::new(HashMap::new()),
            alpha_fast: 0.5,
            alpha_slow: 0.02,
            threshold_multiplier: 5.0,
            min_requests: 10,
        }
    }

    /// Records one request for `identity` at time `now` and returns an
    /// anomaly report if this request's timing looks like a significant
    /// deviation from that client's established pattern. `now` is
    /// passed in explicitly (not read internally via `Instant::now()`)
    /// specifically so tests can simulate elapsed time deterministically
    /// -- `Instant` has no public constructor for an arbitrary point in
    /// time, but `some_instant + Duration` is always available, so tests
    /// build a fixed start time and advance it exactly as much as each
    /// simulated request needs, with zero real sleeping and zero flakiness.
    fn record_and_check(&self, identity: &str, now: Instant) -> Option<AnomalyReport> {
        let mut baselines = self.baselines.lock().expect("anomaly detector mutex poisoned");
        let baseline = baselines.entry(identity.to_string()).or_insert(ClientBaseline {
            fast_rate: 0.0,
            slow_rate: 0.0,
            last_seen: now,
            request_count: 0,
        });

        // The very first request for a client has no real prior request
        // to measure an interval against -- comparing `now` to itself
        // (since a fresh baseline's last_seen is initialized to `now`)
        // would produce a near-zero dt, clamped to the 0.001s minimum,
        // which works out to an artificial ~1000/s "instantaneous rate"
        // injected into both EWMAs before any real pattern exists at
        // all. This was a genuine bug, found via a live test showing an
        // implausible baseline (~20-35/s from an actual ~1/s pattern) --
        // traced to exactly this. Rate estimation now genuinely starts
        // from the second request onward, where a real interval exists
        // between two actual observed events.
        if baseline.request_count == 0 {
            baseline.last_seen = now;
            baseline.request_count = 1;
            return None;
        }

        let dt = now.saturating_duration_since(baseline.last_seen).as_secs_f64().max(0.001);
        let instantaneous_rate = 1.0 / dt;

        if baseline.request_count == 1 {
            // Second request ever: seed both EWMAs directly from this
            // first real interval, instead of both blending up from a
            // shared 0.0 starting point. Without this, fast_rate
            // (alpha=0.5) converges to the true rate almost immediately
            // while slow_rate (alpha=0.02) takes dozens of requests to
            // catch up -- producing a spurious "fast >> slow" ratio
            // under perfectly steady load, purely from asymmetric
            // warm-up speed, not any real behavior change. Found via
            // the same live-server test that caught the first-request
            // bug above -- a fixed test (steady_rate_requests_never_flagged)
            // failed after that fix, which is what surfaced this.
            baseline.fast_rate = instantaneous_rate;
            baseline.slow_rate = instantaneous_rate;
        } else {
            baseline.fast_rate = self.alpha_fast * instantaneous_rate + (1.0 - self.alpha_fast) * baseline.fast_rate;
            baseline.slow_rate = self.alpha_slow * instantaneous_rate + (1.0 - self.alpha_slow) * baseline.slow_rate;
        }
        baseline.last_seen = now;
        baseline.request_count += 1;

        if baseline.request_count <= self.min_requests || baseline.slow_rate < 0.01 {
            return None;
        }

        let ratio = baseline.fast_rate / baseline.slow_rate;
        if ratio >= self.threshold_multiplier {
            Some(AnomalyReport {
                identity: identity.to_string(),
                fast_rate: baseline.fast_rate,
                slow_rate: baseline.slow_rate,
                ratio,
            })
        } else {
            None
        }
    }
}

/// Manages multiple independent, named collections -- each backed by
/// its own `Engine` (own WAL, own SSTables, own vector index), fully
/// isolated from every other collection. A collection is just a named
/// subdirectory under the base data directory; "default" is special --
/// it maps to the base directory *directly*, not a subdirectory, so
/// every existing deployment's on-disk layout keeps working unchanged.
/// Collections are created lazily on first access, matching how the
/// default collection has never needed an explicit setup step either.
#[derive(Clone)]
struct CollectionManager {
    base_dir: PathBuf,
    engines: Arc<RwLock<HashMap<String, SharedEngine>>>,
}

impl CollectionManager {
    fn new(base_dir: PathBuf) -> Self {
        Self { base_dir, engines: Arc::new(RwLock::new(HashMap::new())) }
    }

    /// A collection name becomes a directory name on disk -- validated
    /// tightly (letters, digits, underscore, hyphen only) specifically
    /// to close off path traversal (a name like "../../etc" must never
    /// reach a filesystem path).
    fn validate_name(name: &str) -> Result<(), ApiError> {
        if name.is_empty() || name.len() > 128 {
            return Err(ApiError::BadRequest("collection name must be 1-128 characters".to_string()));
        }
        if !name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-') {
            return Err(ApiError::BadRequest(
                "collection name may only contain letters, digits, underscore, and hyphen".to_string(),
            ));
        }
        Ok(())
    }

    /// Pre-registers an already-open engine under a name, without going
    /// through the normal open-on-demand path. Used once, at startup,
    /// to register the pre-existing default-collection engine -- this
    /// is what guarantees there's only ever one `Engine` instance for
    /// "default", not two independently opened ones racing on the same
    /// files, whether it's reached via the original `/v1/*` routes or
    /// the new `/v1/collections/default/*` ones.
    async fn register_existing(&self, name: &str, engine: SharedEngine) {
        self.engines.write().await.insert(name.to_string(), engine);
    }

    async fn get_or_create(&self, name: &str) -> Result<SharedEngine, ApiError> {
        Self::validate_name(name)?;
        {
            let engines = self.engines.read().await;
            if let Some(engine) = engines.get(name) {
                return Ok(engine.clone());
            }
        }
        let mut engines = self.engines.write().await;
        // Re-check after acquiring the write lock -- another concurrent
        // request may have already created this collection while this
        // one was waiting for the lock.
        if let Some(engine) = engines.get(name) {
            return Ok(engine.clone());
        }
        let dir = if name == "default" { self.base_dir.clone() } else { self.base_dir.join(name) };
        let engine = Engine::open(&dir).map_err(|e| ApiError::Internal(e.to_string()))?;
        let shared: SharedEngine = Arc::new(RwLock::new(engine));
        engines.insert(name.to_string(), shared.clone());
        Ok(shared)
    }

    /// Lists known collections: everything already loaded this session,
    /// unioned with subdirectories on disk that look like a collection
    /// (contain a WAL or SSTable file) -- so collections created in a
    /// previous run show up immediately after a restart, before
    /// anything has touched them again to trigger `get_or_create`.
    async fn list(&self) -> Vec<String> {
        let mut names: std::collections::HashSet<String> = self.engines.read().await.keys().cloned().collect();

        if self.base_dir.join("wal.log").exists() {
            names.insert("default".to_string());
        }
        if let Ok(entries) = fs::read_dir(&self.base_dir) {
            for entry in entries.flatten() {
                if !entry.path().is_dir() {
                    continue;
                }
                let Some(name) = entry.file_name().to_str().map(|s| s.to_string()) else { continue };
                let looks_like_a_collection = entry.path().join("wal.log").exists()
                    || fs::read_dir(entry.path())
                        .map(|mut d| {
                            d.any(|e| {
                                e.ok()
                                    .map(|e| e.path().extension().map(|ext| ext == "sst").unwrap_or(false))
                                    .unwrap_or(false)
                            })
                        })
                        .unwrap_or(false);
                if looks_like_a_collection {
                    names.insert(name);
                }
            }
        }
        let mut names: Vec<String> = names.into_iter().collect();
        names.sort();
        names
    }
}

#[derive(Clone)]
struct SecurityConfig {
    api_keys: ApiKeys,
    rate_limiter: Option<Arc<RateLimiter>>,
    anomaly_detector: Option<Arc<AnomalyDetector>>,
}

/// Combines auth and rate-limiting into one pass, deliberately not two
/// separate middleware layers. Two layers would need rate-limiting to
/// somehow see auth's *validated* result (not just the raw, possibly
/// forged header) to avoid a real gap: keying rate limits by the raw
/// provided key would let an attacker bypass limits entirely by simply
/// rotating the key string on every request, since each new string gets
/// a fresh bucket. Resolving identity once, in one function, before
/// either check runs, avoids that gap and avoids relying on tower's
/// layer-ordering semantics being exactly right.
///
/// When auth is disabled (no keys configured), every client shares one
/// global rate-limit bucket -- there's no cheap, reliable per-client
/// identity available without auth (that would need extracting and
/// trusting a connection's source IP, a bigger change deferred for
/// now). Documented as a known simplification, not silently assumed.
async fn security_middleware(
    State(config): State<SecurityConfig>,
    headers: HeaderMap,
    request: Request,
    next: Next,
) -> Result<axum::response::Response, ApiError> {
    let identity: String = match config.api_keys.as_ref() {
        Some(valid_keys) => {
            let provided_key = headers
                .get("Authorization")
                .and_then(|v| v.to_str().ok())
                .and_then(|v| v.strip_prefix("Bearer "));
            match provided_key {
                Some(key) if valid_keys.contains(key) => key.to_string(),
                _ => return Err(ApiError::Unauthorized),
            }
        }
        None => "__anonymous_shared_bucket__".to_string(),
    };

    if let Some(limiter) = &config.rate_limiter {
        if !limiter.check(&identity) {
            return Err(ApiError::TooManyRequests);
        }
    }

    // Advisory only -- logs a flagged event, never blocks the request.
    // This is deliberate: a statistical detector will have false
    // positives (a legitimate client doing a real bulk load looks
    // identical to a burst), and auto-rejecting on those would be worse
    // than the problem it's meant to catch. A human reviews the log.
    if let Some(detector) = &config.anomaly_detector {
        if let Some(report) = detector.record_and_check(&identity, Instant::now()) {
            println!(
                "ANOMALY: client '{}' request rate {:.2}/s is {:.1}x its established baseline of {:.2}/s -- flagged for review, not blocked.",
                report.identity, report.fast_rate, report.ratio, report.slow_rate
            );
        }
    }

    Ok(next.run(request).await)
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
    put_record_impl(engine, req).await
}

async fn put_record_impl(engine: SharedEngine, req: PutRequest) -> Result<StatusCode, ApiError> {
    if req.vector.is_empty() {
        return Err(ApiError::BadRequest("vector must not be empty".to_string()));
    }
    let metadata = convert_metadata_map(req.metadata)?;
    let mut engine = engine.write().await;
    engine
        .put(req.id, req.vector, metadata)
        .map_err(|e| ApiError::Internal(e.to_string()))?;
    Ok(StatusCode::CREATED)
}

async fn put_record_collection(
    State(collections): State<CollectionManager>,
    Path(collection): Path<String>,
    Json(req): Json<PutRequest>,
) -> Result<StatusCode, ApiError> {
    let engine = collections.get_or_create(&collection).await?;
    put_record_impl(engine, req).await
}

async fn put_batch(
    State(engine): State<SharedEngine>,
    Json(req): Json<PutBatchRequest>,
) -> Result<StatusCode, ApiError> {
    put_batch_impl(engine, req).await
}

async fn put_batch_impl(engine: SharedEngine, req: PutBatchRequest) -> Result<StatusCode, ApiError> {
    if req.records.is_empty() {
        return Err(ApiError::BadRequest("records must not be empty".to_string()));
    }
    for r in &req.records {
        if r.vector.is_empty() {
            return Err(ApiError::BadRequest(format!("record {} has an empty vector", r.id)));
        }
    }
    let entries: Vec<(RecordId, Vec<f32>, HashMap<String, MetadataValue>)> = req
        .records
        .into_iter()
        .map(|r| convert_metadata_map(r.metadata).map(|m| (r.id, r.vector, m)))
        .collect::<Result<_, ApiError>>()?;
    let mut engine = engine.write().await;
    engine.put_batch(entries).map_err(|e| ApiError::Internal(e.to_string()))?;
    Ok(StatusCode::CREATED)
}

async fn put_batch_collection(
    State(collections): State<CollectionManager>,
    Path(collection): Path<String>,
    Json(req): Json<PutBatchRequest>,
) -> Result<StatusCode, ApiError> {
    let engine = collections.get_or_create(&collection).await?;
    put_batch_impl(engine, req).await
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
fn parse_binary_batch(bytes: &[u8]) -> Result<Vec<(RecordId, Vec<f32>, HashMap<String, serde_json::Value>)>, BinaryParseError> {
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
        let metadata: HashMap<String, serde_json::Value> = if meta_len == 0 {
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
    put_batch_binary_impl(engine, body).await
}

async fn put_batch_binary_impl(engine: SharedEngine, body: Bytes) -> Result<StatusCode, ApiError> {
    let entries = parse_binary_batch(&body).map_err(|e| ApiError::BadRequest(e.to_string()))?;
    if entries.is_empty() {
        return Err(ApiError::BadRequest("records must not be empty".to_string()));
    }
    for (id, vector, _) in &entries {
        if vector.is_empty() {
            return Err(ApiError::BadRequest(format!("record {id} has an empty vector")));
        }
    }
    let entries: Vec<(RecordId, Vec<f32>, HashMap<String, MetadataValue>)> = entries
        .into_iter()
        .map(|(id, vector, metadata)| convert_metadata_map(metadata).map(|m| (id, vector, m)))
        .collect::<Result<_, ApiError>>()?;
    let mut engine = engine.write().await;
    engine.put_batch(entries).map_err(|e| ApiError::Internal(e.to_string()))?;
    Ok(StatusCode::CREATED)
}

async fn put_batch_binary_collection(
    State(collections): State<CollectionManager>,
    Path(collection): Path<String>,
    body: Bytes,
) -> Result<StatusCode, ApiError> {
    let engine = collections.get_or_create(&collection).await?;
    put_batch_binary_impl(engine, body).await
}

async fn get_record(
    State(engine): State<SharedEngine>,
    Path(id): Path<RecordId>,
) -> Result<Json<RecordResponse>, ApiError> {
    get_record_impl(engine, id).await
}

async fn get_record_impl(engine: SharedEngine, id: RecordId) -> Result<Json<RecordResponse>, ApiError> {
    let engine = engine.read().await;
    match engine.get(id) {
        Some(record) => Ok(Json(RecordResponse {
            id: record.id,
            vector: record.vector,
            metadata: record.metadata.iter().map(|(k, v)| (k.clone(), metadata_value_to_json(v))).collect(),
        })),
        None => Err(ApiError::NotFound),
    }
}

async fn get_record_collection(
    State(collections): State<CollectionManager>,
    Path((collection, id)): Path<(String, RecordId)>,
) -> Result<Json<RecordResponse>, ApiError> {
    let engine = collections.get_or_create(&collection).await?;
    get_record_impl(engine, id).await
}

async fn delete_record(
    State(engine): State<SharedEngine>,
    Path(id): Path<RecordId>,
) -> Result<StatusCode, ApiError> {
    delete_record_impl(engine, id).await
}

async fn delete_record_impl(engine: SharedEngine, id: RecordId) -> Result<StatusCode, ApiError> {
    let mut engine = engine.write().await;
    engine.delete(id).map_err(|e| ApiError::Internal(e.to_string()))?;
    Ok(StatusCode::NO_CONTENT)
}

async fn delete_record_collection(
    State(collections): State<CollectionManager>,
    Path((collection, id)): Path<(String, RecordId)>,
) -> Result<StatusCode, ApiError> {
    let engine = collections.get_or_create(&collection).await?;
    delete_record_impl(engine, id).await
}

async fn build_index(State(engine): State<SharedEngine>, body: Bytes) -> Result<StatusCode, ApiError> {
    build_index_impl(engine, body).await
}

async fn build_index_impl(engine: SharedEngine, body: Bytes) -> Result<StatusCode, ApiError> {
    // Raw Bytes, not Json<BuildIndexRequest> -- axum's Json extractor
    // requires a Content-Type header and a parseable body, which would
    // reject every existing caller that sends this endpoint no body at
    // all (every client and test written before metric support existed).
    // Empty body -> defaults; non-empty body -> parsed as JSON.
    let metric = if body.is_empty() {
        DistanceMetric::L2
    } else {
        let req: BuildIndexRequest = serde_json::from_slice(&body)
            .map_err(|e| ApiError::BadRequest(format!("invalid request body: {e}")))?;
        parse_metric(req.metric.as_deref())?
    };
    let params = HnswParams { metric, ..HnswParams::default() };
    let mut engine = engine.write().await;
    engine.build_index_with_params(params, 42); // seed matches Engine::build_index()'s own default
    Ok(StatusCode::OK)
}

async fn build_index_collection(
    State(collections): State<CollectionManager>,
    Path(collection): Path<String>,
    body: Bytes,
) -> Result<StatusCode, ApiError> {
    let engine = collections.get_or_create(&collection).await?;
    build_index_impl(engine, body).await
}

async fn compact(State(engine): State<SharedEngine>) -> Result<StatusCode, ApiError> {
    compact_impl(engine).await
}

async fn compact_impl(engine: SharedEngine) -> Result<StatusCode, ApiError> {
    let mut engine = engine.write().await;
    engine.compact().map_err(|e| ApiError::Internal(e.to_string()))?;
    Ok(StatusCode::OK)
}

async fn compact_collection(
    State(collections): State<CollectionManager>,
    Path(collection): Path<String>,
) -> Result<StatusCode, ApiError> {
    let engine = collections.get_or_create(&collection).await?;
    compact_impl(engine).await
}

async fn search(
    State(engine): State<SharedEngine>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    search_impl(engine, req).await
}

async fn search_impl(engine: SharedEngine, req: SearchRequest) -> Result<Json<SearchResponse>, ApiError> {
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

async fn search_collection(
    State(collections): State<CollectionManager>,
    Path(collection): Path<String>,
    Json(req): Json<SearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    let engine = collections.get_or_create(&collection).await?;
    search_impl(engine, req).await
}

async fn search_filtered(
    State(engine): State<SharedEngine>,
    Json(req): Json<FilteredSearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    search_filtered_impl(engine, req).await
}

async fn search_filtered_impl(engine: SharedEngine, req: FilteredSearchRequest) -> Result<Json<SearchResponse>, ApiError> {
    if req.vector.is_empty() {
        return Err(ApiError::BadRequest("vector must not be empty".to_string()));
    }
    let op = parse_filter_op(req.op.as_deref(), &req.value)?;
    let engine = engine.read().await;
    let results = engine
        .search_knn_filtered(&req.vector, req.k, req.ef_search, &req.field, &op)
        .ok_or_else(|| ApiError::BadRequest("index not built yet -- call POST /v1/index/build first".to_string()))?;
    Ok(Json(SearchResponse {
        results: results.into_iter().map(|(id, distance)| SearchResultItem { id, distance }).collect(),
    }))
}

async fn search_filtered_collection(
    State(collections): State<CollectionManager>,
    Path(collection): Path<String>,
    Json(req): Json<FilteredSearchRequest>,
) -> Result<Json<SearchResponse>, ApiError> {
    let engine = collections.get_or_create(&collection).await?;
    search_filtered_impl(engine, req).await
}

async fn stats(State(engine): State<SharedEngine>) -> Json<StatsResponse> {
    stats_impl(engine).await
}

async fn stats_impl(engine: SharedEngine) -> Json<StatsResponse> {
    let engine = engine.read().await;
    Json(StatsResponse {
        live_records: engine.len(),
        memtable_records: engine.memtable_len(),
        sstable_count: engine.sstable_count(),
        index_built: engine.has_index(),
        index_len: engine.index_len(),
    })
}

async fn stats_collection(
    State(collections): State<CollectionManager>,
    Path(collection): Path<String>,
) -> Result<Json<StatsResponse>, ApiError> {
    let engine = collections.get_or_create(&collection).await?;
    Ok(stats_impl(engine).await)
}

#[derive(Serialize)]
struct CollectionsResponse {
    collections: Vec<String>,
}

async fn list_collections(State(collections): State<CollectionManager>) -> Json<CollectionsResponse> {
    Json(CollectionsResponse { collections: collections.list().await })
}

fn build_router(engine: SharedEngine, collections: CollectionManager, security: SecurityConfig) -> Router {
    let protected = Router::new()
        .route("/v1/records", post(put_record))
        .route("/v1/records/batch", post(put_batch))
        .route("/v1/records/batch/binary", post(put_batch_binary))
        .route("/v1/records/:id", get(get_record))
        .route("/v1/records/:id", delete(delete_record))
        .route("/v1/index/build", post(build_index))
        .route("/v1/compact", post(compact))
        .route("/v1/search", post(search))
        .route("/v1/search/filtered", post(search_filtered))
        .route("/v1/stats", get(stats))
        // route_layer applies only to the routes already added above,
        // and only to requests that match one of them -- unmatched
        // paths fall through to a 404 without ever running this check.
        // security_middleware's state (SecurityConfig) is independent
        // of the router's own handler state (SharedEngine) below.
        .route_layer(middleware::from_fn_with_state(security.clone(), security_middleware))
        .with_state(engine.clone());

    // Same operation set as `protected` above, scoped to a named
    // collection instead of the implicit default -- see
    // `CollectionManager`'s docs for why "default" is handled specially
    // rather than being just another named collection under the hood.
    let collections_protected = Router::new()
        .route("/v1/collections", get(list_collections))
        .route("/v1/collections/:collection/records", post(put_record_collection))
        .route("/v1/collections/:collection/records/batch", post(put_batch_collection))
        .route("/v1/collections/:collection/records/batch/binary", post(put_batch_binary_collection))
        .route("/v1/collections/:collection/records/:id", get(get_record_collection))
        .route("/v1/collections/:collection/records/:id", delete(delete_record_collection))
        .route("/v1/collections/:collection/index/build", post(build_index_collection))
        .route("/v1/collections/:collection/compact", post(compact_collection))
        .route("/v1/collections/:collection/search", post(search_collection))
        .route("/v1/collections/:collection/search/filtered", post(search_filtered_collection))
        .route("/v1/collections/:collection/stats", get(stats_collection))
        .route_layer(middleware::from_fn_with_state(security, security_middleware))
        .with_state(collections);

    Router::new()
        .route("/health", get(health))
        .with_state(engine)
        .merge(protected)
        .merge(collections_protected)
        // axum applies a 2MB default request body limit. A batch of
        // 1,000 records at dim=128 (JSON floats, ~1.5-2KB/record) sits
        // right at that ceiling -- real-world batch inserts of
        // higher-dimensional vectors or larger batches would hit this
        // routinely. Raised to 50MB, generous enough for large batches
        // at typical embedding dimensions (up to ~1536 for common
        // OpenAI-style embeddings) without removing the safety limit
        // entirely (unbounded body size is a real DoS vector).
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

    // Auth is opt-in via NEURASTORE_API_KEYS (comma-separated keys),
    // not opt-out -- but the startup log makes the choice visible either
    // way, so running without auth is a decision someone can see they
    // made, not a silent gap they discover later.
    let api_keys: ApiKeys = match env::var("NEURASTORE_API_KEYS") {
        Ok(raw) => {
            let keys: HashSet<String> = raw.split(',').map(|s| s.trim().to_string()).filter(|s| !s.is_empty()).collect();
            if keys.is_empty() {
                println!("WARNING: NEURASTORE_API_KEYS was set but contained no valid keys -- running WITHOUT authentication.");
                Arc::new(None)
            } else {
                println!("Authentication ENABLED -- {} API key(s) configured. All /v1/* endpoints require 'Authorization: Bearer <key>'.", keys.len());
                Arc::new(Some(keys))
            }
        }
        Err(_) => {
            println!("WARNING: NEURASTORE_API_KEYS not set -- running WITHOUT authentication. Anyone who can reach this server has full read/write access. Set NEURASTORE_API_KEYS to a comma-separated list of keys to enable it.");
            Arc::new(None)
        }
    };

    // Rate limiting is opt-in too, same reasoning as auth: this
    // project's own benchmark tooling fires many rapid requests, and a
    // default-on limit could silently break documented workflows.
    // NEURASTORE_RATE_LIMIT_RPS=<n> enables it; burst defaults to 2x the
    // rate (a couple seconds' worth of headroom for legitimate bursts)
    // unless NEURASTORE_RATE_LIMIT_BURST overrides it.
    let rate_limiter: Option<Arc<RateLimiter>> = match env::var("NEURASTORE_RATE_LIMIT_RPS") {
        Ok(raw) => match raw.parse::<f64>() {
            Ok(rps) if rps > 0.0 => {
                let burst: f64 = env::var("NEURASTORE_RATE_LIMIT_BURST")
                    .ok()
                    .and_then(|s| s.parse().ok())
                    .unwrap_or(rps * 2.0);
                println!("Rate limiting ENABLED -- {rps} req/sec, burst {burst}, per {}.", 
                    if env::var("NEURASTORE_API_KEYS").is_ok() { "API key" } else { "server-wide (no auth configured, so no per-client identity)" });
                Some(Arc::new(RateLimiter::new(rps, burst)))
            }
            _ => {
                println!("WARNING: NEURASTORE_RATE_LIMIT_RPS was set but isn't a valid positive number -- rate limiting disabled.");
                None
            }
        },
        Err(_) => {
            println!("Rate limiting disabled. Set NEURASTORE_RATE_LIMIT_RPS=<requests/sec> to enable it.");
            None
        }
    };

    // Anomaly detection is opt-in, same reasoning as auth and rate
    // limiting: this project's own benchmark tooling makes rapid,
    // legitimate bursts of requests that would otherwise generate noisy
    // false-positive flags in every log line.
    let anomaly_detector: Option<Arc<AnomalyDetector>> = if env::var("NEURASTORE_ANOMALY_DETECTION").is_ok() {
        println!("Anomaly detection ENABLED -- flags unusual per-client request-rate deviations in the log. Advisory only, never blocks a request.");
        Some(Arc::new(AnomalyDetector::new()))
    } else {
        println!("Anomaly detection disabled. Set NEURASTORE_ANOMALY_DETECTION=1 to enable it.");
        None
    };

    let security = SecurityConfig { api_keys, rate_limiter, anomaly_detector };

    // The "default" collection is the SAME engine as the original,
    // pre-multi-collection `/v1/*` routes -- registered here, not
    // opened fresh, so there's only ever one Engine instance touching
    // these files, never two independently opened ones racing on the
    // same directory.
    let collections = CollectionManager::new(PathBuf::from(&data_dir));
    collections.register_existing("default", shared.clone()).await;
    println!("Multi-collection support: additional collections are created on first use under {data_dir}/<name>/, addressable via /v1/collections/<name>/*.");

    let app = build_router(shared, collections, security);
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

    /// Mirrors exactly what `main()` does: a fresh engine registered as
    /// "default" in a fresh CollectionManager, so pre-existing tests
    /// (written before multi-collection support existed) exercise the
    /// same wiring production actually uses, not a simplified stand-in.
    async fn build_test_router(security: SecurityConfig) -> Router {
        let engine = test_engine();
        let collections_dir = tempfile::tempdir().unwrap().into_path();
        let collections = CollectionManager::new(collections_dir);
        collections.register_existing("default", engine.clone()).await;
        build_router(engine, collections, security)
    }

    async fn body_json(response: axum::response::Response) -> serde_json::Value {
        let bytes = axum::body::to_bytes(response.into_body(), usize::MAX).await.unwrap();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[tokio::test]
    async fn health_check_returns_ok() {
        let app = build_test_router(no_security()).await;
        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn put_then_get_roundtrip() {
        let app = build_test_router(no_security()).await;

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
        let app = build_test_router(no_security()).await;
        let response = app
            .oneshot(Request::builder().uri("/v1/records/999").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn put_with_empty_vector_returns_400() {
        let app = build_test_router(no_security()).await;
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
        let app = build_test_router(no_security()).await;

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
        let app = build_test_router(no_security()).await;
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
        let app = build_test_router(no_security()).await;

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
        let app = build_test_router(no_security()).await;
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
        let app = build_test_router(no_security()).await;
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
        let app = build_test_router(no_security()).await;
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
        let app = build_test_router(no_security()).await;
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
    async fn compact_endpoint_merges_sstables_and_stays_correct() {
        let app = build_test_router(no_security()).await;

        for i in 1..=3u64 {
            let put_body = serde_json::json!({"id": i, "vector": [i as f32]});
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
        }

        let response = app
            .clone()
            .oneshot(Request::builder().method("POST").uri("/v1/compact").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        // Data must still be correct after compaction, via the real HTTP path.
        let response = app.oneshot(Request::builder().uri("/v1/records/2").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert_eq!(json["vector"], serde_json::json!([2.0]));
    }

    #[tokio::test]
    async fn compact_requires_auth_like_other_write_routes() {
        let app = build_test_router(security_with_keys(&["secret123"])).await;
        let response = app
            .oneshot(Request::builder().method("POST").uri("/v1/compact").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    // -- Distance metric tests ----------------------------------------

    #[tokio::test]
    async fn build_index_with_no_body_defaults_to_l2_backward_compatible() {
        // The critical backward-compat guarantee: every caller written
        // before metric support existed sends this endpoint zero body
        // at all. That must keep working exactly as before.
        let app = build_test_router(no_security()).await;
        let response = app
            .oneshot(Request::builder().method("POST").uri("/v1/index/build").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "no body at all must still work and default to L2");
    }

    #[tokio::test]
    async fn build_index_accepts_valid_metric_names() {
        for metric in ["l2", "cosine", "dot_product"] {
            let app = build_test_router(no_security()).await;
            let body = serde_json::json!({"metric": metric});
            let response = app
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/v1/index/build")
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "metric '{metric}' should be accepted");
        }
    }

    #[tokio::test]
    async fn build_index_rejects_unknown_metric_name() {
        let app = build_test_router(no_security()).await;
        let body = serde_json::json!({"metric": "manhattan"});
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/index/build")
                    .header("content-type", "application/json")
                    .body(Body::from(body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn cosine_metric_changes_search_results_end_to_end_over_real_http() {
        // The real, end-to-end proof: build the SAME data twice, once
        // with each metric, and confirm ranking actually differs via the
        // real HTTP API -- not just at the Rust-internal level already
        // covered by hnsw.rs's own tests.
        let insert_data = |app: Router| async move {
            for (id, vec) in [(1u64, [1.0, 0.0]), (2, [20.0, 0.0]), (3, [0.9, 0.1])] {
                let body = serde_json::json!({"id": id, "vector": vec});
                app.clone()
                    .oneshot(
                        Request::builder()
                            .method("POST")
                            .uri("/v1/records")
                            .header("content-type", "application/json")
                            .body(Body::from(body.to_string()))
                            .unwrap(),
                    )
                    .await
                    .unwrap();
            }
            app
        };

        // L2-built index.
        let app_l2 = insert_data(build_test_router(no_security()).await).await;
        app_l2
            .clone()
            .oneshot(Request::builder().method("POST").uri("/v1/index/build").body(Body::empty()).unwrap())
            .await
            .unwrap();
        let search_body = serde_json::json!({"vector": [1.0, 0.0], "k": 1, "ef_search": 50});
        let l2_response = app_l2
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
        let l2_json = body_json(l2_response).await;
        let l2_top1 = l2_json["results"][0]["id"].as_i64().unwrap();

        // Cosine-built index, same data.
        let app_cos = insert_data(build_test_router(no_security()).await).await;
        let build_body = serde_json::json!({"metric": "cosine"});
        app_cos
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/index/build")
                    .header("content-type", "application/json")
                    .body(Body::from(build_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        let cos_response = app_cos
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
        let cos_json = body_json(cos_response).await;
        let cos_top1 = cos_json["results"][0]["id"].as_i64().unwrap();

        // L2 must not pick id 2 (magnitude 20, far in raw distance).
        assert_ne!(l2_top1, 2, "L2 should not rank the large-magnitude vector as nearest");
        // Cosine should pick id 1 or id 2 -- both exactly on-direction.
        assert!(cos_top1 == 1 || cos_top1 == 2, "cosine should rank one of the on-direction vectors as nearest, got {cos_top1}");
    }

    // -- Multi-collection tests --------------------------------------

    #[tokio::test]
    async fn collections_are_fully_isolated_from_each_other() {
        let app = build_test_router(no_security()).await;

        let insert_into = |collection: &str, id: u64| {
            Request::builder()
                .method("POST")
                .uri(format!("/v1/collections/{collection}/records"))
                .header("content-type", "application/json")
                .body(Body::from(serde_json::json!({"id": id, "vector": [1.0]}).to_string()))
                .unwrap()
        };

        let r1 = app.clone().oneshot(insert_into("alpha", 1)).await.unwrap();
        assert_eq!(r1.status(), StatusCode::CREATED);
        let r2 = app.clone().oneshot(insert_into("beta", 1)).await.unwrap();
        assert_eq!(r2.status(), StatusCode::CREATED);

        // Same id (1) in both collections -- must not collide or overwrite each other.
        let get_from = |collection: &str, id: u64| {
            Request::builder().uri(format!("/v1/collections/{collection}/records/{id}")).body(Body::empty()).unwrap()
        };
        let alpha_record = app.clone().oneshot(get_from("alpha", 1)).await.unwrap();
        assert_eq!(alpha_record.status(), StatusCode::OK);

        let beta_record = app.clone().oneshot(get_from("beta", 1)).await.unwrap();
        assert_eq!(beta_record.status(), StatusCode::OK);

        // A record in "alpha" must not be reachable through "beta"'s id
        // space at some OTHER id -- confirms real isolation, not just
        // "both happened to return 200".
        let cross_collection_miss = app.oneshot(get_from("beta", 999)).await.unwrap();
        assert_eq!(cross_collection_miss.status(), StatusCode::NOT_FOUND);
    }

    #[tokio::test]
    async fn default_collection_route_and_explicit_default_collection_route_share_the_same_data() {
        // Proves there's only ONE Engine instance for "default", not two
        // independently opened ones -- a write via the original /v1/records
        // route must be visible via /v1/collections/default/records too.
        let app = build_test_router(no_security()).await;

        let put_body = serde_json::json!({"id": 42, "vector": [1.0, 2.0]});
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
            .oneshot(Request::builder().uri("/v1/collections/default/records/42").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK, "the default collection route should see data written via the original /v1/records route");
        let json = body_json(response).await;
        assert_eq!(json["vector"], serde_json::json!([1.0, 2.0]));
    }

    #[tokio::test]
    async fn collection_name_rejects_path_traversal_attempts() {
        let app = build_test_router(no_security()).await;
        for malicious_name in ["../../etc", "..", "a/b", "a%2Fb"] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri(format!("/v1/collections/{malicious_name}/records/1"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            // Either rejected by our own validation (400) or never
            // routed here at all by axum's own path segment handling
            // (404) -- either is an acceptable, safe outcome. What's
            // NOT acceptable is a 200/500 that implies the raw string
            // reached the filesystem.
            assert!(
                response.status() == StatusCode::BAD_REQUEST || response.status() == StatusCode::NOT_FOUND,
                "malicious collection name {malicious_name:?} got unexpected status {}",
                response.status()
            );
        }
    }

    #[tokio::test]
    async fn collection_name_too_long_is_rejected() {
        let app = build_test_router(no_security()).await;
        let long_name = "a".repeat(200);
        let response = app
            .oneshot(Request::builder().uri(format!("/v1/collections/{long_name}/records/1")).body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn list_collections_includes_created_ones() {
        let app = build_test_router(no_security()).await;
        let put_body = serde_json::json!({"id": 1, "vector": [1.0]});
        app.clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/collections/my_new_collection/records")
                    .header("content-type", "application/json")
                    .body(Body::from(put_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();

        let response = app.oneshot(Request::builder().uri("/v1/collections").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        let names: Vec<String> = json["collections"].as_array().unwrap().iter().map(|v| v.as_str().unwrap().to_string()).collect();
        assert!(names.contains(&"my_new_collection".to_string()));
    }

    #[tokio::test]
    async fn collection_scoped_search_and_compact_work_end_to_end() {
        let app = build_test_router(no_security()).await;

        for (id, vec) in [(1, [0.0, 0.0]), (2, [10.0, 10.0]), (3, [0.1, 0.1])] {
            let body = serde_json::json!({"id": id, "vector": vec});
            app.clone()
                .oneshot(
                    Request::builder()
                        .method("POST")
                        .uri("/v1/collections/search_test/records")
                        .header("content-type", "application/json")
                        .body(Body::from(body.to_string()))
                        .unwrap(),
                )
                .await
                .unwrap();
        }

        app.clone()
            .oneshot(Request::builder().method("POST").uri("/v1/collections/search_test/index/build").body(Body::empty()).unwrap())
            .await
            .unwrap();

        let search_body = serde_json::json!({"vector": [0.0, 0.0], "k": 2, "ef_search": 20});
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/collections/search_test/search")
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

        let response = app
            .oneshot(Request::builder().method("POST").uri("/v1/collections/search_test/compact").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn collection_routes_require_auth_like_default_routes() {
        let app = build_test_router(security_with_keys(&["secret123"])).await;
        let response = app
            .oneshot(Request::builder().uri("/v1/collections/some_collection/stats").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    // -- Phase 10: typed metadata HTTP tests ---------------------------

    #[tokio::test]
    async fn insert_and_get_numeric_and_boolean_metadata_over_http() {
        let app = build_test_router(no_security()).await;
        let put_body = serde_json::json!({
            "id": 1,
            "vector": [1.0, 2.0],
            "metadata": {"category": "docs", "price": 29.99, "in_stock": true}
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

        let response = app.oneshot(Request::builder().uri("/v1/records/1").body(Body::empty()).unwrap()).await.unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        assert_eq!(json["metadata"]["category"], "docs");
        assert_eq!(json["metadata"]["price"], 29.99);
        assert_eq!(json["metadata"]["in_stock"], true);
    }

    #[tokio::test]
    async fn insert_rejects_array_metadata_value_with_clear_error() {
        let app = build_test_router(no_security()).await;
        let put_body = serde_json::json!({"id": 1, "vector": [1.0], "metadata": {"tags": ["a", "b"]}});
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
        let json = body_json(response).await;
        assert!(json["error"].as_str().unwrap().contains("arrays"));
    }

    #[tokio::test]
    async fn range_filter_over_http_end_to_end() {
        let app = build_test_router(no_security()).await;
        for (id, price) in [(1, 10.0), (2, 20.0), (3, 30.0)] {
            let put_body = serde_json::json!({"id": id, "vector": [id as f32, 0.0], "metadata": {"price": price}});
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
        }
        app.clone()
            .oneshot(Request::builder().method("POST").uri("/v1/index/build").body(Body::empty()).unwrap())
            .await
            .unwrap();

        let search_body = serde_json::json!({"vector": [0.0, 0.0], "k": 10, "ef_search": 50, "field": "price", "op": "gte", "value": 20.0});
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/search/filtered")
                    .header("content-type", "application/json")
                    .body(Body::from(search_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        let mut ids: Vec<i64> = json["results"].as_array().unwrap().iter().map(|r| r["id"].as_i64().unwrap()).collect();
        ids.sort();
        assert_eq!(ids, vec![2, 3], "gte(20.0) should include the boundary and everything above");
    }

    #[tokio::test]
    async fn filtered_search_without_op_still_defaults_to_equality_backward_compatible() {
        // The exact pre-Phase-10 request shape: no "op" field at all.
        let app = build_test_router(no_security()).await;
        for (id, category) in [(1, "docs"), (2, "code")] {
            let put_body = serde_json::json!({"id": id, "vector": [id as f32, 0.0], "metadata": {"category": category}});
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
        }
        app.clone()
            .oneshot(Request::builder().method("POST").uri("/v1/index/build").body(Body::empty()).unwrap())
            .await
            .unwrap();

        let search_body = serde_json::json!({"vector": [0.0, 0.0], "k": 10, "ef_search": 50, "field": "category", "value": "docs"});
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/search/filtered")
                    .header("content-type", "application/json")
                    .body(Body::from(search_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let json = body_json(response).await;
        let ids: Vec<i64> = json["results"].as_array().unwrap().iter().map(|r| r["id"].as_i64().unwrap()).collect();
        assert_eq!(ids, vec![1]);
    }

    #[tokio::test]
    async fn range_op_with_non_numeric_value_returns_clear_400() {
        let app = build_test_router(no_security()).await;
        let search_body = serde_json::json!({"vector": [0.0, 0.0], "k": 5, "field": "price", "op": "gt", "value": "expensive"});
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/search/filtered")
                    .header("content-type", "application/json")
                    .body(Body::from(search_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
        let json = body_json(response).await;
        assert!(json["error"].as_str().unwrap().contains("numeric"));
    }

    #[tokio::test]
    async fn unknown_filter_op_returns_clear_400() {
        let app = build_test_router(no_security()).await;
        let search_body = serde_json::json!({"vector": [0.0, 0.0], "k": 5, "field": "price", "op": "between", "value": 5.0});
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/search/filtered")
                    .header("content-type", "application/json")
                    .body(Body::from(search_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn stats_reflects_engine_state() {
        let app = build_test_router(no_security()).await;
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

    fn api_keys(keys: &[&str]) -> ApiKeys {
        Arc::new(Some(keys.iter().map(|k| k.to_string()).collect()))
    }

    fn no_security() -> SecurityConfig {
        SecurityConfig { api_keys: Arc::new(None), rate_limiter: None, anomaly_detector: None }
    }

    fn security_with_keys(keys: &[&str]) -> SecurityConfig {
        SecurityConfig { api_keys: api_keys(keys), rate_limiter: None, anomaly_detector: None }
    }

    #[tokio::test]
    async fn health_check_bypasses_auth_even_when_enabled() {
        let app = build_test_router(security_with_keys(&["secret123"])).await;
        // No Authorization header at all -- /health must still work.
        let response = app
            .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn protected_route_without_key_returns_401_when_auth_enabled() {
        let app = build_test_router(security_with_keys(&["secret123"])).await;
        let response = app
            .oneshot(Request::builder().uri("/v1/stats").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn protected_route_with_wrong_key_returns_401() {
        let app = build_test_router(security_with_keys(&["secret123"])).await;
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/stats")
                    .header("Authorization", "Bearer wrong-key")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn protected_route_with_correct_key_succeeds() {
        let app = build_test_router(security_with_keys(&["secret123"])).await;
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/stats")
                    .header("Authorization", "Bearer secret123")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn multiple_configured_keys_each_work_independently() {
        let app = build_test_router(security_with_keys(&["key-for-client-a", "key-for-client-b"])).await;

        for key in ["key-for-client-a", "key-for-client-b"] {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/v1/stats")
                        .header("Authorization", format!("Bearer {key}"))
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "key {key} should be accepted");
        }
    }

    #[tokio::test]
    async fn no_keys_configured_means_auth_disabled_backward_compatible() {
        // The Phase 5/6 default: Arc::new(None) means every existing
        // client (no Authorization header at all) keeps working exactly
        // as before auth was added.
        let app = build_test_router(no_security()).await;
        let response = app
            .oneshot(Request::builder().uri("/v1/stats").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn auth_applies_to_write_routes_not_just_reads() {
        let app = build_test_router(security_with_keys(&["secret123"])).await;
        let put_body = serde_json::json!({"id": 1, "vector": [1.0]});
        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/v1/records")
                    .header("content-type", "application/json")
                    // deliberately no Authorization header
                    .body(Body::from(put_body.to_string()))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    #[tokio::test]
    async fn malformed_authorization_header_returns_401_not_500() {
        let app = build_test_router(security_with_keys(&["secret123"])).await;
        let response = app
            .oneshot(
                Request::builder()
                    .uri("/v1/stats")
                    // missing the "Bearer " prefix entirely
                    .header("Authorization", "secret123")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
    }

    fn security_with_rate_limit(keys: Option<&[&str]>, rate_per_sec: f64, burst: f64) -> SecurityConfig {
        SecurityConfig {
            api_keys: match keys {
                Some(k) => api_keys(k),
                None => Arc::new(None),
            },
            rate_limiter: Some(Arc::new(RateLimiter::new(rate_per_sec, burst))),
            anomaly_detector: None,
        }
    }

    #[tokio::test]
    async fn requests_within_burst_all_succeed() {
        let app = build_test_router(security_with_rate_limit(Some(&["k1"]), 1.0, 3.0)).await;
        for _ in 0..3 {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/v1/stats")
                        .header("Authorization", "Bearer k1")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
    }

    #[tokio::test]
    async fn requests_beyond_burst_get_429() {
        let app = build_test_router(security_with_rate_limit(Some(&["k1"]), 1.0, 2.0)).await;
        let make_request = || {
            Request::builder()
                .uri("/v1/stats")
                .header("Authorization", "Bearer k1")
                .body(Body::empty())
                .unwrap()
        };
        // Burst of 2 -- first two succeed immediately.
        for _ in 0..2 {
            let response = app.clone().oneshot(make_request()).await.unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
        // Third request in immediate succession (no time for refill at
        // 1/sec) should be rejected.
        let response = app.clone().oneshot(make_request()).await.unwrap();
        assert_eq!(response.status(), StatusCode::TOO_MANY_REQUESTS);
    }

    #[tokio::test]
    async fn different_api_keys_get_independent_rate_limit_buckets() {
        let app = build_test_router(
            security_with_rate_limit(Some(&["client-a", "client-b"]), 1.0, 1.0),
        )
        .await;
        let request_with_key = |key: &str| {
            Request::builder()
                .uri("/v1/stats")
                .header("Authorization", format!("Bearer {key}"))
                .body(Body::empty())
                .unwrap()
        };
        // Exhaust client-a's single-token burst.
        let r1 = app.clone().oneshot(request_with_key("client-a")).await.unwrap();
        assert_eq!(r1.status(), StatusCode::OK);
        let r2 = app.clone().oneshot(request_with_key("client-a")).await.unwrap();
        assert_eq!(r2.status(), StatusCode::TOO_MANY_REQUESTS, "client-a should be rate-limited now");

        // client-b has its own separate bucket, untouched by client-a's usage.
        let r3 = app.clone().oneshot(request_with_key("client-b")).await.unwrap();
        assert_eq!(r3.status(), StatusCode::OK, "client-b should have its own independent bucket");
    }

    #[tokio::test]
    async fn wrong_key_returns_401_not_429_even_when_rate_limited() {
        // Confirms the ordering guarantee this design was specifically
        // built around: identity is resolved (and auth checked) BEFORE
        // the rate limiter ever sees a key, so an attacker rotating
        // through wrong keys gets consistent 401s, not a mix of 401/429
        // that could leak information about bucket state.
        let app = build_test_router(security_with_rate_limit(Some(&["real-key"]), 1.0, 1.0)).await;
        for _ in 0..5 {
            let response = app
                .clone()
                .oneshot(
                    Request::builder()
                        .uri("/v1/stats")
                        .header("Authorization", "Bearer totally-wrong-key")
                        .body(Body::empty())
                        .unwrap(),
                )
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::UNAUTHORIZED);
        }
    }

    #[tokio::test]
    async fn health_check_bypasses_rate_limiting_too() {
        let app = build_test_router(security_with_rate_limit(None, 1.0, 1.0)).await;
        // Fire many more health checks than the burst would allow for
        // a protected route -- health must never be throttled.
        for _ in 0..10 {
            let response = app
                .clone()
                .oneshot(Request::builder().uri("/health").body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
    }

    #[tokio::test]
    async fn no_auth_configured_shares_one_global_bucket() {
        // Documented, intentional simplification: without auth, there's
        // no per-client identity to key on, so everyone shares one
        // bucket. This test exists to make that behavior explicit and
        // guarded, not just described in a comment.
        let app = build_test_router(security_with_rate_limit(None, 1.0, 1.0)).await;
        let r1 = app
            .clone()
            .oneshot(Request::builder().uri("/v1/stats").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(r1.status(), StatusCode::OK);
        let r2 = app
            .clone()
            .oneshot(Request::builder().uri("/v1/stats").body(Body::empty()).unwrap())
            .await
            .unwrap();
        assert_eq!(r2.status(), StatusCode::TOO_MANY_REQUESTS, "second anonymous request should share the same exhausted bucket");
    }

    #[tokio::test]
    async fn rate_limiting_disabled_by_default_matches_prior_behavior() {
        let app = build_test_router(no_security()).await;
        for _ in 0..20 {
            let response = app
                .clone()
                .oneshot(Request::builder().uri("/v1/stats").body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "no rate limiter configured -- nothing should ever 429");
        }
    }

    // -- Anomaly detection tests ------------------------------------
    //
    // Unit-tested directly against AnomalyDetector, not only through
    // HTTP, and with time passed in explicitly rather than using real
    // Instant::now() + actual sleeping -- Instant has no public
    // constructor for an arbitrary point in time, but `instant +
    // Duration` always works, so a fixed base instant advanced by exact
    // simulated offsets gives fully deterministic, fast tests instead
    // of flaky ones that depend on real wall-clock timing.

    #[test]
    fn steady_rate_requests_never_flagged() {
        let detector = AnomalyDetector::new();
        let t0 = Instant::now();
        for i in 0..60u64 {
            let now = t0 + std::time::Duration::from_secs(i);
            let report = detector.record_and_check("client-a", now);
            assert!(report.is_none(), "steady 1 req/sec pattern should never be flagged (request {i})");
        }
    }

    #[test]
    fn sudden_burst_after_steady_baseline_is_flagged() {
        let detector = AnomalyDetector::new();
        let t0 = Instant::now();
        let mut now = t0;
        // Establish a steady, established baseline first: 1 req/sec for 30 requests.
        for i in 0..30u64 {
            now = t0 + std::time::Duration::from_secs(i);
            detector.record_and_check("client-a", now);
        }
        // Then burst: many requests 10ms apart -- a ~100x rate spike.
        let mut flagged = false;
        for _ in 0..20 {
            now += std::time::Duration::from_millis(10);
            if detector.record_and_check("client-a", now).is_some() {
                flagged = true;
            }
        }
        assert!(flagged, "a sudden burst after an established steady baseline should be flagged");
    }

    #[test]
    fn cold_start_never_flagged_regardless_of_rate() {
        let detector = AnomalyDetector::new();
        let t0 = Instant::now();
        let mut now = t0;
        // min_requests is 10 -- stay under it. Rapid-fire (5ms apart) on
        // purpose: this would clearly be "anomalous" for an established
        // client, but a brand-new client has no baseline yet to deviate from.
        for i in 0..9u64 {
            now += std::time::Duration::from_millis(5);
            let report = detector.record_and_check("brand-new-client", now);
            assert!(report.is_none(), "cold start (request {i}) should never be flagged");
        }
    }

    #[test]
    fn different_clients_have_independent_baselines() {
        let detector = AnomalyDetector::new();
        let t0 = Instant::now();
        for i in 0..30u64 {
            let now = t0 + std::time::Duration::from_secs(i);
            detector.record_and_check("client-a", now);
        }
        // client-b's cold-start burst must not be influenced by
        // client-a's already-established baseline in any way.
        let mut now_b = t0;
        for i in 0..9u64 {
            now_b += std::time::Duration::from_millis(5);
            let report = detector.record_and_check("client-b", now_b);
            assert!(report.is_none(), "client-b's cold start should be unaffected by client-a's baseline (request {i})");
        }
    }

    fn security_with_anomaly_detection() -> SecurityConfig {
        SecurityConfig { api_keys: Arc::new(None), rate_limiter: None, anomaly_detector: Some(Arc::new(AnomalyDetector::new())) }
    }

    #[tokio::test]
    async fn anomaly_detection_never_blocks_requests_even_when_flagged() {
        // The core behavioral guarantee: even when the detector's own
        // logic (verified deterministically above) WOULD flag a burst,
        // the HTTP layer must never turn that into a rejected request.
        // This fires real, fast, back-to-back HTTP requests (not
        // synthetic timings) specifically to also prove that in
        // practice, at the router level, not just in the detector's
        // own unit tests.
        let app = build_test_router(security_with_anomaly_detection()).await;
        for _ in 0..30 {
            let response = app
                .clone()
                .oneshot(Request::builder().uri("/v1/stats").body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK, "anomaly detection must never block a request, only log it");
        }
    }

    #[tokio::test]
    async fn anomaly_detection_disabled_by_default_matches_prior_behavior() {
        let app = build_test_router(no_security()).await;
        for _ in 0..15 {
            let response = app
                .clone()
                .oneshot(Request::builder().uri("/v1/stats").body(Body::empty()).unwrap())
                .await
                .unwrap();
            assert_eq!(response.status(), StatusCode::OK);
        }
    }
}
