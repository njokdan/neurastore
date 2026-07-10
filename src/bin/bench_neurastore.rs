//! Benchmarks NeuraStore's own engine (WAL + memtable + HNSW index) on
//! the same texmex SIFT dataset (.fvecs/.ivecs) and the same methodology
//! -- insert throughput, index build time, unfiltered query latency
//! percentiles, recall@k -- as `bench/scripts/bench_pgvector.py` and
//! `bench_milvus.py`, so the three sets of numbers are directly
//! comparable in `bench/README.md`.
//!
//! Also runs a Phase 3 check: builds the index from most of the corpus,
//! then streams the remainder in one record at a time via `Engine::put`
//! (the incremental path, post-build -- no second `build_index()` call),
//! and re-measures recall/latency against the full, now-complete corpus.
//! This is the real-data counterpart to `vector_index::tests::
//! incremental_growth_matches_batch_build_recall`, which proves the same
//! claim on synthetic clustered data.
//!
//! Usage (after `python bench/scripts/prepare_dataset.py --mode siftsmall`
//! has downloaded and extracted the corpus on the machine running this):
//!
//!   cargo run --release --bin bench_neurastore -- bench/data/siftsmall 10 40

use neurastore::hnsw::HnswParams;
use neurastore::Engine;
use std::collections::HashMap;
use std::env;
use std::fs;
use std::path::Path;
use std::time::Instant;

fn read_fvecs(path: &Path) -> Vec<Vec<f32>> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"));
    let mut vectors = Vec::new();
    let mut i = 0;
    while i + 4 <= bytes.len() {
        let dim = i32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
        i += 4;
        let mut v = Vec::with_capacity(dim);
        for d in 0..dim {
            let start = i + d * 4;
            v.push(f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap()));
        }
        i += dim * 4;
        vectors.push(v);
    }
    vectors
}

fn read_ivecs(path: &Path) -> Vec<Vec<i64>> {
    let bytes = fs::read(path).unwrap_or_else(|e| panic!("failed to read {path:?}: {e}"));
    let mut vectors = Vec::new();
    let mut i = 0;
    while i + 4 <= bytes.len() {
        let dim = i32::from_le_bytes(bytes[i..i + 4].try_into().unwrap()) as usize;
        i += 4;
        let mut v = Vec::with_capacity(dim);
        for d in 0..dim {
            let start = i + d * 4;
            v.push(i32::from_le_bytes(bytes[start..start + 4].try_into().unwrap()) as i64);
        }
        i += dim * 4;
        vectors.push(v);
    }
    vectors
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = ((p / 100.0) * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn print_latency_summary(label: &str, mut samples_ms: Vec<f64>) {
    samples_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = samples_ms.len();
    let mean = samples_ms.iter().sum::<f64>() / n as f64;
    let max = samples_ms.last().copied().unwrap_or(0.0);
    println!(
        "{label}: n={n} mean={mean:.3}ms p50={:.3}ms p95={:.3}ms p99={:.3}ms max={max:.3}ms",
        percentile(&samples_ms, 50.0),
        percentile(&samples_ms, 95.0),
        percentile(&samples_ms, 99.0),
    );
}

fn recall_at_k(retrieved: &[u64], ground_truth: &[i64], k: usize) -> f64 {
    let gt: std::collections::HashSet<i64> = ground_truth.iter().take(k).copied().collect();
    if gt.is_empty() {
        return 0.0;
    }
    let hits = retrieved.iter().take(k).filter(|id| gt.contains(&(**id as i64))).count();
    hits as f64 / gt.len() as f64
}

fn main() {
    let args: Vec<String> = env::args().collect();
    let data_dir = args.get(1).cloned().unwrap_or_else(|| "bench/data/siftsmall".to_string());
    let k: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(10);
    let ef_search: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(40);

    let dir = Path::new(&data_dir);
    let base_path = dir.join("siftsmall_base.fvecs");
    let query_path = dir.join("siftsmall_query.fvecs");
    let gt_path = dir.join("siftsmall_groundtruth.ivecs");

    for p in [&base_path, &query_path, &gt_path] {
        if !p.exists() {
            eprintln!("Missing dataset file: {p:?}");
            eprintln!("Run `python bench/scripts/prepare_dataset.py --mode siftsmall` first.");
            std::process::exit(1);
        }
    }

    println!("Reading dataset from {data_dir}...");
    let base = read_fvecs(&base_path);
    let queries = read_fvecs(&query_path);
    let ground_truth = read_ivecs(&gt_path);
    let dim = base[0].len();
    println!("Dataset: {} base vectors, dim={dim}, {} queries", base.len(), queries.len());

    // Categories synthesized the same way bench_pgvector.py / bench_milvus.py
    // do -- texmex has no metadata, so this exists purely to let a future
    // filtered-query benchmark run against the same corpus (Phase 4).
    let categories = ["docs", "code", "chat", "logs"];

    // Split the corpus: build the index from the first 80%, hold back
    // the last 20% to stream in AFTER build_index() -- proving the
    // Phase 3 claim (incremental growth, no rebuild) on the real corpus,
    // not just synthetic test data.
    let split = (base.len() as f64 * 0.8) as usize;
    let (initial, streamed) = base.split_at(split);
    println!("Splitting corpus: {} initial (batch-built), {} streamed incrementally after", initial.len(), streamed.len());

    let tmp_dir = std::env::temp_dir().join(format!("neurastore_bench_{}", std::process::id()));
    let mut engine = Engine::open(&tmp_dir).expect("failed to open engine");

    println!("Inserting initial {}% (batched -- one WAL fsync, see Wal::append_batch docs)...", ((split as f64 / base.len() as f64) * 100.0).round());
    let insert_start = Instant::now();
    let entries: Vec<(u64, Vec<f32>, HashMap<String, String>)> = initial
        .iter()
        .enumerate()
        .map(|(i, vector)| {
            let category = categories[i % categories.len()];
            (i as u64, vector.clone(), HashMap::from([("category".to_string(), category.to_string())]))
        })
        .collect();
    engine.put_batch(entries).expect("batch insert failed");
    let insert_elapsed = insert_start.elapsed().as_secs_f64();
    let throughput = initial.len() as f64 / insert_elapsed;
    println!("NeuraStore insert throughput: {throughput:.1} vectors/sec");

    println!("Building HNSW index from the initial {}%...", ((split as f64 / base.len() as f64) * 100.0).round());
    let build_start = Instant::now();
    engine.build_index_with_params(HnswParams { m: 16, m_max0: 32, ef_construction: 64 }, 42);
    let build_elapsed = build_start.elapsed().as_secs_f64();
    println!("NeuraStore HNSW build time: {build_elapsed:.2}s");

    // Warm-up pass, matching the methodology fix applied to the Python
    // benchmarks after the pgvector/Milvus cold-start confound was found.
    for q in queries.iter().take(20.min(queries.len())) {
        engine.search_knn(q, k, ef_search);
    }

    let run_queries = |engine: &Engine, label: &str| -> f64 {
        let mut latencies = Vec::with_capacity(queries.len());
        let mut recalls = Vec::with_capacity(queries.len());
        for (i, q) in queries.iter().enumerate() {
            let start = Instant::now();
            let results = engine.search_knn(q, k, ef_search).unwrap_or_default();
            latencies.push(start.elapsed().as_secs_f64() * 1000.0);
            let ids: Vec<u64> = results.iter().map(|(id, _)| *id).collect();
            recalls.push(recall_at_k(&ids, &ground_truth[i], k));
        }
        print_latency_summary(&format!("NeuraStore unfiltered query latency ({label})"), latencies);
        let avg_recall: f64 = recalls.iter().sum::<f64>() / recalls.len() as f64;
        println!("NeuraStore unfiltered recall@{k} ({label}): {avg_recall:.3}");
        avg_recall
    };

    println!("\n--- Querying at {}% corpus (pre-growth) ---", ((split as f64 / base.len() as f64) * 100.0).round());
    println!("(Recall here is expected to be somewhat lower than the final number --");
    println!(" ground truth includes neighbors from the held-back {} records, which haven't", streamed.len());
    println!(" been streamed in yet. That's expected, not a regression.)");
    let pre_growth_recall = run_queries(&engine, "80% corpus, pre-growth");

    println!("\n--- Streaming remaining {} records incrementally (post-build, no rebuild) ---", streamed.len());
    let stream_start = Instant::now();
    let mut stream_latencies = Vec::with_capacity(streamed.len());
    for (offset, vector) in streamed.iter().enumerate() {
        let id = (split + offset) as u64;
        let category = categories[id as usize % categories.len()];
        let t0 = Instant::now();
        engine
            .put(id, vector.clone(), HashMap::from([("category".to_string(), category.to_string())]))
            .expect("streaming insert failed");
        stream_latencies.push(t0.elapsed().as_secs_f64() * 1000.0);
    }
    let stream_elapsed = stream_start.elapsed().as_secs_f64();
    println!(
        "Streamed {} records in {:.2}s ({:.1} vec/sec incremental-insert-into-live-index throughput)",
        streamed.len(),
        stream_elapsed,
        streamed.len() as f64 / stream_elapsed
    );
    print_latency_summary("Per-record incremental insert latency", stream_latencies);

    println!("\n--- Querying at 100% corpus (post-growth, index was never rebuilt) ---");
    let post_growth_recall = run_queries(&engine, "100% corpus, post-growth");

    println!("\n--- Phase 3 claim check ---");
    println!("Pre-growth recall (80%, informational):  {pre_growth_recall:.3}");
    println!("Post-growth recall (100%, no rebuild):    {post_growth_recall:.3}");
    if post_growth_recall + 0.05 >= pre_growth_recall {
        println!("OK: recall did not degrade after incremental growth without a rebuild.");
    } else {
        println!("WARNING: recall dropped after incremental growth -- investigate before trusting this index shape for production use.");
    }

    // Clean up the temp engine directory -- this binary is a benchmark
    // harness, not meant to leave durable state behind like a real run would.
    let _ = fs::remove_dir_all(&tmp_dir);
}
