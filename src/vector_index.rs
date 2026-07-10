//! VectorIndex: bridges the engine's RecordId space to HNSW's internal,
//! dense (0..n) node id space, and (as of Phase 4) tracks per-record
//! metadata for filtered search.
//!
//! Phase 2 gave this a one-shot `build()`. Phase 3 added `insert()` (grow
//! without rebuilding) and `delete()` (soft-delete tombstones). Phase 4
//! adds:
//!
//! - **Update correctness.** HNSW graph nodes are append-only -- there's
//!   no cheap way to modify a node's vector in place. So re-inserting an
//!   existing RecordId (an "update," which `Engine::put` does whenever a
//!   caller writes to an id that already exists) creates a *new* internal
//!   graph node, leaving the *old* one stranded in the graph. Phase 3's
//!   external-RecordId-keyed tombstone set couldn't tell that stale node
//!   apart from the new live one -- both shared the same external id, so
//!   deleting/undeleting one accidentally affected both. Fixed here by
//!   tracking tombstones by *internal* node id instead, and a reverse
//!   map (external id -> current live internal id) that lets `insert()`
//!   tombstone the previous internal node whenever an id is overwritten.
//! - **Metadata-aware filtered search**, pushing a predicate into the
//!   graph traversal (see `hnsw.rs::search_filtered`) instead of
//!   fetching an unfiltered top-k and discarding non-matches after --
//!   that overfetch-then-filter pattern is the exact behavior the
//!   pgvector/Milvus baseline showed paying a real latency tax for
//!   (bench/README.md, Phase 0). A small inverted index
//!   (field -> value -> internal ids) supports a brute-force fallback
//!   for highly selective filters, where computing exact distances over
//!   a small candidate set directly is cheaper than a graph search.

use crate::hnsw::{HnswIndex, HnswParams};
use crate::record::{Record, RecordId};
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

/// Internal-id + distance pair for the brute-force top-k heap in
/// `search_filtered`. Not the same type as `hnsw.rs`'s private
/// `Candidate` (that one's internal to the graph search) -- this is a
/// much smaller, standalone helper for exact-distance ranking.
#[derive(Debug, Clone, Copy)]
struct Candidate {
    dist: f32,
    id: usize,
}
impl PartialEq for Candidate {
    fn eq(&self, other: &Self) -> bool {
        self.dist == other.dist
    }
}
impl Eq for Candidate {}
impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist.partial_cmp(&other.dist).unwrap_or(Ordering::Equal)
    }
}

/// Below this many matching candidates, compute exact distances directly
/// instead of running a graph search. Originally set to 500 as an
/// untuned guess (see git history). Real-data benchmarking on siftsmall
/// (bench/README.md's Phase 4 section) showed that guess was too
/// conservative: brute force over 200 candidates took ~0.06ms, while
/// graph-traversal search over ~2,500 candidates (25% selectivity) was
/// the slowest measured case (4.43x filter tax, worse than pgvector's
/// 2.6x baseline). Raised to 3,000 based on that evidence -- still a
/// guess in the sense that it hasn't been swept/optimized against a
/// range of corpus sizes and dimensions, but now a guess grounded in a
/// real measurement instead of an arbitrary starting number.
const BRUTE_FORCE_THRESHOLD: usize = 3_000;

/// Caps how much of the graph a filtered search will visit before
/// giving up, so a filter matching almost nothing doesn't silently
/// degrade into a full graph scan. See `hnsw.rs::search_layer` for the
/// mechanism this bounds.
const MAX_FILTERED_VISITS: usize = 20_000;

pub struct VectorIndex {
    hnsw: HnswIndex,
    /// internal HNSW node id -> RecordId. Grows by one on every
    /// `insert`, in lockstep with `hnsw`'s internal ids.
    id_map: Vec<RecordId>,
    /// internal HNSW node id -> metadata, parallel to `id_map`. Phase 4:
    /// what makes filtered search possible without going back to the
    /// engine's storage layer for every candidate.
    metadata: Vec<HashMap<String, String>>,
    /// external RecordId -> its CURRENT live internal node id. An
    /// update (re-insert of an existing RecordId) changes this mapping
    /// and tombstones the previous internal id.
    id_to_internal: HashMap<RecordId, usize>,
    /// Internal node ids to exclude from every search result -- covers
    /// both explicit deletes and internal nodes superseded by a later
    /// update to the same external id. See module docs.
    tombstoned: HashSet<usize>,
    /// field -> value -> internal ids currently holding that value.
    /// Supports selectivity estimation and the brute-force fallback
    /// path. Not pruned when a node is tombstoned (cheap to check
    /// `tombstoned` at read time instead of maintaining two structures
    /// in lockstep) -- callers must filter tombstoned ids out of
    /// whatever this returns.
    field_index: HashMap<String, HashMap<String, Vec<usize>>>,
    dim: Option<usize>,
    live_count: usize,
    /// Persists across calls so incremental inserts continue the same
    /// random stream `build()` started, instead of every `insert()` call
    /// re-seeding and producing a biased/repeated level distribution.
    rng: StdRng,
}

impl VectorIndex {
    /// Build a fresh index from a snapshot of records. Records with
    /// mismatched vector dimension are skipped (schema validation at
    /// write time is still not enforced elsewhere as of Phase 4).
    pub fn build(records: &[Record], params: HnswParams, seed: u64) -> Self {
        let mut index = Self {
            hnsw: HnswIndex::new(params),
            id_map: Vec::with_capacity(records.len()),
            metadata: Vec::with_capacity(records.len()),
            id_to_internal: HashMap::new(),
            tombstoned: HashSet::new(),
            field_index: HashMap::new(),
            dim: None,
            live_count: 0,
            rng: StdRng::seed_from_u64(seed),
        };
        for record in records {
            index.insert(record.id, &record.vector, &record.metadata);
        }
        index
    }

    /// Create an empty index ready to grow purely incrementally via
    /// `insert()` -- the counterpart to `build()`'s all-at-once
    /// construction.
    pub fn empty(params: HnswParams, seed: u64) -> Self {
        Self {
            hnsw: HnswIndex::new(params),
            id_map: Vec::new(),
            metadata: Vec::new(),
            id_to_internal: HashMap::new(),
            tombstoned: HashSet::new(),
            field_index: HashMap::new(),
            dim: None,
            live_count: 0,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Add one record to the index without rebuilding, or update it if
    /// `id` already exists (tombstones the previous internal node --
    /// see module docs). Returns `true` if inserted/updated, `false` if
    /// skipped (empty vector, or dimension mismatch against whatever the
    /// index's dimension was first established as).
    pub fn insert(&mut self, id: RecordId, vector: &[f32], metadata: &HashMap<String, String>) -> bool {
        if vector.is_empty() {
            return false;
        }
        match self.dim {
            None => self.dim = Some(vector.len()),
            Some(d) if d != vector.len() => return false,
            _ => {}
        }

        // An update: the id already has a live internal node. Tombstone
        // it -- its vector/metadata are stale the instant the new node
        // exists, and it must never appear in a result again.
        let is_update = if let Some(&old_internal) = self.id_to_internal.get(&id) {
            self.tombstoned.insert(old_internal);
            true
        } else {
            false
        };

        let new_internal = self.hnsw.insert(vector.to_vec(), &mut self.rng);
        self.id_map.push(id);
        self.metadata.push(metadata.clone());
        for (field, value) in metadata {
            self.field_index
                .entry(field.clone())
                .or_default()
                .entry(value.clone())
                .or_default()
                .push(new_internal);
        }
        self.id_to_internal.insert(id, new_internal);
        if !is_update {
            self.live_count += 1;
        }
        true
    }

    /// Soft-delete: the id is filtered out of all future search results,
    /// but its graph node stays in place (see module docs). A no-op if
    /// the id was never inserted or was already deleted.
    pub fn delete(&mut self, id: RecordId) {
        if let Some(&internal) = self.id_to_internal.get(&id) {
            if self.tombstoned.insert(internal) {
                self.live_count = self.live_count.saturating_sub(1);
            }
        }
    }

    pub fn is_deleted(&self, id: RecordId) -> bool {
        match self.id_to_internal.get(&id) {
            Some(&internal) => self.tombstoned.contains(&internal),
            None => true, // never existed -- treat as "not present"
        }
    }

    /// Number of live (non-deleted, non-superseded) entries. O(1) --
    /// tracked incrementally rather than derived from the graph's raw
    /// node count, which includes stale/tombstoned nodes.
    pub fn len(&self) -> usize {
        self.live_count
    }

    pub fn is_empty(&self) -> bool {
        self.live_count == 0
    }

    pub fn dim(&self) -> Option<usize> {
        self.dim
    }

    fn is_live_internal(&self, internal_id: usize) -> bool {
        !self.tombstoned.contains(&internal_id)
    }

    /// Approximate k-NN search, translated back to RecordIds, with
    /// tombstoned (deleted or superseded) nodes filtered out. Distances
    /// returned are squared L2, matching the pgvector/Milvus baseline
    /// metric.
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(RecordId, f32)> {
        if self.tombstoned.is_empty() {
            return self
                .hnsw
                .search(query, k, ef_search)
                .into_iter()
                .map(|(internal_id, dist)| (self.id_map[internal_id], dist))
                .collect();
        }
        let overfetch_k = k + self.tombstoned.len().min(k * 4 + 50);
        let overfetch_ef = ef_search.max(overfetch_k);
        self.hnsw
            .search(query, overfetch_k, overfetch_ef)
            .into_iter()
            .filter(|(internal_id, _)| self.is_live_internal(*internal_id))
            .take(k)
            .map(|(internal_id, dist)| (self.id_map[internal_id], dist))
            .collect()
    }

    /// Filtered k-NN search: only records where `metadata[field] == value`
    /// are eligible results. This is Phase 4's actual point -- the
    /// predicate is either (a) pushed directly into the HNSW graph
    /// traversal (see `hnsw.rs::search_filtered`), so the search adapts
    /// its depth to how selective the filter is instead of discarding an
    /// unfiltered top-k after the fact, or (b) for highly selective
    /// filters, answered by exact brute-force distance computation over
    /// the small matching candidate set, which is cheaper than a graph
    /// search when there are only a handful of candidates anyway.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        field: &str,
        value: &str,
    ) -> Vec<(RecordId, f32)> {
        let candidate_internal_ids: Vec<usize> = self
            .field_index
            .get(field)
            .and_then(|values| values.get(value))
            .map(|ids| ids.iter().copied().filter(|&id| self.is_live_internal(id)).collect())
            .unwrap_or_default();

        if candidate_internal_ids.is_empty() {
            return Vec::new();
        }

        if candidate_internal_ids.len() <= BRUTE_FORCE_THRESHOLD {
            // Real-hardware benchmarking (bench/README.md's Phase 4
            // section) found the sort was never the bottleneck here --
            // the previous heap optimization (below) confirmed that by
            // barely moving the number. The actual cost is the distance
            // arithmetic itself: at 2,500 candidates x 128 dimensions,
            // that's ~320,000 floating-point operations per query. Each
            // candidate's distance is independent of every other, which
            // makes this "embarrassingly parallel" -- computing them
            // across CPU cores (via rayon) is a much lower-risk lever
            // than manual SIMD (no unsafe code, no platform-specific
            // intrinsics) for the exact bottleneck that was measured,
            // not guessed at.
            //
            // Below PARALLEL_THRESHOLD candidates, skip the parallel
            // path -- thread-pool dispatch overhead can exceed the
            // actual work for small candidate sets, and the sequential
            // path already measured well there (0.22x-0.52x tax).
            const PARALLEL_THRESHOLD: usize = 200;

            let scored: Vec<Candidate> = if candidate_internal_ids.len() > PARALLEL_THRESHOLD {
                use rayon::prelude::*;
                // Reverted from chunked back to per-item par_iter after
                // real measurement: chunking reduced task-dispatch
                // overhead in theory, but on real (noisy, shared laptop)
                // hardware it measured WORSE on the typical case (median
                // 1.76x -> 2.32x) despite lower variance -- likely
                // because per-item granularity lets rayon's work-stealing
                // scheduler adapt when a thread gets preempted by
                // background system load, while a whole chunk stalling
                // costs more than the scheduling overhead chunking saved.
                // Documented here rather than silently reverted, since
                // "the safer-looking optimization measured worse" is
                // itself a useful, real finding -- see bench/README.md's
                // Phase 4 section for the actual numbers this is based on.
                candidate_internal_ids
                    .par_iter()
                    .map(|&id| {
                        let v = self.hnsw.vector(id);
                        let dist: f32 = query.iter().zip(v.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                        Candidate { dist, id }
                    })
                    .collect()
            } else {
                candidate_internal_ids
                    .iter()
                    .map(|&id| {
                        let v = self.hnsw.vector(id);
                        let dist: f32 = query.iter().zip(v.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                        Candidate { dist, id }
                    })
                    .collect()
            };

            // Bounded top-k selection via a max-heap of size k, instead
            // of a full sort -- O(n log k) instead of O(n log n). This
            // part stays sequential: k is tiny (10-ish), so there's
            // nothing meaningful to parallelize here, and the earlier
            // measurement showed this step was never the bottleneck
            // anyway. The distance computation above was.
            let mut heap: BinaryHeap<Candidate> = BinaryHeap::with_capacity(k + 1);
            for c in scored {
                if heap.len() < k {
                    heap.push(c);
                } else if heap.peek().map(|worst| c.dist < worst.dist).unwrap_or(true) {
                    heap.pop();
                    heap.push(c);
                }
            }
            let mut out: Vec<Candidate> = heap.into_vec();
            out.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(std::cmp::Ordering::Equal));
            return out.into_iter().map(|c| (self.id_map[c.id], c.dist)).collect();
        }

        // Broad filter: push the predicate into the graph traversal
        // instead of overfetching-then-filtering.
        let field_owned = field.to_string();
        let value_owned = value.to_string();
        let metadata = &self.metadata;
        let tombstoned = &self.tombstoned;
        let filter = move |internal_id: usize| {
            !tombstoned.contains(&internal_id)
                && metadata[internal_id].get(&field_owned).map(|v| v == &value_owned).unwrap_or(false)
        };

        self.hnsw
            .search_filtered(query, k, ef_search, &filter, MAX_FILTERED_VISITS)
            .into_iter()
            .map(|(internal_id, dist)| (self.id_map[internal_id], dist))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rec(id: u64, vector: Vec<f32>) -> Record {
        Record::new(id, vector, HashMap::new(), id)
    }

    #[test]
    fn record_ids_are_preserved_through_search() {
        let records = vec![
            rec(100, vec![0.0, 0.0]),
            rec(200, vec![10.0, 10.0]),
            rec(300, vec![0.1, 0.1]),
        ];
        let index = VectorIndex::build(&records, HnswParams::default(), 1);
        let results = index.search(&[0.0, 0.0], 2, 20);

        let ids: Vec<RecordId> = results.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&100));
        assert!(ids.contains(&300));
        assert!(!ids.contains(&200));
    }

    #[test]
    fn skips_records_with_mismatched_dimension() {
        let records = vec![
            rec(1, vec![1.0, 2.0, 3.0]),
            rec(2, vec![1.0, 2.0]), // wrong dim, should be skipped
            rec(3, vec![4.0, 5.0, 6.0]),
        ];
        let index = VectorIndex::build(&records, HnswParams::default(), 1);
        assert_eq!(index.len(), 2);
        assert_eq!(index.dim(), Some(3));
    }

    #[test]
    fn skips_records_with_empty_vector() {
        let records = vec![rec(1, vec![]), rec(2, vec![1.0, 2.0])];
        let index = VectorIndex::build(&records, HnswParams::default(), 1);
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn incremental_insert_extends_an_existing_index() {
        let mut index = VectorIndex::empty(HnswParams::default(), 1);
        assert!(index.insert(1, &[0.0, 0.0], &HashMap::new()));
        assert_eq!(index.len(), 1);
        assert!(index.insert(2, &[10.0, 10.0], &HashMap::new()));
        assert_eq!(index.len(), 2);
        assert!(index.insert(3, &[0.1, 0.1], &HashMap::new()));
        assert_eq!(index.len(), 3);

        let results = index.search(&[0.0, 0.0], 2, 20);
        let ids: Vec<RecordId> = results.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
    }

    #[test]
    fn incremental_insert_rejects_dimension_mismatch_after_first_vector() {
        let mut index = VectorIndex::empty(HnswParams::default(), 1);
        assert!(index.insert(1, &[1.0, 2.0, 3.0], &HashMap::new()));
        assert!(!index.insert(2, &[1.0, 2.0], &HashMap::new())); // wrong dim
        assert_eq!(index.len(), 1);
    }

    #[test]
    fn delete_removes_id_from_search_results() {
        let mut index = VectorIndex::empty(HnswParams::default(), 1);
        index.insert(1, &[0.0, 0.0], &HashMap::new());
        index.insert(2, &[0.01, 0.01], &HashMap::new());
        index.insert(3, &[0.02, 0.02], &HashMap::new());

        index.delete(2);
        assert!(index.is_deleted(2));
        assert_eq!(index.len(), 2);

        let results = index.search(&[0.0, 0.0], 3, 50);
        let ids: Vec<RecordId> = results.iter().map(|(id, _)| *id).collect();
        assert!(!ids.contains(&2), "deleted id 2 should never appear in search results");
        assert!(ids.contains(&1));
        assert!(ids.contains(&3));
    }

    #[test]
    fn reinserting_a_deleted_id_undeletes_it() {
        let mut index = VectorIndex::empty(HnswParams::default(), 1);
        index.insert(1, &[0.0, 0.0], &HashMap::new());
        index.delete(1);
        assert!(index.is_deleted(1));

        index.insert(1, &[0.0, 0.0], &HashMap::new()); // re-insert (an "update")
        assert!(!index.is_deleted(1), "re-inserting should clear the tombstone");
    }

    #[test]
    fn updating_a_record_supersedes_the_old_graph_node_not_just_the_new_one() {
        // The Phase 4 bug fix, tested directly: update id 1's vector to
        // somewhere far away. A query near the OLD location must not
        // return id 1 anymore -- if the stale internal node were still
        // live, this would incorrectly still match.
        let mut index = VectorIndex::empty(HnswParams::default(), 1);
        index.insert(1, &[0.0, 0.0], &HashMap::new());
        index.insert(2, &[0.01, 0.01], &HashMap::new());

        // Update id 1 to a totally different location.
        index.insert(1, &[100.0, 100.0], &HashMap::new());
        assert_eq!(index.len(), 2, "update should not change the live count");

        // Query near the OLD location of id 1 -- should only find id 2 now.
        let results = index.search(&[0.0, 0.0], 2, 50);
        let found_id1_near_old_location = results.iter().any(|(id, dist)| *id == 1 && *dist < 1.0);
        assert!(
            !found_id1_near_old_location,
            "stale internal node for id 1 should not be findable near its old location: {results:?}"
        );

        // Query near the NEW location -- should find the updated id 1.
        let results_new = index.search(&[100.0, 100.0], 1, 50);
        assert_eq!(results_new[0].0, 1, "updated id 1 should be findable at its new location");
    }

    #[test]
    fn search_filtered_only_returns_matching_metadata() {
        let mut index = VectorIndex::empty(HnswParams::default(), 1);
        for i in 0..20u64 {
            let category = if i % 2 == 0 { "docs" } else { "code" };
            index.insert(i, &[i as f32, 0.0], &HashMap::from([("category".to_string(), category.to_string())]));
        }

        let results = index.search_filtered(&[0.0, 0.0], 5, 50, "category", "docs");
        assert!(!results.is_empty());
        for (id, _) in &results {
            assert_eq!(id % 2, 0, "only 'docs' (even ids) should be returned, got id {id}");
        }
    }

    #[test]
    fn search_filtered_excludes_deleted_records() {
        let mut index = VectorIndex::empty(HnswParams::default(), 1);
        for i in 0..10u64 {
            index.insert(i, &[i as f32, 0.0], &HashMap::from([("category".to_string(), "docs".to_string())]));
        }
        index.delete(3);

        let results = index.search_filtered(&[0.0, 0.0], 10, 50, "category", "docs");
        let ids: Vec<RecordId> = results.iter().map(|(id, _)| *id).collect();
        assert!(!ids.contains(&3), "deleted record should not appear in filtered results");
    }

    #[test]
    fn search_filtered_returns_empty_for_unknown_value() {
        let mut index = VectorIndex::empty(HnswParams::default(), 1);
        index.insert(1, &[0.0, 0.0], &HashMap::from([("category".to_string(), "docs".to_string())]));
        let results = index.search_filtered(&[0.0, 0.0], 5, 50, "category", "nonexistent");
        assert!(results.is_empty());
    }

    #[test]
    fn search_filtered_matches_brute_force_ground_truth_when_broad() {
        // Uses enough matching candidates to exceed BRUTE_FORCE_THRESHOLD
        // and exercise the graph-traversal path, not just the brute-force
        // fallback -- and checks its results against true brute-force
        // computed independently, to catch the traversal-filter logic
        // getting the wrong answer even if it doesn't crash.
        //
        // 10,000 records / 1-in-3 split -> ~3,333 "docs" candidates,
        // comfortably above BRUTE_FORCE_THRESHOLD (3,000 as of the Phase 4
        // real-data tuning -- see that constant's docs) so this keeps
        // testing the traversal path it's named for instead of silently
        // falling back to brute force after the threshold was raised.
        use rand::Rng;
        let mut rng_data = StdRng::seed_from_u64(3);
        let mut index = VectorIndex::empty(HnswParams::default(), 3);
        let mut all_vectors = Vec::new();

        for i in 0..10_000u64 {
            let v: Vec<f32> = (0..16).map(|_| rng_data.gen_range(-10.0..10.0)).collect();
            let category = if i % 3 == 0 { "docs" } else { "other" };
            index.insert(i, &v, &HashMap::from([("category".to_string(), category.to_string())]));
            all_vectors.push((i, v, category));
        }

        let query: Vec<f32> = (0..16).map(|_| rng_data.gen_range(-10.0..10.0)).collect();
        let k = 10;

        let approx = index.search_filtered(&query, k, 100, "category", "docs");
        let approx_ids: HashSet<RecordId> = approx.iter().map(|(id, _)| *id).collect();

        let mut brute: Vec<(RecordId, f32)> = all_vectors
            .iter()
            .filter(|(_, _, cat)| *cat == "docs")
            .map(|(id, v, _)| {
                let d: f32 = query.iter().zip(v.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                (*id, d)
            })
            .collect();
        brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let brute_ids: HashSet<RecordId> = brute.into_iter().take(k).map(|(id, _)| id).collect();

        let recall = approx_ids.intersection(&brute_ids).count() as f64 / brute_ids.len() as f64;
        assert!(recall > 0.7, "filtered graph search recall too low vs brute force ground truth: {recall}");

        // Every returned result must actually match the filter -- this
        // is a hard correctness requirement, not a recall/quality one.
        for (id, _) in &approx {
            let (_, _, cat) = all_vectors.iter().find(|(i, _, _)| i == id).unwrap();
            assert_eq!(*cat, "docs", "filtered search returned a non-matching record");
        }
    }

    #[test]
    fn search_filtered_parallel_path_matches_brute_force_ground_truth() {
        // Specifically targets the rayon-parallelized brute-force branch
        // (PARALLEL_THRESHOLD=200 < candidates <= BRUTE_FORCE_THRESHOLD=3000)
        // -- none of the other filtered-search tests land in this range
        // (the broad one above deliberately exceeds 3,000 to test graph
        // traversal instead), so without this test the parallel code
        // path would only ever be "it compiles," never "it's correct."
        use rand::Rng;
        let mut rng_data = StdRng::seed_from_u64(11);
        let mut index = VectorIndex::empty(HnswParams::default(), 11);
        let mut all_vectors = Vec::new();

        // 3,000 records, 1-in-3 split -> ~1,000 "docs" candidates:
        // comfortably inside the parallel zone (200, 3000].
        for i in 0..3000u64 {
            let v: Vec<f32> = (0..16).map(|_| rng_data.gen_range(-10.0..10.0)).collect();
            let category = if i % 3 == 0 { "docs" } else { "other" };
            index.insert(i, &v, &HashMap::from([("category".to_string(), category.to_string())]));
            all_vectors.push((i, v, category));
        }

        let query: Vec<f32> = (0..16).map(|_| rng_data.gen_range(-10.0..10.0)).collect();
        let k = 10;

        let approx = index.search_filtered(&query, k, 100, "category", "docs");

        let mut brute: Vec<(RecordId, f32)> = all_vectors
            .iter()
            .filter(|(_, _, cat)| *cat == "docs")
            .map(|(id, v, _)| {
                let d: f32 = query.iter().zip(v.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                (*id, d)
            })
            .collect();
        brute.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        let brute_top_k: Vec<RecordId> = brute.into_iter().take(k).map(|(id, _)| id).collect();

        // Brute force (parallel or sequential) is exact, not approximate
        // -- unlike the graph-traversal path, there's no recall
        // tolerance here. The parallel and sequential code paths must
        // produce identical results (same distances, same candidates,
        // just computed on different threads), so this must match
        // ground truth exactly, not "mostly."
        let approx_ids: Vec<RecordId> = approx.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            approx_ids, brute_top_k,
            "parallel brute-force path must match exact ground truth exactly, not approximately"
        );

        for (id, _) in &approx {
            let (_, _, cat) = all_vectors.iter().find(|(i, _, _)| i == id).unwrap();
            assert_eq!(*cat, "docs", "filtered search returned a non-matching record");
        }
    }

    #[test]
    fn incremental_growth_matches_batch_build_recall() {
        // After the Phase 3 refactor, `build()` is literally implemented
        // as a loop of `insert()` calls against a persisted RNG -- there
        // is no separate "batch algorithm" anymore, by design. This
        // test's value is a regression guard: if a future change
        // reintroduces a diverging bulk-build path, this catches it.
        //
        // Seed choice matters and isn't arbitrary: HNSW's recall on
        // small/moderate clustered corpora is genuinely seed-sensitive
        // (see `hnsw::tests::sparse_clusters_can_strand_a_whole_cluster_
        // from_the_entry_point`). Seed 3 here matches `hnsw::tests::
        // recall_is_high_on_clustered_data`, which uses the identical
        // data and has reliably passed >0.85 recall.
        use rand::Rng;

        fn clustered_vectors(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
            let mut rng = StdRng::seed_from_u64(seed);
            let centers: Vec<Vec<f32>> = (0..n_clusters)
                .map(|_| (0..dim).map(|_| rng.gen_range(-10.0..10.0)).collect())
                .collect();
            (0..n)
                .map(|_| {
                    let center = &centers[rng.gen_range(0..n_clusters)];
                    center.iter().map(|c| c + rng.gen_range(-0.5..0.5)).collect()
                })
                .collect()
        }

        let vectors = clustered_vectors(2000, 32, 10, 7);
        let records: Vec<Record> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| rec(i as u64, v.clone()))
            .collect();

        let batch_index = VectorIndex::build(&records, HnswParams::default(), 3);

        let mut incremental_index = VectorIndex::empty(HnswParams::default(), 3);
        for (i, v) in vectors.iter().enumerate() {
            incremental_index.insert(i as u64, v, &HashMap::new());
        }

        let queries = clustered_vectors(50, 32, 10, 8);
        let brute_force_gt = |q: &[f32]| -> Vec<RecordId> {
            let mut scored: Vec<(RecordId, f32)> = vectors
                .iter()
                .enumerate()
                .map(|(i, v)| {
                    let d: f32 = q.iter().zip(v.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                    (i as u64, d)
                })
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            scored.into_iter().take(10).map(|(id, _)| id).collect()
        };

        let recall = |index: &VectorIndex| -> f64 {
            let mut recalls = Vec::new();
            for q in &queries {
                let gt: HashSet<RecordId> = brute_force_gt(q).into_iter().collect();
                let approx: HashSet<RecordId> =
                    index.search(q, 10, 50).into_iter().map(|(id, _)| id).collect();
                recalls.push(gt.intersection(&approx).count() as f64 / gt.len() as f64);
            }
            recalls.iter().sum::<f64>() / recalls.len() as f64
        };

        let batch_recall = recall(&batch_index);
        let incremental_recall = recall(&incremental_index);

        assert!(batch_recall > 0.85, "batch build recall too low: {batch_recall}");
        assert!(incremental_recall > 0.85, "incremental build recall too low: {incremental_recall}");
        assert_eq!(
            batch_recall, incremental_recall,
            "build() and an equivalent manual insert loop should produce bit-identical results"
        );
    }
}
