//! HNSW (Hierarchical Navigable Small World) -- the approximate nearest
//! neighbor index at the core of NeuraStore's vector search.
//!
//! Phase 2 scope: a correct *static* build -- insert a fixed corpus,
//! then query it. It's worth being precise about what "static" means
//! here, since the HNSW insert algorithm itself is inherently
//! incremental (you can call `insert` one vector at a time and it just
//! works). What Phase 2 does *not* yet handle is the harder problem
//! Phase 3 is named for: safe concurrent reads while inserts are
//! happening, and proving recall/latency don't degrade as the graph
//! grows continuously without a rebuild. Phase 2's job is a correct,
//! benchmarkable single-threaded index -- the foundation Phase 3 adds
//! concurrency safety on top of.
//!
//! Distance metric: squared Euclidean (L2). Matches the `vector_l2_ops` /
//! `metric_type: L2` used in the pgvector/Milvus baseline benchmarks, so
//! NeuraStore's numbers are directly comparable to those.
//!
//! Reference: Malkov & Yashunin, "Efficient and robust approximate
//! nearest neighbor search using Hierarchical Navigable Small World
//! graphs" (2016/2018).

use rand::Rng;
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};

#[derive(Debug, Clone, Copy)]
pub struct HnswParams {
    /// Max neighbors per node at layers above 0.
    pub m: usize,
    /// Max neighbors per node at layer 0 (conventionally 2*m -- layer 0
    /// holds every node, so it carries more of the graph's connectivity).
    pub m_max0: usize,
    /// Candidate list size during construction. Higher = better graph
    /// quality (recall), slower build.
    pub ef_construction: usize,
}

impl Default for HnswParams {
    fn default() -> Self {
        // Matches the m=16, ef_construction=64 used in the pgvector/Milvus
        // baseline benchmarks (bench/scripts/bench_pgvector.py,
        // bench_milvus.py) so NeuraStore's index is tuned comparably,
        // not given an unfair advantage via looser parameters.
        Self { m: 16, m_max0: 32, ef_construction: 64 }
    }
}

/// A candidate during search: an internal node id at some distance from
/// the query. Wraps f32 distance in a type that implements Ord (f32
/// alone can't, because of NaN) -- safe here since vector distances are
/// never NaN for well-formed input.
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

/// Reverses ordering so a `BinaryHeap<MinCandidate>` behaves as a
/// min-heap (closest-first), which is what the frontier of unexplored
/// candidates needs during greedy search.
#[derive(Debug, Clone, Copy, PartialEq)]
struct MinCandidate(Candidate);
impl Eq for MinCandidate {}
impl PartialOrd for MinCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MinCandidate {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.cmp(&self.0) // reversed
    }
}

fn squared_l2(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len(), "vector dimension mismatch");
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

struct Node {
    /// Neighbor lists, one per layer this node participates in:
    /// neighbors[0] is layer 0, neighbors[1] is layer 1, etc.
    neighbors: Vec<Vec<usize>>,
}

pub struct HnswIndex {
    params: HnswParams,
    /// All vectors stored contiguously, one after another: vector for
    /// internal id `i` lives at `vectors[i*dim..(i+1)*dim]`. This
    /// replaced a `Vec<Vec<f32>>` (one heap allocation per vector) --
    /// real benchmarking (bench/README.md's Phase 4 section) showed
    /// filtered search spending real time on cache-unfriendly, scattered
    /// memory access when scanning many candidates for a brute-force
    /// distance computation. A flat layout means scanning candidates
    /// walks contiguous memory instead of chasing pointers -- this is
    /// also what the original architecture doc's "hybrid row/columnar
    /// layout" called for, previously only implemented for on-disk
    /// SSTables (`sstable.rs`), not this in-memory structure.
    vectors: Vec<f32>,
    dim: usize,
    count: usize,
    nodes: Vec<Node>,
    entry_point: Option<usize>,
    max_level: usize,
    level_mult: f64,
}

impl HnswIndex {
    pub fn new(params: HnswParams) -> Self {
        let level_mult = 1.0 / (params.m as f64).ln();
        Self {
            params,
            vectors: Vec::new(),
            dim: 0,
            count: 0,
            nodes: Vec::new(),
            entry_point: None,
            max_level: 0,
            level_mult,
        }
    }

    pub fn len(&self) -> usize {
        self.count
    }

    pub fn is_empty(&self) -> bool {
        self.count == 0
    }

    /// Read access to a node's raw vector by internal id. Used by
    /// `VectorIndex`'s brute-force fallback path for highly selective
    /// filtered queries (Phase 4) -- cheaper to compute exact distances
    /// against a small candidate set directly than to run a graph search
    /// for a handful of matches.
    pub fn vector(&self, internal_id: usize) -> &[f32] {
        let start = internal_id * self.dim;
        &self.vectors[start..start + self.dim]
    }

    fn random_level(&self, rng: &mut impl Rng) -> usize {
        let r: f64 = rng.gen_range(f64::EPSILON..1.0);
        (-r.ln() * self.level_mult).floor() as usize
    }

    /// Insert one vector, returning its internal node id (stable for the
    /// lifetime of this index -- callers map their own external ids to
    /// this via `VectorIndex`, not this layer).
    pub fn insert(&mut self, vector: Vec<f32>, rng: &mut impl Rng) -> usize {
        let new_id = self.count;
        let level = self.random_level(rng);
        if self.dim == 0 {
            self.dim = vector.len();
        }
        debug_assert_eq!(
            vector.len(),
            self.dim,
            "HnswIndex::insert called with mismatched dimension -- callers (VectorIndex) must enforce this"
        );
        self.vectors.extend_from_slice(&vector);
        self.count += 1;
        self.nodes.push(Node { neighbors: vec![Vec::new(); level + 1] });

        let Some(entry) = self.entry_point else {
            self.entry_point = Some(new_id);
            self.max_level = level;
            return new_id;
        };

        let query = self.vector(new_id).to_vec();
        let query = query.as_slice();
        let mut cur = entry;

        // Descend from the top layer down to just above the new node's
        // level, greedily moving to the closest neighbor at each layer
        // (ef=1) -- this finds a good entry point for the layers where
        // we'll actually do a full search-and-connect below.
        for lc in (level + 1..=self.max_level).rev() {
            cur = self.search_layer(query, cur, lc, 1, None, self.count)[0].id;
        }

        // For every layer the new node actually lives on, search with
        // ef_construction candidates, connect to the best `m` of them,
        // and prune each neighbor's list so it doesn't grow unbounded.
        for lc in (0..=level.min(self.max_level)).rev() {
            let candidates = self.search_layer(query, cur, lc, self.params.ef_construction, None, self.count);
            let m_layer = if lc == 0 { self.params.m_max0 } else { self.params.m };
            let chosen: Vec<usize> = candidates.iter().take(m_layer).map(|c| c.id).collect();

            self.nodes[new_id].neighbors[lc] = chosen.clone();
            for &neighbor_id in &chosen {
                self.connect_and_prune(neighbor_id, new_id, lc, m_layer);
            }
            if let Some(closest) = candidates.first() {
                cur = closest.id;
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry_point = Some(new_id);
        }

        new_id
    }

    fn connect_and_prune(&mut self, node_id: usize, new_neighbor: usize, layer: usize, m_max: usize) {
        // Clone the neighbor list out first and release the mutable
        // borrow immediately -- `self.vector()` needs an immutable
        // borrow of `self` below, and a method call (unlike direct field
        // access) doesn't let the borrow checker see that `vectors` and
        // `nodes` are disjoint fields, so the mutable borrow of
        // `self.nodes[...]` can't stay alive across those calls.
        let mut neighbors: Vec<usize> = self.nodes[node_id].neighbors[layer].clone();
        if !neighbors.contains(&new_neighbor) {
            neighbors.push(new_neighbor);
        }
        if neighbors.len() > m_max {
            // Simple pruning heuristic: keep the m_max closest to
            // node_id. (The HNSW paper's "heuristic" neighbor selection
            // considering neighbor diversity gives better graph quality;
            // closest-m is the simpler variant and Phase 2's correctness
            // bar -- swapping in the diversity heuristic is a reasonable
            // later optimization, not a correctness requirement.)
            let node_vec = self.vector(node_id).to_vec();
            let mut scored: Vec<(usize, f32)> = neighbors
                .iter()
                .map(|&id| (id, squared_l2(&node_vec, self.vector(id))))
                .collect();
            scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
            scored.truncate(m_max);
            neighbors = scored.into_iter().map(|(id, _)| id).collect();
        }
        self.nodes[node_id].neighbors[layer] = neighbors;
    }

    /// Greedy best-first search within a single layer. Returns up to
    /// `ef` candidates, sorted closest-first.
    ///
    /// `filter`, when present, is the mechanism that actually avoids the
    /// overfetch-then-filter tax pgvector pays (see bench/README.md's
    /// Phase 0 baseline): the graph is still traversed through
    /// *non-matching* nodes (their neighbors might lead to matching
    /// ones -- a matching node reached only through a non-matching
    /// "bridge" node must still be findable), but only *matching* nodes
    /// count toward `results` and the `ef`-sized stopping budget. That
    /// means a highly selective filter naturally makes the search dig
    /// deeper into the graph instead of returning a mostly-empty result
    /// early -- the predicate is part of the search itself, not a
    /// discard step applied after.
    ///
    /// `max_visits` bounds the worst case (a filter matching zero or
    /// almost nothing in the whole graph): without a cap, search would
    /// silently degrade into a full graph traversal. This is the same
    /// "heuristic, not a hard guarantee" tradeoff already documented for
    /// tombstone filtering in `vector_index.rs`.
    fn search_layer(
        &self,
        query: &[f32],
        entry: usize,
        layer: usize,
        ef: usize,
        filter: Option<&dyn Fn(usize) -> bool>,
        max_visits: usize,
    ) -> Vec<Candidate> {
        let matches = |id: usize| filter.map(|f| f(id)).unwrap_or(true);

        let mut visited: HashSet<usize> = HashSet::new();
        visited.insert(entry);

        let entry_dist = squared_l2(query, self.vector(entry));
        let mut candidates: BinaryHeap<MinCandidate> =
            BinaryHeap::from([MinCandidate(Candidate { dist: entry_dist, id: entry })]);
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();
        if matches(entry) {
            results.push(Candidate { dist: entry_dist, id: entry });
        }

        while let Some(MinCandidate(current)) = candidates.pop() {
            if visited.len() > max_visits {
                break;
            }
            // Stop once the closest remaining candidate is farther than
            // our worst kept result and we already have enough MATCHING
            // results -- nothing left in the frontier can improve the
            // answer. Note this only fires once `results` (matching
            // nodes only) reaches `ef`, so a selective filter correctly
            // keeps the search going past where an unfiltered search
            // would have stopped.
            if let Some(worst) = results.peek() {
                if current.dist > worst.dist && results.len() >= ef {
                    break;
                }
            }

            if layer >= self.nodes[current.id].neighbors.len() {
                continue;
            }
            for &neighbor_id in &self.nodes[current.id].neighbors[layer] {
                if visited.insert(neighbor_id) {
                    let dist = squared_l2(query, self.vector(neighbor_id));
                    // Always explore through the neighbor (push to the
                    // frontier) regardless of whether it matches --
                    // it may be the only path to a matching node deeper
                    // in the graph. Only matching neighbors are eligible
                    // to become part of the answer set.
                    candidates.push(MinCandidate(Candidate { dist, id: neighbor_id }));
                    if matches(neighbor_id) {
                        let worst = results.peek().map(|c| c.dist);
                        if results.len() < ef || worst.map(|w| dist < w).unwrap_or(true) {
                            results.push(Candidate { dist, id: neighbor_id });
                            if results.len() > ef {
                                results.pop();
                            }
                        }
                    }
                }
            }
        }

        let mut out: Vec<Candidate> = results.into_vec();
        out.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        out
    }

    /// Approximate k-nearest-neighbor search. `ef_search` controls the
    /// recall/latency tradeoff at query time -- higher finds more true
    /// neighbors at the cost of exploring more of the graph.
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(usize, f32)> {
        let Some(entry) = self.entry_point else { return Vec::new() };
        let mut cur = entry;
        for lc in (1..=self.max_level).rev() {
            cur = self.search_layer(query, cur, lc, 1, None, self.len())[0].id;
        }
        let candidates = self.search_layer(query, cur, 0, ef_search.max(k), None, self.len());
        candidates.into_iter().take(k).map(|c| (c.id, c.dist)).collect()
    }

    /// Approximate k-nearest-neighbor search restricted to nodes where
    /// `filter(internal_id)` returns true -- the predicate is pushed
    /// into the graph traversal itself (see `search_layer`'s docs), not
    /// applied after fetching an unfiltered top-k. `max_visits` bounds
    /// the cost of a highly selective (or impossible) filter; a sensible
    /// default is the graph's total size, capped lower if latency matters
    /// more than exhaustiveness for very selective filters.
    pub fn search_filtered(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        filter: &dyn Fn(usize) -> bool,
        max_visits: usize,
    ) -> Vec<(usize, f32)> {
        let Some(entry) = self.entry_point else { return Vec::new() };
        let mut cur = entry;
        // Upper-layer descent stays unfiltered -- it's coarse navigation
        // toward the right neighborhood, not answer selection, so there's
        // no benefit to restricting it (and doing so could strand the
        // descent if the filter happens to exclude every node at a
        // sparse upper layer -- see the cluster-stranding finding in
        // Phase 2 for why sparse upper layers are already a delicate area).
        for lc in (1..=self.max_level).rev() {
            cur = self.search_layer(query, cur, lc, 1, None, self.len())[0].id;
        }
        let candidates = self.search_layer(query, cur, 0, ef_search.max(k), Some(filter), max_visits);
        candidates.into_iter().take(k).map(|c| (c.id, c.dist)).collect()
    }

    /// Exact brute-force k-NN, for correctness/recall testing against
    /// the approximate `search` above. O(n) per query -- reference only.
    pub fn brute_force(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut scored: Vec<(usize, f32)> = (0..self.count)
            .map(|id| (id, squared_l2(query, self.vector(id))))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand::rngs::StdRng;

    fn recall_at_k(approx: &[(usize, f32)], exact: &[(usize, f32)], k: usize) -> f64 {
        let exact_ids: HashSet<usize> = exact.iter().take(k).map(|(id, _)| *id).collect();
        let approx_ids: HashSet<usize> = approx.iter().take(k).map(|(id, _)| *id).collect();
        exact_ids.intersection(&approx_ids).count() as f64 / exact_ids.len() as f64
    }

    /// Clustered synthetic data: points scattered tightly around a
    /// handful of cluster centers, not uniform random. This matters --
    /// we learned the hard way during the pgvector/Milvus benchmarking
    /// that uniform random high-dimensional vectors have near-equidistant
    /// pairwise distances (curse of dimensionality), which breaks *any*
    /// HNSW implementation's recall, not just this one. Clustered data
    /// has real nearest-neighbor structure, like real embeddings do.
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

    #[test]
    fn empty_index_search_returns_nothing() {
        let index = HnswIndex::new(HnswParams::default());
        assert!(index.search(&[1.0, 2.0], 5, 20).is_empty());
    }

    #[test]
    fn single_vector_index_finds_itself() {
        let mut rng = StdRng::seed_from_u64(1);
        let mut index = HnswIndex::new(HnswParams::default());
        index.insert(vec![1.0, 2.0, 3.0], &mut rng);
        let results = index.search(&[1.0, 2.0, 3.0], 1, 10);
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0, 0);
        assert!(results[0].1 < 1e-6);
    }

    // Note: an earlier version of this test suite tried to assert that
    // querying with a vector's own value must find it within the top-3
    // approximate results. That turned out to be the wrong invariant --
    // investigating the failures surfaced two distinct real phenomena:
    // (1) genuine cluster stranding, documented below, and (2) plain
    // approximate-search imprecision on tightly-packed near-duplicate
    // points, which is normal HNSW behavior, not a bug. `recall_is_high_
    // on_clustered_data` below is the correct correctness bar -- recall@k
    // against brute-force ground truth, the same methodology used
    // throughout this project's pgvector/Milvus benchmarking -- not an
    // exact-self-lookup guarantee the algorithm never actually promised.

    /// Documents a real HNSW characteristic discovered while writing the
    /// test above: with too few points per cluster relative to `m`, the
    /// base-layer graph can fragment into disconnected islands with no
    /// upper-layer bridge between them, and NO amount of ef_search fixes
    /// it -- the target simply isn't reachable from the entry point. This
    /// isn't a bug in the search; it's a real density requirement. Worth
    /// keeping as a named test (rather than deleting once understood) so
    /// the failure mode stays documented for Phase 3, where the corpus
    /// grows incrementally and this exact risk reappears in a different
    /// form -- a newly-created, initially-small cluster could be
    /// similarly stranded until enough points accumulate in it.
    #[test]
    fn sparse_clusters_can_strand_a_whole_cluster_from_the_entry_point() {
        let mut rng = StdRng::seed_from_u64(2);
        let mut index = HnswIndex::new(HnswParams::default());
        // Deliberately under-dense: 40 pts/cluster, the exact scenario
        // that failed above before widening to 200 pts/cluster.
        let vectors = clustered_vectors(200, 16, 5, 42);
        for v in &vectors {
            index.insert(v.clone(), &mut rng);
        }

        let mut misses = 0;
        for (i, v) in vectors.iter().enumerate() {
            // ef_search far larger than the whole corpus -- if a miss
            // still happens here, it's provably a graph connectivity
            // issue, not an under-explored search budget.
            let results = index.search(v, 3, 1000);
            if !results.iter().any(|(id, _)| *id == i) {
                misses += 1;
            }
        }
        // We expect *some* misses here (that's the point being
        // documented) but not zero and not everything -- this assertion
        // just keeps the test meaningful if the RNG or algorithm changes
        // enough to accidentally "fix" this class of failure entirely.
        assert!(misses > 0, "expected this deliberately under-dense setup to reproduce cluster stranding");
    }

    #[test]
    fn self_lookup_succeeds_on_well_separated_points() {
        // Unlike the tightly-clustered tests elsewhere, these points are
        // spread far enough apart (no jitter, real gaps) that there's no
        // near-tie ambiguity -- self genuinely is the unambiguous nearest
        // neighbor by a wide margin, so this checks the search mechanics
        // without the density/tie issues documented above.
        let mut rng = StdRng::seed_from_u64(5);
        let mut index = HnswIndex::new(HnswParams::default());
        let vectors: Vec<Vec<f32>> = (0..100)
            .map(|i| vec![i as f32 * 10.0, (i as f32 * 7.0) % 50.0])
            .collect();
        for v in &vectors {
            index.insert(v.clone(), &mut rng);
        }
        for i in (0..100).step_by(10) {
            let results = index.search(&vectors[i], 1, 50);
            assert_eq!(results[0].0, i, "well-separated point {i} should find itself as nearest neighbor");
        }
    }

    #[test]
    fn recall_is_high_on_clustered_data() {
        let mut rng = StdRng::seed_from_u64(3);
        let mut index = HnswIndex::new(HnswParams::default());
        let vectors = clustered_vectors(2000, 32, 10, 7);
        for v in &vectors {
            index.insert(v.clone(), &mut rng);
        }

        let queries = clustered_vectors(50, 32, 10, 8);
        let mut recalls = Vec::new();
        for q in &queries {
            let approx = index.search(q, 10, 50);
            let exact = index.brute_force(q, 10);
            recalls.push(recall_at_k(&approx, &exact, 10));
        }
        let avg_recall: f64 = recalls.iter().sum::<f64>() / recalls.len() as f64;
        assert!(avg_recall > 0.85, "expected recall@10 > 0.85 on clustered data, got {avg_recall}");
    }

    #[test]
    fn higher_ef_search_never_decreases_recall_much() {
        // Not a strict monotonicity guarantee (HNSW is approximate), but
        // a much higher ef_search should not perform meaningfully worse
        // than a low one -- if it does, something's wrong with the search.
        let mut rng = StdRng::seed_from_u64(4);
        let mut index = HnswIndex::new(HnswParams::default());
        let vectors = clustered_vectors(1000, 16, 8, 11);
        for v in &vectors {
            index.insert(v.clone(), &mut rng);
        }
        let queries = clustered_vectors(30, 16, 8, 12);

        let recall_for_ef = |ef: usize| -> f64 {
            let mut recalls = Vec::new();
            for q in &queries {
                let approx = index.search(q, 10, ef);
                let exact = index.brute_force(q, 10);
                recalls.push(recall_at_k(&approx, &exact, 10));
            }
            recalls.iter().sum::<f64>() / recalls.len() as f64
        };

        let low = recall_for_ef(10);
        let high = recall_for_ef(100);
        assert!(high >= low - 0.05, "higher ef_search recall ({high}) should not be much worse than low ef_search ({low})");
    }
}
