//! VectorIndex: bridges the engine's RecordId space to HNSW's internal,
//! dense (0..n) node id space, and owns the build-from-corpus entry
//! point Phase 2 needs.
//!
//! Phase 2 scope: `build()` is a one-shot, all-at-once construction from
//! a snapshot of live records. There's no incremental "add one record to
//! an existing index" API yet at the engine level -- that's Phase 3's
//! job, along with making it safe to query while it's happening. Calling
//! `build()` again after further writes fully replaces the index; it
//! does not update it in place.

use crate::hnsw::{HnswIndex, HnswParams};
use crate::record::{Record, RecordId};
use rand::rngs::StdRng;
use rand::SeedableRng;

pub struct VectorIndex {
    hnsw: HnswIndex,
    /// internal HNSW node id -> RecordId, so search results can be
    /// translated back to the ids callers (the engine, and beyond it,
    /// users) actually know about.
    id_map: Vec<RecordId>,
    dim: Option<usize>,
}

impl VectorIndex {
    /// Build a fresh index from a snapshot of records. Records with
    /// mismatched vector dimension are skipped with the assumption
    /// (documented, not yet enforced elsewhere) that a NeuraStore
    /// collection has a single fixed dimension -- Phase 2 doesn't add
    /// schema validation at write time yet.
    pub fn build(records: &[Record], params: HnswParams, seed: u64) -> Self {
        let mut hnsw = HnswIndex::new(params);
        let mut id_map = Vec::with_capacity(records.len());
        let mut rng = StdRng::seed_from_u64(seed);
        let mut dim: Option<usize> = None;

        for record in records {
            if record.vector.is_empty() {
                continue;
            }
            match dim {
                None => dim = Some(record.vector.len()),
                Some(d) if d != record.vector.len() => continue, // skip mismatched dim
                _ => {}
            }
            hnsw.insert(record.vector.clone(), &mut rng);
            id_map.push(record.id);
        }

        Self { hnsw, id_map, dim }
    }

    pub fn len(&self) -> usize {
        self.hnsw.len()
    }

    pub fn is_empty(&self) -> bool {
        self.hnsw.is_empty()
    }

    pub fn dim(&self) -> Option<usize> {
        self.dim
    }

    /// Approximate k-NN search, translated back to RecordIds. Distances
    /// returned are squared L2 (matches the pgvector/Milvus baseline
    /// metric) -- take the square root if a true Euclidean distance is
    /// needed for display.
    pub fn search(&self, query: &[f32], k: usize, ef_search: usize) -> Vec<(RecordId, f32)> {
        self.hnsw
            .search(query, k, ef_search)
            .into_iter()
            .map(|(internal_id, dist)| (self.id_map[internal_id], dist))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

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
        // The two vectors near the origin (100, 300) should be the
        // nearest neighbors, not 200 which is far away.
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
}
