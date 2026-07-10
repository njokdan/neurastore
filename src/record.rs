//! Core record type: the atomic unit stored in NeuraStore.
//!
//! Each record couples a structured metadata payload (row-oriented,
//! for filtering) with a dense vector payload (columnar-friendly,
//! for ANN search). Phase 0 keeps both in one struct; Phase 1 will
//! split storage physically while keeping this as the logical view.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type RecordId = u64;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub id: RecordId,
    pub vector: Vec<f32>,
    pub metadata: HashMap<String, String>,
    /// Monotonic write timestamp (logical clock for MVCC-lite reads).
    pub seq: u64,
    /// Tombstone marker for deletes (LSM-style: deletes are writes).
    pub deleted: bool,
}

impl Record {
    pub fn new(id: RecordId, vector: Vec<f32>, metadata: HashMap<String, String>, seq: u64) -> Self {
        Self { id, vector, metadata, seq, deleted: false }
    }

    pub fn tombstone(id: RecordId, seq: u64) -> Self {
        Self { id, vector: Vec::new(), metadata: HashMap::new(), seq, deleted: true }
    }
}
