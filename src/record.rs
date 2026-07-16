//! Core record type: the atomic unit stored in NeuraStore.
//!
//! Each record couples a structured metadata payload (row-oriented,
//! for filtering) with a dense vector payload (columnar-friendly,
//! for ANN search). Phase 0 keeps both in one struct; Phase 1 will
//! split storage physically while keeping this as the logical view.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

pub type RecordId = u64;

/// A single metadata value. String-only metadata (Phases 0-9) meant
/// filtering could only ever be exact-match equality on text -- no
/// numeric range queries ("price > 100"), no boolean flags. This is
/// Phase 10's extension: string, number (f64 -- covers both integers
/// and floats without a separate variant, matching how JSON itself
/// represents numbers), and bool.
///
/// **Deliberately a normal, tagged enum, not `#[serde(untagged)]`** --
/// verified empirically before choosing this: an untagged version
/// serializes fine via bincode but fails on every single deserialize
/// with `DeserializeAnyNotSupported`, since bincode is a
/// non-self-describing format and untagged deserialization needs to
/// try each variant in turn, which a sequential byte reader can't do.
/// The natural-JSON ergonomics (`"docs"`, `29.99`, `true` instead of a
/// tagged wrapper) are handled explicitly at the HTTP boundary instead
/// (see `src/bin/server.rs`), not pushed down into this type -- this
/// type only needs to round-trip correctly through bincode, since it's
/// never JSON-serialized directly.
///
/// **A real, deliberate breaking change, not an oversight**: this
/// changes the on-disk binary format for stored metadata regardless of
/// tagged vs. untagged. Existing SSTable files written before this
/// change will not deserialize correctly afterward. Made now, not
/// later, specifically because NeuraStore has no real production
/// deployments yet -- this is the right time to make a storage-format
/// change, before anyone has real data depending on the old one. See
/// HISTORY.md's Phase 10 section.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum MetadataValue {
    String(String),
    Number(f64),
    Bool(bool),
}

impl MetadataValue {
    /// A canonical string form, used as the `field_index` lookup key so
    /// the existing fast-path equality machinery (built for string-only
    /// metadata) keeps working unchanged for every value type, instead
    /// of needing a redesigned multi-type index structure. Deliberately
    /// distinguishes the three types even when their canonical text
    /// would otherwise collide -- e.g. the string `"true"` and the bool
    /// `true` must not be treated as equal, so each gets a distinct
    /// type-tagged prefix rather than sharing bare text.
    pub fn canonical_key(&self) -> String {
        match self {
            MetadataValue::String(s) => format!("s:{s}"),
            MetadataValue::Number(n) => format!("n:{n}"),
            MetadataValue::Bool(b) => format!("b:{b}"),
        }
    }

    /// The numeric value, if this is a Number -- used by range filters
    /// (`Gt`/`Gte`/`Lt`/`Lte`), which are only meaningful against
    /// numbers. Returns `None` for String/Bool rather than panicking or
    /// attempting a lossy conversion.
    pub fn as_number(&self) -> Option<f64> {
        match self {
            MetadataValue::Number(n) => Some(*n),
            _ => None,
        }
    }

    /// Rough in-memory size estimate, for the memtable's flush-threshold
    /// tracking (`memtable.rs`) -- doesn't need to be exact, just
    /// proportionate, same as the rest of that size estimate already is.
    pub fn approx_size(&self) -> usize {
        match self {
            MetadataValue::String(s) => s.len(),
            MetadataValue::Number(_) => std::mem::size_of::<f64>(),
            MetadataValue::Bool(_) => std::mem::size_of::<bool>(),
        }
    }
}

impl From<&str> for MetadataValue {
    fn from(s: &str) -> Self {
        MetadataValue::String(s.to_string())
    }
}

impl From<String> for MetadataValue {
    fn from(s: String) -> Self {
        MetadataValue::String(s)
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Record {
    pub id: RecordId,
    pub vector: Vec<f32>,
    pub metadata: HashMap<String, MetadataValue>,
    /// Monotonic write timestamp (logical clock for MVCC-lite reads).
    pub seq: u64,
    /// Tombstone marker for deletes (LSM-style: deletes are writes).
    pub deleted: bool,
}

impl Record {
    pub fn new(id: RecordId, vector: Vec<f32>, metadata: HashMap<String, MetadataValue>, seq: u64) -> Self {
        Self { id, vector, metadata, seq, deleted: false }
    }

    pub fn tombstone(id: RecordId, seq: u64) -> Self {
        Self { id, vector: Vec::new(), metadata: HashMap::new(), seq, deleted: true }
    }
}
