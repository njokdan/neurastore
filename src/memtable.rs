//! MemTable: the in-memory, sorted write buffer.
//!
//! All writes land here first (after the WAL). Reads check the memtable
//! before falling through to on-disk SSTables (SSTables arrive in Phase 1).
//! A BTreeMap keeps keys sorted, which Phase 1's flush-to-SSTable step
//! depends on (SSTables are written in sorted order for range scans and
//! future compaction).

use crate::record::{Record, RecordId};
use std::collections::BTreeMap;

#[derive(Default)]
pub struct MemTable {
    inner: BTreeMap<RecordId, Record>,
    /// Approximate size in bytes, used later to decide when to flush.
    approx_bytes: usize,
}

impl MemTable {
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert or overwrite a record. Last-writer-wins by `seq`, so an
    /// out-of-order replay (shouldn't happen, but defend anyway) doesn't
    /// silently regress a later write.
    pub fn insert(&mut self, record: Record) {
        self.approx_bytes += Self::estimate_size(&record);
        match self.inner.get(&record.id) {
            Some(existing) if existing.seq > record.seq => {
                // Stale write, ignore.
            }
            _ => {
                self.inner.insert(record.id, record);
            }
        }
    }

    pub fn delete(&mut self, id: RecordId, seq: u64) {
        self.insert(Record::tombstone(id, seq));
    }

    /// Get a live (non-tombstoned) record. Returns None if absent or deleted.
    pub fn get(&self, id: RecordId) -> Option<&Record> {
        self.inner.get(&id).filter(|r| !r.deleted)
    }

    /// Iterate all live records in key order. Used by Phase 1's flush
    /// path and by brute-force scans until the vector index exists.
    pub fn iter_live(&self) -> impl Iterator<Item = &Record> {
        self.inner.values().filter(|r| !r.deleted)
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    pub fn approx_size_bytes(&self) -> usize {
        self.approx_bytes
    }

    fn estimate_size(record: &Record) -> usize {
        std::mem::size_of::<RecordId>()
            + record.vector.len() * std::mem::size_of::<f32>()
            + record
                .metadata
                .iter()
                .map(|(k, v)| k.len() + v.len())
                .sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn rec(id: u64, seq: u64) -> Record {
        Record::new(id, vec![0.0; 4], HashMap::new(), seq)
    }

    #[test]
    fn insert_and_get() {
        let mut mt = MemTable::new();
        mt.insert(rec(1, 1));
        assert!(mt.get(1).is_some());
        assert!(mt.get(2).is_none());
    }

    #[test]
    fn delete_shadows_read() {
        let mut mt = MemTable::new();
        mt.insert(rec(1, 1));
        mt.delete(1, 2);
        assert!(mt.get(1).is_none());
        assert_eq!(mt.len(), 1); // tombstone still occupies the slot
    }

    #[test]
    fn out_of_order_write_does_not_regress() {
        let mut mt = MemTable::new();
        mt.insert(rec(1, 5));
        mt.insert(rec(1, 2)); // stale, should be ignored
        assert_eq!(mt.get(1).unwrap().seq, 5);
    }

    #[test]
    fn iter_live_skips_tombstones() {
        let mut mt = MemTable::new();
        mt.insert(rec(1, 1));
        mt.insert(rec(2, 1));
        mt.delete(2, 2);
        let live: Vec<_> = mt.iter_live().map(|r| r.id).collect();
        assert_eq!(live, vec![1]);
    }
}
