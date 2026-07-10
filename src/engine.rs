//! Engine: the write/read path that ties WAL + MemTable together.
//!
//! This is Phase 0's whole point: prove that (a) writes are durable
//! across a crash, and (b) reads reflect all acknowledged writes.
//! SSTable flush, HNSW indexing, and query fusion all build on top of
//! this in later phases -- nothing here is optimized yet, it's the
//! correctness baseline everything else gets measured against.

use crate::memtable::MemTable;
use crate::record::{Record, RecordId};
use crate::wal::{Wal, WalError};
use std::collections::HashMap;
use std::path::Path;
use std::sync::atomic::{AtomicU64, Ordering};

pub struct Engine {
    wal: Wal,
    memtable: MemTable,
    seq: AtomicU64,
}

impl Engine {
    /// Open the engine at `path`, replaying the WAL to reconstruct the
    /// memtable if a previous instance crashed or was shut down.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, WalError> {
        let wal = Wal::open(path)?;
        let mut memtable = MemTable::new();
        let mut max_seq = 0u64;

        for record in wal.replay()? {
            max_seq = max_seq.max(record.seq);
            memtable.insert(record);
        }

        Ok(Self {
            wal,
            memtable,
            seq: AtomicU64::new(max_seq),
        })
    }

    fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::SeqCst) + 1
    }

    /// Insert a record. Durable once this returns Ok (WAL fsync'd first,
    /// memtable updated second -- so a crash between the two just means
    /// replay reconstructs the memtable identically on next open).
    pub fn put(
        &mut self,
        id: RecordId,
        vector: Vec<f32>,
        metadata: HashMap<String, String>,
    ) -> Result<(), WalError> {
        let record = Record::new(id, vector, metadata, self.next_seq());
        self.wal.append(&record)?;
        self.memtable.insert(record);
        Ok(())
    }

    pub fn delete(&mut self, id: RecordId) -> Result<(), WalError> {
        let seq = self.next_seq();
        self.wal.append(&Record::tombstone(id, seq))?;
        self.memtable.delete(id, seq);
        Ok(())
    }

    pub fn get(&self, id: RecordId) -> Option<&Record> {
        self.memtable.get(id)
    }

    /// Brute-force scan of all live records. This is the Phase 0
    /// stand-in for vector search -- correctness reference for Phase 2's
    /// HNSW index (its results must match this exactly, just faster).
    pub fn scan_live(&self) -> impl Iterator<Item = &Record> {
        self.memtable.iter_live()
    }

    pub fn len(&self) -> usize {
        self.memtable.len()
    }

    pub fn is_empty(&self) -> bool {
        self.memtable.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_get_delete_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");
        let mut engine = Engine::open(&path).unwrap();

        engine.put(1, vec![1.0, 2.0], HashMap::new()).unwrap();
        assert!(engine.get(1).is_some());

        engine.delete(1).unwrap();
        assert!(engine.get(1).is_none());
    }

    #[test]
    fn recovers_state_after_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");

        {
            let mut engine = Engine::open(&path).unwrap();
            engine.put(1, vec![1.0], HashMap::new()).unwrap();
            engine.put(2, vec![2.0], HashMap::new()).unwrap();
            engine.delete(2).unwrap();
            // engine dropped here -- simulates process exit
        }

        let engine = Engine::open(&path).unwrap();
        assert!(engine.get(1).is_some());
        assert!(engine.get(2).is_none()); // tombstone replayed correctly
        assert_eq!(engine.len(), 2); // tombstone still occupies a slot
    }

    #[test]
    fn seq_counter_continues_after_reopen() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("engine.wal");

        {
            let mut engine = Engine::open(&path).unwrap();
            engine.put(1, vec![1.0], HashMap::new()).unwrap();
        }
        {
            let mut engine = Engine::open(&path).unwrap();
            // A stale write with an old seq must not overwrite the newer one.
            engine.put(1, vec![9.0], HashMap::new()).unwrap();
            assert_eq!(engine.get(1).unwrap().vector, vec![9.0]);
        }
    }
}
