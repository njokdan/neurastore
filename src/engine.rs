//! Engine: the write/read path tying WAL + MemTable + SSTables together.
//!
//! Phase 1 adds the actual LSM behavior on top of Phase 0's durability
//! baseline: once the memtable grows past a threshold (or `flush()` is
//! called explicitly), its contents -- live records AND tombstones --
//! are written out to an immutable SSTable file, and the memtable/WAL
//! are cleared. Reads now have to check multiple places, newest first:
//! memtable -> newest SSTable -> ... -> oldest SSTable. First match
//! wins; a tombstone match means "definitely deleted," not "keep
//! looking," or a delete would resurrect once the memtable clears.

use crate::memtable::MemTable;
use crate::record::{Record, RecordId};
use crate::sstable::{self, SSTableError, SSTableReader};
use crate::vector_index::VectorIndex;
use crate::hnsw::HnswParams;
use crate::wal::{Wal, WalError};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

#[derive(thiserror::Error, Debug)]
pub enum EngineError {
    #[error("wal error: {0}")]
    Wal(#[from] WalError),
    #[error("sstable error: {0}")]
    SSTable(#[from] SSTableError),
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
}

/// Flush the memtable once it holds this many records. A byte-size
/// threshold would be more realistic for production, but a count
/// threshold is simpler to reason about and test deterministically for
/// Phase 1's correctness goal.
const DEFAULT_FLUSH_THRESHOLD: usize = 10_000;

pub struct Engine {
    dir: PathBuf,
    wal: Wal,
    memtable: MemTable,
    /// Oldest first, newest last -- so reversing gives newest-first,
    /// which is the order reads need to check in.
    sstables: Vec<SSTableReader>,
    next_sstable_id: u64,
    seq: AtomicU64,
    flush_threshold: usize,
    /// Phase 2: static vector index, built on demand via `build_index()`
    /// from a snapshot of `scan_live()`. None until the first call, and
    /// silently stale after further writes until rebuilt -- Phase 3
    /// replaces this with an index that updates incrementally instead.
    vector_index: Option<VectorIndex>,
}

impl Engine {
    /// Open (or create) an engine rooted at `dir`. Layout on disk:
    ///   dir/wal.log       -- the write-ahead log
    ///   dir/000001.sst, dir/000002.sst, ... -- SSTables, oldest to newest
    ///
    /// Recovery order matters: SSTables are loaded first (they're the
    /// durable, already-flushed state), then the WAL is replayed on top
    /// (it holds whatever was written since the last flush, so it must
    /// win on conflicts -- MemTable::insert's last-writer-wins-by-seq
    /// handles that naturally since WAL replay always has the higher seq).
    pub fn open<P: AsRef<Path>>(dir: P) -> Result<Self, EngineError> {
        let dir = dir.as_ref().to_path_buf();
        fs::create_dir_all(&dir)?;

        let mut sstable_paths: Vec<PathBuf> = fs::read_dir(&dir)?
            .filter_map(|e| e.ok())
            .map(|e| e.path())
            .filter(|p| p.extension().map(|ext| ext == "sst").unwrap_or(false))
            .collect();
        sstable_paths.sort(); // filenames are zero-padded, so lexicographic == numeric order

        let mut sstables = Vec::with_capacity(sstable_paths.len());
        let mut next_sstable_id = 1u64;
        for path in &sstable_paths {
            sstables.push(SSTableReader::open(path)?);
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                if let Ok(n) = stem.parse::<u64>() {
                    next_sstable_id = next_sstable_id.max(n + 1);
                }
            }
        }

        let wal = Wal::open(dir.join("wal.log"))?;
        let mut memtable = MemTable::new();
        let mut max_seq = 0u64;
        for record in wal.replay()? {
            max_seq = max_seq.max(record.seq);
            memtable.insert(record);
        }
        // Also account for seq numbers already flushed into SSTables, so
        // a fresh memtable after restart doesn't reuse an old seq.
        for sst in &sstables {
            for record in sst.iter() {
                max_seq = max_seq.max(record.seq);
            }
        }

        Ok(Self {
            dir,
            wal,
            memtable,
            sstables,
            next_sstable_id,
            seq: AtomicU64::new(max_seq),
            flush_threshold: DEFAULT_FLUSH_THRESHOLD,
            vector_index: None,
        })
    }

    #[cfg(test)]
    fn with_flush_threshold(mut self, threshold: usize) -> Self {
        self.flush_threshold = threshold;
        self
    }

    fn next_seq(&self) -> u64 {
        self.seq.fetch_add(1, Ordering::SeqCst) + 1
    }

    pub fn put(
        &mut self,
        id: RecordId,
        vector: Vec<f32>,
        metadata: HashMap<String, String>,
    ) -> Result<(), EngineError> {
        let record = Record::new(id, vector, metadata, self.next_seq());
        self.wal.append(&record)?;
        self.memtable.insert(record);
        self.maybe_flush()?;
        Ok(())
    }

    /// Insert many records with a single WAL fsync for the whole batch,
    /// instead of one per record. Meant for bulk loads (initial data
    /// import, `bin/bench_neurastore`, batch APIs) where the caller
    /// already treats the whole set as one logical unit of work -- see
    /// `Wal::append_batch`'s docs for the durability tradeoff this makes
    /// (all-or-nothing on a crash mid-batch, not per-record). For
    /// interactive single writes where each one needs its own durability
    /// guarantee the instant it returns, use `put` instead.
    pub fn put_batch(
        &mut self,
        entries: Vec<(RecordId, Vec<f32>, HashMap<String, String>)>,
    ) -> Result<(), EngineError> {
        let records: Vec<Record> = entries
            .into_iter()
            .map(|(id, vector, metadata)| Record::new(id, vector, metadata, self.next_seq()))
            .collect();
        self.wal.append_batch(&records)?;
        for record in records {
            self.memtable.insert(record);
        }
        self.maybe_flush()?;
        Ok(())
    }

    pub fn delete(&mut self, id: RecordId) -> Result<(), EngineError> {
        let seq = self.next_seq();
        self.wal.append(&Record::tombstone(id, seq))?;
        self.memtable.delete(id, seq);
        self.maybe_flush()?;
        Ok(())
    }

    /// Point lookup: memtable first (it's always the newest data), then
    /// SSTables newest-to-oldest. Stops at the first match, since a
    /// match -- live or tombstone -- is definitive: it's the most recent
    /// write for that id and nothing older can override it.
    pub fn get(&self, id: RecordId) -> Option<Record> {
        if let Some(record) = self.memtable.get_raw(id) {
            return if record.deleted { None } else { Some(record.clone()) };
        }
        for sst in self.sstables.iter().rev() {
            if let Some(record) = sst.get(id) {
                return if record.deleted { None } else { Some(record) };
            }
        }
        None
    }

    /// Full scan of all live records, correctly merged across every
    /// storage level. Phase 1 prioritizes correctness over performance
    /// here -- this is the reference implementation Phase 2's HNSW
    /// index results get checked against, not the fast path itself.
    pub fn scan_live(&self) -> Vec<Record> {
        let mut merged: BTreeMap<RecordId, Record> = BTreeMap::new();
        // Oldest SSTable first, so each subsequent layer correctly
        // overwrites the previous one; memtable (newest) applied last.
        for sst in &self.sstables {
            for record in sst.iter() {
                merged.insert(record.id, record);
            }
        }
        for record in self.memtable.iter_all() {
            merged.insert(record.id, record.clone());
        }
        merged.into_values().filter(|r| !r.deleted).collect()
    }

    /// Number of live, non-deleted records across the whole engine
    /// (memtable + all SSTables, deduplicated). O(n) -- built on
    /// `scan_live`, not tracked incrementally, since Phase 1 hasn't
    /// needed a cheaper answer yet.
    pub fn len(&self) -> usize {
        self.scan_live().len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn memtable_len(&self) -> usize {
        self.memtable.len()
    }

    pub fn sstable_count(&self) -> usize {
        self.sstables.len()
    }

    /// Build (or rebuild) the vector index from a fresh snapshot of all
    /// live records. Phase 2 is deliberately explicit about this instead
    /// of auto-rebuilding on every write -- that would make writes pay
    /// an unpredictable, corpus-size-dependent cost. Call this once
    /// after loading/writing a batch, before querying.
    pub fn build_index(&mut self) {
        self.build_index_with_params(HnswParams::default(), 42);
    }

    pub fn build_index_with_params(&mut self, params: HnswParams, seed: u64) {
        let records = self.scan_live();
        self.vector_index = Some(VectorIndex::build(&records, params, seed));
    }

    pub fn has_index(&self) -> bool {
        self.vector_index.is_some()
    }

    pub fn index_len(&self) -> Option<usize> {
        self.vector_index.as_ref().map(|idx| idx.len())
    }

    /// Approximate k-nearest-neighbor search against the built index.
    /// Returns `None` if `build_index()` hasn't been called yet --
    /// callers should treat that as "index not ready," not "no results."
    pub fn search_knn(&self, query: &[f32], k: usize, ef_search: usize) -> Option<Vec<(RecordId, f32)>> {
        self.vector_index.as_ref().map(|idx| idx.search(query, k, ef_search))
    }

    fn maybe_flush(&mut self) -> Result<(), EngineError> {
        if self.memtable.len() >= self.flush_threshold {
            self.flush()?;
        }
        Ok(())
    }

    /// Flush the current memtable to a new immutable SSTable, then clear
    /// the memtable and WAL. Order matters for crash safety: the
    /// SSTable is fsync'd (via write_sstable's write-to-temp-then-rename)
    /// *before* the WAL is cleared -- so a crash between those two steps
    /// just means replay redoes a flush that already durably happened,
    /// which is safe (SSTable filenames are unique, and the new memtable
    /// after replay would flush to the next id), rather than losing data.
    pub fn flush(&mut self) -> Result<(), EngineError> {
        if self.memtable.is_empty() {
            return Ok(());
        }
        let records: Vec<Record> = self.memtable.iter_all().cloned().collect();
        let filename = format!("{:06}.sst", self.next_sstable_id);
        let path = self.dir.join(&filename);

        sstable::write_sstable(&path, &records)?;
        self.sstables.push(SSTableReader::open(&path)?);
        self.next_sstable_id += 1;

        self.memtable.clear();
        self.wal.clear()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn put_get_delete_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap();

        engine.put(1, vec![1.0, 2.0], HashMap::new()).unwrap();
        assert!(engine.get(1).is_some());

        engine.delete(1).unwrap();
        assert!(engine.get(1).is_none());
    }

    #[test]
    fn recovers_state_after_reopen_without_flush() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut engine = Engine::open(dir.path()).unwrap();
            engine.put(1, vec![1.0], HashMap::new()).unwrap();
            engine.put(2, vec![2.0], HashMap::new()).unwrap();
            engine.delete(2).unwrap();
        }
        let engine = Engine::open(dir.path()).unwrap();
        assert!(engine.get(1).is_some());
        assert!(engine.get(2).is_none());
        assert_eq!(engine.len(), 1);
        assert_eq!(engine.sstable_count(), 0); // never flushed
    }

    #[test]
    fn manual_flush_moves_data_to_sstable_and_clears_memtable() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap();
        for i in 1..=5 {
            engine.put(i, vec![i as f32], HashMap::new()).unwrap();
        }
        assert_eq!(engine.memtable_len(), 5);
        assert_eq!(engine.sstable_count(), 0);

        engine.flush().unwrap();

        assert_eq!(engine.memtable_len(), 0);
        assert_eq!(engine.sstable_count(), 1);
        assert_eq!(engine.len(), 5); // still readable, now from the sstable
        assert!(engine.get(3).is_some());
    }

    #[test]
    fn data_survives_restart_after_flush() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut engine = Engine::open(dir.path()).unwrap();
            for i in 1..=5 {
                engine.put(i, vec![i as f32], HashMap::new()).unwrap();
            }
            engine.flush().unwrap();
        }
        // Reopen from a totally fresh Engine -- no in-memory state carried over.
        let engine = Engine::open(dir.path()).unwrap();
        assert_eq!(engine.sstable_count(), 1);
        assert_eq!(engine.len(), 5);
        assert_eq!(engine.get(3).unwrap().vector, vec![3.0]);
    }

    #[test]
    fn newer_memtable_write_shadows_older_sstable_value() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap();
        engine.put(1, vec![1.0], HashMap::new()).unwrap();
        engine.flush().unwrap();

        // Overwrite the same id after the flush -- this write lives only
        // in the memtable/WAL now, and must win over the flushed value.
        engine.put(1, vec![99.0], HashMap::new()).unwrap();
        assert_eq!(engine.get(1).unwrap().vector, vec![99.0]);
        assert_eq!(engine.len(), 1); // still one logical record, not two
    }

    #[test]
    fn delete_after_flush_shadows_older_sstable_value() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap();
        engine.put(1, vec![1.0], HashMap::new()).unwrap();
        engine.flush().unwrap();

        engine.delete(1).unwrap();
        assert!(engine.get(1).is_none());
        assert_eq!(engine.len(), 0);

        // And the delete itself must survive a second flush + restart.
        engine.flush().unwrap();
        drop(engine);
        let engine = Engine::open(dir.path()).unwrap();
        assert!(engine.get(1).is_none());
    }

    #[test]
    fn multiple_flushes_produce_multiple_sstables_and_all_are_readable() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap();

        engine.put(1, vec![1.0], HashMap::new()).unwrap();
        engine.flush().unwrap();
        engine.put(2, vec![2.0], HashMap::new()).unwrap();
        engine.flush().unwrap();
        engine.put(3, vec![3.0], HashMap::new()).unwrap();
        // leave record 3 in the memtable, unflushed

        assert_eq!(engine.sstable_count(), 2);
        assert_eq!(engine.memtable_len(), 1);
        assert_eq!(engine.len(), 3);
        assert!(engine.get(1).is_some());
        assert!(engine.get(2).is_some());
        assert!(engine.get(3).is_some());
    }

    #[test]
    fn automatic_flush_triggers_at_threshold() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap().with_flush_threshold(3);

        for i in 1..=3 {
            engine.put(i, vec![i as f32], HashMap::new()).unwrap();
        }
        // The 3rd put should have crossed the threshold and triggered a flush.
        assert_eq!(engine.sstable_count(), 1);
        assert_eq!(engine.memtable_len(), 0);
        assert_eq!(engine.len(), 3);
    }

    #[test]
    fn put_batch_inserts_all_records_and_recovers_after_restart() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut engine = Engine::open(dir.path()).unwrap();
            let entries: Vec<(RecordId, Vec<f32>, HashMap<String, String>)> = (1..=200)
                .map(|i| (i, vec![i as f32], HashMap::new()))
                .collect();
            engine.put_batch(entries).unwrap();
            assert_eq!(engine.len(), 200);
        }
        // Restart -- batch must have been fsync'd durably, same as
        // individual puts, just with one fsync instead of 200.
        let engine = Engine::open(dir.path()).unwrap();
        assert_eq!(engine.len(), 200);
        assert_eq!(engine.get(100).unwrap().vector, vec![100.0]);
    }

    #[test]
    fn put_batch_is_meaningfully_faster_than_individual_puts() {
        let dir = tempfile::tempdir().unwrap();

        let individual_start = std::time::Instant::now();
        {
            let mut engine = Engine::open(dir.path().join("individual")).unwrap();
            for i in 1..=300u64 {
                engine.put(i, vec![i as f32], HashMap::new()).unwrap();
            }
        }
        let individual_elapsed = individual_start.elapsed();

        let batch_start = std::time::Instant::now();
        {
            let mut engine = Engine::open(dir.path().join("batch")).unwrap();
            let entries: Vec<(RecordId, Vec<f32>, HashMap<String, String>)> =
                (1..=300).map(|i| (i, vec![i as f32], HashMap::new())).collect();
            engine.put_batch(entries).unwrap();
        }
        let batch_elapsed = batch_start.elapsed();

        assert!(
            batch_elapsed < individual_elapsed,
            "expected put_batch ({batch_elapsed:?}) to meaningfully beat {} individual puts ({individual_elapsed:?})",
            300
        );
    }

    #[test]
    fn scales_to_100k_records_and_reads_back_correctly() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap().with_flush_threshold(10_000);

        for i in 0..100_000u64 {
            engine
                .put(i, vec![i as f32, (i % 7) as f32], HashMap::from([
                    ("category".to_string(), format!("cat{}", i % 4)),
                ]))
                .unwrap();
        }
        engine.flush().unwrap(); // flush whatever's left under the threshold

        assert_eq!(engine.len(), 100_000);
        assert!(engine.sstable_count() >= 10);

        // Spot-check a handful of ids across the range, not just the edges.
        for id in [0u64, 1, 9_999, 10_000, 50_000, 99_999] {
            let record = engine.get(id).expect("record should exist");
            assert_eq!(record.id, id);
            assert_eq!(record.vector[0], id as f32);
        }
        assert!(engine.get(100_000).is_none());
    }
}
