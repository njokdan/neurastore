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
//!
//! Phase 3: the vector index (`vector_index: Option<Arc<RwLock<VectorIndex>>>`)
//! is now kept incrementally in sync with every `put`/`put_batch`/`delete`
//! once it's been built once -- no more full `build_index()` rebuild
//! required after the first call. The `Arc<RwLock<_>>` wrapper is what
//! makes concurrent access safe: `index_handle()` hands out a cloneable,
//! independently lockable reference so multiple threads can hold read
//! locks for `search` simultaneously, while a writer briefly takes a
//! write lock to insert or delete. Coarse-grained (a writer blocks all
//! readers for the duration of one graph insert), not lock-free -- see
//! `vector_index.rs` for why that tradeoff was chosen deliberately.

use crate::hnsw::HnswParams;
use crate::memtable::MemTable;
use crate::record::{MetadataValue, Record, RecordId};
use crate::sstable::{self, SSTableError, SSTableReader};
use crate::vector_index::{FilterOp, VectorIndex};
use crate::wal::{Wal, WalError};
use std::collections::{BTreeMap, HashMap};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

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
    /// Phase 3: incrementally-maintained vector index. `None` until the
    /// first `build_index()` call; after that, every `put`/`put_batch`/
    /// `delete` updates it in place (see those methods) instead of
    /// requiring a rebuild. `Arc<RwLock<_>>` so `index_handle()` can hand
    /// out a reference safe to use concurrently from other threads.
    vector_index: Option<Arc<RwLock<VectorIndex>>>,
    /// Params/seed used for the most recent `build_index_with_params()`
    /// call, so `compact()` can rebuild the index (if one exists) with
    /// the same configuration instead of silently reverting to defaults
    /// -- a real gap that would otherwise bite anyone who'd built their
    /// index with custom params before ever calling `compact()`.
    index_params: Option<(HnswParams, u64)>,
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
            index_params: None,
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
        metadata: HashMap<String, MetadataValue>,
    ) -> Result<(), EngineError> {
        let record = Record::new(id, vector, metadata, self.next_seq());
        self.wal.append(&record)?;
        // Keep the vector index in sync incrementally, if one exists --
        // this is the Phase 3 change: no more "call build_index() again
        // after every write." A fresh id extends the graph; an id that
        // already existed there is treated as an update (the insert
        // implicitly clears any prior tombstone -- see vector_index.rs).
        if let Some(index) = &self.vector_index {
            index.write().expect("vector index lock poisoned").insert(id, &record.vector, &record.metadata);
        }
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
        entries: Vec<(RecordId, Vec<f32>, HashMap<String, MetadataValue>)>,
    ) -> Result<(), EngineError> {
        let records: Vec<Record> = entries
            .into_iter()
            .map(|(id, vector, metadata)| Record::new(id, vector, metadata, self.next_seq()))
            .collect();
        self.wal.append_batch(&records)?;
        if let Some(index) = &self.vector_index {
            // One write-lock acquisition for the whole batch, not one
            // per record -- same "amortize the lock/lock-adjacent cost"
            // principle as the WAL batching fix, applied to the index.
            let mut guard = index.write().expect("vector index lock poisoned");
            for record in &records {
                guard.insert(record.id, &record.vector, &record.metadata);
            }
        }
        for record in records {
            self.memtable.insert(record);
        }
        self.maybe_flush()?;
        Ok(())
    }

    pub fn delete(&mut self, id: RecordId) -> Result<(), EngineError> {
        let seq = self.next_seq();
        self.wal.append(&Record::tombstone(id, seq))?;
        if let Some(index) = &self.vector_index {
            index.write().expect("vector index lock poisoned").delete(id);
        }
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

    /// Build the vector index from a fresh snapshot of all live records.
    /// After this first call, `put`/`put_batch`/`delete` keep it in sync
    /// incrementally -- calling this again fully replaces it (useful for
    /// periodic re-optimization or reclaiming space from accumulated
    /// tombstones, but not required for correctness after the first call).
    pub fn build_index(&mut self) {
        self.build_index_with_params(HnswParams::default(), 42);
    }

    pub fn build_index_with_params(&mut self, params: HnswParams, seed: u64) {
        let records = self.scan_live();
        self.vector_index = Some(Arc::new(RwLock::new(VectorIndex::build(&records, params, seed))));
        self.index_params = Some((params, seed));
    }

    pub fn has_index(&self) -> bool {
        self.vector_index.is_some()
    }

    pub fn index_len(&self) -> Option<usize> {
        self.vector_index.as_ref().map(|idx| idx.read().expect("vector index lock poisoned").len())
    }

    /// A cloneable, independently-lockable handle to the vector index,
    /// safe to send to another thread and use concurrently with this
    /// `Engine` (whose other parts -- WAL, memtable, SSTables -- are
    /// *not* thread-safe; only the vector index is, as of Phase 3). This
    /// is what makes "query from one thread while inserting from
    /// another" possible: hand the handle to a reader thread, keep
    /// calling `put`/`delete` on the `Engine` (or another handle) from
    /// elsewhere. Returns `None` if `build_index()` hasn't been called yet.
    pub fn index_handle(&self) -> Option<Arc<RwLock<VectorIndex>>> {
        self.vector_index.clone()
    }

    /// Approximate k-nearest-neighbor search against the built index.
    /// Returns `None` if `build_index()` hasn't been called yet --
    /// callers should treat that as "index not ready," not "no results."
    pub fn search_knn(&self, query: &[f32], k: usize, ef_search: usize) -> Option<Vec<(RecordId, f32)>> {
        self.vector_index
            .as_ref()
            .map(|idx| idx.read().expect("vector index lock poisoned").search(query, k, ef_search))
    }

    /// Phase 4: hybrid vector + structured-filter search -- only records
    /// where `metadata[field] == value` are eligible. The predicate is
    /// pushed into the graph traversal (or answered via exact brute
    /// force for highly selective filters), not applied by discarding an
    /// unfiltered top-k after fetching it. See `vector_index.rs` for the
    /// mechanism. Returns `None` if `build_index()` hasn't been called yet.
    pub fn search_knn_filtered(
        &self,
        query: &[f32],
        k: usize,
        ef_search: usize,
        field: &str,
        op: &FilterOp,
    ) -> Option<Vec<(RecordId, f32)>> {
        self.vector_index.as_ref().map(|idx| {
            idx.read()
                .expect("vector index lock poisoned")
                .search_filtered(query, k, ef_search, field, op)
        })
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

    /// Reclaims space accumulated from deletes and updates. Since Phase
    /// 3, every delete or update has left the old data behind forever
    /// (a soft-delete tombstone on disk, a stale node in the HNSW
    /// graph) -- correct, but a long-running server just accumulates
    /// waste indefinitely with nothing to reclaim it. This does both
    /// halves of that in one call:
    ///
    /// 1. **SSTable compaction**: merges every on-disk SSTable into one,
    ///    dropping superseded (updated-over) record versions. Tombstones
    ///    are deliberately KEPT, not dropped, even though the record
    ///    they shadow is gone after a full merge -- this is a crash-
    ///    safety choice, not an oversight. If the process crashes after
    ///    the new compacted file is written but before the old files are
    ///    deleted, `Engine::open()` will load both old and new files and
    ///    merge them by position (newest wins) -- if the tombstone
    ///    weren't there, a deleted record could incorrectly "reappear"
    ///    from an older file in that exact crash window. A tombstone is
    ///    a few bytes; keeping it is a small, permanent cost for a large
    ///    correctness guarantee.
    /// 2. **Index rebuild**: if a vector index already exists, rebuilds
    ///    it from the now-compacted live data, using whichever params
    ///    were last used to build it (not silently reverting to
    ///    defaults) -- this drops stale/tombstoned HNSW graph nodes that
    ///    have been dead weight (still traversed, just filtered from
    ///    results) since whenever they were deleted or superseded.
    ///
    /// Safe to call with 0 or 1 SSTables (a no-op) or no index built yet
    /// (skips that half). Flushes the memtable first if it's non-empty,
    /// so compaction operates on the fullest possible picture.
    pub fn compact(&mut self) -> Result<(), EngineError> {
        self.flush()?;

        if self.sstables.len() > 1 {
            // Merge oldest-to-newest, same ordering rule `scan_live`
            // uses -- a later SSTable's entry for a given id always
            // wins. Unlike `scan_live`, tombstones are kept in the
            // output (see the doc comment above for why).
            let mut merged: BTreeMap<RecordId, Record> = BTreeMap::new();
            for sst in &self.sstables {
                for record in sst.iter() {
                    merged.insert(record.id, record);
                }
            }
            let compacted_records: Vec<Record> = merged.into_values().collect();

            let old_paths: Vec<PathBuf> = self.sstables.iter().map(|s| s.path().to_path_buf()).collect();

            let filename = format!("{:06}.sst", self.next_sstable_id);
            let new_path = self.dir.join(&filename);
            sstable::write_sstable(&new_path, &compacted_records)?;
            self.next_sstable_id += 1;

            // Only after the new file is durably written: swap it in,
            // then clean up the old files. If the process dies before
            // this cleanup finishes, the stale files just linger --
            // safe or retry, not a correctness problem (see doc comment).
            self.sstables = vec![SSTableReader::open(&new_path)?];
            for old_path in old_paths {
                let _ = fs::remove_file(old_path); // best-effort; a leftover file is safe, not silently wrong
            }
        }

        if let Some((params, seed)) = self.index_params {
            self.build_index_with_params(params, seed);
        }

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
            let entries: Vec<(RecordId, Vec<f32>, HashMap<String, MetadataValue>)> = (1..=200)
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
            let entries: Vec<(RecordId, Vec<f32>, HashMap<String, MetadataValue>)> =
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
    fn index_stays_in_sync_incrementally_without_rebuild() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap();

        engine.put(1, vec![0.0, 0.0], HashMap::new()).unwrap();
        engine.put(2, vec![10.0, 10.0], HashMap::new()).unwrap();
        engine.build_index(); // first build, from a snapshot
        assert_eq!(engine.index_len(), Some(2));

        // These happen AFTER build_index() -- the Phase 3 claim is that
        // no second build_index() call is needed for them to show up in
        // search results.
        engine.put(3, vec![0.1, 0.1], HashMap::new()).unwrap();
        engine.put(4, vec![10.1, 10.1], HashMap::new()).unwrap();
        assert_eq!(engine.index_len(), Some(4), "index should have grown incrementally, not stayed at 2");

        let results = engine.search_knn(&[0.0, 0.0], 2, 20).unwrap();
        let ids: Vec<RecordId> = results.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&1));
        assert!(ids.contains(&3), "record 3, added after build_index(), should be findable without a rebuild");
    }

    #[test]
    fn delete_after_build_removes_id_from_search_without_rebuild() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap();

        engine.put(1, vec![0.0, 0.0], HashMap::new()).unwrap();
        engine.put(2, vec![0.01, 0.01], HashMap::new()).unwrap();
        engine.build_index();

        engine.delete(2).unwrap();
        let results = engine.search_knn(&[0.0, 0.0], 5, 20).unwrap();
        let ids: Vec<RecordId> = results.iter().map(|(id, _)| *id).collect();
        assert!(!ids.contains(&2), "deleted record should not appear in search results without a rebuild");
        assert!(ids.contains(&1));
    }

    #[test]
    fn put_batch_after_build_extends_index_in_one_lock_acquisition() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap();
        engine.put(1, vec![0.0, 0.0], HashMap::new()).unwrap();
        engine.build_index();

        let entries: Vec<(RecordId, Vec<f32>, HashMap<String, MetadataValue>)> =
            (2..=50).map(|i| (i, vec![i as f32, i as f32], HashMap::new())).collect();
        engine.put_batch(entries).unwrap();

        assert_eq!(engine.index_len(), Some(50));
    }

    #[test]
    fn concurrent_reads_and_writes_are_actually_thread_safe() {
        // The real Phase 3 concurrency claim, proven with real OS
        // threads (not just "the types happen to allow it"). One writer
        // thread inserts continuously; several reader threads query
        // concurrently the whole time. If anything about the locking
        // were wrong, this would panic (poisoned lock), deadlock (test
        // hangs past cargo's default timeout), or -- if some unsafe
        // shortcut had been taken instead of RwLock -- corrupt data in a
        // way `len()` at the end would catch.
        use std::thread;

        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap();
        // Seed with an initial batch so readers have something to search
        // from the moment threads start, not just an empty index.
        let seed_entries: Vec<(RecordId, Vec<f32>, HashMap<String, MetadataValue>)> =
            (0..200).map(|i| (i, vec![i as f32, (i % 13) as f32], HashMap::new())).collect();
        engine.put_batch(seed_entries).unwrap();
        engine.build_index();

        let handle = engine.index_handle().expect("index should exist after build_index");

        thread::scope(|scope| {
            // 4 concurrent reader threads, searching continuously.
            for _ in 0..4 {
                let reader_handle = handle.clone();
                scope.spawn(move || {
                    for i in 0..300 {
                        let query = vec![(i % 200) as f32, ((i % 200) % 13) as f32];
                        let guard = reader_handle.read().expect("read lock should not be poisoned");
                        let _ = guard.search(&query, 5, 20);
                        // Lock released at end of each iteration -- this
                        // is what lets the writer thread interleave in.
                    }
                });
            }

            // 1 writer thread, inserting new records the whole time the
            // readers are also running.
            let writer_handle = handle.clone();
            scope.spawn(move || {
                for i in 200..400u64 {
                    let mut guard = writer_handle.write().expect("write lock should not be poisoned");
                    guard.insert(i, &[i as f32, (i % 13) as f32], &HashMap::new());
                }
            });
        });
        // thread::scope only returns once every spawned thread has
        // finished -- reaching here at all (no panic, no hang) is most
        // of the proof. The length check below confirms the writes that
        // happened *during* concurrent reading weren't lost or corrupted.

        let final_len = handle.read().unwrap().len();
        assert_eq!(final_len, 400, "expected all 200 seed + 200 concurrently-inserted records to be present");
    }

    #[test]
    fn incremental_and_rebuilt_index_report_consistent_state() {
        // Sanity check that build_index() (full rebuild) and the
        // incremental path agree on final state when applied to the same
        // sequence of writes -- they should, since build_index() just
        // re-derives from the same scan_live() the incremental path was
        // already keeping in sync with.
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap();

        for i in 1..=100u64 {
            engine.put(i, vec![i as f32], HashMap::new()).unwrap();
        }
        engine.build_index();
        for i in 101..=150u64 {
            engine.put(i, vec![i as f32], HashMap::new()).unwrap();
        }
        engine.delete(5).unwrap();
        engine.delete(105).unwrap();

        let incremental_len = engine.index_len().unwrap();

        // Force a full rebuild from the current live state and compare.
        engine.build_index();
        let rebuilt_len = engine.index_len().unwrap();

        assert_eq!(incremental_len, rebuilt_len);
        assert_eq!(rebuilt_len, 148); // 150 inserted - 2 deleted
    }

    #[test]
    fn search_knn_filtered_only_returns_matching_category() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap();

        for i in 0..20u64 {
            let category = if i % 2 == 0 { "docs" } else { "code" };
            engine
                .put(i, vec![i as f32, 0.0], HashMap::from([("category".to_string(), MetadataValue::String(category.to_string()))]))
                .unwrap();
        }
        engine.build_index();

        let results = engine.search_knn_filtered(&[0.0, 0.0], 5, 50, "category", &FilterOp::Eq(MetadataValue::String("docs".to_string()))).unwrap();
        assert!(!results.is_empty());
        for (id, _) in &results {
            assert_eq!(id % 2, 0, "only 'docs' (even ids) should be returned");
        }
    }

    #[test]
    fn search_knn_filtered_stays_correct_after_incremental_writes() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap();

        for i in 0..10u64 {
            engine
                .put(i, vec![i as f32, 0.0], HashMap::from([("category".to_string(), MetadataValue::String("docs".to_string()))]))
                .unwrap();
        }
        engine.build_index();

        // Writes after build_index() -- filtered search must see these
        // without a rebuild, same as unfiltered search already proved in
        // Phase 3.
        engine
            .put(10, vec![10.0, 0.0], HashMap::from([("category".to_string(), MetadataValue::String("docs".to_string()))]))
            .unwrap();
        engine
            .put(11, vec![11.0, 0.0], HashMap::from([("category".to_string(), MetadataValue::String("code".to_string()))]))
            .unwrap();

        let results = engine.search_knn_filtered(&[10.0, 0.0], 5, 50, "category", &FilterOp::Eq(MetadataValue::String("docs".to_string()))).unwrap();
        let ids: Vec<RecordId> = results.iter().map(|(id, _)| *id).collect();
        assert!(ids.contains(&10), "record added after build_index() should be findable via filtered search");
        assert!(!ids.contains(&11), "non-matching record should not appear");
    }

    #[test]
    fn search_knn_filtered_update_correctness() {
        // Direct engine-level check of the Phase 4 bug fix: updating a
        // record's category must make it stop matching its old filter
        // and start matching its new one.
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap();
        engine
            .put(1, vec![0.0, 0.0], HashMap::from([("category".to_string(), MetadataValue::String("docs".to_string()))]))
            .unwrap();
        engine.build_index();

        // Update: same id, different category.
        engine
            .put(1, vec![0.0, 0.0], HashMap::from([("category".to_string(), MetadataValue::String("code".to_string()))]))
            .unwrap();

        let docs_results = engine.search_knn_filtered(&[0.0, 0.0], 5, 50, "category", &FilterOp::Eq(MetadataValue::String("docs".to_string()))).unwrap();
        assert!(docs_results.is_empty(), "id 1 should no longer match 'docs' after being updated to 'code'");

        let code_results = engine.search_knn_filtered(&[0.0, 0.0], 5, 50, "category", &FilterOp::Eq(MetadataValue::String("code".to_string()))).unwrap();
        assert!(code_results.iter().any(|(id, _)| *id == 1), "id 1 should match 'code' after the update");
    }

    #[test]
    fn compact_is_safe_noop_with_zero_or_one_sstables() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap();
        engine.compact().unwrap(); // zero sstables, zero records -- must not error

        engine.put(1, vec![1.0], HashMap::new()).unwrap();
        engine.flush().unwrap();
        assert_eq!(engine.sstable_count(), 1);
        engine.compact().unwrap(); // one sstable -- nothing to merge, must not error
        assert_eq!(engine.get(1).unwrap().vector, vec![1.0]);
    }

    #[test]
    fn compact_merges_multiple_sstables_into_one_and_keeps_all_live_data() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap();

        for i in 1..=5u64 {
            engine.put(i, vec![i as f32], HashMap::new()).unwrap();
            engine.flush().unwrap(); // one sstable per record -- 5 total
        }
        assert_eq!(engine.sstable_count(), 5);

        engine.compact().unwrap();
        assert_eq!(engine.sstable_count(), 1, "compaction should merge everything into a single sstable");
        for i in 1..=5u64 {
            assert_eq!(engine.get(i).unwrap().vector, vec![i as f32]);
        }
        assert_eq!(engine.len(), 5);
    }

    #[test]
    fn compact_drops_superseded_versions_but_keeps_final_state() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap();

        engine.put(1, vec![1.0], HashMap::new()).unwrap();
        engine.flush().unwrap();
        engine.put(1, vec![99.0], HashMap::new()).unwrap(); // update
        engine.flush().unwrap();
        assert_eq!(engine.sstable_count(), 2);

        engine.compact().unwrap();
        assert_eq!(engine.sstable_count(), 1);
        assert_eq!(engine.get(1).unwrap().vector, vec![99.0], "only the final, most recent version should survive compaction");
        assert_eq!(engine.len(), 1, "not two logical records -- the stale version must be gone, not just shadowed");
    }

    #[test]
    fn compact_flushes_pending_memtable_first() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap();
        engine.put(1, vec![1.0], HashMap::new()).unwrap();
        engine.flush().unwrap();

        // This record is deliberately left in the memtable, unflushed,
        // when compact() is called -- it must still end up correctly
        // captured in the compacted output, not silently dropped.
        engine.put(2, vec![2.0], HashMap::new()).unwrap();
        assert_eq!(engine.memtable_len(), 1);

        engine.compact().unwrap();
        assert_eq!(engine.memtable_len(), 0, "compact() should flush the memtable as part of its work");
        assert_eq!(engine.get(2).unwrap().vector, vec![2.0]);
        assert_eq!(engine.len(), 2);
    }

    #[test]
    fn compacted_sstable_retains_tombstone_markers_not_just_absence() {
        // The specific, deliberate crash-safety design choice documented
        // on compact(): a delete's tombstone is kept in the compacted
        // output, not dropped, even though after a full merge the record
        // it shadows is already gone. This test inspects the compacted
        // file's raw contents directly (not just engine.get(), which
        // would look identical whether the tombstone survived or the
        // record was simply never merged in) to confirm the tombstone
        // itself is really there.
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap();
        engine.put(1, vec![1.0], HashMap::new()).unwrap();
        engine.flush().unwrap();
        engine.delete(1).unwrap();
        engine.flush().unwrap();
        assert_eq!(engine.sstable_count(), 2);

        engine.compact().unwrap();
        assert_eq!(engine.sstable_count(), 1);

        // Read the compacted file directly, bypassing the engine's own
        // get()/scan_live() merge logic, to see the raw stored record.
        let compacted_path = engine.dir.join("000003.sst"); // 2 flushes + 1 compacted file = next id is 3
        let reader = sstable::SSTableReader::open(&compacted_path).unwrap();
        let raw = reader.get(1).expect("the tombstone record itself should still be present in the file");
        assert!(raw.deleted, "the record read from the compacted file must be a tombstone, not simply absent");
    }

    #[test]
    fn compact_rebuilds_index_when_one_exists_using_previously_used_params() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap();
        for i in 1..=5u64 {
            engine.put(i, vec![i as f32, 0.0], HashMap::new()).unwrap();
        }
        // Build with deliberately non-default params, to confirm compact()
        // doesn't silently revert to HnswParams::default() on rebuild.
        let custom_params = HnswParams { m: 8, m_max0: 16, ef_construction: 32, metric: crate::hnsw::DistanceMetric::L2 };
        engine.build_index_with_params(custom_params, 7);
        assert_eq!(engine.index_len(), Some(5));

        engine.delete(3).unwrap();
        engine.flush().unwrap();
        engine.compact().unwrap();

        assert!(engine.has_index(), "compact() should not tear down an existing index");
        assert_eq!(engine.index_len(), Some(4), "rebuilt index should reflect the post-compaction live count (5 - 1 deleted)");
        assert!(engine.search_knn(&[3.0, 0.0], 5, 20).unwrap().iter().all(|(id, _)| *id != 3), "deleted record must not reappear in the rebuilt index");
    }

    #[test]
    fn compact_does_not_touch_index_when_none_was_built() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap();
        engine.put(1, vec![1.0], HashMap::new()).unwrap();
        engine.flush().unwrap();
        engine.compact().unwrap();
        assert!(!engine.has_index(), "compact() must not build an index that was never requested");
    }

    #[test]
    fn data_survives_restart_after_compact() {
        let dir = tempfile::tempdir().unwrap();
        {
            let mut engine = Engine::open(dir.path()).unwrap();
            for i in 1..=5u64 {
                engine.put(i, vec![i as f32], HashMap::new()).unwrap();
                engine.flush().unwrap();
            }
            engine.delete(2).unwrap();
            engine.flush().unwrap();
            engine.compact().unwrap();
        }
        // Fresh Engine, no in-memory state carried over -- proves the
        // compacted file on disk is itself correctly durable and readable.
        let engine = Engine::open(dir.path()).unwrap();
        assert_eq!(engine.sstable_count(), 1);
        assert!(engine.get(2).is_none(), "delete must survive both compaction and a restart");
        for i in [1u64, 3, 4, 5] {
            assert_eq!(engine.get(i).unwrap().vector, vec![i as f32]);
        }
        assert_eq!(engine.len(), 4);
    }

    #[test]
    fn scales_to_100k_records_and_reads_back_correctly() {
        let dir = tempfile::tempdir().unwrap();
        let mut engine = Engine::open(dir.path()).unwrap().with_flush_threshold(10_000);

        for i in 0..100_000u64 {
            engine
                .put(i, vec![i as f32, (i % 7) as f32], HashMap::from([
                    ("category".to_string(), MetadataValue::String(format!("cat{}", i % 4))),
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
