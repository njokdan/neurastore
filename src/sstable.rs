//! SSTable: an immutable, sorted, on-disk table produced by flushing a
//! MemTable. This is where the "hybrid row/columnar layout" from the
//! architecture doc physically happens: metadata (structured fields,
//! looked up per-row for filtering) and vectors (looked up as flat
//! float arrays for future ANN scans) are stored in two separate
//! contiguous blobs, not interleaved -- so a future vector scan doesn't
//! have to skip over metadata bytes it doesn't need, and a metadata
//! filter doesn't have to skip over vector bytes it doesn't need.
//!
//! File layout on disk:
//!
//!   [magic: 4 bytes "NSST"]
//!   [index_len: u64 LE][index: bincode(Vec<IndexEntry>)]
//!   [meta_blob_len: u64 LE][meta_blob: concatenated bincode(HashMap<String,String>) per record]
//!   [vector_blob_len: u64 LE][vector_blob: concatenated raw f32 LE bytes per record]
//!
//! `index` is sorted by RecordId (inherited from MemTable's BTreeMap
//! iteration order), which is what lets `get()` binary-search instead
//! of scanning, and is a precondition for range scans and compaction
//! merges later.

use crate::record::{MetadataValue, Record, RecordId};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

const MAGIC: &[u8; 4] = b"NSST";

#[derive(thiserror::Error, Debug)]
pub enum SSTableError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] bincode::Error),
    #[error("bad magic bytes -- not an SSTable file, or corrupted")]
    BadMagic,
    #[error("truncated or corrupted SSTable file")]
    Truncated,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct IndexEntry {
    id: RecordId,
    seq: u64,
    deleted: bool,
    meta_offset: u64,
    meta_len: u32,
    vec_offset: u64,
    vec_len: u32,
}

/// Writes a sorted slice of records (tombstones included -- deletes must
/// survive a flush so they still shadow older SSTables after the
/// memtable that produced them is cleared) to a new SSTable file.
pub fn write_sstable(path: &Path, records: &[Record]) -> Result<(), SSTableError> {
    let mut index = Vec::with_capacity(records.len());
    let mut meta_blob = Vec::new();
    let mut vector_blob = Vec::new();

    for record in records {
        let meta_bytes = bincode::serialize(&record.metadata)?;
        let meta_offset = meta_blob.len() as u64;
        meta_blob.extend_from_slice(&meta_bytes);

        let vec_offset = vector_blob.len() as u64;
        for f in &record.vector {
            vector_blob.extend_from_slice(&f.to_le_bytes());
        }

        index.push(IndexEntry {
            id: record.id,
            seq: record.seq,
            deleted: record.deleted,
            meta_offset,
            meta_len: meta_bytes.len() as u32,
            vec_offset,
            vec_len: (record.vector.len() * 4) as u32,
        });
    }

    let index_bytes = bincode::serialize(&index)?;

    // Write to a temp file and rename, so a crash mid-write never leaves
    // a partially-written file at the final path for the reader to trip
    // over (SSTables are supposed to be immutable once they exist).
    let tmp_path = path.with_extension("sst.tmp");
    {
        let mut file = File::create(&tmp_path)?;
        file.write_all(MAGIC)?;
        file.write_all(&(index_bytes.len() as u64).to_le_bytes())?;
        file.write_all(&index_bytes)?;
        file.write_all(&(meta_blob.len() as u64).to_le_bytes())?;
        file.write_all(&meta_blob)?;
        file.write_all(&(vector_blob.len() as u64).to_le_bytes())?;
        file.write_all(&vector_blob)?;
        file.flush()?;
        file.sync_all()?;
    }
    fs::rename(&tmp_path, path)?;
    Ok(())
}

/// A loaded SSTable, held fully in memory for Phase 1. Phase 2+ can
/// switch this to mmap or on-demand seeks once file sizes matter; the
/// point right now is a correct, simple read path to build the index
/// and query fusion against.
pub struct SSTableReader {
    path: PathBuf,
    index: Vec<IndexEntry>,
    meta_blob: Vec<u8>,
    vector_blob: Vec<u8>,
}

impl SSTableReader {
    pub fn open(path: &Path) -> Result<Self, SSTableError> {
        let bytes = fs::read(path)?;
        let mut cursor = 0usize;

        let take = |cursor: &mut usize, n: usize| -> Result<&[u8], SSTableError> {
            if *cursor + n > bytes.len() {
                return Err(SSTableError::Truncated);
            }
            let slice = &bytes[*cursor..*cursor + n];
            *cursor += n;
            Ok(slice)
        };

        let magic = take(&mut cursor, 4)?;
        if magic != MAGIC {
            return Err(SSTableError::BadMagic);
        }

        let index_len = u64::from_le_bytes(take(&mut cursor, 8)?.try_into().unwrap()) as usize;
        let index_bytes = take(&mut cursor, index_len)?;
        let index: Vec<IndexEntry> = bincode::deserialize(index_bytes)?;

        let meta_len = u64::from_le_bytes(take(&mut cursor, 8)?.try_into().unwrap()) as usize;
        let meta_blob = take(&mut cursor, meta_len)?.to_vec();

        let vec_len = u64::from_le_bytes(take(&mut cursor, 8)?.try_into().unwrap()) as usize;
        let vector_blob = take(&mut cursor, vec_len)?.to_vec();

        Ok(Self { path: path.to_path_buf(), index, meta_blob, vector_blob })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Point lookup. Returns the record (tombstone included) if present
    /// in this SSTable. Binary search relies on `index` being sorted by
    /// id, which `write_sstable` preserves from the memtable's order.
    pub fn get(&self, id: RecordId) -> Option<Record> {
        let pos = self.index.binary_search_by_key(&id, |e| e.id).ok()?;
        Some(self.reconstruct(&self.index[pos]))
    }

    /// Iterate every record in this SSTable in sorted id order,
    /// including tombstones (callers decide whether to filter them --
    /// the engine's merge needs to see tombstones to correctly shadow
    /// older SSTables).
    pub fn iter(&self) -> impl Iterator<Item = Record> + '_ {
        self.index.iter().map(move |e| self.reconstruct(e))
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    fn reconstruct(&self, entry: &IndexEntry) -> Record {
        let metadata: HashMap<String, MetadataValue> = if entry.meta_len == 0 {
            HashMap::new()
        } else {
            let start = entry.meta_offset as usize;
            let end = start + entry.meta_len as usize;
            bincode::deserialize(&self.meta_blob[start..end])
                .expect("corrupt metadata blob in sstable")
        };

        let vector = if entry.vec_len == 0 {
            Vec::new()
        } else {
            let start = entry.vec_offset as usize;
            let end = start + entry.vec_len as usize;
            self.vector_blob[start..end]
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect()
        };

        Record { id: entry.id, vector, metadata, seq: entry.seq, deleted: entry.deleted }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap as StdHashMap;

    fn rec(id: u64, seq: u64, dim: usize) -> Record {
        Record::new(
            id,
            (0..dim).map(|i| (id as f32) + (i as f32) * 0.1).collect(),
            StdHashMap::from([("category".to_string(), MetadataValue::String(format!("cat{}", id % 3)))]),
            seq,
        )
    }

    #[test]
    fn write_and_read_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("000001.sst");
        let records: Vec<Record> = (1..=50).map(|id| rec(id, id, 8)).collect();

        write_sstable(&path, &records).unwrap();
        let reader = SSTableReader::open(&path).unwrap();

        assert_eq!(reader.len(), 50);
        let r = reader.get(25).unwrap();
        assert_eq!(r.id, 25);
        assert_eq!(r.vector.len(), 8);
        assert_eq!(r.metadata.get("category").unwrap(), &MetadataValue::String("cat1".to_string()));
    }

    #[test]
    fn tombstones_survive_the_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("000001.sst");
        let records = vec![rec(1, 1, 4), Record::tombstone(2, 2), rec(3, 3, 4)];

        write_sstable(&path, &records).unwrap();
        let reader = SSTableReader::open(&path).unwrap();

        let tombstone = reader.get(2).unwrap();
        assert!(tombstone.deleted);
        assert!(tombstone.vector.is_empty());
    }

    #[test]
    fn iter_yields_sorted_order() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("000001.sst");
        let records: Vec<Record> = (1..=20).map(|id| rec(id, id, 4)).collect();
        write_sstable(&path, &records).unwrap();

        let reader = SSTableReader::open(&path).unwrap();
        let ids: Vec<u64> = reader.iter().map(|r| r.id).collect();
        assert_eq!(ids, (1..=20).collect::<Vec<u64>>());
    }

    #[test]
    fn get_missing_id_returns_none() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("000001.sst");
        write_sstable(&path, &[rec(1, 1, 4)]).unwrap();
        let reader = SSTableReader::open(&path).unwrap();
        assert!(reader.get(999).is_none());
    }

    #[test]
    fn rejects_bad_magic() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.sst");
        fs::write(&path, b"not an sstable at all").unwrap();
        let result = SSTableReader::open(&path);
        assert!(matches!(result, Err(SSTableError::BadMagic) | Err(SSTableError::Truncated)));
    }

    #[test]
    fn handles_records_with_no_vector_or_metadata() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("empty.sst");
        let r = Record::new(1, vec![], StdHashMap::new(), 1);
        write_sstable(&path, &[r]).unwrap();
        let reader = SSTableReader::open(&path).unwrap();
        let got = reader.get(1).unwrap();
        assert!(got.vector.is_empty());
        assert!(got.metadata.is_empty());
    }
}
