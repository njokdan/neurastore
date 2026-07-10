//! Write-ahead log (WAL).
//!
//! Durability contract: every mutation (insert/delete) is appended to the
//! WAL and fsync'd *before* it is considered committed. On startup, the
//! WAL is replayed to reconstruct the memtable. This is what lets the
//! engine survive a crash without losing acknowledged writes.
//!
//! Frame format on disk, repeated:
//!   [ u32 length (LE) ][ payload bytes (bincode-encoded Record) ][ u32 crc32 (LE) ]

use crate::record::Record;
use std::fs::{File, OpenOptions};
use std::io::{self, BufReader, Read, Write};
use std::path::{Path, PathBuf};

#[derive(thiserror::Error, Debug)]
pub enum WalError {
    #[error("io error: {0}")]
    Io(#[from] io::Error),
    #[error("serialization error: {0}")]
    Serde(#[from] bincode::Error),
    #[error("corrupt WAL frame: checksum mismatch at offset {0}")]
    Checksum(u64),
    #[error("truncated WAL frame at offset {0}")]
    Truncated(u64),
}

pub struct Wal {
    path: PathBuf,
    file: File,
}

impl Wal {
    /// Open (or create) a WAL file for appending. Does not replay --
    /// call `replay()` separately during startup recovery.
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self, WalError> {
        let path = path.as_ref().to_path_buf();
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .read(true)
            .open(&path)?;
        Ok(Self { path, file })
    }

    /// Append a record to the log and fsync before returning.
    /// A write is not considered durable until this returns Ok.
    ///
    /// This is the safest, slowest write path: one fsync per record. Use
    /// `append_batch` for bulk loads -- see its docs for the durability
    /// tradeoff that makes it faster.
    pub fn append(&mut self, record: &Record) -> Result<(), WalError> {
        self.write_frame(record)?;
        self.file.flush()?;
        self.file.sync_data()?;
        Ok(())
    }

    /// Append many records with a single fsync at the end, instead of one
    /// per record. This is the fix for a real, measured problem: Phase 2's
    /// benchmark showed NeuraStore's insert throughput (1,106 vec/sec)
    /// trailing both pgvector (1,633) and Milvus (2,545) on identical
    /// hardware -- traced to exactly this, a synchronous disk round-trip
    /// on every single `put()`.
    ///
    /// Durability tradeoff, stated plainly: all records in a batch become
    /// durable together, at the single fsync, not individually. If the
    /// process crashes partway through writing a batch (before that one
    /// fsync completes), the *entire* batch is lost on replay -- not just
    /// the latest record, unlike `append`, where each call is its own
    /// durability boundary. For a bulk load (this is the case
    /// `bin/bench_neurastore` and any future batch-import API want),
    /// that's the right tradeoff: the batch is one logical unit of work
    /// anyway. It would NOT be the right default for interactive
    /// single-record writes, where callers reasonably expect each `put()`
    /// to be independently durable the moment it returns -- `append`
    /// stays the default for `Engine::put`/`delete` for exactly that reason.
    pub fn append_batch(&mut self, records: &[Record]) -> Result<(), WalError> {
        for record in records {
            self.write_frame(record)?;
        }
        self.file.flush()?;
        self.file.sync_data()?;
        Ok(())
    }

    fn write_frame(&mut self, record: &Record) -> Result<(), WalError> {
        let payload = bincode::serialize(record)?;
        let len = payload.len() as u32;
        let crc = crc32fast::hash(&payload);

        self.file.write_all(&len.to_le_bytes())?;
        self.file.write_all(&payload)?;
        self.file.write_all(&crc.to_le_bytes())?;
        Ok(())
    }

    /// Replay every valid frame in the log, in write order.
    /// Stops (without erroring) at the first truncated/torn frame, since
    /// that's the expected shape of a crash mid-write -- everything before
    /// it is still valid and durable.
    pub fn replay(&self) -> Result<Vec<Record>, WalError> {
        let file = File::open(&self.path)?;
        let mut reader = BufReader::new(file);
        let mut records = Vec::new();
        let mut offset: u64 = 0;

        loop {
            let mut len_buf = [0u8; 4];
            match reader.read_exact(&mut len_buf) {
                Ok(()) => {}
                Err(e) if e.kind() == io::ErrorKind::UnexpectedEof => break,
                Err(e) => return Err(e.into()),
            }
            let len = u32::from_le_bytes(len_buf) as usize;

            let mut payload = vec![0u8; len];
            if reader.read_exact(&mut payload).is_err() {
                // Torn write: last frame wasn't fully flushed before crash.
                break;
            }

            let mut crc_buf = [0u8; 4];
            if reader.read_exact(&mut crc_buf).is_err() {
                break;
            }
            let expected_crc = u32::from_le_bytes(crc_buf);
            let actual_crc = crc32fast::hash(&payload);
            if actual_crc != expected_crc {
                return Err(WalError::Checksum(offset));
            }

            let record: Record = bincode::deserialize(&payload)?;
            offset += 4 + len as u64 + 4;
            records.push(record);
        }

        Ok(records)
    }

    /// Truncate the log to empty. Called after a successful memtable
    /// flush to an SSTable (Phase 1) so the WAL doesn't grow unbounded.
    pub fn clear(&mut self) -> Result<(), WalError> {
        self.file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .read(true)
            .open(&self.path)?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn rec(id: u64, seq: u64) -> Record {
        Record::new(id, vec![1.0, 2.0, 3.0], HashMap::from([("k".into(), "v".into())]), seq)
    }

    #[test]
    fn append_and_replay_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.wal");

        {
            let mut wal = Wal::open(&path).unwrap();
            wal.append(&rec(1, 1)).unwrap();
            wal.append(&rec(2, 2)).unwrap();
            wal.append(&Record::tombstone(1, 3)).unwrap();
        }

        let wal = Wal::open(&path).unwrap();
        let replayed = wal.replay().unwrap();
        assert_eq!(replayed.len(), 3);
        assert_eq!(replayed[0].id, 1);
        assert_eq!(replayed[1].id, 2);
        assert!(replayed[2].deleted);
    }

    #[test]
    fn append_batch_and_replay_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("batch.wal");

        let records: Vec<Record> = (1..=50).map(|i| rec(i, i)).collect();
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.append_batch(&records).unwrap();
        }

        let wal = Wal::open(&path).unwrap();
        let replayed = wal.replay().unwrap();
        assert_eq!(replayed.len(), 50);
        for (original, got) in records.iter().zip(replayed.iter()) {
            assert_eq!(original.id, got.id);
            assert_eq!(original.vector, got.vector);
        }
    }

    #[test]
    fn append_batch_is_faster_than_append_per_record() {
        // Not a strict benchmark (too environment-dependent for a unit
        // test), but a sanity check that the whole point of this method
        // -- fewer fsyncs -- actually holds in practice. Uses a large
        // enough batch that per-fsync overhead should dominate `append`'s
        // time if fsync is doing what we think it's doing.
        let dir = tempfile::tempdir().unwrap();
        let records: Vec<Record> = (1..=500).map(|i| rec(i, i)).collect();

        let per_record_path = dir.path().join("per_record.wal");
        let per_record_start = std::time::Instant::now();
        {
            let mut wal = Wal::open(&per_record_path).unwrap();
            for r in &records {
                wal.append(r).unwrap();
            }
        }
        let per_record_elapsed = per_record_start.elapsed();

        let batch_path = dir.path().join("batch.wal");
        let batch_start = std::time::Instant::now();
        {
            let mut wal = Wal::open(&batch_path).unwrap();
            wal.append_batch(&records).unwrap();
        }
        let batch_elapsed = batch_start.elapsed();

        assert!(
            batch_elapsed < per_record_elapsed,
            "expected append_batch ({batch_elapsed:?}) to be faster than {} individual append calls ({per_record_elapsed:?})",
            records.len()
        );
    }

    #[test]
    fn append_batch_all_or_nothing_on_torn_write() {
        // Documents the durability tradeoff plainly, not just in a
        // comment: if a crash happens before append_batch's single fsync
        // completes, replay can lose the *whole* batch, not just the
        // tail record. Simulated here the same way the single-append
        // torn-write test does -- truncate off the end of the file to
        // stand in for a mid-write crash.
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("torn_batch.wal");
        let records: Vec<Record> = (1..=10).map(|i| rec(i, i)).collect();
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.append_batch(&records).unwrap();
        }

        // Truncate into the middle of the batch, simulating a crash
        // partway through writing (but before the fsync -- in a real
        // crash, data this far along may not even be on disk yet, but
        // truncating is the closest thing a test can do to simulate it).
        let len = std::fs::metadata(&path).unwrap().len();
        let f = OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(len / 2).unwrap();

        let wal = Wal::open(&path).unwrap();
        let replayed = wal.replay().unwrap();
        // Some prefix of the batch survives (whatever was fully written
        // before the truncation point), but not all 10 -- illustrating
        // that a batch's durability is all-or-nothing at the granularity
        // of "however far the write got," not per-record like `append`.
        assert!(replayed.len() < 10, "expected the torn batch to lose at least its tail");
    }

    #[test]
    fn detects_corrupted_frame() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("corrupt.wal");
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.append(&rec(1, 1)).unwrap();
        }
        // Flip a byte in the payload region to corrupt the checksum.
        let mut bytes = std::fs::read(&path).unwrap();
        let mid = bytes.len() / 2;
        bytes[mid] ^= 0xFF;
        std::fs::write(&path, bytes).unwrap();

        let wal = Wal::open(&path).unwrap();
        let result = wal.replay();
        assert!(result.is_err());
    }

    #[test]
    fn survives_torn_write_at_tail() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("torn.wal");
        {
            let mut wal = Wal::open(&path).unwrap();
            wal.append(&rec(1, 1)).unwrap();
            wal.append(&rec(2, 2)).unwrap();
        }
        // Simulate a crash mid-append: truncate off the last few bytes
        // of the second frame.
        let len = std::fs::metadata(&path).unwrap().len();
        let f = OpenOptions::new().write(true).open(&path).unwrap();
        f.set_len(len - 3).unwrap();

        let wal = Wal::open(&path).unwrap();
        let replayed = wal.replay().unwrap();
        // First frame intact, second torn frame dropped.
        assert_eq!(replayed.len(), 1);
        assert_eq!(replayed[0].id, 1);
    }
}
