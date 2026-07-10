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
    pub fn append(&mut self, record: &Record) -> Result<(), WalError> {
        let payload = bincode::serialize(record)?;
        let len = payload.len() as u32;
        let crc = crc32fast::hash(&payload);

        self.file.write_all(&len.to_le_bytes())?;
        self.file.write_all(&payload)?;
        self.file.write_all(&crc.to_le_bytes())?;
        self.file.flush()?;
        self.file.sync_data()?;
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
