//! NeuraStore: a unified storage/query engine for AI-native workloads.
//!
//! Phase 0: durable write path (WAL + MemTable), crash recovery.
//! Phase 1 (current): LSM-style flush to immutable SSTables with a
//! hybrid row/columnar physical layout (metadata and vectors stored in
//! separate blobs), multi-level reads (memtable -> newest..oldest
//! SSTable), and correctness at 100K+ records.
//!
//! Still ahead: a real vector index (Phase 2/3), query fusion (Phase 4),
//! and a network-facing API (Phase 5).

pub mod engine;
pub mod memtable;
pub mod record;
pub mod sstable;
pub mod wal;

pub use engine::Engine;
pub use record::Record;
