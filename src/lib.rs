//! NeuraStore: a unified storage/query engine for AI-native workloads.
//!
//! Phase 0: durable write path (WAL + MemTable), crash recovery.
//! Phase 1: LSM-style flush to immutable SSTables with a hybrid
//! row/columnar physical layout, multi-level reads, 100K+ record scale.
//! Phase 2 (current): a static HNSW vector index, built from a snapshot
//! of live records, benchmarked for recall/latency against the
//! pgvector/Milvus baseline established in Phase 0.
//!
//! Still ahead: incremental/concurrent-safe indexing (Phase 3), query
//! fusion of vector search + structured filters (Phase 4), and a
//! network-facing API (Phase 5).

pub mod engine;
pub mod hnsw;
pub mod memtable;
pub mod record;
pub mod sstable;
pub mod vector_index;
pub mod wal;

pub use engine::Engine;
pub use record::Record;
