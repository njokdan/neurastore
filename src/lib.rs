//! NeuraStore: a unified storage/query engine for AI-native workloads.
//!
//! Phase 0 scope (this crate, today): a durable write path (WAL) and an
//! in-memory sorted store (MemTable), wired together by `Engine`, with
//! crash-recovery correctness proven by tests. Everything else on the
//! roadmap -- SSTables, HNSW, filtered ANN fusion, gRPC API -- builds on
//! top of this foundation in later phases.

pub mod engine;
pub mod memtable;
pub mod record;
pub mod wal;

pub use engine::Engine;
pub use record::Record;
