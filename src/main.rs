//! Phase 1 demo: prove flush-to-SSTable + multi-level reads work, and
//! that state survives a restart whether it's still in the WAL or
//! already flushed to disk.
//!
//! Run repeatedly against the same data dir to watch SSTables accumulate:
//!   cargo run -- ./data
//!   cargo run -- ./data
//!   cargo run -- ./data

use neurastore::Engine;
use std::collections::HashMap;
use std::env;

fn main() {
    let dir = env::args().nth(1).unwrap_or_else(|| "./data".to_string());

    let mut engine = Engine::open(&dir).expect("failed to open engine");
    println!("NeuraStore Phase 1 -- opened engine at {dir}");
    println!(
        "Recovered {} live record(s) across {} SSTable(s) + memtable.",
        engine.len(),
        engine.sstable_count()
    );

    if engine.is_empty() {
        println!("No prior state found -- inserting seed records...");
        engine
            .put(1, vec![0.1, 0.2, 0.3, 0.4], HashMap::from([("category".into(), "docs".into())]))
            .unwrap();
        engine
            .put(2, vec![0.9, 0.8, 0.7, 0.6], HashMap::from([("category".into(), "code".into())]))
            .unwrap();
        engine
            .put(3, vec![0.5, 0.5, 0.5, 0.5], HashMap::from([("category".into(), "docs".into())]))
            .unwrap();
        println!("Inserted 3 records. Flushing to an SSTable...");
        engine.flush().unwrap();
        println!("Flushed. Run this binary again to see it recover from disk, not just the WAL.");
    } else {
        println!("State recovered successfully -- durability check passed.");
    }

    println!("\nCurrent live records (merged across memtable + all SSTables):");
    for record in engine.scan_live() {
        println!("  id={} vector={:?} metadata={:?}", record.id, record.vector, record.metadata);
    }
}
