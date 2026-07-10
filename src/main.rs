//! Phase 0 demo: prove the write path is durable across a process restart.
//!
//! Run twice in a row against the same data dir:
//!   1st run:  inserts records, prints them, exits (simulating a stop).
//!   2nd run:  reopens the same WAL, shows the state survived intact.

use neurastore::Engine;
use std::collections::HashMap;
use std::env;

fn main() {
    let path = env::args()
        .nth(1)
        .unwrap_or_else(|| "./data/neurastore.wal".to_string());

    if let Some(parent) = std::path::Path::new(&path).parent() {
        std::fs::create_dir_all(parent).expect("failed to create data dir");
    }

    let mut engine = Engine::open(&path).expect("failed to open engine");
    println!("NeuraStore Phase 0 -- opened WAL at {path}");
    println!("Recovered {} live record(s) from previous run(s).", engine.len());

    if engine.is_empty() {
        println!("No prior state found -- inserting seed records...");
        engine
            .put(
                1,
                vec![0.1, 0.2, 0.3, 0.4],
                HashMap::from([("category".into(), "docs".into())]),
            )
            .unwrap();
        engine
            .put(
                2,
                vec![0.9, 0.8, 0.7, 0.6],
                HashMap::from([("category".into(), "code".into())]),
            )
            .unwrap();
        engine
            .put(
                3,
                vec![0.5, 0.5, 0.5, 0.5],
                HashMap::from([("category".into(), "docs".into())]),
            )
            .unwrap();
        println!("Inserted 3 records. Run this binary again to prove they survive restart.");
    } else {
        println!("State recovered successfully -- durability check passed.");
    }

    println!("\nCurrent live records:");
    for record in engine.scan_live() {
        println!(
            "  id={} vector={:?} metadata={:?}",
            record.id, record.vector, record.metadata
        );
    }
}
