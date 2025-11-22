//! Secure main entry point binary
//! 
//! This is the production-ready secure entry point for the hive mind system
//! with comprehensive security controls enabled.

use hive_mind_rust::secure_main::secure_main;
use std::process;

#[tokio::main]
async fn main() {
    if let Err(e) = secure_main().await {
        eprintln!("Fatal security error: {}", e);
        process::exit(1);
    }
}