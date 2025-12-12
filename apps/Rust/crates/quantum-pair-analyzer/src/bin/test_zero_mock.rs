// Zero-Mock Enforcement Test Runner Binary
// Copyright (c) 2025 TENGRI Trading Swarm
// Execute comprehensive tests for zero-mock enforcement

use quantum_pair_analyzer::anti_mock::test_runner::run_zero_mock_tests;
use anyhow::Result;
use tracing::info;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    info!("🚀 Zero-Mock Enforcement Test Runner");
    info!("=====================================");
    
    // Run comprehensive tests
    run_zero_mock_tests().await?;
    
    info!("✅ Zero-Mock Enforcement tests completed successfully!");
    
    Ok(())
}