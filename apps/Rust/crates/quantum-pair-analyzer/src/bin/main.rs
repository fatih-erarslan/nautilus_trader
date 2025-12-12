// Quantum-Enhanced Pair Selection Analyzer - Main Binary
// Copyright (c) 2025 TENGRI Trading Swarm

use std::env;
use std::process;
use anyhow::Result;
use tracing::{info, error};
use quantum_pair_analyzer::{QuantumPairAnalyzer, AnalyzerConfig};

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt()
        .with_env_filter("quantum_pair_analyzer=info")
        .init();

    info!("Starting Quantum Pair Analyzer");

    // Load configuration
    let config = match load_config() {
        Ok(config) => config,
        Err(e) => {
            error!("Failed to load configuration: {}", e);
            process::exit(1);
        }
    };

    // Initialize analyzer
    let analyzer = match QuantumPairAnalyzer::new(config).await {
        Ok(analyzer) => analyzer,
        Err(e) => {
            error!("Failed to initialize analyzer: {}", e);
            process::exit(1);
        }
    };

    // Run analysis
    let optimal_pairs = analyzer.find_optimal_pairs(10, None).await?;
    
    info!("Found {} optimal pairs:", optimal_pairs.len());
    for (i, pair) in optimal_pairs.iter().enumerate() {
        info!("{}. {} - Score: {:.4}, Confidence: {:.2}%", 
              i + 1, 
              pair.pair_id.symbol(), 
              pair.score, 
              pair.confidence * 100.0);
    }

    Ok(())
}

fn load_config() -> Result<AnalyzerConfig> {
    // In a real implementation, this would load from a config file
    // For now, return default configuration
    Ok(AnalyzerConfig::default())
}