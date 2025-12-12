//! Unified QBMIA Demo - TENGRI Compliant
//! 
//! This example demonstrates the unified QBMIA core with:
//! - Real market data integration
//! - GPU-only quantum simulation
//! - Authentic biological intelligence
//! - Real system performance monitoring

use qbmia_unified::{
    UnifiedQbmia, UnifiedConfig, RealMarketDataSource,
    error::Result,
};
use tracing::{info, error};
use std::env;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    tracing_subscriber::fmt::init();
    
    info!("🚀 Starting Unified QBMIA Demo - TENGRI Compliant");
    
    // Check for API key
    let api_key = env::var("ALPHA_VANTAGE_API_KEY")
        .unwrap_or_else(|_| {
            println!("⚠️  No ALPHA_VANTAGE_API_KEY found, using demo key (limited functionality)");
            "DEMO_KEY".to_string()
        });
    
    // Configure with real market data sources
    let config = UnifiedConfig {
        num_qubits: 6, // 6 qubits for demo
        market_sources: vec![
            RealMarketDataSource {
                endpoint: "https://www.alphavantage.co/query".to_string(),
                api_key,
                rate_limit: 5, // 5 requests per minute
                last_request: None,
            }
        ],
        monitoring_enabled: true,
        gpu_enabled: true,
    };
    
    // Initialize unified QBMIA core
    match UnifiedQbmia::new(config).await {
        Ok(mut qbmia) => {
            info!("✅ Unified QBMIA Core initialized successfully");
            
            // Test symbols for analysis
            let symbols = vec!["AAPL".to_string(), "MSFT".to_string()];
            
            info!("🔍 Analyzing symbols: {:?}", symbols);
            
            // Perform unified analysis
            match qbmia.analyze(&symbols).await {
                Ok(result) => {
                    info!("✅ Analysis completed successfully");
                    
                    // Display results
                    println!("\n📊 UNIFIED QBMIA ANALYSIS RESULTS");
                    println!("=" .repeat(50));
                    
                    println!("\n🏪 Market Data:");
                    for (i, data) in result.market_data.iter().enumerate() {
                        println!("  {}. {} - Price: ${:.2}, Volume: {:.0}", 
                                i + 1, data.symbol, data.price, data.volume);
                    }
                    
                    println!("\n⚛️  Quantum Probabilities:");
                    for (i, prob) in result.quantum_probabilities.iter().take(8).enumerate() {
                        println!("  State |{:03b}⟩: {:.4}", i, prob);
                    }
                    
                    println!("\n🧠 Biological Response:");
                    println!("  Dopamine Level: {:.3}", result.biological_response.dopamine_response);
                    println!("  Plasticity Change: {:.3}", result.biological_response.plasticity_change);
                    println!("  Adaptation Signal: {}", result.biological_response.adaptation_signal);
                    
                    println!("\n🖥️  System Performance:");
                    println!("  CPU Usage: {:.1}%", result.system_metrics.cpu_usage);
                    println!("  Memory Usage: {:.1}%", result.system_metrics.memory_usage_percent);
                    if let Some(ref gpu) = result.system_metrics.gpu_metrics {
                        println!("  GPU Utilization: {:.1}%", gpu.utilization);
                        println!("  GPU Temperature: {:.1}°C", gpu.temperature);
                    }
                    
                    println!("\n⏱️  Execution Time: {:.2}ms", result.execution_time_ms);
                    println!("📅 Timestamp: {}", result.timestamp);
                    
                    println!("\n✅ TENGRI COMPLIANCE: All data sources are real, no mock data used");
                    
                },
                Err(e) => {
                    error!("❌ Analysis failed: {}", e);
                    
                    // Provide helpful error messages
                    match e {
                        qbmia_unified::error::QBMIAError::Hardware(ref msg) if msg.contains("CUDA") => {
                            println!("\n💡 GPU Requirements:");
                            println!("   QBMIA requires CUDA-capable GPU for quantum simulation");
                            println!("   Install CUDA toolkit and drivers for full functionality");
                        },
                        qbmia_unified::error::QBMIAError::NetworkError(_) => {
                            println!("\n💡 Network Issues:");
                            println!("   Check internet connection and API key validity");
                            println!("   Set ALPHA_VANTAGE_API_KEY environment variable");
                        },
                        qbmia_unified::error::QBMIAError::TengriViolation(ref msg) => {
                            println!("\n🚨 TENGRI VIOLATION DETECTED:");
                            println!("   {}", msg);
                            println!("   This indicates mock data usage - strictly forbidden");
                        },
                        _ => {
                            println!("\n❌ Unexpected error occurred");
                        }
                    }
                }
            }
        },
        Err(e) => {
            error!("❌ Failed to initialize QBMIA Core: {}", e);
            
            match e {
                qbmia_unified::error::QBMIAError::Hardware(ref msg) if msg.contains("CUDA") => {
                    println!("\n🔧 Hardware Requirements:");
                    println!("   ✅ TENGRI Compliance: GPU-only quantum simulation required");
                    println!("   ❌ CUDA not available - please install CUDA toolkit");
                    println!("   📋 Minimum: NVIDIA GPU with CUDA Compute Capability 3.5+");
                },
                _ => {
                    println!("\n❌ Initialization failed: {}", e);
                }
            }
        }
    }
    
    info!("🏁 Unified QBMIA Demo completed");
    Ok(())
}

/// Display system requirements
fn display_requirements() {
    println!("\n📋 QBMIA SYSTEM REQUIREMENTS");
    println!("=" .repeat(40));
    println!("🖥️  Hardware:");
    println!("   • NVIDIA GPU with CUDA support");
    println!("   • 8GB+ RAM");
    println!("   • Multi-core CPU");
    
    println!("\n🔗 Network:");
    println!("   • Internet connection for market data");
    println!("   • Alpha Vantage API key (recommended)");
    
    println!("\n⚡ Software:");
    println!("   • CUDA Toolkit 11.0+");
    println!("   • NVIDIA drivers");
    
    println!("\n🛡️  TENGRI Compliance:");
    println!("   • Zero tolerance for mock data");
    println!("   • Real market data sources only");
    println!("   • GPU-only quantum computation");
    println!("   • Authentic biological algorithms");
}