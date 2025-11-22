//! Basic test example for Tengri trading strategy

use tengri::config::TengriConfig;
use tengri::strategy::TengriStrategy;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("info")
        .init();

    println!("🚀 Testing Tengri Trading Strategy");
    
    // Load configuration from example file
    let config_path = "examples/config.toml";
    match TengriConfig::from_file(config_path) {
        Ok(config) => {
            println!("✅ Configuration loaded successfully");
            println!("   Strategy: {}", config.strategy.name);
            println!("   Instruments: {:?}", config.strategy.instruments);
            println!("   Mode: {:?}", config.strategy.mode);
            
            // Test strategy creation
            match TengriStrategy::new(config).await {
                Ok(_strategy) => {
                    println!("✅ Strategy created successfully");
                    println!("🎯 Tengri system is ready for trading!");
                }
                Err(e) => {
                    println!("❌ Failed to create strategy: {}", e);
                }
            }
        }
        Err(e) => {
            println!("❌ Failed to load configuration: {}", e);
            println!("💡 Make sure examples/config.toml exists and is properly formatted");
        }
    }

    Ok(())
}