//! Example QUIC coordinator server

use neural_trader_swarm::{QuicSwarmCoordinator, CoordinatorConfig};
use std::sync::Arc;
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("neural_trader_swarm=debug,quinn=info")
        .init();

    println!("🚀 Starting QUIC Swarm Coordinator...");

    // Parse address
    let addr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:5000".to_string())
        .parse()?;

    // Configure coordinator
    let config = CoordinatorConfig {
        max_concurrent_bidi_streams: 1000,
        max_idle_timeout: std::time::Duration::from_secs(300),
        keep_alive_interval: std::time::Duration::from_secs(10),
        task_timeout: std::time::Duration::from_secs(300),
        enable_metrics: true,
    };

    // Create coordinator
    let coordinator = Arc::new(QuicSwarmCoordinator::new(addr, config).await?);

    println!("✅ Coordinator listening on {}", addr);
    println!("📊 Metrics enabled");
    println!("🔐 TLS 1.3 encryption enabled");
    println!();
    println!("Waiting for agent connections...");

    // Spawn metrics reporter
    let coord_clone = coordinator.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(5));
        loop {
            interval.tick().await;
            let stats = coord_clone.stats();
            println!(
                "📈 Uptime: {:?} | Active agents: {}",
                stats.uptime, stats.active_agents
            );
        }
    });

    // Run coordinator
    coordinator.run().await?;

    Ok(())
}
