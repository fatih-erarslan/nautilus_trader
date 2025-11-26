//! Example QUIC agent client

use neural_trader_swarm::{QuicSwarmAgent, AgentType};
use tracing_subscriber;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter("neural_trader_swarm=debug,quinn=info")
        .init();

    println!("🤖 Starting QUIC Swarm Agent...");

    // Parse coordinator address
    let coordinator_addr = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "127.0.0.1:5000".to_string())
        .parse()?;

    // Parse agent ID
    let agent_id = std::env::args()
        .nth(2)
        .unwrap_or_else(|| format!("agent-{}", uuid::Uuid::new_v4()));

    // Parse agent type
    let agent_type = match std::env::args().nth(3).as_deref() {
        Some("pattern") => AgentType::PatternMatcher,
        Some("strategy") => AgentType::StrategyCorrelator,
        Some("neural") => AgentType::NeuralTrainer,
        _ => AgentType::Worker,
    };

    println!("🔗 Connecting to coordinator at {}", coordinator_addr);
    println!("🆔 Agent ID: {}", agent_id);
    println!("🏷️  Agent Type: {:?}", agent_type);

    // Connect to coordinator
    let mut agent = QuicSwarmAgent::connect(agent_id, agent_type, coordinator_addr).await?;

    println!("✅ Connected to coordinator");
    println!("🔐 TLS 1.3 secure connection established");
    println!();
    println!("Waiting for tasks...");

    // Spawn heartbeat sender
    let agent_clone = agent.clone();
    tokio::spawn(async move {
        let mut interval = tokio::time::interval(std::time::Duration::from_secs(10));
        loop {
            interval.tick().await;
            if let Err(e) = agent_clone.send_heartbeat(0.5, 0).await {
                eprintln!("Failed to send heartbeat: {}", e);
            }
        }
    });

    // Run agent
    agent.run().await?;

    Ok(())
}
