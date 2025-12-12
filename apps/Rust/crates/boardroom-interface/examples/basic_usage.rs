//! Basic usage example of the boardroom interface

use anyhow::Result;
use boardroom_interface::{
    Agent, AgentCapability, AgentId, AgentInfo, AgentState, Boardroom, BoardroomConfig,
    ConsensusResult, Message, MessageType, VotingPolicy,
};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Example trading agent
struct TradingAgent {
    id: AgentId,
    name: String,
    state: Arc<RwLock<AgentState>>,
}

impl TradingAgent {
    fn new(name: impl Into<String>) -> Self {
        Self {
            id: AgentId::new(),
            name: name.into(),
            state: Arc::new(RwLock::new(AgentState::Initializing)),
        }
    }
}

#[async_trait::async_trait]
impl Agent for TradingAgent {
    fn id(&self) -> AgentId {
        self.id
    }

    async fn info(&self) -> AgentInfo {
        AgentInfo::new(self.id, &self.name, "tcp://localhost:5556")
            .with_capability(AgentCapability::Trading)
            .with_capability(AgentCapability::MarketAnalysis)
    }

    async fn initialize(&mut self) -> Result<()> {
        println!("{}: Initializing...", self.name);
        *self.state.write().await = AgentState::Ready;
        Ok(())
    }

    async fn start(&mut self) -> Result<()> {
        println!("{}: Started", self.name);
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        println!("{}: Stopping...", self.name);
        *self.state.write().await = AgentState::Stopped;
        Ok(())
    }

    async fn handle_message(&mut self, message: Message) -> Result<()> {
        println!("{}: Received message: {:?}", self.name, message.message_type);
        
        match message.message_type {
            MessageType::Request { method, params } => {
                println!("  Method: {}, Params: {}", method, params);
                // In real implementation, would send response
            }
            MessageType::ConsensusRequest { proposal, .. } => {
                println!("  Consensus proposal: {}", proposal);
                // In real implementation, would vote
            }
            _ => {}
        }
        
        Ok(())
    }

    async fn state(&self) -> AgentState {
        *self.state.read().await
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();

    println!("=== Boardroom Interface Example ===\n");

    // Create boardroom configuration
    let config = BoardroomConfig {
        name: "example-boardroom".to_string(),
        heartbeat_interval_ms: 10000,
        ..Default::default()
    };

    println!("1. Creating boardroom with config: {:?}", config.name);
    let boardroom = Arc::new(Boardroom::new(config).await?);

    // Start the boardroom
    println!("\n2. Starting boardroom...");
    boardroom.start().await?;

    // Create and register agents
    println!("\n3. Creating and registering agents...");
    
    let agent1 = Box::new(TradingAgent::new("trader-1"));
    let agent1_id = agent1.id();
    boardroom.register_agent(agent1).await?;
    
    let agent2 = Box::new(TradingAgent::new("trader-2"));
    let agent2_id = agent2.id();
    boardroom.register_agent(agent2).await?;
    
    let agent3 = Box::new(TradingAgent::new("trader-3"));
    let agent3_id = agent3.id();
    boardroom.register_agent(agent3).await?;

    // Give agents time to initialize
    tokio::time::sleep(tokio::time::Duration::from_millis(500)).await;

    // Find agents by capability
    println!("\n4. Finding agents with Trading capability...");
    let traders = boardroom.find_agents(&AgentCapability::Trading).await?;
    println!("   Found {} trading agents", traders.len());

    // Subscribe agents to topics
    println!("\n5. Subscribing agents to topics...");
    boardroom.subscribe_to_topic(agent1_id, "market-data".to_string()).await?;
    boardroom.subscribe_to_topic(agent2_id, "market-data".to_string()).await?;
    boardroom.subscribe_to_topic(agent3_id, "market-data".to_string()).await?;

    // Send direct message
    println!("\n6. Sending direct message to agent1...");
    let message = Message::request(
        agent2_id,
        "get_balance",
        serde_json::json!({"currency": "USD"}),
    )
    .to(agent1_id);
    boardroom.send_message(message).await?;

    // Broadcast message
    println!("\n7. Broadcasting market data update...");
    let broadcast = Message::broadcast(
        agent1_id,
        "market-data",
        serde_json::json!({
            "symbol": "BTC/USD",
            "price": 50000,
            "volume": 1234.56
        }),
    );
    boardroom.broadcast_to_topic("market-data", broadcast).await?;

    // Request consensus
    println!("\n8. Requesting consensus for position limit increase...");
    let proposal = serde_json::json!({
        "action": "increase_position_limit",
        "current_limit": 50000,
        "new_limit": 100000,
        "reason": "Market conditions favorable"
    });

    let participants = vec![agent1_id, agent2_id, agent3_id];
    let mut consensus_rx = boardroom
        .request_consensus(
            proposal,
            participants,
            VotingPolicy::SimpleMajority,
            Some(5000), // 5 second timeout
        )
        .await?;

    // In a real scenario, agents would vote through the consensus mechanism
    println!("   Waiting for consensus result...");
    
    // For demo purposes, we'll simulate the timeout
    tokio::select! {
        result = consensus_rx.recv() => {
            if let Ok(result) = result {
                print_consensus_result(&result);
            }
        }
        _ = tokio::time::sleep(tokio::time::Duration::from_secs(2)) => {
            println!("   Consensus still pending (would timeout in real scenario)");
        }
    }

    // Get agent info
    println!("\n9. Getting agent information...");
    if let Some(info) = boardroom.get_agent(agent1_id).await? {
        println!("   Agent: {} ({})", info.name, info.id);
        println!("   State: {:?}", info.state);
        println!("   Capabilities: {:?}", info.capabilities);
    }

    // Give some time for messages to process
    tokio::time::sleep(tokio::time::Duration::from_secs(1)).await;

    // Stop boardroom
    println!("\n10. Stopping boardroom...");
    boardroom.stop().await?;

    println!("\nExample completed successfully!");
    Ok(())
}

fn print_consensus_result(result: &ConsensusResult) {
    println!("   Consensus Result:");
    println!("     - Approved: {}", result.approved);
    println!("     - Votes For: {}", result.votes_for);
    println!("     - Votes Against: {}", result.votes_against);
    println!("     - Participation: {:.1}%", result.participation_rate * 100.0);
    println!("     - Time: {}ms", result.completion_time_ms);
}