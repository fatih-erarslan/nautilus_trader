# Boardroom Interface

A multi-agent collaboration framework providing service discovery, consensus mechanisms, message routing, and fault-tolerant communication for the ATS CP Trader ecosystem.

## Features

- **Multi-Transport Support**: Redis and ZeroMQ implementations for flexible deployment
- **Service Discovery**: Dynamic agent registration and capability-based discovery
- **Consensus Mechanisms**: Multiple voting policies including BFT consensus
- **Load Balancing**: Intelligent message routing with multiple strategies
- **Fault Tolerance**: Automatic reconnection and stale agent cleanup
- **Async/Await**: Fully asynchronous implementation using Tokio

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Boardroom Interface                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │  Agent   │  │  Agent   │  │  Agent   │  │  Agent  │ │
│  │ (Trader) │  │(Analyzer)│  │ (Whale)  │  │  (QAR)  │ │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬────┘ │
│       │              │              │              │      │
│  ┌────┴──────────────┴──────────────┴──────────────┴───┐ │
│  │              Message Router & Load Balancer          │ │
│  └────┬──────────────┬──────────────┬──────────────┬───┘ │
│       │              │              │              │      │
│  ┌────┴────┐  ┌─────┴────┐  ┌─────┴─────┐  ┌────┴────┐ │
│  │Discovery│  │Consensus │  │ Transport │  │ Routing │ │
│  │Service  │  │ Manager  │  │(Redis/ZMQ)│  │Strategy │ │
│  └─────────┘  └──────────┘  └───────────┘  └─────────┘ │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

## Usage

### Basic Setup

```rust
use boardroom_interface::{Boardroom, BoardroomConfig, AgentCapability};

#[tokio::main]
async fn main() -> Result<()> {
    // Create boardroom with default config
    let config = BoardroomConfig::default();
    let boardroom = Boardroom::new(config).await?;
    
    // Start the boardroom
    boardroom.start().await?;
    
    // Register agents
    let agent = MyTradingAgent::new();
    boardroom.register_agent(Box::new(agent)).await?;
    
    Ok(())
}
```

### Service Discovery

```rust
// Find agents by capability
let traders = boardroom.find_agents(&AgentCapability::Trading).await?;
let analyzers = boardroom.find_agents(&AgentCapability::MarketAnalysis).await?;

// Subscribe to topics
boardroom.subscribe_to_topic(agent_id, "market-data".to_string()).await?;
```

### Consensus Mechanisms

```rust
// Request consensus from multiple agents
let proposal = serde_json::json!({
    "action": "increase_position_limit",
    "new_limit": 100000
});

let participants = vec![agent1_id, agent2_id, agent3_id];
let mut result_rx = boardroom.request_consensus(
    proposal,
    participants,
    VotingPolicy::SimpleMajority,
    Some(30000), // 30 second timeout
).await?;

// Wait for consensus result
let result = result_rx.recv().await?;
if result.approved {
    println!("Consensus reached: {} votes for", result.votes_for);
}
```

### Message Routing

```rust
// Send message with automatic routing
let message = Message::request(
    sender_id,
    "analyze_market",
    serde_json::json!({"symbol": "BTC/USD"}),
);
boardroom.send_message(message).await?;

// Broadcast to all subscribers
let notification = Message::broadcast(
    sender_id,
    "price-update",
    serde_json::json!({"symbol": "BTC/USD", "price": 50000}),
);
boardroom.broadcast_to_topic("market-data", notification).await?;
```

## Transport Configuration

### Redis Transport

```rust
use boardroom_interface::{BoardroomConfig, TransportType, TransportConfig, RedisConfig};

let config = BoardroomConfig {
    transport_type: TransportType::Redis,
    transport_config: TransportConfig::Redis(RedisConfig {
        url: "redis://localhost:6379".to_string(),
        pool_size: 10,
        timeout_ms: 5000,
    }),
    ..Default::default()
};
```

### ZeroMQ Transport

```rust
use boardroom_interface::{BoardroomConfig, TransportType, TransportConfig, ZmqConfig, ZmqSocketType};

let config = BoardroomConfig {
    transport_type: TransportType::ZeroMQ,
    transport_config: TransportConfig::ZeroMQ(ZmqConfig {
        endpoint: "tcp://127.0.0.1:5555".to_string(),
        socket_type: ZmqSocketType::Router,
        high_water_mark: 1000,
        timeout_ms: 5000,
    }),
    ..Default::default()
};
```

## Routing Strategies

- **Direct**: Send to specific agent
- **RoundRobin**: Distribute evenly across agents
- **LeastLoaded**: Send to agent with lowest load
- **Random**: Random agent selection
- **CapabilityBased**: Route based on required capabilities

## Agent Implementation

```rust
use boardroom_interface::{Agent, AgentId, AgentInfo, AgentState, Message};

struct MyAgent {
    id: AgentId,
    // ... other fields
}

#[async_trait::async_trait]
impl Agent for MyAgent {
    fn id(&self) -> AgentId {
        self.id
    }

    async fn info(&self) -> AgentInfo {
        AgentInfo::new(self.id, "my-agent", "tcp://localhost:5556")
            .with_capability(AgentCapability::Trading)
            .with_capability(AgentCapability::RiskAssessment)
    }

    async fn initialize(&mut self) -> Result<()> {
        // Initialize agent resources
        Ok(())
    }

    async fn start(&mut self) -> Result<()> {
        // Start agent processing
        Ok(())
    }

    async fn stop(&mut self) -> Result<()> {
        // Cleanup resources
        Ok(())
    }

    async fn handle_message(&mut self, message: Message) -> Result<()> {
        // Process incoming messages
        match message.message_type {
            MessageType::Request { method, params } => {
                // Handle request
            }
            _ => {}
        }
        Ok(())
    }

    async fn state(&self) -> AgentState {
        AgentState::Ready
    }
}
```

## Testing

Run the test suite:

```bash
cargo test
```

Run integration tests (requires Redis):

```bash
cargo test --features integration-tests
```

## Performance

The boardroom interface is designed for high-performance multi-agent systems:

- Async message passing with minimal overhead
- Efficient load balancing with O(1) agent selection
- Lock-free data structures where possible
- Configurable batching and buffering

## License

This project is part of the ATS CP Trader system.