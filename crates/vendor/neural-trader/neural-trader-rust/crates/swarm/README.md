# Neural Trader QUIC Swarm Coordination

High-performance, low-latency swarm coordination using QUIC protocol with ReasoningBank integration for adaptive learning.

## Features

- **Sub-millisecond latency**: <1ms p99 latency for agent coordination
- **Massive concurrency**: Support for 1000+ concurrent bidirectional streams
- **Built-in encryption**: TLS 1.3 by default
- **0-RTT connection resumption**: Instant reconnection for agents
- **ReasoningBank integration**: Adaptive learning from coordination patterns
- **Stream multiplexing**: Multiple independent communication channels per agent

## Architecture

```
┌─────────────────────────────────────────────────────┐
│         QUIC Swarm Coordinator (Server)            │
└─────────────────────────────────────────────────────┘
          │                    │                    │
    ┌─────┴─────┐      ┌─────┴─────┐      ┌─────┴─────┐
    │  Agent 1  │      │  Agent 2  │      │  Agent N  │
    │ (Client)  │      │ (Client)  │      │ (Client)  │
    └───────────┘      └───────────┘      └───────────┘
```

Each connection supports multiple bidirectional streams:
- Pattern matching results
- Strategy correlations
- ReasoningBank experiences
- Neural gradients
- Task assignments

## Usage

### Coordinator (Server)

```rust
use neural_trader_swarm::{QuicSwarmCoordinator, CoordinatorConfig};
use std::sync::Arc;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let addr = "127.0.0.1:5000".parse()?;
    let config = CoordinatorConfig::default();

    let coordinator = Arc::new(
        QuicSwarmCoordinator::new(addr, config).await?
    );

    // Run coordinator
    coordinator.run().await?;

    Ok(())
}
```

### Agent (Client)

```rust
use neural_trader_swarm::{QuicSwarmAgent, AgentType};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let coordinator_addr = "127.0.0.1:5000".parse()?;

    let mut agent = QuicSwarmAgent::connect(
        "agent-1".to_string(),
        AgentType::PatternMatcher,
        coordinator_addr,
    ).await?;

    // Run agent
    agent.run().await?;

    Ok(())
}
```

## Performance

| Metric | Value |
|--------|-------|
| Latency (p50) | <0.5ms |
| Latency (p99) | <1.0ms |
| Throughput | 1Gbps+ |
| Concurrent Streams | 1000+ |
| Connection Setup | 0-RTT |

## ReasoningBank Integration

The coordinator integrates with ReasoningBank to:
- Record agent experiences and outcomes
- Judge prediction accuracy
- Suggest adaptive parameter changes
- Track agent performance over time

Example:

```rust
// Agent sends pattern match result
let result = PatternMatchResult {
    pattern_type: "price_action".to_string(),
    similarity: 0.87,
    expected_outcome: 0.75,
    actual_outcome: Some(0.78),
    // ...
};

agent.send_pattern_result(result).await?;

// Coordinator records in ReasoningBank
// If similarity > 0.85 and outcome available:
//   - Record experience
//   - Judge accuracy
//   - Suggest adaptations if needed
```

## Development

Run tests:
```bash
cargo test -p neural-trader-swarm
```

Run benchmarks:
```bash
cargo bench -p neural-trader-swarm --features benchmarks
```

## License

MIT OR Apache-2.0
