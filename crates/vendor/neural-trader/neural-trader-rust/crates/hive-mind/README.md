# nt-hive-mind

Distributed multi-agent coordination system for Neural Trader, implementing collective intelligence, consensus building, and fault-tolerant task orchestration.

## Features

- **Queen Coordinator**: Central orchestration with intelligent task delegation
- **Worker Agents**: Specialized agents (Researcher, Coder, Tester, etc.)
- **Distributed Memory**: Shared memory with conflict resolution
- **Consensus Building**: Multiple algorithms (Majority, Unanimous, Weighted, Byzantine)
- **Fault Tolerance**: Self-healing and recovery mechanisms
- **Collective Intelligence**: Democratic decision-making

## Architecture

```
┌─────────────────────────────────────────┐
│           HiveMind System               │
├─────────────────────────────────────────┤
│                                         │
│  ┌──────────┐         ┌──────────────┐ │
│  │  Queen   │────────▶│   Workers    │ │
│  │          │         │ (Specialized)│ │
│  └──────────┘         └──────────────┘ │
│       │                      │          │
│       ▼                      ▼          │
│  ┌──────────────────────────────────┐  │
│  │    Distributed Memory            │  │
│  │  (Shared State & Results)        │  │
│  └──────────────────────────────────┘  │
│                   │                     │
│                   ▼                     │
│  ┌──────────────────────────────────┐  │
│  │    Consensus Builder             │  │
│  │  (Democratic Decision Making)    │  │
│  └──────────────────────────────────┘  │
└─────────────────────────────────────────┘
```

## Usage

### Basic Setup

```rust
use nt_hive_mind::{HiveMind, HiveMindConfig, AgentType};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Create hive mind with default configuration
    let config = HiveMindConfig::default();
    let mut hive = HiveMind::new(config)?;

    // Spawn specialized worker agents
    hive.spawn_worker(AgentType::Researcher, "research-agent".to_string()).await?;
    hive.spawn_worker(AgentType::Coder, "coder-agent".to_string()).await?;
    hive.spawn_worker(AgentType::Tester, "tester-agent".to_string()).await?;

    // Orchestrate a complex task
    let result = hive.orchestrate_task(
        "Research market trends, implement a trading strategy, and test it"
    ).await?;

    println!("Task result: {:?}", result);

    // Get status
    let status = hive.status().await;
    println!("Hive status: {:?}", status);

    // Shutdown gracefully
    hive.shutdown().await?;

    Ok(())
}
```

### Custom Configuration

```rust
use nt_hive_mind::{
    HiveMindConfig, QueenConfig, MemoryConfig,
    ConsensusConfig, ConsensusAlgorithm
};

let config = HiveMindConfig {
    max_workers: 20,
    queen_config: QueenConfig {
        name: "Trading-Queen".to_string(),
        max_concurrent_tasks: 50,
        intelligent_delegation: true,
    },
    memory_config: MemoryConfig {
        max_entries: 50_000,
        persistence: true,
        cache_size_mb: 512,
    },
    consensus_config: ConsensusConfig {
        threshold: 0.75, // 75% agreement required
        algorithm: ConsensusAlgorithm::Byzantine,
        timeout_secs: 120,
    },
    fault_tolerance: true,
    collective_intelligence: true,
};

let hive = HiveMind::new(config)?;
```

## Agent Types

The system supports multiple specialized agent types:

- **Researcher**: Research and analysis tasks
- **Coder**: Code implementation and development
- **Tester**: Testing and validation
- **Architect**: System design and architecture
- **Reviewer**: Code review and auditing
- **Optimizer**: Performance optimization
- **Documenter**: Documentation creation
- **Coordinator**: General coordination (can handle any task)
- **Custom(String)**: Custom agent types

## Consensus Algorithms

### Majority Voting
```rust
ConsensusAlgorithm::Majority  // Simple majority (default 67%)
```

### Unanimous Agreement
```rust
ConsensusAlgorithm::Unanimous  // All agents must agree
```

### Weighted Voting
```rust
ConsensusAlgorithm::Weighted  // Based on agent expertise
```

### Byzantine Fault Tolerant
```rust
ConsensusAlgorithm::Byzantine  // BFT consensus (2f+1 out of 3f+1)
```

## Memory Management

The distributed memory system provides:

- **Shared State**: All agents can read/write to shared memory
- **Conflict Resolution**: CRDT-like conflict handling
- **Task Results**: Automatic storage of task execution results
- **Persistence**: Optional disk persistence
- **Statistics**: Memory usage tracking

```rust
// Access memory directly
let memory = hive.memory();

// Store data
memory.store("key".to_string(), "value".to_string()).await?;

// Retrieve data
let value = memory.retrieve("key").await?;

// Get statistics
let stats = memory.stats().await;
println!("Memory usage: {:.2}%", stats.usage_percent);
```

## Testing

Run the test suite:

```bash
cargo test -p nt-hive-mind
```

Run with logging:

```bash
RUST_LOG=debug cargo test -p nt-hive-mind -- --nocapture
```

## Performance

The hive mind system is designed for high performance:

- **Concurrent Execution**: Tasks run in parallel across workers
- **Lock-Free Data Structures**: DashMap for minimal contention
- **Async/Await**: Tokio-based async runtime
- **Memory Efficient**: Configurable cache sizes and limits

## Examples

See the `examples/` directory for complete examples:

- `basic_hive.rs`: Simple hive mind setup
- `consensus_demo.rs`: Consensus algorithm examples
- `memory_coordination.rs`: Distributed memory usage
- `fault_tolerance.rs`: Fault tolerance demonstrations

## Integration

The nt-hive-mind crate integrates with:

- **nt-core**: Core trading types
- **nt-neural**: Neural network coordination
- **nt-strategies**: Strategy development workflows
- **mcp-server**: MCP server coordination

## License

Licensed under either of Apache License, Version 2.0 or MIT license at your option.
