# Hive Mind Rust Backend

A comprehensive collective intelligence system for the Ximera trading platform, implemented in Rust with ultra-low latency and high-performance requirements.

## 🎯 Overview

The Hive Mind system provides distributed consensus, shared memory, neural pattern recognition, and agent coordination capabilities for the Ximera trading platform. It's designed to maintain sub-microsecond latency requirements while providing fault-tolerant collective intelligence.

## 🚀 Features

### Core Capabilities
- **Distributed Consensus**: Multi-algorithm support (Raft, PBFT, Gossip, Hybrid)
- **Collective Memory**: Knowledge graph with semantic search and intelligent caching
- **Neural Coordination**: Pattern recognition and federated learning
- **Agent Management**: Dynamic agent spawning and coordination
- **P2P Networking**: libp2p-based mesh networking with fault tolerance
- **Performance Monitoring**: Comprehensive metrics and alerting

### Performance Targets
- **Consensus Latency**: < 1ms for local cluster decisions
- **Memory Operations**: < 100μs for cache hits
- **Network Throughput**: > 100K messages/second
- **Agent Coordination**: < 10ms for task distribution

## 📁 Architecture

```
src/
├── lib.rs              # Main library interface
├── core.rs             # Core hive mind coordinator
├── consensus.rs        # Distributed consensus algorithms
├── memory.rs           # Collective memory and knowledge graph
├── neural.rs           # Neural pattern recognition
├── network.rs          # P2P networking and communication
├── agents.rs           # Agent management and coordination
├── metrics.rs          # Performance monitoring and metrics
├── config.rs           # Configuration management
├── error.rs            # Error handling
└── utils.rs            # Utility functions
```

## 🛠️ Installation

### Prerequisites
- Rust 1.70+ with Cargo
- tokio async runtime
- libp2p for networking
- Additional dependencies listed in `Cargo.toml`

### Build from Source
```bash
# Clone the repository
git clone https://github.com/your-org/ximera-backend
cd ximera-backend/src/backend/hive-mind-rust

# Build the project
cargo build --release

# Run tests
cargo test

# Run benchmarks
cargo bench
```

## ⚙️ Configuration

### Basic Configuration
```toml
[instance]
instance_id = "550e8400-e29b-41d4-a716-446655440000"

[network]
listen_addr = "0.0.0.0"
p2p_port = 8080
api_port = 8090
max_peers = 50

[consensus]
algorithm = "raft"
min_nodes = 3
timeout = "10s"
byzantine_threshold = 0.33

[memory]
max_pool_size = 1073741824  # 1GB
cleanup_interval = "5m"
replication_factor = 3

[neural]
enable_pattern_recognition = true
architecture = "transformer"
training_enabled = true

[agents]
max_agents = 100
spawning_strategy = "adaptive"
heartbeat_interval = "30s"

[metrics]
enabled = true
collection_interval = "60s"
prometheus_port = 9090
```

### Configuration Templates
Generate configuration templates:
```bash
# Minimal configuration
./target/release/hive-mind config --template minimal -o minimal.toml

# Production configuration
./target/release/hive-mind config --template production -o production.toml

# Development configuration
./target/release/hive-mind config --template development -o development.toml
```

## 🚀 Usage

### Starting the System
```bash
# Start with default configuration
./target/release/hive-mind start

# Start with custom configuration
./target/release/hive-mind start -c config.toml

# Start in daemon mode
./target/release/hive-mind start --daemon -p /var/run/hive-mind.pid

# Start with debug logging
./target/release/hive-mind start --log-level debug
```

### System Management
```bash
# Check system status
./target/release/hive-mind status

# Stop the system
./target/release/hive-mind stop -p /var/run/hive-mind.pid

# Validate configuration
./target/release/hive-mind validate -c config.toml

# Interactive shell
./target/release/hive-mind shell
```

### Benchmarking
```bash
# Run all benchmarks for 60 seconds
./target/release/hive-mind benchmark --duration 60

# Run specific benchmark
./target/release/hive-mind benchmark --bench-type consensus --duration 30

# Save results to file
./target/release/hive-mind benchmark --output results.json
```

## 🔧 API Usage

### Programmatic Interface
```rust
use hive_mind_rust::{HiveMind, HiveMindBuilder, HiveMindConfig};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create configuration
    let config = HiveMindConfig::default();
    
    // Build and start hive mind
    let hive_mind = HiveMindBuilder::new(config).build().await?;
    hive_mind.start().await?;
    
    // Submit a proposal for consensus
    let proposal = serde_json::json!({
        "action": "trade",
        "symbol": "BTC/USDT",
        "quantity": 1.0
    });
    let proposal_id = hive_mind.submit_proposal(proposal).await?;
    
    // Store knowledge in collective memory
    hive_mind.store_knowledge("market_analysis", serde_json::json!({
        "trend": "bullish",
        "confidence": 0.85
    })).await?;
    
    // Get neural insights
    let market_data = vec![100.0, 101.5, 102.0, 101.8, 103.2];
    let insights = hive_mind.get_neural_insights(&market_data).await?;
    
    // Spawn a specialized agent
    let agent_id = hive_mind.spawn_agent(vec![
        "market_analysis".to_string(),
        "risk_assessment".to_string()
    ]).await?;
    
    // Stop the system
    hive_mind.stop().await?;
    
    Ok(())
}
```

### REST API (when integrated)
```bash
# Submit consensus proposal
curl -X POST http://localhost:8090/api/consensus/propose \
  -H "Content-Type: application/json" \
  -d '{"action": "trade", "symbol": "BTC/USDT"}'

# Query collective memory
curl "http://localhost:8090/api/memory/search?q=market_analysis"

# Get system status
curl http://localhost:8090/api/status

# List active agents
curl http://localhost:8090/api/agents

# Get performance metrics
curl http://localhost:8090/metrics
```

## 📊 Monitoring

### Metrics Collection
The system provides comprehensive metrics through:
- **Prometheus endpoint**: `http://localhost:9090/metrics`
- **Internal metrics API**: Programmatic access to performance data
- **Real-time monitoring**: Live system health and performance tracking

### Key Metrics
- `hive_mind_consensus_operations_total` - Consensus operations
- `hive_mind_memory_operations_total` - Memory operations
- `hive_mind_neural_computations_total` - Neural computations
- `hive_mind_active_agents` - Number of active agents
- `hive_mind_network_messages_total` - Network messages
- `hive_mind_coordination_efficiency` - Agent coordination efficiency

### Alerting
Configure alerts based on:
- Performance thresholds
- Error rates
- Resource utilization
- Consensus failures
- Agent health issues

## 🧪 Testing

### Unit Tests
```bash
# Run all tests
cargo test

# Run specific test module
cargo test consensus

# Run tests with output
cargo test -- --nocapture

# Run tests in parallel
cargo test -- --test-threads=4
```

### Integration Tests
```bash
# Run integration tests
cargo test --test integration_tests

# Run performance tests
cargo test --test performance_tests
```

### Benchmarks
```bash
# Run all benchmarks
cargo bench

# Run specific benchmark
cargo bench consensus_benchmark

# Generate benchmark reports
cargo bench -- --output-format html
```

## 🔒 Security

### Security Features
- **End-to-end encryption** for all communications
- **Mutual authentication** between nodes
- **Byzantine fault tolerance** against malicious actors
- **Secure key rotation** and management
- **Input validation** and sanitization

### Security Configuration
```toml
[security]
enable_encryption = true
encryption_algorithm = "chacha20_poly1305"
key_rotation_interval = "24h"

[security.authentication]
method = "ed25519"
token_expiration = "1h"
enable_mutual_auth = true
```

## 🐛 Troubleshooting

### Common Issues

#### High Memory Usage
```bash
# Check memory configuration
./target/release/hive-mind validate -c config.toml

# Monitor memory metrics
curl http://localhost:9090/api/metrics | grep memory
```

#### Consensus Failures
```bash
# Check consensus logs
./target/release/hive-mind start --log-level debug

# Verify network connectivity
./target/release/hive-mind status --format json
```

#### Agent Coordination Issues
```bash
# List active agents
curl http://localhost:8090/api/agents

# Check agent performance
curl http://localhost:8090/api/agents/metrics
```

### Performance Tuning

#### For High-Frequency Trading
```toml
[consensus]
algorithm = "raft"
timeout = "1s"
min_nodes = 3

[memory]
max_pool_size = 2147483648  # 2GB
cleanup_interval = "1m"

[agents]
max_agents = 50
spawning_strategy = "pre_spawn"
```

#### For Memory-Constrained Environments
```toml
[memory]
max_pool_size = 134217728  # 128MB
enable_compression = true

[agents]
max_agents = 10
spawning_strategy = "on_demand"
```

## 🤝 Integration with Ximera

### Trading Engine Integration
```rust
use hive_mind_rust::HiveMind;
use ximera_trading::{TradingEngine, MarketData};

async fn integrate_with_trading_engine(
    hive_mind: &HiveMind,
    trading_engine: &TradingEngine,
) -> Result<(), Box<dyn std::error::Error>> {
    // Subscribe to market data
    let mut market_stream = trading_engine.market_data_stream().await?;
    
    while let Some(market_data) = market_stream.next().await {
        // Analyze market data with neural insights
        let insights = hive_mind.get_neural_insights(&market_data.prices).await?;
        
        // Submit trading decision to consensus
        let proposal = serde_json::json!({
            "action": "trade",
            "symbol": market_data.symbol,
            "decision": insights,
            "timestamp": chrono::Utc::now()
        });
        
        let proposal_id = hive_mind.submit_proposal(proposal).await?;
        
        // Store market insights in collective memory
        hive_mind.store_knowledge(
            &format!("market_insights_{}", market_data.symbol),
            insights
        ).await?;
    }
    
    Ok(())
}
```

## 📈 Performance Benchmarks

### Consensus Performance
- **Local consensus**: < 1ms latency
- **Network consensus**: < 10ms latency
- **Throughput**: > 10,000 proposals/second

### Memory Performance
- **Cache hits**: < 100μs latency
- **Knowledge search**: < 1ms for complex queries
- **Throughput**: > 100,000 operations/second

### Neural Performance
- **Pattern recognition**: < 5ms for 1000-point datasets
- **Model inference**: < 10ms for trained models
- **Training updates**: < 100ms for federated learning

## 🛣️ Roadmap

### Version 0.2.0
- [ ] WebAssembly support for browser deployment
- [ ] Enhanced neural architectures (GPT, BERT)
- [ ] Advanced consensus algorithms (HotStuff, Tendermint)
- [ ] GPU acceleration for neural processing

### Version 0.3.0
- [ ] Multi-cloud deployment support
- [ ] Advanced anomaly detection
- [ ] Real-time adaptation and learning
- [ ] Enhanced security features

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Ximera trading platform team
- Rust community for excellent crates
- libp2p team for networking foundation
- Tokio team for async runtime

## 📞 Support

For support, please:
1. Check the [troubleshooting section](#troubleshooting)
2. Review existing [issues](https://github.com/your-org/ximera-backend/issues)
3. Create a new issue with detailed information
4. Join our [Discord community](https://discord.gg/ximera)

---

**Note**: This is part of the larger Ximera trading platform ecosystem. For complete deployment, refer to the main Ximera documentation.