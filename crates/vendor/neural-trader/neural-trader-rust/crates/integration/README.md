# Neural Trader Integration Layer

This crate provides the main integration layer that unifies all 17 neural-trader crates into a cohesive, production-ready trading system.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Neural Trader                            │
│                   (Main Facade)                             │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         │                    │                    │
    ┌────▼────┐        ┌─────▼─────┐       ┌─────▼─────┐
    │ Trading │        │ Analytics │       │   Risk    │
    │ Service │        │  Service  │       │  Service  │
    └────┬────┘        └─────┬─────┘       └─────┬─────┘
         │                   │                    │
    ┌────▼───────────────────▼────────────────────▼─────┐
    │            Coordination Layer                      │
    │  (BrokerPool, StrategyManager, ModelRegistry)     │
    └────┬───────────────────┬────────────────────┬─────┘
         │                   │                    │
    ┌────▼────┐        ┌─────▼─────┐       ┌─────▼─────┐
    │ Brokers │        │Strategies │       │  Neural   │
    │ (11)    │        │   (7+)    │       │ Models(3) │
    └─────────┘        └───────────┘       └───────────┘
```

## Components

### Core Components

- **NeuralTrader**: Main facade providing high-level API
- **Config**: Unified configuration management
- **Runtime**: Async runtime management
- **Error**: Unified error handling

### Services Layer

- **TradingService**: Coordinates strategies, brokers, and risk
- **AnalyticsService**: Performance tracking and reporting
- **RiskService**: Portfolio risk management
- **NeuralService**: Model training and inference

### Coordination Layer

- **BrokerPool**: Manages connections to 11 brokers
- **StrategyManager**: Orchestrates 7+ trading strategies
- **ModelRegistry**: Neural model lifecycle management
- **MemoryCoordinator**: Cross-service memory sharing

### API Layer

- **REST API**: HTTP endpoints for external access
- **WebSocket**: Real-time data streaming
- **CLI**: Command-line interface

## Usage

### As a Library

```rust
use neural_trader_integration::{Config, NeuralTrader};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load configuration
    let config = Config::from_file("config.toml")?;

    // Initialize the system
    let trader = NeuralTrader::new(config).await?;

    // Start trading
    trader.start_trading().await?;

    // Execute a strategy
    let result = trader.execute_strategy("momentum").await?;
    println!("Execution result: {:?}", result);

    // Get portfolio
    let portfolio = trader.get_portfolio().await?;
    println!("Portfolio value: ${}", portfolio.total_value);

    // Analyze risk
    let risk_report = trader.analyze_risk().await?;
    println!("VaR 95%: ${}", risk_report.var_95);

    Ok(())
}
```

### As a Binary (CLI)

```bash
# Start trading system
neural-trader start

# Execute a strategy
neural-trader execute momentum

# Get portfolio status
neural-trader portfolio

# Perform risk analysis
neural-trader risk

# Train a model
neural-trader train lstm --data ./data/training.csv

# Generate performance report
neural-trader report --period month

# Health check
neural-trader health

# Start API server
neural-trader serve --port 8080
```

### REST API

```bash
# Health check
curl http://localhost:8080/health

# Get portfolio
curl http://localhost:8080/portfolio

# Risk analysis
curl http://localhost:8080/risk

# List strategies
curl http://localhost:8080/strategies

# Execute strategy
curl -X POST http://localhost:8080/strategies/momentum/execute

# Train model
curl -X POST http://localhost:8080/models/train \
  -H "Content-Type: application/json" \
  -d '{"model_type": "lstm", "training_data": "./data/train.csv"}'
```

## Configuration

Configuration can be loaded from:

1. Config file (`config.toml`)
2. Environment variables (prefix: `NT_`)
3. CLI arguments

Example configuration:

```toml
[brokers]
enabled = ["alpaca", "binance"]

[brokers.alpaca]
api_key = "YOUR_KEY"
api_secret = "YOUR_SECRET"
paper_trading = true

[strategies]
enabled = ["momentum", "mean_reversion"]

[risk]
max_position_size = 100000
max_portfolio_heat = 0.02
enable_gpu = true

[neural]
models = ["lstm", "transformer"]
```

## Integration Points

This crate integrates:

1. **napi-bindings** - Node.js interface
2. **mcp-server** - MCP protocol
3. **execution** - 11 broker implementations
4. **neural** - 3 neural models
5. **strategies** - 7+ trading strategies
6. **risk** - Risk management (VaR, Kelly, limits)
7. **multi-market** - Sports, crypto, prediction markets
8. **memory** - AgentDB + ReasoningBank
9. **distributed** - E2B + federations
10. **testing** - Test infrastructure

## Performance

- **Latency**: <100ms end-to-end
- **Throughput**: 1000+ requests/sec
- **Zero-copy**: Where possible
- **Async**: Throughout the stack
- **GPU**: Enabled for risk and neural components

## Testing

```bash
# Run all tests
cargo test

# Run integration tests
cargo test --test integration_tests

# Run benchmarks
cargo bench
```

## Development

### Adding a New Broker

1. Add configuration in `src/config.rs`
2. Implement connection in `src/coordination/broker_pool.rs`
3. Add to enabled brokers list

### Adding a New Strategy

1. Add configuration in `src/config.rs`
2. Implement strategy in `src/coordination/strategy_manager.rs`
3. Add to enabled strategies list

### Adding a New API Endpoint

1. Add route in `src/api/rest.rs`
2. Implement handler
3. Update documentation

## License

MIT
