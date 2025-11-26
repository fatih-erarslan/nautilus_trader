# Neural Trader Integration - Quick Start Guide

## 5-Minute Setup

### 1. Configuration

Copy the example configuration:

```bash
cd crates/integration
cp example.config.toml config.toml
```

Edit `config.toml` with your broker credentials:

```toml
[brokers.alpaca]
api_key = "YOUR_ALPACA_API_KEY"
api_secret = "YOUR_ALPACA_API_SECRET"
paper_trading = true
```

### 2. Build

```bash
cargo build --release
```

### 3. Run

**Option A: Start Trading**
```bash
./target/release/neural-trader start
```

**Option B: API Server**
```bash
./target/release/neural-trader serve --port 8080
```

**Option C: Single Strategy**
```bash
./target/release/neural-trader execute momentum
```

## CLI Commands

### Trading Operations

```bash
# Start trading system
neural-trader start

# Start in daemon mode
neural-trader start --daemon

# Stop trading
neural-trader stop

# Execute specific strategy
neural-trader execute momentum
```

### Portfolio & Risk

```bash
# Get portfolio status
neural-trader portfolio

# Perform risk analysis
neural-trader risk
```

### Neural Models

```bash
# Train a model
neural-trader train lstm --data ./data/training.csv

# List models
neural-trader models
```

### Reporting

```bash
# Generate monthly report
neural-trader report --period month

# Daily report
neural-trader report --period day
```

### System Management

```bash
# Health check
neural-trader health

# Start API server
neural-trader serve --host 0.0.0.0 --port 8080
```

## REST API Examples

### Health Check

```bash
curl http://localhost:8080/health
```

Response:
```json
{
  "broker_pool": {
    "status": "healthy",
    "message": "1/1 brokers healthy"
  },
  "strategy_manager": {
    "status": "healthy",
    "message": "1 strategies active"
  }
}
```

### Get Portfolio

```bash
curl http://localhost:8080/portfolio
```

Response:
```json
{
  "total_value": "100000.00",
  "cash": "50000.00",
  "positions": [
    {
      "symbol": "AAPL",
      "quantity": "100",
      "average_price": "150.00",
      "current_price": "155.00",
      "unrealized_pl": "500.00"
    }
  ]
}
```

### Risk Analysis

```bash
curl http://localhost:8080/risk
```

Response:
```json
{
  "var_95": "2500.00",
  "var_99": "5000.00",
  "max_drawdown": "0.15",
  "sharpe_ratio": "1.5",
  "alerts": []
}
```

### Execute Strategy

```bash
curl -X POST http://localhost:8080/strategies/momentum/execute
```

Response:
```json
{
  "strategy_name": "momentum",
  "orders": [
    {
      "symbol": "AAPL",
      "side": "buy",
      "quantity": "100",
      "status": "filled"
    }
  ],
  "profit_loss": "0.00"
}
```

## Library Usage

### Basic Example

```rust
use neural_trader_integration::{Config, NeuralTrader};

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Load config
    let config = Config::from_file("config.toml")?;

    // Initialize
    let trader = NeuralTrader::new(config).await?;

    // Start trading
    trader.start_trading().await?;

    // Get portfolio
    let portfolio = trader.get_portfolio().await?;
    println!("Portfolio value: ${}", portfolio.total_value);

    Ok(())
}
```

### Builder Pattern

```rust
use neural_trader_integration::NeuralTraderBuilder;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let trader = NeuralTraderBuilder::new()
        .with_config_file("config.toml")?
        .build()
        .await?;

    trader.start_trading().await?;

    Ok(())
}
```

### Execute Strategy

```rust
let result = trader.execute_strategy("momentum").await?;

println!("Strategy: {}", result.strategy_name);
println!("Orders: {}", result.orders.len());
println!("P&L: ${}", result.profit_loss);
```

### Risk Analysis

```rust
let risk = trader.analyze_risk().await?;

println!("VaR 95%: ${}", risk.var_95);
println!("Max Drawdown: {:.2}%", risk.max_drawdown * 100.0);
println!("Sharpe Ratio: {:.2}", risk.sharpe_ratio);

for alert in &risk.alerts {
    println!("[{:?}] {}", alert.level, alert.message);
}
```

### Train Model

```rust
use neural_trader_integration::types::ModelTrainingConfig;

let config = ModelTrainingConfig {
    model_type: "lstm".to_string(),
    training_data: "./data/train.csv".to_string(),
    parameters: serde_json::json!({}),
    validation_split: 0.2,
    epochs: 100,
};

let model_id = trader.train_model(config).await?;
println!("Model trained: {}", model_id);
```

### Generate Report

```rust
use neural_trader_integration::types::TimePeriod;

let report = trader.generate_report(TimePeriod::Month).await?;

println!("Total Return: {:.2}%", report.total_return * 100.0);
println!("Sharpe Ratio: {:.2}", report.sharpe_ratio);
println!("Max Drawdown: {:.2}%", report.max_drawdown * 100.0);
println!("Win Rate: {:.2}%", report.win_rate * 100.0);
```

## Environment Variables

Override configuration with environment variables:

```bash
# Broker settings
export NT_BROKERS__ALPACA__API_KEY="your_key"
export NT_BROKERS__ALPACA__API_SECRET="your_secret"

# Risk settings
export NT_RISK__MAX_POSITION_SIZE=50000
export NT_RISK__ENABLE_GPU=true

# API settings
export NT_API__REST__PORT=9000

# Run
neural-trader start
```

## Docker Quick Start

### Build Image

```dockerfile
# Dockerfile
FROM rust:1.75 as builder
WORKDIR /app
COPY . .
RUN cargo build --release --bin neural-trader

FROM debian:bookworm-slim
RUN apt-get update && apt-get install -y ca-certificates && rm -rf /var/lib/apt/lists/*
COPY --from=builder /app/target/release/neural-trader /usr/local/bin/
CMD ["neural-trader", "serve"]
```

```bash
docker build -t neural-trader .
```

### Run Container

```bash
docker run -d \
  --name neural-trader \
  -p 8080:8080 \
  -v $(pwd)/config.toml:/app/config.toml \
  -e NT_BROKERS__ALPACA__API_KEY=your_key \
  neural-trader
```

### Docker Compose

```yaml
version: '3.8'
services:
  neural-trader:
    build: .
    ports:
      - "8080:8080"
      - "9090:9090"
    volumes:
      - ./config.toml:/app/config.toml
      - ./data:/app/data
    environment:
      - NT_BROKERS__ALPACA__API_KEY=${ALPACA_KEY}
      - NT_BROKERS__ALPACA__API_SECRET=${ALPACA_SECRET}
```

```bash
docker-compose up -d
```

## Common Tasks

### Add New Broker

1. Add credentials to `config.toml`:
```toml
[brokers]
enabled = ["alpaca", "binance"]

[brokers.binance]
api_key = "your_key"
api_secret = "your_secret"
testnet = true
```

2. Restart:
```bash
neural-trader stop
neural-trader start
```

### Enable New Strategy

```toml
[strategies]
enabled = ["momentum", "mean_reversion"]

[strategies.mean_reversion]
z_score_threshold = 2.0
lookback_period = 30
```

### Adjust Risk Parameters

```toml
[risk]
max_position_size = 50000
max_portfolio_heat = 0.01
max_drawdown = 0.15
var_confidence = 0.99
```

## Troubleshooting

### Issue: Broker Connection Failed

**Solution**: Check credentials and network:
```bash
# Test connection
neural-trader health

# Check logs
tail -f logs/neural-trader.log
```

### Issue: Strategy Not Found

**Solution**: Ensure strategy is enabled:
```toml
[strategies]
enabled = ["momentum", "mean_reversion"]
```

### Issue: GPU Not Detected

**Solution**: Check GPU availability:
```bash
# Disable GPU if not available
export NT_RISK__ENABLE_GPU=false
```

### Issue: Port Already in Use

**Solution**: Change port:
```bash
neural-trader serve --port 9000
```

## Next Steps

1. **Configure Brokers**: Add your broker credentials
2. **Select Strategies**: Choose and configure strategies
3. **Set Risk Limits**: Configure risk parameters
4. **Test Paper Trading**: Verify with paper trading first
5. **Monitor Performance**: Track with `/health` endpoint
6. **Scale Up**: Deploy with Docker/Kubernetes

## Resources

- **Full Documentation**: `docs/integration-architecture.md`
- **API Reference**: Auto-generated from code
- **Configuration Guide**: `example.config.toml`
- **Examples**: `examples/` directory

## Support

- GitHub Issues: https://github.com/ruvnet/neural-trader/issues
- Documentation: https://docs.neural-trader.io
- Community: https://discord.gg/neural-trader
