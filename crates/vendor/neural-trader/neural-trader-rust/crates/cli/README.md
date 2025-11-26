# Neural Trader CLI

Command-line interface for the Neural Trading platform.

## Installation

```bash
cargo build --release --package nt-cli
```

The binary will be available at `target/release/nt-cli`.

## Usage

### Display Version

```bash
neural-trader --version
```

### Display Help

```bash
neural-trader --help
```

### List Available Strategies

```bash
# List all strategies
neural-trader list-strategies

# List with details
neural-trader list-strategies --detailed

# Filter by category
neural-trader list-strategies --category "Mean Reversion"

# Output as JSON
neural-trader list-strategies --json
```

### List Available Brokers

```bash
# List all brokers
neural-trader list-brokers

# List with details
neural-trader list-brokers --detailed

# Filter by type
neural-trader list-brokers --type stocks
neural-trader list-brokers --type crypto

# Output as JSON
neural-trader list-brokers --json
```

### Run Backtesting

```bash
# Basic backtest
neural-trader backtest \
  --strategy pairs-trading \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --symbols AAPL,MSFT \
  --initial-capital 100000

# Run in E2B sandbox
neural-trader backtest \
  --strategy mean-reversion \
  --start 2023-01-01 \
  --symbols SPY \
  --sandbox \
  --cpu 4 \
  --memory 8

# Save results to file
neural-trader backtest \
  --strategy momentum \
  --start 2023-01-01 \
  --symbols TSLA \
  --output backtest-results.json
```

### Paper Trading

```bash
# Start paper trading
neural-trader trade \
  --strategy mean-reversion \
  --broker alpaca \
  --symbols AAPL,GOOGL \
  --paper \
  --capital 100000

# With risk management
neural-trader trade \
  --strategy momentum \
  --broker alpaca \
  --paper \
  --max-position 10000 \
  --stop-loss 2.0 \
  --take-profit 5.0
```

### Live Trading

```bash
# ⚠️ WARNING: This uses real money!
neural-trader trade \
  --strategy pairs-trading \
  --broker alpaca \
  --symbols AAPL,MSFT \
  --max-position 5000

# Dry run (no actual orders)
neural-trader trade \
  --strategy momentum \
  --broker ibkr \
  --symbols SPY \
  --dry-run
```

### Manage Secrets

```bash
# Set API key
neural-trader secrets set ALPACA_API_KEY your-api-key-here

# List stored secrets
neural-trader secrets list

# Remove secret
neural-trader secrets remove ALPACA_API_KEY
```

### Check Status

```bash
# View running trading agents
neural-trader status

# View detailed status
neural-trader status --detailed
```

## Configuration

The CLI supports configuration files in TOML format:

```bash
# Use custom config file
neural-trader --config my-config.toml backtest --strategy pairs-trading

# Use specific profile
neural-trader --profile production trade --strategy momentum --broker alpaca
```

Example configuration (`config.toml`):

```toml
[default]
broker = "alpaca"
initial_capital = 100000
max_position_size = 10000
stop_loss_pct = 2.0
take_profit_pct = 5.0

[production]
broker = "ibkr"
initial_capital = 500000
max_position_size = 50000
```

## Global Options

- `--config <file>` - Path to configuration file
- `--profile <name>` - Profile name to use from config
- `--verbose` - Enable verbose logging
- `--quiet` - Suppress output
- `--json` - Output in JSON format
- `--pretty` - Pretty-print JSON output

## Available Strategies

1. **pairs-trading** - Statistical arbitrage using cointegration
2. **mean-reversion** - Bollinger Bands mean reversion
3. **momentum** - RSI and MACD momentum strategy
4. **market-making** - Automated market making
5. **breakout** - Breakout detection with volume

## Available Brokers

1. **alpaca** - Stocks and crypto (US)
2. **ibkr** - Interactive Brokers (Global, all asset classes)
3. **polygon** - Market data provider (US)
4. **ccxt** - Cryptocurrency exchanges (Global)
5. **oanda** - Forex trading (Global)
6. **questrade** - Canadian broker

## Examples

### Full Backtest Workflow

```bash
# 1. List available strategies
neural-trader list-strategies

# 2. Run backtest
neural-trader backtest \
  --strategy pairs-trading \
  --start 2023-01-01 \
  --end 2023-12-31 \
  --symbols AAPL,MSFT \
  --output results.json

# 3. Start paper trading if backtest is good
neural-trader trade \
  --strategy pairs-trading \
  --broker alpaca \
  --symbols AAPL,MSFT \
  --paper
```

### Multi-Asset Portfolio

```bash
neural-trader trade \
  --strategy momentum \
  --broker alpaca \
  --symbols SPY,QQQ,IWM,DIA \
  --paper \
  --capital 200000 \
  --max-position 50000
```

### Options Trading

```bash
neural-trader trade \
  --strategy volatility-arbitrage \
  --broker ibkr \
  --symbols AAPL \
  --paper
```

## Environment Variables

- `ANTHROPIC_API_KEY` - Claude API key for AI features
- `ALPACA_API_KEY` - Alpaca API key
- `ALPACA_API_SECRET` - Alpaca API secret
- `IBKR_USERNAME` - Interactive Brokers username
- `IBKR_PASSWORD` - Interactive Brokers password
- `POLYGON_API_KEY` - Polygon.io API key

## Development

```bash
# Run CLI in development mode
cargo run --package nt-cli -- --help

# Run specific command
cargo run --package nt-cli -- list-strategies

# Run with verbose logging
cargo run --package nt-cli -- --verbose backtest --strategy pairs-trading
```

## Architecture

The CLI is structured as follows:

```
crates/cli/
├── src/
│   ├── main.rs              # Entry point and argument parsing
│   └── commands/            # Command implementations
│       ├── backtest.rs      # Backtesting command
│       ├── init.rs          # Initialize project
│       ├── list_brokers.rs  # List broker integrations
│       ├── list_strategies.rs # List trading strategies
│       ├── live.rs          # Live trading command
│       ├── paper.rs         # Paper trading command
│       ├── secrets.rs       # Secrets management
│       ├── status.rs        # Status monitoring
│       └── trade.rs         # Unified trading command
└── Cargo.toml
```

## Dependencies

- `clap` - Command-line argument parsing
- `colored` - Terminal colors
- `dialoguer` - Interactive prompts
- `indicatif` - Progress bars
- `tokio` - Async runtime
- `serde_json` - JSON serialization
- `anyhow` - Error handling

## Testing

```bash
# Run CLI tests
cargo test --package nt-cli

# Test specific command
cargo run --package nt-cli -- list-strategies --json
```

## Contributing

When adding new commands:

1. Create command file in `src/commands/`
2. Add module export to `src/commands/mod.rs`
3. Add command variant to `Commands` enum in `main.rs`
4. Implement command execution in match statement
5. Update this README with usage examples

## License

MIT OR Apache-2.0
