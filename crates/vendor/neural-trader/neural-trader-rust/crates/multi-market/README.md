# Multi-Market Trading Support

Comprehensive trading support across sports betting, prediction markets, and cryptocurrency markets.

## Features

### 🏈 Sports Betting
- **The Odds API Integration**: Real-time odds for 40+ sports
- **Kelly Criterion**: Optimal bet sizing calculator
- **Arbitrage Detection**: Cross-bookmaker opportunities
- **Syndicate Management**: Multi-person betting pools with profit distribution
- **Live Streaming**: Real-time odds updates via WebSocket and polling

### 🎲 Prediction Markets
- **Polymarket Integration**: CLOB API v2 client
- **Sentiment Analysis**: Market probability tracking and manipulation detection
- **Expected Value Calculator**: EV-based opportunity detection
- **Order Book Analysis**: Depth, liquidity, and market impact analysis
- **Market Making**: Automated market making strategies

### 💰 Cryptocurrency
- **DeFi Integration**: Beefy Finance, Yearn, yield farming protocols
- **Cross-Exchange Arbitrage**: Price differences across CEXs
- **Yield Optimization**: Auto-compounding and LP strategies
- **Gas Optimization**: Dynamic gas price optimization
- **MEV Protection**: Flashbots integration and private RPC

## Quick Start

```toml
[dependencies]
multi-market = { path = "../multi-market", features = ["all"] }
```

### Sports Betting Example

```rust
use multi_market::sports::*;
use rust_decimal_macros::dec;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize Odds API client
    let odds_client = OddsApiClient::new("your_api_key");

    // Get live odds
    let events = odds_client
        .get_odds(Sport::BasketballNba, &[Market::H2h], &["us"])
        .await?;

    // Kelly Criterion optimization
    let kelly = KellyOptimizer::new(dec!(10000), dec!(0.25));
    let opportunity = BettingOpportunity {
        event_id: "event_1".to_string(),
        outcome: "Lakers Win".to_string(),
        odds: dec!(2.5),
        win_probability: dec!(0.5),
        max_stake: None,
    };

    if let Some(result) = kelly.calculate(&opportunity)? {
        println!("Optimal stake: ${}", result.optimal_stake);
    }

    // Create betting syndicate
    let mut syndicate = Syndicate::new(
        "my_syndicate".to_string(),
        "Pro Bettors".to_string(),
        "Elite sports betting group".to_string(),
    );

    syndicate.add_member(
        "member1".to_string(),
        "John Doe".to_string(),
        "john@example.com".to_string(),
        MemberRole::Manager,
        dec!(5000),
    )?;

    Ok(())
}
```

### Prediction Markets Example

```rust
use multi_market::prediction::*;
use rust_decimal_macros::dec;

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize Polymarket client
    let polymarket = PolymarketClient::new("your_api_key");

    // Get active markets
    let markets = polymarket.get_markets().await?;

    // Calculate expected value
    let ev_calc = ExpectedValueCalculator::new(dec!(10000));
    for market in &markets {
        let opportunities = ev_calc.find_opportunities(market)?;
        for opp in opportunities {
            println!(
                "EV Opportunity: {} - Expected profit: ${}",
                opp.outcome, opp.profit_per_dollar
            );
        }
    }

    // Analyze sentiment
    let sentiment_analyzer = SentimentAnalyzer::new();
    for market in &markets {
        let sentiment = sentiment_analyzer.analyze(market)?;
        println!(
            "Market: {} - Sentiment: {}",
            market.question, sentiment.sentiment.overall
        );
    }

    Ok(())
}
```

### Cryptocurrency Example

```rust
use multi_market::crypto::*;
use rust_decimal_macros::dec;

#[tokio::main]
async fn main() -> Result<()> {
    // DeFi yield optimization
    let defi = DefiManager::new();
    let best_vaults = defi.find_best_yields(dec!(10000)).await?;

    for vault in best_vaults {
        println!(
            "Vault: {} - APY: {}% - TVL: ${}",
            vault.name, vault.apy, vault.tvl
        );

        let returns = defi.calculate_returns(&vault, dec!(1000), 365);
        println!("Expected 1-year return: ${}", returns);
    }

    // Cross-exchange arbitrage
    let arb_engine = ArbitrageEngine::new(dec!(1.0));
    if let Some(opp) = arb_engine.detect_arbitrage(
        "BTC/USD",
        dec!(50000),
        dec!(49500),
        "binance",
        "coinbase",
    ) {
        println!(
            "Arbitrage: {} -> {} - Profit: ${}",
            opp.exchange_a, opp.exchange_b, opp.base.profit_amount
        );
    }

    // Gas optimization
    let gas_optimizer = GasOptimizer::new(dec!(3000));
    let estimate = gas_optimizer.estimate_gas("swap");
    println!("Gas cost: {} ETH (${} USD)", estimate.total_cost_eth, estimate.total_cost_usd);

    Ok(())
}
```

## Architecture

```
multi-market/
├── src/
│   ├── sports/           # Sports betting module
│   │   ├── odds_api.rs   # The Odds API client
│   │   ├── kelly.rs      # Kelly Criterion calculator
│   │   ├── arbitrage.rs  # Arbitrage detector
│   │   ├── syndicate.rs  # Syndicate management
│   │   └── streaming.rs  # Real-time odds streaming
│   ├── prediction/       # Prediction markets module
│   │   ├── polymarket.rs # Polymarket CLOB API
│   │   ├── sentiment.rs  # Sentiment analysis
│   │   ├── expected_value.rs # EV calculator
│   │   ├── orderbook.rs  # Order book analysis
│   │   └── strategies.rs # Trading strategies
│   ├── crypto/           # Cryptocurrency module
│   │   ├── defi.rs       # DeFi protocol integration
│   │   ├── arbitrage.rs  # Cross-exchange arbitrage
│   │   ├── yield_farming.rs # Yield optimization
│   │   ├── gas.rs        # Gas optimization
│   │   └── strategies.rs # Trading strategies
│   ├── types.rs          # Common types
│   └── error.rs          # Error handling
└── tests/
    └── integration_test.rs
```

## Testing

```bash
# Run all tests
cargo test

# Run specific module tests
cargo test --features sports
cargo test --features prediction
cargo test --features crypto

# Run with live API (requires .env)
cargo test --features all -- --ignored
```

## Environment Variables

Create a `.env` file in your project root:

```env
# Sports Betting
ODDS_API_KEY=your_odds_api_key

# Prediction Markets
POLYMARKET_API_KEY=your_polymarket_key

# Cryptocurrency
ETHEREUM_RPC_URL=https://eth-mainnet.g.alchemy.com/v2/...
BSC_RPC_URL=https://bsc-dataseed.binance.org/
```

## Performance

- **Sports Betting**: 5 req/sec rate limiting, sub-100ms response times
- **Prediction Markets**: WebSocket streaming for real-time updates
- **Cryptocurrency**: Gas optimization reduces costs by 20-40%

## Risk Management Integration

Multi-market integrates with the `risk-management` crate for:
- Position sizing and leverage limits
- Portfolio-level risk metrics
- Stop-loss and take-profit automation
- Correlation analysis across markets

## AgentDB Integration

State and memory persistence via `agentdb`:
- Trading history and performance tracking
- Strategy optimization and backtesting
- Pattern recognition and learning

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md)

## License

MIT
