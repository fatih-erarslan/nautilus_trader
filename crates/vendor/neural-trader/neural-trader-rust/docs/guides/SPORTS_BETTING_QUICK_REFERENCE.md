# Sports Betting Quick Reference

## Function Summary

| Function | Purpose | Status |
|----------|---------|--------|
| `get_sports_events()` | Fetch upcoming sports events | ✅ Implemented |
| `get_sports_odds()` | Get odds from multiple bookmakers | ✅ Implemented |
| `find_sports_arbitrage()` | Identify arbitrage opportunities | ✅ Implemented |
| `calculate_kelly_criterion()` | Calculate optimal bet size | ✅ Preserved (Original) |
| `execute_sports_bet()` | Place bets with validation | ✅ Implemented |

## Quick Examples

### 1. Get Events
```rust
let events = get_sports_events("basketball".to_string(), Some(7)).await?;
// Returns: Vec<SportsEvent> with games in next 7 days
```

### 2. Get Odds
```rust
let odds = get_sports_odds("soccer".to_string()).await?;
// Returns: Vec<BettingOdds> from 5 bookmakers (Bet365, DraftKings, etc.)
```

### 3. Find Arbitrage
```rust
let arb = find_sports_arbitrage("football".to_string(), Some(0.01)).await?;
// Returns: Vec<ArbitrageOpportunity> with profit margin >= 1%
// Each opportunity includes optimal stake distribution
```

### 4. Kelly Criterion
```rust
let kelly = calculate_kelly_criterion(
    0.55,    // 55% win probability
    2.0,     // 2.0 decimal odds
    1000.0   // $1000 bankroll
).await?;
// Returns: KellyCriterion with suggested_stake (max 25% of bankroll)
```

### 5. Execute Bet (Validation Mode)
```rust
let result = execute_sports_bet(
    "market-123".to_string(),
    "Team A".to_string(),
    100.0,   // $100 stake
    2.5,     // 2.5 odds
    Some(true)  // validate_only = true (safe mode)
).await?;
// Returns: BetExecution with status = "validated"
```

### 6. Execute Bet (Real Execution)
```rust
let result = execute_sports_bet(
    "market-123".to_string(),
    "Team A".to_string(),
    100.0,
    2.5,
    Some(false)  // validate_only = false (real execution)
).await?;
// Returns: BetExecution with status = "accepted"
```

## Error Handling

```rust
match execute_sports_bet(...).await {
    Ok(execution) => {
        println!("Bet placed: {}", execution.bet_id);
        println!("Potential return: ${}", execution.potential_return);
    }
    Err(e) => {
        // Handle NeuralTraderError::Sports variants
        eprintln!("Bet failed: {}", e);
    }
}
```

## Common Workflows

### Arbitrage Betting Workflow
```rust
// 1. Find opportunities
let opportunities = find_sports_arbitrage("basketball".to_string(), Some(0.02)).await?;

// 2. For each opportunity
for opp in opportunities {
    // 3. Place bet on home team
    let bet1 = execute_sports_bet(
        opp.event_id.clone(),
        "home".to_string(),
        opp.bet_home.stake,
        opp.bet_home.odds,
        Some(false)
    ).await?;

    // 4. Place bet on away team
    let bet2 = execute_sports_bet(
        opp.event_id,
        "away".to_string(),
        opp.bet_away.stake,
        opp.bet_away.odds,
        Some(false)
    ).await?;

    // Guaranteed profit: opp.profit_margin * total_stake
}
```

### Kelly Criterion Betting Workflow
```rust
// 1. Get odds
let odds = get_sports_odds("baseball".to_string()).await?;

// 2. Calculate probability (your own analysis)
let win_probability = 0.60;  // 60% chance

// 3. Use Kelly Criterion
let kelly = calculate_kelly_criterion(
    win_probability,
    odds[0].home_odds,
    5000.0  // $5000 bankroll
).await?;

// 4. Place bet with Kelly-recommended stake
let bet = execute_sports_bet(
    odds[0].event_id.clone(),
    "home".to_string(),
    kelly.suggested_stake,
    odds[0].home_odds,
    Some(false)
).await?;
```

## Validation Rules

| Parameter | Rule | Error Message |
|-----------|------|---------------|
| stake | > 0 | "Stake must be greater than 0" |
| stake | ≤ 100,000 | "Stake exceeds maximum allowed amount" |
| odds | ≥ 1.0 | "Odds must be at least 1.0" |
| market_id | not empty | "Market ID and selection are required" |
| selection | not empty | "Market ID and selection are required" |

## Data Types

### Input Types
- `sport: String` - "basketball", "soccer", "football", "baseball"
- `days_ahead: Option<u32>` - Default: 7
- `min_profit_margin: Option<f64>` - Default: 0.01 (1%)
- `probability: f64` - 0.0 to 1.0
- `odds: f64` - Decimal odds (≥ 1.0)
- `bankroll: f64` - Total available capital
- `stake: f64` - Bet amount (> 0, ≤ 100,000)
- `validate_only: Option<bool>` - Default: true

### Return Types
- `Vec<SportsEvent>` - List of upcoming events
- `Vec<BettingOdds>` - Odds from multiple bookmakers
- `Vec<ArbitrageOpportunity>` - Arbitrage opportunities
- `KellyCriterion` - Kelly calculation result
- `BetExecution` - Bet placement result

## Constants

```rust
// Maximum stake per bet
const MAX_STAKE: f64 = 100000.0;

// Kelly Criterion cap (prevents over-betting)
const KELLY_CAP: f64 = 0.25;  // 25% of bankroll

// Default lookback for events
const DEFAULT_DAYS_AHEAD: u32 = 7;

// Minimum arbitrage profit margin
const DEFAULT_MIN_PROFIT: f64 = 0.01;  // 1%
```

## Supported Sports

- `basketball` - NBA teams
- `soccer` / `football` - European football clubs
- `baseball` - MLB teams
- Other sports - Generic team names

## Supported Bookmakers

1. Bet365
2. DraftKings
3. FanDuel
4. Caesars
5. BetMGM

## Market Types

1. **Moneyline** - Win/loss betting
2. **Spread** - Point spread
3. **Totals** - Over/under

## File Locations

- **Implementation**: `/workspaces/neural-trader/neural-trader-rust/packages/neural-trader-backend/src/sports.rs`
- **Error Types**: `/workspaces/neural-trader/neural-trader-rust/packages/neural-trader-backend/src/error.rs`
- **Tests**: Lines 380-517 in sports.rs

## Testing

```bash
# Run all sports tests
cargo test --lib sports

# Run specific test
cargo test --lib test_calculate_kelly_criterion

# Run with output
cargo test --lib sports -- --nocapture
```

## Integration Points

### Current (Mock)
- Generates realistic mock data
- Simulates bookmaker odds
- Safe for testing and validation

### Future (Production)
- Replace mock with The Odds API
- Integrate real bookmaker APIs
- Add WebSocket for live odds
- Implement bet settlement tracking

## Performance

| Function | Complexity | Typical Time |
|----------|-----------|--------------|
| get_sports_events | O(n) | < 1ms |
| get_sports_odds | O(n*m) | < 10ms |
| find_sports_arbitrage | O(n*m²) | < 50ms |
| calculate_kelly_criterion | O(1) | < 1μs |
| execute_sports_bet | O(1) | < 1ms |

## Safety Features

✅ Stake limits enforced
✅ Validation-only mode default
✅ All inputs validated
✅ Strong type safety
✅ Comprehensive error messages
✅ Execution logging
✅ Kelly cap at 25%

## Production Checklist

- [ ] Add real API keys to environment
- [ ] Enable bookmaker API integration
- [ ] Configure rate limiting
- [ ] Set up bet history database
- [ ] Enable real-time odds updates
- [ ] Add authentication layer
- [ ] Configure monitoring alerts
- [ ] Test with small stakes first
- [ ] Set ENABLE_REAL_BETTING=true
- [ ] Review all stake limits

## Support

For detailed implementation information, see:
`/workspaces/neural-trader/neural-trader-rust/docs/SPORTS_BETTING_IMPLEMENTATION.md`
