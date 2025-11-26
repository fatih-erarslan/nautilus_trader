# Neural Trader Syndicate - Rust Implementation

Comprehensive syndicate management system for collaborative sports betting with 100% feature parity with the Python implementation.

## Features

### Capital Management
- **Fund Allocation Engine**: Automated bankroll management with multiple allocation strategies
  - Kelly Criterion (fractional)
  - Fixed Percentage
  - Dynamic Confidence-based
  - Risk Parity
  - Martingale
  - Anti-Martingale
- **Bankroll Rules**: Configurable risk limits and constraints
- **Exposure Tracking**: Real-time monitoring of capital allocation
- **Profit Distribution**: Multiple distribution models (Hybrid, Proportional, Performance-weighted, Tiered)
- **Withdrawal Management**: Request processing with rules and penalties

### Member Management
- **Role-based Permissions**: 5 role types with 18 permission flags
- **Investment Tiers**: Bronze, Silver, Gold, Platinum
- **Performance Tracking**: Win rate, ROI, accuracy scoring
- **Voting Weight Calculation**: Capital + Performance + Tenure based
- **Member Statistics**: Comprehensive betting performance metrics

### Voting System
- **Proposal Creation**: Create votes with configurable voting periods
- **Weighted Voting**: Capital and performance-based voting weights
- **Vote Finalization**: Automatic outcome determination
- **Vote Tracking**: Full history and status monitoring

### Collaboration
- **Channels**: Public and private communication channels
- **Messaging**: Real-time syndicate communication
- **Member Coordination**: Organized collaboration tools

## Installation

Add to `Cargo.toml`:

```toml
[dependencies]
nt-syndicate = { path = "./crates/nt-syndicate" }
```

## Usage

### Fund Allocation

```rust
use nt_syndicate::{FundAllocationEngine, BettingOpportunity, AllocationStrategy};

let mut engine = FundAllocationEngine::new(
    "syndicate-123".to_string(),
    "100000.00".to_string()
)?;

let opportunity = BettingOpportunity {
    sport: "football".to_string(),
    event: "Team A vs Team B".to_string(),
    bet_type: "moneyline".to_string(),
    selection: "Team A".to_string(),
    odds: 2.0,
    probability: 0.55,
    edge: 0.10,
    confidence: 0.80,
    model_agreement: 0.90,
    time_until_event_secs: 3600,
    liquidity: 50000.0,
    is_live: false,
    is_parlay: false,
};

let result = engine.allocate_funds(opportunity, AllocationStrategy::KellyCriterion)?;
```

### Member Management

```rust
use nt_syndicate::{MemberManager, MemberRole};

let manager = MemberManager::new("syndicate-123".to_string());

let member = manager.add_member(
    "John Doe".to_string(),
    "john@example.com".to_string(),
    MemberRole::ContributingMember,
    "5000.00".to_string(),
)?;
```

### Voting

```rust
use nt_syndicate::VotingSystem;

let voting = VotingSystem::new("syndicate-123".to_string());

let vote_id = voting.create_vote(
    "strategy_change".to_string(),
    r#"{"description": "Switch to Kelly Criterion"}"#.to_string(),
    proposer_id,
    Some(48),
)?;

voting.cast_vote(vote_id, member_id, "approve".to_string(), 0.15)?;
```

## Architecture

```
nt-syndicate/
├── src/
│   ├── lib.rs           # Main exports and NAPI bindings
│   ├── types.rs         # Core types and enums (250+ lines)
│   ├── capital.rs       # Capital management (400+ lines)
│   ├── members.rs       # Member management (350+ lines)
│   ├── voting.rs        # Voting system (200+ lines)
│   └── collaboration.rs # Collaboration tools (150+ lines)
├── Cargo.toml           # Dependencies and build config
└── build.rs             # NAPI build script
```

## Type System

### Enums
- `AllocationStrategy`: 6 allocation methods
- `DistributionModel`: 4 distribution models
- `MemberRole`: 5 role types
- `MemberTier`: 4 investment tiers

### Structs
- `BankrollRules`: 9 risk management parameters
- `MemberPermissions`: 18 permission flags
- `BettingOpportunity`: Complete bet details
- `AllocationResult`: Allocation decision with reasoning
- `Member`: Full member profile
- `MemberStatistics`: Performance metrics
- `WithdrawalRequest`: Withdrawal processing

## Performance

- **Thread-safe**: Uses `DashMap` for concurrent access
- **Efficient**: Rust decimal precision with no floating point errors
- **Fast**: Compiled to native code for maximum performance
- **Memory-safe**: Zero-cost abstractions and borrow checker

## Testing

```bash
cargo test -p nt-syndicate
```

## Node.js Bindings

Build for Node.js:

```bash
npm install
npm run build
```

Usage in Node.js:

```javascript
const { FundAllocationEngine } = require('./index.node');

const engine = new FundAllocationEngine("syndicate-123", "100000.00");
const result = engine.allocateFunds(opportunity, "KellyCriterion");
```

## License

MIT
