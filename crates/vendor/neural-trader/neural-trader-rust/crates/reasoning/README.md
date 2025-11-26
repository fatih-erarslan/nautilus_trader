# ReasoningBank Self-Learning Engine

A Rust implementation of the ReasoningBank pattern learning system for adaptive trading strategies.

## Features

- **Experience Recording**: Track all pattern matching attempts with predicted and actual outcomes
- **Verdict Judgment**: Evaluate prediction quality with 0-1 quality scores
- **Memory Distillation**: Extract successful patterns into long-term memory
- **Adaptive Thresholds**: Automatically adjust matching parameters based on performance
- **Trajectory Analysis**: Build performance histories for pattern types
- **Financial Metrics**: Calculate Sharpe ratio, Sortino ratio, max drawdown, and more

## Architecture

```
PatternLearningEngine
├── Experience Recording
│   ├── Record pattern matches
│   ├── Store in AgentDB
│   └── Cache in memory
├── Outcome Tracking
│   ├── Update with actual results
│   ├── Calculate accuracy
│   └── Generate verdicts
├── Verdict Judgment
│   ├── Direction correctness
│   ├── Magnitude error
│   └── Quality scoring (0-1)
├── Memory Distillation
│   ├── Extract successful patterns
│   ├── Store in long-term memory
│   └── Build pattern clusters
└── Adaptive Learning
    ├── Suggest threshold changes
    ├── Apply adaptations
    └── Monitor performance
```

## Usage

### Basic Example

```rust
use reasoning::{PatternLearningEngine, PatternExperience, MarketContext};
use std::sync::Arc;

// Create engine with storage backend
let storage = Arc::new(MyStorage::new());
let engine = PatternLearningEngine::new(storage);

// Record a pattern match
let experience = PatternExperience {
    id: "exp_001".to_string(),
    pattern_type: "head_and_shoulders".to_string(),
    pattern_vector: vec![0.1, 0.2, 0.3, 0.4],
    similarity: 0.87,
    confidence: 0.79,
    predicted_outcome: 0.05, // Expect 5% return
    actual_outcome: None,    // Will fill later
    market_context: MarketContext {
        symbol: "BTC-USD".to_string(),
        timeframe: "4h".to_string(),
        volatility: 0.025,
        volume: 5000000.0,
        trend: "bullish".to_string(),
        sentiment: 0.65,
    },
    timestamp: Utc::now(),
};

engine.record_experience(experience).await?;

// Later, update with actual outcome
let verdict = engine.update_outcome("exp_001", 0.048).await?;

println!("Quality score: {:.2}", verdict.quality_score);
println!("Direction correct: {}", verdict.direction_correct);
println!("Should learn: {}", verdict.should_learn);
```

### Building Trajectories

```rust
// Analyze pattern performance over time
let trajectory = engine.build_trajectory("head_and_shoulders").await?;

println!("Samples: {}", trajectory.sample_count);
println!("Win rate: {:.1}%", trajectory.success_rate * 100.0);
println!("Avg return: {:.4}", trajectory.avg_return);
println!("Sharpe ratio: {:.2}", trajectory.sharpe_ratio);
```

### Adaptive Threshold Adjustment

```rust
// Automatically adapt thresholds based on recent performance
let thresholds = engine.adapt_thresholds(100).await?;

println!("Similarity threshold: {:.2}", thresholds.similarity_threshold);
println!("Confidence threshold: {:.2}", thresholds.confidence_threshold);
```

## Quality Scoring

The verdict judgment system uses a 0-1 quality score:

- **0.9-1.0**: Excellent prediction (direction + magnitude accurate)
- **0.8-0.9**: Good prediction (direction correct, magnitude close)
- **0.7-0.8**: Acceptable prediction (direction correct, some magnitude error)
- **0.5-0.7**: Poor prediction (direction correct, large magnitude error)
- **0.0-0.5**: Failed prediction (direction wrong)

Patterns with scores > 0.8 are distilled into long-term memory.
Patterns with scores < 0.4 trigger threshold adaptations.

## Performance Metrics

Available financial metrics:

- **Sharpe Ratio**: Risk-adjusted return (annualized)
- **Sortino Ratio**: Downside deviation only
- **Max Drawdown**: Largest peak-to-trough decline
- **Calmar Ratio**: Return / max drawdown
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / gross loss
- **VaR**: Value at Risk at confidence level
- **CVaR**: Conditional VaR (tail risk)

## Learning Curve

Expected performance improvement:

```
Success Rate Over Time
────────────────────────
Initial:  50-55% (baseline)
Week 1:   55-60% (basic learning)
Week 2:   60-65% (pattern recognition)
Month 1:  65-70% (refined thresholds)
Month 3:  70-75% (mature system)
```

## Integration

### With AgentDB

Implement the `PatternStorage` trait:

```rust
use reasoning::PatternStorage;

#[async_trait]
impl PatternStorage for MyAgentDB {
    async fn insert(&self, collection: &str, data: &JsonValue, vector: Option<&[f32]>) -> Result<String> {
        // Store in AgentDB
    }

    async fn query(&self, collection: &str, filter: &str, limit: usize) -> Result<Vec<JsonValue>> {
        // Query from AgentDB
    }

    async fn query_similar(&self, collection: &str, vector: &[f32], limit: usize) -> Result<Vec<JsonValue>> {
        // Vector similarity search
    }
}
```

### With Midstreamer

Use midstreamer for DTW/LCS pattern comparison:

```rust
let similarity = midstreamer.compare_dtw(
    &current_pattern,
    &historical_pattern.pattern_vector,
    None,
).await?;

if similarity.similarity >= thresholds.similarity_threshold {
    // Pattern matches - generate signal
}
```

## Testing

Run tests:

```bash
cargo test -p reasoning
```

Run benchmarks:

```bash
cargo bench -p reasoning
```

## License

MIT
