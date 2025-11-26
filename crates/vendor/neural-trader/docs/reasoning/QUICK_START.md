# ReasoningBank Quick Start Guide

Get started with the ReasoningBank self-learning engine in 5 minutes.

## Installation

Add to your `Cargo.toml`:

```toml
[dependencies]
reasoning = { path = "../reasoning" }
```

## Basic Usage

### 1. Create the Learning Engine

```rust
use reasoning::PatternLearningEngine;
use std::sync::Arc;

// Implement storage backend (see integration guide)
let storage = Arc::new(MyStorage::new());
let engine = PatternLearningEngine::new(storage);
```

### 2. Record a Pattern Match

```rust
use reasoning::{PatternExperience, MarketContext};
use chrono::Utc;

let experience = PatternExperience {
    id: uuid::Uuid::new_v4().to_string(),
    pattern_type: "double_bottom".to_string(),
    pattern_vector: vec![0.1, 0.2, 0.3, 0.4, 0.5],
    similarity: 0.85,
    confidence: 0.78,
    predicted_outcome: 0.04,  // Expect 4% return
    actual_outcome: None,      // Will fill later
    market_context: MarketContext {
        symbol: "BTC-USD".to_string(),
        timeframe: "1h".to_string(),
        volatility: 0.025,
        volume: 1_000_000.0,
        trend: "bullish".to_string(),
        sentiment: 0.6,
    },
    timestamp: Utc::now(),
};

engine.record_experience(experience).await?;
```

### 3. Update with Actual Outcome

```rust
// After trade completes, update with actual return
let verdict = engine
    .update_outcome("experience_id", 0.038)
    .await?;

// Check prediction quality
println!("Quality score: {:.2}", verdict.quality_score);
println!("Direction correct: {}", verdict.direction_correct);
println!("Should learn from this: {}", verdict.should_learn);

// Suggested adaptations
for adaptation in verdict.suggested_changes {
    println!("Adapt {}: {:.2} -> {:.2}",
        adaptation.parameter,
        adaptation.current_value,
        adaptation.suggested_value
    );
}
```

### 4. Build Performance Trajectory

```rust
// Analyze pattern performance over time
let trajectory = engine
    .build_trajectory("double_bottom")
    .await?;

println!("Pattern: {}", trajectory.pattern_type);
println!("Samples: {}", trajectory.sample_count);
println!("Win rate: {:.1}%", trajectory.success_rate * 100.0);
println!("Avg return: {:.4}", trajectory.avg_return);
println!("Sharpe ratio: {:.2}", trajectory.sharpe_ratio);
println!("Best: {:.4}, Worst: {:.4}",
    trajectory.best_return,
    trajectory.worst_return
);
```

### 5. Adaptive Threshold Adjustment

```rust
// Automatically adjust thresholds based on recent performance
let thresholds = engine.adapt_thresholds(100).await?;

println!("Current thresholds:");
println!("  Similarity: {:.2}", thresholds.similarity_threshold);
println!("  Confidence: {:.2}", thresholds.confidence_threshold);
println!("  Min samples: {}", thresholds.min_sample_size);
```

## Complete Example

```rust
use reasoning::{
    PatternLearningEngine, PatternExperience, MarketContext,
};
use std::sync::Arc;
use chrono::Utc;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Setup
    let storage = Arc::new(MyStorage::new());
    let engine = PatternLearningEngine::new(storage);

    // Simulate pattern detection and trading
    for i in 0..100 {
        // Detect pattern
        let exp_id = format!("exp_{:03}", i);
        let predicted_return = 0.05;

        let experience = PatternExperience {
            id: exp_id.clone(),
            pattern_type: "head_and_shoulders".to_string(),
            pattern_vector: vec![0.1, 0.2, 0.3, 0.4],
            similarity: 0.85,
            confidence: 0.75,
            predicted_outcome: predicted_return,
            actual_outcome: None,
            market_context: MarketContext {
                symbol: "BTC-USD".to_string(),
                timeframe: "4h".to_string(),
                volatility: 0.02,
                volume: 5_000_000.0,
                trend: "bearish".to_string(),
                sentiment: 0.4,
            },
            timestamp: Utc::now(),
        };

        // Record experience
        engine.record_experience(experience).await?;

        // Simulate trade execution and outcome
        let actual_return = predicted_return * 0.95 + (rand::random::<f64>() - 0.5) * 0.02;

        // Update with outcome
        let verdict = engine.update_outcome(&exp_id, actual_return).await?;

        println!("Trade {}: predicted={:.4}, actual={:.4}, quality={:.2}",
            i,
            predicted_return,
            actual_return,
            verdict.quality_score
        );

        // Adapt every 20 trades
        if i > 0 && i % 20 == 0 {
            let thresholds = engine.adapt_thresholds(20).await?;
            println!("Adapted thresholds: similarity={:.2}, confidence={:.2}",
                thresholds.similarity_threshold,
                thresholds.confidence_threshold
            );
        }
    }

    // Final trajectory
    let trajectory = engine
        .build_trajectory("head_and_shoulders")
        .await?;

    println!("\nFinal Performance:");
    println!("  Win rate: {:.1}%", trajectory.success_rate * 100.0);
    println!("  Avg return: {:.4}", trajectory.avg_return);
    println!("  Sharpe ratio: {:.2}", trajectory.sharpe_ratio);

    Ok(())
}
```

## Financial Metrics

Calculate various risk and performance metrics:

```rust
use reasoning::metrics::*;

let returns = vec![0.01, 0.02, -0.01, 0.015, 0.03, -0.005];

// Risk-adjusted returns
let sharpe = calculate_sharpe_ratio(&returns);
let sortino = calculate_sortino_ratio(&returns);
let calmar = calculate_calmar_ratio(&returns);

// Risk metrics
let max_dd = calculate_max_drawdown(&returns);
let var_95 = calculate_var(&returns, 0.95);
let cvar_95 = calculate_cvar(&returns, 0.95);

// Performance metrics
let win_rate = calculate_win_rate(&returns);
let profit_factor = calculate_profit_factor(&returns);

println!("Sharpe: {:.2}, Sortino: {:.2}, Calmar: {:.2}", sharpe, sortino, calmar);
println!("Max DD: {:.2}%, VaR(95%): {:.4}, CVaR(95%): {:.4}",
    max_dd * 100.0, var_95, cvar_95);
println!("Win rate: {:.1}%, Profit factor: {:.2}",
    win_rate * 100.0, profit_factor);
```

## Integration with AgentDB

Implement the `PatternStorage` trait:

```rust
use reasoning::PatternStorage;
use async_trait::async_trait;
use anyhow::Result;
use serde_json::Value as JsonValue;

pub struct AgentDBStorage {
    client: AgentDBClient,
}

#[async_trait]
impl PatternStorage for AgentDBStorage {
    async fn insert(
        &self,
        collection: &str,
        data: &JsonValue,
        vector: Option<&[f32]>,
    ) -> Result<String> {
        self.client.insert(collection, data, vector).await
    }

    async fn query(
        &self,
        collection: &str,
        filter: &str,
        limit: usize,
    ) -> Result<Vec<JsonValue>> {
        self.client.query(collection, filter, limit).await
    }

    async fn query_similar(
        &self,
        collection: &str,
        vector: &[f32],
        limit: usize,
    ) -> Result<Vec<JsonValue>> {
        self.client.query_similar(collection, vector, limit).await
    }

    async fn update(
        &self,
        collection: &str,
        id: &str,
        data: &JsonValue,
    ) -> Result<()> {
        self.client.update(collection, id, data).await
    }
}
```

## Best Practices

### 1. Experience Recording
- Record immediately after pattern detection
- Include complete market context
- Use unique IDs for tracking

### 2. Outcome Updates
- Update as soon as trade completes
- Include slippage and fees in actual return
- Handle partial fills appropriately

### 3. Threshold Adaptation
- Run adaptation regularly (every 20-50 trades)
- Use appropriate lookback window (50-200 trades)
- Monitor adaptation frequency

### 4. Trajectory Analysis
- Build trajectories periodically (daily/weekly)
- Compare across different pattern types
- Track improvement over time

### 5. Quality Thresholds
- Learn from patterns with quality > 0.7
- Adapt when quality < 0.4
- Investigate patterns with quality 0.4-0.7

## Troubleshooting

### Low Quality Scores

**Problem:** Consistent low quality scores (< 0.5)

**Solutions:**
1. Increase similarity threshold
2. Increase confidence threshold
3. Improve pattern detection algorithm
4. Add more market context features

### No Adaptation Happening

**Problem:** Thresholds not changing

**Solutions:**
1. Ensure enough completed experiences (> 20)
2. Check if performance is in middle range (55-70%)
3. Increase adaptation sensitivity
4. Review suggested changes in verdicts

### Memory Usage Growing

**Problem:** Too many experiences in memory

**Solutions:**
1. Reduce `max_memory_size` (default 1000)
2. Implement storage pruning
3. Archive old experiences
4. Use trajectory summaries instead

## Next Steps

- Read the [Implementation Guide](./IMPLEMENTATION_COMPLETE.md)
- Review the [API Documentation](../README.md)
- Explore [Integration Examples](../../examples/)
- Check [Performance Benchmarks](../BENCHMARKS.md)

---

**Quick Reference:**
- Experience recording: `engine.record_experience(exp).await?`
- Outcome update: `engine.update_outcome(id, actual).await?`
- Trajectory: `engine.build_trajectory(pattern_type).await?`
- Adapt: `engine.adapt_thresholds(lookback).await?`
