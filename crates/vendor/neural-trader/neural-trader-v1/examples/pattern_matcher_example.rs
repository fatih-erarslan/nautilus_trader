//! Pattern Matcher Strategy Example
//!
//! Demonstrates:
//! - Strategy initialization with AgentDB
//! - Real-time signal generation
//! - Pattern storage with outcomes
//! - Performance monitoring

use nt_strategies::{
    pattern_matcher::{PatternBasedStrategy, PatternMatcherConfig, PatternMetadata, PatternPerformance},
    Strategy, MarketData, Portfolio, Direction,
};
use nt_core::types::{Bar, Symbol};
use rust_decimal::Decimal;
use chrono::Utc;
use std::str::FromStr;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::INFO)
        .init();

    println!("\n=== DTW Pattern Matcher Strategy Example ===\n");

    // 1. Configure strategy
    let config = PatternMatcherConfig {
        window_size: 20,
        min_similarity: 0.75,
        top_k: 50,
        min_confidence: 0.65,
        lookback_hours: Some(720), // 30 days
        collection: "price_patterns".to_string(),
        use_wasm: false, // Set to true when WASM bindings are ready
        outcome_horizon: 5,
    };

    println!("Configuration:");
    println!("  Window size: {} bars", config.window_size);
    println!("  Min similarity: {:.0}%", config.min_similarity * 100.0);
    println!("  Top-K patterns: {}", config.top_k);
    println!("  Min confidence: {:.0}%", config.min_confidence * 100.0);
    println!("  WASM enabled: {}\n", config.use_wasm);

    // 2. Initialize strategy
    let strategy = PatternBasedStrategy::new(
        "pattern_matcher_example".to_string(),
        config,
        "http://localhost:8765".to_string(),
    )
    .await?;

    println!("✓ Strategy initialized");
    strategy.validate_config()?;
    println!("✓ Configuration validated\n");

    // 3. Create sample market data
    let symbol = "AAPL";
    let bars = generate_sample_bars(100, 150.0);
    let market_data = MarketData::new(symbol.to_string(), bars.clone());

    println!("Market Data:");
    println!("  Symbol: {}", symbol);
    println!("  Bars: {}", market_data.bars.len());
    println!("  Current price: ${:.2}\n", market_data.price.unwrap());

    // 4. Generate trading signals
    println!("Generating signals...");
    let portfolio = Portfolio::new(Decimal::from(100000));

    let start = std::time::Instant::now();
    let signals = strategy.process(&market_data, &portfolio).await?;
    let elapsed = start.elapsed();

    println!("✓ Signal generation completed in {:.2}ms\n", elapsed.as_millis());

    // 5. Display signals
    if signals.is_empty() {
        println!("No signals generated (no similar patterns found or confidence too low)");
    } else {
        for signal in &signals {
            println!("=== TRADING SIGNAL ===");
            println!("Direction: {}", signal.direction);
            println!("Symbol: {}", signal.symbol);
            println!("Confidence: {:.1}%", signal.confidence.unwrap_or(0.0) * 100.0);
            println!("Entry Price: ${:.2}", signal.entry_price.unwrap());
            println!("Stop Loss: ${:.2}", signal.stop_loss.unwrap());
            println!("Take Profit: ${:.2}", signal.take_profit.unwrap());
            println!("\nReasoning:");
            println!("{}", signal.reasoning.as_ref().unwrap());
            println!("\nFeatures (for neural training):");
            println!("  {:?}", signal.features);
            println!();
        }
    }

    // 6. Simulate pattern storage (learning)
    println!("=== PATTERN LEARNING ===\n");
    println!("Storing historical patterns with outcomes...");

    for i in 0..5 {
        let pattern = generate_sample_pattern(20);
        let outcome = (i as f64 * 0.01) - 0.02; // -2%, -1%, 0%, 1%, 2%

        let metadata = PatternMetadata {
            regime: if outcome > 0.0 { "bullish" } else { "bearish" }.to_string(),
            volatility: 0.15 + (i as f64 * 0.01),
            volume_profile: "normal".to_string(),
            quality_score: 0.80 + (i as f64 * 0.03),
            performance: Some(PatternPerformance {
                match_count: 1,
                success_rate: if outcome > 0.0 { 1.0 } else { 0.0 },
                avg_return: outcome,
                sharpe_ratio: 1.2 + (i as f64 * 0.1),
            }),
        };

        match strategy.store_pattern_with_outcome(
            symbol,
            pattern,
            outcome,
            metadata,
        ).await {
            Ok(_) => {
                println!("  ✓ Pattern {} stored (outcome: {:.2}%)", i + 1, outcome * 100.0);
            }
            Err(e) => {
                println!("  ✗ Pattern {} failed: {}", i + 1, e);
            }
        }
    }

    // 7. Display performance metrics
    println!("\n=== PERFORMANCE METRICS ===\n");
    let metrics = strategy.metrics();
    println!("Patterns matched: {}", metrics.patterns_matched);
    println!("Signals generated: {}", metrics.signals_generated);
    println!("Avg DTW time: {:.2}μs", metrics.avg_dtw_time_us);
    println!("Avg signal time: {:.2}μs", metrics.avg_signal_time_us);
    println!("Cache hit rate: {:.1}%", metrics.cache_hit_rate * 100.0);

    // 8. Strategy metadata
    println!("\n=== STRATEGY METADATA ===\n");
    let metadata = strategy.metadata();
    println!("Name: {}", metadata.name);
    println!("Version: {}", metadata.version);
    println!("Description: {}", metadata.description);
    println!("Tags: {:?}", metadata.tags);
    println!("Min capital: ${:.2}", metadata.min_capital);
    println!("Max drawdown: {:.1}%", metadata.max_drawdown_threshold * 100.0);

    println!("\n=== Example Complete ===\n");

    Ok(())
}

/// Generate sample bars for testing
fn generate_sample_bars(count: usize, start_price: f64) -> Vec<Bar> {
    let mut bars = Vec::with_capacity(count);
    let mut price = start_price;

    for i in 0..count {
        // Add some randomness
        let change = (i as f64 * 0.01).sin() * 2.0;
        price += change;

        let high = price + (price * 0.005);
        let low = price - (price * 0.005);
        let open = if i > 0 {
            bars[i - 1].close
        } else {
            Decimal::from_f64_retain(price).unwrap()
        };

        let bar = Bar {
            symbol: Symbol::from_str("AAPL").unwrap(),
            timestamp: Utc::now().timestamp() - ((count - i) as i64 * 300), // 5-min bars
            open,
            high: Decimal::from_f64_retain(high).unwrap(),
            low: Decimal::from_f64_retain(low).unwrap(),
            close: Decimal::from_f64_retain(price).unwrap(),
            volume: Decimal::from(1000000 + (i * 10000)),
        };

        bars.push(bar);
    }

    bars
}

/// Generate sample normalized pattern
fn generate_sample_pattern(size: usize) -> Vec<f64> {
    (0..size)
        .map(|i| {
            let x = (i as f64) / (size as f64);
            (x * std::f64::consts::PI * 2.0).sin() * 0.5 + 0.5
        })
        .collect()
}
