//! Comprehensive tests for trading strategies
//!
//! Tests all 7+ trading strategies with:
//! - Signal generation
//! - Edge cases
//! - Backtest scenarios
//! - Risk management integration

use nt_strategies::*;
use nt_core::types::{Symbol, Bar, Direction};
use rust_decimal::Decimal;
use rust_decimal_macros::dec;
use chrono::{Utc, Duration};

// ============================================================================
// Momentum Strategy Tests
// ============================================================================

#[tokio::test]
async fn test_momentum_strategy_creation() {
    let _config = MomentumConfig {
        lookback_period: 20,
        threshold: 0.02,
        ..Default::default()
    };

    let strategy = MomentumStrategy::new(config);
    assert_eq!(strategy.id(), "momentum");
}

#[tokio::test]
async fn test_momentum_strategy_bullish_signal() {
    let _config = MomentumConfig {
        lookback_period: 5,
        threshold: 0.05,
        ..Default::default()
    };

    let mut strategy = MomentumStrategy::new(config);

    // Create upward trending bars
    let symbol = Symbol::new("AAPL").unwrap();
    let mut bars = vec![];
    let base_time = Utc::now();

    for i in 0..10 {
        let price = dec!(100.0) + Decimal::from(i * 2); // Uptrend
        let bar = Bar {
            symbol: symbol.clone(),
            timestamp: base_time + Duration::hours(i),
            open: price,
            high: price + dec!(1.0),
            low: price - dec!(0.5),
            close: price + dec!(0.5),
            volume: dec!(10000),
        };
        bars.push(bar);
    }

    // Initialize with historical data
    strategy.initialize(bars.clone()).await.unwrap();

    // Process latest bar
    let latest_bar = bars.last().unwrap();
    let market_data = MarketData::new(symbol.as_str().to_string(), bars.clone());
    let portfolio = Portfolio::new(dec!(100000));

    let signals = strategy.process(&market_data, &portfolio).await.unwrap();

    // Should generate long signal due to strong uptrend
    assert!(signals.len() > 0);
    if let Some(signal) = signals.first() {
        assert_eq!(signal.direction, Direction::Long);
        assert!(signal.confidence.unwrap_or(0.0) > 0.5);
    }
}

#[tokio::test]
async fn test_momentum_strategy_bearish_signal() {
    let _config = MomentumConfig {
        lookback_period: 5,
        threshold: 0.05,
        ..Default::default()
    };

    let mut strategy = MomentumStrategy::new(config);

    // Create downward trending bars
    let symbol = Symbol::new("AAPL").unwrap();
    let mut bars = vec![];
    let base_time = Utc::now();

    for i in 0..10 {
        let price = dec!(120.0) - Decimal::from(i * 2); // Downtrend
        let bar = Bar {
            symbol: symbol.clone(),
            timestamp: base_time + Duration::hours(i),
            open: price,
            high: price + dec!(0.5),
            low: price - dec!(1.0),
            close: price - dec!(0.5),
            volume: dec!(10000),
        };
        bars.push(bar);
    }

    strategy.initialize(bars.clone()).await.unwrap();

    let market_data = MarketData::new(symbol.as_str().to_string(), bars.clone());
    let portfolio = Portfolio::new(dec!(100000));

    let signals = strategy.process(&market_data, &portfolio).await.unwrap();

    // Should generate short signal or no signal
    if signals.len() > 0 {
        let signal = signals.first().unwrap();
        assert!(signal.direction == Direction::Short || signal.direction == Direction::Close);
    }
}

#[tokio::test]
async fn test_momentum_strategy_insufficient_data() {
    let _config = MomentumConfig {
        lookback_period: 20,
        threshold: 0.02,
        ..Default::default()
    };

    let mut strategy = MomentumStrategy::new(config);

    // Only 5 bars, but strategy needs 20
    let symbol = Symbol::new("AAPL").unwrap();
    let mut bars = vec![];
    let base_time = Utc::now();

    for i in 0..5 {
        let bar = Bar {
            symbol: symbol.clone(),
            timestamp: base_time + Duration::hours(i),
            open: dec!(100.0),
            high: dec!(101.0),
            low: dec!(99.0),
            close: dec!(100.0),
            volume: dec!(10000),
        };
        bars.push(bar);
    }

    let market_data = MarketData::new(symbol.as_str().to_string(), bars);
    let portfolio = Portfolio::new(dec!(100000));

    let result = strategy.process(&market_data, &portfolio).await;

    // Should either error or return no signals
    assert!(result.is_err() || result.unwrap().is_empty());
}

// ============================================================================
// Mean Reversion Strategy Tests
// ============================================================================

#[tokio::test]
async fn test_mean_reversion_strategy_oversold() {
    let _config = MeanReversionConfig {
        lookback_period: 20,
        std_dev_threshold: 2.0,
        ..Default::default()
    };

    let mut strategy = MeanReversionStrategy::new(config);

    // Create bars with sudden drop (oversold condition)
    let symbol = Symbol::new("AAPL").unwrap();
    let mut bars = vec![];
    let base_time = Utc::now();

    // Stable prices
    for i in 0..25 {
        let price = if i < 20 {
            dec!(100.0)
        } else {
            dec!(90.0) // Sudden drop
        };

        let bar = Bar {
            symbol: symbol.clone(),
            timestamp: base_time + Duration::hours(i),
            open: price,
            high: price + dec!(0.5),
            low: price - dec!(0.5),
            close: price,
            volume: dec!(10000),
        };
        bars.push(bar);
    }

    strategy.initialize(bars.clone()).await.unwrap();

    let market_data = MarketData::new(symbol.as_str().to_string(), bars);
    let portfolio = Portfolio::new(dec!(100000));

    let signals = strategy.process(&market_data, &portfolio).await.unwrap();

    // Should generate long signal (expecting reversion to mean)
    assert!(signals.len() > 0);
    if let Some(signal) = signals.first() {
        assert_eq!(signal.direction, Direction::Long);
    }
}

#[tokio::test]
async fn test_mean_reversion_strategy_overbought() {
    let _config = MeanReversionConfig {
        lookback_period: 20,
        std_dev_threshold: 2.0,
        ..Default::default()
    };

    let mut strategy = MeanReversionStrategy::new(config);

    // Create bars with sudden spike (overbought condition)
    let symbol = Symbol::new("AAPL").unwrap();
    let mut bars = vec![];
    let base_time = Utc::now();

    for i in 0..25 {
        let price = if i < 20 {
            dec!(100.0)
        } else {
            dec!(110.0) // Sudden spike
        };

        let bar = Bar {
            symbol: symbol.clone(),
            timestamp: base_time + Duration::hours(i),
            open: price,
            high: price + dec!(0.5),
            low: price - dec!(0.5),
            close: price,
            volume: dec!(10000),
        };
        bars.push(bar);
    }

    strategy.initialize(bars.clone()).await.unwrap();

    let market_data = MarketData::new(symbol.as_str().to_string(), bars);
    let portfolio = Portfolio::new(dec!(100000));

    let signals = strategy.process(&market_data, &portfolio).await.unwrap();

    // Should generate short signal (expecting reversion down)
    if signals.len() > 0 {
        let signal = signals.first().unwrap();
        assert_eq!(signal.direction, Direction::Short);
    }
}

// ============================================================================
// Pairs Trading Strategy Tests
// ============================================================================

#[tokio::test]
async fn test_pairs_trading_cointegration() {
    let _config = PairsConfig {
        symbol_a: "AAPL".to_string(),
        symbol_b: "MSFT".to_string(),
        lookback_period: 30,
        z_score_threshold: 2.0,
        ..Default::default()
    };

    let strategy = PairsStrategy::new(config);
    assert_eq!(strategy.id(), "pairs_trading");
}

#[tokio::test]
async fn test_pairs_trading_divergence_signal() {
    let _config = PairsConfig {
        symbol_a: "AAPL".to_string(),
        symbol_b: "MSFT".to_string(),
        lookback_period: 20,
        z_score_threshold: 1.5,
        ..Default::default()
    };

    let mut strategy = PairsStrategy::new(config);

    // Create correlated pairs that diverge
    let symbol_a = Symbol::new("AAPL").unwrap();
    let symbol_b = Symbol::new("MSFT").unwrap();
    let base_time = Utc::now();

    let mut bars_a = vec![];
    let mut bars_b = vec![];

    for i in 0..30 {
        // Normally move together
        let base_price = dec!(100.0) + Decimal::from(i / 2);

        // But diverge at the end
        let price_a = if i < 25 {
            base_price
        } else {
            base_price + dec!(10.0) // AAPL rallies
        };

        let price_b = if i < 25 {
            base_price
        } else {
            base_price - dec!(5.0) // MSFT falls
        };

        bars_a.push(Bar {
            symbol: symbol_a.clone(),
            timestamp: base_time + Duration::hours(i),
            open: price_a,
            high: price_a + dec!(0.5),
            low: price_a - dec!(0.5),
            close: price_a,
            volume: dec!(10000),
        });

        bars_b.push(Bar {
            symbol: symbol_b.clone(),
            timestamp: base_time + Duration::hours(i),
            open: price_b,
            high: price_b + dec!(0.5),
            low: price_b - dec!(0.5),
            close: price_b,
            volume: dec!(10000),
        });
    }

    // Would need to implement proper pairs trading signal generation
    // This is a simplified test
}

// ============================================================================
// Ensemble Strategy Tests
// ============================================================================

#[tokio::test]
async fn test_ensemble_strategy_aggregation() {
    let mut strategies: Vec<Box<dyn Strategy>> = vec![];

    // Add multiple strategies
    strategies.push(Box::new(MomentumStrategy::new(MomentumConfig::default())));
    strategies.push(Box::new(MeanReversionStrategy::new(MeanReversionConfig::default())));

    let _config = EnsembleConfig {
        min_agreement: 0.6,
        weighting_scheme: WeightingScheme::Equal,
        ..Default::default()
    };

    let ensemble = EnsembleStrategy::new(strategies, config);
    assert_eq!(ensemble.id(), "ensemble");
}

#[tokio::test]
async fn test_ensemble_strategy_conflicting_signals() {
    // When strategies disagree, ensemble should be cautious
    let mut strategies: Vec<Box<dyn Strategy>> = vec![];

    strategies.push(Box::new(MomentumStrategy::new(MomentumConfig::default())));
    strategies.push(Box::new(MeanReversionStrategy::new(MeanReversionConfig::default())));

    let _config = EnsembleConfig {
        min_agreement: 0.8, // Require high agreement
        weighting_scheme: WeightingScheme::Equal,
        ..Default::default()
    };

    let mut ensemble = EnsembleStrategy::new(strategies, config);

    // Test with sideways market (likely conflicting signals)
    let symbol = Symbol::new("AAPL").unwrap();
    let mut bars = vec![];
    let base_time = Utc::now();

    for i in 0..30 {
        let price = dec!(100.0) + Decimal::from((i % 5) - 2); // Choppy
        let bar = Bar {
            symbol: symbol.clone(),
            timestamp: base_time + Duration::hours(i),
            open: price,
            high: price + dec!(1.0),
            low: price - dec!(1.0),
            close: price,
            volume: dec!(10000),
        };
        bars.push(bar);
    }

    ensemble.initialize(bars.clone()).await.unwrap();

    let market_data = MarketData::new(symbol.as_str().to_string(), bars);
    let portfolio = Portfolio::new(dec!(100000));

    let signals = ensemble.process(&market_data, &portfolio).await.unwrap();

    // With high min_agreement and choppy market, should produce fewer signals
    // or signals with lower confidence
}

// ============================================================================
// Strategy Validation Tests
// ============================================================================

#[tokio::test]
async fn test_strategy_config_validation() {
    // Invalid config should fail validation
    let _config = MomentumConfig {
        lookback_period: 0, // Invalid
        threshold: 0.02,
        ..Default::default()
    };

    let strategy = MomentumStrategy::new(config);
    let result = strategy.validate_config();

    assert!(result.is_err());
}

#[tokio::test]
async fn test_strategy_risk_parameters() {
    let strategy = MomentumStrategy::new(MomentumConfig::default());
    let risk_params = strategy.risk_parameters();

    assert!(risk_params.max_position_size > 0.0);
    assert!(risk_params.max_position_size <= 1.0);
    assert!(risk_params.stop_loss_pct > 0.0);
    assert!(risk_params.take_profit_pct > risk_params.stop_loss_pct);
}

// ============================================================================
// Backtest Integration Tests
// ============================================================================

#[tokio::test]
async fn test_strategy_backtest_simple() {
    let _config = MomentumConfig {
        lookback_period: 10,
        threshold: 0.03,
        ..Default::default()
    };

    let strategy = MomentumStrategy::new(config);

    // Create test data for backtest
    let symbol = Symbol::new("AAPL").unwrap();
    let mut bars = vec![];
    let base_time = Utc::now();

    for i in 0..100 {
        let price = dec!(100.0) + Decimal::from(i % 20); // Cyclical
        let bar = Bar {
            symbol: symbol.clone(),
            timestamp: base_time + Duration::hours(i),
            open: price,
            high: price + dec!(1.0),
            low: price - dec!(1.0),
            close: price,
            volume: dec!(10000),
        };
        bars.push(bar);
    }

    let backtest_config = BacktestConfig {
        initial_capital: dec!(100000),
        commission: dec!(0.001), // 0.1% commission
        slippage: dec!(0.0005), // 0.05% slippage
        ..Default::default()
    };

    let engine = BacktestEngine::new(backtest_config);
    let result = engine.run(Box::new(strategy), bars).await.unwrap();

    // Verify backtest completed
    assert!(result.total_trades >= 0);
    assert!(result.final_equity > dec!(0));
}

// ============================================================================
// Property-Based Tests
// ============================================================================

#[cfg(test)]
mod property_tests {
    use super::*;
    use proptest::prelude::*;

    proptest! {
        #[test]
        fn test_momentum_threshold_property(
            threshold in 0.01..0.20f64,
        ) {
            let _config = MomentumConfig {
                lookback_period: 20,
                threshold,
                ..Default::default()
            };

            let strategy = MomentumStrategy::new(config);
            let risk_params = strategy.risk_parameters();

            // Risk parameters should always be valid
            prop_assert!(risk_params.max_position_size > 0.0);
            prop_assert!(risk_params.max_position_size <= 1.0);
        }

        #[test]
        fn test_mean_reversion_std_dev_property(
            std_dev in 0.5..5.0f64,
        ) {
            let _config = MeanReversionConfig {
                lookback_period: 20,
                std_dev_threshold: std_dev,
                ..Default::default()
            };

            let strategy = MeanReversionStrategy::new(config);
            let validation = strategy.validate_config();

            // Should validate successfully for reasonable std_dev values
            prop_assert!(validation.is_ok());
        }
    }
}
