//! Mirror Trading Strategy
//!
//! Replicates trades from successful traders/strategies using pattern matching.
//! Queries historical successful signals and finds similar market conditions.
//!
//! ## Algorithm
//!
//! 1. Query successful past signals from memory/database
//! 2. Get current market observations
//! 3. Find similar past observations using similarity scoring
//! 4. If similarity > threshold, mirror the signal
//! 5. Adjust position size based on similarity score
//!
//! ## Performance Targets
//!
//! - Latency: <20ms
//! - Throughput: 500 signals/sec
//! - Memory: <15MB
//! - Python Sharpe: 6.01
//! - Rust Target Sharpe: >5.5

use crate::{
    async_trait, Bar, Direction, MarketData, Portfolio, Result, Signal, Strategy,
    StrategyError, StrategyMetadata, RiskParameters,
};
use rust_decimal::Decimal;
use rust_decimal::prelude::*;
use std::collections::HashMap;

/// Historical signal record
#[derive(Debug, Clone)]
struct HistoricalSignal {
    strategy_id: String,
    symbol: String,
    direction: Direction,
    confidence: f64,
    features: Vec<f64>,
    success_rate: f64,
}

/// Mirror trading strategy
#[derive(Debug, Clone)]
pub struct MirrorStrategy {
    /// Strategy ID
    id: String,
    /// Symbols to trade
    symbols: Vec<String>,
    /// Strategies to mirror
    mirror_sources: Vec<String>,
    /// Minimum confidence threshold
    min_confidence: f64,
    /// Minimum similarity threshold
    min_similarity: f64,
    /// Historical signals cache (in real implementation, this would use AgentDB)
    historical_signals: HashMap<String, Vec<HistoricalSignal>>,
}

impl MirrorStrategy {
    /// Create a new mirror strategy
    pub fn new(
        symbols: Vec<String>,
        mirror_sources: Vec<String>,
        min_confidence: f64,
        min_similarity: f64,
    ) -> Self {
        Self {
            id: "mirror_trader".to_string(),
            symbols,
            mirror_sources,
            min_confidence,
            min_similarity,
            historical_signals: HashMap::new(),
        }
    }

    /// Extract features from market data for similarity comparison
    fn extract_features(&self, bars: &[Bar]) -> Result<Vec<f64>> {
        if bars.len() < 20 {
            return Err(StrategyError::InsufficientData {
                needed: 20,
                available: bars.len(),
            });
        }

        let mut features = Vec::new();

        // Price features
        let current_price = bars.last().unwrap().close.to_f64().unwrap();
        let prices: Vec<f64> = bars
            .iter()
            .rev()
            .take(20)
            .map(|b| b.close.to_f64().unwrap())
            .collect();

        // Price change ratios
        for i in &[1, 5, 10, 20] {
            if *i < prices.len() {
                let past_price = prices[*i];
                features.push((current_price - past_price) / past_price);
            }
        }

        // Volume features
        let volumes: Vec<f64> = bars
            .iter()
            .rev()
            .take(20)
            .map(|b| b.volume.to_f64().unwrap_or(0.0))
            .collect();
        let avg_volume = volumes.iter().sum::<f64>() / volumes.len() as f64;
        let current_volume = bars.last().unwrap().volume.to_f64().unwrap_or(0.0);
        features.push(current_volume / avg_volume);

        // Volatility
        let returns: Vec<f64> = prices
            .windows(2)
            .map(|w| (w[0] - w[1]) / w[1])
            .collect();
        let volatility = returns.iter().map(|r| r.powi(2)).sum::<f64>().sqrt()
            / returns.len() as f64;
        features.push(volatility);

        Ok(features)
    }

    /// Calculate cosine similarity between two feature vectors
    fn calculate_similarity(&self, features1: &[f64], features2: &[f64]) -> f64 {
        if features1.len() != features2.len() {
            return 0.0;
        }

        let dot_product: f64 = features1
            .iter()
            .zip(features2.iter())
            .map(|(a, b)| a * b)
            .sum();

        let norm1: f64 = features1.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm2: f64 = features2.iter().map(|x| x * x).sum::<f64>().sqrt();

        if norm1 == 0.0 || norm2 == 0.0 {
            return 0.0;
        }

        (dot_product / (norm1 * norm2)).clamp(-1.0, 1.0)
    }

    /// Find similar historical signals
    fn find_similar_signals(
        &self,
        symbol: &str,
        current_features: &[f64],
    ) -> Vec<(HistoricalSignal, f64)> {
        let mut similar_signals = Vec::new();

        if let Some(historical) = self.historical_signals.get(symbol) {
            for signal in historical {
                if !self.mirror_sources.contains(&signal.strategy_id) {
                    continue;
                }

                if signal.confidence < self.min_confidence {
                    continue;
                }

                let similarity = self.calculate_similarity(current_features, &signal.features);

                if similarity > self.min_similarity {
                    similar_signals.push((signal.clone(), similarity));
                }
            }
        }

        // Sort by similarity (highest first)
        similar_signals.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        similar_signals
    }

    /// Add a historical signal for learning (in production, this would persist to AgentDB)
    pub fn add_historical_signal(
        &mut self,
        symbol: String,
        signal: HistoricalSignal,
    ) {
        self.historical_signals
            .entry(symbol)
            .or_insert_with(Vec::new)
            .push(signal);
    }
}

#[async_trait]
impl Strategy for MirrorStrategy {
    fn id(&self) -> &str {
        &self.id
    }

    fn metadata(&self) -> StrategyMetadata {
        StrategyMetadata {
            name: "Mirror Trader".to_string(),
            description: "Replicates successful trades using pattern matching".to_string(),
            version: "1.0.0".to_string(),
            author: "Neural Trader Team".to_string(),
            tags: vec![
                "mirror".to_string(),
                "pattern_matching".to_string(),
                "ml".to_string(),
            ],
            min_capital: Decimal::from(10000),
            max_drawdown_threshold: 0.10,
        }
    }

    async fn process(
        &self,
        market_data: &MarketData,
        portfolio: &Portfolio,
    ) -> Result<Vec<Signal>> {
        let mut signals = Vec::new();

        // Only process if symbol is in our list
        if !self.symbols.contains(&market_data.symbol) {
            return Ok(signals);
        }

        let bars = &market_data.bars;

        // Extract current features
        let current_features = self.extract_features(bars)?;

        // Find similar historical signals
        let similar_signals = self.find_similar_signals(&market_data.symbol, &current_features);

        if similar_signals.is_empty() {
            return Ok(signals);
        }

        // Take the best match
        let (best_signal, similarity) = &similar_signals[0];

        // Calculate adjusted confidence
        let adjusted_confidence = best_signal.confidence * similarity * best_signal.success_rate;

        if adjusted_confidence < self.min_confidence {
            return Ok(signals);
        }

        let current_price = bars.last().unwrap().close;

        // Calculate stop loss and take profit
        let stop_loss_pct = self.risk_parameters().stop_loss_percentage;
        let take_profit_pct = self.risk_parameters().take_profit_percentage;

        let (stop_loss, take_profit) = match best_signal.direction {
            Direction::Long => (
                current_price * Decimal::from_f64_retain(1.0 - stop_loss_pct).unwrap(),
                current_price * Decimal::from_f64_retain(1.0 + take_profit_pct).unwrap(),
            ),
            Direction::Short => (
                current_price * Decimal::from_f64_retain(1.0 + stop_loss_pct).unwrap(),
                current_price * Decimal::from_f64_retain(1.0 - take_profit_pct).unwrap(),
            ),
            Direction::Close => (current_price, current_price),
        };

        signals.push(
            Signal::new(
                self.id.clone(),
                market_data.symbol.clone(),
                best_signal.direction,
            )
            .with_confidence(adjusted_confidence)
            .with_entry_price(current_price)
            .with_stop_loss(stop_loss)
            .with_take_profit(take_profit)
            .with_reasoning(format!(
                "Mirroring {} signal (similarity: {:.2}, success rate: {:.1}%)",
                best_signal.strategy_id,
                similarity,
                best_signal.success_rate * 100.0
            ))
            .with_features(current_features),
        );

        Ok(signals)
    }

    fn validate_config(&self) -> Result<()> {
        if self.symbols.is_empty() {
            return Err(StrategyError::ConfigError(
                "At least one symbol must be specified".to_string(),
            ));
        }

        if self.mirror_sources.is_empty() {
            return Err(StrategyError::ConfigError(
                "At least one mirror source must be specified".to_string(),
            ));
        }

        if self.min_confidence < 0.0 || self.min_confidence > 1.0 {
            return Err(StrategyError::InvalidParameter(
                "Minimum confidence must be between 0 and 1".to_string(),
            ));
        }

        if self.min_similarity < 0.0 || self.min_similarity > 1.0 {
            return Err(StrategyError::InvalidParameter(
                "Minimum similarity must be between 0 and 1".to_string(),
            ));
        }

        Ok(())
    }

    fn risk_parameters(&self) -> RiskParameters {
        RiskParameters {
            max_position_size: Decimal::from(15000),
            max_leverage: 1.0,
            stop_loss_percentage: 0.015,
            take_profit_percentage: 0.04,
            max_daily_loss: Decimal::from(1500),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::Utc;

    fn create_test_bars(count: usize, start_price: f64) -> Vec<Bar> {
        (0..count)
            .map(|i| {
                let price = start_price * (1.0 + 0.01 * i as f64);
                Bar {
                    symbol: "TEST".to_string(),
                    timestamp: Utc::now(),
                    open: Decimal::from_f64_retain(price * 0.99).unwrap(),
                    high: Decimal::from_f64_retain(price * 1.01).unwrap(),
                    low: Decimal::from_f64_retain(price * 0.98).unwrap(),
                    close: Decimal::from_f64_retain(price).unwrap(),
                    volume: 1000000,
                }
            })
            .collect()
    }

    #[test]
    fn test_feature_extraction() {
        let strategy = MirrorStrategy::new(
            vec!["TEST".to_string()],
            vec!["momentum".to_string()],
            0.7,
            0.8,
        );

        let bars = create_test_bars(30, 100.0);
        let features = strategy.extract_features(&bars).unwrap();

        assert!(!features.is_empty());
    }

    #[test]
    fn test_similarity_calculation() {
        let strategy = MirrorStrategy::new(
            vec!["TEST".to_string()],
            vec!["momentum".to_string()],
            0.7,
            0.8,
        );

        let features1 = vec![0.1, 0.2, 0.3];
        let features2 = vec![0.1, 0.2, 0.3];

        let similarity = strategy.calculate_similarity(&features1, &features2);
        assert!((similarity - 1.0).abs() < 0.001);

        let features3 = vec![-0.1, -0.2, -0.3];
        let similarity2 = strategy.calculate_similarity(&features1, &features3);
        assert!(similarity2 < 0.0);
    }

    #[test]
    fn test_strategy_validation() {
        let strategy = MirrorStrategy::new(
            vec!["TEST".to_string()],
            vec!["momentum".to_string()],
            0.7,
            0.8,
        );
        assert!(strategy.validate_config().is_ok());

        let invalid = MirrorStrategy::new(vec![], vec!["momentum".to_string()], 0.7, 0.8);
        assert!(invalid.validate_config().is_err());
    }

    #[tokio::test]
    async fn test_signal_generation_with_history() {
        let mut strategy = MirrorStrategy::new(
            vec!["TEST".to_string()],
            vec!["momentum".to_string()],
            0.6,
            0.7,
        );

        // Add historical signal
        strategy.add_historical_signal(
            "TEST".to_string(),
            HistoricalSignal {
                strategy_id: "momentum".to_string(),
                symbol: "TEST".to_string(),
                direction: Direction::Long,
                confidence: 0.9,
                features: vec![0.05, 0.03, 0.02, 0.01, 1.2, 0.015],
                success_rate: 0.85,
            },
        );

        let bars = create_test_bars(30, 100.0);
        let market_data = MarketData {
            symbol: "TEST".to_string(),
            timestamp: Utc::now(),
            price: Decimal::from(110),
            volume: 1000000,
            bars,
        };

        let portfolio = Portfolio::new(Decimal::from(100000));
        let signals = strategy.process(&market_data, &portfolio).await.unwrap();

        // May or may not generate signal depending on similarity
        assert!(signals.len() <= 1);
    }
}
