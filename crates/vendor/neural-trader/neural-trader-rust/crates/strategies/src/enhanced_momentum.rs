//! Enhanced Momentum Strategy
//!
//! Advanced momentum strategy combining base momentum with ML signals and news sentiment.
//! Enhances traditional momentum trading with additional data sources.
//!
//! ## Algorithm
//!
//! 1. Get base momentum signals
//! 2. Enhance with news sentiment analysis
//! 3. Apply ML-based confidence adjustment
//! 4. Generate final signal with combined confidence
//!
//! ## Performance Targets
//!
//! - Latency: <50ms (includes news API calls)
//! - Throughput: 200 signals/sec
//! - Memory: <30MB
//! - Python Sharpe: 3.20
//! - Rust Target Sharpe: >3.0

use crate::{
    async_trait, Direction, MarketData, Portfolio, Result, Signal, Strategy,
    StrategyError, StrategyMetadata, RiskParameters,
    momentum::MomentumStrategy,
};
use rust_decimal::Decimal;

/// Enhanced momentum strategy with ML and sentiment
#[derive(Debug, Clone)]
pub struct EnhancedMomentumStrategy {
    /// Strategy ID
    id: String,
    /// Base momentum strategy
    base_momentum: MomentumStrategy,
    /// Sentiment weight (0.0-1.0)
    sentiment_weight: f64,
    /// ML model weight (0.0-1.0)
    ml_weight: f64,
}

impl EnhancedMomentumStrategy {
    /// Create a new enhanced momentum strategy
    pub fn new(
        symbols: Vec<String>,
        period: usize,
        entry_threshold: f64,
        exit_threshold: f64,
        sentiment_weight: f64,
        ml_weight: f64,
    ) -> Self {
        Self {
            id: "enhanced_momentum".to_string(),
            base_momentum: MomentumStrategy::new(symbols, period, entry_threshold, exit_threshold),
            sentiment_weight,
            ml_weight,
        }
    }

    /// Simulate sentiment analysis (placeholder for actual sentiment API)
    async fn analyze_sentiment(&self, symbol: &str) -> Result<f64> {
        // In production, this would call news APIs and sentiment analysis
        // For now, return neutral sentiment
        Ok(0.0) // Range: -1.0 (bearish) to 1.0 (bullish)
    }

    /// Simulate ML prediction (placeholder for actual ML model)
    async fn predict_ml_signal(&self, _features: &[f64]) -> Result<f64> {
        // In production, this would run inference on trained ML model
        // For now, return neutral prediction
        Ok(0.5) // Range: 0.0 to 1.0
    }

    /// Calculate enhanced confidence
    fn calculate_enhanced_confidence(
        &self,
        base_confidence: f64,
        sentiment_score: f64,
        ml_score: f64,
    ) -> f64 {
        let base_weight = 1.0 - self.sentiment_weight - self.ml_weight;

        let weighted_confidence = base_confidence * base_weight
            + (sentiment_score + 1.0) / 2.0 * self.sentiment_weight // Convert [-1,1] to [0,1]
            + ml_score * self.ml_weight;

        weighted_confidence.clamp(0.0, 1.0)
    }
}

#[async_trait]
impl Strategy for EnhancedMomentumStrategy {
    fn id(&self) -> &str {
        &self.id
    }

    fn metadata(&self) -> StrategyMetadata {
        StrategyMetadata {
            name: "Enhanced Momentum".to_string(),
            description: "Advanced momentum with ML signals and news sentiment".to_string(),
            version: "1.0.0".to_string(),
            author: "Neural Trader Team".to_string(),
            tags: vec![
                "momentum".to_string(),
                "ml".to_string(),
                "sentiment".to_string(),
                "enhanced".to_string(),
            ],
            min_capital: Decimal::from(10000),
            max_drawdown_threshold: 0.12,
        }
    }

    async fn process(
        &self,
        market_data: &MarketData,
        portfolio: &Portfolio,
    ) -> Result<Vec<Signal>> {
        // Get base momentum signals
        let mut signals = self.base_momentum.process(market_data, portfolio).await?;

        if signals.is_empty() {
            return Ok(signals);
        }

        // Enhance each signal
        for signal in &mut signals {
            // Get sentiment score
            let sentiment_score = self.analyze_sentiment(&signal.symbol).await?;

            // Get ML prediction
            let ml_score = self.predict_ml_signal(&signal.features).await?;

            // Calculate enhanced confidence
            let base_conf = signal.confidence.unwrap_or(0.5);
            let enhanced_confidence = self.calculate_enhanced_confidence(
                base_conf,
                sentiment_score,
                ml_score,
            );

            // Check for contradictory signals
            let momentum_direction = match signal.direction {
                Direction::Long => 1.0,
                Direction::Short => -1.0,
                Direction::Close => 0.0,
            };

            // If sentiment strongly contradicts momentum, reduce confidence
            if momentum_direction * sentiment_score < -0.5 {
                signal.confidence = Some(enhanced_confidence * 0.5);
                let warning = " [WARNING: Sentiment contradicts momentum]";
                signal.reasoning = Some(signal.reasoning.as_deref().unwrap_or("").to_string() + warning);
            } else {
                signal.confidence = Some(enhanced_confidence);
            }

            // Update reasoning
            let update = format!(" | Sentiment: {:.2}, ML: {:.2}", sentiment_score, ml_score);
            signal.reasoning = Some(format!("{}{}", signal.reasoning.as_deref().unwrap_or(""), update));
        }

        Ok(signals)
    }

    fn validate_config(&self) -> Result<()> {
        self.base_momentum.validate_config()?;

        if self.sentiment_weight < 0.0 || self.sentiment_weight > 1.0 {
            return Err(StrategyError::InvalidParameter(
                "Sentiment weight must be between 0 and 1".to_string(),
            ));
        }

        if self.ml_weight < 0.0 || self.ml_weight > 1.0 {
            return Err(StrategyError::InvalidParameter(
                "ML weight must be between 0 and 1".to_string(),
            ));
        }

        if self.sentiment_weight + self.ml_weight > 1.0 {
            return Err(StrategyError::InvalidParameter(
                "Sum of sentiment and ML weights must not exceed 1".to_string(),
            ));
        }

        Ok(())
    }

    fn risk_parameters(&self) -> RiskParameters {
        self.base_momentum.risk_parameters()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_strategy_validation() {
        let strategy = EnhancedMomentumStrategy::new(
            vec!["TEST".to_string()],
            20,
            2.0,
            0.5,
            0.3,
            0.2,
        );
        assert!(strategy.validate_config().is_ok());

        let invalid = EnhancedMomentumStrategy::new(
            vec!["TEST".to_string()],
            20,
            2.0,
            0.5,
            0.6, // Too high
            0.6, // Together exceed 1.0
        );
        assert!(invalid.validate_config().is_err());
    }

    #[tokio::test]
    async fn test_sentiment_analysis() {
        let strategy = EnhancedMomentumStrategy::new(
            vec!["TEST".to_string()],
            20,
            2.0,
            0.5,
            0.3,
            0.2,
        );

        let sentiment = strategy.analyze_sentiment("TEST").await.unwrap();
        assert!(sentiment >= -1.0 && sentiment <= 1.0);
    }

    #[test]
    fn test_confidence_calculation() {
        let strategy = EnhancedMomentumStrategy::new(
            vec!["TEST".to_string()],
            20,
            2.0,
            0.5,
            0.3,
            0.2,
        );

        let confidence = strategy.calculate_enhanced_confidence(0.8, 0.5, 0.7);
        assert!(confidence >= 0.0 && confidence <= 1.0);
    }
}
