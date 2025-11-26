//! Ensemble Strategy
//!
//! Combines signals from multiple strategies using various fusion methods.
//! Supports weighted average, voting, and stacking approaches.
//!
//! ## Fusion Methods
//!
//! 1. **Weighted Average**: Combines confidences with strategy weights
//! 2. **Voting**: Majority vote on direction, average confidence
//! 3. **Stacking**: Uses meta-model to combine signals (placeholder)
//!
//! ## Algorithm
//!
//! 1. Collect signals from all strategies
//! 2. Group signals by symbol
//! 3. Apply fusion method to combine signals
//! 4. Generate final ensemble signal

use crate::{
    async_trait, Direction, MarketData, Portfolio, Result, Signal, Strategy,
    StrategyError, StrategyMetadata, RiskParameters,
};
use rust_decimal::Decimal;
use std::collections::HashMap;

/// Signal fusion method
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FusionMethod {
    /// Weighted average of confidences
    WeightedAverage,
    /// Majority voting on direction
    Voting,
    /// Meta-model stacking (placeholder)
    Stacking,
}

/// Ensemble strategy combining multiple strategies
pub struct EnsembleStrategy {
    /// Strategy ID
    id: String,
    /// Child strategies
    strategies: Vec<Box<dyn Strategy>>,
    /// Strategy weights (must sum to 1.0)
    weights: Vec<f64>,
    /// Fusion method
    fusion_method: FusionMethod,
    /// Minimum confidence threshold for ensemble signal
    min_confidence: f64,
}

impl EnsembleStrategy {
    /// Create a new ensemble strategy
    pub fn new(
        strategies: Vec<Box<dyn Strategy>>,
        weights: Vec<f64>,
        fusion_method: FusionMethod,
        min_confidence: f64,
    ) -> Result<Self> {
        if strategies.len() != weights.len() {
            return Err(StrategyError::ConfigError(
                "Number of strategies must match number of weights".to_string(),
            ));
        }

        let weight_sum: f64 = weights.iter().sum();
        if (weight_sum - 1.0).abs() > 0.001 {
            return Err(StrategyError::ConfigError(format!(
                "Weights must sum to 1.0, got {}",
                weight_sum
            )));
        }

        Ok(Self {
            id: "ensemble".to_string(),
            strategies,
            weights,
            fusion_method,
            min_confidence,
        })
    }

    /// Fuse signals using weighted average
    fn weighted_average_fusion(&self, grouped_signals: HashMap<String, Vec<(Signal, f64)>>) -> Vec<Signal> {
        let mut result = Vec::new();

        for (symbol, weighted_signals) in grouped_signals {
            if weighted_signals.is_empty() {
                continue;
            }

            let total_weight: f64 = weighted_signals.iter().map(|(_, w)| w).sum();

            // Calculate weighted average confidence
            let avg_confidence: f64 = weighted_signals
                .iter()
                .map(|(s, w)| s.confidence.unwrap_or(0.5) * w)
                .sum::<f64>()
                / total_weight;

            if avg_confidence < self.min_confidence {
                continue;
            }

            // Weighted vote on direction
            let mut direction_weights: HashMap<Direction, f64> = HashMap::new();
            for (signal, weight) in &weighted_signals {
                *direction_weights.entry(signal.direction).or_default() += weight * signal.confidence.unwrap_or(0.5);
            }

            let (direction, _) = direction_weights
                .into_iter()
                .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .unwrap();

            // Average prices
            let avg_entry_price = weighted_signals
                .iter()
                .filter_map(|(s, w)| s.entry_price.map(|p| (p, w)))
                .fold(Decimal::ZERO, |acc, (p, w)| acc + p * Decimal::from_f64_retain(*w).unwrap())
                / Decimal::from_f64_retain(total_weight).unwrap();

            result.push(
                Signal::new(self.id.clone(), symbol, direction)
                    .with_confidence(avg_confidence)
                    .with_entry_price(avg_entry_price)
                    .with_reasoning(format!(
                        "Ensemble of {} strategies (weighted avg)",
                        weighted_signals.len()
                    )),
            );
        }

        result
    }

    /// Fuse signals using voting
    fn voting_fusion(&self, grouped_signals: HashMap<String, Vec<(Signal, f64)>>) -> Vec<Signal> {
        let mut result = Vec::new();

        for (symbol, weighted_signals) in grouped_signals {
            if weighted_signals.is_empty() {
                continue;
            }

            // Count votes for each direction
            let mut direction_votes: HashMap<Direction, usize> = HashMap::new();
            for (signal, _) in &weighted_signals {
                *direction_votes.entry(signal.direction).or_default() += 1;
            }

            let (direction, votes) = direction_votes
                .into_iter()
                .max_by_key(|(_, v)| *v)
                .unwrap();

            // Average confidence of signals that voted for winning direction
            let matching_signals: Vec<_> = weighted_signals
                .iter()
                .filter(|(s, _)| s.direction == direction)
                .collect();

            let avg_confidence: f64 = matching_signals
                .iter()
                .map(|(s, _)| s.confidence.unwrap_or(0.5))
                .sum::<f64>()
                / matching_signals.len() as f64;

            if avg_confidence < self.min_confidence {
                continue;
            }

            result.push(
                Signal::new(self.id.clone(), symbol, direction)
                    .with_confidence(avg_confidence)
                    .with_reasoning(format!(
                        "Ensemble of {} strategies ({} votes for {:?})",
                        weighted_signals.len(),
                        votes,
                        direction
                    )),
            );
        }

        result
    }
}

#[async_trait]
impl Strategy for EnsembleStrategy {
    fn id(&self) -> &str {
        &self.id
    }

    fn metadata(&self) -> StrategyMetadata {
        StrategyMetadata {
            name: "Ensemble".to_string(),
            description: "Combines multiple strategies using signal fusion".to_string(),
            version: "1.0.0".to_string(),
            author: "Neural Trader Team".to_string(),
            tags: vec![
                "ensemble".to_string(),
                "fusion".to_string(),
                "meta".to_string(),
            ],
            min_capital: Decimal::from(15000),
            max_drawdown_threshold: 0.12,
        }
    }

    async fn process(
        &self,
        market_data: &MarketData,
        portfolio: &Portfolio,
    ) -> Result<Vec<Signal>> {
        // Collect signals from all strategies
        let mut all_signals: Vec<Vec<Signal>> = Vec::new();

        for strategy in &self.strategies {
            let signals = strategy.process(market_data, portfolio).await?;
            all_signals.push(signals);
        }

        // Group signals by symbol with weights
        let mut grouped: HashMap<String, Vec<(Signal, f64)>> = HashMap::new();

        for (signals, &weight) in all_signals.iter().zip(self.weights.iter()) {
            for signal in signals {
                grouped
                    .entry(signal.symbol.clone())
                    .or_default()
                    .push((signal.clone(), weight));
            }
        }

        // Apply fusion method
        let fused_signals = match self.fusion_method {
            FusionMethod::WeightedAverage => self.weighted_average_fusion(grouped),
            FusionMethod::Voting => self.voting_fusion(grouped),
            FusionMethod::Stacking => {
                // Placeholder: Stacking would use a meta-model
                self.weighted_average_fusion(grouped)
            }
        };

        Ok(fused_signals)
    }

    fn validate_config(&self) -> Result<()> {
        if self.strategies.is_empty() {
            return Err(StrategyError::ConfigError(
                "At least one strategy must be specified".to_string(),
            ));
        }

        if self.min_confidence < 0.0 || self.min_confidence > 1.0 {
            return Err(StrategyError::InvalidParameter(
                "Minimum confidence must be between 0 and 1".to_string(),
            ));
        }

        // Validate all child strategies
        for strategy in &self.strategies {
            strategy.validate_config()?;
        }

        Ok(())
    }

    fn risk_parameters(&self) -> RiskParameters {
        // Use average of child strategies' risk parameters
        let mut total = RiskParameters::default();
        let count = self.strategies.len() as f64;

        for strategy in &self.strategies {
            let params = strategy.risk_parameters();
            total.max_position_size += params.max_position_size;
            total.max_leverage += params.max_leverage;
            total.stop_loss_percentage += params.stop_loss_percentage;
            total.take_profit_percentage += params.take_profit_percentage;
            total.max_daily_loss += params.max_daily_loss;
        }

        RiskParameters {
            max_position_size: total.max_position_size / Decimal::from_f64_retain(count).unwrap(),
            max_leverage: total.max_leverage / count,
            stop_loss_percentage: total.stop_loss_percentage / count,
            take_profit_percentage: total.take_profit_percentage / count,
            max_daily_loss: total.max_daily_loss / Decimal::from_f64_retain(count).unwrap(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MomentumStrategy;

    #[test]
    fn test_ensemble_creation() {
        let strategy1 = Box::new(MomentumStrategy::new(vec!["TEST".to_string()], 20, 2.0, 0.5))
            as Box<dyn Strategy>;
        let strategy2 = Box::new(MomentumStrategy::new(vec!["TEST".to_string()], 10, 1.5, 0.3))
            as Box<dyn Strategy>;

        let ensemble = EnsembleStrategy::new(
            vec![strategy1, strategy2],
            vec![0.6, 0.4],
            FusionMethod::WeightedAverage,
            0.5,
        );

        assert!(ensemble.is_ok());
    }

    #[test]
    fn test_invalid_weights() {
        let strategy1 = Box::new(MomentumStrategy::new(vec!["TEST".to_string()], 20, 2.0, 0.5))
            as Box<dyn Strategy>;

        let ensemble = EnsembleStrategy::new(
            vec![strategy1],
            vec![0.5], // Doesn't sum to 1.0
            FusionMethod::WeightedAverage,
            0.5,
        );

        assert!(ensemble.is_err());
    }
}
