//! Performance-Based Weight Calculation for Consensus Voting
//!
//! Dynamic weight assignment system that adapts voting influence based on
//! organism performance, reliability, and market conditions.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{Duration, SystemTime};
use tracing::{debug, instrument};
use uuid::Uuid;

use super::{PerformanceScore, VotingWeight};

/// Maximum weight multiplier to prevent extreme influence
pub const MAX_WEIGHT_MULTIPLIER: f64 = 3.0;

/// Minimum weight to ensure all organisms have some influence
pub const MIN_WEIGHT_MULTIPLIER: f64 = 0.1;

/// Performance measurement window (seconds)
pub const PERFORMANCE_WINDOW_SECS: u64 = 3600; // 1 hour

/// Weight decay rate per update cycle
pub const WEIGHT_DECAY_RATE: f64 = 0.95;

/// Factors contributing to organism weight calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightFactors {
    /// Recent performance score (0.0 to 1.0)
    pub performance: f64,

    /// Reliability/consistency score (0.0 to 1.0)
    pub reliability: f64,

    /// Market adaptation score (0.0 to 1.0)
    pub adaptation: f64,

    /// Voting accuracy (how often organism votes with majority)
    pub accuracy: f64,

    /// Response time factor (faster = higher weight)
    pub responsiveness: f64,

    /// Resource efficiency (lower resource usage = higher weight)
    pub efficiency: f64,

    /// Contribution to system stability
    pub stability_contribution: f64,

    /// Emergence participation (involvement in collective intelligence)
    pub emergence_factor: f64,
}

impl Default for WeightFactors {
    fn default() -> Self {
        Self {
            performance: 0.5,
            reliability: 0.5,
            adaptation: 0.5,
            accuracy: 0.5,
            responsiveness: 0.5,
            efficiency: 0.5,
            stability_contribution: 0.5,
            emergence_factor: 0.5,
        }
    }
}

/// Weight calculation configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightConfig {
    /// Weights for different factors (must sum to 1.0)
    pub factor_weights: FactorWeights,

    /// Time decay rate for historical performance
    pub decay_rate: f64,

    /// Minimum samples required for stable weight calculation
    pub min_samples: usize,

    /// Maximum age of performance data to consider (seconds)
    pub max_age_secs: u64,

    /// Learning rate for adaptive weight adjustment
    pub learning_rate: f64,
}

/// Relative importance of different weight factors
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FactorWeights {
    pub performance: f64,
    pub reliability: f64,
    pub adaptation: f64,
    pub accuracy: f64,
    pub responsiveness: f64,
    pub efficiency: f64,
    pub stability_contribution: f64,
    pub emergence_factor: f64,
}

impl Default for FactorWeights {
    fn default() -> Self {
        Self {
            performance: 0.25,
            reliability: 0.20,
            adaptation: 0.15,
            accuracy: 0.15,
            responsiveness: 0.10,
            efficiency: 0.05,
            stability_contribution: 0.05,
            emergence_factor: 0.05,
        }
    }
}

impl Default for WeightConfig {
    fn default() -> Self {
        Self {
            factor_weights: FactorWeights::default(),
            decay_rate: WEIGHT_DECAY_RATE,
            min_samples: 5,
            max_age_secs: PERFORMANCE_WINDOW_SECS,
            learning_rate: 0.1,
        }
    }
}

/// Historical performance record
#[derive(Debug, Clone, Serialize, Deserialize)]
struct PerformanceRecord {
    timestamp: SystemTime,
    factors: WeightFactors,
    calculated_weight: f64,
    market_conditions: Option<MarketSnapshot>,
}

/// Market conditions snapshot for context-aware weighting
#[derive(Debug, Clone, Serialize, Deserialize)]
struct MarketSnapshot {
    volatility: f64,
    volume: f64,
    trend_strength: f64,
    noise_level: f64,
}

/// Weight calculation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WeightStatistics {
    pub total_organisms: usize,
    pub active_organisms: usize,
    pub average_weight: f64,
    pub weight_distribution: HashMap<String, usize>, // Weight ranges
    pub top_performers: Vec<(Uuid, f64)>,
    pub last_update: SystemTime,
}

/// Performance-based weight calculator
#[derive(Debug)]
pub struct PerformanceWeights {
    config: WeightConfig,
    organism_history: HashMap<Uuid, Vec<PerformanceRecord>>,
    current_weights: HashMap<Uuid, f64>,
    weight_statistics: WeightStatistics,
    last_cleanup: SystemTime,
}

/// Weight calculator trait
pub trait WeightCalculator {
    /// Calculate weight for an organism based on performance factors
    fn calculate_weight(
        &self,
        organism_id: Uuid,
        factors: &WeightFactors,
    ) -> Result<VotingWeight, WeightError>;

    /// Update weights for multiple organisms
    fn update_weights(
        &mut self,
        performances: HashMap<Uuid, PerformanceScore>,
    ) -> Result<(), WeightError>;

    /// Get current weight for an organism
    fn get_weight(&self, organism_id: &Uuid) -> Option<VotingWeight>;

    /// Get weight statistics
    fn get_statistics(&self) -> WeightStatistics;
}

impl PerformanceWeights {
    /// Create new performance weights calculator
    pub fn new() -> Self {
        Self::with_config(WeightConfig::default())
    }

    /// Create with custom configuration
    pub fn with_config(config: WeightConfig) -> Self {
        Self {
            config,
            organism_history: HashMap::new(),
            current_weights: HashMap::new(),
            weight_statistics: WeightStatistics {
                total_organisms: 0,
                active_organisms: 0,
                average_weight: 1.0,
                weight_distribution: HashMap::new(),
                top_performers: Vec::new(),
                last_update: SystemTime::now(),
            },
            last_cleanup: SystemTime::now(),
        }
    }

    /// Calculate comprehensive weight for an organism
    #[instrument(skip(self, factors))]
    fn calculate_comprehensive_weight(
        &self,
        organism_id: Uuid,
        factors: &WeightFactors,
    ) -> Result<f64, WeightError> {
        // Get historical context if available
        let historical_factor = self.calculate_historical_factor(organism_id)?;

        // Calculate base weight from current factors
        let base_weight = self.calculate_base_weight(factors)?;

        // Apply historical weighting
        let weighted_score = base_weight * 0.7 + historical_factor * 0.3;

        // Apply market context adjustment
        let market_adjusted = self.apply_market_adjustment(weighted_score, organism_id)?;

        // Normalize and apply bounds
        let final_weight = market_adjusted.clamp(MIN_WEIGHT_MULTIPLIER, MAX_WEIGHT_MULTIPLIER);

        debug!(
            "Calculated weight {} for organism {}: base={:.3}, historical={:.3}, final={:.3}",
            final_weight, organism_id, base_weight, historical_factor, final_weight
        );

        Ok(final_weight)
    }

    /// Calculate base weight from current performance factors
    fn calculate_base_weight(&self, factors: &WeightFactors) -> Result<f64, WeightError> {
        let config = &self.config.factor_weights;

        let weight = factors.performance * config.performance
            + factors.reliability * config.reliability
            + factors.adaptation * config.adaptation
            + factors.accuracy * config.accuracy
            + factors.responsiveness * config.responsiveness
            + factors.efficiency * config.efficiency
            + factors.stability_contribution * config.stability_contribution
            + factors.emergence_factor * config.emergence_factor;

        // Validate weight calculation
        if weight < 0.0 || weight > 1.0 {
            return Err(WeightError::InvalidWeightCalculation {
                calculated_weight: weight,
                factors: format!("{:?}", factors),
            });
        }

        Ok(weight)
    }

    /// Calculate historical performance factor
    fn calculate_historical_factor(&self, organism_id: Uuid) -> Result<f64, WeightError> {
        let history = match self.organism_history.get(&organism_id) {
            Some(h) if h.len() >= self.config.min_samples => h,
            _ => return Ok(0.5), // Neutral factor for new/insufficient data
        };

        let now = SystemTime::now();
        let cutoff_time = now - Duration::from_secs(self.config.max_age_secs);

        // Filter recent records and calculate weighted average
        let mut total_weight = 0.0;
        let mut weighted_sum = 0.0;

        for record in history.iter().rev() {
            if record.timestamp < cutoff_time {
                break; // Records are sorted by time
            }

            // Calculate time-based decay
            let age = now
                .duration_since(record.timestamp)
                .map_err(|_| WeightError::InvalidTimestamp)?
                .as_secs() as f64;

            let time_weight = (self.config.decay_rate).powf(age / 3600.0); // Hourly decay

            weighted_sum += record.calculated_weight * time_weight;
            total_weight += time_weight;
        }

        if total_weight > 0.0 {
            Ok(weighted_sum / total_weight)
        } else {
            Ok(0.5) // Default neutral factor
        }
    }

    /// Apply market condition adjustments
    fn apply_market_adjustment(
        &self,
        base_weight: f64,
        organism_id: Uuid,
    ) -> Result<f64, WeightError> {
        // Get latest market conditions from history
        let latest_market = self
            .organism_history
            .get(&organism_id)
            .and_then(|history| history.last())
            .and_then(|record| record.market_conditions.as_ref());

        let adjustment_factor = match latest_market {
            Some(market) => {
                // Adjust based on market conditions
                let volatility_factor = if market.volatility > 0.7 {
                    1.1 // High volatility favors proven performers
                } else {
                    1.0
                };

                let volume_factor = if market.volume > 0.8 {
                    1.05 // High volume rewards responsive organisms
                } else {
                    1.0
                };

                volatility_factor * volume_factor
            }
            None => 1.0, // No adjustment without market data
        };

        Ok(base_weight * adjustment_factor)
    }

    /// Update performance record for an organism
    fn update_performance_record(
        &mut self,
        organism_id: Uuid,
        factors: WeightFactors,
        weight: f64,
    ) {
        let record = PerformanceRecord {
            timestamp: SystemTime::now(),
            factors,
            calculated_weight: weight,
            market_conditions: self.get_current_market_snapshot(),
        };

        let history = self
            .organism_history
            .entry(organism_id)
            .or_insert_with(Vec::new);
        history.push(record);

        // Keep only recent records (max 1000 per organism)
        if history.len() > 1000 {
            history.remove(0);
        }
    }

    /// Get current market conditions snapshot
    fn get_current_market_snapshot(&self) -> Option<MarketSnapshot> {
        // In a real implementation, this would fetch actual market data
        // For now, return a placeholder
        Some(MarketSnapshot {
            volatility: 0.5,
            volume: 0.6,
            trend_strength: 0.4,
            noise_level: 0.3,
        })
    }

    /// Clean up old performance records
    fn cleanup_old_records(&mut self) {
        let now = SystemTime::now();
        let cutoff = now - Duration::from_secs(self.config.max_age_secs * 2); // Keep 2x window for history

        for history in self.organism_history.values_mut() {
            history.retain(|record| record.timestamp >= cutoff);
        }

        // Remove organisms with no recent data
        self.organism_history
            .retain(|_, history| !history.is_empty());
        self.last_cleanup = now;
    }

    /// Update weight statistics
    fn update_statistics(&mut self) {
        let total_organisms = self.current_weights.len();
        let active_organisms = self
            .current_weights
            .values()
            .filter(|&&weight| weight > MIN_WEIGHT_MULTIPLIER)
            .count();

        let average_weight = if total_organisms > 0 {
            self.current_weights.values().sum::<f64>() / total_organisms as f64
        } else {
            1.0
        };

        // Calculate weight distribution
        let mut weight_distribution = HashMap::new();
        for &weight in self.current_weights.values() {
            let bucket = match weight {
                w if w < 0.5 => "0.0-0.5",
                w if w < 1.0 => "0.5-1.0",
                w if w < 1.5 => "1.0-1.5",
                w if w < 2.0 => "1.5-2.0",
                _ => "2.0+",
            };
            *weight_distribution.entry(bucket.to_string()).or_insert(0) += 1;
        }

        // Find top performers
        let mut top_performers: Vec<(Uuid, f64)> = self
            .current_weights
            .iter()
            .map(|(&id, &weight)| (id, weight))
            .collect();
        top_performers.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        top_performers.truncate(10); // Keep top 10

        self.weight_statistics = WeightStatistics {
            total_organisms,
            active_organisms,
            average_weight,
            weight_distribution,
            top_performers,
            last_update: SystemTime::now(),
        };
    }

    /// Adaptive weight adjustment based on system feedback
    pub fn adaptive_adjustment(&mut self, feedback: HashMap<Uuid, f64>) -> Result<(), WeightError> {
        for (organism_id, feedback_score) in feedback {
            if let Some(current_weight) = self.current_weights.get_mut(&organism_id) {
                // Adjust weight based on feedback
                let adjustment = (feedback_score - 0.5) * self.config.learning_rate;
                let new_weight = (*current_weight + adjustment)
                    .clamp(MIN_WEIGHT_MULTIPLIER, MAX_WEIGHT_MULTIPLIER);

                debug!(
                    "Adaptive adjustment for {}: {:.3} -> {:.3} (feedback: {:.3})",
                    organism_id, current_weight, new_weight, feedback_score
                );

                *current_weight = new_weight;
            }
        }

        self.update_statistics();
        Ok(())
    }

    /// Calculate relative weight distribution
    pub fn get_relative_weights(&self) -> HashMap<Uuid, f64> {
        let total_weight: f64 = self.current_weights.values().sum();

        if total_weight <= 0.0 {
            return HashMap::new();
        }

        self.current_weights
            .iter()
            .map(|(&id, &weight)| (id, weight / total_weight))
            .collect()
    }

    /// Get weight percentile for an organism
    pub fn get_weight_percentile(&self, organism_id: &Uuid) -> Option<f64> {
        let weight = self.current_weights.get(organism_id)?;

        let lower_count = self
            .current_weights
            .values()
            .filter(|&&w| w < *weight)
            .count();

        let total = self.current_weights.len();
        if total > 0 {
            Some(lower_count as f64 / total as f64)
        } else {
            None
        }
    }

    /// Export weight history for analysis
    pub fn export_history(&self, organism_id: &Uuid) -> Option<Vec<PerformanceRecord>> {
        self.organism_history.get(organism_id).cloned()
    }

    /// Import pre-calculated weights (for testing or migration)
    pub fn import_weights(&mut self, weights: HashMap<Uuid, f64>) -> Result<(), WeightError> {
        for (organism_id, weight) in weights {
            if weight < 0.0 || weight > MAX_WEIGHT_MULTIPLIER {
                return Err(WeightError::InvalidWeightValue {
                    organism_id,
                    weight,
                    valid_range: (0.0, MAX_WEIGHT_MULTIPLIER),
                });
            }

            self.current_weights.insert(organism_id, weight);
        }

        self.update_statistics();
        Ok(())
    }

    /// Update weights for multiple organisms (direct implementation)
    pub fn update_weights(
        &mut self,
        performances: HashMap<Uuid, PerformanceScore>,
    ) -> Result<(), WeightError> {
        // Delegate to trait implementation
        <Self as WeightCalculator>::update_weights(self, performances)
    }
}

impl WeightCalculator for PerformanceWeights {
    fn calculate_weight(
        &self,
        organism_id: Uuid,
        factors: &WeightFactors,
    ) -> Result<VotingWeight, WeightError> {
        self.calculate_comprehensive_weight(organism_id, factors)
    }

    fn update_weights(
        &mut self,
        performances: HashMap<Uuid, PerformanceScore>,
    ) -> Result<(), WeightError> {
        // Clean up old records periodically
        let now = SystemTime::now();
        if now
            .duration_since(self.last_cleanup)
            .unwrap_or_default()
            .as_secs()
            > 3600
        {
            self.cleanup_old_records();
        }

        for (organism_id, performance_score) in performances {
            // Convert performance score to weight factors
            let factors = WeightFactors {
                performance: performance_score,
                reliability: self.estimate_reliability(organism_id),
                adaptation: self.estimate_adaptation(organism_id),
                accuracy: self.estimate_accuracy(organism_id),
                responsiveness: self.estimate_responsiveness(organism_id),
                efficiency: self.estimate_efficiency(organism_id),
                stability_contribution: self.estimate_stability(organism_id),
                emergence_factor: self.estimate_emergence(organism_id),
            };

            let weight = self.calculate_comprehensive_weight(organism_id, &factors)?;

            self.current_weights.insert(organism_id, weight);
            self.update_performance_record(organism_id, factors, weight);
        }

        self.update_statistics();
        Ok(())
    }

    fn get_weight(&self, organism_id: &Uuid) -> Option<VotingWeight> {
        self.current_weights.get(organism_id).copied()
    }

    fn get_statistics(&self) -> WeightStatistics {
        self.weight_statistics.clone()
    }
}

impl PerformanceWeights {
    /// Estimate organism reliability from historical data
    fn estimate_reliability(&self, organism_id: Uuid) -> f64 {
        self.organism_history
            .get(&organism_id)
            .map(|history| {
                if history.len() < 3 {
                    return 0.5; // Default for insufficient data
                }

                // Calculate coefficient of variation (lower = more reliable)
                let weights: Vec<f64> = history.iter().map(|r| r.calculated_weight).collect();
                let mean = weights.iter().sum::<f64>() / weights.len() as f64;
                let variance =
                    weights.iter().map(|w| (w - mean).powi(2)).sum::<f64>() / weights.len() as f64;

                let cv = if mean > 0.0 {
                    variance.sqrt() / mean
                } else {
                    1.0
                };
                (1.0 - cv).clamp(0.0, 1.0)
            })
            .unwrap_or(0.5)
    }

    /// Estimate adaptation capability
    fn estimate_adaptation(&self, organism_id: Uuid) -> f64 {
        self.organism_history
            .get(&organism_id)
            .map(|history| {
                if history.len() < 5 {
                    return 0.5;
                }

                // Look at performance trend over different market conditions
                let mut adaptation_scores = Vec::new();

                for window in history.windows(3) {
                    if let (Some(first_market), Some(last_market)) = (
                        window[0].market_conditions.as_ref(),
                        window[2].market_conditions.as_ref(),
                    ) {
                        let market_change = (first_market.volatility - last_market.volatility)
                            .abs()
                            + (first_market.volume - last_market.volume).abs();

                        let performance_change =
                            window[2].calculated_weight - window[0].calculated_weight;

                        // Good adaptation = positive performance change with market change
                        if market_change > 0.1 {
                            let adaptation_score = if performance_change > 0.0 { 0.8 } else { 0.2 };
                            adaptation_scores.push(adaptation_score);
                        }
                    }
                }

                if adaptation_scores.is_empty() {
                    0.5
                } else {
                    adaptation_scores.iter().sum::<f64>() / adaptation_scores.len() as f64
                }
            })
            .unwrap_or(0.5)
    }

    /// Estimate voting accuracy (placeholder)
    fn estimate_accuracy(&self, _organism_id: Uuid) -> f64 {
        // In a real implementation, this would track how often the organism
        // votes with the eventual consensus
        0.7 // Placeholder
    }

    /// Estimate responsiveness (placeholder)
    fn estimate_responsiveness(&self, _organism_id: Uuid) -> f64 {
        // Would track response times to voting requests
        0.6 // Placeholder
    }

    /// Estimate efficiency (placeholder)
    fn estimate_efficiency(&self, _organism_id: Uuid) -> f64 {
        // Would track resource usage vs. performance
        0.8 // Placeholder
    }

    /// Estimate stability contribution (placeholder)
    fn estimate_stability(&self, _organism_id: Uuid) -> f64 {
        // Would track how the organism contributes to system stability
        0.5 // Placeholder
    }

    /// Estimate emergence participation (placeholder)
    fn estimate_emergence(&self, _organism_id: Uuid) -> f64 {
        // Would track participation in emergent behaviors
        0.4 // Placeholder
    }
}

/// Errors that can occur during weight calculation
#[derive(Debug, thiserror::Error)]
pub enum WeightError {
    #[error("Invalid weight calculation: weight={calculated_weight}, factors={factors}")]
    InvalidWeightCalculation {
        calculated_weight: f64,
        factors: String,
    },

    #[error(
        "Invalid weight value {weight} for organism {organism_id}, valid range: {valid_range:?}"
    )]
    InvalidWeightValue {
        organism_id: Uuid,
        weight: f64,
        valid_range: (f64, f64),
    },

    #[error("Invalid timestamp in performance record")]
    InvalidTimestamp,

    #[error("Insufficient performance data for organism {organism_id}")]
    InsufficientData { organism_id: Uuid },

    #[error("Weight calculation overflow/underflow")]
    CalculationOverflow,

    #[error("Configuration error: {0}")]
    ConfigurationError(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_performance_weights_creation() {
        let pw = PerformanceWeights::new();
        assert_eq!(pw.current_weights.len(), 0);
        assert_eq!(pw.weight_statistics.total_organisms, 0);
    }

    #[test]
    fn test_base_weight_calculation() {
        let pw = PerformanceWeights::new();

        let factors = WeightFactors {
            performance: 0.8,
            reliability: 0.7,
            adaptation: 0.6,
            accuracy: 0.9,
            responsiveness: 0.5,
            efficiency: 0.8,
            stability_contribution: 0.7,
            emergence_factor: 0.4,
        };

        let weight = pw.calculate_base_weight(&factors).unwrap();
        assert!(weight >= 0.0 && weight <= 1.0);

        // Should be weighted average based on factor weights
        let expected = 0.8 * 0.25
            + 0.7 * 0.20
            + 0.6 * 0.15
            + 0.9 * 0.15
            + 0.5 * 0.10
            + 0.8 * 0.05
            + 0.7 * 0.05
            + 0.4 * 0.05;
        assert!((weight - expected).abs() < 0.001);
    }

    #[test]
    fn test_weight_bounds() {
        let pw = PerformanceWeights::new();
        let organism_id = Uuid::new_v4();

        // Test extreme high factors
        let high_factors = WeightFactors {
            performance: 1.0,
            reliability: 1.0,
            adaptation: 1.0,
            accuracy: 1.0,
            responsiveness: 1.0,
            efficiency: 1.0,
            stability_contribution: 1.0,
            emergence_factor: 1.0,
        };

        let high_weight = pw
            .calculate_comprehensive_weight(organism_id, &high_factors)
            .unwrap();
        assert!(high_weight <= MAX_WEIGHT_MULTIPLIER);

        // Test extreme low factors
        let low_factors = WeightFactors {
            performance: 0.0,
            reliability: 0.0,
            adaptation: 0.0,
            accuracy: 0.0,
            responsiveness: 0.0,
            efficiency: 0.0,
            stability_contribution: 0.0,
            emergence_factor: 0.0,
        };

        let low_weight = pw
            .calculate_comprehensive_weight(organism_id, &low_factors)
            .unwrap();
        assert!(low_weight >= MIN_WEIGHT_MULTIPLIER);
    }

    #[test]
    fn test_invalid_weight_calculation() {
        let pw = PerformanceWeights::new();

        // Create factors that would result in invalid weight (shouldn't happen with normal factors)
        let invalid_factors = WeightFactors {
            performance: -1.0, // Invalid negative value
            ..WeightFactors::default()
        };

        // The base weight calculation should handle this gracefully or return an error
        // In this case, it might still work because we clamp values
        let result = pw.calculate_base_weight(&invalid_factors);
        // The result depends on implementation - it might error or handle gracefully
        match result {
            Ok(weight) => assert!(weight >= 0.0 && weight <= 1.0),
            Err(_) => {} // Error is acceptable for invalid inputs
        }
    }

    #[test]
    fn test_weight_update() {
        let mut pw = PerformanceWeights::new();

        let mut performances = HashMap::new();
        let organism1 = Uuid::new_v4();
        let organism2 = Uuid::new_v4();

        performances.insert(organism1, 0.8);
        performances.insert(organism2, 0.6);

        pw.update_weights(performances).unwrap();

        assert_eq!(pw.current_weights.len(), 2);
        assert!(pw.get_weight(&organism1).is_some());
        assert!(pw.get_weight(&organism2).is_some());

        // Organism1 should have higher weight than organism2
        let weight1 = pw.get_weight(&organism1).unwrap();
        let weight2 = pw.get_weight(&organism2).unwrap();
        assert!(weight1 >= weight2); // Higher performance should get higher or equal weight
    }

    #[test]
    fn test_relative_weights() {
        let mut pw = PerformanceWeights::new();

        let mut performances = HashMap::new();
        let organism1 = Uuid::new_v4();
        let organism2 = Uuid::new_v4();

        performances.insert(organism1, 0.9);
        performances.insert(organism2, 0.3);

        pw.update_weights(performances).unwrap();

        let relative_weights = pw.get_relative_weights();

        // Relative weights should sum to 1.0
        let total: f64 = relative_weights.values().sum();
        assert!((total - 1.0).abs() < 0.001);

        // Higher performing organism should have higher relative weight
        let rel_weight1 = relative_weights.get(&organism1).unwrap();
        let rel_weight2 = relative_weights.get(&organism2).unwrap();
        assert!(rel_weight1 > rel_weight2);
    }

    #[test]
    fn test_adaptive_adjustment() {
        let mut pw = PerformanceWeights::new();
        let organism_id = Uuid::new_v4();

        // Set initial weight
        let mut performances = HashMap::new();
        performances.insert(organism_id, 0.5);
        pw.update_weights(performances).unwrap();

        let initial_weight = pw.get_weight(&organism_id).unwrap();

        // Apply positive feedback
        let mut feedback = HashMap::new();
        feedback.insert(organism_id, 0.8); // Good feedback
        pw.adaptive_adjustment(feedback).unwrap();

        let adjusted_weight = pw.get_weight(&organism_id).unwrap();
        assert!(adjusted_weight >= initial_weight); // Should increase with positive feedback
    }

    #[test]
    fn test_weight_statistics() {
        let mut pw = PerformanceWeights::new();

        // Add several organisms with different performances
        let mut performances = HashMap::new();
        for i in 0..5 {
            let organism_id = Uuid::new_v4();
            performances.insert(organism_id, i as f64 * 0.2); // 0.0, 0.2, 0.4, 0.6, 0.8
        }

        pw.update_weights(performances).unwrap();

        let stats = pw.get_statistics();
        assert_eq!(stats.total_organisms, 5);
        assert_eq!(stats.active_organisms, 5); // All should be above minimum
        assert!(stats.average_weight > 0.0);
        assert_eq!(stats.top_performers.len(), 5); // Should have all 5

        // Top performer should have the highest weight
        let top_weight = stats.top_performers[0].1;
        let bottom_weight = stats.top_performers[4].1;
        assert!(top_weight >= bottom_weight);
    }

    #[test]
    fn test_weight_percentile() {
        let mut pw = PerformanceWeights::new();

        let mut performances = HashMap::new();
        let organisms: Vec<Uuid> = (0..10).map(|_| Uuid::new_v4()).collect();

        for (i, organism_id) in organisms.iter().enumerate() {
            performances.insert(*organism_id, i as f64 * 0.1); // 0.0 to 0.9
        }

        pw.update_weights(performances).unwrap();

        // Test percentiles
        let bottom_percentile = pw.get_weight_percentile(&organisms[0]).unwrap();
        let top_percentile = pw.get_weight_percentile(&organisms[9]).unwrap();

        assert!(bottom_percentile < top_percentile);
        assert!(bottom_percentile >= 0.0 && bottom_percentile <= 1.0);
        assert!(top_percentile >= 0.0 && top_percentile <= 1.0);
    }

    #[test]
    fn test_factor_weights_sum() {
        let factor_weights = FactorWeights::default();

        let sum = factor_weights.performance
            + factor_weights.reliability
            + factor_weights.adaptation
            + factor_weights.accuracy
            + factor_weights.responsiveness
            + factor_weights.efficiency
            + factor_weights.stability_contribution
            + factor_weights.emergence_factor;

        // Factor weights should sum to 1.0
        assert!((sum - 1.0).abs() < 0.001);
    }
}
