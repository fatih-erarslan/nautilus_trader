//! Market Regime Detection Module
//!
//! Advanced market regime detection using multiple methods and quantum-enhanced classification.

use crate::core::{QarResult, FactorMap, StandardFactors};
use crate::error::QarError;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Regime detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeResult {
    /// Detected market regime
    pub regime: super::MarketRegime,
    /// Confidence in regime detection
    pub confidence: f64,
    /// Regime stability
    pub stability: f64,
    /// Regime characteristics
    pub characteristics: RegimeCharacteristics,
    /// Regime history
    pub regime_transitions: Vec<RegimeTransition>,
    /// Multi-model consensus
    pub model_consensus: ModelConsensus,
}

/// Market regime characteristics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeCharacteristics {
    /// Average volatility in regime
    pub avg_volatility: f64,
    /// Average return in regime
    pub avg_return: f64,
    /// Trend persistence
    pub trend_persistence: f64,
    /// Volume profile
    pub volume_profile: VolumeProfile,
    /// Correlation structure
    pub correlation_structure: CorrelationStructure,
}

/// Volume profile for regime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VolumeProfile {
    /// Average volume level
    pub avg_volume: f64,
    /// Volume volatility
    pub volume_volatility: f64,
    /// Volume-price relationship
    pub volume_price_correlation: f64,
}

/// Correlation structure in regime
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CorrelationStructure {
    /// Average cross-asset correlation
    pub avg_correlation: f64,
    /// Correlation stability
    pub correlation_stability: f64,
    /// Regime-specific correlations
    pub regime_correlations: HashMap<String, f64>,
}

/// Regime transition information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeTransition {
    /// Timestamp of transition
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// Previous regime
    pub from_regime: super::MarketRegime,
    /// New regime
    pub to_regime: super::MarketRegime,
    /// Transition probability
    pub transition_probability: f64,
    /// Transition duration
    pub transition_duration: std::time::Duration,
}

/// Multi-model consensus for regime detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelConsensus {
    /// Markov regime switching model result
    pub markov_model: RegimeClassification,
    /// Hidden Markov Model result
    pub hmm_model: RegimeClassification,
    /// Threshold model result
    pub threshold_model: RegimeClassification,
    /// Machine learning model result
    pub ml_model: RegimeClassification,
    /// Consensus confidence
    pub consensus_confidence: f64,
}

/// Individual model classification
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeClassification {
    /// Predicted regime
    pub regime: super::MarketRegime,
    /// Model confidence
    pub confidence: f64,
    /// Model-specific metrics
    pub metrics: HashMap<String, f64>,
}

/// Regime detector
pub struct RegimeDetector {
    config: super::AnalysisConfig,
    detection_params: RegimeDetectionParameters,
    regime_history: Vec<super::MarketRegime>,
    transition_matrix: TransitionMatrix,
    model_states: ModelStates,
    history: Vec<RegimeResult>,
}

/// Regime detection parameters
#[derive(Debug, Clone)]
pub struct RegimeDetectionParameters {
    /// Lookback window for regime detection
    pub lookback_window: usize,
    /// Volatility threshold for crisis regime
    pub crisis_volatility_threshold: f64,
    /// Trend threshold for bull/bear regimes
    pub trend_threshold: f64,
    /// Minimum regime duration
    pub min_regime_duration: usize,
    /// Transition smoothing factor
    pub transition_smoothing: f64,
}

/// Transition matrix for Markov model
#[derive(Debug, Clone)]
pub struct TransitionMatrix {
    /// Transition probabilities between regimes
    pub probabilities: HashMap<(super::MarketRegime, super::MarketRegime), f64>,
    /// Regime steady-state probabilities
    pub steady_state: HashMap<super::MarketRegime, f64>,
}

/// Model states for various detection methods
#[derive(Debug)]
pub struct ModelStates {
    /// Markov model state
    pub markov_state: MarkovState,
    /// HMM model state
    pub hmm_state: HmmState,
    /// Threshold model state
    pub threshold_state: ThresholdState,
}

/// Markov model state
#[derive(Debug)]
pub struct MarkovState {
    /// Current regime probabilities
    pub regime_probabilities: HashMap<super::MarketRegime, f64>,
    /// Transition counts
    pub transition_counts: HashMap<(super::MarketRegime, super::MarketRegime), usize>,
}

/// Hidden Markov Model state
#[derive(Debug)]
pub struct HmmState {
    /// Hidden state probabilities
    pub hidden_states: Vec<f64>,
    /// Emission probabilities
    pub emission_probabilities: HashMap<super::MarketRegime, Vec<f64>>,
    /// Viterbi path
    pub viterbi_path: Vec<super::MarketRegime>,
}

/// Threshold model state
#[derive(Debug)]
pub struct ThresholdState {
    /// Current thresholds
    pub thresholds: HashMap<String, f64>,
    /// Threshold crossings
    pub crossings: Vec<ThresholdCrossing>,
}

/// Threshold crossing event
#[derive(Debug, Clone)]
pub struct ThresholdCrossing {
    /// Variable that crossed threshold
    pub variable: String,
    /// Threshold value
    pub threshold: f64,
    /// Crossing direction
    pub direction: CrossingDirection,
    /// Timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Threshold crossing direction
#[derive(Debug, Clone)]
pub enum CrossingDirection {
    Above,
    Below,
}

impl Default for RegimeDetectionParameters {
    fn default() -> Self {
        Self {
            lookback_window: 50,
            crisis_volatility_threshold: 0.05,
            trend_threshold: 0.02,
            min_regime_duration: 5,
            transition_smoothing: 0.1,
        }
    }
}

impl Default for TransitionMatrix {
    fn default() -> Self {
        let mut probabilities = HashMap::new();
        let regimes = [
            super::MarketRegime::Bull,
            super::MarketRegime::Bear,
            super::MarketRegime::Consolidation,
            super::MarketRegime::Transition,
            super::MarketRegime::Crisis,
        ];

        // Initialize with uniform probabilities
        for &from_regime in &regimes {
            for &to_regime in &regimes {
                let prob = if from_regime == to_regime { 0.7 } else { 0.075 };
                probabilities.insert((from_regime, to_regime), prob);
            }
        }

        let mut steady_state = HashMap::new();
        for &regime in &regimes {
            steady_state.insert(regime, 0.2);
        }

        Self {
            probabilities,
            steady_state,
        }
    }
}

impl RegimeDetector {
    /// Create a new regime detector
    pub fn new(config: super::AnalysisConfig) -> QarResult<Self> {
        let model_states = ModelStates {
            markov_state: MarkovState {
                regime_probabilities: HashMap::new(),
                transition_counts: HashMap::new(),
            },
            hmm_state: HmmState {
                hidden_states: Vec::new(),
                emission_probabilities: HashMap::new(),
                viterbi_path: Vec::new(),
            },
            threshold_state: ThresholdState {
                thresholds: HashMap::new(),
                crossings: Vec::new(),
            },
        };

        Ok(Self {
            config,
            detection_params: RegimeDetectionParameters::default(),
            regime_history: Vec::new(),
            transition_matrix: TransitionMatrix::default(),
            model_states,
            history: Vec::new(),
        })
    }

    /// Detect market regime from factors
    pub async fn detect(&mut self, factors: &FactorMap) -> QarResult<RegimeResult> {
        // Extract market indicators from factors
        let indicators = self.extract_indicators(factors)?;
        
        // Run multiple detection models
        let model_consensus = self.run_detection_models(&indicators)?;
        
        // Determine consensus regime
        let regime = self.determine_consensus_regime(&model_consensus);
        
        // Calculate confidence and stability
        let confidence = model_consensus.consensus_confidence;
        let stability = self.calculate_regime_stability(&regime);
        
        // Calculate regime characteristics
        let characteristics = self.calculate_regime_characteristics(&indicators, &regime)?;
        
        // Update regime history and detect transitions
        let regime_transitions = self.update_regime_history(regime.clone());

        let result = RegimeResult {
            regime,
            confidence,
            stability,
            characteristics,
            regime_transitions,
            model_consensus,
        };

        // Store in history
        self.add_to_history(result.clone());

        Ok(result)
    }

    /// Extract market indicators from factors
    fn extract_indicators(&self, factors: &FactorMap) -> QarResult<MarketIndicators> {
        Ok(MarketIndicators {
            trend: factors.get_factor(&StandardFactors::Trend)?,
            volatility: factors.get_factor(&StandardFactors::Volatility)?,
            momentum: factors.get_factor(&StandardFactors::Momentum)?,
            volume: factors.get_factor(&StandardFactors::Volume)?,
            sentiment: factors.get_factor(&StandardFactors::Sentiment)?,
            liquidity: factors.get_factor(&StandardFactors::Liquidity)?,
            risk: factors.get_factor(&StandardFactors::Risk)?,
            efficiency: factors.get_factor(&StandardFactors::Efficiency)?,
        })
    }

    /// Run multiple detection models
    fn run_detection_models(&mut self, indicators: &MarketIndicators) -> QarResult<ModelConsensus> {
        let markov_model = self.run_markov_model(indicators)?;
        let hmm_model = self.run_hmm_model(indicators)?;
        let threshold_model = self.run_threshold_model(indicators)?;
        let ml_model = self.run_ml_model(indicators)?;

        // Calculate consensus confidence
        let models = vec![&markov_model, &hmm_model, &threshold_model, &ml_model];
        let consensus_confidence = self.calculate_consensus_confidence(&models);

        Ok(ModelConsensus {
            markov_model,
            hmm_model,
            threshold_model,
            ml_model,
            consensus_confidence,
        })
    }

    /// Run Markov regime switching model
    fn run_markov_model(&mut self, indicators: &MarketIndicators) -> QarResult<RegimeClassification> {
        // Update transition counts based on regime history
        self.update_transition_counts();

        // Calculate regime probabilities based on current indicators
        let mut regime_scores = HashMap::new();
        
        // Bull market scoring
        let bull_score = (indicators.trend * 0.4 + 
                         indicators.momentum * 0.3 + 
                         indicators.sentiment * 0.2 + 
                         (1.0 - indicators.volatility) * 0.1).max(0.0).min(1.0);
        regime_scores.insert(super::MarketRegime::Bull, bull_score);

        // Bear market scoring
        let bear_score = ((1.0 - indicators.trend) * 0.4 + 
                         (1.0 - indicators.momentum) * 0.3 + 
                         (1.0 - indicators.sentiment) * 0.2 + 
                         indicators.volatility * 0.1).max(0.0).min(1.0);
        regime_scores.insert(super::MarketRegime::Bear, bear_score);

        // Crisis regime scoring
        let crisis_score = (indicators.volatility * 0.5 + 
                           indicators.risk * 0.3 + 
                           (1.0 - indicators.liquidity) * 0.2).max(0.0).min(1.0);
        regime_scores.insert(super::MarketRegime::Crisis, crisis_score);

        // Consolidation regime scoring
        let consolidation_score = ((1.0 - indicators.volatility) * 0.4 + 
                                  indicators.efficiency * 0.3 + 
                                  indicators.liquidity * 0.3).max(0.0).min(1.0);
        regime_scores.insert(super::MarketRegime::Consolidation, consolidation_score);

        // Transition regime scoring
        let transition_score = 0.5; // Neutral scoring
        regime_scores.insert(super::MarketRegime::Transition, transition_score);

        // Find best regime
        let (best_regime, best_score) = regime_scores.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(k, v)| (k.clone(), *v))
            .unwrap_or((super::MarketRegime::Transition, 0.5));

        let mut metrics = HashMap::new();
        metrics.insert("bull_score".to_string(), regime_scores[&super::MarketRegime::Bull]);
        metrics.insert("bear_score".to_string(), regime_scores[&super::MarketRegime::Bear]);
        metrics.insert("crisis_score".to_string(), regime_scores[&super::MarketRegime::Crisis]);

        Ok(RegimeClassification {
            regime: best_regime,
            confidence: best_score,
            metrics,
        })
    }

    /// Run Hidden Markov Model
    fn run_hmm_model(&mut self, indicators: &MarketIndicators) -> QarResult<RegimeClassification> {
        // Simplified HMM implementation
        let observation = vec![
            indicators.trend,
            indicators.volatility,
            indicators.momentum,
            indicators.volume,
        ];

        // Calculate emission likelihoods for each regime
        let mut regime_likelihoods = HashMap::new();
        
        // Bull market: high trend, low volatility, high momentum
        let bull_likelihood = self.calculate_gaussian_likelihood(&observation, 
            &[0.7, 0.3, 0.7, 0.6], &[0.1, 0.1, 0.1, 0.1]);
        regime_likelihoods.insert(super::MarketRegime::Bull, bull_likelihood);

        // Bear market: low trend, high volatility, low momentum
        let bear_likelihood = self.calculate_gaussian_likelihood(&observation,
            &[0.3, 0.7, 0.3, 0.4], &[0.1, 0.1, 0.1, 0.1]);
        regime_likelihoods.insert(super::MarketRegime::Bear, bear_likelihood);

        // Crisis: very high volatility, low liquidity
        let crisis_likelihood = self.calculate_gaussian_likelihood(&observation,
            &[0.2, 0.9, 0.2, 0.3], &[0.2, 0.1, 0.2, 0.2]);
        regime_likelihoods.insert(super::MarketRegime::Crisis, crisis_likelihood);

        // Consolidation: medium trend, low volatility
        let consolidation_likelihood = self.calculate_gaussian_likelihood(&observation,
            &[0.5, 0.2, 0.5, 0.5], &[0.1, 0.05, 0.1, 0.1]);
        regime_likelihoods.insert(super::MarketRegime::Consolidation, consolidation_likelihood);

        let transition_likelihood = 0.1; // Low likelihood
        regime_likelihoods.insert(super::MarketRegime::Transition, transition_likelihood);

        // Find most likely regime
        let (best_regime, best_likelihood) = regime_likelihoods.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(k, v)| (k.clone(), *v))
            .unwrap_or((super::MarketRegime::Transition, 0.1));

        // Normalize likelihood to confidence
        let total_likelihood: f64 = regime_likelihoods.values().sum();
        let confidence = if total_likelihood > 0.0 {
            best_likelihood / total_likelihood
        } else {
            0.5
        };

        let mut metrics = HashMap::new();
        for (regime, likelihood) in regime_likelihoods {
            metrics.insert(format!("{:?}_likelihood", regime), likelihood);
        }

        Ok(RegimeClassification {
            regime: best_regime,
            confidence,
            metrics,
        })
    }

    /// Run threshold model
    fn run_threshold_model(&mut self, indicators: &MarketIndicators) -> QarResult<RegimeClassification> {
        // Define thresholds for regime classification
        let volatility_crisis_threshold = self.detection_params.crisis_volatility_threshold;
        let trend_bull_threshold = self.detection_params.trend_threshold;
        let trend_bear_threshold = -self.detection_params.trend_threshold;

        // Check threshold crossings
        self.check_threshold_crossings(indicators);

        let regime = if indicators.volatility > volatility_crisis_threshold && indicators.risk > 0.7 {
            super::MarketRegime::Crisis
        } else if indicators.trend > trend_bull_threshold && indicators.momentum > 0.6 {
            super::MarketRegime::Bull
        } else if indicators.trend < trend_bear_threshold && indicators.momentum < 0.4 {
            super::MarketRegime::Bear
        } else if indicators.volatility < 0.3 && indicators.efficiency > 0.6 {
            super::MarketRegime::Consolidation
        } else {
            super::MarketRegime::Transition
        };

        // Calculate confidence based on how far from thresholds
        let confidence = match regime {
            super::MarketRegime::Crisis => {
                let vol_excess = (indicators.volatility - volatility_crisis_threshold) / volatility_crisis_threshold;
                (0.5 + vol_excess).min(1.0).max(0.5)
            },
            super::MarketRegime::Bull => {
                let trend_excess = (indicators.trend - trend_bull_threshold) / trend_bull_threshold;
                (0.5 + trend_excess).min(1.0).max(0.5)
            },
            super::MarketRegime::Bear => {
                let trend_deficit = (trend_bear_threshold - indicators.trend) / trend_bear_threshold.abs();
                (0.5 + trend_deficit).min(1.0).max(0.5)
            },
            _ => 0.6,
        };

        let mut metrics = HashMap::new();
        metrics.insert("volatility_threshold".to_string(), volatility_crisis_threshold);
        metrics.insert("trend_bull_threshold".to_string(), trend_bull_threshold);
        metrics.insert("trend_bear_threshold".to_string(), trend_bear_threshold);

        Ok(RegimeClassification {
            regime,
            confidence,
            metrics,
        })
    }

    /// Run machine learning model (simplified)
    fn run_ml_model(&self, indicators: &MarketIndicators) -> QarResult<RegimeClassification> {
        // Simplified ML model using weighted features
        let features = vec![
            indicators.trend,
            indicators.volatility,
            indicators.momentum,
            indicators.volume,
            indicators.sentiment,
            indicators.liquidity,
            indicators.risk,
            indicators.efficiency,
        ];

        // Learned weights for each regime (simplified)
        let bull_weights = vec![0.3, -0.2, 0.25, 0.1, 0.2, 0.1, -0.15, 0.1];
        let bear_weights = vec![-0.3, 0.2, -0.25, -0.1, -0.2, -0.1, 0.15, -0.1];
        let crisis_weights = vec![-0.1, 0.4, -0.1, -0.2, -0.3, -0.25, 0.3, -0.2];
        let consolidation_weights = vec![0.0, -0.3, 0.0, 0.1, 0.1, 0.2, -0.1, 0.3];

        let bull_score = self.calculate_linear_score(&features, &bull_weights);
        let bear_score = self.calculate_linear_score(&features, &bear_weights);
        let crisis_score = self.calculate_linear_score(&features, &crisis_weights);
        let consolidation_score = self.calculate_linear_score(&features, &consolidation_weights);

        let scores = vec![
            (super::MarketRegime::Bull, bull_score),
            (super::MarketRegime::Bear, bear_score),
            (super::MarketRegime::Crisis, crisis_score),
            (super::MarketRegime::Consolidation, consolidation_score),
            (super::MarketRegime::Transition, 0.0),
        ];

        let (best_regime, best_score) = scores.iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(k, v)| (k.clone(), *v))
            .unwrap_or((super::MarketRegime::Transition, 0.0));

        // Apply softmax to get confidence
        let exp_scores: Vec<f64> = scores.iter().map(|(_, score)| score.exp()).collect();
        let sum_exp: f64 = exp_scores.iter().sum();
        let confidence = if sum_exp > 0.0 {
            best_score.exp() / sum_exp
        } else {
            0.5
        };

        let mut metrics = HashMap::new();
        metrics.insert("bull_score".to_string(), bull_score);
        metrics.insert("bear_score".to_string(), bear_score);
        metrics.insert("crisis_score".to_string(), crisis_score);
        metrics.insert("consolidation_score".to_string(), consolidation_score);

        Ok(RegimeClassification {
            regime: best_regime,
            confidence,
            metrics,
        })
    }

    /// Helper methods
    fn calculate_gaussian_likelihood(&self, observation: &[f64], mean: &[f64], std: &[f64]) -> f64 {
        if observation.len() != mean.len() || observation.len() != std.len() {
            return 0.0;
        }

        let mut log_likelihood = 0.0;
        for i in 0..observation.len() {
            let diff = observation[i] - mean[i];
            let variance = std[i].powi(2);
            log_likelihood -= 0.5 * (diff.powi(2) / variance + variance.ln() + (2.0 * std::f64::consts::PI).ln());
        }

        log_likelihood.exp()
    }

    fn calculate_linear_score(&self, features: &[f64], weights: &[f64]) -> f64 {
        if features.len() != weights.len() {
            return 0.0;
        }

        features.iter().zip(weights.iter()).map(|(f, w)| f * w).sum()
    }

    fn update_transition_counts(&mut self) {
        if self.regime_history.len() < 2 {
            return;
        }

        for window in self.regime_history.windows(2) {
            let from_regime = window[0].clone();
            let to_regime = window[1].clone();
            let key = (from_regime, to_regime);
            
            *self.model_states.markov_state.transition_counts.entry(key).or_insert(0) += 1;
        }
    }

    fn check_threshold_crossings(&mut self, indicators: &MarketIndicators) {
        let current_time = chrono::Utc::now();
        
        // Check volatility threshold crossing
        let vol_threshold = self.detection_params.crisis_volatility_threshold;
        if indicators.volatility > vol_threshold {
            self.model_states.threshold_state.crossings.push(ThresholdCrossing {
                variable: "volatility".to_string(),
                threshold: vol_threshold,
                direction: CrossingDirection::Above,
                timestamp: current_time,
            });
        }

        // Maintain crossing history
        let max_crossings = 100;
        if self.model_states.threshold_state.crossings.len() > max_crossings {
            self.model_states.threshold_state.crossings.remove(0);
        }
    }

    fn determine_consensus_regime(&self, consensus: &ModelConsensus) -> super::MarketRegime {
        let models = vec![
            &consensus.markov_model,
            &consensus.hmm_model,
            &consensus.threshold_model,
            &consensus.ml_model,
        ];

        // Count votes for each regime
        let mut regime_votes = HashMap::new();
        for model in &models {
            *regime_votes.entry(model.regime.clone()).or_insert(0) += 1;
        }

        // Return regime with most votes, or use confidence weighting if tied
        regime_votes.iter()
            .max_by(|(_, a), (_, b)| a.cmp(b))
            .map(|(regime, _)| regime.clone())
            .unwrap_or(super::MarketRegime::Transition)
    }

    fn calculate_consensus_confidence(&self, models: &[&RegimeClassification]) -> f64 {
        if models.is_empty() {
            return 0.0;
        }

        // Calculate average confidence
        let avg_confidence = models.iter().map(|m| m.confidence).sum::<f64>() / models.len() as f64;
        
        // Calculate regime agreement
        let regime_counts = models.iter().fold(HashMap::new(), |mut acc, model| {
            *acc.entry(model.regime.clone()).or_insert(0) += 1;
            acc
        });
        
        let max_agreement = regime_counts.values().max().unwrap_or(&0);
        let agreement_ratio = *max_agreement as f64 / models.len() as f64;
        
        // Combine average confidence with agreement
        (avg_confidence + agreement_ratio) / 2.0
    }

    fn calculate_regime_stability(&self, regime: &super::MarketRegime) -> f64 {
        if self.regime_history.len() < 3 {
            return 0.5;
        }

        // Count recent regime consistency
        let lookback = 10.min(self.regime_history.len());
        let recent_regimes = &self.regime_history[self.regime_history.len() - lookback..];
        
        let same_regime_count = recent_regimes.iter()
            .filter(|&r| r == regime)
            .count();
        
        same_regime_count as f64 / recent_regimes.len() as f64
    }

    fn calculate_regime_characteristics(&self, indicators: &MarketIndicators, regime: &super::MarketRegime) -> QarResult<RegimeCharacteristics> {
        // Calculate characteristics based on current indicators and regime
        let avg_volatility = indicators.volatility;
        let avg_return = match regime {
            super::MarketRegime::Bull => indicators.trend * 0.1,
            super::MarketRegime::Bear => -indicators.trend * 0.1,
            _ => 0.0,
        };
        
        let trend_persistence = indicators.momentum;
        
        let volume_profile = VolumeProfile {
            avg_volume: indicators.volume,
            volume_volatility: indicators.volatility * 0.5,
            volume_price_correlation: 0.3, // Simplified
        };
        
        let mut regime_correlations = HashMap::new();
        regime_correlations.insert("trend_volatility".to_string(), -0.2);
        regime_correlations.insert("momentum_volume".to_string(), 0.4);
        
        let correlation_structure = CorrelationStructure {
            avg_correlation: 0.3,
            correlation_stability: 0.7,
            regime_correlations,
        };

        Ok(RegimeCharacteristics {
            avg_volatility,
            avg_return,
            trend_persistence,
            volume_profile,
            correlation_structure,
        })
    }

    fn update_regime_history(&mut self, new_regime: super::MarketRegime) -> Vec<RegimeTransition> {
        let mut transitions = Vec::new();
        
        if let Some(&last_regime) = self.regime_history.last() {
            if last_regime != new_regime {
                // Detect regime transition
                transitions.push(RegimeTransition {
                    timestamp: chrono::Utc::now(),
                    from_regime: last_regime,
                    to_regime: new_regime.clone(),
                    transition_probability: 0.8, // Simplified
                    transition_duration: std::time::Duration::from_secs(3600), // Simplified
                });
            }
        }

        self.regime_history.push(new_regime);
        
        // Maintain history size
        if self.regime_history.len() > self.config.max_history {
            self.regime_history.remove(0);
        }

        transitions
    }

    fn add_to_history(&mut self, result: RegimeResult) {
        self.history.push(result);
        
        if self.history.len() > self.config.max_history {
            self.history.remove(0);
        }
    }

    /// Get detection history
    pub fn get_history(&self) -> &[RegimeResult] {
        &self.history
    }

    /// Get latest detection
    pub fn get_latest(&self) -> Option<&RegimeResult> {
        self.history.last()
    }

    /// Get regime history
    pub fn get_regime_history(&self) -> &[super::MarketRegime] {
        &self.regime_history
    }
}

/// Market indicators structure
#[derive(Debug, Clone)]
pub struct MarketIndicators {
    pub trend: f64,
    pub volatility: f64,
    pub momentum: f64,
    pub volume: f64,
    pub sentiment: f64,
    pub liquidity: f64,
    pub risk: f64,
    pub efficiency: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::StandardFactors;

    #[tokio::test]
    async fn test_regime_detector() {
        let config = super::super::AnalysisConfig::default();
        let mut detector = RegimeDetector::new(config).unwrap();

        let mut factors = std::collections::HashMap::new();
        factors.insert(StandardFactors::Trend.to_string(), 0.8);
        factors.insert(StandardFactors::Volatility.to_string(), 0.2);
        factors.insert(StandardFactors::Momentum.to_string(), 0.7);
        factors.insert(StandardFactors::Volume.to_string(), 0.6);
        factors.insert(StandardFactors::Sentiment.to_string(), 0.8);
        factors.insert(StandardFactors::Liquidity.to_string(), 0.7);
        factors.insert(StandardFactors::Risk.to_string(), 0.3);
        factors.insert(StandardFactors::Efficiency.to_string(), 0.6);
        
        let factor_map = FactorMap::new(factors).unwrap();
        let result = detector.detect(&factor_map).await;
        
        assert!(result.is_ok());
        let regime_result = result.unwrap();
        assert!(regime_result.confidence >= 0.0 && regime_result.confidence <= 1.0);
        assert!(regime_result.stability >= 0.0 && regime_result.stability <= 1.0);
    }

    #[test]
    fn test_gaussian_likelihood() {
        let config = super::super::AnalysisConfig::default();
        let detector = RegimeDetector::new(config).unwrap();
        
        let observation = vec![0.8, 0.2, 0.7];
        let mean = vec![0.8, 0.2, 0.7];
        let std = vec![0.1, 0.1, 0.1];
        
        let likelihood = detector.calculate_gaussian_likelihood(&observation, &mean, &std);
        assert!(likelihood > 0.0);
        
        // Should be higher likelihood for closer match
        let far_mean = vec![0.2, 0.8, 0.3];
        let far_likelihood = detector.calculate_gaussian_likelihood(&observation, &far_mean, &std);
        assert!(likelihood > far_likelihood);
    }

    #[test]
    fn test_linear_score() {
        let config = super::super::AnalysisConfig::default();
        let detector = RegimeDetector::new(config).unwrap();
        
        let features = vec![1.0, 0.5, 0.8];
        let weights = vec![0.3, 0.2, 0.5];
        
        let score = detector.calculate_linear_score(&features, &weights);
        let expected = 1.0 * 0.3 + 0.5 * 0.2 + 0.8 * 0.5;
        assert!((score - expected).abs() < 0.01);
    }

    #[test]
    fn test_transition_matrix_default() {
        let matrix = TransitionMatrix::default();
        
        // Check that probabilities sum to 1 for each regime
        let regimes = [
            super::super::MarketRegime::Bull,
            super::super::MarketRegime::Bear,
            super::super::MarketRegime::Consolidation,
            super::super::MarketRegime::Transition,
            super::super::MarketRegime::Crisis,
        ];

        for &from_regime in &regimes {
            let sum: f64 = regimes.iter()
                .map(|&to_regime| matrix.probabilities.get(&(from_regime, to_regime)).unwrap_or(&0.0))
                .sum();
            assert!((sum - 1.0).abs() < 0.01);
        }
    }

    #[tokio::test]
    async fn test_regime_transition_detection() {
        let config = super::super::AnalysisConfig::default();
        let mut detector = RegimeDetector::new(config).unwrap();

        // First detection - Bull market
        let mut bull_factors = std::collections::HashMap::new();
        bull_factors.insert(StandardFactors::Trend.to_string(), 0.9);
        bull_factors.insert(StandardFactors::Volatility.to_string(), 0.2);
        bull_factors.insert(StandardFactors::Momentum.to_string(), 0.8);
        bull_factors.insert(StandardFactors::Volume.to_string(), 0.7);
        bull_factors.insert(StandardFactors::Sentiment.to_string(), 0.9);
        bull_factors.insert(StandardFactors::Liquidity.to_string(), 0.8);
        bull_factors.insert(StandardFactors::Risk.to_string(), 0.2);
        bull_factors.insert(StandardFactors::Efficiency.to_string(), 0.7);
        
        let bull_factor_map = FactorMap::new(bull_factors).unwrap();
        let _ = detector.detect(&bull_factor_map).await.unwrap();

        // Second detection - Bear market (should trigger transition)
        let mut bear_factors = std::collections::HashMap::new();
        bear_factors.insert(StandardFactors::Trend.to_string(), 0.1);
        bear_factors.insert(StandardFactors::Volatility.to_string(), 0.8);
        bear_factors.insert(StandardFactors::Momentum.to_string(), 0.2);
        bear_factors.insert(StandardFactors::Volume.to_string(), 0.4);
        bear_factors.insert(StandardFactors::Sentiment.to_string(), 0.2);
        bear_factors.insert(StandardFactors::Liquidity.to_string(), 0.3);
        bear_factors.insert(StandardFactors::Risk.to_string(), 0.8);
        bear_factors.insert(StandardFactors::Efficiency.to_string(), 0.3);
        
        let bear_factor_map = FactorMap::new(bear_factors).unwrap();
        let result = detector.detect(&bear_factor_map).await.unwrap();

        // Should have detected a transition
        assert!(!result.regime_transitions.is_empty());
        assert_eq!(detector.get_regime_history().len(), 2);
    }
}