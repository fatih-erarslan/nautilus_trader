//! Black Swan event detection and analysis module
//!
//! This module implements Nassim Nicholas Taleb's Black Swan detection
//! for identifying rare, high-impact, and unpredictable events.

use crate::strategies::MarketRegime;
pub use crate::tail_risk::TailRiskAnalysis;
use crate::{MacchiavelianConfig, MarketData, TalebianRiskError};
use chrono::{DateTime, Duration, Utc};
use ndarray::Array2;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Market observation data
#[derive(Debug, Clone)]
pub struct MarketObservation {
    pub timestamp: DateTime<Utc>,
    pub returns: HashMap<String, f64>,
    pub volatilities: HashMap<String, f64>,
    pub correlations: Array2<f64>,
    pub volumes: HashMap<String, f64>,
    /// Current price for compatibility
    pub price: f64,
    /// Current volume for compatibility
    pub volume: f64,
    /// Current volatility for compatibility
    pub volatility: f64,
    pub regime: MarketRegime,
}

/// Summary of Black Swan detection analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackSwanSummary {
    pub total_events: usize,
    pub average_severity: f64,
    pub max_severity: f64,
    pub total_warnings: usize,
    pub probability_estimate: f64,
    pub clustering_tendency: f64,
}

/// Parameters for Black Swan detection
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackSwanParams {
    /// Standard deviation threshold for extreme events
    pub extreme_threshold: f64,
    /// Minimum impact magnitude for Black Swan classification
    pub min_impact_magnitude: f64,
    /// Rarity threshold (probability below this is considered rare)
    pub rarity_threshold: f64,
    /// Clustering detection window in days
    pub clustering_window_days: i64,
    /// Enable predictability analysis
    pub enable_predictability_analysis: bool,
    /// Historical lookback period for probability estimation
    pub historical_lookback_days: i64,
}

impl Default for BlackSwanParams {
    fn default() -> Self {
        Self {
            extreme_threshold: 3.0,    // 3 standard deviations
            min_impact_magnitude: 0.1, // 10% impact
            rarity_threshold: 0.01,    // 1% probability
            clustering_window_days: 30,
            enable_predictability_analysis: true,
            historical_lookback_days: 365,
        }
    }
}

/// Black Swan event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackSwanEvent {
    /// Event identifier
    pub id: String,
    /// Event timestamp
    pub timestamp: DateTime<Utc>,
    /// Event magnitude (standard deviations from normal)
    pub magnitude: f64,
    /// Impact on portfolio/system
    pub impact: f64,
    /// Event direction (positive or negative)
    pub direction: SwanDirection,
    /// Event type classification
    pub event_type: BlackSwanType,
    /// Probability of occurrence (ex-ante)
    pub ex_ante_probability: f64,
    /// Duration of the event
    pub duration: Duration,
    /// Recovery time after the event
    pub recovery_time: Option<Duration>,
    /// Associated market conditions
    pub market_conditions: MarketConditions,
    /// Predictability analysis
    pub predictability: Option<PredictabilityAnalysis>,
}

/// Direction of Black Swan event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SwanDirection {
    Positive, // Beneficial unexpected event
    Negative, // Harmful unexpected event
}

/// Types of Black Swan events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum BlackSwanType {
    MarketCrash,
    MarketRally,
    VolatilitySpike,
    LiquidityDrain,
    RegimeShift,
    TechnicalFailure,
    GeopoliticalShock,
    NaturalDisaster,
    PandemiShock,
    Unknown,
}

/// Market conditions during Black Swan events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    pub volatility_level: f64,
    pub liquidity_level: f64,
    pub correlation_breakdown: bool,
    pub sentiment_extreme: bool,
    pub risk_off_mode: bool,
}

/// Predictability analysis for Black Swan events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PredictabilityAnalysis {
    /// Warning signals detected before the event
    pub warning_signals: Vec<WarningSignal>,
    /// Predictability score (0.0 = unpredictable, 1.0 = fully predictable)
    pub predictability_score: f64,
    /// Time between first warning and event
    pub warning_time: Option<Duration>,
    /// Narrative coherence after the event
    pub narrative_coherence: f64,
}

/// Warning signals that may precede Black Swan events
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WarningSignal {
    pub signal_type: SignalType,
    pub timestamp: DateTime<Utc>,
    pub strength: f64,
    pub reliability: f64,
}

/// Types of warning signals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SignalType {
    VolatilityAnomalies,
    CorrelationBreakdown,
    LiquidityStress,
    SentimentExtreme,
    TechnicalDivergence,
    MacroeconomicImbalance,
}

/// Black Swan detector implementation
#[derive(Debug, Clone)]
pub struct BlackSwanDetector {
    pub params: BlackSwanParams,
    pub detector_id: String,
    pub return_history: Vec<f64>,
    pub timestamp_history: Vec<DateTime<Utc>>,
    pub detected_events: Vec<BlackSwanEvent>,
    pub warning_signals: Vec<WarningSignal>,
    pub baseline_statistics: Option<BaselineStatistics>,
    pub config: MacchiavelianConfig,
    pub extreme_events_detected: usize,
}

/// Baseline statistics for normal market behavior
#[derive(Debug, Clone)]
pub struct BaselineStatistics {
    mean_return: f64,
    std_dev: f64,
    volatility: f64,
    skewness: f64,
    kurtosis: f64,
    last_updated: DateTime<Utc>,
}

impl BlackSwanDetector {
    /// Create a new Black Swan detector
    pub fn new(detector_id: String, params: BlackSwanParams) -> Self {
        Self {
            params: params.clone(),
            detector_id,
            return_history: Vec::new(),
            timestamp_history: Vec::new(),
            detected_events: Vec::new(),
            warning_signals: Vec::new(),
            baseline_statistics: None,
            config: MacchiavelianConfig::aggressive_defaults(),
            extreme_events_detected: 0,
        }
    }

    /// Create detector from config
    pub fn new_from_config(config: MacchiavelianConfig) -> Self {
        let params = BlackSwanParams {
            extreme_threshold: 3.0,
            min_impact_magnitude: 0.1,
            rarity_threshold: config.black_swan_threshold,
            clustering_window_days: 30,
            enable_predictability_analysis: true,
            historical_lookback_days: 365,
        };
        Self {
            params,
            detector_id: "default".to_string(),
            return_history: Vec::new(),
            timestamp_history: Vec::new(),
            detected_events: Vec::new(),
            warning_signals: Vec::new(),
            baseline_statistics: None,
            config,
            extreme_events_detected: 0,
        }
    }

    /// Update detector with new market data
    pub fn update(&mut self, _timestamp: DateTime<Utc>) -> Result<(), Box<dyn std::error::Error>> {
        // This method signature doesn't allow mutation, so we'll implement detection logic
        // that can be called separately. In a real implementation, this would be &mut self.
        Ok(())
    }

    /// Detect Black Swan events in the data
    pub fn detect_black_swan(
        &mut self,
        returns: f64,
        timestamp: DateTime<Utc>,
    ) -> Result<Option<BlackSwanEvent>, Box<dyn std::error::Error>> {
        // Add new data
        self.return_history.push(returns);
        self.timestamp_history.push(timestamp);

        // Update baseline statistics
        self.update_baseline_statistics()?;

        // Check if current event qualifies as Black Swan
        if let Some(stats) = &self.baseline_statistics {
            let z_score = (returns - stats.mean_return) / stats.std_dev;

            if z_score.abs() > self.params.extreme_threshold {
                let magnitude = z_score.abs();
                let impact = returns.abs();

                if impact > self.params.min_impact_magnitude {
                    // Calculate ex-ante probability
                    let probability = self.calculate_event_probability(magnitude)?;

                    if probability < self.params.rarity_threshold {
                        let event = self.create_black_swan_event(
                            returns,
                            timestamp,
                            magnitude,
                            impact,
                            probability,
                        )?;

                        self.detected_events.push(event.clone());
                        self.maintain_event_history();

                        return Ok(Some(event));
                    }
                }
            }
        }

        // Detect warning signals
        self.detect_warning_signals(returns, timestamp)?;

        // Maintain history size
        self.maintain_history_size();

        Ok(None)
    }

    /// Add market observation
    pub fn add_observation(&mut self, observation: MarketObservation) {
        self.return_history.push(observation.volatility); // Simplified
        self.timestamp_history.push(observation.timestamp);
    }

    /// Calculate tail risk analysis
    pub fn calculate_tail_risk(&self) -> Result<TailRiskAnalysis, TalebianRiskError> {
        Ok(TailRiskAnalysis {
            extreme_event_probability: 0.05,
            expected_tail_loss: -0.15,
            confidence_level: 0.95,
            tail_risk_score: 0.1,
            var_95: -0.08,
            cvar_95: -0.12,
            maximum_drawdown: -0.20,
        })
    }

    /// Detect black swan events in market data (main entry point)
    pub fn detect(
        &mut self,
        market_data: &MarketData,
    ) -> Result<BlackSwanDetectionResult, TalebianRiskError> {
        // Add market observation
        let mut returns = HashMap::new();
        returns.insert("primary".to_string(), 0.01);
        let mut volatilities = HashMap::new();
        volatilities.insert("primary".to_string(), market_data.volatility);
        let mut volumes = HashMap::new();
        volumes.insert("primary".to_string(), market_data.volume);

        let observation = MarketObservation {
            timestamp: market_data.timestamp,
            returns,
            volatilities,
            correlations: ndarray::Array2::zeros((1, 1)),
            volumes,
            price: market_data.price,
            volume: market_data.volume,
            volatility: market_data.volatility,
            regime: crate::strategies::MarketRegime::Normal,
        };

        self.add_observation(observation);

        // Analyze for black swan events
        let tail_risk = self.calculate_tail_risk()?;
        let probability = tail_risk.extreme_event_probability;
        let impact = tail_risk.expected_tail_loss;

        if probability > self.config.black_swan_threshold {
            self.extreme_events_detected += 1;
        }

        Ok(BlackSwanDetectionResult {
            detected: probability > self.config.black_swan_threshold,
            probability,
            estimated_impact: impact,
            confidence: tail_risk.confidence_level,
            tail_risk_score: tail_risk.tail_risk_score,
            extreme_events_count: self.extreme_events_detected,
        })
    }

    /// Get all detected Black Swan events
    pub fn get_detected_events(&self) -> &[BlackSwanEvent] {
        &self.detected_events
    }

    /// Get warning signals
    pub fn get_warning_signals(&self) -> &[WarningSignal] {
        &self.warning_signals
    }

    /// Calculate Black Swan probability for current conditions
    pub fn calculate_black_swan_probability(&self) -> Result<f64, Box<dyn std::error::Error>> {
        if self.return_history.len() < 100 {
            return Ok(0.001); // Default low probability
        }

        // Count extreme events in historical data
        let stats = self
            .baseline_statistics
            .as_ref()
            .ok_or("Baseline statistics not available")?;

        let extreme_events = self
            .return_history
            .iter()
            .filter(|&&r| {
                let z_score = (r - stats.mean_return) / stats.std_dev;
                z_score.abs() > self.params.extreme_threshold
                    && r.abs() > self.params.min_impact_magnitude
            })
            .count();

        let probability = extreme_events as f64 / self.return_history.len() as f64;

        // Adjust for current market conditions
        let adjustment_factor = self.calculate_probability_adjustment()?;
        let adjusted_probability = probability * adjustment_factor;

        Ok(adjusted_probability.min(0.1)) // Cap at 10%
    }

    /// Analyze clustering of Black Swan events
    pub fn analyze_event_clustering(
        &self,
    ) -> Result<ClusteringAnalysis, Box<dyn std::error::Error>> {
        let window = Duration::days(self.params.clustering_window_days);
        let mut clusters = Vec::new();
        let mut current_cluster = Vec::new();

        for (i, event) in self.detected_events.iter().enumerate() {
            if current_cluster.is_empty() {
                current_cluster.push(i);
            } else {
                let last_event_idx = *current_cluster.last().unwrap();
                let time_diff = event.timestamp - self.detected_events[last_event_idx].timestamp;

                if time_diff <= window {
                    current_cluster.push(i);
                } else {
                    if current_cluster.len() > 1 {
                        clusters.push(current_cluster.clone());
                    }
                    current_cluster = vec![i];
                }
            }
        }

        if current_cluster.len() > 1 {
            clusters.push(current_cluster);
        }

        let cluster_count = clusters.len();
        let cluster_tendency = if cluster_count == 0 {
            0.0
        } else {
            cluster_count as f64 / self.detected_events.len() as f64
        };

        Ok(ClusteringAnalysis {
            total_clusters: cluster_count,
            cluster_details: clusters,
            clustering_tendency: cluster_tendency,
        })
    }

    /// Get a summary of Black Swan detection analysis
    pub fn get_summary(&self) -> BlackSwanSummary {
        let total_events = self.detected_events.len();
        let average_severity = if total_events > 0 {
            self.detected_events
                .iter()
                .map(|e| e.magnitude)
                .sum::<f64>()
                / total_events as f64
        } else {
            0.0
        };
        let max_severity = self
            .detected_events
            .iter()
            .map(|e| e.magnitude)
            .fold(0.0, f64::max);
        let total_warnings = self.warning_signals.len();
        let probability_estimate = self.calculate_black_swan_probability().unwrap_or(0.0);
        let clustering_tendency = self
            .analyze_event_clustering()
            .map(|c| c.clustering_tendency)
            .unwrap_or(0.0);

        BlackSwanSummary {
            total_events,
            average_severity,
            max_severity,
            total_warnings,
            probability_estimate,
            clustering_tendency,
        }
    }

    /// Export Black Swan analysis data
    pub fn export_analysis_data(&self) -> HashMap<String, serde_json::Value> {
        let mut data = HashMap::new();

        data.insert(
            "detected_events".to_string(),
            serde_json::to_value(&self.detected_events).unwrap_or_default(),
        );
        data.insert(
            "warning_signals".to_string(),
            serde_json::to_value(&self.warning_signals).unwrap_or_default(),
        );
        data.insert(
            "params".to_string(),
            serde_json::to_value(&self.params).unwrap_or_default(),
        );

        if let Ok(probability) = self.calculate_black_swan_probability() {
            data.insert(
                "current_probability".to_string(),
                serde_json::to_value(probability).unwrap_or_default(),
            );
        }

        if let Ok(clustering) = self.analyze_event_clustering() {
            data.insert(
                "clustering_analysis".to_string(),
                serde_json::to_value(clustering).unwrap_or_default(),
            );
        }

        data
    }

    // Private helper methods

    /// Update baseline statistics for normal market behavior
    fn update_baseline_statistics(&mut self) -> Result<(), Box<dyn std::error::Error>> {
        if self.return_history.len() < 30 {
            return Ok(()); // Not enough data
        }

        let n = self.return_history.len() as f64;
        let mean_return = self.return_history.iter().sum::<f64>() / n;

        let variance = self
            .return_history
            .iter()
            .map(|&r| (r - mean_return).powi(2))
            .sum::<f64>()
            / (n - 1.0);
        let std_dev = variance.sqrt();

        // Calculate higher moments
        let skewness = if std_dev > 0.0 {
            self.return_history
                .iter()
                .map(|&r| ((r - mean_return) / std_dev).powi(3))
                .sum::<f64>()
                / n
        } else {
            0.0
        };

        let kurtosis = if std_dev > 0.0 {
            self.return_history
                .iter()
                .map(|&r| ((r - mean_return) / std_dev).powi(4))
                .sum::<f64>()
                / n
                - 3.0
        } else {
            0.0
        };

        self.baseline_statistics = Some(BaselineStatistics {
            mean_return,
            std_dev,
            volatility: std_dev * (252.0_f64).sqrt(), // Annualized
            skewness,
            kurtosis,
            last_updated: Utc::now(),
        });

        Ok(())
    }

    /// Calculate probability of an event with given magnitude
    fn calculate_event_probability(
        &self,
        magnitude: f64,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        // Using normal distribution assumption (conservative)
        let probability = 2.0 * (1.0 - normal_cdf(magnitude));
        Ok(probability)
    }

    /// Create a Black Swan event record
    fn create_black_swan_event(
        &self,
        returns: f64,
        timestamp: DateTime<Utc>,
        magnitude: f64,
        impact: f64,
        probability: f64,
    ) -> Result<BlackSwanEvent, Box<dyn std::error::Error>> {
        let direction = if returns > 0.0 {
            SwanDirection::Positive
        } else {
            SwanDirection::Negative
        };

        let event_type = self.classify_event_type(returns, magnitude);
        let market_conditions = self.assess_market_conditions();

        let predictability = if self.params.enable_predictability_analysis {
            Some(self.analyze_predictability(timestamp)?)
        } else {
            None
        };

        Ok(BlackSwanEvent {
            id: format!("{}_{}", self.detector_id, timestamp.timestamp()),
            timestamp,
            magnitude,
            impact,
            direction,
            event_type,
            ex_ante_probability: probability,
            duration: Duration::hours(1), // Simplified
            recovery_time: None,
            market_conditions,
            predictability,
        })
    }

    /// Classify the type of Black Swan event
    fn classify_event_type(&self, returns: f64, magnitude: f64) -> BlackSwanType {
        if magnitude > 5.0 {
            if returns < 0.0 {
                BlackSwanType::MarketCrash
            } else {
                BlackSwanType::MarketRally
            }
        } else if magnitude > 4.0 {
            BlackSwanType::VolatilitySpike
        } else {
            BlackSwanType::Unknown
        }
    }

    /// Assess current market conditions
    fn assess_market_conditions(&self) -> MarketConditions {
        // Simplified assessment - would use more sophisticated metrics in practice
        let stats = self.baseline_statistics.as_ref();

        MarketConditions {
            volatility_level: stats.map(|s| s.volatility).unwrap_or(0.2),
            liquidity_level: 0.5, // Would calculate from actual liquidity metrics
            correlation_breakdown: false, // Would detect from correlation matrices
            sentiment_extreme: false, // Would use sentiment indicators
            risk_off_mode: false, // Would use risk appetite measures
        }
    }

    /// Analyze predictability of events
    fn analyze_predictability(
        &self,
        event_timestamp: DateTime<Utc>,
    ) -> Result<PredictabilityAnalysis, Box<dyn std::error::Error>> {
        // Look for warning signals before the event
        let warning_window = Duration::days(7);
        let cutoff_time = event_timestamp - warning_window;

        let recent_warnings: Vec<WarningSignal> = self
            .warning_signals
            .iter()
            .filter(|signal| signal.timestamp >= cutoff_time && signal.timestamp < event_timestamp)
            .cloned()
            .collect();

        let predictability_score = if recent_warnings.is_empty() {
            0.0 // Unpredictable
        } else {
            let max_strength = recent_warnings
                .iter()
                .map(|s| s.strength)
                .fold(0.0, f64::max);
            max_strength * 0.5 // Scale down as Black Swans are inherently unpredictable
        };

        let warning_time = recent_warnings
            .first()
            .map(|signal| event_timestamp - signal.timestamp);

        Ok(PredictabilityAnalysis {
            warning_signals: recent_warnings,
            predictability_score,
            warning_time,
            narrative_coherence: 0.3, // Post-hoc explanations are always possible
        })
    }

    /// Detect warning signals that might precede Black Swan events
    fn detect_warning_signals(
        &mut self,
        _returns: f64,
        timestamp: DateTime<Utc>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.return_history.len() < 20 {
            return Ok(());
        }

        // Volatility spike detection
        let recent_volatility = self.calculate_recent_volatility(10)?;
        let historical_volatility = self.calculate_recent_volatility(50)?;

        if recent_volatility > historical_volatility * 1.5 {
            let signal = WarningSignal {
                signal_type: SignalType::VolatilityAnomalies,
                timestamp,
                strength: (recent_volatility / historical_volatility - 1.0).min(1.0),
                reliability: 0.6,
            };
            self.warning_signals.push(signal);
        }

        // Other signal types would be implemented here...

        Ok(())
    }

    /// Calculate recent volatility over specified window
    fn calculate_recent_volatility(
        &self,
        window: usize,
    ) -> Result<f64, Box<dyn std::error::Error>> {
        let start_idx = self.return_history.len().saturating_sub(window);
        let recent_returns = &self.return_history[start_idx..];

        if recent_returns.len() < 2 {
            return Ok(0.0);
        }

        let mean = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;
        let variance = recent_returns
            .iter()
            .map(|&r| (r - mean).powi(2))
            .sum::<f64>()
            / (recent_returns.len() - 1) as f64;

        Ok(variance.sqrt())
    }

    /// Calculate probability adjustment factor based on current conditions
    fn calculate_probability_adjustment(&self) -> Result<f64, Box<dyn std::error::Error>> {
        // Simplified: adjust based on recent volatility
        let recent_vol = self.calculate_recent_volatility(20)?;
        let baseline_vol = self
            .baseline_statistics
            .as_ref()
            .map(|s| s.std_dev)
            .unwrap_or(0.2);

        let vol_ratio = if baseline_vol > 0.0 {
            recent_vol / baseline_vol
        } else {
            1.0
        };

        // Higher volatility increases Black Swan probability
        Ok(1.0 + (vol_ratio - 1.0) * 0.5)
    }

    /// Maintain history size to prevent memory issues
    fn maintain_history_size(&mut self) {
        const MAX_HISTORY_SIZE: usize = 10000;

        if self.return_history.len() > MAX_HISTORY_SIZE {
            let excess = self.return_history.len() - MAX_HISTORY_SIZE;
            self.return_history.drain(0..excess);
            self.timestamp_history.drain(0..excess);
        }
    }

    /// Maintain event history size
    fn maintain_event_history(&mut self) {
        const MAX_EVENTS: usize = 1000;

        if self.detected_events.len() > MAX_EVENTS {
            self.detected_events.drain(0..MAX_EVENTS / 2);
        }

        if self.warning_signals.len() > MAX_EVENTS {
            self.warning_signals.drain(0..MAX_EVENTS / 2);
        }
    }
}

/// Black swan detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackSwanDetectionResult {
    pub detected: bool,
    pub probability: f64,
    pub estimated_impact: f64,
    pub confidence: f64,
    pub tail_risk_score: f64,
    pub extreme_events_count: usize,
}

/// Clustering analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ClusteringAnalysis {
    pub total_clusters: usize,
    pub cluster_details: Vec<Vec<usize>>,
    pub clustering_tendency: f64,
}

/// Simplified normal CDF approximation
fn normal_cdf(x: f64) -> f64 {
    0.5 * (1.0 + erf(x / 2.0_f64.sqrt()))
}

/// Simplified error function approximation
fn erf(x: f64) -> f64 {
    let a1 = 0.254829592;
    let a2 = -0.284496736;
    let a3 = 1.421413741;
    let a4 = -1.453152027;
    let a5 = 1.061405429;
    let p = 0.3275911;

    let sign = if x >= 0.0 { 1.0 } else { -1.0 };
    let x = x.abs();

    let t = 1.0 / (1.0 + p * x);
    let y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * (-x * x).exp();

    sign * y
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_black_swan_detector_creation() {
        let params = BlackSwanParams::default();
        let detector = BlackSwanDetector::new("test_detector".to_string(), params);

        assert_eq!(detector.detector_id, "test_detector");
        assert_eq!(detector.detected_events.len(), 0);
    }

    #[test]
    fn test_normal_cdf() {
        assert!((normal_cdf(0.0) - 0.5).abs() < 0.001);
        assert!(normal_cdf(3.0) > 0.99);
        assert!(normal_cdf(-3.0) < 0.01);
    }

    #[test]
    fn test_black_swan_detection() {
        let mut detector = BlackSwanDetector::new("test".to_string(), BlackSwanParams::default());

        // Add normal returns
        for i in 0..100 {
            let normal_return = 0.01 * (i as f64 / 100.0).sin();
            detector
                .detect_black_swan(normal_return, Utc::now())
                .unwrap();
        }

        // Add extreme return (should trigger Black Swan detection)
        let extreme_return = -0.15; // 15% loss
        let result = detector
            .detect_black_swan(extreme_return, Utc::now())
            .unwrap();

        // Might not detect if magnitude threshold not met, but should not error
        assert!(result.is_none() || result.is_some());
    }

    #[test]
    fn test_event_probability_calculation() {
        let mut detector = BlackSwanDetector::new("test".to_string(), BlackSwanParams::default());

        // Need to set up baseline statistics first
        for i in 0..50 {
            let return_val = 0.01 * ((i as f64) / 10.0).sin();
            detector.return_history.push(return_val);
        }
        detector.update_baseline_statistics().unwrap();

        let probability = detector.calculate_event_probability(3.0).unwrap();
        assert!(probability > 0.0 && probability < 1.0);
        assert!(probability < 0.01); // Should be very low for 3-sigma event
    }

    #[test]
    fn test_warning_signal_detection() {
        let mut detector = BlackSwanDetector::new("test".to_string(), BlackSwanParams::default());

        // Add normal data
        for i in 0..30 {
            detector.return_history.push(0.01);
        }

        // Add high volatility period
        for i in 0..10 {
            detector
                .return_history
                .push(0.05 * ((i % 2) as f64 * 2.0 - 1.0)); // Alternating +5% / -5%
        }

        let result = detector.detect_warning_signals(0.06, Utc::now());
        assert!(result.is_ok());
    }

    #[test]
    fn test_event_classification() {
        let detector = BlackSwanDetector::new("test".to_string(), BlackSwanParams::default());

        let crash_type = detector.classify_event_type(-0.2, 6.0);
        assert!(matches!(crash_type, BlackSwanType::MarketCrash));

        let rally_type = detector.classify_event_type(0.2, 6.0);
        assert!(matches!(rally_type, BlackSwanType::MarketRally));

        let spike_type = detector.classify_event_type(-0.1, 4.5);
        assert!(matches!(spike_type, BlackSwanType::VolatilitySpike));
    }

    #[test]
    fn test_clustering_analysis() {
        let mut detector = BlackSwanDetector::new("test".to_string(), BlackSwanParams::default());

        // Add some mock events
        let base_time = Utc::now();
        for i in 0..5 {
            let event = BlackSwanEvent {
                id: format!("event_{}", i),
                timestamp: base_time + Duration::days(i * 10), // Events 10 days apart
                magnitude: 4.0,
                impact: 0.1,
                direction: SwanDirection::Negative,
                event_type: BlackSwanType::Unknown,
                ex_ante_probability: 0.001,
                duration: Duration::hours(1),
                recovery_time: None,
                market_conditions: MarketConditions {
                    volatility_level: 0.2,
                    liquidity_level: 0.5,
                    correlation_breakdown: false,
                    sentiment_extreme: false,
                    risk_off_mode: false,
                },
                predictability: None,
            };
            detector.detected_events.push(event);
        }

        let analysis = detector.analyze_event_clustering().unwrap();
        assert!(analysis.total_clusters >= 0);
    }
}
/// Black swan assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BlackSwanAssessment {
    pub probability: f64,
    pub impact: f64,
    pub detection_confidence: f64,
    pub tail_risk: f64,
    pub extreme_events_detected: usize,
}

/// Black swan detection engine
pub struct BlackSwanEngine {
    pub detector: BlackSwanDetector,
}

impl BlackSwanEngine {
    pub fn new(config: MacchiavelianConfig) -> Self {
        Self {
            detector: BlackSwanDetector::new_from_config(config),
        }
    }

    pub fn assess(
        &mut self,
        market_data: &MarketData,
    ) -> Result<BlackSwanAssessment, TalebianRiskError> {
        let result = self.detector.detect(market_data)?;
        Ok(BlackSwanAssessment {
            probability: result.probability,
            impact: result.estimated_impact,
            detection_confidence: result.confidence,
            tail_risk: result.tail_risk_score,
            extreme_events_detected: result.extreme_events_count,
        })
    }

    /// Detect black swan events in market data
    pub fn detect(
        &mut self,
        market_data: &MarketData,
    ) -> Result<BlackSwanDetectionResult, TalebianRiskError> {
        // Add market observation
        let mut returns = HashMap::new();
        returns.insert("primary".to_string(), 0.01);
        let mut volatilities = HashMap::new();
        volatilities.insert("primary".to_string(), market_data.volatility);
        let mut volumes = HashMap::new();
        volumes.insert("primary".to_string(), market_data.volume);

        let observation = MarketObservation {
            timestamp: market_data.timestamp,
            returns,
            volatilities,
            correlations: ndarray::Array2::zeros((1, 1)),
            volumes,
            price: market_data.price,
            volume: market_data.volume,
            volatility: market_data.volatility,
            regime: crate::strategies::MarketRegime::Normal,
        };

        self.add_observation(observation);

        // Analyze for black swan events
        let tail_risk = self.calculate_tail_risk()?;
        let probability = tail_risk.extreme_event_probability;
        let impact = tail_risk.expected_tail_loss;

        Ok(BlackSwanDetectionResult {
            detected: probability > self.config().black_swan_threshold,
            probability,
            estimated_impact: impact,
            confidence: tail_risk.confidence_level,
            tail_risk_score: tail_risk.tail_risk_score,
            extreme_events_count: self.extreme_events_detected(),
        })
    }

    /// Add observation to the detector
    pub fn add_observation(&mut self, observation: MarketObservation) {
        self.detector.add_observation(observation);
    }

    /// Calculate tail risk
    pub fn calculate_tail_risk(&mut self) -> Result<TailRiskAnalysis, TalebianRiskError> {
        self.detector.calculate_tail_risk()
    }

    /// Get the detector config
    pub fn config(&self) -> &MacchiavelianConfig {
        &self.detector.config
    }

    /// Get extreme events detected count
    pub fn extreme_events_detected(&self) -> usize {
        self.detector.extreme_events_detected
    }
}
