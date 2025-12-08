//! Antifragility measurement and analysis module
//!
//! This module implements Nassim Nicholas Taleb's concept of antifragility,
//! measuring how systems benefit from stress and volatility.

use crate::{MacchiavelianConfig, MarketData, TalebianRiskError};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Parameters for antifragility measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntifragilityParams {
    /// Volatility threshold for stress detection
    pub volatility_threshold: f64,
    /// Minimum observation period for measurement
    pub min_observation_period: usize,
    /// Stress test intensity (0.0 to 1.0)
    pub stress_test_intensity: f64,
    /// Convexity measurement window
    pub convexity_window: usize,
    /// Enable hormesis effect detection
    pub enable_hormesis: bool,
    /// Adaptation speed parameter
    pub adaptation_speed: f64,
}

impl Default for AntifragilityParams {
    fn default() -> Self {
        Self {
            volatility_threshold: 0.2,
            min_observation_period: 100,
            stress_test_intensity: 0.8,
            convexity_window: 50,
            enable_hormesis: true,
            adaptation_speed: 0.1,
        }
    }
}

/// Antifragility measurement result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntifragilityMeasurement {
    /// Overall antifragility score (0.0 to 1.0, 0.5 = neutral)
    pub overall_score: f64,
    /// Convexity to volatility
    pub convexity: f64,
    /// Benefit from volatility
    pub volatility_benefit: f64,
    /// Response to stress events
    pub stress_response: f64,
    /// Hormesis effect strength
    pub hormesis_effect: f64,
    /// Tail event benefit
    pub tail_benefit: f64,
    /// Regime adaptation capability
    pub regime_adaptation: f64,
    /// Antifragility level description
    pub level_description: String,
    /// Component scores breakdown
    pub component_scores: HashMap<String, f64>,
}

/// Antifragility measurer implementation
#[derive(Debug, Clone)]
pub struct AntifragilityMeasurer {
    params: AntifragilityParams,
    portfolio_id: String,
    return_history: Vec<f64>,
    volatility_history: Vec<f64>,
    stress_events: Vec<StressEvent>,
    adaptation_history: Vec<AdaptationMeasurement>,
    last_measurement: Option<AntifragilityMeasurement>,
}

/// Stress event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StressEvent {
    pub timestamp: DateTime<Utc>,
    pub stress_intensity: f64,
    pub duration: i64, // seconds
    pub pre_stress_performance: f64,
    pub during_stress_performance: f64,
    pub post_stress_performance: f64,
    pub recovery_time: Option<i64>,
}

/// Adaptation measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationMeasurement {
    pub timestamp: DateTime<Utc>,
    pub regime_change: RegimeChange,
    pub adaptation_speed: f64,
    pub adaptation_quality: f64,
    pub learning_effect: f64,
}

/// Types of regime changes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum RegimeChange {
    VolatilityIncrease {
        from: f64,
        to: f64,
    },
    VolatilityDecrease {
        from: f64,
        to: f64,
    },
    TrendChange {
        from: TrendDirection,
        to: TrendDirection,
    },
    MarketCrash {
        severity: f64,
    },
    Recovery {
        strength: f64,
    },
}

/// Market trend directions
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Bullish,
    Bearish,
    Sideways,
}

impl AntifragilityMeasurer {
    /// Create a new antifragility measurer
    pub fn new(portfolio_id: String, params: AntifragilityParams) -> Self {
        Self {
            params,
            portfolio_id,
            return_history: Vec::new(),
            volatility_history: Vec::new(),
            stress_events: Vec::new(),
            adaptation_history: Vec::new(),
            last_measurement: None,
        }
    }

    /// Update with new market data
    pub fn update(
        &mut self,
        returns: f64,
        volatility: f64,
        timestamp: DateTime<Utc>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        self.return_history.push(returns);
        self.volatility_history.push(volatility);

        // Detect stress events
        if self.is_stress_event(volatility) {
            self.record_stress_event(timestamp)?;
        }

        // Detect regime changes
        if let Some(regime_change) = self.detect_regime_change() {
            self.record_adaptation(regime_change, timestamp)?;
        }

        // Maintain history size
        self.maintain_history_size();

        Ok(())
    }

    /// Measure antifragility
    pub fn measure_antifragility(
        &mut self,
    ) -> Result<AntifragilityMeasurement, Box<dyn std::error::Error>> {
        if self.return_history.len() < self.params.min_observation_period {
            return Err("Insufficient data for antifragility measurement".into());
        }

        // Calculate component scores
        let convexity = self.calculate_convexity()?;
        let volatility_benefit = self.calculate_volatility_benefit()?;
        let stress_response = self.calculate_stress_response()?;
        let hormesis_effect = if self.params.enable_hormesis {
            self.calculate_hormesis_effect()?
        } else {
            0.0
        };
        let tail_benefit = self.calculate_tail_benefit()?;
        let regime_adaptation = self.calculate_regime_adaptation()?;

        // Calculate overall score (weighted average)
        let weights = [0.2, 0.2, 0.2, 0.1, 0.15, 0.15]; // Sum = 1.0
        let scores = [
            convexity,
            volatility_benefit,
            stress_response,
            hormesis_effect,
            tail_benefit,
            regime_adaptation,
        ];
        let overall_score = weights.iter().zip(scores.iter()).map(|(w, s)| w * s).sum();

        // Create component scores map
        let mut component_scores = HashMap::new();
        component_scores.insert("convexity".to_string(), convexity);
        component_scores.insert("volatility_benefit".to_string(), volatility_benefit);
        component_scores.insert("stress_response".to_string(), stress_response);
        component_scores.insert("hormesis_effect".to_string(), hormesis_effect);
        component_scores.insert("tail_benefit".to_string(), tail_benefit);
        component_scores.insert("regime_adaptation".to_string(), regime_adaptation);

        let level_description = self.classify_antifragility_level(overall_score);

        let measurement = AntifragilityMeasurement {
            overall_score,
            convexity,
            volatility_benefit,
            stress_response,
            hormesis_effect,
            tail_benefit,
            regime_adaptation,
            level_description,
            component_scores,
        };

        self.last_measurement = Some(measurement.clone());
        Ok(measurement)
    }

    /// Get stress events
    pub fn get_stress_events(&self) -> &[StressEvent] {
        &self.stress_events
    }

    /// Get adaptation history
    pub fn get_adaptation_history(&self) -> &[AdaptationMeasurement] {
        &self.adaptation_history
    }

    /// Get last measurement
    pub fn get_last_measurement(&self) -> Option<&AntifragilityMeasurement> {
        self.last_measurement.as_ref()
    }

    // Private helper methods

    /// Check if current volatility indicates a stress event
    fn is_stress_event(&self, current_volatility: f64) -> bool {
        if self.volatility_history.len() < 20 {
            return false;
        }

        let recent_avg_vol = self.volatility_history.iter().rev().take(20).sum::<f64>() / 20.0;

        current_volatility > recent_avg_vol * (1.0 + self.params.volatility_threshold)
    }

    /// Record a stress event
    fn record_stress_event(
        &mut self,
        timestamp: DateTime<Utc>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        if self.return_history.len() < 10 {
            return Ok(());
        }

        // Calculate performance before, during, and after stress
        let stress_window = 5; // 5 periods
        let pre_stress_start = self.return_history.len().saturating_sub(stress_window * 3);
        let stress_start = self.return_history.len().saturating_sub(stress_window * 2);
        let post_stress_start = self.return_history.len().saturating_sub(stress_window);

        let pre_stress_returns = &self.return_history[pre_stress_start..stress_start];
        let stress_returns = &self.return_history[stress_start..post_stress_start];
        let post_stress_returns = &self.return_history[post_stress_start..];

        let pre_stress_performance =
            pre_stress_returns.iter().sum::<f64>() / pre_stress_returns.len() as f64;
        let during_stress_performance =
            stress_returns.iter().sum::<f64>() / stress_returns.len() as f64;
        let post_stress_performance =
            post_stress_returns.iter().sum::<f64>() / post_stress_returns.len() as f64;

        let current_volatility = self.volatility_history.last().copied().unwrap_or(0.0);
        let stress_intensity = (current_volatility / self.params.volatility_threshold).min(1.0);

        let stress_event = StressEvent {
            timestamp,
            stress_intensity,
            duration: (stress_window as i64) * 86400, // Assume daily data
            pre_stress_performance,
            during_stress_performance,
            post_stress_performance,
            recovery_time: None, // Would be calculated later
        };

        self.stress_events.push(stress_event);
        Ok(())
    }

    /// Detect regime changes
    fn detect_regime_change(&self) -> Option<RegimeChange> {
        if self.volatility_history.len() < 40 {
            return None;
        }

        let current_window = 20;
        let previous_window = 20;

        let current_vol = self
            .volatility_history
            .iter()
            .rev()
            .take(current_window)
            .sum::<f64>()
            / current_window as f64;

        let previous_vol = self
            .volatility_history
            .iter()
            .rev()
            .skip(current_window)
            .take(previous_window)
            .sum::<f64>()
            / previous_window as f64;

        let vol_change_threshold = 0.3; // 30% change
        if (current_vol - previous_vol) / previous_vol > vol_change_threshold {
            Some(RegimeChange::VolatilityIncrease {
                from: previous_vol,
                to: current_vol,
            })
        } else if (previous_vol - current_vol) / previous_vol > vol_change_threshold {
            Some(RegimeChange::VolatilityDecrease {
                from: previous_vol,
                to: current_vol,
            })
        } else {
            None
        }
    }

    /// Record adaptation measurement
    fn record_adaptation(
        &mut self,
        regime_change: RegimeChange,
        timestamp: DateTime<Utc>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        // Simplified adaptation measurement
        let adaptation_speed = self.params.adaptation_speed;
        let adaptation_quality = 0.7; // Would be calculated based on performance
        let learning_effect = 0.1; // Would be calculated based on historical adaptations

        let adaptation = AdaptationMeasurement {
            timestamp,
            regime_change,
            adaptation_speed,
            adaptation_quality,
            learning_effect,
        };

        self.adaptation_history.push(adaptation);
        Ok(())
    }

    /// Calculate convexity to volatility
    fn calculate_convexity(&self) -> Result<f64, Box<dyn std::error::Error>> {
        let window = self.params.convexity_window.min(self.return_history.len());
        if window < 10 {
            return Ok(0.5); // Neutral
        }

        let recent_returns = &self.return_history[self.return_history.len() - window..];
        let recent_volatility = &self.volatility_history[self.volatility_history.len() - window..];

        // Calculate correlation between volatility and returns (positive = antifragile)
        let mean_returns = recent_returns.iter().sum::<f64>() / recent_returns.len() as f64;
        let mean_vol = recent_volatility.iter().sum::<f64>() / recent_volatility.len() as f64;

        let covariance = recent_returns
            .iter()
            .zip(recent_volatility.iter())
            .map(|(r, v)| (r - mean_returns) * (v - mean_vol))
            .sum::<f64>()
            / (recent_returns.len() - 1) as f64;

        let vol_variance = recent_volatility
            .iter()
            .map(|v| (v - mean_vol).powi(2))
            .sum::<f64>()
            / (recent_volatility.len() - 1) as f64;

        let returns_variance = recent_returns
            .iter()
            .map(|r| (r - mean_returns).powi(2))
            .sum::<f64>()
            / (recent_returns.len() - 1) as f64;

        let correlation = if vol_variance > 0.0 && returns_variance > 0.0 {
            covariance / (vol_variance.sqrt() * returns_variance.sqrt())
        } else {
            0.0
        };

        // Convert correlation to 0-1 scale (0.5 = neutral)
        Ok((correlation + 1.0) / 2.0)
    }

    /// Calculate volatility benefit
    fn calculate_volatility_benefit(&self) -> Result<f64, Box<dyn std::error::Error>> {
        if self.volatility_history.len() < 50 {
            return Ok(0.5);
        }

        // Compare performance in high vs low volatility periods
        let volatility_threshold =
            self.volatility_history.iter().sum::<f64>() / self.volatility_history.len() as f64;

        let mut high_vol_returns = Vec::new();
        let mut low_vol_returns = Vec::new();

        for (i, &vol) in self.volatility_history.iter().enumerate() {
            if let Some(&ret) = self.return_history.get(i) {
                if vol > volatility_threshold {
                    high_vol_returns.push(ret);
                } else {
                    low_vol_returns.push(ret);
                }
            }
        }

        if high_vol_returns.is_empty() || low_vol_returns.is_empty() {
            return Ok(0.5);
        }

        let high_vol_avg = high_vol_returns.iter().sum::<f64>() / high_vol_returns.len() as f64;
        let low_vol_avg = low_vol_returns.iter().sum::<f64>() / low_vol_returns.len() as f64;

        // Antifragile if performs better in high volatility
        let benefit_ratio = if low_vol_avg != 0.0 {
            high_vol_avg / low_vol_avg
        } else {
            1.0
        };

        // Convert to 0-1 scale
        Ok((benefit_ratio.ln() + 1.0).max(0.0).min(1.0))
    }

    /// Calculate stress response
    fn calculate_stress_response(&self) -> Result<f64, Box<dyn std::error::Error>> {
        if self.stress_events.is_empty() {
            return Ok(0.5);
        }

        let mut stress_responses = Vec::new();
        for event in &self.stress_events {
            // Calculate how well portfolio performed during stress
            let stress_performance = event.during_stress_performance;
            let baseline_performance = event.pre_stress_performance;

            let response = if baseline_performance != 0.0 {
                stress_performance / baseline_performance
            } else {
                1.0
            };

            stress_responses.push(response);
        }

        let avg_response = stress_responses.iter().sum::<f64>() / stress_responses.len() as f64;

        // Convert to 0-1 scale (>1 is good, <1 is bad)
        Ok((avg_response.ln() + 1.0).max(0.0).min(1.0))
    }

    /// Calculate hormesis effect
    fn calculate_hormesis_effect(&self) -> Result<f64, Box<dyn std::error::Error>> {
        // Simplified: measure if small stresses lead to improved performance
        if self.stress_events.len() < 3 {
            return Ok(0.5);
        }

        let small_stress_events: Vec<_> = self
            .stress_events
            .iter()
            .filter(|event| event.stress_intensity < 0.5)
            .collect();

        if small_stress_events.is_empty() {
            return Ok(0.5);
        }

        let hormesis_score = small_stress_events
            .iter()
            .map(|event| {
                let improvement = event.post_stress_performance - event.pre_stress_performance;
                improvement.max(0.0)
            })
            .sum::<f64>()
            / small_stress_events.len() as f64;

        Ok(hormesis_score.min(1.0))
    }

    /// Calculate tail benefit
    fn calculate_tail_benefit(&self) -> Result<f64, Box<dyn std::error::Error>> {
        if self.return_history.len() < 100 {
            return Ok(0.5);
        }

        let mut sorted_returns = self.return_history.clone();
        sorted_returns.sort_by(|a, b| a.partial_cmp(b).unwrap());

        // Look at extreme events (bottom and top 5%)
        let tail_size = (self.return_history.len() as f64 * 0.05) as usize;
        let bottom_tail = &sorted_returns[..tail_size];
        let top_tail = &sorted_returns[sorted_returns.len() - tail_size..];

        let bottom_avg = bottom_tail.iter().sum::<f64>() / bottom_tail.len() as f64;
        let top_avg = top_tail.iter().sum::<f64>() / top_tail.len() as f64;

        // Antifragile systems have positive skew (benefit from positive tail events)
        let tail_benefit = top_avg + bottom_avg.abs(); // Both tails contribute differently

        Ok((tail_benefit * 10.0).min(1.0).max(0.0)) // Scale appropriately
    }

    /// Calculate regime adaptation capability
    fn calculate_regime_adaptation(&self) -> Result<f64, Box<dyn std::error::Error>> {
        if self.adaptation_history.is_empty() {
            return Ok(0.5);
        }

        let avg_adaptation_quality = self
            .adaptation_history
            .iter()
            .map(|adaptation| adaptation.adaptation_quality)
            .sum::<f64>()
            / self.adaptation_history.len() as f64;

        Ok(avg_adaptation_quality)
    }

    /// Classify antifragility level
    fn classify_antifragility_level(&self, score: f64) -> String {
        match score {
            x if x < 0.2 => "Highly Fragile".to_string(),
            x if x < 0.4 => "Fragile".to_string(),
            x if x < 0.6 => "Robust (Neutral)".to_string(),
            x if x < 0.8 => "Antifragile".to_string(),
            _ => "Highly Antifragile".to_string(),
        }
    }

    /// Maintain history size to prevent memory issues
    fn maintain_history_size(&mut self) {
        const MAX_HISTORY_SIZE: usize = 10000;

        if self.return_history.len() > MAX_HISTORY_SIZE {
            let excess = self.return_history.len() - MAX_HISTORY_SIZE;
            self.return_history.drain(0..excess);
            self.volatility_history.drain(0..excess);
        }

        if self.stress_events.len() > 100 {
            self.stress_events.drain(0..50);
        }

        if self.adaptation_history.len() > 100 {
            self.adaptation_history.drain(0..50);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_antifragility_measurer_creation() {
        let params = AntifragilityParams::default();
        let measurer = AntifragilityMeasurer::new("test_portfolio".to_string(), params);

        assert_eq!(measurer.portfolio_id, "test_portfolio");
        assert_eq!(measurer.return_history.len(), 0);
    }

    #[test]
    fn test_update_with_data() {
        let mut measurer =
            AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());

        let result = measurer.update(0.01, 0.15, Utc::now());
        assert!(result.is_ok());
        assert_eq!(measurer.return_history.len(), 1);
        assert_eq!(measurer.volatility_history.len(), 1);
    }

    #[test]
    fn test_measure_antifragility_insufficient_data() {
        let mut measurer =
            AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());

        // Add only a few data points
        for i in 0..10 {
            measurer.update(i as f64 * 0.01, 0.15, Utc::now()).unwrap();
        }

        let result = measurer.measure_antifragility();
        assert!(result.is_err());
    }

    #[test]
    fn test_measure_antifragility_sufficient_data() {
        let mut measurer =
            AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());

        // Add sufficient data points
        for i in 0..150 {
            let returns = if i % 10 == 0 { -0.05 } else { 0.01 }; // Some negative shocks
            let volatility = if i % 10 == 0 { 0.3 } else { 0.15 }; // Higher volatility during shocks
            measurer.update(returns, volatility, Utc::now()).unwrap();
        }

        let result = measurer.measure_antifragility();
        assert!(result.is_ok());

        let measurement = result.unwrap();
        assert!(measurement.overall_score >= 0.0 && measurement.overall_score <= 1.0);
        assert!(!measurement.level_description.is_empty());
        assert_eq!(measurement.component_scores.len(), 6);
    }

    #[test]
    fn test_stress_event_detection() {
        let mut measurer =
            AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());

        // Add normal volatility data
        for _ in 0..30 {
            measurer.update(0.01, 0.15, Utc::now()).unwrap();
        }

        // Add high volatility (should trigger stress event)
        measurer.update(0.01, 0.35, Utc::now()).unwrap();

        assert!(!measurer.stress_events.is_empty());
    }

    #[test]
    fn test_regime_change_detection() {
        let mut measurer =
            AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());

        // Add low volatility period
        for _ in 0..25 {
            measurer.update(0.01, 0.10, Utc::now()).unwrap();
        }

        // Add high volatility period (should trigger regime change)
        for _ in 0..25 {
            measurer.update(0.01, 0.25, Utc::now()).unwrap();
        }

        assert!(!measurer.adaptation_history.is_empty());
    }

    #[test]
    fn test_convexity_calculation() {
        let mut measurer =
            AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());

        // Add data with positive correlation between volatility and returns (antifragile pattern)
        for i in 0..60 {
            let vol = 0.1 + (i as f64 * 0.01);
            let ret = 0.005 + (i as f64 * 0.0001); // Returns increase with volatility
            measurer.update(ret, vol, Utc::now()).unwrap();
        }

        let convexity = measurer.calculate_convexity().unwrap();
        assert!(convexity > 0.5); // Should be above neutral for antifragile pattern
    }

    #[test]
    fn test_classification_levels() {
        let measurer =
            AntifragilityMeasurer::new("test".to_string(), AntifragilityParams::default());

        assert_eq!(measurer.classify_antifragility_level(0.1), "Highly Fragile");
        assert_eq!(measurer.classify_antifragility_level(0.3), "Fragile");
        assert_eq!(
            measurer.classify_antifragility_level(0.5),
            "Robust (Neutral)"
        );
        assert_eq!(measurer.classify_antifragility_level(0.7), "Antifragile");
        assert_eq!(
            measurer.classify_antifragility_level(0.9),
            "Highly Antifragile"
        );
    }
}
/// Antifragility assessment result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntifragilityAssessment {
    pub score: f64,
    pub fragility_index: f64,
    pub robustness: f64,
    pub volatility_benefit: f64,
    pub stress_response: f64,
    pub confidence: f64,
}

/// Antifragility analysis engine
pub struct AntifragilityEngine {
    measurer: AntifragilityMeasurer,
    config: MacchiavelianConfig,
}

impl AntifragilityEngine {
    pub fn new(config: MacchiavelianConfig) -> Self {
        let params = AntifragilityParams {
            volatility_threshold: config.antifragility_threshold,
            min_observation_period: config.antifragility_window,
            stress_test_intensity: 0.8,
            convexity_window: 50,
            enable_hormesis: true,
            adaptation_speed: 0.1,
        };
        Self {
            measurer: AntifragilityMeasurer::new("antifragility_engine".to_string(), params),
            config,
        }
    }

    pub fn assess(
        &mut self,
        market_data: &MarketData,
    ) -> Result<AntifragilityAssessment, TalebianRiskError> {
        // Update with market data
        self.measurer
            .update(0.01, market_data.volatility, market_data.timestamp)
            .map_err(|e| TalebianRiskError::CalculationError(e.to_string()))?;

        let measurement = self
            .measurer
            .measure_antifragility()
            .map_err(|e| TalebianRiskError::CalculationError(e.to_string()))?;
        Ok(AntifragilityAssessment {
            score: measurement.overall_score,
            fragility_index: 1.0 - measurement.overall_score,
            robustness: measurement.overall_score,
            volatility_benefit: measurement.volatility_benefit,
            stress_response: measurement.stress_response,
            confidence: 0.8,
        })
    }

    pub fn calculate_antifragility(
        &mut self,
        market_data: &MarketData,
    ) -> Result<crate::stubs::AntifragilityAssessment, TalebianRiskError> {
        let assessment = self.assess(market_data)?;
        Ok(crate::stubs::AntifragilityAssessment {
            score: assessment.score,
            antifragility_score: assessment.score,
            fragility_index: assessment.fragility_index,
            robustness: assessment.robustness,
            volatility_benefit: assessment.volatility_benefit,
            stress_response: assessment.stress_response,
            confidence: assessment.confidence,
        })
    }
}
