//! Market analysis implementation using quantum algorithms
//!
//! This module provides advanced market analysis capabilities using quantum computing
//! for enhanced pattern recognition and regime detection.

use crate::core::{
    QarResult, QarError, StandardFactors, FactorMap, MarketPhase, constants,
    CircuitParams, ExecutionContext, QuantumResult
};
use async_trait::async_trait;
use std::collections::HashMap;
use super::types::*;
use super::traits::*;
use super::circuits::{QftCircuit, PatternRecognitionCircuit};

/// Quantum market analyzer
#[derive(Debug)]
pub struct QuantumMarketAnalyzer {
    /// QFT circuit for spectral analysis
    qft_circuit: QftCircuit,
    /// Pattern recognition circuit
    pattern_circuit: PatternRecognitionCircuit,
    /// Analysis cache
    analysis_cache: HashMap<String, RegimeAnalysis>,
    /// Historical analyses for trend detection
    history: Vec<(chrono::DateTime<chrono::Utc>, RegimeAnalysis)>,
}

impl QuantumMarketAnalyzer {
    /// Create a new quantum market analyzer
    pub fn new(num_qubits: usize) -> Self {
        let qft_circuit = QftCircuit::new(num_qubits);
        let pattern_circuit = PatternRecognitionCircuit::new(
            num_qubits,
            constants::pattern::ENCODING_PRECISION,
        );

        Self {
            qft_circuit,
            pattern_circuit,
            analysis_cache: HashMap::new(),
            history: Vec::new(),
        }
    }

    /// Extract factor values as normalized vector
    fn extract_normalized_factors(&self, factors: &FactorMap) -> Vec<f64> {
        let mut values: Vec<f64> = StandardFactors::all_factors()
            .iter()
            .map(|factor| factors.get(factor).unwrap_or(0.0))
            .collect();

        // Normalize to [0, 1] range
        let max_val = values.iter().fold(0.0, |a, &b| a.max(b.abs()));
        if max_val > 0.0 {
            for value in &mut values {
                *value = (*value + max_val) / (2.0 * max_val); // Shift and scale to [0, 1]
            }
        }

        values
    }

    /// Perform spectral analysis using QFT
    async fn perform_spectral_analysis(&self, factors: &FactorMap) -> QarResult<Vec<f64>> {
        let normalized_factors = self.extract_normalized_factors(factors);
        
        // Pad to circuit size
        let expected_size = 1 << self.qft_circuit.num_qubits();
        let mut padded_factors = normalized_factors;
        padded_factors.resize(expected_size, 0.0);

        let params = CircuitParams::new(padded_factors, self.qft_circuit.num_qubits());
        let context = ExecutionContext::default();

        let result = self.qft_circuit.execute(&params, &context).await?;
        Ok(result.expectation_values)
    }

    /// Calculate regime stability
    fn calculate_regime_stability(&self, current: &RegimeAnalysis) -> f64 {
        if self.history.is_empty() {
            return 0.5; // Neutral stability for first analysis
        }

        let recent_analyses: Vec<&RegimeAnalysis> = self.history
            .iter()
            .rev()
            .take(10) // Last 10 analyses
            .map(|(_, analysis)| analysis)
            .collect();

        if recent_analyses.is_empty() {
            return 0.5;
        }

        // Calculate phase consistency
        let same_phase_count = recent_analyses
            .iter()
            .filter(|analysis| analysis.phase == current.phase)
            .count();

        let phase_stability = same_phase_count as f64 / recent_analyses.len() as f64;

        // Calculate confidence trend
        let avg_confidence: f64 = recent_analyses
            .iter()
            .map(|a| a.confidence)
            .sum::<f64>() / recent_analyses.len() as f64;

        let confidence_stability = if avg_confidence > 0.0 {
            (current.confidence / avg_confidence).min(1.0)
        } else {
            0.5
        };

        // Combine metrics
        (phase_stability + confidence_stability) / 2.0
    }

    /// Detect regime transitions
    fn detect_regime_transition(&self, current: &RegimeAnalysis) -> bool {
        if self.history.len() < 2 {
            return false;
        }

        let previous = &self.history[self.history.len() - 1].1;
        
        // Check for phase change
        if current.phase != previous.phase {
            return true;
        }

        // Check for significant confidence drop
        if current.confidence < previous.confidence * 0.7 {
            return true;
        }

        // Check for volatility spike
        if current.volatility > previous.volatility * 1.5 {
            return true;
        }

        false
    }

    /// Calculate market momentum from spectral analysis
    fn calculate_momentum(&self, spectral_power: &[f64]) -> f64 {
        if spectral_power.len() < 3 {
            return 0.0;
        }

        let low_freq = spectral_power[0];
        let mid_freq = spectral_power[spectral_power.len() / 2];
        let high_freq = spectral_power[spectral_power.len() - 1];

        // Momentum favors low frequency components
        let total = low_freq + mid_freq + high_freq;
        if total > 0.0 {
            (low_freq - high_freq) / total
        } else {
            0.0
        }
    }

    /// Detect trend direction and strength
    fn analyze_trend(&self, spectral_power: &[f64]) -> (f64, f64) {
        if spectral_power.is_empty() {
            return (0.0, 0.0);
        }

        // Calculate spectral centroid (frequency center of mass)
        let total_power: f64 = spectral_power.iter().sum();
        if total_power == 0.0 {
            return (0.0, 0.0);
        }

        let centroid: f64 = spectral_power
            .iter()
            .enumerate()
            .map(|(i, &power)| i as f64 * power)
            .sum::<f64>() / total_power;

        let normalized_centroid = centroid / spectral_power.len() as f64;

        // Direction: low frequencies = upward trend, high frequencies = downward trend
        let direction = 1.0 - 2.0 * normalized_centroid; // Maps [0, 1] to [1, -1]

        // Strength: concentration of power
        let variance: f64 = spectral_power
            .iter()
            .enumerate()
            .map(|(i, &power)| {
                let freq_diff = i as f64 - centroid;
                freq_diff * freq_diff * power
            })
            .sum::<f64>() / total_power;

        let strength = 1.0 / (1.0 + variance); // Higher variance = lower strength

        (direction, strength)
    }

    /// Calculate volatility clustering
    fn analyze_volatility_clustering(&self, spectral_power: &[f64]) -> f64 {
        if spectral_power.len() < 4 {
            return 0.0;
        }

        // High frequency components indicate volatility clustering
        let high_freq_start = 3 * spectral_power.len() / 4;
        let high_freq_power: f64 = spectral_power.iter().skip(high_freq_start).sum();
        let total_power: f64 = spectral_power.iter().sum();

        if total_power > 0.0 {
            high_freq_power / total_power
        } else {
            0.0
        }
    }

    /// Estimate cycle period from spectral analysis
    fn estimate_cycle_period(&self, spectral_power: &[f64]) -> Option<f64> {
        if spectral_power.len() < 3 {
            return None;
        }

        // Find dominant frequency (excluding DC component)
        let mut max_power = 0.0;
        let mut dominant_freq = 0;

        for (i, &power) in spectral_power.iter().enumerate().skip(1) {
            if power > max_power {
                max_power = power;
                dominant_freq = i;
            }
        }

        if dominant_freq > 0 && max_power > 0.0 {
            Some(spectral_power.len() as f64 / dominant_freq as f64)
        } else {
            None
        }
    }
}

#[async_trait]
impl MarketAnalyzer for QuantumMarketAnalyzer {
    async fn analyze_regime(&self, factors: &FactorMap) -> QarResult<RegimeAnalysis> {
        // Generate cache key
        let cache_key = format!("{:?}", factors);
        
        // Check cache first
        if let Some(cached) = self.analysis_cache.get(&cache_key) {
            return Ok(cached.clone());
        }

        // Perform spectral analysis
        let spectral_power = self.perform_spectral_analysis(factors).await?;

        // Determine market phase
        let phase = self.determine_market_phase(&spectral_power);
        let confidence = self.calculate_regime_confidence(&spectral_power);
        let strength = self.calculate_regime_strength(&spectral_power);
        let volatility = self.estimate_volatility(&spectral_power);
        let noise_level = self.estimate_noise_level(&spectral_power);

        // Calculate phase coherence
        let phase_coherence = self.calculate_phase_coherence(&spectral_power);

        let mut analysis = RegimeAnalysis {
            phase,
            confidence,
            strength,
            volatility,
            noise_level,
            spectral_power: spectral_power.clone(),
            phase_coherence,
        };

        // Add to history for trend analysis
        let mut self_mut = unsafe { &mut *(self as *const _ as *mut Self) };
        self_mut.history.push((chrono::Utc::now(), analysis.clone()));

        // Keep only recent history
        if self_mut.history.len() > constants::DEFAULT_MEMORY_LENGTH {
            self_mut.history.remove(0);
        }

        // Cache the result
        self_mut.analysis_cache.insert(cache_key, analysis.clone());

        Ok(analysis)
    }

    async fn detect_patterns(&self, factors: &FactorMap) -> QarResult<Vec<PatternMatch>> {
        let normalized_factors = self.extract_normalized_factors(factors);
        
        // Pad to circuit requirements
        let expected_size = 1 << self.pattern_circuit.num_qubits();
        let mut padded_factors = normalized_factors;
        padded_factors.resize(expected_size, 0.0);

        let params = CircuitParams::new(padded_factors, self.pattern_circuit.num_qubits());
        let context = ExecutionContext::default();

        let result = self.pattern_circuit.execute(&params, &context).await?;

        // Convert results to pattern matches
        let mut patterns = Vec::new();
        for (i, &similarity) in result.expectation_values.iter().enumerate() {
            if similarity >= constants::MIN_PATTERN_CONFIDENCE {
                let pattern_id = match i {
                    0 => "bull_trend".to_string(),
                    1 => "bear_trend".to_string(),
                    2 => "sideways_market".to_string(),
                    _ => format!("pattern_{}", i),
                };

                let mut metadata = HashMap::new();
                metadata.insert("similarity_score".to_string(), similarity.to_string());
                metadata.insert("pattern_index".to_string(), i.to_string());

                let mut pattern = PatternMatch::new(pattern_id, similarity, similarity);
                pattern.metadata = metadata;
                patterns.push(pattern);
            }
        }

        // Sort by similarity (highest first)
        patterns.sort_by(|a, b| b.similarity.partial_cmp(&a.similarity).unwrap_or(std::cmp::Ordering::Equal));

        Ok(patterns)
    }

    async fn predict_direction(&self, factors: &FactorMap) -> QarResult<MarketPrediction> {
        // Perform spectral analysis
        let spectral_power = self.perform_spectral_analysis(factors).await?;

        // Analyze trend
        let (direction, strength) = self.analyze_trend(&spectral_power);
        let momentum = self.calculate_momentum(&spectral_power);

        // Combine signals
        let predicted_direction = (direction + momentum) / 2.0;
        let confidence = strength * self.calculate_regime_confidence(&spectral_power);

        // Estimate time horizon based on dominant cycle
        let time_horizon_ms = if let Some(period) = self.estimate_cycle_period(&spectral_power) {
            (period * 60000.0) as u64 // Convert to milliseconds (assuming period is in minutes)
        } else {
            3600000 // Default 1 hour
        };

        // Create supporting factors map
        let mut supporting_factors = HashMap::new();
        for (i, factor) in StandardFactors::all_factors().iter().enumerate() {
            if i < spectral_power.len() {
                supporting_factors.insert(format!("{:?}", factor), spectral_power[i]);
            }
        }

        Ok(MarketPrediction {
            direction: predicted_direction,
            confidence,
            time_horizon_ms,
            factors: supporting_factors,
        })
    }

    async fn calculate_volatility(&self, factors: &FactorMap) -> QarResult<f64> {
        let spectral_power = self.perform_spectral_analysis(factors).await?;
        
        // Volatility from spectral analysis
        let spectral_volatility = self.estimate_volatility(&spectral_power);
        
        // Volatility clustering component
        let clustering = self.analyze_volatility_clustering(&spectral_power);
        
        // Combine metrics
        let total_volatility = spectral_volatility + clustering * 0.5;
        
        Ok(total_volatility.min(1.0)) // Cap at 1.0
    }
}

// Helper methods for quantum market analyzer
impl QuantumMarketAnalyzer {
    /// Determine market phase from spectral power distribution
    fn determine_market_phase(&self, spectral_power: &[f64]) -> MarketPhase {
        if spectral_power.is_empty() {
            return MarketPhase::Uncertain;
        }

        let total_power: f64 = spectral_power.iter().sum();
        if total_power == 0.0 {
            return MarketPhase::Sideways;
        }

        // Analyze frequency distribution
        let third = spectral_power.len() / 3;
        let low_freq_power: f64 = spectral_power.iter().take(third).sum();
        let mid_freq_power: f64 = spectral_power.iter().skip(third).take(third).sum();
        let high_freq_power: f64 = spectral_power.iter().skip(2 * third).sum();

        let low_ratio = low_freq_power / total_power;
        let mid_ratio = mid_freq_power / total_power;
        let high_ratio = high_freq_power / total_power;

        // Decision thresholds from constants
        if low_ratio > constants::market::PHASE_TRANSITION_THRESHOLD {
            MarketPhase::Growth
        } else if high_ratio > constants::market::PHASE_TRANSITION_THRESHOLD {
            MarketPhase::Decline
        } else if mid_ratio > constants::market::PHASE_STABILITY_THRESHOLD {
            MarketPhase::Sideways
        } else {
            MarketPhase::Uncertain
        }
    }

    /// Calculate regime confidence using entropy
    fn calculate_regime_confidence(&self, spectral_power: &[f64]) -> f64 {
        if spectral_power.is_empty() {
            return 0.0;
        }

        let total_power: f64 = spectral_power.iter().sum();
        if total_power == 0.0 {
            return 0.0;
        }

        // Calculate Shannon entropy
        let entropy: f64 = spectral_power.iter()
            .map(|&p| {
                let prob = p / total_power;
                if prob > constants::math::EPSILON {
                    -prob * prob.ln()
                } else {
                    0.0
                }
            })
            .sum();

        let max_entropy = (spectral_power.len() as f64).ln();
        if max_entropy > constants::math::EPSILON {
            (1.0 - (entropy / max_entropy)).max(0.0)
        } else {
            0.0
        }
    }

    /// Calculate regime strength
    fn calculate_regime_strength(&self, spectral_power: &[f64]) -> f64 {
        if spectral_power.is_empty() {
            return 0.0;
        }

        let max_power = spectral_power.iter().fold(0.0, |a, &b| a.max(b));
        let total_power: f64 = spectral_power.iter().sum();

        if total_power > constants::math::EPSILON {
            max_power / total_power
        } else {
            0.0
        }
    }

    /// Estimate volatility from spectral variance
    fn estimate_volatility(&self, spectral_power: &[f64]) -> f64 {
        if spectral_power.len() < 2 {
            return 0.0;
        }

        let mean = spectral_power.iter().sum::<f64>() / spectral_power.len() as f64;
        let variance = spectral_power.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / spectral_power.len() as f64;

        variance.sqrt()
    }

    /// Estimate noise level from high-frequency components
    fn estimate_noise_level(&self, spectral_power: &[f64]) -> f64 {
        if spectral_power.is_empty() {
            return 1.0;
        }

        // High frequency components indicate noise
        let high_freq_start = 2 * spectral_power.len() / 3;
        let high_freq_power: f64 = spectral_power.iter().skip(high_freq_start).sum();
        let total_power: f64 = spectral_power.iter().sum();

        if total_power > constants::math::EPSILON {
            (high_freq_power / total_power).min(1.0)
        } else {
            1.0
        }
    }

    /// Calculate phase coherence
    fn calculate_phase_coherence(&self, spectral_power: &[f64]) -> f64 {
        if spectral_power.len() < 2 {
            return 0.0;
        }

        let mean = spectral_power.iter().sum::<f64>() / spectral_power.len() as f64;
        let variance = spectral_power.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / spectral_power.len() as f64;

        if variance > constants::math::EPSILON {
            1.0 / (1.0 + variance)
        } else {
            1.0
        }
    }

    /// Clear analysis cache
    pub fn clear_cache(&mut self) {
        self.analysis_cache.clear();
    }

    /// Get analysis history
    pub fn get_history(&self) -> &[(chrono::DateTime<chrono::Utc>, RegimeAnalysis)] {
        &self.history
    }

    /// Get cache statistics
    pub fn get_cache_stats(&self) -> (usize, usize) {
        (self.analysis_cache.len(), constants::DEFAULT_CIRCUIT_CACHE_SIZE)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_quantum_market_analyzer_creation() {
        let analyzer = QuantumMarketAnalyzer::new(3);
        assert_eq!(analyzer.qft_circuit.num_qubits(), 3);
        assert_eq!(analyzer.pattern_circuit.num_qubits(), 3);
    }

    #[tokio::test]
    async fn test_regime_analysis() {
        let analyzer = QuantumMarketAnalyzer::new(3);
        
        let mut factors = FactorMap::new();
        factors.set(StandardFactors::Momentum, 0.8).unwrap();
        factors.set(StandardFactors::Volume, 0.6).unwrap();
        factors.set(StandardFactors::Volatility, 0.4).unwrap();

        let result = analyzer.analyze_regime(&factors).await;
        assert!(result.is_ok());
        
        let analysis = result.unwrap();
        assert!(analysis.confidence >= 0.0 && analysis.confidence <= 1.0);
        assert!(analysis.strength >= 0.0 && analysis.strength <= 1.0);
        assert!(!analysis.spectral_power.is_empty());
    }

    #[tokio::test]
    async fn test_pattern_detection() {
        let analyzer = QuantumMarketAnalyzer::new(3);
        
        let mut factors = FactorMap::new();
        factors.set(StandardFactors::Momentum, 0.9).unwrap();
        factors.set(StandardFactors::Volume, 0.8).unwrap();

        let result = analyzer.detect_patterns(&factors).await;
        assert!(result.is_ok());
        
        let patterns = result.unwrap();
        // Should return some patterns, even if empty is valid
        assert!(patterns.len() <= 10); // Reasonable upper bound
    }

    #[tokio::test]
    async fn test_direction_prediction() {
        let analyzer = QuantumMarketAnalyzer::new(3);
        
        let mut factors = FactorMap::new();
        factors.set(StandardFactors::Momentum, 0.7).unwrap();
        factors.set(StandardFactors::Trend, 0.6).unwrap();

        let result = analyzer.predict_direction(&factors).await;
        assert!(result.is_ok());
        
        let prediction = result.unwrap();
        assert!(prediction.direction >= -1.0 && prediction.direction <= 1.0);
        assert!(prediction.confidence >= 0.0 && prediction.confidence <= 1.0);
        assert!(prediction.time_horizon_ms > 0);
    }

    #[tokio::test]
    async fn test_volatility_calculation() {
        let analyzer = QuantumMarketAnalyzer::new(3);
        
        let mut factors = FactorMap::new();
        factors.set(StandardFactors::Volatility, 0.8).unwrap();
        factors.set(StandardFactors::Volume, 0.6).unwrap();

        let result = analyzer.calculate_volatility(&factors).await;
        assert!(result.is_ok());
        
        let volatility = result.unwrap();
        assert!(volatility >= 0.0 && volatility <= 1.0);
    }

    #[test]
    fn test_market_phase_determination() {
        let analyzer = QuantumMarketAnalyzer::new(3);
        
        // Test growth pattern (low frequency dominance)
        let growth_spectrum = vec![0.6, 0.3, 0.1];
        let phase = analyzer.determine_market_phase(&growth_spectrum);
        assert_eq!(phase, MarketPhase::Growth);
        
        // Test decline pattern (high frequency dominance)
        let decline_spectrum = vec![0.1, 0.2, 0.7];
        let phase = analyzer.determine_market_phase(&decline_spectrum);
        assert_eq!(phase, MarketPhase::Decline);
        
        // Test sideways pattern (balanced distribution)
        let sideways_spectrum = vec![0.3, 0.4, 0.3];
        let phase = analyzer.determine_market_phase(&sideways_spectrum);
        assert_eq!(phase, MarketPhase::Sideways);
    }

    #[test]
    fn test_trend_analysis() {
        let analyzer = QuantumMarketAnalyzer::new(3);
        
        // Strong upward trend (low frequency dominant)
        let upward_spectrum = vec![0.8, 0.15, 0.05];
        let (direction, strength) = analyzer.analyze_trend(&upward_spectrum);
        assert!(direction > 0.0); // Positive direction
        assert!(strength > 0.5); // Strong trend
        
        // Strong downward trend (high frequency dominant)
        let downward_spectrum = vec![0.05, 0.15, 0.8];
        let (direction, strength) = analyzer.analyze_trend(&downward_spectrum);
        assert!(direction < 0.0); // Negative direction
        assert!(strength > 0.5); // Strong trend
    }

    #[test]
    fn test_volatility_estimation() {
        let analyzer = QuantumMarketAnalyzer::new(3);
        
        // High volatility (high variance)
        let volatile_spectrum = vec![0.1, 0.8, 0.1];
        let volatility = analyzer.estimate_volatility(&volatile_spectrum);
        assert!(volatility > 0.2);
        
        // Low volatility (low variance)
        let stable_spectrum = vec![0.33, 0.34, 0.33];
        let volatility = analyzer.estimate_volatility(&stable_spectrum);
        assert!(volatility < 0.2);
    }
}