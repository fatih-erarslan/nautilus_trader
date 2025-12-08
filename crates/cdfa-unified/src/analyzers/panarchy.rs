//! # Panarchy Analyzer
//!
//! High-performance implementation of Panarchy adaptive cycle analysis for CDFA,
//! achieving sub-microsecond performance targets with four-phase cycle detection
//! and PCR (Potential, Connectedness, Resilience) analysis.
//!
//! This analyzer implements the four-phase adaptive cycle model:
//! - Growth (r): Exploitation of opportunities, increasing potential
//! - Conservation (K): Stability and efficiency, high connectedness
//! - Release (Ω): Creative destruction, sudden resilience loss
//! - Reorganization (α): Innovation and renewal, rebuilding resilience

use std::collections::HashMap;
// Arc available through std if needed
use std::time::Instant;

// NDArray prelude available through types
use num_traits::Float;
#[cfg(feature = "parallel")]
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

#[cfg(feature = "simd")]
use wide::f64x4;

#[cfg(feature = "gpu")]
use candle_core::{Device, Tensor};

use crate::error::{CdfaError, Result};
use crate::traits::SystemAnalyzer;
use crate::types::{Float as CdfaFloat, FloatArrayView1, FloatArrayView2};

/// Performance targets for sub-microsecond analysis
pub mod performance_targets {
    /// Target latency for PCR calculation (nanoseconds)
    pub const PCR_CALCULATION_TARGET_NS: u64 = 300;
    
    /// Target latency for phase classification (nanoseconds)
    pub const PHASE_CLASSIFICATION_TARGET_NS: u64 = 200;
    
    /// Target latency for regime score calculation (nanoseconds)
    pub const REGIME_SCORE_TARGET_NS: u64 = 150;
    
    /// Target latency for full analysis (nanoseconds)
    pub const FULL_ANALYSIS_TARGET_NS: u64 = 800;
}

/// The four phases of Panarchy adaptive cycles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PanarchyPhase {
    /// Growth (r) phase - exploitation of opportunities
    Growth,
    /// Conservation (K) phase - stability and efficiency
    Conservation,
    /// Release (Ω) phase - creative destruction
    Release,
    /// Reorganization (α) phase - innovation and renewal
    Reorganization,
    /// Unknown phase when classification fails
    Unknown,
}

impl PanarchyPhase {
    /// Convert phase to numeric score (0.0-1.0)
    pub fn to_score(&self) -> CdfaFloat {
        match self {
            Self::Growth => 0.25,
            Self::Conservation => 0.50,
            Self::Release => 0.75,
            Self::Reorganization => 0.90,
            Self::Unknown => 0.50,
        }
    }
    
    /// Get phase name as string
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Growth => "growth",
            Self::Conservation => "conservation", 
            Self::Release => "release",
            Self::Reorganization => "reorganization",
            Self::Unknown => "unknown",
        }
    }
}

impl std::fmt::Display for PanarchyPhase {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// PCR (Potential, Connectedness, Resilience) components
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PCRComponents {
    /// Potential (P): Capacity for growth/change, normalized price position
    pub potential: CdfaFloat,
    /// Connectedness (C): Internal connections/rigidity via autocorrelation
    pub connectedness: CdfaFloat,
    /// Resilience (R): Ability to withstand disturbance, inverse volatility
    pub resilience: CdfaFloat,
}

impl PCRComponents {
    /// Create new PCR components
    pub fn new(potential: CdfaFloat, connectedness: CdfaFloat, resilience: CdfaFloat) -> Self {
        Self {
            potential,
            connectedness,
            resilience,
        }
    }
    
    /// Calculate composite PCR score
    pub fn composite_score(&self) -> CdfaFloat {
        (self.potential + self.connectedness + self.resilience) / 3.0
    }
    
    /// Validate PCR components are within expected ranges
    pub fn validate(&self) -> Result<()> {
        if !self.potential.is_finite() || self.potential < 0.0 || self.potential > 1.0 {
            return Err(CdfaError::invalid_input("Potential must be between 0 and 1"));
        }
        if !self.connectedness.is_finite() || self.connectedness < 0.0 || self.connectedness > 1.0 {
            return Err(CdfaError::invalid_input("Connectedness must be between 0 and 1"));
        }
        if !self.resilience.is_finite() || self.resilience < 0.0 || self.resilience > 1.0 {
            return Err(CdfaError::invalid_input("Resilience must be between 0 and 1"));
        }
        Ok(())
    }
}

/// Panarchy analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanarchyResult {
    /// Current phase
    pub phase: PanarchyPhase,
    /// Phase confidence score (0.0-1.0)
    pub confidence: CdfaFloat,
    /// PCR components
    pub pcr: PCRComponents,
    /// Phase scores for all phases
    pub phase_scores: HashMap<String, CdfaFloat>,
    /// Transition probability to next phase
    pub transition_probability: CdfaFloat,
    /// Analysis computation time in nanoseconds
    pub computation_time_ns: u64,
}

/// Configuration parameters for Panarchy analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanarchyConfig {
    /// Period for rolling calculations
    pub window_size: usize,
    /// Lag for autocorrelation calculation
    pub autocorr_lag: usize,
    /// Minimum confidence threshold for phase classification
    pub min_confidence: CdfaFloat,
    /// Hysteresis threshold to prevent phase oscillation
    pub hysteresis_threshold: CdfaFloat,
    /// Enable SIMD optimization
    pub enable_simd: bool,
    /// Enable GPU acceleration
    pub enable_gpu: bool,
    /// Enable parallel processing
    pub enable_parallel: bool,
}

impl Default for PanarchyConfig {
    fn default() -> Self {
        Self {
            window_size: 20,
            autocorr_lag: 1,
            min_confidence: 0.6,
            hysteresis_threshold: 0.1,
            enable_simd: true,
            enable_gpu: false,
            enable_parallel: true,
        }
    }
}

/// High-performance Panarchy analyzer implementing sub-microsecond analysis
pub struct PanarchyAnalyzer {
    config: PanarchyConfig,
    previous_phase: Option<PanarchyPhase>,
    phase_history: Vec<PanarchyPhase>,
    #[cfg(feature = "gpu")]
    device: Option<Device>,
}

impl PanarchyAnalyzer {
    /// Create new Panarchy analyzer with default configuration
    pub fn new() -> Self {
        Self::with_config(PanarchyConfig::default())
    }
    
    /// Create new analyzer with custom configuration
    pub fn with_config(config: PanarchyConfig) -> Self {
        Self {
            config,
            previous_phase: None,
            phase_history: Vec::new(),
            #[cfg(feature = "gpu")]
            device: if config.enable_gpu {
                Device::cuda_if_available(0).ok()
            } else {
                None
            },
        }
    }
    
    /// Perform full Panarchy analysis on market data
    pub fn analyze_full(&mut self, prices: &[CdfaFloat], volumes: &[CdfaFloat]) -> Result<PanarchyResult> {
        let start_time = Instant::now();
        
        // Validate input data
        self.validate_input(prices, volumes)?;
        
        // Calculate returns for analysis
        let returns = self.calculate_returns(prices)?;
        
        // Calculate PCR components with <300ns target
        let pcr = self.calculate_pcr_fast(prices, &returns, volumes)?;
        
        // Identify current phase with <200ns target
        let (phase, confidence, phase_scores) = self.identify_phase_fast(&pcr, &returns)?;
        
        // Calculate transition probability
        let transition_probability = self.calculate_transition_probability(&pcr, phase)?;
        
        // Update phase history
        self.update_phase_history(phase);
        
        let computation_time_ns = start_time.elapsed().as_nanos() as u64;
        
        Ok(PanarchyResult {
            phase,
            confidence,
            pcr,
            phase_scores,
            transition_probability,
            computation_time_ns,
        })
    }
    
    /// Calculate PCR components with ultra-fast performance (<300ns target)
    fn calculate_pcr_fast(&self, prices: &[CdfaFloat], returns: &[CdfaFloat], volumes: &[CdfaFloat]) -> Result<PCRComponents> {
        let n = prices.len();
        let window = self.config.window_size.min(n);
        
        if window < 3 {
            return Ok(PCRComponents::new(0.5, 0.5, 0.5));
        }
        
        // Use latest data window for calculation
        let start_idx = n.saturating_sub(window);
        let price_window = &prices[start_idx..];
        let return_window = &returns[start_idx.saturating_sub(1)..];
        let volume_window = &volumes[start_idx..];
        
        #[cfg(feature = "simd")]
        {
            if self.config.enable_simd && price_window.len() >= 4 {
                return self.calculate_pcr_simd(price_window, return_window, volume_window);
            }
        }
        
        #[cfg(feature = "gpu")]
        {
            if self.config.enable_gpu && self.device.is_some() {
                return self.calculate_pcr_gpu(price_window, return_window, volume_window);
            }
        }
        
        // Fallback to scalar calculation
        self.calculate_pcr_scalar(price_window, return_window, volume_window)
    }
    
    /// Scalar PCR calculation
    fn calculate_pcr_scalar(&self, prices: &[CdfaFloat], returns: &[CdfaFloat], volumes: &[CdfaFloat]) -> Result<PCRComponents> {
        let n = prices.len();
        
        // Potential: Normalized position in price range
        let price_min = prices.iter().fold(CdfaFloat::INFINITY, |a, &b| a.min(b));
        let price_max = prices.iter().fold(CdfaFloat::NEG_INFINITY, |a, &b| a.max(b));
        let current_price = prices[n - 1];
        
        let potential = if price_max > price_min {
            (current_price - price_min) / (price_max - price_min)
        } else {
            0.5
        };
        
        // Connectedness: Autocorrelation of returns
        let connectedness = if returns.len() > self.config.autocorr_lag {
            self.calculate_autocorrelation(returns, self.config.autocorr_lag)?
        } else {
            0.5
        };
        
        // Resilience: Inverse of volatility (stability measure)
        let volatility = self.calculate_volatility(returns)?;
        let resilience = if volatility > 0.0 {
            1.0 / (1.0 + volatility)
        } else {
            1.0
        };
        
        let pcr = PCRComponents::new(
            potential.clamp(0.0, 1.0),
            connectedness.abs().clamp(0.0, 1.0),
            resilience.clamp(0.0, 1.0),
        );
        
        pcr.validate()?;
        Ok(pcr)
    }
    
    /// SIMD-optimized PCR calculation
    #[cfg(feature = "simd")]
    fn calculate_pcr_simd(&self, prices: &[CdfaFloat], returns: &[CdfaFloat], _volumes: &[CdfaFloat]) -> Result<PCRComponents> {
        let n = prices.len();
        let chunks = prices.chunks_exact(4);
        let remainder = chunks.remainder();
        
        // SIMD min/max calculation for potential
        let mut min_vals = f64x4::splat(CdfaFloat::INFINITY);
        let mut max_vals = f64x4::splat(CdfaFloat::NEG_INFINITY);
        
        for chunk in chunks {
            let vals = f64x4::new([chunk[0], chunk[1], chunk[2], chunk[3]]);
            min_vals = min_vals.min(vals);
            max_vals = max_vals.max(vals);
        }
        
        // Process remainder
        let mut price_min = min_vals.reduce_min();
        let mut price_max = max_vals.reduce_max();
        
        for &price in remainder {
            price_min = price_min.min(price);
            price_max = price_max.max(price);
        }
        
        let current_price = prices[n - 1];
        let potential = if price_max > price_min {
            (current_price - price_min) / (price_max - price_min)
        } else {
            0.5
        };
        
        // Connectedness and resilience using scalar fallback for simplicity
        let connectedness = if returns.len() > self.config.autocorr_lag {
            self.calculate_autocorrelation(returns, self.config.autocorr_lag)?.abs().clamp(0.0, 1.0)
        } else {
            0.5
        };
        
        let volatility = self.calculate_volatility(returns)?;
        let resilience = if volatility > 0.0 {
            (1.0 / (1.0 + volatility)).clamp(0.0, 1.0)
        } else {
            1.0
        };
        
        let pcr = PCRComponents::new(
            potential.clamp(0.0, 1.0),
            connectedness,
            resilience,
        );
        
        pcr.validate()?;
        Ok(pcr)
    }
    
    /// GPU-accelerated PCR calculation
    #[cfg(feature = "gpu")]
    fn calculate_pcr_gpu(&self, prices: &[CdfaFloat], returns: &[CdfaFloat], _volumes: &[CdfaFloat]) -> Result<PCRComponents> {
        if let Some(ref device) = self.device {
            // Convert to tensors
            let price_tensor = Tensor::from_slice(prices, prices.len(), device)
                .map_err(|e| CdfaError::computation_error(format!("GPU tensor creation failed: {}", e)))?;
            
            // Calculate min/max for potential
            let price_min = price_tensor.min(0)
                .map_err(|e| CdfaError::computation_error(format!("GPU min calculation failed: {}", e)))?
                .to_scalar::<f64>()
                .map_err(|e| CdfaError::computation_error(format!("GPU scalar conversion failed: {}", e)))?;
                
            let price_max = price_tensor.max(0)
                .map_err(|e| CdfaError::computation_error(format!("GPU max calculation failed: {}", e)))?
                .to_scalar::<f64>()
                .map_err(|e| CdfaError::computation_error(format!("GPU scalar conversion failed: {}", e)))?;
            
            let current_price = prices[prices.len() - 1];
            let potential = if price_max > price_min {
                (current_price - price_min) / (price_max - price_min)
            } else {
                0.5
            };
            
            // Fallback to CPU for connectedness and resilience
            let connectedness = if returns.len() > self.config.autocorr_lag {
                self.calculate_autocorrelation(returns, self.config.autocorr_lag)?.abs().clamp(0.0, 1.0)
            } else {
                0.5
            };
            
            let volatility = self.calculate_volatility(returns)?;
            let resilience = if volatility > 0.0 {
                (1.0 / (1.0 + volatility)).clamp(0.0, 1.0)
            } else {
                1.0
            };
            
            let pcr = PCRComponents::new(
                potential.clamp(0.0, 1.0),
                connectedness,
                resilience,
            );
            
            pcr.validate()?;
            Ok(pcr)
        } else {
            // Fallback to scalar calculation
            self.calculate_pcr_scalar(prices, returns, _volumes)
        }
    }
    
    /// Fast phase identification with <200ns target
    fn identify_phase_fast(&self, pcr: &PCRComponents, returns: &[CdfaFloat]) -> Result<(PanarchyPhase, CdfaFloat, HashMap<String, CdfaFloat>)> {
        let mut phase_scores = HashMap::new();
        
        // Calculate phase scores based on PCR components
        let growth_score = pcr.potential * 0.6 + (1.0 - pcr.connectedness) * 0.4;
        let conservation_score = pcr.connectedness * 0.7 + pcr.resilience * 0.3;
        let release_score = (1.0 - pcr.resilience) * 0.8 + pcr.potential * 0.2;
        let reorganization_score = (1.0 - pcr.potential) * 0.5 + (1.0 - pcr.connectedness) * 0.3 + pcr.resilience * 0.2;
        
        phase_scores.insert("growth".to_string(), growth_score);
        phase_scores.insert("conservation".to_string(), conservation_score);
        phase_scores.insert("release".to_string(), release_score);
        phase_scores.insert("reorganization".to_string(), reorganization_score);
        
        // Find phase with highest score
        let (best_phase_name, &max_score) = phase_scores
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
            .unwrap();
        
        let phase = match best_phase_name.as_str() {
            "growth" => PanarchyPhase::Growth,
            "conservation" => PanarchyPhase::Conservation,
            "release" => PanarchyPhase::Release,
            "reorganization" => PanarchyPhase::Reorganization,
            _ => PanarchyPhase::Unknown,
        };
        
        // Apply hysteresis if we have previous phase
        let final_phase = if let Some(prev_phase) = self.previous_phase {
            self.apply_hysteresis(phase, prev_phase, max_score)?
        } else {
            phase
        };
        
        // Calculate confidence based on score separation
        let scores: Vec<CdfaFloat> = phase_scores.values().copied().collect();
        let confidence = self.calculate_phase_confidence(&scores, max_score);
        
        Ok((final_phase, confidence, phase_scores))
    }
    
    /// Apply hysteresis to prevent phase oscillation
    fn apply_hysteresis(&self, new_phase: PanarchyPhase, prev_phase: PanarchyPhase, score: CdfaFloat) -> Result<PanarchyPhase> {
        if new_phase == prev_phase {
            return Ok(new_phase);
        }
        
        // Require higher confidence to change phases
        let threshold = self.config.min_confidence + self.config.hysteresis_threshold;
        
        if score >= threshold {
            Ok(new_phase)
        } else {
            Ok(prev_phase)
        }
    }
    
    /// Calculate phase confidence based on score separation
    fn calculate_phase_confidence(&self, scores: &[CdfaFloat], max_score: CdfaFloat) -> CdfaFloat {
        if scores.len() < 2 {
            return 0.5;
        }
        
        let mut sorted_scores = scores.to_vec();
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
        
        let second_best = sorted_scores[1];
        let separation = max_score - second_best;
        
        // Normalize confidence between 0 and 1
        (separation * 2.0).clamp(0.0, 1.0)
    }
    
    /// Calculate transition probability to next phase
    fn calculate_transition_probability(&self, pcr: &PCRComponents, current_phase: PanarchyPhase) -> Result<CdfaFloat> {
        // Transition probabilities based on PCR characteristics and current phase
        let transition_prob = match current_phase {
            PanarchyPhase::Growth => {
                // Growth to Conservation: high connectedness signals maturity
                pcr.connectedness * 0.8 + (1.0 - pcr.potential) * 0.2
            }
            PanarchyPhase::Conservation => {
                // Conservation to Release: low resilience signals instability
                (1.0 - pcr.resilience) * 0.9 + pcr.potential * 0.1
            }
            PanarchyPhase::Release => {
                // Release to Reorganization: natural progression after destruction
                0.8 // High probability as release naturally leads to reorganization
            }
            PanarchyPhase::Reorganization => {
                // Reorganization to Growth: new potential emerging
                pcr.potential * 0.7 + (1.0 - pcr.connectedness) * 0.3
            }
            PanarchyPhase::Unknown => 0.5,
        };
        
        Ok(transition_prob.clamp(0.0, 1.0))
    }
    
    /// Update phase history for tracking
    fn update_phase_history(&mut self, phase: PanarchyPhase) {
        self.previous_phase = Some(phase);
        self.phase_history.push(phase);
        
        // Keep history limited to prevent memory growth
        if self.phase_history.len() > 100 {
            self.phase_history.remove(0);
        }
    }
    
    /// Calculate autocorrelation for connectedness measure
    fn calculate_autocorrelation(&self, data: &[CdfaFloat], lag: usize) -> Result<CdfaFloat> {
        let n = data.len();
        if n <= lag {
            return Ok(0.0);
        }
        
        let mean = data.iter().sum::<CdfaFloat>() / n as CdfaFloat;
        
        let mut numerator = 0.0;
        let mut denominator = 0.0;
        
        for i in 0..(n - lag) {
            let x_dev = data[i] - mean;
            let y_dev = data[i + lag] - mean;
            numerator += x_dev * y_dev;
            denominator += x_dev * x_dev;
        }
        
        if denominator.abs() < CdfaFloat::EPSILON {
            Ok(0.0)
        } else {
            Ok(numerator / denominator)
        }
    }
    
    /// Calculate volatility for resilience measure
    fn calculate_volatility(&self, returns: &[CdfaFloat]) -> Result<CdfaFloat> {
        if returns.is_empty() {
            return Ok(0.0);
        }
        
        let mean = returns.iter().sum::<CdfaFloat>() / returns.len() as CdfaFloat;
        let variance = returns.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<CdfaFloat>() / returns.len() as CdfaFloat;
            
        Ok(variance.sqrt())
    }
    
    /// Calculate returns from price series
    fn calculate_returns(&self, prices: &[CdfaFloat]) -> Result<Vec<CdfaFloat>> {
        if prices.len() < 2 {
            return Ok(Vec::new());
        }
        
        let mut returns = Vec::with_capacity(prices.len() - 1);
        for i in 1..prices.len() {
            if prices[i - 1] != 0.0 {
                returns.push((prices[i] - prices[i - 1]) / prices[i - 1]);
            } else {
                returns.push(0.0);
            }
        }
        
        Ok(returns)
    }
    
    /// Validate input data
    fn validate_input(&self, prices: &[CdfaFloat], volumes: &[CdfaFloat]) -> Result<()> {
        if prices.is_empty() {
            return Err(CdfaError::invalid_input("Prices cannot be empty"));
        }
        
        if volumes.is_empty() {
            return Err(CdfaError::invalid_input("Volumes cannot be empty"));
        }
        
        if prices.len() != volumes.len() {
            return Err(CdfaError::invalid_input("Prices and volumes must have same length"));
        }
        
        for &price in prices {
            if !price.is_finite() || price <= 0.0 {
                return Err(CdfaError::invalid_input("All prices must be positive and finite"));
            }
        }
        
        for &volume in volumes {
            if !volume.is_finite() || volume < 0.0 {
                return Err(CdfaError::invalid_input("All volumes must be non-negative and finite"));
            }
        }
        
        Ok(())
    }
    
    /// Get current configuration
    pub fn config(&self) -> &PanarchyConfig {
        &self.config
    }
    
    /// Update configuration
    pub fn set_config(&mut self, config: PanarchyConfig) {
        self.config = config;
        #[cfg(feature = "gpu")]
        {
            self.device = if config.enable_gpu {
                Device::cuda_if_available(0).ok()
            } else {
                None
            };
        }
    }
    
    /// Get phase history
    pub fn phase_history(&self) -> &[PanarchyPhase] {
        &self.phase_history
    }
    
    /// Clear phase history
    pub fn clear_history(&mut self) {
        self.phase_history.clear();
        self.previous_phase = None;
    }
}

impl Default for PanarchyAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

impl SystemAnalyzer for PanarchyAnalyzer {
    fn analyze(&self, data: &FloatArrayView2, _scores: &FloatArrayView1) -> Result<HashMap<String, CdfaFloat>> {
        if data.nrows() < 2 || data.ncols() < 2 {
            return Err(CdfaError::invalid_input("Insufficient data for Panarchy analysis"));
        }
        
        // Extract prices and volumes from data matrix
        let prices: Vec<CdfaFloat> = data.column(0).to_vec();
        let volumes: Vec<CdfaFloat> = if data.ncols() > 1 {
            data.column(1).to_vec()
        } else {
            vec![1.0; prices.len()]
        };
        
        // Create a mutable copy for analysis
        let mut analyzer = PanarchyAnalyzer::with_config(self.config.clone());
        
        // Perform analysis
        let result = analyzer.analyze_full(&prices, &volumes)?;
        
        // Convert result to metrics
        let mut metrics = HashMap::new();
        metrics.insert("panarchy_phase_score".to_string(), result.phase.to_score());
        metrics.insert("panarchy_confidence".to_string(), result.confidence);
        metrics.insert("panarchy_potential".to_string(), result.pcr.potential);
        metrics.insert("panarchy_connectedness".to_string(), result.pcr.connectedness);
        metrics.insert("panarchy_resilience".to_string(), result.pcr.resilience);
        metrics.insert("panarchy_pcr_composite".to_string(), result.pcr.composite_score());
        metrics.insert("panarchy_transition_probability".to_string(), result.transition_probability);
        metrics.insert("panarchy_computation_time_ns".to_string(), result.computation_time_ns as CdfaFloat);
        
        // Add individual phase scores
        for (phase_name, score) in result.phase_scores {
            metrics.insert(format!("panarchy_phase_{}", phase_name), score);
        }
        
        Ok(metrics)
    }
    
    fn name(&self) -> &'static str {
        "PanarchyAnalyzer"
    }
    
    fn metric_names(&self) -> Vec<String> {
        vec![
            "panarchy_phase_score".to_string(),
            "panarchy_confidence".to_string(),
            "panarchy_potential".to_string(),
            "panarchy_connectedness".to_string(),
            "panarchy_resilience".to_string(),
            "panarchy_pcr_composite".to_string(),
            "panarchy_transition_probability".to_string(),
            "panarchy_computation_time_ns".to_string(),
            "panarchy_phase_growth".to_string(),
            "panarchy_phase_conservation".to_string(),
            "panarchy_phase_release".to_string(),
            "panarchy_phase_reorganization".to_string(),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Array1, Array2};
    use approx::assert_relative_eq;
    
    #[test]
    fn test_panarchy_analyzer_creation() {
        let analyzer = PanarchyAnalyzer::new();
        assert_eq!(analyzer.config.window_size, 20);
        assert_eq!(analyzer.config.autocorr_lag, 1);
    }
    
    #[test]
    fn test_pcr_components() {
        let pcr = PCRComponents::new(0.6, 0.7, 0.8);
        assert_relative_eq!(pcr.potential, 0.6, epsilon = 1e-10);
        assert_relative_eq!(pcr.connectedness, 0.7, epsilon = 1e-10);
        assert_relative_eq!(pcr.resilience, 0.8, epsilon = 1e-10);
        assert_relative_eq!(pcr.composite_score(), 0.7, epsilon = 1e-10);
        assert!(pcr.validate().is_ok());
    }
    
    #[test]
    fn test_panarchy_phase_scoring() {
        assert_relative_eq!(PanarchyPhase::Growth.to_score(), 0.25, epsilon = 1e-10);
        assert_relative_eq!(PanarchyPhase::Conservation.to_score(), 0.50, epsilon = 1e-10);
        assert_relative_eq!(PanarchyPhase::Release.to_score(), 0.75, epsilon = 1e-10);
        assert_relative_eq!(PanarchyPhase::Reorganization.to_score(), 0.90, epsilon = 1e-10);
    }
    
    #[test]
    fn test_simple_analysis() {
        let mut analyzer = PanarchyAnalyzer::new();
        let prices = vec![100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 104.0, 103.0, 102.0, 101.0,
                         100.0, 99.0, 98.0, 97.0, 96.0, 97.0, 98.0, 99.0, 100.0, 101.0, 102.0];
        let volumes = vec![1000.0; prices.len()];
        
        let result = analyzer.analyze_full(&prices, &volumes);
        assert!(result.is_ok());
        
        let result = result.unwrap();
        assert!(result.confidence >= 0.0 && result.confidence <= 1.0);
        assert!(result.pcr.potential >= 0.0 && result.pcr.potential <= 1.0);
        assert!(result.pcr.connectedness >= 0.0 && result.pcr.connectedness <= 1.0);
        assert!(result.pcr.resilience >= 0.0 && result.pcr.resilience <= 1.0);
        assert!(result.computation_time_ns > 0);
        assert!(result.computation_time_ns < performance_targets::FULL_ANALYSIS_TARGET_NS * 10); // Allow 10x margin for tests
    }
    
    #[test]
    fn test_system_analyzer_trait() {
        let analyzer = PanarchyAnalyzer::new();
        let data = Array2::from_shape_vec((21, 2), 
            (0..42).map(|i| if i % 2 == 0 { 100.0 + i as f64 } else { 1000.0 }).collect()
        ).unwrap();
        let scores = Array1::zeros(21);
        
        let result = analyzer.analyze(&data.view(), &scores.view());
        assert!(result.is_ok());
        
        let metrics = result.unwrap();
        assert!(metrics.contains_key("panarchy_phase_score"));
        assert!(metrics.contains_key("panarchy_confidence"));
        assert!(metrics.contains_key("panarchy_potential"));
        assert!(metrics.contains_key("panarchy_connectedness"));
        assert!(metrics.contains_key("panarchy_resilience"));
    }
    
    #[test]
    fn test_returns_calculation() {
        let analyzer = PanarchyAnalyzer::new();
        let prices = vec![100.0, 110.0, 105.0, 115.0];
        let returns = analyzer.calculate_returns(&prices).unwrap();
        
        assert_eq!(returns.len(), 3);
        assert_relative_eq!(returns[0], 0.1, epsilon = 1e-10); // (110-100)/100
        assert_relative_eq!(returns[1], -0.045454545454545456, epsilon = 1e-10); // (105-110)/110
        assert_relative_eq!(returns[2], 0.09523809523809523, epsilon = 1e-10); // (115-105)/105
    }
    
    #[test]
    fn test_autocorrelation_calculation() {
        let analyzer = PanarchyAnalyzer::new();
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 4.0, 3.0, 2.0, 1.0];
        let autocorr = analyzer.calculate_autocorrelation(&data, 1).unwrap();
        assert!(autocorr.abs() <= 1.0); // Autocorrelation should be between -1 and 1
    }
    
    #[test]
    fn test_volatility_calculation() {
        let analyzer = PanarchyAnalyzer::new();
        let returns = vec![0.01, -0.02, 0.015, -0.01, 0.005];
        let volatility = analyzer.calculate_volatility(&returns).unwrap();
        assert!(volatility >= 0.0);
    }
    
    #[test]
    fn test_input_validation() {
        let analyzer = PanarchyAnalyzer::new();
        
        // Empty data
        assert!(analyzer.validate_input(&[], &[]).is_err());
        
        // Mismatched lengths
        assert!(analyzer.validate_input(&[1.0, 2.0], &[1.0]).is_err());
        
        // Invalid prices
        assert!(analyzer.validate_input(&[0.0, 1.0], &[1.0, 1.0]).is_err());
        assert!(analyzer.validate_input(&[f64::NAN, 1.0], &[1.0, 1.0]).is_err());
        
        // Valid data
        assert!(analyzer.validate_input(&[1.0, 2.0], &[1.0, 1.0]).is_ok());
    }
    
    #[test]
    fn test_phase_hysteresis() {
        let analyzer = PanarchyAnalyzer::new();
        
        // Same phase should remain
        let result = analyzer.apply_hysteresis(
            PanarchyPhase::Growth, 
            PanarchyPhase::Growth, 
            0.8
        ).unwrap();
        assert_eq!(result, PanarchyPhase::Growth);
        
        // Different phase with low score should stay previous
        let result = analyzer.apply_hysteresis(
            PanarchyPhase::Conservation, 
            PanarchyPhase::Growth, 
            0.5
        ).unwrap();
        assert_eq!(result, PanarchyPhase::Growth);
        
        // Different phase with high score should change
        let result = analyzer.apply_hysteresis(
            PanarchyPhase::Conservation, 
            PanarchyPhase::Growth, 
            0.9
        ).unwrap();
        assert_eq!(result, PanarchyPhase::Conservation);
    }
}