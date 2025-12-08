//! Type definitions for SOC analysis

use ndarray::Array1;
use serde::{Deserialize, Serialize};

/// Parameters for SOC calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SOCParameters {
    // Sample Entropy parameters
    pub sample_entropy_m: usize,
    pub sample_entropy_r: f64,
    pub sample_entropy_min_points: usize,
    
    // Entropy Rate parameters
    pub entropy_rate_lag: usize,
    pub n_bins: usize,
    
    // SOC regime thresholds
    pub critical_threshold_complexity: f64,
    pub critical_threshold_equilibrium: f64,
    pub critical_threshold_fragility: f64,
    
    pub stable_threshold_equilibrium: f64,
    pub stable_threshold_fragility: f64,
    pub stable_threshold_entropy: f64,
    
    pub unstable_threshold_equilibrium: f64,
    pub unstable_threshold_fragility: f64,
    pub unstable_threshold_entropy: f64,
}

impl Default for SOCParameters {
    fn default() -> Self {
        Self {
            // Sample Entropy params
            sample_entropy_m: 2,
            sample_entropy_r: 0.2,
            sample_entropy_min_points: 20,
            
            // Entropy Rate params
            entropy_rate_lag: 1,
            n_bins: 10,
            
            // SOC regime thresholds
            critical_threshold_complexity: 0.7,
            critical_threshold_equilibrium: 0.3,
            critical_threshold_fragility: 0.6,
            
            stable_threshold_equilibrium: 0.7,
            stable_threshold_fragility: 0.3,
            stable_threshold_entropy: 0.6,
            
            unstable_threshold_equilibrium: 0.3,
            unstable_threshold_fragility: 0.7,
            unstable_threshold_entropy: 0.7,
        }
    }
}

/// SOC regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SOCRegime {
    Critical,
    Stable, 
    Unstable,
    Unknown,
}

impl SOCRegime {
    pub fn as_str(&self) -> &'static str {
        match self {
            SOCRegime::Critical => "critical",
            SOCRegime::Stable => "stable",
            SOCRegime::Unstable => "unstable", 
            SOCRegime::Unknown => "unknown",
        }
    }
    
    pub fn from_str(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "critical" => SOCRegime::Critical,
            "stable" => SOCRegime::Stable,
            "unstable" => SOCRegime::Unstable,
            _ => SOCRegime::Unknown,
        }
    }
}

/// Represents an avalanche event in SOC analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvalancheEvent {
    pub start_index: usize,
    pub end_index: usize,
    pub magnitude: f64,
    pub duration: usize,
    pub intensity: f64,
}

/// SOC analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SOCResult {
    // Core metrics
    pub sample_entropy: f64,
    pub entropy_rate: f64,
    pub complexity_measure: f64,
    
    // Regime classification
    pub regime: SOCRegime,
    pub regime_confidence: f64,
    
    // Equilibrium and fragility
    pub equilibrium_score: f64,
    pub fragility_score: f64,
    
    // Avalanche detection
    pub avalanche_events: Vec<AvalancheEvent>,
    pub avalanche_frequency: f64,
    pub total_avalanche_magnitude: f64,
    
    // Performance metrics
    pub computation_time_ns: u64,
    
    // Time series of intermediate values (optional)
    pub entropy_series: Option<Array1<f64>>,
    pub regime_series: Option<Array1<i32>>,
}

impl SOCResult {
    pub fn new() -> Self {
        Self {
            sample_entropy: 0.0,
            entropy_rate: 0.0,
            complexity_measure: 0.0,
            regime: SOCRegime::Unknown,
            regime_confidence: 0.0,
            equilibrium_score: 0.0,
            fragility_score: 0.0,
            avalanche_events: Vec::new(),
            avalanche_frequency: 0.0,
            total_avalanche_magnitude: 0.0,
            computation_time_ns: 0,
            entropy_series: None,
            regime_series: None,
        }
    }
    
    pub fn is_critical(&self) -> bool {
        self.regime == SOCRegime::Critical
    }
    
    pub fn is_stable(&self) -> bool {
        self.regime == SOCRegime::Stable
    }
    
    pub fn is_unstable(&self) -> bool {
        self.regime == SOCRegime::Unstable
    }
    
    pub fn avalanche_count(&self) -> usize {
        self.avalanche_events.len()
    }
    
    pub fn average_avalanche_magnitude(&self) -> f64 {
        if self.avalanche_events.is_empty() {
            0.0
        } else {
            self.total_avalanche_magnitude / self.avalanche_events.len() as f64
        }
    }
}

impl Default for SOCResult {
    fn default() -> Self {
        Self::new()
    }
}

/// Performance metrics for SOC analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SOCPerformanceMetrics {
    pub sample_entropy_time_ns: u64,
    pub entropy_rate_time_ns: u64,
    pub regime_classification_time_ns: u64,
    pub avalanche_detection_time_ns: u64,
    pub total_time_ns: u64,
    pub throughput_ops_per_sec: f64,
    pub memory_usage_bytes: usize,
}

impl SOCPerformanceMetrics {
    pub fn meets_targets(&self) -> bool {
        use crate::perf::*;
        
        self.sample_entropy_time_ns <= SAMPLE_ENTROPY_TARGET_NS
            && self.entropy_rate_time_ns <= ENTROPY_RATE_TARGET_NS
            && self.regime_classification_time_ns <= REGIME_CLASSIFICATION_TARGET_NS
            && self.total_time_ns <= FULL_ANALYSIS_TARGET_NS
    }
}