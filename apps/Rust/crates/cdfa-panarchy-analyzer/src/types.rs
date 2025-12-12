//! Core types for Panarchy analysis

use serde::{Deserialize, Serialize};
use std::fmt;

/// The four phases of Panarchy adaptive cycles
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MarketPhase {
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

impl MarketPhase {
    /// Convert from string representation
    pub fn from_string(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "growth" => Self::Growth,
            "conservation" => Self::Conservation,
            "release" => Self::Release,
            "reorganization" => Self::Reorganization,
            _ => Self::Unknown,
        }
    }
    
    /// Get numeric score for phase (0.0-1.0)
    pub fn to_score(&self) -> f64 {
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

impl fmt::Display for MarketPhase {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.as_str())
    }
}

/// Parameters for Panarchy calculations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanarchyParameters {
    /// Lag for autocorrelation calculation
    pub autocorr_lag: usize,
    /// Period for ADX calculation
    pub adx_period: usize,
    /// Number of regimes to identify
    pub n_regimes: usize,
    /// Weight for momentum in potential calculation
    pub p_momentum_weight: f64,
    /// Window for regime smoothing
    pub regime_smoothing_window: usize,
    /// Minimum score threshold for hysteresis
    pub hysteresis_min_score_threshold: f64,
    /// Minimum score difference for phase change
    pub hysteresis_min_score_diff: f64,
    
    /// Weights for growth phase indicators
    pub weights_growth: PhaseWeights,
    /// Weights for conservation phase indicators
    pub weights_conservation: PhaseWeights,
    /// Weights for release phase indicators
    pub weights_release: PhaseWeights,
    /// Weights for reorganization phase indicators
    pub weights_reorganization: PhaseWeights,
}

impl Default for PanarchyParameters {
    fn default() -> Self {
        Self {
            autocorr_lag: 1,
            adx_period: 14,
            n_regimes: 4,
            p_momentum_weight: 0.5,
            regime_smoothing_window: 3,
            hysteresis_min_score_threshold: 0.35,
            hysteresis_min_score_diff: 0.10,
            weights_growth: PhaseWeights::growth_default(),
            weights_conservation: PhaseWeights::conservation_default(),
            weights_release: PhaseWeights::release_default(),
            weights_reorganization: PhaseWeights::reorganization_default(),
        }
    }
}

/// Weights for phase determination
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseWeights {
    pub primary: f64,
    pub secondary: f64,
    pub tertiary: f64,
}

impl PhaseWeights {
    pub fn growth_default() -> Self {
        Self {
            primary: 0.4,    // r_high_c_low
            secondary: 0.4,  // momentum_pos
            tertiary: 0.2,   // potential_rising
        }
    }
    
    pub fn conservation_default() -> Self {
        Self {
            primary: 0.5,    // p_high_c_high_r_low
            secondary: 0.3,  // momentum_stable
            tertiary: 0.2,   // potential_stable
        }
    }
    
    pub fn release_default() -> Self {
        Self {
            primary: 0.3,    // r_low_c_high
            secondary: 0.4,  // momentum_neg
            tertiary: 0.3,   // potential_falling
        }
    }
    
    pub fn reorganization_default() -> Self {
        Self {
            primary: 0.4,    // p_low_c_low_r_high
            secondary: 0.4,  // momentum_improving
            tertiary: 0.2,   // potential_low
        }
    }
}

/// PCR (Potential, Connectedness, Resilience) components
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PCRComponents {
    /// Potential - capacity for growth/change (0.0-1.0)
    pub potential: f64,
    /// Connectedness - internal connections/rigidity (0.0-1.0)
    pub connectedness: f64,
    /// Resilience - ability to withstand disturbance (0.0-1.0)
    pub resilience: f64,
}

impl PCRComponents {
    /// Create new PCR components with validation
    pub fn new(potential: f64, connectedness: f64, resilience: f64) -> Self {
        Self {
            potential: potential.clamp(0.0, 1.0),
            connectedness: connectedness.clamp(0.0, 1.0),
            resilience: resilience.clamp(0.0, 1.0),
        }
    }
    
    /// Calculate phase likelihood scores
    pub fn phase_scores(&self) -> PhaseScores {
        let p = self.potential;
        let c = self.connectedness;
        let r = self.resilience;
        
        PhaseScores {
            growth: (p * (1.1 - c) * r).max(0.0),
            conservation: (p * c * (1.1 - r)).max(0.0),
            release: ((1.1 - p) * c * (1.1 - r)).max(0.0),
            reorganization: ((1.1 - p) * (1.1 - c) * r).max(0.0),
        }
    }
}

/// Phase likelihood scores
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct PhaseScores {
    pub growth: f64,
    pub conservation: f64,
    pub release: f64,
    pub reorganization: f64,
}

impl PhaseScores {
    /// Normalize scores to sum to 1.0
    pub fn normalize(&mut self) {
        let total = self.growth + self.conservation + self.release + self.reorganization;
        if total > 0.0 {
            self.growth /= total;
            self.conservation /= total;
            self.release /= total;
            self.reorganization /= total;
        }
    }
    
    /// Get the dominant phase
    pub fn dominant_phase(&self) -> MarketPhase {
        let mut max_score = self.growth;
        let mut phase = MarketPhase::Growth;
        
        if self.conservation > max_score {
            max_score = self.conservation;
            phase = MarketPhase::Conservation;
        }
        if self.release > max_score {
            max_score = self.release;
            phase = MarketPhase::Release;
        }
        if self.reorganization > max_score {
            phase = MarketPhase::Reorganization;
        }
        
        phase
    }
    
    /// Calculate regime score (0.0-1.0)
    pub fn regime_score(&self) -> f64 {
        (self.reorganization * 0.10) +
        (self.growth * 0.35) +
        (self.conservation * 0.65) +
        (self.release * 0.90)
    }
}

/// Result of Panarchy analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanarchyResult {
    /// Current market phase
    pub phase: MarketPhase,
    /// PCR components
    pub pcr: PCRComponents,
    /// Phase likelihood scores
    pub phase_scores: PhaseScores,
    /// Regime score (0-100)
    pub regime_score: f64,
    /// Analysis confidence (0.0-1.0)
    pub confidence: f64,
    /// Signal strength (0.0-1.0)
    pub signal: f64,
    /// Number of data points analyzed
    pub data_points: usize,
    /// Computation time in nanoseconds
    pub computation_time_ns: u64,
}

/// Market data point for analysis
#[derive(Debug, Clone, Copy)]
pub struct MarketData {
    pub open: f64,
    pub high: f64,
    pub low: f64,
    pub close: f64,
    pub volume: f64,
}

/// Configuration for regime score calculation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegimeScoreConfig {
    /// Weights for different components
    pub panarchy_weight: f64,
    pub soc_weight: f64,
    pub volatility_weight: f64,
    pub fragility_weight: f64,
    pub adx_weight: f64,
}

impl Default for RegimeScoreConfig {
    fn default() -> Self {
        Self {
            panarchy_weight: 0.30,
            soc_weight: 0.25,
            volatility_weight: 0.25,
            fragility_weight: 0.10,
            adx_weight: 0.10,
        }
    }
}

/// Performance targets in nanoseconds
pub mod performance {
    /// Target for PCR calculation
    pub const PCR_CALCULATION_TARGET_NS: u64 = 300;
    /// Target for phase classification
    pub const PHASE_CLASSIFICATION_TARGET_NS: u64 = 200;
    /// Target for regime score calculation
    pub const REGIME_SCORE_TARGET_NS: u64 = 150;
    /// Target for full analysis
    pub const FULL_ANALYSIS_TARGET_NS: u64 = 800;
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_market_phase_conversion() {
        assert_eq!(MarketPhase::from_string("growth"), MarketPhase::Growth);
        assert_eq!(MarketPhase::from_string("CONSERVATION"), MarketPhase::Conservation);
        assert_eq!(MarketPhase::from_string("unknown"), MarketPhase::Unknown);
    }
    
    #[test]
    fn test_pcr_components_validation() {
        let pcr = PCRComponents::new(1.5, -0.5, 0.5);
        assert_eq!(pcr.potential, 1.0);
        assert_eq!(pcr.connectedness, 0.0);
        assert_eq!(pcr.resilience, 0.5);
    }
    
    #[test]
    fn test_phase_scores_normalization() {
        let mut scores = PhaseScores {
            growth: 0.2,
            conservation: 0.3,
            release: 0.4,
            reorganization: 0.1,
        };
        scores.normalize();
        let total = scores.growth + scores.conservation + scores.release + scores.reorganization;
        assert!((total - 1.0).abs() < 1e-10);
    }
}