//! Core types for antifragility analysis

use ndarray::Array1;
use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Result of volatility analysis
#[derive(Debug, Clone)]
pub struct VolatilityResult {
    /// Combined volatility estimate
    pub combined_vol: Array1<f64>,
    /// Volatility regime score (0-1)
    pub vol_regime: Array1<f64>,
    /// Smoothed volatility rate of change
    pub vol_roc_smoothed: Array1<f64>,
    /// Yang-Zhang volatility
    pub yz_volatility: Array1<f64>,
    /// GARCH-like volatility
    pub garch_volatility: Array1<f64>,
    /// Parkinson volatility
    pub parkinson_volatility: Array1<f64>,
    /// ATR-based volatility
    pub atr_volatility: Array1<f64>,
}

impl Default for VolatilityResult {
    fn default() -> Self {
        Self {
            combined_vol: Array1::zeros(0),
            vol_regime: Array1::zeros(0),
            vol_roc_smoothed: Array1::zeros(0),
            yz_volatility: Array1::zeros(0),
            garch_volatility: Array1::zeros(0),
            parkinson_volatility: Array1::zeros(0),
            atr_volatility: Array1::zeros(0),
        }
    }
}

/// Result of performance analysis
#[derive(Debug, Clone)]
pub struct PerformanceResult {
    /// Performance acceleration
    pub acceleration: Array1<f64>,
    /// Smoothed performance rate of change
    pub perf_roc_smoothed: Array1<f64>,
    /// Performance momentum
    pub momentum: Array1<f64>,
    /// Log performance returns
    pub log_perf_returns: Array1<f64>,
}

impl Default for PerformanceResult {
    fn default() -> Self {
        Self {
            acceleration: Array1::zeros(0),
            perf_roc_smoothed: Array1::zeros(0),
            momentum: Array1::zeros(0),
            log_perf_returns: Array1::zeros(0),
        }
    }
}

/// Complete antifragility analysis result
#[derive(Debug, Clone)]
pub struct AnalysisResult {
    /// Main antifragility index (0-1, higher = more antifragile)
    pub antifragility_index: f64,
    /// Fragility score (0-1, higher = more fragile)
    pub fragility_score: f64,
    /// Convexity component score
    pub convexity_score: f64,
    /// Asymmetry component score
    pub asymmetry_score: f64,
    /// Recovery velocity component score
    pub recovery_score: f64,
    /// Benefit ratio component score
    pub benefit_ratio_score: f64,
    /// Detailed volatility analysis
    pub volatility: VolatilityResult,
    /// Detailed performance analysis
    pub performance: PerformanceResult,
    /// Number of data points analyzed
    pub data_points: usize,
    /// Time taken for calculation
    pub calculation_time: Duration,
}

impl AnalysisResult {
    /// Get a summary of the analysis
    pub fn summary(&self) -> String {
        format!(
            "Antifragility Analysis Summary:\n\
             - Antifragility Index: {:.4}\n\
             - Fragility Score: {:.4}\n\
             - Convexity: {:.4}\n\
             - Asymmetry: {:.4}\n\
             - Recovery: {:.4}\n\
             - Benefit Ratio: {:.4}\n\
             - Data Points: {}\n\
             - Calculation Time: {:?}",
            self.antifragility_index,
            self.fragility_score,
            self.convexity_score,
            self.asymmetry_score,
            self.recovery_score,
            self.benefit_ratio_score,
            self.data_points,
            self.calculation_time
        )
    }
    
    /// Check if the system is antifragile
    pub fn is_antifragile(&self) -> bool {
        self.antifragility_index > 0.6
    }
    
    /// Check if the system is fragile
    pub fn is_fragile(&self) -> bool {
        self.fragility_score > 0.6
    }
    
    /// Check if the system is robust (neither fragile nor antifragile)
    pub fn is_robust(&self) -> bool {
        !self.is_antifragile() && !self.is_fragile()
    }
    
    /// Get the dominant characteristic
    pub fn dominant_characteristic(&self) -> &'static str {
        if self.is_antifragile() {
            "Antifragile"
        } else if self.is_fragile() {
            "Fragile"
        } else {
            "Robust"
        }
    }
}

/// Signal strength enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SignalStrength {
    VeryWeak,
    Weak,
    Moderate,
    Strong,
    VeryStrong,
}

impl From<f64> for SignalStrength {
    fn from(value: f64) -> Self {
        match value {
            x if x < 0.2 => SignalStrength::VeryWeak,
            x if x < 0.4 => SignalStrength::Weak,
            x if x < 0.6 => SignalStrength::Moderate,
            x if x < 0.8 => SignalStrength::Strong,
            _ => SignalStrength::VeryStrong,
        }
    }
}

/// Volatility regime classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VolatilityRegime {
    Low,
    Normal,
    High,
    Extreme,
}

impl From<f64> for VolatilityRegime {
    fn from(value: f64) -> Self {
        match value {
            x if x < 0.25 => VolatilityRegime::Low,
            x if x < 0.5 => VolatilityRegime::Normal,
            x if x < 0.75 => VolatilityRegime::High,
            _ => VolatilityRegime::Extreme,
        }
    }
}

/// Market phase classification
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MarketPhase {
    Bull,
    Bear,
    Sideways,
    Volatile,
}

/// Component analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentAnalysis {
    /// Raw component score
    pub raw_score: f64,
    /// Normalized component score (0-1)
    pub normalized_score: f64,
    /// Component weight in final calculation
    pub weight: f64,
    /// Weighted contribution to final score
    pub weighted_contribution: f64,
    /// Signal strength classification
    pub signal_strength: SignalStrength,
    /// Statistical confidence in the measurement
    pub confidence: f64,
}

impl ComponentAnalysis {
    pub fn new(raw_score: f64, weight: f64) -> Self {
        let normalized_score = raw_score.clamp(0.0, 1.0);
        let weighted_contribution = normalized_score * weight;
        let signal_strength = SignalStrength::from(normalized_score);
        
        Self {
            raw_score,
            normalized_score,
            weight,
            weighted_contribution,
            signal_strength,
            confidence: 0.8, // Default confidence
        }
    }
}

/// Detailed analysis result with component breakdown
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DetailedAnalysisResult {
    /// Basic analysis result
    pub basic: AnalysisResult,
    /// Detailed component analysis
    pub components: ComponentBreakdown,
    /// Market phase classification
    pub market_phase: MarketPhase,
    /// Volatility regime classification
    pub volatility_regime: VolatilityRegime,
    /// Statistical measures
    pub statistics: StatisticalMeasures,
}

/// Component breakdown for detailed analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentBreakdown {
    /// Convexity component analysis
    pub convexity: ComponentAnalysis,
    /// Asymmetry component analysis
    pub asymmetry: ComponentAnalysis,
    /// Recovery component analysis
    pub recovery: ComponentAnalysis,
    /// Benefit ratio component analysis
    pub benefit_ratio: ComponentAnalysis,
}

/// Statistical measures for analysis validation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatisticalMeasures {
    /// Mean of the antifragility index over time
    pub mean_antifragility: f64,
    /// Standard deviation of the antifragility index
    pub std_antifragility: f64,
    /// Skewness of the antifragility distribution
    pub skewness: f64,
    /// Kurtosis of the antifragility distribution
    pub kurtosis: f64,
    /// Sharpe ratio equivalent for antifragility
    pub antifragility_ratio: f64,
    /// Stability measure (1 - coefficient of variation)
    pub stability: f64,
}

/// Configuration for different analysis modes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalysisConfig {
    /// Whether to include detailed component breakdown
    pub include_components: bool,
    /// Whether to calculate statistical measures
    pub include_statistics: bool,
    /// Whether to classify market phases
    pub classify_market_phase: bool,
    /// Whether to use SIMD optimizations
    pub use_simd: bool,
    /// Whether to use parallel processing
    pub use_parallel: bool,
    /// Minimum confidence threshold for results
    pub min_confidence: f64,
}

impl Default for AnalysisConfig {
    fn default() -> Self {
        Self {
            include_components: true,
            include_statistics: true,
            classify_market_phase: true,
            use_simd: true,
            use_parallel: true,
            min_confidence: 0.7,
        }
    }
}

/// Real-time analysis state for streaming data
#[derive(Debug, Clone)]
pub struct RealTimeState {
    /// Recent price history
    pub price_buffer: Vec<f64>,
    /// Recent volume history
    pub volume_buffer: Vec<f64>,
    /// Rolling analysis results
    pub analysis_history: Vec<AnalysisResult>,
    /// Buffer size for analysis
    pub buffer_size: usize,
    /// Update frequency (number of new data points before reanalysis)
    pub update_frequency: usize,
    /// Current position in the update cycle
    pub update_counter: usize,
}

impl RealTimeState {
    pub fn new(buffer_size: usize, update_frequency: usize) -> Self {
        Self {
            price_buffer: Vec::with_capacity(buffer_size),
            volume_buffer: Vec::with_capacity(buffer_size),
            analysis_history: Vec::new(),
            buffer_size,
            update_frequency,
            update_counter: 0,
        }
    }
    
    pub fn add_data_point(&mut self, price: f64, volume: f64) {
        self.price_buffer.push(price);
        self.volume_buffer.push(volume);
        
        // Maintain buffer size
        if self.price_buffer.len() > self.buffer_size {
            self.price_buffer.remove(0);
            self.volume_buffer.remove(0);
        }
        
        self.update_counter += 1;
    }
    
    pub fn should_update(&self) -> bool {
        self.update_counter >= self.update_frequency
    }
    
    pub fn reset_counter(&mut self) {
        self.update_counter = 0;
    }
    
    pub fn is_ready(&self) -> bool {
        self.price_buffer.len() >= self.buffer_size / 2
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_signal_strength_conversion() {
        assert_eq!(SignalStrength::from(0.1), SignalStrength::VeryWeak);
        assert_eq!(SignalStrength::from(0.3), SignalStrength::Weak);
        assert_eq!(SignalStrength::from(0.5), SignalStrength::Moderate);
        assert_eq!(SignalStrength::from(0.7), SignalStrength::Strong);
        assert_eq!(SignalStrength::from(0.9), SignalStrength::VeryStrong);
    }
    
    #[test]
    fn test_volatility_regime_conversion() {
        assert_eq!(VolatilityRegime::from(0.1), VolatilityRegime::Low);
        assert_eq!(VolatilityRegime::from(0.3), VolatilityRegime::Normal);
        assert_eq!(VolatilityRegime::from(0.6), VolatilityRegime::High);
        assert_eq!(VolatilityRegime::from(0.8), VolatilityRegime::Extreme);
    }
    
    #[test]
    fn test_analysis_result_classification() {
        let mut result = AnalysisResult {
            antifragility_index: 0.7,
            fragility_score: 0.2,
            convexity_score: 0.6,
            asymmetry_score: 0.5,
            recovery_score: 0.8,
            benefit_ratio_score: 0.4,
            volatility: VolatilityResult::default(),
            performance: PerformanceResult::default(),
            data_points: 100,
            calculation_time: Duration::from_millis(1),
        };
        
        assert!(result.is_antifragile());
        assert!(!result.is_fragile());
        assert!(!result.is_robust());
        assert_eq!(result.dominant_characteristic(), "Antifragile");
        
        result.antifragility_index = 0.3;
        result.fragility_score = 0.7;
        
        assert!(!result.is_antifragile());
        assert!(result.is_fragile());
        assert!(!result.is_robust());
        assert_eq!(result.dominant_characteristic(), "Fragile");
    }
    
    #[test]
    fn test_component_analysis() {
        let component = ComponentAnalysis::new(0.75, 0.4);
        
        assert_eq!(component.raw_score, 0.75);
        assert_eq!(component.normalized_score, 0.75);
        assert_eq!(component.weight, 0.4);
        assert_eq!(component.weighted_contribution, 0.3);
        assert_eq!(component.signal_strength, SignalStrength::Strong);
    }
    
    #[test]
    fn test_real_time_state() {
        let mut state = RealTimeState::new(100, 10);
        
        // Add some data points
        for i in 0..50 {
            state.add_data_point(100.0 + i as f64, 1000.0);
        }
        
        assert!(state.is_ready());
        assert_eq!(state.price_buffer.len(), 50);
        assert_eq!(state.volume_buffer.len(), 50);
        
        // Add more data to test buffer management
        for i in 50..150 {
            state.add_data_point(100.0 + i as f64, 1000.0);
        }
        
        assert_eq!(state.price_buffer.len(), 100);
        assert_eq!(state.volume_buffer.len(), 100);
    }
}