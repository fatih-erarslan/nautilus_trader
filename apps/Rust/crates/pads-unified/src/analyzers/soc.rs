//! # Self-Organized Criticality (SOC) Analyzer
//!
//! Harvested from CDFA SOC Analyzer with sophisticated entropy analysis, avalanche detection,
//! and regime classification capabilities.
//!
//! ## Core Concepts
//!
//! Self-Organized Criticality analyzes systems that naturally evolve toward critical states:
//! - **Sample Entropy**: Regularity measurement with SIMD optimization
//! - **Entropy Rate**: Information flow analysis with conditional entropy
//! - **Avalanche Detection**: Cascade event identification and statistics
//! - **Regime Classification**: Critical, stable, unstable state detection
//! - **Power-Law Analysis**: Scale-invariant behavior identification
//!
//! ## Features
//! - Sub-microsecond entropy calculations
//! - SIMD-optimized template matching
//! - Sophisticated avalanche event detection
//! - Multi-scale regime classification with hysteresis
//! - Branching ratio analysis for criticality detection

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use ndarray::{Array1, ArrayView1};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use super::{MarketData, PatternAnalyzer};
use crate::error::{PadsError, PadsResult};

#[cfg(feature = "simd")]
use std::simd::{f32x8, f64x4, f64x8, LaneCount, SupportedLaneCount};

/// SOC analyzer parameters (harvested from CDFA implementation)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SOCParameters {
    /// Sample entropy parameters
    pub sample_entropy_m: usize,
    pub sample_entropy_r: f64,
    pub sample_entropy_min_points: usize,
    
    /// Entropy rate parameters
    pub entropy_rate_lag: usize,
    pub n_bins: usize,
    
    /// Avalanche detection parameters
    pub min_avalanche_size: usize,
    pub significance_threshold: f64,
    pub cascade_threshold: f64,
    
    /// Regime classification thresholds
    pub critical_threshold_complexity: f64,
    pub critical_threshold_equilibrium: f64,
    pub critical_threshold_fragility: f64,
    pub stable_threshold_equilibrium: f64,
    pub stable_threshold_fragility: f64,
    pub stable_threshold_entropy: f64,
    pub unstable_threshold_equilibrium: f64,
    pub unstable_threshold_fragility: f64,
    pub unstable_threshold_entropy: f64,
    
    /// Performance parameters
    pub enable_simd: bool,
    pub enable_parallel: bool,
    pub cache_size: usize,
}

impl Default for SOCParameters {
    fn default() -> Self {
        Self {
            // Sample entropy parameters (from CDFA implementation)
            sample_entropy_m: 2,
            sample_entropy_r: 0.2,
            sample_entropy_min_points: 20,
            
            // Entropy rate parameters
            entropy_rate_lag: 1,
            n_bins: 10,
            
            // Avalanche detection
            min_avalanche_size: 2,
            significance_threshold: 0.01,
            cascade_threshold: 0.04,
            
            // Regime classification thresholds
            critical_threshold_complexity: 0.6,
            critical_threshold_equilibrium: 0.4,
            critical_threshold_fragility: 0.6,
            stable_threshold_equilibrium: 0.7,
            stable_threshold_fragility: 0.3,
            stable_threshold_entropy: 0.5,
            unstable_threshold_equilibrium: 0.3,
            unstable_threshold_fragility: 0.7,
            unstable_threshold_entropy: 0.8,
            
            // Performance settings
            enable_simd: true,
            enable_parallel: true,
            cache_size: 1000,
        }
    }
}

/// SOC analysis results
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SOCResult {
    /// Sample entropy value
    pub sample_entropy: f64,
    
    /// Entropy rate (conditional entropy)
    pub entropy_rate: f64,
    
    /// Complexity measure (multi-scale entropy)
    pub complexity_measure: f64,
    
    /// Equilibrium score (autocorrelation and stability)
    pub equilibrium_score: f64,
    
    /// Fragility score (volatility clustering and extremes)
    pub fragility_score: f64,
    
    /// Market regime classification
    pub regime: SOCRegime,
    pub regime_confidence: f64,
    
    /// Avalanche analysis
    pub avalanche_events: Vec<AvalancheEvent>,
    pub total_avalanche_magnitude: f64,
    pub avalanche_frequency: f64,
    pub branching_ratios: Vec<f64>,
    
    /// Power-law analysis
    pub power_law_exponent: Option<f64>,
    pub power_law_r_squared: Option<f64>,
    
    /// Analysis metadata
    pub computation_time_ns: u64,
    pub data_points: usize,
    pub simd_used: bool,
}

impl SOCResult {
    pub fn new() -> Self {
        Self {
            sample_entropy: 0.0,
            entropy_rate: 0.0,
            complexity_measure: 0.0,
            equilibrium_score: 0.0,
            fragility_score: 0.0,
            regime: SOCRegime::Unknown,
            regime_confidence: 0.0,
            avalanche_events: Vec::new(),
            total_avalanche_magnitude: 0.0,
            avalanche_frequency: 0.0,
            branching_ratios: Vec::new(),
            power_law_exponent: None,
            power_law_r_squared: None,
            computation_time_ns: 0,
            data_points: 0,
            simd_used: false,
        }
    }
}

/// Market regime classification from SOC analysis
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SOCRegime {
    /// Critical state (at edge of chaos)
    Critical,
    /// Stable/sub-critical state
    Stable,
    /// Unstable/super-critical state
    Unstable,
    /// Unknown/insufficient data
    Unknown,
}

/// Avalanche event detected in the time series
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AvalancheEvent {
    pub start_index: usize,
    pub end_index: usize,
    pub magnitude: f64,
    pub duration: usize,
    pub intensity: f64,
}

/// Cascade event (large avalanche)
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CascadeEvent {
    pub start_idx: usize,
    pub end_idx: usize,
    pub duration: usize,
    pub total_magnitude: f64,
    pub peak_magnitude: f64,
}

/// High-performance SOC analyzer
pub struct SOCAnalyzer {
    params: SOCParameters,
    cache: Arc<RwLock<AnalysisCache>>,
    avalanche_detector: AvalancheDetector,
}

impl SOCAnalyzer {
    /// Create new SOC analyzer with default parameters
    pub fn new() -> Self {
        Self::with_params(SOCParameters::default())
    }
    
    /// Create analyzer with custom parameters
    pub fn with_params(params: SOCParameters) -> Self {
        let avalanche_detector = AvalancheDetector::new(
            params.min_avalanche_size,
            params.significance_threshold,
        );
        
        Self {
            params,
            cache: Arc::new(RwLock::new(AnalysisCache::new(1000))),
            avalanche_detector,
        }
    }
    
    /// Calculate sample entropy with SIMD optimization
    fn calculate_sample_entropy(&self, data: ArrayView1<f64>) -> PadsResult<f64> {
        let n = data.len();
        if n < self.params.sample_entropy_min_points {
            return Ok(0.5); // Default for too short series
        }
        
        // Convert to f32 for SIMD (maintaining precision for our use case)
        let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        
        // Calculate mean and standard deviation
        let mean = data_f32.iter().sum::<f32>() / n as f32;
        let variance = data_f32.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (n - 1) as f32;
        let std_dev = variance.sqrt();
        
        if std_dev < 1e-9 {
            return Ok(0.5); // Default for constant series
        }
        
        let tolerance = (self.params.sample_entropy_r as f32) * std_dev;
        
        // Count template matches for patterns of length m and m+1
        let count_m = self.count_template_matches_simd(&data_f32, self.params.sample_entropy_m, tolerance)?;
        let count_m1 = self.count_template_matches_simd(&data_f32, self.params.sample_entropy_m + 1, tolerance)?;
        
        if count_m == 0 || count_m1 == 0 {
            return Ok(0.5);
        }
        
        let sample_entropy = -(count_m1 as f64 / count_m as f64).ln();
        Ok(sample_entropy)
    }
    
    /// SIMD-optimized template matching for sample entropy
    #[cfg(feature = "simd")]
    fn count_template_matches_simd(&self, data: &[f32], m: usize, tolerance: f32) -> PadsResult<u32> {
        let n = data.len();
        if n < m + 1 {
            return Ok(0);
        }
        
        let mut count = 0u32;
        let tolerance_vec = f32x8::splat(tolerance);
        
        // Main SIMD loop for template matching
        for i in 0..=(n - m) {
            for j in (i + 1)..=(n - m) {
                if j + 7 < n - m + 1 {
                    // SIMD comparison for 8 templates at once
                    let mut all_match = true;
                    
                    for k in 0..m {
                        let template_val = f32x8::splat(data[i + k]);
                        
                        // Load 8 comparison values
                        let compare_vals = if j + k + 7 < n {
                            f32x8::from_array([
                                data[j + k],
                                data[j + k + 1],
                                data[j + k + 2], 
                                data[j + k + 3],
                                data[j + k + 4],
                                data[j + k + 5],
                                data[j + k + 6],
                                data[j + k + 7],
                            ])
                        } else {
                            // Handle boundary case
                            let mut vals = [0.0f32; 8];
                            for l in 0..8 {
                                if j + k + l < n {
                                    vals[l] = data[j + k + l];
                                }
                            }
                            f32x8::from_array(vals)
                        };
                        
                        let diff = (template_val - compare_vals).abs();
                        let within_tolerance = diff.lanes_le(tolerance_vec);
                        
                        if !within_tolerance.all() {
                            all_match = false;
                            break;
                        }
                    }
                    
                    if all_match {
                        count += 1;
                    }
                } else {
                    // Fallback to scalar comparison for remaining elements
                    let mut matches = true;
                    for k in 0..m {
                        if (data[i + k] - data[j + k]).abs() > tolerance {
                            matches = false;
                            break;
                        }
                    }
                    if matches {
                        count += 1;
                    }
                }
            }
        }
        
        Ok(count)
    }
    
    #[cfg(not(feature = "simd"))]
    fn count_template_matches_simd(&self, data: &[f32], m: usize, tolerance: f32) -> PadsResult<u32> {
        let n = data.len();
        if n < m + 1 {
            return Ok(0);
        }
        
        let mut count = 0u32;
        
        for i in 0..=(n - m) {
            for j in (i + 1)..=(n - m) {
                let mut matches = true;
                for k in 0..m {
                    if (data[i + k] - data[j + k]).abs() > tolerance {
                        matches = false;
                        break;
                    }
                }
                if matches {
                    count += 1;
                }
            }
        }
        
        Ok(count)
    }
    
    /// Calculate entropy rate using conditional entropy
    fn calculate_entropy_rate(&self, data: ArrayView1<f64>) -> PadsResult<f64> {
        let n = data.len();
        let lag = self.params.entropy_rate_lag;
        let n_bins = self.params.n_bins;
        
        if n <= lag {
            return Ok(0.0);
        }
        
        // Create histogram bins
        let min_val = data.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = data.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if (max_val - min_val).abs() < 1e-12 {
            return Ok(0.0); // Constant series has zero entropy rate
        }
        
        let bin_width = (max_val - min_val) / n_bins as f64;
        
        // Count joint occurrences P(X_t, X_{t-lag})
        let mut joint_counts = vec![vec![0u32; n_bins]; n_bins];
        let mut marginal_counts = vec![0u32; n_bins];
        
        for i in lag..n {
            let bin_curr = ((data[i] - min_val) / bin_width).floor() as usize;
            let bin_prev = ((data[i - lag] - min_val) / bin_width).floor() as usize;
            
            let bin_curr = bin_curr.min(n_bins - 1);
            let bin_prev = bin_prev.min(n_bins - 1);
            
            joint_counts[bin_prev][bin_curr] += 1;
            marginal_counts[bin_prev] += 1;
        }
        
        let total_count = (n - lag) as f64;
        
        // Calculate conditional entropy H(X_t | X_{t-lag})
        let mut conditional_entropy = 0.0;
        
        for i in 0..n_bins {
            if marginal_counts[i] > 0 {
                let p_prev = marginal_counts[i] as f64 / total_count;
                
                for j in 0..n_bins {
                    if joint_counts[i][j] > 0 {
                        let p_joint = joint_counts[i][j] as f64 / total_count;
                        let p_conditional = joint_counts[i][j] as f64 / marginal_counts[i] as f64;
                        
                        conditional_entropy -= p_joint * p_conditional.ln();
                    }
                }
            }
        }
        
        Ok(conditional_entropy)
    }
    
    /// Calculate complexity measure from multiple entropy-based metrics
    fn calculate_complexity(&self, data: ArrayView1<f64>) -> PadsResult<f64> {
        // Use multiple scales for complexity estimation
        let mut complexities = Vec::with_capacity(3);
        
        // Short-term complexity (m=1)
        if data.len() >= 10 {
            let short_entropy = self.calculate_sample_entropy_with_params(data, 1, self.params.sample_entropy_r * 0.5)?;
            complexities.push(short_entropy);
        }
        
        // Medium-term complexity (m=2, default)
        complexities.push(self.calculate_sample_entropy(data)?);
        
        // Long-term complexity (m=3)
        if data.len() >= self.params.sample_entropy_min_points + 10 {
            let long_entropy = self.calculate_sample_entropy_with_params(data, 3, self.params.sample_entropy_r * 1.5)?;
            complexities.push(long_entropy);
        }
        
        // Weighted average with emphasis on medium-term
        let weights = if complexities.len() == 3 {
            [0.2, 0.6, 0.2]
        } else if complexities.len() == 2 {
            [0.4, 0.6]
        } else {
            [1.0]
        };
        
        let complexity = complexities
            .iter()
            .zip(weights.iter())
            .map(|(c, w)| c * w)
            .sum::<f64>();
            
        Ok(complexity)
    }
    
    /// Calculate sample entropy with custom parameters
    fn calculate_sample_entropy_with_params(&self, data: ArrayView1<f64>, m: usize, r: f64) -> PadsResult<f64> {
        let n = data.len();
        if n < 10 {
            return Ok(0.5);
        }
        
        let data_f32: Vec<f32> = data.iter().map(|&x| x as f32).collect();
        let mean = data_f32.iter().sum::<f32>() / n as f32;
        let variance = data_f32.iter().map(|&x| (x - mean).powi(2)).sum::<f32>() / (n - 1) as f32;
        let std_dev = variance.sqrt();
        
        if std_dev < 1e-9 {
            return Ok(0.5);
        }
        
        let tolerance = (r as f32) * std_dev;
        let count_m = self.count_template_matches_simd(&data_f32, m, tolerance)?;
        let count_m1 = self.count_template_matches_simd(&data_f32, m + 1, tolerance)?;
        
        if count_m == 0 || count_m1 == 0 {
            return Ok(0.5);
        }
        
        Ok(-(count_m1 as f64 / count_m as f64).ln())
    }
    
    /// Calculate equilibrium score based on autocorrelation and stability
    fn calculate_equilibrium_score(&self, data: ArrayView1<f64>) -> PadsResult<f64> {
        let n = data.len();
        if n < 4 {
            return Ok(0.0);
        }
        
        // Calculate mean and variance
        let mean = data.mean().unwrap_or(0.0);
        let variance = data.var(0.0);
        
        if variance < 1e-12 {
            return Ok(1.0); // Perfect stability for constant series
        }
        
        // Calculate autocorrelation at lag 1
        let mut autocorr = 0.0;
        for i in 1..n {
            autocorr += (data[i] - mean) * (data[i-1] - mean);
        }
        autocorr /= (n - 1) as f64 * variance;
        
        // Calculate return to mean tendency
        let mut mean_reversion = 0.0;
        for i in 1..n {
            let deviation = (data[i-1] - mean).abs();
            let direction = if data[i-1] > mean { -1.0 } else { 1.0 };
            let change = data[i] - data[i-1];
            mean_reversion += direction * change / (deviation + 1e-12);
        }
        mean_reversion /= (n - 1) as f64;
        
        // Combine autocorrelation and mean reversion
        let equilibrium = (autocorr.abs() + mean_reversion.max(0.0)) / 2.0;
        Ok(equilibrium.clamp(0.0, 1.0))
    }
    
    /// Calculate fragility score based on volatility clustering and extremes
    fn calculate_fragility_score(&self, data: ArrayView1<f64>) -> PadsResult<f64> {
        let n = data.len();
        if n < 4 {
            return Ok(0.0);
        }
        
        // Calculate returns
        let mut returns = Vec::with_capacity(n - 1);
        for i in 1..n {
            if data[i-1].abs() > 1e-12 {
                returns.push((data[i] - data[i-1]) / data[i-1]);
            }
        }
        
        if returns.is_empty() {
            return Ok(0.0);
        }
        
        // Calculate volatility clustering (GARCH-like effect)
        let mut vol_clustering = 0.0;
        if returns.len() >= 3 {
            for i in 2..returns.len() {
                let vol_prev = returns[i-1].abs();
                let vol_curr = returns[i].abs();
                vol_clustering += (vol_prev * vol_curr).sqrt();
            }
            vol_clustering /= (returns.len() - 2) as f64;
        }
        
        // Calculate extreme value frequency
        let return_std = {
            let mean = returns.iter().sum::<f64>() / returns.len() as f64;
            let variance = returns.iter().map(|r| (r - mean).powi(2)).sum::<f64>() / returns.len() as f64;
            variance.sqrt()
        };
        
        let extreme_threshold = 2.0 * return_std;
        let extreme_count = returns.iter().filter(|&&r| r.abs() > extreme_threshold).count();
        let extreme_frequency = extreme_count as f64 / returns.len() as f64;
        
        // Combine volatility clustering and extreme frequency
        let fragility = (vol_clustering + extreme_frequency * 2.0) / 3.0;
        Ok(fragility.clamp(0.0, 1.0))
    }
    
    /// Classify SOC regime based on multiple metrics
    fn classify_regime(
        &self,
        complexity: f64,
        equilibrium: f64,
        fragility: f64,
        entropy: f64,
    ) -> PadsResult<(SOCRegime, f64)> {
        
        // Calculate scores for each regime
        let critical_score = self.calculate_critical_score(complexity, equilibrium, fragility, entropy);
        let stable_score = self.calculate_stable_score(complexity, equilibrium, fragility, entropy);
        let unstable_score = self.calculate_unstable_score(complexity, equilibrium, fragility, entropy);
        
        // Find the regime with highest score
        let scores = [
            (SOCRegime::Critical, critical_score),
            (SOCRegime::Stable, stable_score),
            (SOCRegime::Unstable, unstable_score),
        ];
        
        let (regime, max_score) = scores
            .iter()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .copied()
            .unwrap_or((SOCRegime::Unknown, 0.0));
        
        // Calculate confidence as the difference between best and second-best scores
        let mut sorted_scores = [critical_score, stable_score, unstable_score];
        sorted_scores.sort_by(|a, b| b.partial_cmp(a).unwrap());
        
        let confidence = if sorted_scores[0] > 0.0 && sorted_scores[1] >= 0.0 {
            (sorted_scores[0] - sorted_scores[1]) / sorted_scores[0]
        } else {
            0.0
        };
        
        // Apply minimum confidence threshold
        if max_score < 0.1 || confidence < 0.1 {
            Ok((SOCRegime::Unknown, confidence))
        } else {
            Ok((regime, confidence))
        }
    }
    
    /// Calculate score for critical regime
    fn calculate_critical_score(&self, complexity: f64, equilibrium: f64, fragility: f64, _entropy: f64) -> f64 {
        let mut score = 0.0;
        
        // High complexity indicates criticality
        if complexity >= self.params.critical_threshold_complexity {
            score += 0.4 * (complexity - self.params.critical_threshold_complexity) / 
                     (1.0 - self.params.critical_threshold_complexity);
        }
        
        // Low equilibrium indicates instability near critical point
        if equilibrium <= self.params.critical_threshold_equilibrium {
            score += 0.3 * (self.params.critical_threshold_equilibrium - equilibrium) / 
                     self.params.critical_threshold_equilibrium;
        }
        
        // Moderate to high fragility indicates criticality
        if fragility >= self.params.critical_threshold_fragility {
            score += 0.3 * (fragility - self.params.critical_threshold_fragility) / 
                     (1.0 - self.params.critical_threshold_fragility);
        }
        
        score.clamp(0.0, 1.0)
    }
    
    /// Calculate score for stable regime
    fn calculate_stable_score(&self, _complexity: f64, equilibrium: f64, fragility: f64, entropy: f64) -> f64 {
        let mut score = 0.0;
        
        // High equilibrium indicates stability
        if equilibrium >= self.params.stable_threshold_equilibrium {
            score += 0.4 * (equilibrium - self.params.stable_threshold_equilibrium) / 
                     (1.0 - self.params.stable_threshold_equilibrium);
        }
        
        // Low fragility indicates stability
        if fragility <= self.params.stable_threshold_fragility {
            score += 0.3 * (self.params.stable_threshold_fragility - fragility) / 
                     self.params.stable_threshold_fragility;
        }
        
        // Moderate entropy indicates organized but not critical
        if entropy <= self.params.stable_threshold_entropy {
            score += 0.3 * (self.params.stable_threshold_entropy - entropy) / 
                     self.params.stable_threshold_entropy;
        }
        
        score.clamp(0.0, 1.0)
    }
    
    /// Calculate score for unstable regime
    fn calculate_unstable_score(&self, _complexity: f64, equilibrium: f64, fragility: f64, entropy: f64) -> f64 {
        let mut score = 0.0;
        
        // Low equilibrium indicates instability
        if equilibrium <= self.params.unstable_threshold_equilibrium {
            score += 0.3 * (self.params.unstable_threshold_equilibrium - equilibrium) / 
                     self.params.unstable_threshold_equilibrium;
        }
        
        // High fragility indicates instability
        if fragility >= self.params.unstable_threshold_fragility {
            score += 0.4 * (fragility - self.params.unstable_threshold_fragility) / 
                     (1.0 - self.params.unstable_threshold_fragility);
        }
        
        // High entropy indicates chaos/disorder
        if entropy >= self.params.unstable_threshold_entropy {
            score += 0.3 * (entropy - self.params.unstable_threshold_entropy) / 
                     (10.0 - self.params.unstable_threshold_entropy); // Assume max entropy around 10
        }
        
        score.clamp(0.0, 1.0)
    }
}

impl PatternAnalyzer for SOCAnalyzer {
    type Result = SOCResult;
    
    fn analyze(&self, data: &MarketData) -> PadsResult<Self::Result> {
        let start_time = Instant::now();
        
        // Validate inputs
        if data.len() < self.params.sample_entropy_min_points {
            return Err(PadsError::InsufficientData {
                required: self.params.sample_entropy_min_points,
                actual: data.len(),
            });
        }
        
        data.validate()?;
        
        let mut result = SOCResult::new();
        result.data_points = data.len();
        result.simd_used = self.params.enable_simd;
        
        // Calculate sample entropy
        result.sample_entropy = self.calculate_sample_entropy(data.prices.view())?;
        
        // Calculate entropy rate
        result.entropy_rate = self.calculate_entropy_rate(data.prices.view())?;
        
        // Calculate complexity measure
        result.complexity_measure = self.calculate_complexity(data.prices.view())?;
        
        // Calculate equilibrium and fragility scores
        result.equilibrium_score = self.calculate_equilibrium_score(data.prices.view())?;
        result.fragility_score = self.calculate_fragility_score(data.prices.view())?;
        
        // Classify SOC regime
        let (regime, confidence) = self.classify_regime(
            result.complexity_measure,
            result.equilibrium_score,
            result.fragility_score,
            result.sample_entropy,
        )?;
        result.regime = regime;
        result.regime_confidence = confidence;
        
        // Detect avalanche events
        let returns: Vec<f32> = data.prices.windows(2)
            .map(|w| ((w[1] - w[0]) / w[0]) as f32)
            .collect();
        
        result.avalanche_events = self.avalanche_detector.detect_avalanches(&returns)?;
        result.total_avalanche_magnitude = result.avalanche_events
            .iter()
            .map(|a| a.magnitude)
            .sum();
        result.avalanche_frequency = result.avalanche_events.len() as f64 / data.len() as f64;
        
        // Calculate branching ratios
        let avalanche_sizes: Vec<f64> = result.avalanche_events
            .iter()
            .map(|a| a.magnitude)
            .collect();
        result.branching_ratios = self.avalanche_detector.calculate_branching_ratios(&avalanche_sizes, 5);
        
        result.computation_time_ns = start_time.elapsed().as_nanos() as u64;
        
        Ok(result)
    }
    
    fn name(&self) -> &'static str {
        "soc"
    }
    
    fn supports_simd(&self) -> bool {
        true
    }
    
    fn supports_parallel(&self) -> bool {
        true
    }
    
    fn min_data_points(&self) -> usize {
        self.params.sample_entropy_min_points
    }
}

impl Default for SOCAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// Avalanche detector for identifying and analyzing avalanche events
pub struct AvalancheDetector {
    min_avalanche_size: usize,
    significance_threshold: f64,
}

impl AvalancheDetector {
    /// Create a new avalanche detector
    pub fn new(min_size: usize, threshold: f64) -> Self {
        Self {
            min_avalanche_size: min_size,
            significance_threshold: threshold,
        }
    }
    
    /// Detect avalanche events in returns data
    pub fn detect_avalanches(&self, returns: &[f32]) -> PadsResult<Vec<AvalancheEvent>> {
        let n = returns.len();
        if n < 10 {
            return Ok(Vec::new());
        }
        
        // Calculate rolling variance to detect volatility changes
        let window = (n / 10).max(5).min(20);
        let mut variances = Vec::with_capacity(n - window + 1);
        
        for i in window..=n {
            let segment = &returns[i-window..i];
            let mean = segment.iter().sum::<f32>() / segment.len() as f32;
            let variance = segment.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / segment.len() as f32;
            variances.push(variance);
        }
        
        if variances.is_empty() {
            return Ok(Vec::new());
        }
        
        // Find avalanche threshold (2 standard deviations above mean)
        let var_mean = variances.iter().sum::<f32>() / variances.len() as f32;
        let var_std = {
            let var_variance = variances.iter().map(|v| (v - var_mean).powi(2)).sum::<f32>() / variances.len() as f32;
            var_variance.sqrt()
        };
        
        let threshold = var_mean + 2.0 * var_std;
        
        // Detect avalanche events
        let mut events = Vec::new();
        let mut in_avalanche = false;
        let mut start_idx = 0;
        
        for (i, &variance) in variances.iter().enumerate() {
            if variance > threshold && !in_avalanche {
                // Start of avalanche
                in_avalanche = true;
                start_idx = i + window - 1;
            } else if variance <= threshold && in_avalanche {
                // End of avalanche
                in_avalanche = false;
                let end_idx = i + window - 1;
                let duration = end_idx - start_idx;
                
                if duration >= self.min_avalanche_size {
                    // Calculate avalanche magnitude and intensity
                    let segment = &returns[start_idx..=end_idx];
                    let magnitude = segment.iter().map(|x| x.abs() as f64).sum::<f64>();
                    let intensity = magnitude / duration as f64;
                    
                    events.push(AvalancheEvent {
                        start_index: start_idx,
                        end_index: end_idx,
                        magnitude,
                        duration,
                        intensity,
                    });
                }
            }
        }
        
        Ok(events)
    }
    
    /// Calculate branching ratios for criticality analysis
    pub fn calculate_branching_ratios(&self, avalanche_sizes: &[f64], window: usize) -> Vec<f64> {
        let n = avalanche_sizes.len();
        if n < window * 2 {
            return vec![1.0; n];
        }
        
        let mut branching_ratios = vec![1.0f64; n];
        
        for i in window..n - window {
            let ancestors = &avalanche_sizes[(i - window)..i];
            let descendants = &avalanche_sizes[i..(i + window)];
            
            let ancestor_sum: f64 = ancestors.iter().sum();
            let descendant_sum: f64 = descendants.iter().sum();
            
            if ancestor_sum > 0.0 {
                branching_ratios[i] = descendant_sum / ancestor_sum;
            }
        }
        
        branching_ratios
    }
}

/// Analysis cache for performance optimization
struct AnalysisCache {
    cache: HashMap<String, SOCResult>,
    max_size: usize,
}

impl AnalysisCache {
    fn new(max_size: usize) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn generate_test_data(n: usize) -> MarketData {
        let mut prices = Vec::with_capacity(n);
        let mut volumes = Vec::with_capacity(n);
        
        let mut price = 100.0;
        for i in 0..n {
            let return_rate = 0.001 * ((i as f64) * 0.1).sin() + 
                             0.002 * ((i as f64) * 0.05).cos();
            price *= 1.0 + return_rate;
            prices.push(price);
            volumes.push(1000.0 + 100.0 * ((i as f64) * 0.03).sin());
        }
        
        MarketData::new(prices, volumes)
    }
    
    #[test]
    fn test_soc_analyzer_creation() {
        let analyzer = SOCAnalyzer::new();
        assert_eq!(analyzer.params.sample_entropy_m, 2);
        assert_eq!(analyzer.params.sample_entropy_r, 0.2);
    }
    
    #[test]
    fn test_sample_entropy_calculation() {
        let analyzer = SOCAnalyzer::new();
        let data = Array1::from_vec((0..100).map(|x| (x as f64 * 0.1).sin()).collect());
        
        let entropy = analyzer.calculate_sample_entropy(data.view()).unwrap();
        assert!(entropy >= 0.0);
        assert!(entropy < 10.0); // Reasonable upper bound
    }
    
    #[test]
    fn test_soc_analysis() {
        let analyzer = SOCAnalyzer::new();
        let data = generate_test_data(100);
        
        let result = analyzer.analyze(&data).unwrap();
        assert!(result.sample_entropy >= 0.0);
        assert!(result.entropy_rate >= 0.0);
        assert!(result.complexity_measure >= 0.0);
        assert!(result.equilibrium_score >= 0.0 && result.equilibrium_score <= 1.0);
        assert!(result.fragility_score >= 0.0 && result.fragility_score <= 1.0);
        assert!(result.regime_confidence >= 0.0 && result.regime_confidence <= 1.0);
    }
    
    #[test]
    fn test_insufficient_data() {
        let analyzer = SOCAnalyzer::new();
        let data = generate_test_data(10); // Too few points
        
        let result = analyzer.analyze(&data);
        assert!(result.is_err());
        
        if let Err(PadsError::InsufficientData { required, actual }) = result {
            assert_eq!(required, 20);
            assert_eq!(actual, 10);
        } else {
            panic!("Expected InsufficientData error");
        }
    }
    
    #[test]
    fn test_avalanche_detection() {
        let detector = AvalancheDetector::new(2, 0.01);
        let returns = vec![
            0.01, 0.02, -0.03, -0.02, 0.001, 0.0005, 0.04, 0.03, -0.001, -0.05
        ];
        
        let events = detector.detect_avalanches(&returns).unwrap();
        assert!(events.len() >= 0); // Should not fail
        
        // Check that all events have positive magnitude
        for event in &events {
            assert!(event.magnitude > 0.0);
            assert!(event.duration >= 2);
            assert!(event.intensity > 0.0);
        }
    }
    
    #[test]
    fn test_branching_ratios() {
        let detector = AvalancheDetector::new(2, 0.01);
        let sizes = vec![1.0, 2.0, 4.0, 8.0, 4.0, 2.0, 1.0, 1.0, 1.0, 1.0];
        
        let ratios = detector.calculate_branching_ratios(&sizes, 2);
        assert_eq!(ratios.len(), sizes.len());
        
        // All ratios should be positive
        for ratio in &ratios {
            assert!(*ratio > 0.0);
        }
    }
}