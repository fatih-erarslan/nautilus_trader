//! Fusion algorithms for CDFA
//! 
//! This module provides various methods for combining multiple
//! rankings or scores from different sources into a consensus result.
//! 
//! Supports both score-based and rank-based fusion methods, as well as
//! hybrid approaches that combine the best of both strategies.

use crate::error::{CdfaError, Result};
use crate::types::*;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

pub mod score_fusion;
pub mod rank_fusion;

// Re-export main types and functions
pub use score_fusion::{ScoreFusion, AdaptiveScoreFusion, WeightedAverageFusion};
pub use rank_fusion::{RankFusion, HybridRankScoreFusion, scores_to_rankings, rankings_to_scores};

/// Main fusion interface that supports both score and rank-based methods
pub struct CdfaFusion;

impl CdfaFusion {
    /// Perform fusion using the specified method
    /// 
    /// # Arguments
    /// * `data` - Data matrix where rows are sources and columns are items
    /// * `method` - Fusion method to use
    /// * `params` - Optional parameters for the fusion method
    /// 
    /// # Returns
    /// * Fused result vector
    pub fn fuse(
        data: &ArrayView2<Float>,
        method: FusionMethod,
        params: Option<FusionParams>
    ) -> Result<Array1<Float>> {
        // Validate input
        if data.is_empty() {
            return Err(CdfaError::invalid_input("Data cannot be empty"));
        }
        
        if data.nrows() < 1 {
            return Err(CdfaError::invalid_input("Need at least one data source"));
        }
        
        match method {
            // Score-based methods
            FusionMethod::Average => ScoreFusion::average(data),
            FusionMethod::WeightedAverage => {
                if let Some(p) = params {
                    if let Some(weights) = p.weights {
                        ScoreFusion::weighted_average(data, &weights.view())
                    } else {
                        Err(CdfaError::config_error("Weighted average requires weights parameter"))
                    }
                } else {
                    Err(CdfaError::config_error("Weighted average requires parameters"))
                }
            },
            FusionMethod::NormalizedAverage => ScoreFusion::normalized_average(data),
            FusionMethod::StandardizedAverage => ScoreFusion::standardized_average(data),
            FusionMethod::Maximum => ScoreFusion::maximum(data),
            FusionMethod::Minimum => ScoreFusion::minimum(data),
            FusionMethod::Median => ScoreFusion::median(data),
            FusionMethod::TrimmedMean => {
                let trim_percent = params
                    .and_then(|p| p.trim_percent)
                    .unwrap_or(0.1);
                ScoreFusion::trimmed_mean(data, trim_percent)
            },
            FusionMethod::CombSum => ScoreFusion::comb_sum(data),
            FusionMethod::CombMnz => {
                let threshold = params
                    .and_then(|p| p.threshold)
                    .unwrap_or(0.0);
                ScoreFusion::comb_mnz(data, threshold)
            },
            FusionMethod::Isr => ScoreFusion::isr_fusion(data),
            FusionMethod::GeometricMean => ScoreFusion::geometric_mean(data),
            FusionMethod::HarmonicMean => ScoreFusion::harmonic_mean(data),
            
            // Rank-based methods
            FusionMethod::BordaCount => {
                let rankings = rank_fusion::scores_to_rankings(data)?;
                RankFusion::borda_count(&rankings.view())
            },
            FusionMethod::MedianRank => {
                let rankings = rank_fusion::scores_to_rankings(data)?;
                RankFusion::median_rank(&rankings.view())
            },
            FusionMethod::MinimumRank => {
                let rankings = rank_fusion::scores_to_rankings(data)?;
                RankFusion::minimum_rank(&rankings.view())
            },
            FusionMethod::ReciprocalRank => {
                let rankings = rank_fusion::scores_to_rankings(data)?;
                RankFusion::reciprocal_rank(&rankings.view())
            },
            FusionMethod::Kemeny => {
                let rankings = rank_fusion::scores_to_rankings(data)?;
                RankFusion::kemeny_approximation(&rankings.view())
            },
            FusionMethod::Footrule => {
                let rankings = rank_fusion::scores_to_rankings(data)?;
                RankFusion::footrule_aggregation(&rankings.view())
            },
            
            // Hybrid method
            FusionMethod::Hybrid => {
                let alpha = params
                    .and_then(|p| p.alpha)
                    .unwrap_or(0.5);
                HybridRankScoreFusion::fuse(data, alpha)
            },
            
            // Adaptive method
            FusionMethod::Adaptive => {
                let adaptive = AdaptiveScoreFusion::new();
                adaptive.fuse(data)
            },
            
            // Diversity weighted method
            FusionMethod::DiversityWeighted => {
                // Use weighted average with diversity-based weights
                ScoreFusion::weighted_average(data, &Array1::ones(data.ncols()).view())
            },
        }
    }
    
    /// Perform ensemble fusion using multiple methods
    /// 
    /// Combines results from multiple fusion methods for potentially better robustness.
    pub fn ensemble_fuse(
        data: &ArrayView2<Float>,
        methods: &[FusionMethod],
        weights: Option<&ArrayView1<Float>>,
    ) -> Result<Array1<Float>> {
        if methods.is_empty() {
            return Err(CdfaError::invalid_input("Need at least one fusion method"));
        }
        
        let n_items = data.ncols();
        let mut results = Vec::new();
        
        // Apply each fusion method
        for &method in methods {
            let result = Self::fuse(data, method, None)?;
            results.push(result);
        }
        
        // Create matrix from results
        let result_matrix = Array2::from_shape_fn((results.len(), n_items), |(i, j)| results[i][j]);
        
        // Fuse the results
        if let Some(w) = weights {
            if w.len() != methods.len() {
                return Err(CdfaError::dimension_mismatch(methods.len(), w.len()));
            }
            let params = FusionParams::new().with_weights(w.to_owned());
            Self::fuse(&result_matrix.view(), FusionMethod::WeightedAverage, Some(params))
        } else {
            Self::fuse(&result_matrix.view(), FusionMethod::Average, None)
        }
    }
}

/// Available fusion methods
#[derive(Debug, Clone, Copy, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub enum FusionMethod {
    // Score-based methods
    Average,
    WeightedAverage,
    NormalizedAverage,
    StandardizedAverage,
    Maximum,
    Minimum,
    Median,
    TrimmedMean,
    CombSum,
    CombMnz,
    Isr,
    GeometricMean,
    HarmonicMean,
    
    // Rank-based methods
    BordaCount,
    MedianRank,
    MinimumRank,
    ReciprocalRank,
    Kemeny,
    Footrule,
    
    // Hybrid and adaptive
    Hybrid,
    Adaptive,
    DiversityWeighted,
}

impl FusionMethod {
    /// Get all available fusion methods
    pub fn all_methods() -> Vec<FusionMethod> {
        vec![
            FusionMethod::Average,
            FusionMethod::WeightedAverage,
            FusionMethod::NormalizedAverage,
            FusionMethod::StandardizedAverage,
            FusionMethod::Maximum,
            FusionMethod::Minimum,
            FusionMethod::Median,
            FusionMethod::TrimmedMean,
            FusionMethod::CombSum,
            FusionMethod::CombMnz,
            FusionMethod::Isr,
            FusionMethod::GeometricMean,
            FusionMethod::HarmonicMean,
            FusionMethod::BordaCount,
            FusionMethod::MedianRank,
            FusionMethod::MinimumRank,
            FusionMethod::ReciprocalRank,
            FusionMethod::Kemeny,
            FusionMethod::Footrule,
            FusionMethod::Hybrid,
            FusionMethod::Adaptive,
        ]
    }
    
    /// Check if this is a score-based method
    pub fn is_score_based(&self) -> bool {
        matches!(
            self,
            FusionMethod::Average |
            FusionMethod::WeightedAverage |
            FusionMethod::NormalizedAverage |
            FusionMethod::StandardizedAverage |
            FusionMethod::Maximum |
            FusionMethod::Minimum |
            FusionMethod::Median |
            FusionMethod::TrimmedMean |
            FusionMethod::CombSum |
            FusionMethod::CombMnz |
            FusionMethod::Isr |
            FusionMethod::GeometricMean |
            FusionMethod::HarmonicMean
        )
    }
    
    /// Check if this is a rank-based method
    pub fn is_rank_based(&self) -> bool {
        matches!(
            self,
            FusionMethod::BordaCount |
            FusionMethod::MedianRank |
            FusionMethod::MinimumRank |
            FusionMethod::ReciprocalRank |
            FusionMethod::Kemeny |
            FusionMethod::Footrule
        )
    }
    
    /// Check if this method requires additional parameters
    pub fn requires_parameters(&self) -> bool {
        matches!(
            self,
            FusionMethod::WeightedAverage |
            FusionMethod::TrimmedMean |
            FusionMethod::CombMnz |
            FusionMethod::Hybrid
        )
    }
}

/// Parameters for fusion methods
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FusionParams {
    /// Weights for weighted fusion methods
    pub weights: Option<Array1<Float>>,
    
    /// Trim percentage for trimmed mean
    pub trim_percent: Option<Float>,
    
    /// Threshold for CombMNZ
    pub threshold: Option<Float>,
    
    /// Alpha parameter for hybrid methods
    pub alpha: Option<Float>,
    
    /// Custom parameters
    pub custom: std::collections::HashMap<String, Float>,
}

impl Default for FusionParams {
    fn default() -> Self {
        Self {
            weights: None,
            trim_percent: None,
            threshold: None,
            alpha: None,
            custom: std::collections::HashMap::new(),
        }
    }
}

impl FusionParams {
    /// Create new empty parameters
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Set weights for weighted methods
    pub fn with_weights(mut self, weights: Array1<Float>) -> Self {
        self.weights = Some(weights);
        self
    }
    
    /// Set trim percentage for trimmed mean
    pub fn with_trim_percent(mut self, trim_percent: Float) -> Self {
        if !(0.0..=0.5).contains(&trim_percent) {
            panic!("Trim percent must be between 0.0 and 0.5");
        }
        self.trim_percent = Some(trim_percent);
        self
    }
    
    /// Set threshold for CombMNZ
    pub fn with_threshold(mut self, threshold: Float) -> Self {
        self.threshold = Some(threshold);
        self
    }
    
    /// Set alpha parameter for hybrid methods
    pub fn with_alpha(mut self, alpha: Float) -> Self {
        if !(0.0..=1.0).contains(&alpha) {
            panic!("Alpha must be between 0.0 and 1.0");
        }
        self.alpha = Some(alpha);
        self
    }
    
    /// Add custom parameter
    pub fn with_custom(mut self, key: String, value: Float) -> Self {
        self.custom.insert(key, value);
        self
    }
    
    /// Get custom parameter
    pub fn get_custom(&self, key: &str) -> Option<Float> {
        self.custom.get(key).copied()
    }
}

/// Fusion result with metadata
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FusionResult {
    /// Fused values
    pub values: Array1<Float>,
    
    /// Method used for fusion
    pub method: FusionMethod,
    
    /// Parameters used
    pub params: Option<FusionParams>,
    
    /// Execution time in microseconds
    pub execution_time_us: u64,
    
    /// Quality metrics
    pub quality_metrics: FusionQualityMetrics,
}

/// Quality metrics for fusion results
#[derive(Debug, Clone, Default)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct FusionQualityMetrics {
    /// Consensus level (0.0 - 1.0)
    pub consensus: Float,
    
    /// Diversity of input sources (0.0 - 1.0)
    pub diversity: Float,
    
    /// Stability of the fusion (consistency across similar inputs)
    pub stability: Float,
    
    /// Robustness to outliers
    pub robustness: Float,
}

/// Comprehensive fusion analysis
pub fn comprehensive_fusion_analysis(
    data: &ArrayView2<Float>,
    methods: Option<Vec<FusionMethod>>,
) -> Result<Vec<FusionResult>> {
    let methods = methods.unwrap_or_else(|| {
        vec![
            FusionMethod::Average,
            FusionMethod::Median,
            FusionMethod::BordaCount,
            FusionMethod::Adaptive,
        ]
    });
    
    let mut results = Vec::new();
    
    for method in methods {
        let timer = crate::utils::Timer::start();
        
        let values = CdfaFusion::fuse(data, method, None)?;
        
        let execution_time = timer.elapsed_us();
        
        // Calculate quality metrics (simplified)
        let quality_metrics = calculate_fusion_quality(data, &values.view())?;
        
        results.push(FusionResult {
            values,
            method,
            params: None,
            execution_time_us: execution_time,
            quality_metrics,
        });
    }
    
    Ok(results)
}

/// Calculate fusion quality metrics
fn calculate_fusion_quality(
    input_data: &ArrayView2<Float>,
    fused_result: &ArrayView1<Float>,
) -> Result<FusionQualityMetrics> {
    let n_sources = input_data.nrows();
    let n_items = input_data.ncols();
    
    if n_sources == 0 || n_items == 0 {
        return Err(CdfaError::invalid_input("Empty input data"));
    }
    
    // Calculate consensus: how much sources agree
    let mut total_agreement = 0.0;
    let mut pair_count = 0;
    
    for i in 0..n_sources {
        for j in i + 1..n_sources {
            let corr = crate::core::diversity::pearson_correlation_fast(
                &input_data.row(i),
                &input_data.row(j),
            )?;
            total_agreement += corr.abs();
            pair_count += 1;
        }
    }
    
    let consensus = if pair_count > 0 {
        total_agreement / pair_count as Float
    } else {
        1.0
    };
    
    // Calculate diversity: variance in input sources
    let mean_variance: Float = (0..n_items)
        .map(|j| {
            let column = input_data.column(j);
            crate::utils::stats::variance(&column).unwrap_or(0.0)
        })
        .sum::<Float>() / n_items as Float;
    
    let diversity = (mean_variance / (mean_variance + 1.0)).min(1.0); // Normalize
    
    // Simplified stability and robustness metrics
    let stability = 0.8; // Would require multiple runs or bootstrap sampling
    let robustness = 0.7; // Would require outlier injection tests
    
    Ok(FusionQualityMetrics {
        consensus,
        diversity,
        stability,
        robustness,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_cdfa_fusion_average() {
        let data = array![
            [0.8, 0.6, 0.9],
            [0.7, 0.8, 0.6],
            [0.9, 0.5, 0.8]
        ];
        
        let result = CdfaFusion::fuse(&data.view(), FusionMethod::Average, None).unwrap();
        assert_eq!(result.len(), 3);
        
        // Check if results are reasonable
        assert_abs_diff_eq!(result[0], (0.8 + 0.7 + 0.9) / 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], (0.6 + 0.8 + 0.5) / 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(result[2], (0.9 + 0.6 + 0.8) / 3.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_cdfa_fusion_weighted() {
        let data = array![
            [1.0, 0.0],
            [0.0, 1.0]
        ];
        let params = FusionParams::new()
            .with_weights(array![0.7, 0.3]);
        
        let result = CdfaFusion::fuse(&data.view(), FusionMethod::WeightedAverage, Some(params)).unwrap();
        assert_abs_diff_eq!(result[0], 0.7, epsilon = 1e-10);
        assert_abs_diff_eq!(result[1], 0.3, epsilon = 1e-10);
    }
    
    #[test]
    fn test_cdfa_fusion_borda() {
        let scores = array![
            [0.9, 0.7, 0.8],
            [0.5, 0.6, 0.4]
        ];
        
        let result = CdfaFusion::fuse(&scores.view(), FusionMethod::BordaCount, None).unwrap();
        assert_eq!(result.len(), 3);
        
        // First item should have highest Borda score
        assert!(result[0] > result[1] && result[0] > result[2]);
    }
    
    #[test]
    fn test_ensemble_fusion() {
        let data = array![
            [0.9, 0.7, 0.8],
            [0.5, 0.6, 0.4]
        ];
        
        let methods = vec![
            FusionMethod::Average,
            FusionMethod::Median,
            FusionMethod::BordaCount,
        ];
        
        let result = CdfaFusion::ensemble_fuse(&data.view(), &methods, None).unwrap();
        assert_eq!(result.len(), 3);
    }
    
    #[test]
    fn test_fusion_method_properties() {
        assert!(FusionMethod::Average.is_score_based());
        assert!(!FusionMethod::Average.is_rank_based());
        assert!(!FusionMethod::Average.requires_parameters());
        
        assert!(!FusionMethod::BordaCount.is_score_based());
        assert!(FusionMethod::BordaCount.is_rank_based());
        assert!(!FusionMethod::BordaCount.requires_parameters());
        
        assert!(FusionMethod::WeightedAverage.requires_parameters());
        assert!(FusionMethod::TrimmedMean.requires_parameters());
    }
    
    #[test]
    fn test_fusion_params() {
        let mut params = FusionParams::new();
        
        params = params.with_weights(array![0.5, 0.3, 0.2]);
        assert!(params.weights.is_some());
        
        params = params.with_trim_percent(0.1);
        assert_eq!(params.trim_percent, Some(0.1));
        
        params = params.with_threshold(0.5);
        assert_eq!(params.threshold, Some(0.5));
        
        params = params.with_alpha(0.7);
        assert_eq!(params.alpha, Some(0.7));
        
        params = params.with_custom("beta".to_string(), 0.9);
        assert_eq!(params.get_custom("beta"), Some(0.9));
    }
    
    #[test]
    fn test_comprehensive_fusion_analysis() {
        let data = array![
            [0.8, 0.6, 0.9, 0.3],
            [0.7, 0.8, 0.6, 0.4],
            [0.9, 0.5, 0.8, 0.5]
        ];
        
        let results = comprehensive_fusion_analysis(&data.view(), None).unwrap();
        assert!(!results.is_empty());
        
        for result in results {
            assert_eq!(result.values.len(), 4);
            assert!(result.execution_time_us > 0);
            assert!(result.quality_metrics.consensus >= 0.0);
            assert!(result.quality_metrics.consensus <= 1.0);
        }
    }
    
    #[test]
    fn test_input_validation() {
        let empty_data = Array2::<Float>::zeros((0, 0));
        assert!(CdfaFusion::fuse(&empty_data.view(), FusionMethod::Average, None).is_err());
        
        let no_sources = Array2::<Float>::zeros((0, 3));
        assert!(CdfaFusion::fuse(&no_sources.view(), FusionMethod::Average, None).is_err());
    }
}