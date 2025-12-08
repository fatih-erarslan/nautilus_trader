//! Fusion algorithms for CDFA
//! 
//! This module provides various methods for combining multiple
//! rankings or scores from different sources into a consensus result.

pub mod score_fusion;
pub mod rank_fusion;

// Re-export main types and functions
pub use score_fusion::{ScoreFusion, AdaptiveScoreFusion};
pub use rank_fusion::{RankFusion, HybridRankScoreFusion, scores_to_rankings, rankings_to_scores};

use ndarray::{Array1, ArrayView2};

/// Main fusion interface that supports both score and rank-based methods
pub struct CdfaFusion;

impl CdfaFusion {
    /// Perform fusion using the specified method
    pub fn fuse(
        data: &ArrayView2<f64>,
        method: FusionMethod,
        params: Option<FusionParams>
    ) -> Result<Array1<f64>, &'static str> {
        match method {
            // Score-based methods
            FusionMethod::Average => ScoreFusion::average(data),
            FusionMethod::WeightedAverage => {
                if let Some(p) = params {
                    if let Some(weights) = p.weights {
                        ScoreFusion::weighted_average(data, &weights.view())
                    } else {
                        Err("Weighted average requires weights parameter")
                    }
                } else {
                    Err("Weighted average requires parameters")
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
            
            // Rank-based methods
            FusionMethod::BordaCount => {
                let rankings = rank_fusion::scores_to_rankings(data);
                RankFusion::borda_count(&rankings.view())
            },
            FusionMethod::MedianRank => {
                let rankings = rank_fusion::scores_to_rankings(data);
                RankFusion::median_rank(&rankings.view())
            },
            FusionMethod::MinimumRank => {
                let rankings = rank_fusion::scores_to_rankings(data);
                RankFusion::minimum_rank(&rankings.view())
            },
            FusionMethod::ReciprocalRank => {
                let rankings = rank_fusion::scores_to_rankings(data);
                RankFusion::reciprocal_rank(&rankings.view())
            },
            FusionMethod::Kemeny => {
                let rankings = rank_fusion::scores_to_rankings(data);
                RankFusion::kemeny_approximation(&rankings.view())
            },
            FusionMethod::Footrule => {
                let rankings = rank_fusion::scores_to_rankings(data);
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
        }
    }
}

/// Available fusion methods
#[derive(Debug, Clone, Copy, PartialEq)]
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
}

/// Parameters for fusion methods
#[derive(Debug, Clone)]
pub struct FusionParams {
    pub weights: Option<Array1<f64>>,
    pub trim_percent: Option<f64>,
    pub threshold: Option<f64>,
    pub alpha: Option<f64>,
}

impl Default for FusionParams {
    fn default() -> Self {
        Self {
            weights: None,
            trim_percent: None,
            threshold: None,
            alpha: None,
        }
    }
}

impl FusionParams {
    pub fn new() -> Self {
        Self::default()
    }
    
    pub fn with_weights(mut self, weights: Array1<f64>) -> Self {
        self.weights = Some(weights);
        self
    }
    
    pub fn with_trim_percent(mut self, trim_percent: f64) -> Self {
        self.trim_percent = Some(trim_percent);
        self
    }
    
    pub fn with_threshold(mut self, threshold: f64) -> Self {
        self.threshold = Some(threshold);
        self
    }
    
    pub fn with_alpha(mut self, alpha: f64) -> Self {
        self.alpha = Some(alpha);
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_cdfa_fusion_average() {
        let data = array![
            [0.8, 0.6, 0.9],
            [0.7, 0.8, 0.6],
            [0.9, 0.5, 0.8]
        ];
        
        let result = CdfaFusion::fuse(&data.view(), FusionMethod::Average, None).unwrap();
        assert_eq!(result.len(), 3);
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
        assert!((result[0] - 0.7).abs() < 1e-10);
        assert!((result[1] - 0.3).abs() < 1e-10);
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
    fn test_cdfa_fusion_hybrid() {
        let data = array![
            [0.9, 0.7, 0.8],
            [0.5, 0.6, 0.4]
        ];
        let params = FusionParams::new().with_alpha(0.5);
        
        let result = CdfaFusion::fuse(&data.view(), FusionMethod::Hybrid, Some(params)).unwrap();
        assert_eq!(result.len(), 3);
    }
}