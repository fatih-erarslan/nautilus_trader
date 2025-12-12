//! Confidence scoring for regime detection

use crate::types::{MarketRegime, RegimeFeatures};
use wide::f32x8;

/// Confidence scorer for regime detection
pub struct ConfidenceScorer {
    /// Feature weights for each regime
    weights: RegimeWeights,
}

/// Weights for different features per regime
struct RegimeWeights {
    trending_bull: FeatureWeights,
    trending_bear: FeatureWeights,
    ranging: FeatureWeights,
    high_volatility: FeatureWeights,
    low_volatility: FeatureWeights,
    transition: FeatureWeights,
}

/// Individual feature weights
struct FeatureWeights {
    trend: f32,
    volatility: f32,
    autocorrelation: f32,
    vwap: f32,
    hurst: f32,
    rsi: f32,
    noise: f32,
    ofi: f32,
}

impl Default for ConfidenceScorer {
    fn default() -> Self {
        Self {
            weights: RegimeWeights {
                trending_bull: FeatureWeights {
                    trend: 0.25,
                    volatility: -0.1,
                    autocorrelation: 0.15,
                    vwap: 0.1,
                    hurst: 0.2,
                    rsi: 0.15,
                    noise: -0.05,
                    ofi: 0.1,
                },
                trending_bear: FeatureWeights {
                    trend: -0.25,
                    volatility: -0.1,
                    autocorrelation: 0.15,
                    vwap: -0.1,
                    hurst: 0.2,
                    rsi: -0.15,
                    noise: -0.05,
                    ofi: -0.1,
                },
                ranging: FeatureWeights {
                    trend: -0.3,
                    volatility: -0.2,
                    autocorrelation: -0.2,
                    vwap: 0.0,
                    hurst: -0.15,
                    rsi: 0.0,
                    noise: 0.1,
                    ofi: -0.05,
                },
                high_volatility: FeatureWeights {
                    trend: 0.0,
                    volatility: 0.4,
                    autocorrelation: -0.1,
                    vwap: 0.0,
                    hurst: -0.1,
                    rsi: 0.0,
                    noise: 0.2,
                    ofi: 0.0,
                },
                low_volatility: FeatureWeights {
                    trend: 0.0,
                    volatility: -0.4,
                    autocorrelation: 0.1,
                    vwap: 0.0,
                    hurst: 0.1,
                    rsi: 0.0,
                    noise: -0.2,
                    ofi: 0.0,
                },
                transition: FeatureWeights {
                    trend: 0.0,
                    volatility: 0.1,
                    autocorrelation: -0.3,
                    vwap: 0.0,
                    hurst: -0.2,
                    rsi: 0.0,
                    noise: 0.3,
                    ofi: 0.1,
                },
            },
        }
    }
}

impl ConfidenceScorer {
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Calculate confidence scores for all regimes
    #[inline(always)]
    pub fn calculate_scores(&self, features: &RegimeFeatures) -> Vec<(MarketRegime, f32)> {
        let mut scores = vec![
            (MarketRegime::TrendingBull, self.score_trending_bull(features)),
            (MarketRegime::TrendingBear, self.score_trending_bear(features)),
            (MarketRegime::Ranging, self.score_ranging(features)),
            (MarketRegime::HighVolatility, self.score_high_volatility(features)),
            (MarketRegime::LowVolatility, self.score_low_volatility(features)),
            (MarketRegime::Transition, self.score_transition(features)),
        ];
        
        // Normalize scores using softmax
        self.normalize_scores(&mut scores);
        scores
    }
    
    #[inline(always)]
    fn score_trending_bull(&self, features: &RegimeFeatures) -> f32 {
        self.calculate_weighted_score(features, &self.weights.trending_bull)
    }
    
    #[inline(always)]
    fn score_trending_bear(&self, features: &RegimeFeatures) -> f32 {
        self.calculate_weighted_score(features, &self.weights.trending_bear)
    }
    
    #[inline(always)]
    fn score_ranging(&self, features: &RegimeFeatures) -> f32 {
        self.calculate_weighted_score(features, &self.weights.ranging)
    }
    
    #[inline(always)]
    fn score_high_volatility(&self, features: &RegimeFeatures) -> f32 {
        self.calculate_weighted_score(features, &self.weights.high_volatility)
    }
    
    #[inline(always)]
    fn score_low_volatility(&self, features: &RegimeFeatures) -> f32 {
        self.calculate_weighted_score(features, &self.weights.low_volatility)
    }
    
    #[inline(always)]
    fn score_transition(&self, features: &RegimeFeatures) -> f32 {
        self.calculate_weighted_score(features, &self.weights.transition)
    }
    
    #[inline(always)]
    fn calculate_weighted_score(&self, features: &RegimeFeatures, weights: &FeatureWeights) -> f32 {
        // Normalize features for scoring
        let norm_trend = features.trend_strength.tanh();
        let norm_vol = (features.volatility * 100.0).tanh();
        let norm_auto = features.autocorrelation;
        let norm_vwap = (features.vwap_ratio - 1.0).tanh();
        let norm_hurst = (features.hurst_exponent - 0.5) * 2.0;
        let norm_rsi = (features.rsi - 50.0) / 50.0;
        let norm_noise = features.microstructure_noise;
        let norm_ofi = features.order_flow_imbalance;
        
        // Calculate weighted sum using SIMD
        let feature_vec = f32x8::from([
            norm_trend,
            norm_vol,
            norm_auto,
            norm_vwap,
            norm_hurst,
            norm_rsi,
            norm_noise,
            norm_ofi,
        ]);
        
        let weight_vec = f32x8::from([
            weights.trend,
            weights.volatility,
            weights.autocorrelation,
            weights.vwap,
            weights.hurst,
            weights.rsi,
            weights.noise,
            weights.ofi,
        ]);
        
        (feature_vec * weight_vec).reduce_add()
    }
    
    #[inline(always)]
    fn normalize_scores(&self, scores: &mut Vec<(MarketRegime, f32)>) {
        // Apply softmax for probability-like scores
        let max_score = scores.iter().map(|(_, s)| *s).fold(f32::NEG_INFINITY, f32::max);
        
        let mut exp_sum = 0.0;
        for (_, score) in scores.iter_mut() {
            *score = (*score - max_score).exp();
            exp_sum += *score;
        }
        
        if exp_sum > 0.0 {
            for (_, score) in scores.iter_mut() {
                *score /= exp_sum;
            }
        }
    }
    
    /// Get transition probabilities between regimes
    pub fn get_transition_probabilities(
        &self,
        current_regime: MarketRegime,
        features: &RegimeFeatures,
    ) -> Vec<(MarketRegime, f32)> {
        // Simplified transition matrix based on market dynamics
        let transition_probs = match current_regime {
            MarketRegime::TrendingBull => vec![
                (MarketRegime::TrendingBull, 0.7),
                (MarketRegime::Ranging, 0.15),
                (MarketRegime::TrendingBear, 0.05),
                (MarketRegime::HighVolatility, 0.05),
                (MarketRegime::Transition, 0.05),
            ],
            MarketRegime::TrendingBear => vec![
                (MarketRegime::TrendingBear, 0.7),
                (MarketRegime::Ranging, 0.15),
                (MarketRegime::TrendingBull, 0.05),
                (MarketRegime::HighVolatility, 0.05),
                (MarketRegime::Transition, 0.05),
            ],
            MarketRegime::Ranging => vec![
                (MarketRegime::Ranging, 0.6),
                (MarketRegime::TrendingBull, 0.15),
                (MarketRegime::TrendingBear, 0.15),
                (MarketRegime::LowVolatility, 0.05),
                (MarketRegime::Transition, 0.05),
            ],
            MarketRegime::HighVolatility => vec![
                (MarketRegime::HighVolatility, 0.5),
                (MarketRegime::Transition, 0.2),
                (MarketRegime::TrendingBull, 0.1),
                (MarketRegime::TrendingBear, 0.1),
                (MarketRegime::Ranging, 0.1),
            ],
            MarketRegime::LowVolatility => vec![
                (MarketRegime::LowVolatility, 0.6),
                (MarketRegime::Ranging, 0.25),
                (MarketRegime::Transition, 0.1),
                (MarketRegime::HighVolatility, 0.05),
            ],
            MarketRegime::Transition => vec![
                (MarketRegime::TrendingBull, 0.25),
                (MarketRegime::TrendingBear, 0.25),
                (MarketRegime::Ranging, 0.25),
                (MarketRegime::HighVolatility, 0.15),
                (MarketRegime::LowVolatility, 0.1),
            ],
        };
        
        // Adjust probabilities based on current features
        let mut adjusted_probs = transition_probs;
        
        // Volatility adjustments
        if features.volatility > 0.02 {
            for (regime, prob) in &mut adjusted_probs {
                match regime {
                    MarketRegime::HighVolatility => *prob *= 1.2,
                    MarketRegime::LowVolatility => *prob *= 0.8,
                    _ => {}
                }
            }
        }
        
        // Trend adjustments
        if features.trend_strength.abs() > 0.5 {
            for (regime, prob) in &mut adjusted_probs {
                match regime {
                    MarketRegime::TrendingBull if features.trend_strength > 0.0 => *prob *= 1.2,
                    MarketRegime::TrendingBear if features.trend_strength < 0.0 => *prob *= 1.2,
                    MarketRegime::Ranging => *prob *= 0.8,
                    _ => {}
                }
            }
        }
        
        // Normalize
        let sum: f32 = adjusted_probs.iter().map(|(_, p)| p).sum();
        if sum > 0.0 {
            for (_, prob) in &mut adjusted_probs {
                *prob /= sum;
            }
        }
        
        adjusted_probs
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_confidence_scoring() {
        let scorer = ConfidenceScorer::new();
        let features = RegimeFeatures {
            trend_strength: 0.8,
            volatility: 0.01,
            autocorrelation: 0.6,
            vwap_ratio: 1.02,
            hurst_exponent: 0.7,
            rsi: 65.0,
            microstructure_noise: 0.1,
            order_flow_imbalance: 0.3,
        };
        
        let scores = scorer.calculate_scores(&features);
        
        // Check that scores sum to 1 (probability distribution)
        let sum: f32 = scores.iter().map(|(_, s)| s).sum();
        assert!((sum - 1.0).abs() < 1e-6);
        
        // Check that trending bull has highest score for these features
        let (best_regime, _) = scores.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()).unwrap();
        assert_eq!(*best_regime, MarketRegime::TrendingBull);
    }
}