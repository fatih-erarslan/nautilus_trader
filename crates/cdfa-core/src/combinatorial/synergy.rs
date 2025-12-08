//! Synergy Detection for Algorithm Interactions
//! 
//! Analyzes how different algorithms complement each other,
//! detecting synergistic effects and interaction patterns.

use super::{CombinatorialResult, CombinatorialError};
use crate::diversity::{pearson_correlation, spearman_correlation, kendall_tau};
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis};
use std::collections::HashMap;

/// Detects and analyzes synergistic relationships between algorithms
pub struct SynergyDetector {
    correlation_threshold: f64,
    complementarity_threshold: f64,
    interaction_cache: HashMap<String, AlgorithmInteraction>,
    analysis_depth: AnalysisDepth,
}

/// Depth of synergy analysis
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum AnalysisDepth {
    Basic,      // Correlation analysis only
    Standard,   // Correlation + complementarity
    Advanced,   // Full interaction modeling
    Expert,     // Advanced + predictive modeling
}

/// Metrics quantifying synergy between algorithms
#[derive(Debug, Clone)]
pub struct SynergyMetrics {
    /// Pearson correlation between algorithm outputs
    pub correlation: f64,
    /// Spearman rank correlation
    pub rank_correlation: f64,
    /// Kendall tau correlation
    pub kendall_tau: f64,
    /// Complementarity score (how well they cover different aspects)
    pub complementarity: f64,
    /// Diversity contribution (how much each adds to the ensemble)
    pub diversity_contribution: f64,
    /// Interaction strength (non-linear effects)
    pub interaction_strength: f64,
    /// Stability of the synergy over different datasets
    pub stability_score: f64,
    /// Overall confidence in the synergy measurement
    pub confidence_score: f64,
    /// Redundancy level (overlap in functionality)
    pub redundancy: f64,
}

/// Interaction analysis between two algorithms
#[derive(Debug, Clone)]
pub struct AlgorithmInteraction {
    pub algorithm_a: String,
    pub algorithm_b: String,
    pub synergy_metrics: SynergyMetrics,
    pub interaction_type: InteractionType,
    pub confidence_score: f64,
    pub recommendations: Vec<String>,
}

/// Types of interactions between algorithms
#[derive(Debug, Clone, PartialEq)]
pub enum InteractionType {
    /// Algorithms produce similar results (potentially redundant)
    Redundant,
    /// Algorithms complement each other well
    Complementary,
    /// Algorithms have strong synergistic effects
    Synergistic,
    /// Algorithms interfere with each other
    Antagonistic,
    /// Algorithms are independent
    Independent,
}

/// Matrix of all pairwise interactions in the algorithm set
#[derive(Debug, Clone)]
pub struct InteractionMatrix {
    pub algorithms: Vec<String>,
    pub interactions: Array2<f64>,
    pub interaction_types: Array2<InteractionType>,
    pub summary_stats: MatrixSummary,
}

/// Summary statistics for interaction matrix
#[derive(Debug, Clone)]
pub struct MatrixSummary {
    pub avg_synergy: f64,
    pub max_synergy: f64,
    pub min_synergy: f64,
    pub redundancy_count: usize,
    pub synergistic_count: usize,
    pub complementary_count: usize,
}

impl SynergyDetector {
    /// Create a new synergy detector with default parameters
    pub fn new() -> Self {
        Self {
            correlation_threshold: 0.7,
            complementarity_threshold: 0.3,
            interaction_cache: HashMap::new(),
            analysis_depth: AnalysisDepth::Standard,
        }
    }
    
    /// Create detector with custom configuration
    pub fn with_config(
        correlation_threshold: f64,
        complementarity_threshold: f64,
        analysis_depth: AnalysisDepth,
    ) -> Self {
        Self {
            correlation_threshold,
            complementarity_threshold,
            interaction_cache: HashMap::new(),
            analysis_depth,
        }
    }
    
    /// Analyze synergy between two algorithm outputs
    pub fn analyze_pair(
        &mut self,
        algorithm_a: &str,
        algorithm_b: &str,
        output_a: &ArrayView1<f64>,
        output_b: &ArrayView1<f64>,
    ) -> CombinatorialResult<AlgorithmInteraction> {
        // Check cache first
        let cache_key = format!("{}_{}", algorithm_a, algorithm_b);
        if let Some(cached) = self.interaction_cache.get(&cache_key) {
            return Ok(cached.clone());
        }
        
        // Validate inputs
        if output_a.len() != output_b.len() {
            return Err(CombinatorialError::SynergyError {
                message: "Algorithm outputs must have the same length".to_string(),
            });
        }
        
        if output_a.len() < 3 {
            return Err(CombinatorialError::SynergyError {
                message: "Insufficient data points for synergy analysis".to_string(),
            });
        }
        
        // Calculate correlation metrics
        let correlation = pearson_correlation(output_a, output_b)
            .map_err(|e| CombinatorialError::SynergyError {
                message: format!("Pearson correlation failed: {}", e),
            })?;
            
        let rank_correlation = spearman_correlation(output_a, output_b)
            .map_err(|e| CombinatorialError::SynergyError {
                message: format!("Spearman correlation failed: {}", e),
            })?;
            
        let kendall_tau_val = kendall_tau(output_a, output_b)
            .map_err(|e| CombinatorialError::SynergyError {
                message: format!("Kendall tau failed: {}", e),
            })?;
        
        // Calculate complementarity
        let complementarity = self.calculate_complementarity(output_a, output_b)?;
        
        // Calculate diversity contribution
        let diversity_contribution = self.calculate_diversity_contribution(output_a, output_b)?;
        
        // Calculate interaction strength based on analysis depth
        let interaction_strength = match self.analysis_depth {
            AnalysisDepth::Basic => 0.0,
            _ => self.calculate_interaction_strength(output_a, output_b)?,
        };
        
        // Calculate stability score
        let stability_score = match self.analysis_depth {
            AnalysisDepth::Basic | AnalysisDepth::Standard => 0.5, // Default value
            _ => self.calculate_stability_score(output_a, output_b)?,
        };
        
        // Calculate redundancy
        let redundancy = self.calculate_redundancy(correlation, rank_correlation)?;
        
        let synergy_metrics = SynergyMetrics {
            correlation,
            rank_correlation,
            kendall_tau: kendall_tau_val,
            complementarity,
            diversity_contribution,
            interaction_strength,
            stability_score,
            confidence_score: (complementarity + diversity_contribution + stability_score) / 3.0,
            redundancy,
        };
        
        // Determine interaction type
        let interaction_type = self.classify_interaction(&synergy_metrics)?;
        
        // Calculate confidence score
        let confidence_score = self.calculate_confidence(&synergy_metrics)?;
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(&synergy_metrics, &interaction_type)?;
        
        let interaction = AlgorithmInteraction {
            algorithm_a: algorithm_a.to_string(),
            algorithm_b: algorithm_b.to_string(),
            synergy_metrics,
            interaction_type,
            confidence_score,
            recommendations,
        };
        
        // Cache the result
        self.interaction_cache.insert(cache_key, interaction.clone());
        
        Ok(interaction)
    }
    
    /// Analyze synergy across multiple algorithm outputs
    pub fn analyze_matrix(
        &mut self,
        algorithm_names: &[String],
        outputs: &ArrayView2<f64>,
    ) -> CombinatorialResult<InteractionMatrix> {
        let n_algorithms = algorithm_names.len();
        
        if outputs.nrows() != n_algorithms {
            return Err(CombinatorialError::SynergyError {
                message: "Number of algorithms must match output rows".to_string(),
            });
        }
        
        let mut interactions = Array2::zeros((n_algorithms, n_algorithms));
        let mut interaction_types = Array2::from_elem(
            (n_algorithms, n_algorithms),
            InteractionType::Independent,
        );
        
        let mut synergy_sum = 0.0;
        let mut synergy_count = 0;
        let mut max_synergy = f64::NEG_INFINITY;
        let mut min_synergy = f64::INFINITY;
        let mut redundancy_count = 0;
        let mut synergistic_count = 0;
        let mut complementary_count = 0;
        
        // Analyze all pairs
        for i in 0..n_algorithms {
            for j in i + 1..n_algorithms {
                let output_i = outputs.row(i);
                let output_j = outputs.row(j);
                
                let interaction = self.analyze_pair(
                    &algorithm_names[i],
                    &algorithm_names[j],
                    &output_i,
                    &output_j,
                )?;
                
                let synergy_score = self.calculate_overall_synergy_score(&interaction.synergy_metrics);
                
                interactions[[i, j]] = synergy_score;
                interactions[[j, i]] = synergy_score; // Symmetric matrix
                
                interaction_types[[i, j]] = interaction.interaction_type.clone();
                interaction_types[[j, i]] = interaction.interaction_type.clone();
                
                // Update statistics
                synergy_sum += synergy_score;
                synergy_count += 1;
                max_synergy = max_synergy.max(synergy_score);
                min_synergy = min_synergy.min(synergy_score);
                
                match interaction.interaction_type {
                    InteractionType::Redundant => redundancy_count += 1,
                    InteractionType::Synergistic => synergistic_count += 1,
                    InteractionType::Complementary => complementary_count += 1,
                    _ => {}
                }
            }
        }
        
        // Set diagonal to 1.0 (algorithm synergy with itself)
        for i in 0..n_algorithms {
            interactions[[i, i]] = 1.0;
        }
        
        let summary_stats = MatrixSummary {
            avg_synergy: if synergy_count > 0 { synergy_sum / synergy_count as f64 } else { 0.0 },
            max_synergy,
            min_synergy,
            redundancy_count,
            synergistic_count,
            complementary_count,
        };
        
        Ok(InteractionMatrix {
            algorithms: algorithm_names.to_vec(),
            interactions,
            interaction_types,
            summary_stats,
        })
    }
    
    /// Find optimal algorithm combinations based on synergy
    pub fn find_optimal_combinations(
        &mut self,
        algorithm_names: &[String],
        outputs: &ArrayView2<f64>,
        k: usize,
    ) -> CombinatorialResult<Vec<Vec<String>>> {
        if k > algorithm_names.len() {
            return Err(CombinatorialError::InvalidK {
                k,
                max: algorithm_names.len(),
            });
        }
        
        let interaction_matrix = self.analyze_matrix(algorithm_names, outputs)?;
        
        // Generate all k-combinations
        use itertools::Itertools;
        let combinations: Vec<Vec<usize>> = (0..algorithm_names.len())
            .combinations(k)
            .collect();
        
        // Score each combination
        let mut scored_combinations: Vec<(Vec<String>, f64)> = combinations
            .into_iter()
            .map(|indices| {
                let combination: Vec<String> = indices.iter()
                    .map(|&i| algorithm_names[i].clone())
                    .collect();
                
                let score = self.score_combination(&indices, &interaction_matrix);
                (combination, score)
            })
            .collect();
        
        // Sort by score (descending)
        scored_combinations.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        // Return top combinations (limit to reasonable number)
        let max_combinations = 10.min(scored_combinations.len());
        Ok(scored_combinations
            .into_iter()
            .take(max_combinations)
            .map(|(combination, _)| combination)
            .collect())
    }
    
    // Private helper methods
    
    fn calculate_complementarity(
        &self,
        output_a: &ArrayView1<f64>,
        output_b: &ArrayView1<f64>,
    ) -> CombinatorialResult<f64> {
        // Complementarity is high when algorithms perform well on different items
        // Calculate correlation of relative performance differences
        
        let mean_a = output_a.mean().unwrap_or(0.0);
        let mean_b = output_b.mean().unwrap_or(0.0);
        
        let diff_a: Array1<f64> = output_a.mapv(|x| x - mean_a);
        let diff_b: Array1<f64> = output_b.mapv(|x| x - mean_b);
        
        // Negative correlation of differences indicates complementarity
        let correlation = pearson_correlation(&diff_a.view(), &diff_b.view())
            .unwrap_or(0.0);
        
        // Convert to complementarity score (0 to 1)
        Ok((1.0 - correlation.abs()).max(0.0))
    }
    
    fn calculate_diversity_contribution(
        &self,
        output_a: &ArrayView1<f64>,
        output_b: &ArrayView1<f64>,
    ) -> CombinatorialResult<f64> {
        // Measure how much each algorithm contributes to overall diversity
        let combined = (output_a + output_b) / 2.0;
        
        let var_a = output_a.var(0.0);
        let var_b = output_b.var(0.0);
        let var_combined = combined.var(0.0);
        
        // Diversity contribution is related to variance preservation
        let contribution = (var_combined / (var_a + var_b).max(1e-10)).min(1.0);
        Ok(contribution)
    }
    
    fn calculate_interaction_strength(
        &self,
        output_a: &ArrayView1<f64>,
        output_b: &ArrayView1<f64>,
    ) -> CombinatorialResult<f64> {
        // Detect non-linear interactions
        // For now, use a simple product-based measure
        let product: Array1<f64> = output_a * output_b;
        let sum: Array1<f64> = output_a + output_b;
        
        let product_mean = product.mean().unwrap_or(0.0);
        let sum_mean = sum.mean().unwrap_or(0.0);
        
        // Interaction strength is deviation from additivity
        let interaction = (product_mean - sum_mean * 0.5).abs();
        Ok(interaction.min(1.0))
    }
    
    fn calculate_stability_score(
        &self,
        output_a: &ArrayView1<f64>,
        output_b: &ArrayView1<f64>,
    ) -> CombinatorialResult<f64> {
        // Assess stability by looking at variance of correlations over subsets
        // For simplicity, use coefficient of variation
        
        let len = output_a.len();
        if len < 6 {
            return Ok(0.5); // Default for small datasets
        }
        
        let subset_size = len / 3;
        let mut correlations = Vec::new();
        
        for i in 0..3 {
            let start = i * subset_size;
            let end = if i == 2 { len } else { (i + 1) * subset_size };
            
            let subset_a = output_a.slice(ndarray::s![start..end]);
            let subset_b = output_b.slice(ndarray::s![start..end]);
            
            if let Ok(corr) = pearson_correlation(&subset_a, &subset_b) {
                correlations.push(corr);
            }
        }
        
        if correlations.is_empty() {
            return Ok(0.5);
        }
        
        let mean_corr = correlations.iter().sum::<f64>() / correlations.len() as f64;
        let var_corr = correlations.iter()
            .map(|&x| (x - mean_corr).powi(2))
            .sum::<f64>() / correlations.len() as f64;
        
        let cv = if mean_corr.abs() > 1e-10 {
            var_corr.sqrt() / mean_corr.abs()
        } else {
            1.0
        };
        
        // Stability is inverse of coefficient of variation
        Ok((1.0 / (1.0 + cv)).max(0.0).min(1.0))
    }
    
    fn calculate_redundancy(
        &self,
        correlation: f64,
        rank_correlation: f64,
    ) -> CombinatorialResult<f64> {
        // High correlation indicates potential redundancy
        let avg_correlation = (correlation.abs() + rank_correlation.abs()) / 2.0;
        Ok(avg_correlation)
    }
    
    fn classify_interaction(&self, metrics: &SynergyMetrics) -> CombinatorialResult<InteractionType> {
        let avg_correlation = (metrics.correlation.abs() + metrics.rank_correlation.abs()) / 2.0;
        
        if avg_correlation > self.correlation_threshold {
            Ok(InteractionType::Redundant)
        } else if metrics.complementarity > self.complementarity_threshold &&
                  metrics.interaction_strength > 0.3 {
            Ok(InteractionType::Synergistic)
        } else if metrics.complementarity > self.complementarity_threshold {
            Ok(InteractionType::Complementary)
        } else if avg_correlation < -0.5 {
            Ok(InteractionType::Antagonistic)
        } else {
            Ok(InteractionType::Independent)
        }
    }
    
    fn calculate_confidence(&self, metrics: &SynergyMetrics) -> CombinatorialResult<f64> {
        // Confidence based on consistency across different metrics
        let stability_weight = 0.3;
        let consistency_weight = 0.4;
        let strength_weight = 0.3;
        
        let stability_score = metrics.stability_score;
        
        // Consistency: how well different correlation measures agree
        let correlations = vec![
            metrics.correlation,
            metrics.rank_correlation,
            metrics.kendall_tau,
        ];
        
        let mean_corr = correlations.iter().sum::<f64>() / correlations.len() as f64;
        let consistency = 1.0 - correlations.iter()
            .map(|&x| (x - mean_corr).abs())
            .sum::<f64>() / correlations.len() as f64;
        
        let strength = (metrics.complementarity + metrics.interaction_strength) / 2.0;
        
        let confidence = stability_weight * stability_score +
                        consistency_weight * consistency +
                        strength_weight * strength;
        
        Ok(confidence.max(0.0).min(1.0))
    }
    
    fn generate_recommendations(
        &self,
        metrics: &SynergyMetrics,
        interaction_type: &InteractionType,
    ) -> CombinatorialResult<Vec<String>> {
        let mut recommendations = Vec::new();
        
        match interaction_type {
            InteractionType::Redundant => {
                recommendations.push("Consider removing one algorithm to reduce redundancy".to_string());
                recommendations.push("If keeping both, use different weights in fusion".to_string());
            },
            InteractionType::Synergistic => {
                recommendations.push("Excellent combination - prioritize in fusion".to_string());
                recommendations.push("Consider equal or higher weights for both algorithms".to_string());
            },
            InteractionType::Complementary => {
                recommendations.push("Good complementary pair - use balanced weighting".to_string());
                recommendations.push("Consider ensemble methods that leverage diversity".to_string());
            },
            InteractionType::Antagonistic => {
                recommendations.push("Algorithms may interfere - use with caution".to_string());
                recommendations.push("Consider sequential rather than parallel application".to_string());
            },
            InteractionType::Independent => {
                recommendations.push("Algorithms are independent - standard fusion applies".to_string());
            },
        }
        
        if metrics.confidence_score < 0.5 {
            recommendations.push("Low confidence - collect more data for validation".to_string());
        }
        
        if metrics.stability_score < 0.3 {
            recommendations.push("Unstable interaction - monitor performance over time".to_string());
        }
        
        Ok(recommendations)
    }
    
    fn calculate_overall_synergy_score(&self, metrics: &SynergyMetrics) -> f64 {
        // Weighted combination of synergy factors
        let complementarity_weight = 0.3;
        let diversity_weight = 0.2;
        let interaction_weight = 0.2;
        let stability_weight = 0.2;
        let anti_redundancy_weight = 0.1;
        
        complementarity_weight * metrics.complementarity +
        diversity_weight * metrics.diversity_contribution +
        interaction_weight * metrics.interaction_strength +
        stability_weight * metrics.stability_score +
        anti_redundancy_weight * (1.0 - metrics.redundancy)
    }
    
    fn score_combination(&self, indices: &[usize], matrix: &InteractionMatrix) -> f64 {
        if indices.len() < 2 {
            return 0.0;
        }
        
        let mut total_score = 0.0;
        let mut pair_count = 0;
        
        // Sum pairwise synergy scores
        for i in 0..indices.len() {
            for j in i + 1..indices.len() {
                let idx_i = indices[i];
                let idx_j = indices[j];
                total_score += matrix.interactions[[idx_i, idx_j]];
                pair_count += 1;
            }
        }
        
        if pair_count > 0 {
            total_score / pair_count as f64
        } else {
            0.0
        }
    }
}

impl Default for SynergyDetector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_synergy_detector_creation() {
        let detector = SynergyDetector::new();
        assert_eq!(detector.correlation_threshold, 0.7);
        assert_eq!(detector.analysis_depth, AnalysisDepth::Standard);
    }
    
    #[test]
    fn test_analyze_pair_basic() {
        let mut detector = SynergyDetector::new();
        
        let output_a = array![0.8, 0.6, 0.9, 0.3, 0.7];
        let output_b = array![0.7, 0.8, 0.6, 0.4, 0.9];
        
        let result = detector.analyze_pair(
            "algorithm_a",
            "algorithm_b",
            &output_a.view(),
            &output_b.view(),
        );
        
        assert!(result.is_ok());
        let interaction = result.unwrap();
        assert_eq!(interaction.algorithm_a, "algorithm_a");
        assert_eq!(interaction.algorithm_b, "algorithm_b");
    }
    
    #[test]
    fn test_interaction_matrix() {
        let mut detector = SynergyDetector::new();
        
        let algorithms = vec!["alg1".to_string(), "alg2".to_string(), "alg3".to_string()];
        let outputs = array![
            [0.8, 0.6, 0.9, 0.3, 0.7],
            [0.7, 0.8, 0.6, 0.4, 0.9],
            [0.9, 0.5, 0.8, 0.5, 0.6]
        ];
        
        let result = detector.analyze_matrix(&algorithms, &outputs.view());
        assert!(result.is_ok());
        
        let matrix = result.unwrap();
        assert_eq!(matrix.algorithms.len(), 3);
        assert_eq!(matrix.interactions.shape(), &[3, 3]);
    }
    
    #[test]
    fn test_find_optimal_combinations() {
        let mut detector = SynergyDetector::new();
        
        let algorithms = vec!["alg1".to_string(), "alg2".to_string(), "alg3".to_string()];
        let outputs = array![
            [0.8, 0.6, 0.9, 0.3, 0.7],
            [0.7, 0.8, 0.6, 0.4, 0.9],
            [0.9, 0.5, 0.8, 0.5, 0.6]
        ];
        
        let result = detector.find_optimal_combinations(&algorithms, &outputs.view(), 2);
        assert!(result.is_ok());
        
        let combinations = result.unwrap();
        assert!(!combinations.is_empty());
        assert!(combinations.iter().all(|combo| combo.len() == 2));
    }
}