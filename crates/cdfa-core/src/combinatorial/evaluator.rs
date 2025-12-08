//! Fusion Evaluator for K-Combinations Assessment
//! 
//! Evaluates the performance of different algorithm combinations,
//! providing comprehensive metrics and performance profiling.

use super::{CombinatorialResult, CombinatorialError, PerformanceBenchmark};
use crate::fusion::{CdfaFusion, FusionMethod, FusionParams};
use ndarray::{Array1, Array2, ArrayView2};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Evaluates fusion performance for algorithm combinations
pub struct FusionEvaluator {
    benchmark_enabled: bool,
    performance_cache: HashMap<String, PerformanceProfile>,
    evaluation_config: EvaluationConfig,
}

/// Configuration for fusion evaluation
#[derive(Debug, Clone)]
pub struct EvaluationConfig {
    /// Number of benchmark iterations for timing
    pub benchmark_iterations: usize,
    /// Enable statistical significance testing
    pub enable_statistical_tests: bool,
    /// Memory profiling enabled
    pub enable_memory_profiling: bool,
    /// Cache evaluation results
    pub enable_caching: bool,
    /// Evaluation timeout in milliseconds
    pub timeout_ms: u64,
}

impl Default for EvaluationConfig {
    fn default() -> Self {
        Self {
            benchmark_iterations: 100,
            enable_statistical_tests: true,
            enable_memory_profiling: true,
            enable_caching: true,
            timeout_ms: 1000, // 1 second timeout
        }
    }
}

/// Comprehensive evaluation metrics for a fusion combination
#[derive(Debug, Clone)]
pub struct EvaluationMetrics {
    /// Fusion accuracy metrics
    pub accuracy: AccuracyMetrics,
    /// Performance timing metrics
    pub performance: PerformanceMetrics,
    /// Memory usage metrics
    pub memory: MemoryMetrics,
    /// Statistical significance metrics
    pub statistics: StatisticalMetrics,
    /// Quality assessment
    pub quality: QualityMetrics,
}

/// Accuracy-related metrics
#[derive(Debug, Clone)]
pub struct AccuracyMetrics {
    /// Mean absolute error compared to ground truth
    pub mean_absolute_error: f64,
    /// Root mean square error
    pub root_mean_square_error: f64,
    /// Correlation with ground truth
    pub correlation_with_truth: f64,
    /// Ranking quality (if applicable)
    pub ranking_quality: Option<f64>,
    /// Consistency across different data subsets
    pub consistency_score: f64,
}

/// Performance timing metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    /// Average execution time in nanoseconds
    pub avg_execution_time_ns: u64,
    /// Minimum execution time
    pub min_execution_time_ns: u64,
    /// Maximum execution time
    pub max_execution_time_ns: u64,
    /// Standard deviation of execution times
    pub std_dev_ns: f64,
    /// Throughput (combinations per second)
    pub throughput: f64,
    /// Meets <1μs target
    pub meets_target: bool,
}

/// Memory usage metrics
#[derive(Debug, Clone)]
pub struct MemoryMetrics {
    /// Peak memory usage in bytes
    pub peak_memory_bytes: usize,
    /// Average memory usage
    pub avg_memory_bytes: usize,
    /// Memory efficiency score (output quality per byte)
    pub efficiency_score: f64,
    /// Number of allocations
    pub allocation_count: usize,
}

/// Statistical significance metrics
#[derive(Debug, Clone)]
pub struct StatisticalMetrics {
    /// Confidence interval for performance
    pub confidence_interval_95: (f64, f64),
    /// P-value for performance difference vs baseline
    pub p_value: Option<f64>,
    /// Effect size (Cohen's d)
    pub effect_size: Option<f64>,
    /// Statistical power
    pub statistical_power: Option<f64>,
}

/// Quality assessment metrics
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Overall quality score (0.0 to 1.0)
    pub overall_score: f64,
    /// Robustness to noisy data
    pub robustness_score: f64,
    /// Diversity preservation
    pub diversity_preservation: f64,
    /// Computational efficiency
    pub computational_efficiency: f64,
    /// Scalability assessment
    pub scalability_score: f64,
}

/// Performance profile for an algorithm combination
#[derive(Debug, Clone)]
pub struct PerformanceProfile {
    pub combination_id: String,
    pub algorithm_ids: Vec<String>,
    pub fusion_method: FusionMethod,
    pub evaluation_metrics: EvaluationMetrics,
    pub benchmark_results: PerformanceBenchmark,
    pub evaluation_timestamp: Instant,
    pub dataset_characteristics: DatasetCharacteristics,
}

/// Characteristics of the dataset used for evaluation
#[derive(Debug, Clone)]
pub struct DatasetCharacteristics {
    pub num_algorithms: usize,
    pub num_items: usize,
    pub data_range: (f64, f64),
    pub noise_level: f64,
    pub sparsity: f64,
}

impl FusionEvaluator {
    /// Create a new fusion evaluator
    pub fn new() -> Self {
        Self {
            benchmark_enabled: true,
            performance_cache: HashMap::new(),
            evaluation_config: EvaluationConfig::default(),
        }
    }
    
    /// Create evaluator with custom configuration
    pub fn with_config(config: EvaluationConfig) -> Self {
        Self {
            benchmark_enabled: true,
            performance_cache: HashMap::new(),
            evaluation_config: config,
        }
    }
    
    /// Evaluate a specific combination of algorithms
    pub fn evaluate_combination(
        &mut self,
        combination_id: &str,
        algorithm_ids: &[String],
        algorithm_outputs: &ArrayView2<f64>,
        fusion_method: FusionMethod,
        ground_truth: Option<&Array1<f64>>,
    ) -> CombinatorialResult<PerformanceProfile> {
        // Check cache first
        if self.evaluation_config.enable_caching {
            if let Some(cached) = self.performance_cache.get(combination_id) {
                return Ok(cached.clone());
            }
        }
        
        // Validate inputs
        if algorithm_outputs.nrows() != algorithm_ids.len() {
            return Err(CombinatorialError::EvaluationFailed {
                reason: "Number of algorithm IDs must match output rows".to_string(),
            });
        }
        
        let start_time = Instant::now();
        
        // Extract dataset characteristics
        let dataset_chars = self.analyze_dataset_characteristics(algorithm_outputs);
        
        // Perform fusion
        let fusion_result = self.perform_fusion_with_timing(algorithm_outputs, fusion_method)?;
        
        // Calculate evaluation metrics
        let evaluation_metrics = self.calculate_evaluation_metrics(
            algorithm_outputs,
            &fusion_result.result,
            ground_truth,
            &fusion_result.timing,
        )?;
        
        // Create benchmark results
        let benchmark_results = self.create_benchmark_results(&fusion_result.timing)?;
        
        let profile = PerformanceProfile {
            combination_id: combination_id.to_string(),
            algorithm_ids: algorithm_ids.to_vec(),
            fusion_method,
            evaluation_metrics,
            benchmark_results,
            evaluation_timestamp: start_time,
            dataset_characteristics: dataset_chars,
        };
        
        // Cache the result
        if self.evaluation_config.enable_caching {
            self.performance_cache.insert(combination_id.to_string(), profile.clone());
        }
        
        Ok(profile)
    }
    
    /// Evaluate multiple k-combinations in parallel
    pub fn evaluate_k_combinations(
        &mut self,
        algorithm_ids: &[String],
        algorithm_outputs: &ArrayView2<f64>,
        k: usize,
        fusion_methods: &[FusionMethod],
        ground_truth: Option<&Array1<f64>>,
    ) -> CombinatorialResult<Vec<PerformanceProfile>> {
        if k > algorithm_ids.len() {
            return Err(CombinatorialError::InvalidK {
                k,
                max: algorithm_ids.len(),
            });
        }
        
        // Generate all k-combinations
        use itertools::Itertools;
        let combinations: Vec<Vec<usize>> = (0..algorithm_ids.len())
            .combinations(k)
            .collect();
        
        let mut profiles = Vec::new();
        
        // Evaluate each combination with each fusion method
        for (combo_idx, indices) in combinations.iter().enumerate() {
            for fusion_method in fusion_methods {
                let combination_id = format!("combo_{}_{:?}", combo_idx, fusion_method);
                let combo_algorithm_ids: Vec<String> = indices.iter()
                    .map(|&i| algorithm_ids[i].clone())
                    .collect();
                
                // Extract relevant outputs
                let combo_outputs = self.extract_combination_outputs(algorithm_outputs, indices)?;
                
                match self.evaluate_combination(
                    &combination_id,
                    &combo_algorithm_ids,
                    &combo_outputs.view(),
                    *fusion_method,
                    ground_truth,
                ) {
                    Ok(profile) => profiles.push(profile),
                    Err(e) => {
                        eprintln!("Warning: Failed to evaluate combination {}: {:?}", combination_id, e);
                        continue;
                    }
                }
            }
        }
        
        Ok(profiles)
    }
    
    /// Rank combinations by performance
    pub fn rank_combinations(
        &self,
        profiles: &[PerformanceProfile],
        ranking_criteria: RankingCriteria,
    ) -> Vec<(usize, f64)> {
        let mut scored_profiles: Vec<(usize, f64)> = profiles
            .iter()
            .enumerate()
            .map(|(idx, profile)| {
                let score = self.calculate_ranking_score(profile, &ranking_criteria);
                (idx, score)
            })
            .collect();
        
        // Sort by score (descending)
        scored_profiles.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        scored_profiles
    }
    
    /// Generate comprehensive performance report
    pub fn generate_performance_report(
        &self,
        profiles: &[PerformanceProfile],
    ) -> PerformanceReport {
        if profiles.is_empty() {
            return PerformanceReport::empty();
        }
        
        let mut best_accuracy = f64::NEG_INFINITY;
        let mut best_performance = u64::MAX;
        let mut best_overall = f64::NEG_INFINITY;
        
        let mut best_accuracy_profile = None;
        let mut best_performance_profile = None;
        let mut best_overall_profile = None;
        
        let mut total_execution_time = 0u64;
        let mut target_meeting_count = 0;
        
        for profile in profiles {
            // Track best accuracy
            let accuracy_score = profile.evaluation_metrics.quality.overall_score;
            if accuracy_score > best_accuracy {
                best_accuracy = accuracy_score;
                best_accuracy_profile = Some(profile);
            }
            
            // Track best performance
            let exec_time = profile.evaluation_metrics.performance.avg_execution_time_ns;
            if exec_time < best_performance {
                best_performance = exec_time;
                best_performance_profile = Some(profile);
            }
            
            // Track best overall (balanced)
            let overall_score = self.calculate_overall_score(profile);
            if overall_score > best_overall {
                best_overall = overall_score;
                best_overall_profile = Some(profile);
            }
            
            total_execution_time += exec_time;
            if profile.evaluation_metrics.performance.meets_target {
                target_meeting_count += 1;
            }
        }
        
        let avg_execution_time = total_execution_time / profiles.len() as u64;
        let target_meeting_rate = target_meeting_count as f64 / profiles.len() as f64;
        
        PerformanceReport {
            total_combinations_evaluated: profiles.len(),
            best_accuracy_combination: best_accuracy_profile.map(|p| p.combination_id.clone()),
            best_performance_combination: best_performance_profile.map(|p| p.combination_id.clone()),
            best_overall_combination: best_overall_profile.map(|p| p.combination_id.clone()),
            average_execution_time_ns: avg_execution_time,
            target_meeting_rate,
            summary_statistics: self.calculate_summary_statistics(profiles),
        }
    }
    
    // Private helper methods
    
    fn perform_fusion_with_timing(
        &self,
        data: &ArrayView2<f64>,
        method: FusionMethod,
    ) -> CombinatorialResult<FusionResult> {
        let start = Instant::now();
        
        let result = CdfaFusion::fuse(data, method, None)
            .map_err(|e| CombinatorialError::EvaluationFailed {
                reason: format!("Fusion failed: {}", e),
            })?;
        
        let execution_time = start.elapsed();
        
        Ok(FusionResult {
            result,
            timing: TimingResult {
                execution_time,
                meets_target: execution_time.as_nanos() < 1_000, // <1μs
            },
        })
    }
    
    fn calculate_evaluation_metrics(
        &self,
        _inputs: &ArrayView2<f64>,
        fusion_result: &Array1<f64>,
        ground_truth: Option<&Array1<f64>>,
        timing: &TimingResult,
    ) -> CombinatorialResult<EvaluationMetrics> {
        // Calculate accuracy metrics
        let accuracy = if let Some(truth) = ground_truth {
            self.calculate_accuracy_metrics(fusion_result, truth)?
        } else {
            AccuracyMetrics {
                mean_absolute_error: 0.0,
                root_mean_square_error: 0.0,
                correlation_with_truth: 0.0,
                ranking_quality: None,
                consistency_score: 0.5,
            }
        };
        
        // Calculate performance metrics
        let performance = PerformanceMetrics {
            avg_execution_time_ns: timing.execution_time.as_nanos() as u64,
            min_execution_time_ns: timing.execution_time.as_nanos() as u64,
            max_execution_time_ns: timing.execution_time.as_nanos() as u64,
            std_dev_ns: 0.0,
            throughput: 1_000_000_000.0 / timing.execution_time.as_nanos() as f64,
            meets_target: timing.meets_target,
        };
        
        // Calculate memory metrics (simplified)
        let memory = MemoryMetrics {
            peak_memory_bytes: fusion_result.len() * 8, // 8 bytes per f64
            avg_memory_bytes: fusion_result.len() * 8,
            efficiency_score: 0.8,
            allocation_count: 1,
        };
        
        // Calculate statistical metrics (simplified)
        let statistics = StatisticalMetrics {
            confidence_interval_95: (0.0, 1.0),
            p_value: None,
            effect_size: None,
            statistical_power: None,
        };
        
        // Calculate quality metrics
        let quality = self.calculate_quality_metrics(fusion_result, &accuracy, &performance)?;
        
        Ok(EvaluationMetrics {
            accuracy,
            performance,
            memory,
            statistics,
            quality,
        })
    }
    
    fn calculate_accuracy_metrics(
        &self,
        fusion_result: &Array1<f64>,
        ground_truth: &Array1<f64>,
    ) -> CombinatorialResult<AccuracyMetrics> {
        if fusion_result.len() != ground_truth.len() {
            return Err(CombinatorialError::EvaluationFailed {
                reason: "Fusion result and ground truth must have same length".to_string(),
            });
        }
        
        let n = fusion_result.len() as f64;
        
        // Mean Absolute Error
        let mae = fusion_result.iter()
            .zip(ground_truth.iter())
            .map(|(pred, truth)| (pred - truth).abs())
            .sum::<f64>() / n;
        
        // Root Mean Square Error
        let rmse = (fusion_result.iter()
            .zip(ground_truth.iter())
            .map(|(pred, truth)| (pred - truth).powi(2))
            .sum::<f64>() / n).sqrt();
        
        // Correlation with ground truth
        let correlation = crate::diversity::pearson_correlation(
            &fusion_result.view(),
            &ground_truth.view(),
        ).unwrap_or(0.0);
        
        // Consistency score (placeholder)
        let consistency_score = 1.0 - (mae / (ground_truth.mean().unwrap_or(1.0) + 1e-10));
        
        Ok(AccuracyMetrics {
            mean_absolute_error: mae,
            root_mean_square_error: rmse,
            correlation_with_truth: correlation,
            ranking_quality: None,
            consistency_score: consistency_score.max(0.0).min(1.0),
        })
    }
    
    fn calculate_quality_metrics(
        &self,
        fusion_result: &Array1<f64>,
        accuracy: &AccuracyMetrics,
        performance: &PerformanceMetrics,
    ) -> CombinatorialResult<QualityMetrics> {
        // Overall score combines accuracy and performance
        let accuracy_score = 1.0 - accuracy.mean_absolute_error.min(1.0);
        let performance_score = if performance.meets_target { 1.0 } else { 0.5 };
        let overall_score = (accuracy_score + performance_score) / 2.0;
        
        // Robustness (based on consistency)
        let robustness_score = accuracy.consistency_score;
        
        // Diversity preservation (placeholder)
        let diversity_preservation = 0.8;
        
        // Computational efficiency
        let target_time_ns = 1000.0; // 1μs
        let efficiency = (target_time_ns / performance.avg_execution_time_ns as f64).min(1.0);
        
        // Scalability (based on algorithmic complexity)
        let scalability_score = 0.7; // Placeholder
        
        Ok(QualityMetrics {
            overall_score,
            robustness_score,
            diversity_preservation,
            computational_efficiency: efficiency,
            scalability_score,
        })
    }
    
    fn analyze_dataset_characteristics(&self, data: &ArrayView2<f64>) -> DatasetCharacteristics {
        let num_algorithms = data.nrows();
        let num_items = data.ncols();
        
        let all_values: Vec<f64> = data.iter().cloned().collect();
        let min_val = all_values.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_val = all_values.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        
        // Estimate noise level as standard deviation
        let mean = all_values.iter().sum::<f64>() / all_values.len() as f64;
        let variance = all_values.iter()
            .map(|&x| (x - mean).powi(2))
            .sum::<f64>() / all_values.len() as f64;
        let noise_level = variance.sqrt();
        
        // Estimate sparsity (fraction of near-zero values)
        let near_zero_count = all_values.iter().filter(|&&x| x.abs() < 0.01).count();
        let sparsity = near_zero_count as f64 / all_values.len() as f64;
        
        DatasetCharacteristics {
            num_algorithms,
            num_items,
            data_range: (min_val, max_val),
            noise_level,
            sparsity,
        }
    }
    
    fn extract_combination_outputs(
        &self,
        full_outputs: &ArrayView2<f64>,
        indices: &[usize],
    ) -> CombinatorialResult<Array2<f64>> {
        let num_items = full_outputs.ncols();
        let mut combo_outputs = Array2::zeros((indices.len(), num_items));
        
        for (new_idx, &orig_idx) in indices.iter().enumerate() {
            if orig_idx >= full_outputs.nrows() {
                return Err(CombinatorialError::EvaluationFailed {
                    reason: format!("Invalid algorithm index: {}", orig_idx),
                });
            }
            combo_outputs.row_mut(new_idx).assign(&full_outputs.row(orig_idx));
        }
        
        Ok(combo_outputs)
    }
    
    fn create_benchmark_results(&self, timing: &TimingResult) -> CombinatorialResult<PerformanceBenchmark> {
        Ok(PerformanceBenchmark {
            fusion_time_ns: timing.execution_time.as_nanos() as u64,
            combination_generation_time_ns: 0,
            synergy_analysis_time_ns: 0,
            total_time_ns: timing.execution_time.as_nanos() as u64,
            memory_usage_bytes: 0,
            cache_hits: 0,
            cache_misses: 0,
        })
    }
    
    fn calculate_ranking_score(&self, profile: &PerformanceProfile, criteria: &RankingCriteria) -> f64 {
        match criteria {
            RankingCriteria::Accuracy => profile.evaluation_metrics.quality.overall_score,
            RankingCriteria::Performance => {
                if profile.evaluation_metrics.performance.meets_target { 1.0 } else { 0.0 }
            },
            RankingCriteria::Balanced => self.calculate_overall_score(profile),
            RankingCriteria::MemoryEfficiency => profile.evaluation_metrics.memory.efficiency_score,
        }
    }
    
    fn calculate_overall_score(&self, profile: &PerformanceProfile) -> f64 {
        let accuracy_weight = 0.4;
        let performance_weight = 0.3;
        let quality_weight = 0.2;
        let efficiency_weight = 0.1;
        
        let accuracy_score = profile.evaluation_metrics.quality.overall_score;
        let performance_score = if profile.evaluation_metrics.performance.meets_target { 1.0 } else { 0.5 };
        let quality_score = profile.evaluation_metrics.quality.robustness_score;
        let efficiency_score = profile.evaluation_metrics.memory.efficiency_score;
        
        accuracy_weight * accuracy_score +
        performance_weight * performance_score +
        quality_weight * quality_score +
        efficiency_weight * efficiency_score
    }
    
    fn calculate_summary_statistics(&self, profiles: &[PerformanceProfile]) -> SummaryStatistics {
        if profiles.is_empty() {
            return SummaryStatistics::default();
        }
        
        let exec_times: Vec<u64> = profiles.iter()
            .map(|p| p.evaluation_metrics.performance.avg_execution_time_ns)
            .collect();
        
        let quality_scores: Vec<f64> = profiles.iter()
            .map(|p| p.evaluation_metrics.quality.overall_score)
            .collect();
        
        let avg_exec_time = exec_times.iter().sum::<u64>() / exec_times.len() as u64;
        let avg_quality = quality_scores.iter().sum::<f64>() / quality_scores.len() as f64;
        
        SummaryStatistics {
            avg_execution_time_ns: avg_exec_time,
            avg_quality_score: avg_quality,
            std_dev_execution_time: 0.0, // Simplified
            std_dev_quality: 0.0,       // Simplified
        }
    }
}

/// Criteria for ranking combinations
#[derive(Debug, Clone, Copy)]
pub enum RankingCriteria {
    Accuracy,
    Performance,
    Balanced,
    MemoryEfficiency,
}

/// Result of fusion operation with timing
struct FusionResult {
    result: Array1<f64>,
    timing: TimingResult,
}

/// Timing result for fusion operation
struct TimingResult {
    execution_time: Duration,
    meets_target: bool,
}

/// Comprehensive performance report
#[derive(Debug, Clone)]
pub struct PerformanceReport {
    pub total_combinations_evaluated: usize,
    pub best_accuracy_combination: Option<String>,
    pub best_performance_combination: Option<String>,
    pub best_overall_combination: Option<String>,
    pub average_execution_time_ns: u64,
    pub target_meeting_rate: f64,
    pub summary_statistics: SummaryStatistics,
}

/// Summary statistics across all evaluated combinations
#[derive(Debug, Clone, Default)]
pub struct SummaryStatistics {
    pub avg_execution_time_ns: u64,
    pub avg_quality_score: f64,
    pub std_dev_execution_time: f64,
    pub std_dev_quality: f64,
}

impl PerformanceReport {
    fn empty() -> Self {
        Self {
            total_combinations_evaluated: 0,
            best_accuracy_combination: None,
            best_performance_combination: None,
            best_overall_combination: None,
            average_execution_time_ns: 0,
            target_meeting_rate: 0.0,
            summary_statistics: SummaryStatistics::default(),
        }
    }
}

impl Default for FusionEvaluator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_fusion_evaluator_creation() {
        let evaluator = FusionEvaluator::new();
        assert!(evaluator.benchmark_enabled);
    }
    
    #[test]
    fn test_evaluate_combination() {
        let mut evaluator = FusionEvaluator::new();
        
        let algorithm_ids = vec!["alg1".to_string(), "alg2".to_string()];
        let outputs = array![
            [0.8, 0.6, 0.9, 0.3, 0.7],
            [0.7, 0.8, 0.6, 0.4, 0.9]
        ];
        
        let result = evaluator.evaluate_combination(
            "test_combo",
            &algorithm_ids,
            &outputs.view(),
            FusionMethod::Average,
            None,
        );
        
        assert!(result.is_ok());
        let profile = result.unwrap();
        assert_eq!(profile.combination_id, "test_combo");
        assert_eq!(profile.algorithm_ids.len(), 2);
    }
    
    #[test]
    fn test_k_combinations_evaluation() {
        let mut evaluator = FusionEvaluator::new();
        
        let algorithm_ids = vec!["alg1".to_string(), "alg2".to_string(), "alg3".to_string()];
        let outputs = array![
            [0.8, 0.6, 0.9, 0.3, 0.7],
            [0.7, 0.8, 0.6, 0.4, 0.9],
            [0.9, 0.5, 0.8, 0.5, 0.6]
        ];
        
        let fusion_methods = vec![FusionMethod::Average, FusionMethod::BordaCount];
        
        let result = evaluator.evaluate_k_combinations(
            &algorithm_ids,
            &outputs.view(),
            2,
            &fusion_methods,
            None,
        );
        
        assert!(result.is_ok());
        let profiles = result.unwrap();
        assert!(!profiles.is_empty());
        // 3 choose 2 = 3 combinations * 2 methods = 6 profiles
        assert_eq!(profiles.len(), 6);
    }
}