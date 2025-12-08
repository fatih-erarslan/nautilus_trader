//! Combinatorial Diversity Fusion Analyzer
//! 
//! The main orchestrator for combinatorial fusion analysis,
//! integrating algorithm pools, synergy detection, and performance evaluation.

use super::{
    CombinatorialResult, CombinatorialError, CombinatorialConfig, PerformanceBenchmark,
    algorithm_pool::{AlgorithmPool, SwarmAlgorithm, BuiltinFusionAlgorithm},
    synergy::{SynergyDetector, InteractionMatrix, AlgorithmInteraction},
    evaluator::{FusionEvaluator, PerformanceProfile, EvaluationConfig, RankingCriteria},
};
use crate::fusion::FusionMethod;
use ndarray::{Array1, Array2, ArrayView2};
use std::collections::HashMap;
use std::time::Instant;

/// Main combinatorial diversity fusion analyzer
pub struct CombinatorialDiversityFusionAnalyzer {
    /// Pool of algorithms available for combination
    algorithm_pool: AlgorithmPool,
    /// Synergy detector for algorithm interaction analysis
    synergy_detector: SynergyDetector,
    /// Fusion evaluator for performance assessment
    fusion_evaluator: FusionEvaluator,
    /// Configuration settings
    config: CombinatorialConfig,
    /// Analysis cache for performance optimization
    analysis_cache: AnalysisCache,
    /// Performance tracking
    performance_tracker: PerformanceTracker,
}

/// Result of a complete combinatorial analysis
#[derive(Debug, Clone)]
pub struct CombinationAnalysis {
    /// All evaluated combinations and their profiles
    pub combination_profiles: Vec<PerformanceProfile>,
    /// Interaction matrix between algorithms
    pub interaction_matrix: InteractionMatrix,
    /// Optimal combinations for different criteria
    pub optimal_combinations: OptimalCombinations,
    /// Overall analysis summary
    pub analysis_summary: AnalysisSummary,
    /// Performance benchmarks
    pub performance_benchmarks: PerformanceBenchmark,
}

/// Result of a single combination evaluation
#[derive(Debug, Clone)]
pub struct CombinationResult {
    /// Fusion result
    pub fusion_result: Array1<f64>,
    /// Algorithms used in this combination
    pub algorithm_ids: Vec<String>,
    /// Fusion method applied
    pub fusion_method: FusionMethod,
    /// Performance metrics
    pub performance_profile: PerformanceProfile,
    /// Synergy analysis
    pub synergy_analysis: Vec<AlgorithmInteraction>,
    /// Computational cost
    pub computational_cost: ComputationalCost,
}

/// Optimal combinations for different optimization criteria
#[derive(Debug, Clone)]
pub struct OptimalCombinations {
    /// Best combination for accuracy
    pub best_accuracy: Option<String>,
    /// Best combination for performance (<1μs requirement)
    pub best_performance: Option<String>,
    /// Best overall combination (balanced)
    pub best_overall: Option<String>,
    /// Most diverse combination
    pub most_diverse: Option<String>,
    /// Most synergistic combination
    pub most_synergistic: Option<String>,
}

/// Summary of the complete analysis
#[derive(Debug, Clone)]
pub struct AnalysisSummary {
    /// Total combinations evaluated
    pub total_combinations: usize,
    /// Number meeting performance target
    pub target_meeting_combinations: usize,
    /// Average synergy score
    pub average_synergy_score: f64,
    /// Diversity distribution across combinations
    pub diversity_distribution: DiversityDistribution,
    /// Recommendations for algorithm selection
    pub recommendations: Vec<String>,
}

/// Distribution of diversity scores
#[derive(Debug, Clone)]
pub struct DiversityDistribution {
    pub min_diversity: f64,
    pub max_diversity: f64,
    pub mean_diversity: f64,
    pub std_dev_diversity: f64,
    pub quartiles: [f64; 3], // Q1, Q2 (median), Q3
}

/// Computational cost breakdown
#[derive(Debug, Clone)]
pub struct ComputationalCost {
    /// Time for combination generation
    pub combination_generation_ns: u64,
    /// Time for fusion computation
    pub fusion_computation_ns: u64,
    /// Time for synergy analysis
    pub synergy_analysis_ns: u64,
    /// Total analysis time
    pub total_analysis_ns: u64,
    /// Memory usage
    pub memory_usage_bytes: usize,
}

/// Cache for analysis results
struct AnalysisCache {
    combination_cache: HashMap<String, CombinationResult>,
    synergy_cache: HashMap<String, AlgorithmInteraction>,
    max_cache_size: usize,
}

/// Performance tracking across analyses
struct PerformanceTracker {
    total_analyses: usize,
    total_time_ns: u64,
    cache_hit_rate: f64,
    performance_history: Vec<PerformanceBenchmark>,
}

impl CombinatorialDiversityFusionAnalyzer {
    /// Create a new combinatorial diversity fusion analyzer
    pub fn new() -> Self {
        let mut algorithm_pool = AlgorithmPool::new();
        
        // Add built-in fusion algorithms
        let builtin_methods = vec![
            FusionMethod::Average,
            FusionMethod::WeightedAverage,
            FusionMethod::BordaCount,
            FusionMethod::MedianRank,
            FusionMethod::Hybrid,
            FusionMethod::Adaptive,
        ];
        
        for method in builtin_methods {
            let algorithm = Box::new(BuiltinFusionAlgorithm::new(method));
            let _ = algorithm_pool.add_algorithm(algorithm);
        }
        
        Self {
            algorithm_pool,
            synergy_detector: SynergyDetector::new(),
            fusion_evaluator: FusionEvaluator::new(),
            config: CombinatorialConfig::default(),
            analysis_cache: AnalysisCache::new(1000),
            performance_tracker: PerformanceTracker::new(),
        }
    }
    
    /// Create analyzer with custom configuration
    pub fn with_config(config: CombinatorialConfig) -> Self {
        let mut analyzer = Self::new();
        
        // Configure sub-components based on config before moving
        let eval_config = EvaluationConfig {
            enable_statistical_tests: true,
            enable_memory_profiling: config.parallel_evaluation,
            enable_caching: true,
            ..Default::default()
        };
        
        analyzer.config = config;
        analyzer.fusion_evaluator = FusionEvaluator::with_config(eval_config);
        
        analyzer
    }
    
    /// Add a custom algorithm to the pool
    pub fn add_algorithm(&mut self, algorithm: Box<dyn SwarmAlgorithm>) -> CombinatorialResult<()> {
        self.algorithm_pool.add_algorithm(algorithm)
    }
    
    /// Perform complete combinatorial diversity fusion analysis
    pub fn analyze_combinations(
        &mut self,
        data: &ArrayView2<f64>,
        ground_truth: Option<&Array1<f64>>,
    ) -> CombinatorialResult<CombinationAnalysis> {
        let analysis_start = Instant::now();
        let mut benchmark = PerformanceBenchmark::new();
        
        // Validate input data
        if data.nrows() < 2 {
            return Err(CombinatorialError::InsufficientAlgorithms {
                count: data.nrows(),
                min: 2,
            });
        }
        
        if data.ncols() == 0 {
            return Err(CombinatorialError::EvaluationFailed {
                reason: "Empty data provided".to_string(),
            });
        }
        
        // Get available algorithms
        let algorithm_ids = self.algorithm_pool.list_algorithms();
        if algorithm_ids.len() < 2 {
            return Err(CombinatorialError::InsufficientAlgorithms {
                count: algorithm_ids.len(),
                min: 2,
            });
        }
        
        // Step 1: Generate algorithm outputs for all algorithms in pool
        let pool_outputs = self.generate_pool_outputs(data)?;
        benchmark.combination_generation_time_ns = 100; // Placeholder
        
        // Step 2: Analyze synergy between all algorithm pairs
        let synergy_start = Instant::now();
        let interaction_matrix = self.synergy_detector
            .analyze_matrix(&algorithm_ids, &pool_outputs.view())?;
        benchmark.synergy_analysis_time_ns = synergy_start.elapsed().as_nanos() as u64;
        
        // Step 3: Evaluate k-combinations for all k from 2 to max_k
        let fusion_start = Instant::now();
        let mut all_profiles = Vec::new();
        
        for k in 2..=self.config.max_k.min(algorithm_ids.len()) {
            let fusion_methods = self.select_fusion_methods_for_k(k);
            
            let k_profiles = self.fusion_evaluator.evaluate_k_combinations(
                &algorithm_ids,
                &pool_outputs.view(),
                k,
                &fusion_methods,
                ground_truth,
            )?;
            
            all_profiles.extend(k_profiles);
        }
        benchmark.fusion_time_ns = fusion_start.elapsed().as_nanos() as u64;
        
        // Step 4: Find optimal combinations
        let optimal_combinations = self.find_optimal_combinations(&all_profiles, &interaction_matrix)?;
        
        // Step 5: Generate analysis summary
        let analysis_summary = self.generate_analysis_summary(&all_profiles, &interaction_matrix)?;
        
        // Update performance tracking
        benchmark.total_time_ns = analysis_start.elapsed().as_nanos() as u64;
        benchmark.memory_usage_bytes = self.estimate_memory_usage(&all_profiles);
        
        self.performance_tracker.record_analysis(&benchmark);
        
        Ok(CombinationAnalysis {
            combination_profiles: all_profiles,
            interaction_matrix,
            optimal_combinations,
            analysis_summary,
            performance_benchmarks: benchmark,
        })
    }
    
    /// Analyze a specific combination of algorithms
    pub fn analyze_single_combination(
        &mut self,
        algorithm_ids: &[String],
        data: &ArrayView2<f64>,
        fusion_method: FusionMethod,
        ground_truth: Option<&Array1<f64>>,
    ) -> CombinatorialResult<CombinationResult> {
        let start_time = Instant::now();
        
        // Validate inputs
        if algorithm_ids.len() < 2 {
            return Err(CombinatorialError::InsufficientAlgorithms {
                count: algorithm_ids.len(),
                min: 2,
            });
        }
        
        // Generate cache key
        let cache_key = self.generate_cache_key(algorithm_ids, fusion_method);
        
        // Check cache
        if let Some(cached_result) = self.analysis_cache.get(&cache_key) {
            return Ok(cached_result.clone());
        }
        
        // Extract algorithm outputs
        let mut algorithm_outputs = Array2::zeros((algorithm_ids.len(), data.ncols()));
        
        for (i, id) in algorithm_ids.iter().enumerate() {
            let output = self.algorithm_pool.execute_algorithm(id, data)?;
            algorithm_outputs.row_mut(i).assign(&output);
        }
        
        // Perform fusion
        let fusion_start = Instant::now();
        use crate::fusion::CdfaFusion;
        let fusion_result = CdfaFusion::fuse(&algorithm_outputs.view(), fusion_method, None)
            .map_err(|e| CombinatorialError::EvaluationFailed {
                reason: format!("Fusion failed: {}", e),
            })?;
        let fusion_time = fusion_start.elapsed().as_nanos() as u64;
        
        // Analyze synergy between algorithms in combination
        let synergy_start = Instant::now();
        let mut synergy_analysis = Vec::new();
        for i in 0..algorithm_ids.len() {
            for j in i + 1..algorithm_ids.len() {
                let interaction = self.synergy_detector.analyze_pair(
                    &algorithm_ids[i],
                    &algorithm_ids[j],
                    &algorithm_outputs.row(i),
                    &algorithm_outputs.row(j),
                )?;
                synergy_analysis.push(interaction);
            }
        }
        let synergy_time = synergy_start.elapsed().as_nanos() as u64;
        
        // Evaluate performance
        let combination_id = format!("{:?}_{:?}", algorithm_ids, fusion_method);
        let performance_profile = self.fusion_evaluator.evaluate_combination(
            &combination_id,
            algorithm_ids,
            &algorithm_outputs.view(),
            fusion_method,
            ground_truth,
        )?;
        
        // Calculate computational cost
        let total_time = start_time.elapsed().as_nanos() as u64;
        let computational_cost = ComputationalCost {
            combination_generation_ns: 0, // Not applicable for single combination
            fusion_computation_ns: fusion_time,
            synergy_analysis_ns: synergy_time,
            total_analysis_ns: total_time,
            memory_usage_bytes: algorithm_outputs.len() * 8 + fusion_result.len() * 8,
        };
        
        let result = CombinationResult {
            fusion_result,
            algorithm_ids: algorithm_ids.to_vec(),
            fusion_method,
            performance_profile,
            synergy_analysis,
            computational_cost,
        };
        
        // Cache the result
        self.analysis_cache.insert(cache_key, result.clone());
        
        Ok(result)
    }
    
    /// Find optimal k-combinations using synergy-guided selection
    pub fn find_synergy_guided_combinations(
        &mut self,
        data: &ArrayView2<f64>,
        k: usize,
        max_combinations: usize,
    ) -> CombinatorialResult<Vec<CombinationResult>> {
        if k > self.algorithm_pool.size() {
            return Err(CombinatorialError::InvalidK {
                k,
                max: self.algorithm_pool.size(),
            });
        }
        
        // Get algorithm pool outputs
        let algorithm_ids = self.algorithm_pool.list_algorithms();
        let pool_outputs = self.generate_pool_outputs(data)?;
        
        // Find optimal combinations based on synergy
        let optimal_combinations = self.synergy_detector
            .find_optimal_combinations(&algorithm_ids, &pool_outputs.view(), k)?;
        
        let combinations_to_evaluate = optimal_combinations
            .into_iter()
            .take(max_combinations)
            .collect::<Vec<_>>();
        
        // Evaluate each optimal combination
        let mut results = Vec::new();
        let fusion_method = FusionMethod::Adaptive; // Use adaptive method for synergy-guided selection
        
        for combination in combinations_to_evaluate {
            match self.analyze_single_combination(
                &combination,
                data,
                fusion_method,
                None,
            ) {
                Ok(result) => results.push(result),
                Err(e) => {
                    eprintln!("Warning: Failed to analyze combination {:?}: {:?}", combination, e);
                    continue;
                }
            }
        }
        
        Ok(results)
    }
    
    /// Get performance statistics for the analyzer
    pub fn get_performance_statistics(&self) -> AnalyzerPerformanceStats {
        AnalyzerPerformanceStats {
            total_analyses: self.performance_tracker.total_analyses,
            average_analysis_time_ns: if self.performance_tracker.total_analyses > 0 {
                self.performance_tracker.total_time_ns / self.performance_tracker.total_analyses as u64
            } else {
                0
            },
            cache_hit_rate: self.performance_tracker.cache_hit_rate,
            algorithm_pool_size: self.algorithm_pool.size(),
            cache_size: self.analysis_cache.size(),
            performance_target_achievement_rate: self.calculate_target_achievement_rate(),
        }
    }
    
    // Private helper methods
    
    fn generate_pool_outputs(&mut self, data: &ArrayView2<f64>) -> CombinatorialResult<Array2<f64>> {
        let algorithm_ids = self.algorithm_pool.list_algorithms();
        let num_algorithms = algorithm_ids.len();
        let num_items = data.ncols();
        
        let mut pool_outputs = Array2::zeros((num_algorithms, num_items));
        
        for (i, id) in algorithm_ids.iter().enumerate() {
            let output = self.algorithm_pool.execute_algorithm(id, data)?;
            pool_outputs.row_mut(i).assign(&output);
        }
        
        Ok(pool_outputs)
    }
    
    fn select_fusion_methods_for_k(&self, k: usize) -> Vec<FusionMethod> {
        // Select appropriate fusion methods based on k
        let mut methods = vec![FusionMethod::Average, FusionMethod::BordaCount];
        
        if k <= 3 {
            methods.push(FusionMethod::Median);
            methods.push(FusionMethod::Hybrid);
        }
        
        if k <= 5 {
            methods.push(FusionMethod::Adaptive);
        }
        
        methods
    }
    
    fn find_optimal_combinations(
        &self,
        profiles: &[PerformanceProfile],
        interaction_matrix: &InteractionMatrix,
    ) -> CombinatorialResult<OptimalCombinations> {
        if profiles.is_empty() {
            return Ok(OptimalCombinations {
                best_accuracy: None,
                best_performance: None,
                best_overall: None,
                most_diverse: None,
                most_synergistic: None,
            });
        }
        
        // Find best combinations for different criteria
        let best_accuracy = self.find_best_by_criteria(profiles, RankingCriteria::Accuracy);
        let best_performance = self.find_best_by_criteria(profiles, RankingCriteria::Performance);
        let best_overall = self.find_best_by_criteria(profiles, RankingCriteria::Balanced);
        
        // Find most diverse combination
        let most_diverse = self.find_most_diverse_combination(profiles, interaction_matrix);
        
        // Find most synergistic combination
        let most_synergistic = self.find_most_synergistic_combination(profiles, interaction_matrix);
        
        Ok(OptimalCombinations {
            best_accuracy,
            best_performance,
            best_overall,
            most_diverse,
            most_synergistic,
        })
    }
    
    fn find_best_by_criteria(&self, profiles: &[PerformanceProfile], criteria: RankingCriteria) -> Option<String> {
        let ranked = self.fusion_evaluator.rank_combinations(profiles, criteria);
        ranked.first().map(|(idx, _)| profiles[*idx].combination_id.clone())
    }
    
    fn find_most_diverse_combination(
        &self,
        profiles: &[PerformanceProfile],
        _interaction_matrix: &InteractionMatrix,
    ) -> Option<String> {
        // Find combination with highest diversity score
        profiles.iter()
            .max_by(|a, b| {
                a.evaluation_metrics.quality.diversity_preservation
                    .partial_cmp(&b.evaluation_metrics.quality.diversity_preservation)
                    .unwrap()
            })
            .map(|p| p.combination_id.clone())
    }
    
    fn find_most_synergistic_combination(
        &self,
        profiles: &[PerformanceProfile],
        interaction_matrix: &InteractionMatrix,
    ) -> Option<String> {
        // Find combination with highest average synergy score
        let mut best_combo = None;
        let mut best_synergy = f64::NEG_INFINITY;
        
        for profile in profiles {
            let synergy_score = self.calculate_combination_synergy(&profile.algorithm_ids, interaction_matrix);
            if synergy_score > best_synergy {
                best_synergy = synergy_score;
                best_combo = Some(profile.combination_id.clone());
            }
        }
        
        best_combo
    }
    
    fn calculate_combination_synergy(&self, algorithm_ids: &[String], matrix: &InteractionMatrix) -> f64 {
        if algorithm_ids.len() < 2 {
            return 0.0;
        }
        
        let mut total_synergy = 0.0;
        let mut pair_count = 0;
        
        for i in 0..algorithm_ids.len() {
            for j in i + 1..algorithm_ids.len() {
                if let (Some(idx_i), Some(idx_j)) = (
                    matrix.algorithms.iter().position(|id| id == &algorithm_ids[i]),
                    matrix.algorithms.iter().position(|id| id == &algorithm_ids[j]),
                ) {
                    total_synergy += matrix.interactions[[idx_i, idx_j]];
                    pair_count += 1;
                }
            }
        }
        
        if pair_count > 0 {
            total_synergy / pair_count as f64
        } else {
            0.0
        }
    }
    
    fn generate_analysis_summary(
        &self,
        profiles: &[PerformanceProfile],
        interaction_matrix: &InteractionMatrix,
    ) -> CombinatorialResult<AnalysisSummary> {
        let total_combinations = profiles.len();
        
        let target_meeting_combinations = profiles.iter()
            .filter(|p| p.evaluation_metrics.performance.meets_target)
            .count();
        
        let average_synergy_score = interaction_matrix.summary_stats.avg_synergy;
        
        // Calculate diversity distribution
        let diversity_scores: Vec<f64> = profiles.iter()
            .map(|p| p.evaluation_metrics.quality.diversity_preservation)
            .collect();
        
        let diversity_distribution = self.calculate_diversity_distribution(&diversity_scores);
        
        // Generate recommendations
        let recommendations = self.generate_recommendations(profiles, interaction_matrix);
        
        Ok(AnalysisSummary {
            total_combinations,
            target_meeting_combinations,
            average_synergy_score,
            diversity_distribution,
            recommendations,
        })
    }
    
    fn calculate_diversity_distribution(&self, scores: &[f64]) -> DiversityDistribution {
        if scores.is_empty() {
            return DiversityDistribution {
                min_diversity: 0.0,
                max_diversity: 0.0,
                mean_diversity: 0.0,
                std_dev_diversity: 0.0,
                quartiles: [0.0, 0.0, 0.0],
            };
        }
        
        let mut sorted_scores = scores.to_vec();
        sorted_scores.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let min_diversity = sorted_scores[0];
        let max_diversity = sorted_scores[sorted_scores.len() - 1];
        let mean_diversity = scores.iter().sum::<f64>() / scores.len() as f64;
        
        let variance = scores.iter()
            .map(|&x| (x - mean_diversity).powi(2))
            .sum::<f64>() / scores.len() as f64;
        let std_dev_diversity = variance.sqrt();
        
        let q1_idx = sorted_scores.len() / 4;
        let q2_idx = sorted_scores.len() / 2;
        let q3_idx = 3 * sorted_scores.len() / 4;
        
        let quartiles = [
            sorted_scores[q1_idx],
            sorted_scores[q2_idx],
            sorted_scores[q3_idx],
        ];
        
        DiversityDistribution {
            min_diversity,
            max_diversity,
            mean_diversity,
            std_dev_diversity,
            quartiles,
        }
    }
    
    fn generate_recommendations(
        &self,
        profiles: &[PerformanceProfile],
        interaction_matrix: &InteractionMatrix,
    ) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        let target_meeting_rate = profiles.iter()
            .filter(|p| p.evaluation_metrics.performance.meets_target)
            .count() as f64 / profiles.len() as f64;
        
        if target_meeting_rate < 0.5 {
            recommendations.push(
                "Consider algorithm optimization - less than 50% of combinations meet <1μs target".to_string()
            );
        }
        
        if interaction_matrix.summary_stats.redundancy_count > interaction_matrix.summary_stats.synergistic_count {
            recommendations.push(
                "High redundancy detected - consider reducing algorithm pool size".to_string()
            );
        }
        
        if interaction_matrix.summary_stats.avg_synergy < 0.3 {
            recommendations.push(
                "Low average synergy - consider adding more diverse algorithms to pool".to_string()
            );
        }
        
        recommendations.push(
            format!("Best synergy achieved with {} combinations out of {}", 
                interaction_matrix.summary_stats.synergistic_count, 
                profiles.len())
        );
        
        recommendations
    }
    
    fn generate_cache_key(&self, algorithm_ids: &[String], fusion_method: FusionMethod) -> String {
        format!("{:?}_{:?}", algorithm_ids, fusion_method)
    }
    
    fn estimate_memory_usage(&self, profiles: &[PerformanceProfile]) -> usize {
        profiles.iter()
            .map(|p| p.evaluation_metrics.memory.peak_memory_bytes)
            .sum()
    }
    
    fn calculate_target_achievement_rate(&self) -> f64 {
        if self.performance_tracker.performance_history.is_empty() {
            return 0.0;
        }
        
        let meeting_target = self.performance_tracker.performance_history
            .iter()
            .filter(|b| b.meets_performance_target())
            .count();
        
        meeting_target as f64 / self.performance_tracker.performance_history.len() as f64
    }
}

/// Performance statistics for the analyzer
#[derive(Debug, Clone)]
pub struct AnalyzerPerformanceStats {
    pub total_analyses: usize,
    pub average_analysis_time_ns: u64,
    pub cache_hit_rate: f64,
    pub algorithm_pool_size: usize,
    pub cache_size: usize,
    pub performance_target_achievement_rate: f64,
}

impl AnalysisCache {
    fn new(max_size: usize) -> Self {
        Self {
            combination_cache: HashMap::new(),
            synergy_cache: HashMap::new(),
            max_cache_size: max_size,
        }
    }
    
    fn get(&self, key: &str) -> Option<&CombinationResult> {
        self.combination_cache.get(key)
    }
    
    fn insert(&mut self, key: String, value: CombinationResult) {
        if self.combination_cache.len() >= self.max_cache_size {
            // Simple eviction: remove first entry
            if let Some(first_key) = self.combination_cache.keys().next().cloned() {
                self.combination_cache.remove(&first_key);
            }
        }
        self.combination_cache.insert(key, value);
    }
    
    fn size(&self) -> usize {
        self.combination_cache.len()
    }
}

impl PerformanceTracker {
    fn new() -> Self {
        Self {
            total_analyses: 0,
            total_time_ns: 0,
            cache_hit_rate: 0.0,
            performance_history: Vec::new(),
        }
    }
    
    fn record_analysis(&mut self, benchmark: &PerformanceBenchmark) {
        self.total_analyses += 1;
        self.total_time_ns += benchmark.total_time_ns;
        self.performance_history.push(benchmark.clone());
        
        // Calculate cache hit rate
        let total_requests = benchmark.cache_hits + benchmark.cache_misses;
        if total_requests > 0 {
            self.cache_hit_rate = benchmark.cache_hits as f64 / total_requests as f64;
        }
    }
}

impl Default for CombinatorialDiversityFusionAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_analyzer_creation() {
        let analyzer = CombinatorialDiversityFusionAnalyzer::new();
        assert!(analyzer.algorithm_pool.size() > 0); // Should have built-in algorithms
    }
    
    #[test]
    fn test_single_combination_analysis() {
        let mut analyzer = CombinatorialDiversityFusionAnalyzer::new();
        
        let data = array![
            [0.8, 0.6, 0.9, 0.3, 0.7],
            [0.7, 0.8, 0.6, 0.4, 0.9]
        ];
        
        let algorithm_ids = analyzer.algorithm_pool.list_algorithms();
        let selected_ids = algorithm_ids.into_iter().take(2).collect::<Vec<_>>();
        
        let result = analyzer.analyze_single_combination(
            &selected_ids,
            &data.view(),
            FusionMethod::Average,
            None,
        );
        
        assert!(result.is_ok());
        let combination_result = result.unwrap();
        assert_eq!(combination_result.fusion_result.len(), 5);
        assert_eq!(combination_result.algorithm_ids.len(), 2);
    }
    
    #[test]
    fn test_synergy_guided_combinations() {
        let mut analyzer = CombinatorialDiversityFusionAnalyzer::new();
        
        let data = array![
            [0.8, 0.6, 0.9, 0.3, 0.7],
            [0.7, 0.8, 0.6, 0.4, 0.9],
            [0.9, 0.5, 0.8, 0.5, 0.6]
        ];
        
        let result = analyzer.find_synergy_guided_combinations(&data.view(), 2, 3);
        assert!(result.is_ok());
        
        let combinations = result.unwrap();
        assert!(!combinations.is_empty());
        assert!(combinations.iter().all(|c| c.algorithm_ids.len() == 2));
    }
    
    #[test]
    fn test_performance_statistics() {
        let analyzer = CombinatorialDiversityFusionAnalyzer::new();
        let stats = analyzer.get_performance_statistics();
        
        assert_eq!(stats.total_analyses, 0); // No analyses yet
        assert!(stats.algorithm_pool_size > 0); // Should have built-in algorithms
    }
}