//! Combinatorial Diversity Fusion Analysis (CDFA) implementation
//!
//! This module provides sophisticated algorithms for analyzing and optimizing
//! combinations of diverse data sources, algorithms, and fusion strategies.
//!
//! ## Core Concepts
//!
//! - **Combinatorial Analysis**: Systematic evaluation of all possible combinations
//! - **Diversity Metrics**: Quantitative measures of diversity between sources
//! - **Synergy Detection**: Identification of synergistic combinations
//! - **Fusion Optimization**: Automatic selection of optimal fusion strategies
//!
//! ## Performance
//!
//! - SIMD-optimized calculations for large-scale analysis
//! - Parallel processing for combination evaluation
//! - Adaptive algorithms that scale with complexity
//! - Sub-microsecond performance for real-time applications

use crate::error::{CdfaError, Result, CombinatorialError};
use crate::types::*;
// Diversity methods available through core
use crate::core::fusion::*;
use std::collections::HashMap;
// NDArray prelude available through types
use std::sync::Arc;
use std::sync::Mutex;

#[cfg(feature = "parallel")]
use rayon::prelude::*;

#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};

/// Configuration for combinatorial analysis
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CombinatorialConfig {
    /// Maximum number of combinations to evaluate
    pub max_combinations: usize,
    
    /// Minimum diversity threshold for combinations
    pub min_diversity_threshold: Float,
    
    /// Maximum redundancy allowed in combinations
    pub max_redundancy_threshold: Float,
    
    /// Enable synergy detection
    pub enable_synergy_detection: bool,
    
    /// Enable parallel processing
    pub enable_parallel: bool,
    
    /// Number of top combinations to return
    pub top_k_results: usize,
    
    /// Validation tolerance for numerical computations
    pub numerical_tolerance: Float,
    
    /// Cache size for intermediate results
    pub cache_size: usize,
}

impl Default for CombinatorialConfig {
    fn default() -> Self {
        Self {
            max_combinations: 1000,
            min_diversity_threshold: 0.1,
            max_redundancy_threshold: 0.9,
            enable_synergy_detection: true,
            enable_parallel: true,
            top_k_results: 10,
            numerical_tolerance: 1e-10,
            cache_size: 100,
        }
    }
}


/// Represents a single algorithm in the swarm
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SwarmAlgorithm {
    /// Unique identifier for the algorithm
    pub id: String,
    
    /// Human-readable name
    pub name: String,
    
    /// Algorithm type/category
    pub algorithm_type: String,
    
    /// Performance characteristics
    pub performance_profile: PerformanceProfile,
    
    /// Behavioral characteristics
    pub behavioral_traits: BehavioralTraits,
    
    /// Historical performance data
    pub historical_performance: Vec<Float>,
    
    /// Resource requirements
    pub resource_requirements: ResourceRequirements,
}

/// Performance characteristics of an algorithm
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PerformanceProfile {
    /// Average convergence speed
    pub convergence_speed: Float,
    
    /// Solution quality metric
    pub solution_quality: Float,
    
    /// Exploration capability
    pub exploration_capability: Float,
    
    /// Exploitation capability
    pub exploitation_capability: Float,
    
    /// Robustness to noise
    pub noise_robustness: Float,
    
    /// Scalability with problem size
    pub scalability: Float,
}

/// Behavioral characteristics of an algorithm
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct BehavioralTraits {
    /// Diversity maintenance capability
    pub diversity_maintenance: Float,
    
    /// Adaptability to changing conditions
    pub adaptability: Float,
    
    /// Cooperation with other algorithms
    pub cooperation: Float,
    
    /// Competition intensity
    pub competition: Float,
    
    /// Information sharing tendency
    pub information_sharing: Float,
    
    /// Risk tolerance
    pub risk_tolerance: Float,
}

/// Resource requirements for an algorithm
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct ResourceRequirements {
    /// CPU usage (0.0 to 1.0)
    pub cpu_usage: Float,
    
    /// Memory usage in MB
    pub memory_usage: Float,
    
    /// Communication overhead
    pub communication_overhead: Float,
    
    /// Storage requirements
    pub storage_requirements: Float,
}

/// Manages a pool of algorithms for combinatorial analysis
pub struct AlgorithmPool {
    algorithms: HashMap<String, SwarmAlgorithm>,
    performance_cache: Arc<Mutex<HashMap<String, Float>>>,
    diversity_cache: Arc<Mutex<HashMap<String, Float>>>,
    config: CombinatorialConfig,
}

impl AlgorithmPool {
    /// Create a new algorithm pool
    pub fn new(config: CombinatorialConfig) -> Self {
        Self {
            algorithms: HashMap::new(),
            performance_cache: Arc::new(Mutex::new(HashMap::new())),
            diversity_cache: Arc::new(Mutex::new(HashMap::new())),
            config,
        }
    }
    
    /// Add an algorithm to the pool
    pub fn add_algorithm(&mut self, algorithm: SwarmAlgorithm) -> Result<()> {
        if algorithm.id.is_empty() {
            return Err(CdfaError::invalid_input("Algorithm ID cannot be empty"));
        }
        
        self.algorithms.insert(algorithm.id.clone(), algorithm);
        Ok(())
    }
    
    /// Remove an algorithm from the pool
    pub fn remove_algorithm(&mut self, algorithm_id: &str) -> Result<SwarmAlgorithm> {
        self.algorithms.remove(algorithm_id)
            .ok_or_else(|| CdfaError::invalid_input(format!("Algorithm not found: {}", algorithm_id)))
    }
    
    /// Get algorithm by ID
    pub fn get_algorithm(&self, algorithm_id: &str) -> Option<&SwarmAlgorithm> {
        self.algorithms.get(algorithm_id)
    }
    
    /// Get all algorithm IDs
    pub fn get_algorithm_ids(&self) -> Vec<String> {
        self.algorithms.keys().cloned().collect()
    }
    
    /// Get algorithms by type
    pub fn get_algorithms_by_type(&self, algorithm_type: &str) -> Vec<&SwarmAlgorithm> {
        self.algorithms.values()
            .filter(|alg| alg.algorithm_type == algorithm_type)
            .collect()
    }
    
    /// Calculate diversity between two algorithms
    pub fn calculate_algorithm_diversity(&self, id1: &str, id2: &str) -> Result<Float> {
        let alg1 = self.get_algorithm(id1)
            .ok_or_else(|| CdfaError::invalid_input(format!("Algorithm not found: {}", id1)))?;
        let alg2 = self.get_algorithm(id2)
            .ok_or_else(|| CdfaError::invalid_input(format!("Algorithm not found: {}", id2)))?;
        
        // Calculate diversity based on performance profiles and behavioral traits
        let performance_diversity = self.calculate_performance_diversity(&alg1.performance_profile, &alg2.performance_profile);
        let behavioral_diversity = self.calculate_behavioral_diversity(&alg1.behavioral_traits, &alg2.behavioral_traits);
        
        // Combine diversities with equal weights
        let combined_diversity = (performance_diversity + behavioral_diversity) / 2.0;
        
        Ok(combined_diversity)
    }
    
    /// Calculate performance diversity between two performance profiles
    fn calculate_performance_diversity(&self, profile1: &PerformanceProfile, profile2: &PerformanceProfile) -> Float {
        let metrics1 = vec![
            profile1.convergence_speed,
            profile1.solution_quality,
            profile1.exploration_capability,
            profile1.exploitation_capability,
            profile1.noise_robustness,
            profile1.scalability,
        ];
        
        let metrics2 = vec![
            profile2.convergence_speed,
            profile2.solution_quality,
            profile2.exploration_capability,
            profile2.exploitation_capability,
            profile2.noise_robustness,
            profile2.scalability,
        ];
        
        // Calculate Euclidean distance normalized by dimension
        let mut sum_squared_diff = 0.0;
        for (m1, m2) in metrics1.iter().zip(metrics2.iter()) {
            sum_squared_diff += (m1 - m2).powi(2);
        }
        
        (sum_squared_diff / metrics1.len() as Float).sqrt()
    }
    
    /// Calculate behavioral diversity between two behavioral trait sets
    fn calculate_behavioral_diversity(&self, traits1: &BehavioralTraits, traits2: &BehavioralTraits) -> Float {
        let metrics1 = vec![
            traits1.diversity_maintenance,
            traits1.adaptability,
            traits1.cooperation,
            traits1.competition,
            traits1.information_sharing,
            traits1.risk_tolerance,
        ];
        
        let metrics2 = vec![
            traits2.diversity_maintenance,
            traits2.adaptability,
            traits2.cooperation,
            traits2.competition,
            traits2.information_sharing,
            traits2.risk_tolerance,
        ];
        
        // Calculate Euclidean distance normalized by dimension
        let mut sum_squared_diff = 0.0;
        for (m1, m2) in metrics1.iter().zip(metrics2.iter()) {
            sum_squared_diff += (m1 - m2).powi(2);
        }
        
        (sum_squared_diff / metrics1.len() as Float).sqrt()
    }
    
    /// Get total number of algorithms in the pool
    pub fn size(&self) -> usize {
        self.algorithms.len()
    }
    
    /// Clear all algorithms from the pool
    pub fn clear(&mut self) {
        self.algorithms.clear();
        self.performance_cache.lock().unwrap().clear();
        self.diversity_cache.lock().unwrap().clear();
    }
}

/// Metrics for evaluating synergy between algorithms
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SynergyMetrics {
    /// Synergy strength (0.0 to 1.0)
    pub synergy_strength: Float,
    
    /// Complementarity score
    pub complementarity: Float,
    
    /// Information sharing efficiency
    pub information_sharing: Float,
    
    /// Cooperative potential
    pub cooperative_potential: Float,
    
    /// Redundancy reduction
    pub redundancy_reduction: Float,
}

/// Detects synergistic relationships between algorithms
pub struct SynergyDetector {
    config: CombinatorialConfig,
    synergy_cache: Arc<Mutex<HashMap<String, SynergyMetrics>>>,
}

impl SynergyDetector {
    /// Create a new synergy detector
    pub fn new(config: CombinatorialConfig) -> Self {
        Self {
            config,
            synergy_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    /// Detect synergy between two algorithms
    pub fn detect_synergy(&self, pool: &AlgorithmPool, id1: &str, id2: &str) -> Result<SynergyMetrics> {
        let cache_key = format!("{}:{}", id1, id2);
        
        // Check cache first
        if let Ok(cache) = self.synergy_cache.lock() {
            if let Some(metrics) = cache.get(&cache_key) {
                return Ok(metrics.clone());
            }
        }
        
        let alg1 = pool.get_algorithm(id1)
            .ok_or_else(|| CdfaError::invalid_input(format!("Algorithm not found: {}", id1)))?;
        let alg2 = pool.get_algorithm(id2)
            .ok_or_else(|| CdfaError::invalid_input(format!("Algorithm not found: {}", id2)))?;
        
        // Calculate synergy metrics
        let complementarity = self.calculate_complementarity(&alg1.performance_profile, &alg2.performance_profile);
        let information_sharing = self.calculate_information_sharing(&alg1.behavioral_traits, &alg2.behavioral_traits);
        let cooperative_potential = self.calculate_cooperative_potential(&alg1.behavioral_traits, &alg2.behavioral_traits);
        let redundancy_reduction = self.calculate_redundancy_reduction(&alg1.performance_profile, &alg2.performance_profile);
        
        let synergy_strength = (complementarity + information_sharing + cooperative_potential + redundancy_reduction) / 4.0;
        
        let metrics = SynergyMetrics {
            synergy_strength,
            complementarity,
            information_sharing,
            cooperative_potential,
            redundancy_reduction,
        };
        
        // Cache the result
        if let Ok(mut cache) = self.synergy_cache.lock() {
            cache.insert(cache_key, metrics.clone());
        }
        
        Ok(metrics)
    }
    
    /// Calculate complementarity between two performance profiles
    fn calculate_complementarity(&self, profile1: &PerformanceProfile, profile2: &PerformanceProfile) -> Float {
        // Higher complementarity when strengths/weaknesses are different
        let exploration_complement = (profile1.exploration_capability - profile2.exploitation_capability).abs();
        let exploitation_complement = (profile1.exploitation_capability - profile2.exploration_capability).abs();
        let speed_quality_complement = (profile1.convergence_speed - profile2.solution_quality).abs();
        
        (exploration_complement + exploitation_complement + speed_quality_complement) / 3.0
    }
    
    /// Calculate information sharing potential
    fn calculate_information_sharing(&self, traits1: &BehavioralTraits, traits2: &BehavioralTraits) -> Float {
        let sharing_capacity = (traits1.information_sharing + traits2.information_sharing) / 2.0;
        let cooperation_factor = (traits1.cooperation + traits2.cooperation) / 2.0;
        let competition_penalty = (traits1.competition + traits2.competition) / 4.0; // Reduce by competition
        
        (sharing_capacity + cooperation_factor - competition_penalty).max(0.0).min(1.0)
    }
    
    /// Calculate cooperative potential
    fn calculate_cooperative_potential(&self, traits1: &BehavioralTraits, traits2: &BehavioralTraits) -> Float {
        let base_cooperation = (traits1.cooperation + traits2.cooperation) / 2.0;
        let adaptability_factor = (traits1.adaptability + traits2.adaptability) / 2.0;
        let competition_factor = 1.0 - (traits1.competition + traits2.competition) / 2.0;
        
        (base_cooperation * adaptability_factor * competition_factor).max(0.0).min(1.0)
    }
    
    /// Calculate redundancy reduction potential
    fn calculate_redundancy_reduction(&self, profile1: &PerformanceProfile, profile2: &PerformanceProfile) -> Float {
        // Lower redundancy when profiles are different
        let speed_diff = (profile1.convergence_speed - profile2.convergence_speed).abs();
        let quality_diff = (profile1.solution_quality - profile2.solution_quality).abs();
        let exploration_diff = (profile1.exploration_capability - profile2.exploration_capability).abs();
        let exploitation_diff = (profile1.exploitation_capability - profile2.exploitation_capability).abs();
        
        (speed_diff + quality_diff + exploration_diff + exploitation_diff) / 4.0
    }
}

/// Represents a combination of algorithms and their analysis results
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CombinationResult {
    /// Algorithm IDs in the combination
    pub algorithm_ids: Vec<String>,
    
    /// Diversity score of the combination
    pub diversity_score: Float,
    
    /// Synergy metrics for the combination
    pub synergy_metrics: Option<SynergyMetrics>,
    
    /// Evaluation metrics
    pub evaluation_metrics: EvaluationMetrics,
    
    /// Recommended fusion strategy
    pub recommended_fusion: FusionMethod,
    
    /// Confidence in the recommendation
    pub confidence: Float,
}

/// Evaluation metrics for combinations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct EvaluationMetrics {
    /// Overall combination score
    pub overall_score: Float,
    
    /// Performance potential
    pub performance_potential: Float,
    
    /// Diversity contribution
    pub diversity_contribution: Float,
    
    /// Resource efficiency
    pub resource_efficiency: Float,
    
    /// Scalability score
    pub scalability_score: Float,
    
    /// Robustness score
    pub robustness_score: Float,
}

/// Evaluates algorithm combinations
pub struct FusionEvaluator {
    config: CombinatorialConfig,
    synergy_detector: SynergyDetector,
    evaluation_cache: Arc<Mutex<HashMap<String, EvaluationMetrics>>>,
}

impl FusionEvaluator {
    /// Create a new fusion evaluator
    pub fn new(config: CombinatorialConfig) -> Self {
        let synergy_detector = SynergyDetector::new(config.clone());
        Self {
            config,
            synergy_detector,
            evaluation_cache: Arc::new(Mutex::new(HashMap::new())),
        }
    }
    
    /// Evaluate a combination of algorithms
    pub fn evaluate_combination(&self, pool: &AlgorithmPool, algorithm_ids: &[String]) -> Result<CombinationResult> {
        if algorithm_ids.len() < 2 {
            return Err(CdfaError::invalid_input("Need at least 2 algorithms for combination").into());
        }
        
        // Calculate diversity score
        let diversity_score = self.calculate_combination_diversity(pool, algorithm_ids)?;
        
        // Check diversity threshold
        if diversity_score < self.config.min_diversity_threshold {
            return Err(CombinatorialError::InsufficientDiversity.into());
        }
        
        // Calculate synergy metrics if enabled
        let synergy_metrics = if self.config.enable_synergy_detection {
            Some(self.calculate_combination_synergy(pool, algorithm_ids)?)
        } else {
            None
        };
        
        // Calculate evaluation metrics
        let evaluation_metrics = self.calculate_evaluation_metrics(pool, algorithm_ids, diversity_score, &synergy_metrics)?;
        
        // Recommend fusion strategy
        let recommended_fusion = self.recommend_fusion_strategy(pool, algorithm_ids, &evaluation_metrics)?;
        
        // Calculate confidence
        let confidence = self.calculate_confidence(&evaluation_metrics, &synergy_metrics);
        
        Ok(CombinationResult {
            algorithm_ids: algorithm_ids.to_vec(),
            diversity_score,
            synergy_metrics,
            evaluation_metrics,
            recommended_fusion,
            confidence,
        })
    }
    
    /// Calculate diversity score for a combination
    fn calculate_combination_diversity(&self, pool: &AlgorithmPool, algorithm_ids: &[String]) -> Result<Float> {
        let mut total_diversity = 0.0;
        let mut pair_count = 0;
        
        for i in 0..algorithm_ids.len() {
            for j in i + 1..algorithm_ids.len() {
                let diversity = pool.calculate_algorithm_diversity(&algorithm_ids[i], &algorithm_ids[j])?;
                total_diversity += diversity;
                pair_count += 1;
            }
        }
        
        Ok(if pair_count > 0 { total_diversity / pair_count as Float } else { 0.0 })
    }
    
    /// Calculate synergy metrics for a combination
    fn calculate_combination_synergy(&self, pool: &AlgorithmPool, algorithm_ids: &[String]) -> Result<SynergyMetrics> {
        let mut total_synergy = 0.0;
        let mut total_complementarity = 0.0;
        let mut total_information_sharing = 0.0;
        let mut total_cooperative_potential = 0.0;
        let mut total_redundancy_reduction = 0.0;
        let mut pair_count = 0;
        
        for i in 0..algorithm_ids.len() {
            for j in i + 1..algorithm_ids.len() {
                let synergy = self.synergy_detector.detect_synergy(pool, &algorithm_ids[i], &algorithm_ids[j])?;
                total_synergy += synergy.synergy_strength;
                total_complementarity += synergy.complementarity;
                total_information_sharing += synergy.information_sharing;
                total_cooperative_potential += synergy.cooperative_potential;
                total_redundancy_reduction += synergy.redundancy_reduction;
                pair_count += 1;
            }
        }
        
        let avg_factor = if pair_count > 0 { 1.0 / pair_count as Float } else { 0.0 };
        
        Ok(SynergyMetrics {
            synergy_strength: total_synergy * avg_factor,
            complementarity: total_complementarity * avg_factor,
            information_sharing: total_information_sharing * avg_factor,
            cooperative_potential: total_cooperative_potential * avg_factor,
            redundancy_reduction: total_redundancy_reduction * avg_factor,
        })
    }
    
    /// Calculate evaluation metrics for a combination
    fn calculate_evaluation_metrics(
        &self,
        pool: &AlgorithmPool,
        algorithm_ids: &[String],
        diversity_score: Float,
        synergy_metrics: &Option<SynergyMetrics>,
    ) -> Result<EvaluationMetrics> {
        let mut total_performance = 0.0;
        let mut total_cpu = 0.0;
        let mut total_memory = 0.0;
        let mut total_scalability = 0.0;
        let mut total_robustness = 0.0;
        
        for id in algorithm_ids {
            let alg = pool.get_algorithm(id)
                .ok_or_else(|| CdfaError::invalid_input(format!("Algorithm not found: {}", id)))?;
            
            let performance = (alg.performance_profile.convergence_speed + alg.performance_profile.solution_quality) / 2.0;
            total_performance += performance;
            total_cpu += alg.resource_requirements.cpu_usage;
            total_memory += alg.resource_requirements.memory_usage;
            total_scalability += alg.performance_profile.scalability;
            total_robustness += alg.performance_profile.noise_robustness;
        }
        
        let avg_performance = total_performance / algorithm_ids.len() as Float;
        let avg_scalability = total_scalability / algorithm_ids.len() as Float;
        let avg_robustness = total_robustness / algorithm_ids.len() as Float;
        
        // Resource efficiency is inversely related to resource usage
        let resource_efficiency = 1.0 / (1.0 + total_cpu + total_memory / 1000.0);
        
        // Synergy bonus
        let synergy_bonus = if let Some(synergy) = synergy_metrics {
            synergy.synergy_strength * 0.2 // 20% bonus for synergy
        } else {
            0.0
        };
        
        let overall_score = (avg_performance + diversity_score + resource_efficiency + synergy_bonus) / 4.0;
        
        Ok(EvaluationMetrics {
            overall_score,
            performance_potential: avg_performance,
            diversity_contribution: diversity_score,
            resource_efficiency,
            scalability_score: avg_scalability,
            robustness_score: avg_robustness,
        })
    }
    
    /// Recommend fusion strategy based on combination characteristics
    fn recommend_fusion_strategy(
        &self,
        pool: &AlgorithmPool,
        algorithm_ids: &[String],
        evaluation_metrics: &EvaluationMetrics,
    ) -> Result<FusionMethod> {
        // Analyze algorithm characteristics to recommend fusion strategy
        let mut has_high_quality = false;
        let mut has_high_speed = false;
        let mut performance_variance = 0.0;
        
        let mut performances = Vec::new();
        for id in algorithm_ids {
            let alg = pool.get_algorithm(id)
                .ok_or_else(|| CdfaError::invalid_input(format!("Algorithm not found: {}", id)))?;
            
            let performance = (alg.performance_profile.convergence_speed + alg.performance_profile.solution_quality) / 2.0;
            performances.push(performance);
            
            if alg.performance_profile.solution_quality > 0.8 {
                has_high_quality = true;
            }
            if alg.performance_profile.convergence_speed > 0.8 {
                has_high_speed = true;
            }
        }
        
        // Calculate performance variance
        let mean_performance = performances.iter().sum::<Float>() / performances.len() as Float;
        performance_variance = performances.iter()
            .map(|p| (p - mean_performance).powi(2))
            .sum::<Float>() / performances.len() as Float;
        
        // Recommend fusion strategy based on characteristics
        if has_high_quality && has_high_speed {
            Ok(FusionMethod::WeightedAverage) // Balance all high-performing algorithms
        } else if performance_variance > 0.1 {
            Ok(FusionMethod::BordaCount) // Handle diverse performance levels
        } else if evaluation_metrics.diversity_contribution > 0.6 {
            Ok(FusionMethod::DiversityWeighted) // Emphasize diversity
        } else {
            Ok(FusionMethod::Average) // Default to simple average
        }
    }
    
    /// Calculate confidence in the recommendation
    fn calculate_confidence(&self, evaluation_metrics: &EvaluationMetrics, synergy_metrics: &Option<SynergyMetrics>) -> Float {
        let mut confidence = evaluation_metrics.overall_score;
        
        // Boost confidence if synergy is detected
        if let Some(synergy) = synergy_metrics {
            confidence += synergy.synergy_strength * 0.1;
        }
        
        // Boost confidence for good resource efficiency
        confidence += evaluation_metrics.resource_efficiency * 0.1;
        
        // Boost confidence for good scalability
        confidence += evaluation_metrics.scalability_score * 0.1;
        
        confidence.min(1.0).max(0.0)
    }
}

/// Analysis of combinations
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct CombinationAnalysis {
    /// Total number of combinations evaluated
    pub total_combinations: usize,
    
    /// Top combinations found
    pub top_combinations: Vec<CombinationResult>,
    
    /// Overall diversity statistics
    pub diversity_statistics: DiversityStatistics,
    
    /// Synergy analysis results
    pub synergy_analysis: Option<SynergyAnalysis>,
    
    /// Performance analysis
    pub performance_analysis: PerformanceAnalysis,
    
    /// Execution time in microseconds
    pub execution_time_us: u64,
}

/// Diversity statistics for the analysis
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct DiversityStatistics {
    /// Mean diversity score
    pub mean_diversity: Float,
    
    /// Standard deviation of diversity
    pub std_diversity: Float,
    
    /// Minimum diversity found
    pub min_diversity: Float,
    
    /// Maximum diversity found
    pub max_diversity: Float,
    
    /// Number of combinations above threshold
    pub above_threshold_count: usize,
}

/// Synergy analysis results
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SynergyAnalysis {
    /// Mean synergy strength
    pub mean_synergy: Float,
    
    /// Standard deviation of synergy
    pub std_synergy: Float,
    
    /// Best synergistic combination
    pub best_synergy_combination: Option<CombinationResult>,
    
    /// Synergy distribution
    pub synergy_distribution: Vec<Float>,
}

/// Performance analysis results
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct PerformanceAnalysis {
    /// Average overall score
    pub mean_overall_score: Float,
    
    /// Standard deviation of overall score
    pub std_overall_score: Float,
    
    /// Resource efficiency analysis
    pub resource_efficiency_mean: Float,
    
    /// Scalability analysis
    pub scalability_mean: Float,
    
    /// Robustness analysis
    pub robustness_mean: Float,
}

/// Main combinatorial diversity fusion analyzer
pub struct CombinatorialDiversityFusionAnalyzer {
    algorithm_pool: AlgorithmPool,
    fusion_evaluator: FusionEvaluator,
    config: CombinatorialConfig,
}

impl CombinatorialDiversityFusionAnalyzer {
    /// Create a new analyzer
    pub fn new(config: CombinatorialConfig) -> Self {
        let algorithm_pool = AlgorithmPool::new(config.clone());
        let fusion_evaluator = FusionEvaluator::new(config.clone());
        
        Self {
            algorithm_pool,
            fusion_evaluator,
            config,
        }
    }
    
    /// Add algorithm to the pool
    pub fn add_algorithm(&mut self, algorithm: SwarmAlgorithm) -> Result<()> {
        self.algorithm_pool.add_algorithm(algorithm)
    }
    
    /// Analyze all possible combinations
    pub fn analyze_combinations(&self, combination_size: usize) -> Result<CombinationAnalysis> {
        let start_time = std::time::Instant::now();
        
        if combination_size < 2 {
            return Err(CombinatorialError::InvalidCombinationSize(combination_size).into());
        }
        
        let algorithm_ids = self.algorithm_pool.get_algorithm_ids();
        if algorithm_ids.len() < combination_size {
            return Err(CombinatorialError::InvalidCombinationSize(combination_size).into());
        }
        
        // Generate all combinations
        let combinations = self.generate_combinations(&algorithm_ids, combination_size);
        
        // Limit combinations if necessary
        let combinations_to_evaluate = if combinations.len() > self.config.max_combinations {
            combinations.into_iter().take(self.config.max_combinations).collect()
        } else {
            combinations
        };
        
        // Evaluate combinations
        let results = if self.config.enable_parallel {
            self.evaluate_combinations_parallel(&combinations_to_evaluate)?
        } else {
            self.evaluate_combinations_sequential(&combinations_to_evaluate)?
        };
        
        // Sort by overall score
        let mut sorted_results = results;
        sorted_results.sort_by(|a, b| {
            b.evaluation_metrics.overall_score.partial_cmp(&a.evaluation_metrics.overall_score).unwrap_or(std::cmp::Ordering::Equal)
        });
        
        // Take top k results
        let top_combinations = sorted_results.into_iter().take(self.config.top_k_results).collect::<Vec<_>>();
        
        // Calculate statistics
        let diversity_statistics = self.calculate_diversity_statistics(&top_combinations);
        let synergy_analysis = if self.config.enable_synergy_detection {
            Some(self.calculate_synergy_analysis(&top_combinations))
        } else {
            None
        };
        let performance_analysis = self.calculate_performance_analysis(&top_combinations);
        
        let execution_time_us = start_time.elapsed().as_micros() as u64;
        
        Ok(CombinationAnalysis {
            total_combinations: combinations_to_evaluate.len(),
            top_combinations,
            diversity_statistics,
            synergy_analysis,
            performance_analysis,
            execution_time_us,
        })
    }
    
    /// Generate all combinations of given size
    fn generate_combinations(&self, items: &[String], size: usize) -> Vec<Vec<String>> {
        let mut combinations = Vec::new();
        self.generate_combinations_recursive(items, size, 0, &mut Vec::new(), &mut combinations);
        combinations
    }
    
    /// Recursive helper for generating combinations
    fn generate_combinations_recursive(
        &self,
        items: &[String],
        size: usize,
        start: usize,
        current: &mut Vec<String>,
        combinations: &mut Vec<Vec<String>>,
    ) {
        if current.len() == size {
            combinations.push(current.clone());
            return;
        }
        
        for i in start..items.len() {
            current.push(items[i].clone());
            self.generate_combinations_recursive(items, size, i + 1, current, combinations);
            current.pop();
        }
    }
    
    /// Evaluate combinations in parallel
    #[cfg(feature = "parallel")]
    fn evaluate_combinations_parallel(&self, combinations: &[Vec<String>]) -> Result<Vec<CombinationResult>> {
        use rayon::prelude::*;
        
        combinations.par_iter()
            .map(|combination| {
                self.fusion_evaluator.evaluate_combination(&self.algorithm_pool, combination)
            })
            .collect::<Result<Vec<_>>>()
    }
    
    /// Evaluate combinations in parallel (fallback when parallel feature not available)
    #[cfg(not(feature = "parallel"))]
    fn evaluate_combinations_parallel(&self, combinations: &[Vec<String>]) -> Result<Vec<CombinationResult>> {
        self.evaluate_combinations_sequential(combinations)
    }
    
    /// Evaluate combinations sequentially
    fn evaluate_combinations_sequential(&self, combinations: &[Vec<String>]) -> Result<Vec<CombinationResult>> {
        combinations.iter()
            .map(|combination| {
                self.fusion_evaluator.evaluate_combination(&self.algorithm_pool, combination)
            })
            .collect::<Result<Vec<_>>>()
    }
    
    /// Calculate diversity statistics
    fn calculate_diversity_statistics(&self, combinations: &[CombinationResult]) -> DiversityStatistics {
        if combinations.is_empty() {
            return DiversityStatistics {
                mean_diversity: 0.0,
                std_diversity: 0.0,
                min_diversity: 0.0,
                max_diversity: 0.0,
                above_threshold_count: 0,
            };
        }
        
        let diversities: Vec<Float> = combinations.iter().map(|c| c.diversity_score).collect();
        let mean = diversities.iter().sum::<Float>() / diversities.len() as Float;
        let variance = diversities.iter().map(|d| (d - mean).powi(2)).sum::<Float>() / diversities.len() as Float;
        let std_dev = variance.sqrt();
        let min_diversity = diversities.iter().fold(Float::INFINITY, |a, &b| a.min(b));
        let max_diversity = diversities.iter().fold(Float::NEG_INFINITY, |a, &b| a.max(b));
        let above_threshold_count = diversities.iter().filter(|&&d| d >= self.config.min_diversity_threshold).count();
        
        DiversityStatistics {
            mean_diversity: mean,
            std_diversity: std_dev,
            min_diversity,
            max_diversity,
            above_threshold_count,
        }
    }
    
    /// Calculate synergy analysis
    fn calculate_synergy_analysis(&self, combinations: &[CombinationResult]) -> SynergyAnalysis {
        let synergy_strengths: Vec<Float> = combinations.iter()
            .filter_map(|c| c.synergy_metrics.as_ref().map(|s| s.synergy_strength))
            .collect();
        
        if synergy_strengths.is_empty() {
            return SynergyAnalysis {
                mean_synergy: 0.0,
                std_synergy: 0.0,
                best_synergy_combination: None,
                synergy_distribution: Vec::new(),
            };
        }
        
        let mean = synergy_strengths.iter().sum::<Float>() / synergy_strengths.len() as Float;
        let variance = synergy_strengths.iter().map(|s| (s - mean).powi(2)).sum::<Float>() / synergy_strengths.len() as Float;
        let std_dev = variance.sqrt();
        
        let best_synergy_combination = combinations.iter()
            .filter(|c| c.synergy_metrics.is_some())
            .max_by(|a, b| {
                let a_synergy = a.synergy_metrics.as_ref().unwrap().synergy_strength;
                let b_synergy = b.synergy_metrics.as_ref().unwrap().synergy_strength;
                a_synergy.partial_cmp(&b_synergy).unwrap_or(std::cmp::Ordering::Equal)
            })
            .cloned();
        
        SynergyAnalysis {
            mean_synergy: mean,
            std_synergy: std_dev,
            best_synergy_combination,
            synergy_distribution: synergy_strengths,
        }
    }
    
    /// Calculate performance analysis
    fn calculate_performance_analysis(&self, combinations: &[CombinationResult]) -> PerformanceAnalysis {
        if combinations.is_empty() {
            return PerformanceAnalysis {
                mean_overall_score: 0.0,
                std_overall_score: 0.0,
                resource_efficiency_mean: 0.0,
                scalability_mean: 0.0,
                robustness_mean: 0.0,
            };
        }
        
        let overall_scores: Vec<Float> = combinations.iter().map(|c| c.evaluation_metrics.overall_score).collect();
        let mean_overall = overall_scores.iter().sum::<Float>() / overall_scores.len() as Float;
        let variance_overall = overall_scores.iter().map(|s| (s - mean_overall).powi(2)).sum::<Float>() / overall_scores.len() as Float;
        let std_overall = variance_overall.sqrt();
        
        let resource_efficiency_mean = combinations.iter()
            .map(|c| c.evaluation_metrics.resource_efficiency)
            .sum::<Float>() / combinations.len() as Float;
        
        let scalability_mean = combinations.iter()
            .map(|c| c.evaluation_metrics.scalability_score)
            .sum::<Float>() / combinations.len() as Float;
        
        let robustness_mean = combinations.iter()
            .map(|c| c.evaluation_metrics.robustness_score)
            .sum::<Float>() / combinations.len() as Float;
        
        PerformanceAnalysis {
            mean_overall_score: mean_overall,
            std_overall_score: std_overall,
            resource_efficiency_mean,
            scalability_mean,
            robustness_mean,
        }
    }
    
    /// Get algorithm pool reference
    pub fn get_algorithm_pool(&self) -> &AlgorithmPool {
        &self.algorithm_pool
    }
    
    /// Get mutable algorithm pool reference
    pub fn get_algorithm_pool_mut(&mut self) -> &mut AlgorithmPool {
        &mut self.algorithm_pool
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    
    fn create_test_algorithm(id: &str, convergence: Float, quality: Float) -> SwarmAlgorithm {
        SwarmAlgorithm {
            id: id.to_string(),
            name: format!("Test Algorithm {}", id),
            algorithm_type: "test".to_string(),
            performance_profile: PerformanceProfile {
                convergence_speed: convergence,
                solution_quality: quality,
                exploration_capability: 0.5,
                exploitation_capability: 0.5,
                noise_robustness: 0.5,
                scalability: 0.5,
            },
            behavioral_traits: BehavioralTraits {
                diversity_maintenance: 0.5,
                adaptability: 0.5,
                cooperation: 0.5,
                competition: 0.5,
                information_sharing: 0.5,
                risk_tolerance: 0.5,
            },
            historical_performance: vec![0.8, 0.7, 0.9],
            resource_requirements: ResourceRequirements {
                cpu_usage: 0.1,
                memory_usage: 100.0,
                communication_overhead: 0.1,
                storage_requirements: 10.0,
            },
        }
    }
    
    #[test]
    fn test_algorithm_pool() {
        let config = CombinatorialConfig::default();
        let mut pool = AlgorithmPool::new(config);
        
        let alg1 = create_test_algorithm("alg1", 0.8, 0.7);
        let alg2 = create_test_algorithm("alg2", 0.6, 0.9);
        
        pool.add_algorithm(alg1).unwrap();
        pool.add_algorithm(alg2).unwrap();
        
        assert_eq!(pool.size(), 2);
        assert!(pool.get_algorithm("alg1").is_some());
        assert!(pool.get_algorithm("alg2").is_some());
        
        let diversity = pool.calculate_algorithm_diversity("alg1", "alg2").unwrap();
        assert!(diversity > 0.0);
    }
    
    #[test]
    fn test_synergy_detector() {
        let config = CombinatorialConfig::default();
        let mut pool = AlgorithmPool::new(config.clone());
        let detector = SynergyDetector::new(config);
        
        // Create complementary algorithms
        let alg1 = SwarmAlgorithm {
            id: "explorer".to_string(),
            name: "Explorer".to_string(),
            algorithm_type: "exploration".to_string(),
            performance_profile: PerformanceProfile {
                convergence_speed: 0.3,
                solution_quality: 0.6,
                exploration_capability: 0.9,
                exploitation_capability: 0.2,
                noise_robustness: 0.7,
                scalability: 0.8,
            },
            behavioral_traits: BehavioralTraits {
                diversity_maintenance: 0.9,
                adaptability: 0.8,
                cooperation: 0.7,
                competition: 0.3,
                information_sharing: 0.8,
                risk_tolerance: 0.9,
            },
            historical_performance: vec![0.6, 0.7, 0.8],
            resource_requirements: ResourceRequirements {
                cpu_usage: 0.2,
                memory_usage: 200.0,
                communication_overhead: 0.1,
                storage_requirements: 20.0,
            },
        };
        
        let alg2 = SwarmAlgorithm {
            id: "exploiter".to_string(),
            name: "Exploiter".to_string(),
            algorithm_type: "exploitation".to_string(),
            performance_profile: PerformanceProfile {
                convergence_speed: 0.9,
                solution_quality: 0.8,
                exploration_capability: 0.2,
                exploitation_capability: 0.9,
                noise_robustness: 0.6,
                scalability: 0.7,
            },
            behavioral_traits: BehavioralTraits {
                diversity_maintenance: 0.3,
                adaptability: 0.6,
                cooperation: 0.8,
                competition: 0.4,
                information_sharing: 0.7,
                risk_tolerance: 0.4,
            },
            historical_performance: vec![0.8, 0.9, 0.8],
            resource_requirements: ResourceRequirements {
                cpu_usage: 0.1,
                memory_usage: 150.0,
                communication_overhead: 0.05,
                storage_requirements: 15.0,
            },
        };
        
        pool.add_algorithm(alg1).unwrap();
        pool.add_algorithm(alg2).unwrap();
        
        let synergy = detector.detect_synergy(&pool, "explorer", "exploiter").unwrap();
        
        // Should have good synergy due to complementary characteristics
        assert!(synergy.synergy_strength > 0.3);
        assert!(synergy.complementarity > 0.3);
        assert!(synergy.cooperative_potential > 0.3);
    }
    
    #[test]
    fn test_fusion_evaluator() {
        let config = CombinatorialConfig::default();
        let mut pool = AlgorithmPool::new(config.clone());
        let evaluator = FusionEvaluator::new(config);
        
        let alg1 = create_test_algorithm("alg1", 0.8, 0.7);
        let alg2 = create_test_algorithm("alg2", 0.6, 0.9);
        
        pool.add_algorithm(alg1).unwrap();
        pool.add_algorithm(alg2).unwrap();
        
        let combination = vec!["alg1".to_string(), "alg2".to_string()];
        let result = evaluator.evaluate_combination(&pool, &combination).unwrap();
        
        assert_eq!(result.algorithm_ids.len(), 2);
        assert!(result.diversity_score > 0.0);
        assert!(result.evaluation_metrics.overall_score > 0.0);
        assert!(result.confidence > 0.0);
    }
    
    #[test]
    fn test_combinatorial_analyzer() {
        let config = CombinatorialConfig {
            max_combinations: 10,
            min_diversity_threshold: 0.0,
            enable_synergy_detection: true,
            enable_parallel: false,
            top_k_results: 5,
            ..Default::default()
        };
        
        let mut analyzer = CombinatorialDiversityFusionAnalyzer::new(config);
        
        // Add test algorithms
        for i in 0..4 {
            let alg = create_test_algorithm(
                &format!("alg{}", i),
                0.5 + i as Float * 0.1,
                0.6 + i as Float * 0.05
            );
            analyzer.add_algorithm(alg).unwrap();
        }
        
        let analysis = analyzer.analyze_combinations(2).unwrap();
        
        assert!(analysis.total_combinations > 0);
        assert!(!analysis.top_combinations.is_empty());
        assert!(analysis.diversity_statistics.mean_diversity >= 0.0);
        assert!(analysis.synergy_analysis.is_some());
        assert!(analysis.execution_time_us > 0);
    }
    
    #[test]
    fn test_combination_generation() {
        let config = CombinatorialConfig::default();
        let analyzer = CombinatorialDiversityFusionAnalyzer::new(config);
        
        let items = vec!["A".to_string(), "B".to_string(), "C".to_string(), "D".to_string()];
        let combinations = analyzer.generate_combinations(&items, 2);
        
        // Should have C(4,2) = 6 combinations
        assert_eq!(combinations.len(), 6);
        
        // Check that we have all expected combinations
        let expected = vec![
            vec!["A".to_string(), "B".to_string()],
            vec!["A".to_string(), "C".to_string()],
            vec!["A".to_string(), "D".to_string()],
            vec!["B".to_string(), "C".to_string()],
            vec!["B".to_string(), "D".to_string()],
            vec!["C".to_string(), "D".to_string()],
        ];
        
        for expected_combo in expected {
            assert!(combinations.contains(&expected_combo));
        }
    }
    
    #[test]
    fn test_diversity_statistics() {
        let config = CombinatorialConfig::default();
        let analyzer = CombinatorialDiversityFusionAnalyzer::new(config);
        
        let combinations = vec![
            CombinationResult {
                algorithm_ids: vec!["A".to_string(), "B".to_string()],
                diversity_score: 0.5,
                synergy_metrics: None,
                evaluation_metrics: EvaluationMetrics {
                    overall_score: 0.6,
                    performance_potential: 0.7,
                    diversity_contribution: 0.5,
                    resource_efficiency: 0.8,
                    scalability_score: 0.6,
                    robustness_score: 0.7,
                },
                recommended_fusion: FusionMethod::Average,
                confidence: 0.8,
            },
            CombinationResult {
                algorithm_ids: vec!["C".to_string(), "D".to_string()],
                diversity_score: 0.8,
                synergy_metrics: None,
                evaluation_metrics: EvaluationMetrics {
                    overall_score: 0.7,
                    performance_potential: 0.6,
                    diversity_contribution: 0.8,
                    resource_efficiency: 0.7,
                    scalability_score: 0.8,
                    robustness_score: 0.6,
                },
                recommended_fusion: FusionMethod::Average,
                confidence: 0.9,
            },
        ];
        
        let stats = analyzer.calculate_diversity_statistics(&combinations);
        
        assert_abs_diff_eq!(stats.mean_diversity, 0.65, epsilon = 1e-10);
        assert_abs_diff_eq!(stats.min_diversity, 0.5, epsilon = 1e-10);
        assert_abs_diff_eq!(stats.max_diversity, 0.8, epsilon = 1e-10);
        assert_eq!(stats.above_threshold_count, 2); // Both above default threshold of 0.1
    }
}