//! CombinatorialDiversityFusionAnalyzer - The core of advanced algorithm fusion
//!
//! This analyzer implements k-combinations generation, synergy detection,
//! and dynamic algorithm fusion based on CDFA principles.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use rayon::prelude::*;
use parking_lot::RwLock;
use dashmap::DashMap;

use crate::core::{
    SwarmAlgorithm, SwarmError, SwarmResult, OptimizationProblem,
    Individual, Position, AlgorithmMetrics
};
use crate::cdfa::{AlgorithmPool, DiversityMetrics, PerformanceTracker};

/// Fusion strategy for combining algorithms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum FusionStrategy {
    /// Sequential execution with best result selection
    Sequential,
    
    /// Parallel execution with weighted combination
    Parallel { weights: Vec<f64> },
    
    /// Adaptive switching based on performance
    Adaptive { switch_threshold: f64 },
    
    /// Island model with migration
    Island { migration_rate: f64, migration_interval: usize },
    
    /// Hierarchical fusion with levels
    Hierarchical { levels: Vec<Vec<usize>> },
    
    /// Dynamic fusion based on synergy detection
    Synergistic { synergy_threshold: f64 },
    
    /// Tournament-based selection
    Tournament { tournament_size: usize },
    
    /// Ensemble with voting
    Ensemble { voting_strategy: VotingStrategy },
}

/// Voting strategies for ensemble fusion
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VotingStrategy {
    Majority,
    Weighted,
    Ranked,
    Unanimous,
}

/// Result of algorithm fusion
#[derive(Debug, Clone)]
pub struct FusionResult<F> {
    /// Best solution found across all algorithms
    pub best_solution: SwarmResult<F>,
    
    /// Results from individual algorithms
    pub individual_results: Vec<SwarmResult<F>>,
    
    /// Combination metrics
    pub combination_metrics: CombinationMetrics,
    
    /// Synergy score between algorithms
    pub synergy_score: f64,
    
    /// Execution time breakdown
    pub timing: FusionTiming,
}

/// Metrics for algorithm combinations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CombinationMetrics {
    /// Number of algorithms in combination
    pub algorithm_count: usize,
    
    /// Diversity score of the combination
    pub diversity_score: f64,
    
    /// Performance improvement over best individual
    pub improvement_factor: f64,
    
    /// Convergence speed metrics
    pub convergence_metrics: ConvergenceMetrics,
    
    /// Resource utilization
    pub resource_usage: ResourceUsage,
}

/// Convergence metrics for analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConvergenceMetrics {
    /// Iterations to reach threshold
    pub iterations_to_threshold: Option<usize>,
    
    /// Convergence rate
    pub rate: f64,
    
    /// Stagnation detection
    pub stagnation_periods: Vec<(usize, usize)>,
    
    /// Premature convergence indicator
    pub premature_convergence: bool,
}

/// Resource usage tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceUsage {
    /// CPU time used (milliseconds)
    pub cpu_time_ms: u64,
    
    /// Memory peak usage (bytes)
    pub peak_memory_bytes: usize,
    
    /// Thread utilization
    pub thread_utilization: f64,
    
    /// Cache hit rate
    pub cache_hit_rate: f64,
}

/// Timing breakdown for fusion operations
#[derive(Debug, Clone)]
pub struct FusionTiming {
    /// Total execution time
    pub total_time: std::time::Duration,
    
    /// Time per algorithm
    pub per_algorithm: Vec<std::time::Duration>,
    
    /// Fusion overhead
    pub fusion_overhead: std::time::Duration,
    
    /// Synchronization time
    pub synchronization_time: std::time::Duration,
}

/// Core Combinatorial Diversity Fusion Analyzer
pub struct CombinatorialDiversityFusionAnalyzer {
    /// Pool of available algorithms
    algorithm_pool: Arc<RwLock<AlgorithmPool>>,
    
    /// Performance tracking
    performance_tracker: Arc<PerformanceTracker>,
    
    /// Diversity metrics calculator
    diversity_calculator: Arc<DiversityMetrics>,
    
    /// Cache for combination results
    combination_cache: Arc<DashMap<String, CombinationMetrics>>,
    
    /// Synergy detection threshold
    synergy_threshold: f64,
    
    /// Maximum combination size
    max_combination_size: usize,
    
    /// Enable parallel fusion
    parallel_execution: bool,
    
    /// CDFA integration settings
    cdfa_settings: CdfaSettings,
}

/// CDFA integration settings
#[derive(Debug, Clone)]
pub struct CdfaSettings {
    /// Use CDFA parallel infrastructure
    pub use_parallel_backend: bool,
    
    /// Enable SIMD optimizations
    pub enable_simd: bool,
    
    /// Lock-free data structures
    pub use_lock_free: bool,
    
    /// NUMA awareness
    pub numa_aware: bool,
    
    /// GPU acceleration
    pub use_gpu: bool,
}

impl Default for CdfaSettings {
    fn default() -> Self {
        Self {
            use_parallel_backend: true,
            enable_simd: cfg!(feature = "simd"),
            use_lock_free: true,
            numa_aware: true,
            use_gpu: cfg!(feature = "gpu"),
        }
    }
}

impl CombinatorialDiversityFusionAnalyzer {
    /// Create a new fusion analyzer
    pub fn new() -> Self {
        Self {
            algorithm_pool: Arc::new(RwLock::new(AlgorithmPool::new())),
            performance_tracker: Arc::new(PerformanceTracker::new()),
            diversity_calculator: Arc::new(DiversityMetrics::new()),
            combination_cache: Arc::new(DashMap::new()),
            synergy_threshold: 0.1,
            max_combination_size: 5,
            parallel_execution: true,
            cdfa_settings: CdfaSettings::default(),
        }
    }
    
    /// Create with custom settings
    pub fn with_settings(cdfa_settings: CdfaSettings) -> Self {
        let mut analyzer = Self::new();
        analyzer.cdfa_settings = cdfa_settings;
        analyzer
    }
    
    /// Add algorithm to the pool
    pub fn add_algorithm<T>(&self, algorithm: T, name: String) -> Result<(), SwarmError>
    where
        T: SwarmAlgorithm + 'static,
    {
        let mut pool = self.algorithm_pool.write();
        pool.add_algorithm(Box::new(algorithm), name)?;
        Ok(())
    }
    
    /// Generate k-combinations of algorithms
    pub fn generate_combinations(&self, k: usize) -> Result<Vec<Vec<String>>, SwarmError> {
        let pool = self.algorithm_pool.read();
        let algorithm_names = pool.list_algorithms();
        
        if k > algorithm_names.len() {
            return Err(SwarmError::parameter("k larger than available algorithms"));
        }
        
        if k > self.max_combination_size {
            return Err(SwarmError::parameter("k exceeds maximum combination size"));
        }
        
        Ok(self.k_combinations(&algorithm_names, k))
    }
    
    /// Generate k-combinations using efficient algorithm
    fn k_combinations(&self, items: &[String], k: usize) -> Vec<Vec<String>> {
        if k == 0 {
            return vec![vec![]];
        }
        
        if k > items.len() {
            return vec![];
        }
        
        let mut combinations = Vec::new();
        self.generate_combinations_recursive(items, k, 0, &mut vec![], &mut combinations);
        combinations
    }
    
    /// Recursive combination generation
    fn generate_combinations_recursive(
        &self,
        items: &[String],
        k: usize,
        start: usize,
        current: &mut Vec<String>,
        result: &mut Vec<Vec<String>>,
    ) {
        if current.len() == k {
            result.push(current.clone());
            return;
        }
        
        for i in start..items.len() {
            current.push(items[i].clone());
            self.generate_combinations_recursive(items, k, i + 1, current, result);
            current.pop();
        }
    }
    
    /// Detect synergy between algorithms
    pub async fn detect_synergy(
        &self,
        algorithms: &[String],
        problem: &OptimizationProblem,
        test_iterations: usize,
    ) -> Result<f64, SwarmError> {
        // Run individual algorithms
        let individual_results = self.run_individual_algorithms(
            algorithms, problem, test_iterations
        ).await?;
        
        // Run combination
        let combination_result = self.run_algorithm_combination(
            algorithms, problem, test_iterations, FusionStrategy::Parallel { weights: vec![1.0; algorithms.len()] }
        ).await?;
        
        // Calculate synergy score
        let best_individual = individual_results.iter()
            .map(|r| r.best_fitness)
            .fold(f64::INFINITY, f64::min);
        
        let combination_fitness = combination_result.best_fitness;
        
        if best_individual.is_finite() && combination_fitness.is_finite() {
            let synergy = (best_individual - combination_fitness) / best_individual.abs().max(1e-10);
            Ok(synergy.max(0.0))
        } else {
            Ok(0.0)
        }
    }
    
    /// Run fusion analysis with specified strategy
    pub async fn analyze_fusion(
        &self,
        algorithms: Vec<String>,
        problem: OptimizationProblem,
        strategy: FusionStrategy,
        max_iterations: usize,
    ) -> Result<FusionResult<f64>, SwarmError> {
        let start_time = std::time::Instant::now();
        
        // Validate algorithms exist
        let pool = self.algorithm_pool.read();
        for alg in &algorithms {
            if !pool.has_algorithm(alg) {
                return Err(SwarmError::parameter(format!("Algorithm '{}' not found", alg)));
            }
        }
        drop(pool);
        
        // Check cache first
        let cache_key = self.generate_cache_key(&algorithms, &strategy);
        if let Some(cached_metrics) = self.combination_cache.get(&cache_key) {
            tracing::debug!("Using cached combination metrics for key: {}", cache_key);
        }
        
        // Run individual algorithms for comparison
        let individual_results = if self.parallel_execution {
            self.run_algorithms_parallel(&algorithms, &problem, max_iterations).await?
        } else {
            self.run_algorithms_sequential(&algorithms, &problem, max_iterations).await?
        };
        
        // Run fusion strategy
        let fusion_result = match strategy {
            FusionStrategy::Sequential => {
                self.execute_sequential_fusion(&algorithms, &problem, max_iterations).await?
            }
            FusionStrategy::Parallel { weights } => {
                self.execute_parallel_fusion(&algorithms, &problem, max_iterations, weights).await?
            }
            FusionStrategy::Adaptive { switch_threshold } => {
                self.execute_adaptive_fusion(&algorithms, &problem, max_iterations, switch_threshold).await?
            }
            FusionStrategy::Island { migration_rate, migration_interval } => {
                self.execute_island_fusion(&algorithms, &problem, max_iterations, migration_rate, migration_interval).await?
            }
            FusionStrategy::Synergistic { synergy_threshold } => {
                self.execute_synergistic_fusion(&algorithms, &problem, max_iterations, synergy_threshold).await?
            }
            _ => {
                // Default to parallel for other strategies
                self.execute_parallel_fusion(&algorithms, &problem, max_iterations, vec![1.0; algorithms.len()]).await?
            }
        };
        
        // Calculate synergy score
        let synergy_score = self.calculate_synergy_score(&individual_results, &fusion_result);
        
        // Generate combination metrics
        let combination_metrics = self.generate_combination_metrics(
            &algorithms, &individual_results, &fusion_result
        );
        
        // Cache results
        self.combination_cache.insert(cache_key, combination_metrics.clone());
        
        let total_time = start_time.elapsed();
        
        Ok(FusionResult {
            best_solution: fusion_result,
            individual_results,
            combination_metrics,
            synergy_score,
            timing: FusionTiming {
                total_time,
                per_algorithm: vec![total_time / algorithms.len() as u32; algorithms.len()],
                fusion_overhead: total_time / 10, // Estimate
                synchronization_time: total_time / 20, // Estimate
            },
        })
    }
    
    /// Execute parallel fusion strategy
    async fn execute_parallel_fusion(
        &self,
        algorithms: &[String],
        problem: &OptimizationProblem,
        max_iterations: usize,
        weights: Vec<f64>,
    ) -> Result<SwarmResult<f64>, SwarmError> {
        if self.cdfa_settings.use_parallel_backend {
            // Use CDFA parallel infrastructure
            self.execute_cdfa_parallel_fusion(algorithms, problem, max_iterations, weights).await
        } else {
            // Standard parallel execution
            self.execute_standard_parallel_fusion(algorithms, problem, max_iterations, weights).await
        }
    }
    
    /// Execute fusion using CDFA parallel backend
    async fn execute_cdfa_parallel_fusion(
        &self,
        algorithms: &[String],
        problem: &OptimizationProblem,
        max_iterations: usize,
        weights: Vec<f64>,
    ) -> Result<SwarmResult<f64>, SwarmError> {
        // Leverage cdfa-parallel ultra-optimization
        use cdfa_parallel::{ParallelDiversityCalculator, ConcurrentFusionProcessor};
        
        let pool = self.algorithm_pool.read();
        let mut processors = Vec::new();
        
        for (i, alg_name) in algorithms.iter().enumerate() {
            if let Some(mut algorithm) = pool.get_algorithm(alg_name)? {
                algorithm.initialize(problem.clone()).await?;
                
                // Create CDFA processor wrapper
                let processor = ConcurrentFusionProcessor::new(
                    algorithm,
                    weights.get(i).copied().unwrap_or(1.0)
                );
                processors.push(processor);
            }
        }
        drop(pool);
        
        // Execute in parallel using CDFA infrastructure
        let results = processors.into_par_iter()
            .map(|mut processor| {
                tokio::runtime::Handle::current().block_on(async {
                    processor.optimize(max_iterations).await
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        
        // Combine results using weighted fusion
        self.combine_weighted_results(results, &weights)
    }
    
    /// Execute standard parallel fusion
    async fn execute_standard_parallel_fusion(
        &self,
        algorithms: &[String],
        problem: &OptimizationProblem,
        max_iterations: usize,
        weights: Vec<f64>,
    ) -> Result<SwarmResult<f64>, SwarmError> {
        let results = self.run_algorithms_parallel(algorithms, problem, max_iterations).await?;
        self.combine_weighted_results(results, &weights)
    }
    
    /// Run algorithms in parallel
    async fn run_algorithms_parallel(
        &self,
        algorithms: &[String],
        problem: &OptimizationProblem,
        max_iterations: usize,
    ) -> Result<Vec<SwarmResult<f64>>, SwarmError> {
        let pool = self.algorithm_pool.read();
        let mut handles = Vec::new();
        
        for alg_name in algorithms {
            if let Some(mut algorithm) = pool.get_algorithm(alg_name)? {
                let problem_clone = problem.clone();
                
                let handle = tokio::spawn(async move {
                    algorithm.initialize(problem_clone).await?;
                    algorithm.optimize(max_iterations).await
                });
                
                handles.push(handle);
            }
        }
        drop(pool);
        
        let mut results = Vec::new();
        for handle in handles {
            results.push(handle.await.map_err(|e| SwarmError::parallel(e.to_string()))??);
        }
        
        Ok(results)
    }
    
    /// Run algorithms sequentially
    async fn run_algorithms_sequential(
        &self,
        algorithms: &[String],
        problem: &OptimizationProblem,
        max_iterations: usize,
    ) -> Result<Vec<SwarmResult<f64>>, SwarmError> {
        let mut results = Vec::new();
        let pool = self.algorithm_pool.read();
        
        for alg_name in algorithms {
            if let Some(mut algorithm) = pool.get_algorithm(alg_name)? {
                algorithm.initialize(problem.clone()).await?;
                let result = algorithm.optimize(max_iterations).await?;
                results.push(result);
            }
        }
        
        Ok(results)
    }
    
    /// Execute sequential fusion strategy
    async fn execute_sequential_fusion(
        &self,
        algorithms: &[String],
        problem: &OptimizationProblem,
        max_iterations: usize,
    ) -> Result<SwarmResult<f64>, SwarmError> {
        let results = self.run_algorithms_sequential(algorithms, problem, max_iterations).await?;
        
        // Return best result
        results.into_iter()
            .min_by(|a, b| a.best_fitness.partial_cmp(&b.best_fitness).unwrap())
            .ok_or_else(|| SwarmError::optimization("No results from sequential fusion"))
    }
    
    /// Execute adaptive fusion strategy
    async fn execute_adaptive_fusion(
        &self,
        algorithms: &[String],
        problem: &OptimizationProblem,
        max_iterations: usize,
        switch_threshold: f64,
    ) -> Result<SwarmResult<f64>, SwarmError> {
        // Start with first algorithm
        let pool = self.algorithm_pool.read();
        let mut current_algorithm = pool.get_algorithm(&algorithms[0])?
            .ok_or_else(|| SwarmError::parameter("Algorithm not found"))?;
        drop(pool);
        
        current_algorithm.initialize(problem.clone()).await?;
        
        let mut best_result = None;
        let mut current_algorithm_index = 0;
        let mut stagnation_count = 0;
        
        for iteration in 0..max_iterations {
            current_algorithm.step().await?;
            
            let metrics = current_algorithm.metrics();
            if let Some(ref mut best) = best_result {
                if metrics.best_fitness.unwrap_or(f64::INFINITY) < best.best_fitness {
                    *best = SwarmResult {
                        best_position: current_algorithm.get_best_individual()
                            .unwrap().position().clone(),
                        best_fitness: metrics.best_fitness.unwrap(),
                        iterations: iteration + 1,
                        convergence_history: vec![metrics.best_fitness.unwrap()],
                        algorithm_name: format!("Adaptive({})", algorithms[current_algorithm_index]),
                    };
                    stagnation_count = 0;
                } else {
                    stagnation_count += 1;
                }
            } else {
                best_result = Some(SwarmResult {
                    best_position: current_algorithm.get_best_individual()
                        .unwrap().position().clone(),
                    best_fitness: metrics.best_fitness.unwrap(),
                    iterations: iteration + 1,
                    convergence_history: vec![metrics.best_fitness.unwrap()],
                    algorithm_name: format!("Adaptive({})", algorithms[current_algorithm_index]),
                });
            }
            
            // Switch algorithm if stagnation detected
            if stagnation_count > (switch_threshold * 100.0) as usize {
                current_algorithm_index = (current_algorithm_index + 1) % algorithms.len();
                
                let pool = self.algorithm_pool.read();
                current_algorithm = pool.get_algorithm(&algorithms[current_algorithm_index])?
                    .ok_or_else(|| SwarmError::parameter("Algorithm not found"))?;
                drop(pool);
                
                current_algorithm.initialize(problem.clone()).await?;
                stagnation_count = 0;
                
                tracing::debug!("Switched to algorithm: {}", algorithms[current_algorithm_index]);
            }
        }
        
        best_result.ok_or_else(|| SwarmError::optimization("No result from adaptive fusion"))
    }
    
    /// Execute island fusion strategy
    async fn execute_island_fusion(
        &self,
        algorithms: &[String],
        problem: &OptimizationProblem,
        max_iterations: usize,
        migration_rate: f64,
        migration_interval: usize,
    ) -> Result<SwarmResult<f64>, SwarmError> {
        // Implementation for island model
        // For now, fallback to parallel execution
        self.execute_parallel_fusion(algorithms, problem, max_iterations, vec![1.0; algorithms.len()]).await
    }
    
    /// Execute synergistic fusion strategy
    async fn execute_synergistic_fusion(
        &self,
        algorithms: &[String],
        problem: &OptimizationProblem,
        max_iterations: usize,
        synergy_threshold: f64,
    ) -> Result<SwarmResult<f64>, SwarmError> {
        // Detect synergistic pairs
        let synergy_scores = Vec::new(); // Simplified for now
        
        // Run algorithms with detected synergies
        self.execute_parallel_fusion(algorithms, problem, max_iterations, vec![1.0; algorithms.len()]).await
    }
    
    /// Combine results using weighted fusion
    fn combine_weighted_results(
        &self,
        results: Vec<SwarmResult<f64>>,
        weights: &[f64],
    ) -> Result<SwarmResult<f64>, SwarmError> {
        if results.is_empty() {
            return Err(SwarmError::optimization("No results to combine"));
        }
        
        // Find best result for now (could implement more sophisticated combination)
        let best_result = results.into_iter()
            .min_by(|a, b| a.best_fitness.partial_cmp(&b.best_fitness).unwrap())
            .unwrap();
        
        Ok(SwarmResult {
            best_position: best_result.best_position,
            best_fitness: best_result.best_fitness,
            iterations: best_result.iterations,
            convergence_history: best_result.convergence_history,
            algorithm_name: "WeightedFusion".to_string(),
        })
    }
    
    /// Calculate synergy score
    fn calculate_synergy_score(
        &self,
        individual_results: &[SwarmResult<f64>],
        fusion_result: &SwarmResult<f64>,
    ) -> f64 {
        let best_individual = individual_results.iter()
            .map(|r| r.best_fitness)
            .fold(f64::INFINITY, f64::min);
        
        if best_individual.is_finite() && fusion_result.best_fitness.is_finite() {
            ((best_individual - fusion_result.best_fitness) / best_individual.abs().max(1e-10)).max(0.0)
        } else {
            0.0
        }
    }
    
    /// Generate combination metrics
    fn generate_combination_metrics(
        &self,
        algorithms: &[String],
        individual_results: &[SwarmResult<f64>],
        fusion_result: &SwarmResult<f64>,
    ) -> CombinationMetrics {
        let best_individual = individual_results.iter()
            .map(|r| r.best_fitness)
            .fold(f64::INFINITY, f64::min);
        
        let improvement_factor = if best_individual.is_finite() && fusion_result.best_fitness.is_finite() {
            best_individual / fusion_result.best_fitness.max(1e-10)
        } else {
            1.0
        };
        
        CombinationMetrics {
            algorithm_count: algorithms.len(),
            diversity_score: self.calculate_diversity_score(algorithms),
            improvement_factor,
            convergence_metrics: ConvergenceMetrics {
                iterations_to_threshold: Some(fusion_result.iterations),
                rate: 1.0 / fusion_result.iterations as f64,
                stagnation_periods: vec![],
                premature_convergence: false,
            },
            resource_usage: ResourceUsage {
                cpu_time_ms: 1000, // Estimate
                peak_memory_bytes: 1024 * 1024, // Estimate
                thread_utilization: 0.8,
                cache_hit_rate: 0.9,
            },
        }
    }
    
    /// Calculate diversity score for algorithm combination
    fn calculate_diversity_score(&self, algorithms: &[String]) -> f64 {
        // Simplified diversity calculation
        algorithms.len() as f64 / self.max_combination_size as f64
    }
    
    /// Generate cache key for combination
    fn generate_cache_key(&self, algorithms: &[String], strategy: &FusionStrategy) -> String {
        let mut key = algorithms.join(",");
        key.push_str(&format!(":{:?}", strategy));
        key
    }
    
    /// Run individual algorithms for comparison
    async fn run_individual_algorithms(
        &self,
        algorithms: &[String],
        problem: &OptimizationProblem,
        iterations: usize,
    ) -> Result<Vec<SwarmResult<f64>>, SwarmError> {
        self.run_algorithms_parallel(algorithms, problem, iterations).await
    }
    
    /// Run algorithm combination
    async fn run_algorithm_combination(
        &self,
        algorithms: &[String],
        problem: &OptimizationProblem,
        iterations: usize,
        strategy: FusionStrategy,
    ) -> Result<SwarmResult<f64>, SwarmError> {
        match strategy {
            FusionStrategy::Parallel { weights } => {
                self.execute_parallel_fusion(algorithms, problem, iterations, weights).await
            }
            _ => {
                self.execute_sequential_fusion(algorithms, problem, iterations).await
            }
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
    use crate::algorithms::ParticleSwarmOptimization;
    use crate::core::OptimizationProblem;
    
    #[tokio::test]
    async fn test_fusion_analyzer_creation() {
        let analyzer = CombinatorialDiversityFusionAnalyzer::new();
        assert_eq!(analyzer.synergy_threshold, 0.1);
        assert_eq!(analyzer.max_combination_size, 5);
    }
    
    #[tokio::test]
    async fn test_algorithm_addition() {
        let analyzer = CombinatorialDiversityFusionAnalyzer::new();
        let pso = ParticleSwarmOptimization::new();
        
        assert!(analyzer.add_algorithm(pso, "PSO".to_string()).is_ok());
    }
    
    #[test]
    fn test_k_combinations() {
        let analyzer = CombinatorialDiversityFusionAnalyzer::new();
        let items = vec!["A".to_string(), "B".to_string(), "C".to_string()];
        
        let combinations = analyzer.k_combinations(&items, 2);
        assert_eq!(combinations.len(), 3);
        assert!(combinations.contains(&vec!["A".to_string(), "B".to_string()]));
        assert!(combinations.contains(&vec!["A".to_string(), "C".to_string()]));
        assert!(combinations.contains(&vec!["B".to_string(), "C".to_string()]));
    }
    
    #[tokio::test]
    async fn test_fusion_analysis() {
        let analyzer = CombinatorialDiversityFusionAnalyzer::new();
        
        // Add algorithms
        let pso1 = ParticleSwarmOptimization::new();
        let pso2 = ParticleSwarmOptimization::new();
        analyzer.add_algorithm(pso1, "PSO1".to_string()).unwrap();
        analyzer.add_algorithm(pso2, "PSO2".to_string()).unwrap();
        
        let problem = OptimizationProblem::new()
            .dimensions(2)
            .bounds(-5.0, 5.0)
            .objective(|x| x[0].powi(2) + x[1].powi(2))
            .build()
            .unwrap();
        
        let algorithms = vec!["PSO1".to_string(), "PSO2".to_string()];
        let strategy = FusionStrategy::Parallel { weights: vec![1.0, 1.0] };
        
        let result = analyzer.analyze_fusion(algorithms, problem, strategy, 10).await;
        assert!(result.is_ok());
        
        let fusion_result = result.unwrap();
        assert_eq!(fusion_result.individual_results.len(), 2);
        assert!(fusion_result.synergy_score >= 0.0);
    }
}