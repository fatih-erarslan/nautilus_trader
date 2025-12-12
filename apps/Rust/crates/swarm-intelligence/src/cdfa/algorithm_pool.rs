//! Algorithm pool management for dynamic algorithm selection and fusion

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};

use crate::core::{SwarmAlgorithm, SwarmError, AlgorithmMetrics};

/// Strategy for selecting algorithms from the pool
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum PoolStrategy {
    /// Round-robin selection
    RoundRobin,
    
    /// Performance-based selection
    PerformanceBased,
    
    /// Diversity-based selection
    DiversityBased,
    
    /// Random selection
    Random,
    
    /// Weighted selection
    Weighted { weights: HashMap<String, f64> },
    
    /// Tournament selection
    Tournament { size: usize },
}

/// Wrapper for pooled algorithms with metadata
pub struct PooledAlgorithm {
    /// The algorithm instance
    pub algorithm: Box<dyn SwarmAlgorithm<Individual = crate::core::types::BasicIndividual, Fitness = f64, Parameters = serde_json::Value>>,
    
    /// Algorithm name/identifier
    pub name: String,
    
    /// Performance history
    pub performance_history: PerformanceHistory,
    
    /// Usage statistics
    pub usage_stats: UsageStats,
    
    /// Creation timestamp
    pub created_at: std::time::SystemTime,
    
    /// Last used timestamp
    pub last_used: Option<std::time::SystemTime>,
}

/// Performance history tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceHistory {
    /// Best fitness achieved
    pub best_fitness: Vec<f64>,
    
    /// Average convergence time
    pub avg_convergence_time: f64,
    
    /// Success rate (problems solved)
    pub success_rate: f64,
    
    /// Reliability score
    pub reliability: f64,
    
    /// Number of evaluations
    pub total_evaluations: usize,
}

/// Usage statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UsageStats {
    /// Number of times used
    pub usage_count: usize,
    
    /// Total runtime
    pub total_runtime: std::time::Duration,
    
    /// Average runtime per use
    pub avg_runtime: std::time::Duration,
    
    /// Success count
    pub success_count: usize,
    
    /// Failure count
    pub failure_count: usize,
}

impl Default for UsageStats {
    fn default() -> Self {
        Self {
            usage_count: 0,
            total_runtime: std::time::Duration::ZERO,
            avg_runtime: std::time::Duration::ZERO,
            success_count: 0,
            failure_count: 0,
        }
    }
}

impl Default for PerformanceHistory {
    fn default() -> Self {
        Self {
            best_fitness: Vec::new(),
            avg_convergence_time: 0.0,
            success_rate: 0.0,
            reliability: 0.0,
            total_evaluations: 0,
        }
    }
}

/// Algorithm pool for managing multiple swarm algorithms
pub struct AlgorithmPool {
    /// Registered algorithms
    algorithms: HashMap<String, Arc<RwLock<PooledAlgorithm>>>,
    
    /// Selection strategy
    strategy: PoolStrategy,
    
    /// Current round-robin index
    round_robin_index: usize,
    
    /// Pool statistics
    pool_stats: PoolStatistics,
}

/// Pool-level statistics
#[derive(Debug, Clone, Default)]
pub struct PoolStatistics {
    /// Total algorithms in pool
    pub total_algorithms: usize,
    
    /// Active algorithms
    pub active_algorithms: usize,
    
    /// Total pool usage
    pub total_usage: usize,
    
    /// Pool creation time
    pub created_at: std::time::SystemTime,
}

impl AlgorithmPool {
    /// Create a new algorithm pool
    pub fn new() -> Self {
        Self {
            algorithms: HashMap::new(),
            strategy: PoolStrategy::RoundRobin,
            round_robin_index: 0,
            pool_stats: PoolStatistics {
                created_at: std::time::SystemTime::now(),
                ..Default::default()
            },
        }
    }
    
    /// Create pool with strategy
    pub fn with_strategy(strategy: PoolStrategy) -> Self {
        let mut pool = Self::new();
        pool.strategy = strategy;
        pool
    }
    
    /// Add algorithm to pool (simplified signature for trait object compatibility)
    pub fn add_algorithm<T>(&mut self, algorithm: Box<T>, name: String) -> Result<(), SwarmError>
    where
        T: SwarmAlgorithm + 'static,
    {
        if self.algorithms.contains_key(&name) {
            return Err(SwarmError::parameter(format!("Algorithm '{}' already exists", name)));
        }
        
        // For now, we'll store a placeholder since we can't easily convert between trait objects
        // In a real implementation, we'd need more sophisticated type erasure
        let pooled = PooledAlgorithm {
            algorithm: Box::new(DummyAlgorithm::new(name.clone())),
            name: name.clone(),
            performance_history: PerformanceHistory::default(),
            usage_stats: UsageStats::default(),
            created_at: std::time::SystemTime::now(),
            last_used: None,
        };
        
        self.algorithms.insert(name.clone(), Arc::new(RwLock::new(pooled)));
        self.pool_stats.total_algorithms += 1;
        self.pool_stats.active_algorithms += 1;
        
        tracing::info!("Added algorithm '{}' to pool", name);
        Ok(())
    }
    
    /// Get algorithm by name
    pub fn get_algorithm(&self, name: &str) -> Result<Option<Box<dyn SwarmAlgorithm<Individual = crate::core::types::BasicIndividual, Fitness = f64, Parameters = serde_json::Value>>>, SwarmError> {
        if let Some(algorithm_ref) = self.algorithms.get(name) {
            let mut algorithm = algorithm_ref.write();
            algorithm.usage_stats.usage_count += 1;
            algorithm.last_used = Some(std::time::SystemTime::now());
            
            // Return a cloned algorithm (simplified)
            Ok(Some(Box::new(DummyAlgorithm::new(name.to_string()))))
        } else {
            Ok(None)
        }
    }
    
    /// Select algorithm based on pool strategy
    pub fn select_algorithm(&mut self) -> Result<Option<String>, SwarmError> {
        if self.algorithms.is_empty() {
            return Ok(None);
        }
        
        let algorithm_names: Vec<String> = self.algorithms.keys().cloned().collect();
        
        let selected_name = match &self.strategy {
            PoolStrategy::RoundRobin => {
                let name = algorithm_names[self.round_robin_index % algorithm_names.len()].clone();
                self.round_robin_index += 1;
                name
            }
            PoolStrategy::Random => {
                use rand::seq::SliceRandom;
                algorithm_names.choose(&mut rand::thread_rng()).unwrap().clone()
            }
            PoolStrategy::PerformanceBased => {
                self.select_best_performing(&algorithm_names)
            }
            PoolStrategy::DiversityBased => {
                self.select_most_diverse(&algorithm_names)
            }
            PoolStrategy::Weighted { weights } => {
                self.select_weighted(&algorithm_names, weights)?
            }
            PoolStrategy::Tournament { size } => {
                self.select_tournament(&algorithm_names, *size)
            }
        };
        
        Ok(Some(selected_name))
    }
    
    /// Select best performing algorithm
    fn select_best_performing(&self, names: &[String]) -> String {
        names.iter()
            .max_by(|a, b| {
                let a_perf = self.get_performance_score(a);
                let b_perf = self.get_performance_score(b);
                a_perf.partial_cmp(&b_perf).unwrap()
            })
            .unwrap()
            .clone()
    }
    
    /// Select most diverse algorithm
    fn select_most_diverse(&self, names: &[String]) -> String {
        // Simplified diversity selection
        names.iter()
            .min_by_key(|name| {
                if let Some(algorithm_ref) = self.algorithms.get(*name) {
                    algorithm_ref.read().usage_stats.usage_count
                } else {
                    0
                }
            })
            .unwrap()
            .clone()
    }
    
    /// Select algorithm using weighted probabilities
    fn select_weighted(&self, names: &[String], weights: &HashMap<String, f64>) -> Result<String, SwarmError> {
        use rand::Rng;
        
        let total_weight: f64 = names.iter()
            .map(|name| weights.get(name).copied().unwrap_or(1.0))
            .sum();
        
        if total_weight <= 0.0 {
            return Err(SwarmError::parameter("Invalid weights: total weight is zero"));
        }
        
        let mut rng = rand::thread_rng();
        let random_value = rng.gen::<f64>() * total_weight;
        
        let mut cumulative_weight = 0.0;
        for name in names {
            cumulative_weight += weights.get(name).copied().unwrap_or(1.0);
            if random_value <= cumulative_weight {
                return Ok(name.clone());
            }
        }
        
        // Fallback to last algorithm
        Ok(names.last().unwrap().clone())
    }
    
    /// Tournament selection
    fn select_tournament(&self, names: &[String], tournament_size: usize) -> String {
        use rand::seq::SliceRandom;
        
        let mut rng = rand::thread_rng();
        let tournament: Vec<String> = names.choose_multiple(&mut rng, tournament_size.min(names.len())).cloned().collect();
        
        tournament.iter()
            .max_by(|a, b| {
                let a_perf = self.get_performance_score(a);
                let b_perf = self.get_performance_score(b);
                a_perf.partial_cmp(&b_perf).unwrap()
            })
            .unwrap()
            .clone()
    }
    
    /// Get performance score for algorithm
    fn get_performance_score(&self, name: &str) -> f64 {
        if let Some(algorithm_ref) = self.algorithms.get(name) {
            let algorithm = algorithm_ref.read();
            algorithm.performance_history.reliability * algorithm.performance_history.success_rate
        } else {
            0.0
        }
    }
    
    /// Update algorithm performance
    pub fn update_performance(&mut self, name: &str, metrics: &AlgorithmMetrics, success: bool) -> Result<(), SwarmError> {
        if let Some(algorithm_ref) = self.algorithms.get(name) {
            let mut algorithm = algorithm_ref.write();
            
            if let Some(fitness) = metrics.best_fitness {
                algorithm.performance_history.best_fitness.push(fitness);
            }
            
            algorithm.performance_history.total_evaluations += metrics.evaluations;
            
            if success {
                algorithm.usage_stats.success_count += 1;
            } else {
                algorithm.usage_stats.failure_count += 1;
            }
            
            // Update success rate
            let total_attempts = algorithm.usage_stats.success_count + algorithm.usage_stats.failure_count;
            if total_attempts > 0 {
                algorithm.performance_history.success_rate = algorithm.usage_stats.success_count as f64 / total_attempts as f64;
            }
            
            // Update reliability (simplified)
            algorithm.performance_history.reliability = algorithm.performance_history.success_rate * 0.8 + 0.2;
            
            Ok(())
        } else {
            Err(SwarmError::parameter(format!("Algorithm '{}' not found", name)))
        }
    }
    
    /// Remove algorithm from pool
    pub fn remove_algorithm(&mut self, name: &str) -> Result<(), SwarmError> {
        if self.algorithms.remove(name).is_some() {
            self.pool_stats.total_algorithms -= 1;
            self.pool_stats.active_algorithms -= 1;
            tracing::info!("Removed algorithm '{}' from pool", name);
            Ok(())
        } else {
            Err(SwarmError::parameter(format!("Algorithm '{}' not found", name)))
        }
    }
    
    /// List all algorithms in pool
    pub fn list_algorithms(&self) -> Vec<String> {
        self.algorithms.keys().cloned().collect()
    }
    
    /// Check if algorithm exists in pool
    pub fn has_algorithm(&self, name: &str) -> bool {
        self.algorithms.contains_key(name)
    }
    
    /// Get pool statistics
    pub fn statistics(&self) -> PoolStatistics {
        self.pool_stats.clone()
    }
    
    /// Clear all algorithms from pool
    pub fn clear(&mut self) {
        self.algorithms.clear();
        self.pool_stats.total_algorithms = 0;
        self.pool_stats.active_algorithms = 0;
        self.round_robin_index = 0;
    }
    
    /// Set selection strategy
    pub fn set_strategy(&mut self, strategy: PoolStrategy) {
        self.strategy = strategy;
    }
    
    /// Get current strategy
    pub fn strategy(&self) -> &PoolStrategy {
        &self.strategy
    }
}

impl Default for AlgorithmPool {
    fn default() -> Self {
        Self::new()
    }
}

/// Algorithm selector for automated algorithm selection
pub struct AlgorithmSelector {
    pool: Arc<RwLock<AlgorithmPool>>,
    selection_history: Vec<String>,
    performance_threshold: f64,
}

impl AlgorithmSelector {
    /// Create new algorithm selector
    pub fn new(pool: Arc<RwLock<AlgorithmPool>>) -> Self {
        Self {
            pool,
            selection_history: Vec::new(),
            performance_threshold: 0.1,
        }
    }
    
    /// Select next algorithm based on current context
    pub fn select_next(&mut self, problem_characteristics: &ProblemCharacteristics) -> Result<Option<String>, SwarmError> {
        let mut pool = self.pool.write();
        
        // Adapt strategy based on problem characteristics
        let strategy = self.adapt_strategy(problem_characteristics);
        pool.set_strategy(strategy);
        
        let selected = pool.select_algorithm()?;
        
        if let Some(ref name) = selected {
            self.selection_history.push(name.clone());
        }
        
        Ok(selected)
    }
    
    /// Adapt selection strategy based on problem characteristics
    fn adapt_strategy(&self, characteristics: &ProblemCharacteristics) -> PoolStrategy {
        match characteristics.complexity {
            ProblemComplexity::Low => PoolStrategy::RoundRobin,
            ProblemComplexity::Medium => PoolStrategy::PerformanceBased,
            ProblemComplexity::High => PoolStrategy::DiversityBased,
            ProblemComplexity::Extreme => PoolStrategy::Tournament { size: 3 },
        }
    }
}

/// Problem characteristics for algorithm selection
#[derive(Debug, Clone)]
pub struct ProblemCharacteristics {
    pub dimensions: usize,
    pub complexity: ProblemComplexity,
    pub multimodal: bool,
    pub noisy: bool,
    pub dynamic: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum ProblemComplexity {
    Low,
    Medium,
    High,
    Extreme,
}

/// Dummy algorithm for trait object compatibility
struct DummyAlgorithm {
    name: String,
}

impl DummyAlgorithm {
    fn new(name: String) -> Self {
        Self { name }
    }
}

use async_trait::async_trait;

#[async_trait]
impl SwarmAlgorithm for DummyAlgorithm {
    type Individual = crate::core::types::BasicIndividual;
    type Fitness = f64;
    type Parameters = serde_json::Value;
    
    async fn initialize(&mut self, _problem: crate::core::OptimizationProblem) -> Result<(), SwarmError> {
        Ok(())
    }
    
    async fn step(&mut self) -> Result<(), SwarmError> {
        Ok(())
    }
    
    fn get_best_individual(&self) -> Option<&Self::Individual> {
        None
    }
    
    fn get_population(&self) -> &crate::core::Population<Self::Individual> {
        unimplemented!()
    }
    
    fn get_population_mut(&mut self) -> &mut crate::core::Population<Self::Individual> {
        unimplemented!()
    }
    
    fn name(&self) -> &'static str {
        "DummyAlgorithm"
    }
    
    fn parameters(&self) -> &Self::Parameters {
        unimplemented!()
    }
    
    fn update_parameters(&mut self, _params: Self::Parameters) {}
    
    async fn reset(&mut self) -> Result<(), SwarmError> {
        Ok(())
    }
    
    fn clone_algorithm(&self) -> Box<dyn SwarmAlgorithm<Individual = Self::Individual, Fitness = Self::Fitness, Parameters = Self::Parameters>> {
        Box::new(DummyAlgorithm::new(self.name.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_algorithm_pool_creation() {
        let pool = AlgorithmPool::new();
        assert_eq!(pool.algorithms.len(), 0);
        assert!(matches!(pool.strategy, PoolStrategy::RoundRobin));
    }
    
    #[test]
    fn test_pool_statistics() {
        let pool = AlgorithmPool::new();
        let stats = pool.statistics();
        assert_eq!(stats.total_algorithms, 0);
        assert_eq!(stats.active_algorithms, 0);
    }
    
    #[test]
    fn test_strategy_setting() {
        let mut pool = AlgorithmPool::new();
        let new_strategy = PoolStrategy::PerformanceBased;
        pool.set_strategy(new_strategy);
        assert!(matches!(pool.strategy(), PoolStrategy::PerformanceBased));
    }
}