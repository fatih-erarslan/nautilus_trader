//! Algorithm Pool Management
//! 
//! Manages a collection of algorithms for combinatorial fusion analysis,
//! providing dynamic selection, performance tracking, and swarm integration.

use super::{CombinatorialResult, CombinatorialError};
use crate::fusion::FusionMethod;
use ndarray::{Array1, Array2, ArrayView2};
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Trait for algorithms that can be used in combinatorial fusion
pub trait SwarmAlgorithm: Send + Sync {
    /// Unique identifier for the algorithm
    fn id(&self) -> &str;
    
    /// Human-readable name
    fn name(&self) -> &str;
    
    /// Algorithm description
    fn description(&self) -> &str;
    
    /// Execute the algorithm on given data
    fn execute(&self, data: &ArrayView2<f64>) -> CombinatorialResult<Array1<f64>>;
    
    /// Get algorithm metadata
    fn metadata(&self) -> &AlgorithmMetadata;
    
    /// Update performance metrics
    fn update_performance(&mut self, metrics: &PerformanceMetrics);
    
    /// Check if algorithm supports parallel execution
    fn supports_parallel(&self) -> bool { false }
    
    /// Check if algorithm supports SIMD operations
    fn supports_simd(&self) -> bool { false }
}

/// Metadata for algorithms in the pool
#[derive(Debug, Clone)]
pub struct AlgorithmMetadata {
    pub id: String,
    pub name: String,
    pub description: String,
    pub version: String,
    pub author: String,
    pub creation_time: Instant,
    pub category: AlgorithmCategory,
    pub complexity: Complexity,
    pub expected_performance: ExpectedPerformance,
}

/// Algorithm categories for classification
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum AlgorithmCategory {
    ScoreFusion,
    RankFusion,
    Hybrid,
    Statistical,
    MachineLearning,
    Quantum,
    Adaptive,
    Custom(String),
}

/// Computational complexity classification
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Complexity {
    Constant,     // O(1)
    Logarithmic,  // O(log n)
    Linear,       // O(n)
    Quadratic,    // O(n²)
    Exponential,  // O(2^n)
}

/// Expected performance characteristics
#[derive(Debug, Clone)]
pub struct ExpectedPerformance {
    pub typical_runtime_ns: u64,
    pub memory_usage_bytes: usize,
    pub accuracy_score: f64,  // 0.0 to 1.0
    pub stability_score: f64, // 0.0 to 1.0
}

/// Runtime performance metrics
#[derive(Debug, Clone)]
pub struct PerformanceMetrics {
    pub execution_time: Duration,
    pub memory_allocated: usize,
    pub memory_peak: usize,
    pub accuracy: Option<f64>,
    pub error_rate: f64,
    pub cache_performance: CacheMetrics,
}

/// Cache performance tracking
#[derive(Debug, Clone)]
pub struct CacheMetrics {
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
}

/// Algorithm pool managing multiple fusion algorithms
pub struct AlgorithmPool {
    algorithms: HashMap<String, Box<dyn SwarmAlgorithm>>,
    performance_history: HashMap<String, Vec<PerformanceMetrics>>,
    selection_strategy: SelectionStrategy,
    cache: ResultsCache,
}

/// Strategy for selecting algorithms from the pool
#[derive(Debug, Clone)]
pub enum SelectionStrategy {
    /// Select by performance score
    Performance,
    /// Select by diversity contribution
    Diversity,
    /// Balanced selection (performance + diversity)
    Balanced { performance_weight: f64 },
    /// Random selection for exploration
    Random,
    /// Custom selection function
    Custom,
}

/// Cache for algorithm results to improve performance
struct ResultsCache {
    cache: HashMap<String, (Array1<f64>, Instant)>,
    max_size: usize,
    ttl: Duration,
}

impl AlgorithmPool {
    /// Create a new algorithm pool
    pub fn new() -> Self {
        Self {
            algorithms: HashMap::new(),
            performance_history: HashMap::new(),
            selection_strategy: SelectionStrategy::Balanced { performance_weight: 0.7 },
            cache: ResultsCache::new(1000, Duration::from_secs(300)), // 5-minute TTL
        }
    }
    
    /// Add an algorithm to the pool
    pub fn add_algorithm(&mut self, algorithm: Box<dyn SwarmAlgorithm>) -> CombinatorialResult<()> {
        let id = algorithm.id().to_string();
        
        if self.algorithms.contains_key(&id) {
            return Err(CombinatorialError::EvaluationFailed {
                reason: format!("Algorithm with ID '{}' already exists", id),
            });
        }
        
        self.algorithms.insert(id.clone(), algorithm);
        self.performance_history.insert(id, Vec::new());
        Ok(())
    }
    
    /// Remove an algorithm from the pool
    pub fn remove_algorithm(&mut self, id: &str) -> CombinatorialResult<()> {
        if !self.algorithms.contains_key(id) {
            return Err(CombinatorialError::EvaluationFailed {
                reason: format!("Algorithm with ID '{}' not found", id),
            });
        }
        
        self.algorithms.remove(id);
        self.performance_history.remove(id);
        Ok(())
    }
    
    /// Get algorithm by ID
    pub fn get_algorithm(&self, id: &str) -> Option<&dyn SwarmAlgorithm> {
        self.algorithms.get(id).map(|a| a.as_ref())
    }
    
    /// List all algorithm IDs
    pub fn list_algorithms(&self) -> Vec<String> {
        self.algorithms.keys().cloned().collect()
    }
    
    /// Get number of algorithms in pool
    pub fn size(&self) -> usize {
        self.algorithms.len()
    }
    
    /// Select k algorithms based on current strategy
    pub fn select_algorithms(&self, k: usize, data: Option<&ArrayView2<f64>>) -> CombinatorialResult<Vec<String>> {
        if k > self.algorithms.len() {
            return Err(CombinatorialError::InsufficientAlgorithms {
                count: self.algorithms.len(),
                min: k,
            });
        }
        
        match self.selection_strategy {
            SelectionStrategy::Performance => self.select_by_performance(k),
            SelectionStrategy::Diversity => self.select_by_diversity(k, data),
            SelectionStrategy::Balanced { performance_weight } => {
                self.select_balanced(k, performance_weight, data)
            },
            SelectionStrategy::Random => self.select_random(k),
            SelectionStrategy::Custom => self.select_custom(k, data),
        }
    }
    
    /// Execute algorithm and track performance
    pub fn execute_algorithm(&mut self, id: &str, data: &ArrayView2<f64>) -> CombinatorialResult<Array1<f64>> {
        // Check cache first
        if let Some(cached) = self.cache.get(&format!("{}_data_hash", id)) {
            return Ok(cached);
        }
        
        let algorithm = self.algorithms.get_mut(id)
            .ok_or_else(|| CombinatorialError::EvaluationFailed {
                reason: format!("Algorithm '{}' not found", id),
            })?;
        
        let start = Instant::now();
        let result = algorithm.execute(data)?;
        let execution_time = start.elapsed();
        
        // Update performance metrics
        let metrics = PerformanceMetrics {
            execution_time,
            memory_allocated: 0, // Would need memory profiling
            memory_peak: 0,
            accuracy: None,
            error_rate: 0.0,
            cache_performance: CacheMetrics { hits: 0, misses: 0, evictions: 0 },
        };
        
        algorithm.update_performance(&metrics);
        self.performance_history.get_mut(id).unwrap().push(metrics);
        
        // Cache result
        self.cache.insert(format!("{}_data_hash", id), result.clone());
        
        Ok(result)
    }
    
    /// Get performance statistics for an algorithm
    pub fn get_performance_stats(&self, id: &str) -> Option<AlgorithmStats> {
        let history = self.performance_history.get(id)?;
        if history.is_empty() {
            return None;
        }
        
        let avg_execution_time = history.iter()
            .map(|m| m.execution_time.as_nanos() as f64)
            .sum::<f64>() / history.len() as f64;
            
        let avg_memory = history.iter()
            .map(|m| m.memory_allocated as f64)
            .sum::<f64>() / history.len() as f64;
            
        let avg_error_rate = history.iter()
            .map(|m| m.error_rate)
            .sum::<f64>() / history.len() as f64;
        
        Some(AlgorithmStats {
            executions: history.len(),
            avg_execution_time_ns: avg_execution_time as u64,
            avg_memory_usage: avg_memory as usize,
            avg_error_rate,
            last_execution: history.last().unwrap().execution_time,
        })
    }
    
    // Private selection methods
    fn select_by_performance(&self, k: usize) -> CombinatorialResult<Vec<String>> {
        let mut scored_algorithms: Vec<(String, f64)> = self.algorithms
            .keys()
            .filter_map(|id| {
                self.get_performance_stats(id).map(|stats| {
                    // Lower execution time and error rate = higher score
                    let score = 1.0 / (1.0 + stats.avg_execution_time_ns as f64 / 1000.0 + stats.avg_error_rate);
                    (id.clone(), score)
                })
            })
            .collect();
        
        scored_algorithms.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        Ok(scored_algorithms.into_iter().take(k).map(|(id, _)| id).collect())
    }
    
    fn select_by_diversity(&self, k: usize, _data: Option<&ArrayView2<f64>>) -> CombinatorialResult<Vec<String>> {
        // For now, select algorithms from different categories
        let mut selected = Vec::new();
        let mut categories_used = std::collections::HashSet::new();
        
        for id in self.algorithms.keys() {
            if selected.len() >= k {
                break;
            }
            
            let algorithm = self.algorithms.get(id).unwrap();
            let category = &algorithm.metadata().category;
            
            if !categories_used.contains(category) {
                selected.push(id.clone());
                categories_used.insert(category.clone());
            }
        }
        
        // Fill remaining slots with any available algorithms
        for id in self.algorithms.keys() {
            if selected.len() >= k {
                break;
            }
            if !selected.contains(id) {
                selected.push(id.clone());
            }
        }
        
        Ok(selected)
    }
    
    fn select_balanced(&self, k: usize, performance_weight: f64, data: Option<&ArrayView2<f64>>) -> CombinatorialResult<Vec<String>> {
        let performance_selected = self.select_by_performance((k as f64 * performance_weight).ceil() as usize)?;
        let diversity_selected = self.select_by_diversity((k as f64 * (1.0 - performance_weight)).ceil() as usize, data)?;
        
        let mut combined = performance_selected;
        for id in diversity_selected {
            if combined.len() >= k {
                break;
            }
            if !combined.contains(&id) {
                combined.push(id);
            }
        }
        
        combined.truncate(k);
        Ok(combined)
    }
    
    fn select_random(&self, k: usize) -> CombinatorialResult<Vec<String>> {
        use rand::seq::SliceRandom;
        let mut rng = rand::thread_rng();
        let mut ids: Vec<String> = self.algorithms.keys().cloned().collect();
        ids.shuffle(&mut rng);
        Ok(ids.into_iter().take(k).collect())
    }
    
    fn select_custom(&self, k: usize, _data: Option<&ArrayView2<f64>>) -> CombinatorialResult<Vec<String>> {
        // Placeholder for custom selection logic
        self.select_by_performance(k)
    }
}

/// Algorithm performance statistics
#[derive(Debug, Clone)]
pub struct AlgorithmStats {
    pub executions: usize,
    pub avg_execution_time_ns: u64,
    pub avg_memory_usage: usize,
    pub avg_error_rate: f64,
    pub last_execution: Duration,
}

impl ResultsCache {
    fn new(max_size: usize, ttl: Duration) -> Self {
        Self {
            cache: HashMap::new(),
            max_size,
            ttl,
        }
    }
    
    fn get(&self, key: &str) -> Option<Array1<f64>> {
        if let Some((result, timestamp)) = self.cache.get(key) {
            if timestamp.elapsed() <= self.ttl {
                return Some(result.clone());
            }
        }
        None
    }
    
    fn insert(&mut self, key: String, value: Array1<f64>) {
        if self.cache.len() >= self.max_size {
            // Simple LRU: remove oldest entry
            if let Some(oldest_key) = self.cache.keys().next().cloned() {
                self.cache.remove(&oldest_key);
            }
        }
        self.cache.insert(key, (value, Instant::now()));
    }
}

// Built-in algorithm implementations for common fusion methods
pub struct BuiltinFusionAlgorithm {
    metadata: AlgorithmMetadata,
    method: FusionMethod,
    performance_metrics: Vec<PerformanceMetrics>,
}

impl BuiltinFusionAlgorithm {
    pub fn new(method: FusionMethod) -> Self {
        let metadata = AlgorithmMetadata {
            id: format!("builtin_{:?}", method).to_lowercase(),
            name: format!("{:?} Fusion", method),
            description: format!("Built-in {} fusion algorithm", format!("{:?}", method).to_lowercase()),
            version: "1.0.0".to_string(),
            author: "CDFA Core Team".to_string(),
            creation_time: Instant::now(),
            category: match method {
                FusionMethod::Average | FusionMethod::WeightedAverage | FusionMethod::NormalizedAverage => 
                    AlgorithmCategory::ScoreFusion,
                FusionMethod::BordaCount | FusionMethod::MedianRank => 
                    AlgorithmCategory::RankFusion,
                FusionMethod::Hybrid => 
                    AlgorithmCategory::Hybrid,
                FusionMethod::Adaptive => 
                    AlgorithmCategory::Adaptive,
                _ => AlgorithmCategory::Statistical,
            },
            complexity: Complexity::Linear,
            expected_performance: ExpectedPerformance {
                typical_runtime_ns: 100,
                memory_usage_bytes: 1024,
                accuracy_score: 0.85,
                stability_score: 0.9,
            },
        };
        
        Self {
            metadata,
            method,
            performance_metrics: Vec::new(),
        }
    }
}

impl SwarmAlgorithm for BuiltinFusionAlgorithm {
    fn id(&self) -> &str {
        &self.metadata.id
    }
    
    fn name(&self) -> &str {
        &self.metadata.name
    }
    
    fn description(&self) -> &str {
        &self.metadata.description
    }
    
    fn execute(&self, data: &ArrayView2<f64>) -> CombinatorialResult<Array1<f64>> {
        use crate::fusion::CdfaFusion;
        
        CdfaFusion::fuse(data, self.method, None)
            .map_err(|e| CombinatorialError::EvaluationFailed {
                reason: format!("Fusion execution failed: {}", e),
            })
    }
    
    fn metadata(&self) -> &AlgorithmMetadata {
        &self.metadata
    }
    
    fn update_performance(&mut self, metrics: &PerformanceMetrics) {
        self.performance_metrics.push(metrics.clone());
    }
    
    fn supports_parallel(&self) -> bool {
        matches!(self.method, 
            FusionMethod::Average | 
            FusionMethod::WeightedAverage | 
            FusionMethod::NormalizedAverage |
            FusionMethod::BordaCount
        )
    }
    
    fn supports_simd(&self) -> bool {
        matches!(self.method, 
            FusionMethod::Average | 
            FusionMethod::WeightedAverage
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;
    
    #[test]
    fn test_algorithm_pool_creation() {
        let pool = AlgorithmPool::new();
        assert_eq!(pool.size(), 0);
    }
    
    #[test]
    fn test_add_builtin_algorithm() {
        let mut pool = AlgorithmPool::new();
        let algorithm = Box::new(BuiltinFusionAlgorithm::new(FusionMethod::Average));
        
        assert!(pool.add_algorithm(algorithm).is_ok());
        assert_eq!(pool.size(), 1);
    }
    
    #[test]
    fn test_algorithm_execution() {
        let mut pool = AlgorithmPool::new();
        let algorithm = Box::new(BuiltinFusionAlgorithm::new(FusionMethod::Average));
        let id = algorithm.id().to_string();
        
        pool.add_algorithm(algorithm).unwrap();
        
        let data = array![
            [0.8, 0.6, 0.9],
            [0.7, 0.8, 0.6]
        ];
        
        let result = pool.execute_algorithm(&id, &data.view()).unwrap();
        assert_eq!(result.len(), 3);
    }
}