//! Evolution engine for parasitic organisms - Complete genetic algorithm implementation
//! Sub-millisecond performance with ZERO mocks policy
//! Real market-driven organism evolution with neural capabilities

pub mod genetic_algorithm;
pub mod fitness_evaluator;
pub mod mutation_engine;
pub mod crossover_engine;
pub mod selection_pressure;
pub mod population_manager;
pub mod neural_evolution;

#[cfg(test)]
pub mod tests;

// Re-export main components for easier access
pub use genetic_algorithm::{GeneticAlgorithm, GeneticAlgorithmConfig, EvolutionResult, EvolutionStatistics};
pub use fitness_evaluator::{FitnessEvaluator, FitnessEvaluationConfig, MarketConditions, FitnessScore};
pub use mutation_engine::{MutationEngine, MutationEngineConfig, MutationResult, MutationStatistics, MutationStrategy};

use std::sync::{Arc, atomic::{AtomicU64, Ordering}};
use dashmap::DashMap;
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock;

use crate::organisms::{ParasiticOrganism, OrganismGenetics};

/// Comprehensive evolution engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionEngineConfig {
    pub genetic_algorithm: GeneticAlgorithmConfig,
    pub fitness_evaluation: FitnessEvaluationConfig,
    pub mutation_engine: MutationEngineConfig,
    pub enable_neural_evolution: bool,
    pub performance_target_ms: f64,
    pub max_generations: Option<u64>,
    pub convergence_threshold: f64,
}

impl Default for EvolutionEngineConfig {
    fn default() -> Self {
        Self {
            genetic_algorithm: GeneticAlgorithmConfig::default(),
            fitness_evaluation: FitnessEvaluationConfig::default(),
            mutation_engine: MutationEngineConfig::default(),
            enable_neural_evolution: true,
            performance_target_ms: 1.0, // Sub-millisecond target
            max_generations: Some(1000),
            convergence_threshold: 0.001,
        }
    }
}

/// Comprehensive evolution status combining all subsystems
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComprehensiveEvolutionStatus {
    pub genetic_algorithm_stats: EvolutionStatistics,
    pub mutation_stats: MutationStatistics,
    pub current_generation: u64,
    pub population_size: usize,
    pub convergence_progress: f64,
    pub performance_metrics: EvolutionPerformanceMetrics,
    pub neural_evolution_active: bool,
}

/// Performance metrics for the evolution engine
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionPerformanceMetrics {
    pub average_evolution_time_ms: f64,
    pub total_evolution_time_ms: f64,
    pub mutations_per_second: f64,
    pub evaluations_per_second: f64,
    pub memory_efficiency_score: f64,
    pub sub_millisecond_compliance: bool,
}

/// Main Evolution Engine integrating all components
pub struct EvolutionEngine {
    config: Arc<RwLock<EvolutionEngineConfig>>,
    genetic_algorithm: genetic_algorithm::GeneticAlgorithm,
    fitness_evaluator: fitness_evaluator::FitnessEvaluator,
    mutation_engine: mutation_engine::MutationEngine,
    current_generation: Arc<AtomicU64>,
    performance_metrics: Arc<RwLock<EvolutionPerformanceMetrics>>,
    is_converged: Arc<std::sync::Mutex<bool>>,
}

impl EvolutionEngine {
    /// Create new evolution engine with comprehensive configuration
    pub fn new(config: EvolutionEngineConfig) -> Self {
        let genetic_algorithm = genetic_algorithm::GeneticAlgorithm::new(config.genetic_algorithm.clone());
        let fitness_evaluator = fitness_evaluator::FitnessEvaluator::new(config.fitness_evaluation.clone());
        let mutation_engine = mutation_engine::MutationEngine::new(config.mutation_engine.clone());
        
        let initial_performance = EvolutionPerformanceMetrics {
            average_evolution_time_ms: 0.0,
            total_evolution_time_ms: 0.0,
            mutations_per_second: 0.0,
            evaluations_per_second: 0.0,
            memory_efficiency_score: 1.0,
            sub_millisecond_compliance: true,
        };
        
        Self {
            config: Arc::new(RwLock::new(config)),
            genetic_algorithm,
            fitness_evaluator,
            mutation_engine,
            current_generation: Arc::new(AtomicU64::new(0)),
            performance_metrics: Arc::new(RwLock::new(initial_performance)),
            is_converged: Arc::new(std::sync::Mutex::new(false)),
        }
    }
    
    /// Main evolution cycle - orchestrates all components with sub-millisecond performance
    pub async fn evolve_organisms(
        &mut self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
        market_conditions: &MarketConditions,
    ) -> Result<ComprehensiveEvolutionStatus, Box<dyn std::error::Error + Send + Sync>> {
        let cycle_start = std::time::Instant::now();
        
        // Check convergence before proceeding
        if *self.is_converged.lock().unwrap() {
            return Ok(self.get_status(organisms, market_conditions).await);
        }
        
        // Step 1: Fitness Evaluation (parallel)
        let fitness_start = std::time::Instant::now();
        let population_fitness = self.fitness_evaluator.evaluate_population_fitness(organisms, market_conditions).await?;
        let fitness_time = fitness_start.elapsed();
        
        // Calculate population diversity for adaptive parameters
        let population_diversity = self.calculate_population_diversity(organisms).await;
        
        // Step 2: Genetic Algorithm Evolution (selection, crossover, elimination)
        let ga_start = std::time::Instant::now();
        let evolution_result = self.genetic_algorithm.evolve_population(organisms).await?;
        let ga_time = ga_start.elapsed();
        
        // Step 3: Mutation Phase (adaptive based on diversity)
        let mutation_start = std::time::Instant::now();
        let mutation_result = self.mutation_engine.apply_mutations(organisms, population_diversity).await?;
        let mutation_time = mutation_start.elapsed();
        
        // Update generation counter
        let generation = self.current_generation.fetch_add(1, Ordering::SeqCst) + 1;
        
        // Check convergence
        let convergence_progress = self.check_convergence(&population_fitness, population_diversity).await;
        
        // Update performance metrics
        let total_cycle_time = cycle_start.elapsed();
        self.update_performance_metrics(
            total_cycle_time,
            fitness_time,
            ga_time,
            mutation_time,
            population_fitness.len(),
            mutation_result.mutations_applied,
        ).await;
        
        // Create comprehensive status
        let status = ComprehensiveEvolutionStatus {
            genetic_algorithm_stats: self.genetic_algorithm.get_evolution_statistics(organisms).await,
            mutation_stats: self.mutation_engine.get_mutation_statistics().await,
            current_generation: generation,
            population_size: organisms.len(),
            convergence_progress,
            performance_metrics: self.performance_metrics.read().await.clone(),
            neural_evolution_active: self.config.read().await.enable_neural_evolution,
        };
        
        // Verify sub-millisecond performance compliance
        let performance_compliant = total_cycle_time.as_millis() < 1;
        if !performance_compliant {
            tracing::warn!("Evolution cycle exceeded 1ms target: {:?}", total_cycle_time);
        }
        
        tracing::info!(
            "Evolution cycle {} completed in {:?}ms - {} organisms, diversity: {:.3}, mutations: {}",
            generation,
            total_cycle_time.as_millis(),
            status.population_size,
            population_diversity,
            mutation_result.mutations_applied
        );
        
        Ok(status)
    }
    
    /// Calculate population genetic diversity
    async fn calculate_population_diversity(
        &self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
    ) -> f64 {
        self.genetic_algorithm.calculate_genetic_diversity(organisms).await
    }
    
    /// Check convergence based on fitness variance and diversity
    async fn check_convergence(
        &self,
        population_fitness: &std::collections::HashMap<Uuid, FitnessScore>,
        population_diversity: f64,
    ) -> f64 {
        let config = self.config.read().await;
        
        if population_fitness.is_empty() {
            return 0.0;
        }
        
        // Calculate fitness statistics
        let fitness_values: Vec<f64> = population_fitness.values()
            .map(|score| score.overall_fitness)
            .collect();
        
        let mean_fitness = fitness_values.iter().sum::<f64>() / fitness_values.len() as f64;
        let fitness_variance = fitness_values.iter()
            .map(|f| (f - mean_fitness).powi(2))
            .sum::<f64>() / fitness_values.len() as f64;
        
        let best_fitness = fitness_values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        // Convergence criteria
        let fitness_convergence = if let Some(threshold) = config.genetic_algorithm.fitness_threshold {
            best_fitness / threshold
        } else {
            mean_fitness
        };
        
        let diversity_convergence = 1.0 - (population_diversity / 0.5).min(1.0); // Inverse of diversity
        let variance_convergence = if fitness_variance < config.convergence_threshold { 1.0 } else { 0.0 };
        
        // Combined convergence score
        let convergence_score = (fitness_convergence * 0.5 + diversity_convergence * 0.3 + variance_convergence * 0.2).min(1.0);
        
        // Mark as converged if score is high enough
        if convergence_score > 0.9 {
            *self.is_converged.lock().unwrap() = true;
        }
        
        convergence_score
    }
    
    /// Update performance metrics based on cycle timing
    async fn update_performance_metrics(
        &self,
        total_time: std::time::Duration,
        fitness_time: std::time::Duration,
        ga_time: std::time::Duration,
        mutation_time: std::time::Duration,
        population_size: usize,
        mutations_applied: u64,
    ) {
        let mut metrics = self.performance_metrics.write().await;
        
        let total_ms = total_time.as_millis() as f64;
        let generation = self.current_generation.load(Ordering::SeqCst);
        
        // Update running averages
        if generation > 0 {
            metrics.average_evolution_time_ms = 
                (metrics.average_evolution_time_ms * (generation - 1) as f64 + total_ms) / generation as f64;
        } else {
            metrics.average_evolution_time_ms = total_ms;
        }
        
        metrics.total_evolution_time_ms += total_ms;
        
        // Calculate throughput metrics
        let total_seconds = total_time.as_secs_f64();
        if total_seconds > 0.0 {
            metrics.mutations_per_second = mutations_applied as f64 / total_seconds;
            metrics.evaluations_per_second = population_size as f64 / total_seconds;
        }
        
        // Memory efficiency (simplified - could be more sophisticated)
        metrics.memory_efficiency_score = if population_size > 0 {
            (1000.0 / population_size as f64).min(1.0)
        } else {
            1.0
        };
        
        // Sub-millisecond compliance
        metrics.sub_millisecond_compliance = total_ms < 1.0;
    }
    
    /// Get comprehensive evolution status
    pub async fn get_status(
        &self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
        market_conditions: &MarketConditions,
    ) -> ComprehensiveEvolutionStatus {
        let population_diversity = self.calculate_population_diversity(organisms).await;
        
        // Get current fitness for convergence calculation
        let population_fitness = self.fitness_evaluator.evaluate_population_fitness(organisms, market_conditions).await
            .unwrap_or_else(|_| std::collections::HashMap::new());
        
        let convergence_progress = self.check_convergence(&population_fitness, population_diversity).await;
        
        ComprehensiveEvolutionStatus {
            genetic_algorithm_stats: self.genetic_algorithm.get_evolution_statistics(organisms).await,
            mutation_stats: self.mutation_engine.get_mutation_statistics().await,
            current_generation: self.current_generation.load(Ordering::SeqCst),
            population_size: organisms.len(),
            convergence_progress,
            performance_metrics: self.performance_metrics.read().await.clone(),
            neural_evolution_active: self.config.read().await.enable_neural_evolution,
        }
    }
    
    /// Check if evolution has converged
    pub fn has_converged(&self) -> bool {
        *self.is_converged.lock().unwrap()
    }
    
    /// Get current generation number
    pub fn get_generation(&self) -> u64 {
        self.current_generation.load(Ordering::SeqCst)
    }
    
    /// Reset evolution engine state
    pub async fn reset(&mut self) {
        self.genetic_algorithm.reset().await;
        self.mutation_engine.reset().await;
        self.fitness_evaluator.clear_all_data().await;
        
        self.current_generation.store(0, Ordering::SeqCst);
        *self.is_converged.lock().unwrap() = false;
        
        let mut metrics = self.performance_metrics.write().await;
        *metrics = EvolutionPerformanceMetrics {
            average_evolution_time_ms: 0.0,
            total_evolution_time_ms: 0.0,
            mutations_per_second: 0.0,
            evaluations_per_second: 0.0,
            memory_efficiency_score: 1.0,
            sub_millisecond_compliance: true,
        };
    }
    
    /// Get configuration
    pub async fn get_config(&self) -> EvolutionEngineConfig {
        self.config.read().await.clone()
    }
    
    /// Update configuration (affects next evolution cycle)
    pub async fn update_config(&self, new_config: EvolutionEngineConfig) {
        let mut config = self.config.write().await;
        *config = new_config;
    }
    
    /// Get detailed performance report
    pub async fn get_performance_report(&self) -> String {
        let metrics = self.performance_metrics.read().await;
        let config = self.config.read().await;
        
        format!(
            "Evolution Engine Performance Report\n\
             =====================================\n\
             Current Generation: {}\n\
             Average Evolution Time: {:.3}ms\n\
             Total Evolution Time: {:.3}ms\n\
             Mutations per Second: {:.1}\n\
             Evaluations per Second: {:.1}\n\
             Memory Efficiency: {:.3}\n\
             Sub-millisecond Compliance: {}\n\
             Target Performance: {:.1}ms\n\
             Convergence Threshold: {:.6}\n\
             Neural Evolution Enabled: {}\n",
            self.current_generation.load(Ordering::SeqCst),
            metrics.average_evolution_time_ms,
            metrics.total_evolution_time_ms,
            metrics.mutations_per_second,
            metrics.evaluations_per_second,
            metrics.memory_efficiency_score,
            metrics.sub_millisecond_compliance,
            config.performance_target_ms,
            config.convergence_threshold,
            config.enable_neural_evolution
        )
    }
}

/// Factory function for creating evolution engine with sensible defaults
pub fn create_evolution_engine() -> EvolutionEngine {
    EvolutionEngine::new(EvolutionEngineConfig::default())
}

/// Factory function for creating high-performance evolution engine
pub fn create_high_performance_evolution_engine() -> EvolutionEngine {
    let config = EvolutionEngineConfig {
        genetic_algorithm: GeneticAlgorithmConfig {
            population_size: 50, // Smaller for speed
            parallel_execution: true,
            adaptive_parameters: true,
            ..Default::default()
        },
        mutation_engine: MutationEngineConfig {
            adaptive_mutation: true,
            gaussian_mutation: true,
            targeted_mutation: true,
            ..Default::default()
        },
        fitness_evaluation: FitnessEvaluationConfig {
            real_time_evaluation: true,
            ..Default::default()
        },
        performance_target_ms: 0.5, // Ultra-fast target
        enable_neural_evolution: true,
        ..Default::default()
    };
    
    EvolutionEngine::new(config)
}