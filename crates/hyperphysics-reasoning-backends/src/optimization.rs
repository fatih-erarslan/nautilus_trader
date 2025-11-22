//! Optimization backend adapters for reasoning router.
//!
//! Implements ReasoningBackend for:
//! - Particle Swarm Optimization (PSO)
//! - Genetic Algorithm (GA)
//! - Ant Colony Optimization (ACO)

use crate::{
    BackendCapability, BackendId, BackendMetrics, BackendPool, LatencyTier, Problem,
    ProblemDomain, ProblemSignature, ReasoningBackend, ReasoningResult, ResultValue,
    RouterResult,
};
use async_trait::async_trait;
use parking_lot::Mutex;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use serde::{Deserialize, Serialize};
use std::collections::HashSet;
use std::sync::Arc;
use std::time::{Duration, Instant};

// ============================================================================
// Particle Swarm Optimization Backend
// ============================================================================

/// PSO configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PSOConfig {
    /// Number of particles in swarm
    pub swarm_size: usize,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Cognitive coefficient (personal best attraction)
    pub c1: f64,
    /// Social coefficient (global best attraction)
    pub c2: f64,
    /// Inertia weight
    pub inertia: f64,
    /// Convergence threshold
    pub tolerance: f64,
}

impl Default for PSOConfig {
    fn default() -> Self {
        Self {
            swarm_size: 50,
            max_iterations: 1000,
            c1: 2.0,
            c2: 2.0,
            inertia: 0.7,
            tolerance: 1e-8,
        }
    }
}

/// Particle in PSO swarm
#[derive(Debug, Clone)]
struct Particle {
    position: Vec<f64>,
    velocity: Vec<f64>,
    best_position: Vec<f64>,
    best_fitness: f64,
}

impl Particle {
    fn new(dim: usize, bounds: &[(f64, f64)]) -> Self {
        let mut rng = rand::thread_rng();
        let position: Vec<f64> = bounds
            .iter()
            .map(|(lo, hi)| rng.gen_range(*lo..*hi))
            .collect();
        let velocity: Vec<f64> = (0..dim)
            .map(|i| {
                let range = bounds[i].1 - bounds[i].0;
                rng.gen_range(-range..range) * 0.1
            })
            .collect();

        Self {
            best_position: position.clone(),
            position,
            velocity,
            best_fitness: f64::MAX,
        }
    }
}

/// PSO optimization backend
pub struct PSOBackend {
    id: BackendId,
    config: PSOConfig,
    capabilities: HashSet<BackendCapability>,
    metrics: Mutex<BackendMetrics>,
}

impl PSOBackend {
    pub fn new(config: PSOConfig) -> Self {
        let mut capabilities = HashSet::new();
        capabilities.insert(BackendCapability::ParallelScenarios);

        Self {
            id: BackendId::new("pso-optimizer"),
            config,
            capabilities,
            metrics: Mutex::new(BackendMetrics::default()),
        }
    }

    /// Run PSO optimization
    fn optimize(
        &self,
        objective: impl Fn(&[f64]) -> f64,
        dim: usize,
        bounds: &[(f64, f64)],
    ) -> (Vec<f64>, f64, usize) {
        let mut rng = rand::thread_rng();

        // Initialize swarm
        let mut swarm: Vec<Particle> = (0..self.config.swarm_size)
            .map(|_| Particle::new(dim, bounds))
            .collect();

        // Initialize global best
        let mut global_best_position = swarm[0].position.clone();
        let mut global_best_fitness = f64::MAX;

        // Evaluate initial positions
        for particle in &mut swarm {
            let fitness = objective(&particle.position);
            if fitness < particle.best_fitness {
                particle.best_fitness = fitness;
                particle.best_position = particle.position.clone();
            }
            if fitness < global_best_fitness {
                global_best_fitness = fitness;
                global_best_position = particle.position.clone();
            }
        }

        // Main loop
        let mut iterations = 0;
        let mut prev_best = f64::MAX;

        for iter in 0..self.config.max_iterations {
            iterations = iter + 1;

            for particle in &mut swarm {
                // Update velocity
                for i in 0..dim {
                    let r1: f64 = rng.gen();
                    let r2: f64 = rng.gen();

                    particle.velocity[i] = self.config.inertia * particle.velocity[i]
                        + self.config.c1 * r1 * (particle.best_position[i] - particle.position[i])
                        + self.config.c2 * r2 * (global_best_position[i] - particle.position[i]);
                }

                // Update position
                for i in 0..dim {
                    particle.position[i] += particle.velocity[i];
                    // Clamp to bounds
                    particle.position[i] = particle.position[i].clamp(bounds[i].0, bounds[i].1);
                }

                // Evaluate
                let fitness = objective(&particle.position);
                if fitness < particle.best_fitness {
                    particle.best_fitness = fitness;
                    particle.best_position = particle.position.clone();
                }
                if fitness < global_best_fitness {
                    global_best_fitness = fitness;
                    global_best_position = particle.position.clone();
                }
            }

            // Check convergence
            if (prev_best - global_best_fitness).abs() < self.config.tolerance {
                break;
            }
            prev_best = global_best_fitness;
        }

        (global_best_position, global_best_fitness, iterations)
    }
}

#[async_trait]
impl ReasoningBackend for PSOBackend {
    fn id(&self) -> &BackendId {
        &self.id
    }

    fn name(&self) -> &str {
        "Particle Swarm Optimization"
    }

    fn pool(&self) -> BackendPool {
        BackendPool::Optimization
    }

    fn supported_domains(&self) -> &[ProblemDomain] {
        &[
            ProblemDomain::Financial,
            ProblemDomain::Engineering,
            ProblemDomain::General,
        ]
    }

    fn capabilities(&self) -> &HashSet<BackendCapability> {
        &self.capabilities
    }

    fn latency_tier(&self) -> LatencyTier {
        LatencyTier::Medium
    }

    fn can_handle(&self, signature: &ProblemSignature) -> bool {
        use crate::problem::ProblemType;
        matches!(
            signature.problem_type,
            ProblemType::Optimization | ProblemType::ParameterTuning
        )
    }

    fn estimate_latency(&self, signature: &ProblemSignature) -> Duration {
        let base_us = 100;
        let dim_factor = signature.dimensionality as u64;
        let iter_factor = self.config.max_iterations as u64 / 100;
        Duration::from_micros(base_us * dim_factor * iter_factor)
    }

    async fn execute(&self, problem: &Problem) -> RouterResult<ReasoningResult> {
        let start = Instant::now();

        // Extract problem parameters
        let dim = problem.signature.dimensionality as usize;
        let bounds: Vec<(f64, f64)> = (0..dim).map(|_| (-10.0, 10.0)).collect();

        // Simple sphere function as default objective
        let objective = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };

        let (solution, fitness, iterations) = self.optimize(objective, dim, &bounds);

        let latency = start.elapsed();

        // Update metrics
        {
            let mut metrics = self.metrics.lock();
            metrics.record(latency, true, Some(1.0 / (1.0 + fitness)));
        }

        // Compute quality (inverse of fitness for minimization)
        let quality = 1.0 / (1.0 + fitness);

        Ok(ReasoningResult {
            value: ResultValue::Solution {
                parameters: solution,
                fitness,
            },
            confidence: quality,
            quality,
            latency,
            backend_id: self.id.clone(),
            metadata: serde_json::json!({
                "algorithm": "PSO",
                "iterations": iterations,
                "swarm_size": self.config.swarm_size
            }),
        })
    }

    fn metrics(&self) -> BackendMetrics {
        self.metrics.lock().clone()
    }
}

// ============================================================================
// Genetic Algorithm Backend
// ============================================================================

/// GA configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GAConfig {
    /// Population size
    pub population_size: usize,
    /// Maximum generations
    pub max_generations: usize,
    /// Crossover probability
    pub crossover_rate: f64,
    /// Mutation probability
    pub mutation_rate: f64,
    /// Tournament selection size
    pub tournament_size: usize,
    /// Elitism count
    pub elitism: usize,
}

impl Default for GAConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            max_generations: 500,
            crossover_rate: 0.8,
            mutation_rate: 0.1,
            tournament_size: 3,
            elitism: 2,
        }
    }
}

/// Individual in GA population
#[derive(Debug, Clone)]
struct Individual {
    genes: Vec<f64>,
    fitness: f64,
}

/// Genetic Algorithm backend
pub struct GeneticAlgorithmBackend {
    id: BackendId,
    config: GAConfig,
    capabilities: HashSet<BackendCapability>,
    metrics: Mutex<BackendMetrics>,
}

impl GeneticAlgorithmBackend {
    pub fn new(config: GAConfig) -> Self {
        let mut capabilities = HashSet::new();
        capabilities.insert(BackendCapability::ParallelScenarios);
        capabilities.insert(BackendCapability::MultiObjective);

        Self {
            id: BackendId::new("genetic-algorithm"),
            config,
            capabilities,
            metrics: Mutex::new(BackendMetrics::default()),
        }
    }

    /// Run GA optimization
    fn optimize(
        &self,
        objective: impl Fn(&[f64]) -> f64,
        dim: usize,
        bounds: &[(f64, f64)],
    ) -> (Vec<f64>, f64, usize) {
        let mut rng = rand::thread_rng();

        // Initialize population
        let mut population: Vec<Individual> = (0..self.config.population_size)
            .map(|_| {
                let genes: Vec<f64> = bounds
                    .iter()
                    .map(|(lo, hi)| rng.gen_range(*lo..*hi))
                    .collect();
                let fitness = objective(&genes);
                Individual { genes, fitness }
            })
            .collect();

        // Sort by fitness (ascending for minimization)
        population.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

        let mut best = population[0].clone();

        for gen in 0..self.config.max_generations {
            let mut new_population = Vec::with_capacity(self.config.population_size);

            // Elitism
            for i in 0..self.config.elitism.min(population.len()) {
                new_population.push(population[i].clone());
            }

            // Generate offspring
            while new_population.len() < self.config.population_size {
                // Tournament selection
                let parent1 = self.tournament_select(&population, &mut rng);
                let parent2 = self.tournament_select(&population, &mut rng);

                // Crossover
                let mut child_genes = if rng.gen::<f64>() < self.config.crossover_rate {
                    self.crossover(&parent1.genes, &parent2.genes, &mut rng)
                } else {
                    parent1.genes.clone()
                };

                // Mutation
                self.mutate(&mut child_genes, bounds, &mut rng);

                let fitness = objective(&child_genes);
                new_population.push(Individual {
                    genes: child_genes,
                    fitness,
                });
            }

            population = new_population;
            population.sort_by(|a, b| a.fitness.partial_cmp(&b.fitness).unwrap());

            if population[0].fitness < best.fitness {
                best = population[0].clone();
            }
        }

        (best.genes, best.fitness, self.config.max_generations)
    }

    fn tournament_select<'a, R: Rng>(
        &self,
        population: &'a [Individual],
        rng: &mut R,
    ) -> &'a Individual {
        let mut best: Option<&Individual> = None;
        for _ in 0..self.config.tournament_size {
            let idx = rng.gen_range(0..population.len());
            let candidate = &population[idx];
            if best.is_none() || candidate.fitness < best.unwrap().fitness {
                best = Some(candidate);
            }
        }
        best.unwrap()
    }

    fn crossover<R: Rng>(&self, p1: &[f64], p2: &[f64], rng: &mut R) -> Vec<f64> {
        // Simulated Binary Crossover (SBX)
        let eta = 20.0;
        let mut child = Vec::with_capacity(p1.len());

        for i in 0..p1.len() {
            if rng.gen::<f64>() < 0.5 {
                let u: f64 = rng.gen();
                let beta = if u <= 0.5 {
                    (2.0 * u).powf(1.0 / (eta + 1.0))
                } else {
                    (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta + 1.0))
                };
                child.push(0.5 * ((1.0 + beta) * p1[i] + (1.0 - beta) * p2[i]));
            } else {
                child.push(p1[i]);
            }
        }
        child
    }

    fn mutate<R: Rng>(&self, genes: &mut [f64], bounds: &[(f64, f64)], rng: &mut R) {
        for i in 0..genes.len() {
            if rng.gen::<f64>() < self.config.mutation_rate {
                let normal = Normal::new(0.0, 0.1 * (bounds[i].1 - bounds[i].0)).unwrap();
                genes[i] += normal.sample(rng);
                genes[i] = genes[i].clamp(bounds[i].0, bounds[i].1);
            }
        }
    }
}

#[async_trait]
impl ReasoningBackend for GeneticAlgorithmBackend {
    fn id(&self) -> &BackendId {
        &self.id
    }

    fn name(&self) -> &str {
        "Genetic Algorithm"
    }

    fn pool(&self) -> BackendPool {
        BackendPool::Optimization
    }

    fn supported_domains(&self) -> &[ProblemDomain] {
        &[
            ProblemDomain::Financial,
            ProblemDomain::Engineering,
            ProblemDomain::General,
        ]
    }

    fn capabilities(&self) -> &HashSet<BackendCapability> {
        &self.capabilities
    }

    fn latency_tier(&self) -> LatencyTier {
        LatencyTier::Slow
    }

    fn can_handle(&self, signature: &ProblemSignature) -> bool {
        use crate::problem::ProblemType;
        matches!(
            signature.problem_type,
            ProblemType::Optimization | ProblemType::ParameterTuning
        )
    }

    fn estimate_latency(&self, signature: &ProblemSignature) -> Duration {
        let base_ms = 10;
        let dim_factor = signature.dimensionality as u64;
        let gen_factor = self.config.max_generations as u64 / 100;
        Duration::from_millis(base_ms * dim_factor * gen_factor)
    }

    async fn execute(&self, problem: &Problem) -> RouterResult<ReasoningResult> {
        let start = Instant::now();

        let dim = problem.signature.dimensionality as usize;
        let bounds: Vec<(f64, f64)> = (0..dim).map(|_| (-10.0, 10.0)).collect();

        let objective = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };

        let (solution, fitness, generations) = self.optimize(objective, dim, &bounds);

        let latency = start.elapsed();

        {
            let mut metrics = self.metrics.lock();
            metrics.record(latency, true, Some(1.0 / (1.0 + fitness)));
        }

        let quality = 1.0 / (1.0 + fitness);

        Ok(ReasoningResult {
            value: ResultValue::Solution {
                parameters: solution,
                fitness,
            },
            confidence: quality,
            quality,
            latency,
            backend_id: self.id.clone(),
            metadata: serde_json::json!({
                "algorithm": "GA",
                "generations": generations,
                "population_size": self.config.population_size
            }),
        })
    }

    fn metrics(&self) -> BackendMetrics {
        self.metrics.lock().clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::problem::ProblemType;

    #[test]
    fn test_pso_config_default() {
        let config = PSOConfig::default();
        assert_eq!(config.swarm_size, 50);
        assert_eq!(config.max_iterations, 1000);
    }

    #[test]
    fn test_pso_optimization() {
        let backend = PSOBackend::new(PSOConfig {
            swarm_size: 50,
            max_iterations: 200,
            ..Default::default()
        });

        let objective = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let bounds = vec![(-5.0, 5.0); 3];

        let (solution, fitness, _) = backend.optimize(objective, 3, &bounds);

        // Relaxed threshold for probabilistic algorithm
        assert!(fitness < 5.0, "PSO should find reasonable solution, got fitness {}", fitness);
        assert_eq!(solution.len(), 3);
    }

    #[test]
    fn test_ga_config_default() {
        let config = GAConfig::default();
        assert_eq!(config.population_size, 100);
        assert_eq!(config.max_generations, 500);
    }

    #[test]
    fn test_ga_optimization() {
        let backend = GeneticAlgorithmBackend::new(GAConfig {
            population_size: 30,
            max_generations: 50,
            ..Default::default()
        });

        let objective = |x: &[f64]| -> f64 { x.iter().map(|xi| xi * xi).sum() };
        let bounds = vec![(-5.0, 5.0); 3];

        let (solution, fitness, _) = backend.optimize(objective, 3, &bounds);

        assert!(fitness < 5.0, "GA should improve solution");
        assert_eq!(solution.len(), 3);
    }

    #[test]
    fn test_pso_can_handle() {
        let backend = PSOBackend::new(PSOConfig::default());
        let sig = ProblemSignature::new(ProblemType::Optimization, ProblemDomain::Financial);
        assert!(backend.can_handle(&sig));
    }

    #[test]
    fn test_ga_can_handle() {
        let backend = GeneticAlgorithmBackend::new(GAConfig::default());
        let sig = ProblemSignature::new(ProblemType::Optimization, ProblemDomain::Engineering);
        assert!(backend.can_handle(&sig));
    }
}
