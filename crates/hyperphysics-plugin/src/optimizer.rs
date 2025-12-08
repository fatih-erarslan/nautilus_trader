//! Single-strategy optimizer
//!
//! Provides a simple interface for single-algorithm optimization.

use std::collections::HashMap;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use nalgebra::DVector;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{HyperPhysicsError, Result, Strategy, OptimizationConfig, ConvergenceMetrics};

/// Optimization result
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct OptimizationResult {
    /// Best solution found
    pub solution: Vec<f64>,
    /// Best fitness value
    pub fitness: f64,
    /// Convergence history
    pub history: Vec<f64>,
    /// Strategy used
    pub strategy: Strategy,
    /// Metrics
    pub metrics: ConvergenceMetrics,
}

/// Agent in the optimizer population
#[derive(Debug, Clone)]
struct Agent {
    position: DVector<f64>,
    velocity: DVector<f64>,
    fitness: f64,
    personal_best: DVector<f64>,
    personal_best_fitness: f64,
    metadata: HashMap<String, f64>,
}

impl Agent {
    fn new(dim: usize) -> Self {
        Self {
            position: DVector::zeros(dim),
            velocity: DVector::zeros(dim),
            fitness: f64::INFINITY,
            personal_best: DVector::zeros(dim),
            personal_best_fitness: f64::INFINITY,
            metadata: HashMap::new(),
        }
    }
    
    fn random(bounds: &[(f64, f64)], rng: &mut impl Rng) -> Self {
        let dim = bounds.len();
        let mut agent = Self::new(dim);
        
        for (i, &(min, max)) in bounds.iter().enumerate() {
            agent.position[i] = rng.gen_range(min..max);
            agent.velocity[i] = (max - min) * 0.1 * (rng.gen::<f64>() - 0.5);
        }
        agent.personal_best = agent.position.clone();
        agent
    }
}

/// Main optimizer
pub struct Optimizer {
    config: OptimizationConfig,
    agents: Vec<Agent>,
    global_best: DVector<f64>,
    global_best_fitness: f64,
    iteration: usize,
    evaluations: usize,
    history: Vec<f64>,
    rng: ChaCha8Rng,
}

impl Optimizer {
    /// Create a new optimizer with configuration
    pub fn new(config: OptimizationConfig) -> Result<Self> {
        if config.dimensions == 0 {
            return Err(HyperPhysicsError::Config("Dimensions must be > 0".into()));
        }
        
        let mut rng = ChaCha8Rng::from_entropy();
        let agents: Vec<Agent> = (0..config.population_size)
            .map(|_| Agent::random(&config.bounds, &mut rng))
            .collect();
        
        let dim = config.dimensions;
        
        Ok(Self {
            config,
            agents,
            global_best: DVector::zeros(dim),
            global_best_fitness: f64::INFINITY,
            iteration: 0,
            evaluations: 0,
            history: Vec::new(),
            rng,
        })
    }
    
    /// Create optimizer with default config for given dimensions
    pub fn with_dimensions(dimensions: usize) -> Result<Self> {
        let mut config = OptimizationConfig::default();
        config.dimensions = dimensions;
        config.bounds = vec![(-100.0, 100.0); dimensions];
        Self::new(config)
    }
    
    /// Run optimization with the given objective function
    pub fn optimize<F>(&mut self, objective: F) -> Result<OptimizationResult>
    where
        F: Fn(&[f64]) -> f64,
    {
        let start = std::time::Instant::now();
        let maximize = self.config.maximize;
        let max_iterations = self.config.max_iterations;
        let strategy = self.config.strategy;
        
        // Adjust for maximization
        let eval = |x: &[f64]| {
            if maximize {
                -objective(x)
            } else {
                objective(x)
            }
        };
        
        // Initial evaluation
        for agent in &mut self.agents {
            agent.fitness = eval(agent.position.as_slice());
            self.evaluations += 1;
            
            if agent.fitness < agent.personal_best_fitness {
                agent.personal_best = agent.position.clone();
                agent.personal_best_fitness = agent.fitness;
            }
            
            if agent.fitness < self.global_best_fitness {
                self.global_best = agent.position.clone();
                self.global_best_fitness = agent.fitness;
            }
        }
        
        // Main loop
        while self.iteration < max_iterations {
            // Strategy-specific update
            match strategy {
                Strategy::ParticleSwarm => self.step_pso(),
                Strategy::GreyWolf => self.step_grey_wolf(),
                Strategy::Whale => self.step_whale(),
                Strategy::Cuckoo => self.step_cuckoo(),
                Strategy::DifferentialEvolution => self.step_de(),
                Strategy::Adaptive => self.step_adaptive(),
                _ => self.step_pso(), // Default to PSO
            }
            
            // Evaluate
            for agent in &mut self.agents {
                agent.fitness = eval(agent.position.as_slice());
                self.evaluations += 1;
                
                if agent.fitness < agent.personal_best_fitness {
                    agent.personal_best = agent.position.clone();
                    agent.personal_best_fitness = agent.fitness;
                }
                
                if agent.fitness < self.global_best_fitness {
                    self.global_best = agent.position.clone();
                    self.global_best_fitness = agent.fitness;
                }
            }
            
            self.history.push(self.global_best_fitness);
            self.iteration += 1;
            
            // Early stopping
            if self.check_convergence() {
                break;
            }
        }
        
        let final_fitness = if self.config.maximize {
            -self.global_best_fitness
        } else {
            self.global_best_fitness
        };
        
        let convergence_rate = if self.history.len() > 1 {
            let first = self.history.first().unwrap();
            let last = self.history.last().unwrap();
            (first - last) / self.history.len() as f64
        } else {
            0.0
        };
        
        Ok(OptimizationResult {
            solution: self.global_best.as_slice().to_vec(),
            fitness: final_fitness,
            history: self.history.clone().into_iter()
                .map(|f| if self.config.maximize { -f } else { f })
                .collect(),
            strategy: self.config.strategy,
            metrics: ConvergenceMetrics {
                final_fitness,
                iterations: self.iteration,
                evaluations: self.evaluations,
                convergence_rate,
                diversity: self.compute_diversity(),
                time_ms: start.elapsed().as_millis() as u64,
            },
        })
    }
    
    /// PSO step
    fn step_pso(&mut self) {
        let w = self.config.params.get("inertia").copied().unwrap_or(0.7);
        let c1 = self.config.params.get("cognitive").copied().unwrap_or(1.5);
        let c2 = self.config.params.get("social").copied().unwrap_or(1.5);
        
        let global_best = self.global_best.clone();
        
        for agent in &mut self.agents {
            for i in 0..agent.position.len() {
                let r1 = self.rng.gen::<f64>();
                let r2 = self.rng.gen::<f64>();
                
                agent.velocity[i] = w * agent.velocity[i]
                    + c1 * r1 * (agent.personal_best[i] - agent.position[i])
                    + c2 * r2 * (global_best[i] - agent.position[i]);
                
                agent.position[i] += agent.velocity[i];
                
                let (min, max) = self.config.bounds[i];
                agent.position[i] = agent.position[i].clamp(min, max);
            }
        }
    }
    
    /// Grey Wolf step
    fn step_grey_wolf(&mut self) {
        let a = 2.0 - 2.0 * (self.iteration as f64 / self.config.max_iterations as f64);
        
        // Sort to get alpha, beta, delta
        let mut sorted: Vec<_> = self.agents.iter().enumerate().collect();
        sorted.sort_by(|a, b| a.1.fitness.partial_cmp(&b.1.fitness).unwrap());
        
        let alpha = sorted[0].1.position.clone();
        let beta = sorted[1].1.position.clone();
        let delta = sorted[2].1.position.clone();
        
        for agent in &mut self.agents {
            for i in 0..agent.position.len() {
                let r1 = self.rng.gen::<f64>();
                let r2 = self.rng.gen::<f64>();
                let a1 = 2.0 * a * r1 - a;
                let c1 = 2.0 * r2;
                let d_alpha = (c1 * alpha[i] - agent.position[i]).abs();
                let x1 = alpha[i] - a1 * d_alpha;
                
                let r1 = self.rng.gen::<f64>();
                let r2 = self.rng.gen::<f64>();
                let a2 = 2.0 * a * r1 - a;
                let c2 = 2.0 * r2;
                let d_beta = (c2 * beta[i] - agent.position[i]).abs();
                let x2 = beta[i] - a2 * d_beta;
                
                let r1 = self.rng.gen::<f64>();
                let r2 = self.rng.gen::<f64>();
                let a3 = 2.0 * a * r1 - a;
                let c3 = 2.0 * r2;
                let d_delta = (c3 * delta[i] - agent.position[i]).abs();
                let x3 = delta[i] - a3 * d_delta;
                
                agent.position[i] = (x1 + x2 + x3) / 3.0;
                
                let (min, max) = self.config.bounds[i];
                agent.position[i] = agent.position[i].clamp(min, max);
            }
        }
    }
    
    /// Whale step
    fn step_whale(&mut self) {
        let a = 2.0 - 2.0 * (self.iteration as f64 / self.config.max_iterations as f64);
        let b = 1.0;
        let global_best = self.global_best.clone();
        
        for agent in &mut self.agents {
            let p = self.rng.gen::<f64>();
            let l = self.rng.gen_range(-1.0..1.0);
            
            for i in 0..agent.position.len() {
                if p < 0.5 {
                    let r = self.rng.gen::<f64>();
                    let a_coeff = 2.0 * a * r - a;
                    let c = 2.0 * self.rng.gen::<f64>();
                    let d = (c * global_best[i] - agent.position[i]).abs();
                    agent.position[i] = global_best[i] - a_coeff * d;
                } else {
                    let d = (global_best[i] - agent.position[i]).abs();
                    agent.position[i] = d * (b * l * 2.0 * std::f64::consts::PI).cos().exp() 
                        * (b * l * 2.0 * std::f64::consts::PI).cos() + global_best[i];
                }
                
                let (min, max) = self.config.bounds[i];
                agent.position[i] = agent.position[i].clamp(min, max);
            }
        }
    }
    
    /// Cuckoo Search step
    fn step_cuckoo(&mut self) {
        let pa = self.config.params.get("abandon_prob").copied().unwrap_or(0.25);
        let global_best = self.global_best.clone();
        let bounds = self.config.bounds.clone();
        let n = self.agents.len();
        
        // Pre-compute Lévy flight steps
        let steps: Vec<f64> = (0..n).map(|_| self.levy_flight()).collect();
        
        // Lévy flights
        for (idx, agent) in self.agents.iter_mut().enumerate() {
            let step = steps[idx];
            for i in 0..agent.position.len() {
                agent.position[i] += step * (agent.position[i] - global_best[i]);
                let (min, max) = bounds[i];
                agent.position[i] = agent.position[i].clamp(min, max);
            }
        }
        
        // Abandon worst nests
        let mut sorted: Vec<_> = (0..self.agents.len()).collect();
        sorted.sort_by(|&a, &b| {
            self.agents[b].fitness.partial_cmp(&self.agents[a].fitness).unwrap()
        });
        
        let abandon_count = (pa * self.agents.len() as f64) as usize;
        for &idx in sorted.iter().take(abandon_count) {
            self.agents[idx] = Agent::random(&self.config.bounds, &mut self.rng);
        }
    }
    
    /// Lévy flight
    fn levy_flight(&mut self) -> f64 {
        let beta = 1.5;
        let sigma = ((1.0f64 + beta).exp() * beta.sin() * std::f64::consts::PI / 2.0
            / (((1.0 + beta) / 2.0).exp() * beta * 2.0f64.powf((beta - 1.0) / 2.0)))
            .powf(1.0 / beta);
        
        let u: f64 = Normal::new(0.0, sigma).unwrap().sample(&mut self.rng);
        let v: f64 = Normal::new(0.0, 1.0).unwrap().sample(&mut self.rng);
        
        u / v.abs().powf(1.0 / beta)
    }
    
    /// Differential Evolution step
    fn step_de(&mut self) {
        let f = self.config.params.get("mutation_factor").copied().unwrap_or(0.8);
        let cr = self.config.params.get("crossover_rate").copied().unwrap_or(0.9);
        
        let n = self.agents.len();
        let dim = self.config.dimensions;
        let bounds = self.config.bounds.clone();
        
        for i in 0..n {
            // Select 3 random distinct agents
            let mut indices: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            let a = indices.remove(self.rng.gen_range(0..indices.len()));
            let b = indices.remove(self.rng.gen_range(0..indices.len()));
            let c = indices.remove(self.rng.gen_range(0..indices.len()));
            
            // Mutation
            let mut mutant = DVector::zeros(dim);
            for j in 0..dim {
                mutant[j] = self.agents[a].position[j] 
                    + f * (self.agents[b].position[j] - self.agents[c].position[j]);
            }
            
            // Crossover
            let j_rand = self.rng.gen_range(0..dim);
            for j in 0..dim {
                if self.rng.gen::<f64>() > cr && j != j_rand {
                    mutant[j] = self.agents[i].position[j];
                }
                let (min, max) = bounds[j];
                mutant[j] = mutant[j].clamp(min, max);
            }
            
            self.agents[i].velocity = mutant.clone() - &self.agents[i].position;
            self.agents[i].position = mutant;
        }
    }
    
    /// Adaptive step
    fn step_adaptive(&mut self) {
        // Select strategy based on progress
        let progress = self.iteration as f64 / self.config.max_iterations as f64;
        let improvement = if self.history.len() >= 10 {
            let recent = &self.history[self.history.len() - 10..];
            (recent[0] - recent[9]) / (recent[0].abs() + 1e-10)
        } else {
            0.1
        };
        
        if improvement < 0.001 && progress > 0.3 {
            self.step_cuckoo(); // Stuck - explore
        } else if progress < 0.3 {
            self.step_whale(); // Early - explore
        } else if progress < 0.7 {
            self.step_grey_wolf(); // Middle - balanced
        } else {
            self.step_pso(); // Late - exploit
        }
    }
    
    /// Check convergence
    fn check_convergence(&self) -> bool {
        if self.history.len() < 20 {
            return false;
        }
        
        let recent = &self.history[self.history.len() - 20..];
        let improvement = (recent[0] - recent[19]).abs() / (recent[0].abs() + 1e-10);
        
        improvement < self.config.tolerance
    }
    
    /// Compute population diversity
    fn compute_diversity(&self) -> f64 {
        if self.agents.is_empty() {
            return 0.0;
        }
        
        let n = self.agents.len() as f64;
        let dim = self.config.dimensions;
        
        let mut centroid = DVector::zeros(dim);
        for agent in &self.agents {
            centroid += &agent.position;
        }
        centroid /= n;
        
        let mut total_dist = 0.0;
        for agent in &self.agents {
            let diff = &agent.position - &centroid;
            total_dist += diff.norm();
        }
        
        total_dist / n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_optimizer_sphere() {
        let mut config = OptimizationConfig::default();
        config.dimensions = 5;
        config.bounds = vec![(-5.0, 5.0); 5];
        config.max_iterations = 200;
        config.strategy = Strategy::GreyWolf;
        
        let mut optimizer = Optimizer::new(config).unwrap();
        let result = optimizer.optimize(|x| x.iter().map(|xi| xi * xi).sum()).unwrap();
        
        assert!(result.fitness < 0.1);
    }
    
    #[test]
    fn test_optimizer_maximize() {
        let mut config = OptimizationConfig::default();
        config.dimensions = 2;
        config.bounds = vec![(-5.0, 5.0); 2];
        config.max_iterations = 100;
        config.maximize = true;
        
        let mut optimizer = Optimizer::new(config).unwrap();
        // Maximize -x² (peak at 0)
        let result = optimizer.optimize(|x| -x.iter().map(|xi| xi * xi).sum::<f64>()).unwrap();
        
        assert!(result.fitness > -0.1);
    }
}
