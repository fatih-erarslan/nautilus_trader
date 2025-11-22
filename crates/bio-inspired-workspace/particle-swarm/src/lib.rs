//! Particle Swarm Optimization (PSO) Algorithm Implementation
//! 
//! This crate provides a basic implementation of the Particle Swarm Optimization algorithm,
//! inspired by the flocking behavior of birds and schooling behavior of fish.

use async_trait::async_trait;
use rand::Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Particle Swarm Optimization algorithm parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleSwarmParameters {
    pub population_size: usize,
    pub max_iterations: u32,
    pub w: f64,  // inertia weight
    pub c1: f64, // cognitive acceleration coefficient
    pub c2: f64, // social acceleration coefficient
    pub max_velocity: f64,
}

impl Default for ParticleSwarmParameters {
    fn default() -> Self {
        Self {
            population_size: 50,
            max_iterations: 1000,
            w: 0.7,
            c1: 1.5,
            c2: 1.5,
            max_velocity: 10.0,
        }
    }
}

/// Individual particle in the swarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Particle {
    pub position: Vec<f64>,
    pub velocity: Vec<f64>,
    pub fitness: f64,
    pub best_position: Vec<f64>,
    pub best_fitness: f64,
}

impl Particle {
    pub fn new(dimension: usize) -> Self {
        let mut rng = rand::thread_rng();
        let position: Vec<f64> = (0..dimension)
            .map(|_| rng.gen_range(0.0..1.0))
            .collect();
        let velocity: Vec<f64> = (0..dimension)
            .map(|_| rng.gen_range(-1.0..1.0))
            .collect();
        
        Self {
            position: position.clone(),
            velocity,
            fitness: 0.0,
            best_position: position,
            best_fitness: 0.0,
        }
    }
    
    pub fn update_velocity(&mut self, global_best: &[f64], parameters: &ParticleSwarmParameters) {
        let mut rng = rand::thread_rng();
        
        for i in 0..self.velocity.len() {
            let r1 = rng.gen::<f64>();
            let r2 = rng.gen::<f64>();
            
            self.velocity[i] = parameters.w * self.velocity[i]
                + parameters.c1 * r1 * (self.best_position[i] - self.position[i])
                + parameters.c2 * r2 * (global_best[i] - self.position[i]);
            
            // Clamp velocity
            self.velocity[i] = self.velocity[i].clamp(-parameters.max_velocity, parameters.max_velocity);
        }
    }
    
    pub fn update_position(&mut self, bounds: &[(f64, f64)]) {
        for i in 0..self.position.len() {
            self.position[i] += self.velocity[i];
            
            // Clamp position to bounds
            if i < bounds.len() {
                self.position[i] = self.position[i].clamp(bounds[i].0, bounds[i].1);
            }
        }
    }
}

/// Particle Swarm Optimization algorithm
#[derive(Debug)]
pub struct ParticleSwarmOptimizer {
    pub parameters: ParticleSwarmParameters,
    pub particles: Vec<Particle>,
    pub global_best_position: Vec<f64>,
    pub global_best_fitness: f64,
    pub iteration: u32,
}

impl ParticleSwarmOptimizer {
    pub fn new(parameters: ParticleSwarmParameters, dimension: usize) -> Self {
        let particles = (0..parameters.population_size)
            .map(|_| Particle::new(dimension))
            .collect();
        
        Self {
            parameters,
            particles,
            global_best_position: vec![0.0; dimension],
            global_best_fitness: f64::INFINITY,
            iteration: 0,
        }
    }
    
    pub fn initialize_population(&mut self, bounds: &[(f64, f64)]) {
        let mut rng = rand::thread_rng();
        
        for particle in &mut self.particles {
            particle.position = bounds.iter()
                .map(|(min, max)| rng.gen_range(*min..*max))
                .collect();
            particle.best_position = particle.position.clone();
            particle.fitness = 0.0;
            particle.best_fitness = f64::INFINITY;
        }
    }
    
    pub fn update_swarm(&mut self) {
        // Update global best
        for particle in &self.particles {
            if particle.fitness < self.global_best_fitness {
                self.global_best_fitness = particle.fitness;
                self.global_best_position = particle.position.clone();
            }
        }
        
        // Update particles
        for particle in &mut self.particles {
            particle.update_velocity(&self.global_best_position, &self.parameters);
        }
    }
    
    pub fn get_best_solution(&self) -> Option<&Vec<f64>> {
        if self.global_best_fitness < f64::INFINITY {
            Some(&self.global_best_position)
        } else {
            None
        }
    }
    
    pub fn get_population_diversity(&self) -> f64 {
        if self.particles.is_empty() {
            return 0.0;
        }
        
        let mut diversity = 0.0;
        let n = self.particles.len();
        
        for i in 0..n {
            for j in i + 1..n {
                let distance: f64 = self.particles[i].position.iter()
                    .zip(&self.particles[j].position)
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                diversity += distance;
            }
        }
        
        diversity / (n * (n - 1) / 2) as f64
    }
    
    pub fn is_converged(&self) -> bool {
        self.get_population_diversity() < 1e-6
    }
}

/// Objective function trait for particle swarm optimization
#[async_trait]
pub trait ParticleSwarmObjective: Send + Sync {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, ParticleSwarmError>;
    fn get_bounds(&self) -> Vec<(f64, f64)>;
    fn get_dimension(&self) -> usize;
}

/// Particle Swarm Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleSwarmResult {
    pub best_solution: Vec<f64>,
    pub best_fitness: f64,
    pub iterations: u32,
    pub convergence_history: Vec<f64>,
    pub success: bool,
}

/// Particle Swarm Optimization errors
#[derive(Error, Debug)]
pub enum ParticleSwarmError {
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    #[error("Evaluation error: {0}")]
    EvaluationError(String),
}

/// Simple objective function for testing
pub struct SimpleObjective {
    pub dimension: usize,
    pub bounds: Vec<(f64, f64)>,
}

impl SimpleObjective {
    pub fn new(dimension: usize) -> Self {
        Self {
            dimension,
            bounds: vec![(-10.0, 10.0); dimension],
        }
    }
}

#[async_trait]
impl ParticleSwarmObjective for SimpleObjective {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, ParticleSwarmError> {
        // Simple sphere function: minimize sum of squares
        let fitness = solution.iter().map(|x| x.powi(2)).sum::<f64>();
        Ok(fitness)
    }
    
    fn get_bounds(&self) -> Vec<(f64, f64)> {
        self.bounds.clone()
    }
    
    fn get_dimension(&self) -> usize {
        self.dimension
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_particle_creation() {
        let particle = Particle::new(3);
        assert_eq!(particle.position.len(), 3);
        assert_eq!(particle.velocity.len(), 3);
    }
    
    #[test]
    fn test_parameters_default() {
        let params = ParticleSwarmParameters::default();
        assert_eq!(params.population_size, 50);
        assert_eq!(params.max_iterations, 1000);
    }
    
    #[test]
    fn test_optimizer_creation() {
        let params = ParticleSwarmParameters::default();
        let optimizer = ParticleSwarmOptimizer::new(params, 3);
        assert_eq!(optimizer.particles.len(), 50);
        assert_eq!(optimizer.global_best_position.len(), 3);
    }
    
    #[tokio::test]
    async fn test_simple_objective() {
        let objective = SimpleObjective::new(2);
        let result = objective.evaluate(&[1.0, 2.0]).await.unwrap();
        assert_eq!(result, 5.0); // 1^2 + 2^2 = 5
    }
}