//! Ant Colony Optimization (ACO) Algorithm Implementation
//! 
//! This crate provides a basic implementation of the Ant Colony Optimization algorithm,
//! inspired by the foraging behavior of ants and their pheromone trail communication.

use async_trait::async_trait;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use thiserror::Error;

/// Ant Colony Optimization algorithm parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntColonyParameters {
    pub population_size: usize,
    pub max_iterations: u32,
    pub pheromone_evaporation: f64,
    pub alpha: f64, // pheromone importance
    pub beta: f64,  // heuristic importance
    pub rho: f64,   // pheromone decay rate
}

impl Default for AntColonyParameters {
    fn default() -> Self {
        Self {
            population_size: 50,
            max_iterations: 1000,
            pheromone_evaporation: 0.1,
            alpha: 1.0,
            beta: 2.0,
            rho: 0.5,
        }
    }
}

/// Individual ant in the colony
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ant {
    pub position: Vec<f64>,
    pub fitness: f64,
    pub path: Vec<usize>,
}

impl Ant {
    pub fn new(dimension: usize) -> Self {
        let mut rng = rand::thread_rng();
        let position = (0..dimension)
            .map(|_| rng.gen_range(0.0..1.0))
            .collect();
        
        Self {
            position,
            fitness: 0.0,
            path: Vec::new(),
        }
    }
}

/// Ant Colony Optimization algorithm
#[derive(Debug)]
pub struct AntColonyOptimizer {
    pub parameters: AntColonyParameters,
    pub ants: Vec<Ant>,
    pub pheromone_matrix: Vec<Vec<f64>>,
    pub best_solution: Option<Ant>,
    pub iteration: u32,
}

impl AntColonyOptimizer {
    pub fn new(parameters: AntColonyParameters, dimension: usize) -> Self {
        let ants = (0..parameters.population_size)
            .map(|_| Ant::new(dimension))
            .collect();
        
        let pheromone_matrix = vec![vec![1.0; dimension]; dimension];
        
        Self {
            parameters,
            ants,
            pheromone_matrix,
            best_solution: None,
            iteration: 0,
        }
    }
    
    pub fn initialize_population(&mut self, bounds: &[(f64, f64)]) {
        let mut rng = rand::thread_rng();
        
        for ant in &mut self.ants {
            ant.position = bounds.iter()
                .map(|(min, max)| rng.gen_range(*min..*max))
                .collect();
            ant.fitness = 0.0;
        }
    }
    
    pub fn update_pheromones(&mut self) {
        // Evaporate pheromones
        for i in 0..self.pheromone_matrix.len() {
            for j in 0..self.pheromone_matrix[i].len() {
                self.pheromone_matrix[i][j] *= 1.0 - self.parameters.pheromone_evaporation;
            }
        }
        
        // Deposit new pheromones
        for ant in &self.ants {
            if ant.fitness > 0.0 {
                let pheromone_amount = 1.0 / ant.fitness;
                for i in 0..ant.path.len() - 1 {
                    let from = ant.path[i];
                    let to = ant.path[i + 1];
                    if from < self.pheromone_matrix.len() && to < self.pheromone_matrix[from].len() {
                        self.pheromone_matrix[from][to] += pheromone_amount;
                    }
                }
            }
        }
    }
    
    pub fn get_best_solution(&self) -> Option<&Ant> {
        self.best_solution.as_ref()
    }
    
    pub fn get_population_diversity(&self) -> f64 {
        if self.ants.is_empty() {
            return 0.0;
        }
        
        let mut diversity = 0.0;
        let n = self.ants.len();
        
        for i in 0..n {
            for j in i + 1..n {
                let distance: f64 = self.ants[i].position.iter()
                    .zip(&self.ants[j].position)
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

/// Objective function trait for ant colony optimization
#[async_trait]
pub trait AntColonyObjective: Send + Sync {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, AntColonyError>;
    fn get_bounds(&self) -> Vec<(f64, f64)>;
    fn get_dimension(&self) -> usize;
}

/// Ant Colony Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntColonyResult {
    pub best_solution: Vec<f64>,
    pub best_fitness: f64,
    pub iterations: u32,
    pub convergence_history: Vec<f64>,
    pub success: bool,
}

/// Ant Colony Optimization errors
#[derive(Error, Debug)]
pub enum AntColonyError {
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
impl AntColonyObjective for SimpleObjective {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, AntColonyError> {
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
    fn test_ant_creation() {
        let ant = Ant::new(3);
        assert_eq!(ant.position.len(), 3);
        assert_eq!(ant.fitness, 0.0);
    }
    
    #[test]
    fn test_parameters_default() {
        let params = AntColonyParameters::default();
        assert_eq!(params.population_size, 50);
        assert_eq!(params.max_iterations, 1000);
    }
    
    #[test]
    fn test_optimizer_creation() {
        let params = AntColonyParameters::default();
        let optimizer = AntColonyOptimizer::new(params, 3);
        assert_eq!(optimizer.ants.len(), 50);
        assert_eq!(optimizer.pheromone_matrix.len(), 3);
    }
    
    #[tokio::test]
    async fn test_simple_objective() {
        let objective = SimpleObjective::new(2);
        let result = objective.evaluate(&[1.0, 2.0]).await.unwrap();
        assert_eq!(result, 5.0); // 1^2 + 2^2 = 5
    }
}