//! Differential Evolution (DE) Algorithm Implementation

use async_trait::async_trait;
use rand::Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialEvolutionParameters {
    pub population_size: usize,
    pub max_generations: u32,
    pub mutation_factor: f64,
    pub crossover_rate: f64,
}

impl Default for DifferentialEvolutionParameters {
    fn default() -> Self {
        Self {
            population_size: 50,
            max_generations: 1000,
            mutation_factor: 0.8,
            crossover_rate: 0.9,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Individual {
    pub parameters: Vec<f64>,
    pub fitness: f64,
}

impl Individual {
    pub fn new(dimension: usize) -> Self {
        let mut rng = rand::thread_rng();
        let parameters = (0..dimension).map(|_| rng.gen_range(0.0..1.0)).collect();
        Self { parameters, fitness: 0.0 }
    }
}

#[derive(Debug)]
pub struct DifferentialEvolutionOptimizer {
    pub parameters: DifferentialEvolutionParameters,
    pub population: Vec<Individual>,
    pub best_solution: Option<Individual>,
    pub generation: u32,
}

impl DifferentialEvolutionOptimizer {
    pub fn new(parameters: DifferentialEvolutionParameters, dimension: usize) -> Self {
        let population = (0..parameters.population_size)
            .map(|_| Individual::new(dimension))
            .collect();
        
        Self {
            parameters,
            population,
            best_solution: None,
            generation: 0,
        }
    }
    
    pub fn get_best_solution(&self) -> Option<&Individual> {
        self.best_solution.as_ref()
    }
    
    pub fn get_population_diversity(&self) -> f64 {
        if self.population.is_empty() { return 0.0; }
        
        let mut diversity = 0.0;
        let n = self.population.len();
        
        for i in 0..n {
            for j in i + 1..n {
                let distance: f64 = self.population[i].parameters.iter()
                    .zip(&self.population[j].parameters)
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

#[derive(Error, Debug)]
pub enum DifferentialEvolutionError {
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    #[error("Evaluation error: {0}")]
    EvaluationError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DifferentialEvolutionResult {
    pub best_solution: Vec<f64>,
    pub best_fitness: f64,
    pub generations: u32,
    pub convergence_history: Vec<f64>,
    pub success: bool,
}

#[async_trait]
pub trait DifferentialEvolutionObjective: Send + Sync {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, DifferentialEvolutionError>;
    fn get_bounds(&self) -> Vec<(f64, f64)>;
    fn get_dimension(&self) -> usize;
}

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
impl DifferentialEvolutionObjective for SimpleObjective {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, DifferentialEvolutionError> {
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
    fn test_individual_creation() {
        let individual = Individual::new(3);
        assert_eq!(individual.parameters.len(), 3);
    }
    
    #[test]
    fn test_parameters_default() {
        let params = DifferentialEvolutionParameters::default();
        assert_eq!(params.population_size, 50);
    }
}