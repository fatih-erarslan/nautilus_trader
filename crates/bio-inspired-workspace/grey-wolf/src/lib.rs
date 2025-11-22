//! Grey Wolf Optimizer (GWO) Algorithm Implementation

use async_trait::async_trait;
use rand::Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GreyWolfParameters {
    pub population_size: usize,
    pub max_iterations: u32,
    pub a_decay: f64,
}

impl Default for GreyWolfParameters {
    fn default() -> Self {
        Self {
            population_size: 50,
            max_iterations: 1000,
            a_decay: 2.0,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Wolf {
    pub position: Vec<f64>,
    pub fitness: f64,
    pub rank: WolfRank,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum WolfRank {
    Alpha,
    Beta,
    Delta,
    Omega,
}

impl Wolf {
    pub fn new(dimension: usize) -> Self {
        let mut rng = rand::thread_rng();
        let position = (0..dimension).map(|_| rng.gen_range(0.0..1.0)).collect();
        Self { position, fitness: 0.0, rank: WolfRank::Omega }
    }
}

#[derive(Debug)]
pub struct GreyWolfOptimizer {
    pub parameters: GreyWolfParameters,
    pub wolves: Vec<Wolf>,
    pub alpha: Option<Wolf>,
    pub beta: Option<Wolf>,
    pub delta: Option<Wolf>,
    pub iteration: u32,
}

impl GreyWolfOptimizer {
    pub fn new(parameters: GreyWolfParameters, dimension: usize) -> Self {
        let wolves = (0..parameters.population_size)
            .map(|_| Wolf::new(dimension))
            .collect();
        
        Self {
            parameters,
            wolves,
            alpha: None,
            beta: None,
            delta: None,
            iteration: 0,
        }
    }
    
    pub fn get_best_solution(&self) -> Option<&Wolf> {
        self.alpha.as_ref()
    }
    
    pub fn get_population_diversity(&self) -> f64 {
        if self.wolves.is_empty() { return 0.0; }
        
        let mut diversity = 0.0;
        let n = self.wolves.len();
        
        for i in 0..n {
            for j in i + 1..n {
                let distance: f64 = self.wolves[i].position.iter()
                    .zip(&self.wolves[j].position)
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
pub enum GreyWolfError {
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),
    #[error("Invalid parameters: {0}")]
    InvalidParameters(String),
    #[error("Evaluation error: {0}")]
    EvaluationError(String),
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GreyWolfResult {
    pub best_solution: Vec<f64>,
    pub best_fitness: f64,
    pub iterations: u32,
    pub convergence_history: Vec<f64>,
    pub success: bool,
}

#[async_trait]
pub trait GreyWolfObjective: Send + Sync {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, GreyWolfError>;
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
impl GreyWolfObjective for SimpleObjective {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, GreyWolfError> {
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
    fn test_wolf_creation() {
        let wolf = Wolf::new(3);
        assert_eq!(wolf.position.len(), 3);
        assert_eq!(wolf.rank, WolfRank::Omega);
    }
    
    #[test]
    fn test_parameters_default() {
        let params = GreyWolfParameters::default();
        assert_eq!(params.population_size, 50);
    }
}