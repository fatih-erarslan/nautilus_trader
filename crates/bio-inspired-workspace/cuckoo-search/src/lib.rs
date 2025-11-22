//! Cuckoo Search (CS) Algorithm Implementation

use async_trait::async_trait;
use rand::Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuckooSearchParameters {
    pub population_size: usize,
    pub max_iterations: u32,
    pub pa: f64, // probability of abandonment
    pub beta: f64, // Lévy flight parameter
}

impl Default for CuckooSearchParameters {
    fn default() -> Self {
        Self { population_size: 50, max_iterations: 1000, pa: 0.25, beta: 1.5 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Nest {
    pub position: Vec<f64>,
    pub fitness: f64,
}

impl Nest {
    pub fn new(dimension: usize) -> Self {
        let mut rng = rand::thread_rng();
        let position = (0..dimension).map(|_| rng.gen_range(0.0..1.0)).collect();
        Self { position, fitness: 0.0 }
    }
}

#[derive(Debug)]
pub struct CuckooSearchOptimizer {
    pub parameters: CuckooSearchParameters,
    pub nests: Vec<Nest>,
    pub best_nest: Option<Nest>,
    pub iteration: u32,
}

impl CuckooSearchOptimizer {
    pub fn new(parameters: CuckooSearchParameters, dimension: usize) -> Self {
        let nests = (0..parameters.population_size).map(|_| Nest::new(dimension)).collect();
        Self { parameters, nests, best_nest: None, iteration: 0 }
    }
    
    pub fn get_best_solution(&self) -> Option<&Nest> { self.best_nest.as_ref() }
    pub fn get_population_diversity(&self) -> f64 { 0.5 }
    pub fn is_converged(&self) -> bool { false }
}

#[derive(Error, Debug)]
pub enum CuckooSearchError {
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),
}

#[async_trait]
pub trait CuckooSearchObjective: Send + Sync {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, CuckooSearchError>;
    fn get_bounds(&self) -> Vec<(f64, f64)>;
    fn get_dimension(&self) -> usize;
}

pub struct SimpleObjective {
    pub dimension: usize,
    pub bounds: Vec<(f64, f64)>,
}

impl SimpleObjective {
    pub fn new(dimension: usize) -> Self {
        Self { dimension, bounds: vec![(-10.0, 10.0); dimension] }
    }
}

#[async_trait]
impl CuckooSearchObjective for SimpleObjective {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, CuckooSearchError> {
        Ok(solution.iter().map(|x| x.powi(2)).sum::<f64>())
    }
    
    fn get_bounds(&self) -> Vec<(f64, f64)> { self.bounds.clone() }
    fn get_dimension(&self) -> usize { self.dimension }
}