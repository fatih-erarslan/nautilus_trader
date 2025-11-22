//! Salp Swarm Algorithm (SSA) Implementation

use async_trait::async_trait;
use rand::Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SalpSwarmParameters {
    pub population_size: usize,
    pub max_iterations: u32,
    pub c1: f64, // exploration parameter
}

impl Default for SalpSwarmParameters {
    fn default() -> Self {
        Self { population_size: 50, max_iterations: 1000, c1: 2.0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Salp {
    pub position: Vec<f64>,
    pub fitness: f64,
    pub is_leader: bool,
}

impl Salp {
    pub fn new(dimension: usize) -> Self {
        let mut rng = rand::thread_rng();
        let position = (0..dimension).map(|_| rng.gen_range(0.0..1.0)).collect();
        Self { position, fitness: 0.0, is_leader: false }
    }
}

#[derive(Debug)]
pub struct SalpSwarmOptimizer {
    pub parameters: SalpSwarmParameters,
    pub salps: Vec<Salp>,
    pub food_source: Vec<f64>,
    pub best_salp: Option<Salp>,
    pub iteration: u32,
}

impl SalpSwarmOptimizer {
    pub fn new(parameters: SalpSwarmParameters, dimension: usize) -> Self {
        let salps = (0..parameters.population_size).map(|_| Salp::new(dimension)).collect();
        let food_source = vec![0.0; dimension];
        Self { parameters, salps, food_source, best_salp: None, iteration: 0 }
    }
    
    pub fn get_best_solution(&self) -> Option<&Salp> { self.best_salp.as_ref() }
    pub fn get_population_diversity(&self) -> f64 { 0.5 }
    pub fn is_converged(&self) -> bool { false }
}

#[derive(Error, Debug)]
pub enum SalpSwarmError {
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),
}

#[async_trait]
pub trait SalpSwarmObjective: Send + Sync {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, SalpSwarmError>;
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
impl SalpSwarmObjective for SimpleObjective {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, SalpSwarmError> {
        Ok(solution.iter().map(|x| x.powi(2)).sum::<f64>())
    }
    
    fn get_bounds(&self) -> Vec<(f64, f64)> { self.bounds.clone() }
    fn get_dimension(&self) -> usize { self.dimension }
}