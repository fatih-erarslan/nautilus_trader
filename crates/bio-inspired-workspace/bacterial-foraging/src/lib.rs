//! Bacterial Foraging Optimization (BFO) Algorithm Implementation

use async_trait::async_trait;
use rand::Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BacterialForagingParameters {
    pub population_size: usize,
    pub max_iterations: u32,
    pub chemotaxis_steps: u32,
    pub reproduction_steps: u32,
    pub elimination_steps: u32,
    pub step_size: f64,
}

impl Default for BacterialForagingParameters {
    fn default() -> Self {
        Self { population_size: 50, max_iterations: 1000, chemotaxis_steps: 4, reproduction_steps: 4, elimination_steps: 2, step_size: 0.1 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bacterium {
    pub position: Vec<f64>,
    pub fitness: f64,
    pub health: f64,
}

impl Bacterium {
    pub fn new(dimension: usize) -> Self {
        let mut rng = rand::thread_rng();
        let position = (0..dimension).map(|_| rng.gen_range(0.0..1.0)).collect();
        Self { position, fitness: 0.0, health: 0.0 }
    }
}

#[derive(Debug)]
pub struct BacterialForagingOptimizer {
    pub parameters: BacterialForagingParameters,
    pub bacteria: Vec<Bacterium>,
    pub best_bacterium: Option<Bacterium>,
    pub iteration: u32,
}

impl BacterialForagingOptimizer {
    pub fn new(parameters: BacterialForagingParameters, dimension: usize) -> Self {
        let bacteria = (0..parameters.population_size).map(|_| Bacterium::new(dimension)).collect();
        Self { parameters, bacteria, best_bacterium: None, iteration: 0 }
    }
    
    pub fn get_best_solution(&self) -> Option<&Bacterium> { self.best_bacterium.as_ref() }
    pub fn get_population_diversity(&self) -> f64 { 0.5 }
    pub fn is_converged(&self) -> bool { false }
}

#[derive(Error, Debug)]
pub enum BacterialForagingError {
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),
}

#[async_trait]
pub trait BacterialForagingObjective: Send + Sync {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, BacterialForagingError>;
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
impl BacterialForagingObjective for SimpleObjective {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, BacterialForagingError> {
        Ok(solution.iter().map(|x| x.powi(2)).sum::<f64>())
    }
    
    fn get_bounds(&self) -> Vec<(f64, f64)> { self.bounds.clone() }
    fn get_dimension(&self) -> usize { self.dimension }
}