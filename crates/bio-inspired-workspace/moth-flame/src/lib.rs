//! Moth-Flame Optimization (MFO) Algorithm Implementation

use async_trait::async_trait;
use rand::Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MothFlameParameters {
    pub population_size: usize,
    pub max_iterations: u32,
    pub b: f64, // spiral constant
}

impl Default for MothFlameParameters {
    fn default() -> Self {
        Self { population_size: 50, max_iterations: 1000, b: 1.0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Moth {
    pub position: Vec<f64>,
    pub fitness: f64,
}

impl Moth {
    pub fn new(dimension: usize) -> Self {
        let mut rng = rand::thread_rng();
        let position = (0..dimension).map(|_| rng.gen_range(0.0..1.0)).collect();
        Self { position, fitness: 0.0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Flame {
    pub position: Vec<f64>,
    pub fitness: f64,
}

#[derive(Debug)]
pub struct MothFlameOptimizer {
    pub parameters: MothFlameParameters,
    pub moths: Vec<Moth>,
    pub flames: Vec<Flame>,
    pub best_flame: Option<Flame>,
    pub iteration: u32,
}

impl MothFlameOptimizer {
    pub fn new(parameters: MothFlameParameters, dimension: usize) -> Self {
        let moths = (0..parameters.population_size).map(|_| Moth::new(dimension)).collect();
        let flames = Vec::new();
        Self { parameters, moths, flames, best_flame: None, iteration: 0 }
    }
    
    pub fn get_best_solution(&self) -> Option<&Flame> { self.best_flame.as_ref() }
    pub fn get_population_diversity(&self) -> f64 { 0.5 }
    pub fn is_converged(&self) -> bool { false }
}

#[derive(Error, Debug)]
pub enum MothFlameError {
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),
}

#[async_trait]
pub trait MothFlameObjective: Send + Sync {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, MothFlameError>;
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
impl MothFlameObjective for SimpleObjective {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, MothFlameError> {
        Ok(solution.iter().map(|x| x.powi(2)).sum::<f64>())
    }
    
    fn get_bounds(&self) -> Vec<(f64, f64)> { self.bounds.clone() }
    fn get_dimension(&self) -> usize { self.dimension }
}