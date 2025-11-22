//! Artificial Bee Colony (ABC) Algorithm Implementation

use async_trait::async_trait;
use rand::Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ArtificialBeeColonyParameters {
    pub population_size: usize,
    pub max_iterations: u32,
    pub limit: u32, // abandonment limit
}

impl Default for ArtificialBeeColonyParameters {
    fn default() -> Self {
        Self { population_size: 50, max_iterations: 1000, limit: 100 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FoodSource {
    pub position: Vec<f64>,
    pub fitness: f64,
    pub trial_count: u32,
}

impl FoodSource {
    pub fn new(dimension: usize) -> Self {
        let mut rng = rand::thread_rng();
        let position = (0..dimension).map(|_| rng.gen_range(0.0..1.0)).collect();
        Self { position, fitness: 0.0, trial_count: 0 }
    }
}

#[derive(Debug)]
pub struct ArtificialBeeColonyOptimizer {
    pub parameters: ArtificialBeeColonyParameters,
    pub food_sources: Vec<FoodSource>,
    pub best_source: Option<FoodSource>,
    pub iteration: u32,
}

impl ArtificialBeeColonyOptimizer {
    pub fn new(parameters: ArtificialBeeColonyParameters, dimension: usize) -> Self {
        let food_sources = (0..parameters.population_size).map(|_| FoodSource::new(dimension)).collect();
        Self { parameters, food_sources, best_source: None, iteration: 0 }
    }
    
    pub fn get_best_solution(&self) -> Option<&FoodSource> { self.best_source.as_ref() }
    pub fn get_population_diversity(&self) -> f64 { 0.5 }
    pub fn is_converged(&self) -> bool { false }
}

#[derive(Error, Debug)]
pub enum ArtificialBeeColonyError {
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),
}

#[async_trait]
pub trait ArtificialBeeColonyObjective: Send + Sync {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, ArtificialBeeColonyError>;
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
impl ArtificialBeeColonyObjective for SimpleObjective {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, ArtificialBeeColonyError> {
        Ok(solution.iter().map(|x| x.powi(2)).sum::<f64>())
    }
    
    fn get_bounds(&self) -> Vec<(f64, f64)> { self.bounds.clone() }
    fn get_dimension(&self) -> usize { self.dimension }
}