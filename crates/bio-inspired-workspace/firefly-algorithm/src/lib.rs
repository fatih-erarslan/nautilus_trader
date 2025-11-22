//! Firefly Algorithm (FA) Implementation

use async_trait::async_trait;
use rand::Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireflyAlgorithmParameters {
    pub population_size: usize,
    pub max_iterations: u32,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
}

impl Default for FireflyAlgorithmParameters {
    fn default() -> Self {
        Self { population_size: 50, max_iterations: 1000, alpha: 0.2, beta: 1.0, gamma: 1.0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Firefly {
    pub position: Vec<f64>,
    pub fitness: f64,
    pub brightness: f64,
}

impl Firefly {
    pub fn new(dimension: usize) -> Self {
        let mut rng = rand::thread_rng();
        let position = (0..dimension).map(|_| rng.gen_range(0.0..1.0)).collect();
        Self { position, fitness: 0.0, brightness: 0.0 }
    }
}

#[derive(Debug)]
pub struct FireflyAlgorithmOptimizer {
    pub parameters: FireflyAlgorithmParameters,
    pub fireflies: Vec<Firefly>,
    pub best_firefly: Option<Firefly>,
    pub iteration: u32,
}

impl FireflyAlgorithmOptimizer {
    pub fn new(parameters: FireflyAlgorithmParameters, dimension: usize) -> Self {
        let fireflies = (0..parameters.population_size).map(|_| Firefly::new(dimension)).collect();
        Self { parameters, fireflies, best_firefly: None, iteration: 0 }
    }
    
    pub fn get_best_solution(&self) -> Option<&Firefly> { self.best_firefly.as_ref() }
    pub fn get_population_diversity(&self) -> f64 { 0.5 }
    pub fn is_converged(&self) -> bool { false }
}

#[derive(Error, Debug)]
pub enum FireflyAlgorithmError {
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),
}

#[async_trait]
pub trait FireflyAlgorithmObjective: Send + Sync {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, FireflyAlgorithmError>;
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
impl FireflyAlgorithmObjective for SimpleObjective {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, FireflyAlgorithmError> {
        Ok(solution.iter().map(|x| x.powi(2)).sum::<f64>())
    }
    
    fn get_bounds(&self) -> Vec<(f64, f64)> { self.bounds.clone() }
    fn get_dimension(&self) -> usize { self.dimension }
}