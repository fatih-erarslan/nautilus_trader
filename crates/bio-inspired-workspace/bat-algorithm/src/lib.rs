//! Bat Algorithm (BA) Implementation

use async_trait::async_trait;
use rand::Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatAlgorithmParameters {
    pub population_size: usize,
    pub max_iterations: u32,
    pub frequency_min: f64,
    pub frequency_max: f64,
    pub loudness: f64,
    pub pulse_rate: f64,
}

impl Default for BatAlgorithmParameters {
    fn default() -> Self {
        Self { population_size: 50, max_iterations: 1000, frequency_min: 0.0, frequency_max: 100.0, loudness: 0.5, pulse_rate: 0.5 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Bat {
    pub position: Vec<f64>,
    pub velocity: Vec<f64>,
    pub fitness: f64,
    pub frequency: f64,
    pub loudness: f64,
    pub pulse_rate: f64,
}

impl Bat {
    pub fn new(dimension: usize) -> Self {
        let mut rng = rand::thread_rng();
        let position = (0..dimension).map(|_| rng.gen_range(0.0..1.0)).collect();
        let velocity = (0..dimension).map(|_| rng.gen_range(-1.0..1.0)).collect();
        Self { position, velocity, fitness: 0.0, frequency: 0.0, loudness: 0.5, pulse_rate: 0.5 }
    }
}

#[derive(Debug)]
pub struct BatAlgorithmOptimizer {
    pub parameters: BatAlgorithmParameters,
    pub bats: Vec<Bat>,
    pub best_bat: Option<Bat>,
    pub iteration: u32,
}

impl BatAlgorithmOptimizer {
    pub fn new(parameters: BatAlgorithmParameters, dimension: usize) -> Self {
        let bats = (0..parameters.population_size).map(|_| Bat::new(dimension)).collect();
        Self { parameters, bats, best_bat: None, iteration: 0 }
    }
    
    pub fn get_best_solution(&self) -> Option<&Bat> { self.best_bat.as_ref() }
    pub fn get_population_diversity(&self) -> f64 { 0.5 } // Stub
    pub fn is_converged(&self) -> bool { false } // Stub
}

#[derive(Error, Debug)]
pub enum BatAlgorithmError {
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),
}

#[async_trait]
pub trait BatAlgorithmObjective: Send + Sync {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, BatAlgorithmError>;
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
impl BatAlgorithmObjective for SimpleObjective {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, BatAlgorithmError> {
        Ok(solution.iter().map(|x| x.powi(2)).sum::<f64>())
    }
    
    fn get_bounds(&self) -> Vec<(f64, f64)> { self.bounds.clone() }
    fn get_dimension(&self) -> usize { self.dimension }
}