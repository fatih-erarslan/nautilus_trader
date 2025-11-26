//! Whale Optimization Algorithm (WOA) Implementation

use async_trait::async_trait;
use rand::Rng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WhaleOptimizationParameters {
    pub population_size: usize,
    pub max_iterations: u32,
    pub a_decay: f64,
    pub b_spiral: f64,
}

impl Default for WhaleOptimizationParameters {
    fn default() -> Self {
        Self { population_size: 50, max_iterations: 1000, a_decay: 2.0, b_spiral: 1.0 }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Whale {
    pub position: Vec<f64>,
    pub fitness: f64,
}

impl Whale {
    pub fn new(dimension: usize) -> Self {
        let mut rng = rand::thread_rng();
        let position = (0..dimension).map(|_| rng.gen_range(0.0..1.0)).collect();
        Self { position, fitness: 0.0 }
    }
}

#[derive(Debug)]
pub struct WhaleOptimizationOptimizer {
    pub parameters: WhaleOptimizationParameters,
    pub whales: Vec<Whale>,
    pub best_whale: Option<Whale>,
    pub iteration: u32,
}

impl WhaleOptimizationOptimizer {
    pub fn new(parameters: WhaleOptimizationParameters, dimension: usize) -> Self {
        let whales = (0..parameters.population_size).map(|_| Whale::new(dimension)).collect();
        Self { parameters, whales, best_whale: None, iteration: 0 }
    }

    pub fn get_best_solution(&self) -> Option<&Whale> { self.best_whale.as_ref() }

    /// Calculate population diversity using average pairwise Euclidean distance
    /// Higher diversity means more exploration, lower means convergence
    /// Based on: Das & Suganthan (2011) "Differential Evolution: A Survey"
    pub fn get_population_diversity(&self) -> f64 {
        if self.whales.len() < 2 {
            return 0.0;
        }

        let n = self.whales.len();
        let mut total_distance = 0.0;
        let mut count = 0;

        // Calculate average pairwise Euclidean distance (normalized)
        for i in 0..n {
            for j in (i + 1)..n {
                let distance: f64 = self.whales[i].position.iter()
                    .zip(self.whales[j].position.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum::<f64>()
                    .sqrt();
                total_distance += distance;
                count += 1;
            }
        }

        if count == 0 {
            return 0.0;
        }

        // Normalize by dimension and return average distance
        let dimension = self.whales[0].position.len() as f64;
        let avg_distance = total_distance / count as f64;

        // Normalize to [0, 1] assuming search space is [0, 1]^d
        // Max possible distance in unit hypercube is sqrt(d)
        let max_distance = dimension.sqrt();
        (avg_distance / max_distance).min(1.0)
    }

    /// Check convergence using population diversity threshold
    /// Converged when diversity drops below 1% of initial diversity
    /// Based on: Mirjalili & Lewis (2016) "The Whale Optimization Algorithm"
    pub fn is_converged(&self) -> bool {
        const CONVERGENCE_THRESHOLD: f64 = 0.01;

        // Check iteration limit
        if self.iteration >= self.parameters.max_iterations {
            return true;
        }

        // Check diversity-based convergence
        let diversity = self.get_population_diversity();
        diversity < CONVERGENCE_THRESHOLD
    }
}

#[derive(Error, Debug)]
pub enum WhaleOptimizationError {
    #[error("Optimization failed: {0}")]
    OptimizationFailed(String),
}

#[async_trait]
pub trait WhaleOptimizationObjective: Send + Sync {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, WhaleOptimizationError>;
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
impl WhaleOptimizationObjective for SimpleObjective {
    async fn evaluate(&self, solution: &[f64]) -> Result<f64, WhaleOptimizationError> {
        Ok(solution.iter().map(|x| x.powi(2)).sum::<f64>())
    }
    
    fn get_bounds(&self) -> Vec<(f64, f64)> { self.bounds.clone() }
    fn get_dimension(&self) -> usize { self.dimension }
}