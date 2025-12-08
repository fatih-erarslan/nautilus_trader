//! Fluent builder API for HyperPhysics
//!
//! Provides a convenient builder pattern for common operations.

use crate::{
    Result, Strategy, Topology, ProblemType,
    OptimizationConfig, optimizer::{Optimizer, OptimizationResult},
};

/// Main entry point for HyperPhysics functionality
pub struct HyperPhysics;

impl HyperPhysics {
    /// Start building an optimization task
    ///
    /// # Example
    ///
    /// ```rust
    /// use hyperphysics_plugin::prelude::*;
    ///
    /// let result = HyperPhysics::optimize()
    ///     .dimensions(5)
    ///     .bounds(-10.0, 10.0)
    ///     .strategy(Strategy::GreyWolf)
    ///     .minimize(|x| x.iter().map(|xi| xi * xi).sum())
    ///     .unwrap();
    /// ```
    pub fn optimize() -> OptimizeBuilder {
        OptimizeBuilder::new()
    }
    
    /// Quick optimization with defaults
    ///
    /// # Example
    ///
    /// ```rust
    /// use hyperphysics_plugin::prelude::*;
    ///
    /// let result = HyperPhysics::quick_optimize(5, |x| {
    ///     x.iter().map(|xi| xi * xi).sum()
    /// }).unwrap();
    /// ```
    pub fn quick_optimize<F>(dimensions: usize, objective: F) -> Result<OptimizationResult>
    where
        F: Fn(&[f64]) -> f64,
    {
        Self::optimize()
            .dimensions(dimensions)
            .bounds(-100.0, 100.0)
            .minimize(objective)
    }
    
    /// Benchmark an objective function with multiple strategies
    pub fn benchmark<F>(dimensions: usize, objective: F) -> Result<Vec<(Strategy, OptimizationResult)>>
    where
        F: Fn(&[f64]) -> f64 + Clone,
    {
        let strategies = [
            Strategy::ParticleSwarm,
            Strategy::GreyWolf,
            Strategy::Whale,
            Strategy::Cuckoo,
            Strategy::DifferentialEvolution,
        ];
        
        let mut results = Vec::new();
        
        for strategy in strategies {
            let result = Self::optimize()
                .dimensions(dimensions)
                .bounds(-100.0, 100.0)
                .strategy(strategy)
                .iterations(500)
                .minimize(objective.clone())?;
            
            results.push((strategy, result));
        }
        
        // Sort by fitness
        results.sort_by(|a, b| a.1.fitness.partial_cmp(&b.1.fitness).unwrap());
        
        Ok(results)
    }
}

/// Builder for optimization tasks
pub struct OptimizeBuilder {
    config: OptimizationConfig,
}

impl OptimizeBuilder {
    /// Create a new builder with defaults
    pub fn new() -> Self {
        Self {
            config: OptimizationConfig::default(),
        }
    }
    
    /// Set the number of dimensions
    pub fn dimensions(mut self, dim: usize) -> Self {
        self.config.dimensions = dim;
        // Resize bounds if they don't match
        if self.config.bounds.len() != dim {
            let default_bound = self.config.bounds.first().copied().unwrap_or((-100.0, 100.0));
            self.config.bounds = vec![default_bound; dim];
        }
        self
    }
    
    /// Set uniform bounds for all dimensions
    pub fn bounds(mut self, min: f64, max: f64) -> Self {
        self.config.bounds = vec![(min, max); self.config.dimensions];
        self
    }
    
    /// Set custom bounds per dimension
    pub fn custom_bounds(mut self, bounds: Vec<(f64, f64)>) -> Self {
        self.config.dimensions = bounds.len();
        self.config.bounds = bounds;
        self
    }
    
    /// Set the optimization strategy
    pub fn strategy(mut self, strategy: Strategy) -> Self {
        self.config.strategy = strategy;
        self
    }
    
    /// Set the problem type for auto-strategy selection
    pub fn problem_type(mut self, problem_type: ProblemType) -> Self {
        self.config.problem_type = problem_type;
        self.config.strategy = problem_type.recommended_strategy();
        self
    }
    
    /// Set the swarm topology
    pub fn topology(mut self, topology: Topology) -> Self {
        self.config.topology = topology;
        self
    }
    
    /// Set population size
    pub fn population(mut self, size: usize) -> Self {
        self.config.population_size = size;
        self
    }
    
    /// Set maximum iterations
    pub fn iterations(mut self, max: usize) -> Self {
        self.config.max_iterations = max;
        self
    }
    
    /// Set convergence tolerance
    pub fn tolerance(mut self, tol: f64) -> Self {
        self.config.tolerance = tol;
        self
    }
    
    /// Set a custom parameter
    pub fn param(mut self, name: &str, value: f64) -> Self {
        self.config.params.insert(name.to_string(), value);
        self
    }
    
    /// Run minimization
    pub fn minimize<F>(self, objective: F) -> Result<OptimizationResult>
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut config = self.config;
        config.maximize = false;
        
        let mut optimizer = Optimizer::new(config)?;
        optimizer.optimize(objective)
    }
    
    /// Run maximization
    pub fn maximize<F>(self, objective: F) -> Result<OptimizationResult>
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut config = self.config;
        config.maximize = true;
        
        let mut optimizer = Optimizer::new(config)?;
        optimizer.optimize(objective)
    }
    
    /// Get the current configuration
    pub fn config(&self) -> &OptimizationConfig {
        &self.config
    }
    
    /// Build and return the optimizer without running
    pub fn build(self) -> Result<Optimizer> {
        Optimizer::new(self.config)
    }
}

impl Default for OptimizeBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|xi| xi * xi).sum()
    }
    
    #[test]
    fn test_builder_minimize() {
        let result = HyperPhysics::optimize()
            .dimensions(5)
            .bounds(-5.0, 5.0)
            .strategy(Strategy::GreyWolf)
            .iterations(200)
            .minimize(sphere)
            .unwrap();
        
        assert!(result.fitness < 0.1);
    }
    
    #[test]
    fn test_quick_optimize() {
        let result = HyperPhysics::quick_optimize(3, sphere).unwrap();
        assert!(result.fitness < 1.0);
    }
}
