//! Core types and data structures for swarm intelligence algorithms

use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use nalgebra::{DVector, DMatrix};
use ndarray::{Array1, Array2};

/// A position in the optimization space
pub type Position = DVector<f64>;

/// A velocity vector for particle-based algorithms
pub type Velocity = DVector<f64>;

/// Fitness value type
pub type Fitness = f64;

/// Trait representing an individual in a swarm population
pub trait Individual: Clone + Send + Sync + Debug {
    /// Get the position of this individual
    fn position(&self) -> &Position;
    
    /// Get a mutable reference to the position
    fn position_mut(&mut self) -> &mut Position;
    
    /// Get the fitness value
    fn fitness(&self) -> &Fitness;
    
    /// Set the fitness value
    fn set_fitness(&mut self, fitness: Fitness);
    
    /// Update the position
    fn update_position(&mut self, new_position: Position);
    
    /// Get the dimensionality of this individual
    fn dimensions(&self) -> usize {
        self.position().len()
    }
}

/// Basic implementation of an individual
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BasicIndividual {
    position: Position,
    fitness: Fitness,
}

impl BasicIndividual {
    /// Create a new individual with given position
    pub fn new(position: Position) -> Self {
        Self {
            position,
            fitness: f64::INFINITY, // Initialize with worst possible fitness
        }
    }
    
    /// Create a random individual within bounds
    pub fn random(dimensions: usize, lower_bound: f64, upper_bound: f64) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let position = Position::from_fn(dimensions, |_, _| {
            rng.gen_range(lower_bound..=upper_bound)
        });
        
        Self::new(position)
    }
}

impl Individual for BasicIndividual {
    fn position(&self) -> &Position {
        &self.position
    }
    
    fn position_mut(&mut self) -> &mut Position {
        &mut self.position
    }
    
    fn fitness(&self) -> &Fitness {
        &self.fitness
    }
    
    fn set_fitness(&mut self, fitness: Fitness) {
        self.fitness = fitness;
    }
    
    fn update_position(&mut self, new_position: Position) {
        self.position = new_position;
    }
}

/// Particle for PSO algorithm with velocity
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Particle {
    position: Position,
    velocity: Velocity,
    fitness: Fitness,
    best_position: Position,
    best_fitness: Fitness,
}

impl Particle {
    /// Create a new particle
    pub fn new(position: Position, velocity: Velocity) -> Self {
        let best_position = position.clone();
        Self {
            position,
            velocity,
            fitness: f64::INFINITY,
            best_position,
            best_fitness: f64::INFINITY,
        }
    }
    
    /// Create a random particle within bounds
    pub fn random(dimensions: usize, pos_bounds: (f64, f64), vel_bounds: (f64, f64)) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        let position = Position::from_fn(dimensions, |_, _| {
            rng.gen_range(pos_bounds.0..=pos_bounds.1)
        });
        
        let velocity = Velocity::from_fn(dimensions, |_, _| {
            rng.gen_range(vel_bounds.0..=vel_bounds.1)
        });
        
        Self::new(position, velocity)
    }
    
    /// Get velocity
    pub fn velocity(&self) -> &Velocity {
        &self.velocity
    }
    
    /// Get mutable velocity
    pub fn velocity_mut(&mut self) -> &mut Velocity {
        &mut self.velocity
    }
    
    /// Get personal best position
    pub fn best_position(&self) -> &Position {
        &self.best_position
    }
    
    /// Get personal best fitness
    pub fn best_fitness(&self) -> Fitness {
        self.best_fitness
    }
    
    /// Update personal best if current is better
    pub fn update_personal_best(&mut self) {
        if self.fitness < self.best_fitness {
            self.best_position = self.position.clone();
            self.best_fitness = self.fitness;
        }
    }
    
    /// Update velocity and position
    pub fn update(&mut self, global_best: &Position, inertia: f64, c1: f64, c2: f64) {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        
        for i in 0..self.velocity.len() {
            let r1: f64 = rng.gen();
            let r2: f64 = rng.gen();
            
            self.velocity[i] = inertia * self.velocity[i]
                + c1 * r1 * (self.best_position[i] - self.position[i])
                + c2 * r2 * (global_best[i] - self.position[i]);
            
            self.position[i] += self.velocity[i];
        }
    }
}

impl Individual for Particle {
    fn position(&self) -> &Position {
        &self.position
    }
    
    fn position_mut(&mut self) -> &mut Position {
        &mut self.position
    }
    
    fn fitness(&self) -> &Fitness {
        &self.fitness
    }
    
    fn set_fitness(&mut self, fitness: Fitness) {
        self.fitness = fitness;
    }
    
    fn update_position(&mut self, new_position: Position) {
        self.position = new_position;
    }
}

/// A population of individuals
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Population<T: Individual> {
    pub individuals: Vec<T>,
}

impl<T: Individual> Population<T> {
    /// Create a new empty population
    pub fn new() -> Self {
        Self {
            individuals: Vec::new(),
        }
    }
    
    /// Create a population with given capacity
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            individuals: Vec::with_capacity(capacity),
        }
    }
    
    /// Add an individual to the population
    pub fn add(&mut self, individual: T) {
        self.individuals.push(individual);
    }
    
    /// Get population size
    pub fn size(&self) -> usize {
        self.individuals.len()
    }
    
    /// Check if population is empty
    pub fn is_empty(&self) -> bool {
        self.individuals.is_empty()
    }
    
    /// Get the best individual (minimum fitness)
    pub fn best(&self) -> Option<&T> {
        self.individuals.iter()
            .min_by(|a, b| a.fitness().partial_cmp(b.fitness()).unwrap())
    }
    
    /// Get the worst individual (maximum fitness)
    pub fn worst(&self) -> Option<&T> {
        self.individuals.iter()
            .max_by(|a, b| a.fitness().partial_cmp(b.fitness()).unwrap())
    }
    
    /// Calculate average fitness
    pub fn average_fitness(&self) -> Option<f64> {
        if self.individuals.is_empty() {
            return None;
        }
        
        let sum: f64 = self.individuals.iter().map(|i| *i.fitness()).sum();
        Some(sum / self.individuals.len() as f64)
    }
    
    /// Calculate population diversity (standard deviation of distances from centroid)
    pub fn diversity(&self) -> Option<f64> {
        if self.individuals.len() < 2 {
            return None;
        }
        
        let dimensions = self.individuals[0].dimensions();
        
        // Calculate centroid
        let mut centroid = Position::zeros(dimensions);
        for individual in &self.individuals {
            centroid += individual.position();
        }
        centroid /= self.individuals.len() as f64;
        
        // Calculate distances from centroid
        let distances: Vec<f64> = self.individuals.iter()
            .map(|i| (i.position() - &centroid).norm())
            .collect();
        
        // Calculate standard deviation
        let mean_distance = distances.iter().sum::<f64>() / distances.len() as f64;
        let variance = distances.iter()
            .map(|d| (d - mean_distance).powi(2))
            .sum::<f64>() / distances.len() as f64;
        
        Some(variance.sqrt())
    }
    
    /// Sort population by fitness (best first)
    pub fn sort_by_fitness(&mut self) {
        self.individuals.sort_by(|a, b| 
            a.fitness().partial_cmp(b.fitness()).unwrap()
        );
    }
    
    /// Get iterator over individuals
    pub fn iter(&self) -> std::slice::Iter<T> {
        self.individuals.iter()
    }
    
    /// Get mutable iterator over individuals
    pub fn iter_mut(&mut self) -> std::slice::IterMut<T> {
        self.individuals.iter_mut()
    }
}

impl<T: Individual> Default for Population<T> {
    fn default() -> Self {
        Self::new()
    }
}

/// Optimization problem definition
#[derive(Debug, Clone)]
pub struct OptimizationProblem {
    /// Problem dimensions
    pub dimensions: usize,
    
    /// Lower bounds for each dimension
    pub lower_bounds: Position,
    
    /// Upper bounds for each dimension
    pub upper_bounds: Position,
    
    /// Objective function
    pub objective: Box<dyn Fn(&Position) -> f64 + Send + Sync>,
    
    /// Whether this is a minimization problem
    pub minimize: bool,
}

impl OptimizationProblem {
    /// Create a new optimization problem
    pub fn new() -> OptimizationProblemBuilder {
        OptimizationProblemBuilder::new()
    }
    
    /// Evaluate a position
    pub fn evaluate(&self, position: &Position) -> f64 {
        let fitness = (self.objective)(position);
        if self.minimize {
            fitness
        } else {
            -fitness  // Convert maximization to minimization
        }
    }
    
    /// Evaluate multiple positions in parallel
    pub fn evaluate_parallel(&self, positions: &[Position]) -> Vec<f64> {
        use rayon::prelude::*;
        positions.par_iter().map(|pos| self.evaluate(pos)).collect()
    }
    
    /// Check if a position is within bounds
    pub fn is_within_bounds(&self, position: &Position) -> bool {
        position.iter().zip(self.lower_bounds.iter()).zip(self.upper_bounds.iter())
            .all(|((p, l), u)| *p >= *l && *p <= *u)
    }
    
    /// Clamp a position to bounds
    pub fn clamp_to_bounds(&self, position: &mut Position) {
        for (i, p) in position.iter_mut().enumerate() {
            *p = p.clamp(self.lower_bounds[i], self.upper_bounds[i]);
        }
    }
}

impl Default for OptimizationProblem {
    fn default() -> Self {
        Self {
            dimensions: 2,
            lower_bounds: Position::from_element(2, -10.0),
            upper_bounds: Position::from_element(2, 10.0),
            objective: Box::new(|x| x.iter().map(|xi| xi.powi(2)).sum()),
            minimize: true,
        }
    }
}

/// Builder for optimization problems
pub struct OptimizationProblemBuilder {
    dimensions: Option<usize>,
    lower_bounds: Option<Position>,
    upper_bounds: Option<Position>,
    objective: Option<Box<dyn Fn(&Position) -> f64 + Send + Sync>>,
    minimize: bool,
}

impl OptimizationProblemBuilder {
    pub fn new() -> Self {
        Self {
            dimensions: None,
            lower_bounds: None,
            upper_bounds: None,
            objective: None,
            minimize: true,
        }
    }
    
    pub fn dimensions(mut self, dims: usize) -> Self {
        self.dimensions = Some(dims);
        self
    }
    
    pub fn bounds(mut self, lower: f64, upper: f64) -> Self {
        if let Some(dims) = self.dimensions {
            self.lower_bounds = Some(Position::from_element(dims, lower));
            self.upper_bounds = Some(Position::from_element(dims, upper));
        }
        self
    }
    
    pub fn lower_bounds(mut self, bounds: Position) -> Self {
        self.lower_bounds = Some(bounds);
        self
    }
    
    pub fn upper_bounds(mut self, bounds: Position) -> Self {
        self.upper_bounds = Some(bounds);
        self
    }
    
    pub fn objective<F>(mut self, f: F) -> Self
    where
        F: Fn(&Position) -> f64 + Send + Sync + 'static,
    {
        self.objective = Some(Box::new(f));
        self
    }
    
    pub fn minimize(mut self, min: bool) -> Self {
        self.minimize = min;
        self
    }
    
    pub fn build(self) -> Result<OptimizationProblem, String> {
        let dimensions = self.dimensions.ok_or("Dimensions not specified")?;
        let lower_bounds = self.lower_bounds.unwrap_or_else(|| Position::from_element(dimensions, -10.0));
        let upper_bounds = self.upper_bounds.unwrap_or_else(|| Position::from_element(dimensions, 10.0));
        let objective = self.objective.ok_or("Objective function not specified")?;
        
        if lower_bounds.len() != dimensions || upper_bounds.len() != dimensions {
            return Err("Bounds dimension mismatch".to_string());
        }
        
        Ok(OptimizationProblem {
            dimensions,
            lower_bounds,
            upper_bounds,
            objective,
            minimize: self.minimize,
        })
    }
}

/// Result of swarm optimization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmResult<F> {
    /// Best position found
    pub best_position: Position,
    
    /// Best fitness achieved
    pub best_fitness: F,
    
    /// Number of iterations performed
    pub iterations: usize,
    
    /// Convergence history
    pub convergence_history: Vec<F>,
    
    /// Algorithm name
    pub algorithm_name: String,
}

/// System information for performance tuning
#[derive(Debug, Clone)]
pub struct SystemInfo {
    pub cpu_cores: usize,
    pub memory_gb: f64,
    pub has_simd: bool,
    pub has_gpu: bool,
    pub numa_nodes: usize,
}

impl SystemInfo {
    pub fn collect() -> Self {
        Self {
            cpu_cores: num_cpus::get(),
            memory_gb: Self::get_memory_gb(),
            has_simd: Self::detect_simd(),
            has_gpu: Self::detect_gpu(),
            numa_nodes: Self::detect_numa_nodes(),
        }
    }
    
    fn get_memory_gb() -> f64 {
        // Simple approximation - in real implementation would use system APIs
        8.0 // Default assumption
    }
    
    fn detect_simd() -> bool {
        #[cfg(feature = "simd")]
        return true;
        
        #[cfg(not(feature = "simd"))]
        false
    }
    
    fn detect_gpu() -> bool {
        #[cfg(feature = "gpu")]
        return true;
        
        #[cfg(not(feature = "gpu"))]
        false
    }
    
    fn detect_numa_nodes() -> usize {
        1 // Default - would use actual NUMA detection in real implementation
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_basic_individual() {
        let position = Position::from_vec(vec![1.0, 2.0, 3.0]);
        let mut individual = BasicIndividual::new(position.clone());
        
        assert_eq!(individual.position(), &position);
        assert_eq!(*individual.fitness(), f64::INFINITY);
        assert_eq!(individual.dimensions(), 3);
        
        individual.set_fitness(5.0);
        assert_eq!(*individual.fitness(), 5.0);
    }
    
    #[test]
    fn test_particle() {
        let position = Position::from_vec(vec![1.0, 2.0]);
        let velocity = Velocity::from_vec(vec![0.1, -0.1]);
        let mut particle = Particle::new(position.clone(), velocity.clone());
        
        assert_eq!(particle.position(), &position);
        assert_eq!(particle.velocity(), &velocity);
        assert_eq!(particle.best_fitness(), f64::INFINITY);
        
        particle.set_fitness(3.0);
        particle.update_personal_best();
        assert_eq!(particle.best_fitness(), 3.0);
    }
    
    #[test]
    fn test_population() {
        let mut population = Population::new();
        assert!(population.is_empty());
        
        let individual1 = BasicIndividual::new(Position::from_vec(vec![1.0, 2.0]));
        let mut individual2 = BasicIndividual::new(Position::from_vec(vec![3.0, 4.0]));
        individual2.set_fitness(2.0);
        
        population.add(individual1);
        population.add(individual2);
        
        assert_eq!(population.size(), 2);
        assert_eq!(*population.best().unwrap().fitness(), 2.0);
    }
    
    #[test]
    fn test_optimization_problem() {
        let problem = OptimizationProblem::new()
            .dimensions(2)
            .bounds(-5.0, 5.0)
            .objective(|x| x[0].powi(2) + x[1].powi(2))
            .build()
            .unwrap();
        
        assert_eq!(problem.dimensions, 2);
        
        let position = Position::from_vec(vec![1.0, 2.0]);
        let fitness = problem.evaluate(&position);
        assert_relative_eq!(fitness, 5.0, epsilon = 1e-10);
    }
}