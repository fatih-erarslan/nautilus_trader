//! Ant Colony Optimization (ACO) implementation
//!
//! ACO is inspired by the foraging behavior of ants. Ants deposit pheromones
//! on paths and prefer routes with higher pheromone concentrations, leading
//! to emergent optimization through collective behavior.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use rayon::prelude::*;
use rand::prelude::*;
use ndarray::{Array2, Array1};

use crate::core::{
    SwarmAlgorithm, SwarmError, SwarmResult, OptimizationProblem,
    Population, Individual, BasicIndividual, Position, AlgorithmMetrics,
    AdaptiveAlgorithm, ParallelAlgorithm, AdaptationStrategy
};
use crate::validate_parameter;

/// ACO algorithm variants
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum AcoVariant {
    /// Classic Ant System (AS)
    AntSystem,
    /// Elitist Ant System (EAS)
    ElitistAntSystem,
    /// Rank-based Ant System (ASrank)
    RankBasedAntSystem,
    /// Max-Min Ant System (MMAS)
    MaxMinAntSystem,
    /// Ant Colony System (ACS)
    AntColonySystem,
}

/// ACO algorithm parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AcoParameters {
    /// Number of ants in the colony
    pub num_ants: usize,
    
    /// Pheromone importance factor (alpha)
    pub alpha: f64,
    
    /// Heuristic importance factor (beta)
    pub beta: f64,
    
    /// Pheromone evaporation rate (rho)
    pub evaporation_rate: f64,
    
    /// Pheromone deposit amount (Q)
    pub pheromone_deposit: f64,
    
    /// Initial pheromone level
    pub initial_pheromone: f64,
    
    /// ACO variant to use
    pub variant: AcoVariant,
    
    /// Local search probability
    pub local_search_probability: f64,
    
    /// Elitist ants count (for EAS)
    pub num_elitist_ants: usize,
    
    /// Max pheromone level (for MMAS)
    pub max_pheromone: f64,
    
    /// Min pheromone level (for MMAS)
    pub min_pheromone: f64,
    
    /// Exploration vs exploitation balance (for ACS)
    pub q0: f64,
    
    /// Local pheromone update parameter (for ACS)
    pub xi: f64,
    
    /// Solution construction step size
    pub step_size: f64,
    
    /// Enable adaptive parameters
    pub adaptive: bool,
}

impl Default for AcoParameters {
    fn default() -> Self {
        Self {
            num_ants: 50,
            alpha: 1.0,
            beta: 2.0,
            evaporation_rate: 0.1,
            pheromone_deposit: 1.0,
            initial_pheromone: 0.1,
            variant: AcoVariant::AntSystem,
            local_search_probability: 0.1,
            num_elitist_ants: 5,
            max_pheromone: 1.0,
            min_pheromone: 0.01,
            q0: 0.9,
            xi: 0.1,
            step_size: 0.1,
            adaptive: false,
        }
    }
}

impl AcoParameters {
    /// Validate parameters
    pub fn validate(&self) -> Result<(), SwarmError> {
        validate_parameter!(self.num_ants, "num_ants", 2, 1000);
        validate_parameter!(self.alpha, "alpha", 0.0, 10.0);
        validate_parameter!(self.beta, "beta", 0.0, 10.0);
        validate_parameter!(self.evaporation_rate, "evaporation_rate", 0.0, 1.0);
        validate_parameter!(self.pheromone_deposit, "pheromone_deposit", 0.0, 10.0);
        validate_parameter!(self.initial_pheromone, "initial_pheromone", 0.0, 1.0);
        validate_parameter!(self.local_search_probability, "local_search_probability", 0.0, 1.0);
        validate_parameter!(self.step_size, "step_size", 0.01, 1.0);
        
        if self.max_pheromone < self.min_pheromone {
            return Err(SwarmError::parameter("max_pheromone must be >= min_pheromone"));
        }
        
        Ok(())
    }
    
    /// Create builder for ACO parameters
    pub fn builder() -> AcoParametersBuilder {
        AcoParametersBuilder::new()
    }
}

/// Builder for ACO parameters
pub struct AcoParametersBuilder {
    params: AcoParameters,
}

impl AcoParametersBuilder {
    pub fn new() -> Self {
        Self {
            params: AcoParameters::default(),
        }
    }
    
    pub fn num_ants(mut self, count: usize) -> Self {
        self.params.num_ants = count;
        self
    }
    
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.params.alpha = alpha;
        self
    }
    
    pub fn beta(mut self, beta: f64) -> Self {
        self.params.beta = beta;
        self
    }
    
    pub fn evaporation_rate(mut self, rate: f64) -> Self {
        self.params.evaporation_rate = rate;
        self
    }
    
    pub fn variant(mut self, variant: AcoVariant) -> Self {
        self.params.variant = variant;
        self
    }
    
    pub fn adaptive(mut self, adaptive: bool) -> Self {
        self.params.adaptive = adaptive;
        self
    }
    
    pub fn build(self) -> Result<AcoParameters, SwarmError> {
        self.params.validate()?;
        Ok(self.params)
    }
}

/// Artificial ant representation
#[derive(Debug, Clone)]
pub struct Ant {
    /// Current position
    position: Position,
    
    /// Fitness value
    fitness: f64,
    
    /// Path taken by the ant
    path: Vec<Position>,
    
    /// Path quality
    path_quality: f64,
    
    /// Local pheromone level
    local_pheromone: f64,
}

impl Ant {
    /// Create a new ant at random position
    pub fn new(dimensions: usize, bounds: (f64, f64)) -> Self {
        let mut rng = thread_rng();
        let position = Position::from_fn(dimensions, |_, _| {
            rng.gen_range(bounds.0..=bounds.1)
        });
        
        Self {
            position,
            fitness: f64::INFINITY,
            path: Vec::new(),
            path_quality: 0.0,
            local_pheromone: 0.0,
        }
    }
    
    /// Construct solution using ACO probabilistic rules
    pub fn construct_solution(
        &mut self,
        pheromone_matrix: &Array2<f64>,
        heuristic_matrix: &Array2<f64>,
        alpha: f64,
        beta: f64,
        step_size: f64,
        bounds: (f64, f64),
    ) -> Result<(), SwarmError> {
        let mut rng = thread_rng();
        let dimensions = self.position.len();
        
        // Clear previous path
        self.path.clear();
        self.path.push(self.position.clone());
        
        // Construct solution step by step
        for step in 0..dimensions {
            let current_idx = self.discretize_position(&self.position, &bounds)?;
            
            // Calculate transition probabilities
            let probabilities = self.calculate_transition_probabilities(
                current_idx,
                pheromone_matrix,
                heuristic_matrix,
                alpha,
                beta,
            );
            
            // Select next position based on probabilities
            let next_idx = self.select_next_position(&probabilities, &mut rng);
            let next_position = self.continuous_position_from_index(next_idx, &bounds, step_size);
            
            // Move to next position
            self.position = next_position;
            self.path.push(self.position.clone());
        }
        
        Ok(())
    }
    
    /// Calculate transition probabilities
    fn calculate_transition_probabilities(
        &self,
        current_idx: usize,
        pheromone_matrix: &Array2<f64>,
        heuristic_matrix: &Array2<f64>,
        alpha: f64,
        beta: f64,
    ) -> Vec<f64> {
        let n = pheromone_matrix.nrows();
        let mut probabilities = vec![0.0; n];
        let mut total = 0.0;
        
        for j in 0..n {
            let pheromone = pheromone_matrix[[current_idx, j]];
            let heuristic = heuristic_matrix[[current_idx, j]];
            
            if heuristic > 0.0 {
                let prob = pheromone.powf(alpha) * heuristic.powf(beta);
                probabilities[j] = prob;
                total += prob;
            }
        }
        
        // Normalize probabilities
        if total > 0.0 {
            for prob in &mut probabilities {
                *prob /= total;
            }
        }
        
        probabilities
    }
    
    /// Select next position using roulette wheel selection
    fn select_next_position(&self, probabilities: &[f64], rng: &mut ThreadRng) -> usize {
        let r: f64 = rng.gen();
        let mut cumulative = 0.0;
        
        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative += prob;
            if r <= cumulative {
                return i;
            }
        }
        
        // Fallback to random selection
        rng.gen_range(0..probabilities.len())
    }
    
    /// Discretize continuous position for pheromone matrix indexing
    fn discretize_position(&self, position: &Position, bounds: &(f64, f64)) -> Result<usize, SwarmError> {
        let normalized = (position[0] - bounds.0) / (bounds.1 - bounds.0);
        let discretized = (normalized * 100.0).clamp(0.0, 99.0) as usize;
        Ok(discretized)
    }
    
    /// Convert discrete index back to continuous position
    fn continuous_position_from_index(&self, index: usize, bounds: &(f64, f64), step_size: f64) -> Position {
        let mut rng = thread_rng();
        let base_value = bounds.0 + (index as f64 / 100.0) * (bounds.1 - bounds.0);
        
        let dimensions = self.position.len();
        Position::from_fn(dimensions, |i, _| {
            if i == 0 {
                base_value + rng.gen_range(-step_size..step_size)
            } else {
                self.position[i] + rng.gen_range(-step_size..step_size)
            }
        })
    }
    
    /// Get path length (for traveling salesman-like problems)
    pub fn path_length(&self) -> f64 {
        if self.path.len() < 2 {
            return 0.0;
        }
        
        let mut length = 0.0;
        for i in 1..self.path.len() {
            length += (&self.path[i] - &self.path[i-1]).norm();
        }
        length
    }
    
    /// Update path quality based on fitness
    pub fn update_path_quality(&mut self, fitness: f64) {
        self.fitness = fitness;
        self.path_quality = if fitness.is_finite() && fitness > 0.0 {
            1.0 / fitness
        } else {
            0.0
        };
    }
}

impl Individual for Ant {
    fn position(&self) -> &Position {
        &self.position
    }
    
    fn position_mut(&mut self) -> &mut Position {
        &mut self.position
    }
    
    fn fitness(&self) -> &f64 {
        &self.fitness
    }
    
    fn set_fitness(&mut self, fitness: f64) {
        self.fitness = fitness;
        self.update_path_quality(fitness);
    }
    
    fn update_position(&mut self, new_position: Position) {
        self.position = new_position;
    }
}

/// Ant Colony Optimization algorithm
#[derive(Debug, Clone)]
pub struct AntColonyOptimization {
    /// Algorithm parameters
    parameters: AcoParameters,
    
    /// Colony of ants
    colony: Population<Ant>,
    
    /// Pheromone matrix (discretized space)
    pheromone_matrix: Array2<f64>,
    
    /// Heuristic information matrix
    heuristic_matrix: Array2<f64>,
    
    /// Best solution found
    best_solution: Option<Arc<Ant>>,
    
    /// Best fitness
    best_fitness: f64,
    
    /// Current iteration
    iteration: usize,
    
    /// Optimization problem
    problem: Option<Arc<OptimizationProblem>>,
    
    /// Performance metrics
    metrics: AlgorithmMetrics,
    
    /// Matrix size for discretization
    matrix_size: usize,
}

impl AntColonyOptimization {
    /// Create a new ACO algorithm with default parameters
    pub fn new() -> Self {
        Self::with_parameters(AcoParameters::default())
    }
    
    /// Create ACO with specific parameters
    pub fn with_parameters(parameters: AcoParameters) -> Self {
        let matrix_size = 100; // Default discretization
        
        Self {
            parameters,
            colony: Population::new(),
            pheromone_matrix: Array2::zeros((matrix_size, matrix_size)),
            heuristic_matrix: Array2::zeros((matrix_size, matrix_size)),
            best_solution: None,
            best_fitness: f64::INFINITY,
            iteration: 0,
            problem: None,
            metrics: AlgorithmMetrics::default(),
            matrix_size,
        }
    }
    
    /// Builder pattern for ACO construction
    pub fn builder() -> AcoBuilder {
        AcoBuilder::new()
    }
    
    /// Initialize pheromone matrix
    fn initialize_pheromone_matrix(&mut self) {
        self.pheromone_matrix.fill(self.parameters.initial_pheromone);
    }
    
    /// Initialize heuristic matrix
    fn initialize_heuristic_matrix(&mut self) -> Result<(), SwarmError> {
        let problem = self.problem.as_ref()
            .ok_or_else(|| SwarmError::initialization("Problem not set"))?;
        
        // Calculate heuristic information based on problem structure
        for i in 0..self.matrix_size {
            for j in 0..self.matrix_size {
                // Simple heuristic: inverse distance or problem-specific
                let heuristic = if i != j {
                    1.0 / (1.0 + (i as f64 - j as f64).abs())
                } else {
                    0.0
                };
                self.heuristic_matrix[[i, j]] = heuristic;
            }
        }
        
        Ok(())
    }
    
    /// Construct solutions for all ants
    async fn construct_solutions(&mut self) -> Result<(), SwarmError> {
        let problem = self.problem.as_ref().unwrap();
        let bounds = (problem.lower_bounds.min(), problem.upper_bounds.max());
        
        for ant in self.colony.iter_mut() {
            ant.construct_solution(
                &self.pheromone_matrix,
                &self.heuristic_matrix,
                self.parameters.alpha,
                self.parameters.beta,
                self.parameters.step_size,
                bounds,
            )?;
        }
        
        Ok(())
    }
    
    /// Evaluate all ant solutions
    async fn evaluate_solutions(&mut self) -> Result<(), SwarmError> {
        let problem = self.problem.as_ref().unwrap();
        
        // Evaluate in parallel
        let positions: Vec<Position> = self.colony.iter().map(|ant| ant.position().clone()).collect();
        let fitnesses = problem.evaluate_parallel(&positions);
        
        for (ant, fitness) in self.colony.iter_mut().zip(fitnesses.iter()) {
            ant.set_fitness(*fitness);
            
            // Update global best
            if *fitness < self.best_fitness {
                self.best_fitness = *fitness;
                self.best_solution = Some(Arc::new(ant.clone()));
            }
        }
        
        self.metrics.evaluations += fitnesses.len();
        Ok(())
    }
    
    /// Update pheromone trails
    fn update_pheromones(&mut self) -> Result<(), SwarmError> {
        match self.parameters.variant {
            AcoVariant::AntSystem => self.update_pheromones_ant_system(),
            AcoVariant::ElitistAntSystem => self.update_pheromones_elitist(),
            AcoVariant::RankBasedAntSystem => self.update_pheromones_rank_based(),
            AcoVariant::MaxMinAntSystem => self.update_pheromones_max_min(),
            AcoVariant::AntColonySystem => self.update_pheromones_ant_colony_system(),
        }
    }
    
    /// Standard Ant System pheromone update
    fn update_pheromones_ant_system(&mut self) -> Result<(), SwarmError> {
        // Evaporation
        self.pheromone_matrix.mapv_inplace(|x| x * (1.0 - self.parameters.evaporation_rate));
        
        // Pheromone deposit by all ants
        for ant in self.colony.iter() {
            if ant.path_quality > 0.0 {
                self.deposit_pheromone_on_path(ant, self.parameters.pheromone_deposit * ant.path_quality)?;
            }
        }
        
        Ok(())
    }
    
    /// Elitist Ant System pheromone update
    fn update_pheromones_elitist(&mut self) -> Result<(), SwarmError> {
        // Standard update
        self.update_pheromones_ant_system()?;
        
        // Additional deposit by best ant
        if let Some(ref best_ant) = self.best_solution {
            let elite_deposit = self.parameters.pheromone_deposit * 
                              self.parameters.num_elitist_ants as f64 * 
                              best_ant.path_quality;
            self.deposit_pheromone_on_path(best_ant, elite_deposit)?;
        }
        
        Ok(())
    }
    
    /// Rank-based Ant System pheromone update
    fn update_pheromones_rank_based(&mut self) -> Result<(), SwarmError> {
        // Evaporation
        self.pheromone_matrix.mapv_inplace(|x| x * (1.0 - self.parameters.evaporation_rate));
        
        // Sort ants by fitness
        let mut ranked_ants: Vec<_> = self.colony.iter().collect();
        ranked_ants.sort_by(|a, b| a.fitness().partial_cmp(b.fitness()).unwrap());
        
        // Deposit pheromone based on rank
        for (rank, ant) in ranked_ants.iter().take(self.parameters.num_elitist_ants).enumerate() {
            let weight = (self.parameters.num_elitist_ants - rank) as f64;
            let deposit = self.parameters.pheromone_deposit * weight * ant.path_quality;
            self.deposit_pheromone_on_path(ant, deposit)?;
        }
        
        Ok(())
    }
    
    /// Max-Min Ant System pheromone update
    fn update_pheromones_max_min(&mut self) -> Result<(), SwarmError> {
        // Evaporation
        self.pheromone_matrix.mapv_inplace(|x| x * (1.0 - self.parameters.evaporation_rate));
        
        // Only best ant deposits pheromone
        if let Some(ref best_ant) = self.best_solution {
            let deposit = self.parameters.pheromone_deposit * best_ant.path_quality;
            self.deposit_pheromone_on_path(best_ant, deposit)?;
        }
        
        // Enforce pheromone bounds
        self.pheromone_matrix.mapv_inplace(|x| {
            x.clamp(self.parameters.min_pheromone, self.parameters.max_pheromone)
        });
        
        Ok(())
    }
    
    /// Ant Colony System pheromone update
    fn update_pheromones_ant_colony_system(&mut self) -> Result<(), SwarmError> {
        // Global pheromone update - only best ant
        if let Some(ref best_ant) = self.best_solution {
            let deposit = (1.0 - self.parameters.evaporation_rate) * best_ant.path_quality;
            self.deposit_pheromone_on_path(best_ant, deposit)?;
        }
        
        Ok(())
    }
    
    /// Deposit pheromone on ant's path
    fn deposit_pheromone_on_path(&mut self, ant: &Ant, amount: f64) -> Result<(), SwarmError> {
        if ant.path.len() < 2 {
            return Ok(());
        }
        
        let problem = self.problem.as_ref().unwrap();
        let bounds = (problem.lower_bounds.min(), problem.upper_bounds.max());
        
        for i in 1..ant.path.len() {
            let prev_idx = self.discretize_position(&ant.path[i-1], bounds)?;
            let curr_idx = self.discretize_position(&ant.path[i], bounds)?;
            
            self.pheromone_matrix[[prev_idx, curr_idx]] += amount;
            self.pheromone_matrix[[curr_idx, prev_idx]] += amount; // Symmetric
        }
        
        Ok(())
    }
    
    /// Discretize position for matrix indexing
    fn discretize_position(&self, position: &Position, bounds: (f64, f64)) -> Result<usize, SwarmError> {
        let normalized = (position[0] - bounds.0) / (bounds.1 - bounds.0);
        let discretized = (normalized * (self.matrix_size - 1) as f64).clamp(0.0, (self.matrix_size - 1) as f64) as usize;
        Ok(discretized)
    }
    
    /// Perform local search
    fn local_search(&mut self) -> Result<(), SwarmError> {
        let mut rng = thread_rng();
        
        for ant in self.colony.iter_mut() {
            if rng.gen::<f64>() < self.parameters.local_search_probability {
                // Simple local search: small random perturbation
                let old_position = ant.position().clone();
                let old_fitness = *ant.fitness();
                
                // Perturb position
                let perturbation = Position::from_fn(old_position.len(), |_, _| {
                    rng.gen_range(-0.1..0.1)
                });
                let new_position = &old_position + perturbation;
                
                // Evaluate new position
                if let Some(ref problem) = self.problem {
                    let new_fitness = problem.evaluate(&new_position);
                    
                    // Accept if better
                    if new_fitness < old_fitness {
                        ant.update_position(new_position);
                        ant.set_fitness(new_fitness);
                        
                        if new_fitness < self.best_fitness {
                            self.best_fitness = new_fitness;
                            self.best_solution = Some(Arc::new(ant.clone()));
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Calculate population diversity
    fn calculate_diversity(&self) -> f64 {
        self.colony.diversity().unwrap_or(0.0)
    }
    
    /// Check convergence
    fn has_converged(&self) -> bool {
        let diversity = self.calculate_diversity();
        diversity < 1e-6 || self.iteration > 1000
    }
}

#[async_trait]
impl SwarmAlgorithm for AntColonyOptimization {
    type Individual = Ant;
    type Fitness = f64;
    type Parameters = AcoParameters;
    
    async fn initialize(&mut self, problem: OptimizationProblem) -> Result<(), SwarmError> {
        self.parameters.validate()?;
        
        // Store problem
        self.problem = Some(Arc::new(problem));
        let problem_ref = self.problem.as_ref().unwrap();
        
        // Initialize colony
        self.colony = Population::with_capacity(self.parameters.num_ants);
        let bounds = (problem_ref.lower_bounds.min(), problem_ref.upper_bounds.max());
        
        for _ in 0..self.parameters.num_ants {
            let ant = Ant::new(problem_ref.dimensions, bounds);
            self.colony.add(ant);
        }
        
        // Initialize matrices
        self.initialize_pheromone_matrix();
        self.initialize_heuristic_matrix()?;
        
        // Initial evaluation
        self.evaluate_solutions().await?;
        
        // Initialize metrics
        self.metrics = AlgorithmMetrics {
            iteration: 0,
            best_fitness: Some(self.best_fitness),
            average_fitness: self.colony.average_fitness(),
            diversity: Some(self.calculate_diversity()),
            convergence_rate: None,
            evaluations: self.parameters.num_ants,
            time_per_iteration: None,
            memory_usage: None,
        };
        
        tracing::info!(
            "ACO initialized with {} ants, variant: {:?}",
            self.parameters.num_ants,
            self.parameters.variant
        );
        
        Ok(())
    }
    
    async fn step(&mut self) -> Result<(), SwarmError> {
        let start_time = std::time::Instant::now();
        
        self.iteration += 1;
        
        // Construct solutions
        self.construct_solutions().await?;
        
        // Evaluate solutions
        self.evaluate_solutions().await?;
        
        // Update pheromones
        self.update_pheromones()?;
        
        // Local search
        self.local_search()?;
        
        // Update metrics
        self.metrics.iteration = self.iteration;
        self.metrics.best_fitness = Some(self.best_fitness);
        self.metrics.average_fitness = self.colony.average_fitness();
        self.metrics.diversity = Some(self.calculate_diversity());
        self.metrics.time_per_iteration = Some(start_time.elapsed().as_micros() as u64);
        
        Ok(())
    }
    
    fn get_best_individual(&self) -> Option<&Self::Individual> {
        self.best_solution.as_ref().map(|arc| arc.as_ref())
    }
    
    fn get_population(&self) -> &Population<Self::Individual> {
        &self.colony
    }
    
    fn get_population_mut(&mut self) -> &mut Population<Self::Individual> {
        &mut self.colony
    }
    
    fn has_converged(&self) -> bool {
        self.has_converged()
    }
    
    fn name(&self) -> &'static str {
        "AntColonyOptimization"
    }
    
    fn parameters(&self) -> &Self::Parameters {
        &self.parameters
    }
    
    fn update_parameters(&mut self, params: Self::Parameters) {
        self.parameters = params;
    }
    
    fn metrics(&self) -> AlgorithmMetrics {
        self.metrics.clone()
    }
    
    async fn reset(&mut self) -> Result<(), SwarmError> {
        self.colony = Population::new();
        self.best_solution = None;
        self.best_fitness = f64::INFINITY;
        self.iteration = 0;
        self.metrics = AlgorithmMetrics::default();
        self.initialize_pheromone_matrix();
        Ok(())
    }
    
    fn clone_algorithm(&self) -> Box<dyn SwarmAlgorithm<
        Individual = Self::Individual,
        Fitness = Self::Fitness,
        Parameters = Self::Parameters
    >> {
        Box::new(self.clone())
    }
}

impl AdaptiveAlgorithm for AntColonyOptimization {
    fn adapt_parameters(&mut self, performance_metrics: &AlgorithmMetrics) {
        if !self.parameters.adaptive {
            return;
        }
        
        // Adapt evaporation rate based on diversity
        if let Some(diversity) = performance_metrics.diversity {
            if diversity < 0.1 {
                // Low diversity - increase evaporation to encourage exploration
                self.parameters.evaporation_rate = 
                    (self.parameters.evaporation_rate + 0.01).min(0.5);
            } else if diversity > 0.5 {
                // High diversity - decrease evaporation to encourage exploitation
                self.parameters.evaporation_rate = 
                    (self.parameters.evaporation_rate - 0.01).max(0.01);
            }
        }
        
        // Adapt alpha and beta based on convergence
        if self.iteration > 100 {
            let progress = self.iteration as f64 / 1000.0;
            self.parameters.alpha = 1.0 + progress; // Increase pheromone importance
            self.parameters.beta = 2.0 - progress;  // Decrease heuristic importance
        }
    }
    
    fn adaptation_strategy(&self) -> AdaptationStrategy {
        AdaptationStrategy::Feedback { sensitivity: 0.1 }
    }
}

impl ParallelAlgorithm for AntColonyOptimization {
    async fn parallel_step(&mut self, _thread_count: usize) -> Result<(), SwarmError> {
        // ACO can be parallelized at the ant level
        // For now, use the standard step implementation
        self.step().await
    }
    
    fn optimal_thread_count(&self) -> usize {
        (self.parameters.num_ants / 10).max(1).min(num_cpus::get())
    }
}

/// Builder for ACO algorithm
pub struct AcoBuilder {
    parameters: AcoParameters,
}

impl AcoBuilder {
    pub fn new() -> Self {
        Self {
            parameters: AcoParameters::default(),
        }
    }
    
    pub fn num_ants(mut self, count: usize) -> Self {
        self.parameters.num_ants = count;
        self
    }
    
    pub fn alpha(mut self, alpha: f64) -> Self {
        self.parameters.alpha = alpha;
        self
    }
    
    pub fn beta(mut self, beta: f64) -> Self {
        self.parameters.beta = beta;
        self
    }
    
    pub fn evaporation_rate(mut self, rate: f64) -> Self {
        self.parameters.evaporation_rate = rate;
        self
    }
    
    pub fn variant(mut self, variant: AcoVariant) -> Self {
        self.parameters.variant = variant;
        self
    }
    
    pub fn adaptive(mut self, adaptive: bool) -> Self {
        self.parameters.adaptive = adaptive;
        self
    }
    
    pub fn build(self) -> Result<AntColonyOptimization, SwarmError> {
        self.parameters.validate()?;
        Ok(AntColonyOptimization::with_parameters(self.parameters))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::OptimizationProblem;
    use approx::assert_relative_eq;
    
    #[tokio::test]
    async fn test_aco_initialization() {
        let mut aco = AntColonyOptimization::new();
        
        let problem = OptimizationProblem::new()
            .dimensions(2)
            .bounds(-10.0, 10.0)
            .objective(|x| x[0].powi(2) + x[1].powi(2))
            .build()
            .unwrap();
        
        assert!(aco.initialize(problem).await.is_ok());
        assert_eq!(aco.colony.size(), 50);
        assert!(aco.best_solution.is_some());
    }
    
    #[tokio::test]
    async fn test_aco_optimization() {
        let mut aco = AntColonyOptimization::builder()
            .num_ants(20)
            .variant(AcoVariant::AntSystem)
            .build()
            .unwrap();
        
        let problem = OptimizationProblem::new()
            .dimensions(2)
            .bounds(-5.0, 5.0)
            .objective(|x| x[0].powi(2) + x[1].powi(2))
            .build()
            .unwrap();
        
        let result = aco.optimize(30).await.unwrap();
        
        // Should find reasonable solution
        assert!(result.best_fitness < 50.0);
        assert!(result.iterations <= 30);
        assert_eq!(result.algorithm_name, "AntColonyOptimization");
    }
    
    #[test]
    fn test_aco_parameters() {
        let params = AcoParameters::builder()
            .num_ants(100)
            .alpha(1.5)
            .beta(2.5)
            .evaporation_rate(0.2)
            .variant(AcoVariant::MaxMinAntSystem)
            .adaptive(true)
            .build()
            .unwrap();
        
        assert_eq!(params.num_ants, 100);
        assert_relative_eq!(params.alpha, 1.5);
        assert_relative_eq!(params.beta, 2.5);
        assert!(params.adaptive);
        assert!(matches!(params.variant, AcoVariant::MaxMinAntSystem));
    }
    
    #[test]
    fn test_parameter_validation() {
        let result = AcoParameters::builder()
            .num_ants(0) // Invalid
            .build();
        
        assert!(result.is_err());
    }
    
    #[test]
    fn test_ant_creation() {
        let ant = Ant::new(3, (-5.0, 5.0));
        
        assert_eq!(ant.position().len(), 3);
        assert_eq!(*ant.fitness(), f64::INFINITY);
        assert!(ant.path.is_empty());
    }
}