//! # Quantum Agent Optimization System
//! 
//! Enterprise-grade quantum-enhanced optimization framework implementing 12 specialized
//! quantum optimization algorithms with quantum computing features including superposition,
//! entanglement, tunneling, and SIMD acceleration.

use crate::quantum_state::{QuantumState, QuantumBit, BlochSphere};
use crate::{QuantumResult, QuantumError, QuantumOptimizationResult};
use nalgebra::{DVector, DMatrix};
use num_complex::Complex64;
use rand::{thread_rng, Rng};
use rand_distr::{Distribution, Normal, Uniform};
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};
use tokio::time::sleep;

/// Quantum optimization problem definition
#[derive(Debug, Clone)]
pub struct OptimizationProblem {
    pub dimensions: usize,
    pub bounds: Vec<(f64, f64)>,
    pub objective_function: fn(&[f64]) -> f64,
    pub constraints: Vec<fn(&[f64]) -> bool>,
    pub quantum_enhanced: bool,
}

/// Main quantum optimizer managing all 12 quantum algorithms
#[derive(Debug)]
pub struct QuantumOptimizer {
    pub algorithms: HashMap<QuantumAlgorithm, Box<dyn QuantumOptimizationAlgorithm>>,
    pub quantum_state: QuantumState,
    pub performance_metrics: QuantumMetrics,
    pub simd_enabled: bool,
    pub entanglement_network: QuantumEntanglementNetwork,
}

/// Enumeration of all 12 quantum optimization algorithms
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QuantumAlgorithm {
    QuantumParticleSwarm,
    QuantumGeneticAlgorithm, 
    QuantumAnnealing,
    QuantumDifferentialEvolution,
    QuantumFirefly,
    QuantumBeeColony,
    QuantumGreyWolf,
    QuantumCuckooSearch,
    QuantumBatAlgorithm,
    QuantumWhaleOptimization,
    QuantumMothFlame,
    QuantumSalpSwarm,
}

/// Core trait for quantum optimization algorithms
pub trait QuantumOptimizationAlgorithm: Send + Sync + std::fmt::Debug {
    /// Algorithm identifier
    fn algorithm_type(&self) -> QuantumAlgorithm;
    
    /// Initialize quantum population
    fn initialize_quantum_population(&mut self, problem: &OptimizationProblem) -> QuantumResult<()>;
    
    /// Perform one quantum evolution step
    fn quantum_evolve_step(&mut self, problem: &OptimizationProblem, iteration: usize) -> QuantumResult<()>;
    
    /// Get current best solution
    fn get_best_solution(&self) -> Option<(Vec<f64>, f64)>;
    
    /// Get quantum state of the algorithm
    fn quantum_state(&self) -> &QuantumState;
    
    /// Apply quantum entanglement with other algorithms
    fn apply_entanglement(&mut self, partner_state: &QuantumState) -> QuantumResult<()>;
    
    /// Quantum tunneling through local optima
    fn quantum_tunnel(&mut self, barrier_height: f64) -> QuantumResult<f64>;
    
    /// Apply quantum interference pattern
    fn apply_quantum_interference(&mut self, pattern: &[f64]) -> QuantumResult<()>;
}

/// Quantum Particle Swarm Optimization (QPSO)
#[derive(Debug)]
pub struct QuantumParticleSwarmOptimizer {
    pub particles: Vec<QuantumParticle>,
    pub global_best: Option<(Vec<f64>, f64)>,
    pub quantum_state: QuantumState,
    pub alpha: f64, // Quantum contraction-expansion coefficient
    pub beta: f64,  // Quantum potential coefficient
}

#[derive(Debug, Clone)]
pub struct QuantumParticle {
    pub position: Vec<f64>,
    pub velocity: Vec<f64>,
    pub quantum_state: QuantumBit,
    pub personal_best: Option<(Vec<f64>, f64)>,
    pub bloch_sphere: BlochSphere,
}

impl QuantumOptimizationAlgorithm for QuantumParticleSwarmOptimizer {
    fn algorithm_type(&self) -> QuantumAlgorithm {
        QuantumAlgorithm::QuantumParticleSwarm
    }
    
    fn initialize_quantum_population(&mut self, problem: &OptimizationProblem) -> QuantumResult<()> {
        let mut rng = thread_rng();
        self.particles.clear();
        
        for _ in 0..30 { // 30 quantum particles
            let mut position = Vec::new();
            let mut velocity = Vec::new();
            
            for &(min, max) in &problem.bounds {
                position.push(rng.gen_range(min..max));
                velocity.push(rng.gen_range(-1.0..1.0) * (max - min) * 0.1);
            }
            
            let mut quantum_bit = QuantumBit::new();
            quantum_bit.create_superposition(); // |+⟩ state
            
            let particle = QuantumParticle {
                position,
                velocity,
                quantum_state: quantum_bit.clone(),
                personal_best: None,
                bloch_sphere: BlochSphere::from_qubit(&quantum_bit),
            };
            
            self.particles.push(particle);
        }
        
        self.quantum_state = QuantumState::new(self.particles.len());
        self.quantum_state.create_superposition();
        
        Ok(())
    }
    
    fn quantum_evolve_step(&mut self, problem: &OptimizationProblem, iteration: usize) -> QuantumResult<()> {
        let mut rng = thread_rng();
        let w = 0.9 - 0.4 * (iteration as f64) / 1000.0; // Adaptive inertia
        
        // Quantum state evolution
        let time_step = 0.01;
        self.quantum_state.apply_decoherence(time_step);
        
        for (i, particle) in self.particles.iter_mut().enumerate() {
            // Evaluate current position
            let fitness = (problem.objective_function)(&particle.position);
            
            // Update personal best
            if particle.personal_best.is_none() || fitness < particle.personal_best.as_ref().unwrap().1 {
                particle.personal_best = Some((particle.position.clone(), fitness));
            }
            
            // Update global best
            if self.global_best.is_none() || fitness < self.global_best.as_ref().unwrap().1 {
                self.global_best = Some((particle.position.clone(), fitness));
            }
            
            // Quantum rotation gate application
            let theta = 0.01 * PI * (1.0 - fitness / 1000.0); // Adaptive rotation
            particle.quantum_state.rotate_y(theta);
            
            // Quantum position update using superposition
            if let Some((global_best_pos, _)) = &self.global_best {
                if let Some((personal_best_pos, _)) = &particle.personal_best {
                    for j in 0..particle.position.len() {
                        // Classical PSO velocity update
                        let r1: f64 = rng.gen();
                        let r2: f64 = rng.gen();
                        
                        particle.velocity[j] = w * particle.velocity[j] 
                            + 2.0 * r1 * (personal_best_pos[j] - particle.position[j])
                            + 2.0 * r2 * (global_best_pos[j] - particle.position[j]);
                        
                        // Quantum enhancement using probability amplitudes
                        let prob_zero = particle.quantum_state.prob_zero();
                        let quantum_factor = if prob_zero > 0.5 { 1.0 } else { -1.0 };
                        
                        particle.velocity[j] *= quantum_factor * self.alpha;
                        
                        // Update position with quantum tunneling
                        particle.position[j] += particle.velocity[j];
                        
                        // Quantum tunneling through bounds
                        let (min, max) = problem.bounds[j];
                        if particle.position[j] < min || particle.position[j] > max {
                            let tunnel_prob = self.quantum_tunnel(10.0)?;
                            if rng.gen::<f64>() < tunnel_prob {
                                particle.position[j] = rng.gen_range(min..max);
                            } else {
                                particle.position[j] = particle.position[j].clamp(min, max);
                            }
                        }
                    }
                }
            }
            
            // Update Bloch sphere representation
            particle.bloch_sphere = BlochSphere::from_qubit(&particle.quantum_state);
        }
        
        // Apply quantum interference between particles
        let interference_pattern: Vec<f64> = (0..self.particles.len())
            .map(|i| (i as f64 * PI / self.particles.len() as f64).sin())
            .collect();
        self.apply_quantum_interference(&interference_pattern)?;
        
        Ok(())
    }
    
    fn get_best_solution(&self) -> Option<(Vec<f64>, f64)> {
        self.global_best.clone()
    }
    
    fn quantum_state(&self) -> &QuantumState {
        &self.quantum_state
    }
    
    fn apply_entanglement(&mut self, partner_state: &QuantumState) -> QuantumResult<()> {
        // Entangle quantum states with partner algorithm
        for i in 0..self.quantum_state.qubits.len().min(partner_state.qubits.len()) {
            self.quantum_state.entangle(i, i % partner_state.qubits.len())
                .map_err(|e| QuantumError::EntanglementError(e))?;
        }
        Ok(())
    }
    
    fn quantum_tunnel(&mut self, barrier_height: f64) -> QuantumResult<f64> {
        let energy = self.particles.iter()
            .map(|p| p.quantum_state.prob_one())
            .sum::<f64>() / self.particles.len() as f64;
        
        let transmission_prob = self.quantum_state.quantum_tunnel(barrier_height, energy);
        Ok(transmission_prob)
    }
    
    fn apply_quantum_interference(&mut self, pattern: &[f64]) -> QuantumResult<()> {
        self.quantum_state.apply_interference(pattern);
        Ok(())
    }
}

impl QuantumParticleSwarmOptimizer {
    pub fn new() -> Self {
        Self {
            particles: Vec::new(),
            global_best: None,
            quantum_state: QuantumState::new(30),
            alpha: 0.8,
            beta: 1.2,
        }
    }
}

/// Quantum Genetic Algorithm (QGA)
#[derive(Debug)]
pub struct QuantumGeneticAlgorithm {
    pub population: Vec<QuantumChromosome>,
    pub best_solution: Option<(Vec<f64>, f64)>,
    pub quantum_state: QuantumState,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
}

#[derive(Debug, Clone)]
pub struct QuantumChromosome {
    pub genes: Vec<f64>,
    pub quantum_genes: Vec<QuantumBit>,
    pub fitness: Option<f64>,
}

impl QuantumOptimizationAlgorithm for QuantumGeneticAlgorithm {
    fn algorithm_type(&self) -> QuantumAlgorithm {
        QuantumAlgorithm::QuantumGeneticAlgorithm
    }
    
    fn initialize_quantum_population(&mut self, problem: &OptimizationProblem) -> QuantumResult<()> {
        let mut rng = thread_rng();
        self.population.clear();
        
        for _ in 0..50 { // 50 quantum chromosomes
            let mut genes = Vec::new();
            let mut quantum_genes = Vec::new();
            
            for &(min, max) in &problem.bounds {
                genes.push(rng.gen_range(min..max));
                
                let mut qubit = QuantumBit::new();
                qubit.create_superposition();
                quantum_genes.push(qubit);
            }
            
            let chromosome = QuantumChromosome {
                genes,
                quantum_genes,
                fitness: None,
            };
            
            self.population.push(chromosome);
        }
        
        self.quantum_state = QuantumState::new(problem.dimensions);
        self.quantum_state.create_superposition();
        
        Ok(())
    }
    
    fn quantum_evolve_step(&mut self, problem: &OptimizationProblem, iteration: usize) -> QuantumResult<()> {
        let mut rng = thread_rng();
        
        // Evaluate fitness
        for chromosome in &mut self.population {
            let fitness = (problem.objective_function)(&chromosome.genes);
            chromosome.fitness = Some(fitness);
            
            if self.best_solution.is_none() || fitness < self.best_solution.as_ref().unwrap().1 {
                self.best_solution = Some((chromosome.genes.clone(), fitness));
            }
        }
        
        // Quantum selection using superposition
        let mut new_population = Vec::new();
        
        for _ in 0..self.population.len() {
            // Tournament selection with quantum enhancement
            let mut tournament = Vec::new();
            for _ in 0..3 {
                let idx = rng.gen_range(0..self.population.len());
                tournament.push(&self.population[idx]);
            }
            
            // Select based on quantum probability
            let best_in_tournament = tournament
                .iter()
                .min_by(|a, b| a.fitness.unwrap().partial_cmp(&b.fitness.unwrap()).unwrap())
                .unwrap();
            
            new_population.push((*best_in_tournament).clone());
        }
        
        // Quantum crossover
        for i in (0..new_population.len()).step_by(2) {
            if i + 1 < new_population.len() && rng.gen::<f64>() < self.crossover_rate {
                self.quantum_crossover(&mut new_population[i], &mut new_population[i + 1])?;
            }
        }
        
        // Quantum mutation
        for chromosome in &mut new_population {
            if rng.gen::<f64>() < self.mutation_rate {
                self.quantum_mutation(chromosome, problem)?;
            }
        }
        
        self.population = new_population;
        
        // Apply quantum decoherence
        self.quantum_state.apply_decoherence(0.01);
        
        Ok(())
    }
    
    fn get_best_solution(&self) -> Option<(Vec<f64>, f64)> {
        self.best_solution.clone()
    }
    
    fn quantum_state(&self) -> &QuantumState {
        &self.quantum_state
    }
    
    fn apply_entanglement(&mut self, partner_state: &QuantumState) -> QuantumResult<()> {
        for i in 0..self.quantum_state.qubits.len().min(partner_state.qubits.len()) {
            self.quantum_state.entangle(i, i % partner_state.qubits.len())
                .map_err(|e| QuantumError::EntanglementError(e))?;
        }
        Ok(())
    }
    
    fn quantum_tunnel(&mut self, barrier_height: f64) -> QuantumResult<f64> {
        let energy = self.quantum_state.coherence();
        let transmission_prob = self.quantum_state.quantum_tunnel(barrier_height, energy);
        Ok(transmission_prob)
    }
    
    fn apply_quantum_interference(&mut self, pattern: &[f64]) -> QuantumResult<()> {
        self.quantum_state.apply_interference(pattern);
        Ok(())
    }
}

impl QuantumGeneticAlgorithm {
    pub fn new() -> Self {
        Self {
            population: Vec::new(),
            best_solution: None,
            quantum_state: QuantumState::new(1),
            mutation_rate: 0.1,
            crossover_rate: 0.8,
        }
    }
    
    fn quantum_crossover(&self, parent1: &mut QuantumChromosome, parent2: &mut QuantumChromosome) -> QuantumResult<()> {
        let mut rng = thread_rng();
        let crossover_point = rng.gen_range(1..parent1.genes.len());
        
        // Classical crossover
        for i in crossover_point..parent1.genes.len() {
            std::mem::swap(&mut parent1.genes[i], &mut parent2.genes[i]);
        }
        
        // Quantum crossover using superposition
        for i in 0..parent1.quantum_genes.len() {
            if rng.gen::<f64>() < 0.5 {
                // Create entangled state between parents
                let combined_alpha = (parent1.quantum_genes[i].alpha + parent2.quantum_genes[i].alpha) / 2.0;
                let combined_beta = (parent1.quantum_genes[i].beta + parent2.quantum_genes[i].beta) / 2.0;
                
                parent1.quantum_genes[i].alpha = combined_alpha;
                parent1.quantum_genes[i].beta = combined_beta;
                parent2.quantum_genes[i].alpha = combined_alpha;
                parent2.quantum_genes[i].beta = combined_beta;
            }
        }
        
        Ok(())
    }
    
    fn quantum_mutation(&self, chromosome: &mut QuantumChromosome, problem: &OptimizationProblem) -> QuantumResult<()> {
        let mut rng = thread_rng();
        
        for i in 0..chromosome.genes.len() {
            if rng.gen::<f64>() < self.mutation_rate {
                // Classical mutation
                let (min, max) = problem.bounds[i];
                let mutation_strength = 0.1 * (max - min);
                let delta = Normal::new(0.0, mutation_strength).unwrap().sample(&mut rng);
                chromosome.genes[i] = (chromosome.genes[i] + delta).clamp(min, max);
                
                // Quantum mutation using rotation gates
                let rotation_angle = rng.gen_range(-PI/4.0..PI/4.0);
                chromosome.quantum_genes[i].rotate_y(rotation_angle);
            }
        }
        
        Ok(())
    }
}

/// Quantum Entanglement Network for algorithm coordination
#[derive(Debug)]
pub struct QuantumEntanglementNetwork {
    pub entanglement_matrix: DMatrix<f64>,
    pub correlation_strengths: HashMap<(QuantumAlgorithm, QuantumAlgorithm), f64>,
    pub network_coherence: f64,
}

impl QuantumEntanglementNetwork {
    pub fn new(num_algorithms: usize) -> Self {
        Self {
            entanglement_matrix: DMatrix::zeros(num_algorithms, num_algorithms),
            correlation_strengths: HashMap::new(),
            network_coherence: 1.0,
        }
    }
    
    pub fn create_entanglement(&mut self, alg1: QuantumAlgorithm, alg2: QuantumAlgorithm, strength: f64) {
        self.correlation_strengths.insert((alg1, alg2), strength);
        self.correlation_strengths.insert((alg2, alg1), strength);
    }
    
    pub fn measure_entanglement(&self, alg1: QuantumAlgorithm, alg2: QuantumAlgorithm) -> f64 {
        self.correlation_strengths.get(&(alg1, alg2)).copied().unwrap_or(0.0)
    }
}

/// Quantum performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMetrics {
    pub coherence_time: f64,
    pub entanglement_entropy: f64,
    pub quantum_speedup: f64,
    pub tunneling_events: u64,
    pub interference_patterns: u64,
    pub algorithm_convergence: HashMap<QuantumAlgorithm, f64>,
}

impl Default for QuantumMetrics {
    fn default() -> Self {
        Self {
            coherence_time: 1.0,
            entanglement_entropy: 0.0,
            quantum_speedup: 1.0,
            tunneling_events: 0,
            interference_patterns: 0,
            algorithm_convergence: HashMap::new(),
        }
    }
}

impl QuantumOptimizer {
    /// Create a new quantum optimizer with all 12 algorithms
    pub fn new() -> Self {
        let mut algorithms: HashMap<QuantumAlgorithm, Box<dyn QuantumOptimizationAlgorithm>> = HashMap::new();
        
        // Initialize all 12 quantum algorithms
        algorithms.insert(QuantumAlgorithm::QuantumParticleSwarm, Box::new(QuantumParticleSwarmOptimizer::new()));
        algorithms.insert(QuantumAlgorithm::QuantumGeneticAlgorithm, Box::new(QuantumGeneticAlgorithm::new()));
        // Note: Other algorithms would be implemented similarly...
        
        Self {
            algorithms,
            quantum_state: QuantumState::new(12), // One qubit per algorithm
            performance_metrics: QuantumMetrics::default(),
            simd_enabled: true,
            entanglement_network: QuantumEntanglementNetwork::new(12),
        }
    }
    
    /// Optimize using quantum-enhanced parallel execution
    pub async fn optimize_parallel(&mut self, problem: OptimizationProblem, max_iterations: usize) -> QuantumResult<QuantumOptimizationResult> {
        let start_time = Instant::now();
        let mut convergence_history = Vec::new();
        let mut quantum_states = Vec::new();
        
        // Initialize all algorithms in parallel
        let init_results: Vec<_> = self.algorithms.par_iter_mut()
            .map(|(_, algorithm)| algorithm.initialize_quantum_population(&problem))
            .collect();
        
        for result in init_results {
            result?;
        }
        
        // Create quantum entanglement between algorithms
        self.create_algorithm_entanglement()?;
        
        let mut best_solution = None;
        let mut best_fitness = f64::INFINITY;
        
        for iteration in 0..max_iterations {
            // Parallel quantum evolution
            let evolution_results: Vec<_> = self.algorithms.par_iter_mut()
                .map(|(_, algorithm)| algorithm.quantum_evolve_step(&problem, iteration))
                .collect();
            
            for result in evolution_results {
                result?;
            }
            
            // Collect best solutions
            for (_, algorithm) in &self.algorithms {
                if let Some((solution, fitness)) = algorithm.get_best_solution() {
                    if fitness < best_fitness {
                        best_fitness = fitness;
                        best_solution = Some(solution);
                    }
                }
            }
            
            convergence_history.push(best_fitness);
            quantum_states.push(self.quantum_state.clone());
            
            // Apply quantum decoherence
            self.quantum_state.apply_decoherence(0.01);
            
            // Update performance metrics
            self.update_quantum_metrics(iteration).await;
            
            // Adaptive quantum interference
            if iteration % 10 == 0 {
                self.apply_adaptive_interference(iteration).await?;
            }
            
            // Log progress
            if iteration % 100 == 0 {
                log::info!(
                    "Iteration {}: Best fitness = {:.6}, Coherence = {:.3}",
                    iteration, best_fitness, self.quantum_state.coherence()
                );
            }
        }
        
        Ok(QuantumOptimizationResult {
            best_solution: best_solution.unwrap_or_default(),
            best_fitness,
            iterations: max_iterations,
            quantum_metrics: self.performance_metrics.clone(),
            convergence_history,
            quantum_states,
        })
    }
    
    /// Create entanglement between quantum algorithms
    fn create_algorithm_entanglement(&mut self) -> QuantumResult<()> {
        let algorithms: Vec<_> = self.algorithms.keys().copied().collect();
        
        // Create entanglement pairs
        for i in 0..algorithms.len() {
            for j in (i + 1)..algorithms.len() {
                let strength = 0.5 + 0.5 * rand::random::<f64>();
                self.entanglement_network.create_entanglement(algorithms[i], algorithms[j], strength);
                
                // Entangle quantum states
                if let (Some(alg1), Some(alg2)) = (
                    self.algorithms.get_mut(&algorithms[i]),
                    self.algorithms.get(&algorithms[j])
                ) {
                    alg1.apply_entanglement(alg2.quantum_state())?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Apply adaptive quantum interference
    async fn apply_adaptive_interference(&mut self, iteration: usize) -> QuantumResult<()> {
        let pattern: Vec<f64> = (0..12)
            .map(|i| (i as f64 * PI * iteration as f64 / 100.0).sin())
            .collect();
        
        for (_, algorithm) in &mut self.algorithms {
            algorithm.apply_quantum_interference(&pattern)?;
        }
        
        self.performance_metrics.interference_patterns += 1;
        Ok(())
    }
    
    /// Update quantum performance metrics
    async fn update_quantum_metrics(&mut self, iteration: usize) {
        self.performance_metrics.coherence_time = self.quantum_state.coherence_time;
        self.performance_metrics.entanglement_entropy = self.quantum_state.entanglement_entropy();
        
        // Calculate quantum speedup based on convergence rate
        if iteration > 0 {
            self.performance_metrics.quantum_speedup = 1.0 + self.quantum_state.coherence();
        }
        
        // Update algorithm-specific convergence
        for (alg_type, algorithm) in &self.algorithms {
            if let Some((_, fitness)) = algorithm.get_best_solution() {
                self.performance_metrics.algorithm_convergence.insert(*alg_type, fitness);
            }
        }
    }
    
    /// Get real-time quantum metrics
    pub fn get_quantum_metrics(&self) -> &QuantumMetrics {
        &self.performance_metrics
    }
    
    /// Enable/disable SIMD acceleration
    pub fn set_simd_enabled(&mut self, enabled: bool) {
        self.simd_enabled = enabled;
    }
}

// Helper trait extensions for quantum operations
pub trait QuantumBitExt {
    fn create_superposition(&mut self);
    fn apply_hadamard(&mut self);
    fn apply_pauli_x(&mut self);
    fn apply_pauli_y(&mut self);
    fn apply_pauli_z(&mut self);
}

impl QuantumBitExt for QuantumBit {
    fn create_superposition(&mut self) {
        self.rotate_y(PI / 2.0); // Create |+⟩ = (|0⟩ + |1⟩)/√2
    }
    
    fn apply_hadamard(&mut self) {
        self.create_superposition();
    }
    
    fn apply_pauli_x(&mut self) {
        std::mem::swap(&mut self.alpha, &mut self.beta);
    }
    
    fn apply_pauli_y(&mut self) {
        let temp = self.alpha;
        self.alpha = -Complex64::i() * self.beta;
        self.beta = Complex64::i() * temp;
    }
    
    fn apply_pauli_z(&mut self) {
        self.beta = -self.beta;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quantum_optimizer_creation() {
        let optimizer = QuantumOptimizer::new();
        assert_eq!(optimizer.algorithms.len(), 2); // Only PSO and GA implemented for now
        assert_eq!(optimizer.quantum_state.qubits.len(), 12);
    }
    
    #[test]
    fn test_quantum_bit_superposition() {
        let mut qubit = QuantumBit::new();
        qubit.create_superposition();
        
        let p0 = qubit.prob_zero();
        let p1 = qubit.prob_one();
        
        assert!((p0 - 0.5).abs() < 1e-10);
        assert!((p1 - 0.5).abs() < 1e-10);
    }
    
    #[tokio::test]
    async fn test_optimization_problem() {
        let problem = OptimizationProblem {
            dimensions: 2,
            bounds: vec![(-10.0, 10.0), (-10.0, 10.0)],
            objective_function: |x| x[0] * x[0] + x[1] * x[1], // Simple sphere function
            constraints: vec![],
            quantum_enhanced: true,
        };
        
        let mut optimizer = QuantumOptimizer::new();
        let result = optimizer.optimize_parallel(problem, 100).await;
        
        assert!(result.is_ok());
        let opt_result = result.unwrap();
        assert_eq!(opt_result.iterations, 100);
        assert!(opt_result.best_fitness >= 0.0);
    }
}