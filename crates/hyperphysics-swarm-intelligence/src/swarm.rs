//! Meta-Swarm System
//!
//! Orchestrates multiple swarm strategies, topologies, and the pBit lattice
//! to create emergent intelligent behaviors.

use std::collections::HashMap;
use std::sync::Arc;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::{SwarmResult, SwarmIntelligenceError};
use crate::lattice::{PBitLattice, LatticeConfig, SpatioTemporalState};
use crate::topology::{SwarmTopology, TopologyType, TopologyMetrics};
use crate::strategy::{BiomimeticStrategy, StrategyType, StrategyConfig, StrategyResult};
use crate::evolution::{EvolutionEngine, Genome, Fitness, EvolutionConfig};

/// Configuration for the meta-swarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    /// Number of agents
    pub agent_count: usize,
    /// Active strategies
    pub strategies: Vec<StrategyType>,
    /// Primary topology
    pub topology: TopologyType,
    /// Lattice configuration
    pub lattice_config: LatticeConfig,
    /// Problem dimensions
    pub dimensions: usize,
    /// Search bounds
    pub bounds: Vec<(f64, f64)>,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Enable evolution
    pub enable_evolution: bool,
    /// Evolution config
    pub evolution_config: EvolutionConfig,
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            agent_count: 50,
            strategies: vec![
                StrategyType::ParticleSwarm,
                StrategyType::GreyWolf,
                StrategyType::WhaleOptimization,
            ],
            topology: TopologyType::Hyperbolic,
            lattice_config: LatticeConfig::default(),
            dimensions: 10,
            bounds: vec![(-100.0, 100.0); 10],
            max_iterations: 1000,
            enable_evolution: true,
            evolution_config: EvolutionConfig::default(),
        }
    }
}

/// An agent in the meta-swarm
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmAgent {
    /// Unique identifier
    pub id: Uuid,
    /// Current position
    pub position: Vec<f64>,
    /// Current velocity
    pub velocity: Vec<f64>,
    /// Personal best position
    pub personal_best: Vec<f64>,
    /// Personal best fitness
    pub personal_best_fitness: f64,
    /// Current fitness
    pub fitness: f64,
    /// Assigned strategy
    pub strategy: StrategyType,
    /// Lattice coordinates (if embedded)
    pub lattice_position: Option<(usize, usize, usize)>,
    /// Agent-specific parameters
    pub parameters: HashMap<String, f64>,
    /// Neighbors in topology
    pub neighbors: Vec<Uuid>,
    /// Trust scores for neighbors
    pub trust_scores: HashMap<Uuid, f64>,
    /// Contribution to swarm intelligence
    pub contribution: f64,
}

impl SwarmAgent {
    /// Create a new agent
    pub fn new(dimensions: usize) -> Self {
        Self {
            id: Uuid::new_v4(),
            position: vec![0.0; dimensions],
            velocity: vec![0.0; dimensions],
            personal_best: vec![0.0; dimensions],
            personal_best_fitness: f64::INFINITY,
            fitness: f64::INFINITY,
            strategy: StrategyType::ParticleSwarm,
            lattice_position: None,
            parameters: HashMap::new(),
            neighbors: Vec::new(),
            trust_scores: HashMap::new(),
            contribution: 0.0,
        }
    }
}

/// Complete swarm state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmState {
    /// Current iteration
    pub iteration: usize,
    /// Global best position
    pub global_best: Vec<f64>,
    /// Global best fitness
    pub global_best_fitness: f64,
    /// All agents
    pub agents: Vec<SwarmAgent>,
    /// Lattice state
    pub lattice_state: SpatioTemporalState,
    /// Topology metrics
    pub topology_metrics: TopologyMetrics,
    /// Strategy performance
    pub strategy_performance: HashMap<StrategyType, f64>,
    /// Convergence history
    pub convergence: Vec<f64>,
    /// Diversity history
    pub diversity: Vec<f64>,
    /// Active genome (if evolving)
    pub active_genome: Option<Genome>,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// The main meta-swarm orchestrator
pub struct MetaSwarm {
    config: SwarmConfig,
    agents: Vec<SwarmAgent>,
    lattice: PBitLattice,
    topology: SwarmTopology,
    strategies: HashMap<StrategyType, BiomimeticStrategy>,
    evolution_engine: Option<EvolutionEngine>,
    global_best: Vec<f64>,
    global_best_fitness: f64,
    iteration: usize,
    convergence: Vec<f64>,
    diversity: Vec<f64>,
    strategy_performance: HashMap<StrategyType, Vec<f64>>,
    active_genome: Option<Genome>,
}

impl std::fmt::Debug for MetaSwarm {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MetaSwarm")
            .field("iteration", &self.iteration)
            .field("global_best_fitness", &self.global_best_fitness)
            .field("num_agents", &self.agents.len())
            .field("num_strategies", &self.strategies.len())
            .finish()
    }
}

impl MetaSwarm {
    /// Create a new meta-swarm
    pub fn new(config: SwarmConfig) -> SwarmResult<Self> {
        // Initialize lattice
        let lattice = PBitLattice::new(config.lattice_config.clone())?;
        
        // Initialize topology
        let topology = SwarmTopology::new(config.topology, config.agent_count)?;
        
        // Initialize agents
        let agents: Vec<SwarmAgent> = topology.agents().iter().map(|&id| {
            let mut agent = SwarmAgent::new(config.dimensions);
            agent.id = id;
            agent.neighbors = topology.get_neighbors(id);
            agent
        }).collect();
        
        // Initialize strategies
        let mut strategies = HashMap::new();
        for strategy_type in &config.strategies {
            let strategy_config = StrategyConfig {
                strategy_type: *strategy_type,
                population_size: config.agent_count / config.strategies.len(),
                swarm_size: None,
                max_iterations: config.max_iterations,
                bounds: config.bounds.clone(),
                params: HashMap::new(),
            };
            strategies.insert(*strategy_type, BiomimeticStrategy::new(strategy_config)?);
        }
        
        // Initialize evolution engine
        let evolution_engine = if config.enable_evolution {
            Some(EvolutionEngine::new(config.evolution_config.clone()))
        } else {
            None
        };
        
        let dim = config.dimensions;
        
        Ok(Self {
            config,
            agents,
            lattice,
            topology,
            strategies,
            evolution_engine,
            global_best: vec![0.0; dim],
            global_best_fitness: f64::INFINITY,
            iteration: 0,
            convergence: Vec::new(),
            diversity: Vec::new(),
            strategy_performance: HashMap::new(),
            active_genome: None,
        })
    }
    
    /// Optimize using the meta-swarm
    pub fn optimize<F>(&mut self, objective: F) -> SwarmResult<SwarmState>
    where
        F: Fn(&[f64]) -> f64 + Clone,
    {
        // Initialize agent positions randomly
        let mut rng = rand::thread_rng();
        for agent in &mut self.agents {
            for (i, &(min, max)) in self.config.bounds.iter().enumerate() {
                agent.position[i] = min + rand::Rng::gen::<f64>(&mut rng) * (max - min);
                agent.velocity[i] = (max - min) * 0.1 * (rand::Rng::gen::<f64>(&mut rng) - 0.5);
            }
            agent.personal_best = agent.position.clone();
        }
        
        // Main optimization loop
        while self.iteration < self.config.max_iterations {
            // Evaluate all agents
            for agent in &mut self.agents {
                agent.fitness = objective(&agent.position);
                
                if agent.fitness < agent.personal_best_fitness {
                    agent.personal_best = agent.position.clone();
                    agent.personal_best_fitness = agent.fitness;
                }
                
                if agent.fitness < self.global_best_fitness {
                    self.global_best = agent.position.clone();
                    self.global_best_fitness = agent.fitness;
                }
            }
            
            // Update strategies based on topology neighborhoods
            self.update_with_strategies();
            
            // Sweep lattice
            self.lattice.sweep();
            
            // Apply lattice influence to agents
            self.apply_lattice_influence();
            
            // Record metrics
            self.convergence.push(self.global_best_fitness);
            self.diversity.push(self.compute_diversity());
            
            // Track strategy performance
            self.track_strategy_performance();
            
            self.iteration += 1;
            
            // Adaptive strategy switching
            if self.iteration % 100 == 0 {
                self.adapt_strategies();
            }
        }
        
        Ok(self.get_state())
    }
    
    /// Update agents using their assigned strategies
    fn update_with_strategies(&mut self) {
        let strategy_count = self.config.strategies.len();
        let strategies = self.config.strategies.clone();
        let bounds = self.config.bounds.clone();
        let global_best = self.global_best.clone();
        let max_iterations = self.config.max_iterations;
        let iteration = self.iteration;
        
        // Pre-compute neighborhood bests
        let neighborhood_bests: Vec<Vec<f64>> = self.agents.iter()
            .map(|a| self.get_neighborhood_best(a.id))
            .collect();
        
        for (i, agent) in self.agents.iter_mut().enumerate() {
            // Assign strategy based on agent index (round-robin)
            let strategy_idx = i % strategy_count;
            agent.strategy = strategies[strategy_idx];
            
            // Get neighborhood best
            let neighborhood_best = &neighborhood_bests[i];
            
            // Apply strategy-specific update
            match agent.strategy {
                StrategyType::ParticleSwarm => {
                    Self::pso_update_static(agent, neighborhood_best);
                }
                StrategyType::GreyWolf => {
                    Self::grey_wolf_update_static(agent, &global_best, iteration, max_iterations);
                }
                StrategyType::WhaleOptimization => {
                    Self::whale_update_static(agent, &global_best, iteration, max_iterations);
                }
                _ => {
                    Self::pso_update_static(agent, neighborhood_best);
                }
            }
            
            // Enforce bounds
            for (j, &(min, max)) in bounds.iter().enumerate() {
                agent.position[j] = agent.position[j].clamp(min, max);
            }
        }
    }
    
    /// PSO update for an agent (static version to avoid borrow issues)
    fn pso_update_static(agent: &mut SwarmAgent, neighborhood_best: &[f64]) {
        let w = 0.7;
        let c1 = 1.5;
        let c2 = 1.5;
        
        for i in 0..agent.position.len() {
            let r1 = rand::random::<f64>();
            let r2 = rand::random::<f64>();
            
            agent.velocity[i] = w * agent.velocity[i]
                + c1 * r1 * (agent.personal_best[i] - agent.position[i])
                + c2 * r2 * (neighborhood_best[i] - agent.position[i]);
            
            agent.position[i] += agent.velocity[i];
        }
    }
    
    /// Grey wolf update for an agent (static version)
    fn grey_wolf_update_static(agent: &mut SwarmAgent, global_best: &[f64], iteration: usize, max_iterations: usize) {
        let a = 2.0 - 2.0 * (iteration as f64 / max_iterations as f64);
        
        for i in 0..agent.position.len() {
            let r1 = rand::random::<f64>();
            let r2 = rand::random::<f64>();
            
            let a_coeff = 2.0 * a * r1 - a;
            let c = 2.0 * r2;
            
            let d = (c * global_best[i] - agent.position[i]).abs();
            agent.position[i] = global_best[i] - a_coeff * d;
        }
    }
    
    /// Whale update for an agent (static version)
    fn whale_update_static(agent: &mut SwarmAgent, global_best: &[f64], iteration: usize, max_iterations: usize) {
        let a = 2.0 - 2.0 * (iteration as f64 / max_iterations as f64);
        let p = rand::random::<f64>();
        
        for i in 0..agent.position.len() {
            if p < 0.5 {
                let r = rand::random::<f64>();
                let a_coeff = 2.0 * a * r - a;
                let c = 2.0 * rand::random::<f64>();
                let d = (c * global_best[i] - agent.position[i]).abs();
                agent.position[i] = global_best[i] - a_coeff * d;
            } else {
                let l = rand::random::<f64>() * 2.0 - 1.0;
                let d = (global_best[i] - agent.position[i]).abs();
                agent.position[i] = d * (std::f64::consts::PI * l).cos().exp() * (2.0 * std::f64::consts::PI * l).cos() 
                    + global_best[i];
            }
        }
    }
    
    /// Get best position in agent's neighborhood
    fn get_neighborhood_best(&self, agent_id: Uuid) -> Vec<f64> {
        let neighbors = self.topology.get_neighbors(agent_id);
        
        let mut best = self.global_best.clone();
        let mut best_fitness = self.global_best_fitness;
        
        for &neighbor_id in &neighbors {
            if let Some(neighbor) = self.agents.iter().find(|a| a.id == neighbor_id) {
                if neighbor.personal_best_fitness < best_fitness {
                    best = neighbor.personal_best.clone();
                    best_fitness = neighbor.personal_best_fitness;
                }
            }
        }
        
        best
    }
    
    /// Apply pBit lattice influence to agents
    fn apply_lattice_influence(&mut self) {
        let lattice_state = self.lattice.get_state();
        let (lx, ly, lz) = self.lattice.dimensions();
        
        for agent in &mut self.agents {
            // Map agent position to lattice coordinates
            let norm_pos: Vec<f64> = agent.position.iter()
                .zip(&self.config.bounds)
                .map(|(p, (min, max))| (p - min) / (max - min))
                .collect();
            
            // Get lattice node influence
            let lx_idx = ((norm_pos[0] * lx as f64) as usize).min(lx - 1);
            let ly_idx = ((norm_pos.get(1).unwrap_or(&0.5) * ly as f64) as usize).min(ly - 1);
            let lz_idx = ((norm_pos.get(2).unwrap_or(&0.5) * lz as f64) as usize).min(lz - 1);
            
            agent.lattice_position = Some((lx_idx, ly_idx, lz_idx));
            
            // Apply spin influence as perturbation
            let spin_idx = lx_idx * ly * lz + ly_idx * lz + lz_idx;
            if let Some(&spin) = lattice_state.spins.get(spin_idx) {
                // Spin affects search direction
                let spin_influence = spin * 0.01 * self.lattice.temperature();
                for (i, v) in agent.velocity.iter_mut().enumerate() {
                    *v += spin_influence * (rand::random::<f64>() - 0.5);
                }
            }
        }
    }
    
    /// Track strategy performance
    fn track_strategy_performance(&mut self) {
        for strategy in &self.config.strategies {
            let agents_with_strategy: Vec<_> = self.agents.iter()
                .filter(|a| a.strategy == *strategy)
                .collect();
            
            if !agents_with_strategy.is_empty() {
                let avg_improvement: f64 = agents_with_strategy.iter()
                    .map(|a| a.personal_best_fitness)
                    .sum::<f64>() / agents_with_strategy.len() as f64;
                
                self.strategy_performance
                    .entry(*strategy)
                    .or_default()
                    .push(avg_improvement);
            }
        }
    }
    
    /// Adapt strategies based on performance
    fn adapt_strategies(&mut self) {
        // Find best performing strategy
        let mut best_strategy = self.config.strategies[0];
        let mut best_performance = f64::INFINITY;
        
        for (strategy, performance) in &self.strategy_performance {
            if let Some(&last) = performance.last() {
                if last < best_performance {
                    best_performance = last;
                    best_strategy = *strategy;
                }
            }
        }
        
        // Shift more agents to best strategy
        let total = self.agents.len();
        let shift_count = total / 4;
        
        for (i, agent) in self.agents.iter_mut().enumerate() {
            if i < shift_count {
                agent.strategy = best_strategy;
            }
        }
        
        // Reduce lattice temperature for exploitation
        let current_temp = self.lattice.temperature();
        self.lattice.set_temperature(current_temp * 0.95);
    }
    
    /// Compute population diversity
    fn compute_diversity(&self) -> f64 {
        if self.agents.is_empty() { return 0.0; }
        
        let n = self.agents.len() as f64;
        let dim = self.config.dimensions;
        
        // Compute centroid
        let mut centroid = vec![0.0; dim];
        for agent in &self.agents {
            for (i, &p) in agent.position.iter().enumerate() {
                centroid[i] += p;
            }
        }
        for c in &mut centroid {
            *c /= n;
        }
        
        // Average distance from centroid
        let mut total_dist = 0.0;
        for agent in &self.agents {
            let dist: f64 = agent.position.iter()
                .zip(&centroid)
                .map(|(p, c)| (p - c).powi(2))
                .sum::<f64>()
                .sqrt();
            total_dist += dist;
        }
        
        total_dist / n
    }
    
    /// Get current state
    pub fn get_state(&self) -> SwarmState {
        let strategy_performance: HashMap<StrategyType, f64> = self.strategy_performance
            .iter()
            .map(|(s, p)| (*s, p.last().copied().unwrap_or(f64::INFINITY)))
            .collect();
        
        SwarmState {
            iteration: self.iteration,
            global_best: self.global_best.clone(),
            global_best_fitness: self.global_best_fitness,
            agents: self.agents.clone(),
            lattice_state: self.lattice.get_state(),
            topology_metrics: self.topology.metrics(),
            strategy_performance,
            convergence: self.convergence.clone(),
            diversity: self.diversity.clone(),
            active_genome: self.active_genome.clone(),
            timestamp: Utc::now(),
        }
    }
    
    /// Get best solution
    pub fn best(&self) -> (&[f64], f64) {
        (&self.global_best, self.global_best_fitness)
    }
    
    /// Evolve the swarm strategies
    pub fn evolve<F>(&mut self, objective: F, generations: usize) -> SwarmResult<Genome>
    where
        F: Fn(&[f64]) -> f64 + Clone,
    {
        if self.evolution_engine.is_none() {
            return Err(SwarmIntelligenceError::EvolutionError(
                "Evolution not enabled".to_string()
            ));
        }
        
        let engine = self.evolution_engine.as_mut().unwrap();
        let bounds = self.config.bounds.clone();
        let obj = objective.clone();
        
        let best = engine.evolve(|genome| {
            // Create strategy from genome
            let strategy_config = genome.to_strategy_config(bounds.clone());
            let mut strategy = BiomimeticStrategy::new(strategy_config)?;
            
            // Run optimization
            let result = strategy.optimize(&obj)?;
            
            // Compute fitness
            Ok(Fitness {
                objective: result.best_fitness,
                convergence_speed: result.iterations as f64,
                diversity: result.diversity,
                robustness: 1.0 / (1.0 + result.convergence.windows(2)
                    .map(|w| (w[1] - w[0]).abs())
                    .sum::<f64>()),
                efficiency: 1.0 / (1.0 + result.evaluations as f64 / 1000.0),
                combined: 0.0,
            })
        })?;
        
        self.active_genome = Some(best.clone());
        Ok(best)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|&xi| xi * xi).sum()
    }
    
    #[test]
    fn test_meta_swarm_creation() {
        let config = SwarmConfig {
            agent_count: 20,
            dimensions: 5,
            bounds: vec![(-5.0, 5.0); 5],
            max_iterations: 100,
            enable_evolution: false,
            ..Default::default()
        };
        
        let swarm = MetaSwarm::new(config).unwrap();
        assert_eq!(swarm.agents.len(), 20);
    }
    
    #[test]
    fn test_meta_swarm_optimization() {
        let config = SwarmConfig {
            agent_count: 30,
            dimensions: 5,
            bounds: vec![(-5.0, 5.0); 5],
            max_iterations: 200,
            enable_evolution: false,
            ..Default::default()
        };
        
        let mut swarm = MetaSwarm::new(config).unwrap();
        let state = swarm.optimize(sphere).unwrap();
        
        assert!(state.global_best_fitness < 1.0);
    }
}
