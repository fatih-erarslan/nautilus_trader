//! Multi-strategy swarm optimization
//!
//! Provides a meta-swarm that combines multiple strategies with
//! different network topologies.

use std::collections::HashMap;
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use uuid::Uuid;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use crate::{Result, Strategy, Topology, SwarmConfig, ConvergenceMetrics};

/// Swarm optimization result
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SwarmResult {
    /// Best solution found
    pub solution: Vec<f64>,
    /// Best fitness value
    pub fitness: f64,
    /// Convergence history
    pub history: Vec<f64>,
    /// Strategy performance
    pub strategy_performance: HashMap<Strategy, f64>,
    /// Metrics
    pub metrics: ConvergenceMetrics,
}

/// Agent in the swarm
#[derive(Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub struct SwarmAgent {
    /// Unique ID
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
    pub strategy: Strategy,
    /// Neighbor IDs
    pub neighbors: Vec<Uuid>,
}

impl SwarmAgent {
    fn new(dimensions: usize) -> Self {
        Self {
            id: Uuid::new_v4(),
            position: vec![0.0; dimensions],
            velocity: vec![0.0; dimensions],
            personal_best: vec![0.0; dimensions],
            personal_best_fitness: f64::INFINITY,
            fitness: f64::INFINITY,
            strategy: Strategy::ParticleSwarm,
            neighbors: Vec::new(),
        }
    }
    
    fn random(bounds: &[(f64, f64)], rng: &mut impl Rng) -> Self {
        let mut agent = Self::new(bounds.len());
        for (i, &(min, max)) in bounds.iter().enumerate() {
            agent.position[i] = rng.gen_range(min..max);
            agent.velocity[i] = (max - min) * 0.1 * (rng.gen::<f64>() - 0.5);
        }
        agent.personal_best = agent.position.clone();
        agent
    }
}

/// Multi-strategy swarm
pub struct Swarm {
    config: SwarmConfig,
    agents: Vec<SwarmAgent>,
    global_best: Vec<f64>,
    global_best_fitness: f64,
    iteration: usize,
    evaluations: usize,
    history: Vec<f64>,
    strategy_performance: HashMap<Strategy, Vec<f64>>,
    rng: ChaCha8Rng,
}

impl Swarm {
    /// Create a new swarm with configuration
    pub fn new(config: SwarmConfig) -> Result<Self> {
        let mut rng = ChaCha8Rng::from_entropy();
        
        // Create agents
        let agents: Vec<SwarmAgent> = (0..config.agent_count)
            .map(|_| SwarmAgent::random(&config.bounds, &mut rng))
            .collect();
        
        let dim = config.dimensions;
        
        Ok(Self {
            config,
            agents,
            global_best: vec![0.0; dim],
            global_best_fitness: f64::INFINITY,
            iteration: 0,
            evaluations: 0,
            history: Vec::new(),
            strategy_performance: HashMap::new(),
            rng,
        })
    }
    
    /// Run swarm optimization
    pub fn optimize<F>(&mut self, objective: F) -> Result<SwarmResult>
    where
        F: Fn(&[f64]) -> f64,
    {
        let start = std::time::Instant::now();
        
        // Build topology
        self.build_topology();
        
        // Assign strategies
        self.assign_strategies();
        
        // Initial evaluation
        for agent in &mut self.agents {
            agent.fitness = objective(&agent.position);
            self.evaluations += 1;
            
            if agent.fitness < agent.personal_best_fitness {
                agent.personal_best = agent.position.clone();
                agent.personal_best_fitness = agent.fitness;
            }
            
            if agent.fitness < self.global_best_fitness {
                self.global_best = agent.position.clone();
                self.global_best_fitness = agent.fitness;
            }
        }
        
        // Main loop
        while self.iteration < self.config.max_iterations {
            // Update each agent based on its strategy
            self.update_agents();
            
            // Evaluate
            for agent in &mut self.agents {
                agent.fitness = objective(&agent.position);
                self.evaluations += 1;
                
                if agent.fitness < agent.personal_best_fitness {
                    agent.personal_best = agent.position.clone();
                    agent.personal_best_fitness = agent.fitness;
                }
                
                if agent.fitness < self.global_best_fitness {
                    self.global_best = agent.position.clone();
                    self.global_best_fitness = agent.fitness;
                }
            }
            
            self.history.push(self.global_best_fitness);
            self.track_strategy_performance();
            self.iteration += 1;
            
            // Adapt strategies periodically
            if self.iteration % 100 == 0 {
                self.adapt_strategies();
            }
        }
        
        let convergence_rate = if self.history.len() > 1 {
            let first = self.history.first().unwrap();
            let last = self.history.last().unwrap();
            (first - last) / self.history.len() as f64
        } else {
            0.0
        };
        
        let strategy_perf: HashMap<Strategy, f64> = self.strategy_performance
            .iter()
            .map(|(s, v)| (*s, v.last().copied().unwrap_or(f64::INFINITY)))
            .collect();
        
        Ok(SwarmResult {
            solution: self.global_best.clone(),
            fitness: self.global_best_fitness,
            history: self.history.clone(),
            strategy_performance: strategy_perf,
            metrics: ConvergenceMetrics {
                final_fitness: self.global_best_fitness,
                iterations: self.iteration,
                evaluations: self.evaluations,
                convergence_rate,
                diversity: self.compute_diversity(),
                time_ms: start.elapsed().as_millis() as u64,
            },
        })
    }
    
    /// Build network topology
    fn build_topology(&mut self) {
        let agent_ids: Vec<Uuid> = self.agents.iter().map(|a| a.id).collect();
        let n = agent_ids.len();
        
        match self.config.topology {
            Topology::Star => {
                // All connect to first agent
                for agent in self.agents.iter_mut().skip(1) {
                    agent.neighbors = vec![agent_ids[0]];
                }
                self.agents[0].neighbors = agent_ids[1..].to_vec();
            }
            Topology::Ring => {
                for i in 0..n {
                    let prev = (i + n - 1) % n;
                    let next = (i + 1) % n;
                    self.agents[i].neighbors = vec![agent_ids[prev], agent_ids[next]];
                }
            }
            Topology::Mesh => {
                for i in 0..n {
                    self.agents[i].neighbors = agent_ids.iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, id)| *id)
                        .collect();
                }
            }
            _ => {
                // Default to mesh for other topologies
                for i in 0..n {
                    self.agents[i].neighbors = agent_ids.iter()
                        .enumerate()
                        .filter(|(j, _)| *j != i)
                        .map(|(_, id)| *id)
                        .collect();
                }
            }
        }
    }
    
    /// Assign strategies to agents
    fn assign_strategies(&mut self) {
        let strategies = &self.config.strategies;
        if strategies.is_empty() {
            return;
        }
        
        for (i, agent) in self.agents.iter_mut().enumerate() {
            agent.strategy = strategies[i % strategies.len()];
        }
    }
    
    /// Update agents based on their strategies
    fn update_agents(&mut self) {
        let global_best = self.global_best.clone();
        let bounds = self.config.bounds.clone();
        let max_iterations = self.config.max_iterations;
        let iteration = self.iteration;
        
        // Collect neighborhood bests
        let neighborhood_bests: Vec<Vec<f64>> = self.agents.iter()
            .map(|agent| self.get_neighborhood_best(agent))
            .collect();
        
        for (i, agent) in self.agents.iter_mut().enumerate() {
            let neighborhood_best = &neighborhood_bests[i];
            
            match agent.strategy {
                Strategy::ParticleSwarm => {
                    Self::pso_update(agent, neighborhood_best, &mut self.rng);
                }
                Strategy::GreyWolf => {
                    Self::grey_wolf_update(agent, &global_best, iteration, max_iterations, &mut self.rng);
                }
                Strategy::Whale => {
                    Self::whale_update(agent, &global_best, iteration, max_iterations, &mut self.rng);
                }
                _ => {
                    Self::pso_update(agent, neighborhood_best, &mut self.rng);
                }
            }
            
            // Enforce bounds
            for (j, &(min, max)) in bounds.iter().enumerate() {
                agent.position[j] = agent.position[j].clamp(min, max);
            }
        }
    }
    
    /// PSO update
    fn pso_update(agent: &mut SwarmAgent, neighborhood_best: &[f64], rng: &mut impl Rng) {
        let w = 0.7;
        let c1 = 1.5;
        let c2 = 1.5;
        
        for i in 0..agent.position.len() {
            let r1 = rng.gen::<f64>();
            let r2 = rng.gen::<f64>();
            
            agent.velocity[i] = w * agent.velocity[i]
                + c1 * r1 * (agent.personal_best[i] - agent.position[i])
                + c2 * r2 * (neighborhood_best[i] - agent.position[i]);
            
            agent.position[i] += agent.velocity[i];
        }
    }
    
    /// Grey Wolf update
    fn grey_wolf_update(agent: &mut SwarmAgent, global_best: &[f64], iteration: usize, max_iterations: usize, rng: &mut impl Rng) {
        let a = 2.0 - 2.0 * (iteration as f64 / max_iterations as f64);
        
        for i in 0..agent.position.len() {
            let r1 = rng.gen::<f64>();
            let r2 = rng.gen::<f64>();
            let a_coeff = 2.0 * a * r1 - a;
            let c = 2.0 * r2;
            
            let d = (c * global_best[i] - agent.position[i]).abs();
            agent.position[i] = global_best[i] - a_coeff * d;
        }
    }
    
    /// Whale update
    fn whale_update(agent: &mut SwarmAgent, global_best: &[f64], iteration: usize, max_iterations: usize, rng: &mut impl Rng) {
        let a = 2.0 - 2.0 * (iteration as f64 / max_iterations as f64);
        let p = rng.gen::<f64>();
        
        for i in 0..agent.position.len() {
            if p < 0.5 {
                let r = rng.gen::<f64>();
                let a_coeff = 2.0 * a * r - a;
                let c = 2.0 * rng.gen::<f64>();
                let d = (c * global_best[i] - agent.position[i]).abs();
                agent.position[i] = global_best[i] - a_coeff * d;
            } else {
                let l = rng.gen_range(-1.0..1.0);
                let d = (global_best[i] - agent.position[i]).abs();
                agent.position[i] = d * (std::f64::consts::PI * l).cos().exp() 
                    * (2.0 * std::f64::consts::PI * l).cos() + global_best[i];
            }
        }
    }
    
    /// Get neighborhood best
    fn get_neighborhood_best(&self, agent: &SwarmAgent) -> Vec<f64> {
        let mut best = self.global_best.clone();
        let mut best_fitness = self.global_best_fitness;
        
        for neighbor_id in &agent.neighbors {
            if let Some(neighbor) = self.agents.iter().find(|a| a.id == *neighbor_id) {
                if neighbor.personal_best_fitness < best_fitness {
                    best = neighbor.personal_best.clone();
                    best_fitness = neighbor.personal_best_fitness;
                }
            }
        }
        
        best
    }
    
    /// Track strategy performance
    fn track_strategy_performance(&mut self) {
        for strategy in &self.config.strategies {
            let agents: Vec<_> = self.agents.iter()
                .filter(|a| a.strategy == *strategy)
                .collect();
            
            if !agents.is_empty() {
                let avg: f64 = agents.iter()
                    .map(|a| a.personal_best_fitness)
                    .sum::<f64>() / agents.len() as f64;
                
                self.strategy_performance
                    .entry(*strategy)
                    .or_default()
                    .push(avg);
            }
        }
    }
    
    /// Adapt strategies based on performance
    fn adapt_strategies(&mut self) {
        // Find best performing strategy
        let mut best_strategy = self.config.strategies[0];
        let mut best_perf = f64::INFINITY;
        
        for (strategy, perf) in &self.strategy_performance {
            if let Some(&last) = perf.last() {
                if last < best_perf {
                    best_perf = last;
                    best_strategy = *strategy;
                }
            }
        }
        
        // Shift more agents to best strategy
        let shift_count = self.agents.len() / 4;
        for agent in self.agents.iter_mut().take(shift_count) {
            agent.strategy = best_strategy;
        }
    }
    
    /// Compute diversity
    fn compute_diversity(&self) -> f64 {
        if self.agents.is_empty() {
            return 0.0;
        }
        
        let n = self.agents.len() as f64;
        let dim = self.config.dimensions;
        
        let mut centroid = vec![0.0; dim];
        for agent in &self.agents {
            for (i, &p) in agent.position.iter().enumerate() {
                centroid[i] += p;
            }
        }
        for c in &mut centroid {
            *c /= n;
        }
        
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
}

/// Builder for swarm
pub struct SwarmBuilder {
    config: SwarmConfig,
}

impl SwarmBuilder {
    /// Create a new builder
    pub fn new() -> Self {
        Self {
            config: SwarmConfig::default(),
        }
    }
    
    /// Set agent count
    pub fn agents(mut self, count: usize) -> Self {
        self.config.agent_count = count;
        self
    }
    
    /// Set dimensions
    pub fn dimensions(mut self, dim: usize) -> Self {
        self.config.dimensions = dim;
        if self.config.bounds.len() != dim {
            let default = self.config.bounds.first().copied().unwrap_or((-100.0, 100.0));
            self.config.bounds = vec![default; dim];
        }
        self
    }
    
    /// Set bounds
    pub fn bounds(mut self, min: f64, max: f64) -> Self {
        self.config.bounds = vec![(min, max); self.config.dimensions];
        self
    }
    
    /// Set strategies
    pub fn strategies(mut self, strategies: Vec<Strategy>) -> Self {
        self.config.strategies = strategies;
        self
    }
    
    /// Set topology
    pub fn topology(mut self, topology: Topology) -> Self {
        self.config.topology = topology;
        self
    }
    
    /// Set iterations
    pub fn iterations(mut self, max: usize) -> Self {
        self.config.max_iterations = max;
        self
    }
    
    /// Build the swarm
    pub fn build(self) -> Result<Swarm> {
        Swarm::new(self.config)
    }
    
    /// Build and run optimization
    pub fn minimize<F>(self, objective: F) -> Result<SwarmResult>
    where
        F: Fn(&[f64]) -> f64,
    {
        let mut swarm = self.build()?;
        swarm.optimize(objective)
    }
}

impl Default for SwarmBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_swarm_optimize() {
        let result = SwarmBuilder::new()
            .agents(30)
            .dimensions(5)
            .bounds(-5.0, 5.0)
            .strategies(vec![Strategy::ParticleSwarm, Strategy::GreyWolf])
            .iterations(200)
            .minimize(|x| x.iter().map(|xi| xi * xi).sum())
            .unwrap();
        
        assert!(result.fitness < 1.0);
    }
}
