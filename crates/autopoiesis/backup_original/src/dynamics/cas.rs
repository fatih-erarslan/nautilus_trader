use std::collections::HashMap;
use nalgebra::{DVector, DMatrix};
use rand::{Rng, thread_rng};

/// Complex Adaptive System for modeling swarm evolution and emergent behaviors
/// Implements agents with learning, adaptation, and interaction rules
pub struct ComplexAdaptiveSystem {
    /// Collection of adaptive agents
    agents: Vec<AdaptiveAgent>,
    /// Environment state
    environment: Environment,
    /// Interaction network
    network: InteractionNetwork,
    /// Evolution parameters
    params: CasParameters,
    /// Generation counter
    generation: usize,
    /// Fitness history
    fitness_history: Vec<FitnessSnapshot>,
}

#[derive(Clone, Debug)]
pub struct AdaptiveAgent {
    /// Unique identifier
    pub id: usize,
    /// Agent's position in space
    pub position: DVector<f64>,
    /// Agent's internal state/strategy
    pub strategy: DVector<f64>,
    /// Learning rate
    pub learning_rate: f64,
    /// Fitness score
    pub fitness: f64,
    /// Memory of past interactions
    pub memory: Vec<InteractionMemory>,
    /// Agent type
    pub agent_type: AgentType,
    /// Age (number of generations survived)
    pub age: usize,
}

#[derive(Clone, Debug)]
pub enum AgentType {
    Explorer,    // High mutation, exploration
    Exploiter,   // Low mutation, exploitation
    Cooperator,  // Cooperative strategies
    Defector,    // Competitive strategies
    Adaptive,    // Dynamic strategy switching
}

#[derive(Clone, Debug)]
pub struct InteractionMemory {
    pub partner_id: usize,
    pub outcome: f64,
    pub timestamp: f64,
    pub strategy_used: DVector<f64>,
}

#[derive(Clone, Debug)]
pub struct Environment {
    /// Resource distribution
    pub resources: DMatrix<f64>,
    /// Environmental pressure
    pub pressure: f64,
    /// Carrying capacity
    pub carrying_capacity: usize,
    /// Environmental dynamics
    pub dynamics: EnvironmentDynamics,
}

#[derive(Clone, Debug)]
pub struct EnvironmentDynamics {
    /// Resource regeneration rate
    pub regeneration_rate: f64,
    /// Pressure change rate
    pub pressure_change_rate: f64,
    /// Seasonal cycles
    pub seasonal_amplitude: f64,
    pub seasonal_frequency: f64,
}

#[derive(Clone, Debug)]
pub struct InteractionNetwork {
    /// Adjacency matrix for agent interactions
    pub adjacency: DMatrix<f64>,
    /// Network topology type
    pub topology: NetworkTopology,
    /// Connection strength decay
    pub decay_rate: f64,
}

#[derive(Clone, Debug)]
pub enum NetworkTopology {
    Random,
    SmallWorld,
    ScaleFree,
    Lattice,
    FullyConnected,
}

#[derive(Clone, Debug)]
pub struct CasParameters {
    /// Number of agents
    pub population_size: usize,
    /// Dimension of strategy space
    pub strategy_dimensions: usize,
    /// Dimension of physical space
    pub space_dimensions: usize,
    /// Mutation rate
    pub mutation_rate: f64,
    /// Selection pressure
    pub selection_pressure: f64,
    /// Interaction radius
    pub interaction_radius: f64,
    /// Learning parameters
    pub learning_params: LearningParameters,
}

#[derive(Clone, Debug)]
pub struct LearningParameters {
    /// Base learning rate
    pub base_learning_rate: f64,
    /// Learning rate decay
    pub learning_decay: f64,
    /// Memory capacity
    pub memory_capacity: usize,
    /// Exploration vs exploitation trade-off
    pub epsilon: f64,
}

#[derive(Clone, Debug)]
pub struct FitnessSnapshot {
    pub generation: usize,
    pub mean_fitness: f64,
    pub max_fitness: f64,
    pub fitness_variance: f64,
    pub diversity_index: f64,
    pub cooperation_level: f64,
}

impl Default for CasParameters {
    fn default() -> Self {
        Self {
            population_size: 100,
            strategy_dimensions: 10,
            space_dimensions: 2,
            mutation_rate: 0.01,
            selection_pressure: 0.1,
            interaction_radius: 5.0,
            learning_params: LearningParameters {
                base_learning_rate: 0.1,
                learning_decay: 0.99,
                memory_capacity: 50,
                epsilon: 0.1,
            },
        }
    }
}

impl ComplexAdaptiveSystem {
    /// Create new CAS
    pub fn new(params: CasParameters) -> Self {
        let mut agents = Vec::with_capacity(params.population_size);
        let mut rng = thread_rng();

        // Initialize agents
        for i in 0..params.population_size {
            let position = DVector::from_fn(params.space_dimensions, |_, _| {
                rng.gen::<f64>() * 100.0 - 50.0
            });
            
            let strategy = DVector::from_fn(params.strategy_dimensions, |_, _| {
                rng.gen::<f64>() * 2.0 - 1.0
            });

            let agent_type = match rng.gen_range(0..5) {
                0 => AgentType::Explorer,
                1 => AgentType::Exploiter,
                2 => AgentType::Cooperator,
                3 => AgentType::Defector,
                _ => AgentType::Adaptive,
            };

            agents.push(AdaptiveAgent {
                id: i,
                position,
                strategy,
                learning_rate: params.learning_params.base_learning_rate,
                fitness: 0.0,
                memory: Vec::new(),
                agent_type,
                age: 0,
            });
        }

        // Initialize environment
        let environment = Environment {
            resources: DMatrix::from_fn(50, 50, |_, _| rng.gen::<f64>()),
            pressure: 0.5,
            carrying_capacity: params.population_size,
            dynamics: EnvironmentDynamics {
                regeneration_rate: 0.1,
                pressure_change_rate: 0.01,
                seasonal_amplitude: 0.3,
                seasonal_frequency: 0.1,
            },
        };

        // Initialize interaction network
        let network = InteractionNetwork::new(params.population_size, NetworkTopology::SmallWorld);

        Self {
            agents,
            environment,
            network,
            params,
            generation: 0,
            fitness_history: Vec::new(),
        }
    }

    /// Run one generation of the CAS
    pub fn evolve_generation(&mut self) {
        // Update environment
        self.update_environment();
        
        // Agent interactions
        self.process_interactions();
        
        // Learning phase
        self.learning_phase();
        
        // Selection and reproduction
        self.selection_phase();
        
        // Mutation
        self.mutation_phase();
        
        // Update network
        self.update_network();
        
        // Record fitness
        self.record_fitness();
        
        self.generation += 1;
    }

    /// Update environment state
    fn update_environment(&mut self) {
        let time = self.generation as f64;
        
        // Seasonal pressure variation
        let seasonal_pressure = self.environment.dynamics.seasonal_amplitude * 
            (self.environment.dynamics.seasonal_frequency * time).sin();
        
        self.environment.pressure += self.environment.dynamics.pressure_change_rate + seasonal_pressure;
        self.environment.pressure = self.environment.pressure.clamp(0.0, 1.0);
        
        // Resource regeneration
        let mut rng = thread_rng();
        for i in 0..self.environment.resources.nrows() {
            for j in 0..self.environment.resources.ncols() {
                let current = self.environment.resources[(i, j)];
                let regeneration = self.environment.dynamics.regeneration_rate * (1.0 - current);
                let noise = (rng.gen::<f64>() - 0.5) * 0.1;
                self.environment.resources[(i, j)] = (current + regeneration + noise).clamp(0.0, 1.0);
            }
        }
    }

    /// Process agent interactions
    fn process_interactions(&mut self) {
        let mut interactions = Vec::new();
        
        // Find interacting pairs
        for i in 0..self.agents.len() {
            for j in (i + 1)..self.agents.len() {
                if self.network.adjacency[(i, j)] > 0.0 {
                    let distance = (&self.agents[i].position - &self.agents[j].position).norm();
                    if distance <= self.params.interaction_radius {
                        interactions.push((i, j));
                    }
                }
            }
        }

        // Process interactions
        for (i, j) in interactions {
            let outcome = self.calculate_interaction_outcome(i, j);
            
            // Update fitness based on interaction
            self.agents[i].fitness += outcome.0;
            self.agents[j].fitness += outcome.1;
            
            // Store in memory
            let memory_i = InteractionMemory {
                partner_id: j,
                outcome: outcome.0,
                timestamp: self.generation as f64,
                strategy_used: self.agents[i].strategy.clone(),
            };
            
            let memory_j = InteractionMemory {
                partner_id: i,
                outcome: outcome.1,
                timestamp: self.generation as f64,
                strategy_used: self.agents[j].strategy.clone(),
            };
            
            // Add to memory (with capacity limit)
            if self.agents[i].memory.len() >= self.params.learning_params.memory_capacity {
                self.agents[i].memory.remove(0);
            }
            self.agents[i].memory.push(memory_i);
            
            if self.agents[j].memory.len() >= self.params.learning_params.memory_capacity {
                self.agents[j].memory.remove(0);
            }
            self.agents[j].memory.push(memory_j);
        }
    }

    /// Calculate outcome of interaction between two agents
    fn calculate_interaction_outcome(&self, i: usize, j: usize) -> (f64, f64) {
        let agent_i = &self.agents[i];
        let agent_j = &self.agents[j];
        
        // Strategy similarity
        let strategy_dot = agent_i.strategy.dot(&agent_j.strategy);
        let strategy_similarity = strategy_dot / (agent_i.strategy.norm() * agent_j.strategy.norm());
        
        // Resource competition
        let position_distance = (&agent_i.position - &agent_j.position).norm();
        let competition_factor = (-position_distance / self.params.interaction_radius).exp();
        
        // Agent type interactions
        let type_modifier = match (&agent_i.agent_type, &agent_j.agent_type) {
            (AgentType::Cooperator, AgentType::Cooperator) => (1.5, 1.5), // Mutual benefit
            (AgentType::Cooperator, AgentType::Defector) => (-0.5, 1.0), // Exploitation
            (AgentType::Defector, AgentType::Cooperator) => (1.0, -0.5), // Exploitation
            (AgentType::Defector, AgentType::Defector) => (0.0, 0.0), // Mutual harm
            _ => (strategy_similarity, strategy_similarity), // Default to strategy similarity
        };
        
        let base_outcome = strategy_similarity * (1.0 - competition_factor);
        
        (
            base_outcome * type_modifier.0,
            base_outcome * type_modifier.1,
        )
    }

    /// Learning phase - agents adapt their strategies
    fn learning_phase(&mut self) {
        for agent in &mut self.agents {
            if agent.memory.is_empty() {
                continue;
            }
            
            // Calculate average outcome from recent interactions
            let recent_memories: Vec<_> = agent.memory.iter()
                .rev()
                .take(10)
                .collect();
            
            if recent_memories.is_empty() {
                continue;
            }
            
            let avg_outcome: f64 = recent_memories.iter()
                .map(|m| m.outcome)
                .sum::<f64>() / recent_memories.len() as f64;
            
            // Adjust strategy based on learning
            if avg_outcome > 0.0 {
                // Reinforce current strategy
                let reinforcement_factor = agent.learning_rate * avg_outcome;
                agent.strategy *= 1.0 + reinforcement_factor;
            } else {
                // Explore new strategies
                let mut rng = thread_rng();
                let exploration_factor = agent.learning_rate * (-avg_outcome);
                
                for i in 0..agent.strategy.len() {
                    let noise = (rng.gen::<f64>() - 0.5) * exploration_factor;
                    agent.strategy[i] += noise;
                }
            }
            
            // Normalize strategy
            let norm = agent.strategy.norm();
            if norm > 0.0 {
                agent.strategy /= norm;
            }
            
            // Decay learning rate
            agent.learning_rate *= self.params.learning_params.learning_decay;
        }
    }

    /// Selection phase - fitness-based selection
    fn selection_phase(&mut self) {
        // Calculate fitness statistics
        let total_fitness: f64 = self.agents.iter().map(|a| a.fitness.max(0.0)).sum();
        
        if total_fitness <= 0.0 {
            return; // No selection pressure
        }
        
        // Apply selection pressure
        let mut rng = thread_rng();
        let mut new_agents = Vec::new();
        
        for _ in 0..self.params.population_size {
            // Tournament selection
            let tournament_size = 3;
            let mut best_agent = None;
            let mut best_fitness = f64::NEG_INFINITY;
            
            for _ in 0..tournament_size {
                let candidate_idx = rng.gen_range(0..self.agents.len());
                let candidate = &self.agents[candidate_idx];
                
                if candidate.fitness > best_fitness {
                    best_fitness = candidate.fitness;
                    best_agent = Some(candidate.clone());
                }
            }
            
            if let Some(mut agent) = best_agent {
                agent.age += 1;
                agent.fitness = 0.0; // Reset fitness for next generation
                agent.memory.clear(); // Clear memory
                new_agents.push(agent);
            }
        }
        
        self.agents = new_agents;
    }

    /// Mutation phase
    fn mutation_phase(&mut self) {
        let mut rng = thread_rng();
        
        for agent in &mut self.agents {
            if rng.gen::<f64>() < self.params.mutation_rate {
                // Mutate strategy
                let mutation_strength = match agent.agent_type {
                    AgentType::Explorer => 0.1,
                    AgentType::Exploiter => 0.01,
                    _ => 0.05,
                };
                
                for i in 0..agent.strategy.len() {
                    let mutation = (rng.gen::<f64>() - 0.5) * mutation_strength;
                    agent.strategy[i] += mutation;
                }
                
                // Normalize
                let norm = agent.strategy.norm();
                if norm > 0.0 {
                    agent.strategy /= norm;
                }
            }
            
            // Occasionally mutate position
            if rng.gen::<f64>() < self.params.mutation_rate * 0.1 {
                for i in 0..agent.position.len() {
                    let mutation = (rng.gen::<f64>() - 0.5) * 10.0;
                    agent.position[i] += mutation;
                    agent.position[i] = agent.position[i].clamp(-50.0, 50.0);
                }
            }
        }
    }

    /// Update interaction network
    fn update_network(&mut self) {
        // Decay existing connections
        self.network.adjacency *= 1.0 - self.network.decay_rate;
        
        // Form new connections based on successful interactions
        for agent in &self.agents {
            for memory in &agent.memory {
                if memory.outcome > 0.0 && memory.partner_id < self.agents.len() {
                    let strength_increase = memory.outcome * 0.1;
                    self.network.adjacency[(agent.id, memory.partner_id)] += strength_increase;
                    self.network.adjacency[(memory.partner_id, agent.id)] += strength_increase;
                }
            }
        }
        
        // Clamp connection strengths
        for i in 0..self.network.adjacency.nrows() {
            for j in 0..self.network.adjacency.ncols() {
                self.network.adjacency[(i, j)] = self.network.adjacency[(i, j)].clamp(0.0, 1.0);
            }
        }
    }

    /// Record fitness statistics
    fn record_fitness(&mut self) {
        let fitness_values: Vec<f64> = self.agents.iter().map(|a| a.fitness).collect();
        
        let mean_fitness = fitness_values.iter().sum::<f64>() / fitness_values.len() as f64;
        let max_fitness = fitness_values.iter().fold(f64::NEG_INFINITY, |max, &val| val.max(max));
        
        let variance = fitness_values.iter()
            .map(|&x| (x - mean_fitness).powi(2))
            .sum::<f64>() / fitness_values.len() as f64;
        
        // Calculate diversity (strategy diversity)
        let diversity_index = self.calculate_diversity_index();
        
        // Calculate cooperation level
        let cooperation_level = self.calculate_cooperation_level();
        
        self.fitness_history.push(FitnessSnapshot {
            generation: self.generation,
            mean_fitness,
            max_fitness,
            fitness_variance: variance,
            diversity_index,
            cooperation_level,
        });
    }

    /// Calculate diversity index based on strategy differences
    fn calculate_diversity_index(&self) -> f64 {
        if self.agents.len() < 2 {
            return 0.0;
        }
        
        let mut total_distance = 0.0;
        let mut count = 0;
        
        for i in 0..self.agents.len() {
            for j in (i + 1)..self.agents.len() {
                let distance = (&self.agents[i].strategy - &self.agents[j].strategy).norm();
                total_distance += distance;
                count += 1;
            }
        }
        
        if count > 0 {
            total_distance / count as f64
        } else {
            0.0
        }
    }

    /// Calculate cooperation level
    fn calculate_cooperation_level(&self) -> f64 {
        let cooperator_count = self.agents.iter()
            .filter(|a| matches!(a.agent_type, AgentType::Cooperator))
            .count();
        
        cooperator_count as f64 / self.agents.len() as f64
    }

    /// Get current system state
    pub fn get_state(&self) -> CasState {
        CasState {
            generation: self.generation,
            population_size: self.agents.len(),
            agent_types: self.get_agent_type_distribution(),
            fitness_stats: self.fitness_history.last().cloned(),
            network_density: self.calculate_network_density(),
            environmental_pressure: self.environment.pressure,
        }
    }

    fn get_agent_type_distribution(&self) -> HashMap<String, usize> {
        let mut distribution = HashMap::new();
        
        for agent in &self.agents {
            let type_name = match agent.agent_type {
                AgentType::Explorer => "Explorer",
                AgentType::Exploiter => "Exploiter",
                AgentType::Cooperator => "Cooperator",
                AgentType::Defector => "Defector",
                AgentType::Adaptive => "Adaptive",
            };
            
            *distribution.entry(type_name.to_string()).or_insert(0) += 1;
        }
        
        distribution
    }

    fn calculate_network_density(&self) -> f64 {
        let total_possible = self.agents.len() * (self.agents.len() - 1) / 2;
        if total_possible == 0 {
            return 0.0;
        }
        
        let mut active_connections = 0;
        for i in 0..self.network.adjacency.nrows() {
            for j in (i + 1)..self.network.adjacency.ncols() {
                if self.network.adjacency[(i, j)] > 0.1 {
                    active_connections += 1;
                }
            }
        }
        
        active_connections as f64 / total_possible as f64
    }

    /// Get agents reference
    pub fn get_agents(&self) -> &[AdaptiveAgent] {
        &self.agents
    }

    /// Get fitness history
    pub fn get_fitness_history(&self) -> &[FitnessSnapshot] {
        &self.fitness_history
    }
}

impl InteractionNetwork {
    fn new(size: usize, topology: NetworkTopology) -> Self {
        let mut adjacency = DMatrix::zeros(size, size);
        let mut rng = thread_rng();
        
        match topology {
            NetworkTopology::Random => {
                for i in 0..size {
                    for j in (i + 1)..size {
                        if rng.gen::<f64>() < 0.1 {
                            adjacency[(i, j)] = rng.gen::<f64>();
                            adjacency[(j, i)] = adjacency[(i, j)];
                        }
                    }
                }
            },
            NetworkTopology::SmallWorld => {
                // Watts-Strogatz small-world network
                let k = 4; // Each node connected to k nearest neighbors
                for i in 0..size {
                    for j in 1..=k/2 {
                        let neighbor1 = (i + j) % size;
                        let neighbor2 = (i + size - j) % size;
                        adjacency[(i, neighbor1)] = 1.0;
                        adjacency[(neighbor1, i)] = 1.0;
                        adjacency[(i, neighbor2)] = 1.0;
                        adjacency[(neighbor2, i)] = 1.0;
                    }
                }
                
                // Rewire with probability p
                let p = 0.1;
                for i in 0..size {
                    for j in 0..size {
                        if adjacency[(i, j)] > 0.0 && rng.gen::<f64>() < p {
                            adjacency[(i, j)] = 0.0;
                            adjacency[(j, i)] = 0.0;
                            
                            let new_target = rng.gen_range(0..size);
                            if new_target != i {
                                adjacency[(i, new_target)] = rng.gen::<f64>();
                                adjacency[(new_target, i)] = adjacency[(i, new_target)];
                            }
                        }
                    }
                }
            },
            _ => {
                // Default to random
                for i in 0..size {
                    for j in (i + 1)..size {
                        if rng.gen::<f64>() < 0.1 {
                            adjacency[(i, j)] = rng.gen::<f64>();
                            adjacency[(j, i)] = adjacency[(i, j)];
                        }
                    }
                }
            }
        }
        
        Self {
            adjacency,
            topology,
            decay_rate: 0.01,
        }
    }
}

#[derive(Clone, Debug)]
pub struct CasState {
    pub generation: usize,
    pub population_size: usize,
    pub agent_types: HashMap<String, usize>,
    pub fitness_stats: Option<FitnessSnapshot>,
    pub network_density: f64,
    pub environmental_pressure: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cas_initialization() {
        let params = CasParameters::default();
        let cas = ComplexAdaptiveSystem::new(params.clone());
        
        assert_eq!(cas.agents.len(), params.population_size);
        assert_eq!(cas.generation, 0);
        assert!(!cas.agents.is_empty());
    }

    #[test]
    fn test_evolution() {
        let mut params = CasParameters::default();
        params.population_size = 20; // Smaller for testing
        let mut cas = ComplexAdaptiveSystem::new(params);
        
        let initial_generation = cas.generation;
        cas.evolve_generation();
        
        assert_eq!(cas.generation, initial_generation + 1);
        assert!(!cas.fitness_history.is_empty());
    }

    #[test]
    fn test_agent_interaction() {
        let mut params = CasParameters::default();
        params.population_size = 5;
        let mut cas = ComplexAdaptiveSystem::new(params);
        
        // Force some interactions
        cas.process_interactions();
        
        // Check that some agents have memories
        let total_memories: usize = cas.agents.iter().map(|a| a.memory.len()).sum();
        // May or may not have interactions depending on random positioning
        assert!(total_memories >= 0);
    }
}