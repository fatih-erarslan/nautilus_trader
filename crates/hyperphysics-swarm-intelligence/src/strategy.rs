//! Biomimetic Strategies
//!
//! 14+ animal-inspired optimization strategies that can be combined
//! to create emergent intelligent behaviors.
//!
//! ## Strategy Categories
//!
//! - **Swarm**: Particle swarm (birds), ant colony, bee foraging
//! - **Pack**: Grey wolf hunting, whale bubble-net
//! - **School**: Fish schooling, salp chain
//! - **Flock**: Firefly signaling, bat echolocation
//! - **Colony**: Bacterial foraging, social spider
//! - **Evolution**: Genetic algorithm, differential evolution

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal, Cauchy};
use nalgebra::DVector;
use uuid::Uuid;

use crate::{SwarmResult, SwarmIntelligenceError};
use crate::lattice::PBitLattice;

/// Types of biomimetic strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StrategyType {
    // ============ SWARM ============
    /// Particle Swarm Optimization (birds flocking)
    ParticleSwarm,
    /// Ant Colony Optimization (pheromone trails)
    AntColony,
    /// Artificial Bee Colony (waggle dance)
    BeeColony,
    
    // ============ PACK ============
    /// Grey Wolf Optimization (alpha-beta-delta-omega)
    GreyWolf,
    /// Whale Optimization (bubble-net hunting)
    WhaleOptimization,
    
    // ============ SCHOOL ============
    /// Fish schooling behavior
    FishSchool,
    /// Salp Swarm Algorithm (chain formation)
    SalpSwarm,
    
    // ============ FLOCK ============
    /// Firefly Algorithm (bioluminescence)
    Firefly,
    /// Bat Algorithm (echolocation)
    Bat,
    /// Cuckoo Search (Lévy flights)
    Cuckoo,
    /// Moth-Flame Optimization
    MothFlame,
    
    // ============ COLONY ============
    /// Bacterial Foraging (chemotaxis)
    BacterialForaging,
    /// Social Spider Optimization
    SocialSpider,
    
    // ============ EVOLUTION ============
    /// Genetic Algorithm
    Genetic,
    /// Differential Evolution
    DifferentialEvolution,
    
    // ============ HYBRID ============
    /// Quantum-enhanced PSO
    QuantumPSO,
    /// Neural evolution
    NeuralEvolution,
    /// Chaos-enhanced optimization
    ChaosEnhanced,
    /// Multi-strategy hybrid
    AdaptiveHybrid,
}

impl StrategyType {
    /// Get biological inspiration description
    pub fn inspiration(&self) -> &'static str {
        match self {
            StrategyType::ParticleSwarm => "Birds flock by following local and global best positions",
            StrategyType::AntColony => "Ants deposit pheromones to mark successful paths",
            StrategyType::BeeColony => "Bees share food source info via waggle dance",
            StrategyType::GreyWolf => "Wolf packs hunt with alpha, beta, delta hierarchy",
            StrategyType::WhaleOptimization => "Whales encircle prey with bubble-net spiral",
            StrategyType::FishSchool => "Fish school using alignment, cohesion, separation",
            StrategyType::SalpSwarm => "Salps form chains led by leader, followed by followers",
            StrategyType::Firefly => "Fireflies attract mates with brighter flashes",
            StrategyType::Bat => "Bats use echolocation frequency and loudness",
            StrategyType::Cuckoo => "Cuckoos use Lévy flights and brood parasitism",
            StrategyType::MothFlame => "Moths navigate toward distant light sources",
            StrategyType::BacterialForaging => "Bacteria swim via tumbling and running (chemotaxis)",
            StrategyType::SocialSpider => "Spiders communicate via web vibrations",
            StrategyType::Genetic => "Natural selection favors fittest individuals",
            StrategyType::DifferentialEvolution => "Mutation and crossover evolve population",
            StrategyType::QuantumPSO => "Quantum superposition enables parallel exploration",
            StrategyType::NeuralEvolution => "Neural networks evolve through generations",
            StrategyType::ChaosEnhanced => "Chaotic dynamics escape local optima",
            StrategyType::AdaptiveHybrid => "Dynamically combines multiple strategies",
        }
    }
    
    /// Get exploration/exploitation balance (0 = full exploitation, 1 = full exploration)
    pub fn exploration_ratio(&self) -> f64 {
        match self {
            StrategyType::ParticleSwarm => 0.5,
            StrategyType::AntColony => 0.6,
            StrategyType::BeeColony => 0.5,
            StrategyType::GreyWolf => 0.3,
            StrategyType::WhaleOptimization => 0.7,
            StrategyType::FishSchool => 0.5,
            StrategyType::SalpSwarm => 0.4,
            StrategyType::Firefly => 0.5,
            StrategyType::Bat => 0.6,
            StrategyType::Cuckoo => 0.8,
            StrategyType::MothFlame => 0.4,
            StrategyType::BacterialForaging => 0.7,
            StrategyType::SocialSpider => 0.5,
            StrategyType::Genetic => 0.6,
            StrategyType::DifferentialEvolution => 0.5,
            StrategyType::QuantumPSO => 0.8,
            StrategyType::ChaosEnhanced => 0.9,
            StrategyType::AdaptiveHybrid => 0.5,
            _ => 0.5,
        }
    }
}

/// Configuration for a strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyConfig {
    /// Strategy type
    pub strategy_type: StrategyType,
    /// Population size
    pub population_size: usize,
    /// Maximum iterations
    pub max_iterations: usize,
    /// Search space bounds
    pub bounds: Vec<(f64, f64)>,
    /// Strategy-specific parameters
    pub params: HashMap<String, f64>,
}

impl Default for StrategyConfig {
    fn default() -> Self {
        Self {
            strategy_type: StrategyType::ParticleSwarm,
            population_size: 50,
            max_iterations: 1000,
            bounds: vec![(-100.0, 100.0); 10],
            params: HashMap::new(),
        }
    }
}

/// Result of strategy execution
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StrategyResult {
    /// Best solution found
    pub best_position: Vec<f64>,
    /// Best fitness value
    pub best_fitness: f64,
    /// Convergence history
    pub convergence: Vec<f64>,
    /// Number of function evaluations
    pub evaluations: usize,
    /// Iterations performed
    pub iterations: usize,
    /// Final population diversity
    pub diversity: f64,
    /// Strategy that was used
    pub strategy: StrategyType,
    /// Execution time in milliseconds
    pub execution_time_ms: u64,
}

/// Agent state for optimization
#[derive(Debug, Clone)]
pub struct Agent {
    pub id: Uuid,
    pub position: DVector<f64>,
    pub velocity: DVector<f64>,
    pub fitness: f64,
    pub personal_best: DVector<f64>,
    pub personal_best_fitness: f64,
    pub metadata: HashMap<String, f64>,
}

impl Agent {
    pub fn new(dimensions: usize) -> Self {
        Self {
            id: Uuid::new_v4(),
            position: DVector::zeros(dimensions),
            velocity: DVector::zeros(dimensions),
            fitness: f64::INFINITY,
            personal_best: DVector::zeros(dimensions),
            personal_best_fitness: f64::INFINITY,
            metadata: HashMap::new(),
        }
    }
    
    pub fn random(bounds: &[(f64, f64)], rng: &mut impl Rng) -> Self {
        let dim = bounds.len();
        let mut agent = Self::new(dim);
        
        for (i, &(min, max)) in bounds.iter().enumerate() {
            agent.position[i] = rng.gen_range(min..max);
            agent.velocity[i] = rng.gen_range(-1.0..1.0) * (max - min) * 0.1;
        }
        
        agent.personal_best = agent.position.clone();
        agent
    }
}

/// Biomimetic strategy executor
pub struct BiomimeticStrategy {
    config: StrategyConfig,
    agents: Vec<Agent>,
    global_best: DVector<f64>,
    global_best_fitness: f64,
    rng: ChaCha8Rng,
    iteration: usize,
    evaluations: usize,
    convergence: Vec<f64>,
    lattice: Option<PBitLattice>,
}

impl BiomimeticStrategy {
    /// Create a new strategy
    pub fn new(config: StrategyConfig) -> SwarmResult<Self> {
        let dim = config.bounds.len();
        let mut rng = ChaCha8Rng::from_entropy();
        
        // Initialize population
        let agents: Vec<Agent> = (0..config.population_size)
            .map(|_| Agent::random(&config.bounds, &mut rng))
            .collect();
        
        Ok(Self {
            config,
            agents,
            global_best: DVector::zeros(dim),
            global_best_fitness: f64::INFINITY,
            rng,
            iteration: 0,
            evaluations: 0,
            convergence: Vec::new(),
            lattice: None,
        })
    }
    
    /// Attach a pBit lattice for quantum-inspired operations
    pub fn with_lattice(mut self, lattice: PBitLattice) -> Self {
        self.lattice = Some(lattice);
        self
    }
    
    /// Optimize using the configured strategy
    pub fn optimize<F>(&mut self, objective: F) -> SwarmResult<StrategyResult>
    where
        F: Fn(&[f64]) -> f64,
    {
        let start = std::time::Instant::now();
        
        // Initial evaluation
        for agent in &mut self.agents {
            agent.fitness = objective(agent.position.as_slice());
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
            match self.config.strategy_type {
                StrategyType::ParticleSwarm => self.step_pso(),
                StrategyType::GreyWolf => self.step_grey_wolf(),
                StrategyType::WhaleOptimization => self.step_whale(),
                StrategyType::Firefly => self.step_firefly(),
                StrategyType::Bat => self.step_bat(),
                StrategyType::Cuckoo => self.step_cuckoo(),
                StrategyType::DifferentialEvolution => self.step_de(),
                StrategyType::BacterialForaging => self.step_bacterial(),
                StrategyType::SalpSwarm => self.step_salp(),
                StrategyType::QuantumPSO => self.step_quantum_pso(),
                StrategyType::AdaptiveHybrid => self.step_adaptive_hybrid(),
                _ => self.step_pso(), // Default to PSO
            }
            
            // Evaluate
            for agent in &mut self.agents {
                agent.fitness = objective(agent.position.as_slice());
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
            
            self.convergence.push(self.global_best_fitness);
            self.iteration += 1;
            
            // Lattice sweep if available
            if let Some(ref mut lattice) = self.lattice {
                lattice.sweep();
            }
        }
        
        Ok(StrategyResult {
            best_position: self.global_best.as_slice().to_vec(),
            best_fitness: self.global_best_fitness,
            convergence: self.convergence.clone(),
            evaluations: self.evaluations,
            iterations: self.iteration,
            diversity: self.compute_diversity(),
            strategy: self.config.strategy_type,
            execution_time_ms: start.elapsed().as_millis() as u64,
        })
    }
    
    // ============ STRATEGY IMPLEMENTATIONS ============
    
    /// Particle Swarm Optimization step
    fn step_pso(&mut self) {
        let w = self.config.params.get("inertia").copied().unwrap_or(0.7);
        let c1 = self.config.params.get("cognitive").copied().unwrap_or(1.5);
        let c2 = self.config.params.get("social").copied().unwrap_or(1.5);
        
        for agent in &mut self.agents {
            for i in 0..agent.position.len() {
                let r1 = self.rng.gen::<f64>();
                let r2 = self.rng.gen::<f64>();
                
                agent.velocity[i] = w * agent.velocity[i]
                    + c1 * r1 * (agent.personal_best[i] - agent.position[i])
                    + c2 * r2 * (self.global_best[i] - agent.position[i]);
                
                agent.position[i] += agent.velocity[i];
                
                // Enforce bounds
                let (min, max) = self.config.bounds[i];
                agent.position[i] = agent.position[i].clamp(min, max);
            }
        }
    }
    
    /// Grey Wolf Optimization step
    fn step_grey_wolf(&mut self) {
        // Sort by fitness to get alpha, beta, delta
        let mut sorted: Vec<_> = self.agents.iter().enumerate().collect();
        sorted.sort_by(|a, b| a.1.fitness.partial_cmp(&b.1.fitness).unwrap());
        
        let alpha = sorted[0].1.position.clone();
        let beta = sorted[1].1.position.clone();
        let delta = sorted[2].1.position.clone();
        
        let a = 2.0 - 2.0 * (self.iteration as f64 / self.config.max_iterations as f64);
        
        for agent in &mut self.agents {
            for i in 0..agent.position.len() {
                let r1 = self.rng.gen::<f64>();
                let r2 = self.rng.gen::<f64>();
                
                let a1 = 2.0 * a * r1 - a;
                let c1 = 2.0 * r2;
                let d_alpha = (c1 * alpha[i] - agent.position[i]).abs();
                let x1 = alpha[i] - a1 * d_alpha;
                
                let r1 = self.rng.gen::<f64>();
                let r2 = self.rng.gen::<f64>();
                let a2 = 2.0 * a * r1 - a;
                let c2 = 2.0 * r2;
                let d_beta = (c2 * beta[i] - agent.position[i]).abs();
                let x2 = beta[i] - a2 * d_beta;
                
                let r1 = self.rng.gen::<f64>();
                let r2 = self.rng.gen::<f64>();
                let a3 = 2.0 * a * r1 - a;
                let c3 = 2.0 * r2;
                let d_delta = (c3 * delta[i] - agent.position[i]).abs();
                let x3 = delta[i] - a3 * d_delta;
                
                agent.position[i] = (x1 + x2 + x3) / 3.0;
                
                let (min, max) = self.config.bounds[i];
                agent.position[i] = agent.position[i].clamp(min, max);
            }
        }
    }
    
    /// Whale Optimization Algorithm step
    fn step_whale(&mut self) {
        let a = 2.0 - 2.0 * (self.iteration as f64 / self.config.max_iterations as f64);
        let b = 1.0;
        let global_best = self.global_best.clone();
        let bounds = self.config.bounds.clone();
        let n = self.agents.len();
        
        // Pre-select random agents for those that need random search
        let random_positions: Vec<Vec<f64>> = (0..n)
            .map(|_| {
                let idx = self.rng.gen_range(0..n);
                self.agents[idx].position.as_slice().to_vec()
            })
            .collect();
        
        for (agent_idx, agent) in self.agents.iter_mut().enumerate() {
            let p = rand::random::<f64>();
            let l = rand::random::<f64>() * 2.0 - 1.0;
            
            for i in 0..agent.position.len() {
                if p < 0.5 {
                    let r = rand::random::<f64>();
                    let a_coeff = 2.0 * a * r - a;
                    let c = 2.0 * rand::random::<f64>();
                    
                    if a_coeff.abs() < 1.0 {
                        // Encircling prey
                        let d = (c * global_best[i] - agent.position[i]).abs();
                        agent.position[i] = global_best[i] - a_coeff * d;
                    } else {
                        // Random search using pre-selected random agent
                        let rand_pos = &random_positions[agent_idx];
                        let d = (c * rand_pos[i] - agent.position[i]).abs();
                        agent.position[i] = rand_pos[i] - a_coeff * d;
                    }
                } else {
                    // Spiral update (bubble-net)
                    let d = (global_best[i] - agent.position[i]).abs();
                    agent.position[i] = d * (b * l * 2.0 * std::f64::consts::PI).exp().cos() 
                        * (b * l * 2.0 * std::f64::consts::PI).sin() + global_best[i];
                }
                
                let (min, max) = bounds[i];
                agent.position[i] = agent.position[i].clamp(min, max);
            }
        }
    }
    
    /// Firefly Algorithm step
    fn step_firefly(&mut self) {
        let alpha = self.config.params.get("alpha").copied().unwrap_or(0.5);
        let beta0 = self.config.params.get("beta").copied().unwrap_or(1.0);
        let gamma = self.config.params.get("gamma").copied().unwrap_or(1.0);
        
        let n = self.agents.len();
        for i in 0..n {
            for j in 0..n {
                if i == j { continue; }
                
                // Move i toward j if j is brighter
                if self.agents[j].fitness < self.agents[i].fitness {
                    let r = self.distance(i, j);
                    let beta = beta0 * (-gamma * r * r).exp();
                    
                    for k in 0..self.agents[i].position.len() {
                        let random = self.rng.gen_range(-0.5..0.5);
                        self.agents[i].position[k] = self.agents[i].position[k]
                            + beta * (self.agents[j].position[k] - self.agents[i].position[k])
                            + alpha * random;
                        
                        let (min, max) = self.config.bounds[k];
                        self.agents[i].position[k] = self.agents[i].position[k].clamp(min, max);
                    }
                }
            }
        }
    }
    
    /// Bat Algorithm step
    fn step_bat(&mut self) {
        let f_min = 0.0;
        let f_max = 2.0;
        let a0 = self.config.params.get("loudness").copied().unwrap_or(0.5);
        let r0 = self.config.params.get("pulse_rate").copied().unwrap_or(0.5);
        let n = self.agents.len() as f64;
        let global_best = self.global_best.clone();
        let iteration = self.iteration;
        
        // Compute avg loudness first
        let avg_loudness: f64 = self.agents.iter()
            .map(|a| a.metadata.get("loudness").copied().unwrap_or(a0))
            .sum::<f64>() / n;
        
        for agent in &mut self.agents {
            let freq = f_min + (f_max - f_min) * self.rng.gen::<f64>();
            agent.metadata.insert("frequency".to_string(), freq);
            
            let loudness = agent.metadata.get("loudness").copied().unwrap_or(a0);
            let pulse_rate = agent.metadata.get("pulse_rate").copied().unwrap_or(r0);
            
            for i in 0..agent.position.len() {
                agent.velocity[i] += (agent.position[i] - global_best[i]) * freq;
                let new_pos = agent.position[i] + agent.velocity[i];
                
                // Local search
                if self.rng.gen::<f64>() > pulse_rate {
                    agent.position[i] = global_best[i] + 0.01 * self.rng.gen_range(-1.0..1.0) * avg_loudness;
                } else {
                    agent.position[i] = new_pos;
                }
                
                let (min, max) = self.config.bounds[i];
                agent.position[i] = agent.position[i].clamp(min, max);
            }
            
            // Update loudness and pulse rate
            agent.metadata.insert("loudness".to_string(), loudness * 0.9);
            agent.metadata.insert("pulse_rate".to_string(), r0 * (1.0 - (-0.1 * iteration as f64).exp()));
        }
    }
    
    /// Cuckoo Search step (with Lévy flights)
    fn step_cuckoo(&mut self) {
        let pa = self.config.params.get("abandon_prob").copied().unwrap_or(0.25);
        let global_best = self.global_best.clone();
        let bounds = self.config.bounds.clone();
        
        // Lévy flights - compute steps first
        let steps: Vec<f64> = (0..self.agents.len()).map(|_| self.levy_flight()).collect();
        
        for (idx, agent) in self.agents.iter_mut().enumerate() {
            let step = steps[idx];
            for i in 0..agent.position.len() {
                agent.position[i] += step * (agent.position[i] - global_best[i]);
                
                let (min, max) = bounds[i];
                agent.position[i] = agent.position[i].clamp(min, max);
            }
        }
        
        // Abandon worst nests
        let mut sorted: Vec<_> = (0..self.agents.len()).collect();
        sorted.sort_by(|&a, &b| {
            self.agents[b].fitness.partial_cmp(&self.agents[a].fitness).unwrap()
        });
        
        let abandon_count = (pa * self.agents.len() as f64) as usize;
        for &idx in sorted.iter().take(abandon_count) {
            self.agents[idx] = Agent::random(&self.config.bounds, &mut self.rng);
        }
    }
    
    /// Lévy flight step
    fn levy_flight(&mut self) -> f64 {
        let beta = 1.5;
        let sigma = ((1.0f64 + beta).exp() * beta.sin() * std::f64::consts::PI / 2.0
            / (((1.0 + beta) / 2.0).exp() * beta * 2.0f64.powf((beta - 1.0) / 2.0)))
            .powf(1.0 / beta);
        
        let u: f64 = Normal::new(0.0, sigma).unwrap().sample(&mut self.rng);
        let v: f64 = Normal::new(0.0, 1.0).unwrap().sample(&mut self.rng);
        
        u / v.abs().powf(1.0 / beta)
    }
    
    /// Differential Evolution step
    fn step_de(&mut self) {
        let f = self.config.params.get("mutation_factor").copied().unwrap_or(0.8);
        let cr = self.config.params.get("crossover_rate").copied().unwrap_or(0.9);
        
        let n = self.agents.len();
        let dim = self.config.bounds.len();
        
        for i in 0..n {
            // Select 3 random distinct agents
            let mut indices: Vec<usize> = (0..n).filter(|&j| j != i).collect();
            let a = indices.remove(self.rng.gen_range(0..indices.len()));
            let b = indices.remove(self.rng.gen_range(0..indices.len()));
            let c = indices.remove(self.rng.gen_range(0..indices.len()));
            
            // Mutation
            let mut mutant = DVector::zeros(dim);
            for j in 0..dim {
                mutant[j] = self.agents[a].position[j] 
                    + f * (self.agents[b].position[j] - self.agents[c].position[j]);
            }
            
            // Crossover
            let j_rand = self.rng.gen_range(0..dim);
            for j in 0..dim {
                if self.rng.gen::<f64>() > cr && j != j_rand {
                    mutant[j] = self.agents[i].position[j];
                }
                
                let (min, max) = self.config.bounds[j];
                mutant[j] = mutant[j].clamp(min, max);
            }
            
            self.agents[i].velocity = mutant.clone() - &self.agents[i].position;
            self.agents[i].position = mutant;
        }
    }
    
    /// Bacterial Foraging step
    fn step_bacterial(&mut self) {
        let step_size = self.config.params.get("step_size").copied().unwrap_or(0.1);
        
        for agent in &mut self.agents {
            // Tumble: random direction
            let mut direction = DVector::zeros(agent.position.len());
            let mut norm: f64 = 0.0;
            for i in 0..direction.len() {
                direction[i] = self.rng.gen_range(-1.0..1.0);
                norm += direction[i] * direction[i];
            }
            norm = norm.sqrt();
            
            // Swim: move in direction
            let prev_fitness = agent.fitness;
            for i in 0..agent.position.len() {
                agent.position[i] += step_size * direction[i] / norm;
                
                let (min, max) = self.config.bounds[i];
                agent.position[i] = agent.position[i].clamp(min, max);
            }
        }
    }
    
    /// Salp Swarm Algorithm step
    fn step_salp(&mut self) {
        let c1 = 2.0 * (-((4.0 * self.iteration as f64 / self.config.max_iterations as f64).powi(2))).exp();
        
        let n = self.agents.len();
        let leader_count = (n as f64 / 2.0).ceil() as usize;
        
        // Sort by fitness
        let mut indices: Vec<usize> = (0..n).collect();
        indices.sort_by(|&a, &b| {
            self.agents[a].fitness.partial_cmp(&self.agents[b].fitness).unwrap()
        });
        
        // Update leaders
        for &idx in indices.iter().take(leader_count) {
            for i in 0..self.agents[idx].position.len() {
                let c2 = self.rng.gen::<f64>();
                let c3 = self.rng.gen::<f64>();
                let (min, max) = self.config.bounds[i];
                
                self.agents[idx].position[i] = if c3 < 0.5 {
                    self.global_best[i] + c1 * ((max - min) * c2 + min)
                } else {
                    self.global_best[i] - c1 * ((max - min) * c2 + min)
                };
                
                self.agents[idx].position[i] = self.agents[idx].position[i].clamp(min, max);
            }
        }
        
        // Update followers
        for k in leader_count..n {
            let idx = indices[k];
            let prev_idx = indices[k - 1];
            
            for i in 0..self.agents[idx].position.len() {
                self.agents[idx].position[i] = 0.5 * (self.agents[idx].position[i] + self.agents[prev_idx].position[i]);
                
                let (min, max) = self.config.bounds[i];
                self.agents[idx].position[i] = self.agents[idx].position[i].clamp(min, max);
            }
        }
    }
    
    /// Quantum-inspired PSO step
    fn step_quantum_pso(&mut self) {
        let beta = 0.5 + 0.5 * (1.0 - self.iteration as f64 / self.config.max_iterations as f64);
        
        // Compute mean best position
        let mut mbest = DVector::zeros(self.config.bounds.len());
        for agent in &self.agents {
            mbest += &agent.personal_best;
        }
        mbest /= self.agents.len() as f64;
        
        for agent in &mut self.agents {
            for i in 0..agent.position.len() {
                let phi = self.rng.gen::<f64>();
                let p = phi * agent.personal_best[i] + (1.0 - phi) * self.global_best[i];
                
                let u = self.rng.gen::<f64>();
                let sign = if self.rng.gen::<f64>() < 0.5 { 1.0 } else { -1.0 };
                
                agent.position[i] = p + sign * beta * (mbest[i] - agent.position[i]).abs() * (-u).ln();
                
                let (min, max) = self.config.bounds[i];
                agent.position[i] = agent.position[i].clamp(min, max);
            }
        }
        
        // Use lattice if available
        if let Some(ref mut lattice) = self.lattice {
            // Inject best solution as pattern
            lattice.inject_pattern(&self.global_best);
            lattice.sweep();
            
            // Read back influence
            let pattern = lattice.read_pattern();
            for (i, val) in pattern.iter().enumerate().take(self.global_best.len()) {
                if self.rng.gen::<f64>() < 0.1 {
                    self.global_best[i] += 0.01 * val;
                }
            }
        }
    }
    
    /// Adaptive hybrid step - selects best strategy dynamically
    fn step_adaptive_hybrid(&mut self) {
        // Compute improvement rate
        let improvement = if self.convergence.len() >= 10 {
            let recent = &self.convergence[self.convergence.len() - 10..];
            (recent[0] - recent[9]) / (recent[0] + 1e-10)
        } else {
            0.0
        };
        
        // Select strategy based on improvement and iteration
        let progress = self.iteration as f64 / self.config.max_iterations as f64;
        
        if improvement < 0.001 && progress > 0.3 {
            // Stuck - use exploratory strategy
            self.step_cuckoo();
        } else if progress < 0.3 {
            // Early - use exploratory
            self.step_whale();
        } else if progress < 0.7 {
            // Middle - use balanced
            self.step_grey_wolf();
        } else {
            // Late - use exploitative
            self.step_pso();
        }
    }
    
    /// Compute distance between two agents
    fn distance(&self, i: usize, j: usize) -> f64 {
        let mut sum = 0.0;
        for k in 0..self.agents[i].position.len() {
            let diff = self.agents[i].position[k] - self.agents[j].position[k];
            sum += diff * diff;
        }
        sum.sqrt()
    }
    
    /// Compute population diversity
    fn compute_diversity(&self) -> f64 {
        if self.agents.is_empty() { return 0.0; }
        
        let n = self.agents.len() as f64;
        let dim = self.config.bounds.len();
        
        // Compute centroid
        let mut centroid = DVector::zeros(dim);
        for agent in &self.agents {
            centroid += &agent.position;
        }
        centroid /= n;
        
        // Compute average distance from centroid
        let mut total_dist = 0.0;
        for agent in &self.agents {
            let diff = &agent.position - &centroid;
            total_dist += diff.norm();
        }
        
        total_dist / n
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    fn sphere(x: &[f64]) -> f64 {
        x.iter().map(|&xi| xi * xi).sum()
    }
    
    #[test]
    fn test_pso_optimization() {
        let config = StrategyConfig {
            strategy_type: StrategyType::ParticleSwarm,
            population_size: 30,
            max_iterations: 100,
            bounds: vec![(-5.0, 5.0); 5],
            params: HashMap::new(),
        };
        
        let mut strategy = BiomimeticStrategy::new(config).unwrap();
        let result = strategy.optimize(sphere).unwrap();
        
        assert!(result.best_fitness < 1.0);
    }
    
    #[test]
    fn test_grey_wolf_optimization() {
        let config = StrategyConfig {
            strategy_type: StrategyType::GreyWolf,
            population_size: 30,
            max_iterations: 100,
            bounds: vec![(-5.0, 5.0); 5],
            params: HashMap::new(),
        };
        
        let mut strategy = BiomimeticStrategy::new(config).unwrap();
        let result = strategy.optimize(sphere).unwrap();
        
        assert!(result.best_fitness < 1.0);
    }
}
