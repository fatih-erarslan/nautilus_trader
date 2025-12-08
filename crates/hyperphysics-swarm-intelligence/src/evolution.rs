//! Evolution Engine
//!
//! Records successful strategies and evolves them to create
//! new emergent behaviors and intellect.
//!
//! ## Concepts
//!
//! - **Genome**: Encoded strategy configuration + weights
//! - **Fitness**: Performance metrics across multiple objectives
//! - **Generation**: Population of genomes that compete and reproduce
//! - **Evolution**: Selection, crossover, mutation to improve over time

use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use rand::{Rng, SeedableRng};
use rand_chacha::ChaCha8Rng;
use rand_distr::{Distribution, Normal};
use uuid::Uuid;
use chrono::{DateTime, Utc};

use crate::{SwarmResult, SwarmIntelligenceError};
use crate::strategy::{StrategyType, StrategyConfig, StrategyResult};
use crate::topology::TopologyType;

/// A genome encoding a swarm strategy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Genome {
    /// Unique identifier
    pub id: Uuid,
    /// Parent IDs (for lineage tracking)
    pub parents: Vec<Uuid>,
    /// Generation number
    pub generation: usize,
    
    /// Strategy type weights (which strategies to use)
    pub strategy_weights: HashMap<StrategyType, f64>,
    /// Topology type weights
    pub topology_weights: HashMap<TopologyType, f64>,
    
    /// Strategy parameters
    pub parameters: HashMap<String, f64>,
    
    /// Combination rules (how to merge strategy outputs)
    pub combination_weights: Vec<f64>,
    
    /// Adaptation rates
    pub adaptation_rate: f64,
    pub mutation_rate: f64,
    
    /// Creation timestamp
    pub created_at: DateTime<Utc>,
    
    /// Fitness scores (multi-objective)
    pub fitness: Fitness,
    
    /// Usage count
    pub usage_count: usize,
    
    /// Success rate
    pub success_rate: f64,
}

impl Genome {
    /// Create a random genome
    pub fn random(rng: &mut impl Rng) -> Self {
        let mut strategy_weights = HashMap::new();
        for strategy in [
            StrategyType::ParticleSwarm,
            StrategyType::GreyWolf,
            StrategyType::WhaleOptimization,
            StrategyType::Firefly,
            StrategyType::Cuckoo,
            StrategyType::DifferentialEvolution,
            StrategyType::QuantumPSO,
        ] {
            strategy_weights.insert(strategy, rng.gen::<f64>());
        }
        
        let mut topology_weights = HashMap::new();
        for topology in [
            TopologyType::Star,
            TopologyType::Ring,
            TopologyType::Mesh,
            TopologyType::Hyperbolic,
            TopologyType::SmallWorld,
        ] {
            topology_weights.insert(topology, rng.gen::<f64>());
        }
        
        let mut parameters = HashMap::new();
        parameters.insert("inertia".to_string(), rng.gen_range(0.4..0.9));
        parameters.insert("cognitive".to_string(), rng.gen_range(1.0..2.5));
        parameters.insert("social".to_string(), rng.gen_range(1.0..2.5));
        parameters.insert("mutation_factor".to_string(), rng.gen_range(0.5..1.0));
        parameters.insert("crossover_rate".to_string(), rng.gen_range(0.6..0.95));
        parameters.insert("exploration_decay".to_string(), rng.gen_range(0.9..0.99));
        parameters.insert("temperature".to_string(), rng.gen_range(0.5..2.0));
        parameters.insert("lattice_coupling".to_string(), rng.gen_range(0.1..1.0));
        
        Self {
            id: Uuid::new_v4(),
            parents: Vec::new(),
            generation: 0,
            strategy_weights,
            topology_weights,
            parameters,
            combination_weights: (0..10).map(|_| rng.gen::<f64>()).collect(),
            adaptation_rate: rng.gen_range(0.01..0.1),
            mutation_rate: rng.gen_range(0.01..0.2),
            created_at: Utc::now(),
            fitness: Fitness::default(),
            usage_count: 0,
            success_rate: 0.0,
        }
    }
    
    /// Crossover with another genome
    pub fn crossover(&self, other: &Genome, rng: &mut impl Rng) -> Genome {
        let mut child = Genome::random(rng);
        child.parents = vec![self.id, other.id];
        child.generation = self.generation.max(other.generation) + 1;
        
        // Blend strategy weights
        for (strategy, &weight) in &self.strategy_weights {
            let other_weight = other.strategy_weights.get(strategy).copied().unwrap_or(0.5);
            let blend = rng.gen::<f64>();
            child.strategy_weights.insert(*strategy, blend * weight + (1.0 - blend) * other_weight);
        }
        
        // Blend topology weights
        for (topology, &weight) in &self.topology_weights {
            let other_weight = other.topology_weights.get(topology).copied().unwrap_or(0.5);
            let blend = rng.gen::<f64>();
            child.topology_weights.insert(*topology, blend * weight + (1.0 - blend) * other_weight);
        }
        
        // Blend parameters
        for (key, &value) in &self.parameters {
            let other_value = other.parameters.get(key).copied().unwrap_or(value);
            let blend = rng.gen::<f64>();
            child.parameters.insert(key.clone(), blend * value + (1.0 - blend) * other_value);
        }
        
        // Blend combination weights
        for i in 0..child.combination_weights.len().min(self.combination_weights.len()) {
            let blend = rng.gen::<f64>();
            let other_w = other.combination_weights.get(i).copied().unwrap_or(0.5);
            child.combination_weights[i] = blend * self.combination_weights[i] + (1.0 - blend) * other_w;
        }
        
        // Average adaptation rates
        child.adaptation_rate = (self.adaptation_rate + other.adaptation_rate) / 2.0;
        child.mutation_rate = (self.mutation_rate + other.mutation_rate) / 2.0;
        
        child
    }
    
    /// Mutate the genome
    pub fn mutate(&mut self, rng: &mut impl Rng) {
        let normal = Normal::new(0.0, 0.1).unwrap();
        
        // Mutate strategy weights
        for weight in self.strategy_weights.values_mut() {
            if rng.gen::<f64>() < self.mutation_rate {
                *weight = (*weight + normal.sample(rng)).clamp(0.0, 1.0);
            }
        }
        
        // Mutate topology weights
        for weight in self.topology_weights.values_mut() {
            if rng.gen::<f64>() < self.mutation_rate {
                *weight = (*weight + normal.sample(rng)).clamp(0.0, 1.0);
            }
        }
        
        // Mutate parameters
        for value in self.parameters.values_mut() {
            if rng.gen::<f64>() < self.mutation_rate {
                *value = (*value * (1.0 + normal.sample(rng))).max(0.001);
            }
        }
        
        // Mutate combination weights
        for weight in &mut self.combination_weights {
            if rng.gen::<f64>() < self.mutation_rate {
                *weight = (*weight + normal.sample(rng)).clamp(0.0, 1.0);
            }
        }
        
        // Self-adaptive mutation
        if rng.gen::<f64>() < 0.1 {
            self.mutation_rate = (self.mutation_rate * (1.0 + normal.sample(rng))).clamp(0.01, 0.5);
        }
    }
    
    /// Get dominant strategy
    pub fn dominant_strategy(&self) -> StrategyType {
        self.strategy_weights
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(s, _)| *s)
            .unwrap_or(StrategyType::ParticleSwarm)
    }
    
    /// Get dominant topology
    pub fn dominant_topology(&self) -> TopologyType {
        self.topology_weights
            .iter()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(t, _)| *t)
            .unwrap_or(TopologyType::Mesh)
    }
    
    /// Convert to strategy config
    pub fn to_strategy_config(&self, bounds: Vec<(f64, f64)>) -> StrategyConfig {
        StrategyConfig {
            strategy_type: self.dominant_strategy(),
            population_size: 50,
            max_iterations: 1000,
            bounds,
            params: self.parameters.clone(),
        }
    }
}

/// Multi-objective fitness scores
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Fitness {
    /// Primary objective value (lower is better)
    pub objective: f64,
    /// Convergence speed (iterations to threshold)
    pub convergence_speed: f64,
    /// Solution diversity maintained
    pub diversity: f64,
    /// Robustness across runs
    pub robustness: f64,
    /// Computational efficiency
    pub efficiency: f64,
    /// Combined weighted score
    pub combined: f64,
}

impl Fitness {
    /// Compute combined fitness with weights
    pub fn compute_combined(&mut self, weights: &[f64; 5]) {
        self.combined = weights[0] * (1.0 / (1.0 + self.objective))
            + weights[1] * (1.0 / (1.0 + self.convergence_speed))
            + weights[2] * self.diversity
            + weights[3] * self.robustness
            + weights[4] * self.efficiency;
    }
    
    /// Dominates another fitness (Pareto dominance)
    pub fn dominates(&self, other: &Fitness) -> bool {
        self.objective <= other.objective
            && self.convergence_speed <= other.convergence_speed
            && self.diversity >= other.diversity
            && self.robustness >= other.robustness
            && self.efficiency >= other.efficiency
            && (self.objective < other.objective
                || self.convergence_speed < other.convergence_speed
                || self.diversity > other.diversity
                || self.robustness > other.robustness
                || self.efficiency > other.efficiency)
    }
}

/// A generation of genomes
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Generation {
    /// Generation number
    pub number: usize,
    /// Population of genomes
    pub population: Vec<Genome>,
    /// Best genome
    pub best: Option<Genome>,
    /// Pareto front
    pub pareto_front: Vec<Genome>,
    /// Statistics
    pub stats: GenerationStats,
    /// Timestamp
    pub timestamp: DateTime<Utc>,
}

/// Generation statistics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct GenerationStats {
    pub min_fitness: f64,
    pub max_fitness: f64,
    pub mean_fitness: f64,
    pub std_fitness: f64,
    pub diversity: f64,
    pub improvement: f64,
}

/// Evolution engine configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionConfig {
    /// Population size
    pub population_size: usize,
    /// Number of generations
    pub max_generations: usize,
    /// Elite count (preserved each generation)
    pub elite_count: usize,
    /// Tournament size for selection
    pub tournament_size: usize,
    /// Crossover probability
    pub crossover_prob: f64,
    /// Mutation probability (per gene)
    pub mutation_prob: f64,
    /// Fitness weights [objective, speed, diversity, robustness, efficiency]
    pub fitness_weights: [f64; 5],
    /// Number of evaluations per genome
    pub evaluations_per_genome: usize,
}

impl Default for EvolutionConfig {
    fn default() -> Self {
        Self {
            population_size: 50,
            max_generations: 100,
            elite_count: 5,
            tournament_size: 3,
            crossover_prob: 0.8,
            mutation_prob: 0.1,
            fitness_weights: [0.4, 0.2, 0.15, 0.15, 0.1],
            evaluations_per_genome: 3,
        }
    }
}

/// Evolution engine for strategy evolution
pub struct EvolutionEngine {
    config: EvolutionConfig,
    population: Vec<Genome>,
    generation: usize,
    history: Vec<Generation>,
    rng: ChaCha8Rng,
    best_ever: Option<Genome>,
}

impl EvolutionEngine {
    /// Create new evolution engine
    pub fn new(config: EvolutionConfig) -> Self {
        let mut rng = ChaCha8Rng::from_entropy();
        
        let population: Vec<Genome> = (0..config.population_size)
            .map(|_| Genome::random(&mut rng))
            .collect();
        
        Self {
            config,
            population,
            generation: 0,
            history: Vec::new(),
            rng,
            best_ever: None,
        }
    }
    
    /// Run evolution with an evaluator function
    pub fn evolve<F>(&mut self, evaluator: F) -> SwarmResult<Genome>
    where
        F: Fn(&Genome) -> SwarmResult<Fitness>,
    {
        for gen in 0..self.config.max_generations {
            // Evaluate population
            for genome in &mut self.population {
                let mut total_fitness = Fitness::default();
                
                for _ in 0..self.config.evaluations_per_genome {
                    let fitness = evaluator(genome)?;
                    total_fitness.objective += fitness.objective;
                    total_fitness.convergence_speed += fitness.convergence_speed;
                    total_fitness.diversity += fitness.diversity;
                    total_fitness.robustness += fitness.robustness;
                    total_fitness.efficiency += fitness.efficiency;
                }
                
                let n = self.config.evaluations_per_genome as f64;
                total_fitness.objective /= n;
                total_fitness.convergence_speed /= n;
                total_fitness.diversity /= n;
                total_fitness.robustness /= n;
                total_fitness.efficiency /= n;
                total_fitness.compute_combined(&self.config.fitness_weights);
                
                genome.fitness = total_fitness;
                genome.usage_count += 1;
            }
            
            // Sort by combined fitness
            self.population.sort_by(|a, b| {
                b.fitness.combined.partial_cmp(&a.fitness.combined).unwrap()
            });
            
            // Update best ever
            if let Some(best) = self.population.first() {
                if self.best_ever.is_none() 
                    || best.fitness.combined > self.best_ever.as_ref().unwrap().fitness.combined 
                {
                    self.best_ever = Some(best.clone());
                }
            }
            
            // Record generation
            let stats = self.compute_stats();
            let pareto_front = self.compute_pareto_front();
            
            self.history.push(Generation {
                number: self.generation,
                population: self.population.clone(),
                best: self.population.first().cloned(),
                pareto_front,
                stats,
                timestamp: Utc::now(),
            });
            
            // Selection and reproduction
            let mut new_population = Vec::with_capacity(self.config.population_size);
            
            // Elite preservation
            for genome in self.population.iter().take(self.config.elite_count) {
                new_population.push(genome.clone());
            }
            
            // Fill rest with offspring
            while new_population.len() < self.config.population_size {
                let parent1 = self.tournament_select();
                let parent2 = self.tournament_select();
                
                let mut child = if self.rng.gen::<f64>() < self.config.crossover_prob {
                    parent1.crossover(&parent2, &mut self.rng)
                } else {
                    parent1.clone()
                };
                
                child.mutate(&mut self.rng);
                new_population.push(child);
            }
            
            self.population = new_population;
            self.generation += 1;
        }
        
        self.best_ever.clone().ok_or_else(|| {
            SwarmIntelligenceError::EvolutionError("No solution found".to_string())
        })
    }
    
    /// Tournament selection
    fn tournament_select(&mut self) -> Genome {
        let mut best: Option<&Genome> = None;
        
        for _ in 0..self.config.tournament_size {
            let idx = self.rng.gen_range(0..self.population.len());
            let candidate = &self.population[idx];
            
            if best.is_none() || candidate.fitness.combined > best.unwrap().fitness.combined {
                best = Some(candidate);
            }
        }
        
        best.unwrap().clone()
    }
    
    /// Compute generation statistics
    fn compute_stats(&self) -> GenerationStats {
        if self.population.is_empty() {
            return GenerationStats::default();
        }
        
        let fitnesses: Vec<f64> = self.population.iter()
            .map(|g| g.fitness.combined)
            .collect();
        
        let min_fitness = fitnesses.iter().cloned().fold(f64::INFINITY, f64::min);
        let max_fitness = fitnesses.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let mean_fitness = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
        
        let variance = fitnesses.iter()
            .map(|f| (f - mean_fitness).powi(2))
            .sum::<f64>() / fitnesses.len() as f64;
        let std_fitness = variance.sqrt();
        
        let improvement = if let Some(prev) = self.history.last() {
            mean_fitness - prev.stats.mean_fitness
        } else {
            0.0
        };
        
        GenerationStats {
            min_fitness,
            max_fitness,
            mean_fitness,
            std_fitness,
            diversity: std_fitness / (mean_fitness + 1e-10),
            improvement,
        }
    }
    
    /// Compute Pareto front
    fn compute_pareto_front(&self) -> Vec<Genome> {
        let mut front = Vec::new();
        
        for genome in &self.population {
            let mut dominated = false;
            
            for other in &self.population {
                if other.fitness.dominates(&genome.fitness) {
                    dominated = true;
                    break;
                }
            }
            
            if !dominated {
                front.push(genome.clone());
            }
        }
        
        front
    }
    
    /// Get evolution history
    pub fn history(&self) -> &[Generation] {
        &self.history
    }
    
    /// Get current best
    pub fn best(&self) -> Option<&Genome> {
        self.best_ever.as_ref()
    }
    
    /// Get current generation
    pub fn generation(&self) -> usize {
        self.generation
    }
    
    /// Save state to JSON
    pub fn save_state(&self) -> SwarmResult<String> {
        let state = EvolutionState {
            config: self.config.clone(),
            population: self.population.clone(),
            generation: self.generation,
            best_ever: self.best_ever.clone(),
            history_summary: self.history.iter().map(|g| GenerationSummary {
                number: g.number,
                best_fitness: g.best.as_ref().map(|b| b.fitness.combined).unwrap_or(0.0),
                mean_fitness: g.stats.mean_fitness,
                diversity: g.stats.diversity,
            }).collect(),
        };
        
        serde_json::to_string_pretty(&state)
            .map_err(|e| SwarmIntelligenceError::EvolutionError(e.to_string()))
    }
    
    /// Load state from JSON
    pub fn load_state(json: &str) -> SwarmResult<Self> {
        let state: EvolutionState = serde_json::from_str(json)
            .map_err(|e| SwarmIntelligenceError::EvolutionError(e.to_string()))?;
        
        Ok(Self {
            config: state.config,
            population: state.population,
            generation: state.generation,
            history: Vec::new(),
            rng: ChaCha8Rng::from_entropy(),
            best_ever: state.best_ever,
        })
    }
}

/// Serializable evolution state
#[derive(Debug, Clone, Serialize, Deserialize)]
struct EvolutionState {
    config: EvolutionConfig,
    population: Vec<Genome>,
    generation: usize,
    best_ever: Option<Genome>,
    history_summary: Vec<GenerationSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct GenerationSummary {
    number: usize,
    best_fitness: f64,
    mean_fitness: f64,
    diversity: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_genome_creation() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let genome = Genome::random(&mut rng);
        
        assert!(!genome.strategy_weights.is_empty());
        assert!(!genome.parameters.is_empty());
    }
    
    #[test]
    fn test_crossover() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let parent1 = Genome::random(&mut rng);
        let parent2 = Genome::random(&mut rng);
        
        let child = parent1.crossover(&parent2, &mut rng);
        
        assert_eq!(child.parents.len(), 2);
        assert_eq!(child.generation, 1);
    }
    
    #[test]
    fn test_mutation() {
        let mut rng = ChaCha8Rng::seed_from_u64(42);
        let mut genome = Genome::random(&mut rng);
        genome.mutation_rate = 1.0; // Force mutation
        
        let original = genome.clone();
        genome.mutate(&mut rng);
        
        // Something should have changed
        assert_ne!(
            format!("{:?}", original.strategy_weights),
            format!("{:?}", genome.strategy_weights)
        );
    }
}
