//! Real Genetic Algorithm implementation for parasitic organism evolution
//! Sub-millisecond performance with zero mocks policy
//! Advanced selection, crossover, and mutation strategies

use std::sync::{Arc, Mutex, atomic::{AtomicU64, Ordering}};
use std::collections::HashMap;
use dashmap::DashMap;
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc};
use rand::{Rng, seq::SliceRandom, thread_rng};
use rand::distributions::{Distribution, Standard};
use rand_distr::Normal;
use tokio::sync::RwLock;
use rayon::prelude::*;

use crate::organisms::{ParasiticOrganism, OrganismGenetics, OrganismFactory, OrganismType};

/// Genetic Algorithm configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneticAlgorithmConfig {
    pub population_size: usize,
    pub elite_percentage: f64,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub selection_pressure: f64,
    pub max_generations: u64,
    pub convergence_threshold: f64,
    pub parallel_execution: bool,
    pub adaptive_parameters: bool,
}

impl Default for GeneticAlgorithmConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            elite_percentage: 0.1,
            mutation_rate: 0.1,
            crossover_rate: 0.8,
            selection_pressure: 1.2,
            max_generations: 1000,
            convergence_threshold: 0.001,
            parallel_execution: true,
            adaptive_parameters: true,
        }
    }
}

/// Evolution result for a single generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionResult {
    pub generation_completed: bool,
    pub operations_count: u64,
    pub selection_time_nanos: u64,
    pub crossover_time_nanos: u64,
    pub mutation_time_nanos: u64,
    pub total_time_nanos: u64,
    pub fitness_improvement: f64,
    pub genetic_diversity: f64,
}

/// Evolution statistics tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionStatistics {
    pub current_generation: u64,
    pub population_size: usize,
    pub average_fitness: f64,
    pub best_fitness: f64,
    pub worst_fitness: f64,
    pub genetic_diversity: f64,
    pub operations_performed: u64,
    pub total_evolution_time_ms: f64,
    pub convergence_progress: f64,
}

/// Real Genetic Algorithm implementation
pub struct GeneticAlgorithm {
    config: Arc<RwLock<GeneticAlgorithmConfig>>,
    current_generation: Arc<AtomicU64>,
    best_fitness_ever: Arc<Mutex<f64>>,
    convergence_reached: Arc<Mutex<bool>>,
    generations_without_improvement: Arc<AtomicU64>,
    performance_metrics: Arc<RwLock<HashMap<String, f64>>>,
    evolution_history: Arc<RwLock<Vec<EvolutionStatistics>>>,
    thread_pool: Option<rayon::ThreadPool>,
}

impl GeneticAlgorithm {
    pub fn new(config: GeneticAlgorithmConfig) -> Self {
        let thread_pool = if config.parallel_execution {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_cpus::get())
                .build()
                .ok()
        } else {
            None
        };
        
        Self {
            config: Arc::new(RwLock::new(config)),
            current_generation: Arc::new(AtomicU64::new(0)),
            best_fitness_ever: Arc::new(Mutex::new(0.0)),
            convergence_reached: Arc::new(Mutex::new(false)),
            generations_without_improvement: Arc::new(AtomicU64::new(0)),
            performance_metrics: Arc::new(RwLock::new(HashMap::new())),
            evolution_history: Arc::new(RwLock::new(Vec::new())),
            thread_pool,
        }
    }
    
    /// Main evolution cycle - optimized for sub-millisecond performance
    pub async fn evolve_population(
        &mut self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
    ) -> Result<EvolutionResult, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = std::time::Instant::now();
        
        if *self.convergence_reached.lock().unwrap() {
            return Ok(EvolutionResult {
                generation_completed: false,
                operations_count: 0,
                selection_time_nanos: 0,
                crossover_time_nanos: 0,
                mutation_time_nanos: 0,
                total_time_nanos: start_time.elapsed().as_nanos() as u64,
                fitness_improvement: 0.0,
                genetic_diversity: self.calculate_genetic_diversity(organisms).await,
            });
        }
        
        // Collect fitness scores for the current population
        let fitness_scores = self.collect_fitness_scores(organisms).await;
        if fitness_scores.is_empty() {
            return Err("No organisms in population".into());
        }
        
        let previous_best = *self.best_fitness_ever.lock().unwrap();
        
        // Selection phase
        let selection_start = std::time::Instant::now();
        let selected_parents = self.selection_phase(&fitness_scores).await?;
        let selection_time = selection_start.elapsed().as_nanos() as u64;
        
        // Crossover phase
        let crossover_start = std::time::Instant::now();
        let crossover_count = self.crossover_phase(organisms, &selected_parents).await?;
        let crossover_time = crossover_start.elapsed().as_nanos() as u64;
        
        // Mutation phase
        let mutation_start = std::time::Instant::now();
        let mutation_count = self.mutation_phase(organisms).await?;
        let mutation_time = mutation_start.elapsed().as_nanos() as u64;
        
        // Update generation and check for improvement
        let current_best = fitness_scores.iter()
            .map(|(_, fitness)| *fitness)
            .fold(f64::NEG_INFINITY, f64::max);
        
        {
            let mut best = self.best_fitness_ever.lock().unwrap();
            if current_best > *best {
                *best = current_best;
                self.generations_without_improvement.store(0, Ordering::SeqCst);
            } else {
                self.generations_without_improvement.fetch_add(1, Ordering::SeqCst);
            }
        }
        
        // Check convergence
        self.check_convergence(&fitness_scores).await;
        
        // Adaptive parameter adjustment
        let config = self.config.read().await;
        if config.adaptive_parameters {
            drop(config);
            self.adapt_parameters(&fitness_scores).await;
        }
        
        let generation = self.current_generation.fetch_add(1, Ordering::SeqCst) + 1;
        let total_time = start_time.elapsed().as_nanos() as u64;
        
        // Update performance metrics
        {
            let mut metrics = self.performance_metrics.write().await;
            metrics.insert("last_evolution_nanos".to_string(), total_time as f64);
            metrics.insert("selection_ratio".to_string(), selection_time as f64 / total_time as f64);
            metrics.insert("crossover_ratio".to_string(), crossover_time as f64 / total_time as f64);
            metrics.insert("mutation_ratio".to_string(), mutation_time as f64 / total_time as f64);
        }
        
        // Store evolution statistics
        let stats = self.calculate_evolution_statistics(organisms).await;
        self.evolution_history.write().await.push(stats);
        
        Ok(EvolutionResult {
            generation_completed: true,
            operations_count: crossover_count + mutation_count,
            selection_time_nanos: selection_time,
            crossover_time_nanos: crossover_time,
            mutation_time_nanos: mutation_time,
            total_time_nanos: total_time,
            fitness_improvement: current_best - previous_best,
            genetic_diversity: self.calculate_genetic_diversity(organisms).await,
        })
    }
    
    /// Tournament selection - real implementation with configurable pressure
    pub async fn tournament_selection(
        &self,
        fitness_scores: &[(Uuid, f64)],
        tournament_size: usize,
    ) -> Result<Vec<Uuid>, Box<dyn std::error::Error + Send + Sync>> {
        let config = self.config.read().await;
        let selection_count = (fitness_scores.len() as f64 * config.elite_percentage) as usize;
        let mut selected = Vec::with_capacity(selection_count);
        let mut rng = thread_rng();
        
        for _ in 0..selection_count {
            // Select tournament participants
            let tournament: Vec<&(Uuid, f64)> = fitness_scores
                .choose_multiple(&mut rng, tournament_size.min(fitness_scores.len()))
                .collect();
            
            // Select winner based on fitness and selection pressure
            if let Some(winner) = tournament.iter()
                .max_by(|a, b| {
                    let adjusted_fitness_a = a.1.powf(config.selection_pressure);
                    let adjusted_fitness_b = b.1.powf(config.selection_pressure);
                    adjusted_fitness_a.partial_cmp(&adjusted_fitness_b).unwrap()
                }) {
                selected.push(winner.0);
            }
        }
        
        Ok(selected)
    }
    
    /// Roulette wheel selection - real probabilistic implementation
    pub async fn roulette_wheel_selection(
        &self,
        fitness_scores: &[(Uuid, f64)],
    ) -> Result<Vec<Uuid>, Box<dyn std::error::Error + Send + Sync>> {
        let config = self.config.read().await;
        let selection_count = (fitness_scores.len() as f64 * config.elite_percentage) as usize;
        
        // Calculate total fitness (ensuring non-negative values)
        let min_fitness = fitness_scores.iter().map(|(_, f)| *f).fold(f64::INFINITY, f64::min);
        let offset = if min_fitness < 0.0 { -min_fitness + 0.01 } else { 0.0 };
        
        let adjusted_scores: Vec<(Uuid, f64)> = fitness_scores.iter()
            .map(|(id, fitness)| (*id, *fitness + offset))
            .collect();
        
        let total_fitness: f64 = adjusted_scores.iter().map(|(_, f)| *f).sum();
        
        if total_fitness <= 0.0 {
            // Fallback to random selection if no positive fitness
            let mut rng = thread_rng();
            return Ok(fitness_scores.choose_multiple(&mut rng, selection_count)
                .map(|(id, _)| *id)
                .collect());
        }
        
        let mut selected = Vec::with_capacity(selection_count);
        let mut rng = thread_rng();
        
        for _ in 0..selection_count {
            let spin = rng.gen::<f64>() * total_fitness;
            let mut cumulative = 0.0;
            
            for (id, fitness) in &adjusted_scores {
                cumulative += fitness;
                if cumulative >= spin {
                    selected.push(*id);
                    break;
                }
            }
        }
        
        Ok(selected)
    }
    
    /// Calculate genetic diversity using average pairwise distance
    pub async fn calculate_genetic_diversity(
        &self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
    ) -> f64 {
        if organisms.len() < 2 {
            return 0.0;
        }
        
        let genetics_vec: Vec<OrganismGenetics> = organisms.iter()
            .map(|entry| entry.value().get_genetics())
            .collect();
        
        if genetics_vec.len() < 2 {
            return 0.0;
        }
        
        // Calculate pairwise genetic distances
        let total_distance: f64 = genetics_vec.par_iter()
            .enumerate()
            .map(|(i, g1)| {
                genetics_vec.par_iter()
                    .skip(i + 1)
                    .map(|g2| self.genetic_distance(g1, g2))
                    .sum::<f64>()
            })
            .sum();
        
        let comparisons = (genetics_vec.len() * (genetics_vec.len() - 1)) / 2;
        if comparisons > 0 {
            total_distance / comparisons as f64
        } else {
            0.0
        }
    }
    
    /// Calculate genetic distance between two organisms
    fn genetic_distance(&self, g1: &OrganismGenetics, g2: &OrganismGenetics) -> f64 {
        let differences = [
            (g1.aggression - g2.aggression).abs(),
            (g1.adaptability - g2.adaptability).abs(),
            (g1.efficiency - g2.efficiency).abs(),
            (g1.resilience - g2.resilience).abs(),
            (g1.reaction_speed - g2.reaction_speed).abs(),
            (g1.risk_tolerance - g2.risk_tolerance).abs(),
            (g1.cooperation - g2.cooperation).abs(),
            (g1.stealth - g2.stealth).abs(),
        ];
        
        // Normalized Euclidean distance
        differences.iter().map(|d| d * d).sum::<f64>().sqrt() / 8.0_f64.sqrt()
    }
    
    /// Collect fitness scores from all organisms - optimized for speed
    async fn collect_fitness_scores(
        &self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
    ) -> Vec<(Uuid, f64)> {
        organisms.par_iter()
            .map(|entry| (*entry.key(), entry.value().fitness()))
            .collect()
    }
    
    /// Selection phase - choose parents for reproduction
    async fn selection_phase(
        &self,
        fitness_scores: &[(Uuid, f64)],
    ) -> Result<Vec<Uuid>, Box<dyn std::error::Error + Send + Sync>> {
        let config = self.config.read().await;
        
        // Use tournament selection by default, could be made configurable
        let tournament_size = (fitness_scores.len() / 10).max(3).min(7);
        self.tournament_selection(fitness_scores, tournament_size).await
    }
    
    /// Crossover phase - generate offspring from selected parents
    async fn crossover_phase(
        &self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
        selected_parents: &[Uuid],
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        if selected_parents.len() < 2 {
            return Ok(0);
        }
        
        let config = self.config.read().await;
        let mut crossover_count = 0;
        let mut rng = thread_rng();
        
        // Perform crossovers in parallel if enabled
        if config.parallel_execution && self.thread_pool.is_some() {
            let crossover_pairs = (selected_parents.len() / 2).max(1);
            let offspring: Result<Vec<_>, _> = (0..crossover_pairs)
                .into_par_iter()
                .map(|_| {
                    let parent1_id = selected_parents.choose(&mut thread_rng()).unwrap();
                    let parent2_id = selected_parents.choose(&mut thread_rng()).unwrap();
                    
                    if parent1_id != parent2_id {
                        if let (Some(p1), Some(p2)) = (organisms.get(parent1_id), organisms.get(parent2_id)) {
                            return self.perform_crossover(&**p1.value(), &**p2.value());
                        }
                    }
                    Err("Crossover failed".into())
                })
                .collect();
            
            match offspring {
                Ok(children) => {
                    for child in children.into_iter().flatten() {
                        let id = child.id();
                        organisms.insert(id, child);
                        crossover_count += 1;
                    }
                },
                Err(e) => return Err(e),
            }
        } else {
            // Sequential crossover
            let crossover_pairs = (selected_parents.len() / 2).max(1);
            
            for _ in 0..crossover_pairs {
                if rng.gen::<f64>() < config.crossover_rate {
                    let parent1_id = selected_parents.choose(&mut rng).unwrap();
                    let parent2_id = selected_parents.choose(&mut rng).unwrap();
                    
                    if parent1_id != parent2_id {
                        if let (Some(p1), Some(p2)) = (organisms.get(parent1_id), organisms.get(parent2_id)) {
                            if let Ok(Some(child)) = self.perform_crossover(&**p1.value(), &**p2.value()) {
                                let id = child.id();
                                organisms.insert(id, child);
                                crossover_count += 1;
                            }
                        }
                    }
                }
            }
        }
        
        Ok(crossover_count)
    }
    
    /// Perform single-point crossover between two parents
    fn perform_crossover(
        &self,
        parent1: &dyn ParasiticOrganism,
        parent2: &dyn ParasiticOrganism,
    ) -> Result<Option<Box<dyn ParasiticOrganism + Send + Sync>>, Box<dyn std::error::Error + Send + Sync>> {
        let g1 = parent1.get_genetics();
        let g2 = parent2.get_genetics();
        
        // Single-point crossover on genetic traits
        let crossover_point = thread_rng().gen_range(1..8);
        let mut child_genetics = g1.clone();
        
        let parent2_genes = [g2.aggression, g2.adaptability, g2.efficiency, g2.resilience,
                            g2.reaction_speed, g2.risk_tolerance, g2.cooperation, g2.stealth];
        let child_genes = [&mut child_genetics.aggression, &mut child_genetics.adaptability,
                          &mut child_genetics.efficiency, &mut child_genetics.resilience,
                          &mut child_genetics.reaction_speed, &mut child_genetics.risk_tolerance,
                          &mut child_genetics.cooperation, &mut child_genetics.stealth];
        
        for i in crossover_point..child_genes.len() {
            *child_genes[i] = parent2_genes[i];
        }
        
        // Create offspring of same type as parent1
        let mut offspring = OrganismFactory::create_organism(parent1.organism_type())?;
        offspring.set_genetics(child_genetics);
        
        Ok(Some(offspring))
    }
    
    /// Mutation phase - introduce genetic variation
    async fn mutation_phase(
        &self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
    ) -> Result<u64, Box<dyn std::error::Error + Send + Sync>> {
        let config = self.config.read().await;
        let organism_ids: Vec<Uuid> = organisms.iter().map(|entry| *entry.key()).collect();
        let expected_mutations = (organism_ids.len() as f64 * config.mutation_rate) as usize;
        
        let mut mutation_count = 0;
        let mut rng = thread_rng();
        
        // Apply mutations
        for _ in 0..expected_mutations {
            if let Some(&organism_id) = organism_ids.choose(&mut rng) {
                if let Some(mut organism_ref) = organisms.get_mut(&organism_id) {
                    let mut genetics = organism_ref.value().get_genetics();
                    
                    // Gaussian mutation on random trait
                    let trait_index = rng.gen_range(0..8);
                    let mutation_strength = 0.1;
                    let normal = Normal::new(0.0, mutation_strength).unwrap();
                    let mutation_delta = normal.sample(&mut rng);
                    
                    match trait_index {
                        0 => genetics.aggression = (genetics.aggression + mutation_delta).clamp(0.0, 1.0),
                        1 => genetics.adaptability = (genetics.adaptability + mutation_delta).clamp(0.0, 1.0),
                        2 => genetics.efficiency = (genetics.efficiency + mutation_delta).clamp(0.0, 1.0),
                        3 => genetics.resilience = (genetics.resilience + mutation_delta).clamp(0.0, 1.0),
                        4 => genetics.reaction_speed = (genetics.reaction_speed + mutation_delta).clamp(0.0, 1.0),
                        5 => genetics.risk_tolerance = (genetics.risk_tolerance + mutation_delta).clamp(0.0, 1.0),
                        6 => genetics.cooperation = (genetics.cooperation + mutation_delta).clamp(0.0, 1.0),
                        7 => genetics.stealth = (genetics.stealth + mutation_delta).clamp(0.0, 1.0),
                        _ => {}
                    }
                    
                    organism_ref.value_mut().set_genetics(genetics);
                    mutation_count += 1;
                }
            }
        }
        
        Ok(mutation_count)
    }
    
    /// Check convergence based on fitness variance and improvement stagnation
    async fn check_convergence(&self, fitness_scores: &[(Uuid, f64)]) {
        if fitness_scores.len() < 2 {
            return;
        }
        
        let config = self.config.read().await;
        let fitnesses: Vec<f64> = fitness_scores.iter().map(|(_, f)| *f).collect();
        let mean = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
        let variance = fitnesses.iter()
            .map(|f| (f - mean).powi(2))
            .sum::<f64>() / fitnesses.len() as f64;
        
        let stagnation_limit = 20; // Generations without improvement
        let is_converged = variance < config.convergence_threshold ||
                          self.generations_without_improvement.load(Ordering::SeqCst) >= stagnation_limit;
        
        *self.convergence_reached.lock().unwrap() = is_converged;
    }
    
    /// Adaptive parameter adjustment based on population dynamics
    async fn adapt_parameters(&self, fitness_scores: &[(Uuid, f64)]) {
        let mut config = self.config.write().await;
        
        if fitness_scores.len() < 2 {
            return;
        }
        
        let fitnesses: Vec<f64> = fitness_scores.iter().map(|(_, f)| *f).collect();
        let mean = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
        let variance = fitnesses.iter()
            .map(|f| (f - mean).powi(2))
            .sum::<f64>() / fitnesses.len() as f64;
        
        // Adjust mutation rate based on variance
        if variance < 0.01 {
            // Low diversity - increase mutation
            config.mutation_rate = (config.mutation_rate * 1.1).min(0.3);
        } else if variance > 0.1 {
            // High diversity - decrease mutation
            config.mutation_rate = (config.mutation_rate * 0.95).max(0.01);
        }
        
        // Adjust selection pressure based on improvement rate
        let stagnation = self.generations_without_improvement.load(Ordering::SeqCst);
        if stagnation > 5 {
            // Increase pressure to drive improvement
            config.selection_pressure = (config.selection_pressure * 1.05).min(3.0);
        } else if stagnation == 0 {
            // Good improvement - can reduce pressure
            config.selection_pressure = (config.selection_pressure * 0.98).max(1.0);
        }
    }
    
    /// Calculate comprehensive evolution statistics
    async fn calculate_evolution_statistics(
        &self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
    ) -> EvolutionStatistics {
        let fitness_scores = self.collect_fitness_scores(organisms).await;
        
        if fitness_scores.is_empty() {
            return EvolutionStatistics {
                current_generation: self.current_generation.load(Ordering::SeqCst),
                population_size: 0,
                average_fitness: 0.0,
                best_fitness: 0.0,
                worst_fitness: 0.0,
                genetic_diversity: 0.0,
                operations_performed: 0,
                total_evolution_time_ms: 0.0,
                convergence_progress: 0.0,
            };
        }
        
        let fitnesses: Vec<f64> = fitness_scores.iter().map(|(_, f)| *f).collect();
        let average_fitness = fitnesses.iter().sum::<f64>() / fitnesses.len() as f64;
        let best_fitness = fitnesses.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let worst_fitness = fitnesses.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        let genetic_diversity = self.calculate_genetic_diversity(organisms).await;
        
        // Calculate convergence progress
        let config = self.config.read().await;
        let best_ever = *self.best_fitness_ever.lock().unwrap();
        let convergence_progress = if let Some(target) = config.fitness_threshold {
            (best_ever / target).min(1.0)
        } else {
            genetic_diversity
        };
        
        let metrics = self.performance_metrics.read().await;
        let total_time = metrics.get("last_evolution_nanos").cloned().unwrap_or(0.0) / 1_000_000.0;
        
        EvolutionStatistics {
            current_generation: self.current_generation.load(Ordering::SeqCst),
            population_size: fitness_scores.len(),
            average_fitness,
            best_fitness,
            worst_fitness,
            genetic_diversity,
            operations_performed: self.current_generation.load(Ordering::SeqCst) * fitness_scores.len() as u64,
            total_evolution_time_ms: total_time,
            convergence_progress,
        }
    }
    
    // Public getters
    pub async fn get_config(&self) -> GeneticAlgorithmConfig {
        self.config.read().await.clone()
    }
    
    pub fn get_generation(&self) -> u64 {
        self.current_generation.load(Ordering::SeqCst)
    }
    
    pub fn has_converged(&self) -> bool {
        *self.convergence_reached.lock().unwrap()
    }
    
    pub async fn get_evolution_statistics(
        &self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
    ) -> EvolutionStatistics {
        self.calculate_evolution_statistics(organisms).await
    }
    
    pub async fn get_performance_metrics(&self) -> HashMap<String, f64> {
        self.performance_metrics.read().await.clone()
    }
    
    pub async fn get_evolution_history(&self) -> Vec<EvolutionStatistics> {
        self.evolution_history.read().await.clone()
    }
    
    /// Reset evolution state for new experiment
    pub async fn reset(&mut self) {
        self.current_generation.store(0, Ordering::SeqCst);
        *self.best_fitness_ever.lock().unwrap() = 0.0;
        *self.convergence_reached.lock().unwrap() = false;
        self.generations_without_improvement.store(0, Ordering::SeqCst);
        self.performance_metrics.write().await.clear();
        self.evolution_history.write().await.clear();
    }
}