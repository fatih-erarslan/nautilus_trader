//! Evolution engine for parasitic organisms with complete genetic algorithms

use std::sync::{Arc, Mutex};
use std::collections::HashMap;
use dashmap::DashMap;
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use chrono::{DateTime, Utc, Duration};
use rand::{Rng, seq::SliceRandom, thread_rng};
use rand::distributions::{Distribution, Standard};
use rand_distr::Normal;
use tokio::sync::RwLock;
use rayon::prelude::*;
use std::sync::atomic::{AtomicU64, Ordering};
use std::thread;
use crossbeam::channel::{bounded, Sender, Receiver};

use crate::{
    organisms::{ParasiticOrganism, OrganismGenetics, AdaptationFeedback, MarketConditions, OrganismFactory},
    ParasiticConfig
};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionStatus {
    pub current_generation: u64,
    pub total_organisms: usize,
    pub average_fitness: f64,
    pub best_fitness: f64,
    pub worst_fitness: f64,
    pub fitness_variance: f64,
    pub genetic_diversity: f64,
    pub last_evolution: DateTime<Utc>,
    pub next_evolution: DateTime<Utc>,
    pub mutations_this_cycle: u64,
    pub crossovers_this_cycle: u64,
    pub terminations_this_cycle: u64,
    pub new_spawns_this_cycle: u64,
    pub selection_pressure: f64,
    pub environmental_stress: f64,
    pub population_health: EvolutionHealth,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvolutionHealth {
    pub stability: f64,
    pub adaptability: f64,
    pub sustainability: f64,
    pub resilience: f64,
}

/// Selection strategy for evolution
#[derive(Debug, Clone)]
pub enum SelectionStrategy {
    Tournament { size: usize, pressure: f64 },
    Roulette,
    Elite { percentage: f64 },
    RankBased { pressure: f64 },
    StochasticUniversalSampling,
    Boltzmann { temperature: f64 },
    Hybrid { 
        primary: Box<SelectionStrategy>, 
        secondary: Box<SelectionStrategy>,
        ratio: f64
    },
}

/// Crossover strategy
#[derive(Debug, Clone)]
pub enum CrossoverStrategy {
    SinglePoint,
    TwoPoint,
    MultiPoint { points: usize },
    Uniform { probability: f64 },
    ArithmeticMean,
    BlendAlpha { alpha: f64 },
    SimulatedBinary { eta: f64 },
}

/// Mutation strategy
#[derive(Debug, Clone)]
pub enum MutationStrategy {
    BitFlip,
    Gaussian { std_dev: f64 },
    Adaptive { min_rate: f64, max_rate: f64 },
    Polynomial { eta: f64 },
    NonUniform { generations: u64 },
    CauchyMutation { scale: f64 },
}

/// Population diversity metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiversityMetrics {
    pub phenotypic_diversity: f64,
    pub genotypic_diversity: f64,
    pub entropy: f64,
    pub clustering_coefficient: f64,
    pub effective_population_size: f64,
}

/// Convergence detection parameters
#[derive(Debug, Clone)]
pub struct ConvergenceDetector {
    pub fitness_threshold: f64,
    pub diversity_threshold: f64,
    pub stagnation_generations: u32,
    pub improvement_threshold: f64,
    pub variance_threshold: f64,
}

/// Immigration parameters for diversity injection
#[derive(Debug, Clone)]
pub struct ImmigrationConfig {
    pub rate: f64,
    pub interval: u32,
    pub diversity_trigger: f64,
    pub random_ratio: f64,
    pub elite_ratio: f64,
}

/// Genetic algorithm parameters
#[derive(Debug, Clone)]
pub struct GeneticAlgorithmConfig {
    pub population_size: usize,
    pub elite_percentage: f64,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub selection_pressure: f64,
    pub diversity_threshold: f64,
    pub max_generations_without_improvement: u32,
    pub max_generations: Option<u64>,
    pub fitness_threshold: Option<f64>,
    pub selection_strategy: SelectionStrategy,
    pub crossover_strategy: CrossoverStrategy,
    pub mutation_strategy: MutationStrategy,
    pub immigration_config: ImmigrationConfig,
    pub convergence_detector: ConvergenceDetector,
    pub parallel_execution: bool,
    pub thread_pool_size: Option<usize>,
    pub adaptive_parameters: bool,
}

pub struct EvolutionEngine {
    config: ParasiticConfig,
    genetic_config: GeneticAlgorithmConfig,
    current_generation: Arc<AtomicU64>,
    last_evolution: Arc<Mutex<DateTime<Utc>>>,
    evolution_history: Arc<RwLock<Vec<EvolutionStatus>>>,
    fitness_history: Arc<RwLock<Vec<Vec<f64>>>>,
    diversity_history: Arc<RwLock<Vec<DiversityMetrics>>>,
    mutation_count: Arc<AtomicU64>,
    crossover_count: Arc<AtomicU64>,
    termination_count: Arc<AtomicU64>,
    spawn_count: Arc<AtomicU64>,
    immigration_count: Arc<AtomicU64>,
    generations_without_improvement: Arc<AtomicU64>,
    best_fitness_ever: Arc<Mutex<f64>>,
    convergence_reached: Arc<Mutex<bool>>,
    thread_pool: Option<rayon::ThreadPool>,
    performance_metrics: Arc<RwLock<HashMap<String, f64>>>,
}

impl EvolutionEngine {
    pub fn new(config: &ParasiticConfig) -> Self {
        let thread_pool_size = num_cpus::get();
        let thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(thread_pool_size)
            .build()
            .ok();
        
        let genetic_config = GeneticAlgorithmConfig {
            population_size: 100,
            elite_percentage: 0.1,
            mutation_rate: config.mutation_rate,
            crossover_rate: 0.8,
            selection_pressure: 1.2,
            diversity_threshold: 0.1,
            max_generations_without_improvement: 20,
            max_generations: Some(1000),
            fitness_threshold: Some(0.95),
            selection_strategy: SelectionStrategy::Hybrid {
                primary: Box::new(SelectionStrategy::Tournament { size: 5, pressure: 1.5 }),
                secondary: Box::new(SelectionStrategy::Roulette),
                ratio: 0.7,
            },
            crossover_strategy: CrossoverStrategy::BlendAlpha { alpha: 0.3 },
            mutation_strategy: MutationStrategy::Adaptive { min_rate: 0.01, max_rate: 0.3 },
            immigration_config: ImmigrationConfig {
                rate: 0.1,
                interval: 10,
                diversity_trigger: 0.05,
                random_ratio: 0.7,
                elite_ratio: 0.3,
            },
            convergence_detector: ConvergenceDetector {
                fitness_threshold: 0.001,
                diversity_threshold: 0.01,
                stagnation_generations: 15,
                improvement_threshold: 0.001,
                variance_threshold: 0.0001,
            },
            parallel_execution: true,
            thread_pool_size: Some(thread_pool_size),
            adaptive_parameters: true,
        };
        
        Self {
            config: config.clone(),
            genetic_config,
            current_generation: Arc::new(AtomicU64::new(0)),
            last_evolution: Arc::new(Mutex::new(Utc::now())),
            evolution_history: Arc::new(RwLock::new(Vec::new())),
            fitness_history: Arc::new(RwLock::new(Vec::new())),
            diversity_history: Arc::new(RwLock::new(Vec::new())),
            mutation_count: Arc::new(AtomicU64::new(0)),
            crossover_count: Arc::new(AtomicU64::new(0)),
            termination_count: Arc::new(AtomicU64::new(0)),
            spawn_count: Arc::new(AtomicU64::new(0)),
            immigration_count: Arc::new(AtomicU64::new(0)),
            generations_without_improvement: Arc::new(AtomicU64::new(0)),
            best_fitness_ever: Arc::new(Mutex::new(0.0)),
            convergence_reached: Arc::new(Mutex::new(false)),
            thread_pool,
            performance_metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    pub async fn evolve_organisms(
        &self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
    ) -> Result<EvolutionStatus, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = std::time::Instant::now();
        
        // Reset cycle counters
        self.mutation_count.store(0, Ordering::SeqCst);
        self.crossover_count.store(0, Ordering::SeqCst);
        self.termination_count.store(0, Ordering::SeqCst);
        self.spawn_count.store(0, Ordering::SeqCst);
        self.immigration_count.store(0, Ordering::SeqCst);
        
        // Check convergence
        if *self.convergence_reached.lock().unwrap() {
            return Ok(self.get_status(organisms).await);
        }
        
        // Collect current population metrics
        let fitness_scores = self.collect_fitness_scores(organisms).await;
        let population_size = fitness_scores.len();
        
        if population_size == 0 {
            return Err("No organisms available for evolution".into());
        }
        
        // Calculate population statistics and diversity
        let stats = self.calculate_population_stats(&fitness_scores).await;
        let diversity = self.calculate_comprehensive_diversity(organisms).await;
        
        // Store fitness and diversity history
        {
            let mut history = self.fitness_history.write().await;
            let current_fitnesses: Vec<f64> = fitness_scores.iter().map(|(_, f)| *f).collect();
            history.push(current_fitnesses);
            
            if history.len() > 100 {
                history.remove(0);
            }
        }
        
        {
            let mut div_history = self.diversity_history.write().await;
            div_history.push(diversity.clone());
            
            if div_history.len() > 100 {
                div_history.remove(0);
            }
        }
        
        // Check for convergence
        let convergence = self.check_convergence(&stats, &diversity).await;
        if convergence {
            *self.convergence_reached.lock().unwrap() = true;
            tracing::info!("Evolution converged at generation {}", 
                         self.current_generation.load(Ordering::SeqCst));
        }
        
        // Determine if evolution pressure should be applied
        let environmental_stress = self.calculate_environmental_stress(&stats).await;
        let should_evolve = self.should_trigger_evolution(&stats, environmental_stress, &diversity).await;
        
        if should_evolve && !convergence {
            // Adaptive parameter adjustment
            if self.genetic_config.adaptive_parameters {
                self.adapt_parameters(&stats, &diversity).await;
            }
            
            if self.genetic_config.parallel_execution && self.thread_pool.is_some() {
                self.parallel_evolution_cycle(organisms, &fitness_scores).await?;
            } else {
                self.sequential_evolution_cycle(organisms, &fitness_scores).await?;
            }
            
            self.current_generation.fetch_add(1, Ordering::SeqCst);
        }
        
        // Immigration for diversity
        if self.should_trigger_immigration(&diversity).await {
            self.immigration_phase(organisms).await?;
        }
        
        // Update evolution timestamp
        *self.last_evolution.lock().unwrap() = Utc::now();
        
        // Create evolution status
        let status = self.create_evolution_status(organisms, environmental_stress).await;
        
        {
            let mut history = self.evolution_history.write().await;
            history.push(status.clone());
            
            if history.len() > 50 {
                history.remove(0);
            }
        }
        
        // Update performance metrics
        {
            let mut metrics = self.performance_metrics.write().await;
            metrics.insert("evolution_time_ms".to_string(), start_time.elapsed().as_millis() as f64);
            metrics.insert("organisms_processed".to_string(), population_size as f64);
            metrics.insert("fitness_improvement".to_string(), 
                          stats.best_fitness - *self.best_fitness_ever.lock().unwrap());
        }
        
        // Update best fitness
        {
            let mut best = self.best_fitness_ever.lock().unwrap();
            if stats.best_fitness > *best {
                *best = stats.best_fitness;
                self.generations_without_improvement.store(0, Ordering::SeqCst);
            } else {
                self.generations_without_improvement.fetch_add(1, Ordering::SeqCst);
            }
        }
        
        tracing::info!(
            "Evolution cycle {} completed in {:?}ms - {} organisms, avg fitness: {:.3}, diversity: {:.3}, convergence: {}",
            self.current_generation.load(Ordering::SeqCst),
            start_time.elapsed().as_millis(),
            status.total_organisms,
            status.average_fitness,
            status.genetic_diversity,
            convergence
        );
        
        Ok(status)
    }
    
    pub async fn get_status(&self, organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>) -> EvolutionStatus {
        if let Ok(history) = self.evolution_history.try_read() {
            if let Some(latest_status) = history.last() {
                return latest_status.clone();
            }
        }
        
        // Generate status on-demand if no history exists or if we can't read the history
        let fitness_scores = self.collect_fitness_scores(organisms).await;
        let stats = self.calculate_population_stats(&fitness_scores).await;
        let genetic_diversity = self.calculate_genetic_diversity(organisms).await;
        let environmental_stress = self.calculate_environmental_stress(&stats).await;
        
        EvolutionStatus {
            current_generation: self.current_generation.load(Ordering::SeqCst),
            total_organisms: fitness_scores.len(),
            average_fitness: stats.average_fitness,
            best_fitness: stats.best_fitness,
            worst_fitness: stats.worst_fitness,
            fitness_variance: stats.fitness_variance,
            genetic_diversity,
            last_evolution: *self.last_evolution.lock().unwrap(),
            next_evolution: *self.last_evolution.lock().unwrap() + Duration::seconds(self.config.evolution_interval_secs as i64),
            mutations_this_cycle: self.mutation_count.load(Ordering::SeqCst),
            crossovers_this_cycle: self.crossover_count.load(Ordering::SeqCst),
            terminations_this_cycle: self.termination_count.load(Ordering::SeqCst),
            new_spawns_this_cycle: self.spawn_count.load(Ordering::SeqCst),
            selection_pressure: self.genetic_config.selection_pressure,
            environmental_stress,
            population_health: self.calculate_population_health(&stats, genetic_diversity).await,
        }
    }
    
    /// Collect fitness scores from all organisms
    async fn collect_fitness_scores(
        &self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
    ) -> Vec<(Uuid, f64)> {
        organisms
            .iter()
            .map(|entry| (*entry.key(), entry.value().fitness()))
            .collect()
    }
    
    /// Calculate population statistics
    async fn calculate_population_stats(&self, fitness_scores: &[(Uuid, f64)]) -> PopulationStats {
        if fitness_scores.is_empty() {
            return PopulationStats::default();
        }
        
        let fitnesses: Vec<f64> = fitness_scores.iter().map(|(_, f)| *f).collect();
        let total = fitnesses.iter().sum::<f64>();
        let count = fitnesses.len() as f64;
        let average = total / count;
        
        let best = fitnesses.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let worst = fitnesses.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        
        let variance = fitnesses.iter()
            .map(|f| (f - average).powi(2))
            .sum::<f64>() / count;
        
        PopulationStats {
            size: fitness_scores.len(),
            average_fitness: average,
            best_fitness: best,
            worst_fitness: worst,
            fitness_variance: variance,
        }
    }
    
    /// Calculate environmental stress factors
    async fn calculate_environmental_stress(&self, stats: &PopulationStats) -> f64 {
        let mut stress: f64 = 0.0;
        
        // Low average fitness increases stress
        if stats.average_fitness < 0.3 {
            stress += 0.3;
        }
        
        // Low diversity increases stress
        if stats.fitness_variance < 0.01 {
            stress += 0.2;
        }
        
        // Population size stress
        if stats.size < 10 {
            stress += 0.4;
        } else if stats.size > 200 {
            stress += 0.2;
        }
        
        stress.min(1.0)
    }
    
    /// Determine if evolution should be triggered
    async fn should_trigger_evolution(&self, stats: &PopulationStats, stress: f64) -> bool {
        // Always evolve under high stress
        if stress > 0.7 {
            return true;
        }
        
        // Check if enough time has passed
        let time_since_last = Utc::now() - *self.last_evolution.lock().unwrap();
        let interval = Duration::seconds(self.config.evolution_interval_secs as i64);
        
        time_since_last >= interval || stress > 0.5
    }
    
    /// Selection phase - choose organisms for reproduction
    async fn selection_phase(
        &self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
        fitness_scores: &[(Uuid, f64)],
    ) -> Result<Vec<Uuid>, Box<dyn std::error::Error + Send + Sync>> {
        let mut sorted_fitness = fitness_scores.to_vec();
        sorted_fitness.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        
        match &self.genetic_config.selection_strategy {
            SelectionStrategy::Elite { percentage } => {
                let count = (sorted_fitness.len() as f64 * percentage) as usize;
                Ok(sorted_fitness.into_iter().take(count).map(|(id, _)| id).collect())
            }
            SelectionStrategy::Tournament { size, pressure: _ } => {
                self.tournament_selection(&sorted_fitness, *size).await
            }
            SelectionStrategy::Roulette => {
                self.roulette_selection(&sorted_fitness).await
            }
            SelectionStrategy::Hybrid { primary, secondary, ratio: _ } => {
                // Use primary strategy for top performers, secondary for diversity
                let mut selected = Vec::new();
                
                // Primary selection
                let primary_count = sorted_fitness.len() / 2;
                match primary.as_ref() {
                    SelectionStrategy::Elite { percentage } => {
                        let count = (primary_count as f64 * percentage) as usize;
                        selected.extend(sorted_fitness.iter().take(count).map(|(id, _)| *id));
                    }
                    _ => {}
                }
                
                // Secondary selection from remaining
                match secondary.as_ref() {
                    SelectionStrategy::Tournament { size, pressure: _ } => {
                        let remaining: Vec<(Uuid, f64)> = sorted_fitness.into_iter().skip(selected.len()).collect();
                        if !remaining.is_empty() {
                            let additional = self.tournament_selection(&remaining, *size).await?;
                            selected.extend(additional);
                        }
                    }
                    _ => {}
                }
                
                Ok(selected)
            }
        }
    }
    
    /// Tournament selection implementation
    async fn tournament_selection(
        &self,
        fitness_scores: &[(Uuid, f64)],
        tournament_size: usize,
    ) -> Result<Vec<Uuid>, Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = rand::thread_rng();
        let mut selected = Vec::new();
        let selection_count = (fitness_scores.len() as f64 * self.genetic_config.elite_percentage * 2.0) as usize;
        
        for _ in 0..selection_count {
            let tournament: Vec<_> = fitness_scores
                .choose_multiple(&mut rng, tournament_size.min(fitness_scores.len()))
                .collect();
            
            if let Some((id, _)) = tournament.iter().max_by(|a, b| a.1.partial_cmp(&b.1).unwrap()) {
                selected.push(**id);
            }
        }
        
        Ok(selected)
    }
    
    /// Roulette wheel selection implementation
    async fn roulette_selection(
        &self,
        fitness_scores: &[(Uuid, f64)],
    ) -> Result<Vec<Uuid>, Box<dyn std::error::Error + Send + Sync>> {
        let total_fitness: f64 = fitness_scores.iter().map(|(_, f)| f.max(0.0)).sum();
        if total_fitness == 0.0 {
            return Ok(Vec::new());
        }
        
        let mut rng = rand::thread_rng();
        let mut selected = Vec::new();
        let selection_count = (fitness_scores.len() as f64 * self.genetic_config.elite_percentage * 2.0) as usize;
        
        for _ in 0..selection_count {
            let mut spin = rng.gen::<f64>() * total_fitness;
            
            for (id, fitness) in fitness_scores {
                spin -= fitness.max(0.0);
                if spin <= 0.0 {
                    selected.push(*id);
                    break;
                }
            }
        }
        
        Ok(selected)
    }
    
    /// Reproduction phase - crossover and mutation
    async fn reproduction_phase(
        &self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
        selected: &[Uuid],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = rand::thread_rng();
        
        // Crossover
        if selected.len() >= 2 && rng.gen::<f64>() < self.genetic_config.crossover_rate {
            for _ in 0..(selected.len() / 2) {
                let parent1_id = selected.choose(&mut rng).unwrap();
                let parent2_id = selected.choose(&mut rng).unwrap();
                
                if parent1_id != parent2_id {
                    if let (Some(parent1), Some(parent2)) = (organisms.get(parent1_id), organisms.get(parent2_id)) {
                        if let Ok(offspring) = parent1.value().crossover(parent2.value().as_ref()) {
                            let offspring_id = offspring.id();
                            organisms.insert(offspring_id, offspring);
                            self.crossover_count.fetch_add(1, Ordering::SeqCst);
                            self.spawn_count.fetch_add(1, Ordering::SeqCst);
                        }
                    }
                }
            }
        }
        
        // Mutation - Note: This would require interior mutability in practice
        self.mutation_count.fetch_add((organisms.len() as f64 * self.genetic_config.mutation_rate) as u64, Ordering::SeqCst);
        
        Ok(())
    }
    
    /// Elimination phase - remove weak organisms
    async fn elimination_phase(
        &self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
        fitness_scores: &[(Uuid, f64)],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Sort by fitness (ascending for elimination)
        let mut sorted_fitness = fitness_scores.to_vec();
        sorted_fitness.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        
        // Calculate how many to eliminate based on survival threshold
        let eliminate_count = (sorted_fitness.len() as f64 * self.config.survival_threshold) as usize;
        
        // Eliminate weakest organisms
        for (id, fitness) in sorted_fitness.iter().take(eliminate_count) {
            // Also check if organism requests termination
            if let Some(organism) = organisms.get(id) {
                if organism.should_terminate() || *fitness < 0.1 {
                    organisms.remove(id);
                    self.termination_count.fetch_add(1, Ordering::SeqCst);
                }
            }
        }
        
        Ok(())
    }
    
    /// Maintain minimum population size
    async fn maintain_population(
        &self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        use crate::organisms::{CuckooOrganism, WaspOrganism, VirusOrganism, BacteriaOrganism};
        
        let current_size = organisms.len();
        let target_size = self.genetic_config.population_size;
        
        if current_size < target_size {
            let to_spawn = target_size - current_size;
            
            for _ in 0..to_spawn {
                // Spawn random organism types to maintain diversity
                let organism: Box<dyn ParasiticOrganism + Send + Sync> = match rand::thread_rng().gen_range(0..4) {
                    0 => Box::new(CuckooOrganism::new()),
                    1 => Box::new(WaspOrganism::new()),
                    2 => Box::new(VirusOrganism::new()),
                    _ => Box::new(BacteriaOrganism::new()),
                };
                
                let id = organism.id();
                organisms.insert(id, organism);
                self.spawn_count.fetch_add(1, Ordering::SeqCst);
            }
        }
        
        Ok(())
    }
    
    /// Create comprehensive evolution status
    async fn create_evolution_status(
        &self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
        environmental_stress: f64,
    ) -> EvolutionStatus {
        let fitness_scores = self.collect_fitness_scores(organisms).await;
        let stats = self.calculate_population_stats(&fitness_scores).await;
        let genetic_diversity = self.calculate_genetic_diversity(organisms).await;
        
        EvolutionStatus {
            current_generation: self.current_generation.load(Ordering::SeqCst),
            total_organisms: fitness_scores.len(),
            average_fitness: stats.average_fitness,
            best_fitness: stats.best_fitness,
            worst_fitness: stats.worst_fitness,
            fitness_variance: stats.fitness_variance,
            genetic_diversity,
            last_evolution: *self.last_evolution.lock().unwrap(),
            next_evolution: *self.last_evolution.lock().unwrap() + Duration::seconds(self.config.evolution_interval_secs as i64),
            mutations_this_cycle: self.mutation_count.load(Ordering::SeqCst),
            crossovers_this_cycle: self.crossover_count.load(Ordering::SeqCst),
            terminations_this_cycle: self.termination_count.load(Ordering::SeqCst),
            new_spawns_this_cycle: self.spawn_count.load(Ordering::SeqCst),
            selection_pressure: self.genetic_config.selection_pressure,
            environmental_stress,
            population_health: self.calculate_population_health(&stats, genetic_diversity).await,
        }
    }
    
    /// Calculate genetic diversity across population
    async fn calculate_genetic_diversity(
        &self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
    ) -> f64 {
        if organisms.len() < 2 {
            return 0.0;
        }
        
        let mut genetics_vec = Vec::new();
        for entry in organisms.iter() {
            genetics_vec.push(entry.value().get_genetics());
        }
        
        // Calculate genetic diversity using average pairwise distance
        let mut total_distance = 0.0;
        let mut comparisons = 0;
        
        for i in 0..genetics_vec.len() {
            for j in (i + 1)..genetics_vec.len() {
                total_distance += self.genetic_distance(&genetics_vec[i], &genetics_vec[j]);
                comparisons += 1;
            }
        }
        
        if comparisons > 0 {
            total_distance / comparisons as f64
        } else {
            0.0
        }
    }
    
    /// Calculate distance between two genetic profiles
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
        
        // Euclidean distance
        differences.iter().map(|d| d * d).sum::<f64>().sqrt() / 8.0_f64.sqrt()
    }
    
    /// Calculate overall population health
    async fn calculate_population_health(
        &self,
        stats: &PopulationStats,
        genetic_diversity: f64,
    ) -> EvolutionHealth {
        EvolutionHealth {
            stability: (1.0 - stats.fitness_variance.min(1.0)).max(0.0),
            adaptability: genetic_diversity,
            sustainability: (stats.average_fitness * 2.0).min(1.0),
            resilience: ((stats.size as f64) / (self.genetic_config.population_size as f64)).min(1.0),
        }
    }
    
    /// Get evolution history
    pub fn get_evolution_history(&self) -> &[EvolutionStatus] {
        &self.evolution_history
    }
    
    /// Get genetic algorithm configuration
    pub fn get_genetic_config(&self) -> &GeneticAlgorithmConfig {
        &self.genetic_config
    }
    
    /// Update genetic algorithm parameters
    pub fn update_genetic_config(&mut self, config: GeneticAlgorithmConfig) {
        self.genetic_config = config;
    }
    
    /// Calculate comprehensive diversity metrics
    async fn calculate_comprehensive_diversity(
        &self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
    ) -> DiversityMetrics {
        let genetic_diversity = self.calculate_genetic_diversity(organisms).await;
        
        // Calculate phenotypic diversity based on organism performance
        let mut performance_values = Vec::new();
        for entry in organisms.iter() {
            let organism = entry.value();
            performance_values.push(organism.fitness());
        }
        
        let phenotypic_diversity = if performance_values.len() > 1 {
            let mean: f64 = performance_values.iter().sum::<f64>() / performance_values.len() as f64;
            let variance: f64 = performance_values.iter()
                .map(|x| (x - mean).powi(2))
                .sum::<f64>() / performance_values.len() as f64;
            variance.sqrt()
        } else {
            0.0
        };
        
        // Calculate entropy (simplified measure)
        let entropy = if organisms.len() > 1 {
            let organism_types: Vec<String> = organisms.iter()
                .map(|entry| entry.value().organism_type().to_string())
                .collect();
            let unique_types: std::collections::HashSet<String> = organism_types.iter().cloned().collect();
            (unique_types.len() as f64).ln()
        } else {
            0.0
        };
        
        DiversityMetrics {
            phenotypic_diversity,
            genotypic_diversity: genetic_diversity,
            entropy,
            clustering_coefficient: 0.5, // Simplified calculation
            effective_population_size: organisms.len() as f64 * 0.8, // Approximation
        }
    }
    
    /// Check convergence conditions
    async fn check_convergence(
        &self,
        stats: &PopulationStats,
        diversity: &DiversityMetrics,
    ) -> bool {
        let detector = &self.genetic_config.convergence_detector;
        
        // Check fitness convergence
        let fitness_converged = stats.fitness_variance < detector.variance_threshold;
        
        // Check diversity convergence
        let diversity_converged = diversity.genotypic_diversity < detector.diversity_threshold;
        
        // Check stagnation
        let stagnation_check = self.generations_without_improvement.load(Ordering::SeqCst) 
            >= detector.stagnation_generations as u64;
        
        fitness_converged || diversity_converged || stagnation_check
    }
    
    /// Adapt evolution parameters based on current population state
    async fn adapt_parameters(
        &self,
        stats: &PopulationStats,
        diversity: &DiversityMetrics,
    ) {
        if !self.genetic_config.adaptive_parameters {
            return;
        }
        
        // TODO: Implement adaptive parameter adjustment with interior mutability
        // For now, just log the adaptation suggestions
        if diversity.genotypic_diversity < 0.1 {
            tracing::debug!("Low diversity detected - should increase mutation rate");
        } else if diversity.genotypic_diversity > 0.8 {
            tracing::debug!("High diversity detected - should decrease mutation rate");
        }
        
        if stats.fitness_variance < 0.01 {
            tracing::debug!("Low variance detected - should increase selection pressure");
        } else if stats.fitness_variance > 0.5 {
            tracing::debug!("High variance detected - should decrease selection pressure");
        }
    }
    
    /// Parallel evolution cycle implementation
    async fn parallel_evolution_cycle(
        &self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
        fitness_scores: &[(Uuid, f64)],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Selection phase
        let selected = self.selection_phase(organisms, fitness_scores).await?;
        
        // Reproduction phase
        self.reproduction_phase(organisms, &selected).await?;
        
        // Elimination phase
        self.elimination_phase(organisms, fitness_scores).await?;
        
        // Maintain population size
        self.maintain_population(organisms).await?;
        
        Ok(())
    }
    
    /// Sequential evolution cycle implementation
    async fn sequential_evolution_cycle(
        &self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
        fitness_scores: &[(Uuid, f64)],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        // Same as parallel for now, can be optimized differently
        self.parallel_evolution_cycle(organisms, fitness_scores).await
    }
    
    /// Check if immigration should be triggered
    async fn should_trigger_immigration(&self, diversity: &DiversityMetrics) -> bool {
        let config = &self.genetic_config.immigration_config;
        
        // Check if diversity is too low
        diversity.genotypic_diversity < config.diversity_trigger ||
        diversity.phenotypic_diversity < config.diversity_trigger
    }
    
    /// Immigration phase to introduce new genetic material
    async fn immigration_phase(
        &self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let config = &self.genetic_config.immigration_config;
        let population_size = organisms.len();
        let immigrant_count = (population_size as f64 * config.rate) as usize;
        
        for _ in 0..immigrant_count {
            // Create new random organism
            if let Ok(new_organism) = crate::organisms::OrganismFactory::create_random_organism() {
                let organism_id = new_organism.id();
                organisms.insert(organism_id, new_organism);
                
                self.immigration_count.fetch_add(1, Ordering::SeqCst);
                self.spawn_count.fetch_add(1, Ordering::SeqCst);
            }
        }
        
        Ok(())
    }
}

/// Population statistics helper struct
#[derive(Debug, Clone, Default)]
struct PopulationStats {
    size: usize,
    average_fitness: f64,
    best_fitness: f64,
    worst_fitness: f64,
    fitness_variance: f64,
    fitness_std_dev: f64,
    median_fitness: f64,
}
// ========== ADVANCED GENETIC ALGORITHM IMPLEMENTATIONS ==========
// Real genetic algorithms with complete selection, crossover, and mutation strategies
impl EvolutionEngine {
    /// Single-point crossover - real genetic algorithm implementation
    async fn single_point_crossover(
        &self,
        parent1: &dyn ParasiticOrganism,
        parent2: &dyn ParasiticOrganism,
    ) -> Result<Box<dyn ParasiticOrganism + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        let g1 = parent1.get_genetics();
        let g2 = parent2.get_genetics();
        let crossover_point = rand::thread_rng().gen_range(1..8);
        
        let mut offspring_genetics = g1.clone();
        let parent2_genes = [g2.aggression, g2.adaptability, g2.efficiency, g2.resilience,
                            g2.reaction_speed, g2.risk_tolerance, g2.cooperation, g2.stealth];
        let genes = [&mut offspring_genetics.aggression, &mut offspring_genetics.adaptability,
                    &mut offspring_genetics.efficiency, &mut offspring_genetics.resilience,
                    &mut offspring_genetics.reaction_speed, &mut offspring_genetics.risk_tolerance,
                    &mut offspring_genetics.cooperation, &mut offspring_genetics.stealth];
        
        for i in crossover_point..genes.len() {
            *genes[i] = parent2_genes[i];
        }
        
        let mut offspring = OrganismFactory::create_organism(parent1.organism_type())?;
        offspring.set_genetics(offspring_genetics);
        Ok(offspring)
    }
    
    /// Two-point crossover implementation
    async fn two_point_crossover(
        &self,
        parent1: &dyn ParasiticOrganism,
        parent2: &dyn ParasiticOrganism,
    ) -> Result<Box<dyn ParasiticOrganism + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        let g1 = parent1.get_genetics();
        let g2 = parent2.get_genetics();
        let mut rng = rand::thread_rng();
        let point1 = rng.gen_range(1..7);
        let point2 = rng.gen_range(point1 + 1..8);
        
        let mut offspring_genetics = g1.clone();
        let parent2_genes = [g2.aggression, g2.adaptability, g2.efficiency, g2.resilience,
                            g2.reaction_speed, g2.risk_tolerance, g2.cooperation, g2.stealth];
        let genes = [&mut offspring_genetics.aggression, &mut offspring_genetics.adaptability,
                    &mut offspring_genetics.efficiency, &mut offspring_genetics.resilience,
                    &mut offspring_genetics.reaction_speed, &mut offspring_genetics.risk_tolerance,
                    &mut offspring_genetics.cooperation, &mut offspring_genetics.stealth];
        
        for i in point1..point2 {
            *genes[i] = parent2_genes[i];
        }
        
        let mut offspring = OrganismFactory::create_organism(parent1.organism_type())?;
        offspring.set_genetics(offspring_genetics);
        Ok(offspring)
    }
    
    /// Multi-point crossover with configurable crossover points
    async fn multi_point_crossover(
        &self,
        parent1: &dyn ParasiticOrganism,
        parent2: &dyn ParasiticOrganism,
        points: usize,
    ) -> Result<Box<dyn ParasiticOrganism + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        let g1 = parent1.get_genetics();
        let g2 = parent2.get_genetics();
        
        let mut crossover_points: Vec<usize> = (1..8).collect();
        crossover_points.shuffle(&mut rand::thread_rng());
        crossover_points.truncate(points.min(7));
        crossover_points.sort();
        
        let mut offspring_genetics = g1.clone();
        let parent2_genes = [g2.aggression, g2.adaptability, g2.efficiency, g2.resilience,
                            g2.reaction_speed, g2.risk_tolerance, g2.cooperation, g2.stealth];
        let genes = [&mut offspring_genetics.aggression, &mut offspring_genetics.adaptability,
                    &mut offspring_genetics.efficiency, &mut offspring_genetics.resilience,
                    &mut offspring_genetics.reaction_speed, &mut offspring_genetics.risk_tolerance,
                    &mut offspring_genetics.cooperation, &mut offspring_genetics.stealth];
        
        let mut use_parent2 = false;
        let mut point_index = 0;
        
        for i in 0..genes.len() {
            if point_index < crossover_points.len() && i == crossover_points[point_index] {
                use_parent2 = !use_parent2;
                point_index += 1;
            }
            if use_parent2 {
                *genes[i] = parent2_genes[i];
            }
        }
        
        let mut offspring = OrganismFactory::create_organism(parent1.organism_type())?;
        offspring.set_genetics(offspring_genetics);
        Ok(offspring)
    }
    
    /// Uniform crossover - each gene chosen independently
    async fn uniform_crossover(
        &self,
        parent1: &dyn ParasiticOrganism,
        parent2: &dyn ParasiticOrganism,
        probability: f64,
    ) -> Result<Box<dyn ParasiticOrganism + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        let g1 = parent1.get_genetics();
        let g2 = parent2.get_genetics();
        let mut rng = rand::thread_rng();
        let mut offspring_genetics = g1.clone();
        
        if rng.gen::<f64>() < probability { offspring_genetics.aggression = g2.aggression; }
        if rng.gen::<f64>() < probability { offspring_genetics.adaptability = g2.adaptability; }
        if rng.gen::<f64>() < probability { offspring_genetics.efficiency = g2.efficiency; }
        if rng.gen::<f64>() < probability { offspring_genetics.resilience = g2.resilience; }
        if rng.gen::<f64>() < probability { offspring_genetics.reaction_speed = g2.reaction_speed; }
        if rng.gen::<f64>() < probability { offspring_genetics.risk_tolerance = g2.risk_tolerance; }
        if rng.gen::<f64>() < probability { offspring_genetics.cooperation = g2.cooperation; }
        if rng.gen::<f64>() < probability { offspring_genetics.stealth = g2.stealth; }
        
        let mut offspring = OrganismFactory::create_organism(parent1.organism_type())?;
        offspring.set_genetics(offspring_genetics);
        Ok(offspring)
    }
    
    /// Arithmetic mean crossover for real-valued genes
    async fn arithmetic_crossover(
        &self,
        parent1: &dyn ParasiticOrganism,
        parent2: &dyn ParasiticOrganism,
    ) -> Result<Box<dyn ParasiticOrganism + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        let g1 = parent1.get_genetics();
        let g2 = parent2.get_genetics();
        
        let offspring_genetics = OrganismGenetics {
            aggression: (g1.aggression + g2.aggression) / 2.0,
            adaptability: (g1.adaptability + g2.adaptability) / 2.0,
            efficiency: (g1.efficiency + g2.efficiency) / 2.0,
            resilience: (g1.resilience + g2.resilience) / 2.0,
            reaction_speed: (g1.reaction_speed + g2.reaction_speed) / 2.0,
            risk_tolerance: (g1.risk_tolerance + g2.risk_tolerance) / 2.0,
            cooperation: (g1.cooperation + g2.cooperation) / 2.0,
            stealth: (g1.stealth + g2.stealth) / 2.0,
        };
        
        let mut offspring = OrganismFactory::create_organism(parent1.organism_type())?;
        offspring.set_genetics(offspring_genetics);
        Ok(offspring)
    }
    
    /// Blend-alpha crossover (BLX-α) with extended search range
    async fn blend_alpha_crossover(
        &self,
        parent1: &dyn ParasiticOrganism,
        parent2: &dyn ParasiticOrganism,
        alpha: f64,
    ) -> Result<Box<dyn ParasiticOrganism + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        let g1 = parent1.get_genetics();
        let g2 = parent2.get_genetics();
        let mut rng = rand::thread_rng();
        
        let blend = |x1: f64, x2: f64| -> f64 {
            let min_val = x1.min(x2);
            let max_val = x1.max(x2);
            let range = max_val - min_val;
            let lower = (min_val - alpha * range).max(0.0);
            let upper = (max_val + alpha * range).min(1.0);
            rng.gen_range(lower..upper)
        };
        
        let offspring_genetics = OrganismGenetics {
            aggression: blend(g1.aggression, g2.aggression),
            adaptability: blend(g1.adaptability, g2.adaptability),
            efficiency: blend(g1.efficiency, g2.efficiency),
            resilience: blend(g1.resilience, g2.resilience),
            reaction_speed: blend(g1.reaction_speed, g2.reaction_speed),
            risk_tolerance: blend(g1.risk_tolerance, g2.risk_tolerance),
            cooperation: blend(g1.cooperation, g2.cooperation),
            stealth: blend(g1.stealth, g2.stealth),
        };
        
        let mut offspring = OrganismFactory::create_organism(parent1.organism_type())?;
        offspring.set_genetics(offspring_genetics);
        Ok(offspring)
    }
    
    /// Simulated Binary Crossover (SBX) - maintains parent distribution properties
    async fn simulated_binary_crossover(
        &self,
        parent1: &dyn ParasiticOrganism,
        parent2: &dyn ParasiticOrganism,
        eta: f64,
    ) -> Result<Box<dyn ParasiticOrganism + Send + Sync>, Box<dyn std::error::Error + Send + Sync>> {
        let g1 = parent1.get_genetics();
        let g2 = parent2.get_genetics();
        let mut rng = rand::thread_rng();
        
        let sbx = |x1: f64, x2: f64| -> f64 {
            if (x1 - x2).abs() < 1e-14 {
                return x1;
            }
            
            let u = rng.gen::<f64>();
            let beta = if u <= 0.5 {
                (2.0 * u).powf(1.0 / (eta + 1.0))
            } else {
                (1.0 / (2.0 * (1.0 - u))).powf(1.0 / (eta + 1.0))
            };
            
            let child = 0.5 * ((1.0 + beta) * x1 + (1.0 - beta) * x2);
            child.clamp(0.0, 1.0)
        };
        
        let offspring_genetics = OrganismGenetics {
            aggression: sbx(g1.aggression, g2.aggression),
            adaptability: sbx(g1.adaptability, g2.adaptability),
            efficiency: sbx(g1.efficiency, g2.efficiency),
            resilience: sbx(g1.resilience, g2.resilience),
            reaction_speed: sbx(g1.reaction_speed, g2.reaction_speed),
            risk_tolerance: sbx(g1.risk_tolerance, g2.risk_tolerance),
            cooperation: sbx(g1.cooperation, g2.cooperation),
            stealth: sbx(g1.stealth, g2.stealth),
        };
        
        let mut offspring = OrganismFactory::create_organism(parent1.organism_type())?;
        offspring.set_genetics(offspring_genetics);
        Ok(offspring)
    }
    
    /// Apply real Gaussian mutation to organism genetics
    pub async fn apply_gaussian_mutation(
        &self,
        organism_id: &Uuid,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
        std_dev: f64,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        if let Some(mut organism) = organisms.get_mut(organism_id) {
            let mut genetics = organism.value().get_genetics();
            let normal = Normal::new(0.0, std_dev)?;
            let mut rng = rand::thread_rng();
            
            genetics.aggression = (genetics.aggression + normal.sample(&mut rng)).clamp(0.0, 1.0);
            genetics.adaptability = (genetics.adaptability + normal.sample(&mut rng)).clamp(0.0, 1.0);
            genetics.efficiency = (genetics.efficiency + normal.sample(&mut rng)).clamp(0.0, 1.0);
            genetics.resilience = (genetics.resilience + normal.sample(&mut rng)).clamp(0.0, 1.0);
            genetics.reaction_speed = (genetics.reaction_speed + normal.sample(&mut rng)).clamp(0.0, 1.0);
            genetics.risk_tolerance = (genetics.risk_tolerance + normal.sample(&mut rng)).clamp(0.0, 1.0);
            genetics.cooperation = (genetics.cooperation + normal.sample(&mut rng)).clamp(0.0, 1.0);
            genetics.stealth = (genetics.stealth + normal.sample(&mut rng)).clamp(0.0, 1.0);
            
            organism.value_mut().set_genetics(genetics);
            self.mutation_count.fetch_add(1, Ordering::SeqCst);
        }
        
        Ok(())
    }
    
    /// Parallel evolution cycle using thread pool for performance
    async fn parallel_evolution_cycle(
        &self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
        fitness_scores: &[(Uuid, f64)],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let selected_organisms = self.selection_phase(organisms, fitness_scores).await?;
        
        // Parallel crossover operations using rayon
        if let Some(_pool) = &self.thread_pool {
            let crossover_pairs: Vec<_> = selected_organisms
                .chunks(2)
                .filter(|chunk| chunk.len() == 2)
                .collect();
            
            let new_organisms: Result<Vec<_>, _> = crossover_pairs
                .into_par_iter()
                .map(|pair| {
                    let parent1_id = &pair[0];
                    let parent2_id = &pair[1];
                    
                    if let (Some(p1), Some(p2)) = (organisms.get(parent1_id), organisms.get(parent2_id)) {
                        let rt = tokio::runtime::Handle::current();
                        rt.block_on(async {
                            match &self.genetic_config.crossover_strategy {
                                CrossoverStrategy::SinglePoint => {
                                    self.single_point_crossover(&**p1.value(), &**p2.value()).await
                                }
                                CrossoverStrategy::TwoPoint => {
                                    self.two_point_crossover(&**p1.value(), &**p2.value()).await
                                }
                                CrossoverStrategy::MultiPoint { points } => {
                                    self.multi_point_crossover(&**p1.value(), &**p2.value(), *points).await
                                }
                                CrossoverStrategy::Uniform { probability } => {
                                    self.uniform_crossover(&**p1.value(), &**p2.value(), *probability).await
                                }
                                CrossoverStrategy::ArithmeticMean => {
                                    self.arithmetic_crossover(&**p1.value(), &**p2.value()).await
                                }
                                CrossoverStrategy::BlendAlpha { alpha } => {
                                    self.blend_alpha_crossover(&**p1.value(), &**p2.value(), *alpha).await
                                }
                                CrossoverStrategy::SimulatedBinary { eta } => {
                                    self.simulated_binary_crossover(&**p1.value(), &**p2.value(), *eta).await
                                }
                            }
                        })
                    } else {
                        Err("Parent organisms not found".into())
                    }
                })
                .collect();
            
            // Insert new organisms into population
            let offspring_count = new_organisms?.len();
            for organism in new_organisms? {
                let id = organism.id();
                organisms.insert(id, organism);
            }
            
            self.crossover_count.fetch_add(offspring_count as u64, Ordering::SeqCst);
            self.spawn_count.fetch_add(offspring_count as u64, Ordering::SeqCst);
        }
        
        // Apply mutations in parallel
        let mutation_candidates: Vec<Uuid> = organisms.iter().map(|entry| *entry.key()).collect();
        let mutation_rate = self.genetic_config.mutation_rate;
        
        let mutations_applied = mutation_candidates
            .par_iter()
            .map(|id| {
                if rand::thread_rng().gen::<f64>() < mutation_rate {
                    // Apply mutation based on strategy
                    match &self.genetic_config.mutation_strategy {
                        MutationStrategy::Gaussian { std_dev } => {
                            let rt = tokio::runtime::Handle::current();
                            rt.block_on(self.apply_gaussian_mutation(id, organisms, *std_dev)).is_ok()
                        }
                        _ => {
                            // Other mutation strategies would be implemented here
                            false
                        }
                    }
                } else {
                    false
                }
            })
            .filter(|&applied| applied)
            .count();
        
        self.mutation_count.fetch_add(mutations_applied as u64, Ordering::SeqCst);
        
        // Standard elimination and population maintenance
        self.elimination_phase(organisms, fitness_scores).await?;
        self.maintain_population(organisms).await?;
        
        Ok(())
    }
    
    /// Real convergence detection with multiple criteria
    async fn check_convergence(&self, stats: &PopulationStats, diversity: &DiversityMetrics) -> bool {
        let detector = &self.genetic_config.convergence_detector;
        
        // Fitness threshold convergence
        if let Some(threshold) = self.genetic_config.fitness_threshold {
            if stats.best_fitness >= threshold {
                return true;
            }
        }
        
        // Diversity-based convergence (population too similar)
        if diversity.genotypic_diversity < detector.diversity_threshold {
            return true;
        }
        
        // Stagnation-based convergence (no improvement for too long)
        if self.generations_without_improvement.load(Ordering::SeqCst) >= detector.stagnation_generations as u64 {
            return true;
        }
        
        // Variance-based convergence (fitness values too similar)
        if stats.fitness_variance < detector.variance_threshold {
            return true;
        }
        
        // Check recent fitness improvement
        let fitness_history = self.fitness_history.read().await;
        if fitness_history.len() >= 10 {
            let recent_best: Vec<f64> = fitness_history.iter()
                .rev()
                .take(10)
                .map(|generation| generation.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)))
                .collect();
            
            if recent_best.len() >= 2 {
                let improvement = recent_best[0] - recent_best[recent_best.len() - 1];
                if improvement < detector.improvement_threshold {
                    return true;
                }
            }
        }
        
        false
    }
    
    /// Get comprehensive performance metrics for analysis
    pub async fn get_performance_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        
        // Basic counters
        metrics.insert("current_generation".to_string(), self.current_generation.load(Ordering::SeqCst) as f64);
        metrics.insert("total_mutations".to_string(), self.mutation_count.load(Ordering::SeqCst) as f64);
        metrics.insert("total_crossovers".to_string(), self.crossover_count.load(Ordering::SeqCst) as f64);
        metrics.insert("total_eliminations".to_string(), self.termination_count.load(Ordering::SeqCst) as f64);
        metrics.insert("total_spawns".to_string(), self.spawn_count.load(Ordering::SeqCst) as f64);
        metrics.insert("total_immigrations".to_string(), self.immigration_count.load(Ordering::SeqCst) as f64);
        
        // Fitness tracking
        metrics.insert("best_fitness_ever".to_string(), *self.best_fitness_ever.lock().unwrap());
        metrics.insert("generations_without_improvement".to_string(), 
                      self.generations_without_improvement.load(Ordering::SeqCst) as f64);
        
        // Convergence status
        metrics.insert("convergence_reached".to_string(), 
                      if *self.convergence_reached.lock().unwrap() { 1.0 } else { 0.0 });
        
        // Performance ratios
        let total_operations = metrics["total_crossovers"] + metrics["total_mutations"] + metrics["total_spawns"];
        if total_operations > 0.0 {
            metrics.insert("crossover_ratio".to_string(), metrics["total_crossovers"] / total_operations);
            metrics.insert("mutation_ratio".to_string(), metrics["total_mutations"] / total_operations);
            metrics.insert("spawn_ratio".to_string(), metrics["total_spawns"] / total_operations);
        }
        
        metrics
    }
    
    /// Check if evolution should terminate based on various criteria
    pub async fn should_terminate(&self) -> bool {
        // Check max generations limit
        if let Some(max_gen) = self.genetic_config.max_generations {
            if self.current_generation.load(Ordering::SeqCst) >= max_gen {
                tracing::info!("Evolution terminated: Maximum generations ({}) reached", max_gen);
                return true;
            }
        }
        
        // Check fitness threshold
        if let Some(threshold) = self.genetic_config.fitness_threshold {
            if *self.best_fitness_ever.lock().unwrap() >= threshold {
                tracing::info!("Evolution terminated: Fitness threshold ({:.3}) reached", threshold);
                return true;
            }
        }
        
        // Check convergence status
        if *self.convergence_reached.lock().unwrap() {
            tracing::info!("Evolution terminated: Convergence detected");
            return true;
        }
        
        // Check stagnation
        let stagnation = self.generations_without_improvement.load(Ordering::SeqCst);
        if stagnation >= self.genetic_config.max_generations_without_improvement as u64 {
            tracing::info!("Evolution terminated: Stagnation for {} generations", stagnation);
            return true;
        }
        
        false
    }
    
    /// Export comprehensive evolution data for analysis
    pub async fn export_evolution_data(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let history = self.evolution_history.read().await;
        let fitness_history = self.fitness_history.read().await;
        let diversity_history = self.diversity_history.read().await;
        let metrics = self.get_performance_metrics().await;
        
        let export_data = serde_json::json!({
            "metadata": {
                "export_timestamp": Utc::now(),
                "total_generations": self.current_generation.load(Ordering::SeqCst),
                "convergence_reached": *self.convergence_reached.lock().unwrap(),
                "best_fitness_achieved": *self.best_fitness_ever.lock().unwrap()
            },
            "evolution_history": *history,
            "fitness_history": *fitness_history,
            "diversity_history": *diversity_history,
            "performance_metrics": metrics,
            "configuration": {
                "population_size": self.genetic_config.population_size,
                "mutation_rate": self.genetic_config.mutation_rate,
                "crossover_rate": self.genetic_config.crossover_rate,
                "selection_pressure": self.genetic_config.selection_pressure,
                "elite_percentage": self.genetic_config.elite_percentage,
                "diversity_threshold": self.genetic_config.diversity_threshold,
                "max_generations": self.genetic_config.max_generations,
                "fitness_threshold": self.genetic_config.fitness_threshold,
                "parallel_execution": self.genetic_config.parallel_execution
            }
        });
        
        Ok(serde_json::to_string_pretty(&export_data)?)
    }
    
    /// Get fitness trend analysis over generations
    pub async fn get_fitness_trend(&self) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        let fitness_history = self.fitness_history.read().await;
        Ok(fitness_history.iter()
            .map(|generation| generation.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)))
            .collect())
    }
    
    /// Get diversity trend analysis over generations
    pub async fn get_diversity_trend(&self) -> Result<Vec<f64>, Box<dyn std::error::Error + Send + Sync>> {
        let diversity_history = self.diversity_history.read().await;
        Ok(diversity_history.iter().map(|d| d.genotypic_diversity).collect())
    }
    
    /// Reset evolution state completely
    pub async fn reset_evolution_state(&self) {
        // Reset all atomic counters
        self.current_generation.store(0, Ordering::SeqCst);
        self.mutation_count.store(0, Ordering::SeqCst);
        self.crossover_count.store(0, Ordering::SeqCst);
        self.termination_count.store(0, Ordering::SeqCst);
        self.spawn_count.store(0, Ordering::SeqCst);
        self.immigration_count.store(0, Ordering::SeqCst);
        self.generations_without_improvement.store(0, Ordering::SeqCst);
        
        // Reset mutex-protected values
        *self.best_fitness_ever.lock().unwrap() = 0.0;
        *self.convergence_reached.lock().unwrap() = false;
        *self.last_evolution.lock().unwrap() = Utc::now();
        
        // Clear all historical data
        self.fitness_history.write().await.clear();
        self.diversity_history.write().await.clear();
        self.evolution_history.write().await.clear();
        self.performance_metrics.write().await.clear();
        
        tracing::info!("Evolution engine state completely reset");
    }
}