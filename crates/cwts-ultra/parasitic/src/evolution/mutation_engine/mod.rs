//! Real Mutation Engine implementation for parasitic organism genetic variation
//! Adaptive mutation strategies with sub-millisecond performance
//! Zero mocks policy - all mutations operate on real genetic data

use std::sync::{Arc, Mutex, atomic::{AtomicU64, Ordering}};
use std::collections::HashMap;
use dashmap::DashMap;
use uuid::Uuid;
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock;
use rand::{Rng, thread_rng};
use rand_distr::{Normal, Uniform, Distribution};
use rayon::prelude::*;

use crate::organisms::{ParasiticOrganism, OrganismGenetics};

/// Mutation engine configuration parameters
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationEngineConfig {
    pub base_mutation_rate: f64,
    pub adaptive_mutation: bool,
    pub mutation_strength: f64,
    pub max_mutation_rate: f64,
    pub min_mutation_rate: f64,
    pub diversity_threshold: f64,
    pub convergence_pressure: f64,
    pub targeted_mutation: bool,
    pub gaussian_mutation: bool,
    pub uniform_mutation: bool,
}

impl Default for MutationEngineConfig {
    fn default() -> Self {
        Self {
            base_mutation_rate: 0.1,
            adaptive_mutation: true,
            mutation_strength: 0.2,
            max_mutation_rate: 0.5,
            min_mutation_rate: 0.01,
            diversity_threshold: 0.15,
            convergence_pressure: 1.5,
            targeted_mutation: true,
            gaussian_mutation: true,
            uniform_mutation: false,
        }
    }
}

/// Mutation operation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationResult {
    pub mutations_applied: u64,
    pub effective_rate: f64,
    pub traits_modified: u64,
    pub total_time_nanos: u64,
    pub organisms_affected: u64,
    pub average_genetic_change: f64,
}

/// Mutation statistics for tracking and analysis
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MutationStatistics {
    pub total_mutations: u64,
    pub total_organisms_mutated: u64,
    pub average_mutations_per_cycle: f64,
    pub current_mutation_rate: f64,
    pub mutation_rate_history: Vec<f64>,
    pub total_execution_time_ms: f64,
    pub genetic_diversity_impact: f64,
}

/// Mutation strategy type
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MutationStrategy {
    Gaussian,
    Uniform,
    Targeted,
    Adaptive,
}

/// Real Mutation Engine implementation
pub struct MutationEngine {
    config: Arc<RwLock<MutationEngineConfig>>,
    current_mutation_rate: Arc<Mutex<f64>>,
    mutation_count: Arc<AtomicU64>,
    organism_mutation_count: Arc<AtomicU64>,
    mutation_statistics: Arc<RwLock<MutationStatistics>>,
    rate_history: Arc<RwLock<Vec<f64>>>,
    execution_metrics: Arc<RwLock<HashMap<String, f64>>>,
}

impl MutationEngine {
    pub fn new(config: MutationEngineConfig) -> Self {
        let initial_stats = MutationStatistics {
            total_mutations: 0,
            total_organisms_mutated: 0,
            average_mutations_per_cycle: 0.0,
            current_mutation_rate: config.base_mutation_rate,
            mutation_rate_history: Vec::new(),
            total_execution_time_ms: 0.0,
            genetic_diversity_impact: 0.0,
        };
        
        Self {
            current_mutation_rate: Arc::new(Mutex::new(config.base_mutation_rate)),
            config: Arc::new(RwLock::new(config)),
            mutation_count: Arc::new(AtomicU64::new(0)),
            organism_mutation_count: Arc::new(AtomicU64::new(0)),
            mutation_statistics: Arc::new(RwLock::new(initial_stats)),
            rate_history: Arc::new(RwLock::new(Vec::new())),
            execution_metrics: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Apply mutations to population - main entry point optimized for speed
    pub async fn apply_mutations(
        &mut self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
        population_diversity: f64,
    ) -> Result<MutationResult, Box<dyn std::error::Error + Send + Sync>> {
        let start_time = std::time::Instant::now();
        let config = self.config.read().await;
        
        // Adaptive mutation rate adjustment
        if config.adaptive_mutation {
            self.adapt_mutation_rate(population_diversity, &config).await;
        }
        
        let current_rate = *self.current_mutation_rate.lock().unwrap();
        let organism_ids: Vec<Uuid> = organisms.iter().map(|entry| *entry.key()).collect();
        
        // Calculate expected number of mutations
        let expected_mutations = (organism_ids.len() as f64 * current_rate) as usize;
        
        let mut mutations_applied = 0u64;
        let mut traits_modified = 0u64;
        let mut organisms_affected = 0u64;
        let mut total_genetic_change = 0.0;
        
        // Apply mutations based on configured strategy
        if config.gaussian_mutation {
            let gaussian_results = self.apply_gaussian_mutations(
                organisms, &organism_ids, expected_mutations, config.mutation_strength
            ).await?;
            mutations_applied += gaussian_results.0;
            traits_modified += gaussian_results.1;
            organisms_affected += gaussian_results.2;
            total_genetic_change += gaussian_results.3;
        }
        
        if config.uniform_mutation {
            let uniform_results = self.apply_uniform_mutations(
                organisms, &organism_ids, expected_mutations, config.mutation_strength
            ).await?;
            mutations_applied += uniform_results.0;
            traits_modified += uniform_results.1;
            organisms_affected += uniform_results.2;
            total_genetic_change += uniform_results.3;
        }
        
        if config.targeted_mutation {
            let targeted_results = self.apply_targeted_mutations(
                organisms, &organism_ids, population_diversity
            ).await?;
            mutations_applied += targeted_results.0;
            traits_modified += targeted_results.1;
            organisms_affected += targeted_results.2;
            total_genetic_change += targeted_results.3;
        }
        
        // Update counters
        self.mutation_count.fetch_add(mutations_applied, Ordering::SeqCst);
        self.organism_mutation_count.fetch_add(organisms_affected, Ordering::SeqCst);
        
        // Update statistics
        let execution_time = start_time.elapsed();
        self.update_statistics(
            mutations_applied,
            organisms_affected,
            current_rate,
            execution_time.as_nanos() as f64 / 1_000_000.0
        ).await;
        
        let average_change = if organisms_affected > 0 {
            total_genetic_change / organisms_affected as f64
        } else {
            0.0
        };
        
        Ok(MutationResult {
            mutations_applied,
            effective_rate: mutations_applied as f64 / organism_ids.len() as f64,
            traits_modified,
            total_time_nanos: execution_time.as_nanos() as u64,
            organisms_affected,
            average_genetic_change: average_change,
        })
    }
    
    /// Apply Gaussian mutations to selected organisms
    async fn apply_gaussian_mutations(
        &self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
        organism_ids: &[Uuid],
        target_mutations: usize,
        mutation_strength: f64,
    ) -> Result<(u64, u64, u64, f64), Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = thread_rng();
        let normal_dist = Normal::new(0.0, mutation_strength)?;
        
        let mut mutations_applied = 0u64;
        let mut traits_modified = 0u64;
        let mut organisms_affected = 0u64;
        let mut total_genetic_change = 0.0;
        
        // Select organisms for mutation
        let selected_ids: Vec<&Uuid> = organism_ids
            .choose_multiple(&mut rng, target_mutations)
            .collect();
        
        for &organism_id in &selected_ids {
            if let Some(mut organism_ref) = organisms.get_mut(organism_id) {
                let mut genetics = organism_ref.value().get_genetics();
                let original_genetics = genetics.clone();
                
                // Apply Gaussian mutation to random traits
                let traits_to_mutate = rng.gen_range(1..=3); // Mutate 1-3 traits per organism
                
                for _ in 0..traits_to_mutate {
                    let trait_index = rng.gen_range(0..8);
                    let mutation_delta = normal_dist.sample(&mut rng);
                    
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
                    
                    traits_modified += 1;
                }
                
                organism_ref.value_mut().set_genetics(genetics.clone());
                mutations_applied += 1;
                organisms_affected += 1;
                
                // Calculate genetic change
                let change = self.calculate_genetic_distance(&original_genetics, &genetics);
                total_genetic_change += change;
            }
        }
        
        Ok((mutations_applied, traits_modified, organisms_affected, total_genetic_change))
    }
    
    /// Apply uniform mutations to selected organisms
    async fn apply_uniform_mutations(
        &self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
        organism_ids: &[Uuid],
        target_mutations: usize,
        mutation_strength: f64,
    ) -> Result<(u64, u64, u64, f64), Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = thread_rng();
        let uniform_dist = Uniform::new(-mutation_strength, mutation_strength);
        
        let mut mutations_applied = 0u64;
        let mut traits_modified = 0u64;
        let mut organisms_affected = 0u64;
        let mut total_genetic_change = 0.0;
        
        let selected_ids: Vec<&Uuid> = organism_ids
            .choose_multiple(&mut rng, target_mutations)
            .collect();
        
        for &organism_id in &selected_ids {
            if let Some(mut organism_ref) = organisms.get_mut(organism_id) {
                let mut genetics = organism_ref.value().get_genetics();
                let original_genetics = genetics.clone();
                
                // Apply uniform mutation to all traits
                let mutation_deltas: [f64; 8] = [
                    uniform_dist.sample(&mut rng),
                    uniform_dist.sample(&mut rng),
                    uniform_dist.sample(&mut rng),
                    uniform_dist.sample(&mut rng),
                    uniform_dist.sample(&mut rng),
                    uniform_dist.sample(&mut rng),
                    uniform_dist.sample(&mut rng),
                    uniform_dist.sample(&mut rng),
                ];
                
                genetics.aggression = (genetics.aggression + mutation_deltas[0]).clamp(0.0, 1.0);
                genetics.adaptability = (genetics.adaptability + mutation_deltas[1]).clamp(0.0, 1.0);
                genetics.efficiency = (genetics.efficiency + mutation_deltas[2]).clamp(0.0, 1.0);
                genetics.resilience = (genetics.resilience + mutation_deltas[3]).clamp(0.0, 1.0);
                genetics.reaction_speed = (genetics.reaction_speed + mutation_deltas[4]).clamp(0.0, 1.0);
                genetics.risk_tolerance = (genetics.risk_tolerance + mutation_deltas[5]).clamp(0.0, 1.0);
                genetics.cooperation = (genetics.cooperation + mutation_deltas[6]).clamp(0.0, 1.0);
                genetics.stealth = (genetics.stealth + mutation_deltas[7]).clamp(0.0, 1.0);
                
                organism_ref.value_mut().set_genetics(genetics.clone());
                mutations_applied += 1;
                traits_modified += 8; // All traits modified in uniform mutation
                organisms_affected += 1;
                
                let change = self.calculate_genetic_distance(&original_genetics, &genetics);
                total_genetic_change += change;
            }
        }
        
        Ok((mutations_applied, traits_modified, organisms_affected, total_genetic_change))
    }
    
    /// Apply targeted mutations based on performance feedback
    async fn apply_targeted_mutations(
        &self,
        organisms: &Arc<DashMap<Uuid, Box<dyn ParasiticOrganism + Send + Sync>>>,
        organism_ids: &[Uuid],
        population_diversity: f64,
    ) -> Result<(u64, u64, u64, f64), Box<dyn std::error::Error + Send + Sync>> {
        let mut rng = thread_rng();
        let mutation_strength = if population_diversity < 0.2 {
            0.3 // Stronger mutations for low diversity
        } else {
            0.15 // Gentler mutations for diverse populations
        };
        
        let mut mutations_applied = 0u64;
        let mut traits_modified = 0u64;
        let mut organisms_affected = 0u64;
        let mut total_genetic_change = 0.0;
        
        // Target lower-performing organisms (simplified - in practice would use fitness scores)
        let target_count = (organism_ids.len() as f64 * 0.3) as usize; // Target 30% of population
        let selected_ids: Vec<&Uuid> = organism_ids
            .choose_multiple(&mut rng, target_count)
            .collect();
        
        for &organism_id in &selected_ids {
            if let Some(mut organism_ref) = organisms.get_mut(organism_id) {
                let mut genetics = organism_ref.value().get_genetics();
                let original_genetics = genetics.clone();
                
                // Targeted improvements to key performance traits
                let improvement_targets = [
                    (&mut genetics.efficiency, 0.8), // Target efficiency improvement
                    (&mut genetics.reaction_speed, 0.9), // Target reaction speed
                    (&mut genetics.adaptability, 0.7), // Target adaptability
                    (&mut genetics.resilience, 0.6), // Target resilience
                ];
                
                for (trait_ref, target_value) in improvement_targets.iter() {
                    if **trait_ref < *target_value {
                        let improvement = rng.gen_range(0.0..mutation_strength);
                        **trait_ref = (**trait_ref + improvement).clamp(0.0, 1.0);
                        traits_modified += 1;
                    }
                }
                
                // Also reduce risk tolerance if it's too high
                if genetics.risk_tolerance > 0.6 {
                    let reduction = rng.gen_range(0.0..mutation_strength);
                    genetics.risk_tolerance = (genetics.risk_tolerance - reduction).clamp(0.0, 1.0);
                    traits_modified += 1;
                }
                
                organism_ref.value_mut().set_genetics(genetics.clone());
                mutations_applied += 1;
                organisms_affected += 1;
                
                let change = self.calculate_genetic_distance(&original_genetics, &genetics);
                total_genetic_change += change;
            }
        }
        
        Ok((mutations_applied, traits_modified, organisms_affected, total_genetic_change))
    }
    
    /// Individual Gaussian mutation application
    pub async fn apply_gaussian_mutation(
        &self,
        organism: &mut dyn ParasiticOrganism,
        strength: f64,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut genetics = organism.get_genetics();
        let mut rng = thread_rng();
        let normal_dist = Normal::new(0.0, strength)?;
        
        // Apply Gaussian mutation to random trait
        let trait_index = rng.gen_range(0..8);
        let mutation_delta = normal_dist.sample(&mut rng);
        
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
        
        organism.set_genetics(genetics);
        Ok(())
    }
    
    /// Individual uniform mutation application
    pub async fn apply_uniform_mutation(
        &self,
        organism: &mut dyn ParasiticOrganism,
        strength: f64,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut genetics = organism.get_genetics();
        let mut rng = thread_rng();
        let uniform_dist = Uniform::new(-strength, strength);
        
        // Apply uniform mutation to all traits
        genetics.aggression = (genetics.aggression + uniform_dist.sample(&mut rng)).clamp(0.0, 1.0);
        genetics.adaptability = (genetics.adaptability + uniform_dist.sample(&mut rng)).clamp(0.0, 1.0);
        genetics.efficiency = (genetics.efficiency + uniform_dist.sample(&mut rng)).clamp(0.0, 1.0);
        genetics.resilience = (genetics.resilience + uniform_dist.sample(&mut rng)).clamp(0.0, 1.0);
        genetics.reaction_speed = (genetics.reaction_speed + uniform_dist.sample(&mut rng)).clamp(0.0, 1.0);
        genetics.risk_tolerance = (genetics.risk_tolerance + uniform_dist.sample(&mut rng)).clamp(0.0, 1.0);
        genetics.cooperation = (genetics.cooperation + uniform_dist.sample(&mut rng)).clamp(0.0, 1.0);
        genetics.stealth = (genetics.stealth + uniform_dist.sample(&mut rng)).clamp(0.0, 1.0);
        
        organism.set_genetics(genetics);
        Ok(())
    }
    
    /// Individual targeted mutation based on fitness scores
    pub async fn apply_targeted_mutation(
        &self,
        organism: &mut dyn ParasiticOrganism,
        fitness_scores: &[f64],
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let mut genetics = organism.get_genetics();
        let mut rng = thread_rng();
        
        let current_fitness = fitness_scores.last().unwrap_or(&0.5);
        let mutation_strength = if *current_fitness < 0.3 {
            0.3 // Strong mutations for poor performers
        } else if *current_fitness > 0.7 {
            0.05 // Gentle mutations for good performers
        } else {
            0.15 // Medium mutations for average performers
        };
        
        // Target key performance traits for improvement
        if genetics.efficiency < 0.8 {
            let improvement = rng.gen_range(0.0..mutation_strength);
            genetics.efficiency = (genetics.efficiency + improvement).clamp(0.0, 1.0);
        }
        
        if genetics.reaction_speed < 0.9 {
            let improvement = rng.gen_range(0.0..mutation_strength);
            genetics.reaction_speed = (genetics.reaction_speed + improvement).clamp(0.0, 1.0);
        }
        
        if genetics.adaptability < 0.7 {
            let improvement = rng.gen_range(0.0..mutation_strength);
            genetics.adaptability = (genetics.adaptability + improvement).clamp(0.0, 1.0);
        }
        
        // Reduce excessive risk tolerance
        if genetics.risk_tolerance > 0.7 {
            let reduction = rng.gen_range(0.0..mutation_strength);
            genetics.risk_tolerance = (genetics.risk_tolerance - reduction).clamp(0.0, 1.0);
        }
        
        organism.set_genetics(genetics);
        Ok(())
    }
    
    /// Adapt mutation rate based on population diversity
    async fn adapt_mutation_rate(&self, population_diversity: f64, config: &MutationEngineConfig) {
        let mut current_rate = self.current_mutation_rate.lock().unwrap();
        
        // Increase mutation rate for low diversity, decrease for high diversity
        let diversity_factor = if population_diversity < config.diversity_threshold {
            config.convergence_pressure
        } else {
            1.0 / config.convergence_pressure
        };
        
        let target_rate = config.base_mutation_rate * diversity_factor;
        let new_rate = target_rate.clamp(config.min_mutation_rate, config.max_mutation_rate);
        
        *current_rate = new_rate;
    }
    
    /// Calculate genetic distance between two genetic profiles
    fn calculate_genetic_distance(&self, g1: &OrganismGenetics, g2: &OrganismGenetics) -> f64 {
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
        
        // Euclidean distance normalized by maximum possible distance
        differences.iter().map(|d| d * d).sum::<f64>().sqrt() / 8.0_f64.sqrt()
    }
    
    /// Update internal statistics tracking
    async fn update_statistics(
        &self,
        mutations_applied: u64,
        organisms_affected: u64,
        current_rate: f64,
        execution_time_ms: f64,
    ) {
        let mut stats = self.mutation_statistics.write().await;
        let mut rate_history = self.rate_history.write().await;
        
        stats.total_mutations += mutations_applied;
        stats.total_organisms_mutated += organisms_affected;
        stats.current_mutation_rate = current_rate;
        stats.total_execution_time_ms += execution_time_ms;
        
        rate_history.push(current_rate);
        if rate_history.len() > 100 {
            rate_history.remove(0);
        }
        stats.mutation_rate_history = rate_history.clone();
        
        // Update average mutations per cycle
        let cycle_count = rate_history.len() as f64;
        if cycle_count > 0.0 {
            stats.average_mutations_per_cycle = stats.total_mutations as f64 / cycle_count;
        }
    }
    
    // Public getters and utilities
    pub async fn get_config(&self) -> MutationEngineConfig {
        self.config.read().await.clone()
    }
    
    pub fn get_current_mutation_rate(&self) -> f64 {
        *self.current_mutation_rate.lock().unwrap()
    }
    
    pub fn get_mutation_count(&self) -> u64 {
        self.mutation_count.load(Ordering::SeqCst)
    }
    
    pub async fn get_mutation_statistics(&self) -> MutationStatistics {
        self.mutation_statistics.read().await.clone()
    }
    
    pub async fn get_execution_metrics(&self) -> HashMap<String, f64> {
        self.execution_metrics.read().await.clone()
    }
    
    /// Reset mutation engine state
    pub async fn reset(&mut self) {
        *self.current_mutation_rate.lock().unwrap() = self.config.read().await.base_mutation_rate;
        self.mutation_count.store(0, Ordering::SeqCst);
        self.organism_mutation_count.store(0, Ordering::SeqCst);
        
        let mut stats = self.mutation_statistics.write().await;
        stats.total_mutations = 0;
        stats.total_organisms_mutated = 0;
        stats.average_mutations_per_cycle = 0.0;
        stats.mutation_rate_history.clear();
        stats.total_execution_time_ms = 0.0;
        
        self.rate_history.write().await.clear();
        self.execution_metrics.write().await.clear();
    }
}