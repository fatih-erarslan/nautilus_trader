//! # Parasitic Organisms Module
//!
//! This module defines different types of parasitic organisms that can infect
//! and exploit trading pairs. Each organism has unique characteristics and
//! strategies for extracting value from market inefficiencies.

pub mod anglerfish;
pub mod anglerfish_lure;
#[cfg(test)]
pub mod anglerfish_lure_test;
pub mod bacteria;
pub mod cordyceps;
pub mod cuckoo;
pub mod electric_eel;
pub mod komodo_dragon;
pub mod lancet_liver_fluke;
pub mod mycelial_network;
pub mod octopus;
pub mod platypus;
pub mod tardigrade;
pub mod toxoplasma;
pub mod vampire_bat;
pub mod virus;
pub mod wasp;

pub use anglerfish::{AnglerfishConfig, AnglerfishOrganism, AnglerfishStatus};
pub use anglerfish_lure::{
    AnglerfishLure, ArtificialActivityGenerator, HoneyPotCreator, TraderAttractor,
};
pub use bacteria::BacteriaOrganism;
pub use cordyceps::{CordycepsConfig, CordycepsOrganism, CordycepsStatus};
pub use cuckoo::CuckooOrganism;
pub use electric_eel::{ElectricEelConfig, ElectricEelOrganism, ElectricEelStatus};
pub use komodo_dragon::{KomodoConfig, KomodoDragonOrganism, KomodoStatus};
pub use lancet_liver_fluke::{LancetFlukeConfig, LancetFlukeOrganism, LancetFlukeStatus};
pub use mycelial_network::{MycelialConfig, MycelialNetworkOrganism, MycelialNetworkStatus};
pub use octopus::OctopusCamouflage;
pub use platypus::{PlatypusConfig, PlatypusOrganism, PlatypusStatus};
pub use tardigrade::{
    DormancyTrigger, MarketExtremeDetector, RevivalConditions, TardigradeConfig,
    TardigradeOrganism, TardigradeStatus, TardigradeSurvival,
};
pub use toxoplasma::{ToxoplasmaConfig, ToxoplasmaOrganism, ToxoplasmaStatus};
pub use vampire_bat::{VampireBatConfig, VampireBatOrganism, VampireBatStatus};
pub use virus::VirusOrganism;
pub use wasp::WaspOrganism;

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use uuid::Uuid;

/// Base trait for all parasitic organisms
#[async_trait]
pub trait ParasiticOrganism: Send + Sync {
    /// Unique identifier for this organism
    fn id(&self) -> Uuid;

    /// Get organism type name
    fn organism_type(&self) -> &'static str;

    /// Get current fitness score
    fn fitness(&self) -> f64;

    /// Calculate infection strength based on vulnerability
    fn calculate_infection_strength(&self, vulnerability: f64) -> f64;

    /// Infect a trading pair
    async fn infect_pair(
        &self,
        pair_id: &str,
        vulnerability: f64,
    ) -> Result<InfectionResult, OrganismError>;

    /// Update organism based on performance feedback
    async fn adapt(&mut self, feedback: AdaptationFeedback) -> Result<(), OrganismError>;

    /// Mutate organism genetics
    fn mutate(&mut self, rate: f64);

    /// Crossover with another organism
    fn crossover(
        &self,
        other: &dyn ParasiticOrganism,
    ) -> Result<Box<dyn ParasiticOrganism + Send + Sync>, OrganismError>;

    /// Get organism's current genetics
    fn get_genetics(&self) -> OrganismGenetics;

    /// Set organism genetics
    fn set_genetics(&mut self, genetics: OrganismGenetics);

    /// Check if organism should be terminated due to poor performance
    fn should_terminate(&self) -> bool;

    /// Get resource consumption metrics
    fn resource_consumption(&self) -> ResourceMetrics;

    /// Get strategy parameters
    fn get_strategy_params(&self) -> HashMap<String, f64>;
}

/// Result of an infection attempt
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InfectionResult {
    pub success: bool,
    pub infection_id: Uuid,
    pub initial_profit: f64,
    pub estimated_duration: u64, // seconds
    pub resource_usage: ResourceMetrics,
}

/// Feedback for organism adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptationFeedback {
    pub performance_score: f64,
    pub profit_generated: f64,
    pub trades_executed: u64,
    pub success_rate: f64,
    pub avg_latency_ns: u64,
    pub market_conditions: MarketConditions,
    pub competition_level: f64,
}

/// Current market conditions affecting parasitic behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketConditions {
    pub volatility: f64,
    pub volume: f64,
    pub spread: f64,
    pub trend_strength: f64,
    pub noise_level: f64,
}

/// Genetics defining organism behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganismGenetics {
    /// Aggressiveness in seeking opportunities
    pub aggression: f64,

    /// Ability to adapt to changing conditions  
    pub adaptability: f64,

    /// Efficiency in resource utilization
    pub efficiency: f64,

    /// Resistance to market stress
    pub resilience: f64,

    /// Speed of decision making
    pub reaction_speed: f64,

    /// Risk tolerance
    pub risk_tolerance: f64,

    /// Cooperation with other organisms
    pub cooperation: f64,

    /// Stealth to avoid detection
    pub stealth: f64,
}

impl Default for OrganismGenetics {
    fn default() -> Self {
        Self {
            aggression: 0.5,
            adaptability: 0.5,
            efficiency: 0.5,
            resilience: 0.5,
            reaction_speed: 0.5,
            risk_tolerance: 0.5,
            cooperation: 0.5,
            stealth: 0.5,
        }
    }
}

impl OrganismGenetics {
    /// Generate random genetics
    pub fn random() -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        Self {
            aggression: rng.gen_range(0.0..1.0),
            adaptability: rng.gen_range(0.0..1.0),
            efficiency: rng.gen_range(0.0..1.0),
            resilience: rng.gen_range(0.0..1.0),
            reaction_speed: rng.gen_range(0.0..1.0),
            risk_tolerance: rng.gen_range(0.0..1.0),
            cooperation: rng.gen_range(0.0..1.0),
            stealth: rng.gen_range(0.0..1.0),
        }
    }

    /// Mutate genetics with given rate
    pub fn mutate(&mut self, rate: f64) {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < rate {
            self.aggression = (self.aggression + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
        if rng.gen::<f64>() < rate {
            self.adaptability = (self.adaptability + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
        if rng.gen::<f64>() < rate {
            self.efficiency = (self.efficiency + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
        if rng.gen::<f64>() < rate {
            self.resilience = (self.resilience + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
        if rng.gen::<f64>() < rate {
            self.reaction_speed = (self.reaction_speed + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
        if rng.gen::<f64>() < rate {
            self.risk_tolerance = (self.risk_tolerance + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
        if rng.gen::<f64>() < rate {
            self.cooperation = (self.cooperation + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
        if rng.gen::<f64>() < rate {
            self.stealth = (self.stealth + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
    }

    /// Crossover with another genetics
    pub fn crossover(&self, other: &OrganismGenetics) -> OrganismGenetics {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        OrganismGenetics {
            aggression: if rng.gen::<bool>() {
                self.aggression
            } else {
                other.aggression
            },
            adaptability: if rng.gen::<bool>() {
                self.adaptability
            } else {
                other.adaptability
            },
            efficiency: if rng.gen::<bool>() {
                self.efficiency
            } else {
                other.efficiency
            },
            resilience: if rng.gen::<bool>() {
                self.resilience
            } else {
                other.resilience
            },
            reaction_speed: if rng.gen::<bool>() {
                self.reaction_speed
            } else {
                other.reaction_speed
            },
            risk_tolerance: if rng.gen::<bool>() {
                self.risk_tolerance
            } else {
                other.risk_tolerance
            },
            cooperation: if rng.gen::<bool>() {
                self.cooperation
            } else {
                other.cooperation
            },
            stealth: if rng.gen::<bool>() {
                self.stealth
            } else {
                other.stealth
            },
        }
    }
}

/// Resource consumption metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResourceMetrics {
    pub cpu_usage: f64,
    pub memory_mb: f64,
    pub network_bandwidth_kbps: f64,
    pub api_calls_per_second: f64,
    pub latency_overhead_ns: u64,
}

impl Default for ResourceMetrics {
    fn default() -> Self {
        Self {
            cpu_usage: 0.0,
            memory_mb: 0.0,
            network_bandwidth_kbps: 0.0,
            api_calls_per_second: 0.0,
            latency_overhead_ns: 0,
        }
    }
}

/// Errors that can occur with organisms
#[derive(Debug, thiserror::Error)]
pub enum OrganismError {
    #[error("Infection failed: {0}")]
    InfectionFailed(String),

    #[error("Adaptation failed: {0}")]
    AdaptationFailed(String),

    #[error("Crossover failed: {0}")]
    CrossoverFailed(String),

    #[error("Invalid genetics: {0}")]
    InvalidGenetics(String),

    #[error("Resource exhausted: {0}")]
    ResourceExhausted(String),

    #[error("Market conditions unsuitable: {0}")]
    UnsuitableConditions(String),
}

/// Base implementation for common organism functionality
#[derive(Debug, Clone)]
pub struct BaseOrganism {
    pub id: Uuid,
    pub genetics: OrganismGenetics,
    pub fitness: f64,
    pub creation_time: DateTime<Utc>,
    pub last_update: DateTime<Utc>,
    pub total_infections: u64,
    pub successful_infections: u64,
    pub total_profit: f64,
    pub performance_history: Vec<f64>,
}

impl BaseOrganism {
    pub fn new() -> Self {
        Self {
            id: Uuid::new_v4(),
            genetics: OrganismGenetics::random(),
            fitness: 0.5, // Start with neutral fitness
            creation_time: Utc::now(),
            last_update: Utc::now(),
            total_infections: 0,
            successful_infections: 0,
            total_profit: 0.0,
            performance_history: Vec::new(),
        }
    }

    pub fn update_fitness(&mut self, performance_score: f64) {
        // Exponentially weighted moving average for fitness
        const ALPHA: f64 = 0.1;
        self.fitness = ALPHA * performance_score + (1.0 - ALPHA) * self.fitness;
        self.performance_history.push(performance_score);

        // Keep only last 100 performance measurements
        if self.performance_history.len() > 100 {
            self.performance_history.remove(0);
        }

        self.last_update = Utc::now();
    }

    pub fn calculate_base_infection_strength(&self, vulnerability: f64) -> f64 {
        // Base infection strength formula combining genetics and vulnerability
        let genetic_factor = (self.genetics.aggression * 0.3
            + self.genetics.efficiency * 0.2
            + self.genetics.reaction_speed * 0.2
            + self.genetics.risk_tolerance * 0.3);

        genetic_factor * vulnerability * (1.0 + self.fitness * 0.5)
    }

    pub fn should_terminate_base(&self) -> bool {
        // Terminate if fitness is very low and hasn't improved
        if self.fitness < 0.1 && self.performance_history.len() >= 10 {
            let recent_avg = self.performance_history.iter().rev().take(10).sum::<f64>() / 10.0;
            return recent_avg < 0.15;
        }
        false
    }
}

/// Factory for creating different organism types
pub struct OrganismFactory;

impl OrganismFactory {
    /// Create a new organism of the specified type
    pub fn create_organism(
        organism_type: &str,
    ) -> Result<Box<dyn ParasiticOrganism + Send + Sync>, OrganismError> {
        match organism_type.to_lowercase().as_str() {
            "cuckoo" => Ok(Box::new(CuckooOrganism::new())),
            "wasp" => Ok(Box::new(WaspOrganism::new())),
            "virus" => Ok(Box::new(VirusOrganism::new())),
            "bacteria" => Ok(Box::new(BacteriaOrganism::new())),
            "cordyceps" => {
                let config = CordycepsConfig::default();
                Ok(Box::new(CordycepsOrganism::new(config)?))
            }
            "vampire_bat" => Ok(Box::new(VampireBatOrganism::new())),
            "lancet_liver_fluke" => Ok(Box::new(LancetFlukeOrganism::new())),
            "toxoplasma" => Ok(Box::new(ToxoplasmaOrganism::new())),
            "mycelial_network" => Ok(Box::new(MycelialNetworkOrganism::new())),
            "anglerfish" => {
                let config = AnglerfishConfig::default();
                Ok(Box::new(AnglerfishOrganism::new(config)?))
            }
            "komodo_dragon" => {
                let config = KomodoConfig::default();
                Ok(Box::new(KomodoDragonOrganism::new(config)?))
            }
            "tardigrade" => {
                let config = TardigradeConfig::default();
                Ok(Box::new(TardigradeOrganism::new(config)?))
            }
            "electric_eel" => {
                let config = ElectricEelConfig::default();
                Ok(Box::new(ElectricEelOrganism::new(config)?))
            }
            "platypus" => {
                let config = PlatypusConfig::default();
                Ok(Box::new(PlatypusOrganism::new(config)?))
            }
            _ => Err(OrganismError::InvalidGenetics(format!(
                "Unknown organism type: {}",
                organism_type
            ))),
        }
    }

    /// Get list of all available organism types
    pub fn available_types() -> Vec<&'static str> {
        vec![
            "cuckoo",
            "wasp",
            "virus",
            "bacteria",
            "cordyceps",
            "vampire_bat",
            "lancet_liver_fluke",
            "toxoplasma",
            "mycelial_network",
            "anglerfish",
            "komodo_dragon",
            "tardigrade",
            "electric_eel",
            "platypus",
        ]
    }

    /// Create a random organism from available types
    pub fn create_random_organism(
    ) -> Result<Box<dyn ParasiticOrganism + Send + Sync>, OrganismError> {
        let types = Self::available_types();
        let random_type = types[fastrand::usize(0..types.len())];
        Self::create_organism(random_type)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_genetics_mutation() {
        let mut genetics = OrganismGenetics::default();
        let original = genetics.clone();

        genetics.mutate(1.0); // 100% mutation rate

        // At least one gene should have changed
        assert_ne!(genetics.aggression, original.aggression);
    }

    #[test]
    fn test_genetics_crossover() {
        let genetics1 = OrganismGenetics {
            aggression: 1.0,
            ..Default::default()
        };
        let genetics2 = OrganismGenetics {
            aggression: 0.0,
            ..Default::default()
        };

        let offspring = genetics1.crossover(&genetics2);

        // Offspring should have one of the parent's aggression values
        assert!(offspring.aggression == 1.0 || offspring.aggression == 0.0);
    }

    #[test]
    fn test_base_organism_creation() {
        let organism = BaseOrganism::new();

        assert_eq!(organism.fitness, 0.5);
        assert_eq!(organism.total_infections, 0);
        assert_eq!(organism.total_profit, 0.0);
    }
}
