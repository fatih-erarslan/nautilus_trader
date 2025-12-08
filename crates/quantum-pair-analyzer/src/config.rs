// Configuration Management - Production Settings Only
// Copyright (c) 2025 TENGRI Trading Swarm

use std::collections::HashMap;
use std::time::Duration;
use serde::{Deserialize, Serialize};
use anyhow::Result;

use crate::errors::ConfigError;
use crate::data::{ExchangeConfig, NewsConfig};

/// Swarm algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SwarmConfig {
    pub cuckoo_search: CuckooConfig,
    pub firefly: FireflyConfig,
    pub ant_colony: AntColonyConfig,
    pub particle_swarm: ParticleSwarmConfig,
    pub genetic_algorithm: GeneticConfig,
    pub max_iterations: u32,
    pub convergence_threshold: f64,
    pub population_size: usize,
}

/// Cuckoo Search algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CuckooConfig {
    pub population_size: usize,
    pub levy_alpha: f64,
    pub abandonment_rate: f64,
    pub max_generations: u32,
    pub step_size: f64,
}

/// Firefly algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FireflyConfig {
    pub population_size: usize,
    pub alpha: f64,
    pub beta: f64,
    pub gamma: f64,
    pub max_generations: u32,
}

/// Ant Colony Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AntColonyConfig {
    pub num_ants: usize,
    pub alpha: f64,
    pub beta: f64,
    pub rho: f64,
    pub q: f64,
    pub max_iterations: u32,
}

/// Particle Swarm Optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParticleSwarmConfig {
    pub num_particles: usize,
    pub inertia_weight: f64,
    pub cognitive_coefficient: f64,
    pub social_coefficient: f64,
    pub max_velocity: f64,
    pub max_iterations: u32,
}

/// Genetic Algorithm configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneticConfig {
    pub population_size: usize,
    pub mutation_rate: f64,
    pub crossover_rate: f64,
    pub elite_size: usize,
    pub max_generations: u32,
}

/// Quantum computing configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumConfig {
    pub enabled: bool,
    pub device_hierarchy: Vec<String>,
    pub num_qubits: usize,
    pub circuit_depth: usize,
    pub optimization_steps: u32,
    pub convergence_tolerance: f64,
}

/// Sentiment analysis configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SentimentConfig {
    pub enabled: bool,
    pub models: Vec<String>,
    pub fusion_method: String,
    pub update_frequency: Duration,
    pub confidence_threshold: f64,
}

/// Time frame for analysis
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeFrame {
    OneMinute,
    FiveMinutes,
    FifteenMinutes,
    OneHour,
    FourHours,
    OneDay,
    OneWeek,
}

impl TimeFrame {
    pub fn to_seconds(&self) -> u64 {
        match self {
            TimeFrame::OneMinute => 60,
            TimeFrame::FiveMinutes => 300,
            TimeFrame::FifteenMinutes => 900,
            TimeFrame::OneHour => 3600,
            TimeFrame::FourHours => 14400,
            TimeFrame::OneDay => 86400,
            TimeFrame::OneWeek => 604800,
        }
    }
    
    pub fn to_duration(&self) -> Duration {
        Duration::from_secs(self.to_seconds())
    }
}

impl Default for SwarmConfig {
    fn default() -> Self {
        Self {
            cuckoo_search: CuckooConfig::default(),
            firefly: FireflyConfig::default(),
            ant_colony: AntColonyConfig::default(),
            particle_swarm: ParticleSwarmConfig::default(),
            genetic_algorithm: GeneticConfig::default(),
            max_iterations: 1000,
            convergence_threshold: 1e-6,
            population_size: 50,
        }
    }
}

impl Default for CuckooConfig {
    fn default() -> Self {
        Self {
            population_size: 25,
            levy_alpha: 1.5,
            abandonment_rate: 0.25,
            max_generations: 1000,
            step_size: 0.01,
        }
    }
}

impl Default for FireflyConfig {
    fn default() -> Self {
        Self {
            population_size: 20,
            alpha: 0.2,
            beta: 1.0,
            gamma: 0.97,
            max_generations: 1000,
        }
    }
}

impl Default for AntColonyConfig {
    fn default() -> Self {
        Self {
            num_ants: 50,
            alpha: 1.0,
            beta: 2.0,
            rho: 0.1,
            q: 100.0,
            max_iterations: 1000,
        }
    }
}

impl Default for ParticleSwarmConfig {
    fn default() -> Self {
        Self {
            num_particles: 30,
            inertia_weight: 0.5,
            cognitive_coefficient: 1.5,
            social_coefficient: 1.5,
            max_velocity: 0.1,
            max_iterations: 1000,
        }
    }
}

impl Default for GeneticConfig {
    fn default() -> Self {
        Self {
            population_size: 100,
            mutation_rate: 0.01,
            crossover_rate: 0.8,
            elite_size: 10,
            max_generations: 1000,
        }
    }
}

impl Default for QuantumConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            device_hierarchy: vec![
                "lightning.gpu".to_string(),
                "lightning-kokkos".to_string(),
                "lightning.qubit".to_string(),
            ],
            num_qubits: 12,
            circuit_depth: 6,
            optimization_steps: 100,
            convergence_tolerance: 1e-6,
        }
    }
}

impl Default for SentimentConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            models: vec![
                "roberta".to_string(),
                "distilbert".to_string(),
                "xgboost".to_string(),
            ],
            fusion_method: "rsc".to_string(),
            update_frequency: Duration::from_secs(300),
            confidence_threshold: 0.7,
        }
    }
}

impl Default for crate::AnalyzerConfig {
    fn default() -> Self {
        Self {
            exchange_configs: HashMap::new(),
            news_configs: HashMap::new(),
            swarm_config: SwarmConfig::default(),
            quantum_config: QuantumConfig::default(),
            sentiment_config: SentimentConfig::default(),
            parallelism: num_cpus::get(),
            simd_enabled: true,
            quantum_enabled: true,
            enforce_real_data: true,
            validation_strictness: crate::ValidationStrictness::Strict,
            correlation_window: 252, // 1 year of trading days
            regime_lookback: 100,
            sentiment_horizon: Duration::from_secs(86400), // 24 hours
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_timeframe_conversion() {
        assert_eq!(TimeFrame::OneMinute.to_seconds(), 60);
        assert_eq!(TimeFrame::OneHour.to_seconds(), 3600);
        assert_eq!(TimeFrame::OneDay.to_seconds(), 86400);
    }
    
    #[test]
    fn test_default_configs() {
        let swarm_config = SwarmConfig::default();
        assert!(swarm_config.max_iterations > 0);
        assert!(swarm_config.population_size > 0);
        
        let quantum_config = QuantumConfig::default();
        assert!(quantum_config.enabled);
        assert!(!quantum_config.device_hierarchy.is_empty());
    }
    
    #[test]
    fn test_config_serialization() {
        let config = SwarmConfig::default();
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: SwarmConfig = serde_json::from_str(&serialized).unwrap();
        assert_eq!(config.max_iterations, deserialized.max_iterations);
    }
}