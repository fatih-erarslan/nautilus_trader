//! Selection Pressure module for market-driven organism selection
//! Adaptive selection based on market conditions and organism fitness

use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use uuid::Uuid;

use crate::organisms::ParasiticOrganism;
use crate::evolution::fitness_evaluator::MarketConditions;

/// Selection pressure configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SelectionPressureConfig {
    pub base_pressure: f64,
    pub market_volatility_factor: f64,
    pub adaptive_pressure: bool,
    pub elite_protection: f64,
}

impl Default for SelectionPressureConfig {
    fn default() -> Self {
        Self {
            base_pressure: 1.2,
            market_volatility_factor: 0.5,
            adaptive_pressure: true,
            elite_protection: 0.1,
        }
    }
}

/// Selection pressure engine (placeholder)
pub struct SelectionPressure {
    config: SelectionPressureConfig,
}

impl SelectionPressure {
    pub fn new(config: SelectionPressureConfig) -> Self {
        Self { config }
    }
    
    /// Calculate selection pressure based on market conditions
    pub async fn calculate_pressure(
        &self,
        market_conditions: &MarketConditions,
        population_fitness: &HashMap<Uuid, f64>,
    ) -> f64 {
        let mut pressure = self.config.base_pressure;
        
        // Adjust based on market volatility
        if self.config.adaptive_pressure {
            pressure *= 1.0 + (market_conditions.volatility * self.config.market_volatility_factor);
        }
        
        pressure.clamp(0.5, 3.0)
    }
}