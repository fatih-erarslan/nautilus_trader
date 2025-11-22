//! Cuckoo parasitic organism - stealth order book manipulation

use async_trait::async_trait;
use uuid::Uuid;
// Removed unused import: HashMap
// use chrono::{DateTime, Utc}; // Removed: unused imports

use super::{
    AdaptationFeedback, BaseOrganism, InfectionResult, OrganismError, OrganismGenetics,
    ParasiticOrganism, ResourceMetrics,
};
use std::collections::HashMap;

/// Cuckoo organism - specializes in stealth order manipulation
pub struct CuckooOrganism {
    base: BaseOrganism,
    stealth_factor: f64,
    manipulation_strength: f64,
}

impl CuckooOrganism {
    pub fn new() -> Self {
        let mut base = BaseOrganism::new();
        // Cuckoos have high stealth and efficiency by default
        base.genetics.stealth = 0.8;
        base.genetics.efficiency = 0.7;
        base.genetics.aggression = 0.3; // Low aggression for stealth

        Self {
            base,
            stealth_factor: 0.8,
            manipulation_strength: 0.6,
        }
    }
}

#[async_trait]
impl ParasiticOrganism for CuckooOrganism {
    fn id(&self) -> Uuid {
        self.base.id
    }

    fn organism_type(&self) -> &'static str {
        "cuckoo"
    }

    fn fitness(&self) -> f64 {
        self.base.fitness
    }

    fn calculate_infection_strength(&self, vulnerability: f64) -> f64 {
        self.base.calculate_base_infection_strength(vulnerability) * self.stealth_factor
    }

    async fn infect_pair(
        &self,
        pair_id: &str,
        vulnerability: f64,
    ) -> Result<InfectionResult, OrganismError> {
        if vulnerability < 0.3 {
            return Err(OrganismError::UnsuitableConditions(
                "Low vulnerability - cuckoo requires stealth opportunities".to_string(),
            ));
        }

        let infection_strength = self.calculate_infection_strength(vulnerability);

        Ok(InfectionResult {
            success: true,
            infection_id: Uuid::new_v4(),
            initial_profit: infection_strength * 100.0, // Base profit calculation
            estimated_duration: (3600.0 * (1.0 + vulnerability)) as u64, // 1-2 hours
            resource_usage: ResourceMetrics {
                cpu_usage: 15.0, // Low CPU - stealth operations
                memory_mb: 32.0,
                network_bandwidth_kbps: 128.0,
                api_calls_per_second: 10.0,
                latency_overhead_ns: 50_000, // 50µs overhead
            },
        })
    }

    async fn adapt(&mut self, feedback: AdaptationFeedback) -> Result<(), OrganismError> {
        self.base.update_fitness(feedback.performance_score);

        // Adapt stealth based on detection risk
        if feedback.success_rate < 0.8 {
            self.stealth_factor = (self.stealth_factor * 1.1).min(1.0);
            self.base.genetics.stealth = (self.base.genetics.stealth * 1.05).min(1.0);
        }

        Ok(())
    }

    fn mutate(&mut self, rate: f64) {
        self.base.genetics.mutate(rate);

        // Cuckoo-specific mutations
        use rand::Rng;
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < rate {
            self.stealth_factor =
                (self.stealth_factor + rng.gen_range(-0.05..0.05)).clamp(0.0, 1.0);
        }
        if rng.gen::<f64>() < rate {
            self.manipulation_strength =
                (self.manipulation_strength + rng.gen_range(-0.05..0.05)).clamp(0.0, 1.0);
        }
    }

    fn crossover(
        &self,
        other: &dyn ParasiticOrganism,
    ) -> Result<Box<dyn ParasiticOrganism + Send + Sync>, OrganismError> {
        if other.organism_type() != "cuckoo" {
            return Err(OrganismError::CrossoverFailed(
                "Can only crossover with same organism type".to_string(),
            ));
        }

        let mut offspring = CuckooOrganism::new();
        offspring.base.genetics = self.base.genetics.crossover(&other.get_genetics());

        // Average the cuckoo-specific traits
        offspring.stealth_factor = (self.stealth_factor + self.stealth_factor) / 2.0; // Would need access to other's traits
        offspring.manipulation_strength =
            (self.manipulation_strength + self.manipulation_strength) / 2.0;

        Ok(Box::new(offspring))
    }

    fn get_genetics(&self) -> OrganismGenetics {
        self.base.genetics.clone()
    }

    fn set_genetics(&mut self, genetics: OrganismGenetics) {
        self.base.genetics = genetics;
    }

    fn should_terminate(&self) -> bool {
        self.base.should_terminate_base()
    }

    fn resource_consumption(&self) -> ResourceMetrics {
        ResourceMetrics {
            cpu_usage: 10.0 + self.stealth_factor * 10.0,
            memory_mb: 24.0 + self.manipulation_strength * 16.0,
            network_bandwidth_kbps: 64.0 + self.base.genetics.reaction_speed * 128.0,
            api_calls_per_second: 5.0 + self.base.genetics.aggression * 15.0,
            latency_overhead_ns: 30_000 + (self.stealth_factor * 50_000.0) as u64,
        }
    }

    fn get_strategy_params(&self) -> std::collections::HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("stealth_factor".to_string(), self.stealth_factor);
        params.insert(
            "manipulation_strength".to_string(),
            self.manipulation_strength,
        );
        params.insert("order_size_ratio".to_string(), 0.15); // Small orders for stealth
        params.insert("cancel_probability".to_string(), 0.7); // High cancel rate
        params.insert("detection_avoidance".to_string(), 0.9); // Very high
        params
    }
}
