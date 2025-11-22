//! Wasp parasitic organism - aggressive high-frequency trading

use async_trait::async_trait;
use rand::Rng;
use std::collections::HashMap;
use uuid::Uuid;

use super::{
    AdaptationFeedback, BaseOrganism, InfectionResult, OrganismError, OrganismGenetics,
    ParasiticOrganism, ResourceMetrics,
};

/// Wasp organism - aggressive HFT with territorial behavior
/// Specializes in rapid execution and territory control
#[derive(Debug, Clone)]
pub struct WaspOrganism {
    base: BaseOrganism,
    aggression_multiplier: f64,
    territory_radius: f64,
    attack_frequency: f64,
    energy_consumption: f64,
}

impl WaspOrganism {
    pub fn new() -> Self {
        let mut base = BaseOrganism::new();
        // Wasp genetic predispositions
        base.genetics.aggression = 0.9;
        base.genetics.reaction_speed = 0.85;
        base.genetics.risk_tolerance = 0.75;
        base.genetics.efficiency = 0.6; // High energy consumption
        base.genetics.stealth = 0.2; // Low stealth - aggressive

        Self {
            base,
            aggression_multiplier: 1.5,
            territory_radius: 0.01, // 1% price range
            attack_frequency: 50.0, // 50 attacks per second
            energy_consumption: 0.8,
        }
    }

    /// Calculate territorial advantage based on market conditions
    fn calculate_territorial_advantage(&self, market_volatility: f64) -> f64 {
        // Higher volatility = more opportunities for aggressive tactics
        (market_volatility * self.territory_radius * self.aggression_multiplier).min(2.0)
    }

    /// Determine if territory is contested by other organisms
    fn assess_competition(&self, _pair_id: &str) -> f64 {
        // Mock implementation - would analyze active organisms in the area
        0.3 // 30% competition level
    }
}

#[async_trait]
impl ParasiticOrganism for WaspOrganism {
    fn id(&self) -> Uuid {
        self.base.id
    }
    fn organism_type(&self) -> &'static str {
        "wasp"
    }
    fn fitness(&self) -> f64 {
        self.base.fitness
    }

    fn calculate_infection_strength(&self, vulnerability: f64) -> f64 {
        self.base.calculate_base_infection_strength(vulnerability) * self.aggression_multiplier
    }

    async fn infect_pair(
        &self,
        pair_id: &str,
        vulnerability: f64,
    ) -> Result<InfectionResult, OrganismError> {
        // Wasps require high vulnerability and low competition
        if vulnerability < 0.4 {
            return Err(OrganismError::UnsuitableConditions(format!(
                "Insufficient vulnerability ({:.2}) for wasp aggression - minimum 0.4 required",
                vulnerability
            )));
        }

        let competition_level = self.assess_competition(pair_id);
        if competition_level > 0.7 {
            return Err(OrganismError::InfectionFailed(
                "Territory too contested for wasp establishment".to_string(),
            ));
        }

        let territorial_advantage = self.calculate_territorial_advantage(vulnerability);
        let infection_strength =
            self.calculate_infection_strength(vulnerability) * territorial_advantage;

        // High-intensity, short-duration infection
        let base_duration = 1800; // 30 minutes base
        let actual_duration = (base_duration as f64 * (2.0 - vulnerability)) as u64;

        Ok(InfectionResult {
            success: true,
            infection_id: Uuid::new_v4(),
            initial_profit: infection_strength * 200.0,
            estimated_duration: actual_duration,
            resource_usage: ResourceMetrics {
                cpu_usage: 85.0 + (self.attack_frequency * 0.5),
                memory_mb: 128.0 + (territorial_advantage * 64.0),
                network_bandwidth_kbps: 2048.0 + (self.attack_frequency * 20.0),
                api_calls_per_second: self.attack_frequency,
                latency_overhead_ns: 5_000, // 5µs - very fast reactions
            },
        })
    }

    async fn adapt(&mut self, feedback: AdaptationFeedback) -> Result<(), OrganismError> {
        self.base.update_fitness(feedback.performance_score);

        // Wasp-specific adaptations based on market feedback
        if feedback.success_rate < 0.7 {
            // Increase aggression when success rate drops
            self.aggression_multiplier = (self.aggression_multiplier * 1.05).min(2.5);
            self.base.genetics.aggression = (self.base.genetics.aggression * 1.02).min(1.0);
        }

        if feedback.avg_latency_ns > 10_000 {
            // Optimize for speed when latency is high
            self.attack_frequency = (self.attack_frequency * 1.1).min(100.0);
            self.base.genetics.reaction_speed = (self.base.genetics.reaction_speed * 1.05).min(1.0);
        }

        // Adjust territory based on competition
        if feedback.competition_level > 0.6 {
            self.territory_radius = (self.territory_radius * 0.95).max(0.005);
        } else if feedback.competition_level < 0.3 {
            self.territory_radius = (self.territory_radius * 1.05).min(0.05);
        }

        // Energy management
        if feedback.profit_generated < 50.0 {
            self.energy_consumption = (self.energy_consumption * 0.98).max(0.5);
        }

        Ok(())
    }

    fn mutate(&mut self, rate: f64) {
        self.base.genetics.mutate(rate);

        // Wasp-specific mutations
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < rate {
            self.aggression_multiplier =
                (self.aggression_multiplier + rng.gen_range(-0.1..0.1)).clamp(1.0, 3.0);
        }
        if rng.gen::<f64>() < rate {
            self.territory_radius =
                (self.territory_radius + rng.gen_range(-0.002..0.002)).clamp(0.001, 0.1);
        }
        if rng.gen::<f64>() < rate {
            self.attack_frequency =
                (self.attack_frequency + rng.gen_range(-5.0..5.0)).clamp(10.0, 100.0);
        }
        if rng.gen::<f64>() < rate {
            self.energy_consumption =
                (self.energy_consumption + rng.gen_range(-0.05..0.05)).clamp(0.3, 1.0);
        }
    }

    fn crossover(
        &self,
        other: &dyn ParasiticOrganism,
    ) -> Result<Box<dyn ParasiticOrganism + Send + Sync>, OrganismError> {
        if other.organism_type() != "wasp" {
            return Err(OrganismError::CrossoverFailed(
                "Wasp can only crossover with another wasp organism".to_string(),
            ));
        }

        // Cast to wasp to access wasp-specific traits
        let other_wasp =
            unsafe { &*(other as *const dyn ParasiticOrganism as *const WaspOrganism) };

        let mut offspring = WaspOrganism::new();
        offspring.base.genetics = self.base.genetics.crossover(&other.get_genetics());

        // Crossover wasp-specific traits
        use rand::Rng;
        let mut rng = rand::thread_rng();

        offspring.aggression_multiplier = if rng.gen::<bool>() {
            self.aggression_multiplier
        } else {
            other_wasp.aggression_multiplier
        };

        offspring.territory_radius = (self.territory_radius + other_wasp.territory_radius) / 2.0;
        offspring.attack_frequency = if rng.gen::<bool>() {
            self.attack_frequency
        } else {
            other_wasp.attack_frequency
        };
        offspring.energy_consumption =
            (self.energy_consumption + other_wasp.energy_consumption) / 2.0;

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
            cpu_usage: 80.0,
            memory_mb: 120.0,
            network_bandwidth_kbps: 1024.0,
            api_calls_per_second: 45.0,
            latency_overhead_ns: 8_000,
        }
    }

    fn get_strategy_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert(
            "aggression_multiplier".to_string(),
            self.aggression_multiplier,
        );
        params.insert("territory_radius".to_string(), self.territory_radius);
        params.insert("attack_frequency".to_string(), self.attack_frequency);
        params.insert("energy_consumption".to_string(), self.energy_consumption);
        params.insert("order_size_multiplier".to_string(), 1.5); // Large aggressive orders
        params.insert("cancel_aggressiveness".to_string(), 0.9); // Aggressive cancellations
        params.insert("market_impact_tolerance".to_string(), 0.8); // High impact tolerance
        params.insert("territorial_defense".to_string(), 0.95); // Strong defense
        params
    }
}
