//! Virus parasitic organism - self-replicating across correlated pairs

use async_trait::async_trait;
use rand::Rng;
use std::collections::HashMap;
use uuid::Uuid;

use super::{
    AdaptationFeedback, BaseOrganism, InfectionResult, OrganismError, OrganismGenetics,
    ParasiticOrganism, ResourceMetrics,
};

/// Virus organism - spreads across correlation networks
/// Specializes in rapid replication and network effects
#[derive(Debug, Clone)]
pub struct VirusOrganism {
    base: BaseOrganism,
    replication_rate: f64,
    mutation_resistance: f64,
    network_spread_factor: f64,
    viral_load: f64,
    incubation_time: u64, // milliseconds
}

impl VirusOrganism {
    pub fn new() -> Self {
        let mut base = BaseOrganism::new();
        // Virus genetic characteristics
        base.genetics.adaptability = 0.95; // Extremely adaptable
        base.genetics.efficiency = 0.8; // High efficiency in resource use
        base.genetics.resilience = 0.7; // Moderate resilience
        base.genetics.reaction_speed = 0.6; // Slower initial reaction
        base.genetics.stealth = 0.9; // Very stealthy

        Self {
            base,
            replication_rate: 0.6,
            mutation_resistance: 0.8,
            network_spread_factor: 1.5,
            viral_load: 1.0,
            incubation_time: 5000, // 5 seconds
        }
    }

    /// Calculate network spread potential
    fn calculate_spread_potential(&self, correlation_strength: f64) -> f64 {
        correlation_strength * self.network_spread_factor * self.viral_load
    }

    /// Determine if virus can successfully replicate
    fn can_replicate(&self, host_resistance: f64) -> bool {
        self.replication_rate * self.mutation_resistance > host_resistance
    }

    /// Calculate incubation time based on conditions
    fn calculate_incubation(&self, market_stress: f64) -> u64 {
        // Higher market stress = shorter incubation
        (self.incubation_time as f64 * (2.0 - market_stress)).max(1000.0) as u64
    }
}

#[async_trait]
impl ParasiticOrganism for VirusOrganism {
    fn id(&self) -> Uuid {
        self.base.id
    }
    fn organism_type(&self) -> &'static str {
        "virus"
    }
    fn fitness(&self) -> f64 {
        self.base.fitness
    }

    fn calculate_infection_strength(&self, vulnerability: f64) -> f64 {
        self.base.calculate_base_infection_strength(vulnerability) * (1.0 + self.replication_rate)
    }

    async fn infect_pair(
        &self,
        pair_id: &str,
        vulnerability: f64,
    ) -> Result<InfectionResult, OrganismError> {
        // Viruses can infect with lower vulnerability but need time to spread
        if vulnerability < 0.2 {
            return Err(OrganismError::UnsuitableConditions(format!(
                "Insufficient vulnerability ({:.2}) for viral infection - minimum 0.2 required",
                vulnerability
            )));
        }

        // Mock correlation strength - in real implementation would analyze pair correlations
        let correlation_strength = 0.7;

        if !self.can_replicate(1.0 - vulnerability) {
            return Err(OrganismError::InfectionFailed(
                "Host resistance too high for viral replication".to_string(),
            ));
        }

        let spread_potential = self.calculate_spread_potential(correlation_strength);
        let infection_strength = self.calculate_infection_strength(vulnerability);
        let incubation = self.calculate_incubation(vulnerability);

        // Longer duration but network effects
        let base_duration = 7200; // 2 hours
        let network_multiplier = 1.0 + (spread_potential * 0.5);
        let actual_duration = (base_duration as f64 * network_multiplier) as u64;

        Ok(InfectionResult {
            success: true,
            infection_id: Uuid::new_v4(),
            initial_profit: infection_strength * 75.0 * network_multiplier,
            estimated_duration: actual_duration,
            resource_usage: ResourceMetrics {
                cpu_usage: 30.0 + (self.viral_load * 20.0),
                memory_mb: 64.0 + (spread_potential * 32.0),
                network_bandwidth_kbps: 512.0 + (self.replication_rate * 256.0),
                api_calls_per_second: 15.0 + (self.network_spread_factor * 10.0),
                latency_overhead_ns: incubation * 1000, // Convert to ns
            },
        })
    }

    async fn adapt(&mut self, feedback: AdaptationFeedback) -> Result<(), OrganismError> {
        self.base.update_fitness(feedback.performance_score);

        // Viral adaptation based on environmental pressure
        if feedback.success_rate < 0.6 {
            // Increase mutation resistance when facing challenges
            self.mutation_resistance = (self.mutation_resistance * 1.1).min(1.0);
            self.base.genetics.adaptability = (self.base.genetics.adaptability * 1.02).min(1.0);
        }

        // Adapt replication based on host resistance
        let estimated_resistance = 1.0 - feedback.success_rate;
        if estimated_resistance > 0.7 {
            self.replication_rate = (self.replication_rate * 1.05).min(1.0);
        }

        // Network adaptation
        if feedback.profit_generated > 100.0 {
            // Successful network effects - enhance spread factor
            self.network_spread_factor = (self.network_spread_factor * 1.05).min(3.0);
        } else {
            // Poor network performance - increase viral load
            self.viral_load = (self.viral_load * 1.03).min(2.0);
        }

        // Stealth adaptation
        if feedback.competition_level > 0.8 {
            self.incubation_time = (self.incubation_time as f64 * 1.1).min(30000.0) as u64;
            self.base.genetics.stealth = (self.base.genetics.stealth * 1.02).min(1.0);
        }

        Ok(())
    }

    fn mutate(&mut self, rate: f64) {
        // Viruses are highly mutable
        let enhanced_rate = rate * (1.0 + self.mutation_resistance);
        self.base.genetics.mutate(enhanced_rate);

        // Virus-specific mutations
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < enhanced_rate {
            self.replication_rate =
                (self.replication_rate + rng.gen_range(-0.1..0.1)).clamp(0.1, 1.0);
        }
        if rng.gen::<f64>() < enhanced_rate {
            self.mutation_resistance =
                (self.mutation_resistance + rng.gen_range(-0.05..0.05)).clamp(0.5, 1.0);
        }
        if rng.gen::<f64>() < enhanced_rate {
            self.network_spread_factor =
                (self.network_spread_factor + rng.gen_range(-0.1..0.1)).clamp(0.5, 4.0);
        }
        if rng.gen::<f64>() < enhanced_rate {
            self.viral_load = (self.viral_load + rng.gen_range(-0.1..0.1)).clamp(0.5, 3.0);
        }
        if rng.gen::<f64>() < enhanced_rate {
            let time_change = rng.gen_range(-1000..1000);
            self.incubation_time =
                ((self.incubation_time as i64) + time_change).clamp(500, 60000) as u64;
        }
    }

    fn crossover(
        &self,
        other: &dyn ParasiticOrganism,
    ) -> Result<Box<dyn ParasiticOrganism + Send + Sync>, OrganismError> {
        if other.organism_type() != "virus" {
            return Err(OrganismError::CrossoverFailed(
                "Virus can only crossover with another virus organism".to_string(),
            ));
        }

        // Cast to virus to access virus-specific traits
        let other_virus =
            unsafe { &*(other as *const dyn ParasiticOrganism as *const VirusOrganism) };

        let mut offspring = VirusOrganism::new();
        offspring.base.genetics = self.base.genetics.crossover(&other.get_genetics());

        // Viral recombination - viruses can exchange genetic material
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Average traits for stability
        offspring.replication_rate = (self.replication_rate + other_virus.replication_rate) / 2.0;
        offspring.mutation_resistance =
            (self.mutation_resistance + other_virus.mutation_resistance) / 2.0;

        // Random selection for discrete traits
        offspring.network_spread_factor = if rng.gen::<bool>() {
            self.network_spread_factor
        } else {
            other_virus.network_spread_factor
        };

        offspring.viral_load = if rng.gen::<bool>() {
            self.viral_load
        } else {
            other_virus.viral_load
        };

        offspring.incubation_time = if rng.gen::<bool>() {
            self.incubation_time
        } else {
            other_virus.incubation_time
        };

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
            cpu_usage: 25.0 + (self.viral_load * 15.0),
            memory_mb: 48.0 + (self.network_spread_factor * 24.0),
            network_bandwidth_kbps: 384.0 + (self.replication_rate * 192.0),
            api_calls_per_second: 12.0 + (self.viral_load * 8.0),
            latency_overhead_ns: self.incubation_time * 1000, // Convert to ns
        }
    }

    fn get_strategy_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("replication_rate".to_string(), self.replication_rate);
        params.insert("mutation_resistance".to_string(), self.mutation_resistance);
        params.insert(
            "network_spread_factor".to_string(),
            self.network_spread_factor,
        );
        params.insert("viral_load".to_string(), self.viral_load);
        params.insert(
            "incubation_time_ms".to_string(),
            self.incubation_time as f64,
        );
        params.insert("correlation_threshold".to_string(), 0.5); // Minimum correlation for spread
        params.insert("host_jump_probability".to_string(), 0.8); // Probability of jumping to correlated pairs
        params.insert("immune_evasion".to_string(), 0.85); // Ability to evade detection
        params
    }
}
