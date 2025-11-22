//! Bacteria parasitic organism - cooperative clustering with resource sharing

use async_trait::async_trait;
use rand::Rng;
use std::collections::HashMap;
use uuid::Uuid;

use super::{
    AdaptationFeedback, BaseOrganism, InfectionResult, OrganismError, OrganismGenetics,
    ParasiticOrganism, ResourceMetrics,
};

/// Bacteria organism - forms cooperative colonies
/// Specializes in collective behavior and resource efficiency
#[derive(Debug, Clone)]
pub struct BacteriaOrganism {
    base: BaseOrganism,
    cooperation_strength: f64,
    colony_size: u32,
    resource_sharing_efficiency: f64,
    biofilm_strength: f64,
    quorum_sensing_threshold: f64,
}

impl BacteriaOrganism {
    pub fn new() -> Self {
        let mut base = BaseOrganism::new();
        // Bacteria genetic characteristics
        base.genetics.cooperation = 0.95; // Extremely cooperative
        base.genetics.efficiency = 0.85; // Very efficient
        base.genetics.resilience = 0.8; // High resilience through cooperation
        base.genetics.adaptability = 0.7; // Moderate adaptability
        base.genetics.aggression = 0.2; // Low individual aggression
        base.genetics.stealth = 0.6; // Moderate stealth through numbers

        Self {
            base,
            cooperation_strength: 0.8,
            colony_size: 10,
            resource_sharing_efficiency: 0.9,
            biofilm_strength: 0.7,
            quorum_sensing_threshold: 0.6,
        }
    }

    /// Calculate collective efficiency based on colony size
    fn calculate_collective_efficiency(&self) -> f64 {
        // Efficiency increases with cooperation and colony size
        let size_factor = (self.colony_size as f64).log2() / 10.0;
        (self.cooperation_strength + size_factor).min(2.0)
    }

    /// Determine if quorum sensing threshold is reached
    fn quorum_reached(&self, local_density: f64) -> bool {
        local_density >= self.quorum_sensing_threshold
    }

    /// Calculate biofilm defensive strength
    fn biofilm_defense(&self) -> f64 {
        self.biofilm_strength * self.cooperation_strength * (self.colony_size as f64 / 10.0)
    }

    /// Calculate resource sharing benefit
    fn resource_sharing_benefit(&self) -> f64 {
        self.resource_sharing_efficiency * self.cooperation_strength
    }
}

#[async_trait]
impl ParasiticOrganism for BacteriaOrganism {
    fn id(&self) -> Uuid {
        self.base.id
    }
    fn organism_type(&self) -> &'static str {
        "bacteria"
    }
    fn fitness(&self) -> f64 {
        self.base.fitness
    }

    fn calculate_infection_strength(&self, vulnerability: f64) -> f64 {
        self.base.calculate_base_infection_strength(vulnerability) * self.cooperation_strength
    }

    async fn infect_pair(
        &self,
        pair_id: &str,
        vulnerability: f64,
    ) -> Result<InfectionResult, OrganismError> {
        // Bacteria can work with lower individual vulnerability but need sustainable conditions
        if vulnerability < 0.15 {
            return Err(OrganismError::UnsuitableConditions(
                format!("Insufficient vulnerability ({:.2}) for bacterial colonization - minimum 0.15 required", vulnerability)
            ));
        }

        // Mock local density - in real implementation would check for other bacteria
        let local_density = 0.4;

        let collective_efficiency = self.calculate_collective_efficiency();
        let defense_strength = self.biofilm_defense();
        let sharing_benefit = self.resource_sharing_benefit();

        // Check if conditions support colony formation
        let colony_viability = vulnerability * collective_efficiency * (1.0 + sharing_benefit);
        if colony_viability < 0.3 {
            return Err(OrganismError::InfectionFailed(
                "Conditions insufficient for sustainable bacterial colony".to_string(),
            ));
        }

        let infection_strength =
            self.calculate_infection_strength(vulnerability) * collective_efficiency;

        // Long-duration, sustainable infection
        let base_duration = 14400; // 4 hours
        let sustainability_factor = 1.0 + defense_strength + sharing_benefit;
        let actual_duration = (base_duration as f64 * sustainability_factor) as u64;

        // Resource efficiency through cooperation
        let resource_efficiency = 1.0 + sharing_benefit;

        Ok(InfectionResult {
            success: true,
            infection_id: Uuid::new_v4(),
            initial_profit: infection_strength * 50.0 * sustainability_factor,
            estimated_duration: actual_duration,
            resource_usage: ResourceMetrics {
                cpu_usage: (25.0 / resource_efficiency) + (self.colony_size as f64 * 2.0),
                memory_mb: (48.0 / resource_efficiency) + (collective_efficiency * 24.0),
                network_bandwidth_kbps: 256.0 + (self.cooperation_strength * 128.0),
                api_calls_per_second: 8.0 + (self.colony_size as f64 * 0.5),
                latency_overhead_ns: 25_000
                    + (if self.quorum_reached(local_density) {
                        5000
                    } else {
                        15000
                    }),
            },
        })
    }

    async fn adapt(&mut self, feedback: AdaptationFeedback) -> Result<(), OrganismError> {
        self.base.update_fitness(feedback.performance_score);

        // Bacterial colony adaptation
        if feedback.success_rate > 0.8 {
            // Successful cooperation - expand colony
            self.colony_size = (self.colony_size + 1).min(50);
            self.cooperation_strength = (self.cooperation_strength * 1.02).min(1.0);
        } else if feedback.success_rate < 0.5 {
            // Poor performance - strengthen cooperation
            self.cooperation_strength = (self.cooperation_strength * 1.05).min(1.0);
            self.resource_sharing_efficiency = (self.resource_sharing_efficiency * 1.03).min(1.0);
        }

        // Adapt to resource scarcity
        if feedback.profit_generated < 25.0 {
            // Improve resource efficiency
            self.resource_sharing_efficiency = (self.resource_sharing_efficiency * 1.05).min(1.0);
            self.base.genetics.efficiency = (self.base.genetics.efficiency * 1.03).min(1.0);
        }

        // Defensive adaptations
        if feedback.competition_level > 0.6 {
            // Strengthen biofilm defense
            self.biofilm_strength = (self.biofilm_strength * 1.1).min(1.0);
            self.base.genetics.resilience = (self.base.genetics.resilience * 1.02).min(1.0);
        }

        // Colony size management based on resources
        let resource_pressure = feedback.avg_latency_ns as f64 / 50_000.0; // Normalize
        if resource_pressure > 1.5 {
            // Resource strain - reduce colony size
            self.colony_size = (self.colony_size.saturating_sub(1)).max(5);
        }

        // Quorum sensing adaptation
        if feedback.market_conditions.volatility > 0.05 {
            // High volatility - lower threshold for collective action
            self.quorum_sensing_threshold = (self.quorum_sensing_threshold * 0.98).max(0.3);
        }

        Ok(())
    }

    fn mutate(&mut self, rate: f64) {
        self.base.genetics.mutate(rate);

        // Bacterial mutations are often beneficial due to horizontal gene transfer
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < rate {
            self.cooperation_strength =
                (self.cooperation_strength + rng.gen_range(-0.02..0.05)).clamp(0.5, 1.0);
        }
        if rng.gen::<f64>() < rate {
            // Colony size can change through division or death
            let size_change = rng.gen_range(-2..3);
            self.colony_size = ((self.colony_size as i32) + size_change).clamp(3, 100) as u32;
        }
        if rng.gen::<f64>() < rate {
            self.resource_sharing_efficiency =
                (self.resource_sharing_efficiency + rng.gen_range(-0.02..0.03)).clamp(0.6, 1.0);
        }
        if rng.gen::<f64>() < rate {
            self.biofilm_strength =
                (self.biofilm_strength + rng.gen_range(-0.05..0.05)).clamp(0.3, 1.0);
        }
        if rng.gen::<f64>() < rate {
            self.quorum_sensing_threshold =
                (self.quorum_sensing_threshold + rng.gen_range(-0.05..0.05)).clamp(0.2, 0.9);
        }
    }

    fn crossover(
        &self,
        other: &dyn ParasiticOrganism,
    ) -> Result<Box<dyn ParasiticOrganism + Send + Sync>, OrganismError> {
        if other.organism_type() != "bacteria" {
            return Err(OrganismError::CrossoverFailed(
                "Bacteria can only crossover with another bacteria organism".to_string(),
            ));
        }

        // Cast to bacteria to access bacteria-specific traits
        let other_bacteria =
            unsafe { &*(other as *const dyn ParasiticOrganism as *const BacteriaOrganism) };

        let mut offspring = BacteriaOrganism::new();
        offspring.base.genetics = self.base.genetics.crossover(&other.get_genetics());

        // Bacterial horizontal gene transfer - mix traits more freely
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // Average cooperative traits for stability
        offspring.cooperation_strength =
            (self.cooperation_strength + other_bacteria.cooperation_strength) / 2.0;
        offspring.resource_sharing_efficiency =
            (self.resource_sharing_efficiency + other_bacteria.resource_sharing_efficiency) / 2.0;

        // Take best traits for efficiency
        offspring.biofilm_strength = self.biofilm_strength.max(other_bacteria.biofilm_strength);

        // Random selection for discrete traits
        offspring.colony_size = if rng.gen::<bool>() {
            self.colony_size
        } else {
            other_bacteria.colony_size
        };

        offspring.quorum_sensing_threshold = if rng.gen::<bool>() {
            self.quorum_sensing_threshold
        } else {
            other_bacteria.quorum_sensing_threshold
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
        let efficiency_factor = self.resource_sharing_benefit();
        let base_consumption = ResourceMetrics {
            cpu_usage: 20.0,
            memory_mb: 40.0,
            network_bandwidth_kbps: 200.0,
            api_calls_per_second: 6.0,
            latency_overhead_ns: 30_000,
        };

        // Apply efficiency gains from cooperation
        ResourceMetrics {
            cpu_usage: base_consumption.cpu_usage / (1.0 + efficiency_factor)
                + (self.colony_size as f64 * 1.5),
            memory_mb: base_consumption.memory_mb / (1.0 + efficiency_factor)
                + (self.colony_size as f64 * 2.0),
            network_bandwidth_kbps: base_consumption.network_bandwidth_kbps
                + (self.cooperation_strength * 100.0),
            api_calls_per_second: base_consumption.api_calls_per_second
                + (self.colony_size as f64 * 0.3),
            latency_overhead_ns: base_consumption.latency_overhead_ns
                - ((efficiency_factor * 10_000.0) as u64),
        }
    }

    fn get_strategy_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert(
            "cooperation_strength".to_string(),
            self.cooperation_strength,
        );
        params.insert("colony_size".to_string(), self.colony_size as f64);
        params.insert(
            "resource_sharing_efficiency".to_string(),
            self.resource_sharing_efficiency,
        );
        params.insert("biofilm_strength".to_string(), self.biofilm_strength);
        params.insert(
            "quorum_sensing_threshold".to_string(),
            self.quorum_sensing_threshold,
        );
        params.insert(
            "collective_efficiency".to_string(),
            self.calculate_collective_efficiency(),
        );
        params.insert("defensive_strength".to_string(), self.biofilm_defense());
        params.insert("sustainability_factor".to_string(), 1.5); // High sustainability
        params.insert("horizontal_gene_transfer".to_string(), 0.8); // High genetic exchange rate
        params
    }
}
