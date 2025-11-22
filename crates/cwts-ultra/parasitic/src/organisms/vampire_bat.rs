//! Vampire Bat parasitic organism - liquidity draining specialist
//!
//! The Vampire Bat organism specializes in gradually draining liquidity from
//! trading pairs by exploiting spread inefficiencies and order book imbalances.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};
use uuid::Uuid;

use super::{
    AdaptationFeedback, BaseOrganism, InfectionResult, OrganismError, OrganismGenetics,
    ParasiticOrganism, ResourceMetrics,
};

/// Configuration for Vampire Bat organism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VampireBatConfig {
    /// Maximum liquidity to drain per cycle (ratio)
    pub max_drain_ratio: f64,
    /// Feeding frequency in milliseconds
    pub feeding_interval_ms: u64,
    /// Blood tracking radius for prey detection
    pub blood_scent_radius: f64,
    /// Echolocation precision for market sensing
    pub echolocation_precision: f64,
    /// Colony coordination strength
    pub colony_coordination: f64,
    /// Stealth flight mode to avoid detection
    pub stealth_mode: bool,
}

impl Default for VampireBatConfig {
    fn default() -> Self {
        Self {
            max_drain_ratio: 0.15,
            feeding_interval_ms: 500,
            blood_scent_radius: 0.05, // 5% spread detection
            echolocation_precision: 0.9,
            colony_coordination: 0.7,
            stealth_mode: true,
        }
    }
}

/// Vampire Bat organism - liquidity draining specialist
#[derive(Debug)]
pub struct VampireBatOrganism {
    base: BaseOrganism,
    config: VampireBatConfig,
    /// Current blood level (accumulated profit)
    blood_level: f64,
    /// Prey tracking state
    tracked_prey: Vec<PreyTarget>,
    /// Last feeding time
    last_feeding: DateTime<Utc>,
    /// Colony members for coordination
    colony_members: Vec<Uuid>,
    /// Echolocation memory for market patterns
    echolocation_memory: Vec<MarketEcho>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PreyTarget {
    pub pair_id: String,
    pub vulnerability_score: f64,
    pub liquidity_estimate: f64,
    pub last_scan_time: DateTime<Utc>,
    pub feeding_sessions: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MarketEcho {
    pub pair_id: String,
    pub echo_strength: f64,
    pub pattern_signature: Vec<f64>,
    pub timestamp: DateTime<Utc>,
}

impl VampireBatOrganism {
    pub fn new() -> Self {
        Self::with_config(VampireBatConfig::default())
    }

    pub fn with_config(config: VampireBatConfig) -> Self {
        let mut base = BaseOrganism::new();

        // Vampire bats excel at persistence and precision
        base.genetics.resilience = 0.85;
        base.genetics.efficiency = 0.80;
        base.genetics.stealth = if config.stealth_mode { 0.90 } else { 0.60 };
        base.genetics.cooperation = config.colony_coordination;
        base.genetics.adaptability = 0.75;
        base.genetics.aggression = 0.45; // Moderate aggression - methodical

        Self {
            base,
            config,
            blood_level: 0.0,
            tracked_prey: Vec::new(),
            last_feeding: Utc::now(),
            colony_members: Vec::new(),
            echolocation_memory: Vec::new(),
        }
    }

    /// Echolocation - scan for market vulnerabilities
    pub async fn echolocate_market(
        &mut self,
        pairs: Vec<String>,
    ) -> Result<Vec<PreyTarget>, OrganismError> {
        debug!("🦇 Vampire bat echolocating {} pairs", pairs.len());

        let mut new_prey = Vec::new();

        for pair_id in pairs {
            // Simulate echolocation analysis
            let vulnerability = self.analyze_liquidity_vulnerability(&pair_id).await?;

            if vulnerability > 0.3 {
                // Minimum blood scent threshold
                let prey = PreyTarget {
                    pair_id: pair_id.clone(),
                    vulnerability_score: vulnerability,
                    liquidity_estimate: vulnerability * 1000.0, // Rough liquidity estimate
                    last_scan_time: Utc::now(),
                    feeding_sessions: 0,
                };

                new_prey.push(prey.clone());

                // Store echolocation memory
                let echo = MarketEcho {
                    pair_id,
                    echo_strength: vulnerability,
                    pattern_signature: vec![vulnerability, self.config.echolocation_precision],
                    timestamp: Utc::now(),
                };

                self.echolocation_memory.push(echo);

                // Keep memory bounded
                if self.echolocation_memory.len() > 100 {
                    self.echolocation_memory.remove(0);
                }
            }
        }

        self.tracked_prey.extend(new_prey.clone());
        info!("🦇 Vampire bat found {} new prey targets", new_prey.len());

        Ok(new_prey)
    }

    /// Analyze liquidity vulnerability using echolocation
    async fn analyze_liquidity_vulnerability(&self, pair_id: &str) -> Result<f64, OrganismError> {
        // Advanced vulnerability analysis combining multiple factors
        let base_vulnerability = fastrand::f64() * 0.8 + 0.2;

        // Factor in echolocation precision
        let precision_factor = self.config.echolocation_precision;

        // Check historical patterns from memory
        let historical_factor = self
            .echolocation_memory
            .iter()
            .filter(|echo| echo.pair_id == pair_id)
            .last()
            .map(|echo| echo.echo_strength * 0.3)
            .unwrap_or(0.0);

        let vulnerability = (base_vulnerability * precision_factor + historical_factor).min(1.0);

        Ok(vulnerability)
    }

    /// Feed on a prey target (drain liquidity gradually)
    pub async fn feed_on_prey(&mut self, prey_id: &str) -> Result<f64, OrganismError> {
        let now = Utc::now();
        let time_since_last_feeding = (now - self.last_feeding).num_milliseconds() as u64;

        // Check feeding interval
        if time_since_last_feeding < self.config.feeding_interval_ms {
            return Err(OrganismError::ResourceExhausted(
                "Too early to feed again".to_string(),
            ));
        }

        // Find prey target
        let prey_idx = self
            .tracked_prey
            .iter_mut()
            .position(|prey| prey.pair_id == prey_id)
            .ok_or_else(|| OrganismError::InfectionFailed("Prey not found".to_string()))?;

        let prey = &mut self.tracked_prey[prey_idx];

        // Calculate feeding amount based on vulnerability and config
        let feeding_amount = prey.vulnerability_score
            * self.config.max_drain_ratio
            * (1.0 + self.base.genetics.efficiency * 0.5);

        // Increase blood level
        self.blood_level += feeding_amount;
        prey.feeding_sessions += 1;
        prey.last_scan_time = now;
        self.last_feeding = now;

        info!(
            "🩸 Vampire bat fed on {} - drained {:.4} liquidity",
            prey_id, feeding_amount
        );

        // Remove depleted prey (vulnerability drops after feeding)
        prey.vulnerability_score *= 0.85; // Reduce vulnerability
        if prey.vulnerability_score < 0.1 {
            self.tracked_prey.remove(prey_idx);
            debug!("🦇 Prey {} depleted, removed from tracking", prey_id);
        }

        Ok(feeding_amount)
    }

    /// Colony coordination - share prey information
    pub async fn coordinate_with_colony(
        &mut self,
        colony_members: Vec<Uuid>,
    ) -> Result<(), OrganismError> {
        self.colony_members = colony_members;

        // In a real implementation, this would communicate with other vampire bats
        // to share prey information and coordinate feeding schedules
        info!(
            "🦇 Coordinating with {} colony members",
            self.colony_members.len()
        );

        Ok(())
    }

    /// Get current hunting status
    pub fn get_hunting_status(&self) -> VampireBatStatus {
        VampireBatStatus {
            blood_level: self.blood_level,
            tracked_prey_count: self.tracked_prey.len(),
            colony_size: self.colony_members.len(),
            echolocation_memory_size: self.echolocation_memory.len(),
            last_feeding: self.last_feeding,
            stealth_active: self.config.stealth_mode,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VampireBatStatus {
    pub blood_level: f64,
    pub tracked_prey_count: usize,
    pub colony_size: usize,
    pub echolocation_memory_size: usize,
    pub last_feeding: DateTime<Utc>,
    pub stealth_active: bool,
}

#[async_trait]
impl ParasiticOrganism for VampireBatOrganism {
    fn id(&self) -> Uuid {
        self.base.id
    }

    fn organism_type(&self) -> &'static str {
        "vampire_bat"
    }

    fn fitness(&self) -> f64 {
        // Fitness includes blood level and successful prey tracking
        let base_fitness = self.base.fitness;
        let blood_bonus = (self.blood_level / 100.0).min(0.3);
        let prey_bonus = (self.tracked_prey.len() as f64 / 10.0).min(0.2);

        (base_fitness + blood_bonus + prey_bonus).min(1.0)
    }

    fn calculate_infection_strength(&self, vulnerability: f64) -> f64 {
        let base_strength = self.base.calculate_base_infection_strength(vulnerability);

        // Vampire bats are more effective at higher blood levels
        let blood_multiplier = 1.0 + (self.blood_level / 50.0).min(0.5);

        // Echolocation precision enhances effectiveness
        let precision_multiplier = 1.0 + (self.config.echolocation_precision - 0.5) * 0.4;

        base_strength * blood_multiplier * precision_multiplier
    }

    async fn infect_pair(
        &self,
        pair_id: &str,
        vulnerability: f64,
    ) -> Result<InfectionResult, OrganismError> {
        // Vampire bats need reasonable vulnerability to "smell blood"
        if vulnerability < self.config.blood_scent_radius {
            return Err(OrganismError::UnsuitableConditions(format!(
                "Vulnerability {:.3} below blood scent threshold {:.3}",
                vulnerability, self.config.blood_scent_radius
            )));
        }

        let infection_strength = self.calculate_infection_strength(vulnerability);

        // Estimate feeding duration based on liquidity
        let estimated_duration = (7200.0 * (2.0 - vulnerability)) as u64; // 2-4 hours

        Ok(InfectionResult {
            success: true,
            infection_id: Uuid::new_v4(),
            initial_profit: infection_strength * 80.0, // Conservative initial profit
            estimated_duration,
            resource_usage: ResourceMetrics {
                cpu_usage: 20.0 + self.config.echolocation_precision * 15.0,
                memory_mb: 48.0 + self.tracked_prey.len() as f64 * 2.0,
                network_bandwidth_kbps: 256.0,
                api_calls_per_second: 15.0,
                latency_overhead_ns: 75_000, // 75µs for echolocation processing
            },
        })
    }

    async fn adapt(&mut self, feedback: AdaptationFeedback) -> Result<(), OrganismError> {
        self.base.update_fitness(feedback.performance_score);

        // Adapt echolocation precision based on success rate
        if feedback.success_rate > 0.8 {
            self.config.echolocation_precision =
                (self.config.echolocation_precision * 1.02).min(1.0);
        } else if feedback.success_rate < 0.6 {
            self.config.echolocation_precision =
                (self.config.echolocation_precision * 0.98).max(0.5);
        }

        // Adapt feeding interval based on profit
        if feedback.profit_generated > self.blood_level * 0.1 {
            // Successful feeding - can be more aggressive
            self.config.feeding_interval_ms =
                ((self.config.feeding_interval_ms as f64 * 0.95) as u64).max(100);
        } else {
            // Poor feeding - be more patient
            self.config.feeding_interval_ms =
                ((self.config.feeding_interval_ms as f64 * 1.05) as u64).min(2000);
        }

        // Increase blood level from successful adaptations
        if feedback.performance_score > 0.7 {
            self.blood_level += feedback.profit_generated * 0.1;
        }

        Ok(())
    }

    fn mutate(&mut self, rate: f64) {
        self.base.genetics.mutate(rate);

        use rand::Rng;
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < rate {
            self.config.max_drain_ratio =
                (self.config.max_drain_ratio + rng.gen_range(-0.02..0.02)).clamp(0.05, 0.3);
        }
        if rng.gen::<f64>() < rate {
            self.config.echolocation_precision =
                (self.config.echolocation_precision + rng.gen_range(-0.05..0.05)).clamp(0.5, 1.0);
        }
        if rng.gen::<f64>() < rate {
            self.config.colony_coordination =
                (self.config.colony_coordination + rng.gen_range(-0.1..0.1)).clamp(0.0, 1.0);
        }
        if rng.gen::<f64>() < rate {
            self.config.feeding_interval_ms =
                (self.config.feeding_interval_ms as f64 * rng.gen_range(0.8..1.2)).max(50.0) as u64;
        }
    }

    fn crossover(
        &self,
        other: &dyn ParasiticOrganism,
    ) -> Result<Box<dyn ParasiticOrganism + Send + Sync>, OrganismError> {
        if other.organism_type() != "vampire_bat" {
            return Err(OrganismError::CrossoverFailed(
                "Can only crossover with same organism type".to_string(),
            ));
        }

        let mut offspring = VampireBatOrganism::new();
        offspring.base.genetics = self.base.genetics.crossover(&other.get_genetics());

        // Average the vampire bat specific traits
        offspring.config.max_drain_ratio =
            (self.config.max_drain_ratio + self.config.max_drain_ratio) / 2.0;
        offspring.config.echolocation_precision =
            (self.config.echolocation_precision + self.config.echolocation_precision) / 2.0;
        offspring.config.colony_coordination =
            (self.config.colony_coordination + self.config.colony_coordination) / 2.0;

        Ok(Box::new(offspring))
    }

    fn get_genetics(&self) -> OrganismGenetics {
        self.base.genetics.clone()
    }

    fn set_genetics(&mut self, genetics: OrganismGenetics) {
        self.base.genetics = genetics;
    }

    fn should_terminate(&self) -> bool {
        // Terminate if blood level is critically low and base fitness is poor
        let critical_blood = self.blood_level < 1.0 && self.base.fitness < 0.2;
        critical_blood || self.base.should_terminate_base()
    }

    fn resource_consumption(&self) -> ResourceMetrics {
        let base_consumption = 15.0;
        let echolocation_cost = self.config.echolocation_precision * 20.0;
        let memory_cost = self.tracked_prey.len() as f64 * 1.5;

        ResourceMetrics {
            cpu_usage: base_consumption + echolocation_cost,
            memory_mb: 32.0 + memory_cost,
            network_bandwidth_kbps: 128.0 + self.base.genetics.reaction_speed * 128.0,
            api_calls_per_second: 8.0 + (self.tracked_prey.len() as f64 * 2.0),
            latency_overhead_ns: 60_000 + ((1.0 - self.base.genetics.efficiency) * 40_000.0) as u64,
        }
    }

    fn get_strategy_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert("max_drain_ratio".to_string(), self.config.max_drain_ratio);
        params.insert(
            "feeding_interval_ms".to_string(),
            self.config.feeding_interval_ms as f64,
        );
        params.insert(
            "blood_scent_radius".to_string(),
            self.config.blood_scent_radius,
        );
        params.insert(
            "echolocation_precision".to_string(),
            self.config.echolocation_precision,
        );
        params.insert(
            "colony_coordination".to_string(),
            self.config.colony_coordination,
        );
        params.insert("current_blood_level".to_string(), self.blood_level);
        params.insert(
            "tracked_prey_count".to_string(),
            self.tracked_prey.len() as f64,
        );
        params.insert(
            "stealth_mode".to_string(),
            if self.config.stealth_mode { 1.0 } else { 0.0 },
        );
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_vampire_bat_creation() {
        let bat = VampireBatOrganism::new();
        assert_eq!(bat.organism_type(), "vampire_bat");
        assert_eq!(bat.blood_level, 0.0);
        assert!(bat.tracked_prey.is_empty());
    }

    #[tokio::test]
    async fn test_echolocation() {
        let mut bat = VampireBatOrganism::new();
        let pairs = vec!["BTCUSD".to_string(), "ETHUSD".to_string()];

        let prey = bat.echolocate_market(pairs).await.unwrap();
        // Should find some prey based on random vulnerability
        assert!(!bat.echolocation_memory.is_empty());
    }

    #[test]
    fn test_fitness_calculation() {
        let mut bat = VampireBatOrganism::new();
        bat.blood_level = 50.0;
        bat.tracked_prey.push(PreyTarget {
            pair_id: "BTCUSD".to_string(),
            vulnerability_score: 0.8,
            liquidity_estimate: 1000.0,
            last_scan_time: Utc::now(),
            feeding_sessions: 0,
        });

        let fitness = bat.fitness();
        assert!(fitness > 0.5); // Should be higher due to blood level and prey
    }
}
