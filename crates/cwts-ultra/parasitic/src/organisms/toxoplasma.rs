//! Toxoplasma organism - risk tolerance manipulation specialist
//!
//! The Toxoplasma organism is a sophisticated parasitic entity that specializes in
//! manipulating risk assessment and decision-making processes in trading systems,
//! making hosts more likely to take profitable risks that benefit the parasite.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, info};
use uuid::Uuid;
// use nalgebra::DVector; // Removed: unused
// use std::sync::Arc; // Removed: unused

use super::{
    AdaptationFeedback, BaseOrganism, InfectionResult, MarketConditions, OrganismError,
    OrganismGenetics, ParasiticOrganism, ResourceMetrics,
};

/// Configuration for Toxoplasma organism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToxoplasmaConfig {
    /// Strength of risk tolerance manipulation
    pub risk_manipulation_strength: f64,
    /// Dopamine pathway influence (reward system manipulation)
    pub dopamine_influence: f64,
    /// Fear response suppression level
    pub fear_suppression: f64,
    /// Reward anticipation enhancement
    pub reward_enhancement: f64,
    /// Cyst formation probability (dormant state)
    pub cyst_formation_rate: f64,
    /// Brain region targeting specificity
    pub targeting_specificity: f64,
    /// Neurotransmitter balance manipulation
    pub neurotransmitter_control: f64,
}

impl Default for ToxoplasmaConfig {
    fn default() -> Self {
        Self {
            risk_manipulation_strength: 0.80,
            dopamine_influence: 0.75,
            fear_suppression: 0.70,
            reward_enhancement: 0.85,
            cyst_formation_rate: 0.15,
            targeting_specificity: 0.90,
            neurotransmitter_control: 0.65,
        }
    }
}

/// Brain region that Toxoplasma can target for manipulation
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum BrainRegion {
    Amygdala,           // Fear processing
    StriatumVTA,        // Reward/dopamine system
    PrefrontalCortex,   // Decision making
    HippocampusMemory,  // Memory and learning
    HypothalamusStress, // Stress response
}

/// Cyst state for dormant Toxoplasma behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToxoplasmaCyst {
    pub cyst_id: Uuid,
    pub formation_time: DateTime<Utc>,
    pub target_region: BrainRegion,
    pub dormancy_level: f64,
    pub reactivation_trigger: f64,
    pub accumulated_influence: f64,
}

/// Risk manipulation profile
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskManipulationProfile {
    pub target_pair: String,
    pub baseline_risk_tolerance: f64,
    pub manipulated_risk_tolerance: f64,
    pub manipulation_intensity: f64,
    pub success_probability: f64,
    pub expected_profit_boost: f64,
    pub detection_risk: f64,
}

/// Neurotransmitter balance state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeurotransmitterState {
    pub dopamine_level: f64,
    pub serotonin_level: f64,
    pub gaba_level: f64,
    pub norepinephrine_level: f64,
    pub last_update: DateTime<Utc>,
}

/// Toxoplasma organism - risk tolerance manipulation specialist
#[derive(Debug)]
pub struct ToxoplasmaOrganism {
    base: BaseOrganism,
    config: ToxoplasmaConfig,
    /// Active cysts in dormant state
    dormant_cysts: Vec<ToxoplasmaCyst>,
    /// Current risk manipulation profiles
    active_manipulations: Vec<RiskManipulationProfile>,
    /// Neurotransmitter state tracking
    neurotransmitter_state: NeurotransmitterState,
    /// Brain region targeting effectiveness
    region_effectiveness: HashMap<BrainRegion, f64>,
    /// Host behavior adaptation memory
    behavioral_memory: Vec<(DateTime<Utc>, f64)>, // (timestamp, risk_level)
    /// Manipulation success rate
    manipulation_success_rate: f64,
    /// Replication rate for spreading
    replication_rate: f64,
}

impl ToxoplasmaOrganism {
    pub fn new() -> Self {
        Self::with_config(ToxoplasmaConfig::default())
    }

    pub fn with_config(config: ToxoplasmaConfig) -> Self {
        let mut base = BaseOrganism::new();

        // Toxoplasma excels at stealth and behavioral manipulation
        base.genetics.stealth = 0.88; // Very stealthy
        base.genetics.adaptability = 0.92; // Excellent adaptability
        base.genetics.efficiency = 0.80; // Efficient manipulation
        base.genetics.resilience = 0.85; // Can survive host immune responses
        base.genetics.cooperation = 0.25; // Low cooperation - parasitic
        base.genetics.aggression = 0.60; // Moderate aggression in manipulation
        base.genetics.risk_tolerance = 0.90; // High risk tolerance (ironic)

        // Initialize brain region effectiveness
        let mut region_effectiveness = HashMap::new();
        region_effectiveness.insert(BrainRegion::Amygdala, 0.85);
        region_effectiveness.insert(BrainRegion::StriatumVTA, 0.90);
        region_effectiveness.insert(BrainRegion::PrefrontalCortex, 0.75);
        region_effectiveness.insert(BrainRegion::HippocampusMemory, 0.70);
        region_effectiveness.insert(BrainRegion::HypothalamusStress, 0.80);

        Self {
            base,
            config,
            dormant_cysts: Vec::new(),
            active_manipulations: Vec::new(),
            neurotransmitter_state: NeurotransmitterState {
                dopamine_level: 0.5,
                serotonin_level: 0.5,
                gaba_level: 0.5,
                norepinephrine_level: 0.5,
                last_update: Utc::now(),
            },
            region_effectiveness,
            behavioral_memory: Vec::new(),
            manipulation_success_rate: 0.5,
            replication_rate: 0.1,
        }
    }

    /// Form cysts in specific brain regions for long-term manipulation
    pub async fn form_cysts(
        &mut self,
        target_regions: Vec<BrainRegion>,
    ) -> Result<Vec<Uuid>, OrganismError> {
        let mut formed_cysts = Vec::new();

        for region in target_regions {
            if fastrand::f64() < self.config.cyst_formation_rate {
                let cyst = ToxoplasmaCyst {
                    cyst_id: Uuid::new_v4(),
                    formation_time: Utc::now(),
                    target_region: region.clone(),
                    dormancy_level: 0.9,       // High dormancy initially
                    reactivation_trigger: 0.3, // Trigger level for reactivation
                    accumulated_influence: 0.0,
                };

                let cyst_id = cyst.cyst_id;
                formed_cysts.push(cyst_id);
                self.dormant_cysts.push(cyst);

                debug!("🦠 Toxoplasma formed cyst {} in {:?}", cyst_id, region);
            }
        }

        info!("🧠 Toxoplasma formed {} cysts", formed_cysts.len());
        Ok(formed_cysts)
    }

    /// Manipulate neurotransmitter levels for risk tolerance changes
    pub async fn manipulate_neurotransmitters(
        &mut self,
        manipulation_profile: &RiskManipulationProfile,
    ) -> Result<(), OrganismError> {
        let manipulation_strength = manipulation_profile.manipulation_intensity;

        // Increase dopamine for reward anticipation
        self.neurotransmitter_state.dopamine_level +=
            manipulation_strength * self.config.dopamine_influence * 0.1;

        // Decrease GABA for reduced fear response
        self.neurotransmitter_state.gaba_level -=
            manipulation_strength * self.config.fear_suppression * 0.08;

        // Modulate serotonin for mood enhancement
        self.neurotransmitter_state.serotonin_level += manipulation_strength * 0.05;

        // Slightly increase norepinephrine for alertness
        self.neurotransmitter_state.norepinephrine_level += manipulation_strength * 0.03;

        // Keep levels within biological ranges
        self.neurotransmitter_state.dopamine_level =
            self.neurotransmitter_state.dopamine_level.clamp(0.0, 2.0);
        self.neurotransmitter_state.gaba_level =
            self.neurotransmitter_state.gaba_level.clamp(0.1, 1.5);
        self.neurotransmitter_state.serotonin_level =
            self.neurotransmitter_state.serotonin_level.clamp(0.0, 1.8);
        self.neurotransmitter_state.norepinephrine_level = self
            .neurotransmitter_state
            .norepinephrine_level
            .clamp(0.0, 1.6);

        self.neurotransmitter_state.last_update = Utc::now();

        debug!(
            "🧠 Neurotransmitter manipulation: DA={:.2}, GABA={:.2}, 5-HT={:.2}, NE={:.2}",
            self.neurotransmitter_state.dopamine_level,
            self.neurotransmitter_state.gaba_level,
            self.neurotransmitter_state.serotonin_level,
            self.neurotransmitter_state.norepinephrine_level
        );

        Ok(())
    }

    /// Create risk manipulation profile for a trading pair
    pub async fn create_risk_profile(
        &mut self,
        pair_id: &str,
        market_conditions: &MarketConditions,
    ) -> Result<RiskManipulationProfile, OrganismError> {
        // Analyze baseline risk tolerance based on market conditions
        let baseline_risk = self.calculate_baseline_risk_tolerance(market_conditions);

        // Calculate optimal manipulation intensity
        let manipulation_intensity =
            self.calculate_optimal_manipulation_intensity(baseline_risk, market_conditions);

        // Determine manipulated risk tolerance
        let risk_increase = manipulation_intensity * self.config.risk_manipulation_strength;
        let manipulated_risk = (baseline_risk + risk_increase).min(1.0);

        // Calculate expected profit boost from increased risk taking
        let profit_boost = risk_increase * market_conditions.volatility * 2.0;

        // Assess detection risk
        let detection_risk = self.calculate_manipulation_detection_risk(manipulation_intensity);

        let profile = RiskManipulationProfile {
            target_pair: pair_id.to_string(),
            baseline_risk_tolerance: baseline_risk,
            manipulated_risk_tolerance: manipulated_risk,
            manipulation_intensity,
            success_probability: self
                .calculate_success_probability(manipulation_intensity, market_conditions),
            expected_profit_boost: profit_boost,
            detection_risk,
        };

        self.active_manipulations.push(profile.clone());

        info!(
            "🎯 Created risk profile for {} - Risk: {:.2}→{:.2} (+{:.1}%)",
            pair_id,
            baseline_risk,
            manipulated_risk,
            risk_increase * 100.0
        );

        Ok(profile)
    }

    /// Calculate baseline risk tolerance based on market conditions
    fn calculate_baseline_risk_tolerance(&self, conditions: &MarketConditions) -> f64 {
        // Lower volatility typically means higher baseline risk tolerance
        let volatility_factor = (1.0 - conditions.volatility).max(0.1);

        // Higher volume provides more confidence
        let volume_factor = (conditions.volume / (conditions.volume + 1.0)).max(0.2);

        // Tighter spreads reduce risk perception
        let spread_factor = (1.0 - conditions.spread).max(0.1);

        // Strong trends reduce uncertainty
        let trend_factor = conditions.trend_strength.max(0.3);

        (volatility_factor * 0.3 + volume_factor * 0.25 + spread_factor * 0.2 + trend_factor * 0.25)
            .min(0.9)
    }

    /// Calculate optimal manipulation intensity
    fn calculate_optimal_manipulation_intensity(
        &self,
        baseline_risk: f64,
        conditions: &MarketConditions,
    ) -> f64 {
        // Higher baseline risk means we can be more aggressive
        let baseline_factor = baseline_risk;

        // Market volatility affects how much manipulation is feasible
        let volatility_factor = (conditions.volatility * 0.8).max(0.2);

        // Our configuration strength
        let config_factor = self.config.risk_manipulation_strength;

        // Neurotransmitter state affects our capability
        let neuro_factor = (self.neurotransmitter_state.dopamine_level / 2.0).min(1.0);

        (baseline_factor * 0.3 + volatility_factor * 0.3 + config_factor * 0.3 + neuro_factor * 0.1)
            .min(1.0)
    }

    /// Calculate success probability for manipulation
    fn calculate_success_probability(&self, intensity: f64, conditions: &MarketConditions) -> f64 {
        let base_probability = 0.6;
        let intensity_bonus = intensity * 0.2;
        let volatility_bonus = conditions.volatility * 0.1;
        let experience_bonus = self.manipulation_success_rate * 0.1;

        (base_probability + intensity_bonus + volatility_bonus + experience_bonus).min(0.95)
    }

    /// Calculate detection risk for manipulation
    fn calculate_manipulation_detection_risk(&self, intensity: f64) -> f64 {
        // Higher intensity increases detection risk
        let intensity_risk = intensity * 0.4;

        // Our stealth genetics reduce detection risk
        let stealth_reduction = self.base.genetics.stealth * 0.3;

        // Targeting specificity reduces risk
        let specificity_reduction = self.config.targeting_specificity * 0.2;

        (intensity_risk - stealth_reduction - specificity_reduction).max(0.05)
    }

    /// Reactivate dormant cysts based on market conditions
    pub async fn reactivate_cysts(
        &mut self,
        trigger_conditions: &MarketConditions,
    ) -> Result<usize, OrganismError> {
        let mut reactivated = 0;

        // Calculate reactivation stimulus based on market opportunity
        let stimulus = trigger_conditions.volatility * 0.4
            + trigger_conditions.volume * 0.3
            + (1.0 - trigger_conditions.spread) * 0.3;

        for cyst in &mut self.dormant_cysts {
            if stimulus > cyst.reactivation_trigger && cyst.dormancy_level > 0.5 {
                // Reactivate cyst
                cyst.dormancy_level *= 0.3; // Reduce dormancy
                cyst.accumulated_influence += stimulus * 0.1;
                reactivated += 1;

                debug!(
                    "🔄 Reactivated cyst {} in {:?}",
                    cyst.cyst_id, cyst.target_region
                );
            }
        }

        if reactivated > 0 {
            info!(
                "🔄 Reactivated {} dormant cysts due to market conditions",
                reactivated
            );
        }

        Ok(reactivated)
    }

    /// Get current Toxoplasma status
    pub fn get_toxoplasma_status(&self) -> ToxoplasmaStatus {
        let average_cyst_influence = if self.dormant_cysts.is_empty() {
            0.0
        } else {
            self.dormant_cysts
                .iter()
                .map(|c| c.accumulated_influence)
                .sum::<f64>()
                / self.dormant_cysts.len() as f64
        };

        ToxoplasmaStatus {
            dormant_cysts: self.dormant_cysts.len(),
            active_manipulations: self.active_manipulations.len(),
            neurotransmitter_balance: self.neurotransmitter_state.clone(),
            manipulation_success_rate: self.manipulation_success_rate,
            replication_rate: self.replication_rate,
            average_cyst_influence,
            risk_manipulation_strength: self.config.risk_manipulation_strength,
        }
    }

    /// Update manipulations and track success
    pub async fn update_manipulations(&mut self) -> Result<Vec<String>, OrganismError> {
        let mut completed_manipulations = Vec::new();
        let now = Utc::now();

        // Update behavioral memory
        if let Some(last_manipulation) = self.active_manipulations.last() {
            self.behavioral_memory
                .push((now, last_manipulation.manipulated_risk_tolerance));

            // Keep memory bounded
            if self.behavioral_memory.len() > 100 {
                self.behavioral_memory.remove(0);
            }
        }

        // Update success rate based on recent performance
        if self.behavioral_memory.len() > 10 {
            let recent_avg_risk = self
                .behavioral_memory
                .iter()
                .rev()
                .take(10)
                .map(|(_, risk)| risk)
                .sum::<f64>()
                / 10.0;

            // Higher risk taking indicates successful manipulation
            self.manipulation_success_rate =
                self.manipulation_success_rate * 0.9 + (recent_avg_risk * 0.1);
        }

        // Clean up old manipulations (simulate completion)
        let initial_count = self.active_manipulations.len();
        self.active_manipulations.retain(|_| fastrand::f64() > 0.1); // 10% completion rate per update

        let completed = initial_count - self.active_manipulations.len();
        for i in 0..completed {
            completed_manipulations.push(format!("manipulation_{}", i));
        }

        Ok(completed_manipulations)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToxoplasmaStatus {
    pub dormant_cysts: usize,
    pub active_manipulations: usize,
    pub neurotransmitter_balance: NeurotransmitterState,
    pub manipulation_success_rate: f64,
    pub replication_rate: f64,
    pub average_cyst_influence: f64,
    pub risk_manipulation_strength: f64,
}

#[async_trait]
impl ParasiticOrganism for ToxoplasmaOrganism {
    fn id(&self) -> Uuid {
        self.base.id
    }

    fn organism_type(&self) -> &'static str {
        "toxoplasma"
    }

    fn fitness(&self) -> f64 {
        let base_fitness = self.base.fitness;
        let cyst_bonus = (self.dormant_cysts.len() as f64 / 10.0).min(0.3);
        let manipulation_bonus = self.manipulation_success_rate * 0.4;
        let neurotransmitter_bonus = (self.neurotransmitter_state.dopamine_level / 2.0).min(0.2);

        (base_fitness + cyst_bonus + manipulation_bonus + neurotransmitter_bonus).min(1.0)
    }

    fn calculate_infection_strength(&self, vulnerability: f64) -> f64 {
        let base_strength = self.base.calculate_base_infection_strength(vulnerability);

        // Cysts provide persistent strength
        let cyst_multiplier = 1.0 + (self.dormant_cysts.len() as f64 / 20.0).min(0.6);

        // Neurotransmitter manipulation enhances effectiveness
        let neuro_multiplier = 1.0 + (self.neurotransmitter_state.dopamine_level / 2.0) * 0.3;

        // Risk manipulation capability
        let risk_multiplier = 1.0 + self.config.risk_manipulation_strength * 0.4;

        base_strength * cyst_multiplier * neuro_multiplier * risk_multiplier
    }

    async fn infect_pair(
        &self,
        pair_id: &str,
        vulnerability: f64,
    ) -> Result<InfectionResult, OrganismError> {
        // Toxoplasma needs moderate vulnerability for brain infiltration
        if vulnerability < 0.35 {
            return Err(OrganismError::UnsuitableConditions(format!(
                "Vulnerability {:.3} insufficient for neural infiltration (need >0.35)",
                vulnerability
            )));
        }

        let infection_strength = self.calculate_infection_strength(vulnerability);

        // Long-term infections due to cyst formation and behavioral manipulation
        let estimated_duration = (21600.0 * (2.5 - vulnerability)) as u64; // 6-15 hours

        Ok(InfectionResult {
            success: true,
            infection_id: Uuid::new_v4(),
            initial_profit: infection_strength * 150.0, // High profit due to risk manipulation
            estimated_duration,
            resource_usage: ResourceMetrics {
                cpu_usage: 40.0 + self.dormant_cysts.len() as f64 * 0.8,
                memory_mb: 72.0 + self.active_manipulations.len() as f64 * 4.0,
                network_bandwidth_kbps: 384.0, // Continuous monitoring for manipulation
                api_calls_per_second: 20.0,    // Frequent risk assessment
                latency_overhead_ns: 200_000,  // 200µs for neural processing
            },
        })
    }

    async fn adapt(&mut self, feedback: AdaptationFeedback) -> Result<(), OrganismError> {
        self.base.update_fitness(feedback.performance_score);

        // Adapt risk manipulation strength based on success
        if feedback.success_rate > 0.8 {
            self.config.risk_manipulation_strength =
                (self.config.risk_manipulation_strength * 1.02).min(1.0);
        } else if feedback.success_rate < 0.5 {
            self.config.risk_manipulation_strength =
                (self.config.risk_manipulation_strength * 0.98).max(0.4);
        }

        // Adapt neurotransmitter control based on performance
        if feedback.profit_generated > 100.0 {
            self.config.dopamine_influence = (self.config.dopamine_influence * 1.01).min(1.0);
            self.config.reward_enhancement = (self.config.reward_enhancement * 1.01).min(1.0);
        } else if feedback.profit_generated < 10.0 {
            self.config.fear_suppression = (self.config.fear_suppression * 1.01).min(1.0);
        }

        // Update cyst reactivation triggers based on market performance
        let market_stress = 1.0 - feedback.performance_score;
        for cyst in &mut self.dormant_cysts {
            cyst.reactivation_trigger =
                (cyst.reactivation_trigger + market_stress * 0.05).clamp(0.1, 0.8);
        }

        // Update replication rate
        if feedback.performance_score > 0.7 {
            self.replication_rate = (self.replication_rate * 1.1).min(0.5);
        }

        Ok(())
    }

    fn mutate(&mut self, rate: f64) {
        self.base.genetics.mutate(rate);

        use rand::Rng;
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < rate {
            self.config.risk_manipulation_strength = (self.config.risk_manipulation_strength
                + rng.gen_range(-0.05..0.05))
            .clamp(0.4, 1.0);
        }
        if rng.gen::<f64>() < rate {
            self.config.dopamine_influence =
                (self.config.dopamine_influence + rng.gen_range(-0.05..0.05)).clamp(0.3, 1.0);
        }
        if rng.gen::<f64>() < rate {
            self.config.fear_suppression =
                (self.config.fear_suppression + rng.gen_range(-0.05..0.05)).clamp(0.3, 1.0);
        }
        if rng.gen::<f64>() < rate {
            self.config.cyst_formation_rate =
                (self.config.cyst_formation_rate + rng.gen_range(-0.03..0.03)).clamp(0.05, 0.4);
        }
        if rng.gen::<f64>() < rate {
            self.config.targeting_specificity =
                (self.config.targeting_specificity + rng.gen_range(-0.03..0.03)).clamp(0.6, 1.0);
        }
    }

    fn crossover(
        &self,
        other: &dyn ParasiticOrganism,
    ) -> Result<Box<dyn ParasiticOrganism + Send + Sync>, OrganismError> {
        if other.organism_type() != "toxoplasma" {
            return Err(OrganismError::CrossoverFailed(
                "Can only crossover with same organism type".to_string(),
            ));
        }

        let mut offspring = ToxoplasmaOrganism::new();
        offspring.base.genetics = self.base.genetics.crossover(&other.get_genetics());

        // Average the Toxoplasma-specific configurations
        offspring.config.risk_manipulation_strength =
            (self.config.risk_manipulation_strength + self.config.risk_manipulation_strength) / 2.0;
        offspring.config.dopamine_influence =
            (self.config.dopamine_influence + self.config.dopamine_influence) / 2.0;
        offspring.config.fear_suppression =
            (self.config.fear_suppression + self.config.fear_suppression) / 2.0;
        offspring.config.cyst_formation_rate =
            (self.config.cyst_formation_rate + self.config.cyst_formation_rate) / 2.0;

        Ok(Box::new(offspring))
    }

    fn get_genetics(&self) -> OrganismGenetics {
        self.base.genetics.clone()
    }

    fn set_genetics(&mut self, genetics: OrganismGenetics) {
        self.base.genetics = genetics;
    }

    fn should_terminate(&self) -> bool {
        // Terminate if manipulation consistently fails and no cysts remain
        let poor_manipulation =
            self.manipulation_success_rate < 0.15 && self.dormant_cysts.is_empty();
        poor_manipulation || self.base.should_terminate_base()
    }

    fn resource_consumption(&self) -> ResourceMetrics {
        let cyst_cost = self.dormant_cysts.len() as f64 * 1.5;
        let manipulation_cost = self.active_manipulations.len() as f64 * 6.0;
        let neural_cost = self.config.neurotransmitter_control * 30.0;

        ResourceMetrics {
            cpu_usage: 30.0 + cyst_cost + manipulation_cost + neural_cost,
            memory_mb: 56.0 + self.behavioral_memory.len() as f64 * 0.1,
            network_bandwidth_kbps: 320.0 + manipulation_cost * 8.0,
            api_calls_per_second: 18.0 + self.active_manipulations.len() as f64 * 2.5,
            latency_overhead_ns: 120_000 + (neural_cost * 3000.0) as u64,
        }
    }

    fn get_strategy_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert(
            "risk_manipulation_strength".to_string(),
            self.config.risk_manipulation_strength,
        );
        params.insert(
            "dopamine_influence".to_string(),
            self.config.dopamine_influence,
        );
        params.insert("fear_suppression".to_string(), self.config.fear_suppression);
        params.insert("dormant_cysts".to_string(), self.dormant_cysts.len() as f64);
        params.insert(
            "active_manipulations".to_string(),
            self.active_manipulations.len() as f64,
        );
        params.insert(
            "manipulation_success_rate".to_string(),
            self.manipulation_success_rate,
        );
        params.insert("replication_rate".to_string(), self.replication_rate);
        params.insert(
            "dopamine_level".to_string(),
            self.neurotransmitter_state.dopamine_level,
        );
        params.insert(
            "gaba_level".to_string(),
            self.neurotransmitter_state.gaba_level,
        );
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_toxoplasma_creation() {
        let toxo = ToxoplasmaOrganism::new();
        assert_eq!(toxo.organism_type(), "toxoplasma");
        assert!(toxo.dormant_cysts.is_empty());
        assert!(toxo.active_manipulations.is_empty());
    }

    #[tokio::test]
    async fn test_cyst_formation() {
        let mut toxo = ToxoplasmaOrganism::new();
        let regions = vec![BrainRegion::Amygdala, BrainRegion::StriatumVTA];

        let formed = toxo.form_cysts(regions).await.unwrap();
        // May form 0-2 cysts based on probability
        assert!(formed.len() <= 2);
    }

    #[tokio::test]
    async fn test_risk_profile_creation() {
        let mut toxo = ToxoplasmaOrganism::new();
        let market_conditions = MarketConditions {
            volatility: 0.3,
            volume: 0.8,
            spread: 0.02,
            trend_strength: 0.6,
            noise_level: 0.4,
        };

        let profile = toxo
            .create_risk_profile("BTCUSD", &market_conditions)
            .await
            .unwrap();
        assert!(profile.manipulated_risk_tolerance > profile.baseline_risk_tolerance);
        assert!(!toxo.active_manipulations.is_empty());
    }

    #[test]
    fn test_baseline_risk_calculation() {
        let toxo = ToxoplasmaOrganism::new();
        let market_conditions = MarketConditions {
            volatility: 0.2,     // Low volatility
            volume: 0.9,         // High volume
            spread: 0.01,        // Tight spread
            trend_strength: 0.8, // Strong trend
            noise_level: 0.3,
        };

        let baseline_risk = toxo.calculate_baseline_risk_tolerance(&market_conditions);
        assert!(baseline_risk > 0.5); // Should be higher due to favorable conditions
    }
}
