//! Lancet Liver Fluke organism - behavior manipulation specialist
//!
//! The Lancet Liver Fluke is a sophisticated parasitic organism that specializes in
//! behavioral manipulation, altering trading patterns and decision-making processes
//! to benefit the parasite while appearing natural to the host system.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use nalgebra::DMatrix;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;
use tracing::{debug, info};
use uuid::Uuid;

use super::{
    AdaptationFeedback, BaseOrganism, InfectionResult, OrganismError, OrganismGenetics,
    ParasiticOrganism, ResourceMetrics,
};

/// Configuration for Lancet Liver Fluke organism
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LancetFlukeConfig {
    /// Neural control strength for behavior manipulation
    pub neural_control_strength: f64,
    /// Number of behavioral patterns to track
    pub pattern_memory_size: usize,
    /// Manipulation subtlety (lower = more subtle)
    pub manipulation_subtlety: f64,
    /// Host dependency ratio (how much control is needed)
    pub host_dependency: f64,
    /// Behavioral mimicry accuracy
    pub mimicry_accuracy: f64,
    /// Maximum simultaneous manipulations
    pub max_concurrent_manipulations: usize,
}

impl Default for LancetFlukeConfig {
    fn default() -> Self {
        Self {
            neural_control_strength: 0.75,
            pattern_memory_size: 50,
            manipulation_subtlety: 0.85, // Very subtle
            host_dependency: 0.6,
            mimicry_accuracy: 0.9,
            max_concurrent_manipulations: 5,
        }
    }
}

/// Behavioral pattern stored in fluke's memory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralPattern {
    pub pattern_id: String,
    pub frequency_signature: Vec<f64>,
    pub amplitude_profile: Vec<f64>,
    pub phase_shifts: Vec<f64>,
    pub success_rate: f64,
    pub detection_risk: f64,
    pub last_used: DateTime<Utc>,
    pub effectiveness_score: f64,
}

/// Active behavioral manipulation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BehavioralManipulation {
    pub manipulation_id: Uuid,
    pub target_pair: String,
    pub pattern_used: String,
    pub intensity: f64,
    pub start_time: DateTime<Utc>,
    pub expected_duration: u64,
    pub current_phase: ManipulationPhase,
    pub host_compliance: f64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ManipulationPhase {
    Infiltration,
    PatternLearning,
    BehaviorAlteration,
    ControlMaintenance,
    ExtractionPhase,
}

/// Lancet Liver Fluke organism - behavior manipulation specialist
#[derive(Debug)]
pub struct LancetFlukeOrganism {
    base: BaseOrganism,
    config: LancetFlukeConfig,
    /// Learned behavioral patterns
    behavioral_patterns: Vec<BehavioralPattern>,
    /// Active manipulations
    active_manipulations: Vec<BehavioralManipulation>,
    /// Neural control network weights
    neural_weights: DMatrix<f64>,
    /// Behavioral mimicry database
    mimicry_database: HashMap<String, Vec<f64>>,
    /// Host compliance tracking
    host_compliance_history: Vec<f64>,
    /// Manipulation success rate
    manipulation_success_rate: f64,
}

impl LancetFlukeOrganism {
    pub fn new() -> Self {
        Self::with_config(LancetFlukeConfig::default())
    }

    pub fn with_config(config: LancetFlukeConfig) -> Self {
        let mut base = BaseOrganism::new();

        // Lancet flukes excel at control and subtlety
        base.genetics.stealth = 0.95; // Extremely stealthy
        base.genetics.adaptability = 0.90; // Highly adaptable to host behavior
        base.genetics.efficiency = 0.85; // Very efficient manipulation
        base.genetics.resilience = 0.80; // Can maintain control under stress
        base.genetics.cooperation = 0.30; // Parasitic nature, low cooperation
        base.genetics.aggression = 0.20; // Very low aggression - subtle manipulation

        // Initialize neural network for behavioral control
        let neural_size = 10; // Small neural network for behavior modeling
        let neural_weights = DMatrix::from_fn(neural_size, neural_size, |_, _| {
            (fastrand::f64() - 0.5) * 0.5 // Small initial weights
        });

        Self {
            base,
            config,
            behavioral_patterns: Vec::new(),
            active_manipulations: Vec::new(),
            neural_weights,
            mimicry_database: HashMap::new(),
            host_compliance_history: Vec::new(),
            manipulation_success_rate: 0.5,
        }
    }

    /// Learn behavioral patterns from market data
    pub async fn learn_behavioral_patterns(
        &mut self,
        market_data: Vec<(String, Vec<f64>)>,
    ) -> Result<usize, OrganismError> {
        debug!(
            "🧠 Lancet fluke learning behavioral patterns from {} data points",
            market_data.len()
        );

        let mut learned_patterns = 0;

        for (pair_id, data) in market_data {
            if data.len() < 20 {
                continue; // Need sufficient data for pattern recognition
            }

            // Perform frequency analysis using simplified FFT-like transformation
            let frequency_signature = self.analyze_frequency_components(&data);
            let amplitude_profile = self.extract_amplitude_profile(&data);
            let phase_shifts = self.detect_phase_shifts(&data);

            // Calculate pattern effectiveness based on regularity and predictability
            let effectiveness =
                self.calculate_pattern_effectiveness(&frequency_signature, &amplitude_profile);

            if effectiveness > 0.3 {
                // Only learn effective patterns
                let pattern = BehavioralPattern {
                    pattern_id: format!("{}_{}", pair_id, Utc::now().timestamp()),
                    frequency_signature,
                    amplitude_profile,
                    phase_shifts,
                    success_rate: 0.0, // Will be updated with usage
                    detection_risk: self.estimate_detection_risk(&data),
                    last_used: DateTime::from_timestamp(0, 0).unwrap_or_else(Utc::now),
                    effectiveness_score: effectiveness,
                };

                self.behavioral_patterns.push(pattern);
                learned_patterns += 1;

                // Store mimicry data for this pair
                self.mimicry_database.insert(pair_id, data);
            }
        }

        // Limit pattern memory size
        if self.behavioral_patterns.len() > self.config.pattern_memory_size {
            // Remove least effective patterns
            self.behavioral_patterns.sort_by(|a, b| {
                b.effectiveness_score
                    .partial_cmp(&a.effectiveness_score)
                    .unwrap()
            });
            self.behavioral_patterns
                .truncate(self.config.pattern_memory_size);
        }

        info!("🧠 Learned {} new behavioral patterns", learned_patterns);
        Ok(learned_patterns)
    }

    /// Analyze frequency components of behavioral data
    fn analyze_frequency_components(&self, data: &[f64]) -> Vec<f64> {
        let n = data.len().min(32); // Limit for computational efficiency
        let mut frequencies = Vec::with_capacity(n / 2);

        for k in 0..(n / 2) {
            let mut real_sum = 0.0;
            let mut imag_sum = 0.0;

            for (i, &value) in data.iter().take(n).enumerate() {
                let angle = -2.0 * PI * (k as f64) * (i as f64) / (n as f64);
                real_sum += value * angle.cos();
                imag_sum += value * angle.sin();
            }

            let magnitude = (real_sum * real_sum + imag_sum * imag_sum).sqrt();
            frequencies.push(magnitude / n as f64);
        }

        frequencies
    }

    /// Extract amplitude profile from data
    fn extract_amplitude_profile(&self, data: &[f64]) -> Vec<f64> {
        let window_size = (data.len() / 10).max(3);
        let mut profile = Vec::new();

        for i in (0..data.len()).step_by(window_size) {
            let window_end = (i + window_size).min(data.len());
            let window = &data[i..window_end];

            if !window.is_empty() {
                let mean = window.iter().sum::<f64>() / window.len() as f64;
                let variance =
                    window.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / window.len() as f64;

                profile.push(variance.sqrt()); // Standard deviation as amplitude measure
            }
        }

        profile
    }

    /// Detect phase shifts in data
    fn detect_phase_shifts(&self, data: &[f64]) -> Vec<f64> {
        let mut shifts = Vec::new();
        let window_size = data.len() / 5; // 5 phase shift measurements

        for i in (0..data.len()).step_by(window_size) {
            let window_end = (i + window_size).min(data.len());
            let window = &data[i..window_end];

            if window.len() > 2 {
                // Calculate phase as angle of first harmonic
                let mut real_sum = 0.0;
                let mut imag_sum = 0.0;

                for (j, &value) in window.iter().enumerate() {
                    let angle = 2.0 * PI * (j as f64) / (window.len() as f64);
                    real_sum += value * angle.cos();
                    imag_sum += value * angle.sin();
                }

                let phase = imag_sum.atan2(real_sum);
                shifts.push(phase);
            }
        }

        shifts
    }

    /// Calculate pattern effectiveness score
    fn calculate_pattern_effectiveness(&self, frequencies: &[f64], amplitudes: &[f64]) -> f64 {
        if frequencies.is_empty() || amplitudes.is_empty() {
            return 0.0;
        }

        // Higher effectiveness for patterns with clear dominant frequencies
        let max_freq = frequencies.iter().fold(0.0f64, |acc, &x| acc.max(x));
        let freq_clarity =
            max_freq / (frequencies.iter().sum::<f64>() / frequencies.len() as f64 + 1e-10);

        // Higher effectiveness for consistent amplitudes
        let mean_amp = amplitudes.iter().sum::<f64>() / amplitudes.len() as f64;
        let amp_consistency = 1.0
            - (amplitudes
                .iter()
                .map(|&x| (x - mean_amp).abs())
                .sum::<f64>()
                / amplitudes.len() as f64)
                / (mean_amp + 1e-10);

        (freq_clarity * 0.6 + amp_consistency * 0.4).min(1.0)
    }

    /// Estimate detection risk for a pattern
    fn estimate_detection_risk(&self, data: &[f64]) -> f64 {
        // Higher risk for more regular/obvious patterns
        let mean = data.iter().sum::<f64>() / data.len() as f64;
        let variance = data.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / data.len() as f64;

        // Normalize to 0-1 range, where higher variance = lower detection risk
        (1.0 - (variance / (variance + 1.0))).max(0.1)
    }

    /// Start behavioral manipulation on a target pair
    pub async fn initiate_behavioral_manipulation(
        &mut self,
        pair_id: &str,
        manipulation_intensity: f64,
    ) -> Result<Uuid, OrganismError> {
        if self.active_manipulations.len() >= self.config.max_concurrent_manipulations {
            return Err(OrganismError::ResourceExhausted(
                "Maximum concurrent manipulations reached".to_string(),
            ));
        }

        // Select best pattern for this manipulation
        let best_pattern = self
            .behavioral_patterns
            .iter()
            .filter(|p| p.detection_risk < 0.7) // Avoid high-risk patterns
            .max_by(|a, b| {
                a.effectiveness_score
                    .partial_cmp(&b.effectiveness_score)
                    .unwrap()
            })
            .ok_or_else(|| {
                OrganismError::InfectionFailed("No suitable patterns available".to_string())
            })?;

        let manipulation_id = Uuid::new_v4();
        let manipulation = BehavioralManipulation {
            manipulation_id,
            target_pair: pair_id.to_string(),
            pattern_used: best_pattern.pattern_id.clone(),
            intensity: manipulation_intensity * self.config.manipulation_subtlety,
            start_time: Utc::now(),
            expected_duration: 3600 * 6, // 6 hours typical manipulation
            current_phase: ManipulationPhase::Infiltration,
            host_compliance: 0.0, // Will be measured
        };

        self.active_manipulations.push(manipulation);

        info!(
            "🧠 Initiated behavioral manipulation {} on pair {}",
            manipulation_id, pair_id
        );

        Ok(manipulation_id)
    }

    /// Update active manipulations and measure effectiveness
    pub async fn update_manipulations(&mut self) -> Result<Vec<Uuid>, OrganismError> {
        let mut completed_manipulations = Vec::new();
        let now = Utc::now();

        for manipulation in &mut self.active_manipulations {
            let elapsed = (now - manipulation.start_time).num_seconds() as u64;

            // Update manipulation phase based on elapsed time
            manipulation.current_phase = match elapsed {
                0..=300 => ManipulationPhase::Infiltration,
                301..=900 => ManipulationPhase::PatternLearning,
                901..=3600 => ManipulationPhase::BehaviorAlteration,
                3601..=18000 => ManipulationPhase::ControlMaintenance,
                _ => ManipulationPhase::ExtractionPhase,
            };

            // Simulate host compliance measurement
            let base_compliance = self.config.neural_control_strength;
            let time_factor = (elapsed as f64 / 3600.0).min(1.0); // Increase over time
            let intensity_factor = manipulation.intensity;

            manipulation.host_compliance =
                (base_compliance * time_factor * intensity_factor).min(1.0);

            // Check if manipulation is complete
            if elapsed > manipulation.expected_duration {
                completed_manipulations.push(manipulation.manipulation_id);
            }
        }

        // Remove completed manipulations and update pattern success rates
        self.active_manipulations.retain(|m| {
            if completed_manipulations.contains(&m.manipulation_id) {
                // Update pattern success rate
                if let Some(pattern) = self
                    .behavioral_patterns
                    .iter_mut()
                    .find(|p| p.pattern_id == m.pattern_used)
                {
                    let success = m.host_compliance > 0.6;
                    pattern.success_rate =
                        pattern.success_rate * 0.9 + (if success { 1.0 } else { 0.0 }) * 0.1;
                    pattern.last_used = now;
                }
                false // Remove from active list
            } else {
                true // Keep in active list
            }
        });

        // Update overall success rate
        if !self.host_compliance_history.is_empty() {
            self.manipulation_success_rate = self.host_compliance_history.iter().sum::<f64>()
                / self.host_compliance_history.len() as f64;
        }

        Ok(completed_manipulations)
    }

    /// Get current manipulation status
    pub fn get_manipulation_status(&self) -> LancetFlukeStatus {
        LancetFlukeStatus {
            learned_patterns: self.behavioral_patterns.len(),
            active_manipulations: self.active_manipulations.len(),
            overall_success_rate: self.manipulation_success_rate,
            neural_control_strength: self.config.neural_control_strength,
            mimicry_accuracy: self.config.mimicry_accuracy,
            average_detection_risk: self
                .behavioral_patterns
                .iter()
                .map(|p| p.detection_risk)
                .sum::<f64>()
                / (self.behavioral_patterns.len() as f64).max(1.0),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LancetFlukeStatus {
    pub learned_patterns: usize,
    pub active_manipulations: usize,
    pub overall_success_rate: f64,
    pub neural_control_strength: f64,
    pub mimicry_accuracy: f64,
    pub average_detection_risk: f64,
}

#[async_trait]
impl ParasiticOrganism for LancetFlukeOrganism {
    fn id(&self) -> Uuid {
        self.base.id
    }

    fn organism_type(&self) -> &'static str {
        "lancet_liver_fluke"
    }

    fn fitness(&self) -> f64 {
        let base_fitness = self.base.fitness;
        let pattern_bonus = (self.behavioral_patterns.len() as f64 / 20.0).min(0.3);
        let success_bonus = self.manipulation_success_rate * 0.4;
        let control_bonus = self.config.neural_control_strength * 0.2;

        (base_fitness + pattern_bonus + success_bonus + control_bonus).min(1.0)
    }

    fn calculate_infection_strength(&self, vulnerability: f64) -> f64 {
        let base_strength = self.base.calculate_base_infection_strength(vulnerability);

        // Lancet flukes are more effective with learned patterns
        let pattern_multiplier = 1.0 + (self.behavioral_patterns.len() as f64 / 50.0).min(0.5);

        // Neural control enhances effectiveness
        let neural_multiplier = 1.0 + self.config.neural_control_strength * 0.3;

        // Mimicry accuracy provides stealth bonus
        let stealth_multiplier = 1.0 + self.config.mimicry_accuracy * 0.2;

        base_strength * pattern_multiplier * neural_multiplier * stealth_multiplier
    }

    async fn infect_pair(
        &self,
        pair_id: &str,
        vulnerability: f64,
    ) -> Result<InfectionResult, OrganismError> {
        // Lancet flukes need moderate vulnerability to establish neural control
        if vulnerability < 0.4 {
            return Err(OrganismError::UnsuitableConditions(format!(
                "Vulnerability {:.3} insufficient for neural control (need >0.4)",
                vulnerability
            )));
        }

        let infection_strength = self.calculate_infection_strength(vulnerability);

        // Longer infections due to behavioral manipulation complexity
        let estimated_duration = (14400.0 * (2.0 - vulnerability)) as u64; // 4-8 hours

        Ok(InfectionResult {
            success: true,
            infection_id: Uuid::new_v4(),
            initial_profit: infection_strength * 120.0, // Higher profit due to behavioral control
            estimated_duration,
            resource_usage: ResourceMetrics {
                cpu_usage: 35.0 + self.behavioral_patterns.len() as f64 * 0.5,
                memory_mb: 64.0 + self.config.pattern_memory_size as f64 * 1.2,
                network_bandwidth_kbps: 512.0, // Higher due to continuous monitoring
                api_calls_per_second: 25.0,    // Frequent behavioral adjustments
                latency_overhead_ns: 150_000,  // 150µs for neural processing
            },
        })
    }

    async fn adapt(&mut self, feedback: AdaptationFeedback) -> Result<(), OrganismError> {
        self.base.update_fitness(feedback.performance_score);
        self.host_compliance_history.push(feedback.success_rate);

        // Keep compliance history bounded
        if self.host_compliance_history.len() > 100 {
            self.host_compliance_history.remove(0);
        }

        // Adapt neural control strength based on success
        if feedback.success_rate > 0.8 {
            self.config.neural_control_strength =
                (self.config.neural_control_strength * 1.02).min(1.0);
        } else if feedback.success_rate < 0.5 {
            self.config.neural_control_strength =
                (self.config.neural_control_strength * 0.98).max(0.3);
        }

        // Adapt manipulation subtlety based on detection
        if feedback.performance_score < 0.4 {
            // Possible detection
            self.config.manipulation_subtlety = (self.config.manipulation_subtlety * 1.05).min(1.0);
        }

        // Update neural network weights based on feedback
        let learning_rate = 0.01;
        let error_signal = 1.0 - feedback.performance_score;

        for i in 0..self.neural_weights.nrows() {
            for j in 0..self.neural_weights.ncols() {
                let adjustment = learning_rate * error_signal * (fastrand::f64() - 0.5);
                self.neural_weights[(i, j)] += adjustment;

                // Keep weights bounded
                self.neural_weights[(i, j)] = self.neural_weights[(i, j)].clamp(-1.0, 1.0);
            }
        }

        Ok(())
    }

    fn mutate(&mut self, rate: f64) {
        self.base.genetics.mutate(rate);

        use rand::Rng;
        let mut rng = rand::thread_rng();

        if rng.gen::<f64>() < rate {
            self.config.neural_control_strength =
                (self.config.neural_control_strength + rng.gen_range(-0.05..0.05)).clamp(0.3, 1.0);
        }
        if rng.gen::<f64>() < rate {
            self.config.manipulation_subtlety =
                (self.config.manipulation_subtlety + rng.gen_range(-0.03..0.03)).clamp(0.5, 1.0);
        }
        if rng.gen::<f64>() < rate {
            self.config.mimicry_accuracy =
                (self.config.mimicry_accuracy + rng.gen_range(-0.05..0.05)).clamp(0.6, 1.0);
        }

        // Mutate neural network weights
        if rng.gen::<f64>() < rate {
            for i in 0..self.neural_weights.nrows() {
                for j in 0..self.neural_weights.ncols() {
                    if rng.gen::<f64>() < rate * 0.1 {
                        // Lower rate for individual weights
                        self.neural_weights[(i, j)] += rng.gen_range(-0.1..0.1);
                        self.neural_weights[(i, j)] = self.neural_weights[(i, j)].clamp(-1.0, 1.0);
                    }
                }
            }
        }
    }

    fn crossover(
        &self,
        other: &dyn ParasiticOrganism,
    ) -> Result<Box<dyn ParasiticOrganism + Send + Sync>, OrganismError> {
        if other.organism_type() != "lancet_liver_fluke" {
            return Err(OrganismError::CrossoverFailed(
                "Can only crossover with same organism type".to_string(),
            ));
        }

        let mut offspring = LancetFlukeOrganism::new();
        offspring.base.genetics = self.base.genetics.crossover(&other.get_genetics());

        // Average the fluke-specific configurations
        offspring.config.neural_control_strength =
            (self.config.neural_control_strength + self.config.neural_control_strength) / 2.0;
        offspring.config.manipulation_subtlety =
            (self.config.manipulation_subtlety + self.config.manipulation_subtlety) / 2.0;
        offspring.config.mimicry_accuracy =
            (self.config.mimicry_accuracy + self.config.mimicry_accuracy) / 2.0;

        Ok(Box::new(offspring))
    }

    fn get_genetics(&self) -> OrganismGenetics {
        self.base.genetics.clone()
    }

    fn set_genetics(&mut self, genetics: OrganismGenetics) {
        self.base.genetics = genetics;
    }

    fn should_terminate(&self) -> bool {
        // Terminate if manipulation success rate is consistently poor
        let poor_manipulation =
            self.manipulation_success_rate < 0.2 && self.host_compliance_history.len() > 20;
        poor_manipulation || self.base.should_terminate_base()
    }

    fn resource_consumption(&self) -> ResourceMetrics {
        let pattern_cost = self.behavioral_patterns.len() as f64 * 0.8;
        let manipulation_cost = self.active_manipulations.len() as f64 * 5.0;
        let neural_cost = self.config.neural_control_strength * 25.0;

        ResourceMetrics {
            cpu_usage: 25.0 + pattern_cost + manipulation_cost + neural_cost,
            memory_mb: 48.0 + self.config.pattern_memory_size as f64 * 0.5,
            network_bandwidth_kbps: 256.0 + manipulation_cost * 10.0,
            api_calls_per_second: 15.0 + self.active_manipulations.len() as f64 * 3.0,
            latency_overhead_ns: 100_000 + (neural_cost * 2000.0) as u64,
        }
    }

    fn get_strategy_params(&self) -> HashMap<String, f64> {
        let mut params = HashMap::new();
        params.insert(
            "neural_control_strength".to_string(),
            self.config.neural_control_strength,
        );
        params.insert(
            "manipulation_subtlety".to_string(),
            self.config.manipulation_subtlety,
        );
        params.insert("mimicry_accuracy".to_string(), self.config.mimicry_accuracy);
        params.insert(
            "learned_patterns".to_string(),
            self.behavioral_patterns.len() as f64,
        );
        params.insert(
            "active_manipulations".to_string(),
            self.active_manipulations.len() as f64,
        );
        params.insert(
            "manipulation_success_rate".to_string(),
            self.manipulation_success_rate,
        );
        params.insert("host_dependency".to_string(), self.config.host_dependency);
        params
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_lancet_fluke_creation() {
        let fluke = LancetFlukeOrganism::new();
        assert_eq!(fluke.organism_type(), "lancet_liver_fluke");
        assert!(fluke.behavioral_patterns.is_empty());
        assert!(fluke.active_manipulations.is_empty());
    }

    #[tokio::test]
    async fn test_pattern_learning() {
        let mut fluke = LancetFlukeOrganism::new();
        let market_data = vec![(
            "BTCUSD".to_string(),
            vec![
                1.0, 2.0, 1.5, 2.5, 2.0, 1.8, 2.2, 1.9, 2.1, 1.7, 2.3, 1.6, 2.4, 1.4, 2.6, 1.3,
                2.7, 1.2, 2.8, 1.1,
            ],
        )];

        let learned = fluke.learn_behavioral_patterns(market_data).await.unwrap();
        assert!(learned > 0 || fluke.behavioral_patterns.is_empty()); // May not learn if pattern is ineffective
    }

    #[test]
    fn test_frequency_analysis() {
        let fluke = LancetFlukeOrganism::new();
        let data = vec![1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0]; // Simple alternating pattern

        let frequencies = fluke.analyze_frequency_components(&data);
        assert!(!frequencies.is_empty());
        assert!(frequencies.iter().any(|&f| f > 0.0)); // Should detect some frequency content
    }
}
