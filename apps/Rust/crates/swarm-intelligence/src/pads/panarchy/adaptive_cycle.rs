//! # Adaptive Cycle Implementation
//!
//! Core implementation of adaptive cycles based on panarchy theory.
//! Each cycle represents the four phases: Growth, Conservation, Release, Reorganization.

use std::collections::HashMap;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use tracing::{debug, trace, instrument};

use crate::core::{
    PadsResult, PadsError, AdaptiveCyclePhase, PhaseCharacteristics
};

/// Adaptive cycle configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveCycleConfig {
    /// Scale level (0 = fastest, higher = slower)
    pub scale_level: usize,
    
    /// Total cycle duration
    pub cycle_duration: Duration,
    
    /// Phase transition thresholds
    pub transition_thresholds: HashMap<AdaptiveCyclePhase, f64>,
}

/// Adaptive cycle state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveCycleState {
    /// Current phase
    pub current_phase: AdaptiveCyclePhase,
    
    /// Time in current phase
    pub phase_duration: Duration,
    
    /// Phase progress (0.0 to 1.0)
    pub phase_progress: f64,
    
    /// Phase characteristics
    pub characteristics: PhaseCharacteristics,
    
    /// Performance metrics
    pub performance: CyclePerformance,
    
    /// Disturbance level
    pub disturbance: f64,
    
    /// Adaptation pressure
    pub adaptation_pressure: f64,
    
    /// Last update time
    pub last_updated: Instant,
}

/// Cycle performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CyclePerformance {
    /// Resource utilization efficiency
    pub efficiency: f64,
    
    /// Innovation rate
    pub innovation_rate: f64,
    
    /// Stability measure
    pub stability: f64,
    
    /// Adaptability measure
    pub adaptability: f64,
    
    /// Learning rate
    pub learning_rate: f64,
    
    /// Overall performance score
    pub overall_score: f64,
}

impl Default for CyclePerformance {
    fn default() -> Self {
        Self {
            efficiency: 0.5,
            innovation_rate: 0.5,
            stability: 0.5,
            adaptability: 0.5,
            learning_rate: 0.5,
            overall_score: 0.5,
        }
    }
}

/// Main adaptive cycle implementation
#[derive(Debug, Clone)]
pub struct AdaptiveCycle {
    /// Cycle configuration
    config: AdaptiveCycleConfig,
    
    /// Current cycle state
    state: AdaptiveCycleState,
    
    /// Phase history
    phase_history: Vec<PhaseRecord>,
    
    /// Transition rules
    transition_rules: TransitionRules,
    
    /// Performance history
    performance_history: Vec<CyclePerformance>,
}

/// Historical phase record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseRecord {
    /// Phase type
    pub phase: AdaptiveCyclePhase,
    
    /// Phase start time
    pub start_time: Instant,
    
    /// Phase duration
    pub duration: Duration,
    
    /// Performance during this phase
    pub performance: CyclePerformance,
    
    /// Transition trigger
    pub transition_trigger: String,
    
    /// Disturbances encountered
    pub disturbances: Vec<Disturbance>,
}

/// Disturbance event
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Disturbance {
    /// Disturbance type
    pub disturbance_type: DisturbanceType,
    
    /// Magnitude (0.0 to 1.0)
    pub magnitude: f64,
    
    /// Duration
    pub duration: Duration,
    
    /// Timestamp
    pub timestamp: Instant,
    
    /// Source
    pub source: String,
    
    /// Impact on cycle
    pub impact: f64,
}

/// Types of disturbances
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DisturbanceType {
    /// External shock
    External,
    
    /// Internal instability
    Internal,
    
    /// Cross-scale cascade
    CrossScale,
    
    /// Resource depletion
    ResourceDepletion,
    
    /// Innovation pressure
    Innovation,
    
    /// Competitive pressure
    Competition,
}

/// Transition rules for phase changes
#[derive(Debug, Clone)]
pub struct TransitionRules {
    /// Time-based transition rules
    pub time_based: HashMap<AdaptiveCyclePhase, Duration>,
    
    /// Threshold-based transition rules
    pub threshold_based: HashMap<AdaptiveCyclePhase, Vec<ThresholdRule>>,
    
    /// Disturbance-triggered transitions
    pub disturbance_triggered: HashMap<AdaptiveCyclePhase, Vec<DisturbanceRule>>,
}

/// Threshold rule for transitions
#[derive(Debug, Clone)]
pub struct ThresholdRule {
    /// Metric to monitor
    pub metric: String,
    
    /// Threshold value
    pub threshold: f64,
    
    /// Comparison operator
    pub operator: ComparisonOperator,
    
    /// Target phase
    pub target_phase: AdaptiveCyclePhase,
    
    /// Rule weight
    pub weight: f64,
}

/// Disturbance rule for transitions
#[derive(Debug, Clone)]
pub struct DisturbanceRule {
    /// Disturbance type
    pub disturbance_type: DisturbanceType,
    
    /// Minimum magnitude
    pub min_magnitude: f64,
    
    /// Target phase
    pub target_phase: AdaptiveCyclePhase,
    
    /// Probability of transition
    pub probability: f64,
}

/// Comparison operators
#[derive(Debug, Clone)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Equal,
}

impl AdaptiveCycle {
    /// Create a new adaptive cycle
    #[instrument(skip(config))]
    pub async fn new(config: AdaptiveCycleConfig) -> PadsResult<Self> {
        debug!("Creating adaptive cycle for scale {}", config.scale_level);
        
        let state = AdaptiveCycleState {
            current_phase: AdaptiveCyclePhase::Growth,
            phase_duration: Duration::from_secs(0),
            phase_progress: 0.0,
            characteristics: AdaptiveCyclePhase::Growth.characteristics(),
            performance: CyclePerformance::default(),
            disturbance: 0.0,
            adaptation_pressure: 0.0,
            last_updated: Instant::now(),
        };
        
        let transition_rules = Self::create_default_transition_rules(&config);
        
        Ok(Self {
            config,
            state,
            phase_history: Vec::new(),
            transition_rules,
            performance_history: Vec::new(),
        })
    }
    
    /// Start the adaptive cycle
    #[instrument(skip(self))]
    pub async fn start(&mut self) -> PadsResult<()> {
        debug!("Starting adaptive cycle for scale {}", self.config.scale_level);
        self.state.last_updated = Instant::now();
        Ok(())
    }
    
    /// Update the adaptive cycle
    #[instrument(skip(self))]
    pub async fn update(&mut self) -> PadsResult<()> {
        trace!("Updating adaptive cycle for scale {}", self.config.scale_level);
        
        let now = Instant::now();
        let elapsed = now.duration_since(self.state.last_updated);
        
        // Update phase duration
        self.state.phase_duration += elapsed;
        
        // Calculate phase progress
        let expected_phase_duration = self.calculate_expected_phase_duration();
        self.state.phase_progress = (self.state.phase_duration.as_secs_f64() 
            / expected_phase_duration.as_secs_f64()).min(1.0);
        
        // Update characteristics based on phase and progress
        self.update_phase_characteristics().await?;
        
        // Update performance metrics
        self.update_performance_metrics().await?;
        
        // Check for disturbances
        self.check_disturbances().await?;
        
        // Calculate adaptation pressure
        self.calculate_adaptation_pressure().await?;
        
        self.state.last_updated = now;
        
        Ok(())
    }
    
    /// Check if the cycle should transition to the next phase
    #[instrument(skip(self))]
    pub async fn should_transition(&self) -> PadsResult<bool> {
        // Check time-based transitions
        if self.check_time_based_transition().await {
            return Ok(true);
        }
        
        // Check threshold-based transitions
        if self.check_threshold_based_transitions().await? {
            return Ok(true);
        }
        
        // Check disturbance-triggered transitions
        if self.check_disturbance_triggered_transitions().await {
            return Ok(true);
        }
        
        Ok(false)
    }
    
    /// Transition to the next phase
    #[instrument(skip(self))]
    pub async fn transition_phase(&mut self) -> PadsResult<AdaptiveCyclePhase> {
        let old_phase = self.state.current_phase;
        let new_phase = old_phase.next_phase();
        
        debug!("Transitioning from {:?} to {:?} at scale {}", 
               old_phase, new_phase, self.config.scale_level);
        
        // Record the completed phase
        self.record_phase_completion(old_phase).await?;
        
        // Transition to new phase
        self.state.current_phase = new_phase;
        self.state.phase_duration = Duration::from_secs(0);
        self.state.phase_progress = 0.0;
        self.state.characteristics = new_phase.characteristics();
        
        // Reset performance metrics for new phase
        self.reset_performance_for_new_phase(new_phase).await?;
        
        Ok(new_phase)
    }
    
    /// Force transition to a specific phase
    #[instrument(skip(self))]
    pub async fn force_transition(&mut self, target_phase: AdaptiveCyclePhase) -> PadsResult<()> {
        debug!("Force transitioning to {:?} at scale {}", 
               target_phase, self.config.scale_level);
        
        let old_phase = self.state.current_phase;
        
        // Record the interrupted phase
        self.record_phase_completion(old_phase).await?;
        
        // Force transition
        self.state.current_phase = target_phase;
        self.state.phase_duration = Duration::from_secs(0);
        self.state.phase_progress = 0.0;
        self.state.characteristics = target_phase.characteristics();
        
        // Reset performance metrics
        self.reset_performance_for_new_phase(target_phase).await?;
        
        Ok(())
    }
    
    /// Apply cross-scale influence
    #[instrument(skip(self))]
    pub async fn apply_cross_scale_influence(
        &mut self, 
        influence_strength: f64,
        source_phase: AdaptiveCyclePhase,
    ) -> PadsResult<()> {
        debug!("Applying cross-scale influence (strength: {}, source: {:?})", 
               influence_strength, source_phase);
        
        // Increase adaptation pressure based on influence
        self.state.adaptation_pressure += influence_strength * 0.3;
        
        // Potentially trigger phase transition if influence is strong
        if influence_strength > 0.8 {
            // Strong influence might trigger immediate transition
            if source_phase == AdaptiveCyclePhase::Release {
                // Release phase at other scales can trigger release here
                if self.state.current_phase != AdaptiveCyclePhase::Release {
                    self.force_transition(AdaptiveCyclePhase::Release).await?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Get current phase
    pub async fn get_current_phase(&self) -> AdaptiveCyclePhase {
        self.state.current_phase
    }
    
    /// Get current state
    pub async fn get_state(&self) -> AdaptiveCycleState {
        self.state.clone()
    }
    
    /// Calculate resilience of this cycle
    pub async fn calculate_resilience(&self) -> f64 {
        // Resilience is combination of multiple factors
        let stability_factor = self.state.performance.stability;
        let adaptability_factor = self.state.performance.adaptability;
        let disturbance_resistance = 1.0 - self.state.disturbance.min(1.0);
        let phase_resilience = self.state.characteristics.resilience;
        
        // Weighted combination
        (stability_factor * 0.3 + 
         adaptability_factor * 0.3 + 
         disturbance_resistance * 0.2 + 
         phase_resilience * 0.2).clamp(0.0, 1.0)
    }
    
    /// Create default transition rules
    fn create_default_transition_rules(config: &AdaptiveCycleConfig) -> TransitionRules {
        let mut time_based = HashMap::new();
        let cycle_quarter = config.cycle_duration / 4;
        
        // Default time allocations (can be adjusted)
        time_based.insert(AdaptiveCyclePhase::Growth, cycle_quarter * 2);        // 50% of cycle
        time_based.insert(AdaptiveCyclePhase::Conservation, cycle_quarter);      // 25% of cycle
        time_based.insert(AdaptiveCyclePhase::Release, cycle_quarter / 2);       // 12.5% of cycle
        time_based.insert(AdaptiveCyclePhase::Reorganization, cycle_quarter / 2); // 12.5% of cycle
        
        // TODO: Add threshold-based and disturbance-triggered rules
        let threshold_based = HashMap::new();
        let disturbance_triggered = HashMap::new();
        
        TransitionRules {
            time_based,
            threshold_based,
            disturbance_triggered,
        }
    }
    
    /// Calculate expected phase duration
    fn calculate_expected_phase_duration(&self) -> Duration {
        self.transition_rules
            .time_based
            .get(&self.state.current_phase)
            .copied()
            .unwrap_or(self.config.cycle_duration / 4)
    }
    
    /// Update phase characteristics based on progress
    async fn update_phase_characteristics(&mut self) -> PadsResult<()> {
        let base_characteristics = self.state.current_phase.characteristics();
        let progress = self.state.phase_progress;
        
        // Modify characteristics based on phase progress
        self.state.characteristics = PhaseCharacteristics {
            potential: base_characteristics.potential * (1.0 + progress * 0.2),
            connectedness: base_characteristics.connectedness * (1.0 + progress * 0.1),
            resilience: base_characteristics.resilience * (1.0 - progress * 0.1),
            innovation: base_characteristics.innovation * (1.0 + progress * 0.3),
            efficiency: base_characteristics.efficiency * (1.0 + progress * 0.2),
        };
        
        Ok(())
    }
    
    /// Update performance metrics
    async fn update_performance_metrics(&mut self) -> PadsResult<()> {
        // Performance is influenced by phase characteristics and external factors
        let characteristics = &self.state.characteristics;
        let disturbance_impact = 1.0 - self.state.disturbance * 0.5;
        
        self.state.performance = CyclePerformance {
            efficiency: (characteristics.efficiency * disturbance_impact).clamp(0.0, 1.0),
            innovation_rate: (characteristics.innovation * disturbance_impact).clamp(0.0, 1.0),
            stability: (characteristics.resilience * disturbance_impact).clamp(0.0, 1.0),
            adaptability: (characteristics.potential * disturbance_impact).clamp(0.0, 1.0),
            learning_rate: (characteristics.innovation * 0.8 * disturbance_impact).clamp(0.0, 1.0),
            overall_score: 0.0, // Will be calculated below
        };
        
        // Calculate overall score
        self.state.performance.overall_score = (
            self.state.performance.efficiency * 0.2 +
            self.state.performance.innovation_rate * 0.2 +
            self.state.performance.stability * 0.2 +
            self.state.performance.adaptability * 0.2 +
            self.state.performance.learning_rate * 0.2
        ).clamp(0.0, 1.0);
        
        Ok(())
    }
    
    /// Check for environmental disturbances
    async fn check_disturbances(&mut self) -> PadsResult<()> {
        // Simulate disturbance detection (in real system, this would come from monitoring)
        let base_disturbance = match self.state.current_phase {
            AdaptiveCyclePhase::Growth => 0.1,        // Low disturbance
            AdaptiveCyclePhase::Conservation => 0.2,  // Medium disturbance
            AdaptiveCyclePhase::Release => 0.8,       // High disturbance
            AdaptiveCyclePhase::Reorganization => 0.3, // Medium disturbance
        };
        
        // Add some random variation
        let random_factor = rand::random::<f64>() * 0.2 - 0.1; // ±0.1
        self.state.disturbance = (base_disturbance + random_factor).clamp(0.0, 1.0);
        
        Ok(())
    }
    
    /// Calculate adaptation pressure
    async fn calculate_adaptation_pressure(&mut self) -> PadsResult<()> {
        // Adaptation pressure comes from multiple sources
        let disturbance_pressure = self.state.disturbance * 0.5;
        let phase_pressure = match self.state.current_phase {
            AdaptiveCyclePhase::Growth => 0.1,
            AdaptiveCyclePhase::Conservation => 0.8,
            AdaptiveCyclePhase::Release => 0.2,
            AdaptiveCyclePhase::Reorganization => 0.6,
        };
        let progress_pressure = self.state.phase_progress * 0.3;
        
        self.state.adaptation_pressure = (
            disturbance_pressure + phase_pressure + progress_pressure
        ).clamp(0.0, 1.0);
        
        Ok(())
    }
    
    /// Check time-based transition conditions
    async fn check_time_based_transition(&self) -> bool {
        let expected_duration = self.calculate_expected_phase_duration();
        self.state.phase_duration >= expected_duration
    }
    
    /// Check threshold-based transition conditions
    async fn check_threshold_based_transitions(&self) -> PadsResult<bool> {
        // TODO: Implement threshold checking logic
        // For now, return false (no threshold-triggered transitions)
        Ok(false)
    }
    
    /// Check disturbance-triggered transition conditions
    async fn check_disturbance_triggered_transitions(&self) -> bool {
        // High disturbance can trigger release phase
        if self.state.disturbance > 0.7 && 
           self.state.current_phase != AdaptiveCyclePhase::Release {
            return true;
        }
        
        false
    }
    
    /// Record completion of a phase
    async fn record_phase_completion(&mut self, completed_phase: AdaptiveCyclePhase) -> PadsResult<()> {
        let record = PhaseRecord {
            phase: completed_phase,
            start_time: self.state.last_updated - self.state.phase_duration,
            duration: self.state.phase_duration,
            performance: self.state.performance.clone(),
            transition_trigger: "natural".to_string(), // TODO: Track actual trigger
            disturbances: Vec::new(), // TODO: Track disturbances
        };
        
        self.phase_history.push(record);
        
        // Limit history size
        if self.phase_history.len() > 100 {
            self.phase_history.remove(0);
        }
        
        Ok(())
    }
    
    /// Reset performance metrics for new phase
    async fn reset_performance_for_new_phase(&mut self, new_phase: AdaptiveCyclePhase) -> PadsResult<()> {
        // Performance starts at base level for new phase
        let characteristics = new_phase.characteristics();
        
        self.state.performance = CyclePerformance {
            efficiency: characteristics.efficiency,
            innovation_rate: characteristics.innovation,
            stability: characteristics.resilience,
            adaptability: characteristics.potential,
            learning_rate: characteristics.innovation * 0.8,
            overall_score: (characteristics.efficiency + characteristics.innovation + 
                          characteristics.resilience + characteristics.potential) / 4.0,
        };
        
        // Reset disturbance and adaptation pressure
        self.state.disturbance = 0.1;
        self.state.adaptation_pressure = 0.0;
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_adaptive_cycle_creation() {
        let config = AdaptiveCycleConfig {
            scale_level: 0,
            cycle_duration: Duration::from_secs(3600),
            transition_thresholds: HashMap::new(),
        };
        
        let cycle = AdaptiveCycle::new(config).await;
        assert!(cycle.is_ok());
        
        let cycle = cycle.unwrap();
        assert_eq!(cycle.state.current_phase, AdaptiveCyclePhase::Growth);
    }
    
    #[tokio::test]
    async fn test_phase_transition() {
        let config = AdaptiveCycleConfig {
            scale_level: 0,
            cycle_duration: Duration::from_secs(3600),
            transition_thresholds: HashMap::new(),
        };
        
        let mut cycle = AdaptiveCycle::new(config).await.unwrap();
        let initial_phase = cycle.get_current_phase().await;
        
        let new_phase = cycle.transition_phase().await.unwrap();
        assert_eq!(new_phase, initial_phase.next_phase());
    }
    
    #[tokio::test]
    async fn test_resilience_calculation() {
        let config = AdaptiveCycleConfig {
            scale_level: 0,
            cycle_duration: Duration::from_secs(3600),
            transition_thresholds: HashMap::new(),
        };
        
        let cycle = AdaptiveCycle::new(config).await.unwrap();
        let resilience = cycle.calculate_resilience().await;
        
        assert!(resilience >= 0.0 && resilience <= 1.0);
    }
    
    #[tokio::test]
    async fn test_cross_scale_influence() {
        let config = AdaptiveCycleConfig {
            scale_level: 0,
            cycle_duration: Duration::from_secs(3600),
            transition_thresholds: HashMap::new(),
        };
        
        let mut cycle = AdaptiveCycle::new(config).await.unwrap();
        
        let result = cycle.apply_cross_scale_influence(
            0.9,
            AdaptiveCyclePhase::Release,
        ).await;
        
        assert!(result.is_ok());
        assert!(cycle.state.adaptation_pressure > 0.0);
    }
}