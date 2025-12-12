//! # Panarchy Framework
//!
//! Implementation of panarchy theory for adaptive systems modeling.
//! Provides hierarchical adaptive cycle management with cross-scale interactions.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use serde::{Deserialize, Serialize};
use tracing::{info, warn, debug, trace, instrument};

use crate::core::{
    PadsResult, PadsError, AdaptiveCyclePhase, SystemHealth, PhaseCharacteristics
};
use crate::core::traits::{
    PanarchyModel, PhaseMetrics, EmergentPattern, Adaptive, AdaptationState
};

pub mod adaptive_cycle;
pub mod cross_scale;
pub mod resilience;
pub mod emergence;

pub use adaptive_cycle::*;
pub use cross_scale::*;
pub use resilience::*;
pub use emergence::*;

/// Main panarchy framework implementation
#[derive(Debug)]
pub struct PanarchyFramework {
    /// System configuration
    config: PanarchyConfig,
    
    /// Current panarchy state
    state: Arc<RwLock<PanarchyState>>,
    
    /// Adaptive cycles for different scales
    cycles: HashMap<usize, Arc<RwLock<AdaptiveCycle>>>,
    
    /// Cross-scale interactions
    interactions: Arc<RwLock<CrossScaleInteractions>>,
    
    /// Resilience engine
    resilience: Arc<RwLock<ResilienceEngine>>,
    
    /// Emergence detector
    emergence: Arc<RwLock<EmergenceDetector>>,
    
    /// Historical data
    history: Arc<RwLock<PanarchyHistory>>,
}

/// Panarchy configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanarchyConfig {
    /// Number of hierarchical scales
    pub scale_count: usize,
    
    /// Cycle duration for each scale
    pub scale_durations: HashMap<usize, Duration>,
    
    /// Cross-scale interaction strength
    pub interaction_strength: f64,
    
    /// Resilience monitoring enabled
    pub resilience_monitoring: bool,
    
    /// Emergence detection enabled
    pub emergence_detection: bool,
    
    /// Phase transition thresholds
    pub transition_thresholds: HashMap<AdaptiveCyclePhase, f64>,
    
    /// Maximum cycles to track in history
    pub history_limit: usize,
}

impl Default for PanarchyConfig {
    fn default() -> Self {
        let mut scale_durations = HashMap::new();
        scale_durations.insert(0, Duration::from_secs(60));    // Tactical: 1 min
        scale_durations.insert(1, Duration::from_secs(900));   // Operational: 15 min
        scale_durations.insert(2, Duration::from_secs(3600));  // Strategic: 1 hour
        scale_durations.insert(3, Duration::from_secs(14400)); // Meta: 4 hours
        
        let mut transition_thresholds = HashMap::new();
        transition_thresholds.insert(AdaptiveCyclePhase::Growth, 0.8);
        transition_thresholds.insert(AdaptiveCyclePhase::Conservation, 0.9);
        transition_thresholds.insert(AdaptiveCyclePhase::Release, 0.7);
        transition_thresholds.insert(AdaptiveCyclePhase::Reorganization, 0.6);
        
        Self {
            scale_count: 4,
            scale_durations,
            interaction_strength: 0.7,
            resilience_monitoring: true,
            emergence_detection: true,
            transition_thresholds,
            history_limit: 1000,
        }
    }
}

/// Overall panarchy state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanarchyState {
    /// Current phases for each scale
    pub scale_phases: HashMap<usize, AdaptiveCyclePhase>,
    
    /// Phase characteristics for each scale
    pub scale_characteristics: HashMap<usize, PhaseCharacteristics>,
    
    /// System-wide resilience
    pub system_resilience: f64,
    
    /// Active emergent patterns
    pub emergent_patterns: Vec<EmergentPattern>,
    
    /// Cross-scale interaction strength
    pub interaction_level: f64,
    
    /// Overall system health
    pub health: SystemHealth,
    
    /// Last update timestamp
    pub last_updated: Instant,
}

impl Default for PanarchyState {
    fn default() -> Self {
        Self {
            scale_phases: HashMap::new(),
            scale_characteristics: HashMap::new(),
            system_resilience: 0.8,
            emergent_patterns: Vec::new(),
            interaction_level: 0.5,
            health: SystemHealth::Healthy,
            last_updated: Instant::now(),
        }
    }
}

/// Historical panarchy data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PanarchyHistory {
    /// Historical states
    pub states: Vec<PanarchyState>,
    
    /// Phase transitions
    pub transitions: Vec<PhaseTransition>,
    
    /// Emergence events
    pub emergence_events: Vec<EmergenceEvent>,
    
    /// Resilience measurements
    pub resilience_history: Vec<ResilienceMeasurement>,
}

impl Default for PanarchyHistory {
    fn default() -> Self {
        Self {
            states: Vec::new(),
            transitions: Vec::new(),
            emergence_events: Vec::new(),
            resilience_history: Vec::new(),
        }
    }
}

/// Phase transition record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PhaseTransition {
    /// Scale level
    pub scale: usize,
    
    /// From phase
    pub from_phase: AdaptiveCyclePhase,
    
    /// To phase
    pub to_phase: AdaptiveCyclePhase,
    
    /// Transition timestamp
    pub timestamp: Instant,
    
    /// Transition trigger
    pub trigger: TransitionTrigger,
    
    /// Transition confidence
    pub confidence: f64,
    
    /// Impact on other scales
    pub cross_scale_impact: HashMap<usize, f64>,
}

/// What triggered a phase transition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TransitionTrigger {
    /// Natural progression based on time
    Natural,
    
    /// Threshold-based transition
    Threshold { metric: String, value: f64 },
    
    /// Cross-scale influence
    CrossScale { from_scale: usize },
    
    /// External disturbance
    External { source: String },
    
    /// System adaptation
    Adaptive { reason: String },
}

/// Emergence event record
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergenceEvent {
    /// Event identifier
    pub id: String,
    
    /// Emergence timestamp
    pub timestamp: Instant,
    
    /// Emergent pattern detected
    pub pattern: EmergentPattern,
    
    /// Contributing scales
    pub contributing_scales: Vec<usize>,
    
    /// Event significance
    pub significance: f64,
    
    /// Predicted duration
    pub predicted_duration: Option<Duration>,
}

/// Resilience measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResilienceMeasurement {
    /// Measurement timestamp
    pub timestamp: Instant,
    
    /// Overall resilience score
    pub overall_resilience: f64,
    
    /// Scale-specific resilience
    pub scale_resilience: HashMap<usize, f64>,
    
    /// Resilience components
    pub components: ResilienceComponents,
    
    /// Measurement confidence
    pub confidence: f64,
}

/// Components of system resilience
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResilienceComponents {
    /// Adaptive capacity
    pub adaptive_capacity: f64,
    
    /// Absorption capacity
    pub absorption_capacity: f64,
    
    /// Recovery capability
    pub recovery_capability: f64,
    
    /// Transformation potential
    pub transformation_potential: f64,
    
    /// Learning capability
    pub learning_capability: f64,
}

impl PanarchyFramework {
    /// Create a new panarchy framework
    #[instrument(skip(config))]
    pub async fn new(config: PanarchyConfig) -> PadsResult<Self> {
        info!("Initializing panarchy framework with {} scales", config.scale_count);
        
        let state = Arc::new(RwLock::new(PanarchyState::default()));
        let mut cycles = HashMap::new();
        
        // Initialize adaptive cycles for each scale
        for scale in 0..config.scale_count {
            let cycle_config = AdaptiveCycleConfig {
                scale_level: scale,
                cycle_duration: config.scale_durations.get(&scale)
                    .copied()
                    .unwrap_or(Duration::from_secs(3600)),
                transition_thresholds: config.transition_thresholds.clone(),
            };
            
            let cycle = AdaptiveCycle::new(cycle_config).await?;
            cycles.insert(scale, Arc::new(RwLock::new(cycle)));
        }
        
        let interactions = Arc::new(RwLock::new(
            CrossScaleInteractions::new(config.scale_count, config.interaction_strength).await?
        ));
        
        let resilience = Arc::new(RwLock::new(
            ResilienceEngine::new().await?
        ));
        
        let emergence = Arc::new(RwLock::new(
            EmergenceDetector::new().await?
        ));
        
        let history = Arc::new(RwLock::new(PanarchyHistory::default()));
        
        // Initialize state
        {
            let mut state_guard = state.write().await;
            for scale in 0..config.scale_count {
                state_guard.scale_phases.insert(scale, AdaptiveCyclePhase::Growth);
                state_guard.scale_characteristics.insert(
                    scale, 
                    AdaptiveCyclePhase::Growth.characteristics()
                );
            }
        }
        
        info!("Panarchy framework initialized successfully");
        
        Ok(Self {
            config,
            state,
            cycles,
            interactions,
            resilience,
            emergence,
            history,
        })
    }
    
    /// Start the panarchy framework
    #[instrument(skip(self))]
    pub async fn start(&mut self) -> PadsResult<()> {
        info!("Starting panarchy framework");
        
        // Start all adaptive cycles
        for (scale, cycle) in &self.cycles {
            debug!("Starting adaptive cycle for scale {}", scale);
            cycle.write().await.start().await?;
        }
        
        // Start monitoring tasks
        if self.config.resilience_monitoring {
            self.start_resilience_monitoring().await?;
        }
        
        if self.config.emergence_detection {
            self.start_emergence_detection().await?;
        }
        
        self.start_cross_scale_coordination().await?;
        
        info!("Panarchy framework started successfully");
        Ok(())
    }
    
    /// Update the panarchy system
    #[instrument(skip(self))]
    pub async fn update(&mut self) -> PadsResult<()> {
        trace!("Updating panarchy framework");
        
        // Update all adaptive cycles
        for (scale, cycle) in &self.cycles {
            let mut cycle_guard = cycle.write().await;
            cycle_guard.update().await?;
            
            // Check for phase transitions
            if cycle_guard.should_transition().await? {
                let old_phase = cycle_guard.get_current_phase().await;
                let new_phase = cycle_guard.transition_phase().await?;
                
                self.record_phase_transition(
                    *scale,
                    old_phase,
                    new_phase,
                    TransitionTrigger::Natural,
                ).await?;
            }
        }
        
        // Update cross-scale interactions
        self.interactions.write().await.update(&self.cycles).await?;
        
        // Update system state
        self.update_system_state().await?;
        
        // Detect emergence
        if self.config.emergence_detection {
            self.detect_emergence().await?;
        }
        
        // Assess resilience
        if self.config.resilience_monitoring {
            self.assess_resilience().await?;
        }
        
        Ok(())
    }
    
    /// Get current panarchy state
    pub async fn get_state(&self) -> PanarchyState {
        self.state.read().await.clone()
    }
    
    /// Get adaptive cycle for a specific scale
    pub async fn get_cycle(&self, scale: usize) -> Option<AdaptiveCycle> {
        if let Some(cycle_ref) = self.cycles.get(&scale) {
            Some(cycle_ref.read().await.clone())
        } else {
            None
        }
    }
    
    /// Force a phase transition at a specific scale
    #[instrument(skip(self))]
    pub async fn force_transition(
        &mut self,
        scale: usize,
        target_phase: AdaptiveCyclePhase,
        trigger: TransitionTrigger,
    ) -> PadsResult<()> {
        info!("Forcing transition at scale {} to phase {:?}", scale, target_phase);
        
        if let Some(cycle) = self.cycles.get(&scale) {
            let mut cycle_guard = cycle.write().await;
            let old_phase = cycle_guard.get_current_phase().await;
            
            cycle_guard.force_transition(target_phase).await?;
            
            self.record_phase_transition(
                scale,
                old_phase,
                target_phase,
                trigger,
            ).await?;
            
            // Update cross-scale interactions
            self.propagate_transition_effects(scale, target_phase).await?;
        }
        
        Ok(())
    }
    
    /// Start resilience monitoring
    async fn start_resilience_monitoring(&self) -> PadsResult<()> {
        debug!("Starting resilience monitoring");
        
        let resilience = self.resilience.clone();
        let state = self.state.clone();
        let cycles = self.cycles.clone();
        let history = self.history.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                if let Ok(measurement) = Self::measure_resilience(&cycles, &resilience).await {
                    // Update state
                    {
                        let mut state_guard = state.write().await;
                        state_guard.system_resilience = measurement.overall_resilience;
                    }
                    
                    // Record in history
                    {
                        let mut history_guard = history.write().await;
                        history_guard.resilience_history.push(measurement);
                        
                        // Limit history size
                        if history_guard.resilience_history.len() > 1000 {
                            history_guard.resilience_history.remove(0);
                        }
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Start emergence detection
    async fn start_emergence_detection(&self) -> PadsResult<()> {
        debug!("Starting emergence detection");
        
        let emergence = self.emergence.clone();
        let state = self.state.clone();
        let cycles = self.cycles.clone();
        let history = self.history.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(60));
            
            loop {
                interval.tick().await;
                
                if let Ok(patterns) = Self::detect_emergent_patterns(&cycles, &emergence).await {
                    if !patterns.is_empty() {
                        // Update state
                        {
                            let mut state_guard = state.write().await;
                            state_guard.emergent_patterns = patterns.clone();
                        }
                        
                        // Record emergence events
                        {
                            let mut history_guard = history.write().await;
                            for pattern in patterns {
                                let event = EmergenceEvent {
                                    id: format!("emergence-{}", uuid::Uuid::new_v4()),
                                    timestamp: Instant::now(),
                                    pattern,
                                    contributing_scales: (0..4).collect(),
                                    significance: 0.7,
                                    predicted_duration: Some(Duration::from_secs(300)),
                                };
                                history_guard.emergence_events.push(event);
                            }
                        }
                    }
                }
            }
        });
        
        Ok(())
    }
    
    /// Start cross-scale coordination
    async fn start_cross_scale_coordination(&self) -> PadsResult<()> {
        debug!("Starting cross-scale coordination");
        
        let interactions = self.interactions.clone();
        let cycles = self.cycles.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            
            loop {
                interval.tick().await;
                
                if let Err(e) = interactions.write().await.coordinate(&cycles).await {
                    warn!("Cross-scale coordination error: {}", e);
                }
            }
        });
        
        Ok(())
    }
    
    /// Update system state based on cycles
    async fn update_system_state(&self) -> PadsResult<()> {
        let mut state_guard = self.state.write().await;
        
        // Update phases and characteristics
        for (scale, cycle) in &self.cycles {
            let cycle_guard = cycle.read().await;
            let phase = cycle_guard.get_current_phase().await;
            let characteristics = phase.characteristics();
            
            state_guard.scale_phases.insert(*scale, phase);
            state_guard.scale_characteristics.insert(*scale, characteristics);
        }
        
        // Update interaction level
        state_guard.interaction_level = self.interactions.read().await.get_interaction_strength().await;
        
        // Update timestamp
        state_guard.last_updated = Instant::now();
        
        Ok(())
    }
    
    /// Record a phase transition
    async fn record_phase_transition(
        &self,
        scale: usize,
        from_phase: AdaptiveCyclePhase,
        to_phase: AdaptiveCyclePhase,
        trigger: TransitionTrigger,
    ) -> PadsResult<()> {
        let transition = PhaseTransition {
            scale,
            from_phase,
            to_phase,
            timestamp: Instant::now(),
            trigger,
            confidence: 0.9,
            cross_scale_impact: HashMap::new(), // TODO: Calculate actual impact
        };
        
        let mut history_guard = self.history.write().await;
        history_guard.transitions.push(transition);
        
        // Limit history size
        if history_guard.transitions.len() > self.config.history_limit {
            history_guard.transitions.remove(0);
        }
        
        info!("Recorded phase transition: scale {} from {:?} to {:?}", 
               scale, from_phase, to_phase);
        
        Ok(())
    }
    
    /// Propagate transition effects to other scales
    async fn propagate_transition_effects(
        &self,
        source_scale: usize,
        new_phase: AdaptiveCyclePhase,
    ) -> PadsResult<()> {
        debug!("Propagating transition effects from scale {} (phase {:?})", 
               source_scale, new_phase);
        
        // Calculate influence on other scales
        for (target_scale, cycle) in &self.cycles {
            if *target_scale != source_scale {
                let influence = self.calculate_cross_scale_influence(
                    source_scale, 
                    *target_scale, 
                    new_phase
                ).await;
                
                if influence > 0.5 {
                    // Strong influence - consider forcing transition
                    let mut cycle_guard = cycle.write().await;
                    cycle_guard.apply_cross_scale_influence(influence, new_phase).await?;
                }
            }
        }
        
        Ok(())
    }
    
    /// Calculate cross-scale influence
    async fn calculate_cross_scale_influence(
        &self,
        source_scale: usize,
        target_scale: usize,
        source_phase: AdaptiveCyclePhase,
    ) -> f64 {
        // Implementation of cross-scale influence calculation
        // Based on panarchy theory and scale separation
        
        let scale_difference = (source_scale as i32 - target_scale as i32).abs() as f64;
        let base_influence = self.config.interaction_strength;
        
        // Influence decreases with scale separation
        let distance_factor = 1.0 / (1.0 + scale_difference * 0.5);
        
        // Phase-specific influence modifiers
        let phase_factor = match source_phase {
            AdaptiveCyclePhase::Release => 1.5,     // Release has strong influence
            AdaptiveCyclePhase::Reorganization => 1.2, // Reorganization moderately influential
            AdaptiveCyclePhase::Growth => 0.8,      // Growth has weak influence
            AdaptiveCyclePhase::Conservation => 0.6, // Conservation has minimal influence
        };
        
        base_influence * distance_factor * phase_factor
    }
    
    /// Detect emergence across scales
    async fn detect_emergence(&self) -> PadsResult<()> {
        let patterns = Self::detect_emergent_patterns(&self.cycles, &self.emergence).await?;
        
        if !patterns.is_empty() {
            let mut state_guard = self.state.write().await;
            state_guard.emergent_patterns = patterns;
        }
        
        Ok(())
    }
    
    /// Assess system resilience
    async fn assess_resilience(&self) -> PadsResult<()> {
        let measurement = Self::measure_resilience(&self.cycles, &self.resilience).await?;
        
        let mut state_guard = self.state.write().await;
        state_guard.system_resilience = measurement.overall_resilience;
        
        Ok(())
    }
    
    /// Measure system resilience
    async fn measure_resilience(
        cycles: &HashMap<usize, Arc<RwLock<AdaptiveCycle>>>,
        resilience_engine: &Arc<RwLock<ResilienceEngine>>,
    ) -> PadsResult<ResilienceMeasurement> {
        let mut scale_resilience = HashMap::new();
        let mut total_resilience = 0.0;
        
        for (scale, cycle) in cycles {
            let cycle_guard = cycle.read().await;
            let resilience = cycle_guard.calculate_resilience().await;
            scale_resilience.insert(*scale, resilience);
            total_resilience += resilience;
        }
        
        let overall_resilience = total_resilience / cycles.len() as f64;
        
        let components = ResilienceComponents {
            adaptive_capacity: 0.8,     // TODO: Calculate actual values
            absorption_capacity: 0.7,
            recovery_capability: 0.9,
            transformation_potential: 0.6,
            learning_capability: 0.8,
        };
        
        Ok(ResilienceMeasurement {
            timestamp: Instant::now(),
            overall_resilience,
            scale_resilience,
            components,
            confidence: 0.85,
        })
    }
    
    /// Detect emergent patterns
    async fn detect_emergent_patterns(
        cycles: &HashMap<usize, Arc<RwLock<AdaptiveCycle>>>,
        emergence_detector: &Arc<RwLock<EmergenceDetector>>,
    ) -> PadsResult<Vec<EmergentPattern>> {
        let mut detector_guard = emergence_detector.write().await;
        
        // Collect current state from all cycles
        let mut cycle_states = Vec::new();
        for cycle in cycles.values() {
            let cycle_guard = cycle.read().await;
            cycle_states.push(cycle_guard.get_state().await);
        }
        
        // Detect patterns
        detector_guard.detect_patterns(&cycle_states).await
    }
}

#[async_trait::async_trait]
impl PanarchyModel for PanarchyFramework {
    async fn get_current_phase(&self) -> AdaptiveCyclePhase {
        // Return the phase of the focal scale (usually scale 1 - operational)
        if let Some(cycle) = self.cycles.get(&1) {
            cycle.read().await.get_current_phase().await
        } else {
            AdaptiveCyclePhase::Growth // Default fallback
        }
    }
    
    async fn should_transition(&self) -> PadsResult<bool> {
        // Check if any scale should transition
        for cycle in self.cycles.values() {
            if cycle.read().await.should_transition().await? {
                return Ok(true);
            }
        }
        Ok(false)
    }
    
    async fn transition_phase(&mut self) -> PadsResult<AdaptiveCyclePhase> {
        // Transition the focal scale
        if let Some(cycle) = self.cycles.get(&1) {
            let new_phase = cycle.write().await.transition_phase().await?;
            
            // Propagate effects
            self.propagate_transition_effects(1, new_phase).await?;
            
            Ok(new_phase)
        } else {
            Err(PadsError::SystemState {
                state: "panarchy".to_string(),
                message: "No focal scale cycle available".to_string(),
            })
        }
    }
    
    async fn calculate_phase_characteristics(&self) -> PhaseMetrics {
        let state = self.state.read().await;
        
        // Aggregate characteristics across scales
        let mut total_characteristics = PhaseCharacteristics {
            potential: 0.0,
            connectedness: 0.0,
            resilience: 0.0,
            innovation: 0.0,
            efficiency: 0.0,
        };
        
        let scale_count = state.scale_characteristics.len() as f64;
        for characteristics in state.scale_characteristics.values() {
            total_characteristics.potential += characteristics.potential;
            total_characteristics.connectedness += characteristics.connectedness;
            total_characteristics.resilience += characteristics.resilience;
            total_characteristics.innovation += characteristics.innovation;
            total_characteristics.efficiency += characteristics.efficiency;
        }
        
        if scale_count > 0.0 {
            PhaseMetrics {
                potential: total_characteristics.potential / scale_count,
                connectedness: total_characteristics.connectedness / scale_count,
                resilience: total_characteristics.resilience / scale_count,
                innovation: total_characteristics.innovation / scale_count,
                efficiency: total_characteristics.efficiency / scale_count,
                stability: state.system_resilience,
            }
        } else {
            PhaseMetrics {
                potential: 0.5,
                connectedness: 0.5,
                resilience: 0.5,
                innovation: 0.5,
                efficiency: 0.5,
                stability: 0.5,
            }
        }
    }
    
    async fn assess_resilience(&self) -> PadsResult<f64> {
        Ok(self.state.read().await.system_resilience)
    }
    
    async fn detect_emergence(&self) -> PadsResult<Vec<EmergentPattern>> {
        Ok(self.state.read().await.emergent_patterns.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_panarchy_framework_creation() {
        let config = PanarchyConfig::default();
        let framework = PanarchyFramework::new(config).await;
        assert!(framework.is_ok());
    }
    
    #[tokio::test]
    async fn test_phase_transition() {
        let config = PanarchyConfig::default();
        let mut framework = PanarchyFramework::new(config).await.unwrap();
        
        let result = framework.force_transition(
            0,
            AdaptiveCyclePhase::Conservation,
            TransitionTrigger::External { source: "test".to_string() },
        ).await;
        
        assert!(result.is_ok());
    }
    
    #[tokio::test]
    async fn test_resilience_measurement() {
        let config = PanarchyConfig::default();
        let framework = PanarchyFramework::new(config).await.unwrap();
        
        let resilience = framework.assess_resilience().await;
        assert!(resilience.is_ok());
        assert!(resilience.unwrap() > 0.0);
    }
}