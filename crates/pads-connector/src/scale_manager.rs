//! Scale management for panarchy system

use crate::{
    config::{PadsConfig, ScaleConfig},
    error::{PadsError, Result},
    monitoring::PadsMonitor,
    types::*,
};
use dashmap::DashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info, warn};

/// Manages panarchy scales and transitions
pub struct ScaleManager {
    config: Arc<PadsConfig>,
    monitor: Arc<PadsMonitor>,
    scales: DashMap<ScaleLevel, ScaleState>,
    current_phase: RwLock<AdaptiveCyclePhase>,
    transition_manager: Arc<TransitionManager>,
    adaptive_parameters: RwLock<AdaptiveParameters>,
}

/// Scale state information
#[derive(Debug, Clone)]
struct ScaleState {
    scale: PanarchyScale,
    active_decisions: usize,
    performance_history: Vec<f64>,
    capacity_remaining: f64,
    last_transition: chrono::DateTime<chrono::Utc>,
}

/// Manages scale transitions
struct TransitionManager {
    active_transitions: DashMap<String, TransitionInfo>,
    transition_history: RwLock<Vec<TransitionRecord>>,
}

/// Transition information
#[derive(Debug, Clone)]
struct TransitionInfo {
    id: String,
    from_scale: ScaleLevel,
    to_scale: ScaleLevel,
    started_at: chrono::DateTime<chrono::Utc>,
    progress: f64,
    trigger: TransitionTrigger,
}

/// Transition trigger
#[derive(Debug, Clone)]
enum TransitionTrigger {
    PerformanceThreshold,
    CapacityExhaustion,
    TimeHorizon,
    ExternalShock,
    AdaptiveCycle,
}

/// Transition record for history
#[derive(Debug, Clone)]
struct TransitionRecord {
    transition: TransitionInfo,
    completed_at: chrono::DateTime<chrono::Utc>,
    success: bool,
    impact: f64,
}

/// Adaptive parameters that evolve over time
#[derive(Debug, Clone)]
struct AdaptiveParameters {
    exploitation_rate: f64,
    exploration_rate: f64,
    innovation_threshold: f64,
    resilience_factor: f64,
    learning_rate: f64,
}

impl ScaleManager {
    /// Create new scale manager
    pub async fn new(config: Arc<PadsConfig>, monitor: Arc<PadsMonitor>) -> Result<Self> {
        let scales = DashMap::new();
        
        // Initialize scales
        for level in [ScaleLevel::Micro, ScaleLevel::Meso, ScaleLevel::Macro] {
            let scale_state = Self::create_scale_state(&config.scale_config, level)?;
            scales.insert(level, scale_state);
        }
        
        let transition_manager = Arc::new(TransitionManager {
            active_transitions: DashMap::new(),
            transition_history: RwLock::new(Vec::new()),
        });
        
        let adaptive_parameters = RwLock::new(AdaptiveParameters {
            exploitation_rate: 0.7,
            exploration_rate: 0.3,
            innovation_threshold: 0.6,
            resilience_factor: 0.8,
            learning_rate: 0.1,
        });
        
        Ok(Self {
            config,
            monitor,
            scales,
            current_phase: RwLock::new(AdaptiveCyclePhase::Growth),
            transition_manager,
            adaptive_parameters,
        })
    }
    
    /// Create scale state for a level
    fn create_scale_state(config: &ScaleConfig, level: ScaleLevel) -> Result<ScaleState> {
        let (time_horizon, spatial_extent, connectivity, resilience, potential) = match level {
            ScaleLevel::Micro => (
                std::time::Duration::from_millis(config.micro_scale.time_horizon_ms),
                config.micro_scale.search_radius,
                0.3,
                0.5,
                0.7,
            ),
            ScaleLevel::Meso => (
                std::time::Duration::from_secs(config.meso_scale.time_horizon_secs),
                0.5,
                0.6,
                0.7,
                0.6,
            ),
            ScaleLevel::Macro => (
                std::time::Duration::from_secs(config.macro_scale.time_horizon_mins * 60),
                1.0,
                0.9,
                0.9,
                0.8,
            ),
        };
        
        Ok(ScaleState {
            scale: PanarchyScale {
                level,
                time_horizon,
                spatial_extent,
                connectivity,
                resilience,
                potential,
            },
            active_decisions: 0,
            performance_history: Vec::with_capacity(100),
            capacity_remaining: 1.0,
            last_transition: chrono::Utc::now(),
        })
    }
    
    /// Initialize scales
    pub async fn initialize_scales(&mut self) -> Result<()> {
        info!("Initializing panarchy scales");
        
        // Set initial phase based on system state
        *self.current_phase.write().await = AdaptiveCyclePhase::Growth;
        
        // Initialize scale relationships
        self.setup_scale_relationships().await?;
        
        Ok(())
    }
    
    /// Setup relationships between scales
    async fn setup_scale_relationships(&self) -> Result<()> {
        // Define cross-scale interactions
        debug!("Setting up cross-scale relationships");
        
        // Micro influences Meso through aggregation
        // Meso influences both Micro and Macro through coordination
        // Macro influences lower scales through constraints
        
        Ok(())
    }
    
    /// Determine appropriate scale for decision
    pub async fn determine_scale(&self, decision: &PanarchyDecision) -> Result<PanarchyScale> {
        let params = self.adaptive_parameters.read().await;
        
        // Score each scale based on decision characteristics
        let micro_score = self.score_micro_scale(decision, &params);
        let meso_score = self.score_meso_scale(decision, &params);
        let macro_score = self.score_macro_scale(decision, &params);
        
        // Select scale with highest score
        let (selected_level, score) = if micro_score >= meso_score && micro_score >= macro_score {
            (ScaleLevel::Micro, micro_score)
        } else if meso_score >= macro_score {
            (ScaleLevel::Meso, meso_score)
        } else {
            (ScaleLevel::Macro, macro_score)
        };
        
        debug!("Selected scale {:?} with score {}", selected_level, score);
        
        // Get scale info
        let scale_state = self.scales.get(&selected_level)
            .ok_or_else(|| PadsError::scale("Scale not found"))?;
        
        Ok(scale_state.scale.clone())
    }
    
    /// Score micro scale suitability
    fn score_micro_scale(&self, decision: &PanarchyDecision, params: &AdaptiveParameters) -> f64 {
        let mut score = 0.0;
        
        // High urgency favors micro scale
        score += decision.urgency * 0.3;
        
        // Low uncertainty favors micro scale
        score += (1.0 - decision.uncertainty) * 0.2;
        
        // Exploitation mode favors micro scale
        score += params.exploitation_rate * 0.3;
        
        // Low impact decisions suit micro scale
        score += (1.0 - decision.impact) * 0.2;
        
        score
    }
    
    /// Score meso scale suitability
    fn score_meso_scale(&self, decision: &PanarchyDecision, params: &AdaptiveParameters) -> f64 {
        let mut score = 0.0;
        
        // Medium urgency favors meso scale
        score += (1.0 - (decision.urgency - 0.5).abs() * 2.0) * 0.3;
        
        // Medium uncertainty favors meso scale
        score += (1.0 - (decision.uncertainty - 0.5).abs() * 2.0) * 0.2;
        
        // Balance between exploration and exploitation
        let balance = 1.0 - (params.exploitation_rate - params.exploration_rate).abs();
        score += balance * 0.3;
        
        // Medium impact favors meso scale
        score += (1.0 - (decision.impact - 0.5).abs() * 2.0) * 0.2;
        
        score
    }
    
    /// Score macro scale suitability
    fn score_macro_scale(&self, decision: &PanarchyDecision, params: &AdaptiveParameters) -> f64 {
        let mut score = 0.0;
        
        // Low urgency favors macro scale
        score += (1.0 - decision.urgency) * 0.3;
        
        // High uncertainty favors macro scale
        score += decision.uncertainty * 0.2;
        
        // Exploration mode favors macro scale
        score += params.exploration_rate * 0.3;
        
        // High impact decisions suit macro scale
        score += decision.impact * 0.2;
        
        score
    }
    
    /// Check if scale transition is needed
    pub async fn should_transition(&self, result: &DecisionResult) -> Result<bool> {
        let scale_state = self.scales.get(&result.scale_level)
            .ok_or_else(|| PadsError::scale("Scale not found"))?;
        
        // Check various transition triggers
        if self.check_performance_trigger(&scale_state, result).await? {
            return Ok(true);
        }
        
        if self.check_capacity_trigger(&scale_state).await? {
            return Ok(true);
        }
        
        if self.check_time_trigger(&scale_state).await? {
            return Ok(true);
        }
        
        if self.check_adaptive_cycle_trigger().await? {
            return Ok(true);
        }
        
        Ok(false)
    }
    
    /// Check performance-based transition trigger
    async fn check_performance_trigger(
        &self,
        state: &ScaleState,
        result: &DecisionResult
    ) -> Result<bool> {
        let thresholds = &self.config.scale_config.transition_thresholds;
        let performance = result.metrics.confidence_score * 
                         (if result.success { 1.0 } else { 0.0 });
        
        match state.scale.level {
            ScaleLevel::Micro => Ok(performance < thresholds.micro_to_meso),
            ScaleLevel::Meso => {
                Ok(performance < thresholds.meso_to_micro || 
                   performance > thresholds.meso_to_macro)
            }
            ScaleLevel::Macro => Ok(performance < thresholds.macro_to_meso),
        }
    }
    
    /// Check capacity-based transition trigger
    async fn check_capacity_trigger(&self, state: &ScaleState) -> Result<bool> {
        Ok(state.capacity_remaining < 0.2)
    }
    
    /// Check time-based transition trigger
    async fn check_time_trigger(&self, state: &ScaleState) -> Result<bool> {
        let elapsed = chrono::Utc::now() - state.last_transition;
        let max_duration = state.scale.time_horizon * 10;
        Ok(elapsed > chrono::Duration::from_std(max_duration).unwrap())
    }
    
    /// Check adaptive cycle-based transition trigger
    async fn check_adaptive_cycle_trigger(&self) -> Result<bool> {
        let phase = self.current_phase.read().await;
        match *phase {
            AdaptiveCyclePhase::Conservation => Ok(true), // Ready for release
            AdaptiveCyclePhase::Release => Ok(true), // Ready for reorganization
            _ => Ok(false),
        }
    }
    
    /// Transition to new scale
    pub async fn transition_scale(
        &mut self,
        from: ScaleLevel,
        result: &DecisionResult
    ) -> Result<()> {
        let to = self.determine_target_scale(from, result).await?;
        
        info!("Transitioning from {:?} to {:?}", from, to);
        
        // Create transition record
        let transition = TransitionInfo {
            id: uuid::Uuid::new_v4().to_string(),
            from_scale: from,
            to_scale: to,
            started_at: chrono::Utc::now(),
            progress: 0.0,
            trigger: self.identify_trigger(result).await?,
        };
        
        // Start transition
        self.transition_manager.active_transitions
            .insert(transition.id.clone(), transition.clone());
        
        // Execute transition
        self.execute_transition(transition).await?;
        
        Ok(())
    }
    
    /// Determine target scale for transition
    async fn determine_target_scale(
        &self,
        from: ScaleLevel,
        result: &DecisionResult
    ) -> Result<ScaleLevel> {
        let params = self.adaptive_parameters.read().await;
        
        match from {
            ScaleLevel::Micro => {
                if params.exploration_rate > params.exploitation_rate {
                    Ok(ScaleLevel::Meso)
                } else {
                    Ok(ScaleLevel::Micro) // Stay if still exploiting
                }
            }
            ScaleLevel::Meso => {
                if result.success && params.exploitation_rate > 0.7 {
                    Ok(ScaleLevel::Micro)
                } else if !result.success || params.exploration_rate > 0.7 {
                    Ok(ScaleLevel::Macro)
                } else {
                    Ok(ScaleLevel::Meso)
                }
            }
            ScaleLevel::Macro => {
                if params.exploitation_rate > params.exploration_rate {
                    Ok(ScaleLevel::Meso)
                } else {
                    Ok(ScaleLevel::Macro) // Stay if still exploring
                }
            }
        }
    }
    
    /// Identify transition trigger
    async fn identify_trigger(&self, result: &DecisionResult) -> Result<TransitionTrigger> {
        if !result.success {
            Ok(TransitionTrigger::PerformanceThreshold)
        } else if result.metrics.resource_usage > 0.9 {
            Ok(TransitionTrigger::CapacityExhaustion)
        } else {
            Ok(TransitionTrigger::AdaptiveCycle)
        }
    }
    
    /// Execute scale transition
    async fn execute_transition(&mut self, mut transition: TransitionInfo) -> Result<()> {
        // Update scale states
        if let Some(mut from_state) = self.scales.get_mut(&transition.from_scale) {
            from_state.active_decisions = 0;
            from_state.capacity_remaining = 1.0;
        }
        
        if let Some(mut to_state) = self.scales.get_mut(&transition.to_scale) {
            to_state.last_transition = chrono::Utc::now();
        }
        
        // Update adaptive parameters
        self.update_adaptive_parameters(&transition).await?;
        
        // Complete transition
        transition.progress = 1.0;
        self.transition_manager.active_transitions.remove(&transition.id);
        
        // Record in history
        let record = TransitionRecord {
            transition: transition.clone(),
            completed_at: chrono::Utc::now(),
            success: true,
            impact: 0.8, // Would calculate actual impact
        };
        
        self.transition_manager.transition_history.write().await.push(record);
        
        Ok(())
    }
    
    /// Update adaptive parameters based on transition
    async fn update_adaptive_parameters(&self, transition: &TransitionInfo) -> Result<()> {
        let mut params = self.adaptive_parameters.write().await;
        
        match (transition.from_scale, transition.to_scale) {
            (ScaleLevel::Micro, ScaleLevel::Meso) => {
                // Moving up: increase exploration
                params.exploration_rate = (params.exploration_rate + 0.1).min(0.9);
                params.exploitation_rate = 1.0 - params.exploration_rate;
            }
            (ScaleLevel::Meso, ScaleLevel::Macro) => {
                // Moving to macro: maximize exploration
                params.exploration_rate = (params.exploration_rate + 0.2).min(0.95);
                params.exploitation_rate = 1.0 - params.exploration_rate;
            }
            (ScaleLevel::Macro, ScaleLevel::Meso) => {
                // Moving down: increase exploitation
                params.exploitation_rate = (params.exploitation_rate + 0.1).min(0.9);
                params.exploration_rate = 1.0 - params.exploitation_rate;
            }
            (ScaleLevel::Meso, ScaleLevel::Micro) => {
                // Moving to micro: maximize exploitation
                params.exploitation_rate = (params.exploitation_rate + 0.2).min(0.95);
                params.exploration_rate = 1.0 - params.exploitation_rate;
            }
            _ => {} // Same scale or invalid transition
        }
        
        Ok(())
    }
    
    /// Get adjacent scales
    pub async fn get_adjacent_scales(&self, level: ScaleLevel) -> Result<Vec<PanarchyScale>> {
        let mut adjacent = Vec::new();
        
        match level {
            ScaleLevel::Micro => {
                if let Some(meso) = self.scales.get(&ScaleLevel::Meso) {
                    adjacent.push(meso.scale.clone());
                }
            }
            ScaleLevel::Meso => {
                if let Some(micro) = self.scales.get(&ScaleLevel::Micro) {
                    adjacent.push(micro.scale.clone());
                }
                if let Some(macro) = self.scales.get(&ScaleLevel::Macro) {
                    adjacent.push(macro.scale.clone());
                }
            }
            ScaleLevel::Macro => {
                if let Some(meso) = self.scales.get(&ScaleLevel::Meso) {
                    adjacent.push(meso.scale.clone());
                }
            }
        }
        
        Ok(adjacent)
    }
    
    /// Update strategic parameters from macro insights
    pub async fn update_strategic_parameters(&mut self, insights: MacroInsights) -> Result<()> {
        let mut params = self.adaptive_parameters.write().await;
        
        // Update parameters based on insights
        if insights.strategic_direction == "innovative" {
            params.innovation_threshold = (params.innovation_threshold - 0.1).max(0.3);
        }
        
        // Apply risk adjustments
        for (param, adjustment) in insights.risk_adjustments {
            if param == "resilience" {
                params.resilience_factor = (params.resilience_factor + adjustment).clamp(0.1, 1.0);
            }
        }
        
        Ok(())
    }
    
    /// Get scale management status
    pub async fn get_status(&self) -> Result<ScaleStatus> {
        let mut active_scales = HashMap::new();
        
        for scale_ref in self.scales.iter() {
            let level = *scale_ref.key();
            let state = scale_ref.value();
            
            active_scales.insert(level, ScaleInfo {
                level,
                active_decisions: state.active_decisions,
                capacity_used: 1.0 - state.capacity_remaining,
                performance: state.performance_history.iter().sum::<f64>() / 
                             state.performance_history.len().max(1) as f64,
            });
        }
        
        let current_phase = *self.current_phase.read().await;
        
        let transition_state = self.transition_manager.active_transitions
            .iter()
            .next()
            .map(|t| TransitionState {
                from: t.from_scale,
                to: t.to_scale,
                progress: t.progress,
                estimated_completion: t.started_at + chrono::Duration::seconds(10),
            });
        
        Ok(ScaleStatus {
            active_scales,
            current_phase,
            transition_state,
        })
    }
    
    /// Reset to stable state
    pub async fn reset_to_stable(&mut self) -> Result<()> {
        warn!("Resetting scale manager to stable state");
        
        // Clear active transitions
        self.transition_manager.active_transitions.clear();
        
        // Reset to growth phase
        *self.current_phase.write().await = AdaptiveCyclePhase::Growth;
        
        // Reset adaptive parameters to defaults
        *self.adaptive_parameters.write().await = AdaptiveParameters {
            exploitation_rate: 0.7,
            exploration_rate: 0.3,
            innovation_threshold: 0.6,
            resilience_factor: 0.8,
            learning_rate: 0.1,
        };
        
        // Reset scale states
        for mut scale_ref in self.scales.iter_mut() {
            scale_ref.active_decisions = 0;
            scale_ref.capacity_remaining = 1.0;
            scale_ref.performance_history.clear();
        }
        
        Ok(())
    }
    
    /// Adapt parameters based on feedback
    pub async fn adapt_parameters(&mut self, feedback: AdaptiveFeedback) -> Result<()> {
        let mut params = self.adaptive_parameters.write().await;
        
        // Adjust learning rate based on performance
        if feedback.performance_score > 0.8 {
            params.learning_rate = (params.learning_rate * 0.9).max(0.01);
        } else {
            params.learning_rate = (params.learning_rate * 1.1).min(0.5);
        }
        
        // Apply scale adjustments
        for (level, adjustment) in feedback.scale_adjustments {
            match level {
                ScaleLevel::Micro => {
                    params.exploitation_rate = (params.exploitation_rate + adjustment)
                        .clamp(0.1, 0.9);
                }
                ScaleLevel::Macro => {
                    params.exploration_rate = (params.exploration_rate + adjustment)
                        .clamp(0.1, 0.9);
                }
                _ => {}
            }
        }
        
        // Ensure rates sum to 1
        let total = params.exploitation_rate + params.exploration_rate;
        params.exploitation_rate /= total;
        params.exploration_rate /= total;
        
        Ok(())
    }
    
    /// Get adaptive capacity
    pub async fn get_adaptive_capacity(&self) -> Result<f64> {
        let params = self.adaptive_parameters.read().await;
        
        // Calculate capacity based on current state
        let scale_capacity: f64 = self.scales.iter()
            .map(|s| s.capacity_remaining * s.scale.resilience)
            .sum::<f64>() / self.scales.len() as f64;
        
        let adaptive_capacity = scale_capacity * params.resilience_factor * 
                               (1.0 + params.innovation_threshold);
        
        Ok(adaptive_capacity.clamp(0.0, 1.0))
    }
}

use std::collections::HashMap;

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_scale_determination() {
        let config = Arc::new(PadsConfig::default());
        let monitor = Arc::new(PadsMonitor::new(config.clone()).await.unwrap());
        let manager = ScaleManager::new(config, monitor).await.unwrap();
        
        let decision = PanarchyDecision::test_decision();
        let scale = manager.determine_scale(&decision).await.unwrap();
        assert!(matches!(scale.level, ScaleLevel::Micro | ScaleLevel::Meso | ScaleLevel::Macro));
    }
}