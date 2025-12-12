//! # PADS Core System
//!
//! Core infrastructure for the Panarchy Adaptive Decision System.
//! Provides fundamental types, error handling, and system coordination.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::{RwLock, Mutex};
use serde::{Deserialize, Serialize};
use thiserror::Error;
use tracing::{info, warn, error, debug, trace};

pub mod types;
pub mod config;
pub mod system;
pub mod traits;

pub use types::*;
pub use config::*;
pub use system::*;
pub use traits::*;

/// PADS system errors
#[derive(Error, Debug, Clone)]
pub enum PadsError {
    #[error("Configuration error: {message}")]
    Configuration { message: String },
    
    #[error("Decision layer error: {layer:?} - {message}")]
    DecisionLayer { layer: DecisionLayer, message: String },
    
    #[error("Panarchy cycle error: {phase:?} - {message}")]
    PanarchyCycle { phase: AdaptiveCyclePhase, message: String },
    
    #[error("Integration error: {component} - {message}")]
    Integration { component: String, message: String },
    
    #[error("Performance error: {metric} - {message}")]
    Performance { metric: String, message: String },
    
    #[error("Governance error: {policy} - {message}")]
    Governance { policy: String, message: String },
    
    #[error("System state error: {state} - {message}")]
    SystemState { state: String, message: String },
    
    #[error("Async operation error: {operation} - {message}")]
    AsyncOperation { operation: String, message: String },
    
    #[error("Resource error: {resource} - {message}")]
    Resource { resource: String, message: String },
    
    #[error("Validation error: {field} - {message}")]
    Validation { field: String, message: String },
}

/// Result type for PADS operations
pub type PadsResult<T> = Result<T, PadsError>;

/// Decision layer hierarchy
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DecisionLayer {
    /// Immediate response and execution (microseconds to seconds)
    Tactical,
    /// Short-term optimization and coordination (seconds to minutes)
    Operational,
    /// Medium-term planning and resource allocation (minutes to hours)
    Strategic,
    /// Long-term adaptation and transformation (hours to days)
    MetaStrategic,
}

impl DecisionLayer {
    /// Get the typical time horizon for this decision layer
    pub fn time_horizon(&self) -> Duration {
        match self {
            Self::Tactical => Duration::from_millis(100),      // ~100ms
            Self::Operational => Duration::from_secs(30),      // ~30s
            Self::Strategic => Duration::from_secs(1800),      // ~30min
            Self::MetaStrategic => Duration::from_secs(21600), // ~6h
        }
    }
    
    /// Get the priority level (higher = more urgent)
    pub fn priority(&self) -> u8 {
        match self {
            Self::Tactical => 4,
            Self::Operational => 3,
            Self::Strategic => 2,
            Self::MetaStrategic => 1,
        }
    }
    
    /// Get all decision layers in priority order
    pub fn all_layers() -> Vec<Self> {
        vec![Self::Tactical, Self::Operational, Self::Strategic, Self::MetaStrategic]
    }
}

/// Adaptive cycle phases from panarchy theory
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AdaptiveCyclePhase {
    /// Growth phase (r) - rapid expansion and exploitation
    Growth,
    /// Conservation phase (K) - optimization and consolidation
    Conservation,
    /// Release phase (Ω) - creative destruction and innovation
    Release,
    /// Reorganization phase (α) - renewal and transformation
    Reorganization,
}

impl AdaptiveCyclePhase {
    /// Get the next phase in the adaptive cycle
    pub fn next_phase(&self) -> Self {
        match self {
            Self::Growth => Self::Conservation,
            Self::Conservation => Self::Release,
            Self::Release => Self::Reorganization,
            Self::Reorganization => Self::Growth,
        }
    }
    
    /// Get all phases in cycle order
    pub fn all_phases() -> Vec<Self> {
        vec![Self::Growth, Self::Conservation, Self::Release, Self::Reorganization]
    }
    
    /// Get the typical characteristics of this phase
    pub fn characteristics(&self) -> PhaseCharacteristics {
        match self {
            Self::Growth => PhaseCharacteristics {
                potential: 0.8,
                connectedness: 0.3,
                resilience: 0.6,
                innovation: 0.7,
                efficiency: 0.4,
            },
            Self::Conservation => PhaseCharacteristics {
                potential: 0.3,
                connectedness: 0.9,
                resilience: 0.3,
                innovation: 0.2,
                efficiency: 0.9,
            },
            Self::Release => PhaseCharacteristics {
                potential: 0.9,
                connectedness: 0.1,
                resilience: 0.1,
                innovation: 0.9,
                efficiency: 0.2,
            },
            Self::Reorganization => PhaseCharacteristics {
                potential: 0.6,
                connectedness: 0.4,
                resilience: 0.8,
                innovation: 0.8,
                efficiency: 0.5,
            },
        }
    }
}

/// Phase characteristics for adaptive cycles
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct PhaseCharacteristics {
    /// Available potential for change (0.0 to 1.0)
    pub potential: f64,
    /// Level of system connectedness (0.0 to 1.0)
    pub connectedness: f64,
    /// System resilience to disturbance (0.0 to 1.0)
    pub resilience: f64,
    /// Innovation capacity (0.0 to 1.0)
    pub innovation: f64,
    /// Operational efficiency (0.0 to 1.0)
    pub efficiency: f64,
}

/// Decision context with metadata and constraints
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionContext {
    /// Unique identifier for this decision context
    pub id: String,
    /// Target decision layer
    pub layer: DecisionLayer,
    /// Current adaptive cycle phase
    pub cycle_phase: AdaptiveCyclePhase,
    /// Decision urgency (0.0 to 1.0)
    pub urgency: f64,
    /// Available decision time
    pub time_budget: Duration,
    /// Resource constraints
    pub constraints: HashMap<String, f64>,
    /// Environmental factors
    pub environment: HashMap<String, f64>,
    /// Historical context
    pub history: Vec<DecisionOutcome>,
    /// Uncertainty level (0.0 to 1.0)
    pub uncertainty: f64,
    /// Risk tolerance (0.0 to 1.0)
    pub risk_tolerance: f64,
    /// Created timestamp
    pub created_at: Instant,
}

impl DecisionContext {
    /// Create a new decision context
    pub fn new(
        id: String,
        layer: DecisionLayer,
        cycle_phase: AdaptiveCyclePhase,
    ) -> Self {
        Self {
            id,
            layer,
            cycle_phase,
            urgency: 0.5,
            time_budget: layer.time_horizon(),
            constraints: HashMap::new(),
            environment: HashMap::new(),
            history: Vec::new(),
            uncertainty: 0.5,
            risk_tolerance: 0.5,
            created_at: Instant::now(),
        }
    }
    
    /// Check if the decision context is still valid (within time budget)
    pub fn is_valid(&self) -> bool {
        self.created_at.elapsed() < self.time_budget
    }
    
    /// Get remaining time budget
    pub fn remaining_time(&self) -> Duration {
        self.time_budget.saturating_sub(self.created_at.elapsed())
    }
}

/// Outcome of a decision for learning and adaptation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionOutcome {
    /// Decision identifier
    pub decision_id: String,
    /// Context that led to this decision
    pub context: DecisionContext,
    /// Chosen action/alternative
    pub action: String,
    /// Performance metrics
    pub performance: HashMap<String, f64>,
    /// Success indicator (0.0 to 1.0)
    pub success_score: f64,
    /// Lessons learned
    pub lessons: Vec<String>,
    /// Timestamp of decision execution
    pub executed_at: Instant,
    /// Outcome evaluation timestamp
    pub evaluated_at: Option<Instant>,
}

/// System health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum SystemHealth {
    /// System operating optimally
    Healthy,
    /// Minor issues, still operational
    Degraded,
    /// Significant issues, limited functionality
    Compromised,
    /// System failure, requires intervention
    Failed,
}

impl SystemHealth {
    /// Check if system is operational
    pub fn is_operational(&self) -> bool {
        matches!(self, Self::Healthy | Self::Degraded)
    }
    
    /// Get health score (0.0 to 1.0)
    pub fn score(&self) -> f64 {
        match self {
            Self::Healthy => 1.0,
            Self::Degraded => 0.7,
            Self::Compromised => 0.3,
            Self::Failed => 0.0,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_decision_layer_hierarchy() {
        assert!(DecisionLayer::Tactical.priority() > DecisionLayer::Operational.priority());
        assert!(DecisionLayer::Operational.priority() > DecisionLayer::Strategic.priority());
        assert!(DecisionLayer::Strategic.priority() > DecisionLayer::MetaStrategic.priority());
    }
    
    #[test]
    fn test_adaptive_cycle_phases() {
        let phase = AdaptiveCyclePhase::Growth;
        assert_eq!(phase.next_phase(), AdaptiveCyclePhase::Conservation);
        
        let characteristics = phase.characteristics();
        assert!(characteristics.potential > 0.5);
        assert!(characteristics.innovation > 0.5);
    }
    
    #[test]
    fn test_decision_context_validity() {
        let context = DecisionContext::new(
            "test-001".to_string(),
            DecisionLayer::Tactical,
            AdaptiveCyclePhase::Growth,
        );
        
        assert!(context.is_valid());
        assert!(context.remaining_time() <= context.time_budget);
    }
    
    #[test]
    fn test_system_health() {
        assert!(SystemHealth::Healthy.is_operational());
        assert!(SystemHealth::Degraded.is_operational());
        assert!(!SystemHealth::Compromised.is_operational());
        assert!(!SystemHealth::Failed.is_operational());
        
        assert_eq!(SystemHealth::Healthy.score(), 1.0);
        assert_eq!(SystemHealth::Failed.score(), 0.0);
    }
}