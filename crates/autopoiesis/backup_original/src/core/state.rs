//! System state module for autopoietic systems

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

/// The overall state of an autopoietic system
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SystemState {
    /// Internal state variables
    pub variables: HashMap<String, StateVariable>,
    
    /// System phase (e.g., initializing, active, adapting)
    pub phase: SystemPhase,
    
    /// Stability measure (0.0 = unstable, 1.0 = fully stable)
    pub stability: f64,
    
    /// Complexity measure
    pub complexity: f64,
    
    /// Energy level of the system
    pub energy: f64,
    
    /// Timestamp of last update
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Individual state variable
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct StateVariable {
    /// Variable name
    pub name: String,
    
    /// Current value
    pub value: serde_json::Value,
    
    /// Rate of change
    pub rate_of_change: f64,
    
    /// Historical values (limited to last N entries)
    pub history: Vec<(chrono::DateTime<chrono::Utc>, serde_json::Value)>,
}

/// System phase enumeration
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum SystemPhase {
    /// System is initializing
    Initializing,
    
    /// System is in stable operation
    Stable,
    
    /// System is adapting to changes
    Adapting,
    
    /// System is in a critical transition
    CriticalTransition,
    
    /// System is reorganizing
    Reorganizing,
    
    /// System is in error state
    Error(String),
}

impl Default for SystemState {
    fn default() -> Self {
        Self {
            variables: HashMap::new(),
            phase: SystemPhase::Initializing,
            stability: 0.5,
            complexity: 0.5,
            energy: 1.0,
            last_updated: chrono::Utc::now(),
        }
    }
}

impl SystemState {
    /// Create a new system state
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Update a state variable
    pub fn update_variable(&mut self, name: &str, value: serde_json::Value) {
        let now = chrono::Utc::now();
        
        if let Some(var) = self.variables.get_mut(name) {
            // Calculate rate of change
            if let Some((last_time, last_value)) = var.history.last() {
                if let (Some(current), Some(previous)) = (value.as_f64(), last_value.as_f64()) {
                    let time_diff = (now - *last_time).num_seconds() as f64;
                    if time_diff > 0.0 {
                        var.rate_of_change = (current - previous) / time_diff;
                    }
                }
            }
            
            // Update value and history
            var.value = value.clone();
            var.history.push((now, value));
            
            // Keep history limited
            if var.history.len() > 100 {
                var.history.remove(0);
            }
        } else {
            // Create new variable
            self.variables.insert(
                name.to_string(),
                StateVariable {
                    name: name.to_string(),
                    value: value.clone(),
                    rate_of_change: 0.0,
                    history: vec![(now, value)],
                },
            );
        }
        
        self.last_updated = now;
    }
    
    /// Calculate overall system health
    pub fn health(&self) -> f64 {
        (self.stability + self.energy) / 2.0
    }
    
    /// Check if system is in a healthy state
    pub fn is_healthy(&self) -> bool {
        matches!(self.phase, SystemPhase::Stable | SystemPhase::Adapting) 
            && self.stability > 0.3 
            && self.energy > 0.2
    }
}

/// Thread-safe state container
pub type SharedSystemState = Arc<RwLock<SystemState>>;

/// Create a new shared system state
pub fn shared_state() -> SharedSystemState {
    Arc::new(RwLock::new(SystemState::new()))
}