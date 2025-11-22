//! System boundary module for autopoietic systems

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use async_trait::async_trait;
use crate::Result;

/// The boundary of an autopoietic system that regulates interactions
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SystemBoundary {
    /// Permeability level (0.0 = closed, 1.0 = fully open)
    pub permeability: f64,
    
    /// Active filters for incoming signals
    pub input_filters: HashMap<String, Filter>,
    
    /// Active filters for outgoing signals
    pub output_filters: HashMap<String, Filter>,
    
    /// Boundary integrity (0.0 = compromised, 1.0 = intact)
    pub integrity: f64,
    
    /// Adaptive threshold for automatic regulation
    pub adaptive_threshold: f64,
    
    /// History of boundary crossings
    pub crossing_history: Vec<BoundaryCrossing>,
}

/// Filter for boundary regulation
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Filter {
    /// Filter name
    pub name: String,
    
    /// Filter type
    pub filter_type: FilterType,
    
    /// Filter strength (0.0 = no filtering, 1.0 = complete blocking)
    pub strength: f64,
    
    /// Is the filter active?
    pub active: bool,
    
    /// Adaptive parameters
    pub parameters: HashMap<String, f64>,
}

/// Types of filters
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum FilterType {
    /// Pass through only specific types
    Whitelist,
    
    /// Block specific types
    Blacklist,
    
    /// Transform signals
    Transform,
    
    /// Rate limiting
    RateLimit,
    
    /// Threshold-based
    Threshold,
    
    /// Pattern matching
    Pattern,
}

/// Record of a boundary crossing event
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoundaryCrossing {
    /// Timestamp of the crossing
    pub timestamp: chrono::DateTime<chrono::Utc>,
    
    /// Direction of crossing
    pub direction: CrossingDirection,
    
    /// Type of signal/entity that crossed
    pub signal_type: String,
    
    /// Was the crossing allowed?
    pub allowed: bool,
    
    /// Filters that were applied
    pub applied_filters: Vec<String>,
}

/// Direction of boundary crossing
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum CrossingDirection {
    /// Incoming to the system
    Inbound,
    
    /// Outgoing from the system
    Outbound,
}

impl Default for SystemBoundary {
    fn default() -> Self {
        Self {
            permeability: 0.5,
            input_filters: HashMap::new(),
            output_filters: HashMap::new(),
            integrity: 1.0,
            adaptive_threshold: 0.5,
            crossing_history: Vec::new(),
        }
    }
}

impl SystemBoundary {
    /// Create a new system boundary
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Add an input filter
    pub fn add_input_filter(&mut self, filter: Filter) {
        self.input_filters.insert(filter.name.clone(), filter);
    }
    
    /// Add an output filter
    pub fn add_output_filter(&mut self, filter: Filter) {
        self.output_filters.insert(filter.name.clone(), filter);
    }
    
    /// Check if a signal can pass through the boundary (inbound)
    pub fn can_enter(&self, signal_type: &str, signal_strength: f64) -> bool {
        // Check permeability first
        if self.permeability == 0.0 {
            return false;
        }
        
        // Check integrity
        if self.integrity < 0.1 {
            return true; // Compromised boundary lets everything through
        }
        
        // Apply filters
        let mut allowed = true;
        let mut filter_strength_sum = 0.0;
        
        for filter in self.input_filters.values() {
            if !filter.active {
                continue;
            }
            
            match filter.filter_type {
                FilterType::Blacklist => {
                    if signal_type == filter.name {
                        allowed = false;
                        break;
                    }
                }
                FilterType::Threshold => {
                    if let Some(&threshold) = filter.parameters.get("threshold") {
                        if signal_strength < threshold {
                            allowed = false;
                            break;
                        }
                    }
                }
                _ => {
                    filter_strength_sum += filter.strength;
                }
            }
        }
        
        // Consider overall filtering effect
        if filter_strength_sum >= 1.0 {
            allowed = false;
        }
        
        allowed && signal_strength > self.adaptive_threshold * (1.0 - self.permeability)
    }
    
    /// Check if a signal can pass through the boundary (outbound)
    pub fn can_exit(&self, signal_type: &str, signal_strength: f64) -> bool {
        // Always allow exit if boundary is compromised
        if self.integrity < 0.1 {
            return true;
        }
        
        // Apply output filters
        for filter in self.output_filters.values() {
            if !filter.active {
                continue;
            }
            
            match filter.filter_type {
                FilterType::Blacklist => {
                    if signal_type == filter.name {
                        return false;
                    }
                }
                FilterType::RateLimit => {
                    // Simple rate limit check (could be more sophisticated)
                    let recent_count = self.crossing_history.iter()
                        .filter(|c| {
                            matches!(c.direction, CrossingDirection::Outbound) &&
                            c.signal_type == signal_type &&
                            c.timestamp > chrono::Utc::now() - chrono::Duration::minutes(1)
                        })
                        .count();
                    
                    if let Some(&limit) = filter.parameters.get("limit") {
                        if recent_count >= limit as usize {
                            return false;
                        }
                    }
                }
                _ => {}
            }
        }
        
        true
    }
    
    /// Record a boundary crossing
    pub fn record_crossing(&mut self, direction: CrossingDirection, signal_type: String, allowed: bool) {
        let crossing = BoundaryCrossing {
            timestamp: chrono::Utc::now(),
            direction,
            signal_type,
            allowed,
            applied_filters: match &direction {
                CrossingDirection::Inbound => self.input_filters.keys().cloned().collect(),
                CrossingDirection::Outbound => self.output_filters.keys().cloned().collect(),
            },
        };
        
        self.crossing_history.push(crossing);
        
        // Keep history limited
        if self.crossing_history.len() > 1000 {
            self.crossing_history.remove(0);
        }
    }
    
    /// Adapt boundary based on recent activity
    pub fn adapt(&mut self) {
        // Calculate crossing rate
        let recent_crossings = self.crossing_history.iter()
            .filter(|c| c.timestamp > chrono::Utc::now() - chrono::Duration::minutes(5))
            .count();
        
        let crossing_rate = recent_crossings as f64 / 5.0; // per minute
        
        // Adjust permeability based on activity
        if crossing_rate > 10.0 {
            // Too much activity, reduce permeability
            self.permeability = (self.permeability * 0.9).max(0.1);
        } else if crossing_rate < 1.0 {
            // Too little activity, increase permeability
            self.permeability = (self.permeability * 1.1).min(0.9);
        }
        
        // Adjust adaptive threshold
        let denied_ratio = self.crossing_history.iter()
            .filter(|c| !c.allowed)
            .count() as f64 / self.crossing_history.len().max(1) as f64;
        
        if denied_ratio > 0.5 {
            // Too restrictive, lower threshold
            self.adaptive_threshold *= 0.95;
        } else if denied_ratio < 0.1 {
            // Too permissive, raise threshold
            self.adaptive_threshold *= 1.05;
        }
        
        self.adaptive_threshold = self.adaptive_threshold.clamp(0.1, 0.9);
    }
}

/// Trait for boundary management
#[async_trait]
pub trait BoundaryManager: Send + Sync {
    /// Process an incoming signal at the boundary
    async fn process_inbound(&mut self, signal: &dyn std::any::Any) -> Result<bool>;
    
    /// Process an outgoing signal at the boundary
    async fn process_outbound(&mut self, signal: &dyn std::any::Any) -> Result<bool>;
    
    /// Get current boundary state
    fn boundary_state(&self) -> &SystemBoundary;
    
    /// Update boundary configuration
    async fn update_boundary(&mut self, update: BoundaryUpdate) -> Result<()>;
}

/// Update instruction for boundary
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct BoundaryUpdate {
    /// New permeability level
    pub permeability: Option<f64>,
    
    /// Filters to add
    pub add_filters: Vec<Filter>,
    
    /// Filters to remove
    pub remove_filters: Vec<String>,
    
    /// New adaptive threshold
    pub adaptive_threshold: Option<f64>,
}