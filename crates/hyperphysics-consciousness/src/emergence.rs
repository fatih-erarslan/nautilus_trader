//! Emergence detection and classification for consciousness monitoring
//!
//! Provides types and utilities for tracking consciousness emergence events
//! and classifying emergence levels for real-time visualization.

use std::time::SystemTime;
use serde::{Serialize, Deserialize};

/// Emergence event representing a significant change in consciousness metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmergenceEvent {
    /// Timestamp when the emergence was detected
    pub timestamp: SystemTime,
    /// Φ value at the time of emergence
    pub phi_value: f64,
    /// Number of nodes in the system
    pub node_count: usize,
    /// Classification of emergence level
    pub emergence_level: EmergenceLevel,
    /// Strength of integration (0.0 to 1.0)
    pub integration_strength: f64,
    /// Duration of the emergence event
    pub duration_ms: u64,
    /// Spatial location of emergence (if applicable)
    pub spatial_center: Option<[f64; 3]>,
    /// Description of the emergence pattern
    pub description: String,
}

/// Classification of consciousness emergence levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EmergenceLevel {
    /// No significant emergence detected
    None,
    /// Weak emergence - local integration
    Weak,
    /// Moderate emergence - regional integration
    Moderate,
    /// Strong emergence - global integration
    Strong,
    /// Critical emergence - system-wide phase transition
    Critical,
}

/// Hierarchical consciousness analysis result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HierarchicalResult {
    /// Total system Φ
    pub total_phi: f64,
    /// Φ values at different hierarchical levels
    pub level_phi: Vec<f64>,
    /// Number of nodes at each level
    pub level_sizes: Vec<usize>,
    /// Integration strength between levels
    pub inter_level_integration: Vec<f64>,
    /// Detected emergence events
    pub emergence_events: Vec<EmergenceEvent>,
    /// Overall emergence classification
    pub overall_emergence: EmergenceLevel,
    /// Computation timestamp
    pub timestamp: SystemTime,
}

impl EmergenceLevel {
    /// Convert emergence level to numeric score (0.0 to 1.0)
    pub fn to_score(self) -> f64 {
        match self {
            EmergenceLevel::None => 0.0,
            EmergenceLevel::Weak => 0.2,
            EmergenceLevel::Moderate => 0.5,
            EmergenceLevel::Strong => 0.8,
            EmergenceLevel::Critical => 1.0,
        }
    }
    
    /// Create emergence level from numeric score
    pub fn from_score(score: f64) -> Self {
        if score < 0.1 {
            EmergenceLevel::None
        } else if score < 0.35 {
            EmergenceLevel::Weak
        } else if score < 0.65 {
            EmergenceLevel::Moderate
        } else if score < 0.9 {
            EmergenceLevel::Strong
        } else {
            EmergenceLevel::Critical
        }
    }
    
    /// Get color representation for visualization
    pub fn to_color_rgb(self) -> [f32; 3] {
        match self {
            EmergenceLevel::None => [0.2, 0.2, 0.2],      // Dark gray
            EmergenceLevel::Weak => [0.0, 0.5, 1.0],      // Blue
            EmergenceLevel::Moderate => [0.0, 1.0, 0.5],  // Green
            EmergenceLevel::Strong => [1.0, 0.8, 0.0],    // Orange
            EmergenceLevel::Critical => [1.0, 0.2, 0.2],  // Red
        }
    }
}

impl EmergenceEvent {
    /// Create a new emergence event
    pub fn new(
        phi_value: f64,
        node_count: usize,
        integration_strength: f64,
        description: String,
    ) -> Self {
        let emergence_level = EmergenceLevel::from_score(integration_strength);
        
        Self {
            timestamp: SystemTime::now(),
            phi_value,
            node_count,
            emergence_level,
            integration_strength,
            duration_ms: 0,
            spatial_center: None,
            description,
        }
    }
    
    /// Check if this is a significant emergence event
    pub fn is_significant(&self) -> bool {
        self.emergence_level != EmergenceLevel::None && 
        self.integration_strength > 0.3
    }
    
    /// Get age of the event in milliseconds
    pub fn age_ms(&self) -> u64 {
        self.timestamp
            .elapsed()
            .unwrap_or_default()
            .as_millis() as u64
    }
}

impl HierarchicalResult {
    /// Create a new hierarchical result
    pub fn new(
        total_phi: f64,
        level_phi: Vec<f64>,
        level_sizes: Vec<usize>,
    ) -> Self {
        let inter_level_integration = Self::calculate_inter_level_integration(&level_phi);
        let overall_emergence = Self::classify_overall_emergence(total_phi, &level_phi);
        
        Self {
            total_phi,
            level_phi,
            level_sizes,
            inter_level_integration,
            emergence_events: Vec::new(),
            overall_emergence,
            timestamp: SystemTime::now(),
        }
    }
    
    /// Calculate integration strength between hierarchical levels
    fn calculate_inter_level_integration(level_phi: &[f64]) -> Vec<f64> {
        if level_phi.len() < 2 {
            return Vec::new();
        }
        
        level_phi
            .windows(2)
            .map(|window| {
                let lower = window[0];
                let upper = window[1];
                if lower > 0.0 {
                    upper / lower
                } else {
                    0.0
                }
            })
            .collect()
    }
    
    /// Classify overall emergence level from hierarchical Φ values
    fn classify_overall_emergence(total_phi: f64, level_phi: &[f64]) -> EmergenceLevel {
        if level_phi.is_empty() {
            return EmergenceLevel::None;
        }
        
        // Calculate emergence score based on total Φ and level distribution
        let max_level_phi = level_phi.iter().fold(0.0f64, |a, &b| a.max(b));
        let phi_ratio = if max_level_phi > 0.0 {
            total_phi / max_level_phi
        } else {
            0.0
        };
        
        // Higher ratios indicate more integrated (emergent) systems
        let emergence_score = (phi_ratio - 1.0).max(0.0).min(1.0);
        
        EmergenceLevel::from_score(emergence_score)
    }
    
    /// Add an emergence event to this result
    pub fn add_emergence_event(&mut self, event: EmergenceEvent) {
        self.emergence_events.push(event);
    }
    
    /// Get the most recent significant emergence event
    pub fn latest_significant_event(&self) -> Option<&EmergenceEvent> {
        self.emergence_events
            .iter()
            .filter(|event| event.is_significant())
            .max_by_key(|event| event.timestamp)
    }
    
    /// Calculate total integration across all levels
    pub fn total_integration(&self) -> f64 {
        self.level_phi.iter().sum::<f64>()
    }
    
    /// Get the number of hierarchical levels
    pub fn level_count(&self) -> usize {
        self.level_phi.len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_emergence_level_scoring() {
        assert_eq!(EmergenceLevel::from_score(0.0), EmergenceLevel::None);
        assert_eq!(EmergenceLevel::from_score(0.3), EmergenceLevel::Weak);
        assert_eq!(EmergenceLevel::from_score(0.6), EmergenceLevel::Moderate);
        assert_eq!(EmergenceLevel::from_score(0.85), EmergenceLevel::Strong);
        assert_eq!(EmergenceLevel::from_score(1.0), EmergenceLevel::Critical);
    }

    #[test]
    fn test_emergence_event_creation() {
        let event = EmergenceEvent::new(
            0.5,
            1000,
            0.7,
            "Test emergence".to_string(),
        );
        
        assert_eq!(event.phi_value, 0.5);
        assert_eq!(event.node_count, 1000);
        assert_eq!(event.integration_strength, 0.7);
        assert_eq!(event.emergence_level, EmergenceLevel::Strong);
        assert!(event.is_significant());
    }

    #[test]
    fn test_hierarchical_result() {
        let level_phi = vec![0.1, 0.3, 0.5];
        let level_sizes = vec![100, 50, 25];
        
        let result = HierarchicalResult::new(0.9, level_phi, level_sizes);
        
        assert_eq!(result.total_phi, 0.9);
        assert_eq!(result.level_count(), 3);
        assert_eq!(result.total_integration(), 0.9);
        assert_eq!(result.inter_level_integration.len(), 2);
    }
}
