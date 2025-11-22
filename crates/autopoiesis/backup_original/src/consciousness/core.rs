/// Core Consciousness State Implementation
/// 
/// This module provides the fundamental consciousness state representation
/// used throughout the system for real-time market analysis and predictions.

use serde::{Deserialize, Serialize};
use std::time::{SystemTime, UNIX_EPOCH};

/// Primary consciousness state containing all system information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessState {
    /// Level of consciousness coherence (0.0 to 1.0)
    pub coherence_level: f64,
    
    /// Field coherence strength (0.0 to 1.0)
    pub field_coherence: f64,
    
    /// Temporal consistency of the consciousness state
    pub temporal_consistency: f64,
    
    /// Attention focus level
    pub attention_level: f64,
    
    /// System awareness factor
    pub awareness_factor: f64,
    
    /// Current processing timestamp
    pub timestamp: f64,
    
    /// Additional metadata for analysis
    pub metadata: ConsciousnessMetadata,
}

/// Metadata associated with consciousness state
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsciousnessMetadata {
    /// Source of the consciousness data
    pub source: String,
    
    /// Processing context identifier
    pub context_id: Option<String>,
    
    /// Quality assessment score
    pub quality_score: f64,
    
    /// Processing duration in milliseconds
    pub processing_duration_ms: Option<f64>,
}

impl ConsciousnessState {
    /// Create a new consciousness state with default values
    pub fn new() -> Self {
        Self {
            coherence_level: 0.5,
            field_coherence: 0.5,
            temporal_consistency: 0.5,
            attention_level: 0.5,
            awareness_factor: 0.5,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
            metadata: ConsciousnessMetadata::default(),
        }
    }
    
    /// Create consciousness state from market data analysis
    pub fn from_market_data(data_points: &[f64]) -> Self {
        if data_points.is_empty() {
            return Self::new();
        }
        
        // Calculate coherence based on data stability
        let mean = data_points.iter().sum::<f64>() / data_points.len() as f64;
        let variance = data_points.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / data_points.len() as f64;
        
        let coherence_level = (1.0 / (1.0 + variance)).clamp(0.0, 1.0);
        
        // Calculate field coherence based on trend consistency
        let trend_consistency = if data_points.len() > 1 {
            let differences: Vec<f64> = data_points.windows(2)
                .map(|w| w[1] - w[0])
                .collect();
            
            let trend_mean = differences.iter().sum::<f64>() / differences.len() as f64;
            let trend_variance = differences.iter()
                .map(|x| (x - trend_mean).powi(2))
                .sum::<f64>() / differences.len() as f64;
            
            (1.0 / (1.0 + trend_variance)).clamp(0.0, 1.0)
        } else {
            0.5
        };
        
        Self {
            coherence_level,
            field_coherence: trend_consistency,
            temporal_consistency: coherence_level * trend_consistency,
            attention_level: coherence_level,
            awareness_factor: (coherence_level + trend_consistency) / 2.0,
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
            metadata: ConsciousnessMetadata {
                source: "market_data".to_string(),
                context_id: None,
                quality_score: coherence_level,
                processing_duration_ms: None,
            },
        }
    }
    
    /// Update consciousness state with new market information
    pub fn update_with_market_info(&mut self, new_data: &[f64], learning_rate: f64) {
        if new_data.is_empty() {
            return;
        }
        
        let new_state = Self::from_market_data(new_data);
        
        // Exponentially weighted moving average update
        self.coherence_level = self.coherence_level * (1.0 - learning_rate) + 
                               new_state.coherence_level * learning_rate;
        
        self.field_coherence = self.field_coherence * (1.0 - learning_rate) + 
                               new_state.field_coherence * learning_rate;
        
        self.temporal_consistency = self.temporal_consistency * (1.0 - learning_rate) + 
                                    new_state.temporal_consistency * learning_rate;
        
        self.attention_level = self.attention_level * (1.0 - learning_rate) + 
                               new_state.attention_level * learning_rate;
        
        self.awareness_factor = self.awareness_factor * (1.0 - learning_rate) + 
                                new_state.awareness_factor * learning_rate;
        
        self.timestamp = new_state.timestamp;
        self.metadata.quality_score = self.coherence_level;
    }
    
    /// Get overall consciousness quality score
    pub fn quality_score(&self) -> f64 {
        (self.coherence_level + self.field_coherence + self.temporal_consistency + 
         self.attention_level + self.awareness_factor) / 5.0
    }
    
    /// Check if consciousness state is stable
    pub fn is_stable(&self) -> bool {
        self.coherence_level > 0.6 && 
        self.field_coherence > 0.6 && 
        self.temporal_consistency > 0.5
    }
    
    /// Get consciousness state as feature vector for ML
    pub fn to_feature_vector(&self) -> Vec<f64> {
        vec![
            self.coherence_level,
            self.field_coherence,
            self.temporal_consistency,
            self.attention_level,
            self.awareness_factor,
            self.quality_score(),
        ]
    }
    
    /// Create consciousness state from feature vector
    pub fn from_feature_vector(features: &[f64]) -> Result<Self, String> {
        if features.len() < 5 {
            return Err("Feature vector must contain at least 5 elements".to_string());
        }
        
        Ok(Self {
            coherence_level: features[0].clamp(0.0, 1.0),
            field_coherence: features[1].clamp(0.0, 1.0),
            temporal_consistency: features[2].clamp(0.0, 1.0),
            attention_level: features[3].clamp(0.0, 1.0),
            awareness_factor: features[4].clamp(0.0, 1.0),
            timestamp: SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs_f64(),
            metadata: ConsciousnessMetadata::default(),
        })
    }
    
    /// Validate consciousness state values
    pub fn validate(&self) -> Result<(), String> {
        let fields = [
            ("coherence_level", self.coherence_level),
            ("field_coherence", self.field_coherence),
            ("temporal_consistency", self.temporal_consistency),
            ("attention_level", self.attention_level),
            ("awareness_factor", self.awareness_factor),
        ];
        
        for (name, value) in fields {
            if !(0.0..=1.0).contains(&value) {
                return Err(format!("{} must be between 0.0 and 1.0, got {}", name, value));
            }
        }
        
        if self.timestamp < 0.0 {
            return Err("Timestamp must be non-negative".to_string());
        }
        
        Ok(())
    }
}

impl Default for ConsciousnessState {
    fn default() -> Self {
        Self::new()
    }
}

impl ConsciousnessMetadata {
    /// Create new metadata with default values
    pub fn new() -> Self {
        Self {
            source: "unknown".to_string(),
            context_id: None,
            quality_score: 0.5,
            processing_duration_ms: None,
        }
    }
    
    /// Set processing context
    pub fn with_context(mut self, context_id: String) -> Self {
        self.context_id = Some(context_id);
        self
    }
    
    /// Set source information
    pub fn with_source(mut self, source: String) -> Self {
        self.source = source;
        self
    }
    
    /// Set quality score
    pub fn with_quality_score(mut self, score: f64) -> Self {
        self.quality_score = score.clamp(0.0, 1.0);
        self
    }
}

impl Default for ConsciousnessMetadata {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_consciousness_state_creation() {
        let state = ConsciousnessState::new();
        assert!(state.validate().is_ok());
        assert_eq!(state.coherence_level, 0.5);
        assert_eq!(state.field_coherence, 0.5);
    }
    
    #[test]
    fn test_from_market_data() {
        let data = vec![1.0, 1.1, 1.05, 1.2, 1.15];
        let state = ConsciousnessState::from_market_data(&data);
        
        assert!(state.validate().is_ok());
        assert!(state.coherence_level > 0.0);
        assert!(state.field_coherence > 0.0);
        assert_eq!(state.metadata.source, "market_data");
    }
    
    #[test]
    fn test_feature_vector_conversion() {
        let state = ConsciousnessState::new();
        let features = state.to_feature_vector();
        assert_eq!(features.len(), 6);
        
        let restored = ConsciousnessState::from_feature_vector(&features[..5]).unwrap();
        assert!((restored.coherence_level - state.coherence_level).abs() < 1e-10);
    }
    
    #[test]
    fn test_update_with_market_info() {
        let mut state = ConsciousnessState::new();
        let initial_coherence = state.coherence_level;
        
        let new_data = vec![2.0, 2.1, 2.05];
        state.update_with_market_info(&new_data, 0.1);
        
        // Should have changed due to learning
        assert_ne!(state.coherence_level, initial_coherence);
        assert!(state.validate().is_ok());
    }
}