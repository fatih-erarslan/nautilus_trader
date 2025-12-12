//! # PADS Core Types
//!
//! Fundamental data structures and types for the Panarchy Adaptive Decision System.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use serde::{Deserialize, Serialize};
use ndarray::{Array1, Array2};

/// Decision vector representing a point in decision space
pub type DecisionVector = Array1<f64>;

/// Decision matrix for multi-criteria analysis
pub type DecisionMatrix = Array2<f64>;

/// Performance metrics collection
pub type PerformanceMetrics = HashMap<String, f64>;

/// Resource allocation map
pub type ResourceMap = HashMap<String, f64>;

/// Policy configuration map
pub type PolicyMap = HashMap<String, String>;

/// Agent capability set
pub type CapabilitySet = Vec<String>;

/// System state snapshot
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemState {
    /// Current decision layer being processed
    pub active_layer: crate::core::DecisionLayer,
    /// Current adaptive cycle phase
    pub cycle_phase: crate::core::AdaptiveCyclePhase,
    /// System performance metrics
    pub performance: PerformanceMetrics,
    /// Resource utilization
    pub resources: ResourceMap,
    /// Active policies
    pub policies: PolicyMap,
    /// System health status
    pub health: crate::core::SystemHealth,
    /// Timestamp of state capture
    pub timestamp: Instant,
    /// State version for consistency
    pub version: u64,
}

impl SystemState {
    /// Create a new system state
    pub fn new() -> Self {
        Self {
            active_layer: crate::core::DecisionLayer::Tactical,
            cycle_phase: crate::core::AdaptiveCyclePhase::Growth,
            performance: PerformanceMetrics::new(),
            resources: ResourceMap::new(),
            policies: PolicyMap::new(),
            health: crate::core::SystemHealth::Healthy,
            timestamp: Instant::now(),
            version: 1,
        }
    }
    
    /// Update system state with new metrics
    pub fn update(&mut self, metrics: PerformanceMetrics) {
        self.performance.extend(metrics);
        self.timestamp = Instant::now();
        self.version += 1;
    }
    
    /// Check if state is recent (within threshold)
    pub fn is_fresh(&self, threshold: Duration) -> bool {
        self.timestamp.elapsed() < threshold
    }
}

impl Default for SystemState {
    fn default() -> Self {
        Self::new()
    }
}

/// Decision alternative with evaluation criteria
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionAlternative {
    /// Unique identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Description of the alternative
    pub description: String,
    /// Estimated performance across criteria
    pub criteria_scores: HashMap<String, f64>,
    /// Resource requirements
    pub resource_requirements: ResourceMap,
    /// Risk assessment
    pub risk_level: f64,
    /// Implementation complexity (0.0 to 1.0)
    pub complexity: f64,
    /// Expected benefit (0.0 to 1.0)
    pub expected_benefit: f64,
    /// Time to implement
    pub implementation_time: Duration,
    /// Confidence in estimates (0.0 to 1.0)
    pub confidence: f64,
}

impl DecisionAlternative {
    /// Create a new decision alternative
    pub fn new(id: String, name: String, description: String) -> Self {
        Self {
            id,
            name,
            description,
            criteria_scores: HashMap::new(),
            resource_requirements: ResourceMap::new(),
            risk_level: 0.5,
            complexity: 0.5,
            expected_benefit: 0.5,
            implementation_time: Duration::from_secs(60),
            confidence: 0.8,
        }
    }
    
    /// Add a criterion score
    pub fn with_criterion(mut self, criterion: String, score: f64) -> Self {
        self.criteria_scores.insert(criterion, score.clamp(0.0, 1.0));
        self
    }
    
    /// Add resource requirement
    pub fn with_resource(mut self, resource: String, amount: f64) -> Self {
        self.resource_requirements.insert(resource, amount.max(0.0));
        self
    }
    
    /// Set risk level
    pub fn with_risk(mut self, risk: f64) -> Self {
        self.risk_level = risk.clamp(0.0, 1.0);
        self
    }
    
    /// Calculate overall score using weighted criteria
    pub fn calculate_score(&self, weights: &HashMap<String, f64>) -> f64 {
        let mut total_score = 0.0;
        let mut total_weight = 0.0;
        
        for (criterion, score) in &self.criteria_scores {
            if let Some(weight) = weights.get(criterion) {
                total_score += score * weight;
                total_weight += weight;
            }
        }
        
        if total_weight > 0.0 {
            total_score / total_weight
        } else {
            0.0
        }
    }
}

/// Decision criteria with weights and preferences
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DecisionCriteria {
    /// Criterion identifier
    pub id: String,
    /// Human-readable name
    pub name: String,
    /// Importance weight (0.0 to 1.0)
    pub weight: f64,
    /// Optimization direction (true = maximize, false = minimize)
    pub maximize: bool,
    /// Minimum acceptable value
    pub min_threshold: Option<f64>,
    /// Maximum acceptable value
    pub max_threshold: Option<f64>,
    /// Criterion category
    pub category: String,
}

impl DecisionCriteria {
    /// Create a new decision criterion
    pub fn new(id: String, name: String, weight: f64, maximize: bool) -> Self {
        Self {
            id,
            name,
            weight: weight.clamp(0.0, 1.0),
            maximize,
            min_threshold: None,
            max_threshold: None,
            category: "default".to_string(),
        }
    }
    
    /// Set minimum threshold
    pub fn with_min_threshold(mut self, threshold: f64) -> Self {
        self.min_threshold = Some(threshold);
        self
    }
    
    /// Set maximum threshold
    pub fn with_max_threshold(mut self, threshold: f64) -> Self {
        self.max_threshold = Some(threshold);
        self
    }
    
    /// Set category
    pub fn with_category(mut self, category: String) -> Self {
        self.category = category;
        self
    }
    
    /// Check if a value meets the criterion thresholds
    pub fn meets_threshold(&self, value: f64) -> bool {
        if let Some(min) = self.min_threshold {
            if value < min {
                return false;
            }
        }
        
        if let Some(max) = self.max_threshold {
            if value > max {
                return false;
            }
        }
        
        true
    }
    
    /// Normalize a value for this criterion
    pub fn normalize_value(&self, value: f64) -> f64 {
        if self.maximize {
            value
        } else {
            1.0 - value
        }
    }
}

/// Performance measurement point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformancePoint {
    /// Metric name
    pub metric: String,
    /// Measured value
    pub value: f64,
    /// Measurement timestamp
    pub timestamp: Instant,
    /// Measurement confidence (0.0 to 1.0)
    pub confidence: f64,
    /// Context metadata
    pub context: HashMap<String, String>,
}

impl PerformancePoint {
    /// Create a new performance point
    pub fn new(metric: String, value: f64) -> Self {
        Self {
            metric,
            value,
            timestamp: Instant::now(),
            confidence: 1.0,
            context: HashMap::new(),
        }
    }
    
    /// Add context metadata
    pub fn with_context(mut self, key: String, value: String) -> Self {
        self.context.insert(key, value);
        self
    }
    
    /// Set confidence level
    pub fn with_confidence(mut self, confidence: f64) -> Self {
        self.confidence = confidence.clamp(0.0, 1.0);
        self
    }
    
    /// Check if measurement is recent
    pub fn is_recent(&self, threshold: Duration) -> bool {
        self.timestamp.elapsed() < threshold
    }
}

/// Configuration for adaptive behavior
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AdaptiveConfig {
    /// Learning rate for adaptation (0.0 to 1.0)
    pub learning_rate: f64,
    /// Exploration vs exploitation balance (0.0 to 1.0)
    pub exploration_rate: f64,
    /// Memory decay factor (0.0 to 1.0)
    pub memory_decay: f64,
    /// Adaptation sensitivity (0.0 to 1.0)
    pub sensitivity: f64,
    /// Minimum confidence for decisions (0.0 to 1.0)
    pub min_confidence: f64,
    /// Maximum adaptation rate per cycle
    pub max_adaptation_rate: f64,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.1,
            exploration_rate: 0.2,
            memory_decay: 0.01,
            sensitivity: 0.5,
            min_confidence: 0.6,
            max_adaptation_rate: 0.3,
        }
    }
}

/// Event representing a system occurrence
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemEvent {
    /// Event identifier
    pub id: String,
    /// Event type
    pub event_type: String,
    /// Event severity level
    pub severity: EventSeverity,
    /// Event description
    pub description: String,
    /// Associated data
    pub data: HashMap<String, String>,
    /// Event timestamp
    pub timestamp: Instant,
    /// Source component
    pub source: String,
    /// Impact level (0.0 to 1.0)
    pub impact: f64,
}

/// Event severity levels
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EventSeverity {
    /// Informational events
    Info,
    /// Warning events
    Warning,
    /// Error events
    Error,
    /// Critical system events
    Critical,
}

impl EventSeverity {
    /// Get numeric severity level
    pub fn level(&self) -> u8 {
        match self {
            Self::Info => 1,
            Self::Warning => 2,
            Self::Error => 3,
            Self::Critical => 4,
        }
    }
    
    /// Check if severity requires immediate attention
    pub fn requires_attention(&self) -> bool {
        matches!(self, Self::Error | Self::Critical)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_system_state() {
        let mut state = SystemState::new();
        assert_eq!(state.version, 1);
        
        let mut metrics = PerformanceMetrics::new();
        metrics.insert("cpu_usage".to_string(), 0.75);
        state.update(metrics);
        
        assert_eq!(state.version, 2);
        assert_eq!(state.performance.get("cpu_usage"), Some(&0.75));
    }
    
    #[test]
    fn test_decision_alternative() {
        let alt = DecisionAlternative::new(
            "alt-001".to_string(),
            "Test Alternative".to_string(),
            "A test alternative".to_string(),
        )
        .with_criterion("performance".to_string(), 0.8)
        .with_criterion("cost".to_string(), 0.6)
        .with_resource("cpu".to_string(), 0.5)
        .with_risk(0.3);
        
        let mut weights = HashMap::new();
        weights.insert("performance".to_string(), 0.7);
        weights.insert("cost".to_string(), 0.3);
        
        let score = alt.calculate_score(&weights);
        assert!((score - 0.74).abs() < 0.01); // 0.8 * 0.7 + 0.6 * 0.3 = 0.74
    }
    
    #[test]
    fn test_decision_criteria() {
        let criteria = DecisionCriteria::new(
            "perf".to_string(),
            "Performance".to_string(),
            0.8,
            true,
        )
        .with_min_threshold(0.5)
        .with_max_threshold(1.0);
        
        assert!(criteria.meets_threshold(0.7));
        assert!(!criteria.meets_threshold(0.3));
        assert!(!criteria.meets_threshold(1.2));
    }
    
    #[test]
    fn test_performance_point() {
        let point = PerformancePoint::new("latency".to_string(), 150.0)
            .with_context("service".to_string(), "api".to_string())
            .with_confidence(0.95);
        
        assert_eq!(point.metric, "latency");
        assert_eq!(point.value, 150.0);
        assert_eq!(point.confidence, 0.95);
    }
    
    #[test]
    fn test_event_severity() {
        assert!(EventSeverity::Critical > EventSeverity::Error);
        assert!(EventSeverity::Error > EventSeverity::Warning);
        assert!(EventSeverity::Warning > EventSeverity::Info);
        
        assert!(EventSeverity::Critical.requires_attention());
        assert!(!EventSeverity::Info.requires_attention());
    }
}