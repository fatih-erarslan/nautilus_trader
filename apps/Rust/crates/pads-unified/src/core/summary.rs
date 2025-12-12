//! System summary functionality for PADS

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::SystemTime;

/// System summary information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemSummary {
    /// System version
    pub version: String,
    /// Number of decisions made
    pub decision_count: usize,
    /// Average decision latency
    pub avg_decision_latency: std::time::Duration,
    /// Number of agents
    pub agent_count: usize,
    /// Active features
    pub active_features: Vec<String>,
    /// System health score
    pub system_health: f64,
    /// Last update time
    pub last_update: SystemTime,
    /// Additional metrics
    pub metrics: HashMap<String, f64>,
}

impl Default for SystemSummary {
    fn default() -> Self {
        Self {
            version: env!("CARGO_PKG_VERSION").to_string(),
            decision_count: 0,
            avg_decision_latency: std::time::Duration::from_nanos(0),
            agent_count: 12,
            active_features: vec!["core".to_string()],
            system_health: 1.0,
            last_update: SystemTime::now(),
            metrics: HashMap::new(),
        }
    }
}

impl SystemSummary {
    /// Create new system summary
    pub fn new() -> Self {
        Self::default()
    }
    
    /// Update summary with new metrics
    pub fn update_metrics(&mut self, metrics: HashMap<String, f64>) {
        self.metrics = metrics;
        self.last_update = SystemTime::now();
    }
    
    /// Add feature to active features
    pub fn add_feature(&mut self, feature: String) {
        if !self.active_features.contains(&feature) {
            self.active_features.push(feature);
        }
    }
    
    /// Remove feature from active features
    pub fn remove_feature(&mut self, feature: &str) {
        self.active_features.retain(|f| f != feature);
    }
    
    /// Update system health
    pub fn update_health(&mut self, health: f64) {
        self.system_health = health.clamp(0.0, 1.0);
        self.last_update = SystemTime::now();
    }
    
    /// Get uptime
    pub fn uptime(&self) -> std::time::Duration {
        SystemTime::now().duration_since(self.last_update)
            .unwrap_or_else(|_| std::time::Duration::from_secs(0))
    }
    
    /// Check if system is healthy
    pub fn is_healthy(&self) -> bool {
        self.system_health > 0.8
    }
    
    /// Get metric value
    pub fn get_metric(&self, key: &str) -> Option<f64> {
        self.metrics.get(key).copied()
    }
    
    /// Set metric value
    pub fn set_metric(&mut self, key: String, value: f64) {
        self.metrics.insert(key, value);
        self.last_update = SystemTime::now();
    }
}