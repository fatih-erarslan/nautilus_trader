//! Quantum Security Metrics Module

use crate::error::QuantumSecurityError;
use crate::types::*;

/// Security Metrics Manager
#[derive(Debug, Clone)]
pub struct SecurityMetricsManager {
    pub enabled: bool,
    pub total_operations: u64,
    pub error_count: u64,
    pub average_latency_us: f64,
    pub max_latency_us: u64,
    pub min_latency_us: u64,
    pub operations_by_type: std::collections::HashMap<String, u64>,
    pub performance_alerts: u64,
}

impl SecurityMetricsManager {
    pub fn new() -> Self {
        Self { 
            enabled: true,
            total_operations: 0,
            error_count: 0,
            average_latency_us: 0.0,
            max_latency_us: 0,
            min_latency_us: u64::MAX,
            operations_by_type: std::collections::HashMap::new(),
            performance_alerts: 0,
        }
    }
}

impl Default for SecurityMetricsManager {
    fn default() -> Self {
        Self::new()
    }
}