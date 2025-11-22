//! Common types and data structures for TENGRI Market Readiness Sentinel

use serde::{Deserialize, Serialize};
use uuid::Uuid;
use chrono::{DateTime, Utc};
use std::collections::HashMap;

/// Validation status enumeration
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValidationStatus {
    Passed,
    Warning,
    Failed,
    InProgress,
}

/// Individual validation result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ValidationResult {
    pub status: ValidationStatus,
    pub message: String,
    pub details: Option<serde_json::Value>,
    pub timestamp: DateTime<Utc>,
    pub duration_ms: u64,
}

impl ValidationResult {
    pub fn passed(message: String) -> Self {
        Self {
            status: ValidationStatus::Passed,
            message,
            details: None,
            timestamp: Utc::now(),
            duration_ms: 0,
        }
    }

    pub fn warning(message: String) -> Self {
        Self {
            status: ValidationStatus::Warning,
            message,
            details: None,
            timestamp: Utc::now(),
            duration_ms: 0,
        }
    }

    pub fn failed(message: String) -> Self {
        Self {
            status: ValidationStatus::Failed,
            message,
            details: None,
            timestamp: Utc::now(),
            duration_ms: 0,
        }
    }

    pub fn with_details(mut self, details: serde_json::Value) -> Self {
        self.details = Some(details);
        self
    }

    pub fn with_duration(mut self, duration_ms: u64) -> Self {
        self.duration_ms = duration_ms;
        self
    }
}

// Re-export common types from lib.rs
pub use crate::{
    MarketReadinessReport,
    MetricsReport,
    HealthReport,
    PerformanceMetrics,
    SystemMetrics,
    TradingMetrics,
    RiskMetrics,
    HealthStatus,
    Alert,
    AlertSeverity,
};