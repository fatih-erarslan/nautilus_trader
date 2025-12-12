//! Health Monitor Agent
//!
//! This module provides health monitoring capabilities for the MCP orchestration system.

use crate::error::McpError;
use crate::types::*;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use uuid::Uuid;

/// Health Monitor Agent for tracking system health
#[derive(Debug, Clone)]
pub struct HealthMonitor {
    pub id: Uuid,
    pub name: String,
    pub metrics: Arc<RwLock<HashMap<String, f64>>>,
    pub health_checks: Arc<RwLock<HashMap<String, HealthCheck>>>,
    pub enabled: bool,
}

/// Health Check Configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthCheck {
    pub name: String,
    pub endpoint: String,
    pub interval_seconds: u64,
    pub timeout_seconds: u64,
    pub expected_status: u16,
    pub last_check: Option<chrono::DateTime<chrono::Utc>>,
    pub status: HealthStatus,
}

/// Health Status
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub enum HealthStatus {
    Healthy,
    Degraded,
    Unhealthy,
    Unknown,
}

impl HealthMonitor {
    /// Create a new health monitor
    pub fn new(name: String) -> Self {
        Self {
            id: Uuid::new_v4(),
            name,
            metrics: Arc::new(RwLock::new(HashMap::new())),
            health_checks: Arc::new(RwLock::new(HashMap::new())),
            enabled: true,
        }
    }

    /// Add a health check
    pub async fn add_health_check(&self, check: HealthCheck) -> Result<(), McpError> {
        let mut checks = self.health_checks.write().await;
        checks.insert(check.name.clone(), check);
        Ok(())
    }

    /// Get health status
    pub async fn get_health_status(&self) -> Result<HealthStatus, McpError> {
        let checks = self.health_checks.read().await;
        
        if checks.is_empty() {
            return Ok(HealthStatus::Unknown);
        }

        let mut healthy_count = 0;
        let mut total_count = 0;

        for check in checks.values() {
            total_count += 1;
            if check.status == HealthStatus::Healthy {
                healthy_count += 1;
            }
        }

        let health_ratio = healthy_count as f64 / total_count as f64;
        
        if health_ratio == 1.0 {
            Ok(HealthStatus::Healthy)
        } else if health_ratio >= 0.7 {
            Ok(HealthStatus::Degraded)
        } else {
            Ok(HealthStatus::Unhealthy)
        }
    }

    /// Update metrics
    pub async fn update_metric(&self, name: String, value: f64) -> Result<(), McpError> {
        let mut metrics = self.metrics.write().await;
        metrics.insert(name, value);
        Ok(())
    }

    /// Get all metrics
    pub async fn get_metrics(&self) -> Result<HashMap<String, f64>, McpError> {
        let metrics = self.metrics.read().await;
        Ok(metrics.clone())
    }
}

impl Default for HealthMonitor {
    fn default() -> Self {
        Self::new("default-health-monitor".to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_health_monitor_creation() {
        let monitor = HealthMonitor::new("test".to_string());
        assert_eq!(monitor.name, "test");
        assert!(monitor.enabled);
    }

    #[tokio::test]
    async fn test_health_status_calculation() {
        let monitor = HealthMonitor::new("test".to_string());
        
        // No checks should return Unknown
        let status = monitor.get_health_status().await.unwrap();
        assert_eq!(status, HealthStatus::Unknown);

        // Add a healthy check
        let check = HealthCheck {
            name: "test-check".to_string(),
            endpoint: "http://localhost:8080/health".to_string(),
            interval_seconds: 30,
            timeout_seconds: 5,
            expected_status: 200,
            last_check: None,
            status: HealthStatus::Healthy,
        };

        monitor.add_health_check(check).await.unwrap();
        
        let status = monitor.get_health_status().await.unwrap();
        assert_eq!(status, HealthStatus::Healthy);
    }

    #[tokio::test]
    async fn test_metrics_update() {
        let monitor = HealthMonitor::new("test".to_string());
        
        monitor.update_metric("cpu_usage".to_string(), 75.5).await.unwrap();
        monitor.update_metric("memory_usage".to_string(), 60.2).await.unwrap();
        
        let metrics = monitor.get_metrics().await.unwrap();
        assert_eq!(metrics.get("cpu_usage"), Some(&75.5));
        assert_eq!(metrics.get("memory_usage"), Some(&60.2));
    }
}