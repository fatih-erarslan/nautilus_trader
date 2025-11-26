// Health checking for distributed nodes

use crate::federation::AgentId;
use crate::{DistributedError, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;
use tokio::time::{interval, Duration};

/// Health status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HealthStatus {
    /// Node is healthy
    Healthy,

    /// Node is degraded (partial functionality)
    Degraded,

    /// Node is unhealthy
    Unhealthy,

    /// Node status unknown
    Unknown,
}

/// Health report for a node
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthReport {
    /// Node/agent ID
    pub node_id: AgentId,

    /// Health status
    pub status: HealthStatus,

    /// Last check timestamp
    pub last_check: chrono::DateTime<chrono::Utc>,

    /// Response time (ms)
    pub response_time_ms: u64,

    /// Error message (if unhealthy)
    pub error: Option<String>,

    /// Health metrics
    pub metrics: HealthMetrics,
}

/// Health metrics
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct HealthMetrics {
    /// CPU usage (0.0-1.0)
    pub cpu_usage: f64,

    /// Memory usage (0.0-1.0)
    pub memory_usage: f64,

    /// Disk usage (0.0-1.0)
    pub disk_usage: f64,

    /// Network latency (ms)
    pub network_latency_ms: u64,

    /// Uptime (seconds)
    pub uptime_seconds: u64,

    /// Total requests
    pub total_requests: u64,

    /// Failed requests
    pub failed_requests: u64,
}

impl HealthMetrics {
    /// Calculate health score (0.0-1.0, higher is better)
    pub fn health_score(&self) -> f64 {
        let cpu_score = 1.0 - self.cpu_usage;
        let memory_score = 1.0 - self.memory_usage;
        let disk_score = 1.0 - self.disk_usage;
        let latency_score = 1.0 - (self.network_latency_ms as f64 / 1000.0).min(1.0);
        let error_score = if self.total_requests > 0 {
            1.0 - (self.failed_requests as f64 / self.total_requests as f64)
        } else {
            1.0
        };

        (cpu_score + memory_score + disk_score + latency_score + error_score) / 5.0
    }
}

/// Health checker
pub struct HealthChecker {
    /// Health reports
    reports: Arc<RwLock<HashMap<AgentId, HealthReport>>>,

    /// Check interval (seconds)
    check_interval: u64,

    /// Health thresholds
    thresholds: HealthThresholds,
}

/// Health thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthThresholds {
    /// CPU threshold for degraded status
    pub cpu_degraded: f64,

    /// CPU threshold for unhealthy status
    pub cpu_unhealthy: f64,

    /// Memory threshold for degraded status
    pub memory_degraded: f64,

    /// Memory threshold for unhealthy status
    pub memory_unhealthy: f64,

    /// Response time threshold for degraded (ms)
    pub response_time_degraded_ms: u64,

    /// Response time threshold for unhealthy (ms)
    pub response_time_unhealthy_ms: u64,

    /// Error rate threshold for degraded (0.0-1.0)
    pub error_rate_degraded: f64,

    /// Error rate threshold for unhealthy (0.0-1.0)
    pub error_rate_unhealthy: f64,
}

impl Default for HealthThresholds {
    fn default() -> Self {
        Self {
            cpu_degraded: 0.7,
            cpu_unhealthy: 0.9,
            memory_degraded: 0.7,
            memory_unhealthy: 0.9,
            response_time_degraded_ms: 500,
            response_time_unhealthy_ms: 2000,
            error_rate_degraded: 0.05,
            error_rate_unhealthy: 0.15,
        }
    }
}

impl HealthChecker {
    /// Create new health checker
    pub fn new(check_interval: u64, thresholds: HealthThresholds) -> Self {
        Self {
            reports: Arc::new(RwLock::new(HashMap::new())),
            check_interval,
            thresholds,
        }
    }

    /// Register node for health checking
    pub async fn register_node(&self, node_id: AgentId) {
        let report = HealthReport {
            node_id: node_id.clone(),
            status: HealthStatus::Unknown,
            last_check: chrono::Utc::now(),
            response_time_ms: 0,
            error: None,
            metrics: HealthMetrics::default(),
        };

        self.reports.write().await.insert(node_id, report);
    }

    /// Unregister node
    pub async fn unregister_node(&self, node_id: &AgentId) {
        self.reports.write().await.remove(node_id);
    }

    /// Perform health check on a node
    pub async fn check_node(&self, node_id: &AgentId, metrics: HealthMetrics) -> Result<HealthReport> {
        let start = std::time::Instant::now();

        // Determine health status based on metrics
        let status = self.determine_status(&metrics);

        let response_time_ms = start.elapsed().as_millis() as u64;

        let report = HealthReport {
            node_id: node_id.clone(),
            status,
            last_check: chrono::Utc::now(),
            response_time_ms,
            error: if status == HealthStatus::Unhealthy {
                Some("Node metrics exceed unhealthy thresholds".to_string())
            } else {
                None
            },
            metrics,
        };

        // Update stored report
        self.reports.write().await.insert(node_id.clone(), report.clone());

        Ok(report)
    }

    /// Determine health status from metrics
    fn determine_status(&self, metrics: &HealthMetrics) -> HealthStatus {
        let error_rate = if metrics.total_requests > 0 {
            metrics.failed_requests as f64 / metrics.total_requests as f64
        } else {
            0.0
        };

        // Check unhealthy conditions
        if metrics.cpu_usage >= self.thresholds.cpu_unhealthy
            || metrics.memory_usage >= self.thresholds.memory_unhealthy
            || error_rate >= self.thresholds.error_rate_unhealthy
        {
            return HealthStatus::Unhealthy;
        }

        // Check degraded conditions
        if metrics.cpu_usage >= self.thresholds.cpu_degraded
            || metrics.memory_usage >= self.thresholds.memory_degraded
            || error_rate >= self.thresholds.error_rate_degraded
        {
            return HealthStatus::Degraded;
        }

        HealthStatus::Healthy
    }

    /// Get health report for a node
    pub async fn get_report(&self, node_id: &AgentId) -> Option<HealthReport> {
        self.reports.read().await.get(node_id).cloned()
    }

    /// Get all health reports
    pub async fn get_all_reports(&self) -> Vec<HealthReport> {
        self.reports.read().await.values().cloned().collect()
    }

    /// Get healthy nodes
    pub async fn get_healthy_nodes(&self) -> Vec<AgentId> {
        self.reports
            .read()
            .await
            .iter()
            .filter(|(_, report)| report.status == HealthStatus::Healthy)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Get unhealthy nodes
    pub async fn get_unhealthy_nodes(&self) -> Vec<AgentId> {
        self.reports
            .read()
            .await
            .iter()
            .filter(|(_, report)| report.status == HealthStatus::Unhealthy)
            .map(|(id, _)| id.clone())
            .collect()
    }

    /// Start periodic health checking
    pub async fn start_health_checking<F>(&self, metrics_provider: F)
    where
        F: Fn(&AgentId) -> HealthMetrics + Send + Sync + 'static,
    {
        let checker = Arc::new(self.clone_for_loop());
        let mut ticker = interval(Duration::from_secs(checker.check_interval));

        tokio::spawn(async move {
            loop {
                ticker.tick().await;

                let node_ids: Vec<_> = {
                    let reports = checker.reports.read().await;
                    reports.keys().cloned().collect()
                };

                for node_id in node_ids {
                    let metrics = metrics_provider(&node_id);
                    let _ = checker.check_node(&node_id, metrics).await;
                }
            }
        });
    }

    /// Get health statistics
    pub async fn stats(&self) -> HealthStats {
        let reports = self.reports.read().await;

        let total_nodes = reports.len();
        let healthy = reports
            .values()
            .filter(|r| r.status == HealthStatus::Healthy)
            .count();
        let degraded = reports
            .values()
            .filter(|r| r.status == HealthStatus::Degraded)
            .count();
        let unhealthy = reports
            .values()
            .filter(|r| r.status == HealthStatus::Unhealthy)
            .count();

        let avg_health_score = if !reports.is_empty() {
            reports.values().map(|r| r.metrics.health_score()).sum::<f64>() / reports.len() as f64
        } else {
            0.0
        };

        HealthStats {
            total_nodes,
            healthy_nodes: healthy,
            degraded_nodes: degraded,
            unhealthy_nodes: unhealthy,
            avg_health_score,
        }
    }

    /// Helper for cloning for background loop
    fn clone_for_loop(&self) -> Self {
        Self {
            reports: Arc::clone(&self.reports),
            check_interval: self.check_interval,
            thresholds: self.thresholds.clone(),
        }
    }
}

/// Health statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthStats {
    /// Total nodes monitored
    pub total_nodes: usize,

    /// Healthy nodes
    pub healthy_nodes: usize,

    /// Degraded nodes
    pub degraded_nodes: usize,

    /// Unhealthy nodes
    pub unhealthy_nodes: usize,

    /// Average health score
    pub avg_health_score: f64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_health_metrics_score() {
        let mut metrics = HealthMetrics::default();
        metrics.cpu_usage = 0.5;
        metrics.memory_usage = 0.5;

        let score = metrics.health_score();
        assert!(score > 0.0 && score <= 1.0);
    }

    #[tokio::test]
    async fn test_health_checker() {
        let checker = HealthChecker::new(10, HealthThresholds::default());

        checker.register_node("node-1".to_string()).await;

        let mut metrics = HealthMetrics::default();
        metrics.cpu_usage = 0.3;
        metrics.memory_usage = 0.4;

        let report = checker
            .check_node(&"node-1".to_string(), metrics)
            .await
            .unwrap();

        assert_eq!(report.status, HealthStatus::Healthy);
    }

    #[tokio::test]
    async fn test_unhealthy_detection() {
        let checker = HealthChecker::new(10, HealthThresholds::default());

        checker.register_node("node-1".to_string()).await;

        let mut metrics = HealthMetrics::default();
        metrics.cpu_usage = 0.95;

        let report = checker
            .check_node(&"node-1".to_string(), metrics)
            .await
            .unwrap();

        assert_eq!(report.status, HealthStatus::Unhealthy);
    }
}
