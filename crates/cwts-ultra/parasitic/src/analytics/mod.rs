//! Analytics Module
//!
//! Comprehensive analytics system for parasitic pairlist with sub-millisecond
//! performance tracking, organism metrics, system health monitoring, and CQGS compliance.

pub mod compliance;
pub mod dashboard;
pub mod emergence;
pub mod health;
pub mod metrics;
pub mod performance;
pub mod tests;

pub use performance::PerformanceAnalytics;

use crate::organisms::{ParasiticOrganism, ResourceMetrics};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Core performance metric for organisms
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetric {
    pub id: Uuid,
    pub organism_id: Uuid,
    pub organism_type: String,
    pub timestamp: DateTime<Utc>,
    pub latency_ns: u64,
    pub throughput: f64,
    pub success_rate: f64,
    pub resource_usage: ResourceMetrics,
    pub profit: f64,
    pub trades_executed: u64,
}

/// Organism performance data input
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganismPerformanceData {
    pub organism_id: Uuid,
    pub organism_type: String,
    pub timestamp: DateTime<Utc>,
    pub latency_ns: u64,
    pub throughput: f64,
    pub success_rate: f64,
    pub resource_usage: ResourceMetrics,
    pub profit: f64,
    pub trades_executed: u64,
}

/// Metric aggregation time periods
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum MetricAggregation {
    Last1Minute,
    Last5Minutes,
    Last1Hour,
}

/// System health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthStatus {
    pub overall_health: f64, // 0.0 to 1.0
    pub component_health: std::collections::HashMap<String, f64>,
    pub active_alerts: usize,
    pub performance_score: f64,
    pub resource_utilization: f64,
    pub timestamp: DateTime<Utc>,
}

/// Organism analytics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrganismAnalyticsSummary {
    pub organism_id: Uuid,
    pub organism_type: String,
    pub performance_score: f64,
    pub total_trades: u64,
    pub total_profit: f64,
    pub average_latency_ns: u64,
    pub success_rate: f64,
    pub resource_efficiency: f64,
    pub last_active: DateTime<Utc>,
}

/// Analytics error types
#[derive(Debug, thiserror::Error)]
pub enum AnalyticsError {
    #[error("Performance tracking error: {0}")]
    Performance(String),

    #[error("Metric aggregation error: {0}")]
    Aggregation(String),

    #[error("System health check failed: {0}")]
    HealthCheck(String),

    #[error("Organism analytics error: {0}")]
    OrganismAnalytics(String),

    #[error("CQGS compliance violation: {0}")]
    ComplianceViolation(String),

    #[error("Data integrity error: {0}")]
    DataIntegrity(String),
}

/// Main analytics engine coordinating all analytics modules
pub struct AnalyticsEngine {
    performance: PerformanceAnalytics,
    organism_metrics: metrics::OrganismMetrics,
    system_health: health::SystemHealthMonitor,
    emergence_analytics: emergence::EmergenceAnalytics,
    dashboard_generator: dashboard::DashboardDataGenerator,
    compliance_tracker: compliance::CqgsComplianceTracker,
}

impl AnalyticsEngine {
    /// Create new analytics engine
    pub async fn new() -> Result<Self, AnalyticsError> {
        Ok(Self {
            performance: PerformanceAnalytics::new(),
            organism_metrics: metrics::OrganismMetrics::new()
                .await
                .map_err(|e| AnalyticsError::OrganismAnalytics(e.to_string()))?,
            system_health: health::SystemHealthMonitor::new()
                .await
                .map_err(|e| AnalyticsError::HealthCheck(e.to_string()))?,
            emergence_analytics: emergence::EmergenceAnalytics::new()
                .await
                .map_err(|e| AnalyticsError::Performance(e.to_string()))?,
            dashboard_generator: dashboard::DashboardDataGenerator::new()
                .await
                .map_err(|e| AnalyticsError::Performance(e.to_string()))?,
            compliance_tracker: compliance::CqgsComplianceTracker::new()
                .await
                .map_err(|e| AnalyticsError::ComplianceViolation(e.to_string()))?,
        })
    }

    /// Record organism performance data across all analytics modules
    pub async fn record_organism_performance(
        &mut self,
        data: OrganismPerformanceData,
    ) -> Result<(), AnalyticsError> {
        // Performance analytics
        self.performance
            .record_metric(data.clone())
            .await
            .map_err(|e| AnalyticsError::Performance(e.to_string()))?;

        // Organism metrics
        self.organism_metrics
            .update_metrics(&data)
            .await
            .map_err(|e| AnalyticsError::OrganismAnalytics(e.to_string()))?;

        // System health monitoring
        self.system_health
            .record_performance_data(&data)
            .await
            .map_err(|e| AnalyticsError::HealthCheck(e.to_string()))?;

        // Emergence pattern detection
        self.emergence_analytics
            .analyze_patterns(&data)
            .await
            .map_err(|e| AnalyticsError::Performance(e.to_string()))?;

        // Update dashboard data
        self.dashboard_generator
            .update_metrics(&data)
            .await
            .map_err(|e| AnalyticsError::Performance(e.to_string()))?;

        // CQGS compliance tracking
        self.compliance_tracker
            .validate_performance(&data)
            .await
            .map_err(|e| AnalyticsError::ComplianceViolation(e.to_string()))?;

        Ok(())
    }

    /// Get comprehensive analytics summary
    pub async fn get_analytics_summary(&self) -> Result<AnalyticsSummary, AnalyticsError> {
        let performance_stats = self
            .performance
            .aggregate_metrics(MetricAggregation::Last5Minutes)
            .await
            .map_err(|e| AnalyticsError::Aggregation(e.to_string()))?;

        let organism_summaries = self
            .organism_metrics
            .get_all_organism_summaries()
            .await
            .map_err(|e| AnalyticsError::OrganismAnalytics(e.to_string()))?;

        let system_health = self
            .system_health
            .get_current_health()
            .await
            .map_err(|e| AnalyticsError::HealthCheck(e.to_string()))?;

        let emergence_patterns = self.emergence_analytics.get_detected_patterns().await;

        let compliance_status = self
            .compliance_tracker
            .get_compliance_status()
            .await
            .map_err(|e| AnalyticsError::ComplianceViolation(e.to_string()))?;

        Ok(AnalyticsSummary {
            performance_stats,
            organism_summaries,
            system_health,
            emergence_patterns,
            compliance_status,
            timestamp: Utc::now(),
        })
    }

    /// Start real-time analytics monitoring
    pub async fn start_monitoring(&mut self) -> Result<(), AnalyticsError> {
        self.performance
            .start_real_time_monitoring()
            .await
            .map_err(|e| AnalyticsError::Performance(e.to_string()))?;

        self.system_health
            .start_monitoring()
            .await
            .map_err(|e| AnalyticsError::HealthCheck(e.to_string()))?;

        self.emergence_analytics
            .start_pattern_detection()
            .await
            .map_err(|e| AnalyticsError::Performance(e.to_string()))?;

        self.dashboard_generator
            .start_real_time_updates()
            .await
            .map_err(|e| AnalyticsError::Performance(e.to_string()))?;

        self.compliance_tracker
            .start_compliance_monitoring()
            .await
            .map_err(|e| AnalyticsError::ComplianceViolation(e.to_string()))?;

        Ok(())
    }

    /// Get performance analytics reference
    pub fn performance(&self) -> &PerformanceAnalytics {
        &self.performance
    }

    /// Get organism metrics reference
    pub fn organism_metrics(&self) -> &metrics::OrganismMetrics {
        &self.organism_metrics
    }

    /// Get system health monitor reference
    pub fn system_health(&self) -> &health::SystemHealthMonitor {
        &self.system_health
    }
}

/// Comprehensive analytics summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnalyticsSummary {
    pub performance_stats: performance::PerformanceStats,
    pub organism_summaries: Vec<OrganismAnalyticsSummary>,
    pub system_health: SystemHealthStatus,
    pub emergence_patterns: Vec<emergence::EmergencePattern>,
    pub compliance_status: compliance::CqgsComplianceStatus,
    pub timestamp: DateTime<Utc>,
}
