//! Real-time monitoring system for market readiness validation

use std::sync::Arc;
use std::collections::HashMap;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn, error};
use chrono::{DateTime, Utc};
use uuid::Uuid;

use crate::config::MarketReadinessConfig;
use crate::error::MarketReadinessError;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MonitoringAlert {
    pub id: Uuid,
    pub timestamp: DateTime<Utc>,
    pub severity: AlertSeverity,
    pub category: AlertCategory,
    pub title: String,
    pub message: String,
    pub component: String,
    pub metric_name: String,
    pub current_value: f64,
    pub threshold_value: f64,
    pub resolved: bool,
    pub resolution_time: Option<DateTime<Utc>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Emergency,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertCategory {
    Performance,
    Availability,
    Security,
    DataQuality,
    Trading,
    Risk,
    Compliance,
    System,
}

#[derive(Debug)]
pub struct MonitoringSystem {
    config: Arc<MarketReadinessConfig>,
    active_alerts: Arc<RwLock<HashMap<Uuid, MonitoringAlert>>>,
    is_monitoring: Arc<RwLock<bool>>,
}

impl MonitoringSystem {
    pub async fn new(config: Arc<MarketReadinessConfig>) -> Result<Self> {
        Ok(Self {
            config,
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            is_monitoring: Arc::new(RwLock::new(false)),
        })
    }

    pub async fn initialize(&self) -> Result<()> {
        info!("Initializing monitoring system...");
        Ok(())
    }

    pub async fn start_monitoring(&self) -> Result<()> {
        {
            let mut is_monitoring = self.is_monitoring.write().await;
            if *is_monitoring {
                return Ok(());
            }
            *is_monitoring = true;
        }
        
        info!("Starting continuous monitoring...");
        Ok(())
    }

    pub async fn stop_monitoring(&self) -> Result<()> {
        let mut is_monitoring = self.is_monitoring.write().await;
        *is_monitoring = false;
        
        info!("Monitoring stopped");
        Ok(())
    }

    pub async fn get_active_alerts(&self) -> Vec<MonitoringAlert> {
        self.active_alerts.read().await.values().cloned().collect()
    }

    pub async fn create_alert(&self, 
                             severity: AlertSeverity,
                             category: AlertCategory,
                             title: String,
                             message: String,
                             component: String,
                             metric_name: String,
                             current_value: f64,
                             threshold_value: f64) -> Result<Uuid> {
        let alert = MonitoringAlert {
            id: Uuid::new_v4(),
            timestamp: Utc::now(),
            severity,
            category,
            title,
            message,
            component,
            metric_name,
            current_value,
            threshold_value,
            resolved: false,
            resolution_time: None,
        };

        let alert_id = alert.id;
        
        {
            let mut alerts = self.active_alerts.write().await;
            alerts.insert(alert_id, alert.clone());
        }

        // Log alert based on severity
        match alert.severity {
            AlertSeverity::Emergency => error!("🚨 EMERGENCY ALERT: {} - {}", alert.title, alert.message),
            AlertSeverity::Critical => error!("🔴 CRITICAL ALERT: {} - {}", alert.title, alert.message),
            AlertSeverity::Warning => warn!("🟡 WARNING ALERT: {} - {}", alert.title, alert.message),
            AlertSeverity::Info => info!("🔵 INFO ALERT: {} - {}", alert.title, alert.message),
        }

        Ok(alert_id)
    }

    pub async fn resolve_alert(&self, alert_id: Uuid) -> Result<()> {
        let mut alerts = self.active_alerts.write().await;
        
        if let Some(alert) = alerts.get_mut(&alert_id) {
            alert.resolved = true;
            alert.resolution_time = Some(Utc::now());
            info!("Alert resolved: {} - {}", alert.title, alert_id);
        }
        
        Ok(())
    }

    pub async fn check_alert_conditions(&self, metrics: &crate::metrics::MetricsSnapshot) -> Result<()> {
        // Check validation metrics
        if metrics.validation_metrics.validation_success_rate < 0.95 {
            self.create_alert(
                AlertSeverity::Warning,
                AlertCategory::Performance,
                "Low Validation Success Rate".to_string(),
                format!("Validation success rate is {:.2}%", metrics.validation_metrics.validation_success_rate * 100.0),
                "validation_engine".to_string(),
                "validation_success_rate".to_string(),
                metrics.validation_metrics.validation_success_rate,
                0.95
            ).await?;
        }

        // Check performance metrics
        if metrics.performance_metrics.cpu_usage_percent > 80.0 {
            self.create_alert(
                AlertSeverity::Warning,
                AlertCategory::Performance,
                "High CPU Usage".to_string(),
                format!("CPU usage is {:.1}%", metrics.performance_metrics.cpu_usage_percent),
                "system".to_string(),
                "cpu_usage_percent".to_string(),
                metrics.performance_metrics.cpu_usage_percent,
                80.0
            ).await?;
        }

        if metrics.performance_metrics.memory_usage_mb > 8000.0 {
            self.create_alert(
                AlertSeverity::Critical,
                AlertCategory::System,
                "High Memory Usage".to_string(),
                format!("Memory usage is {:.1} MB", metrics.performance_metrics.memory_usage_mb),
                "system".to_string(),
                "memory_usage_mb".to_string(),
                metrics.performance_metrics.memory_usage_mb,
                8000.0
            ).await?;
        }

        // Check trading metrics
        if metrics.trading_metrics.rejection_rate > 0.05 {
            self.create_alert(
                AlertSeverity::Warning,
                AlertCategory::Trading,
                "High Order Rejection Rate".to_string(),
                format!("Order rejection rate is {:.2}%", metrics.trading_metrics.rejection_rate * 100.0),
                "trading_system".to_string(),
                "rejection_rate".to_string(),
                metrics.trading_metrics.rejection_rate,
                0.05
            ).await?;
        }

        if metrics.trading_metrics.average_latency_ms > 100.0 {
            self.create_alert(
                AlertSeverity::Critical,
                AlertCategory::Performance,
                "High Trading Latency".to_string(),
                format!("Average trading latency is {:.1} ms", metrics.trading_metrics.average_latency_ms),
                "trading_system".to_string(),
                "average_latency_ms".to_string(),
                metrics.trading_metrics.average_latency_ms,
                100.0
            ).await?;
        }

        // Check risk metrics
        if metrics.risk_metrics.position_utilization > 0.9 {
            self.create_alert(
                AlertSeverity::Warning,
                AlertCategory::Risk,
                "High Position Utilization".to_string(),
                format!("Position utilization is {:.1}%", metrics.risk_metrics.position_utilization * 100.0),
                "risk_management".to_string(),
                "position_utilization".to_string(),
                metrics.risk_metrics.position_utilization,
                0.9
            ).await?;
        }

        if metrics.risk_metrics.max_drawdown > 0.1 {
            self.create_alert(
                AlertSeverity::Critical,
                AlertCategory::Risk,
                "High Drawdown".to_string(),
                format!("Max drawdown is {:.1}%", metrics.risk_metrics.max_drawdown * 100.0),
                "risk_management".to_string(),
                "max_drawdown".to_string(),
                metrics.risk_metrics.max_drawdown,
                0.1
            ).await?;
        }

        Ok(())
    }

    pub async fn start_alert_monitoring(&self, metrics_collector: Arc<crate::metrics::MetricsCollector>) -> Result<()> {
        let monitoring_system = Arc::new(self.clone());
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(std::time::Duration::from_secs(30));
            
            loop {
                interval.tick().await;
                
                // Get current metrics
                let metrics = metrics_collector.get_current_snapshot().await;
                
                // Check alert conditions
                if let Err(e) = monitoring_system.check_alert_conditions(&metrics).await {
                    error!("Error checking alert conditions: {}", e);
                }
                
                // Auto-resolve old alerts (older than 1 hour)
                if let Err(e) = monitoring_system.auto_resolve_old_alerts().await {
                    error!("Error auto-resolving old alerts: {}", e);
                }
            }
        });
        
        Ok(())
    }

    async fn auto_resolve_old_alerts(&self) -> Result<()> {
        let mut alerts = self.active_alerts.write().await;
        let now = Utc::now();
        let cutoff_time = now - chrono::Duration::hours(1);
        
        for alert in alerts.values_mut() {
            if !alert.resolved && alert.timestamp < cutoff_time {
                alert.resolved = true;
                alert.resolution_time = Some(now);
                info!("Auto-resolved old alert: {}", alert.title);
            }
        }
        
        Ok(())
    }

    pub async fn get_alert_summary(&self) -> Result<AlertSummary> {
        let alerts = self.active_alerts.read().await;
        let active_alerts: Vec<&MonitoringAlert> = alerts.values().filter(|a| !a.resolved).collect();
        
        let emergency_count = active_alerts.iter().filter(|a| matches!(a.severity, AlertSeverity::Emergency)).count();
        let critical_count = active_alerts.iter().filter(|a| matches!(a.severity, AlertSeverity::Critical)).count();
        let warning_count = active_alerts.iter().filter(|a| matches!(a.severity, AlertSeverity::Warning)).count();
        let info_count = active_alerts.iter().filter(|a| matches!(a.severity, AlertSeverity::Info)).count();
        
        Ok(AlertSummary {
            total_active: active_alerts.len(),
            emergency_count,
            critical_count,
            warning_count,
            info_count,
            oldest_alert: active_alerts.iter().min_by_key(|a| a.timestamp).map(|a| a.timestamp),
        })
    }
}

#[derive(Debug, Clone)]
pub struct MonitoringSystem {
    config: Arc<MarketReadinessConfig>,
    active_alerts: Arc<RwLock<HashMap<Uuid, MonitoringAlert>>>,
    is_monitoring: Arc<RwLock<bool>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertSummary {
    pub total_active: usize,
    pub emergency_count: usize,
    pub critical_count: usize,
    pub warning_count: usize,
    pub info_count: usize,
    pub oldest_alert: Option<DateTime<Utc>>,
}