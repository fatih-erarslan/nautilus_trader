//! PADS monitoring and metrics

use crate::{
    config::PadsConfig,
    error::Result,
    types::*,
};
use dashmap::DashMap;
use metrics::{counter, gauge, histogram};
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::{debug, info};
use chrono::{DateTime, Utc};
use std::time::Instant;

/// PADS monitoring system
pub struct PadsMonitor {
    config: Arc<PadsConfig>,
    metrics_registry: Arc<MetricsRegistry>,
    performance_tracker: Arc<RwLock<PerformanceTracker>>,
    alert_manager: Arc<AlertManager>,
}

/// Metrics registry
struct MetricsRegistry {
    counters: DashMap<String, u64>,
    gauges: DashMap<String, f64>,
    histograms: DashMap<String, Vec<f64>>,
}

/// Performance tracker
struct PerformanceTracker {
    decision_latencies: Vec<LatencyRecord>,
    scale_transitions: Vec<TransitionRecord>,
    error_log: Vec<ErrorRecord>,
    resource_usage: Vec<ResourceSnapshot>,
}

/// Latency record
#[derive(Debug, Clone)]
struct LatencyRecord {
    decision_id: String,
    scale: ScaleLevel,
    start: Instant,
    end: Instant,
    latency_ms: u64,
}

/// Transition record
#[derive(Debug, Clone)]
struct TransitionRecord {
    from_scale: ScaleLevel,
    to_scale: ScaleLevel,
    timestamp: DateTime<Utc>,
    duration_ms: u64,
    success: bool,
}

/// Error record
#[derive(Debug, Clone)]
struct ErrorRecord {
    timestamp: DateTime<Utc>,
    error_type: String,
    details: String,
    scale: Option<ScaleLevel>,
}

/// Resource snapshot
#[derive(Debug, Clone)]
struct ResourceSnapshot {
    timestamp: Instant,
    cpu_usage: f64,
    memory_usage: f64,
    active_decisions: usize,
    queue_depth: usize,
}

/// Alert manager
struct AlertManager {
    active_alerts: DashMap<String, Alert>,
    alert_history: RwLock<Vec<Alert>>,
}

/// Alert information
#[derive(Debug, Clone)]
struct Alert {
    id: String,
    alert_type: AlertType,
    severity: AlertSeverity,
    message: String,
    timestamp: DateTime<Utc>,
    resolved: bool,
}

/// Alert type
#[derive(Debug, Clone, Copy)]
enum AlertType {
    HighLatency,
    HighErrorRate,
    ResourceExhaustion,
    ScaleTransitionFailure,
}

/// Alert severity
#[derive(Debug, Clone, Copy)]
enum AlertSeverity {
    Info,
    Warning,
    Error,
    Critical,
}

impl PadsMonitor {
    /// Create new monitor
    pub async fn new(config: Arc<PadsConfig>) -> Result<Self> {
        let metrics_registry = Arc::new(MetricsRegistry {
            counters: DashMap::new(),
            gauges: DashMap::new(),
            histograms: DashMap::new(),
        });
        
        let performance_tracker = Arc::new(RwLock::new(PerformanceTracker {
            decision_latencies: Vec::new(),
            scale_transitions: Vec::new(),
            error_log: Vec::new(),
            resource_usage: Vec::new(),
        }));
        
        let alert_manager = Arc::new(AlertManager {
            active_alerts: DashMap::new(),
            alert_history: RwLock::new(Vec::new()),
        });
        
        Ok(Self {
            config,
            metrics_registry,
            performance_tracker,
            alert_manager,
        })
    }
    
    /// Start monitoring
    pub async fn start(&self) -> Result<()> {
        info!("Starting PADS monitoring");
        
        // Initialize metrics
        self.initialize_metrics();
        
        // Start metrics collection
        self.start_metrics_collection().await?;
        
        // Start alert monitoring
        self.start_alert_monitoring().await?;
        
        Ok(())
    }
    
    /// Initialize metrics
    fn initialize_metrics(&self) {
        // Decision metrics
        counter!("pads_decisions_total", "Total decisions processed");
        counter!("pads_decisions_success", "Successful decisions");
        counter!("pads_decisions_failed", "Failed decisions");
        
        // Scale metrics
        gauge!("pads_active_scale", "Currently active scale");
        counter!("pads_scale_transitions", "Scale transitions");
        histogram!("pads_scale_transition_duration", "Scale transition duration");
        
        // Performance metrics
        histogram!("pads_decision_latency", "Decision processing latency");
        gauge!("pads_cpu_usage", "CPU usage percentage");
        gauge!("pads_memory_usage", "Memory usage percentage");
        
        // Communication metrics
        counter!("pads_messages_sent", "Messages sent");
        counter!("pads_messages_received", "Messages received");
        gauge!("pads_channel_error_rate", "Channel error rate");
        
        // Resilience metrics
        gauge!("pads_circuit_breaker_state", "Circuit breaker state");
        gauge!("pads_adaptive_capacity", "Adaptive capacity");
        counter!("pads_recovery_actions", "Recovery actions taken");
    }
    
    /// Start metrics collection
    async fn start_metrics_collection(&self) -> Result<()> {
        let interval = self.config.monitor_config.metrics_interval;
        let monitor = Arc::downgrade(&Arc::new(self));
        
        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            
            loop {
                interval_timer.tick().await;
                
                if let Some(monitor) = monitor.upgrade() {
                    if let Err(e) = monitor.collect_metrics().await {
                        tracing::error!("Metrics collection failed: {}", e);
                    }
                } else {
                    break;
                }
            }
        });
        
        Ok(())
    }
    
    /// Collect current metrics
    async fn collect_metrics(&self) -> Result<()> {
        debug!("Collecting metrics");
        
        // Collect resource usage
        let cpu_usage = self.get_cpu_usage();
        let memory_usage = self.get_memory_usage();
        
        gauge!("pads_cpu_usage").set(cpu_usage);
        gauge!("pads_memory_usage").set(memory_usage);
        
        // Record snapshot
        let mut tracker = self.performance_tracker.write().await;
        tracker.resource_usage.push(ResourceSnapshot {
            timestamp: Instant::now(),
            cpu_usage,
            memory_usage,
            active_decisions: 0, // Would get actual count
            queue_depth: 0, // Would get actual depth
        });
        
        // Trim old data
        self.trim_old_metrics(&mut tracker).await;
        
        Ok(())
    }
    
    /// Get CPU usage (simplified)
    fn get_cpu_usage(&self) -> f64 {
        // In production, would use actual CPU metrics
        0.45
    }
    
    /// Get memory usage (simplified)
    fn get_memory_usage(&self) -> f64 {
        // In production, would use actual memory metrics
        0.65
    }
    
    /// Trim old metrics data
    async fn trim_old_metrics(&self, tracker: &mut PerformanceTracker) {
        let retention = self.config.monitor_config.retention_period;
        let cutoff = Instant::now() - retention;
        
        // Trim latency records
        tracker.decision_latencies.retain(|r| r.start > cutoff);
        
        // Trim resource snapshots
        tracker.resource_usage.retain(|s| s.timestamp > cutoff);
        
        // Trim old transitions
        let cutoff_dt = Utc::now() - chrono::Duration::from_std(retention).unwrap();
        tracker.scale_transitions.retain(|t| t.timestamp > cutoff_dt);
        tracker.error_log.retain(|e| e.timestamp > cutoff_dt);
    }
    
    /// Start alert monitoring
    async fn start_alert_monitoring(&self) -> Result<()> {
        let interval = std::time::Duration::from_secs(10);
        let monitor = Arc::downgrade(&Arc::new(self));
        
        tokio::spawn(async move {
            let mut interval_timer = tokio::time::interval(interval);
            
            loop {
                interval_timer.tick().await;
                
                if let Some(monitor) = monitor.upgrade() {
                    if let Err(e) = monitor.check_alerts().await {
                        tracing::error!("Alert check failed: {}", e);
                    }
                } else {
                    break;
                }
            }
        });
        
        Ok(())
    }
    
    /// Check for alert conditions
    async fn check_alerts(&self) -> Result<()> {
        let thresholds = &self.config.monitor_config.alert_thresholds;
        
        // Check latency
        let avg_latency = self.get_average_latency().await;
        if avg_latency > thresholds.latency_ms as f64 {
            self.raise_alert(
                AlertType::HighLatency,
                AlertSeverity::Warning,
                format!("Average latency {}ms exceeds threshold", avg_latency)
            ).await;
        }
        
        // Check error rate
        let error_rate = self.get_error_rate().await;
        if error_rate > thresholds.error_rate_percent / 100.0 {
            self.raise_alert(
                AlertType::HighErrorRate,
                AlertSeverity::Error,
                format!("Error rate {:.1}% exceeds threshold", error_rate * 100.0)
            ).await;
        }
        
        // Check resource usage
        let memory_usage = self.get_memory_usage();
        if memory_usage > thresholds.memory_percent / 100.0 {
            self.raise_alert(
                AlertType::ResourceExhaustion,
                AlertSeverity::Warning,
                format!("Memory usage {:.1}% exceeds threshold", memory_usage * 100.0)
            ).await;
        }
        
        Ok(())
    }
    
    /// Get average latency
    async fn get_average_latency(&self) -> f64 {
        let tracker = self.performance_tracker.read().await;
        
        if tracker.decision_latencies.is_empty() {
            return 0.0;
        }
        
        let sum: u64 = tracker.decision_latencies.iter()
            .map(|r| r.latency_ms)
            .sum();
        
        sum as f64 / tracker.decision_latencies.len() as f64
    }
    
    /// Get error rate
    async fn get_error_rate(&self) -> f64 {
        let total = self.metrics_registry.counters
            .get("decisions_total")
            .map(|v| *v)
            .unwrap_or(0);
        
        let failed = self.metrics_registry.counters
            .get("decisions_failed")
            .map(|v| *v)
            .unwrap_or(0);
        
        if total == 0 {
            0.0
        } else {
            failed as f64 / total as f64
        }
    }
    
    /// Raise an alert
    async fn raise_alert(&self, alert_type: AlertType, severity: AlertSeverity, message: String) {
        let alert = Alert {
            id: uuid::Uuid::new_v4().to_string(),
            alert_type,
            severity,
            message: message.clone(),
            timestamp: Utc::now(),
            resolved: false,
        };
        
        self.alert_manager.active_alerts.insert(alert.id.clone(), alert.clone());
        self.alert_manager.alert_history.write().await.push(alert);
        
        match severity {
            AlertSeverity::Critical => tracing::error!("CRITICAL ALERT: {}", message),
            AlertSeverity::Error => tracing::error!("ERROR ALERT: {}", message),
            AlertSeverity::Warning => tracing::warn!("WARNING ALERT: {}", message),
            AlertSeverity::Info => info!("INFO ALERT: {}", message),
        }
    }
    
    /// Record decision start
    pub fn record_decision_start(&self, decision: &PanarchyDecision) {
        counter!("pads_decisions_total").increment(1);
        
        let scale = ScaleLevel::Micro; // Would determine actual scale
        gauge!("pads_active_scale").set(scale as i64 as f64);
    }
    
    /// Record decision completion
    pub fn record_decision_complete(&self, result: &DecisionResult) {
        if result.success {
            counter!("pads_decisions_success").increment(1);
        } else {
            counter!("pads_decisions_failed").increment(1);
        }
        
        histogram!("pads_decision_latency")
            .record(result.metrics.processing_time_ms as f64);
    }
    
    /// Record scale processing
    pub fn record_scale_processing(&self, scale: ScaleLevel) {
        gauge!("pads_active_scale").set(scale as i64 as f64);
    }
    
    /// Record routing latency
    pub fn record_routing_latency(&self, scale: ScaleLevel, latency_ms: u64) {
        histogram!("pads_routing_latency", "scale" => format!("{:?}", scale))
            .record(latency_ms as f64);
    }
    
    /// Record message sent
    pub fn record_message_sent(&self, from: ScaleLevel, to: ScaleLevel) {
        counter!("pads_messages_sent", 
            "from" => format!("{:?}", from),
            "to" => format!("{:?}", to)
        ).increment(1);
    }
    
    /// Record recovery action
    pub fn record_recovery_action(&self, action: &str, value: f64) {
        counter!("pads_recovery_actions", "action" => action).increment(1);
        gauge!("pads_recovery_impact", "action" => action).set(value);
    }
    
    /// Get system metrics
    pub async fn get_metrics(&self) -> Result<SystemMetrics> {
        let tracker = self.performance_tracker.read().await;
        
        let decisions_per_second = self.calculate_decision_rate(&tracker);
        let success_rate = self.calculate_success_rate();
        let avg_latency_ms = self.get_average_latency().await;
        let cpu_usage = self.get_cpu_usage();
        let memory_usage = self.get_memory_usage();
        
        Ok(SystemMetrics {
            decisions_per_second,
            success_rate,
            avg_latency_ms,
            cpu_usage,
            memory_usage,
        })
    }
    
    /// Calculate decision rate
    fn calculate_decision_rate(&self, tracker: &PerformanceTracker) -> f64 {
        if tracker.decision_latencies.is_empty() {
            return 0.0;
        }
        
        let duration = tracker.decision_latencies.last().unwrap().end - 
                      tracker.decision_latencies.first().unwrap().start;
        
        if duration.as_secs() == 0 {
            return 0.0;
        }
        
        tracker.decision_latencies.len() as f64 / duration.as_secs_f64()
    }
    
    /// Calculate success rate
    fn calculate_success_rate(&self) -> f64 {
        let total = self.metrics_registry.counters
            .get("decisions_total")
            .map(|v| *v)
            .unwrap_or(0);
        
        let success = self.metrics_registry.counters
            .get("decisions_success")
            .map(|v| *v)
            .unwrap_or(0);
        
        if total == 0 {
            1.0
        } else {
            success as f64 / total as f64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_monitor_creation() {
        let config = Arc::new(PadsConfig::default());
        let monitor = PadsMonitor::new(config).await.unwrap();
        assert!(monitor.start().await.is_ok());
    }
}