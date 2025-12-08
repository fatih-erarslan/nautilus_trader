//! Real-time Risk Monitoring Dashboard
//! 
//! Provides comprehensive real-time monitoring, alerting, and visualization
//! of risk metrics for the cerebellar trading system.

use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex, RwLock};
use std::time::{SystemTime, UNIX_EPOCH, Duration, Instant};
use serde::{Serialize, Deserialize};
use anyhow::{Result, Context};
use tracing::{debug, info, warn, error};
use tokio::sync::{mpsc, watch, broadcast};
use tokio::time::interval;

use crate::risk_management::{RiskManager, RiskEvent, RiskStatus, RiskMetrics, CircuitBreakerType};

/// Real-time dashboard metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DashboardMetrics {
    /// System health indicators
    pub system_health: SystemHealth,
    /// Risk overview
    pub risk_overview: RiskOverview,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Alert summary
    pub alerts: AlertSummary,
    /// Real-time charts data
    pub charts: ChartData,
    /// Timestamp of last update
    pub last_update: u64
}

/// System health status
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealth {
    /// Overall system status
    pub status: HealthStatus,
    /// Individual component health
    pub components: HashMap<String, ComponentHealth>,
    /// System uptime in seconds
    pub uptime_seconds: u64,
    /// Memory usage percentage
    pub memory_usage_percent: f64,
    /// CPU usage percentage
    pub cpu_usage_percent: f64,
    /// Neural network processing latency (microseconds)
    pub neural_latency_us: f64,
    /// Risk validation latency (microseconds)
    pub risk_validation_latency_us: f64
}

/// Health status levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum HealthStatus {
    Healthy,
    Warning,
    Critical,
    Failed
}

/// Individual component health
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentHealth {
    pub status: HealthStatus,
    pub last_check: u64,
    pub error_count: u64,
    pub message: String
}

/// Risk overview summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskOverview {
    /// Trading status
    pub trading_enabled: bool,
    /// Active risk alerts
    pub active_alerts: u32,
    /// Critical alerts
    pub critical_alerts: u32,
    /// Current drawdown percentage
    pub current_drawdown_percent: f64,
    /// Daily P&L
    pub daily_pnl: f64,
    /// Total portfolio exposure
    pub total_exposure: f64,
    /// VaR utilization percentage
    pub var_utilization_percent: f64,
    /// Active circuit breakers
    pub active_circuit_breakers: Vec<CircuitBreakerType>,
    /// Risk score (0-100, higher is riskier)
    pub risk_score: f64
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Average neural network processing time (microseconds)
    pub avg_neural_processing_us: f64,
    /// Average risk validation time (microseconds)
    pub avg_risk_validation_us: f64,
    /// Trades per second (last minute)
    pub trades_per_second: f64,
    /// Neural network accuracy (recent period)
    pub neural_accuracy_percent: f64,
    /// Risk validation rejection rate
    pub risk_rejection_rate_percent: f64,
    /// System throughput (decisions per second)
    pub throughput_decisions_per_second: f64
}

/// Alert summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertSummary {
    /// Total active alerts
    pub total_active: u32,
    /// Critical alerts
    pub critical: u32,
    /// Warning alerts
    pub warning: u32,
    /// Info alerts
    pub info: u32,
    /// Recent alerts (last 10)
    pub recent_alerts: Vec<AlertInfo>
}

/// Alert information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertInfo {
    pub id: String,
    pub level: AlertLevel,
    pub message: String,
    pub timestamp: u64,
    pub component: String,
    pub acknowledged: bool
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertLevel {
    Critical,
    Warning,
    Info
}

/// Chart data for real-time visualization
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartData {
    /// P&L over time
    pub pnl_chart: TimeSeriesData,
    /// Drawdown over time
    pub drawdown_chart: TimeSeriesData,
    /// VaR over time
    pub var_chart: TimeSeriesData,
    /// Trading velocity over time
    pub velocity_chart: TimeSeriesData,
    /// Neural accuracy over time
    pub accuracy_chart: TimeSeriesData,
    /// Risk score over time
    pub risk_score_chart: TimeSeriesData,
    /// Position exposure breakdown
    pub exposure_breakdown: Vec<ExposureData>
}

/// Time series data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TimeSeriesData {
    pub timestamps: Vec<u64>,
    pub values: Vec<f64>,
    pub max_points: usize
}

impl TimeSeriesData {
    pub fn new(max_points: usize) -> Self {
        Self {
            timestamps: Vec::with_capacity(max_points),
            values: Vec::with_capacity(max_points),
            max_points
        }
    }

    pub fn add_point(&mut self, timestamp: u64, value: f64) {
        self.timestamps.push(timestamp);
        self.values.push(value);
        
        // Keep only the last max_points
        if self.timestamps.len() > self.max_points {
            self.timestamps.remove(0);
            self.values.remove(0);
        }
    }
}

/// Position exposure data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExposureData {
    pub symbol: String,
    pub exposure: f64,
    pub percentage: f64
}

/// Real-time risk dashboard
pub struct RiskDashboard {
    /// Risk manager reference
    risk_manager: Arc<RiskManager>,
    /// Dashboard metrics
    metrics: Arc<Mutex<DashboardMetrics>>,
    /// Alert manager
    alert_manager: Arc<AlertManager>,
    /// Performance tracker
    performance_tracker: Arc<PerformanceTracker>,
    /// Health monitor
    health_monitor: Arc<HealthMonitor>,
    /// Update channel for real-time updates
    update_sender: broadcast::Sender<DashboardMetrics>,
    /// System start time
    system_start_time: Instant
}

impl RiskDashboard {
    /// Create new risk dashboard
    pub fn new(risk_manager: Arc<RiskManager>) -> Self {
        let (update_sender, _) = broadcast::channel(100);
        
        Self {
            risk_manager,
            metrics: Arc::new(Mutex::new(DashboardMetrics::default())),
            alert_manager: Arc::new(AlertManager::new()),
            performance_tracker: Arc::new(PerformanceTracker::new()),
            health_monitor: Arc::new(HealthMonitor::new()),
            update_sender,
            system_start_time: Instant::now()
        }
    }

    /// Start the dashboard monitoring loop
    pub async fn start_monitoring(&self) -> Result<()> {
        info!("Starting risk dashboard monitoring");
        
        let dashboard = self.clone();
        tokio::spawn(async move {
            let mut interval = interval(Duration::from_millis(1000)); // 1 second updates
            
            loop {
                interval.tick().await;
                
                if let Err(e) = dashboard.update_metrics().await {
                    error!("Failed to update dashboard metrics: {}", e);
                }
            }
        });
        
        Ok(())
    }

    /// Update all dashboard metrics
    async fn update_metrics(&self) -> Result<()> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        
        // Get current risk status
        let risk_status = self.risk_manager.get_risk_status();
        let risk_metrics = self.risk_manager.get_metrics();
        
        // Update system health
        let system_health = self.health_monitor.get_health_status().await?;
        
        // Update performance metrics
        let performance_metrics = self.performance_tracker.get_performance_metrics().await?;
        
        // Update alert summary
        let alerts = self.alert_manager.get_alert_summary().await?;
        
        // Calculate risk overview
        let risk_overview = self.calculate_risk_overview(&risk_status, &risk_metrics);
        
        // Update charts
        let charts = self.update_charts(&risk_status, &risk_metrics, timestamp).await?;
        
        // Create updated metrics
        let updated_metrics = DashboardMetrics {
            system_health,
            risk_overview,
            performance_metrics,
            alerts,
            charts,
            last_update: timestamp
        };
        
        // Store metrics
        {
            let mut metrics = self.metrics.lock().unwrap();
            *metrics = updated_metrics.clone();
        }
        
        // Broadcast update to subscribers
        let _ = self.update_sender.send(updated_metrics);
        
        Ok(())
    }

    /// Calculate risk overview from current status
    fn calculate_risk_overview(&self, risk_status: &RiskStatus, risk_metrics: &RiskMetrics) -> RiskOverview {
        // Calculate risk score (0-100, higher is riskier)
        let mut risk_score = 0.0;
        
        // Drawdown component (30% weight)
        risk_score += (risk_status.current_drawdown * 100.0) * 0.3;
        
        // VaR utilization component (25% weight)
        let var_utilization = (risk_status.var.abs() / 100_000.0).min(1.0);
        risk_score += var_utilization * 100.0 * 0.25;
        
        // Trading velocity component (20% weight)
        let velocity_risk = (risk_status.trading_velocity / 100.0).min(1.0);
        risk_score += velocity_risk * 100.0 * 0.20;
        
        // Circuit breaker component (25% weight)
        let breaker_risk = if risk_status.active_circuit_breakers.is_empty() { 
            0.0 
        } else { 
            risk_status.active_circuit_breakers.len() as f64 * 20.0 
        };
        risk_score += breaker_risk * 0.25;
        
        risk_score = risk_score.min(100.0);
        
        RiskOverview {
            trading_enabled: risk_status.trading_enabled,
            active_alerts: self.alert_manager.get_active_alert_count(),
            critical_alerts: self.alert_manager.get_critical_alert_count(),
            current_drawdown_percent: risk_status.current_drawdown * 100.0,
            daily_pnl: risk_status.daily_pnl,
            total_exposure: risk_status.total_exposure,
            var_utilization_percent: var_utilization * 100.0,
            active_circuit_breakers: risk_status.active_circuit_breakers,
            risk_score
        }
    }

    /// Update chart data
    async fn update_charts(&self, risk_status: &RiskStatus, risk_metrics: &RiskMetrics, timestamp: u64) -> Result<ChartData> {
        let mut metrics = self.metrics.lock().unwrap();
        let mut charts = metrics.charts.clone();
        
        // Update time series data
        charts.pnl_chart.add_point(timestamp, risk_status.daily_pnl);
        charts.drawdown_chart.add_point(timestamp, risk_status.current_drawdown * 100.0);
        charts.var_chart.add_point(timestamp, risk_status.var);
        charts.velocity_chart.add_point(timestamp, risk_status.trading_velocity);
        
        // Update neural accuracy (placeholder - would come from actual tracking)
        charts.accuracy_chart.add_point(timestamp, 85.0); // Placeholder
        
        // Update risk score
        let risk_overview = self.calculate_risk_overview(risk_status, risk_metrics);
        charts.risk_score_chart.add_point(timestamp, risk_overview.risk_score);
        
        // Update exposure breakdown (placeholder)
        charts.exposure_breakdown = vec![
            ExposureData {
                symbol: "AAPL".to_string(),
                exposure: 50000.0,
                percentage: 50.0
            },
            ExposureData {
                symbol: "GOOGL".to_string(),
                exposure: 30000.0,
                percentage: 30.0
            },
            ExposureData {
                symbol: "MSFT".to_string(),
                exposure: 20000.0,
                percentage: 20.0
            }
        ];
        
        Ok(charts)
    }

    /// Get current dashboard metrics
    pub fn get_metrics(&self) -> DashboardMetrics {
        self.metrics.lock().unwrap().clone()
    }

    /// Subscribe to real-time updates
    pub fn subscribe_to_updates(&self) -> broadcast::Receiver<DashboardMetrics> {
        self.update_sender.subscribe()
    }

    /// Generate risk report
    pub async fn generate_risk_report(&self) -> Result<RiskReport> {
        let metrics = self.get_metrics();
        let risk_status = self.risk_manager.get_risk_status();
        
        RiskReport::generate(metrics, risk_status).await
    }
}

impl Clone for RiskDashboard {
    fn clone(&self) -> Self {
        Self {
            risk_manager: self.risk_manager.clone(),
            metrics: self.metrics.clone(),
            alert_manager: self.alert_manager.clone(),
            performance_tracker: self.performance_tracker.clone(),
            health_monitor: self.health_monitor.clone(),
            update_sender: self.update_sender.clone(),
            system_start_time: self.system_start_time
        }
    }
}

impl Default for DashboardMetrics {
    fn default() -> Self {
        Self {
            system_health: SystemHealth::default(),
            risk_overview: RiskOverview::default(),
            performance_metrics: PerformanceMetrics::default(),
            alerts: AlertSummary::default(),
            charts: ChartData::default(),
            last_update: 0
        }
    }
}

impl Default for SystemHealth {
    fn default() -> Self {
        Self {
            status: HealthStatus::Healthy,
            components: HashMap::new(),
            uptime_seconds: 0,
            memory_usage_percent: 0.0,
            cpu_usage_percent: 0.0,
            neural_latency_us: 0.0,
            risk_validation_latency_us: 0.0
        }
    }
}

impl Default for RiskOverview {
    fn default() -> Self {
        Self {
            trading_enabled: true,
            active_alerts: 0,
            critical_alerts: 0,
            current_drawdown_percent: 0.0,
            daily_pnl: 0.0,
            total_exposure: 0.0,
            var_utilization_percent: 0.0,
            active_circuit_breakers: Vec::new(),
            risk_score: 0.0
        }
    }
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            avg_neural_processing_us: 0.0,
            avg_risk_validation_us: 0.0,
            trades_per_second: 0.0,
            neural_accuracy_percent: 0.0,
            risk_rejection_rate_percent: 0.0,
            throughput_decisions_per_second: 0.0
        }
    }
}

impl Default for AlertSummary {
    fn default() -> Self {
        Self {
            total_active: 0,
            critical: 0,
            warning: 0,
            info: 0,
            recent_alerts: Vec::new()
        }
    }
}

impl Default for ChartData {
    fn default() -> Self {
        Self {
            pnl_chart: TimeSeriesData::new(300), // 5 minutes at 1 second intervals
            drawdown_chart: TimeSeriesData::new(300),
            var_chart: TimeSeriesData::new(300),
            velocity_chart: TimeSeriesData::new(300),
            accuracy_chart: TimeSeriesData::new(300),
            risk_score_chart: TimeSeriesData::new(300),
            exposure_breakdown: Vec::new()
        }
    }
}

/// Alert management system
pub struct AlertManager {
    /// Active alerts
    active_alerts: RwLock<HashMap<String, AlertInfo>>,
    /// Alert history
    alert_history: Mutex<VecDeque<AlertInfo>>,
    /// Alert configuration
    config: AlertConfig
}

/// Alert configuration
#[derive(Debug, Clone)]
pub struct AlertConfig {
    pub max_active_alerts: usize,
    pub max_history_size: usize,
    pub auto_acknowledge_timeout_ms: u64
}

impl Default for AlertConfig {
    fn default() -> Self {
        Self {
            max_active_alerts: 1000,
            max_history_size: 10000,
            auto_acknowledge_timeout_ms: 300_000 // 5 minutes
        }
    }
}

impl AlertManager {
    pub fn new() -> Self {
        Self {
            active_alerts: RwLock::new(HashMap::new()),
            alert_history: Mutex::new(VecDeque::new()),
            config: AlertConfig::default()
        }
    }

    /// Create a new alert
    pub async fn create_alert(&self, level: AlertLevel, message: String, component: String) -> Result<String> {
        let alert_id = format!("alert_{}", SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos());
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        
        let alert = AlertInfo {
            id: alert_id.clone(),
            level,
            message,
            timestamp,
            component,
            acknowledged: false
        };
        
        // Add to active alerts
        {
            let mut active = self.active_alerts.write().unwrap();
            active.insert(alert_id.clone(), alert.clone());
            
            // Limit active alerts
            if active.len() > self.config.max_active_alerts {
                // Remove oldest alerts
                let mut alerts: Vec<_> = active.values().cloned().collect();
                alerts.sort_by_key(|a| a.timestamp);
                for old_alert in alerts.iter().take(active.len() - self.config.max_active_alerts) {
                    active.remove(&old_alert.id);
                }
            }
        }
        
        // Add to history
        {
            let mut history = self.alert_history.lock().unwrap();
            history.push_back(alert);
            
            if history.len() > self.config.max_history_size {
                history.pop_front();
            }
        }
        
        Ok(alert_id)
    }

    /// Acknowledge an alert
    pub async fn acknowledge_alert(&self, alert_id: &str) -> Result<()> {
        let mut active = self.active_alerts.write().unwrap();
        if let Some(alert) = active.get_mut(alert_id) {
            alert.acknowledged = true;
        }
        Ok(())
    }

    /// Get alert summary
    pub async fn get_alert_summary(&self) -> Result<AlertSummary> {
        let active = self.active_alerts.read().unwrap();
        
        let mut critical = 0;
        let mut warning = 0;
        let mut info = 0;
        
        for alert in active.values() {
            if !alert.acknowledged {
                match alert.level {
                    AlertLevel::Critical => critical += 1,
                    AlertLevel::Warning => warning += 1,
                    AlertLevel::Info => info += 1
                }
            }
        }
        
        // Get recent alerts
        let history = self.alert_history.lock().unwrap();
        let recent_alerts: Vec<AlertInfo> = history.iter()
            .rev()
            .take(10)
            .cloned()
            .collect();
        
        Ok(AlertSummary {
            total_active: active.len() as u32,
            critical,
            warning,
            info,
            recent_alerts
        })
    }

    /// Get active alert count
    pub fn get_active_alert_count(&self) -> u32 {
        self.active_alerts.read().unwrap().len() as u32
    }

    /// Get critical alert count
    pub fn get_critical_alert_count(&self) -> u32 {
        self.active_alerts.read().unwrap()
            .values()
            .filter(|alert| matches!(alert.level, AlertLevel::Critical) && !alert.acknowledged)
            .count() as u32
    }
}

/// Performance tracking system
pub struct PerformanceTracker {
    /// Performance metrics
    metrics: Mutex<PerformanceMetrics>,
    /// Timing samples
    neural_latency_samples: Mutex<VecDeque<f64>>,
    risk_latency_samples: Mutex<VecDeque<f64>>,
    throughput_samples: Mutex<VecDeque<(u64, u64)>> // (timestamp, count)
}

impl PerformanceTracker {
    pub fn new() -> Self {
        Self {
            metrics: Mutex::new(PerformanceMetrics::default()),
            neural_latency_samples: Mutex::new(VecDeque::with_capacity(1000)),
            risk_latency_samples: Mutex::new(VecDeque::with_capacity(1000)),
            throughput_samples: Mutex::new(VecDeque::with_capacity(60)) // 1 minute of samples
        }
    }

    /// Record neural processing latency
    pub fn record_neural_latency(&self, latency_us: f64) {
        let mut samples = self.neural_latency_samples.lock().unwrap();
        samples.push_back(latency_us);
        if samples.len() > 1000 {
            samples.pop_front();
        }
    }

    /// Record risk validation latency
    pub fn record_risk_latency(&self, latency_us: f64) {
        let mut samples = self.risk_latency_samples.lock().unwrap();
        samples.push_back(latency_us);
        if samples.len() > 1000 {
            samples.pop_front();
        }
    }

    /// Record throughput sample
    pub fn record_throughput(&self, decisions_count: u64) {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)
            .unwrap_or_default().as_secs();
        
        let mut samples = self.throughput_samples.lock().unwrap();
        samples.push_back((timestamp, decisions_count));
        
        // Keep only last minute of samples
        let cutoff = timestamp.saturating_sub(60);
        while let Some(&(ts, _)) = samples.front() {
            if ts < cutoff {
                samples.pop_front();
            } else {
                break;
            }
        }
    }

    /// Get performance metrics
    pub async fn get_performance_metrics(&self) -> Result<PerformanceMetrics> {
        let neural_samples = self.neural_latency_samples.lock().unwrap();
        let risk_samples = self.risk_latency_samples.lock().unwrap();
        let throughput_samples = self.throughput_samples.lock().unwrap();
        
        let avg_neural_processing_us = if neural_samples.is_empty() {
            0.0
        } else {
            neural_samples.iter().sum::<f64>() / neural_samples.len() as f64
        };
        
        let avg_risk_validation_us = if risk_samples.is_empty() {
            0.0
        } else {
            risk_samples.iter().sum::<f64>() / risk_samples.len() as f64
        };
        
        let throughput_decisions_per_second = if throughput_samples.len() < 2 {
            0.0
        } else {
            let total_decisions = throughput_samples.iter().map(|(_, count)| count).sum::<u64>();
            let time_span = throughput_samples.back().unwrap().0 - throughput_samples.front().unwrap().0;
            if time_span > 0 {
                total_decisions as f64 / time_span as f64
            } else {
                0.0
            }
        };
        
        Ok(PerformanceMetrics {
            avg_neural_processing_us,
            avg_risk_validation_us,
            trades_per_second: 0.0, // Would be calculated from actual trade data
            neural_accuracy_percent: 85.0, // Placeholder
            risk_rejection_rate_percent: 15.0, // Placeholder
            throughput_decisions_per_second
        })
    }
}

/// System health monitoring
pub struct HealthMonitor {
    /// Component health status
    component_health: RwLock<HashMap<String, ComponentHealth>>,
    /// System start time
    start_time: Instant
}

impl HealthMonitor {
    pub fn new() -> Self {
        Self {
            component_health: RwLock::new(HashMap::new()),
            start_time: Instant::now()
        }
    }

    /// Get system health status
    pub async fn get_health_status(&self) -> Result<SystemHealth> {
        let uptime_seconds = self.start_time.elapsed().as_secs();
        
        // Get system resource usage (placeholder implementation)
        let memory_usage_percent = self.get_memory_usage().await?;
        let cpu_usage_percent = self.get_cpu_usage().await?;
        
        let components = self.component_health.read().unwrap().clone();
        
        // Determine overall status
        let status = if components.values().any(|h| matches!(h.status, HealthStatus::Failed)) {
            HealthStatus::Failed
        } else if components.values().any(|h| matches!(h.status, HealthStatus::Critical)) {
            HealthStatus::Critical
        } else if components.values().any(|h| matches!(h.status, HealthStatus::Warning)) {
            HealthStatus::Warning
        } else {
            HealthStatus::Healthy
        };
        
        Ok(SystemHealth {
            status,
            components,
            uptime_seconds,
            memory_usage_percent,
            cpu_usage_percent,
            neural_latency_us: 0.0, // Would be set from performance tracker
            risk_validation_latency_us: 0.0 // Would be set from performance tracker
        })
    }

    /// Update component health
    pub async fn update_component_health(&self, component: String, status: HealthStatus, message: String) -> Result<()> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        
        let health = ComponentHealth {
            status,
            last_check: timestamp,
            error_count: 0, // Would track actual errors
            message
        };
        
        self.component_health.write().unwrap().insert(component, health);
        Ok(())
    }

    /// Get memory usage (placeholder)
    async fn get_memory_usage(&self) -> Result<f64> {
        // Placeholder implementation - would use actual system monitoring
        Ok(45.0)
    }

    /// Get CPU usage (placeholder)
    async fn get_cpu_usage(&self) -> Result<f64> {
        // Placeholder implementation - would use actual system monitoring
        Ok(25.0)
    }
}

/// Risk report generation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskReport {
    pub generated_at: u64,
    pub system_status: String,
    pub key_metrics: RiskReportMetrics,
    pub alerts: Vec<AlertInfo>,
    pub recommendations: Vec<String>,
    pub charts: Vec<ChartSummary>
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RiskReportMetrics {
    pub current_drawdown: f64,
    pub daily_pnl: f64,
    pub var_utilization: f64,
    pub risk_score: f64,
    pub trading_velocity: f64,
    pub active_positions: usize
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChartSummary {
    pub name: String,
    pub description: String,
    pub current_value: f64,
    pub trend: String, // "increasing", "decreasing", "stable"
    pub last_24h_change: f64
}

impl RiskReport {
    pub async fn generate(metrics: DashboardMetrics, risk_status: RiskStatus) -> Result<Self> {
        let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
        
        let system_status = if risk_status.trading_enabled {
            "OPERATIONAL".to_string()
        } else {
            "TRADING HALTED".to_string()
        };
        
        let key_metrics = RiskReportMetrics {
            current_drawdown: risk_status.current_drawdown,
            daily_pnl: risk_status.daily_pnl,
            var_utilization: (risk_status.var / 100_000.0).min(1.0),
            risk_score: metrics.risk_overview.risk_score,
            trading_velocity: risk_status.trading_velocity,
            active_positions: risk_status.position_count
        };
        
        let recommendations = Self::generate_recommendations(&metrics, &risk_status);
        
        Ok(RiskReport {
            generated_at: timestamp,
            system_status,
            key_metrics,
            alerts: metrics.alerts.recent_alerts,
            recommendations,
            charts: Vec::new() // Would include chart summaries
        })
    }

    fn generate_recommendations(metrics: &DashboardMetrics, risk_status: &RiskStatus) -> Vec<String> {
        let mut recommendations = Vec::new();
        
        if metrics.risk_overview.risk_score > 70.0 {
            recommendations.push("High risk score detected. Consider reducing position sizes.".to_string());
        }
        
        if risk_status.current_drawdown > 0.03 {
            recommendations.push("Elevated drawdown. Review trading strategy performance.".to_string());
        }
        
        if risk_status.trading_velocity > 80.0 {
            recommendations.push("High trading velocity. Monitor for overtrading.".to_string());
        }
        
        if !risk_status.active_circuit_breakers.is_empty() {
            recommendations.push("Circuit breakers active. Review and address underlying issues.".to_string());
        }
        
        if recommendations.is_empty() {
            recommendations.push("All risk metrics within acceptable ranges.".to_string());
        }
        
        recommendations
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::risk_management::RiskLimits;

    #[tokio::test]
    async fn test_dashboard_creation() {
        let risk_manager = Arc::new(RiskManager::new(RiskLimits::default()));
        let dashboard = RiskDashboard::new(risk_manager);
        
        let metrics = dashboard.get_metrics();
        assert_eq!(metrics.last_update, 0);
    }

    #[tokio::test]
    async fn test_alert_manager() {
        let alert_manager = AlertManager::new();
        
        let alert_id = alert_manager.create_alert(
            AlertLevel::Critical,
            "Test alert".to_string(),
            "test_component".to_string()
        ).await.unwrap();
        
        let summary = alert_manager.get_alert_summary().await.unwrap();
        assert_eq!(summary.critical, 1);
        
        alert_manager.acknowledge_alert(&alert_id).await.unwrap();
        
        let summary = alert_manager.get_alert_summary().await.unwrap();
        assert_eq!(summary.critical, 0);
    }

    #[tokio::test]
    async fn test_performance_tracker() {
        let tracker = PerformanceTracker::new();
        
        tracker.record_neural_latency(100.0);
        tracker.record_risk_latency(50.0);
        tracker.record_throughput(10);
        
        let metrics = tracker.get_performance_metrics().await.unwrap();
        assert_eq!(metrics.avg_neural_processing_us, 100.0);
        assert_eq!(metrics.avg_risk_validation_us, 50.0);
    }

    #[test]
    fn test_time_series_data() {
        let mut data = TimeSeriesData::new(3);
        
        data.add_point(1, 10.0);
        data.add_point(2, 20.0);
        data.add_point(3, 30.0);
        assert_eq!(data.values.len(), 3);
        
        data.add_point(4, 40.0);
        assert_eq!(data.values.len(), 3);
        assert_eq!(data.values[0], 20.0); // First point removed
    }
}