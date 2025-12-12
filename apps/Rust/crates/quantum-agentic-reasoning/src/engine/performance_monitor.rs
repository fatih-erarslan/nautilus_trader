//! Performance Monitor Module
//!
//! Comprehensive performance monitoring and analysis for quantum trading operations with real-time metrics.

use crate::core::{QarResult, FactorMap};
use crate::error::QarError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use chrono::{DateTime, Utc, Duration};
use uuid::Uuid;

/// Performance metric types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MetricType {
    Latency,
    Throughput,
    ErrorRate,
    SuccessRate,
    ResourceUtilization,
    MemoryUsage,
    CpuUsage,
    NetworkIO,
    DiskIO,
    QuantumGateTime,
    CircuitDepth,
    Fidelity,
    Coherence,
    TradingReturn,
    Slippage,
    ExecutionTime,
    Custom(String),
}

/// Performance measurement
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMeasurement {
    pub id: String,
    pub component: String,
    pub metric_type: MetricType,
    pub value: f64,
    pub unit: String,
    pub timestamp: DateTime<Utc>,
    pub tags: HashMap<String, String>,
    pub metadata: HashMap<String, String>,
}

/// Performance alert level
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AlertLevel {
    Info,
    Warning,
    Critical,
    Emergency,
}

/// Performance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    pub id: String,
    pub component: String,
    pub metric_type: MetricType,
    pub level: AlertLevel,
    pub message: String,
    pub current_value: f64,
    pub threshold_value: f64,
    pub timestamp: DateTime<Utc>,
    pub acknowledged: bool,
    pub resolved: bool,
}

/// Performance threshold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThreshold {
    pub metric_type: MetricType,
    pub component: Option<String>,
    pub warning_threshold: f64,
    pub critical_threshold: f64,
    pub emergency_threshold: f64,
    pub comparison_operator: ComparisonOperator,
    pub enabled: bool,
}

/// Comparison operators for thresholds
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ComparisonOperator {
    GreaterThan,
    LessThan,
    Equal,
    NotEqual,
    GreaterThanOrEqual,
    LessThanOrEqual,
}

/// Performance statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceStats {
    pub metric_type: MetricType,
    pub component: String,
    pub count: u64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub median: f64,
    pub std_dev: f64,
    pub percentiles: HashMap<u8, f64>, // P50, P95, P99, etc.
    pub trend: TrendDirection,
    pub last_updated: DateTime<Utc>,
}

/// Trend direction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TrendDirection {
    Increasing,
    Decreasing,
    Stable,
    Volatile,
}

/// Performance report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceReport {
    pub id: String,
    pub title: String,
    pub description: String,
    pub period_start: DateTime<Utc>,
    pub period_end: DateTime<Utc>,
    pub component_summaries: Vec<ComponentSummary>,
    pub overall_health_score: f64,
    pub recommendations: Vec<String>,
    pub alerts_summary: AlertsSummary,
    pub generated_at: DateTime<Utc>,
}

/// Component performance summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ComponentSummary {
    pub component: String,
    pub health_score: f64,
    pub key_metrics: Vec<PerformanceStats>,
    pub issues: Vec<String>,
    pub improvements: Vec<String>,
}

/// Alerts summary
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AlertsSummary {
    pub total_alerts: u64,
    pub critical_alerts: u64,
    pub warning_alerts: u64,
    pub unacknowledged_alerts: u64,
    pub resolved_alerts: u64,
}

/// Performance monitor configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMonitorConfig {
    pub sampling_interval_seconds: u64,
    pub retention_period_days: u32,
    pub aggregation_window_minutes: u32,
    pub enable_real_time_alerts: bool,
    pub enable_predictive_analytics: bool,
    pub enable_anomaly_detection: bool,
    pub alert_cooldown_minutes: u32,
    pub max_measurements_per_component: usize,
    pub enable_auto_scaling_recommendations: bool,
}

/// Performance monitor implementation
#[derive(Debug)]
pub struct PerformanceMonitor {
    config: PerformanceMonitorConfig,
    measurements: Arc<RwLock<HashMap<String, VecDeque<PerformanceMeasurement>>>>,
    statistics: Arc<RwLock<HashMap<String, PerformanceStats>>>,
    thresholds: Arc<RwLock<Vec<PerformanceThreshold>>>,
    active_alerts: Arc<RwLock<HashMap<String, PerformanceAlert>>>,
    alert_history: Arc<RwLock<Vec<PerformanceAlert>>>,
    anomaly_detector: Arc<dyn AnomalyDetector + Send + Sync>,
    trend_analyzer: Arc<dyn TrendAnalyzer + Send + Sync>,
    health_calculator: Arc<Mutex<HealthCalculator>>,
}

/// Anomaly detection trait
#[async_trait::async_trait]
pub trait AnomalyDetector {
    async fn detect_anomalies(&self, measurements: &[PerformanceMeasurement]) -> QarResult<Vec<Anomaly>>;
    async fn update_model(&self, measurements: &[PerformanceMeasurement]) -> QarResult<()>;
    async fn get_anomaly_score(&self, measurement: &PerformanceMeasurement) -> QarResult<f64>;
}

/// Trend analysis trait
#[async_trait::async_trait]
pub trait TrendAnalyzer {
    async fn analyze_trend(&self, measurements: &[PerformanceMeasurement]) -> QarResult<TrendDirection>;
    async fn predict_future_values(&self, measurements: &[PerformanceMeasurement], steps: usize) -> QarResult<Vec<f64>>;
    async fn calculate_seasonality(&self, measurements: &[PerformanceMeasurement]) -> QarResult<Vec<f64>>;
}

/// Anomaly detection result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Anomaly {
    pub measurement_id: String,
    pub anomaly_score: f64,
    pub description: String,
    pub severity: AlertLevel,
}

/// Health calculator
#[derive(Debug)]
pub struct HealthCalculator {
    component_weights: HashMap<String, f64>,
    metric_weights: HashMap<MetricType, f64>,
    threshold_penalties: HashMap<AlertLevel, f64>,
}

impl PerformanceMonitor {
    /// Create new performance monitor
    pub fn new(
        config: PerformanceMonitorConfig,
        anomaly_detector: Arc<dyn AnomalyDetector + Send + Sync>,
        trend_analyzer: Arc<dyn TrendAnalyzer + Send + Sync>,
    ) -> Self {
        let health_calculator = HealthCalculator {
            component_weights: HashMap::new(),
            metric_weights: HashMap::new(),
            threshold_penalties: [
                (AlertLevel::Warning, 0.1),
                (AlertLevel::Critical, 0.3),
                (AlertLevel::Emergency, 0.6),
            ].into_iter().collect(),
        };

        Self {
            config,
            measurements: Arc::new(RwLock::new(HashMap::new())),
            statistics: Arc::new(RwLock::new(HashMap::new())),
            thresholds: Arc::new(RwLock::new(Vec::new())),
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            alert_history: Arc::new(RwLock::new(Vec::new())),
            anomaly_detector,
            trend_analyzer,
            health_calculator: Arc::new(Mutex::new(health_calculator)),
        }
    }

    /// Record performance measurement
    pub async fn record_measurement(&self, mut measurement: PerformanceMeasurement) -> QarResult<()> {
        measurement.id = Uuid::new_v4().to_string();
        measurement.timestamp = Utc::now();

        let component_key = format!("{}::{:?}", measurement.component, measurement.metric_type);

        // Store measurement
        {
            let mut measurements = self.measurements.write().await;
            let component_measurements = measurements.entry(component_key.clone()).or_insert_with(VecDeque::new);
            
            component_measurements.push_back(measurement.clone());
            
            // Limit measurements per component
            while component_measurements.len() > self.config.max_measurements_per_component {
                component_measurements.pop_front();
            }
        }

        // Update statistics
        self.update_statistics(&component_key, &measurement).await?;

        // Check thresholds and generate alerts
        if self.config.enable_real_time_alerts {
            self.check_thresholds(&measurement).await?;
        }

        // Detect anomalies
        if self.config.enable_anomaly_detection {
            let anomaly_score = self.anomaly_detector.get_anomaly_score(&measurement).await?;
            if anomaly_score > 0.8 { // Threshold for anomaly
                self.generate_anomaly_alert(&measurement, anomaly_score).await?;
            }
        }

        Ok(())
    }

    /// Update statistics for a measurement
    async fn update_statistics(&self, component_key: &str, measurement: &PerformanceMeasurement) -> QarResult<()> {
        let measurements = self.measurements.read().await;
        let component_measurements = measurements.get(component_key).unwrap();
        
        let values: Vec<f64> = component_measurements.iter().map(|m| m.value).collect();
        
        if values.is_empty() {
            return Ok(());
        }

        let count = values.len() as u64;
        let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        let mean = values.iter().sum::<f64>() / values.len() as f64;
        
        let mut sorted_values = values.clone();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let median = if sorted_values.len() % 2 == 0 {
            (sorted_values[sorted_values.len() / 2 - 1] + sorted_values[sorted_values.len() / 2]) / 2.0
        } else {
            sorted_values[sorted_values.len() / 2]
        };

        let variance = values.iter().map(|&x| (x - mean).powi(2)).sum::<f64>() / values.len() as f64;
        let std_dev = variance.sqrt();

        let mut percentiles = HashMap::new();
        for &p in &[50, 90, 95, 99] {
            let index = ((p as f64 / 100.0) * (sorted_values.len() - 1) as f64).round() as usize;
            percentiles.insert(p, sorted_values[index.min(sorted_values.len() - 1)]);
        }

        let trend = self.trend_analyzer.analyze_trend(component_measurements.make_contiguous()).await?;

        let stats = PerformanceStats {
            metric_type: measurement.metric_type.clone(),
            component: measurement.component.clone(),
            count,
            min,
            max,
            mean,
            median,
            std_dev,
            percentiles,
            trend,
            last_updated: Utc::now(),
        };

        let mut statistics = self.statistics.write().await;
        statistics.insert(component_key.to_string(), stats);

        Ok(())
    }

    /// Check thresholds and generate alerts
    async fn check_thresholds(&self, measurement: &PerformanceMeasurement) -> QarResult<()> {
        let thresholds = self.thresholds.read().await;
        
        for threshold in thresholds.iter() {
            if !threshold.enabled {
                continue;
            }

            // Check if threshold applies to this measurement
            if threshold.metric_type != measurement.metric_type {
                continue;
            }

            if let Some(component) = &threshold.component {
                if component != &measurement.component {
                    continue;
                }
            }

            // Check threshold violation
            let violation_level = self.check_threshold_violation(measurement.value, threshold);
            
            if let Some(level) = violation_level {
                self.generate_threshold_alert(measurement, threshold, level).await?;
            }
        }

        Ok(())
    }

    /// Check if a value violates a threshold
    fn check_threshold_violation(&self, value: f64, threshold: &PerformanceThreshold) -> Option<AlertLevel> {
        let violates = |threshold_value: f64| -> bool {
            match threshold.comparison_operator {
                ComparisonOperator::GreaterThan => value > threshold_value,
                ComparisonOperator::LessThan => value < threshold_value,
                ComparisonOperator::GreaterThanOrEqual => value >= threshold_value,
                ComparisonOperator::LessThanOrEqual => value <= threshold_value,
                ComparisonOperator::Equal => (value - threshold_value).abs() < f64::EPSILON,
                ComparisonOperator::NotEqual => (value - threshold_value).abs() >= f64::EPSILON,
            }
        };

        if violates(threshold.emergency_threshold) {
            Some(AlertLevel::Emergency)
        } else if violates(threshold.critical_threshold) {
            Some(AlertLevel::Critical)
        } else if violates(threshold.warning_threshold) {
            Some(AlertLevel::Warning)
        } else {
            None
        }
    }

    /// Generate threshold alert
    async fn generate_threshold_alert(
        &self,
        measurement: &PerformanceMeasurement,
        threshold: &PerformanceThreshold,
        level: AlertLevel,
    ) -> QarResult<()> {
        let alert_id = Uuid::new_v4().to_string();
        let threshold_value = match level {
            AlertLevel::Emergency => threshold.emergency_threshold,
            AlertLevel::Critical => threshold.critical_threshold,
            AlertLevel::Warning => threshold.warning_threshold,
            AlertLevel::Info => threshold.warning_threshold,
        };

        let alert = PerformanceAlert {
            id: alert_id.clone(),
            component: measurement.component.clone(),
            metric_type: measurement.metric_type.clone(),
            level,
            message: format!(
                "{:?} threshold violated for {} in component {}",
                measurement.metric_type, measurement.value, measurement.component
            ),
            current_value: measurement.value,
            threshold_value,
            timestamp: Utc::now(),
            acknowledged: false,
            resolved: false,
        };

        // Check cooldown
        if self.is_alert_in_cooldown(&alert).await? {
            return Ok(());
        }

        // Store active alert
        {
            let mut active_alerts = self.active_alerts.write().await;
            active_alerts.insert(alert_id, alert.clone());
        }

        // Add to history
        {
            let mut alert_history = self.alert_history.write().await;
            alert_history.push(alert);
        }

        Ok(())
    }

    /// Generate anomaly alert
    async fn generate_anomaly_alert(&self, measurement: &PerformanceMeasurement, anomaly_score: f64) -> QarResult<()> {
        let alert_id = Uuid::new_v4().to_string();
        
        let level = if anomaly_score > 0.95 {
            AlertLevel::Critical
        } else if anomaly_score > 0.9 {
            AlertLevel::Warning
        } else {
            AlertLevel::Info
        };

        let alert = PerformanceAlert {
            id: alert_id.clone(),
            component: measurement.component.clone(),
            metric_type: measurement.metric_type.clone(),
            level,
            message: format!(
                "Anomaly detected in {:?} for component {} (score: {:.2})",
                measurement.metric_type, measurement.component, anomaly_score
            ),
            current_value: measurement.value,
            threshold_value: anomaly_score,
            timestamp: Utc::now(),
            acknowledged: false,
            resolved: false,
        };

        let mut active_alerts = self.active_alerts.write().await;
        active_alerts.insert(alert_id, alert.clone());

        let mut alert_history = self.alert_history.write().await;
        alert_history.push(alert);

        Ok(())
    }

    /// Check if alert is in cooldown period
    async fn is_alert_in_cooldown(&self, alert: &PerformanceAlert) -> QarResult<bool> {
        let cooldown_duration = Duration::minutes(self.config.alert_cooldown_minutes as i64);
        let cutoff_time = Utc::now() - cooldown_duration;

        let alert_history = self.alert_history.read().await;
        
        for historical_alert in alert_history.iter().rev() {
            if historical_alert.timestamp < cutoff_time {
                break;
            }
            
            if historical_alert.component == alert.component &&
               historical_alert.metric_type == alert.metric_type &&
               historical_alert.level == alert.level {
                return Ok(true);
            }
        }

        Ok(false)
    }

    /// Add performance threshold
    pub async fn add_threshold(&self, threshold: PerformanceThreshold) -> QarResult<()> {
        let mut thresholds = self.thresholds.write().await;
        thresholds.push(threshold);
        Ok(())
    }

    /// Get statistics for a component and metric type
    pub async fn get_statistics(&self, component: &str, metric_type: &MetricType) -> QarResult<Option<PerformanceStats>> {
        let component_key = format!("{}::{:?}", component, metric_type);
        let statistics = self.statistics.read().await;
        Ok(statistics.get(&component_key).cloned())
    }

    /// Get active alerts
    pub async fn get_active_alerts(&self) -> QarResult<Vec<PerformanceAlert>> {
        let active_alerts = self.active_alerts.read().await;
        Ok(active_alerts.values().cloned().collect())
    }

    /// Acknowledge alert
    pub async fn acknowledge_alert(&self, alert_id: &str) -> QarResult<()> {
        let mut active_alerts = self.active_alerts.write().await;
        if let Some(alert) = active_alerts.get_mut(alert_id) {
            alert.acknowledged = true;
        }
        Ok(())
    }

    /// Resolve alert
    pub async fn resolve_alert(&self, alert_id: &str) -> QarResult<()> {
        let mut active_alerts = self.active_alerts.write().await;
        if let Some(mut alert) = active_alerts.remove(alert_id) {
            alert.resolved = true;
            
            let mut alert_history = self.alert_history.write().await;
            alert_history.push(alert);
        }
        Ok(())
    }

    /// Generate performance report
    pub async fn generate_report(&self, period_start: DateTime<Utc>, period_end: DateTime<Utc>) -> QarResult<PerformanceReport> {
        let statistics = self.statistics.read().await;
        let alert_history = self.alert_history.read().await;
        
        let mut component_summaries = Vec::new();
        let mut components = HashMap::new();
        
        // Group statistics by component
        for (key, stats) in statistics.iter() {
            let component = &stats.component;
            components.entry(component.clone()).or_insert_with(Vec::new).push(stats.clone());
        }
        
        // Calculate health scores and create summaries
        let health_calculator = self.health_calculator.lock().await;
        for (component, stats) in components {
            let health_score = health_calculator.calculate_component_health(&stats, &alert_history);
            
            let component_summary = ComponentSummary {
                component: component.clone(),
                health_score,
                key_metrics: stats,
                issues: Vec::new(), // Would be populated with actual analysis
                improvements: Vec::new(), // Would be populated with recommendations
            };
            
            component_summaries.push(component_summary);
        }
        
        // Calculate overall health score
        let overall_health_score = if component_summaries.is_empty() {
            1.0
        } else {
            component_summaries.iter().map(|cs| cs.health_score).sum::<f64>() / component_summaries.len() as f64
        };
        
        // Generate alerts summary
        let period_alerts: Vec<&PerformanceAlert> = alert_history
            .iter()
            .filter(|a| a.timestamp >= period_start && a.timestamp <= period_end)
            .collect();
        
        let alerts_summary = AlertsSummary {
            total_alerts: period_alerts.len() as u64,
            critical_alerts: period_alerts.iter().filter(|a| a.level == AlertLevel::Critical || a.level == AlertLevel::Emergency).count() as u64,
            warning_alerts: period_alerts.iter().filter(|a| a.level == AlertLevel::Warning).count() as u64,
            unacknowledged_alerts: period_alerts.iter().filter(|a| !a.acknowledged).count() as u64,
            resolved_alerts: period_alerts.iter().filter(|a| a.resolved).count() as u64,
        };
        
        Ok(PerformanceReport {
            id: Uuid::new_v4().to_string(),
            title: "Quantum Trading Performance Report".to_string(),
            description: "Comprehensive performance analysis for quantum trading operations".to_string(),
            period_start,
            period_end,
            component_summaries,
            overall_health_score,
            recommendations: vec![
                "Consider implementing circuit optimization for better quantum performance".to_string(),
                "Monitor memory usage patterns to prevent resource exhaustion".to_string(),
            ],
            alerts_summary,
            generated_at: Utc::now(),
        })
    }

    /// Clean up old measurements
    pub async fn cleanup_old_measurements(&self) -> QarResult<usize> {
        let cutoff_time = Utc::now() - Duration::days(self.config.retention_period_days as i64);
        let mut cleaned_count = 0;
        
        let mut measurements = self.measurements.write().await;
        for (_, component_measurements) in measurements.iter_mut() {
            let original_len = component_measurements.len();
            component_measurements.retain(|m| m.timestamp > cutoff_time);
            cleaned_count += original_len - component_measurements.len();
        }
        
        Ok(cleaned_count)
    }
}

impl HealthCalculator {
    fn calculate_component_health(&self, stats: &[PerformanceStats], _alerts: &[PerformanceAlert]) -> f64 {
        if stats.is_empty() {
            return 1.0;
        }
        
        // Simplified health calculation
        let mut health_score = 1.0;
        
        for stat in stats {
            // Simple heuristic: penalize high variance and extreme values
            let variance_penalty = (stat.std_dev / stat.mean.max(1.0)).min(0.5);
            health_score -= variance_penalty * 0.1;
        }
        
        health_score.max(0.0).min(1.0)
    }
}

/// Mock implementations for testing
pub struct MockAnomalyDetector;

#[async_trait::async_trait]
impl AnomalyDetector for MockAnomalyDetector {
    async fn detect_anomalies(&self, _measurements: &[PerformanceMeasurement]) -> QarResult<Vec<Anomaly>> {
        Ok(Vec::new())
    }

    async fn update_model(&self, _measurements: &[PerformanceMeasurement]) -> QarResult<()> {
        Ok(())
    }

    async fn get_anomaly_score(&self, _measurement: &PerformanceMeasurement) -> QarResult<f64> {
        Ok(0.1) // Low anomaly score
    }
}

pub struct MockTrendAnalyzer;

#[async_trait::async_trait]
impl TrendAnalyzer for MockTrendAnalyzer {
    async fn analyze_trend(&self, measurements: &[PerformanceMeasurement]) -> QarResult<TrendDirection> {
        if measurements.len() < 2 {
            return Ok(TrendDirection::Stable);
        }
        
        let first_half_avg = measurements[..measurements.len()/2].iter().map(|m| m.value).sum::<f64>() / (measurements.len()/2) as f64;
        let second_half_avg = measurements[measurements.len()/2..].iter().map(|m| m.value).sum::<f64>() / (measurements.len() - measurements.len()/2) as f64;
        
        if second_half_avg > first_half_avg * 1.1 {
            Ok(TrendDirection::Increasing)
        } else if second_half_avg < first_half_avg * 0.9 {
            Ok(TrendDirection::Decreasing)
        } else {
            Ok(TrendDirection::Stable)
        }
    }

    async fn predict_future_values(&self, _measurements: &[PerformanceMeasurement], steps: usize) -> QarResult<Vec<f64>> {
        Ok(vec![1.0; steps]) // Simplified prediction
    }

    async fn calculate_seasonality(&self, _measurements: &[PerformanceMeasurement]) -> QarResult<Vec<f64>> {
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_monitor() -> PerformanceMonitor {
        let config = PerformanceMonitorConfig {
            sampling_interval_seconds: 60,
            retention_period_days: 30,
            aggregation_window_minutes: 5,
            enable_real_time_alerts: true,
            enable_predictive_analytics: true,
            enable_anomaly_detection: true,
            alert_cooldown_minutes: 15,
            max_measurements_per_component: 10000,
            enable_auto_scaling_recommendations: true,
        };

        PerformanceMonitor::new(
            config,
            Arc::new(MockAnomalyDetector),
            Arc::new(MockTrendAnalyzer),
        )
    }

    #[tokio::test]
    async fn test_record_measurement() {
        let monitor = create_test_monitor();
        
        let measurement = PerformanceMeasurement {
            id: String::new(),
            component: "test_component".to_string(),
            metric_type: MetricType::Latency,
            value: 100.0,
            unit: "ms".to_string(),
            timestamp: Utc::now(),
            tags: HashMap::new(),
            metadata: HashMap::new(),
        };

        monitor.record_measurement(measurement).await.unwrap();

        let stats = monitor.get_statistics("test_component", &MetricType::Latency).await.unwrap();
        assert!(stats.is_some());
        assert_eq!(stats.unwrap().count, 1);
    }

    #[tokio::test]
    async fn test_threshold_alerts() {
        let monitor = create_test_monitor();

        let threshold = PerformanceThreshold {
            metric_type: MetricType::Latency,
            component: None,
            warning_threshold: 50.0,
            critical_threshold: 100.0,
            emergency_threshold: 200.0,
            comparison_operator: ComparisonOperator::GreaterThan,
            enabled: true,
        };

        monitor.add_threshold(threshold).await.unwrap();

        let measurement = PerformanceMeasurement {
            id: String::new(),
            component: "test_component".to_string(),
            metric_type: MetricType::Latency,
            value: 150.0, // Above critical threshold
            unit: "ms".to_string(),
            timestamp: Utc::now(),
            tags: HashMap::new(),
            metadata: HashMap::new(),
        };

        monitor.record_measurement(measurement).await.unwrap();

        let alerts = monitor.get_active_alerts().await.unwrap();
        assert!(!alerts.is_empty());
        assert_eq!(alerts[0].level, AlertLevel::Critical);
    }

    #[tokio::test]
    async fn test_statistics_calculation() {
        let monitor = create_test_monitor();

        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        for value in values {
            let measurement = PerformanceMeasurement {
                id: String::new(),
                component: "test_component".to_string(),
                metric_type: MetricType::Throughput,
                value,
                unit: "req/s".to_string(),
                timestamp: Utc::now(),
                tags: HashMap::new(),
                metadata: HashMap::new(),
            };
            monitor.record_measurement(measurement).await.unwrap();
        }

        let stats = monitor.get_statistics("test_component", &MetricType::Throughput).await.unwrap();
        assert!(stats.is_some());
        
        let stats = stats.unwrap();
        assert_eq!(stats.count, 5);
        assert_eq!(stats.min, 10.0);
        assert_eq!(stats.max, 50.0);
        assert_eq!(stats.mean, 30.0);
        assert_eq!(stats.median, 30.0);
    }

    #[tokio::test]
    async fn test_alert_acknowledgment() {
        let monitor = create_test_monitor();

        let threshold = PerformanceThreshold {
            metric_type: MetricType::ErrorRate,
            component: None,
            warning_threshold: 0.05,
            critical_threshold: 0.1,
            emergency_threshold: 0.2,
            comparison_operator: ComparisonOperator::GreaterThan,
            enabled: true,
        };

        monitor.add_threshold(threshold).await.unwrap();

        let measurement = PerformanceMeasurement {
            id: String::new(),
            component: "test_component".to_string(),
            metric_type: MetricType::ErrorRate,
            value: 0.15,
            unit: "rate".to_string(),
            timestamp: Utc::now(),
            tags: HashMap::new(),
            metadata: HashMap::new(),
        };

        monitor.record_measurement(measurement).await.unwrap();

        let alerts = monitor.get_active_alerts().await.unwrap();
        assert!(!alerts.is_empty());
        
        let alert_id = &alerts[0].id;
        monitor.acknowledge_alert(alert_id).await.unwrap();

        let updated_alerts = monitor.get_active_alerts().await.unwrap();
        assert!(updated_alerts[0].acknowledged);
    }

    #[tokio::test]
    async fn test_performance_report() {
        let monitor = create_test_monitor();

        let measurement = PerformanceMeasurement {
            id: String::new(),
            component: "test_component".to_string(),
            metric_type: MetricType::TradingReturn,
            value: 0.05,
            unit: "percentage".to_string(),
            timestamp: Utc::now(),
            tags: HashMap::new(),
            metadata: HashMap::new(),
        };

        monitor.record_measurement(measurement).await.unwrap();

        let report = monitor.generate_report(
            Utc::now() - Duration::hours(1),
            Utc::now(),
        ).await.unwrap();

        assert!(!report.id.is_empty());
        assert!(!report.component_summaries.is_empty());
        assert!(report.overall_health_score >= 0.0 && report.overall_health_score <= 1.0);
    }
}