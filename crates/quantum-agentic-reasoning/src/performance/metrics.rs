//! Performance Metrics Module
//!
//! Comprehensive performance tracking for quantum trading operations with real-time metrics collection.

use crate::core::{QarResult, FactorMap};
use crate::error::QarError;
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque, BTreeMap};
use std::sync::Arc;
use tokio::sync::{Mutex, RwLock};
use chrono::{DateTime, Utc, Duration};
use uuid::Uuid;

/// Performance metric types
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, Hash)]
pub enum MetricType {
    // Execution metrics
    ExecutionTime,
    ThroughputQps,
    LatencyMs,
    CpuUsage,
    MemoryUsage,
    
    // Quantum-specific metrics
    CircuitDepth,
    GateCount,
    QubitUtilization,
    FidelityScore,
    DecoherenceTime,
    
    // Trading metrics
    DecisionLatency,
    SignalStrength,
    PredictionAccuracy,
    RiskScore,
    PortfolioBalance,
    
    // System metrics
    CacheHitRate,
    ErrorRate,
    SuccessRate,
    ResourceUtilization,
    
    // Custom metrics
    Custom(String),
}

/// Metric aggregation types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AggregationType {
    Sum,
    Average,
    Min,
    Max,
    Count,
    Percentile(f64),
    StandardDeviation,
    Rate,
    Histogram,
}

/// Time window for metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum TimeWindow {
    RealTime,
    Last1Minute,
    Last5Minutes,
    Last15Minutes,
    LastHour,
    Last24Hours,
    LastWeek,
    Custom(Duration),
}

/// Metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDataPoint {
    pub timestamp: DateTime<Utc>,
    pub value: f64,
    pub tags: HashMap<String, String>,
    pub metadata: HashMap<String, String>,
}

/// Aggregated metric result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AggregatedMetric {
    pub metric_type: MetricType,
    pub aggregation_type: AggregationType,
    pub window: TimeWindow,
    pub value: f64,
    pub count: u64,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub percentiles: Option<HashMap<String, f64>>,
    pub histogram: Option<Vec<(f64, u64)>>,
}

/// Performance threshold
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceThreshold {
    pub metric_type: MetricType,
    pub warning_threshold: f64,
    pub critical_threshold: f64,
    pub operator: ThresholdOperator,
    pub enabled: bool,
}

/// Threshold operators
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ThresholdOperator {
    GreaterThan,
    LessThan,
    GreaterThanOrEqual,
    LessThanOrEqual,
    Equal,
    NotEqual,
}

/// Performance alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceAlert {
    pub id: String,
    pub metric_type: MetricType,
    pub severity: AlertSeverity,
    pub message: String,
    pub current_value: f64,
    pub threshold_value: f64,
    pub triggered_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
    pub tags: HashMap<String, String>,
}

/// Alert severity levels
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
}

/// Performance dashboard data
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceDashboard {
    pub quantum_metrics: HashMap<MetricType, AggregatedMetric>,
    pub trading_metrics: HashMap<MetricType, AggregatedMetric>,
    pub system_metrics: HashMap<MetricType, AggregatedMetric>,
    pub active_alerts: Vec<PerformanceAlert>,
    pub historical_trends: HashMap<MetricType, Vec<MetricDataPoint>>,
    pub generated_at: DateTime<Utc>,
}

/// Metric collector configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsConfig {
    pub collection_interval_seconds: u64,
    pub retention_duration_hours: u64,
    pub max_data_points_per_metric: usize,
    pub enable_real_time_aggregation: bool,
    pub enable_alerting: bool,
    pub export_to_prometheus: bool,
    pub export_to_influxdb: bool,
    pub batch_size: usize,
}

/// Performance metrics manager
#[derive(Debug)]
pub struct PerformanceMetricsManager {
    config: MetricsConfig,
    metrics_data: Arc<RwLock<HashMap<MetricType, VecDeque<MetricDataPoint>>>>,
    aggregated_metrics: Arc<RwLock<HashMap<(MetricType, TimeWindow), AggregatedMetric>>>,
    thresholds: Arc<RwLock<Vec<PerformanceThreshold>>>,
    active_alerts: Arc<RwLock<HashMap<String, PerformanceAlert>>>,
    collectors: Arc<RwLock<Vec<Arc<dyn MetricCollector + Send + Sync>>>>,
    exporters: Arc<RwLock<Vec<Arc<dyn MetricExporter + Send + Sync>>>>,
    alert_handlers: Arc<RwLock<Vec<Arc<dyn AlertHandler + Send + Sync>>>>,
}

/// Metric collector trait
#[async_trait::async_trait]
pub trait MetricCollector {
    async fn collect_metrics(&self) -> QarResult<Vec<(MetricType, MetricDataPoint)>>;
    fn get_collector_name(&self) -> String;
    fn get_collection_interval(&self) -> Duration;
}

/// Metric exporter trait
#[async_trait::async_trait]
pub trait MetricExporter {
    async fn export_metrics(&self, metrics: &[AggregatedMetric]) -> QarResult<()>;
    fn get_exporter_name(&self) -> String;
}

/// Alert handler trait
#[async_trait::async_trait]
pub trait AlertHandler {
    async fn handle_alert(&self, alert: &PerformanceAlert) -> QarResult<()>;
    fn get_handler_name(&self) -> String;
}

impl PerformanceMetricsManager {
    /// Create new performance metrics manager
    pub fn new(config: MetricsConfig) -> Self {
        Self {
            config,
            metrics_data: Arc::new(RwLock::new(HashMap::new())),
            aggregated_metrics: Arc::new(RwLock::new(HashMap::new())),
            thresholds: Arc::new(RwLock::new(Vec::new())),
            active_alerts: Arc::new(RwLock::new(HashMap::new())),
            collectors: Arc::new(RwLock::new(Vec::new())),
            exporters: Arc::new(RwLock::new(Vec::new())),
            alert_handlers: Arc::new(RwLock::new(Vec::new())),
        }
    }

    /// Record metric data point
    pub async fn record_metric(
        &self,
        metric_type: MetricType,
        value: f64,
        tags: Option<HashMap<String, String>>,
    ) -> QarResult<()> {
        let data_point = MetricDataPoint {
            timestamp: Utc::now(),
            value,
            tags: tags.unwrap_or_default(),
            metadata: HashMap::new(),
        };

        {
            let mut metrics = self.metrics_data.write().await;
            let metric_queue = metrics.entry(metric_type.clone()).or_insert_with(VecDeque::new);
            
            metric_queue.push_back(data_point.clone());
            
            // Limit queue size
            while metric_queue.len() > self.config.max_data_points_per_metric {
                metric_queue.pop_front();
            }
        }

        // Real-time aggregation if enabled
        if self.config.enable_real_time_aggregation {
            self.update_real_time_aggregates(&metric_type).await?;
        }

        // Check thresholds if alerting is enabled
        if self.config.enable_alerting {
            self.check_thresholds(&metric_type, &data_point).await?;
        }

        Ok(())
    }

    /// Get aggregated metric for time window
    pub async fn get_aggregated_metric(
        &self,
        metric_type: &MetricType,
        aggregation: &AggregationType,
        window: &TimeWindow,
    ) -> QarResult<Option<AggregatedMetric>> {
        let metrics = self.metrics_data.read().await;
        
        if let Some(data_points) = metrics.get(metric_type) {
            let filtered_points = self.filter_by_time_window(data_points, window);
            
            if filtered_points.is_empty() {
                return Ok(None);
            }

            let aggregate = self.compute_aggregation(&filtered_points, aggregation, window.clone())?;
            Ok(Some(AggregatedMetric {
                metric_type: metric_type.clone(),
                aggregation_type: aggregation.clone(),
                window: window.clone(),
                value: aggregate.value,
                count: aggregate.count,
                start_time: aggregate.start_time,
                end_time: aggregate.end_time,
                percentiles: aggregate.percentiles,
                histogram: aggregate.histogram,
            }))
        } else {
            Ok(None)
        }
    }

    /// Filter data points by time window
    fn filter_by_time_window(
        &self,
        data_points: &VecDeque<MetricDataPoint>,
        window: &TimeWindow,
    ) -> Vec<MetricDataPoint> {
        let now = Utc::now();
        let cutoff_time = match window {
            TimeWindow::RealTime => return data_points.iter().cloned().collect(),
            TimeWindow::Last1Minute => now - Duration::minutes(1),
            TimeWindow::Last5Minutes => now - Duration::minutes(5),
            TimeWindow::Last15Minutes => now - Duration::minutes(15),
            TimeWindow::LastHour => now - Duration::hours(1),
            TimeWindow::Last24Hours => now - Duration::hours(24),
            TimeWindow::LastWeek => now - Duration::weeks(1),
            TimeWindow::Custom(duration) => now - *duration,
        };

        data_points
            .iter()
            .filter(|point| point.timestamp >= cutoff_time)
            .cloned()
            .collect()
    }

    /// Compute aggregation from data points
    fn compute_aggregation(
        &self,
        points: &[MetricDataPoint],
        aggregation: &AggregationType,
        window: TimeWindow,
    ) -> QarResult<AggregatedMetric> {
        if points.is_empty() {
            return Err(QarError::ValidationError("No data points for aggregation".to_string()));
        }

        let values: Vec<f64> = points.iter().map(|p| p.value).collect();
        let count = values.len() as u64;
        let start_time = points.iter().min_by_key(|p| p.timestamp).unwrap().timestamp;
        let end_time = points.iter().max_by_key(|p| p.timestamp).unwrap().timestamp;

        let value = match aggregation {
            AggregationType::Sum => values.iter().sum(),
            AggregationType::Average => values.iter().sum::<f64>() / values.len() as f64,
            AggregationType::Min => values.iter().fold(f64::INFINITY, |a, &b| a.min(b)),
            AggregationType::Max => values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b)),
            AggregationType::Count => count as f64,
            AggregationType::Percentile(p) => self.calculate_percentile(&values, *p),
            AggregationType::StandardDeviation => self.calculate_std_dev(&values),
            AggregationType::Rate => {
                // Calculate rate per second
                if start_time == end_time {
                    count as f64
                } else {
                    let duration_seconds = (end_time - start_time).num_seconds() as f64;
                    count as f64 / duration_seconds
                }
            },
            AggregationType::Histogram => {
                // Return count for histogram (histogram data computed separately)
                count as f64
            },
        };

        let percentiles = if matches!(aggregation, AggregationType::Histogram) {
            Some([
                ("p50".to_string(), self.calculate_percentile(&values, 50.0)),
                ("p90".to_string(), self.calculate_percentile(&values, 90.0)),
                ("p95".to_string(), self.calculate_percentile(&values, 95.0)),
                ("p99".to_string(), self.calculate_percentile(&values, 99.0)),
            ].into_iter().collect())
        } else {
            None
        };

        let histogram = if matches!(aggregation, AggregationType::Histogram) {
            Some(self.calculate_histogram(&values))
        } else {
            None
        };

        Ok(AggregatedMetric {
            metric_type: MetricType::Custom("placeholder".to_string()), // Will be set by caller
            aggregation_type: aggregation.clone(),
            window,
            value,
            count,
            start_time,
            end_time,
            percentiles,
            histogram,
        })
    }

    /// Calculate percentile
    fn calculate_percentile(&self, values: &[f64], percentile: f64) -> f64 {
        if values.is_empty() {
            return 0.0;
        }
        
        let mut sorted_values = values.to_vec();
        sorted_values.sort_by(|a, b| a.partial_cmp(b).unwrap());
        
        let index = (percentile / 100.0 * (sorted_values.len() - 1) as f64).round() as usize;
        sorted_values[index.min(sorted_values.len() - 1)]
    }

    /// Calculate standard deviation
    fn calculate_std_dev(&self, values: &[f64]) -> f64 {
        if values.len() <= 1 {
            return 0.0;
        }

        let mean = values.iter().sum::<f64>() / values.len() as f64;
        let variance = values.iter()
            .map(|x| (x - mean).powi(2))
            .sum::<f64>() / (values.len() - 1) as f64;
        
        variance.sqrt()
    }

    /// Calculate histogram
    fn calculate_histogram(&self, values: &[f64]) -> Vec<(f64, u64)> {
        if values.is_empty() {
            return Vec::new();
        }

        let min_val = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
        let max_val = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
        
        if min_val == max_val {
            return vec![(min_val, values.len() as u64)];
        }

        let num_buckets = 20;
        let bucket_size = (max_val - min_val) / num_buckets as f64;
        let mut buckets = vec![0u64; num_buckets];

        for &value in values {
            let bucket_index = ((value - min_val) / bucket_size).floor() as usize;
            let bucket_index = bucket_index.min(num_buckets - 1);
            buckets[bucket_index] += 1;
        }

        buckets.into_iter()
            .enumerate()
            .map(|(i, count)| {
                let bucket_start = min_val + i as f64 * bucket_size;
                (bucket_start, count)
            })
            .collect()
    }

    /// Update real-time aggregates
    async fn update_real_time_aggregates(&self, metric_type: &MetricType) -> QarResult<()> {
        let windows = vec![
            TimeWindow::Last1Minute,
            TimeWindow::Last5Minutes,
            TimeWindow::Last15Minutes,
            TimeWindow::LastHour,
        ];

        let aggregations = vec![
            AggregationType::Average,
            AggregationType::Min,
            AggregationType::Max,
            AggregationType::Count,
        ];

        for window in windows {
            for aggregation in &aggregations {
                if let Some(aggregate) = self.get_aggregated_metric(metric_type, aggregation, &window).await? {
                    let mut aggregated = self.aggregated_metrics.write().await;
                    aggregated.insert((metric_type.clone(), window.clone()), aggregate);
                }
            }
        }

        Ok(())
    }

    /// Check thresholds and generate alerts
    async fn check_thresholds(
        &self,
        metric_type: &MetricType,
        data_point: &MetricDataPoint,
    ) -> QarResult<()> {
        let thresholds = self.thresholds.read().await;
        
        for threshold in thresholds.iter() {
            if !threshold.enabled || threshold.metric_type != *metric_type {
                continue;
            }

            let should_alert = match threshold.operator {
                ThresholdOperator::GreaterThan => data_point.value > threshold.critical_threshold,
                ThresholdOperator::LessThan => data_point.value < threshold.critical_threshold,
                ThresholdOperator::GreaterThanOrEqual => data_point.value >= threshold.critical_threshold,
                ThresholdOperator::LessThanOrEqual => data_point.value <= threshold.critical_threshold,
                ThresholdOperator::Equal => (data_point.value - threshold.critical_threshold).abs() < f64::EPSILON,
                ThresholdOperator::NotEqual => (data_point.value - threshold.critical_threshold).abs() >= f64::EPSILON,
            };

            if should_alert {
                let alert = PerformanceAlert {
                    id: Uuid::new_v4().to_string(),
                    metric_type: metric_type.clone(),
                    severity: AlertSeverity::Critical,
                    message: format!(
                        "Metric {} {} threshold {} (current: {})",
                        format!("{:?}", metric_type),
                        match threshold.operator {
                            ThresholdOperator::GreaterThan => "exceeded",
                            ThresholdOperator::LessThan => "below",
                            _ => "violated",
                        },
                        threshold.critical_threshold,
                        data_point.value
                    ),
                    current_value: data_point.value,
                    threshold_value: threshold.critical_threshold,
                    triggered_at: Utc::now(),
                    resolved_at: None,
                    tags: data_point.tags.clone(),
                };

                self.trigger_alert(alert).await?;
            }
        }

        Ok(())
    }

    /// Trigger alert
    async fn trigger_alert(&self, alert: PerformanceAlert) -> QarResult<()> {
        // Store alert
        {
            let mut alerts = self.active_alerts.write().await;
            alerts.insert(alert.id.clone(), alert.clone());
        }

        // Notify alert handlers
        let handlers = self.alert_handlers.read().await;
        for handler in handlers.iter() {
            if let Err(e) = handler.handle_alert(&alert).await {
                log::error!("Alert handler {} failed: {}", handler.get_handler_name(), e);
            }
        }

        Ok(())
    }

    /// Add performance threshold
    pub async fn add_threshold(&self, threshold: PerformanceThreshold) -> QarResult<()> {
        let mut thresholds = self.thresholds.write().await;
        thresholds.push(threshold);
        Ok(())
    }

    /// Register metric collector
    pub async fn register_collector(
        &self,
        collector: Arc<dyn MetricCollector + Send + Sync>,
    ) -> QarResult<()> {
        let mut collectors = self.collectors.write().await;
        collectors.push(collector);
        Ok(())
    }

    /// Register metric exporter
    pub async fn register_exporter(
        &self,
        exporter: Arc<dyn MetricExporter + Send + Sync>,
    ) -> QarResult<()> {
        let mut exporters = self.exporters.write().await;
        exporters.push(exporter);
        Ok(())
    }

    /// Register alert handler
    pub async fn register_alert_handler(
        &self,
        handler: Arc<dyn AlertHandler + Send + Sync>,
    ) -> QarResult<()> {
        let mut handlers = self.alert_handlers.write().await;
        handlers.push(handler);
        Ok(())
    }

    /// Generate performance dashboard
    pub async fn generate_dashboard(&self) -> QarResult<PerformanceDashboard> {
        let quantum_metrics = self.get_metrics_by_category(&[
            MetricType::CircuitDepth,
            MetricType::GateCount,
            MetricType::QubitUtilization,
            MetricType::FidelityScore,
            MetricType::DecoherenceTime,
        ]).await?;

        let trading_metrics = self.get_metrics_by_category(&[
            MetricType::DecisionLatency,
            MetricType::SignalStrength,
            MetricType::PredictionAccuracy,
            MetricType::RiskScore,
            MetricType::PortfolioBalance,
        ]).await?;

        let system_metrics = self.get_metrics_by_category(&[
            MetricType::ExecutionTime,
            MetricType::ThroughputQps,
            MetricType::LatencyMs,
            MetricType::CpuUsage,
            MetricType::MemoryUsage,
            MetricType::CacheHitRate,
            MetricType::ErrorRate,
            MetricType::SuccessRate,
        ]).await?;

        let active_alerts = {
            let alerts = self.active_alerts.read().await;
            alerts.values().cloned().collect()
        };

        let historical_trends = self.get_historical_trends().await?;

        Ok(PerformanceDashboard {
            quantum_metrics,
            trading_metrics,
            system_metrics,
            active_alerts,
            historical_trends,
            generated_at: Utc::now(),
        })
    }

    /// Get metrics by category
    async fn get_metrics_by_category(
        &self,
        metric_types: &[MetricType],
    ) -> QarResult<HashMap<MetricType, AggregatedMetric>> {
        let mut metrics = HashMap::new();
        
        for metric_type in metric_types {
            if let Some(aggregate) = self.get_aggregated_metric(
                metric_type,
                &AggregationType::Average,
                &TimeWindow::LastHour,
            ).await? {
                metrics.insert(metric_type.clone(), aggregate);
            }
        }

        Ok(metrics)
    }

    /// Get historical trends
    async fn get_historical_trends(&self) -> QarResult<HashMap<MetricType, Vec<MetricDataPoint>>> {
        let metrics_data = self.metrics_data.read().await;
        let mut trends = HashMap::new();

        for (metric_type, data_points) in metrics_data.iter() {
            // Get last 100 points for trend analysis
            let trend_points: Vec<MetricDataPoint> = data_points
                .iter()
                .rev()
                .take(100)
                .rev()
                .cloned()
                .collect();
            
            if !trend_points.is_empty() {
                trends.insert(metric_type.clone(), trend_points);
            }
        }

        Ok(trends)
    }

    /// Start metric collection
    pub async fn start_collection(&self) -> QarResult<()> {
        let collectors = self.collectors.read().await;
        
        for collector in collectors.iter() {
            let collector_clone = collector.clone();
            let metrics_manager = self.clone_for_collection();
            
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(collector_clone.get_collection_interval().to_std().unwrap());
                
                loop {
                    interval.tick().await;
                    
                    if let Ok(collected_metrics) = collector_clone.collect_metrics().await {
                        for (metric_type, data_point) in collected_metrics {
                            if let Err(e) = metrics_manager.record_metric(
                                metric_type,
                                data_point.value,
                                Some(data_point.tags),
                            ).await {
                                log::error!("Failed to record metric: {}", e);
                            }
                        }
                    }
                }
            });
        }

        Ok(())
    }

    /// Clone for collection tasks
    fn clone_for_collection(&self) -> Self {
        Self {
            config: self.config.clone(),
            metrics_data: self.metrics_data.clone(),
            aggregated_metrics: self.aggregated_metrics.clone(),
            thresholds: self.thresholds.clone(),
            active_alerts: self.active_alerts.clone(),
            collectors: self.collectors.clone(),
            exporters: self.exporters.clone(),
            alert_handlers: self.alert_handlers.clone(),
        }
    }

    /// Export metrics to registered exporters
    pub async fn export_metrics(&self) -> QarResult<()> {
        let aggregated = self.aggregated_metrics.read().await;
        let metrics: Vec<AggregatedMetric> = aggregated.values().cloned().collect();
        
        let exporters = self.exporters.read().await;
        for exporter in exporters.iter() {
            if let Err(e) = exporter.export_metrics(&metrics).await {
                log::error!("Metric exporter {} failed: {}", exporter.get_exporter_name(), e);
            }
        }

        Ok(())
    }
}

/// Mock implementations for testing
pub struct MockMetricCollector {
    name: String,
}

impl MockMetricCollector {
    pub fn new(name: String) -> Self {
        Self { name }
    }
}

#[async_trait::async_trait]
impl MetricCollector for MockMetricCollector {
    async fn collect_metrics(&self) -> QarResult<Vec<(MetricType, MetricDataPoint)>> {
        Ok(vec![
            (MetricType::CpuUsage, MetricDataPoint {
                timestamp: Utc::now(),
                value: rand::random::<f64>() * 100.0,
                tags: HashMap::new(),
                metadata: HashMap::new(),
            }),
            (MetricType::MemoryUsage, MetricDataPoint {
                timestamp: Utc::now(),
                value: rand::random::<f64>() * 100.0,
                tags: HashMap::new(),
                metadata: HashMap::new(),
            }),
        ])
    }

    fn get_collector_name(&self) -> String {
        self.name.clone()
    }

    fn get_collection_interval(&self) -> Duration {
        Duration::seconds(5)
    }
}

pub struct MockMetricExporter;

#[async_trait::async_trait]
impl MetricExporter for MockMetricExporter {
    async fn export_metrics(&self, _metrics: &[AggregatedMetric]) -> QarResult<()> {
        Ok(())
    }

    fn get_exporter_name(&self) -> String {
        "mock_exporter".to_string()
    }
}

pub struct MockAlertHandler;

#[async_trait::async_trait]
impl AlertHandler for MockAlertHandler {
    async fn handle_alert(&self, alert: &PerformanceAlert) -> QarResult<()> {
        log::warn!("Alert: {} - {}", alert.severity as u8, alert.message);
        Ok(())
    }

    fn get_handler_name(&self) -> String {
        "mock_handler".to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_config() -> MetricsConfig {
        MetricsConfig {
            collection_interval_seconds: 5,
            retention_duration_hours: 24,
            max_data_points_per_metric: 1000,
            enable_real_time_aggregation: true,
            enable_alerting: true,
            export_to_prometheus: false,
            export_to_influxdb: false,
            batch_size: 100,
        }
    }

    #[tokio::test]
    async fn test_metric_recording() {
        let manager = PerformanceMetricsManager::new(create_test_config());
        
        manager.record_metric(
            MetricType::CpuUsage,
            75.5,
            Some([("component".to_string(), "quantum_engine".to_string())].into_iter().collect()),
        ).await.unwrap();

        let aggregate = manager.get_aggregated_metric(
            &MetricType::CpuUsage,
            &AggregationType::Average,
            &TimeWindow::Last1Minute,
        ).await.unwrap();

        assert!(aggregate.is_some());
        let agg = aggregate.unwrap();
        assert_eq!(agg.value, 75.5);
        assert_eq!(agg.count, 1);
    }

    #[tokio::test]
    async fn test_threshold_alerting() {
        let manager = PerformanceMetricsManager::new(create_test_config());
        
        let threshold = PerformanceThreshold {
            metric_type: MetricType::CpuUsage,
            warning_threshold: 70.0,
            critical_threshold: 90.0,
            operator: ThresholdOperator::GreaterThan,
            enabled: true,
        };

        manager.add_threshold(threshold).await.unwrap();

        // This should trigger an alert
        manager.record_metric(MetricType::CpuUsage, 95.0, None).await.unwrap();

        let alerts = manager.active_alerts.read().await;
        assert!(!alerts.is_empty());
    }

    #[tokio::test]
    async fn test_aggregation_types() {
        let manager = PerformanceMetricsManager::new(create_test_config());
        
        let values = vec![10.0, 20.0, 30.0, 40.0, 50.0];
        for value in values {
            manager.record_metric(MetricType::LatencyMs, value, None).await.unwrap();
        }

        // Test average
        let avg = manager.get_aggregated_metric(
            &MetricType::LatencyMs,
            &AggregationType::Average,
            &TimeWindow::Last1Minute,
        ).await.unwrap().unwrap();
        assert_eq!(avg.value, 30.0);

        // Test min/max
        let min = manager.get_aggregated_metric(
            &MetricType::LatencyMs,
            &AggregationType::Min,
            &TimeWindow::Last1Minute,
        ).await.unwrap().unwrap();
        assert_eq!(min.value, 10.0);

        let max = manager.get_aggregated_metric(
            &MetricType::LatencyMs,
            &AggregationType::Max,
            &TimeWindow::Last1Minute,
        ).await.unwrap().unwrap();
        assert_eq!(max.value, 50.0);
    }

    #[tokio::test]
    async fn test_dashboard_generation() {
        let manager = PerformanceMetricsManager::new(create_test_config());
        
        // Record some test metrics
        manager.record_metric(MetricType::CircuitDepth, 10.0, None).await.unwrap();
        manager.record_metric(MetricType::DecisionLatency, 50.0, None).await.unwrap();
        manager.record_metric(MetricType::CpuUsage, 75.0, None).await.unwrap();

        let dashboard = manager.generate_dashboard().await.unwrap();
        
        assert!(!dashboard.quantum_metrics.is_empty() || 
                !dashboard.trading_metrics.is_empty() || 
                !dashboard.system_metrics.is_empty());
    }

    #[tokio::test]
    async fn test_percentile_calculation() {
        let manager = PerformanceMetricsManager::new(create_test_config());
        let values = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
        
        assert_eq!(manager.calculate_percentile(&values, 50.0), 5.0);
        assert_eq!(manager.calculate_percentile(&values, 90.0), 9.0);
        assert_eq!(manager.calculate_percentile(&values, 100.0), 10.0);
    }

    #[tokio::test]
    async fn test_collector_registration() {
        let manager = PerformanceMetricsManager::new(create_test_config());
        let collector = Arc::new(MockMetricCollector::new("test_collector".to_string()));
        
        manager.register_collector(collector).await.unwrap();
        
        let collectors = manager.collectors.read().await;
        assert_eq!(collectors.len(), 1);
        assert_eq!(collectors[0].get_collector_name(), "test_collector");
    }
}