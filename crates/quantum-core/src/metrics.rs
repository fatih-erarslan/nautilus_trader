//! Quantum Performance Metrics and Monitoring System
//!
//! This module provides comprehensive metrics collection, analysis, and monitoring
//! for quantum computing operations including performance, reliability, and resource usage.

use crate::error::{QuantumError, QuantumResult};
use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, RwLock};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock as AsyncRwLock;
use tracing::{info, error};
use chrono::{DateTime, Utc, Duration};


/// Metric types supported by the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetricType {
    Counter,
    Histogram,
    Gauge,
    Summary,
    Rate,
    Percentile,
}

/// Metric aggregation methods
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AggregationMethod {
    Sum,
    Average,
    Min,
    Max,
    Count,
    Median,
    Percentile(u8),
    StdDev,
    Variance,
}

/// Time window for metric aggregation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TimeWindow {
    Second,
    Minute,
    Hour,
    Day,
    Week,
    Month,
    Custom(i64), // Duration in seconds
}

impl TimeWindow {
    /// Convert time window to seconds
    pub fn to_seconds(&self) -> i64 {
        match self {
            TimeWindow::Second => 1,
            TimeWindow::Minute => 60,
            TimeWindow::Hour => 3600,
            TimeWindow::Day => 86400,
            TimeWindow::Week => 604800,
            TimeWindow::Month => 2592000,
            TimeWindow::Custom(seconds) => *seconds,
        }
    }
}

/// Metric data point
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDataPoint {
    /// Timestamp of the measurement
    pub timestamp: DateTime<Utc>,
    /// Measured value
    pub value: f64,
    /// Tags for categorization
    pub tags: HashMap<String, String>,
    /// Additional metadata
    pub metadata: HashMap<String, String>,
}

/// Metric definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricDefinition {
    /// Name of the metric
    pub name: String,
    /// Type of metric (counter, gauge, etc.)
    pub metric_type: MetricType,
    /// Human-readable description
    pub description: String,
    /// Unit of measurement
    pub unit: String,
    /// Aggregation method to use
    pub aggregation_method: AggregationMethod,
    /// Time window for aggregation
    pub time_window: TimeWindow,
    /// How long to retain metric data
    pub retention_period: Duration,
    /// Default tags for this metric
    pub tags: HashMap<String, String>,
    /// Optional thresholds for alerting
    pub thresholds: Option<MetricThresholds>,
}

/// Metric thresholds for alerting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricThresholds {
    /// Warning threshold value
    pub warning: f64,
    /// Critical threshold value
    pub critical: f64,
    /// Target value (optional)
    pub target: Option<f64>,
    /// Direction of threshold checking
    pub direction: ThresholdDirection,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThresholdDirection {
    Above,  // Alert when value goes above threshold
    Below,  // Alert when value goes below threshold
}

/// Quantum-specific metric categories
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumMetrics {
    pub fidelity: f64,
    pub coherence_time_us: f64,
    pub gate_time_ns: f64,
    pub error_rate: f64,
    pub decoherence_rate: f64,
    pub entanglement_measure: f64,
    pub quantum_volume: f64,
    pub gate_count: u64,
    pub depth: u64,
    pub success_probability: f64,
    pub timestamp: DateTime<Utc>,
}

impl Default for QuantumMetrics {
    fn default() -> Self {
        Self {
            fidelity: 1.0,
            coherence_time_us: 100.0,
            gate_time_ns: 50.0,
            error_rate: 0.001,
            decoherence_rate: 0.01,
            entanglement_measure: 0.0,
            quantum_volume: 32.0,
            gate_count: 0,
            depth: 0,
            success_probability: 1.0,
            timestamp: Utc::now(),
        }
    }
}

/// Performance metrics for quantum operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub execution_time_ms: f64,
    pub memory_usage_mb: f64,
    pub cpu_utilization_percent: f64,
    pub gpu_utilization_percent: f64,
    pub throughput_ops_per_second: f64,
    pub latency_ms: f64,
    pub queue_depth: u32,
    pub active_threads: u32,
    pub cache_hit_rate: f64,
    pub network_io_mbps: f64,
    pub disk_io_mbps: f64,
    pub timestamp: DateTime<Utc>,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            execution_time_ms: 0.0,
            memory_usage_mb: 0.0,
            cpu_utilization_percent: 0.0,
            gpu_utilization_percent: 0.0,
            throughput_ops_per_second: 0.0,
            latency_ms: 0.0,
            queue_depth: 0,
            active_threads: 0,
            cache_hit_rate: 0.0,
            network_io_mbps: 0.0,
            disk_io_mbps: 0.0,
            timestamp: Utc::now(),
        }
    }
}

/// Reliability metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliabilityMetrics {
    pub uptime_hours: f64,
    pub availability_percentage: f64,
    pub mtbf_hours: f64,
    pub mttr_hours: f64,
    pub failure_rate: f64,
    pub error_count: u64,
    pub recovery_count: u64,
    pub sla_compliance: f64,
    pub timestamp: DateTime<Utc>,
}

impl Default for ReliabilityMetrics {
    fn default() -> Self {
        Self {
            uptime_hours: 0.0,
            availability_percentage: 100.0,
            mtbf_hours: 720.0, // 30 days
            mttr_hours: 1.0,
            failure_rate: 0.0,
            error_count: 0,
            recovery_count: 0,
            sla_compliance: 100.0,
            timestamp: Utc::now(),
        }
    }
}

/// Aggregated metric result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricAggregation {
    pub name: String,
    pub time_window: TimeWindow,
    pub aggregation_method: AggregationMethod,
    pub value: f64,
    pub count: u64,
    pub min: f64,
    pub max: f64,
    pub start_time: DateTime<Utc>,
    pub end_time: DateTime<Utc>,
    pub tags: HashMap<String, String>,
}

/// Metric alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricAlert {
    pub id: String,
    pub metric_name: String,
    pub threshold_type: ThresholdType,
    pub threshold_value: f64,
    pub current_value: f64,
    pub severity: AlertSeverity,
    pub message: String,
    pub triggered_at: DateTime<Utc>,
    pub resolved_at: Option<DateTime<Utc>>,
    pub tags: HashMap<String, String>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThresholdType {
    Warning,
    Critical,
    Target,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum AlertSeverity {
    Info,
    Warning,
    Critical,
    Fatal,
}

/// Metric storage backend
#[derive(Debug)]
pub struct MetricStorage {
    data_points: Arc<RwLock<HashMap<String, VecDeque<MetricDataPoint>>>>,
    aggregations: Arc<RwLock<HashMap<String, MetricAggregation>>>,
    retention_policies: Arc<RwLock<HashMap<String, Duration>>>,
    max_data_points: usize,
}

impl MetricStorage {
    /// Create new metric storage
    pub fn new(max_data_points: usize) -> Self {
        Self {
            data_points: Arc::new(RwLock::new(HashMap::new())),
            aggregations: Arc::new(RwLock::new(HashMap::new())),
            retention_policies: Arc::new(RwLock::new(HashMap::new())),
            max_data_points,
        }
    }

    /// Store metric data point
    pub fn store_data_point(&self, metric_name: &str, data_point: MetricDataPoint) -> QuantumResult<()> {
        let mut data_points = self.data_points.write().unwrap();
        let points = data_points.entry(metric_name.to_string()).or_insert_with(VecDeque::new);
        
        points.push_back(data_point);
        
        // Enforce max data points limit
        if points.len() > self.max_data_points {
            points.pop_front();
        }
        
        Ok(())
    }

    /// Get metric data points
    pub fn get_data_points(&self, metric_name: &str, since: Option<DateTime<Utc>>) -> Vec<MetricDataPoint> {
        let data_points = self.data_points.read().unwrap();
        
        if let Some(points) = data_points.get(metric_name) {
            match since {
                Some(since_time) => points.iter()
                    .filter(|p| p.timestamp >= since_time)
                    .cloned()
                    .collect(),
                None => points.iter().cloned().collect(),
            }
        } else {
            Vec::new()
        }
    }

    /// Clean up old data points
    pub fn cleanup_old_data(&self) -> QuantumResult<()> {
        let mut data_points = self.data_points.write().unwrap();
        let retention_policies = self.retention_policies.read().unwrap();
        let now = Utc::now();
        
        for (metric_name, points) in data_points.iter_mut() {
            if let Some(retention_period) = retention_policies.get(metric_name) {
                let cutoff_time = now - *retention_period;
                
                // Remove old data points
                points.retain(|p| p.timestamp >= cutoff_time);
            }
        }
        
        Ok(())
    }

    /// Set retention policy
    pub fn set_retention_policy(&self, metric_name: &str, retention_period: Duration) {
        let mut policies = self.retention_policies.write().unwrap();
        policies.insert(metric_name.to_string(), retention_period);
    }

    /// Calculate aggregation
    pub fn calculate_aggregation(
        &self,
        metric_name: &str,
        method: AggregationMethod,
        time_window: TimeWindow,
    ) -> Option<MetricAggregation> {
        let data_points = self.data_points.read().unwrap();
        
        if let Some(points) = data_points.get(metric_name) {
            let now = Utc::now();
            let window_duration = Duration::seconds(time_window.to_seconds());
            let start_time = now - window_duration;
            
            let window_points: Vec<&MetricDataPoint> = points.iter()
                .filter(|p| p.timestamp >= start_time)
                .collect();
            
            if window_points.is_empty() {
                return None;
            }
            
            let values: Vec<f64> = window_points.iter().map(|p| p.value).collect();
            let count = values.len() as u64;
            let min = values.iter().fold(f64::INFINITY, |a, &b| a.min(b));
            let max = values.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
            
            let value = match method {
                AggregationMethod::Sum => values.iter().sum(),
                AggregationMethod::Average => values.iter().sum::<f64>() / values.len() as f64,
                AggregationMethod::Min => min,
                AggregationMethod::Max => max,
                AggregationMethod::Count => count as f64,
                AggregationMethod::Median => {
                    let mut sorted = values.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let mid = sorted.len() / 2;
                    if sorted.len() % 2 == 0 {
                        (sorted[mid - 1] + sorted[mid]) / 2.0
                    } else {
                        sorted[mid]
                    }
                }
                AggregationMethod::Percentile(p) => {
                    let mut sorted = values.clone();
                    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
                    let index = ((p as f64 / 100.0) * (sorted.len() - 1) as f64).round() as usize;
                    sorted[index.min(sorted.len() - 1)]
                }
                AggregationMethod::StdDev => {
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64;
                    variance.sqrt()
                }
                AggregationMethod::Variance => {
                    let mean = values.iter().sum::<f64>() / values.len() as f64;
                    values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / values.len() as f64
                }
            };
            
            Some(MetricAggregation {
                name: metric_name.to_string(),
                time_window,
                aggregation_method: method,
                value,
                count,
                min,
                max,
                start_time,
                end_time: now,
                tags: HashMap::new(),
            })
        } else {
            None
        }
    }
}

/// Quantum metrics collector
#[derive(Debug)]
pub struct QuantumMetricsCollector {
    storage: Arc<MetricStorage>,
    definitions: Arc<RwLock<HashMap<String, MetricDefinition>>>,
    alerts: Arc<AsyncRwLock<Vec<MetricAlert>>>,
    quantum_metrics: Arc<RwLock<QuantumMetrics>>,
    performance_metrics: Arc<RwLock<PerformanceMetrics>>,
    reliability_metrics: Arc<RwLock<ReliabilityMetrics>>,
    collection_interval: Duration,
    enabled: Arc<RwLock<bool>>,
}

impl QuantumMetricsCollector {
    /// Create new metrics collector
    pub fn new(collection_interval: Duration) -> Self {
        let collector = Self {
            storage: Arc::new(MetricStorage::new(10000)),
            definitions: Arc::new(RwLock::new(HashMap::new())),
            alerts: Arc::new(AsyncRwLock::new(Vec::new())),
            quantum_metrics: Arc::new(RwLock::new(QuantumMetrics::default())),
            performance_metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            reliability_metrics: Arc::new(RwLock::new(ReliabilityMetrics::default())),
            collection_interval,
            enabled: Arc::new(RwLock::new(true)),
        };
        
        // Register default metrics
        collector.register_default_metrics();
        
        info!("Quantum metrics collector initialized");
        collector
    }

    /// Register default quantum metrics
    fn register_default_metrics(&self) {
        let default_metrics = vec![
            MetricDefinition {
                name: "quantum_fidelity".to_string(),
                metric_type: MetricType::Gauge,
                description: "Quantum state fidelity".to_string(),
                unit: "ratio".to_string(),
                aggregation_method: AggregationMethod::Average,
                time_window: TimeWindow::Minute,
                retention_period: Duration::hours(24),
                tags: HashMap::new(),
                thresholds: Some(MetricThresholds {
                    warning: 0.95,
                    critical: 0.90,
                    target: Some(0.99),
                    direction: ThresholdDirection::Below,
                }),
            },
            MetricDefinition {
                name: "quantum_error_rate".to_string(),
                metric_type: MetricType::Gauge,
                description: "Quantum operation error rate".to_string(),
                unit: "ratio".to_string(),
                aggregation_method: AggregationMethod::Average,
                time_window: TimeWindow::Minute,
                retention_period: Duration::hours(24),
                tags: HashMap::new(),
                thresholds: Some(MetricThresholds {
                    warning: 0.01,
                    critical: 0.05,
                    target: Some(0.001),
                    direction: ThresholdDirection::Above,
                }),
            },
            MetricDefinition {
                name: "execution_time_ms".to_string(),
                metric_type: MetricType::Histogram,
                description: "Quantum operation execution time".to_string(),
                unit: "milliseconds".to_string(),
                aggregation_method: AggregationMethod::Average,
                time_window: TimeWindow::Minute,
                retention_period: Duration::hours(24),
                tags: HashMap::new(),
                thresholds: Some(MetricThresholds {
                    warning: 1000.0,
                    critical: 5000.0,
                    target: Some(100.0),
                    direction: ThresholdDirection::Above,
                }),
            },
            MetricDefinition {
                name: "memory_usage_mb".to_string(),
                metric_type: MetricType::Gauge,
                description: "Memory usage in megabytes".to_string(),
                unit: "megabytes".to_string(),
                aggregation_method: AggregationMethod::Average,
                time_window: TimeWindow::Minute,
                retention_period: Duration::hours(24),
                tags: HashMap::new(),
                thresholds: Some(MetricThresholds {
                    warning: 800.0,
                    critical: 900.0,
                    target: Some(400.0),
                    direction: ThresholdDirection::Above,
                }),
            },
            MetricDefinition {
                name: "throughput_ops_per_second".to_string(),
                metric_type: MetricType::Gauge,
                description: "Operations per second throughput".to_string(),
                unit: "ops/sec".to_string(),
                aggregation_method: AggregationMethod::Average,
                time_window: TimeWindow::Minute,
                retention_period: Duration::hours(24),
                tags: HashMap::new(),
                thresholds: Some(MetricThresholds {
                    warning: 100.0,
                    critical: 50.0,
                    target: Some(200.0),
                    direction: ThresholdDirection::Below,
                }),
            },
        ];
        
        let mut definitions = self.definitions.write().unwrap();
        for metric in default_metrics {
            definitions.insert(metric.name.clone(), metric);
        }
    }

    /// Register custom metric
    pub fn register_metric(&self, definition: MetricDefinition) -> QuantumResult<()> {
        let metric_name = definition.name.clone();
        let mut definitions = self.definitions.write().unwrap();
        definitions.insert(metric_name.clone(), definition);
        
        info!("Registered metric: {}", metric_name);
        Ok(())
    }

    /// Record quantum fidelity
    pub fn record_quantum_fidelity(&self, fidelity: f64) -> QuantumResult<()> {
        self.record_metric("quantum_fidelity", fidelity, None)?;
        
        let mut quantum_metrics = self.quantum_metrics.write().unwrap();
        quantum_metrics.fidelity = fidelity;
        quantum_metrics.timestamp = Utc::now();
        
        Ok(())
    }

    /// Record quantum error rate
    pub fn record_quantum_error_rate(&self, error_rate: f64) -> QuantumResult<()> {
        self.record_metric("quantum_error_rate", error_rate, None)?;
        
        let mut quantum_metrics = self.quantum_metrics.write().unwrap();
        quantum_metrics.error_rate = error_rate;
        quantum_metrics.timestamp = Utc::now();
        
        Ok(())
    }

    /// Record execution time
    pub fn record_execution_time(&self, time_ms: f64) -> QuantumResult<()> {
        self.record_metric("execution_time_ms", time_ms, None)?;
        
        let mut performance_metrics = self.performance_metrics.write().unwrap();
        performance_metrics.execution_time_ms = time_ms;
        performance_metrics.timestamp = Utc::now();
        
        Ok(())
    }

    /// Record memory usage
    pub fn record_memory_usage(&self, memory_mb: f64) -> QuantumResult<()> {
        self.record_metric("memory_usage_mb", memory_mb, None)?;
        
        let mut performance_metrics = self.performance_metrics.write().unwrap();
        performance_metrics.memory_usage_mb = memory_mb;
        performance_metrics.timestamp = Utc::now();
        
        Ok(())
    }

    /// Record throughput
    pub fn record_throughput(&self, ops_per_second: f64) -> QuantumResult<()> {
        self.record_metric("throughput_ops_per_second", ops_per_second, None)?;
        
        let mut performance_metrics = self.performance_metrics.write().unwrap();
        performance_metrics.throughput_ops_per_second = ops_per_second;
        performance_metrics.timestamp = Utc::now();
        
        Ok(())
    }

    /// Record custom metric
    pub fn record_metric(&self, name: &str, value: f64, tags: Option<HashMap<String, String>>) -> QuantumResult<()> {
        if !*self.enabled.read().unwrap() {
            return Ok(());
        }
        
        let data_point = MetricDataPoint {
            timestamp: Utc::now(),
            value,
            tags: tags.unwrap_or_default(),
            metadata: HashMap::new(),
        };
        
        self.storage.store_data_point(name, data_point)?;
        
        // Check thresholds and generate alerts
        self.check_thresholds(name, value)?;
        
        // Update global metrics registry
        match name {
            "quantum_fidelity" => {
                // metrics::gauge!("quantum_fidelity").set(value);
            },
            "quantum_error_rate" => {
                // metrics::gauge!("quantum_error_rate").set(value);
            },
            "execution_time_ms" => {
                // metrics::histogram!("execution_time_ms").record(value);
            },
            "memory_usage_mb" => {
                // metrics::gauge!("memory_usage_mb").set(value);
            },
            "throughput_ops_per_second" => {
                // metrics::gauge!("throughput_ops_per_second").set(value);
            },
            _ => {
                // metrics::gauge!(name).set(value);
            }
        }
        
        Ok(())
    }

    /// Check metric thresholds
    fn check_thresholds(&self, metric_name: &str, value: f64) -> QuantumResult<()> {
        let definitions = self.definitions.read().unwrap();
        
        if let Some(definition) = definitions.get(metric_name) {
            if let Some(thresholds) = &definition.thresholds {
                let should_alert = match thresholds.direction {
                    ThresholdDirection::Above => value > thresholds.warning || value > thresholds.critical,
                    ThresholdDirection::Below => value < thresholds.warning || value < thresholds.critical,
                };
                
                if should_alert {
                    let severity = if match thresholds.direction {
                        ThresholdDirection::Above => value > thresholds.critical,
                        ThresholdDirection::Below => value < thresholds.critical,
                    } {
                        AlertSeverity::Critical
                    } else {
                        AlertSeverity::Warning
                    };
                    
                    let threshold_value = if severity == AlertSeverity::Critical {
                        thresholds.critical
                    } else {
                        thresholds.warning
                    };
                    
                    let alert = MetricAlert {
                        id: format!("alert_{}", uuid::Uuid::new_v4()),
                        metric_name: metric_name.to_string(),
                        threshold_type: if severity == AlertSeverity::Critical { ThresholdType::Critical } else { ThresholdType::Warning },
                        threshold_value,
                        current_value: value,
                        severity,
                        message: format!("Metric {} {} threshold: {} (current: {})", 
                                       metric_name, 
                                       if severity == AlertSeverity::Critical { "critical" } else { "warning" },
                                       threshold_value,
                                       value),
                        triggered_at: Utc::now(),
                        resolved_at: None,
                        tags: HashMap::new(),
                    };
                    
                    // Store alert (async operation)
                    let alerts = self.alerts.clone();
                    tokio::spawn(async move {
                        let mut alerts = alerts.write().await;
                        alerts.push(alert);
                    });
                }
            }
        }
        
        Ok(())
    }

    /// Get metric aggregation
    pub fn get_aggregation(&self, metric_name: &str, method: AggregationMethod, time_window: TimeWindow) -> Option<MetricAggregation> {
        self.storage.calculate_aggregation(metric_name, method, time_window)
    }

    /// Get metric data points
    pub fn get_data_points(&self, metric_name: &str, since: Option<DateTime<Utc>>) -> Vec<MetricDataPoint> {
        self.storage.get_data_points(metric_name, since)
    }

    /// Get current quantum metrics
    pub fn get_quantum_metrics(&self) -> QuantumMetrics {
        let metrics = self.quantum_metrics.read().unwrap();
        metrics.clone()
    }

    /// Get current performance metrics
    pub fn get_performance_metrics(&self) -> PerformanceMetrics {
        let metrics = self.performance_metrics.read().unwrap();
        metrics.clone()
    }

    /// Get current reliability metrics
    pub fn get_reliability_metrics(&self) -> ReliabilityMetrics {
        let metrics = self.reliability_metrics.read().unwrap();
        metrics.clone()
    }

    /// Get active alerts
    pub async fn get_active_alerts(&self) -> Vec<MetricAlert> {
        let alerts = self.alerts.read().await;
        alerts.iter().filter(|a| a.resolved_at.is_none()).cloned().collect()
    }

    /// Resolve alert
    pub async fn resolve_alert(&self, alert_id: &str) -> QuantumResult<()> {
        let mut alerts = self.alerts.write().await;
        
        if let Some(alert) = alerts.iter_mut().find(|a| a.id == alert_id) {
            alert.resolved_at = Some(Utc::now());
            info!("Resolved alert: {}", alert_id);
            Ok(())
        } else {
            Err(QuantumError::metric_error(format!("Alert not found: {}", alert_id)))
        }
    }

    /// Generate metrics report
    pub fn generate_report(&self, time_window: TimeWindow) -> QuantumResult<MetricsReport> {
        let definitions = self.definitions.read().unwrap();
        let mut metric_summaries = Vec::new();
        
        for (name, definition) in definitions.iter() {
            if let Some(aggregation) = self.get_aggregation(name, definition.aggregation_method, time_window) {
                let summary = MetricSummary {
                    name: name.clone(),
                    description: definition.description.clone(),
                    unit: definition.unit.clone(),
                    current_value: aggregation.value,
                    min_value: aggregation.min,
                    max_value: aggregation.max,
                    count: aggregation.count,
                    aggregation_method: definition.aggregation_method,
                    time_window,
                    thresholds: definition.thresholds.clone(),
                };
                
                metric_summaries.push(summary);
            }
        }
        
        let quantum_metrics = self.get_quantum_metrics();
        let performance_metrics = self.get_performance_metrics();
        let reliability_metrics = self.get_reliability_metrics();
        
        Ok(MetricsReport {
            generated_at: Utc::now(),
            time_window,
            metric_summaries,
            quantum_metrics,
            performance_metrics,
            reliability_metrics,
        })
    }

    /// Enable/disable metrics collection
    pub fn set_enabled(&self, enabled: bool) {
        let mut enabled_state = self.enabled.write().unwrap();
        *enabled_state = enabled;
        
        info!("Metrics collection {}", if enabled { "enabled" } else { "disabled" });
    }

    /// Cleanup old data
    pub fn cleanup(&self) -> QuantumResult<()> {
        self.storage.cleanup_old_data()?;
        info!("Metrics cleanup completed");
        Ok(())
    }

    /// Start background collection
    pub async fn start_background_collection(&self) -> QuantumResult<()> {
        let collector = self.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(tokio::time::Duration::from_secs(collector.collection_interval.num_seconds() as u64));
            
            loop {
                interval.tick().await;
                
                if !*collector.enabled.read().unwrap() {
                    continue;
                }
                
                // Collect system metrics
                if let Err(e) = collector.collect_system_metrics().await {
                    error!("Failed to collect system metrics: {}", e);
                }
                
                // Cleanup old data
                if let Err(e) = collector.cleanup() {
                    error!("Failed to cleanup metrics: {}", e);
                }
            }
        });
        
        info!("Background metrics collection started");
        Ok(())
    }

    /// Collect system metrics
    async fn collect_system_metrics(&self) -> QuantumResult<()> {
        // Simulate collecting system metrics
        let memory_usage = 512.0; // MB
        let cpu_usage = 25.0; // %
        let throughput = 150.0; // ops/sec
        
        self.record_memory_usage(memory_usage)?;
        self.record_metric("cpu_utilization_percent", cpu_usage, None)?;
        self.record_throughput(throughput)?;
        
        Ok(())
    }
}

impl Clone for QuantumMetricsCollector {
    fn clone(&self) -> Self {
        Self {
            storage: Arc::clone(&self.storage),
            definitions: Arc::clone(&self.definitions),
            alerts: Arc::clone(&self.alerts),
            quantum_metrics: Arc::clone(&self.quantum_metrics),
            performance_metrics: Arc::clone(&self.performance_metrics),
            reliability_metrics: Arc::clone(&self.reliability_metrics),
            collection_interval: self.collection_interval,
            enabled: Arc::clone(&self.enabled),
        }
    }
}

/// Metric summary for reporting
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricSummary {
    pub name: String,
    pub description: String,
    pub unit: String,
    pub current_value: f64,
    pub min_value: f64,
    pub max_value: f64,
    pub count: u64,
    pub aggregation_method: AggregationMethod,
    pub time_window: TimeWindow,
    pub thresholds: Option<MetricThresholds>,
}

/// Comprehensive metrics report
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsReport {
    pub generated_at: DateTime<Utc>,
    pub time_window: TimeWindow,
    pub metric_summaries: Vec<MetricSummary>,
    pub quantum_metrics: QuantumMetrics,
    pub performance_metrics: PerformanceMetrics,
    pub reliability_metrics: ReliabilityMetrics,
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[test]
    fn test_metric_storage() {
        let storage = MetricStorage::new(100);
        
        let data_point = MetricDataPoint {
            timestamp: Utc::now(),
            value: 42.0,
            tags: HashMap::new(),
            metadata: HashMap::new(),
        };
        
        assert!(storage.store_data_point("test_metric", data_point).is_ok());
        
        let points = storage.get_data_points("test_metric", None);
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].value, 42.0);
    }

    #[test]
    fn test_metric_aggregation() {
        let storage = MetricStorage::new(100);
        
        // Store multiple data points
        for i in 1..=10 {
            let data_point = MetricDataPoint {
                timestamp: Utc::now(),
                value: i as f64,
                tags: HashMap::new(),
                metadata: HashMap::new(),
            };
            storage.store_data_point("test_metric", data_point).unwrap();
        }
        
        // Test different aggregation methods
        let sum_agg = storage.calculate_aggregation("test_metric", AggregationMethod::Sum, TimeWindow::Hour);
        assert!(sum_agg.is_some());
        assert_eq!(sum_agg.unwrap().value, 55.0); // 1+2+...+10 = 55
        
        let avg_agg = storage.calculate_aggregation("test_metric", AggregationMethod::Average, TimeWindow::Hour);
        assert!(avg_agg.is_some());
        assert_eq!(avg_agg.unwrap().value, 5.5); // 55/10 = 5.5
        
        let max_agg = storage.calculate_aggregation("test_metric", AggregationMethod::Max, TimeWindow::Hour);
        assert!(max_agg.is_some());
        assert_eq!(max_agg.unwrap().value, 10.0);
        
        let min_agg = storage.calculate_aggregation("test_metric", AggregationMethod::Min, TimeWindow::Hour);
        assert!(min_agg.is_some());
        assert_eq!(min_agg.unwrap().value, 1.0);
    }

    #[test]
    fn test_quantum_metrics_collector() {
        let collector = QuantumMetricsCollector::new(Duration::seconds(1));
        
        // Test recording quantum fidelity
        assert!(collector.record_quantum_fidelity(0.99).is_ok());
        let quantum_metrics = collector.get_quantum_metrics();
        assert_eq!(quantum_metrics.fidelity, 0.99);
        
        // Test recording error rate
        assert!(collector.record_quantum_error_rate(0.01).is_ok());
        let quantum_metrics = collector.get_quantum_metrics();
        assert_eq!(quantum_metrics.error_rate, 0.01);
        
        // Test recording execution time
        assert!(collector.record_execution_time(150.0).is_ok());
        let performance_metrics = collector.get_performance_metrics();
        assert_eq!(performance_metrics.execution_time_ms, 150.0);
    }

    #[test]
    fn test_metric_thresholds() {
        let thresholds = MetricThresholds {
            warning: 0.95,
            critical: 0.90,
            target: Some(0.99),
            direction: ThresholdDirection::Below,
        };
        
        assert_eq!(thresholds.warning, 0.95);
        assert_eq!(thresholds.critical, 0.90);
        assert_eq!(thresholds.target, Some(0.99));
        assert_eq!(thresholds.direction, ThresholdDirection::Below);
    }

    #[tokio::test]
    async fn test_alert_generation() {
        let collector = QuantumMetricsCollector::new(Duration::seconds(1));
        
        // Record a value that should trigger an alert
        collector.record_quantum_fidelity(0.85).unwrap(); // Below critical threshold of 0.90
        
        // Give some time for async alert processing
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        let alerts = collector.get_active_alerts().await;
        assert!(!alerts.is_empty());
        
        let alert = &alerts[0];
        assert_eq!(alert.metric_name, "quantum_fidelity");
        assert_eq!(alert.severity, AlertSeverity::Critical);
        assert_eq!(alert.current_value, 0.85);
    }

    #[tokio::test]
    async fn test_alert_resolution() {
        let collector = QuantumMetricsCollector::new(Duration::seconds(1));
        
        // Trigger an alert
        collector.record_quantum_fidelity(0.85).unwrap();
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        let alerts = collector.get_active_alerts().await;
        assert!(!alerts.is_empty());
        
        let alert_id = alerts[0].id.clone();
        
        // Resolve the alert
        assert!(collector.resolve_alert(&alert_id).await.is_ok());
        
        let active_alerts = collector.get_active_alerts().await;
        assert!(active_alerts.is_empty());
    }

    #[test]
    fn test_metrics_report() {
        let collector = QuantumMetricsCollector::new(Duration::seconds(1));
        
        // Record some metrics
        collector.record_quantum_fidelity(0.98).unwrap();
        collector.record_quantum_error_rate(0.02).unwrap();
        collector.record_execution_time(200.0).unwrap();
        collector.record_memory_usage(400.0).unwrap();
        collector.record_throughput(180.0).unwrap();
        
        let report = collector.generate_report(TimeWindow::Hour);
        assert!(report.is_ok());
        
        let report = report.unwrap();
        assert!(!report.metric_summaries.is_empty());
        assert_eq!(report.quantum_metrics.fidelity, 0.98);
        assert_eq!(report.performance_metrics.execution_time_ms, 200.0);
    }

    #[test]
    fn test_custom_metric_registration() {
        let collector = QuantumMetricsCollector::new(Duration::seconds(1));
        
        let custom_metric = MetricDefinition {
            name: "custom_metric".to_string(),
            metric_type: MetricType::Counter,
            description: "A custom metric".to_string(),
            unit: "count".to_string(),
            aggregation_method: AggregationMethod::Sum,
            time_window: TimeWindow::Minute,
            retention_period: Duration::hours(1),
            tags: HashMap::new(),
            thresholds: None,
        };
        
        assert!(collector.register_metric(custom_metric).is_ok());
        
        // Record value for custom metric
        assert!(collector.record_metric("custom_metric", 100.0, None).is_ok());
        
        let data_points = collector.get_data_points("custom_metric", None);
        assert_eq!(data_points.len(), 1);
        assert_eq!(data_points[0].value, 100.0);
    }

    #[test]
    fn test_time_window_conversion() {
        assert_eq!(TimeWindow::Second.to_seconds(), 1);
        assert_eq!(TimeWindow::Minute.to_seconds(), 60);
        assert_eq!(TimeWindow::Hour.to_seconds(), 3600);
        assert_eq!(TimeWindow::Day.to_seconds(), 86400);
        assert_eq!(TimeWindow::Week.to_seconds(), 604800);
        assert_eq!(TimeWindow::Month.to_seconds(), 2592000);
        assert_eq!(TimeWindow::Custom(123).to_seconds(), 123);
    }

    #[test]
    fn test_metric_data_point() {
        let mut tags = HashMap::new();
        tags.insert("device".to_string(), "cpu".to_string());
        
        let data_point = MetricDataPoint {
            timestamp: Utc::now(),
            value: 42.0,
            tags,
            metadata: HashMap::new(),
        };
        
        assert_eq!(data_point.value, 42.0);
        assert_eq!(data_point.tags.get("device"), Some(&"cpu".to_string()));
    }

    #[test]
    fn test_default_metrics() {
        let quantum_metrics = QuantumMetrics::default();
        assert_eq!(quantum_metrics.fidelity, 1.0);
        assert_eq!(quantum_metrics.error_rate, 0.001);
        assert_eq!(quantum_metrics.gate_count, 0);
        
        let performance_metrics = PerformanceMetrics::default();
        assert_eq!(performance_metrics.execution_time_ms, 0.0);
        assert_eq!(performance_metrics.memory_usage_mb, 0.0);
        assert_eq!(performance_metrics.throughput_ops_per_second, 0.0);
        
        let reliability_metrics = ReliabilityMetrics::default();
        assert_eq!(reliability_metrics.availability_percentage, 100.0);
        assert_eq!(reliability_metrics.mtbf_hours, 720.0);
        assert_eq!(reliability_metrics.error_count, 0);
    }

    #[test]
    fn test_metric_aggregation_percentiles() {
        let storage = MetricStorage::new(100);
        
        // Store values 1-100
        for i in 1..=100 {
            let data_point = MetricDataPoint {
                timestamp: Utc::now(),
                value: i as f64,
                tags: HashMap::new(),
                metadata: HashMap::new(),
            };
            storage.store_data_point("test_metric", data_point).unwrap();
        }
        
        // Test 50th percentile (median)
        let p50 = storage.calculate_aggregation("test_metric", AggregationMethod::Percentile(50), TimeWindow::Hour);
        assert!(p50.is_some());
        assert!((p50.unwrap().value - 50.0).abs() < 1.0);
        
        // Test 95th percentile
        let p95 = storage.calculate_aggregation("test_metric", AggregationMethod::Percentile(95), TimeWindow::Hour);
        assert!(p95.is_some());
        assert!((p95.unwrap().value - 95.0).abs() < 1.0);
    }

    #[test]
    fn test_metric_retention() {
        let storage = MetricStorage::new(100);
        
        // Set retention policy
        storage.set_retention_policy("test_metric", Duration::hours(1));
        
        // Store old data point
        let old_data_point = MetricDataPoint {
            timestamp: Utc::now() - Duration::hours(2),
            value: 42.0,
            tags: HashMap::new(),
            metadata: HashMap::new(),
        };
        storage.store_data_point("test_metric", old_data_point).unwrap();
        
        // Store recent data point
        let recent_data_point = MetricDataPoint {
            timestamp: Utc::now(),
            value: 100.0,
            tags: HashMap::new(),
            metadata: HashMap::new(),
        };
        storage.store_data_point("test_metric", recent_data_point).unwrap();
        
        // Clean up old data
        storage.cleanup_old_data().unwrap();
        
        let points = storage.get_data_points("test_metric", None);
        assert_eq!(points.len(), 1);
        assert_eq!(points[0].value, 100.0);
    }

    #[test]
    fn test_collector_enable_disable() {
        let collector = QuantumMetricsCollector::new(Duration::seconds(1));
        
        // Initially enabled
        assert!(collector.record_quantum_fidelity(0.99).is_ok());
        
        // Disable collection
        collector.set_enabled(false);
        assert!(collector.record_quantum_fidelity(0.98).is_ok()); // Should still work but not record
        
        // Re-enable collection
        collector.set_enabled(true);
        assert!(collector.record_quantum_fidelity(0.97).is_ok());
    }

    #[test]
    fn test_metric_types_and_enums() {
        let metric_types = vec![
            MetricType::Counter,
            MetricType::Histogram,
            MetricType::Gauge,
            MetricType::Summary,
            MetricType::Rate,
            MetricType::Percentile,
        ];
        
        let aggregation_methods = vec![
            AggregationMethod::Sum,
            AggregationMethod::Average,
            AggregationMethod::Min,
            AggregationMethod::Max,
            AggregationMethod::Count,
            AggregationMethod::Median,
            AggregationMethod::Percentile(95),
            AggregationMethod::StdDev,
            AggregationMethod::Variance,
        ];
        
        let alert_severities = vec![
            AlertSeverity::Info,
            AlertSeverity::Warning,
            AlertSeverity::Critical,
            AlertSeverity::Fatal,
        ];
        
        // Test serialization/deserialization
        for metric_type in metric_types {
            let serialized = serde_json::to_string(&metric_type).unwrap();
            let deserialized: MetricType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(metric_type, deserialized);
        }
        
        for method in aggregation_methods {
            let serialized = serde_json::to_string(&method).unwrap();
            let deserialized: AggregationMethod = serde_json::from_str(&serialized).unwrap();
            assert_eq!(method, deserialized);
        }
        
        for severity in alert_severities {
            let serialized = serde_json::to_string(&severity).unwrap();
            let deserialized: AlertSeverity = serde_json::from_str(&serialized).unwrap();
            assert_eq!(severity, deserialized);
        }
    }
}