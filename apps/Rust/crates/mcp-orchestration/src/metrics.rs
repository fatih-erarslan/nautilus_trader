//! Metrics collection and monitoring for MCP orchestration.

use crate::error::{OrchestrationError, Result};
use crate::types::{AgentId, AgentType, Timestamp};
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use std::time::Duration;
use tokio::time::interval;
use tracing::{debug, info};

/// Metric value types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum MetricValue {
    /// Counter metric (monotonically increasing)
    Counter(u64),
    /// Gauge metric (can increase or decrease)
    Gauge(f64),
    /// Histogram metric with buckets
    Histogram {
        buckets: Vec<(f64, u64)>, // (upper_bound, count)
        sum: f64,
        count: u64,
    },
    /// Summary metric with quantiles
    Summary {
        quantiles: Vec<(f64, f64)>, // (quantile, value)
        sum: f64,
        count: u64,
    },
}

/// Metric definition
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Metric {
    /// Metric name
    pub name: String,
    /// Metric description
    pub description: String,
    /// Metric labels
    pub labels: HashMap<String, String>,
    /// Metric value
    pub value: MetricValue,
    /// Timestamp when metric was recorded
    pub timestamp: Timestamp,
}

impl Metric {
    /// Create a new counter metric
    pub fn counter<S: Into<String>>(name: S, description: S, value: u64) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            labels: HashMap::new(),
            value: MetricValue::Counter(value),
            timestamp: Timestamp::now(),
        }
    }
    
    /// Create a new gauge metric
    pub fn gauge<S: Into<String>>(name: S, description: S, value: f64) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            labels: HashMap::new(),
            value: MetricValue::Gauge(value),
            timestamp: Timestamp::now(),
        }
    }
    
    /// Add a label to the metric
    pub fn with_label<K, V>(mut self, key: K, value: V) -> Self
    where
        K: Into<String>,
        V: Into<String>,
    {
        self.labels.insert(key.into(), value.into());
        self
    }
    
    /// Get metric value as f64
    pub fn as_f64(&self) -> f64 {
        match &self.value {
            MetricValue::Counter(v) => *v as f64,
            MetricValue::Gauge(v) => *v,
            MetricValue::Histogram { sum, .. } => *sum,
            MetricValue::Summary { sum, .. } => *sum,
        }
    }
}

/// Orchestration metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OrchestrationMetrics {
    /// Agent metrics
    pub agent_metrics: AgentMetrics,
    /// Communication metrics
    pub communication_metrics: CommunicationMetrics,
    /// Task metrics
    pub task_metrics: TaskMetrics,
    /// Memory metrics
    pub memory_metrics: MemoryMetrics,
    /// Health metrics
    pub health_metrics: HealthMetrics,
    /// Recovery metrics
    pub recovery_metrics: RecoveryMetrics,
    /// System metrics
    pub system_metrics: SystemMetrics,
    /// Custom metrics
    pub custom_metrics: HashMap<String, Metric>,
    /// Collection timestamp
    pub timestamp: Timestamp,
}

/// Agent-specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentMetrics {
    /// Total number of agents
    pub total_agents: u64,
    /// Agents by state
    pub agents_by_state: HashMap<String, u64>,
    /// Agents by type
    pub agents_by_type: HashMap<AgentType, u64>,
    /// Agent uptime in seconds
    pub agent_uptime: HashMap<AgentId, u64>,
    /// Agent task processing rate
    pub task_processing_rate: HashMap<AgentId, f64>,
    /// Agent resource utilization
    pub resource_utilization: HashMap<AgentId, f64>,
}

/// Communication metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommunicationMetrics {
    /// Total messages sent
    pub messages_sent: u64,
    /// Total messages received
    pub messages_received: u64,
    /// Messages by type
    pub messages_by_type: HashMap<String, u64>,
    /// Message processing latency percentiles
    pub latency_percentiles: HashMap<String, f64>, // P50, P95, P99
    /// Failed messages
    pub failed_messages: u64,
    /// Retried messages
    pub retried_messages: u64,
    /// Broadcast efficiency
    pub broadcast_efficiency: f64,
    /// Queue depths
    pub queue_depths: HashMap<AgentId, u64>,
}

/// Task metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskMetrics {
    /// Total tasks submitted
    pub tasks_submitted: u64,
    /// Tasks by status
    pub tasks_by_status: HashMap<String, u64>,
    /// Tasks by priority
    pub tasks_by_priority: HashMap<String, u64>,
    /// Task processing time percentiles
    pub processing_time_percentiles: HashMap<String, f64>,
    /// Task queue depth
    pub queue_depth: u64,
    /// Task throughput (tasks per second)
    pub throughput: f64,
    /// Task success rate
    pub success_rate: f64,
}

/// Memory metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryMetrics {
    /// Total memory regions
    pub total_regions: u64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Memory by agent
    pub memory_by_agent: HashMap<AgentId, u64>,
    /// Cache hit ratio
    pub cache_hit_ratio: f64,
    /// Memory operations per second
    pub operations_per_second: f64,
    /// Region access frequency
    pub region_access_frequency: HashMap<String, u64>,
}

/// Health metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HealthMetrics {
    /// System health score (0-100)
    pub system_health_score: f64,
    /// Component health scores
    pub component_health_scores: HashMap<String, f64>,
    /// Health check response times
    pub health_check_response_times: HashMap<String, f64>,
    /// Failed health checks
    pub failed_health_checks: u64,
    /// System uptime in seconds
    pub system_uptime: u64,
}

/// Recovery metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RecoveryMetrics {
    /// Total recovery attempts
    pub recovery_attempts: u64,
    /// Successful recoveries
    pub successful_recoveries: u64,
    /// Recovery success rate
    pub recovery_success_rate: f64,
    /// Recovery time percentiles
    pub recovery_time_percentiles: HashMap<String, f64>,
    /// Circuit breaker states
    pub circuit_breaker_states: HashMap<String, String>,
    /// Graceful degradations active
    pub graceful_degradations: u64,
}

/// System metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    /// CPU usage percentage
    pub cpu_usage: f64,
    /// Memory usage in bytes
    pub memory_usage: u64,
    /// Disk usage percentage
    pub disk_usage: f64,
    /// Network I/O bytes per second
    pub network_io_bps: u64,
    /// Open file descriptors
    pub open_file_descriptors: u64,
    /// Thread count
    pub thread_count: u64,
}

/// Metrics collector trait
pub trait MetricsCollector: Send + Sync {
    /// Record a metric
    fn record_metric(&self, metric: Metric);
    
    /// Get all collected metrics
    fn get_metrics(&self) -> HashMap<String, Metric>;
    
    /// Get metrics by prefix
    fn get_metrics_by_prefix(&self, prefix: &str) -> HashMap<String, Metric>;
    
    /// Clear all metrics
    fn clear_metrics(&self);
}

/// In-memory metrics collector
#[derive(Debug)]
pub struct InMemoryMetricsCollector {
    /// Stored metrics
    metrics: Arc<RwLock<HashMap<String, Metric>>>,
    /// Metric counters
    counters: Arc<RwLock<HashMap<String, AtomicU64>>>,
}

impl InMemoryMetricsCollector {
    /// Create a new in-memory metrics collector
    pub fn new() -> Self {
        Self {
            metrics: Arc::new(RwLock::new(HashMap::new())),
            counters: Arc::new(RwLock::new(HashMap::new())),
        }
    }
    
    /// Increment a counter metric
    pub fn increment_counter(&self, name: &str) {
        let counters = self.counters.read();
        if let Some(counter) = counters.get(name) {
            counter.fetch_add(1, Ordering::Relaxed);
        } else {
            drop(counters);
            let mut counters = self.counters.write();
            counters.insert(name.to_string(), AtomicU64::new(1));
        }
    }
    
    /// Set a gauge metric
    pub fn set_gauge(&self, name: &str, value: f64) {
        let metric = Metric::gauge(name, "", value);
        self.record_metric(metric);
    }
    
    /// Get counter value
    pub fn get_counter(&self, name: &str) -> Option<u64> {
        let counters = self.counters.read();
        counters.get(name).map(|counter| counter.load(Ordering::Relaxed))
    }
}

impl MetricsCollector for InMemoryMetricsCollector {
    fn record_metric(&self, metric: Metric) {
        let mut metrics = self.metrics.write();
        metrics.insert(metric.name.clone(), metric);
    }
    
    fn get_metrics(&self) -> HashMap<String, Metric> {
        let metrics = self.metrics.read();
        metrics.clone()
    }
    
    fn get_metrics_by_prefix(&self, prefix: &str) -> HashMap<String, Metric> {
        let metrics = self.metrics.read();
        metrics
            .iter()
            .filter(|(name, _)| name.starts_with(prefix))
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect()
    }
    
    fn clear_metrics(&self) {
        let mut metrics = self.metrics.write();
        metrics.clear();
        
        let mut counters = self.counters.write();
        counters.clear();
    }
}

/// Prometheus metrics exporter
#[derive(Debug)]
pub struct PrometheusExporter {
    /// Internal metrics collector
    collector: Arc<dyn MetricsCollector>,
    /// Prometheus registry
    registry: prometheus::Registry,
}

impl PrometheusExporter {
    /// Create a new Prometheus exporter
    pub fn new(collector: Arc<dyn MetricsCollector>) -> Self {
        Self {
            collector,
            registry: prometheus::Registry::new(),
        }
    }
    
    /// Export metrics in Prometheus format
    pub fn export_metrics(&self) -> Result<String> {
        let metrics = self.collector.get_metrics();
        let mut output = String::new();
        
        for (name, metric) in metrics {
            // Convert metric to Prometheus format
            let prometheus_name = self.sanitize_metric_name(&name);
            
            // Add help text
            output.push_str(&format!("# HELP {} {}\n", prometheus_name, metric.description));
            
            // Add type
            let metric_type = match metric.value {
                MetricValue::Counter(_) => "counter",
                MetricValue::Gauge(_) => "gauge",
                MetricValue::Histogram { .. } => "histogram",
                MetricValue::Summary { .. } => "summary",
            };
            output.push_str(&format!("# TYPE {} {}\n", prometheus_name, metric_type));
            
            // Add metric value with labels
            let labels = self.format_labels(&metric.labels);
            match metric.value {
                MetricValue::Counter(value) => {
                    output.push_str(&format!("{}{} {}\n", prometheus_name, labels, value));
                }
                MetricValue::Gauge(value) => {
                    output.push_str(&format!("{}{} {}\n", prometheus_name, labels, value));
                }
                MetricValue::Histogram { ref buckets, sum, count } => {
                    for (upper_bound, bucket_count) in buckets {
                        output.push_str(&format!(
                            "{}_bucket{{le="{}"{}}} {}\n",
                            prometheus_name,
                            upper_bound,
                            if labels.is_empty() { "" } else { &format!(",{}", &labels[1..labels.len()-1]) },
                            bucket_count
                        ));
                    }
                    output.push_str(&format!("{}_sum{} {}\n", prometheus_name, labels, sum));
                    output.push_str(&format!("{}_count{} {}\n", prometheus_name, labels, count));
                }
                MetricValue::Summary { ref quantiles, sum, count } => {
                    for (quantile, value) in quantiles {
                        output.push_str(&format!(
                            "{}{{quantile="{}"{}}} {}\n",
                            prometheus_name,
                            quantile,
                            if labels.is_empty() { "" } else { &format!(",{}", &labels[1..labels.len()-1]) },
                            value
                        ));
                    }
                    output.push_str(&format!("{}_sum{} {}\n", prometheus_name, labels, sum));
                    output.push_str(&format!("{}_count{} {}\n", prometheus_name, labels, count));
                }
            }
        }
        
        Ok(output)
    }
    
    /// Sanitize metric name for Prometheus
    fn sanitize_metric_name(&self, name: &str) -> String {
        name.chars()
            .map(|c| if c.is_alphanumeric() || c == '_' { c } else { '_' })
            .collect()
    }
    
    /// Format labels for Prometheus
    fn format_labels(&self, labels: &HashMap<String, String>) -> String {
        if labels.is_empty() {
            return String::new();
        }
        
        let label_pairs: Vec<String> = labels
            .iter()
            .map(|(k, v)| format!("{}="{}"", k, v))
            .collect();
        
        format!("{{{}}}", label_pairs.join(","))
    }
}

/// Metrics aggregator for collecting system-wide metrics
#[derive(Debug)]
pub struct MetricsAggregator {
    /// Metrics collector
    collector: Arc<dyn MetricsCollector>,
    /// Prometheus exporter
    prometheus_exporter: Option<PrometheusExporter>,
    /// Collection interval
    collection_interval: Duration,
}

impl MetricsAggregator {
    /// Create a new metrics aggregator
    pub fn new(
        collector: Arc<dyn MetricsCollector>,
        enable_prometheus: bool,
        collection_interval: Duration,
    ) -> Self {
        let prometheus_exporter = if enable_prometheus {
            Some(PrometheusExporter::new(Arc::clone(&collector)))
        } else {
            None
        };
        
        Self {
            collector,
            prometheus_exporter,
            collection_interval,
        }
    }
    
    /// Start metrics collection
    pub async fn start(&self) -> Result<()> {
        // Start metrics collection loop
        let collector = Arc::clone(&self.collector);
        let collection_interval = self.collection_interval;
        
        tokio::spawn(async move {
            let mut interval = interval(collection_interval);
            
            loop {
                interval.tick().await;
                
                // Collect system metrics
                Self::collect_system_metrics(&collector);
                
                debug!("Metrics collected");
            }
        });
        
        info!("Metrics aggregator started successfully");
        Ok(())
    }
    
    /// Collect system metrics
    fn collect_system_metrics(collector: &Arc<dyn MetricsCollector>) {
        // Simulate system metrics collection
        // In a real implementation, this would use system APIs
        
        // CPU usage
        let cpu_usage = rand::random::<f64>() * 100.0;
        collector.record_metric(
            Metric::gauge("system_cpu_usage_percent", "System CPU usage percentage", cpu_usage)
        );
        
        // Memory usage
        let memory_usage = rand::random::<u64>() % (8 * 1024 * 1024 * 1024); // Up to 8GB
        collector.record_metric(
            Metric::gauge("system_memory_usage_bytes", "System memory usage in bytes", memory_usage as f64)
        );
        
        // Network I/O
        let network_io = rand::random::<u64>() % (1024 * 1024); // Up to 1MB/s
        collector.record_metric(
            Metric::gauge("system_network_io_bytes_per_second", "Network I/O bytes per second", network_io as f64)
        );
        
        // Increment system metrics counter
        if let Ok(in_memory_collector) = collector.as_ref().downcast_ref::<InMemoryMetricsCollector>() {
            in_memory_collector.increment_counter("system_metrics_collected_total");
        }
    }
    
    /// Get orchestration metrics
    pub fn get_orchestration_metrics(&self) -> OrchestrationMetrics {
        OrchestrationMetrics {
            agent_metrics: AgentMetrics {
                total_agents: 0,
                agents_by_state: HashMap::new(),
                agents_by_type: HashMap::new(),
                agent_uptime: HashMap::new(),
                task_processing_rate: HashMap::new(),
                resource_utilization: HashMap::new(),
            },
            communication_metrics: CommunicationMetrics {
                messages_sent: 0,
                messages_received: 0,
                messages_by_type: HashMap::new(),
                latency_percentiles: HashMap::new(),
                failed_messages: 0,
                retried_messages: 0,
                broadcast_efficiency: 0.0,
                queue_depths: HashMap::new(),
            },
            task_metrics: TaskMetrics {
                tasks_submitted: 0,
                tasks_by_status: HashMap::new(),
                tasks_by_priority: HashMap::new(),
                processing_time_percentiles: HashMap::new(),
                queue_depth: 0,
                throughput: 0.0,
                success_rate: 0.0,
            },
            memory_metrics: MemoryMetrics {
                total_regions: 0,
                memory_usage: 0,
                memory_by_agent: HashMap::new(),
                cache_hit_ratio: 0.0,
                operations_per_second: 0.0,
                region_access_frequency: HashMap::new(),
            },
            health_metrics: HealthMetrics {
                system_health_score: 0.0,
                component_health_scores: HashMap::new(),
                health_check_response_times: HashMap::new(),
                failed_health_checks: 0,
                system_uptime: 0,
            },
            recovery_metrics: RecoveryMetrics {
                recovery_attempts: 0,
                successful_recoveries: 0,
                recovery_success_rate: 0.0,
                recovery_time_percentiles: HashMap::new(),
                circuit_breaker_states: HashMap::new(),
                graceful_degradations: 0,
            },
            system_metrics: SystemMetrics {
                cpu_usage: 0.0,
                memory_usage: 0,
                disk_usage: 0.0,
                network_io_bps: 0,
                open_file_descriptors: 0,
                thread_count: 0,
            },
            custom_metrics: HashMap::new(),
            timestamp: Timestamp::now(),
        }
    }
    
    /// Export metrics for Prometheus
    pub fn export_prometheus_metrics(&self) -> Result<String> {
        if let Some(exporter) = &self.prometheus_exporter {
            exporter.export_metrics()
        } else {
            Err(OrchestrationError::metrics("Prometheus exporter not enabled".to_string()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use tokio::time::{sleep, Duration};
    
    #[test]
    fn test_metric_creation() {
        let counter = Metric::counter("test_counter", "Test counter metric", 42)
            .with_label("component", "test")
            .with_label("version", "1.0");
        
        assert_eq!(counter.name, "test_counter");
        assert_eq!(counter.description, "Test counter metric");
        assert_eq!(counter.labels.get("component"), Some(&"test".to_string()));
        assert_eq!(counter.labels.get("version"), Some(&"1.0".to_string()));
        
        match counter.value {
            MetricValue::Counter(value) => assert_eq!(value, 42),
            _ => panic!("Expected counter value"),
        }
    }
    
    #[test]
    fn test_in_memory_metrics_collector() {
        let collector = InMemoryMetricsCollector::new();
        
        // Record a metric
        let metric = Metric::gauge("test_gauge", "Test gauge metric", 3.14);
        collector.record_metric(metric);
        
        // Increment counter
        collector.increment_counter("test_counter");
        collector.increment_counter("test_counter");
        
        // Set gauge
        collector.set_gauge("cpu_usage", 75.5);
        
        // Get metrics
        let metrics = collector.get_metrics();
        assert!(metrics.contains_key("test_gauge"));
        assert!(metrics.contains_key("cpu_usage"));
        
        // Get counter value
        assert_eq!(collector.get_counter("test_counter"), Some(2));
        
        // Get metrics by prefix
        let test_metrics = collector.get_metrics_by_prefix("test_");
        assert_eq!(test_metrics.len(), 1);
        assert!(test_metrics.contains_key("test_gauge"));
        
        // Clear metrics
        collector.clear_metrics();
        let metrics = collector.get_metrics();
        assert!(metrics.is_empty());
        assert_eq!(collector.get_counter("test_counter"), None);
    }
    
    #[test]
    fn test_prometheus_exporter() {
        let collector = Arc::new(InMemoryMetricsCollector::new());
        let exporter = PrometheusExporter::new(collector.clone());
        
        // Record some metrics
        collector.record_metric(
            Metric::counter("http_requests_total", "Total HTTP requests", 1000)
                .with_label("method", "GET")
                .with_label("status", "200")
        );
        
        collector.record_metric(
            Metric::gauge("cpu_usage_percent", "CPU usage percentage", 75.5)
        );
        
        // Export metrics
        let prometheus_output = exporter.export_metrics().unwrap();
        
        assert!(prometheus_output.contains("# HELP http_requests_total Total HTTP requests"));
        assert!(prometheus_output.contains("# TYPE http_requests_total counter"));
        assert!(prometheus_output.contains("http_requests_total{method="GET",status="200"} 1000"));
        
        assert!(prometheus_output.contains("# HELP cpu_usage_percent CPU usage percentage"));
        assert!(prometheus_output.contains("# TYPE cpu_usage_percent gauge"));
        assert!(prometheus_output.contains("cpu_usage_percent 75.5"));
    }
    
    #[tokio::test]
    async fn test_metrics_aggregator() {
        let collector = Arc::new(InMemoryMetricsCollector::new());
        let aggregator = MetricsAggregator::new(
            collector.clone(),
            true,
            Duration::from_millis(100),
        );
        
        aggregator.start().await.unwrap();
        
        // Wait for metrics collection
        sleep(Duration::from_millis(200)).await;
        
        // Check that system metrics were collected
        let metrics = collector.get_metrics();
        assert!(metrics.contains_key("system_cpu_usage_percent"));
        assert!(metrics.contains_key("system_memory_usage_bytes"));
        assert!(metrics.contains_key("system_network_io_bytes_per_second"));
        
        // Check counter
        assert!(collector.get_counter("system_metrics_collected_total").unwrap() > 0);
    }
    
    #[test]
    fn test_orchestration_metrics() {
        let collector = Arc::new(InMemoryMetricsCollector::new());
        let aggregator = MetricsAggregator::new(
            collector,
            false,
            Duration::from_secs(1),
        );
        
        let metrics = aggregator.get_orchestration_metrics();
        
        assert_eq!(metrics.agent_metrics.total_agents, 0);
        assert_eq!(metrics.communication_metrics.messages_sent, 0);
        assert_eq!(metrics.task_metrics.tasks_submitted, 0);
        assert_eq!(metrics.memory_metrics.total_regions, 0);
    }
}