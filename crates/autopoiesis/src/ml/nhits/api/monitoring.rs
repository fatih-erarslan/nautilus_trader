use anyhow::Result;
use prometheus::{
    Counter, Gauge, Histogram, HistogramOpts, IntCounter, IntGauge, Opts, Registry,
};
use serde::{Deserialize, Serialize};
use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};
use tokio::sync::RwLock;
use tracing::{debug, error, info, warn};

/// Metrics registry for the NHITS API
#[derive(Clone)]
pub struct MetricsRegistry {
    /// Prometheus registry
    registry: Registry,
    
    // Counter metrics
    pub requests_total: IntCounter,
    pub forecasts_created: IntCounter,
    pub forecasts_completed: IntCounter,
    pub forecasts_failed: IntCounter,
    pub models_created: IntCounter,
    pub models_trained: IntCounter,
    pub websocket_connections: IntCounter,
    pub websocket_disconnections: IntCounter,
    
    // Gauge metrics
    pub active_models: IntGauge,
    pub active_forecasts: IntGauge,
    pub active_websocket_connections: IntGauge,
    pub memory_usage_bytes: Gauge,
    pub cpu_usage_percent: Gauge,
    
    // Histogram metrics
    pub request_duration: Histogram,
    pub forecast_duration: Histogram,
    pub model_training_duration: Histogram,
    pub websocket_message_size: Histogram,
    
    // Custom metrics storage
    custom_metrics: Arc<RwLock<HashMap<String, f64>>>,
    
    // System metrics collector
    system_metrics: Arc<SystemMetricsCollector>,
}

impl MetricsRegistry {
    /// Create a new metrics registry
    pub fn new() -> Self {
        let registry = Registry::new();
        
        // Initialize counter metrics
        let requests_total = IntCounter::new("nhits_requests_total", "Total number of API requests")
            .expect("metric can be created");
        let forecasts_created = IntCounter::new("nhits_forecasts_created_total", "Total number of forecasts created")
            .expect("metric can be created");
        let forecasts_completed = IntCounter::new("nhits_forecasts_completed_total", "Total number of forecasts completed")
            .expect("metric can be created");
        let forecasts_failed = IntCounter::new("nhits_forecasts_failed_total", "Total number of forecasts failed")
            .expect("metric can be created");
        let models_created = IntCounter::new("nhits_models_created_total", "Total number of models created")
            .expect("metric can be created");
        let models_trained = IntCounter::new("nhits_models_trained_total", "Total number of models trained")
            .expect("metric can be created");
        let websocket_connections = IntCounter::new("nhits_websocket_connections_total", "Total WebSocket connections")
            .expect("metric can be created");
        let websocket_disconnections = IntCounter::new("nhits_websocket_disconnections_total", "Total WebSocket disconnections")
            .expect("metric can be created");
        
        // Initialize gauge metrics
        let active_models = IntGauge::new("nhits_active_models", "Number of active models")
            .expect("metric can be created");
        let active_forecasts = IntGauge::new("nhits_active_forecasts", "Number of active forecast jobs")
            .expect("metric can be created");
        let active_websocket_connections = IntGauge::new("nhits_active_websocket_connections", "Number of active WebSocket connections")
            .expect("metric can be created");
        let memory_usage_bytes = Gauge::new("nhits_memory_usage_bytes", "Memory usage in bytes")
            .expect("metric can be created");
        let cpu_usage_percent = Gauge::new("nhits_cpu_usage_percent", "CPU usage percentage")
            .expect("metric can be created");
        
        // Initialize histogram metrics
        let request_duration = Histogram::with_opts(
            HistogramOpts::new("nhits_request_duration_seconds", "Request duration in seconds")
                .buckets(vec![0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0])
        ).expect("metric can be created");
        
        let forecast_duration = Histogram::with_opts(
            HistogramOpts::new("nhits_forecast_duration_seconds", "Forecast computation duration in seconds")
                .buckets(vec![0.1, 0.5, 1.0, 5.0, 10.0, 30.0, 60.0, 300.0, 600.0])
        ).expect("metric can be created");
        
        let model_training_duration = Histogram::with_opts(
            HistogramOpts::new("nhits_model_training_duration_seconds", "Model training duration in seconds")
                .buckets(vec![1.0, 10.0, 60.0, 300.0, 600.0, 1800.0, 3600.0, 7200.0])
        ).expect("metric can be created");
        
        let websocket_message_size = Histogram::with_opts(
            HistogramOpts::new("nhits_websocket_message_size_bytes", "WebSocket message size in bytes")
                .buckets(vec![64.0, 256.0, 1024.0, 4096.0, 16384.0, 65536.0])
        ).expect("metric can be created");
        
        // Register all metrics
        registry.register(Box::new(requests_total.clone())).expect("collector can be registered");
        registry.register(Box::new(forecasts_created.clone())).expect("collector can be registered");
        registry.register(Box::new(forecasts_completed.clone())).expect("collector can be registered");
        registry.register(Box::new(forecasts_failed.clone())).expect("collector can be registered");
        registry.register(Box::new(models_created.clone())).expect("collector can be registered");
        registry.register(Box::new(models_trained.clone())).expect("collector can be registered");
        registry.register(Box::new(websocket_connections.clone())).expect("collector can be registered");
        registry.register(Box::new(websocket_disconnections.clone())).expect("collector can be registered");
        
        registry.register(Box::new(active_models.clone())).expect("collector can be registered");
        registry.register(Box::new(active_forecasts.clone())).expect("collector can be registered");
        registry.register(Box::new(active_websocket_connections.clone())).expect("collector can be registered");
        registry.register(Box::new(memory_usage_bytes.clone())).expect("collector can be registered");
        registry.register(Box::new(cpu_usage_percent.clone())).expect("collector can be registered");
        
        registry.register(Box::new(request_duration.clone())).expect("collector can be registered");
        registry.register(Box::new(forecast_duration.clone())).expect("collector can be registered");
        registry.register(Box::new(model_training_duration.clone())).expect("collector can be registered");
        registry.register(Box::new(websocket_message_size.clone())).expect("collector can be registered");
        
        Self {
            registry,
            requests_total,
            forecasts_created,
            forecasts_completed,
            forecasts_failed,
            models_created,
            models_trained,
            websocket_connections,
            websocket_disconnections,
            active_models,
            active_forecasts,
            active_websocket_connections,
            memory_usage_bytes,
            cpu_usage_percent,
            request_duration,
            forecast_duration,
            model_training_duration,
            websocket_message_size,
            custom_metrics: Arc::new(RwLock::new(HashMap::new())),
            system_metrics: Arc::new(SystemMetricsCollector::new()),
        }
    }

    /// Initialize metrics collection
    pub async fn init(&self) -> Result<()> {
        info!("Initializing metrics collection");
        
        // Start background system metrics collection
        let system_metrics = self.system_metrics.clone();
        let memory_gauge = self.memory_usage_bytes.clone();
        let cpu_gauge = self.cpu_usage_percent.clone();
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(5));
            
            loop {
                interval.tick().await;
                
                if let Ok(memory) = system_metrics.get_memory_usage().await {
                    memory_gauge.set(memory);
                }
                
                if let Ok(cpu) = system_metrics.get_cpu_usage().await {
                    cpu_gauge.set(cpu);
                }
            }
        });
        
        Ok(())
    }

    /// Export metrics in Prometheus format
    pub async fn export_prometheus(&self) -> Result<String> {
        let encoder = prometheus::TextEncoder::new();
        let metric_families = self.registry.gather();
        encoder.encode_to_string(&metric_families)
            .map_err(|e| anyhow::anyhow!("Failed to encode metrics: {}", e))
    }

    /// Export metrics in JSON format
    pub async fn export_json(&self) -> Result<String> {
        let mut metrics = HashMap::new();
        
        // Counter metrics
        metrics.insert("requests_total", self.requests_total.get() as f64);
        metrics.insert("forecasts_created", self.forecasts_created.get() as f64);
        metrics.insert("forecasts_completed", self.forecasts_completed.get() as f64);
        metrics.insert("forecasts_failed", self.forecasts_failed.get() as f64);
        metrics.insert("models_created", self.models_created.get() as f64);
        metrics.insert("models_trained", self.models_trained.get() as f64);
        metrics.insert("websocket_connections", self.websocket_connections.get() as f64);
        metrics.insert("websocket_disconnections", self.websocket_disconnections.get() as f64);
        
        // Gauge metrics
        metrics.insert("active_models", self.active_models.get() as f64);
        metrics.insert("active_forecasts", self.active_forecasts.get() as f64);
        metrics.insert("active_websocket_connections", self.active_websocket_connections.get() as f64);
        metrics.insert("memory_usage_bytes", self.memory_usage_bytes.get());
        metrics.insert("cpu_usage_percent", self.cpu_usage_percent.get());
        
        // Add custom metrics
        let custom = self.custom_metrics.read().await;
        for (key, value) in custom.iter() {
            metrics.insert(key.as_str(), *value);
        }
        
        serde_json::to_string_pretty(&metrics)
            .map_err(|e| anyhow::anyhow!("Failed to serialize metrics: {}", e))
    }

    /// Increment request counter
    pub async fn increment_requests(&self) {
        self.requests_total.inc();
    }

    /// Increment forecasts created counter
    pub async fn increment_forecasts_created(&self) {
        self.forecasts_created.inc();
    }

    /// Increment forecasts completed counter
    pub async fn increment_forecasts_completed(&self) {
        self.forecasts_completed.inc();
    }

    /// Increment forecasts failed counter
    pub async fn increment_forecasts_failed(&self) {
        self.forecasts_failed.inc();
    }

    /// Increment models created counter
    pub async fn increment_models_created(&self) {
        self.models_created.inc();
    }

    /// Increment models trained counter
    pub async fn increment_models_trained(&self) {
        self.models_trained.inc();
    }

    /// Increment WebSocket connections
    pub async fn increment_websocket_connections(&self) {
        self.websocket_connections.inc();
        self.active_websocket_connections.inc();
    }

    /// Increment WebSocket disconnections
    pub async fn increment_websocket_disconnections(&self) {
        self.websocket_disconnections.inc();
        self.active_websocket_connections.dec();
    }

    /// Update system metrics
    pub async fn update_system_metrics(&self, models: usize, jobs: usize, connections: usize) {
        self.active_models.set(models as i64);
        self.active_forecasts.set(jobs as i64);
        self.active_websocket_connections.set(connections as i64);
    }

    /// Record request duration
    pub async fn record_request_duration(&self, duration: Duration) {
        self.request_duration.observe(duration.as_secs_f64());
    }

    /// Record forecast duration
    pub async fn record_forecast_duration(&self, duration: Duration) {
        self.forecast_duration.observe(duration.as_secs_f64());
    }

    /// Record model training duration
    pub async fn record_training_duration(&self, duration: Duration) {
        self.model_training_duration.observe(duration.as_secs_f64());
    }

    /// Record WebSocket message size
    pub async fn record_websocket_message_size(&self, size: usize) {
        self.websocket_message_size.observe(size as f64);
    }

    /// Set custom metric
    pub async fn set_custom_metric(&self, name: String, value: f64) {
        let mut metrics = self.custom_metrics.write().await;
        metrics.insert(name, value);
    }

    /// Get custom metric
    pub async fn get_custom_metric(&self, name: &str) -> Option<f64> {
        let metrics = self.custom_metrics.read().await;
        metrics.get(name).copied()
    }

    /// Get metrics summary
    pub async fn get_summary(&self) -> MetricsSummary {
        MetricsSummary {
            requests_total: self.requests_total.get(),
            forecasts_created: self.forecasts_created.get(),
            forecasts_completed: self.forecasts_completed.get(),
            forecasts_failed: self.forecasts_failed.get(),
            models_created: self.models_created.get(),
            models_trained: self.models_trained.get(),
            active_models: self.active_models.get(),
            active_forecasts: self.active_forecasts.get(),
            active_websocket_connections: self.active_websocket_connections.get(),
            memory_usage_bytes: self.memory_usage_bytes.get(),
            cpu_usage_percent: self.cpu_usage_percent.get(),
            timestamp: chrono::Utc::now(),
        }
    }
}

/// System metrics collector
pub struct SystemMetricsCollector {
    last_cpu_measurement: Arc<RwLock<Option<(Instant, f64)>>>,
}

impl SystemMetricsCollector {
    pub fn new() -> Self {
        Self {
            last_cpu_measurement: Arc::new(RwLock::new(None)),
        }
    }

    /// Get memory usage in bytes
    pub async fn get_memory_usage(&self) -> Result<f64> {
        // In a real implementation, you would use system APIs
        // For now, simulate memory usage
        #[cfg(target_os = "linux")]
        {
            match std::fs::read_to_string("/proc/self/status") {
                Ok(content) => {
                    for line in content.lines() {
                        if line.starts_with("VmRSS:") {
                            if let Some(kb_str) = line.split_whitespace().nth(1) {
                                if let Ok(kb) = kb_str.parse::<f64>() {
                                    return Ok(kb * 1024.0); // Convert KB to bytes
                                }
                            }
                        }
                    }
                }
                Err(_) => {}
            }
        }
        
        // Fallback: simulate memory usage
        Ok(1024.0 * 1024.0 * 100.0) // 100 MB
    }

    /// Get CPU usage percentage
    pub async fn get_cpu_usage(&self) -> Result<f64> {
        // Simplified CPU usage calculation
        // In production, use proper system monitoring libraries
        
        let mut last_measurement = self.last_cpu_measurement.write().await;
        let now = Instant::now();
        
        // Simulate CPU usage between 0-100%
        let usage = (now.elapsed().as_millis() % 100) as f64 / 100.0 * 50.0;
        
        *last_measurement = Some((now, usage));
        Ok(usage)
    }

    /// Get disk usage
    pub async fn get_disk_usage(&self) -> Result<(f64, f64)> {
        // Return (used_bytes, total_bytes)
        // In a real implementation, use statvfs or similar
        Ok((1024.0 * 1024.0 * 1024.0 * 50.0, 1024.0 * 1024.0 * 1024.0 * 100.0))
    }

    /// Get network statistics
    pub async fn get_network_stats(&self) -> Result<(u64, u64)> {
        // Return (bytes_received, bytes_sent)
        // In a real implementation, read from /proc/net/dev or similar
        Ok((1024 * 1024 * 100, 1024 * 1024 * 50))
    }
}

/// Metrics summary structure
#[derive(Debug, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub requests_total: u64,
    pub forecasts_created: u64,
    pub forecasts_completed: u64,
    pub forecasts_failed: u64,
    pub models_created: u64,
    pub models_trained: u64,
    pub active_models: i64,
    pub active_forecasts: i64,
    pub active_websocket_connections: i64,
    pub memory_usage_bytes: f64,
    pub cpu_usage_percent: f64,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

/// Performance metrics for specific operations
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub operation: String,
    pub duration_ms: f64,
    pub throughput: Option<f64>,
    pub memory_delta: Option<f64>,
    pub error_rate: Option<f64>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

impl PerformanceMetrics {
    pub fn new(operation: String) -> Self {
        Self {
            operation,
            duration_ms: 0.0,
            throughput: None,
            memory_delta: None,
            error_rate: None,
            timestamp: chrono::Utc::now(),
        }
    }

    pub fn with_duration(mut self, duration: Duration) -> Self {
        self.duration_ms = duration.as_millis() as f64;
        self
    }

    pub fn with_throughput(mut self, throughput: f64) -> Self {
        self.throughput = Some(throughput);
        self
    }

    pub fn with_memory_delta(mut self, delta: f64) -> Self {
        self.memory_delta = Some(delta);
        self
    }

    pub fn with_error_rate(mut self, rate: f64) -> Self {
        self.error_rate = Some(rate);
        self
    }
}

/// Health check metrics
#[derive(Debug, Serialize, Deserialize)]
pub struct HealthMetrics {
    pub status: String,
    pub uptime_seconds: f64,
    pub memory_usage: f64,
    pub cpu_usage: f64,
    pub active_connections: i64,
    pub error_rate: f64,
    pub last_error: Option<String>,
    pub timestamp: chrono::DateTime<chrono::Utc>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_metrics_registry_creation() {
        let metrics = MetricsRegistry::new();
        
        // Test counter increments
        metrics.increment_requests().await;
        metrics.increment_forecasts_created().await;
        
        assert_eq!(metrics.requests_total.get(), 1);
        assert_eq!(metrics.forecasts_created.get(), 1);
    }

    #[tokio::test]
    async fn test_system_metrics_collector() {
        let collector = SystemMetricsCollector::new();
        
        let memory = collector.get_memory_usage().await.unwrap();
        let cpu = collector.get_cpu_usage().await.unwrap();
        
        assert!(memory > 0.0);
        assert!(cpu >= 0.0 && cpu <= 100.0);
    }

    #[tokio::test]
    async fn test_custom_metrics() {
        let metrics = MetricsRegistry::new();
        
        metrics.set_custom_metric("test_metric".to_string(), 42.0).await;
        let value = metrics.get_custom_metric("test_metric").await;
        
        assert_eq!(value, Some(42.0));
    }

    #[tokio::test]
    async fn test_metrics_summary() {
        let metrics = MetricsRegistry::new();
        
        metrics.increment_requests().await;
        metrics.increment_models_created().await;
        
        let summary = metrics.get_summary().await;
        
        assert_eq!(summary.requests_total, 1);
        assert_eq!(summary.models_created, 1);
    }

    #[test]
    fn test_performance_metrics() {
        let metrics = PerformanceMetrics::new("test_operation".to_string())
            .with_duration(Duration::from_millis(100))
            .with_throughput(50.0)
            .with_error_rate(0.01);
        
        assert_eq!(metrics.operation, "test_operation");
        assert_eq!(metrics.duration_ms, 100.0);
        assert_eq!(metrics.throughput, Some(50.0));
        assert_eq!(metrics.error_rate, Some(0.01));
    }
}