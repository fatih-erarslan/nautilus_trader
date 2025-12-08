use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use anyhow::{Result, anyhow};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug, instrument};
use prometheus::{Counter, Gauge, Histogram, Registry, Encoder, TextEncoder};
use metrics::{increment_counter, gauge, histogram};
use opentelemetry::{trace::{Tracer, TracerProvider}, Context, KeyValue};
use opentelemetry_jaeger::JaegerTraceExporter;
use tracing_opentelemetry::OpenTelemetrySpanExt;
use sysinfo::{System, SystemExt, CpuExt, NetworkExt, DiskExt};

/// Neural Network Performance Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralMetrics {
    pub inference_latency_ns: u64,
    pub throughput_ops_per_sec: f64,
    pub accuracy_percentage: f64,
    pub memory_usage_mb: f64,
    pub gpu_utilization_percent: f64,
    pub network_io_mbps: f64,
    pub error_rate_percent: f64,
    pub timestamp: u64,
}

/// System Health Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemHealthMetrics {
    pub cpu_usage_percent: f64,
    pub memory_usage_percent: f64,
    pub disk_usage_percent: f64,
    pub network_rx_mbps: f64,
    pub network_tx_mbps: f64,
    pub open_file_descriptors: u64,
    pub thread_count: u32,
    pub process_count: u32,
    pub timestamp: u64,
}

/// Trading System Metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TradingMetrics {
    pub orders_per_second: f64,
    pub order_fill_latency_ms: f64,
    pub market_data_latency_ms: f64,
    pub risk_score: f64,
    pub pnl_realtime: f64,
    pub portfolio_value: f64,
    pub open_positions: u32,
    pub alerts_triggered: u32,
    pub timestamp: u64,
}

/// Distributed Tracing Context
#[derive(Debug, Clone)]
pub struct TraceContext {
    pub trace_id: String,
    pub span_id: String,
    pub operation: String,
    pub start_time: Instant,
    pub tags: HashMap<String, String>,
}

/// Anomaly Detection Alert
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyAlert {
    pub alert_id: String,
    pub severity: AlertSeverity,
    pub component: String,
    pub metric_name: String,
    pub current_value: f64,
    pub threshold: f64,
    pub description: String,
    pub timestamp: u64,
    pub recommended_action: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AlertSeverity {
    Critical,
    High,
    Medium,
    Low,
    Info,
}

/// Comprehensive Observability Manager
pub struct ObservabilityManager {
    metrics_registry: Arc<Registry>,
    neural_metrics: Arc<RwLock<Vec<NeuralMetrics>>>,
    system_metrics: Arc<RwLock<Vec<SystemHealthMetrics>>>,
    trading_metrics: Arc<RwLock<Vec<TradingMetrics>>>,
    active_traces: Arc<RwLock<HashMap<String, TraceContext>>>,
    anomaly_alerts: Arc<RwLock<Vec<AnomalyAlert>>>,
    system_monitor: Arc<Mutex<System>>,
    
    // Prometheus metrics
    neural_latency_histogram: Histogram,
    neural_throughput_gauge: Gauge,
    neural_accuracy_gauge: Gauge,
    neural_error_counter: Counter,
    system_cpu_gauge: Gauge,
    system_memory_gauge: Gauge,
    trading_orders_counter: Counter,
    trading_latency_histogram: Histogram,
    risk_score_gauge: Gauge,
}

impl ObservabilityManager {
    pub fn new() -> Result<Self> {
        let registry = Arc::new(Registry::new());
        
        // Initialize Prometheus metrics
        let neural_latency_histogram = Histogram::with_opts(
            prometheus::HistogramOpts::new(
                "neural_inference_latency_seconds",
                "Neural network inference latency in seconds"
            ).buckets(vec![0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0])
        )?;
        
        let neural_throughput_gauge = Gauge::new(
            "neural_throughput_ops_per_sec",
            "Neural network throughput in operations per second"
        )?;
        
        let neural_accuracy_gauge = Gauge::new(
            "neural_accuracy_percentage",
            "Neural network accuracy percentage"
        )?;
        
        let neural_error_counter = Counter::new(
            "neural_errors_total",
            "Total number of neural network errors"
        )?;
        
        let system_cpu_gauge = Gauge::new(
            "system_cpu_usage_percentage",
            "System CPU usage percentage"
        )?;
        
        let system_memory_gauge = Gauge::new(
            "system_memory_usage_percentage", 
            "System memory usage percentage"
        )?;
        
        let trading_orders_counter = Counter::new(
            "trading_orders_total",
            "Total number of trading orders processed"
        )?;
        
        let trading_latency_histogram = Histogram::with_opts(
            prometheus::HistogramOpts::new(
                "trading_order_latency_seconds",
                "Trading order processing latency in seconds"
            ).buckets(vec![0.0001, 0.001, 0.01, 0.1, 1.0, 10.0])
        )?;
        
        let risk_score_gauge = Gauge::new(
            "trading_risk_score",
            "Current trading risk score"
        )?;
        
        // Register all metrics
        registry.register(Box::new(neural_latency_histogram.clone()))?;
        registry.register(Box::new(neural_throughput_gauge.clone()))?;
        registry.register(Box::new(neural_accuracy_gauge.clone()))?;
        registry.register(Box::new(neural_error_counter.clone()))?;
        registry.register(Box::new(system_cpu_gauge.clone()))?;
        registry.register(Box::new(system_memory_gauge.clone()))?;
        registry.register(Box::new(trading_orders_counter.clone()))?;
        registry.register(Box::new(trading_latency_histogram.clone()))?;
        registry.register(Box::new(risk_score_gauge.clone()))?;
        
        Ok(Self {
            metrics_registry: registry,
            neural_metrics: Arc::new(RwLock::new(Vec::new())),
            system_metrics: Arc::new(RwLock::new(Vec::new())),
            trading_metrics: Arc::new(RwLock::new(Vec::new())),
            active_traces: Arc::new(RwLock::new(HashMap::new())),
            anomaly_alerts: Arc::new(RwLock::new(Vec::new())),
            system_monitor: Arc::new(Mutex::new(System::new_all())),
            neural_latency_histogram,
            neural_throughput_gauge,
            neural_accuracy_gauge,
            neural_error_counter,
            system_cpu_gauge,
            system_memory_gauge,
            trading_orders_counter,
            trading_latency_histogram,
            risk_score_gauge,
        })
    }
    
    /// Record neural network performance metrics
    #[instrument(skip(self))]
    pub async fn record_neural_metrics(&self, metrics: NeuralMetrics) -> Result<()> {
        // Update Prometheus metrics
        self.neural_latency_histogram.observe(metrics.inference_latency_ns as f64 / 1_000_000_000.0);
        self.neural_throughput_gauge.set(metrics.throughput_ops_per_sec);
        self.neural_accuracy_gauge.set(metrics.accuracy_percentage);
        
        if metrics.error_rate_percent > 0.0 {
            self.neural_error_counter.inc();
        }
        
        // Store historical data
        let mut neural_data = self.neural_metrics.write().await;
        neural_data.push(metrics.clone());
        
        // Keep only last 10,000 metrics for memory efficiency
        if neural_data.len() > 10_000 {
            neural_data.drain(0..neural_data.len() - 10_000);
        }
        
        // Check for anomalies
        self.check_neural_anomalies(&metrics).await?;
        
        debug!("Recorded neural metrics: {:?}", metrics);
        Ok(())
    }
    
    /// Collect and record system health metrics
    #[instrument(skip(self))]
    pub async fn collect_system_metrics(&self) -> Result<SystemHealthMetrics> {
        let mut system = self.system_monitor.lock().unwrap();
        system.refresh_all();
        
        let cpu_usage = system.global_cpu_info().cpu_usage() as f64;
        let memory_usage = (system.used_memory() as f64 / system.total_memory() as f64) * 100.0;
        
        let disk_usage = system.disks()
            .iter()
            .map(|disk| {
                let used = disk.total_space() - disk.available_space();
                (used as f64 / disk.total_space() as f64) * 100.0
            })
            .fold(0.0, f64::max);
        
        let (network_rx, network_tx) = system.networks()
            .iter()
            .fold((0, 0), |(rx, tx), (_, data)| {
                (rx + data.received(), tx + data.transmitted())
            });
        
        let metrics = SystemHealthMetrics {
            cpu_usage_percent: cpu_usage,
            memory_usage_percent: memory_usage,
            disk_usage_percent: disk_usage,
            network_rx_mbps: network_rx as f64 / 1_000_000.0,
            network_tx_mbps: network_tx as f64 / 1_000_000.0,
            open_file_descriptors: 0, // Platform-specific implementation needed
            thread_count: 0,
            process_count: system.processes().len() as u32,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis() as u64,
        };
        
        // Update Prometheus metrics
        self.system_cpu_gauge.set(metrics.cpu_usage_percent);
        self.system_memory_gauge.set(metrics.memory_usage_percent);
        
        // Store metrics
        let mut system_data = self.system_metrics.write().await;
        system_data.push(metrics.clone());
        
        if system_data.len() > 10_000 {
            system_data.drain(0..system_data.len() - 10_000);
        }
        
        // Check for system anomalies
        self.check_system_anomalies(&metrics).await?;
        
        Ok(metrics)
    }
    
    /// Record trading system metrics
    #[instrument(skip(self))]
    pub async fn record_trading_metrics(&self, metrics: TradingMetrics) -> Result<()> {
        // Update Prometheus metrics
        self.trading_orders_counter.inc_by(metrics.orders_per_second as u64);
        self.trading_latency_histogram.observe(metrics.order_fill_latency_ms / 1000.0);
        self.risk_score_gauge.set(metrics.risk_score);
        
        // Store historical data
        let mut trading_data = self.trading_metrics.write().await;
        trading_data.push(metrics.clone());
        
        if trading_data.len() > 10_000 {
            trading_data.drain(0..trading_data.len() - 10_000);
        }
        
        // Check for trading anomalies
        self.check_trading_anomalies(&metrics).await?;
        
        info!("Recorded trading metrics: orders/sec={}, latency={}ms, risk_score={}", 
              metrics.orders_per_second, metrics.order_fill_latency_ms, metrics.risk_score);
        Ok(())
    }
    
    /// Start a distributed trace
    #[instrument(skip(self))]
    pub async fn start_trace(&self, operation: &str, tags: HashMap<String, String>) -> Result<String> {
        let trace_id = format!("trace-{}-{}", 
                              SystemTime::now().duration_since(UNIX_EPOCH)?.as_nanos(),
                              rand::random::<u32>());
        let span_id = format!("span-{}", rand::random::<u64>());
        
        let context = TraceContext {
            trace_id: trace_id.clone(),
            span_id,
            operation: operation.to_string(),
            start_time: Instant::now(),
            tags,
        };
        
        self.active_traces.write().await.insert(trace_id.clone(), context);
        
        debug!("Started trace: {} for operation: {}", trace_id, operation);
        Ok(trace_id)
    }
    
    /// End a distributed trace
    #[instrument(skip(self))]
    pub async fn end_trace(&self, trace_id: &str, success: bool, error_msg: Option<String>) -> Result<Duration> {
        let mut traces = self.active_traces.write().await;
        
        if let Some(context) = traces.remove(trace_id) {
            let duration = context.start_time.elapsed();
            
            if success {
                info!("Trace {} completed successfully in {:?}", trace_id, duration);
            } else {
                error!("Trace {} failed in {:?}: {}", trace_id, duration, 
                       error_msg.unwrap_or_default());
            }
            
            Ok(duration)
        } else {
            Err(anyhow!("Trace not found: {}", trace_id))
        }
    }
    
    /// Check for neural network anomalies
    async fn check_neural_anomalies(&self, metrics: &NeuralMetrics) -> Result<()> {
        let mut alerts = Vec::new();
        
        // Latency anomaly detection
        if metrics.inference_latency_ns > 10_000_000 { // > 10ms
            alerts.push(AnomalyAlert {
                alert_id: format!("neural-latency-{}", metrics.timestamp),
                severity: AlertSeverity::High,
                component: "Neural Network".to_string(),
                metric_name: "inference_latency_ns".to_string(),
                current_value: metrics.inference_latency_ns as f64,
                threshold: 10_000_000.0,
                description: "Neural network inference latency exceeded 10ms threshold".to_string(),
                timestamp: metrics.timestamp,
                recommended_action: "Check GPU utilization and model complexity".to_string(),
            });
        }
        
        // Accuracy degradation detection
        if metrics.accuracy_percentage < 95.0 {
            alerts.push(AnomalyAlert {
                alert_id: format!("neural-accuracy-{}", metrics.timestamp),
                severity: AlertSeverity::Critical,
                component: "Neural Network".to_string(),
                metric_name: "accuracy_percentage".to_string(),
                current_value: metrics.accuracy_percentage,
                threshold: 95.0,
                description: "Neural network accuracy below 95% threshold".to_string(),
                timestamp: metrics.timestamp,
                recommended_action: "Retrain model or check input data quality".to_string(),
            });
        }
        
        // Error rate detection
        if metrics.error_rate_percent > 1.0 {
            alerts.push(AnomalyAlert {
                alert_id: format!("neural-errors-{}", metrics.timestamp),
                severity: AlertSeverity::Medium,
                component: "Neural Network".to_string(),
                metric_name: "error_rate_percent".to_string(),
                current_value: metrics.error_rate_percent,
                threshold: 1.0,
                description: "Neural network error rate above 1% threshold".to_string(),
                timestamp: metrics.timestamp,
                recommended_action: "Investigate input validation and model stability".to_string(),
            });
        }
        
        if !alerts.is_empty() {
            let mut alert_storage = self.anomaly_alerts.write().await;
            for alert in alerts {
                warn!("Anomaly detected: {:?}", alert);
                alert_storage.push(alert);
            }
        }
        
        Ok(())
    }
    
    /// Check for system health anomalies
    async fn check_system_anomalies(&self, metrics: &SystemHealthMetrics) -> Result<()> {
        let mut alerts = Vec::new();
        
        // CPU usage anomaly
        if metrics.cpu_usage_percent > 90.0 {
            alerts.push(AnomalyAlert {
                alert_id: format!("system-cpu-{}", metrics.timestamp),
                severity: AlertSeverity::High,
                component: "System".to_string(),
                metric_name: "cpu_usage_percent".to_string(),
                current_value: metrics.cpu_usage_percent,
                threshold: 90.0,
                description: "System CPU usage above 90%".to_string(),
                timestamp: metrics.timestamp,
                recommended_action: "Scale horizontally or optimize CPU-intensive operations".to_string(),
            });
        }
        
        // Memory usage anomaly
        if metrics.memory_usage_percent > 85.0 {
            alerts.push(AnomalyAlert {
                alert_id: format!("system-memory-{}", metrics.timestamp),
                severity: AlertSeverity::High,
                component: "System".to_string(),
                metric_name: "memory_usage_percent".to_string(),
                current_value: metrics.memory_usage_percent,
                threshold: 85.0,
                description: "System memory usage above 85%".to_string(),
                timestamp: metrics.timestamp,
                recommended_action: "Increase memory or optimize memory usage".to_string(),
            });
        }
        
        // Disk usage anomaly
        if metrics.disk_usage_percent > 95.0 {
            alerts.push(AnomalyAlert {
                alert_id: format!("system-disk-{}", metrics.timestamp),
                severity: AlertSeverity::Critical,
                component: "System".to_string(),
                metric_name: "disk_usage_percent".to_string(),
                current_value: metrics.disk_usage_percent,
                threshold: 95.0,
                description: "System disk usage above 95%".to_string(),
                timestamp: metrics.timestamp,
                recommended_action: "Clean up disk space or add storage capacity".to_string(),
            });
        }
        
        if !alerts.is_empty() {
            let mut alert_storage = self.anomaly_alerts.write().await;
            for alert in alerts {
                warn!("System anomaly detected: {:?}", alert);
                alert_storage.push(alert);
            }
        }
        
        Ok(())
    }
    
    /// Check for trading system anomalies
    async fn check_trading_anomalies(&self, metrics: &TradingMetrics) -> Result<()> {
        let mut alerts = Vec::new();
        
        // Order latency anomaly
        if metrics.order_fill_latency_ms > 100.0 {
            alerts.push(AnomalyAlert {
                alert_id: format!("trading-latency-{}", metrics.timestamp),
                severity: AlertSeverity::High,
                component: "Trading".to_string(),
                metric_name: "order_fill_latency_ms".to_string(),
                current_value: metrics.order_fill_latency_ms,
                threshold: 100.0,
                description: "Order fill latency above 100ms".to_string(),
                timestamp: metrics.timestamp,
                recommended_action: "Check market connectivity and order routing".to_string(),
            });
        }
        
        // Risk score anomaly
        if metrics.risk_score > 0.8 {
            alerts.push(AnomalyAlert {
                alert_id: format!("trading-risk-{}", metrics.timestamp),
                severity: AlertSeverity::Critical,
                component: "Trading".to_string(),
                metric_name: "risk_score".to_string(),
                current_value: metrics.risk_score,
                threshold: 0.8,
                description: "Trading risk score above 80%".to_string(),
                timestamp: metrics.timestamp,
                recommended_action: "Reduce position sizes and review risk parameters".to_string(),
            });
        }
        
        if !alerts.is_empty() {
            let mut alert_storage = self.anomaly_alerts.write().await;
            for alert in alerts {
                error!("Trading anomaly detected: {:?}", alert);
                alert_storage.push(alert);
            }
        }
        
        Ok(())
    }
    
    /// Export Prometheus metrics
    pub fn export_prometheus_metrics(&self) -> Result<String> {
        let encoder = TextEncoder::new();
        let metric_families = self.metrics_registry.gather();
        let mut buffer = Vec::new();
        encoder.encode(&metric_families, &mut buffer)?;
        Ok(String::from_utf8(buffer)?)
    }
    
    /// Get current anomaly alerts
    pub async fn get_active_alerts(&self) -> Vec<AnomalyAlert> {
        self.anomaly_alerts.read().await.clone()
    }
    
    /// Get neural metrics summary
    pub async fn get_neural_metrics_summary(&self) -> Result<Option<NeuralMetrics>> {
        let metrics = self.neural_metrics.read().await;
        Ok(metrics.last().cloned())
    }
    
    /// Get system health summary
    pub async fn get_system_health_summary(&self) -> Result<Option<SystemHealthMetrics>> {
        let metrics = self.system_metrics.read().await;
        Ok(metrics.last().cloned())
    }
    
    /// Get trading metrics summary
    pub async fn get_trading_metrics_summary(&self) -> Result<Option<TradingMetrics>> {
        let metrics = self.trading_metrics.read().await;
        Ok(metrics.last().cloned())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_neural_metrics_recording() {
        let observability = ObservabilityManager::new().unwrap();
        
        let metrics = NeuralMetrics {
            inference_latency_ns: 5_000_000,
            throughput_ops_per_sec: 1000.0,
            accuracy_percentage: 97.5,
            memory_usage_mb: 2048.0,
            gpu_utilization_percent: 85.0,
            network_io_mbps: 100.0,
            error_rate_percent: 0.1,
            timestamp: SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_millis() as u64,
        };
        
        let result = observability.record_neural_metrics(metrics).await;
        assert!(result.is_ok());
        
        let summary = observability.get_neural_metrics_summary().await.unwrap();
        assert!(summary.is_some());
    }
    
    #[tokio::test]
    async fn test_system_metrics_collection() {
        let observability = ObservabilityManager::new().unwrap();
        
        let result = observability.collect_system_metrics().await;
        assert!(result.is_ok());
        
        let metrics = result.unwrap();
        assert!(metrics.cpu_usage_percent >= 0.0);
        assert!(metrics.memory_usage_percent >= 0.0);
    }
    
    #[tokio::test]
    async fn test_distributed_tracing() {
        let observability = ObservabilityManager::new().unwrap();
        
        let trace_id = observability.start_trace(
            "test_operation", 
            HashMap::from([("component".to_string(), "test".to_string())])
        ).await.unwrap();
        
        tokio::time::sleep(Duration::from_millis(10)).await;
        
        let duration = observability.end_trace(&trace_id, true, None).await.unwrap();
        assert!(duration.as_millis() >= 10);
    }
}