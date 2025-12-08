//! Metrics and monitoring for ruv_FANN Integration
//!
//! This module provides comprehensive metrics collection and monitoring capabilities
//! for the ruv_FANN neural divergent integration system.

use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};
use serde::{Serialize, Deserialize};
use tokio::sync::RwLock;
use tracing::{info, debug, warn, error};

use crate::config::MetricsConfig;
use crate::error::{RuvFannError, RuvFannResult};
use crate::neural_divergent::DivergentOutput;

/// Comprehensive metrics for ruv_FANN integration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RuvFannMetrics {
    /// Module name
    pub module_name: String,
    
    /// Performance metrics
    pub performance: PerformanceMetrics,
    
    /// Neural network metrics
    pub neural_network: NeuralNetworkMetrics,
    
    /// GPU acceleration metrics
    pub gpu_acceleration: GPUAccelerationMetrics,
    
    /// Memory usage metrics
    pub memory_usage: MemoryUsageMetrics,
    
    /// Error metrics
    pub error_metrics: ErrorMetrics,
    
    /// Latency distribution
    pub latency_distribution: LatencyDistribution,
    
    /// Throughput metrics
    pub throughput: ThroughputMetrics,
    
    /// System metrics
    pub system_metrics: SystemMetrics,
    
    /// Timestamp of last update
    pub last_updated: SystemTime,
}

impl RuvFannMetrics {
    /// Create new metrics instance
    pub async fn new(module_name: &str) -> RuvFannResult<Self> {
        Ok(Self {
            module_name: module_name.to_string(),
            performance: PerformanceMetrics::new(),
            neural_network: NeuralNetworkMetrics::new(),
            gpu_acceleration: GPUAccelerationMetrics::new(),
            memory_usage: MemoryUsageMetrics::new(),
            error_metrics: ErrorMetrics::new(),
            latency_distribution: LatencyDistribution::new(),
            throughput: ThroughputMetrics::new(),
            system_metrics: SystemMetrics::new(),
            last_updated: SystemTime::now(),
        })
    }
    
    /// Record a prediction operation
    pub async fn record_prediction(&mut self, latency: Duration, prediction: &DivergentOutput) -> RuvFannResult<()> {
        self.performance.total_predictions.fetch_add(1, Ordering::Relaxed);
        
        // Update latency metrics
        self.latency_distribution.add_sample(latency);
        
        // Update throughput
        self.throughput.record_operation(Instant::now());
        
        // Update neural network metrics
        self.neural_network.record_divergent_processing(
            prediction.pathway_predictions.len(),
            prediction.divergence_metrics.pathway_diversity,
            prediction.divergence_metrics.convergence_strength,
        );
        
        // Update performance averages
        self.performance.update_averages(latency);
        
        self.last_updated = SystemTime::now();
        
        Ok(())
    }
    
    /// Record an error
    pub async fn record_error(&mut self, error: &RuvFannError) -> RuvFannResult<()> {
        self.error_metrics.record_error(error);
        self.last_updated = SystemTime::now();
        Ok(())
    }
    
    /// Record GPU operation
    pub async fn record_gpu_operation(&mut self, operation_type: &str, duration: Duration) -> RuvFannResult<()> {
        self.gpu_acceleration.record_operation(operation_type, duration);
        self.last_updated = SystemTime::now();
        Ok(())
    }
    
    /// Update memory usage
    pub async fn update_memory_usage(&mut self, used_bytes: u64, total_bytes: u64) -> RuvFannResult<()> {
        self.memory_usage.update(used_bytes, total_bytes);
        self.last_updated = SystemTime::now();
        Ok(())
    }
    
    /// Get summary statistics
    pub async fn get_summary(&self) -> RuvFannResult<MetricsSummary> {
        Ok(MetricsSummary {
            total_predictions: self.performance.total_predictions.load(Ordering::Relaxed),
            average_latency_us: self.performance.average_latency_us.load(Ordering::Relaxed),
            error_rate: self.error_metrics.get_error_rate(),
            throughput_ops_per_sec: self.throughput.get_current_throughput(),
            memory_utilization_percent: self.memory_usage.get_utilization_percent(),
            gpu_utilization_percent: self.gpu_acceleration.get_utilization_percent(),
            uptime: self.get_uptime(),
            last_inference_time: self.get_last_inference_time(),
        })
    }
    
    /// Save metrics to disk
    pub async fn save_to_disk(&self) -> RuvFannResult<()> {
        let filename = format!("ruv_fann_metrics_{}.json", self.module_name);
        let json_data = serde_json::to_string_pretty(self)?;
        tokio::fs::write(&filename, json_data).await
            .map_err(|e| RuvFannError::metrics_error(format!("Failed to save metrics: {}", e)))?;
        
        info!("Metrics saved to {}", filename);
        Ok(())
    }
    
    /// Load metrics from disk
    pub async fn load_from_disk(module_name: &str) -> RuvFannResult<Self> {
        let filename = format!("ruv_fann_metrics_{}.json", module_name);
        let json_data = tokio::fs::read_to_string(&filename).await
            .map_err(|e| RuvFannError::metrics_error(format!("Failed to load metrics: {}", e)))?;
        
        let metrics: Self = serde_json::from_str(&json_data)?;
        info!("Metrics loaded from {}", filename);
        Ok(metrics)
    }
    
    fn get_uptime(&self) -> Duration {
        SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default()
    }
    
    fn get_last_inference_time(&self) -> Option<chrono::DateTime<chrono::Utc>> {
        Some(chrono::DateTime::from(self.last_updated))
    }
}

/// Performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub total_predictions: AtomicU64,
    pub total_latency_us: AtomicU64,
    pub average_latency_us: AtomicU64,
    pub min_latency_us: AtomicU64,
    pub max_latency_us: AtomicU64,
    pub p95_latency_us: AtomicU64,
    pub p99_latency_us: AtomicU64,
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
}

impl PerformanceMetrics {
    fn new() -> Self {
        Self {
            total_predictions: AtomicU64::new(0),
            total_latency_us: AtomicU64::new(0),
            average_latency_us: AtomicU64::new(0),
            min_latency_us: AtomicU64::new(u64::MAX),
            max_latency_us: AtomicU64::new(0),
            p95_latency_us: AtomicU64::new(0),
            p99_latency_us: AtomicU64::new(0),
            cache_hits: AtomicU64::new(0),
            cache_misses: AtomicU64::new(0),
        }
    }
    
    fn update_averages(&self, latency: Duration) {
        let latency_us = latency.as_micros() as u64;
        
        // Update total and average
        let total_latency = self.total_latency_us.fetch_add(latency_us, Ordering::Relaxed) + latency_us;
        let total_predictions = self.total_predictions.load(Ordering::Relaxed);
        
        if total_predictions > 0 {
            self.average_latency_us.store(total_latency / total_predictions, Ordering::Relaxed);
        }
        
        // Update min/max
        let current_min = self.min_latency_us.load(Ordering::Relaxed);
        if latency_us < current_min {
            self.min_latency_us.store(latency_us, Ordering::Relaxed);
        }
        
        let current_max = self.max_latency_us.load(Ordering::Relaxed);
        if latency_us > current_max {
            self.max_latency_us.store(latency_us, Ordering::Relaxed);
        }
    }
}

/// Neural network specific metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NeuralNetworkMetrics {
    pub total_pathways_processed: AtomicU64,
    pub average_pathway_diversity: f64,
    pub average_convergence_strength: f64,
    pub adaptation_events: AtomicU64,
    pub divergent_processing_time_us: AtomicU64,
    pub pathway_contribution_variance: f64,
    pub uncertainty_estimation_accuracy: f64,
}

impl NeuralNetworkMetrics {
    fn new() -> Self {
        Self {
            total_pathways_processed: AtomicU64::new(0),
            average_pathway_diversity: 0.0,
            average_convergence_strength: 0.0,
            adaptation_events: AtomicU64::new(0),
            divergent_processing_time_us: AtomicU64::new(0),
            pathway_contribution_variance: 0.0,
            uncertainty_estimation_accuracy: 0.0,
        }
    }
    
    fn record_divergent_processing(
        &mut self,
        pathway_count: usize,
        diversity: f64,
        convergence_strength: f64,
    ) {
        self.total_pathways_processed.fetch_add(pathway_count as u64, Ordering::Relaxed);
        
        // Update running averages (simplified)
        self.average_pathway_diversity = (self.average_pathway_diversity + diversity) / 2.0;
        self.average_convergence_strength = (self.average_convergence_strength + convergence_strength) / 2.0;
    }
}

/// GPU acceleration metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GPUAccelerationMetrics {
    pub gpu_operations: HashMap<String, AtomicU64>,
    pub gpu_operation_times: HashMap<String, AtomicU64>,
    pub gpu_memory_usage_bytes: AtomicU64,
    pub gpu_utilization_percent: f64,
    pub shader_compilation_time_ms: AtomicU64,
    pub buffer_transfers: AtomicU64,
    pub compute_shader_executions: AtomicU64,
}

impl GPUAccelerationMetrics {
    fn new() -> Self {
        Self {
            gpu_operations: HashMap::new(),
            gpu_operation_times: HashMap::new(),
            gpu_memory_usage_bytes: AtomicU64::new(0),
            gpu_utilization_percent: 0.0,
            shader_compilation_time_ms: AtomicU64::new(0),
            buffer_transfers: AtomicU64::new(0),
            compute_shader_executions: AtomicU64::new(0),
        }
    }
    
    fn record_operation(&mut self, operation_type: &str, duration: Duration) {
        let key = operation_type.to_string();
        
        *self.gpu_operations.entry(key.clone()).or_insert_with(|| AtomicU64::new(0)) += 1;
        *self.gpu_operation_times.entry(key).or_insert_with(|| AtomicU64::new(0)) += duration.as_micros() as u64;
        
        self.compute_shader_executions.fetch_add(1, Ordering::Relaxed);
    }
    
    fn get_utilization_percent(&self) -> f64 {
        self.gpu_utilization_percent
    }
}

/// Memory usage metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryUsageMetrics {
    pub current_usage_bytes: AtomicU64,
    pub peak_usage_bytes: AtomicU64,
    pub total_available_bytes: AtomicU64,
    pub allocation_count: AtomicU64,
    pub deallocation_count: AtomicU64,
    pub pool_efficiency_percent: f64,
    pub garbage_collection_events: AtomicU64,
}

impl MemoryUsageMetrics {
    fn new() -> Self {
        Self {
            current_usage_bytes: AtomicU64::new(0),
            peak_usage_bytes: AtomicU64::new(0),
            total_available_bytes: AtomicU64::new(0),
            allocation_count: AtomicU64::new(0),
            deallocation_count: AtomicU64::new(0),
            pool_efficiency_percent: 0.0,
            garbage_collection_events: AtomicU64::new(0),
        }
    }
    
    fn update(&mut self, used_bytes: u64, total_bytes: u64) {
        self.current_usage_bytes.store(used_bytes, Ordering::Relaxed);
        self.total_available_bytes.store(total_bytes, Ordering::Relaxed);
        
        let current_peak = self.peak_usage_bytes.load(Ordering::Relaxed);
        if used_bytes > current_peak {
            self.peak_usage_bytes.store(used_bytes, Ordering::Relaxed);
        }
    }
    
    fn get_utilization_percent(&self) -> f64 {
        let used = self.current_usage_bytes.load(Ordering::Relaxed);
        let total = self.total_available_bytes.load(Ordering::Relaxed);
        
        if total > 0 {
            (used as f64 / total as f64) * 100.0
        } else {
            0.0
        }
    }
}

/// Error metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ErrorMetrics {
    pub total_errors: AtomicU64,
    pub error_counts_by_type: HashMap<String, AtomicU64>,
    pub critical_errors: AtomicU64,
    pub recoverable_errors: AtomicU64,
    pub error_rate_per_hour: f64,
    pub last_error_timestamp: Option<SystemTime>,
    pub mean_time_between_errors: Duration,
}

impl ErrorMetrics {
    fn new() -> Self {
        Self {
            total_errors: AtomicU64::new(0),
            error_counts_by_type: HashMap::new(),
            critical_errors: AtomicU64::new(0),
            recoverable_errors: AtomicU64::new(0),
            error_rate_per_hour: 0.0,
            last_error_timestamp: None,
            mean_time_between_errors: Duration::from_secs(0),
        }
    }
    
    fn record_error(&mut self, error: &RuvFannError) {
        self.total_errors.fetch_add(1, Ordering::Relaxed);
        
        // Count by error type
        let error_type = format!("{:?}", error).split('(').next().unwrap_or("Unknown").to_string();
        *self.error_counts_by_type.entry(error_type).or_insert_with(|| AtomicU64::new(0)) += 1;
        
        // Count by severity
        if error.is_critical() {
            self.critical_errors.fetch_add(1, Ordering::Relaxed);
        } else if error.is_recoverable() {
            self.recoverable_errors.fetch_add(1, Ordering::Relaxed);
        }
        
        self.last_error_timestamp = Some(SystemTime::now());
    }
    
    fn get_error_rate(&self) -> f64 {
        self.error_rate_per_hour
    }
}

/// Latency distribution tracking
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LatencyDistribution {
    samples: VecDeque<u64>,
    max_samples: usize,
    histogram_buckets: HashMap<String, AtomicU64>,
}

impl LatencyDistribution {
    fn new() -> Self {
        let mut histogram_buckets = HashMap::new();
        
        // Initialize latency buckets (in microseconds)
        let buckets = vec![
            "0-10us", "10-50us", "50-100us", "100-500us", 
            "500us-1ms", "1-5ms", "5-10ms", "10ms+"
        ];
        
        for bucket in buckets {
            histogram_buckets.insert(bucket.to_string(), AtomicU64::new(0));
        }
        
        Self {
            samples: VecDeque::with_capacity(10000),
            max_samples: 10000,
            histogram_buckets,
        }
    }
    
    fn add_sample(&mut self, latency: Duration) {
        let latency_us = latency.as_micros() as u64;
        
        // Add to samples
        if self.samples.len() >= self.max_samples {
            self.samples.pop_front();
        }
        self.samples.push_back(latency_us);
        
        // Update histogram
        let bucket = match latency_us {
            0..=10 => "0-10us",
            11..=50 => "10-50us",
            51..=100 => "50-100us",
            101..=500 => "100-500us",
            501..=1000 => "500us-1ms",
            1001..=5000 => "1-5ms",
            5001..=10000 => "5-10ms",
            _ => "10ms+",
        };
        
        if let Some(counter) = self.histogram_buckets.get(bucket) {
            counter.fetch_add(1, Ordering::Relaxed);
        }
    }
    
    fn get_percentile(&self, percentile: f64) -> Option<u64> {
        if self.samples.is_empty() {
            return None;
        }
        
        let mut sorted_samples: Vec<u64> = self.samples.iter().copied().collect();
        sorted_samples.sort_unstable();
        
        let index = ((percentile / 100.0) * (sorted_samples.len() - 1) as f64) as usize;
        Some(sorted_samples[index])
    }
}

/// Throughput metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ThroughputMetrics {
    operation_timestamps: VecDeque<Instant>,
    max_history: usize,
    current_throughput_ops_per_sec: f64,
    peak_throughput_ops_per_sec: f64,
}

impl ThroughputMetrics {
    fn new() -> Self {
        Self {
            operation_timestamps: VecDeque::with_capacity(10000),
            max_history: 10000,
            current_throughput_ops_per_sec: 0.0,
            peak_throughput_ops_per_sec: 0.0,
        }
    }
    
    fn record_operation(&mut self, timestamp: Instant) {
        // Remove old timestamps (older than 1 second)
        let cutoff = timestamp - Duration::from_secs(1);
        while let Some(&front) = self.operation_timestamps.front() {
            if front < cutoff {
                self.operation_timestamps.pop_front();
            } else {
                break;
            }
        }
        
        // Add new timestamp
        if self.operation_timestamps.len() >= self.max_history {
            self.operation_timestamps.pop_front();
        }
        self.operation_timestamps.push_back(timestamp);
        
        // Update current throughput
        self.current_throughput_ops_per_sec = self.operation_timestamps.len() as f64;
        
        // Update peak throughput
        if self.current_throughput_ops_per_sec > self.peak_throughput_ops_per_sec {
            self.peak_throughput_ops_per_sec = self.current_throughput_ops_per_sec;
        }
    }
    
    fn get_current_throughput(&self) -> f64 {
        self.current_throughput_ops_per_sec
    }
}

/// System-level metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SystemMetrics {
    pub cpu_usage_percent: f64,
    pub memory_usage_percent: f64,
    pub disk_usage_percent: f64,
    pub network_bytes_sent: AtomicU64,
    pub network_bytes_received: AtomicU64,
    pub process_id: u32,
    pub thread_count: AtomicUsize,
    pub file_descriptor_count: AtomicUsize,
}

impl SystemMetrics {
    fn new() -> Self {
        Self {
            cpu_usage_percent: 0.0,
            memory_usage_percent: 0.0,
            disk_usage_percent: 0.0,
            network_bytes_sent: AtomicU64::new(0),
            network_bytes_received: AtomicU64::new(0),
            process_id: std::process::id(),
            thread_count: AtomicUsize::new(1),
            file_descriptor_count: AtomicUsize::new(0),
        }
    }
}

/// Metrics summary for quick status checks
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MetricsSummary {
    pub total_predictions: u64,
    pub average_latency_us: u64,
    pub error_rate: f64,
    pub throughput_ops_per_sec: f64,
    pub memory_utilization_percent: f64,
    pub gpu_utilization_percent: f64,
    pub uptime: Duration,
    pub last_inference_time: Option<chrono::DateTime<chrono::Utc>>,
}

/// Neural divergent specific metrics
#[derive(Debug)]
pub struct NeuralDivergentMetrics {
    module_name: String,
    prediction_count: AtomicU64,
    total_processing_time: AtomicU64,
    pathway_diversity_samples: RwLock<VecDeque<f64>>,
    convergence_samples: RwLock<VecDeque<f64>>,
    adaptation_events: AtomicU64,
    uptime_start: Instant,
}

impl NeuralDivergentMetrics {
    /// Create new neural divergent metrics
    pub async fn new(module_name: &str) -> RuvFannResult<Self> {
        Ok(Self {
            module_name: module_name.to_string(),
            prediction_count: AtomicU64::new(0),
            total_processing_time: AtomicU64::new(0),
            pathway_diversity_samples: RwLock::new(VecDeque::with_capacity(1000)),
            convergence_samples: RwLock::new(VecDeque::with_capacity(1000)),
            adaptation_events: AtomicU64::new(0),
            uptime_start: Instant::now(),
        })
    }
    
    /// Record divergent processing
    pub async fn record_divergent_processing(
        &self,
        processing_time: Duration,
        output: &DivergentOutput,
    ) -> RuvFannResult<()> {
        self.prediction_count.fetch_add(1, Ordering::Relaxed);
        self.total_processing_time.fetch_add(processing_time.as_micros() as u64, Ordering::Relaxed);
        
        // Record pathway diversity
        {
            let mut diversity_samples = self.pathway_diversity_samples.write().await;
            if diversity_samples.len() >= 1000 {
                diversity_samples.pop_front();
            }
            diversity_samples.push_back(output.divergence_metrics.pathway_diversity);
        }
        
        // Record convergence strength
        {
            let mut convergence_samples = self.convergence_samples.write().await;
            if convergence_samples.len() >= 1000 {
                convergence_samples.pop_front();
            }
            convergence_samples.push_back(output.divergence_metrics.convergence_strength);
        }
        
        Ok(())
    }
    
    /// Get summary of neural divergent metrics
    pub async fn get_summary(&self) -> RuvFannResult<MetricsSummary> {
        let total_predictions = self.prediction_count.load(Ordering::Relaxed);
        let total_time_us = self.total_processing_time.load(Ordering::Relaxed);
        
        let average_latency_us = if total_predictions > 0 {
            total_time_us / total_predictions
        } else {
            0
        };
        
        Ok(MetricsSummary {
            total_predictions,
            average_latency_us,
            error_rate: 0.0, // Would be calculated from error tracking
            throughput_ops_per_sec: 0.0, // Would be calculated from timing
            memory_utilization_percent: 0.0, // Would be tracked separately
            gpu_utilization_percent: 0.0, // Would be tracked separately
            uptime: self.uptime_start.elapsed(),
            last_inference_time: Some(chrono::Utc::now()),
        })
    }
    
    /// Save metrics to disk
    pub async fn save_to_disk(&self) -> RuvFannResult<()> {
        let summary = self.get_summary().await?;
        let filename = format!("neural_divergent_metrics_{}.json", self.module_name);
        let json_data = serde_json::to_string_pretty(&summary)?;
        
        tokio::fs::write(&filename, json_data).await
            .map_err(|e| RuvFannError::metrics_error(format!("Failed to save metrics: {}", e)))?;
        
        info!("Neural divergent metrics saved to {}", filename);
        Ok(())
    }
}

impl Clone for NeuralDivergentMetrics {
    fn clone(&self) -> Self {
        Self {
            module_name: self.module_name.clone(),
            prediction_count: AtomicU64::new(self.prediction_count.load(Ordering::Relaxed)),
            total_processing_time: AtomicU64::new(self.total_processing_time.load(Ordering::Relaxed)),
            pathway_diversity_samples: RwLock::new(VecDeque::new()),
            convergence_samples: RwLock::new(VecDeque::new()),
            adaptation_events: AtomicU64::new(self.adaptation_events.load(Ordering::Relaxed)),
            uptime_start: self.uptime_start,
        }
    }
}

/// Metrics collector for aggregating metrics from multiple sources
pub struct MetricsCollector {
    config: MetricsConfig,
    metrics_storage: RwLock<HashMap<String, RuvFannMetrics>>,
    collection_interval: Duration,
    export_interval: Duration,
}

impl MetricsCollector {
    /// Create new metrics collector
    pub fn new(config: MetricsConfig) -> Self {
        Self {
            collection_interval: config.collection_frequency,
            export_interval: config.storage_duration,
            config,
            metrics_storage: RwLock::new(HashMap::new()),
        }
    }
    
    /// Register a metrics source
    pub async fn register_source(&self, name: String, metrics: RuvFannMetrics) -> RuvFannResult<()> {
        let mut storage = self.metrics_storage.write().await;
        storage.insert(name, metrics);
        Ok(())
    }
    
    /// Start automatic metrics collection
    pub async fn start_collection(&self) -> RuvFannResult<()> {
        info!("Starting metrics collection with {}s interval", self.collection_interval.as_secs());
        
        let mut interval = tokio::time::interval(self.collection_interval);
        
        loop {
            interval.tick().await;
            
            if let Err(e) = self.collect_all_metrics().await {
                error!("Failed to collect metrics: {}", e);
            }
            
            if let Err(e) = self.export_metrics_if_needed().await {
                error!("Failed to export metrics: {}", e);
            }
        }
    }
    
    async fn collect_all_metrics(&self) -> RuvFannResult<()> {
        let storage = self.metrics_storage.read().await;
        
        for (name, metrics) in storage.iter() {
            debug!("Collecting metrics for: {}", name);
            // In a real implementation, this would update metrics from their sources
        }
        
        Ok(())
    }
    
    async fn export_metrics_if_needed(&self) -> RuvFannResult<()> {
        if let Some(export_path) = &self.config.export_path {
            let storage = self.metrics_storage.read().await;
            
            for (name, metrics) in storage.iter() {
                let filename = format!("{}/metrics_{}_{}.json", 
                    export_path, 
                    name, 
                    chrono::Utc::now().format("%Y%m%d_%H%M%S")
                );
                
                let json_data = serde_json::to_string_pretty(metrics)?;
                tokio::fs::write(&filename, json_data).await
                    .map_err(|e| RuvFannError::metrics_error(format!("Failed to export metrics: {}", e)))?;
            }
        }
        
        Ok(())
    }
    
    /// Get aggregated metrics summary
    pub async fn get_aggregated_summary(&self) -> RuvFannResult<HashMap<String, MetricsSummary>> {
        let storage = self.metrics_storage.read().await;
        let mut summaries = HashMap::new();
        
        for (name, metrics) in storage.iter() {
            let summary = metrics.get_summary().await?;
            summaries.insert(name.clone(), summary);
        }
        
        Ok(summaries)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_ruv_fann_metrics_creation() {
        let metrics = RuvFannMetrics::new("test_module").await.unwrap();
        assert_eq!(metrics.module_name, "test_module");
    }
    
    #[tokio::test]
    async fn test_neural_divergent_metrics() {
        let metrics = NeuralDivergentMetrics::new("test_divergent").await.unwrap();
        assert_eq!(metrics.module_name, "test_divergent");
        
        let summary = metrics.get_summary().await.unwrap();
        assert_eq!(summary.total_predictions, 0);
    }
    
    #[test]
    fn test_latency_distribution() {
        let mut dist = LatencyDistribution::new();
        
        dist.add_sample(Duration::from_micros(50));
        dist.add_sample(Duration::from_micros(150));
        dist.add_sample(Duration::from_micros(25));
        
        assert_eq!(dist.samples.len(), 3);
        
        let p50 = dist.get_percentile(50.0);
        assert!(p50.is_some());
    }
    
    #[test]
    fn test_throughput_metrics() {
        let mut throughput = ThroughputMetrics::new();
        let now = Instant::now();
        
        throughput.record_operation(now);
        throughput.record_operation(now + Duration::from_millis(100));
        
        assert!(throughput.get_current_throughput() > 0.0);
    }
    
    #[tokio::test]
    async fn test_metrics_collector() {
        let config = MetricsConfig::default();
        let collector = MetricsCollector::new(config);
        
        let metrics = RuvFannMetrics::new("test").await.unwrap();
        collector.register_source("test".to_string(), metrics).await.unwrap();
        
        let summaries = collector.get_aggregated_summary().await.unwrap();
        assert!(summaries.contains_key("test"));
    }
}