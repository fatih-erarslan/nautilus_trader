/*!
Performance monitoring and optimization for neural integration
============================================================
*/

use crate::NeuralConfig;
use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::sync::RwLock;
use tracing::{info, warn, debug};

/// Performance monitoring system
#[derive(Debug, Clone)]
pub struct PerformanceMonitor {
    config: NeuralConfig,
    metrics: Arc<RwLock<PerformanceMetrics>>,
    model_metrics: Arc<RwLock<HashMap<String, ModelPerformanceMetrics>>>,
    start_time: Instant,
}

/// System-wide performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total predictions made
    pub total_predictions: u64,
    /// Total successful predictions
    pub successful_predictions: u64,
    /// Total execution time in microseconds
    pub total_execution_time_us: u64,
    /// Average execution time in microseconds
    pub avg_execution_time_us: f64,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f64,
    /// Predictions per second
    pub predictions_per_second: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// CPU utilization percentage
    pub cpu_utilization: f64,
    /// GPU utilization percentage (if available)
    #[cfg(feature = "gpu")]
    pub gpu_utilization: f64,
    /// Last updated timestamp
    pub last_updated: chrono::DateTime<chrono::Utc>,
}

/// Per-model performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelPerformanceMetrics {
    /// Model identifier
    pub model_id: String,
    /// Total predictions for this model
    pub predictions: u64,
    /// Average execution time in microseconds
    pub avg_execution_time_us: f64,
    /// Min execution time in microseconds
    pub min_execution_time_us: u64,
    /// Max execution time in microseconds
    pub max_execution_time_us: u64,
    /// Success rate for this model
    pub success_rate: f64,
    /// Memory usage in bytes
    pub memory_usage_bytes: usize,
    /// Accuracy score (if available)
    pub accuracy: Option<f64>,
    /// Sharpe ratio (if available)
    pub sharpe_ratio: Option<f64>,
    /// Last prediction timestamp
    pub last_prediction: chrono::DateTime<chrono::Utc>,
}

impl Default for PerformanceMetrics {
    fn default() -> Self {
        Self {
            total_predictions: 0,
            successful_predictions: 0,
            total_execution_time_us: 0,
            avg_execution_time_us: 0.0,
            success_rate: 0.0,
            predictions_per_second: 0.0,
            memory_usage_bytes: 0,
            cpu_utilization: 0.0,
            #[cfg(feature = "gpu")]
            gpu_utilization: 0.0,
            last_updated: chrono::Utc::now(),
        }
    }
}

impl PerformanceMonitor {
    /// Create a new performance monitor
    pub fn new(config: &NeuralConfig) -> Result<Self> {
        Ok(Self {
            config: config.clone(),
            metrics: Arc::new(RwLock::new(PerformanceMetrics::default())),
            model_metrics: Arc::new(RwLock::new(HashMap::new())),
            start_time: Instant::now(),
        })
    }
    
    /// Start the performance monitoring system
    pub async fn start(&self) -> Result<()> {
        info!("Starting performance monitoring system");
        
        // Start background monitoring task
        let monitor = self.clone();
        tokio::spawn(async move {
            monitor.monitoring_loop().await;
        });
        
        info!("Performance monitoring system started");
        Ok(())
    }
    
    /// Record a prediction for performance tracking
    pub async fn record_prediction(&self, model_id: &str, execution_time_us: u64) {
        let mut metrics = self.metrics.write().await;
        let mut model_metrics = self.model_metrics.write().await;
        
        // Update system metrics
        metrics.total_predictions += 1;
        metrics.successful_predictions += 1;
        metrics.total_execution_time_us += execution_time_us;
        metrics.avg_execution_time_us = 
            metrics.total_execution_time_us as f64 / metrics.total_predictions as f64;
        metrics.success_rate = 
            metrics.successful_predictions as f64 / metrics.total_predictions as f64;
        
        // Calculate predictions per second
        let elapsed_seconds = self.start_time.elapsed().as_secs_f64();
        if elapsed_seconds > 0.0 {
            metrics.predictions_per_second = metrics.total_predictions as f64 / elapsed_seconds;
        }
        
        metrics.last_updated = chrono::Utc::now();
        
        // Update model-specific metrics
        let model_metric = model_metrics.entry(model_id.to_string())
            .or_insert_with(|| ModelPerformanceMetrics {
                model_id: model_id.to_string(),
                predictions: 0,
                avg_execution_time_us: 0.0,
                min_execution_time_us: u64::MAX,
                max_execution_time_us: 0,
                success_rate: 0.0,
                memory_usage_bytes: 0,
                accuracy: None,
                sharpe_ratio: None,
                last_prediction: chrono::Utc::now(),
            });
        
        model_metric.predictions += 1;
        model_metric.avg_execution_time_us = 
            (model_metric.avg_execution_time_us * (model_metric.predictions - 1) as f64 
             + execution_time_us as f64) / model_metric.predictions as f64;
        model_metric.min_execution_time_us = 
            model_metric.min_execution_time_us.min(execution_time_us);
        model_metric.max_execution_time_us = 
            model_metric.max_execution_time_us.max(execution_time_us);
        model_metric.success_rate = 1.0; // Assuming success if we're recording
        model_metric.last_prediction = chrono::Utc::now();
        
        // Check if prediction meets latency target
        if execution_time_us > self.config.target_latency_us {
            warn!(
                "Prediction latency {}μs exceeds target {}μs for model {}",
                execution_time_us, self.config.target_latency_us, model_id
            );
        }
        
        debug!(
            "Recorded prediction: model={}, latency={}μs, total_predictions={}",
            model_id, execution_time_us, metrics.total_predictions
        );
    }
    
    /// Record a failed prediction
    pub async fn record_failure(&self, model_id: &str) {
        let mut metrics = self.metrics.write().await;
        let mut model_metrics = self.model_metrics.write().await;
        
        metrics.total_predictions += 1;
        metrics.success_rate = 
            metrics.successful_predictions as f64 / metrics.total_predictions as f64;
        metrics.last_updated = chrono::Utc::now();
        
        // Update model failure rate
        if let Some(model_metric) = model_metrics.get_mut(model_id) {
            model_metric.predictions += 1;
            model_metric.success_rate = 
                (model_metric.success_rate * (model_metric.predictions - 1) as f64) 
                / model_metric.predictions as f64;
        }
        
        warn!("Recorded prediction failure for model: {}", model_id);
    }
    
    /// Get current system performance metrics
    pub async fn get_metrics(&self) -> PerformanceMetrics {
        self.metrics.read().await.clone()
    }
    
    /// Get performance metrics for a specific model
    pub async fn get_model_metrics(&self, model_id: &str) -> Option<ModelPerformanceMetrics> {
        self.model_metrics.read().await.get(model_id).cloned()
    }
    
    /// Get all model performance metrics
    pub async fn get_all_model_metrics(&self) -> HashMap<String, ModelPerformanceMetrics> {
        self.model_metrics.read().await.clone()
    }
    
    /// Generate performance report
    pub async fn generate_report(&self) -> Result<PerformanceReport> {
        let system_metrics = self.get_metrics().await;
        let model_metrics = self.get_all_model_metrics().await;
        
        let report = PerformanceReport {
            timestamp: chrono::Utc::now(),
            uptime_seconds: self.start_time.elapsed().as_secs(),
            system_metrics,
            model_metrics,
            recommendations: self.generate_recommendations().await,
        };
        
        Ok(report)
    }
    
    /// Generate performance optimization recommendations
    async fn generate_recommendations(&self) -> Vec<String> {
        let mut recommendations = Vec::new();
        let metrics = self.get_metrics().await;
        
        // Check latency performance
        if metrics.avg_execution_time_us > self.config.target_latency_us as f64 {
            recommendations.push(format!(
                "Average latency {:.1}μs exceeds target {}μs. Consider enabling GPU acceleration or optimizing models.",
                metrics.avg_execution_time_us, self.config.target_latency_us
            ));
        }
        
        // Check success rate
        if metrics.success_rate < 0.95 {
            recommendations.push(format!(
                "Success rate {:.2}% is below 95%. Review error handling and model stability.",
                metrics.success_rate * 100.0
            ));
        }
        
        // Check memory usage
        if metrics.memory_usage_bytes > self.config.memory_pool_size {
            recommendations.push(format!(
                "Memory usage {}MB exceeds pool size {}MB. Consider increasing pool size or optimizing memory usage.",
                metrics.memory_usage_bytes / (1024 * 1024),
                self.config.memory_pool_size / (1024 * 1024)
            ));
        }
        
        // Check predictions per second
        if metrics.predictions_per_second < 100.0 {
            recommendations.push(
                "Prediction throughput is low. Consider enabling parallel processing or SIMD optimizations.".to_string()
            );
        }
        
        recommendations
    }
    
    /// Background monitoring loop
    async fn monitoring_loop(&self) {
        let mut interval = tokio::time::interval(Duration::from_secs(30));
        
        loop {
            interval.tick().await;
            
            // Update system resource metrics
            if let Err(e) = self.update_system_metrics().await {
                warn!("Failed to update system metrics: {}", e);
            }
            
            // Log performance summary
            let metrics = self.get_metrics().await;
            info!(
                "Performance summary: predictions={}, avg_latency={:.1}μs, success_rate={:.2}%, pps={:.1}",
                metrics.total_predictions,
                metrics.avg_execution_time_us,
                metrics.success_rate * 100.0,
                metrics.predictions_per_second
            );
        }
    }
    
    /// Update system resource metrics
    async fn update_system_metrics(&self) -> Result<()> {
        let mut metrics = self.metrics.write().await;
        
        // Simulate system resource monitoring
        // In a real implementation, this would query actual system resources
        metrics.memory_usage_bytes = self.estimate_memory_usage().await;
        metrics.cpu_utilization = self.estimate_cpu_usage().await;
        
        #[cfg(feature = "gpu")]
        {
            metrics.gpu_utilization = self.estimate_gpu_usage().await;
        }
        
        metrics.last_updated = chrono::Utc::now();
        
        Ok(())
    }
    
    /// Estimate current memory usage
    async fn estimate_memory_usage(&self) -> usize {
        // Simplified memory estimation
        let model_count = self.model_metrics.read().await.len();
        let base_memory = 100 * 1024 * 1024; // 100MB base
        let per_model_memory = 50 * 1024 * 1024; // 50MB per model
        
        base_memory + (model_count * per_model_memory)
    }
    
    /// Estimate current CPU usage
    async fn estimate_cpu_usage(&self) -> f64 {
        // Simplified CPU estimation based on recent activity
        let metrics = self.metrics.read().await;
        let recent_activity = metrics.predictions_per_second / 100.0; // Scale factor
        
        (recent_activity * 100.0).min(100.0).max(0.0)
    }
    
    /// Estimate current GPU usage
    #[cfg(feature = "gpu")]
    async fn estimate_gpu_usage(&self) -> f64 {
        // Simplified GPU estimation
        if self.config.gpu_device_id.is_some() {
            let metrics = self.metrics.read().await;
            (metrics.predictions_per_second / 50.0 * 100.0).min(100.0).max(0.0)
        } else {
            0.0
        }
    }
}

/// Comprehensive performance report
#[derive(Debug, Serialize, Deserialize)]
pub struct PerformanceReport {
    /// Report generation timestamp
    pub timestamp: chrono::DateTime<chrono::Utc>,
    /// System uptime in seconds
    pub uptime_seconds: u64,
    /// System-wide metrics
    pub system_metrics: PerformanceMetrics,
    /// Per-model metrics
    pub model_metrics: HashMap<String, ModelPerformanceMetrics>,
    /// Performance recommendations
    pub recommendations: Vec<String>,
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_performance_monitor_creation() {
        let config = NeuralConfig::default();
        let monitor = PerformanceMonitor::new(&config).unwrap();
        
        let metrics = monitor.get_metrics().await;
        assert_eq!(metrics.total_predictions, 0);
        assert_eq!(metrics.success_rate, 0.0);
    }
    
    #[tokio::test]
    async fn test_record_prediction() {
        let config = NeuralConfig::default();
        let monitor = PerformanceMonitor::new(&config).unwrap();
        
        monitor.record_prediction("test_model", 50).await;
        
        let metrics = monitor.get_metrics().await;
        assert_eq!(metrics.total_predictions, 1);
        assert_eq!(metrics.successful_predictions, 1);
        assert_eq!(metrics.avg_execution_time_us, 50.0);
        assert_eq!(metrics.success_rate, 1.0);
        
        let model_metrics = monitor.get_model_metrics("test_model").await.unwrap();
        assert_eq!(model_metrics.predictions, 1);
        assert_eq!(model_metrics.avg_execution_time_us, 50.0);
    }
    
    #[tokio::test]
    async fn test_record_failure() {
        let config = NeuralConfig::default();
        let monitor = PerformanceMonitor::new(&config).unwrap();
        
        monitor.record_prediction("test_model", 50).await;
        monitor.record_failure("test_model").await;
        
        let metrics = monitor.get_metrics().await;
        assert_eq!(metrics.total_predictions, 2);
        assert_eq!(metrics.successful_predictions, 1);
        assert_eq!(metrics.success_rate, 0.5);
    }
    
    #[tokio::test]
    async fn test_performance_report() {
        let config = NeuralConfig::default();
        let monitor = PerformanceMonitor::new(&config).unwrap();
        
        monitor.record_prediction("model1", 75).await;
        monitor.record_prediction("model2", 125).await;
        
        let report = monitor.generate_report().await.unwrap();
        
        assert_eq!(report.system_metrics.total_predictions, 2);
        assert_eq!(report.model_metrics.len(), 2);
        assert!(report.uptime_seconds >= 0);
    }
}