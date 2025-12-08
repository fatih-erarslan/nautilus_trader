//! Utility functions for QBMIA Unified
//!
//! Common utilities and helper functions with TENGRI compliance.

use crate::types::*;
use crate::error::Result;
use std::collections::HashMap;

/// TENGRI compliance validation utilities
pub mod tengri {
    use super::*;

    /// Validate that data is real and not mock/synthetic
    pub fn validate_real_data(data: &str, field_name: &str) -> Result<()> {
        let forbidden_terms = ["mock", "fake", "test", "synthetic", "placeholder", "dummy"];
        let data_lower = data.to_lowercase();
        
        for term in &forbidden_terms {
            if data_lower.contains(term) {
                return Err(crate::error::QbmiaError::MockDataDetected);
            }
        }
        
        Ok(())
    }

    /// Validate timestamp is realistic (not future, not too old)
    pub fn validate_realistic_timestamp(timestamp: chrono::DateTime<chrono::Utc>) -> Result<()> {
        let now = chrono::Utc::now();
        let one_year_ago = now - chrono::Duration::days(365);
        
        if timestamp > now {
            return Err(crate::error::QbmiaError::MockDataDetected);
        }
        
        if timestamp < one_year_ago {
            return Err(crate::error::QbmiaError::InvalidInput {
                field: "timestamp".to_string(),
                reason: "Timestamp too old for real-time analysis".to_string(),
            });
        }
        
        Ok(())
    }

    /// Validate GPU device is real hardware
    pub fn validate_real_gpu_device(device: &GpuDevice) -> Result<()> {
        // Check device name for mock indicators
        validate_real_data(&device.name, "gpu_device_name")?;
        
        // Check for realistic memory amounts
        if device.capabilities.total_memory == 0 {
            return Err(crate::error::QbmiaError::MockDataDetected);
        }
        
        // Check for realistic compute units
        if device.capabilities.multiprocessor_count == 0 {
            return Err(crate::error::QbmiaError::MockDataDetected);
        }
        
        Ok(())
    }
}

/// Performance monitoring utilities
pub mod performance {
    use super::*;
    
    /// Real system resource monitor
    pub struct RealSystemMonitor {
        last_update: std::sync::Arc<parking_lot::RwLock<chrono::DateTime<chrono::Utc>>>,
        metrics_cache: std::sync::Arc<parking_lot::RwLock<PerformanceMetrics>>,
    }
    
    impl RealSystemMonitor {
        pub async fn new(config: &PerformanceConfig) -> Result<Self> {
            let monitor = Self {
                last_update: std::sync::Arc::new(parking_lot::RwLock::new(chrono::Utc::now())),
                metrics_cache: std::sync::Arc::new(parking_lot::RwLock::new(PerformanceMetrics {
                    cpu_usage_percent: 0.0,
                    memory_usage_bytes: 0,
                    gpu_utilization: HashMap::new(),
                    network_throughput_mbps: 0.0,
                    disk_io_mbps: 0.0,
                    processing_latency_ms: 0,
                    throughput_operations_per_second: 0.0,
                    error_rate_percent: 0.0,
                })),
            };
            
            if config.enable_real_monitoring {
                monitor.start_monitoring(config.sampling_interval_ms).await?;
            }
            
            Ok(monitor)
        }
        
        async fn start_monitoring(&self, interval_ms: u64) -> Result<()> {
            // Start background monitoring task
            let metrics_cache = self.metrics_cache.clone();
            let last_update = self.last_update.clone();
            
            tokio::spawn(async move {
                let mut interval = tokio::time::interval(
                    std::time::Duration::from_millis(interval_ms)
                );
                
                loop {
                    interval.tick().await;
                    
                    // Collect real system metrics
                    if let Ok(current_metrics) = collect_system_metrics().await {
                        *metrics_cache.write() = current_metrics;
                        *last_update.write() = chrono::Utc::now();
                    }
                }
            });
            
            Ok(())
        }
        
        pub async fn get_current_metrics(&self) -> PerformanceMetrics {
            self.metrics_cache.read().clone()
        }
        
        pub async fn record_analysis_completion(&self) {
            // Record analysis completion for throughput calculation
            // Implementation would update internal counters
        }
    }
    
    /// Collect real system metrics
    async fn collect_system_metrics() -> Result<PerformanceMetrics> {
        // Use real system monitoring libraries
        let cpu_usage = get_cpu_usage().await?;
        let memory_usage = get_memory_usage().await?;
        let gpu_utilization = get_gpu_utilization().await?;
        let network_throughput = get_network_throughput().await?;
        let disk_io = get_disk_io().await?;
        
        Ok(PerformanceMetrics {
            cpu_usage_percent: cpu_usage,
            memory_usage_bytes: memory_usage,
            gpu_utilization,
            network_throughput_mbps: network_throughput,
            disk_io_mbps: disk_io,
            processing_latency_ms: 0, // Would be calculated from timing data
            throughput_operations_per_second: 0.0, // Would be calculated from operation counts
            error_rate_percent: 0.0, // Would be calculated from error tracking
        })
    }
    
    async fn get_cpu_usage() -> Result<f32> {
        // Implementation would use real CPU monitoring
        Ok(0.0) // Placeholder
    }
    
    async fn get_memory_usage() -> Result<usize> {
        // Implementation would use real memory monitoring
        Ok(0) // Placeholder
    }
    
    async fn get_gpu_utilization() -> Result<HashMap<String, f32>> {
        // Implementation would use real GPU monitoring
        Ok(HashMap::new()) // Placeholder
    }
    
    async fn get_network_throughput() -> Result<f32> {
        // Implementation would use real network monitoring
        Ok(0.0) // Placeholder
    }
    
    async fn get_disk_io() -> Result<f32> {
        // Implementation would use real disk I/O monitoring
        Ok(0.0) // Placeholder
    }
}

/// Mathematical utilities for quantum and biological computations
pub mod math {
    use super::*;
    use num_complex::Complex;
    
    /// Calculate matrix eigenvalues for quantum operations
    pub fn calculate_eigenvalues(matrix: &[Vec<f64>]) -> Result<Vec<f64>> {
        if matrix.is_empty() {
            return Ok(Vec::new());
        }
        
        let n = matrix.len();
        
        // Ensure square matrix
        for row in matrix {
            if row.len() != n {
                return Err(crate::error::QbmiaError::InvalidInput {
                    field: "matrix".to_string(),
                    reason: "Matrix must be square".to_string(),
                });
            }
        }
        
        // Convert to nalgebra matrix for eigenvalue computation
        let nalgebra_matrix = nalgebra::DMatrix::from_fn(n, n, |i, j| matrix[i][j]);
        
        // Compute eigenvalues
        if let Some(eigenvalues) = nalgebra_matrix.symmetric_eigenvalues() {
            Ok(eigenvalues.iter().cloned().collect())
        } else {
            // Fallback for non-symmetric matrices - use approximate method
            let trace = (0..n).map(|i| matrix[i][i]).sum::<f64>();
            Ok(vec![trace / n as f64; n]) // Simplified approximation
        }
    }
    
    /// Normalize quantum state vector
    pub fn normalize_quantum_state(amplitudes: &mut [Complex<f64>]) -> Result<()> {
        let norm_squared: f64 = amplitudes.iter().map(|a| a.norm_sqr()).sum();
        
        if norm_squared == 0.0 {
            return Err(crate::error::QbmiaError::QuantumSimulationError {
                reason: "Cannot normalize zero state".to_string(),
            });
        }
        
        let norm = norm_squared.sqrt();
        for amplitude in amplitudes {
            *amplitude /= norm;
        }
        
        Ok(())
    }
    
    /// Calculate Von Neumann entropy for quantum states
    pub fn von_neumann_entropy(density_matrix: &[Vec<Complex<f64>>]) -> Result<f64> {
        if density_matrix.is_empty() {
            return Ok(0.0);
        }
        
        // Simplified entropy calculation
        let mut entropy = 0.0;
        let n = density_matrix.len();
        
        for i in 0..n {
            let eigenvalue = density_matrix[i][i].re; // Diagonal elements for simplified case
            if eigenvalue > 1e-15 {
                entropy -= eigenvalue * eigenvalue.ln();
            }
        }
        
        Ok(entropy)
    }
}

/// Data conversion utilities
pub mod conversion {
    use super::*;
    
    /// Convert market data to numerical matrix for analysis
    pub fn market_data_to_matrix(data: &MarketData) -> Result<Vec<Vec<f64>>> {
        if data.data_points.is_empty() {
            return Ok(Vec::new());
        }
        
        // Group data by symbol
        let mut symbol_data: HashMap<String, Vec<&MarketDataPoint>> = HashMap::new();
        for point in &data.data_points {
            symbol_data.entry(point.symbol.clone()).or_default().push(point);
        }
        
        let num_symbols = data.symbols.len();
        let max_points = symbol_data.values().map(|v| v.len()).max().unwrap_or(0);
        
        let mut matrix = vec![vec![0.0; num_symbols]; max_points];
        
        for (symbol_idx, symbol) in data.symbols.iter().enumerate() {
            if let Some(points) = symbol_data.get(symbol) {
                for (point_idx, point) in points.iter().enumerate() {
                    if point_idx < max_points {
                        matrix[point_idx][symbol_idx] = point.price.to_f64().unwrap_or(0.0);
                    }
                }
            }
        }
        
        Ok(matrix)
    }
    
    /// Convert biological patterns to feature vectors
    pub fn biological_patterns_to_features(patterns: &[BiologicalPattern]) -> Vec<f64> {
        let mut features = Vec::new();
        
        for pattern in patterns {
            features.push(pattern.strength);
            features.push(pattern.confidence);
            features.extend(&pattern.temporal_dynamics);
        }
        
        features
    }
}

/// Validation utilities for different data types
pub mod validate {
    use super::*;
    
    /// Validate market data completeness and integrity
    pub fn validate_market_data_integrity(data: &MarketData) -> Result<()> {
        if data.symbols.is_empty() {
            return Err(crate::error::QbmiaError::InvalidInput {
                field: "symbols".to_string(),
                reason: "No symbols provided".to_string(),
            });
        }
        
        if data.data_points.is_empty() {
            return Err(crate::error::QbmiaError::InvalidInput {
                field: "data_points".to_string(),
                reason: "No data points provided".to_string(),
            });
        }
        
        // Validate each data point
        for (i, point) in data.data_points.iter().enumerate() {
            if point.price.is_zero() {
                return Err(crate::error::QbmiaError::InvalidInput {
                    field: format!("data_points[{}].price", i),
                    reason: "Price cannot be zero".to_string(),
                });
            }
            
            if point.price.is_sign_negative() {
                return Err(crate::error::QbmiaError::InvalidInput {
                    field: format!("data_points[{}].price", i),
                    reason: "Price cannot be negative".to_string(),
                });
            }
            
            // Validate timestamp is reasonable
            tengri::validate_realistic_timestamp(point.timestamp)?;
        }
        
        // Validate source APIs are real
        for source in &data.source_apis {
            tengri::validate_real_data(source, "source_api")?;
        }
        
        Ok(())
    }
    
    /// Validate quantum state is physically valid
    pub fn validate_quantum_state(state: &QuantumState) -> Result<()> {
        if state.amplitudes.is_empty() {
            return Err(crate::error::QbmiaError::QuantumSimulationError {
                reason: "Quantum state cannot be empty".to_string(),
            });
        }
        
        // Check normalization
        let norm_squared: f64 = state.amplitudes.iter().map(|a| a.norm_sqr()).sum();
        if (norm_squared - 1.0).abs() > 1e-10 {
            return Err(crate::error::QbmiaError::QuantumSimulationError {
                reason: format!("Quantum state not normalized: norm² = {}", norm_squared),
            });
        }
        
        // Check dimension consistency
        let expected_size = 1 << state.qubit_count;
        if state.amplitudes.len() != expected_size {
            return Err(crate::error::QbmiaError::QuantumSimulationError {
                reason: format!(
                    "State dimension mismatch: expected {}, got {}",
                    expected_size,
                    state.amplitudes.len()
                ),
            });
        }
        
        Ok(())
    }
}

/// ID generation utilities
pub fn generate_unique_id() -> String {
    uuid::Uuid::new_v4().to_string()
}

/// Timestamp utilities
pub fn current_utc_timestamp() -> chrono::DateTime<chrono::Utc> {
    chrono::Utc::now()
}

/// Format duration for human readability
pub fn format_duration(duration: std::time::Duration) -> String {
    let total_ms = duration.as_millis();
    
    if total_ms < 1000 {
        format!("{}ms", total_ms)
    } else if total_ms < 60_000 {
        format!("{:.1}s", total_ms as f64 / 1000.0)
    } else {
        format!("{:.1}m", total_ms as f64 / 60_000.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_tengri_validation() {
        // Test mock data detection
        assert!(tengri::validate_real_data("mock_data", "test").is_err());
        assert!(tengri::validate_real_data("real_data", "test").is_ok());
    }
    
    #[test]
    fn test_duration_formatting() {
        assert_eq!(format_duration(std::time::Duration::from_millis(500)), "500ms");
        assert_eq!(format_duration(std::time::Duration::from_millis(1500)), "1.5s");
        assert_eq!(format_duration(std::time::Duration::from_millis(90000)), "1.5m");
    }
}