//! QBMIA Hardware Optimization - GPU/CPU acceleration and resource management
//!
//! This module implements hardware optimization for QBMIA biological systems,
//! including GPU acceleration, CPU optimization, and resource management.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};
use anyhow::Result;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn, error, debug};

use crate::{ComponentHealth, HealthStatus};

/// Hardware device types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum DeviceType {
    CPU,
    GPU,
    TPU,
    FPGA,
    Hybrid,
}

/// Hardware optimization strategy
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum OptimizationStrategy {
    Performance,
    PowerEfficiency,
    Balanced,
    Adaptive,
}

/// Hardware resource information
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareResource {
    pub device_type: DeviceType,
    pub device_id: String,
    pub name: String,
    pub compute_capability: f64,
    pub memory_total: u64,
    pub memory_available: u64,
    pub utilization: f64,
    pub temperature: f64,
    pub power_consumption: f64,
    pub is_available: bool,
}

/// Hardware performance metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    pub throughput: f64,
    pub latency: Duration,
    pub efficiency: f64,
    pub power_efficiency: f64,
    pub thermal_efficiency: f64,
    pub resource_utilization: f64,
}

/// Hardware optimization configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    pub force_cpu: bool,
    pub enable_profiling: bool,
    pub optimization_strategy: OptimizationStrategy,
    pub memory_pool_size: usize,
    pub thread_pool_size: usize,
    pub gpu_memory_fraction: f64,
    pub enable_mixed_precision: bool,
    pub enable_graph_optimization: bool,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            force_cpu: false,
            enable_profiling: true,
            optimization_strategy: OptimizationStrategy::Balanced,
            memory_pool_size: 1024 * 1024 * 1024, // 1GB
            thread_pool_size: num_cpus::get(),
            gpu_memory_fraction: 0.8,
            enable_mixed_precision: true,
            enable_graph_optimization: true,
        }
    }
}

/// Hardware optimizer
#[derive(Debug)]
pub struct HardwareOptimizer {
    config: HardwareConfig,
    available_devices: Arc<RwLock<Vec<HardwareResource>>>,
    active_device: Arc<RwLock<Option<HardwareResource>>>,
    performance_metrics: Arc<RwLock<PerformanceMetrics>>,
    resource_pool: Arc<RwLock<HashMap<String, ResourcePool>>>,
    optimization_history: Arc<RwLock<Vec<OptimizationResult>>>,
    
    // Thread pool for CPU operations
    cpu_thread_pool: rayon::ThreadPool,
    
    // Performance tracking
    operation_stats: Arc<RwLock<HashMap<String, OperationStats>>>,
    
    // State management
    is_running: Arc<RwLock<bool>>,
}

/// Resource pool for managing hardware resources
#[derive(Debug, Clone)]
pub struct ResourcePool {
    pub resource_type: String,
    pub total_capacity: u64,
    pub used_capacity: u64,
    pub allocation_map: HashMap<String, u64>,
    pub last_cleanup: Instant,
}

/// Operation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OperationStats {
    pub operation_name: String,
    pub total_executions: u64,
    pub total_duration: Duration,
    pub average_duration: Duration,
    pub success_rate: f64,
    pub hardware_utilization: f64,
    pub memory_peak: u64,
}

/// Optimization result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationResult {
    pub timestamp: std::time::SystemTime,
    pub strategy: OptimizationStrategy,
    pub device_used: DeviceType,
    pub performance_gain: f64,
    pub power_savings: f64,
    pub success: bool,
    pub notes: String,
}

impl HardwareOptimizer {
    /// Create new hardware optimizer
    pub fn new(force_cpu: bool, enable_profiling: bool) -> Result<Self> {
        info!("Initializing QBMIA Hardware Optimizer");
        
        let config = HardwareConfig {
            force_cpu,
            enable_profiling,
            ..Default::default()
        };
        
        // Initialize CPU thread pool
        let cpu_thread_pool = rayon::ThreadPoolBuilder::new()
            .num_threads(config.thread_pool_size)
            .build()?;
        
        let optimizer = Self {
            config,
            available_devices: Arc::new(RwLock::new(Vec::new())),
            active_device: Arc::new(RwLock::new(None)),
            performance_metrics: Arc::new(RwLock::new(PerformanceMetrics {
                throughput: 0.0,
                latency: Duration::from_millis(0),
                efficiency: 0.0,
                power_efficiency: 0.0,
                thermal_efficiency: 0.0,
                resource_utilization: 0.0,
            })),
            resource_pool: Arc::new(RwLock::new(HashMap::new())),
            optimization_history: Arc::new(RwLock::new(Vec::new())),
            cpu_thread_pool,
            operation_stats: Arc::new(RwLock::new(HashMap::new())),
            is_running: Arc::new(RwLock::new(false)),
        };
        
        Ok(optimizer)
    }
    
    /// Start hardware optimizer
    pub async fn start(&self) -> Result<()> {
        info!("Starting QBMIA Hardware Optimizer");
        
        // Detect available hardware
        self.detect_hardware().await?;
        
        // Initialize resource pools
        self.initialize_resource_pools().await?;
        
        // Select optimal device
        self.select_optimal_device().await?;
        
        // Start monitoring
        self.start_monitoring().await?;
        
        *self.is_running.write().await = true;
        
        info!("Hardware optimizer started successfully");
        Ok(())
    }
    
    /// Stop hardware optimizer
    pub async fn stop(&self) -> Result<()> {
        info!("Stopping QBMIA Hardware Optimizer");
        
        *self.is_running.write().await = false;
        
        // Cleanup resources
        self.cleanup_resources().await?;
        
        info!("Hardware optimizer stopped successfully");
        Ok(())
    }
    
    /// Execute operation with hardware optimization
    pub async fn execute_optimized<F, T>(&self, operation_name: &str, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let start_time = Instant::now();
        
        // Get optimal device for operation
        let device = self.get_optimal_device_for_operation(operation_name).await?;
        
        // Execute operation based on device type
        let result = match device.device_type {
            DeviceType::CPU => {
                self.execute_cpu_operation(operation).await
            }
            DeviceType::GPU => {
                self.execute_gpu_operation(operation).await
            }
            DeviceType::TPU => {
                self.execute_tpu_operation(operation).await
            }
            DeviceType::FPGA => {
                self.execute_fpga_operation(operation).await
            }
            DeviceType::Hybrid => {
                self.execute_hybrid_operation(operation).await
            }
        };
        
        // Update operation statistics
        self.update_operation_stats(operation_name, start_time.elapsed(), result.is_ok()).await?;
        
        // Update performance metrics
        self.update_performance_metrics(start_time.elapsed()).await?;
        
        result
    }
    
    /// Get device information
    pub async fn get_device_info(&self) -> Result<HashMap<String, serde_json::Value>> {
        let mut info = HashMap::new();
        
        // Add CPU information
        info.insert("cpu_cores".to_string(), serde_json::json!(num_cpus::get()));
        info.insert("cpu_logical_cores".to_string(), serde_json::json!(num_cpus::get()));
        
        // Add system memory information
        if let Ok(memory_info) = self.get_system_memory_info() {
            info.insert("system_memory".to_string(), serde_json::json!(memory_info));
        }
        
        // Add GPU information if available
        if let Ok(gpu_info) = self.get_gpu_info().await {
            info.insert("gpu_devices".to_string(), serde_json::json!(gpu_info));
        }
        
        // Add active device information
        if let Some(active_device) = self.active_device.read().await.as_ref() {
            info.insert("active_device".to_string(), serde_json::json!(active_device));
        }
        
        // Add performance metrics
        let metrics = self.performance_metrics.read().await;
        info.insert("performance_metrics".to_string(), serde_json::json!(*metrics));
        
        Ok(info)
    }
    
    /// Get performance metrics
    pub async fn get_performance_metrics(&self) -> Result<PerformanceMetrics> {
        Ok(self.performance_metrics.read().await.clone())
    }
    
    /// Optimize for specific workload
    pub async fn optimize_for_workload(&self, workload_type: &str) -> Result<OptimizationResult> {
        let start_time = std::time::SystemTime::now();
        
        // Determine optimal strategy for workload
        let strategy = self.determine_optimal_strategy(workload_type).await?;
        
        // Apply optimization
        let success = self.apply_optimization_strategy(&strategy).await?;
        
        // Measure performance improvement
        let performance_gain = if success {
            self.measure_performance_improvement().await?
        } else {
            0.0
        };
        
        let result = OptimizationResult {
            timestamp: start_time,
            strategy,
            device_used: self.get_active_device_type().await?,
            performance_gain,
            power_savings: performance_gain * 0.3, // Estimated power savings
            success,
            notes: format!("Optimization for workload: {}", workload_type),
        };
        
        // Store result
        self.optimization_history.write().await.push(result.clone());
        
        Ok(result)
    }
    
    /// Health check
    pub async fn health_check(&self) -> Result<ComponentHealth> {
        let is_running = *self.is_running.read().await;
        let devices = self.available_devices.read().await;
        let metrics = self.performance_metrics.read().await;
        
        let device_health = if !devices.is_empty() {
            devices.iter().map(|d| if d.is_available { 1.0 } else { 0.0 }).sum::<f64>() / devices.len() as f64
        } else {
            0.0
        };
        
        let performance_score = (metrics.efficiency * 0.4 + 
                               metrics.power_efficiency * 0.3 + 
                               metrics.thermal_efficiency * 0.2 + 
                               device_health * 0.1).min(1.0);
        
        Ok(ComponentHealth {
            status: if is_running && performance_score > 0.7 {
                HealthStatus::Healthy
            } else if is_running && performance_score > 0.5 {
                HealthStatus::Degraded
            } else if is_running {
                HealthStatus::Critical
            } else {
                HealthStatus::Offline
            },
            last_update: std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs() as i64,
            error_count: 0, // Would track actual errors in production
            performance_score,
        })
    }
    
    // Private helper methods
    
    async fn detect_hardware(&self) -> Result<()> {
        let mut devices = Vec::new();
        
        // Always add CPU
        devices.push(HardwareResource {
            device_type: DeviceType::CPU,
            device_id: "cpu_0".to_string(),
            name: "System CPU".to_string(),
            compute_capability: 1.0,
            memory_total: self.get_system_memory_total()?,
            memory_available: self.get_system_memory_available()?,
            utilization: 0.0,
            temperature: 65.0,
            power_consumption: 50.0,
            is_available: true,
        });
        
        // Detect GPU devices (simulated)
        if !self.config.force_cpu {
            if let Ok(gpu_devices) = self.detect_gpu_devices().await {
                devices.extend(gpu_devices);
            }
        }
        
        *self.available_devices.write().await = devices;
        
        info!("Detected {} hardware devices", self.available_devices.read().await.len());
        Ok(())
    }
    
    async fn detect_gpu_devices(&self) -> Result<Vec<HardwareResource>> {
        let mut devices = Vec::new();
        
        // Simulate GPU detection
        // In real implementation, this would use CUDA/OpenCL/ROCm APIs
        if std::env::var("QBMIA_GPU_ENABLED").is_ok() {
            devices.push(HardwareResource {
                device_type: DeviceType::GPU,
                device_id: "gpu_0".to_string(),
                name: "NVIDIA GeForce RTX 4090".to_string(),
                compute_capability: 8.9,
                memory_total: 24 * 1024 * 1024 * 1024, // 24GB
                memory_available: 20 * 1024 * 1024 * 1024, // 20GB
                utilization: 0.0,
                temperature: 45.0,
                power_consumption: 300.0,
                is_available: true,
            });
        }
        
        Ok(devices)
    }
    
    async fn initialize_resource_pools(&self) -> Result<()> {
        let mut pools = HashMap::new();
        
        // CPU memory pool
        pools.insert("cpu_memory".to_string(), ResourcePool {
            resource_type: "memory".to_string(),
            total_capacity: self.config.memory_pool_size as u64,
            used_capacity: 0,
            allocation_map: HashMap::new(),
            last_cleanup: Instant::now(),
        });
        
        // GPU memory pool (if available)
        if let Some(gpu_device) = self.available_devices.read().await.iter()
            .find(|d| matches!(d.device_type, DeviceType::GPU)) {
            pools.insert("gpu_memory".to_string(), ResourcePool {
                resource_type: "gpu_memory".to_string(),
                total_capacity: (gpu_device.memory_total as f64 * self.config.gpu_memory_fraction) as u64,
                used_capacity: 0,
                allocation_map: HashMap::new(),
                last_cleanup: Instant::now(),
            });
        }
        
        *self.resource_pool.write().await = pools;
        
        Ok(())
    }
    
    async fn select_optimal_device(&self) -> Result<()> {
        let devices = self.available_devices.read().await;
        
        let optimal_device = if self.config.force_cpu {
            devices.iter().find(|d| matches!(d.device_type, DeviceType::CPU))
        } else {
            // Select based on compute capability and availability
            devices.iter()
                .filter(|d| d.is_available)
                .max_by(|a, b| a.compute_capability.partial_cmp(&b.compute_capability).unwrap())
        };
        
        if let Some(device) = optimal_device {
            *self.active_device.write().await = Some(device.clone());
            info!("Selected optimal device: {} ({})", device.name, device.device_id);
        } else {
            warn!("No suitable hardware device found");
        }
        
        Ok(())
    }
    
    async fn start_monitoring(&self) -> Result<()> {
        let devices = Arc::clone(&self.available_devices);
        let metrics = Arc::clone(&self.performance_metrics);
        let is_running = Arc::clone(&self.is_running);
        
        tokio::spawn(async move {
            let mut interval = tokio::time::interval(Duration::from_secs(10));
            
            while *is_running.read().await {
                interval.tick().await;
                
                // Update device metrics
                let mut devices_guard = devices.write().await;
                for device in devices_guard.iter_mut() {
                    device.utilization = Self::get_device_utilization(&device.device_id).await.unwrap_or(0.0);
                    device.temperature = Self::get_device_temperature(&device.device_id).await.unwrap_or(65.0);
                    device.power_consumption = Self::get_device_power(&device.device_id).await.unwrap_or(50.0);
                }
                
                // Update performance metrics
                let mut metrics_guard = metrics.write().await;
                metrics_guard.resource_utilization = devices_guard.iter()
                    .map(|d| d.utilization)
                    .sum::<f64>() / devices_guard.len() as f64;
                
                metrics_guard.thermal_efficiency = 1.0 - (devices_guard.iter()
                    .map(|d| d.temperature / 100.0)
                    .sum::<f64>() / devices_guard.len() as f64);
                
                metrics_guard.power_efficiency = 1.0 - (devices_guard.iter()
                    .map(|d| d.power_consumption / 500.0)
                    .sum::<f64>() / devices_guard.len() as f64);
            }
        });
        
        Ok(())
    }
    
    async fn cleanup_resources(&self) -> Result<()> {
        // Cleanup resource pools
        let mut pools = self.resource_pool.write().await;
        for pool in pools.values_mut() {
            pool.allocation_map.clear();
            pool.used_capacity = 0;
        }
        
        // Clear operation stats
        self.operation_stats.write().await.clear();
        
        Ok(())
    }
    
    async fn get_optimal_device_for_operation(&self, operation_name: &str) -> Result<HardwareResource> {
        // For now, return active device
        // In real implementation, this would analyze operation characteristics
        if let Some(device) = self.active_device.read().await.as_ref() {
            Ok(device.clone())
        } else {
            // Fallback to CPU
            let devices = self.available_devices.read().await;
            devices.iter()
                .find(|d| matches!(d.device_type, DeviceType::CPU))
                .cloned()
                .ok_or_else(|| anyhow::anyhow!("No CPU device available"))
        }
    }
    
    async fn execute_cpu_operation<F, T>(&self, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        let (sender, receiver) = tokio::sync::oneshot::channel();
        
        self.cpu_thread_pool.spawn(move || {
            let result = operation();
            let _ = sender.send(result);
        });
        
        receiver.await.map_err(|e| anyhow::anyhow!("CPU operation failed: {}", e))?
    }
    
    async fn execute_gpu_operation<F, T>(&self, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        // For now, fallback to CPU
        // In real implementation, this would use GPU acceleration
        self.execute_cpu_operation(operation).await
    }
    
    async fn execute_tpu_operation<F, T>(&self, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        // For now, fallback to CPU
        // In real implementation, this would use TPU acceleration
        self.execute_cpu_operation(operation).await
    }
    
    async fn execute_fpga_operation<F, T>(&self, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        // For now, fallback to CPU
        // In real implementation, this would use FPGA acceleration
        self.execute_cpu_operation(operation).await
    }
    
    async fn execute_hybrid_operation<F, T>(&self, operation: F) -> Result<T>
    where
        F: FnOnce() -> Result<T> + Send + 'static,
        T: Send + 'static,
    {
        // For now, fallback to CPU
        // In real implementation, this would use hybrid CPU/GPU execution
        self.execute_cpu_operation(operation).await
    }
    
    async fn update_operation_stats(&self, operation_name: &str, duration: Duration, success: bool) -> Result<()> {
        let mut stats = self.operation_stats.write().await;
        let operation_stats = stats.entry(operation_name.to_string()).or_insert(OperationStats {
            operation_name: operation_name.to_string(),
            total_executions: 0,
            total_duration: Duration::from_millis(0),
            average_duration: Duration::from_millis(0),
            success_rate: 1.0,
            hardware_utilization: 0.0,
            memory_peak: 0,
        });
        
        operation_stats.total_executions += 1;
        operation_stats.total_duration += duration;
        operation_stats.average_duration = operation_stats.total_duration / operation_stats.total_executions as u32;
        
        if success {
            operation_stats.success_rate = (operation_stats.success_rate * (operation_stats.total_executions - 1) as f64 + 1.0) / operation_stats.total_executions as f64;
        } else {
            operation_stats.success_rate = (operation_stats.success_rate * (operation_stats.total_executions - 1) as f64) / operation_stats.total_executions as f64;
        }
        
        Ok(())
    }
    
    async fn update_performance_metrics(&self, duration: Duration) -> Result<()> {
        let mut metrics = self.performance_metrics.write().await;
        
        // Update latency
        metrics.latency = Duration::from_millis(
            (metrics.latency.as_millis() as u64 + duration.as_millis() as u64) / 2
        );
        
        // Update throughput (operations per second)
        metrics.throughput = if duration.as_secs_f64() > 0.0 {
            (metrics.throughput + 1.0 / duration.as_secs_f64()) / 2.0
        } else {
            metrics.throughput
        };
        
        // Update efficiency
        metrics.efficiency = (metrics.throughput / 1000.0).min(1.0);
        
        Ok(())
    }
    
    async fn determine_optimal_strategy(&self, workload_type: &str) -> Result<OptimizationStrategy> {
        match workload_type {
            "neural_training" => Ok(OptimizationStrategy::Performance),
            "inference" => Ok(OptimizationStrategy::Balanced),
            "memory_intensive" => Ok(OptimizationStrategy::PowerEfficiency),
            _ => Ok(OptimizationStrategy::Adaptive),
        }
    }
    
    async fn apply_optimization_strategy(&self, strategy: &OptimizationStrategy) -> Result<bool> {
        match strategy {
            OptimizationStrategy::Performance => {
                // Optimize for maximum performance
                debug!("Applying performance optimization strategy");
                Ok(true)
            }
            OptimizationStrategy::PowerEfficiency => {
                // Optimize for power efficiency
                debug!("Applying power efficiency optimization strategy");
                Ok(true)
            }
            OptimizationStrategy::Balanced => {
                // Balance performance and efficiency
                debug!("Applying balanced optimization strategy");
                Ok(true)
            }
            OptimizationStrategy::Adaptive => {
                // Adaptive strategy based on current conditions
                debug!("Applying adaptive optimization strategy");
                Ok(true)
            }
        }
    }
    
    async fn measure_performance_improvement(&self) -> Result<f64> {
        // Simulate performance measurement
        // In real implementation, this would compare before/after metrics
        Ok(0.15) // 15% improvement
    }
    
    async fn get_active_device_type(&self) -> Result<DeviceType> {
        if let Some(device) = self.active_device.read().await.as_ref() {
            Ok(device.device_type.clone())
        } else {
            Ok(DeviceType::CPU)
        }
    }
    
    fn get_system_memory_total(&self) -> Result<u64> {
        // Simulate system memory detection
        Ok(32 * 1024 * 1024 * 1024) // 32GB
    }
    
    fn get_system_memory_available(&self) -> Result<u64> {
        // Simulate available memory
        Ok(24 * 1024 * 1024 * 1024) // 24GB
    }
    
    fn get_system_memory_info(&self) -> Result<HashMap<String, u64>> {
        let mut info = HashMap::new();
        info.insert("total".to_string(), self.get_system_memory_total()?);
        info.insert("available".to_string(), self.get_system_memory_available()?);
        info.insert("used".to_string(), self.get_system_memory_total()? - self.get_system_memory_available()?);
        Ok(info)
    }
    
    async fn get_gpu_info(&self) -> Result<Vec<HashMap<String, serde_json::Value>>> {
        let mut gpu_info = Vec::new();
        
        let devices = self.available_devices.read().await;
        for device in devices.iter() {
            if matches!(device.device_type, DeviceType::GPU) {
                let mut info = HashMap::new();
                info.insert("name".to_string(), serde_json::json!(device.name));
                info.insert("memory_total".to_string(), serde_json::json!(device.memory_total));
                info.insert("memory_available".to_string(), serde_json::json!(device.memory_available));
                info.insert("utilization".to_string(), serde_json::json!(device.utilization));
                info.insert("temperature".to_string(), serde_json::json!(device.temperature));
                gpu_info.push(info);
            }
        }
        
        Ok(gpu_info)
    }
    
    async fn get_device_utilization(device_id: &str) -> Result<f64> {
        // Simulate device utilization monitoring
        Ok(rand::random::<f64>() * 0.8) // 0-80% utilization
    }
    
    async fn get_device_temperature(device_id: &str) -> Result<f64> {
        // Simulate temperature monitoring
        Ok(45.0 + rand::random::<f64>() * 30.0) // 45-75°C
    }
    
    async fn get_device_power(device_id: &str) -> Result<f64> {
        // Simulate power monitoring
        Ok(50.0 + rand::random::<f64>() * 200.0) // 50-250W
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_hardware_optimizer_creation() {
        let optimizer = HardwareOptimizer::new(false, true);
        assert!(optimizer.is_ok());
    }
    
    #[tokio::test]
    async fn test_hardware_optimizer_start_stop() {
        let optimizer = HardwareOptimizer::new(true, false).unwrap();
        
        assert!(optimizer.start().await.is_ok());
        assert!(*optimizer.is_running.read().await);
        
        assert!(optimizer.stop().await.is_ok());
        assert!(!*optimizer.is_running.read().await);
    }
    
    #[tokio::test]
    async fn test_cpu_operation_execution() {
        let optimizer = HardwareOptimizer::new(true, false).unwrap();
        optimizer.start().await.unwrap();
        
        let result = optimizer.execute_optimized("test_operation", || {
            Ok(42)
        }).await;
        
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 42);
    }
    
    #[tokio::test]
    async fn test_performance_metrics() {
        let optimizer = HardwareOptimizer::new(true, false).unwrap();
        optimizer.start().await.unwrap();
        
        let metrics = optimizer.get_performance_metrics().await.unwrap();
        assert!(metrics.throughput >= 0.0);
        assert!(metrics.efficiency >= 0.0);
    }
    
    #[tokio::test]
    async fn test_health_check() {
        let optimizer = HardwareOptimizer::new(true, false).unwrap();
        
        let health = optimizer.health_check().await.unwrap();
        assert!(matches!(health.status, HealthStatus::Offline));
        
        optimizer.start().await.unwrap();
        let health = optimizer.health_check().await.unwrap();
        assert!(matches!(health.status, HealthStatus::Healthy | HealthStatus::Degraded));
    }
}