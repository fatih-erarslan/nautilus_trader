//! Hardware Abstraction Layer for Quantum Computing
//!
//! This module provides hardware abstraction and management for quantum computing
//! operations, including CPU, GPU, and quantum hardware interfaces.

use crate::error::{QuantumError, QuantumResult};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use serde::{Deserialize, Serialize};
use tracing::{debug, info};
use chrono::{DateTime, Utc};


/// Hardware types supported by the system
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardwareType {
    CPU,
    GPU,
    QuantumProcessor,
    FPGA,
    TPU,
    CustomAccelerator,
}

/// Hardware vendor information
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardwareVendor {
    Intel,
    AMD,
    NVIDIA,
    IBM,
    Google,
    Rigetti,
    IonQ,
    Honeywell,
    Xanadu,
    PsiQuantum,
    Unknown,
}

/// Hardware status
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HardwareStatus {
    Available,
    Busy,
    Maintenance,
    Error,
    Offline,
    Initializing,
}

/// Hardware capabilities
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareCapabilities {
    /// Number of compute units available
    pub compute_units: u32,
    /// Total memory in gigabytes
    pub memory_gb: f64,
    /// Memory bandwidth in GB/s
    pub bandwidth_gbps: f64,
    /// Peak performance in GFLOPS
    pub peak_performance_gflops: f64,
    /// Power consumption in watts
    pub power_consumption_watts: f64,
    /// Whether 16-bit floating point is supported
    pub supports_fp16: bool,
    /// Whether 32-bit floating point is supported
    pub supports_fp32: bool,
    /// Whether 64-bit floating point is supported
    pub supports_fp64: bool,
    /// Whether 8-bit integer is supported
    pub supports_int8: bool,
    /// Whether parallel execution is supported
    pub supports_parallel_execution: bool,
    /// Maximum number of parallel tasks
    pub max_parallel_tasks: usize,
    /// List of specialized capabilities
    pub specializations: Vec<String>,
}

impl Default for HardwareCapabilities {
    fn default() -> Self {
        Self {
            compute_units: 1,
            memory_gb: 8.0,
            bandwidth_gbps: 100.0,
            peak_performance_gflops: 1000.0,
            power_consumption_watts: 100.0,
            supports_fp16: true,
            supports_fp32: true,
            supports_fp64: true,
            supports_int8: true,
            supports_parallel_execution: true,
            max_parallel_tasks: 8,
            specializations: Vec::new(),
        }
    }
}

/// Hardware configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareConfig {
    /// Unique device identifier
    pub device_id: String,
    /// Memory limit in gigabytes
    pub memory_limit_gb: Option<f64>,
    /// Power limit in watts
    pub power_limit_watts: Option<f64>,
    /// Thermal limit in celsius
    pub thermal_limit_celsius: Option<f64>,
    /// Whether power saving is enabled
    pub enable_power_saving: bool,
    /// Whether turbo boost is enabled
    pub enable_turbo_boost: bool,
    /// Optimization level to use
    pub optimization_level: OptimizationLevel,
    /// CPU affinity mask
    pub affinity_mask: Option<Vec<bool>>,
    /// Custom hardware parameters
    pub custom_parameters: HashMap<String, String>,
}

impl Default for HardwareConfig {
    fn default() -> Self {
        Self {
            device_id: "default".to_string(),
            memory_limit_gb: None,
            power_limit_watts: None,
            thermal_limit_celsius: None,
            enable_power_saving: false,
            enable_turbo_boost: true,
            optimization_level: OptimizationLevel::Balanced,
            affinity_mask: None,
            custom_parameters: HashMap::new(),
        }
    }
}

/// Optimization levels for hardware
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum OptimizationLevel {
    Power,
    Balanced,
    Performance,
    Maximum,
}

/// Hardware metrics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareMetrics {
    /// Hardware utilization percentage
    pub utilization_percentage: f64,
    /// Memory usage in gigabytes
    pub memory_usage_gb: f64,
    /// Memory utilization percentage
    pub memory_utilization_percentage: f64,
    /// Power consumption in watts
    pub power_consumption_watts: f64,
    /// Temperature in celsius
    pub temperature_celsius: f64,
    /// Clock speed in MHz
    pub clock_speed_mhz: f64,
    /// Throughput in operations per second
    pub throughput_ops_per_second: f64,
    /// Latency in milliseconds
    pub latency_ms: f64,
    /// Error rate (0.0 to 1.0)
    pub error_rate: f64,
    /// Uptime in hours
    pub uptime_hours: f64,
    /// Total operations performed
    pub total_operations: u64,
    /// Number of successful operations
    pub successful_operations: u64,
    /// Number of failed operations
    pub failed_operations: u64,
    /// Timestamp of last update
    pub last_updated: DateTime<Utc>,
}

impl Default for HardwareMetrics {
    fn default() -> Self {
        Self {
            utilization_percentage: 0.0,
            memory_usage_gb: 0.0,
            memory_utilization_percentage: 0.0,
            power_consumption_watts: 0.0,
            temperature_celsius: 25.0,
            clock_speed_mhz: 1000.0,
            throughput_ops_per_second: 0.0,
            latency_ms: 0.0,
            error_rate: 0.0,
            uptime_hours: 0.0,
            total_operations: 0,
            successful_operations: 0,
            failed_operations: 0,
            last_updated: Utc::now(),
        }
    }
}

/// Hardware device abstraction
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HardwareDevice {
    pub id: String,
    pub name: String,
    pub hardware_type: HardwareType,
    pub vendor: HardwareVendor,
    pub model: String,
    pub driver_version: String,
    pub firmware_version: String,
    pub capabilities: HardwareCapabilities,
    pub config: HardwareConfig,
    pub status: HardwareStatus,
    pub metrics: HardwareMetrics,
    pub created_at: DateTime<Utc>,
    pub last_updated: DateTime<Utc>,
    pub metadata: HashMap<String, String>,
}

impl HardwareDevice {
    /// Create a new hardware device
    pub fn new(
        name: String,
        hardware_type: HardwareType,
        vendor: HardwareVendor,
        model: String,
    ) -> Self {
        let id = format!("hw_{}", uuid::Uuid::new_v4());
        let now = Utc::now();
        
        Self {
            id,
            name,
            hardware_type,
            vendor,
            model,
            driver_version: "1.0.0".to_string(),
            firmware_version: "1.0.0".to_string(),
            capabilities: HardwareCapabilities::default(),
            config: HardwareConfig::default(),
            status: HardwareStatus::Initializing,
            metrics: HardwareMetrics::default(),
            created_at: now,
            last_updated: now,
            metadata: HashMap::new(),
        }
    }

    /// Initialize hardware device
    pub async fn initialize(&mut self) -> QuantumResult<()> {
        info!("Initializing hardware device: {}", self.name);
        
        self.status = HardwareStatus::Initializing;
        
        // Simulate initialization
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        // Detect hardware capabilities
        self.detect_capabilities().await?;
        
        // Apply configuration
        self.apply_config().await?;
        
        // Run self-test
        self.run_self_test().await?;
        
        self.status = HardwareStatus::Available;
        self.last_updated = Utc::now();
        
        info!("Hardware device {} initialized successfully", self.name);
        // metrics::counter!("hardware_devices_initialized_total", 1);
        
        Ok(())
    }

    /// Detect hardware capabilities
    async fn detect_capabilities(&mut self) -> QuantumResult<()> {
        debug!("Detecting capabilities for device: {}", self.name);
        
        match self.hardware_type {
            HardwareType::CPU => {
                self.capabilities.compute_units = num_cpus::get() as u32;
                self.capabilities.supports_parallel_execution = true;
                self.capabilities.max_parallel_tasks = num_cpus::get();
                
                // Detect SIMD support
                if std::arch::is_x86_feature_detected!("avx2") {
                    self.capabilities.specializations.push("AVX2".to_string());
                }
                if std::arch::is_x86_feature_detected!("avx512f") {
                    self.capabilities.specializations.push("AVX512".to_string());
                }
            }
            HardwareType::GPU => {
                self.capabilities.compute_units = 1024; // Simulated
                self.capabilities.memory_gb = 8.0;
                self.capabilities.bandwidth_gbps = 500.0;
                self.capabilities.peak_performance_gflops = 10000.0;
                self.capabilities.supports_fp16 = true;
                self.capabilities.specializations.push("CUDA".to_string());
            }
            HardwareType::QuantumProcessor => {
                self.capabilities.compute_units = 32; // Qubits
                self.capabilities.memory_gb = 0.001; // Minimal classical memory
                self.capabilities.specializations.push("QuantumGates".to_string());
                self.capabilities.specializations.push("QuantumMeasurement".to_string());
            }
            HardwareType::FPGA => {
                self.capabilities.compute_units = 256; // Logic elements
                self.capabilities.specializations.push("Reconfigurable".to_string());
            }
            HardwareType::TPU => {
                self.capabilities.compute_units = 128; // Tensor cores
                self.capabilities.peak_performance_gflops = 50000.0;
                self.capabilities.specializations.push("TensorFlow".to_string());
            }
            HardwareType::CustomAccelerator => {
                // Custom detection logic
                self.capabilities.specializations.push("Custom".to_string());
            }
        }
        
        debug!("Capabilities detected: {:?}", self.capabilities);
        Ok(())
    }

    /// Apply hardware configuration
    async fn apply_config(&mut self) -> QuantumResult<()> {
        debug!("Applying configuration for device: {}", self.name);
        
        // Apply memory limits
        if let Some(memory_limit) = self.config.memory_limit_gb {
            self.capabilities.memory_gb = memory_limit.min(self.capabilities.memory_gb);
        }
        
        // Apply power limits
        if let Some(power_limit) = self.config.power_limit_watts {
            self.capabilities.power_consumption_watts = power_limit.min(self.capabilities.power_consumption_watts);
        }
        
        // Apply optimization level
        match self.config.optimization_level {
            OptimizationLevel::Power => {
                self.capabilities.power_consumption_watts *= 0.7;
                self.capabilities.peak_performance_gflops *= 0.8;
            }
            OptimizationLevel::Balanced => {
                // Default settings
            }
            OptimizationLevel::Performance => {
                self.capabilities.peak_performance_gflops *= 1.2;
                self.capabilities.power_consumption_watts *= 1.3;
            }
            OptimizationLevel::Maximum => {
                self.capabilities.peak_performance_gflops *= 1.5;
                self.capabilities.power_consumption_watts *= 1.5;
            }
        }
        
        self.last_updated = Utc::now();
        Ok(())
    }

    /// Run hardware self-test
    async fn run_self_test(&mut self) -> QuantumResult<()> {
        debug!("Running self-test for device: {}", self.name);
        
        // Simulate self-test
        tokio::time::sleep(tokio::time::Duration::from_millis(50)).await;
        
        // Check if device passes self-test
        let test_passed = true; // Simplified
        
        if !test_passed {
            self.status = HardwareStatus::Error;
            return Err(QuantumError::HardwareError { 
                component: self.name.clone(), 
                message: "Self-test failed for device".to_string() 
            });
        }
        
        debug!("Self-test passed for device: {}", self.name);
        Ok(())
    }

    /// Update hardware metrics
    pub fn update_metrics(&mut self, operation_time_ms: f64, success: bool) {
        let now = Utc::now();
        let time_diff = now.signed_duration_since(self.metrics.last_updated).num_milliseconds() as f64;
        
        self.metrics.total_operations += 1;
        if success {
            self.metrics.successful_operations += 1;
        } else {
            self.metrics.failed_operations += 1;
        }
        
        // Update error rate
        self.metrics.error_rate = self.metrics.failed_operations as f64 / self.metrics.total_operations as f64;
        
        // Update latency (exponential moving average)
        self.metrics.latency_ms = 0.9 * self.metrics.latency_ms + 0.1 * operation_time_ms;
        
        // Update throughput
        if time_diff > 0.0 {
            self.metrics.throughput_ops_per_second = 1000.0 / operation_time_ms;
        }
        
        // Simulate other metrics
        self.metrics.utilization_percentage = (self.metrics.total_operations % 100) as f64;
        self.metrics.memory_usage_gb = self.capabilities.memory_gb * 0.7;
        self.metrics.memory_utilization_percentage = 70.0;
        self.metrics.power_consumption_watts = self.capabilities.power_consumption_watts * 0.8;
        self.metrics.temperature_celsius = 25.0 + (self.metrics.utilization_percentage / 10.0);
        self.metrics.clock_speed_mhz = 1000.0 + (self.metrics.utilization_percentage * 5.0);
        
        self.metrics.last_updated = now;
        self.last_updated = now;
        
        // Update telemetry
        // metrics::gauge!("hardware_utilization_percentage", "device_id" => self.id.clone()).set(self.metrics.utilization_percentage);
        // metrics::gauge!("hardware_memory_usage_gb", "device_id" => self.id.clone()).set(self.metrics.memory_usage_gb);
        // metrics::gauge!("hardware_temperature_celsius", "device_id" => self.id.clone()).set(self.metrics.temperature_celsius);
        // metrics::histogram!("hardware_operation_latency_ms", "device_id" => self.id.clone()).record(operation_time_ms);
    }

    /// Check if device is available
    pub fn is_available(&self) -> bool {
        matches!(self.status, HardwareStatus::Available)
    }

    /// Check if device is healthy
    pub fn is_healthy(&self) -> bool {
        self.is_available() && 
        self.metrics.error_rate < 0.1 && 
        self.metrics.temperature_celsius < 85.0 &&
        self.metrics.memory_utilization_percentage < 95.0
    }

    /// Get performance score
    pub fn get_performance_score(&self) -> f64 {
        if !self.is_healthy() {
            return 0.0;
        }
        
        let utilization_score = (100.0 - self.metrics.utilization_percentage) / 100.0;
        let error_score = 1.0 - self.metrics.error_rate;
        let thermal_score = (85.0 - self.metrics.temperature_celsius) / 85.0;
        let memory_score = (100.0 - self.metrics.memory_utilization_percentage) / 100.0;
        
        (utilization_score * 0.3 + error_score * 0.3 + thermal_score * 0.2 + memory_score * 0.2) * 100.0
    }

    /// Reset device
    pub async fn reset(&mut self) -> QuantumResult<()> {
        info!("Resetting hardware device: {}", self.name);
        
        self.status = HardwareStatus::Maintenance;
        
        // Simulate reset
        tokio::time::sleep(tokio::time::Duration::from_millis(200)).await;
        
        // Reset metrics
        self.metrics = HardwareMetrics::default();
        
        // Re-initialize
        self.initialize().await?;
        
        info!("Hardware device {} reset successfully", self.name);
        // metrics::counter!("hardware_devices_reset_total", 1);
        
        Ok(())
    }

    /// Shutdown device
    pub async fn shutdown(&mut self) -> QuantumResult<()> {
        info!("Shutting down hardware device: {}", self.name);
        
        self.status = HardwareStatus::Offline;
        
        // Simulate shutdown
        tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
        
        self.last_updated = Utc::now();
        
        info!("Hardware device {} shutdown successfully", self.name);
        // metrics::counter!("hardware_devices_shutdown_total", 1);
        
        Ok(())
    }

    /// Check device compatibility with operation
    pub fn is_compatible_with_operation(&self, operation_type: &str) -> bool {
        match operation_type {
            "quantum_simulation" => {
                matches!(self.hardware_type, HardwareType::CPU | HardwareType::GPU | HardwareType::QuantumProcessor)
            }
            "matrix_multiplication" => {
                matches!(self.hardware_type, HardwareType::CPU | HardwareType::GPU | HardwareType::TPU)
            }
            "neural_network" => {
                matches!(self.hardware_type, HardwareType::GPU | HardwareType::TPU)
            }
            "quantum_gates" => {
                matches!(self.hardware_type, HardwareType::QuantumProcessor)
            }
            _ => true, // Default compatibility
        }
    }
}

/// Hardware manager for managing multiple hardware devices
#[derive(Debug)]
pub struct HardwareManager {
    devices: Arc<Mutex<HashMap<String, HardwareDevice>>>,
    #[allow(dead_code)]
    scheduler: Arc<Mutex<HardwareScheduler>>,
    #[allow(dead_code)]
    monitor: Arc<Mutex<HardwareMonitor>>,
}

impl HardwareManager {
    /// Create a new hardware manager
    pub fn new() -> QuantumResult<Self> {
        let manager = Self {
            devices: Arc::new(Mutex::new(HashMap::new())),
            scheduler: Arc::new(Mutex::new(HardwareScheduler::new())),
            monitor: Arc::new(Mutex::new(HardwareMonitor::new())),
        };
        
        info!("Hardware manager initialized");
        Ok(manager)
    }

    /// Discover and register hardware devices
    pub async fn discover_devices(&self) -> QuantumResult<()> {
        info!("Discovering hardware devices");
        
        // Discover CPU
        let cpu_device = HardwareDevice::new(
            "CPU".to_string(),
            HardwareType::CPU,
            HardwareVendor::Intel, // Simplified
            "Generic CPU".to_string(),
        );
        
        self.register_device(cpu_device).await?;
        
        // Discover GPU if available
        #[cfg(feature = "cuda")]
        {
            if let Ok(_) = cudarc::driver::CudaDevice::new(0) {
                let gpu_device = HardwareDevice::new(
                    "NVIDIA GPU".to_string(),
                    HardwareType::GPU,
                    HardwareVendor::NVIDIA,
                    "Generic GPU".to_string(),
                );
                
                self.register_device(gpu_device).await?;
            }
        }
        
        // Discover quantum processors (simulated)
        let quantum_device = HardwareDevice::new(
            "Quantum Simulator".to_string(),
            HardwareType::QuantumProcessor,
            HardwareVendor::IBM,
            "Quantum Simulator".to_string(),
        );
        
        self.register_device(quantum_device).await?;
        
        info!("Hardware discovery completed");
        Ok(())
    }

    /// Register a hardware device
    pub async fn register_device(&self, mut device: HardwareDevice) -> QuantumResult<()> {
        device.initialize().await?;
        
        let device_id = device.id.clone();
        
        {
            let mut devices = self.devices.lock().unwrap();
            devices.insert(device_id.clone(), device);
        }
        
        info!("Hardware device {} registered", device_id);
        // metrics::counter!("hardware_devices_registered_total", 1);
        
        Ok(())
    }

    /// Unregister a hardware device
    pub async fn unregister_device(&self, device_id: &str) -> QuantumResult<()> {
        let mut devices = self.devices.lock().unwrap();
        
        if let Some(mut device) = devices.remove(device_id) {
            device.shutdown().await?;
            info!("Hardware device {} unregistered", device_id);
            // metrics::counter!("hardware_devices_unregistered_total", 1);
            Ok(())
        } else {
            Err(QuantumError::hardware_error("device", format!("Device not found: {}", device_id)))
        }
    }

    /// Get device by ID
    pub fn get_device(&self, device_id: &str) -> Option<HardwareDevice> {
        let devices = self.devices.lock().unwrap();
        devices.get(device_id).cloned()
    }

    /// List all devices
    pub fn list_devices(&self) -> Vec<HardwareDevice> {
        let devices = self.devices.lock().unwrap();
        devices.values().cloned().collect()
    }

    /// Find best device for operation
    pub fn find_best_device(&self, operation_type: &str) -> Option<HardwareDevice> {
        let devices = self.devices.lock().unwrap();
        
        let mut best_device = None;
        let mut best_score = 0.0;
        
        for device in devices.values() {
            if device.is_compatible_with_operation(operation_type) && device.is_healthy() {
                let score = device.get_performance_score();
                if score > best_score {
                    best_score = score;
                    best_device = Some(device.clone());
                }
            }
        }
        
        best_device
    }

    /// Get system health report
    pub fn get_health_report(&self) -> HashMap<String, f64> {
        let devices = self.devices.lock().unwrap();
        devices.iter()
            .map(|(id, device)| (id.clone(), device.get_performance_score()))
            .collect()
    }

    /// Get system capabilities
    pub fn get_system_capabilities(&self) -> SystemCapabilities {
        let devices = self.devices.lock().unwrap();
        
        let mut capabilities = SystemCapabilities::default();
        
        for device in devices.values() {
            capabilities.total_devices += 1;
            capabilities.total_compute_units += device.capabilities.compute_units;
            capabilities.total_memory_gb += device.capabilities.memory_gb;
            capabilities.total_performance_gflops += device.capabilities.peak_performance_gflops;
            
            match device.hardware_type {
                HardwareType::CPU => capabilities.cpu_devices += 1,
                HardwareType::GPU => capabilities.gpu_devices += 1,
                HardwareType::QuantumProcessor => capabilities.quantum_devices += 1,
                HardwareType::FPGA => capabilities.fpga_devices += 1,
                HardwareType::TPU => capabilities.tpu_devices += 1,
                HardwareType::CustomAccelerator => capabilities.custom_devices += 1,
            }
        }
        
        capabilities
    }

    /// Update all device metrics
    pub fn update_all_metrics(&self) {
        let mut devices = self.devices.lock().unwrap();
        
        for device in devices.values_mut() {
            // Simulate operation
            device.update_metrics(10.0, true);
        }
    }

    /// Reset all devices
    pub async fn reset_all_devices(&self) -> QuantumResult<()> {
        let device_ids: Vec<String> = {
            let devices = self.devices.lock().unwrap();
            devices.keys().cloned().collect()
        };
        
        for device_id in device_ids {
            let mut devices = self.devices.lock().unwrap();
            if let Some(device) = devices.get_mut(&device_id) {
                device.reset().await?;
            }
        }
        
        info!("All devices reset successfully");
        Ok(())
    }

    /// Shutdown all devices
    pub async fn shutdown_all_devices(&self) -> QuantumResult<()> {
        let device_ids: Vec<String> = {
            let devices = self.devices.lock().unwrap();
            devices.keys().cloned().collect()
        };
        
        for device_id in device_ids {
            let mut devices = self.devices.lock().unwrap();
            if let Some(device) = devices.get_mut(&device_id) {
                device.shutdown().await?;
            }
        }
        
        info!("All devices shutdown successfully");
        Ok(())
    }
}

/// System capabilities summary
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct SystemCapabilities {
    pub total_devices: u32,
    pub cpu_devices: u32,
    pub gpu_devices: u32,
    pub quantum_devices: u32,
    pub fpga_devices: u32,
    pub tpu_devices: u32,
    pub custom_devices: u32,
    pub total_compute_units: u32,
    pub total_memory_gb: f64,
    pub total_performance_gflops: f64,
}

/// Hardware scheduler for task allocation
#[derive(Debug)]
struct HardwareScheduler {
    #[allow(dead_code)]
    task_queue: Vec<ScheduledTask>,
    #[allow(dead_code)]
    allocation_strategy: AllocationStrategy,
}

#[derive(Debug, Clone)]
struct ScheduledTask {
    #[allow(dead_code)]
    id: String,
    #[allow(dead_code)]
    operation_type: String,
    #[allow(dead_code)]
    priority: u32,
    #[allow(dead_code)]
    estimated_time_ms: f64,
    #[allow(dead_code)]
    created_at: DateTime<Utc>,
}

#[derive(Debug, Clone, Copy)]
#[allow(dead_code)]
enum AllocationStrategy {
    RoundRobin,
    BestFit,
    LoadBalanced,
    PriorityBased,
}

impl HardwareScheduler {
    fn new() -> Self {
        Self {
            task_queue: Vec::new(),
            allocation_strategy: AllocationStrategy::LoadBalanced,
        }
    }

    #[allow(dead_code)]
    fn schedule_task(&mut self, task: ScheduledTask) {
        self.task_queue.push(task);
        self.task_queue.sort_by(|a, b| b.priority.cmp(&a.priority));
    }

    #[allow(dead_code)]
    fn get_next_task(&mut self) -> Option<ScheduledTask> {
        self.task_queue.pop()
    }
}

/// Hardware monitor for real-time monitoring
#[derive(Debug)]
struct HardwareMonitor {
    #[allow(dead_code)]
    monitoring_enabled: bool,
    #[allow(dead_code)]
    alert_thresholds: HashMap<String, f64>,
}

impl HardwareMonitor {
    fn new() -> Self {
        let mut thresholds = HashMap::new();
        thresholds.insert("temperature".to_string(), 80.0);
        thresholds.insert("utilization".to_string(), 90.0);
        thresholds.insert("error_rate".to_string(), 0.1);
        thresholds.insert("memory_usage".to_string(), 90.0);
        
        Self {
            monitoring_enabled: true,
            alert_thresholds: thresholds,
        }
    }

    #[allow(dead_code)]
    fn check_alerts(&self, device: &HardwareDevice) -> Vec<String> {
        let mut alerts = Vec::new();
        
        if !self.monitoring_enabled {
            return alerts;
        }
        
        if device.metrics.temperature_celsius > self.alert_thresholds["temperature"] {
            alerts.push(format!("High temperature: {:.1}°C", device.metrics.temperature_celsius));
        }
        
        if device.metrics.utilization_percentage > self.alert_thresholds["utilization"] {
            alerts.push(format!("High utilization: {:.1}%", device.metrics.utilization_percentage));
        }
        
        if device.metrics.error_rate > self.alert_thresholds["error_rate"] {
            alerts.push(format!("High error rate: {:.3}", device.metrics.error_rate));
        }
        
        if device.metrics.memory_utilization_percentage > self.alert_thresholds["memory_usage"] {
            alerts.push(format!("High memory usage: {:.1}%", device.metrics.memory_utilization_percentage));
        }
        
        alerts
    }
}

/// Backend capabilities for different hardware types
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BackendCapabilities {
    pub supported_operations: Vec<String>,
    pub max_qubits: Option<usize>,
    pub max_matrix_size: Option<usize>,
    pub supports_multithreading: bool,
    pub supports_gpu_acceleration: bool,
    pub supports_quantum_simulation: bool,
    pub memory_requirements_gb: f64,
    pub performance_characteristics: HashMap<String, f64>,
}

impl Default for BackendCapabilities {
    fn default() -> Self {
        Self {
            supported_operations: vec![
                "quantum_simulation".to_string(),
                "matrix_multiplication".to_string(),
                "vector_operations".to_string(),
            ],
            max_qubits: Some(32),
            max_matrix_size: Some(1024),
            supports_multithreading: true,
            supports_gpu_acceleration: false,
            supports_quantum_simulation: true,
            memory_requirements_gb: 1.0,
            performance_characteristics: HashMap::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokio_test;

    #[test]
    fn test_hardware_device_creation() {
        let device = HardwareDevice::new(
            "Test Device".to_string(),
            HardwareType::CPU,
            HardwareVendor::Intel,
            "Test Model".to_string(),
        );
        
        assert_eq!(device.name, "Test Device");
        assert_eq!(device.hardware_type, HardwareType::CPU);
        assert_eq!(device.vendor, HardwareVendor::Intel);
        assert_eq!(device.model, "Test Model");
        assert_eq!(device.status, HardwareStatus::Initializing);
    }

    #[tokio::test]
    async fn test_hardware_initialization() {
        let mut device = HardwareDevice::new(
            "Test Device".to_string(),
            HardwareType::CPU,
            HardwareVendor::Intel,
            "Test Model".to_string(),
        );
        
        assert!(device.initialize().await.is_ok());
        assert_eq!(device.status, HardwareStatus::Available);
        assert!(device.is_available());
    }

    #[test]
    fn test_hardware_capabilities() {
        let capabilities = HardwareCapabilities::default();
        
        assert_eq!(capabilities.compute_units, 1);
        assert_eq!(capabilities.memory_gb, 8.0);
        assert!(capabilities.supports_parallel_execution);
        assert!(capabilities.supports_fp32);
        assert!(capabilities.supports_fp64);
    }

    #[test]
    fn test_hardware_metrics_update() {
        let mut device = HardwareDevice::new(
            "Test Device".to_string(),
            HardwareType::CPU,
            HardwareVendor::Intel,
            "Test Model".to_string(),
        );
        
        assert_eq!(device.metrics.total_operations, 0);
        
        device.update_metrics(100.0, true);
        assert_eq!(device.metrics.total_operations, 1);
        assert_eq!(device.metrics.successful_operations, 1);
        assert_eq!(device.metrics.failed_operations, 0);
        assert_eq!(device.metrics.latency_ms, 100.0);
        
        device.update_metrics(200.0, false);
        assert_eq!(device.metrics.total_operations, 2);
        assert_eq!(device.metrics.successful_operations, 1);
        assert_eq!(device.metrics.failed_operations, 1);
        assert_eq!(device.metrics.error_rate, 0.5);
    }

    #[test]
    fn test_device_health_check() {
        let mut device = HardwareDevice::new(
            "Test Device".to_string(),
            HardwareType::CPU,
            HardwareVendor::Intel,
            "Test Model".to_string(),
        );
        
        device.status = HardwareStatus::Available;
        device.metrics.error_rate = 0.05;
        device.metrics.temperature_celsius = 60.0;
        device.metrics.memory_utilization_percentage = 70.0;
        
        assert!(device.is_healthy());
        
        device.metrics.temperature_celsius = 90.0;
        assert!(!device.is_healthy());
    }

    #[test]
    fn test_performance_score() {
        let mut device = HardwareDevice::new(
            "Test Device".to_string(),
            HardwareType::CPU,
            HardwareVendor::Intel,
            "Test Model".to_string(),
        );
        
        device.status = HardwareStatus::Available;
        device.metrics.utilization_percentage = 50.0;
        device.metrics.error_rate = 0.01;
        device.metrics.temperature_celsius = 40.0;
        device.metrics.memory_utilization_percentage = 60.0;
        
        let score = device.get_performance_score();
        assert!(score > 0.0);
        assert!(score <= 100.0);
    }

    #[test]
    fn test_operation_compatibility() {
        let cpu_device = HardwareDevice::new(
            "CPU".to_string(),
            HardwareType::CPU,
            HardwareVendor::Intel,
            "Test CPU".to_string(),
        );
        
        let gpu_device = HardwareDevice::new(
            "GPU".to_string(),
            HardwareType::GPU,
            HardwareVendor::NVIDIA,
            "Test GPU".to_string(),
        );
        
        let quantum_device = HardwareDevice::new(
            "Quantum".to_string(),
            HardwareType::QuantumProcessor,
            HardwareVendor::IBM,
            "Test Quantum".to_string(),
        );
        
        assert!(cpu_device.is_compatible_with_operation("quantum_simulation"));
        assert!(gpu_device.is_compatible_with_operation("matrix_multiplication"));
        assert!(quantum_device.is_compatible_with_operation("quantum_gates"));
        
        assert!(!quantum_device.is_compatible_with_operation("matrix_multiplication"));
    }

    #[tokio::test]
    async fn test_device_reset() {
        let mut device = HardwareDevice::new(
            "Test Device".to_string(),
            HardwareType::CPU,
            HardwareVendor::Intel,
            "Test Model".to_string(),
        );
        
        device.initialize().await.unwrap();
        device.update_metrics(100.0, true);
        
        assert_eq!(device.metrics.total_operations, 1);
        
        device.reset().await.unwrap();
        assert_eq!(device.metrics.total_operations, 0);
        assert_eq!(device.status, HardwareStatus::Available);
    }

    #[tokio::test]
    async fn test_device_shutdown() {
        let mut device = HardwareDevice::new(
            "Test Device".to_string(),
            HardwareType::CPU,
            HardwareVendor::Intel,
            "Test Model".to_string(),
        );
        
        device.initialize().await.unwrap();
        assert_eq!(device.status, HardwareStatus::Available);
        
        device.shutdown().await.unwrap();
        assert_eq!(device.status, HardwareStatus::Offline);
    }

    #[tokio::test]
    async fn test_hardware_manager() {
        let manager = HardwareManager::new().unwrap();
        
        let device = HardwareDevice::new(
            "Test Device".to_string(),
            HardwareType::CPU,
            HardwareVendor::Intel,
            "Test Model".to_string(),
        );
        
        let device_id = device.id.clone();
        
        // Register device
        manager.register_device(device).await.unwrap();
        
        // Check if device is registered
        let devices = manager.list_devices();
        assert_eq!(devices.len(), 1);
        assert_eq!(devices[0].id, device_id);
        
        // Get device by ID
        let retrieved_device = manager.get_device(&device_id);
        assert!(retrieved_device.is_some());
        assert_eq!(retrieved_device.unwrap().id, device_id);
        
        // Unregister device
        manager.unregister_device(&device_id).await.unwrap();
        
        let devices_after_unregister = manager.list_devices();
        assert_eq!(devices_after_unregister.len(), 0);
    }

    #[tokio::test]
    async fn test_device_discovery() {
        let manager = HardwareManager::new().unwrap();
        
        manager.discover_devices().await.unwrap();
        
        let devices = manager.list_devices();
        assert!(devices.len() > 0);
        
        // Should at least have CPU and quantum simulator
        let device_types: Vec<HardwareType> = devices.iter().map(|d| d.hardware_type).collect();
        assert!(device_types.contains(&HardwareType::CPU));
        assert!(device_types.contains(&HardwareType::QuantumProcessor));
    }

    #[tokio::test]
    async fn test_find_best_device() {
        let manager = HardwareManager::new().unwrap();
        
        let cpu_device = HardwareDevice::new(
            "CPU".to_string(),
            HardwareType::CPU,
            HardwareVendor::Intel,
            "Test CPU".to_string(),
        );
        
        let gpu_device = HardwareDevice::new(
            "GPU".to_string(),
            HardwareType::GPU,
            HardwareVendor::NVIDIA,
            "Test GPU".to_string(),
        );
        
        manager.register_device(cpu_device).await.unwrap();
        manager.register_device(gpu_device).await.unwrap();
        
        let best_for_quantum = manager.find_best_device("quantum_simulation");
        assert!(best_for_quantum.is_some());
        
        let best_for_matrix = manager.find_best_device("matrix_multiplication");
        assert!(best_for_matrix.is_some());
    }

    #[test]
    fn test_system_capabilities() {
        let manager = HardwareManager::new().unwrap();
        
        let capabilities = manager.get_system_capabilities();
        assert_eq!(capabilities.total_devices, 0);
        assert_eq!(capabilities.cpu_devices, 0);
        assert_eq!(capabilities.gpu_devices, 0);
        assert_eq!(capabilities.quantum_devices, 0);
    }

    #[test]
    fn test_hardware_config() {
        let config = HardwareConfig::default();
        
        assert_eq!(config.device_id, "default");
        assert_eq!(config.optimization_level, OptimizationLevel::Balanced);
        assert!(!config.enable_power_saving);
        assert!(config.enable_turbo_boost);
    }

    #[test]
    fn test_hardware_scheduler() {
        let mut scheduler = HardwareScheduler::new();
        
        let task = ScheduledTask {
            id: "task1".to_string(),
            operation_type: "quantum_simulation".to_string(),
            priority: 10,
            estimated_time_ms: 100.0,
            created_at: Utc::now(),
        };
        
        scheduler.schedule_task(task);
        
        let next_task = scheduler.get_next_task();
        assert!(next_task.is_some());
        assert_eq!(next_task.unwrap().id, "task1");
    }

    #[test]
    fn test_hardware_monitor() {
        let monitor = HardwareMonitor::new();
        
        let mut device = HardwareDevice::new(
            "Test Device".to_string(),
            HardwareType::CPU,
            HardwareVendor::Intel,
            "Test Model".to_string(),
        );
        
        device.metrics.temperature_celsius = 85.0;
        device.metrics.utilization_percentage = 95.0;
        
        let alerts = monitor.check_alerts(&device);
        assert_eq!(alerts.len(), 2);
        assert!(alerts.iter().any(|a| a.contains("High temperature")));
        assert!(alerts.iter().any(|a| a.contains("High utilization")));
    }

    #[test]
    fn test_backend_capabilities() {
        let capabilities = BackendCapabilities::default();
        
        assert!(capabilities.supported_operations.contains(&"quantum_simulation".to_string()));
        assert!(capabilities.supported_operations.contains(&"matrix_multiplication".to_string()));
        assert_eq!(capabilities.max_qubits, Some(32));
        assert!(capabilities.supports_multithreading);
        assert!(capabilities.supports_quantum_simulation);
    }

    #[test]
    fn test_hardware_enums() {
        let hardware_types = vec![
            HardwareType::CPU,
            HardwareType::GPU,
            HardwareType::QuantumProcessor,
            HardwareType::FPGA,
            HardwareType::TPU,
            HardwareType::CustomAccelerator,
        ];
        
        let vendors = vec![
            HardwareVendor::Intel,
            HardwareVendor::AMD,
            HardwareVendor::NVIDIA,
            HardwareVendor::IBM,
            HardwareVendor::Google,
            HardwareVendor::Rigetti,
            HardwareVendor::IonQ,
            HardwareVendor::Honeywell,
            HardwareVendor::Xanadu,
            HardwareVendor::PsiQuantum,
            HardwareVendor::Unknown,
        ];
        
        let statuses = vec![
            HardwareStatus::Available,
            HardwareStatus::Busy,
            HardwareStatus::Maintenance,
            HardwareStatus::Error,
            HardwareStatus::Offline,
            HardwareStatus::Initializing,
        ];
        
        // Test serialization/deserialization
        for hw_type in hardware_types {
            let serialized = serde_json::to_string(&hw_type).unwrap();
            let deserialized: HardwareType = serde_json::from_str(&serialized).unwrap();
            assert_eq!(hw_type, deserialized);
        }
        
        for vendor in vendors {
            let serialized = serde_json::to_string(&vendor).unwrap();
            let deserialized: HardwareVendor = serde_json::from_str(&serialized).unwrap();
            assert_eq!(vendor, deserialized);
        }
        
        for status in statuses {
            let serialized = serde_json::to_string(&status).unwrap();
            let deserialized: HardwareStatus = serde_json::from_str(&serialized).unwrap();
            assert_eq!(status, deserialized);
        }
    }
}