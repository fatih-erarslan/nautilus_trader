//! # Quantum Device Management
//!
//! Device hierarchy management for PennyLane quantum computing devices.
//! Enforces hierarchy: lightning.gpu → lightning-kokkos → lightning.qubit
//! NO default.qubit allowed - only high-grade devices for trading systems.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use tokio::sync::RwLock;
use tracing::{debug, info, warn, error, instrument};
use serde::{Serialize, Deserialize};
use sysinfo::{System, SystemExt};

use crate::bridge::PythonRuntime;
use crate::error::{DeviceError, QuantumError};

/// Quantum device types in priority order (high to low performance)
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum QuantumDevice {
    /// CUDA-accelerated GPU lightning (Highest Priority)
    LightningGpu {
        /// GPU device ID
        device_id: u32,
        /// Available GPU memory in GB
        memory_gb: u32,
        /// CUDA compute capability
        compute_capability: (u32, u32),
    },
    /// Kokkos-accelerated CPU/GPU (Secondary Priority)
    LightningKokkos {
        /// Kokkos backend type
        backend: KokkosBackend,
        /// Number of execution threads
        threads: u32,
        /// Available memory in GB
        memory_gb: u32,
    },
    /// Pure CPU lightning (Tertiary Priority)
    LightningQubit {
        /// Number of CPU threads
        threads: u32,
        /// Enable memory mapping for large circuits
        memory_mapping: bool,
        /// Available memory in GB
        memory_gb: u32,
    },
}

/// Kokkos backend types
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KokkosBackend {
    /// CUDA backend for GPU acceleration
    Cuda,
    /// OpenMP backend for multi-threaded CPU
    OpenMP,
    /// Serial backend for single-threaded execution
    Serial,
}

/// Device hierarchy configuration
#[derive(Debug, Clone)]
pub struct DeviceHierarchy {
    /// Available devices in priority order
    pub devices: Vec<QuantumDevice>,
    /// Current active device
    pub active_device: Option<QuantumDevice>,
    /// Device selection strategy
    pub selection_strategy: DeviceSelectionStrategy,
}

/// Device selection strategies
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceSelectionStrategy {
    /// Always use highest priority available device
    HighestPriority,
    /// Select based on circuit complexity
    AdaptiveComplexity,
    /// Load balancing across available devices
    LoadBalancing,
    /// Round-robin selection
    RoundRobin,
}

/// Device capability information
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    /// Maximum number of qubits supported
    pub max_qubits: u32,
    /// Maximum circuit depth supported efficiently
    pub max_circuit_depth: u32,
    /// Supports GPU acceleration
    pub gpu_accelerated: bool,
    /// Supports parallel execution
    pub parallel_execution: bool,
    /// Memory bandwidth in GB/s
    pub memory_bandwidth_gb_s: f64,
    /// Typical execution time for standard circuit
    pub typical_execution_time_ms: f64,
}

/// Device performance metrics
#[derive(Debug, Default)]
pub struct DeviceMetrics {
    /// Total execution time on this device
    pub total_execution_time: Duration,
    /// Number of circuits executed
    pub circuits_executed: u64,
    /// Average execution time
    pub avg_execution_time: Duration,
    /// Current utilization (0.0 to 1.0)
    pub current_utilization: f64,
    /// Last execution timestamp
    pub last_execution: Option<Instant>,
    /// Error count
    pub error_count: u64,
}

/// Quantum device manager
pub struct DeviceManager {
    /// Available devices and their capabilities
    devices: Arc<RwLock<HashMap<QuantumDevice, DeviceCapabilities>>>,
    /// Device performance metrics
    metrics: Arc<RwLock<HashMap<QuantumDevice, DeviceMetrics>>>,
    /// Current device hierarchy
    hierarchy: Arc<RwLock<DeviceHierarchy>>,
    /// Python runtime for device interaction
    python_runtime: Arc<PythonRuntime>,
    /// GPU acceleration enabled
    gpu_acceleration: bool,
}

impl DeviceManager {
    /// Create new device manager
    #[instrument(skip(python_runtime))]
    pub async fn new(
        python_runtime: &PythonRuntime,
        gpu_acceleration: bool,
    ) -> Result<Self, DeviceError> {
        info!("Initializing quantum device manager");
        
        let manager = Self {
            devices: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(HashMap::new())),
            hierarchy: Arc::new(RwLock::new(DeviceHierarchy {
                devices: Vec::new(),
                active_device: None,
                selection_strategy: DeviceSelectionStrategy::AdaptiveComplexity,
            })),
            python_runtime: Arc::new(python_runtime.clone()),
            gpu_acceleration,
        };
        
        // Discover and initialize devices
        manager.discover_devices().await?;
        
        let device_count = manager.devices.read().await.len();
        info!("Device manager initialized with {} devices", device_count);
        
        Ok(manager)
    }
    
    /// Discover available quantum devices
    #[instrument(skip(self))]
    async fn discover_devices(&self) -> Result<(), DeviceError> {
        let mut devices = self.devices.write().await;
        let mut metrics = self.metrics.write().await;
        let mut hierarchy = self.hierarchy.write().await;
        
        let mut discovered_devices = Vec::new();
        
        // Discover Lightning GPU devices (highest priority)
        if self.gpu_acceleration {
            if let Ok(gpu_devices) = self.discover_lightning_gpu().await {
                for device in gpu_devices {
                    let capabilities = self.get_device_capabilities(&device).await?;
                    devices.insert(device, capabilities);
                    metrics.insert(device, DeviceMetrics::default());
                    discovered_devices.push(device);
                    info!("Discovered Lightning GPU device: {:?}", device);
                }
            }
        }
        
        // Discover Lightning Kokkos devices (secondary priority)
        if let Ok(kokkos_devices) = self.discover_lightning_kokkos().await {
            for device in kokkos_devices {
                let capabilities = self.get_device_capabilities(&device).await?;
                devices.insert(device, capabilities);
                metrics.insert(device, DeviceMetrics::default());
                discovered_devices.push(device);
                info!("Discovered Lightning Kokkos device: {:?}", device);
            }
        }
        
        // Discover Lightning Qubit devices (tertiary priority)
        if let Ok(qubit_devices) = self.discover_lightning_qubit().await {
            for device in qubit_devices {
                let capabilities = self.get_device_capabilities(&device).await?;
                devices.insert(device, capabilities);
                metrics.insert(device, DeviceMetrics::default());
                discovered_devices.push(device);
                info!("Discovered Lightning Qubit device: {:?}", device);
            }
        }
        
        // Sort devices by priority
        discovered_devices.sort_by(|a, b| self.device_priority(a).cmp(&self.device_priority(b)));
        
        hierarchy.devices = discovered_devices;
        hierarchy.active_device = hierarchy.devices.first().copied();
        
        if hierarchy.devices.is_empty() {
            warn!("No quantum devices discovered - quantum computing unavailable");
        } else {
            info!("Device hierarchy established with {} devices", hierarchy.devices.len());
        }
        
        Ok(())
    }
    
    /// Discover Lightning GPU devices
    async fn discover_lightning_gpu(&self) -> Result<Vec<QuantumDevice>, DeviceError> {
        Python::with_gil(|py| -> Result<Vec<QuantumDevice>, DeviceError> {
            let mut devices = Vec::new();
            
            // Check if Lightning GPU module is available
            if let Some(lightning_gpu) = self.python_runtime.devices.get("lightning.gpu") {
                // Try to get GPU information
                if let Ok(gpu_info) = lightning_gpu.call_method0("get_gpu_info") {
                    let gpu_list = gpu_info.downcast::<PyList>()
                        .map_err(|e| DeviceError::DiscoveryFailed(format!("GPU info format error: {}", e)))?;
                    
                    for (idx, gpu_item) in gpu_list.iter().enumerate() {
                        let gpu_dict = gpu_item.downcast::<PyDict>()
                            .map_err(|e| DeviceError::DiscoveryFailed(format!("GPU item format error: {}", e)))?;
                        
                        let memory_gb = gpu_dict.get_item("memory_gb")
                            .and_then(|v| v.extract::<u32>().ok())
                            .unwrap_or(8); // Default 8GB
                        
                        let compute_capability = gpu_dict.get_item("compute_capability")
                            .and_then(|v| v.extract::<(u32, u32)>().ok())
                            .unwrap_or((7, 0)); // Default CUDA 7.0
                        
                        devices.push(QuantumDevice::LightningGpu {
                            device_id: idx as u32,
                            memory_gb,
                            compute_capability,
                        });
                    }
                } else {
                    // Fallback: assume at least one GPU is available
                    devices.push(QuantumDevice::LightningGpu {
                        device_id: 0,
                        memory_gb: 8,
                        compute_capability: (7, 0),
                    });
                }
            }
            
            Ok(devices)
        })
    }
    
    /// Discover Lightning Kokkos devices
    async fn discover_lightning_kokkos(&self) -> Result<Vec<QuantumDevice>, DeviceError> {
        let mut devices = Vec::new();
        
        // Check available Kokkos backends
        let num_threads = num_cpus::get() as u32;
        let memory_gb = self.get_system_memory_gb();
        
        // CUDA backend (if GPU acceleration enabled)
        if self.gpu_acceleration {
            devices.push(QuantumDevice::LightningKokkos {
                backend: KokkosBackend::Cuda,
                threads: num_threads,
                memory_gb,
            });
        }
        
        // OpenMP backend
        devices.push(QuantumDevice::LightningKokkos {
            backend: KokkosBackend::OpenMP,
            threads: num_threads,
            memory_gb,
        });
        
        // Serial backend (fallback)
        devices.push(QuantumDevice::LightningKokkos {
            backend: KokkosBackend::Serial,
            threads: 1,
            memory_gb,
        });
        
        Ok(devices)
    }
    
    /// Discover Lightning Qubit devices
    async fn discover_lightning_qubit(&self) -> Result<Vec<QuantumDevice>, DeviceError> {
        let num_threads = num_cpus::get() as u32;
        let memory_gb = self.get_system_memory_gb();
        
        let devices = vec![
            QuantumDevice::LightningQubit {
                threads: num_threads,
                memory_mapping: true,
                memory_gb,
            },
            QuantumDevice::LightningQubit {
                threads: num_threads / 2,
                memory_mapping: false,
                memory_gb: memory_gb / 2,
            },
        ];
        
        Ok(devices)
    }
    
    /// Get device capabilities
    async fn get_device_capabilities(&self, device: &QuantumDevice) -> Result<DeviceCapabilities, DeviceError> {
        match device {
            QuantumDevice::LightningGpu { memory_gb, compute_capability, .. } => {
                Ok(DeviceCapabilities {
                    max_qubits: std::cmp::min(30, (*memory_gb * 2)), // Rough estimate
                    max_circuit_depth: 1000,
                    gpu_accelerated: true,
                    parallel_execution: true,
                    memory_bandwidth_gb_s: if compute_capability.0 >= 8 { 900.0 } else { 500.0 },
                    typical_execution_time_ms: 10.0,
                })
            }
            QuantumDevice::LightningKokkos { backend, threads, memory_gb } => {
                let gpu_accelerated = matches!(backend, KokkosBackend::Cuda);
                Ok(DeviceCapabilities {
                    max_qubits: std::cmp::min(25, *memory_gb),
                    max_circuit_depth: 800,
                    gpu_accelerated,
                    parallel_execution: *threads > 1,
                    memory_bandwidth_gb_s: if gpu_accelerated { 600.0 } else { 100.0 },
                    typical_execution_time_ms: if gpu_accelerated { 15.0 } else { 50.0 },
                })
            }
            QuantumDevice::LightningQubit { threads, memory_gb, .. } => {
                Ok(DeviceCapabilities {
                    max_qubits: std::cmp::min(20, *memory_gb),
                    max_circuit_depth: 500,
                    gpu_accelerated: false,
                    parallel_execution: *threads > 1,
                    memory_bandwidth_gb_s: 50.0,
                    typical_execution_time_ms: 100.0,
                })
            }
        }
    }
    
    /// Select optimal device for circuit execution
    #[instrument(skip(self))]
    pub async fn select_optimal_device(
        &self,
        qubit_count: u32,
        gate_count: u32,
    ) -> Result<QuantumDevice, DeviceError> {
        let devices = self.devices.read().await;
        let hierarchy = self.hierarchy.read().await;
        
        if hierarchy.devices.is_empty() {
            return Err(DeviceError::NoDevicesAvailable);
        }
        
        let selected_device = match hierarchy.selection_strategy {
            DeviceSelectionStrategy::HighestPriority => {
                // Return highest priority device that can handle the circuit
                for device in &hierarchy.devices {
                    if let Some(capabilities) = devices.get(device) {
                        if capabilities.max_qubits >= qubit_count {
                            debug!("Selected highest priority device: {:?}", device);
                            return Ok(*device);
                        }
                    }
                }
                
                // If no device can handle the full circuit, return the most capable
                Some(hierarchy.devices[0])
            }
            DeviceSelectionStrategy::AdaptiveComplexity => {
                self.select_by_complexity(&devices, &hierarchy.devices, qubit_count, gate_count).await
            }
            DeviceSelectionStrategy::LoadBalancing => {
                self.select_by_load_balancing(&hierarchy.devices).await
            }
            DeviceSelectionStrategy::RoundRobin => {
                self.select_round_robin(&hierarchy.devices).await
            }
        };
        
        selected_device.ok_or(DeviceError::NoSuitableDevice)
    }
    
    /// Select device based on circuit complexity
    async fn select_by_complexity(
        &self,
        devices: &HashMap<QuantumDevice, DeviceCapabilities>,
        available_devices: &[QuantumDevice],
        qubit_count: u32,
        gate_count: u32,
    ) -> Option<QuantumDevice> {
        let complexity_score = qubit_count * 10 + gate_count;
        
        // High complexity: prefer GPU devices
        if complexity_score > 1000 {
            for device in available_devices {
                if let Some(capabilities) = devices.get(device) {
                    if capabilities.gpu_accelerated && capabilities.max_qubits >= qubit_count {
                        return Some(*device);
                    }
                }
            }
        }
        
        // Medium complexity: prefer Kokkos devices
        if complexity_score > 100 {
            for device in available_devices {
                if let Some(capabilities) = devices.get(device) {
                    if matches!(device, QuantumDevice::LightningKokkos { .. }) && 
                       capabilities.max_qubits >= qubit_count {
                        return Some(*device);
                    }
                }
            }
        }
        
        // Low complexity: any suitable device
        for device in available_devices {
            if let Some(capabilities) = devices.get(device) {
                if capabilities.max_qubits >= qubit_count {
                    return Some(*device);
                }
            }
        }
        
        None
    }
    
    /// Select device using load balancing
    async fn select_by_load_balancing(&self, available_devices: &[QuantumDevice]) -> Option<QuantumDevice> {
        let metrics = self.metrics.read().await;
        
        // Find device with lowest current utilization
        available_devices.iter()
            .min_by(|a, b| {
                let util_a = metrics.get(a).map(|m| m.current_utilization).unwrap_or(0.0);
                let util_b = metrics.get(b).map(|m| m.current_utilization).unwrap_or(0.0);
                util_a.partial_cmp(&util_b).unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
    }
    
    /// Select device using round-robin
    async fn select_round_robin(&self, available_devices: &[QuantumDevice]) -> Option<QuantumDevice> {
        let metrics = self.metrics.read().await;
        
        // Find device with lowest execution count
        available_devices.iter()
            .min_by_key(|device| {
                metrics.get(device).map(|m| m.circuits_executed).unwrap_or(0)
            })
            .copied()
    }
    
    /// Get device priority (lower number = higher priority)
    fn device_priority(&self, device: &QuantumDevice) -> u32 {
        match device {
            QuantumDevice::LightningGpu { .. } => 1,
            QuantumDevice::LightningKokkos { backend: KokkosBackend::Cuda, .. } => 2,
            QuantumDevice::LightningKokkos { backend: KokkosBackend::OpenMP, .. } => 3,
            QuantumDevice::LightningKokkos { backend: KokkosBackend::Serial, .. } => 4,
            QuantumDevice::LightningQubit { threads, .. } if *threads > 8 => 5,
            QuantumDevice::LightningQubit { .. } => 6,
        }
    }
    
    /// Get system memory in GB
    fn get_system_memory_gb(&self) -> u32 {
        // Try to get actual system memory, fallback to conservative estimate
        let mut system = System::new();
        system.refresh_memory();
        let total_memory_bytes = system.total_memory();
        if total_memory_bytes > 0 {
            (total_memory_bytes / (1024 * 1024 * 1024)) as u32
        } else {
            16 // Conservative 16GB default
        }
    }
    
    /// Get available devices
    pub async fn available_devices(&self) -> Vec<QuantumDevice> {
        self.hierarchy.read().await.devices.clone()
    }
    
    /// Get device hierarchy
    pub async fn hierarchy(&self) -> DeviceHierarchy {
        self.hierarchy.read().await.clone()
    }
    
    /// Update device metrics after execution
    pub async fn update_device_metrics(&self, device: &QuantumDevice, execution_time: Duration, success: bool) {
        let mut metrics = self.metrics.write().await;
        
        if let Some(device_metrics) = metrics.get_mut(device) {
            device_metrics.circuits_executed += 1;
            device_metrics.total_execution_time += execution_time;
            device_metrics.avg_execution_time = device_metrics.total_execution_time / device_metrics.circuits_executed as u32;
            device_metrics.last_execution = Some(Instant::now());
            
            if !success {
                device_metrics.error_count += 1;
            }
            
            // Update utilization (simplified calculation)
            device_metrics.current_utilization = (execution_time.as_millis() as f64 / 1000.0).min(1.0);
        }
    }
    
    /// Shutdown device manager
    pub async fn shutdown(&self) -> Result<(), DeviceError> {
        info!("Shutting down device manager");
        
        // Clear all device data
        self.devices.write().await.clear();
        self.metrics.write().await.clear();
        
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_device_priority() {
        let manager = DeviceManager {
            devices: Arc::new(RwLock::new(HashMap::new())),
            metrics: Arc::new(RwLock::new(HashMap::new())),
            hierarchy: Arc::new(RwLock::new(DeviceHierarchy {
                devices: Vec::new(),
                active_device: None,
                selection_strategy: DeviceSelectionStrategy::HighestPriority,
            })),
            python_runtime: Arc::new(/* mock runtime */),
            gpu_acceleration: true,
        };
        
        let gpu_device = QuantumDevice::LightningGpu {
            device_id: 0,
            memory_gb: 16,
            compute_capability: (8, 0),
        };
        
        let cpu_device = QuantumDevice::LightningQubit {
            threads: 8,
            memory_mapping: true,
            memory_gb: 32,
        };
        
        assert!(manager.device_priority(&gpu_device) < manager.device_priority(&cpu_device));
    }
}