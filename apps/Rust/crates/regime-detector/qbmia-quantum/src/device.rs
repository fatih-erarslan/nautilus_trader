//! Quantum device management with PennyLane backend
//!
//! Implements the device hierarchy: lightning.gpu → lightning.kokkos → lightning.qubit

use crate::error::{QuantumError, Result};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use std::sync::Arc;
use parking_lot::{Mutex, RwLock};
use tracing::{debug, info, warn};
use once_cell::sync::OnceCell;

/// PennyLane device types with priority ordering
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum DeviceType {
    /// GPU-accelerated lightning device (highest priority)
    LightningGPU,
    /// Kokkos-based lightning device
    LightningKokkos,
    /// CPU-based lightning device
    LightningQubit,
    /// Default qubit device (avoid if possible)
    DefaultQubit,
}

impl DeviceType {
    /// Get the PennyLane device name
    pub fn device_name(&self) -> &'static str {
        match self {
            DeviceType::LightningGPU => "lightning.gpu",
            DeviceType::LightningKokkos => "lightning.kokkos",
            DeviceType::LightningQubit => "lightning.qubit",
            DeviceType::DefaultQubit => "default.qubit",
        }
    }

    /// Get expected performance characteristics
    pub fn performance_score(&self) -> u32 {
        match self {
            DeviceType::LightningGPU => 100,
            DeviceType::LightningKokkos => 80,
            DeviceType::LightningQubit => 60,
            DeviceType::DefaultQubit => 20,
        }
    }
}

/// Quantum device wrapper for PennyLane devices
pub struct QuantumDevice {
    device_type: DeviceType,
    py_device: PyObject,
    num_qubits: usize,
    shots: Option<usize>,
    performance_metrics: Arc<Mutex<DeviceMetrics>>,
}

/// Device performance metrics
#[derive(Debug, Default)]
struct DeviceMetrics {
    total_circuits: usize,
    total_execution_time_us: u64,
    avg_circuit_time_us: u64,
    last_execution_time_us: u64,
}

impl QuantumDevice {
    /// Create a new quantum device with specified parameters
    pub fn new(device_type: DeviceType, num_qubits: usize, shots: Option<usize>) -> Result<Self> {
        Python::with_gil(|py| {
            let pennylane = py.import("pennylane")?;
            
            // Create device with appropriate parameters
            let kwargs = PyDict::new(py);
            kwargs.set_item("wires", num_qubits)?;
            
            if let Some(s) = shots {
                kwargs.set_item("shots", s)?;
            }
            
            // Device-specific configurations
            match device_type {
                DeviceType::LightningGPU => {
                    // Enable GPU-specific optimizations
                    kwargs.set_item("gpu_device_id", 0)?;
                    kwargs.set_item("batch_obs", true)?;
                }
                DeviceType::LightningKokkos => {
                    // Enable Kokkos threading
                    kwargs.set_item("kokkos_threads", num_cpus::get())?;
                }
                _ => {}
            }
            
            let py_device = pennylane.call_method(
                "device",
                (device_type.device_name(),),
                Some(kwargs)
            )?;
            
            Ok(QuantumDevice {
                device_type,
                py_device: py_device.into(),
                num_qubits,
                shots,
                performance_metrics: Arc::new(Mutex::new(DeviceMetrics::default())),
            })
        })
    }

    /// Execute a quantum circuit on this device
    pub fn execute_circuit(&self, circuit: &PyObject) -> Result<PyObject> {
        let start = std::time::Instant::now();
        
        Python::with_gil(|py| {
            let result = circuit.call_method1(py, "execute", (self.py_device.as_ref(py),))?;
            
            let elapsed_us = start.elapsed().as_micros() as u64;
            
            // Update metrics
            {
                let mut metrics = self.performance_metrics.lock();
                metrics.total_circuits += 1;
                metrics.total_execution_time_us += elapsed_us;
                metrics.avg_circuit_time_us = 
                    metrics.total_execution_time_us / metrics.total_circuits as u64;
                metrics.last_execution_time_us = elapsed_us;
            }
            
            if elapsed_us > 10_000 {
                warn!("Circuit execution took {}μs, exceeding 10ms target", elapsed_us);
            }
            
            Ok(result)
        })
    }

    /// Get device type
    pub fn device_type(&self) -> DeviceType {
        self.device_type
    }

    /// Get average circuit execution time in microseconds
    pub fn avg_execution_time_us(&self) -> u64 {
        self.performance_metrics.lock().avg_circuit_time_us
    }
}

/// Device manager for optimal device selection and fallback
pub struct DeviceManager {
    available_devices: Vec<DeviceType>,
    device_cache: dashmap::DashMap<(DeviceType, usize, Option<usize>), Arc<QuantumDevice>>,
    preferred_device: OnceCell<DeviceType>,
}

impl DeviceManager {
    /// Create a new device manager
    pub fn new() -> Self {
        DeviceManager {
            available_devices: Vec::new(),
            device_cache: dashmap::DashMap::new(),
            preferred_device: OnceCell::new(),
        }
    }

    /// Initialize and detect available devices
    pub fn initialize_devices(&mut self) -> Result<()> {
        info!("Detecting available PennyLane devices");
        
        Python::with_gil(|py| {
            let pennylane = py.import("pennylane")?;
            
            // Check for each device type in priority order
            for device_type in [
                DeviceType::LightningGPU,
                DeviceType::LightningKokkos,
                DeviceType::LightningQubit,
            ] {
                if self.check_device_available(py, pennylane, device_type) {
                    self.available_devices.push(device_type);
                    info!("Device {} is available", device_type.device_name());
                }
            }
            
            // Always add default.qubit as fallback
            self.available_devices.push(DeviceType::DefaultQubit);
            
            // Set preferred device
            if let Some(&preferred) = self.available_devices.first() {
                let _ = self.preferred_device.set(preferred);
                info!("Selected {} as preferred device", preferred.device_name());
            }
            
            Ok(())
        })
    }

    /// Check if a specific device is available
    fn check_device_available(&self, py: Python, pennylane: &PyModule, device_type: DeviceType) -> bool {
        let result = pennylane.call_method1(
            "device",
            (device_type.device_name(), 2) // Test with 2 qubits
        );
        
        match result {
            Ok(_) => true,
            Err(e) => {
                debug!("Device {} not available: {}", device_type.device_name(), e);
                false
            }
        }
    }

    /// Get or create a quantum device with fallback logic
    pub fn get_device(&self, num_qubits: usize, shots: Option<usize>) -> Result<Arc<QuantumDevice>> {
        // Try preferred device first
        if let Some(&preferred) = self.preferred_device.get() {
            let key = (preferred, num_qubits, shots);
            
            if let Some(device) = self.device_cache.get(&key) {
                return Ok(device.clone());
            }
            
            if let Ok(device) = QuantumDevice::new(preferred, num_qubits, shots) {
                let device = Arc::new(device);
                self.device_cache.insert(key, device.clone());
                return Ok(device);
            }
        }
        
        // Fallback through available devices
        for &device_type in &self.available_devices {
            let key = (device_type, num_qubits, shots);
            
            if let Some(device) = self.device_cache.get(&key) {
                return Ok(device.clone());
            }
            
            match QuantumDevice::new(device_type, num_qubits, shots) {
                Ok(device) => {
                    let device = Arc::new(device);
                    self.device_cache.insert(key, device.clone());
                    info!("Using fallback device: {}", device_type.device_name());
                    return Ok(device);
                }
                Err(e) => {
                    warn!("Failed to create device {}: {}", device_type.device_name(), e);
                }
            }
        }
        
        Err(QuantumError::DeviceError("No quantum devices available".to_string()))
    }

    /// Get the optimal device for a given circuit size
    pub fn get_optimal_device(&self, num_qubits: usize) -> Result<Arc<QuantumDevice>> {
        // For large circuits, prefer GPU if available
        if num_qubits > 20 && self.available_devices.contains(&DeviceType::LightningGPU) {
            return self.get_device(num_qubits, None);
        }
        
        // Otherwise use the default selection
        self.get_device(num_qubits, None)
    }

    /// Shutdown and cleanup devices
    pub fn shutdown(&mut self) -> Result<()> {
        self.device_cache.clear();
        self.available_devices.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_device_type_ordering() {
        assert!(DeviceType::LightningGPU < DeviceType::LightningKokkos);
        assert!(DeviceType::LightningKokkos < DeviceType::LightningQubit);
        assert!(DeviceType::LightningQubit < DeviceType::DefaultQubit);
    }

    #[test]
    fn test_performance_scores() {
        assert!(DeviceType::LightningGPU.performance_score() > 
                DeviceType::LightningKokkos.performance_score());
        assert!(DeviceType::LightningKokkos.performance_score() > 
                DeviceType::LightningQubit.performance_score());
    }
}