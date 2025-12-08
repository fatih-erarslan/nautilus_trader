//! PennyLane Backend Integration
//! 
//! Provides seamless integration with PennyLane's device hierarchy:
//! lightning.gpu → lightning.kokkos → lightning.qubit

use crate::{error::Result, types::*};
use quantum_core::{QuantumCircuit, QuantumState, QuantumResult, ComplexAmplitude};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList, PyTuple};
use numpy::{PyArray1, PyArray2, IntoPyArray};
use std::collections::HashMap;
use parking_lot::RwLock;
use tracing::{info, warn, debug};

/// PennyLane device wrapper
pub struct PennyLaneDevice {
    device: PyObject,
    device_type: DeviceType,
    num_qubits: usize,
    shots: Option<usize>,
    capabilities: DeviceCapabilities,
}

/// Available PennyLane device types
#[derive(Debug, Clone, PartialEq)]
pub enum DeviceType {
    LightningGpu,
    LightningKokkos,
    LightningQubit,
    DefaultQubit,
}

/// Device capabilities
#[derive(Debug, Clone)]
pub struct DeviceCapabilities {
    pub supports_gpu: bool,
    pub supports_gradients: bool,
    pub supports_shots: bool,
    pub max_qubits: usize,
    pub native_gates: Vec<String>,
}

/// PennyLane backend coordinator
pub struct PennyLaneBackend {
    devices: RwLock<HashMap<String, PennyLaneDevice>>,
    python_runtime: Python<'static>,
    pennylane_module: PyObject,
    device_hierarchy: Vec<DeviceType>,
}

impl PennyLaneBackend {
    /// Initialize PennyLane backend with device hierarchy
    pub async fn new() -> Result<Self> {
        Python::with_gil(|py| {
            // Import PennyLane
            let pennylane = py.import("pennylane")?;
            
            // Setup device hierarchy
            let device_hierarchy = vec![
                DeviceType::LightningGpu,
                DeviceType::LightningKokkos,
                DeviceType::LightningQubit,
                DeviceType::DefaultQubit,
            ];
            
            let backend = Self {
                devices: RwLock::new(HashMap::new()),
                python_runtime: unsafe { Python::assume_gil_acquired() },
                pennylane_module: pennylane.into(),
                device_hierarchy,
            };
            
            // Initialize available devices
            backend.initialize_devices(py)?;
            
            Ok(backend)
        })
    }
    
    /// Initialize all available devices
    fn initialize_devices(&self, py: Python) -> Result<()> {
        let mut devices = self.devices.write();
        
        // Try to initialize each device in hierarchy order
        for device_type in &self.device_hierarchy {
            match self.create_device(py, device_type, 16) {
                Ok(device) => {
                    let name = format!("{:?}", device_type);
                    info!("Initialized PennyLane device: {}", name);
                    devices.insert(name, device);
                }
                Err(e) => {
                    warn!("Failed to initialize {:?}: {}", device_type, e);
                }
            }
        }
        
        if devices.is_empty() {
            return Err(anyhow::anyhow!("No PennyLane devices could be initialized"));
        }
        
        Ok(())
    }
    
    /// Create a PennyLane device
    fn create_device(
        &self,
        py: Python,
        device_type: &DeviceType,
        num_qubits: usize,
    ) -> Result<PennyLaneDevice> {
        let device_name = match device_type {
            DeviceType::LightningGpu => "lightning.gpu",
            DeviceType::LightningKokkos => "lightning.kokkos",
            DeviceType::LightningQubit => "lightning.qubit",
            DeviceType::DefaultQubit => "default.qubit",
        };
        
        let kwargs = PyDict::new(py);
        kwargs.set_item("wires", num_qubits)?;
        
        // Device-specific configuration
        match device_type {
            DeviceType::LightningGpu => {
                kwargs.set_item("batch_obs", true)?;
                kwargs.set_item("gpu_options", PyDict::new(py))?;
            }
            DeviceType::LightningKokkos => {
                kwargs.set_item("kokkos_args", PyDict::new(py))?;
            }
            _ => {}
        }
        
        let device = self.pennylane_module
            .call_method(py, "device", (device_name,), Some(kwargs))?;
        
        let capabilities = self.get_device_capabilities(py, &device, device_type)?;
        
        Ok(PennyLaneDevice {
            device,
            device_type: device_type.clone(),
            num_qubits,
            shots: None,
            capabilities,
        })
    }
    
    /// Get device capabilities
    fn get_device_capabilities(
        &self,
        py: Python,
        device: &PyObject,
        device_type: &DeviceType,
    ) -> Result<DeviceCapabilities> {
        let supports_gpu = matches!(
            device_type,
            DeviceType::LightningGpu | DeviceType::LightningKokkos
        );
        
        // Query device for supported operations
        let operations = device
            .getattr(py, "operations")?
            .extract::<Vec<String>>(py)
            .unwrap_or_default();
        
        Ok(DeviceCapabilities {
            supports_gpu,
            supports_gradients: true,
            supports_shots: true,
            max_qubits: 32,
            native_gates: operations,
        })
    }
    
    /// Select optimal device for circuit
    pub async fn select_optimal_device(&self, circuit: &QuantumCircuit) -> Result<PennyLaneDevice> {
        let devices = self.devices.read();
        let num_qubits = circuit.num_qubits();
        
        // Find best device based on circuit properties
        for device_type in &self.device_hierarchy {
            let device_name = format!("{:?}", device_type);
            
            if let Some(device) = devices.get(&device_name) {
                if device.num_qubits >= num_qubits {
                    // For small circuits, CPU might be faster
                    if num_qubits < 8 && device_type == &DeviceType::LightningGpu {
                        continue;
                    }
                    
                    return Ok(device.clone());
                }
            }
        }
        
        // Fallback: create new device with required qubits
        Python::with_gil(|py| {
            self.create_device(py, &DeviceType::DefaultQubit, num_qubits)
        })
    }
    
    /// Execute circuit on specific device
    pub async fn execute_on_device(
        &self,
        circuit: &QuantumCircuit,
        device: PennyLaneDevice,
    ) -> Result<QuantumResult> {
        Python::with_gil(|py| {
            let start = std::time::Instant::now();
            
            // Create QNode
            let qnode = self.create_qnode(py, circuit, &device)?;
            
            // Execute circuit
            let result = qnode.call0(py)?;
            
            // Extract results
            let state_vector = if device.shots.is_none() {
                Some(self.extract_state_vector(py, &device, circuit.num_qubits())?)
            } else {
                None
            };
            
            let probabilities = self.extract_probabilities(py, result)?;
            
            let execution_time_ns = start.elapsed().as_nanos() as u64;
            
            // Create quantum state
            let quantum_state = if let Some(sv) = state_vector {
                QuantumState::from_amplitudes(sv)
            } else {
                QuantumState::from_probabilities(probabilities.clone())
            };
            
            Ok(QuantumResult::new(
                quantum_state,
                probabilities,
                format!("{:?}", device.device_type),
                execution_time_ns,
            ))
        })
    }
    
    /// Create QNode for circuit execution
    fn create_qnode(
        &self,
        py: Python,
        circuit: &QuantumCircuit,
        device: &PennyLaneDevice,
    ) -> Result<PyObject> {
        // Define quantum function
        let qml = self.pennylane_module.as_ref(py);
        
        // Create function that applies the circuit
        let circuit_fn = PyModule::from_code(
            py,
            &format!(
                r#"
import pennylane as qml
import numpy as np

def circuit_function():
{}
    return qml.probs(wires=range({}))
"#,
                self.generate_pennylane_code(circuit),
                circuit.num_qubits()
            ),
            "circuit.py",
            "circuit",
        )?;
        
        let func = circuit_fn.getattr("circuit_function")?;
        
        // Create QNode
        qml.call_method1("QNode", (func, &device.device))
    }
    
    /// Generate PennyLane code from quantum circuit
    fn generate_pennylane_code(&self, circuit: &QuantumCircuit) -> String {
        let mut code = String::new();
        
        for gate in circuit.gates() {
            let gate_code = match gate.gate_type() {
                GateType::Hadamard => format!("    qml.Hadamard({})\n", gate.target()),
                GateType::PauliX => format!("    qml.PauliX({})\n", gate.target()),
                GateType::PauliY => format!("    qml.PauliY({})\n", gate.target()),
                GateType::PauliZ => format!("    qml.PauliZ({})\n", gate.target()),
                GateType::CNOT => format!("    qml.CNOT([{}, {}])\n", gate.control().unwrap(), gate.target()),
                GateType::RX(angle) => format!("    qml.RX({}, {})\n", angle, gate.target()),
                GateType::RY(angle) => format!("    qml.RY({}, {})\n", angle, gate.target()),
                GateType::RZ(angle) => format!("    qml.RZ({}, {})\n", angle, gate.target()),
                _ => continue,
            };
            code.push_str(&gate_code);
        }
        
        code
    }
    
    /// Extract state vector from device
    fn extract_state_vector(
        &self,
        py: Python,
        device: &PennyLaneDevice,
        num_qubits: usize,
    ) -> Result<Vec<ComplexAmplitude>> {
        let state = device.device.call_method0(py, "state")?;
        let array: &PyArray1<num_complex::Complex<f64>> = state.extract(py)?;
        
        Ok(array.to_vec().into_iter()
            .map(|c| ComplexAmplitude::new(c.re, c.im))
            .collect())
    }
    
    /// Extract probabilities from result
    fn extract_probabilities(&self, py: Python, result: PyObject) -> Result<Vec<f64>> {
        let probs: Vec<f64> = result.extract(py)?;
        Ok(probs)
    }
    
    /// Check if backend is initialized
    pub async fn is_initialized(&self) -> bool {
        !self.devices.read().is_empty()
    }
    
    /// Get available devices
    pub async fn get_available_devices(&self) -> Vec<String> {
        self.devices.read().keys().cloned().collect()
    }
}

impl Clone for PennyLaneDevice {
    fn clone(&self) -> Self {
        Python::with_gil(|py| {
            Self {
                device: self.device.clone_ref(py),
                device_type: self.device_type.clone(),
                num_qubits: self.num_qubits,
                shots: self.shots,
                capabilities: self.capabilities.clone(),
            }
        })
    }
}

impl PennyLaneDevice {
    pub fn supports_gpu(&self) -> bool {
        self.capabilities.supports_gpu
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[tokio::test]
    async fn test_pennylane_initialization() {
        let backend = PennyLaneBackend::new().await;
        assert!(backend.is_ok());
    }
}