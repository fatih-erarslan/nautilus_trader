// High-performance quantum operations using CUDA kernels
// Optimized for QBMIA trading algorithms

use std::sync::Arc;
use cudarc::driver::{CudaDevice, CudaStream, DevicePtr, LaunchAsync, LaunchConfig, DriverError};
use std::marker::PhantomData;
use std::collections::HashMap;

use super::{QBMIACudaContext, KernelMetrics};

/// Complex number type for quantum states
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub struct Complex32 {
    pub re: f32,
    pub im: f32,
}

impl Complex32 {
    pub fn new(re: f32, im: f32) -> Self {
        Self { re, im }
    }
    
    pub fn norm_squared(&self) -> f32 {
        self.re * self.re + self.im * self.im
    }
    
    pub fn norm(&self) -> f32 {
        self.norm_squared().sqrt()
    }
}

/// Quantum gate types supported by CUDA kernels
#[derive(Debug, Clone)]
pub enum QuantumGate {
    Hadamard { qubit: usize },
    CNOT { control: usize, target: usize },
    RX { qubit: usize, angle: f32 },
    RY { qubit: usize, angle: f32 },
    RZ { qubit: usize, angle: f32 },
    U3 { qubit: usize, theta: f32, phi: f32, lambda: f32 },
}

/// Quantum state vector stored on GPU
pub struct QuantumState {
    state_vector: DevicePtr<Complex32>,
    num_qubits: usize,
    batch_size: usize,
    context: Arc<QBMIACudaContext>,
    is_normalized: bool,
}

impl QuantumState {
    /// Create a new quantum state in |0...0⟩
    pub fn new(
        num_qubits: usize,
        batch_size: usize,
        context: Arc<QBMIACudaContext>,
    ) -> Result<Self, DriverError> {
        let state_dim = 1 << num_qubits;
        let total_size = state_dim * batch_size;
        
        // Allocate GPU memory
        let mut state_vector = context.device().alloc_zeros::<Complex32>(total_size)?;
        
        // Initialize to |0...0⟩ state
        let init_state = vec![Complex32::new(1.0, 0.0); batch_size];
        state_vector.copy_from(&init_state)?;
        
        Ok(Self {
            state_vector,
            num_qubits,
            batch_size,
            context,
            is_normalized: true,
        })
    }
    
    /// Create quantum state from classical data (feature map)
    pub fn from_features(
        features: &[f32],
        num_qubits: usize,
        batch_size: usize,
        context: Arc<QBMIACudaContext>,
    ) -> Result<Self, DriverError> {
        let state_dim = 1 << num_qubits;
        let total_size = state_dim * batch_size;
        
        // Allocate GPU memory
        let state_vector = context.device().alloc_zeros::<Complex32>(total_size)?;
        
        // Upload features to GPU
        let features_gpu = context.device().htod_copy(features.to_vec())?;
        
        // Apply quantum feature map
        let func = context.get_function("quantum_kernels", "launch_quantum_feature_map_f32")?;
        let config = LaunchConfig {
            grid_dim: ((total_size + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        };
        
        unsafe {
            func.launch(
                config,
                (
                    &state_vector,
                    &features_gpu,
                    features.len() / batch_size,
                    num_qubits as i32,
                    batch_size as i32,
                ),
            )?;
        }
        
        context.stream().synchronize()?;
        
        Ok(Self {
            state_vector,
            num_qubits,
            batch_size,
            context,
            is_normalized: false,
        })
    }
    
    /// Apply a quantum gate to the state
    pub fn apply_gate(&mut self, gate: &QuantumGate) -> Result<KernelMetrics, DriverError> {
        let start_event = self.context.device().create_event()?;
        let end_event = self.context.device().create_event()?;
        
        start_event.record(self.context.stream())?;
        
        match gate {
            QuantumGate::Hadamard { qubit } => {
                self.apply_hadamard(*qubit)?;
            }
            QuantumGate::CNOT { control, target } => {
                self.apply_cnot(*control, *target)?;
            }
            QuantumGate::RX { qubit, angle } => {
                self.apply_rx(*qubit, *angle)?;
            }
            QuantumGate::RY { qubit, angle } => {
                self.apply_ry(*qubit, *angle)?;
            }
            QuantumGate::RZ { qubit, angle } => {
                self.apply_rz(*qubit, *angle)?;
            }
            QuantumGate::U3 { qubit, theta, phi, lambda } => {
                // Decompose U3 into RZ-RY-RZ sequence
                self.apply_rz(*qubit, *phi)?;
                self.apply_ry(*qubit, *theta)?;
                self.apply_rz(*qubit, *lambda)?;
            }
        }
        
        end_event.record(self.context.stream())?;
        self.context.stream().synchronize()?;
        
        // Calculate metrics
        let state_size = (1 << self.num_qubits) * self.batch_size * std::mem::size_of::<Complex32>();
        KernelMetrics::from_events(
            format!("{:?}", gate),
            &start_event,
            &end_event,
            state_size * 2, // Read + write
        )
    }
    
    /// Apply Hadamard gate
    fn apply_hadamard(&mut self, qubit: usize) -> Result<(), DriverError> {
        let func = self.context.get_function("quantum_kernels", "launch_hadamard_gate_f32")?;
        
        unsafe {
            func.launch(
                self.get_launch_config(),
                (
                    &self.state_vector,
                    qubit as i32,
                    self.num_qubits as i32,
                    self.batch_size as i32,
                ),
            )?;
        }
        
        self.is_normalized = true;
        Ok(())
    }
    
    /// Apply CNOT gate
    fn apply_cnot(&mut self, control: usize, target: usize) -> Result<(), DriverError> {
        let func = self.context.get_function("quantum_kernels", "launch_cnot_gate_f32")?;
        
        unsafe {
            func.launch(
                self.get_launch_config(),
                (
                    &self.state_vector,
                    control as i32,
                    target as i32,
                    self.num_qubits as i32,
                    self.batch_size as i32,
                ),
            )?;
        }
        
        Ok(())
    }
    
    /// Apply RX rotation gate
    fn apply_rx(&mut self, qubit: usize, angle: f32) -> Result<(), DriverError> {
        let func = self.context.get_function("quantum_kernels", "launch_rx_gate_f32")?;
        
        unsafe {
            func.launch(
                self.get_launch_config(),
                (
                    &self.state_vector,
                    angle,
                    qubit as i32,
                    self.num_qubits as i32,
                    self.batch_size as i32,
                ),
            )?;
        }
        
        Ok(())
    }
    
    /// Apply RY rotation gate
    fn apply_ry(&mut self, qubit: usize, angle: f32) -> Result<(), DriverError> {
        let func = self.context.get_function("quantum_kernels", "launch_ry_gate_f32")?;
        
        unsafe {
            func.launch(
                self.get_launch_config(),
                (
                    &self.state_vector,
                    angle,
                    qubit as i32,
                    self.num_qubits as i32,
                    self.batch_size as i32,
                ),
            )?;
        }
        
        Ok(())
    }
    
    /// Apply RZ rotation gate
    fn apply_rz(&mut self, qubit: usize, angle: f32) -> Result<(), DriverError> {
        let func = self.context.get_function("quantum_kernels", "launch_rz_gate_f32")?;
        
        unsafe {
            func.launch(
                self.get_launch_config(),
                (
                    &self.state_vector,
                    angle,
                    qubit as i32,
                    self.num_qubits as i32,
                    self.batch_size as i32,
                ),
            )?;
        }
        
        Ok(())
    }
    
    /// Normalize the quantum state
    pub fn normalize(&mut self) -> Result<(), DriverError> {
        if self.is_normalized {
            return Ok(());
        }
        
        let func = self.context.get_function("quantum_kernels", "launch_normalize_state_f32")?;
        let config = LaunchConfig {
            grid_dim: (self.batch_size as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
        };
        
        unsafe {
            func.launch(
                config,
                (
                    &self.state_vector,
                    (1 << self.num_qubits) as i32,
                    self.batch_size as i32,
                ),
            )?;
        }
        
        self.is_normalized = true;
        Ok(())
    }
    
    /// Calculate expectation value of an observable
    pub fn expectation_value(&self, observable: &[f32]) -> Result<Vec<f32>, DriverError> {
        // Allocate result array
        let result = self.context.device().alloc_zeros::<f32>(self.batch_size)?;
        let observable_gpu = self.context.device().htod_copy(observable.to_vec())?;
        
        let func = self.context.get_function("quantum_kernels", "launch_expectation_value_f32")?;
        let config = LaunchConfig {
            grid_dim: (self.batch_size as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 256 * std::mem::size_of::<f32>() as u32,
        };
        
        unsafe {
            func.launch(
                config,
                (
                    &result,
                    &self.state_vector,
                    &observable_gpu,
                    self.num_qubits as i32,
                    self.batch_size as i32,
                ),
            )?;
        }
        
        // Copy result back to host
        let mut host_result = vec![0.0f32; self.batch_size];
        result.copy_to(&mut host_result)?;
        
        Ok(host_result)
    }
    
    /// Get the quantum state amplitudes (for debugging)
    pub fn get_amplitudes(&self) -> Result<Vec<Complex32>, DriverError> {
        let total_size = (1 << self.num_qubits) * self.batch_size;
        let mut amplitudes = vec![Complex32::new(0.0, 0.0); total_size];
        self.state_vector.copy_to(&mut amplitudes)?;
        Ok(amplitudes)
    }
    
    /// Get optimal launch configuration for kernels
    fn get_launch_config(&self) -> LaunchConfig {
        let total_elements = (1 << self.num_qubits) * self.batch_size;
        LaunchConfig {
            grid_dim: ((total_elements + 255) / 256, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    }
    
    /// Get number of qubits
    pub fn num_qubits(&self) -> usize {
        self.num_qubits
    }
    
    /// Get batch size
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

/// Quantum circuit for executing multiple gates
pub struct QuantumCircuit {
    gates: Vec<QuantumGate>,
    num_qubits: usize,
    parameters: HashMap<String, f32>,
}

impl QuantumCircuit {
    /// Create a new quantum circuit
    pub fn new(num_qubits: usize) -> Self {
        Self {
            gates: Vec::new(),
            num_qubits,
            parameters: HashMap::new(),
        }
    }
    
    /// Add a gate to the circuit
    pub fn add_gate(&mut self, gate: QuantumGate) {
        self.gates.push(gate);
    }
    
    /// Add a parameterized gate
    pub fn add_parameterized_gate(&mut self, gate_type: &str, qubit: usize, param_name: String) {
        // Store parameter name for later substitution
        self.parameters.insert(param_name.clone(), 0.0);
        
        match gate_type {
            "RX" => self.gates.push(QuantumGate::RX { qubit, angle: 0.0 }),
            "RY" => self.gates.push(QuantumGate::RY { qubit, angle: 0.0 }),
            "RZ" => self.gates.push(QuantumGate::RZ { qubit, angle: 0.0 }),
            _ => panic!("Unsupported parameterized gate: {}", gate_type),
        }
    }
    
    /// Set parameter values
    pub fn set_parameters(&mut self, params: HashMap<String, f32>) {
        self.parameters.extend(params);
        
        // Update gate parameters
        // This is a simplified implementation - in practice, you'd need to
        // track which parameters correspond to which gates
    }
    
    /// Execute the circuit on a quantum state
    pub fn execute(&self, state: &mut QuantumState) -> Result<Vec<KernelMetrics>, DriverError> {
        let mut metrics = Vec::new();
        
        for gate in &self.gates {
            let metric = state.apply_gate(gate)?;
            metrics.push(metric);
        }
        
        Ok(metrics)
    }
    
    /// Create a variational quantum circuit for QBMIA
    pub fn create_qbmia_ansatz(num_qubits: usize, num_layers: usize) -> Self {
        let mut circuit = Self::new(num_qubits);
        
        // Add entangling layers
        for layer in 0..num_layers {
            // RY rotations on all qubits
            for qubit in 0..num_qubits {
                circuit.add_parameterized_gate("RY", qubit, format!("theta_{}_{}", layer, qubit));
            }
            
            // Entangling CNOTs in circular pattern
            for qubit in 0..num_qubits {
                let target = (qubit + 1) % num_qubits;
                circuit.add_gate(QuantumGate::CNOT { 
                    control: qubit, 
                    target 
                });
            }
        }
        
        // Final layer of RY rotations
        for qubit in 0..num_qubits {
            circuit.add_parameterized_gate("RY", qubit, format!("final_theta_{}", qubit));
        }
        
        circuit
    }
    
    /// Get the number of parameters in the circuit
    pub fn num_parameters(&self) -> usize {
        self.parameters.len()
    }
    
    /// Get parameter names
    pub fn parameter_names(&self) -> Vec<String> {
        self.parameters.keys().cloned().collect()
    }
}

/// Quantum gradient computation for variational algorithms
pub struct QuantumGradient {
    context: Arc<QBMIACudaContext>,
    shift_angle: f32,
}

impl QuantumGradient {
    pub fn new(context: Arc<QBMIACudaContext>) -> Self {
        Self {
            context,
            shift_angle: std::f32::consts::PI / 2.0, // Parameter shift rule
        }
    }
    
    /// Compute gradients using parameter shift rule
    pub fn compute_gradients(
        &self,
        circuit: &QuantumCircuit,
        initial_state: &QuantumState,
        observable: &[f32],
        parameters: &[f32],
    ) -> Result<Vec<f32>, DriverError> {
        let num_params = parameters.len();
        let mut gradients = vec![0.0; num_params];
        
        for (param_idx, &param_value) in parameters.iter().enumerate() {
            // Forward evaluation (param + shift)
            let mut forward_params = parameters.to_vec();
            forward_params[param_idx] = param_value + self.shift_angle;
            
            let mut forward_state = self.prepare_state_with_params(
                circuit, 
                initial_state, 
                &forward_params
            )?;
            let forward_exp = forward_state.expectation_value(observable)?[0];
            
            // Backward evaluation (param - shift)
            let mut backward_params = parameters.to_vec();
            backward_params[param_idx] = param_value - self.shift_angle;
            
            let mut backward_state = self.prepare_state_with_params(
                circuit, 
                initial_state, 
                &backward_params
            )?;
            let backward_exp = backward_state.expectation_value(observable)?[0];
            
            // Compute gradient
            gradients[param_idx] = (forward_exp - backward_exp) / (2.0 * self.shift_angle.sin());
        }
        
        Ok(gradients)
    }
    
    /// Helper function to prepare state with given parameters
    fn prepare_state_with_params(
        &self,
        circuit: &QuantumCircuit,
        initial_state: &QuantumState,
        parameters: &[f32],
    ) -> Result<QuantumState, DriverError> {
        // This is a simplified implementation
        // In practice, you'd need to properly handle parameter substitution
        let mut state = QuantumState::new(
            initial_state.num_qubits(),
            initial_state.batch_size(),
            self.context.clone(),
        )?;
        
        // Execute circuit with parameters
        circuit.execute(&mut state)?;
        
        Ok(state)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_quantum_state_creation() {
        if let Ok(context) = QBMIACudaContext::new(0) {
            let context = Arc::new(context);
            
            let state = QuantumState::new(3, 1, context).unwrap();
            assert_eq!(state.num_qubits(), 3);
            assert_eq!(state.batch_size(), 1);
        }
    }
    
    #[test]
    fn test_hadamard_gate() {
        if let Ok(context) = QBMIACudaContext::new(0) {
            let context = Arc::new(context);
            
            let mut state = QuantumState::new(1, 1, context).unwrap();
            let gate = QuantumGate::Hadamard { qubit: 0 };
            let metrics = state.apply_gate(&gate).unwrap();
            
            assert!(metrics.execution_time_us < 100.0); // Should be very fast
            
            // Check that state is properly superposed
            let amplitudes = state.get_amplitudes().unwrap();
            let prob_0 = amplitudes[0].norm_squared();
            let prob_1 = amplitudes[1].norm_squared();
            
            assert!((prob_0 - 0.5).abs() < 1e-6);
            assert!((prob_1 - 0.5).abs() < 1e-6);
        }
    }
    
    #[test]
    fn test_qbmia_circuit() {
        let circuit = QuantumCircuit::create_qbmia_ansatz(4, 2);
        assert!(circuit.num_parameters() > 0);
        assert_eq!(circuit.num_qubits, 4);
    }
}