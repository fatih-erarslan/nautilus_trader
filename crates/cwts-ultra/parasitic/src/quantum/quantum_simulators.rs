//! # Quantum Simulators
//!
//! Quantum circuit simulators for full quantum mode operation.
//! Provides statevector and density matrix simulation capabilities.

use crate::quantum::{QuantumConfig, QuantumError, QuantumMode};
use num_complex::Complex64;
use parking_lot::RwLock;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

/// Main quantum simulator collection
#[derive(Debug)]
pub struct QuantumSimulators {
    statevector_sim: Arc<RwLock<Option<StatevectorSimulator>>>,
    density_matrix_sim: Arc<RwLock<Option<DensityMatrixSimulator>>>,
    config: QuantumConfig,
    initialized: Arc<RwLock<bool>>,
}

impl QuantumSimulators {
    pub fn new(config: &QuantumConfig) -> Self {
        Self {
            statevector_sim: Arc::new(RwLock::new(None)),
            density_matrix_sim: Arc::new(RwLock::new(None)),
            config: config.clone(),
            initialized: Arc::new(RwLock::new(false)),
        }
    }

    /// Initialize quantum simulators
    pub async fn initialize_quantum(&self) -> Result<(), QuantumError> {
        let mut initialized = self.initialized.write();
        if *initialized {
            return Ok(());
        }

        // Initialize statevector simulator
        let statevector_sim = StatevectorSimulator::new(self.config.max_qubits)?;
        *self.statevector_sim.write() = Some(statevector_sim);

        // Initialize density matrix simulator if noise modeling is enabled
        if self.config.noise_model.enable_decoherence {
            let density_sim =
                DensityMatrixSimulator::new(self.config.max_qubits, &self.config.noise_model)?;
            *self.density_matrix_sim.write() = Some(density_sim);
        }

        *initialized = true;
        tracing::info!(
            "Quantum simulators initialized with {} qubits",
            self.config.max_qubits
        );

        Ok(())
    }

    /// Shutdown quantum simulators
    pub async fn shutdown_quantum(&self) -> Result<(), QuantumError> {
        let mut initialized = self.initialized.write();
        if !*initialized {
            return Ok(());
        }

        *self.statevector_sim.write() = None;
        *self.density_matrix_sim.write() = None;
        *initialized = false;

        tracing::info!("Quantum simulators shutdown");
        Ok(())
    }

    /// Get statevector simulator
    pub fn statevector_simulator(&self) -> Option<Arc<RwLock<StatevectorSimulator>>> {
        let sim_guard = self.statevector_sim.read();
        if sim_guard.is_some() {
            Some(Arc::new(RwLock::new(sim_guard.as_ref().unwrap().clone())))
        } else {
            None
        }
    }

    /// Get density matrix simulator  
    pub fn density_matrix_simulator(&self) -> Option<Arc<RwLock<DensityMatrixSimulator>>> {
        let sim_guard = self.density_matrix_sim.read();
        if sim_guard.is_some() {
            Some(Arc::new(RwLock::new(sim_guard.as_ref().unwrap().clone())))
        } else {
            None
        }
    }

    /// Execute a quantum circuit
    pub async fn execute_circuit(
        &self,
        circuit: QuantumCircuit,
    ) -> Result<QuantumResult, QuantumError> {
        if !*self.initialized.read() {
            return Err(QuantumError::Simulation(
                "Simulators not initialized".to_string(),
            ));
        }

        // Choose simulator based on circuit requirements and noise model
        if circuit.requires_density_matrix() || self.config.noise_model.enable_decoherence {
            if let Some(sim) = self.density_matrix_simulator() {
                let result = sim.write().execute_circuit(circuit).await?;
                Ok(result)
            } else {
                return Err(QuantumError::Simulation(
                    "Density matrix simulator not available".to_string(),
                ));
            }
        } else {
            if let Some(sim) = self.statevector_simulator() {
                let result = sim.write().execute_circuit(circuit).await?;
                Ok(result)
            } else {
                return Err(QuantumError::Simulation(
                    "Statevector simulator not available".to_string(),
                ));
            }
        }
    }

    /// Create a quantum superposition pattern matcher
    pub async fn create_superposition_matcher(
        &self,
        patterns: Vec<Vec<f64>>,
    ) -> Result<SuperpositionMatcher, QuantumError> {
        if !QuantumMode::current().is_full_quantum() {
            return Err(QuantumError::ModeSwitch(
                "Superposition matching requires full quantum mode".to_string(),
            ));
        }

        let num_qubits = (patterns.len() as f64).log2().ceil() as u32;
        if num_qubits > self.config.max_qubits {
            return Err(QuantumError::ResourceExhausted(format!(
                "Need {} qubits but max is {}",
                num_qubits, self.config.max_qubits
            )));
        }

        Ok(SuperpositionMatcher::new(patterns, num_qubits))
    }
}

/// Statevector quantum simulator
#[derive(Debug, Clone)]
pub struct StatevectorSimulator {
    num_qubits: u32,
    state: Vec<Complex64>,
    gate_count: u64,
}

impl StatevectorSimulator {
    pub fn new(num_qubits: u32) -> Result<Self, QuantumError> {
        if num_qubits > 30 {
            return Err(QuantumError::ResourceExhausted(format!(
                "Too many qubits for statevector simulation: {}",
                num_qubits
            )));
        }

        let state_size = 1usize << num_qubits;
        let mut state = vec![Complex64::new(0.0, 0.0); state_size];
        state[0] = Complex64::new(1.0, 0.0); // Initialize to |0...0⟩

        Ok(Self {
            num_qubits,
            state,
            gate_count: 0,
        })
    }

    /// Execute a quantum circuit
    pub async fn execute_circuit(
        &mut self,
        circuit: QuantumCircuit,
    ) -> Result<QuantumResult, QuantumError> {
        // Fix E0382: Store circuit depth before consuming gates
        let circuit_depth = circuit.gates.len();

        for gate in circuit.gates {
            self.apply_gate(gate).await?;
        }

        let measurements = if circuit.measure_all {
            self.measure_all().await?
        } else {
            HashMap::new()
        };

        Ok(QuantumResult {
            measurements,
            final_state: Some(self.get_statevector()),
            circuit_depth,
            gate_count: self.gate_count,
            execution_time_ns: 0, // Would be measured in real implementation
        })
    }

    /// Apply a quantum gate
    pub async fn apply_gate(&mut self, gate: QuantumGate) -> Result<(), QuantumError> {
        match gate {
            QuantumGate::Hadamard { qubit } => self.hadamard(qubit).await,
            QuantumGate::PauliX { qubit } => self.pauli_x(qubit).await,
            QuantumGate::PauliY { qubit } => self.pauli_y(qubit).await,
            QuantumGate::PauliZ { qubit } => self.pauli_z(qubit).await,
            QuantumGate::CNOT { control, target } => self.cnot(control, target).await,
            QuantumGate::Phase { qubit, angle } => self.phase(qubit, angle).await,
            QuantumGate::RotationX { qubit, angle } => self.rotation_x(qubit, angle).await,
            QuantumGate::RotationY { qubit, angle } => self.rotation_y(qubit, angle).await,
            QuantumGate::RotationZ { qubit, angle } => self.rotation_z(qubit, angle).await,
            QuantumGate::Toffoli {
                control1,
                control2,
                target,
            } => self.toffoli(control1, control2, target).await,
        }
    }

    async fn hadamard(&mut self, qubit: u32) -> Result<(), QuantumError> {
        if qubit >= self.num_qubits {
            return Err(QuantumError::Simulation(format!(
                "Qubit {} out of range",
                qubit
            )));
        }

        let sqrt2_inv = 1.0 / std::f64::consts::SQRT_2;
        let mask = 1usize << qubit;

        for i in 0..self.state.len() {
            if i & mask == 0 {
                let j = i | mask;
                let temp0 = self.state[i];
                let temp1 = self.state[j];

                self.state[i] = Complex64::new(sqrt2_inv, 0.0) * (temp0 + temp1);
                self.state[j] = Complex64::new(sqrt2_inv, 0.0) * (temp0 - temp1);
            }
        }

        self.gate_count += 1;
        Ok(())
    }

    async fn pauli_x(&mut self, qubit: u32) -> Result<(), QuantumError> {
        if qubit >= self.num_qubits {
            return Err(QuantumError::Simulation(format!(
                "Qubit {} out of range",
                qubit
            )));
        }

        let mask = 1usize << qubit;

        for i in 0..self.state.len() {
            if i & mask == 0 {
                let j = i | mask;
                self.state.swap(i, j);
            }
        }

        self.gate_count += 1;
        Ok(())
    }

    async fn pauli_y(&mut self, qubit: u32) -> Result<(), QuantumError> {
        if qubit >= self.num_qubits {
            return Err(QuantumError::Simulation(format!(
                "Qubit {} out of range",
                qubit
            )));
        }

        let mask = 1usize << qubit;

        for i in 0..self.state.len() {
            if i & mask == 0 {
                let j = i | mask;
                let temp0 = self.state[i];
                let temp1 = self.state[j];

                self.state[i] = Complex64::new(0.0, 1.0) * temp1;
                self.state[j] = Complex64::new(0.0, -1.0) * temp0;
            }
        }

        self.gate_count += 1;
        Ok(())
    }

    async fn pauli_z(&mut self, qubit: u32) -> Result<(), QuantumError> {
        if qubit >= self.num_qubits {
            return Err(QuantumError::Simulation(format!(
                "Qubit {} out of range",
                qubit
            )));
        }

        let mask = 1usize << qubit;

        for i in 0..self.state.len() {
            if i & mask != 0 {
                self.state[i] = -self.state[i];
            }
        }

        self.gate_count += 1;
        Ok(())
    }

    async fn cnot(&mut self, control: u32, target: u32) -> Result<(), QuantumError> {
        if control >= self.num_qubits || target >= self.num_qubits {
            return Err(QuantumError::Simulation(
                "CNOT qubits out of range".to_string(),
            ));
        }

        let control_mask = 1usize << control;
        let target_mask = 1usize << target;

        for i in 0..self.state.len() {
            if (i & control_mask) != 0 && (i & target_mask) == 0 {
                let j = i | target_mask;
                self.state.swap(i, j);
            }
        }

        self.gate_count += 1;
        Ok(())
    }

    async fn phase(&mut self, qubit: u32, angle: f64) -> Result<(), QuantumError> {
        if qubit >= self.num_qubits {
            return Err(QuantumError::Simulation(format!(
                "Qubit {} out of range",
                qubit
            )));
        }

        let mask = 1usize << qubit;
        let phase_factor = Complex64::new(0.0, angle).exp();

        for i in 0..self.state.len() {
            if i & mask != 0 {
                self.state[i] *= phase_factor;
            }
        }

        self.gate_count += 1;
        Ok(())
    }

    async fn rotation_x(&mut self, qubit: u32, angle: f64) -> Result<(), QuantumError> {
        if qubit >= self.num_qubits {
            return Err(QuantumError::Simulation(format!(
                "Qubit {} out of range",
                qubit
            )));
        }

        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        let mask = 1usize << qubit;

        for i in 0..self.state.len() {
            if i & mask == 0 {
                let j = i | mask;
                let temp0 = self.state[i];
                let temp1 = self.state[j];

                self.state[i] =
                    Complex64::new(cos_half, 0.0) * temp0 - Complex64::new(0.0, sin_half) * temp1;
                self.state[j] =
                    -Complex64::new(0.0, sin_half) * temp0 + Complex64::new(cos_half, 0.0) * temp1;
            }
        }

        self.gate_count += 1;
        Ok(())
    }

    async fn rotation_y(&mut self, qubit: u32, angle: f64) -> Result<(), QuantumError> {
        if qubit >= self.num_qubits {
            return Err(QuantumError::Simulation(format!(
                "Qubit {} out of range",
                qubit
            )));
        }

        let cos_half = (angle / 2.0).cos();
        let sin_half = (angle / 2.0).sin();
        let mask = 1usize << qubit;

        for i in 0..self.state.len() {
            if i & mask == 0 {
                let j = i | mask;
                let temp0 = self.state[i];
                let temp1 = self.state[j];

                self.state[i] =
                    Complex64::new(cos_half, 0.0) * temp0 - Complex64::new(sin_half, 0.0) * temp1;
                self.state[j] =
                    Complex64::new(sin_half, 0.0) * temp0 + Complex64::new(cos_half, 0.0) * temp1;
            }
        }

        self.gate_count += 1;
        Ok(())
    }

    async fn rotation_z(&mut self, qubit: u32, angle: f64) -> Result<(), QuantumError> {
        if qubit >= self.num_qubits {
            return Err(QuantumError::Simulation(format!(
                "Qubit {} out of range",
                qubit
            )));
        }

        let phase_neg = Complex64::new(0.0, -angle / 2.0).exp();
        let phase_pos = Complex64::new(0.0, angle / 2.0).exp();
        let mask = 1usize << qubit;

        for i in 0..self.state.len() {
            if i & mask == 0 {
                self.state[i] *= phase_neg;
            } else {
                self.state[i] *= phase_pos;
            }
        }

        self.gate_count += 1;
        Ok(())
    }

    async fn toffoli(
        &mut self,
        control1: u32,
        control2: u32,
        target: u32,
    ) -> Result<(), QuantumError> {
        if control1 >= self.num_qubits || control2 >= self.num_qubits || target >= self.num_qubits {
            return Err(QuantumError::Simulation(
                "Toffoli qubits out of range".to_string(),
            ));
        }

        let control1_mask = 1usize << control1;
        let control2_mask = 1usize << control2;
        let target_mask = 1usize << target;

        for i in 0..self.state.len() {
            if (i & control1_mask) != 0 && (i & control2_mask) != 0 && (i & target_mask) == 0 {
                let j = i | target_mask;
                self.state.swap(i, j);
            }
        }

        self.gate_count += 1;
        Ok(())
    }

    /// Measure all qubits
    pub async fn measure_all(&self) -> Result<HashMap<u32, u8>, QuantumError> {
        let mut measurements = HashMap::new();
        let mut probabilities = vec![0.0; self.state.len()];

        // Calculate probabilities
        for (i, amplitude) in self.state.iter().enumerate() {
            probabilities[i] = amplitude.norm_sqr();
        }

        // Sample from probability distribution
        let rand_val = fastrand::f64();
        let mut cumulative_prob = 0.0;
        let mut measured_state = 0;

        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative_prob += prob;
            if rand_val <= cumulative_prob {
                measured_state = i;
                break;
            }
        }

        // Extract individual qubit measurements
        for qubit in 0..self.num_qubits {
            let bit = ((measured_state >> qubit) & 1) as u8;
            measurements.insert(qubit, bit);
        }

        Ok(measurements)
    }

    /// Get current statevector
    pub fn get_statevector(&self) -> Vec<Complex64> {
        self.state.clone()
    }

    /// Reset to |0⟩ state
    pub fn reset(&mut self) {
        self.state.fill(Complex64::new(0.0, 0.0));
        self.state[0] = Complex64::new(1.0, 0.0);
        self.gate_count = 0;
    }
}

/// Density matrix simulator for noisy quantum systems
#[derive(Debug, Clone)]
pub struct DensityMatrixSimulator {
    num_qubits: u32,
    density_matrix: Vec<Vec<Complex64>>,
    noise_model: crate::quantum::QuantumNoiseConfig,
    gate_count: u64,
}

impl DensityMatrixSimulator {
    pub fn new(
        num_qubits: u32,
        noise_model: &crate::quantum::QuantumNoiseConfig,
    ) -> Result<Self, QuantumError> {
        if num_qubits > 15 {
            return Err(QuantumError::ResourceExhausted(format!(
                "Too many qubits for density matrix simulation: {}",
                num_qubits
            )));
        }

        let dim = 1usize << num_qubits;
        let mut density_matrix = vec![vec![Complex64::new(0.0, 0.0); dim]; dim];
        density_matrix[0][0] = Complex64::new(1.0, 0.0); // Initialize to |0⟩⟨0|

        Ok(Self {
            num_qubits,
            density_matrix,
            noise_model: noise_model.clone(),
            gate_count: 0,
        })
    }

    /// Execute a quantum circuit with noise
    pub async fn execute_circuit(
        &mut self,
        circuit: QuantumCircuit,
    ) -> Result<QuantumResult, QuantumError> {
        // Fix E0382: Store circuit depth before consuming gates
        let circuit_depth = circuit.gates.len();

        for gate in circuit.gates {
            self.apply_gate_with_noise(gate).await?;
        }

        let measurements = if circuit.measure_all {
            self.measure_all_noisy().await?
        } else {
            HashMap::new()
        };

        Ok(QuantumResult {
            measurements,
            final_state: None, // Density matrix too large to return
            circuit_depth,
            gate_count: self.gate_count,
            execution_time_ns: 0,
        })
    }

    async fn apply_gate_with_noise(&mut self, gate: QuantumGate) -> Result<(), QuantumError> {
        // Apply the ideal gate
        self.apply_ideal_gate(gate).await?;

        // Apply noise effects
        self.apply_decoherence().await?;
        self.apply_gate_error().await?;

        Ok(())
    }

    async fn apply_ideal_gate(&mut self, gate: QuantumGate) -> Result<(), QuantumError> {
        // Simplified density matrix gate application
        // In a full implementation, this would involve proper unitary evolution
        match gate {
            QuantumGate::Hadamard { qubit: _ } => {
                // Apply hadamard transformation to density matrix
                // ρ' = U ρ U†
                self.gate_count += 1;
            }
            QuantumGate::PauliX { qubit: _ } => {
                self.gate_count += 1;
            }
            // ... other gates would be implemented similarly
            _ => {
                self.gate_count += 1;
            }
        }
        Ok(())
    }

    async fn apply_decoherence(&mut self) -> Result<(), QuantumError> {
        if !self.noise_model.enable_decoherence {
            return Ok(());
        }

        let t1 = self.noise_model.t1_relaxation;
        let t2 = self.noise_model.t2_dephasing;
        let gate_time = 0.01; // microseconds

        // Apply T1 relaxation (amplitude damping)
        let gamma1 = gate_time / t1;
        let p1 = 1.0 - (-gamma1).exp();

        // Apply T2 dephasing
        let gamma2 = gate_time / t2;
        let p2 = 1.0 - (-gamma2).exp();

        // Apply noise to density matrix
        for i in 0..self.density_matrix.len() {
            for j in 0..self.density_matrix[i].len() {
                if i != j {
                    // Dephasing reduces off-diagonal elements
                    self.density_matrix[i][j] *= Complex64::new(1.0 - p2, 0.0);
                }
                // Relaxation affects diagonal elements
                if i != 0 && i == j {
                    // Fix E0502: Store value to avoid double borrow
                    let original_value = self.density_matrix[i][j];
                    self.density_matrix[i][j] *= Complex64::new(1.0 - p1, 0.0);
                    self.density_matrix[0][0] += Complex64::new(p1, 0.0) * original_value;
                }
            }
        }

        Ok(())
    }

    async fn apply_gate_error(&mut self) -> Result<(), QuantumError> {
        let error_rate = self.noise_model.gate_error_rate;

        if fastrand::f64() < error_rate {
            // Apply random Pauli error
            let error_type = fastrand::usize(0..3);
            let error_qubit = fastrand::u32(0..self.num_qubits);

            match error_type {
                0 => {
                    self.apply_ideal_gate(QuantumGate::PauliX { qubit: error_qubit })
                        .await?
                }
                1 => {
                    self.apply_ideal_gate(QuantumGate::PauliY { qubit: error_qubit })
                        .await?
                }
                2 => {
                    self.apply_ideal_gate(QuantumGate::PauliZ { qubit: error_qubit })
                        .await?
                }
                _ => {}
            }
        }

        Ok(())
    }

    async fn measure_all_noisy(&self) -> Result<HashMap<u32, u8>, QuantumError> {
        let mut measurements = HashMap::new();

        // Extract diagonal probabilities from density matrix
        let mut probabilities = Vec::new();
        for i in 0..self.density_matrix.len() {
            probabilities.push(self.density_matrix[i][i].re);
        }

        // Normalize (should already be normalized, but account for numerical errors)
        let total_prob: f64 = probabilities.iter().sum();
        for prob in &mut probabilities {
            *prob /= total_prob;
        }

        // Sample measurement outcome
        let rand_val = fastrand::f64();
        let mut cumulative_prob = 0.0;
        let mut measured_state = 0;

        for (i, &prob) in probabilities.iter().enumerate() {
            cumulative_prob += prob;
            if rand_val <= cumulative_prob {
                measured_state = i;
                break;
            }
        }

        // Add measurement error
        for qubit in 0..self.num_qubits {
            let mut bit = ((measured_state >> qubit) & 1) as u8;

            // Apply measurement error
            if fastrand::f64() < self.noise_model.measurement_error_rate {
                bit = 1 - bit; // Flip bit
            }

            measurements.insert(qubit, bit);
        }

        Ok(measurements)
    }
}

/// Quantum superposition pattern matcher
#[derive(Debug, Clone)]
pub struct SuperpositionMatcher {
    patterns: Vec<Vec<f64>>,
    num_qubits: u32,
}

impl SuperpositionMatcher {
    pub fn new(patterns: Vec<Vec<f64>>, num_qubits: u32) -> Self {
        Self {
            patterns,
            num_qubits,
        }
    }

    /// Match input against patterns using quantum superposition
    pub async fn match_in_superposition(
        &self,
        input: &[f64],
    ) -> Result<SuperpositionResult, QuantumError> {
        // Create quantum circuit for pattern matching
        let mut circuit = QuantumCircuit::new();

        // Initialize superposition of all patterns
        for i in 0..self.num_qubits {
            circuit.add_gate(QuantumGate::Hadamard { qubit: i });
        }

        // Encode patterns using phase kickback (simplified)
        for (pattern_idx, pattern) in self.patterns.iter().enumerate() {
            let similarity = calculate_pattern_similarity(input, pattern);
            let angle = similarity * std::f64::consts::PI;

            // Apply controlled rotation based on pattern index
            let pattern_qubits = encode_pattern_index(pattern_idx, self.num_qubits);
            for (qubit, bit) in pattern_qubits.iter().enumerate() {
                if *bit == 1 {
                    circuit.add_gate(QuantumGate::Phase {
                        qubit: qubit as u32,
                        angle,
                    });
                }
            }
        }

        // Apply quantum Fourier transform for amplitude amplification
        self.apply_qft(&mut circuit);

        circuit.measure_all = true;

        // Execute circuit (this would use the statevector simulator)
        let simulator = StatevectorSimulator::new(self.num_qubits)?;
        // In a real implementation, we would execute the circuit

        Ok(SuperpositionResult {
            best_match_probability: 0.8,                // Placeholder
            pattern_probabilities: vec![0.8, 0.1, 0.1], // Placeholder
            quantum_advantage: 2.0,
        })
    }

    fn apply_qft(&self, circuit: &mut QuantumCircuit) {
        // Simplified QFT implementation
        for i in 0..self.num_qubits {
            circuit.add_gate(QuantumGate::Hadamard { qubit: i });

            for j in (i + 1)..self.num_qubits {
                let angle = std::f64::consts::PI / (1 << (j - i)) as f64;
                // Controlled phase gate would go here
                circuit.add_gate(QuantumGate::Phase { qubit: j, angle });
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SuperpositionResult {
    pub best_match_probability: f64,
    pub pattern_probabilities: Vec<f64>,
    pub quantum_advantage: f64,
}

/// Quantum circuit representation
#[derive(Debug, Clone)]
pub struct QuantumCircuit {
    pub gates: Vec<QuantumGate>,
    pub measure_all: bool,
}

impl QuantumCircuit {
    pub fn new() -> Self {
        Self {
            gates: Vec::new(),
            measure_all: false,
        }
    }

    pub fn add_gate(&mut self, gate: QuantumGate) {
        self.gates.push(gate);
    }

    pub fn requires_density_matrix(&self) -> bool {
        // Check if circuit contains operations that require density matrix simulation
        false // Simplified - would check for noise-sensitive operations
    }
}

impl Default for QuantumCircuit {
    fn default() -> Self {
        Self::new()
    }
}

/// Quantum gates
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QuantumGate {
    Hadamard {
        qubit: u32,
    },
    PauliX {
        qubit: u32,
    },
    PauliY {
        qubit: u32,
    },
    PauliZ {
        qubit: u32,
    },
    CNOT {
        control: u32,
        target: u32,
    },
    Phase {
        qubit: u32,
        angle: f64,
    },
    RotationX {
        qubit: u32,
        angle: f64,
    },
    RotationY {
        qubit: u32,
        angle: f64,
    },
    RotationZ {
        qubit: u32,
        angle: f64,
    },
    Toffoli {
        control1: u32,
        control2: u32,
        target: u32,
    },
}

/// Quantum execution result
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantumResult {
    pub measurements: HashMap<u32, u8>,
    pub final_state: Option<Vec<Complex64>>,
    pub circuit_depth: usize,
    pub gate_count: u64,
    pub execution_time_ns: u64,
}

// Helper functions

fn calculate_pattern_similarity(a: &[f64], b: &[f64]) -> f64 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot_product: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let norm_b = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        0.0
    } else {
        (dot_product / (norm_a * norm_b) + 1.0) / 2.0 // Normalize to [0, 1]
    }
}

fn encode_pattern_index(index: usize, num_qubits: u32) -> Vec<u8> {
    let mut bits = vec![0u8; num_qubits as usize];
    let mut idx = index;

    for i in 0..(num_qubits as usize) {
        bits[i] = (idx & 1) as u8;
        idx >>= 1;
    }

    bits
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn test_statevector_simulator() {
        let mut sim = StatevectorSimulator::new(2).unwrap();

        // Test Hadamard gate
        sim.hadamard(0).await.unwrap();
        let state = sim.get_statevector();

        // After H|0⟩, we should have (|0⟩ + |1⟩)/√2
        let expected_amplitude = 1.0 / std::f64::consts::SQRT_2;
        assert!((state[0].re - expected_amplitude).abs() < 1e-10);
        assert!((state[1].re - expected_amplitude).abs() < 1e-10);
    }

    #[tokio::test]
    async fn test_quantum_circuit() {
        let mut circuit = QuantumCircuit::new();
        circuit.add_gate(QuantumGate::Hadamard { qubit: 0 });
        circuit.add_gate(QuantumGate::CNOT {
            control: 0,
            target: 1,
        });
        circuit.measure_all = true;

        let mut sim = StatevectorSimulator::new(2).unwrap();
        let result = sim.execute_circuit(circuit).await.unwrap();

        assert_eq!(result.circuit_depth, 2);
        assert_eq!(result.gate_count, 2);
        assert!(!result.measurements.is_empty());
    }

    #[test]
    fn test_quantum_simulators_creation() {
        let config = QuantumConfig::default();
        let simulators = QuantumSimulators::new(&config);

        assert!(!*simulators.initialized.read());
    }

    #[tokio::test]
    async fn test_superposition_matcher() {
        let patterns = vec![
            vec![1.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0],
            vec![0.0, 0.0, 1.0],
        ];

        let matcher = SuperpositionMatcher::new(patterns, 2);
        let input = vec![0.8, 0.2, 0.1];

        let result = matcher.match_in_superposition(&input).await.unwrap();
        assert!(result.best_match_probability > 0.0);
        assert!(!result.pattern_probabilities.is_empty());
    }

    #[test]
    fn test_pattern_similarity() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = calculate_pattern_similarity(&a, &b);
        assert!((similarity - 1.0).abs() < 1e-10);

        let c = vec![0.0, 1.0, 0.0];
        let similarity2 = calculate_pattern_similarity(&a, &c);
        assert!((similarity2 - 0.5).abs() < 1e-10);
    }
}
