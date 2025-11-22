//! Classical simulation of quantum circuits
//!
//! This module provides efficient classical simulation of quantum circuits
//! with support for state vector simulation and measurement.

use crate::{Complex, StateVector, Operator, Result, QuantumError, Circuit};
use ndarray::Array1;
use rand::Rng;
// Removed unused rayon import
// Removed unused serde imports
use std::collections::HashMap;

/// Quantum circuit simulator
#[derive(Debug)]
pub struct Simulator {
    /// Number of qubits
    n_qubits: usize,
    /// Current quantum state
    state: StateVector,
    /// Measurement results cache
    measurement_cache: HashMap<String, Vec<u8>>,
    /// Random number generator
    rng: rand::rngs::ThreadRng,
}

impl Simulator {
    /// Create a new quantum simulator
    pub fn new(n_qubits: usize) -> Self {
        if n_qubits > 20 {
            panic!("Simulator limited to 20 qubits for memory efficiency");
        }
        
        let dim = 1 << n_qubits;
        let mut state = Array1::zeros(dim);
        state[0] = Complex::new(1.0, 0.0); // Start in |0...0⟩
        
        Self {
            n_qubits,
            state,
            measurement_cache: HashMap::new(),
            rng: rand::thread_rng(),
        }
    }
    
    /// Reset simulator to |0...0⟩ state
    pub fn reset(&mut self) {
        self.state.fill(Complex::new(0.0, 0.0));
        self.state[0] = Complex::new(1.0, 0.0);
        self.measurement_cache.clear();
    }
    
    /// Set the quantum state
    pub fn set_state(&mut self, state: StateVector) -> Result<()> {
        if state.len() != (1 << self.n_qubits) {
            return Err(QuantumError::DimensionMismatch {
                expected: 1 << self.n_qubits,
                actual: state.len(),
            });
        }
        
        // Verify normalization
        let norm_sqr: f64 = state.iter().map(|c| c.norm_sqr()).sum();
        if (norm_sqr - 1.0).abs() > 1e-10 {
            return Err(QuantumError::InvalidState);
        }
        
        self.state = state;
        Ok(())
    }
    
    /// Get the current quantum state (read-only)
    pub fn state(&self) -> &StateVector {
        &self.state
    }
    
    /// Get a mutable reference to the quantum state
    pub fn state_mut(&mut self) -> &mut StateVector {
        &mut self.state
    }
    
    /// Execute a quantum circuit
    pub fn execute_circuit(&mut self, circuit: &Circuit) -> Result<()> {
        if circuit.n_qubits != self.n_qubits {
            return Err(QuantumError::DimensionMismatch {
                expected: self.n_qubits,
                actual: circuit.n_qubits,
            });
        }
        
        let final_state = circuit.execute()?;
        self.state = final_state;
        Ok(())
    }
    
    /// Execute a quantum circuit with custom parameters
    pub fn execute_circuit_with_parameters(
        &mut self,
        circuit: &Circuit,
        params: &[f64],
    ) -> Result<()> {
        if circuit.n_qubits != self.n_qubits {
            return Err(QuantumError::DimensionMismatch {
                expected: self.n_qubits,
                actual: circuit.n_qubits,
            });
        }
        
        let final_state = circuit.execute_with_parameters(params)?;
        self.state = final_state;
        Ok(())
    }
    
    /// Measure a single qubit and collapse the state
    pub fn measure_qubit(&mut self, qubit: usize) -> Result<u8> {
        if qubit >= self.n_qubits {
            return Err(QuantumError::InvalidQubit(qubit));
        }
        
        // Calculate probability of measuring |1⟩
        let prob_one = self.calculate_measurement_probability(qubit)?;
        
        // Perform measurement
        let measurement = if self.rng.gen::<f64>() < prob_one {
            1
        } else {
            0
        };
        
        // Collapse the state
        self.collapse_state(qubit, measurement)?;
        
        Ok(measurement)
    }
    
    /// Measure all qubits and return the classical bit string
    pub fn measure_all(&mut self) -> Result<Vec<u8>> {
        let mut measurements = Vec::with_capacity(self.n_qubits);
        
        for qubit in 0..self.n_qubits {
            let measurement = self.measure_qubit(qubit)?;
            measurements.push(measurement);
        }
        
        Ok(measurements)
    }
    
    /// Sample measurements without collapsing the state
    pub fn sample_measurements(&self, shots: usize) -> Result<Vec<Vec<u8>>> {
        let mut samples = Vec::with_capacity(shots);
        let probabilities = self.get_measurement_probabilities();
        
        let mut rng = rand::thread_rng();
        
        for _ in 0..shots {
            let random_val: f64 = rng.gen();
            let mut cumulative_prob = 0.0;
            
            for (state_idx, &prob) in probabilities.iter().enumerate() {
                cumulative_prob += prob;
                if random_val < cumulative_prob {
                    let bitstring = self.state_index_to_bitstring(state_idx);
                    samples.push(bitstring);
                    break;
                }
            }
        }
        
        Ok(samples)
    }
    
    /// Calculate expectation value of an observable
    pub fn expectation_value(&self, observable: &Operator) -> Result<f64> {
        crate::utils::expectation_value(&self.state, observable).map(|c| c.re)
    }
    
    /// Calculate measurement probability for a qubit
    fn calculate_measurement_probability(&self, qubit: usize) -> Result<f64> {
        let qubit_mask = 1 << qubit;
        let mut prob_one = 0.0;
        
        for (i, &amplitude) in self.state.iter().enumerate() {
            if (i & qubit_mask) != 0 {
                prob_one += amplitude.norm_sqr();
            }
        }
        
        Ok(prob_one)
    }
    
    /// Collapse the state after measurement
    fn collapse_state(&mut self, qubit: usize, measurement: u8) -> Result<()> {
        let qubit_mask = 1 << qubit;
        let mut norm = 0.0;
        
        // Zero out amplitudes inconsistent with measurement and calculate normalization
        for i in 0..self.state.len() {
            let bit = ((i & qubit_mask) >> qubit) as u8;
            if bit != measurement {
                self.state[i] = Complex::new(0.0, 0.0);
            } else {
                norm += self.state[i].norm_sqr();
            }
        }
        
        // Normalize
        if norm > 0.0 {
            let norm_sqrt = norm.sqrt();
            for i in 0..self.state.len() {
                let bit = ((i & qubit_mask) >> qubit) as u8;
                if bit == measurement {
                    self.state[i] /= Complex::new(norm_sqrt, 0.0);
                }
            }
        } else {
            return Err(QuantumError::InvalidState);
        }
        
        Ok(())
    }
    
    /// Get probability distribution over all computational basis states
    fn get_measurement_probabilities(&self) -> Vec<f64> {
        self.state.iter().map(|c| c.norm_sqr()).collect()
    }
    
    /// Convert state index to bitstring
    fn state_index_to_bitstring(&self, index: usize) -> Vec<u8> {
        (0..self.n_qubits)
            .map(|i| ((index >> i) & 1) as u8)
            .collect()
    }
}

/// State evolution tracking for debugging and analysis
#[derive(Debug, Clone)]
pub struct StateEvolution {
    /// Evolution steps (state after each gate)
    pub steps: Vec<StateVector>,
    /// Gate names corresponding to each step
    pub gate_names: Vec<String>,
    /// Timestamps for each step (not serializable)
    pub timestamps: Vec<std::time::Instant>,
}

impl StateEvolution {
    /// Create a new state evolution tracker
    pub fn new() -> Self {
        Self {
            steps: Vec::new(),
            gate_names: Vec::new(),
            timestamps: Vec::new(),
        }
    }
    
    /// Add a step to the evolution
    pub fn add_step(&mut self, state: StateVector, gate_name: String) {
        self.steps.push(state);
        self.gate_names.push(gate_name);
        self.timestamps.push(std::time::Instant::now());
    }
    
    /// Get the number of steps
    pub fn len(&self) -> usize {
        self.steps.len()
    }
    
    /// Check if evolution is empty
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }
    
    /// Get state at a specific step
    pub fn state_at_step(&self, step: usize) -> Option<&StateVector> {
        self.steps.get(step)
    }
    
    /// Calculate fidelity between consecutive steps
    pub fn step_fidelities(&self) -> Result<Vec<f64>> {
        let mut fidelities = Vec::with_capacity(self.steps.len().saturating_sub(1));
        
        for i in 1..self.steps.len() {
            let fidelity = crate::utils::fidelity(&self.steps[i - 1], &self.steps[i])?;
            fidelities.push(fidelity);
        }
        
        Ok(fidelities)
    }
}

/// Debugging simulator that tracks state evolution
pub struct DebuggingSimulator {
    simulator: Simulator,
    evolution: StateEvolution,
    track_evolution: bool,
}

impl DebuggingSimulator {
    /// Create a new debugging simulator
    pub fn new(n_qubits: usize) -> Self {
        Self {
            simulator: Simulator::new(n_qubits),
            evolution: StateEvolution::new(),
            track_evolution: true,
        }
    }
    
    /// Enable or disable evolution tracking
    pub fn set_tracking(&mut self, enabled: bool) {
        self.track_evolution = enabled;
        if !enabled {
            self.evolution = StateEvolution::new();
        }
    }
    
    /// Execute circuit with step-by-step tracking
    pub fn execute_circuit_debug(&mut self, circuit: &Circuit) -> Result<()> {
        if self.track_evolution {
            self.evolution.add_step(self.simulator.state.clone(), "Initial".to_string());
        }
        
        // This is a simplified version - in practice, we'd need to modify Circuit
        // to allow step-by-step execution
        self.simulator.execute_circuit(circuit)?;
        
        if self.track_evolution {
            self.evolution.add_step(self.simulator.state.clone(), "Final".to_string());
        }
        
        Ok(())
    }
    
    /// Get the underlying simulator
    pub fn simulator(&self) -> &Simulator {
        &self.simulator
    }
    
    /// Get mutable reference to the underlying simulator
    pub fn simulator_mut(&mut self) -> &mut Simulator {
        &mut self.simulator
    }
    
    /// Get the state evolution
    pub fn evolution(&self) -> &StateEvolution {
        &self.evolution
    }
}

/// Efficient parallel simulator for multiple circuits
pub struct BatchSimulator {
    _n_qubits: usize,
    simulators: Vec<Simulator>,
}

impl BatchSimulator {
    /// Create a new batch simulator
    pub fn new(n_qubits: usize, batch_size: usize) -> Self {
        let simulators: Vec<_> = (0..batch_size)
            .map(|_| Simulator::new(n_qubits))
            .collect();
            
        Self {
            _n_qubits: n_qubits,
            simulators,
        }
    }
    
    /// Execute multiple circuits in parallel
    pub fn execute_batch(&mut self, circuits: Vec<&Circuit>) -> Result<Vec<StateVector>> {
        if circuits.len() > self.simulators.len() {
            return Err(QuantumError::InvalidParameter(
                format!("Too many circuits: {} > {}", circuits.len(), self.simulators.len())
            ));
        }
        
        // Use sequential iteration instead of parallel due to Send constraints
        let mut results = Vec::with_capacity(circuits.len());
        for (circuit, simulator) in circuits.into_iter().zip(self.simulators.iter_mut()) {
            simulator.reset();
            simulator.execute_circuit(circuit)?;
            results.push(simulator.state().clone());
        }
        Ok(results)
    }
    
    /// Execute circuits with different parameters in parallel
    pub fn execute_batch_with_parameters(
        &mut self,
        circuit: &Circuit,
        parameter_sets: Vec<&[f64]>,
    ) -> Result<Vec<StateVector>> {
        if parameter_sets.len() > self.simulators.len() {
            return Err(QuantumError::InvalidParameter(
                format!("Too many parameter sets: {} > {}", parameter_sets.len(), self.simulators.len())
            ));
        }
        
        // Use sequential iteration instead of parallel due to Send constraints
        let mut results = Vec::with_capacity(parameter_sets.len());
        for (params, simulator) in parameter_sets.into_iter().zip(self.simulators.iter_mut()) {
            simulator.reset();
            simulator.execute_circuit_with_parameters(circuit, params)?;
            results.push(simulator.state().clone());
        }
        Ok(results)
    }
    
    /// Sample measurements from multiple circuits in parallel
    pub fn sample_batch(
        &mut self,
        circuits: Vec<&Circuit>,
        shots_per_circuit: usize,
    ) -> Result<Vec<Vec<Vec<u8>>>> {
        if circuits.len() > self.simulators.len() {
            return Err(QuantumError::InvalidParameter(
                format!("Too many circuits: {} > {}", circuits.len(), self.simulators.len())
            ));
        }
        
        // Use sequential iteration instead of parallel due to Send constraints
        let mut results = Vec::with_capacity(circuits.len());
        for (circuit, simulator) in circuits.into_iter().zip(self.simulators.iter_mut()) {
            simulator.reset();
            simulator.execute_circuit(circuit)?;
            results.push(simulator.sample_measurements(shots_per_circuit)?);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{Circuit, gates::*, constants};
    use approx::assert_abs_diff_eq;
    
    #[test]
    fn test_simulator_creation() {
        let sim = Simulator::new(2);
        assert_eq!(sim.n_qubits, 2);
        assert_eq!(sim.state().len(), 4);
        assert_abs_diff_eq!(sim.state()[0].re, 1.0);
    }
    
    #[test]
    fn test_circuit_execution() {
        let mut sim = Simulator::new(1);
        let mut circuit = Circuit::new(1);
        circuit.add_gate(Box::new(Hadamard::new(0))).unwrap();
        
        sim.execute_circuit(&circuit).unwrap();
        
        let state = sim.state();
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        assert_abs_diff_eq!(state[0].re, sqrt2_inv, epsilon = 1e-10);
        assert_abs_diff_eq!(state[1].re, sqrt2_inv, epsilon = 1e-10);
    }
    
    #[test]
    fn test_measurement() {
        let mut sim = Simulator::new(1);
        sim.set_state(constants::plus_state()).unwrap();
        
        // Measure many times to check probability distribution
        let mut zero_count = 0;
        let mut one_count = 0;
        
        for _ in 0..1000 {
            sim.set_state(constants::plus_state()).unwrap();
            let measurement = sim.measure_qubit(0).unwrap();
            if measurement == 0 {
                zero_count += 1;
            } else {
                one_count += 1;
            }
        }
        
        // Should be approximately 50/50
        assert!((zero_count as f64 - 500.0).abs() < 100.0);
        assert!((one_count as f64 - 500.0).abs() < 100.0);
    }
    
    #[test]
    fn test_expectation_value() {
        let mut sim = Simulator::new(1);
        let mut circuit = Circuit::new(1);
        circuit.add_gate(Box::new(PauliX::new(0))).unwrap();
        
        sim.execute_circuit(&circuit).unwrap();
        
        let pauli_z = constants::pauli_z();
        let expectation = sim.expectation_value(&pauli_z).unwrap();
        
        assert_abs_diff_eq!(expectation, -1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_sampling() {
        let mut sim = Simulator::new(2);
        sim.set_state(crate::utils::random_state(2));
        
        let samples = sim.sample_measurements(100).unwrap();
        assert_eq!(samples.len(), 100);
        assert!(samples.iter().all(|s| s.len() == 2));
    }
    
    #[test]
    fn test_batch_simulator() {
        let mut batch_sim = BatchSimulator::new(1, 2);
        
        let mut circuit1 = Circuit::new(1);
        circuit1.add_gate(Box::new(Hadamard::new(0))).unwrap();
        
        let mut circuit2 = Circuit::new(1);
        circuit2.add_gate(Box::new(PauliX::new(0))).unwrap();
        
        let results = batch_sim.execute_batch(vec![&circuit1, &circuit2]).unwrap();
        assert_eq!(results.len(), 2);
        
        // First result should be |+⟩ state
        let sqrt2_inv = 1.0 / 2.0_f64.sqrt();
        assert_abs_diff_eq!(results[0][0].re, sqrt2_inv, epsilon = 1e-10);
        assert_abs_diff_eq!(results[0][1].re, sqrt2_inv, epsilon = 1e-10);
        
        // Second result should be |1⟩ state
        assert_abs_diff_eq!(results[1][0].re, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(results[1][1].re, 1.0, epsilon = 1e-10);
    }
    
    #[test]
    fn test_state_evolution() {
        let mut evolution = StateEvolution::new();
        
        let state1 = constants::zero_state();
        let state2 = constants::plus_state();
        
        evolution.add_step(state1, "Initial".to_string());
        evolution.add_step(state2, "Hadamard".to_string());
        
        assert_eq!(evolution.len(), 2);
        assert_eq!(evolution.gate_names[0], "Initial");
        assert_eq!(evolution.gate_names[1], "Hadamard");
        
        let fidelities = evolution.step_fidelities().unwrap();
        assert_eq!(fidelities.len(), 1);
    }
}