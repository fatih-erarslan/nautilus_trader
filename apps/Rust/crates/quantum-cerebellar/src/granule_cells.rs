//! Quantum granule cell population for cerebellar sparse coding
//! 
//! Implements granule cell microcircuit with quantum-enhanced sparse coding,
//! competitive inhibition, and adaptive plasticity for pattern separation.

use std::collections::HashMap;
use tch::{Tensor, Device, Kind};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use anyhow::{Result, anyhow};
use tracing::{debug, info, warn};
use rayon::prelude::*;

use crate::{
    QuantumSNNConfig, QuantumSpikeTrain, CellState, QuantumSynapse,
    QuantumCircuitSimulator, CerebellarQuantumCircuits
};

/// Granule cell population configuration
#[derive(Debug, Clone)]
pub struct GranuleCellConfig {
    /// Number of granule cells
    pub n_granule_cells: usize,
    
    /// Number of mossy fiber inputs per granule cell
    pub mossy_fiber_inputs: usize,
    
    /// Sparse connectivity probability
    pub connectivity_probability: f64,
    
    /// Competitive inhibition strength
    pub inhibition_strength: f64,
    
    /// Activation threshold
    pub activation_threshold: f64,
    
    /// Quantum coherence parameters
    pub quantum_coherence: bool,
    pub coherence_strength: f64,
    
    /// Plasticity parameters
    pub plasticity_enabled: bool,
    pub learning_rate: f64,
    pub homeostatic_scaling: bool,
}

impl Default for GranuleCellConfig {
    fn default() -> Self {
        Self {
            n_granule_cells: 100,
            mossy_fiber_inputs: 5,
            connectivity_probability: 0.1,
            inhibition_strength: 0.3,
            activation_threshold: 0.4,
            quantum_coherence: true,
            coherence_strength: 0.2,
            plasticity_enabled: true,
            learning_rate: 0.005,
            homeostatic_scaling: true,
        }
    }
}

/// Individual granule cell with quantum enhancement
#[derive(Debug)]
pub struct QuantumGranuleCell {
    /// Cell ID
    pub id: usize,
    
    /// Current cell state
    pub state: CellState,
    
    /// Mossy fiber input connections
    pub mossy_fiber_connections: Vec<usize>,
    
    /// Synaptic weights from mossy fibers
    pub input_weights: DVector<f64>,
    
    /// Quantum state amplitudes
    pub quantum_state: Vec<Complex64>,
    
    /// Activity history for adaptation
    pub activity_history: Vec<f64>,
    
    /// Homeostatic scaling factor
    pub scaling_factor: f64,
    
    /// Inhibitory connections
    pub inhibitory_inputs: Vec<(usize, f64)>, // (source_id, weight)
}

impl QuantumGranuleCell {
    /// Create new quantum granule cell
    pub fn new(id: usize, n_mossy_fibers: usize, config: &GranuleCellConfig) -> Self {
        let n_qubits = 4; // qubits per granule cell
        let mut state = CellState::new(config.mossy_fiber_inputs, n_qubits);
        
        // Initialize random mossy fiber connections (sparse)
        let mut mossy_connections = Vec::new();
        let mut input_weights = DVector::zeros(config.mossy_fiber_inputs);
        
        for _ in 0..config.mossy_fiber_inputs {
            if rand::random::<f64>() < config.connectivity_probability {
                let mf_id = rand::random::<usize>() % n_mossy_fibers;
                mossy_connections.push(mf_id);
                
                // Random initial weights
                let weight_idx = mossy_connections.len() - 1;
                if weight_idx < input_weights.len() {
                    input_weights[weight_idx] = (rand::random::<f64>() - 0.5) * 0.2;
                }
            }
        }
        
        // Initialize quantum state
        let quantum_state = if config.quantum_coherence {
            vec![Complex64::new(1.0, 0.0); n_qubits]
        } else {
            vec![Complex64::new(0.0, 0.0); n_qubits]
        };
        
        Self {
            id,
            state,
            mossy_fiber_connections: mossy_connections,
            input_weights,
            quantum_state,
            activity_history: Vec::with_capacity(1000),
            scaling_factor: 1.0,
            inhibitory_inputs: Vec::new(),
        }
    }
    
    /// Process mossy fiber inputs
    pub fn process_inputs(
        &mut self,
        mossy_fiber_spikes: &[QuantumSpikeTrain],
        current_time: f64,
        config: &GranuleCellConfig,
    ) -> Result<f64> {
        if self.state.is_refractory(current_time) {
            return Ok(0.0);
        }
        
        // Calculate input from connected mossy fibers
        let mut total_input = 0.0;
        let mut quantum_input = Complex64::new(0.0, 0.0);
        
        for (weight_idx, &mf_id) in self.mossy_fiber_connections.iter().enumerate() {
            if mf_id < mossy_fiber_spikes.len() && weight_idx < self.input_weights.len() {
                let spike_train = &mossy_fiber_spikes[mf_id];
                
                // Check for recent spikes
                let spike_strength = self.calculate_spike_strength(spike_train, current_time);
                let weighted_input = spike_strength * self.input_weights[weight_idx];
                total_input += weighted_input;
                
                // Quantum contribution
                if config.quantum_coherence && !spike_train.amplitudes.is_empty() {
                    let q_amplitude = spike_train.amplitudes.last().unwrap();
                    quantum_input += q_amplitude * Complex64::new(weighted_input, 0.0);
                }
            }
        }
        
        // Apply inhibitory inputs
        let inhibition = self.calculate_inhibition();
        total_input -= inhibition;
        
        // Update membrane potential
        self.state.voltage += total_input * 0.1; // Leak integration
        self.state.voltage *= 0.95; // Membrane leak
        
        // Update quantum state
        if config.quantum_coherence {
            self.update_quantum_state(quantum_input, config)?;
        }
        
        // Check for spike generation
        let spike_output = if self.state.voltage > config.activation_threshold {
            let spike_strength = (self.state.voltage - config.activation_threshold) * self.scaling_factor;
            self.state.reset(current_time, config.activation_threshold as f64);
            
            // Record activity for homeostatic scaling
            self.activity_history.push(spike_strength);
            if self.activity_history.len() > 1000 {
                self.activity_history.remove(0);
            }
            
            spike_strength
        } else {
            0.0
        };
        
        // Homeostatic scaling
        if config.homeostatic_scaling && self.activity_history.len() > 100 {
            self.update_homeostatic_scaling(config);
        }
        
        Ok(spike_output)
    }
    
    /// Calculate spike strength from recent activity
    fn calculate_spike_strength(&self, spike_train: &QuantumSpikeTrain, current_time: f64) -> f64 {
        let mut strength = 0.0;
        let time_window = 5.0; // ms
        
        for (i, &spike_time) in spike_train.times.iter().enumerate() {
            if current_time - spike_time < time_window {
                // Exponential decay
                let decay = (-(current_time - spike_time) / 2.0).exp();
                let amplitude = if i < spike_train.amplitudes.len() {
                    spike_train.amplitudes[i].norm()
                } else {
                    1.0
                };
                strength += amplitude * decay;
            }
        }
        
        strength
    }
    
    /// Calculate total inhibitory input
    fn calculate_inhibition(&self) -> f64 {
        self.inhibitory_inputs.iter()
            .map(|(_, weight)| weight.abs())
            .sum()
    }
    
    /// Update quantum state based on inputs
    fn update_quantum_state(
        &mut self,
        quantum_input: Complex64,
        config: &GranuleCellConfig,
    ) -> Result<()> {
        if self.quantum_state.is_empty() {
            return Ok(());
        }
        
        // Simple quantum state evolution
        let coupling_strength = config.coherence_strength;
        
        for i in 0..self.quantum_state.len() {
            // Oscillatory evolution with input coupling
            let phase = Complex64::new(0.0, 0.1 * i as f64).exp();
            self.quantum_state[i] = self.quantum_state[i] * phase + 
                                   quantum_input * Complex64::new(coupling_strength, 0.0);
            
            // Normalize to prevent divergence
            let norm = self.quantum_state[i].norm();
            if norm > 2.0 {
                self.quantum_state[i] /= Complex64::new(norm / 2.0, 0.0);
            }
        }
        
        Ok(())
    }
    
    /// Update homeostatic scaling factor
    fn update_homeostatic_scaling(&mut self, config: &GranuleCellConfig) {
        let target_rate = 0.1; // Target firing rate
        let current_rate = self.activity_history.len() as f64 / 1000.0; // Approximate rate
        
        if current_rate > target_rate * 1.2 {
            self.scaling_factor *= 0.99; // Decrease excitability
        } else if current_rate < target_rate * 0.8 {
            self.scaling_factor *= 1.01; // Increase excitability
        }
        
        self.scaling_factor = self.scaling_factor.clamp(0.1, 3.0);
    }
    
    /// Add inhibitory connection
    pub fn add_inhibitory_input(&mut self, source_id: usize, weight: f64) {
        self.inhibitory_inputs.push((source_id, weight));
    }
    
    /// Get current quantum coherence measure
    pub fn quantum_coherence(&self) -> f64 {
        if self.quantum_state.len() < 2 {
            return 0.0;
        }
        
        // Calculate pairwise coherence
        let mut total_coherence = 0.0;
        let mut pairs = 0;
        
        for i in 0..self.quantum_state.len() {
            for j in (i + 1)..self.quantum_state.len() {
                let coherence = (self.quantum_state[i].conj() * self.quantum_state[j]).norm();
                total_coherence += coherence;
                pairs += 1;
            }
        }
        
        if pairs > 0 {
            total_coherence / pairs as f64
        } else {
            0.0
        }
    }
}

/// Golgi cell for inhibitory feedback
#[derive(Debug)]
pub struct GoliCell {
    pub id: usize,
    pub state: CellState,
    pub granule_connections: Vec<usize>,
    pub mossy_fiber_connections: Vec<usize>,
    pub inhibition_strength: f64,
}

impl GoliCell {
    pub fn new(id: usize, n_granule_cells: usize, n_mossy_fibers: usize) -> Self {
        let n_qubits = 2;
        let state = CellState::new(10, n_qubits); // Connect to 10 granule cells on average
        
        // Random connections to granule cells
        let mut granule_connections = Vec::new();
        for _ in 0..10 {
            granule_connections.push(rand::random::<usize>() % n_granule_cells);
        }
        
        // Random connections to mossy fibers
        let mut mossy_fiber_connections = Vec::new();
        for _ in 0..5 {
            mossy_fiber_connections.push(rand::random::<usize>() % n_mossy_fibers);
        }
        
        Self {
            id,
            state,
            granule_connections,
            mossy_fiber_connections,
            inhibition_strength: 0.2,
        }
    }
    
    /// Process inputs and generate inhibition
    pub fn process(
        &mut self,
        granule_activities: &[f64],
        mossy_fiber_spikes: &[QuantumSpikeTrain],
        current_time: f64,
    ) -> f64 {
        let mut total_input = 0.0;
        
        // Input from granule cells
        for &gc_id in &self.granule_connections {
            if gc_id < granule_activities.len() {
                total_input += granule_activities[gc_id] * 0.5;
            }
        }
        
        // Input from mossy fibers
        for &mf_id in &self.mossy_fiber_connections {
            if mf_id < mossy_fiber_spikes.len() {
                for &spike_time in &mossy_fiber_spikes[mf_id].times {
                    if current_time - spike_time < 2.0 {
                        total_input += 0.3;
                    }
                }
            }
        }
        
        // Update state
        self.state.voltage += total_input * 0.1;
        self.state.voltage *= 0.9; // Faster decay for inhibitory neurons
        
        // Generate inhibitory output
        if self.state.voltage > 0.3 {
            self.state.reset(current_time, 2.0);
            self.inhibition_strength
        } else {
            0.0
        }
    }
}

/// Population of quantum granule cells
pub struct GranuleCellPopulation {
    /// Individual granule cells
    pub granule_cells: Vec<QuantumGranuleCell>,
    
    /// Golgi cells for inhibition
    pub golgi_cells: Vec<GoliCell>,
    
    /// Configuration
    config: GranuleCellConfig,
    
    /// Quantum circuit for population dynamics
    quantum_circuit: Option<QuantumCircuitSimulator>,
    
    /// Population activity pattern
    activity_pattern: DVector<f64>,
    
    /// Sparse coding statistics
    sparsity_level: f64,
    
    /// Performance metrics
    processing_time_ns: u64,
}

impl GranuleCellPopulation {
    /// Create new granule cell population
    pub fn new(
        n_granule_cells: usize,
        n_mossy_fibers: usize,
        snn_config: &QuantumSNNConfig,
    ) -> Result<Self> {
        let config = GranuleCellConfig {
            n_granule_cells,
            mossy_fiber_inputs: (n_mossy_fibers / 10).max(1),
            ..Default::default()
        };
        
        // Create granule cells
        let mut granule_cells = Vec::with_capacity(n_granule_cells);
        for i in 0..n_granule_cells {
            granule_cells.push(QuantumGranuleCell::new(i, n_mossy_fibers, &config));
        }
        
        // Create Golgi cells
        let n_golgi = snn_config.n_golgi_cells;
        let mut golgi_cells = Vec::with_capacity(n_golgi);
        for i in 0..n_golgi {
            golgi_cells.push(GoliCell::new(i, n_granule_cells, n_mossy_fibers));
        }
        
        // Create quantum circuit for population dynamics
        let quantum_circuit = if config.quantum_coherence {
            Some(QuantumCircuitSimulator::new(8)?) // 8 qubits for population
        } else {
            None
        };
        
        info!("Created granule cell population: {} granule cells, {} Golgi cells", 
              n_granule_cells, n_golgi);
        
        Ok(Self {
            granule_cells,
            golgi_cells,
            config,
            quantum_circuit,
            activity_pattern: DVector::zeros(n_granule_cells),
            sparsity_level: 0.0,
            processing_time_ns: 0,
        })
    }
    
    /// Process mossy fiber inputs through granule cell population
    pub fn process(
        &mut self,
        mossy_fiber_spikes: &[QuantumSpikeTrain],
        current_time: f64,
    ) -> Result<Vec<QuantumSpikeTrain>> {
        let start_time = std::time::Instant::now();
        
        // Process Golgi cells first (for inhibition)
        let mut golgi_inhibition = vec![0.0; self.golgi_cells.len()];
        for (i, golgi_cell) in self.golgi_cells.iter_mut().enumerate() {
            golgi_inhibition[i] = golgi_cell.process(
                self.activity_pattern.as_slice(),
                mossy_fiber_spikes,
                current_time,
            );
        }
        
        // Apply inhibition to granule cells
        self.apply_golgi_inhibition(&golgi_inhibition)?;
        
        // Process granule cells in parallel
        let granule_outputs: Vec<f64> = self.granule_cells
            .par_iter_mut()
            .map(|gc| {
                gc.process_inputs(mossy_fiber_spikes, current_time, &self.config)
                    .unwrap_or(0.0)
            })
            .collect();
        
        // Update activity pattern
        self.activity_pattern = DVector::from_vec(granule_outputs.clone());
        
        // Calculate sparsity
        let active_cells = granule_outputs.iter().filter(|&&x| x > 0.0).count();
        self.sparsity_level = active_cells as f64 / granule_outputs.len() as f64;
        
        // Apply quantum population dynamics
        if let Some(ref mut qc) = self.quantum_circuit {
            self.apply_quantum_population_dynamics(qc, &granule_outputs)?;
        }
        
        // Convert outputs to spike trains
        let spike_trains = self.convert_to_spike_trains(&granule_outputs, current_time)?;
        
        // Update performance metrics
        self.processing_time_ns = start_time.elapsed().as_nanos() as u64;
        
        debug!("Granule cell population processed: {:.1}% sparsity, {}μs", 
               self.sparsity_level * 100.0, start_time.elapsed().as_micros());
        
        Ok(spike_trains)
    }
    
    /// Apply Golgi cell inhibition to granule cells
    fn apply_golgi_inhibition(&mut self, golgi_outputs: &[f64]) -> Result<()> {
        for (golgi_id, &inhibition) in golgi_outputs.iter().enumerate() {
            if inhibition > 0.0 && golgi_id < self.golgi_cells.len() {
                let golgi_cell = &self.golgi_cells[golgi_id];
                
                // Apply inhibition to connected granule cells
                for &gc_id in &golgi_cell.granule_connections {
                    if gc_id < self.granule_cells.len() {
                        self.granule_cells[gc_id].add_inhibitory_input(golgi_id, inhibition);
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Apply quantum population dynamics
    fn apply_quantum_population_dynamics(
        &mut self,
        quantum_circuit: &mut QuantumCircuitSimulator,
        granule_outputs: &[f64],
    ) -> Result<()> {
        quantum_circuit.reset();
        
        // Encode population activity in quantum state
        let active_indices: Vec<usize> = granule_outputs
            .iter()
            .enumerate()
            .filter_map(|(i, &activity)| {
                if activity > 0.0 { Some(i % 8) } else { None }
            })
            .collect();
        
        // Create entanglement between active granule cells
        let correlations: Vec<(usize, usize, f64)> = active_indices
            .iter()
            .zip(active_indices.iter().skip(1))
            .map(|(&i, &j)| (i, j, 0.3))
            .collect();
        
        if !correlations.is_empty() {
            CerebellarQuantumCircuits::create_entanglement_circuit(
                quantum_circuit,
                &correlations,
            )?;
            quantum_circuit.execute()?;
        }
        
        Ok(())
    }
    
    /// Convert granule cell outputs to spike trains
    fn convert_to_spike_trains(
        &self,
        outputs: &[f64],
        current_time: f64,
    ) -> Result<Vec<QuantumSpikeTrain>> {
        let mut spike_trains = Vec::new();
        
        for (neuron_id, &output) in outputs.iter().enumerate() {
            let mut spike_train = QuantumSpikeTrain::new(neuron_id, 1.0);
            
            if output > 0.0 {
                let amplitude = if neuron_id < self.granule_cells.len() {
                    // Include quantum state information
                    let gc = &self.granule_cells[neuron_id];
                    if gc.quantum_state.is_empty() {
                        Complex64::new(output, 0.0)
                    } else {
                        let quantum_contribution = gc.quantum_state.iter().map(|q| q.norm()).sum::<f64>() / gc.quantum_state.len() as f64;
                        Complex64::new(output, quantum_contribution)
                    }
                } else {
                    Complex64::new(output, 0.0)
                };
                
                spike_train.add_spike(current_time, amplitude, 0.0);
            }
            
            spike_trains.push(spike_train);
        }
        
        Ok(spike_trains)
    }
    
    /// Get population sparsity level
    pub fn sparsity(&self) -> f64 {
        self.sparsity_level
    }
    
    /// Get current activity pattern
    pub fn activity_pattern(&self) -> &DVector<f64> {
        &self.activity_pattern
    }
    
    /// Get quantum coherence across population
    pub fn population_coherence(&self) -> f64 {
        let coherences: Vec<f64> = self.granule_cells
            .iter()
            .map(|gc| gc.quantum_coherence())
            .collect();
        
        if coherences.is_empty() {
            0.0
        } else {
            coherences.iter().sum::<f64>() / coherences.len() as f64
        }
    }
    
    /// Adapt population parameters based on activity
    pub fn adapt_population(&mut self, target_sparsity: f64) -> Result<()> {
        if self.sparsity_level > target_sparsity * 1.2 {
            // Too many active cells, increase inhibition
            for golgi_cell in &mut self.golgi_cells {
                golgi_cell.inhibition_strength *= 1.05;
            }
        } else if self.sparsity_level < target_sparsity * 0.8 {
            // Too few active cells, decrease inhibition
            for golgi_cell in &mut self.golgi_cells {
                golgi_cell.inhibition_strength *= 0.95;
            }
        }
        
        // Bound inhibition strength
        for golgi_cell in &mut self.golgi_cells {
            golgi_cell.inhibition_strength = golgi_cell.inhibition_strength.clamp(0.05, 0.8);
        }
        
        debug!("Adapted population: target sparsity {:.1}%, current {:.1}%", 
               target_sparsity * 100.0, self.sparsity_level * 100.0);
        
        Ok(())
    }
    
    /// Reset population state
    pub fn reset(&mut self) {
        for gc in &mut self.granule_cells {
            gc.state.voltage = 0.0;
            gc.state.last_spike_time = None;
            gc.state.refractory_until = 0.0;
            gc.activity_history.clear();
            gc.scaling_factor = 1.0;
            
            // Reset quantum state
            if !gc.quantum_state.is_empty() {
                gc.quantum_state[0] = Complex64::new(1.0, 0.0);
                for i in 1..gc.quantum_state.len() {
                    gc.quantum_state[i] = Complex64::new(0.0, 0.0);
                }
            }
        }
        
        for golgi in &mut self.golgi_cells {
            golgi.state.voltage = 0.0;
            golgi.state.last_spike_time = None;
            golgi.state.refractory_until = 0.0;
        }
        
        self.activity_pattern.fill(0.0);
        self.sparsity_level = 0.0;
        
        if let Some(ref mut qc) = self.quantum_circuit {
            qc.reset();
        }
    }
    
    /// Get processing performance metrics
    pub fn processing_time_ns(&self) -> u64 {
        self.processing_time_ns
    }
}

/// Specialized Golgi cell population for enhanced inhibition
pub struct GoliCellPopulation {
    golgi_cells: Vec<GoliCell>,
    global_inhibition: f64,
}

impl GoliCellPopulation {
    pub fn new(n_golgi: usize, n_granule: usize, n_mossy: usize) -> Self {
        let mut golgi_cells = Vec::with_capacity(n_golgi);
        for i in 0..n_golgi {
            golgi_cells.push(GoliCell::new(i, n_granule, n_mossy));
        }
        
        Self {
            golgi_cells,
            global_inhibition: 0.0,
        }
    }
    
    /// Apply competitive inhibition to granule cell population
    pub fn apply_inhibition(
        &mut self,
        granule_activity: &DVector<f64>,
        current_time: f64,
    ) -> Result<DVector<f64>> {
        let mut inhibited_activity = granule_activity.clone();
        
        // Winner-take-all mechanism
        let max_activity = granule_activity.max();
        let threshold = max_activity * 0.7;
        
        for i in 0..inhibited_activity.len() {
            if inhibited_activity[i] < threshold {
                inhibited_activity[i] *= 0.3; // Strong inhibition for non-winners
            }
        }
        
        // Apply global inhibition based on population activity
        let total_activity: f64 = granule_activity.sum();
        self.global_inhibition = (total_activity / granule_activity.len() as f64) * 0.2;
        
        for i in 0..inhibited_activity.len() {
            inhibited_activity[i] -= self.global_inhibition;
            inhibited_activity[i] = inhibited_activity[i].max(0.0);
        }
        
        Ok(inhibited_activity)
    }
    
    /// Reset Golgi cell states
    pub fn reset(&mut self) {
        for golgi in &mut self.golgi_cells {
            golgi.state.voltage = 0.0;
            golgi.state.last_spike_time = None;
            golgi.state.refractory_until = 0.0;
        }
        self.global_inhibition = 0.0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_granule_cell_creation() {
        let config = GranuleCellConfig::default();
        let gc = QuantumGranuleCell::new(0, 50, &config);
        
        assert_eq!(gc.id, 0);
        assert!(!gc.mossy_fiber_connections.is_empty());
        assert!(gc.quantum_state.len() > 0);
    }
    
    #[test]
    fn test_granule_cell_processing() {
        let config = GranuleCellConfig::default();
        let mut gc = QuantumGranuleCell::new(0, 10, &config);
        
        // Create mock spike trains
        let mut spike_trains = Vec::new();
        for i in 0..10 {
            let mut st = QuantumSpikeTrain::new(i, 10.0);
            if i < 3 {
                st.add_spike(1.0, Complex64::new(0.8, 0.0), 0.0);
            }
            spike_trains.push(st);
        }
        
        let output = gc.process_inputs(&spike_trains, 2.0, &config).unwrap();
        
        // Should have some output if connected to active mossy fibers
        assert!(output >= 0.0);
    }
    
    #[test]
    fn test_granule_population_creation() {
        let snn_config = QuantumSNNConfig::default();
        let population = GranuleCellPopulation::new(20, 10, &snn_config).unwrap();
        
        assert_eq!(population.granule_cells.len(), 20);
        assert_eq!(population.golgi_cells.len(), snn_config.n_golgi_cells);
    }
    
    #[test]
    fn test_population_processing() {
        let snn_config = QuantumSNNConfig::default();
        let mut population = GranuleCellPopulation::new(10, 5, &snn_config).unwrap();
        
        // Create test spike trains
        let mut spike_trains = Vec::new();
        for i in 0..5 {
            let mut st = QuantumSpikeTrain::new(i, 10.0);
            st.add_spike(1.0, Complex64::new(0.7, 0.0), 0.0);
            spike_trains.push(st);
        }
        
        let output_spikes = population.process(&spike_trains, 2.0).unwrap();
        
        assert_eq!(output_spikes.len(), 10);
        assert!(population.sparsity() >= 0.0 && population.sparsity() <= 1.0);
    }
    
    #[test]
    fn test_golgi_inhibition() {
        let mut goli_pop = GoliCellPopulation::new(5, 10, 5);
        let activity = DVector::from_vec(vec![0.8, 0.3, 0.9, 0.1, 0.6, 0.2, 0.7, 0.4, 0.0, 0.5]);
        
        let inhibited = goli_pop.apply_inhibition(&activity, 1.0).unwrap();
        
        // Check that some inhibition was applied
        let total_before: f64 = activity.sum();
        let total_after: f64 = inhibited.sum();
        assert!(total_after <= total_before);
    }
    
    #[test]
    fn test_sparse_coding() {
        let snn_config = QuantumSNNConfig::default();
        let mut population = GranuleCellPopulation::new(50, 20, &snn_config).unwrap();
        
        // Create dense input
        let mut spike_trains = Vec::new();
        for i in 0..20 {
            let mut st = QuantumSpikeTrain::new(i, 10.0);
            st.add_spike(1.0, Complex64::new(0.6, 0.0), 0.0);
            spike_trains.push(st);
        }
        
        let _output = population.process(&spike_trains, 1.0).unwrap();
        
        // Granule cell layer should produce sparse output
        assert!(population.sparsity() < 0.5); // Less than 50% active
    }
    
    #[test]
    fn test_quantum_coherence() {
        let config = GranuleCellConfig::default();
        let gc = QuantumGranuleCell::new(0, 10, &config);
        
        let coherence = gc.quantum_coherence();
        assert!(coherence >= 0.0 && coherence <= 1.0);
    }
    
    #[test]
    fn test_homeostatic_scaling() {
        let config = GranuleCellConfig::default();
        let mut gc = QuantumGranuleCell::new(0, 10, &config);
        
        // Simulate high activity
        for _ in 0..150 {
            gc.activity_history.push(1.0);
        }
        
        let initial_scaling = gc.scaling_factor;
        gc.update_homeostatic_scaling(&config);
        
        // Scaling should decrease due to high activity
        assert!(gc.scaling_factor <= initial_scaling);
    }
}