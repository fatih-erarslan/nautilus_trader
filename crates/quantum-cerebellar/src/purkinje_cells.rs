//! Quantum Purkinje cell population for cerebellar motor learning
//! 
//! Implements sophisticated Purkinje cell dynamics with parallel fiber integration,
//! climbing fiber modulation, complex spike generation, and quantum-enhanced plasticity.

use std::collections::{HashMap, VecDeque};
use tch::{Tensor, Device, Kind};
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use anyhow::{Result, anyhow};
use tracing::{debug, info, warn};
use serde::{Serialize, Deserialize};

use crate::{
    QuantumSNNConfig, QuantumSpikeTrain, CellState, QuantumSynapse,
    QuantumCircuitSimulator, CerebellarQuantumCircuits
};

/// Purkinje cell configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PurkinjeCellConfig {
    /// Number of parallel fiber inputs
    pub n_parallel_fibers: usize,
    
    /// Dendritic tree complexity
    pub dendritic_branches: usize,
    
    /// Simple spike threshold
    pub simple_spike_threshold: f64,
    
    /// Complex spike threshold
    pub complex_spike_threshold: f64,
    
    /// Parallel fiber weight decay
    pub pf_weight_decay: f64,
    
    /// Climbing fiber strength
    pub cf_strength: f64,
    
    /// Intrinsic plasticity parameters
    pub intrinsic_plasticity: bool,
    pub plasticity_rate: f64,
    
    /// Quantum enhancement
    pub quantum_dendritic_integration: bool,
    pub quantum_plasticity: bool,
    pub coherence_window: f64,
}

impl Default for PurkinjeCellConfig {
    fn default() -> Self {
        Self {
            n_parallel_fibers: 100,
            dendritic_branches: 10,
            simple_spike_threshold: 0.6,
            complex_spike_threshold: 0.8,
            pf_weight_decay: 0.001,
            cf_strength: 2.0,
            intrinsic_plasticity: true,
            plasticity_rate: 0.01,
            quantum_dendritic_integration: true,
            quantum_plasticity: true,
            coherence_window: 50.0,
        }
    }
}

/// Dendritic compartment for spatial integration
#[derive(Debug, Clone)]
pub struct DendriticCompartment {
    /// Compartment ID
    pub id: usize,
    
    /// Membrane potential
    pub voltage: f64,
    
    /// Calcium concentration
    pub calcium: f64,
    
    /// Parallel fiber synapses
    pub pf_synapses: Vec<ParallelFiberSynapse>,
    
    /// Quantum state for dendritic integration
    pub quantum_state: Vec<Complex64>,
    
    /// Plasticity state variables
    pub ltp_tag: f64,
    pub ltd_tag: f64,
}

impl DendriticCompartment {
    pub fn new(id: usize, n_pf_inputs: usize) -> Self {
        let mut pf_synapses = Vec::with_capacity(n_pf_inputs);
        for i in 0..n_pf_inputs {
            pf_synapses.push(ParallelFiberSynapse::new(i, id));
        }
        
        Self {
            id,
            voltage: 0.0,
            calcium: 0.0,
            pf_synapses,
            quantum_state: vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 0.0)],
            ltp_tag: 0.0,
            ltd_tag: 0.0,
        }
    }
    
    /// Integrate parallel fiber inputs
    pub fn integrate_pf_inputs(
        &mut self,
        pf_activity: &[f64],
        config: &PurkinjeCellConfig,
    ) -> f64 {
        let mut total_input = 0.0;
        let mut quantum_input = Complex64::new(0.0, 0.0);
        
        // Process each parallel fiber synapse
        for synapse in &mut self.pf_synapses {
            if synapse.pf_id < pf_activity.len() {
                let pf_strength = pf_activity[synapse.pf_id];
                let synaptic_current = synapse.process_input(pf_strength);
                total_input += synaptic_current;
                
                // Quantum contribution
                if config.quantum_dendritic_integration {
                    quantum_input += Complex64::new(synaptic_current, 0.0) * 
                                   Complex64::new(0.0, synapse.weight).exp();
                }
            }
        }
        
        // Update dendritic voltage
        self.voltage += total_input * 0.1;
        self.voltage *= 0.95; // Membrane leak
        
        // Update calcium concentration
        if total_input > 0.1 {
            self.calcium += total_input * 0.05;
        }
        self.calcium *= 0.98; // Calcium decay
        
        // Quantum dendritic integration
        if config.quantum_dendritic_integration {
            self.update_quantum_state(quantum_input);
        }
        
        self.voltage
    }
    
    /// Update quantum state for dendritic integration
    fn update_quantum_state(&mut self, input: Complex64) {
        if self.quantum_state.len() >= 2 {
            // Quantum oscillator dynamics
            let phase = Complex64::new(0.0, 0.1).exp();
            self.quantum_state[0] = self.quantum_state[0] * phase + input * Complex64::new(0.1, 0.0);
            self.quantum_state[1] = self.quantum_state[1] * phase.conj() + input.conj() * Complex64::new(0.1, 0.0);
            
            // Normalize
            let norm = (self.quantum_state[0].norm_sqr() + self.quantum_state[1].norm_sqr()).sqrt();
            if norm > 1e-10 {
                self.quantum_state[0] /= Complex64::new(norm, 0.0);
                self.quantum_state[1] /= Complex64::new(norm, 0.0);
            }
        }
    }
    
    /// Apply climbing fiber modulation
    pub fn climbing_fiber_modulation(&mut self, cf_strength: f64, config: &PurkinjeCellConfig) {
        // Calcium spike from climbing fiber
        self.calcium += cf_strength * config.cf_strength;
        
        // Modulate all parallel fiber synapses
        for synapse in &mut self.pf_synapses {
            synapse.apply_cf_modulation(cf_strength, self.calcium, config);
        }
        
        // Set plasticity tags
        if cf_strength > 0.5 {
            self.ltp_tag = 1.0;
            self.ltd_tag = 0.0;
        } else if cf_strength < -0.5 {
            self.ltp_tag = 0.0;
            self.ltd_tag = 1.0;
        }
    }
}

/// Parallel fiber synapse with quantum plasticity
#[derive(Debug, Clone)]
pub struct ParallelFiberSynapse {
    /// Parallel fiber ID
    pub pf_id: usize,
    
    /// Target dendritic compartment
    pub dendrite_id: usize,
    
    /// Synaptic weight
    pub weight: f64,
    
    /// Plasticity state
    pub plasticity_trace: f64,
    
    /// Quantum coupling strength
    pub quantum_coupling: Complex64,
    
    /// Recent activity trace
    pub activity_trace: f64,
    
    /// Eligibility trace for plasticity
    pub eligibility: f64,
}

impl ParallelFiberSynapse {
    pub fn new(pf_id: usize, dendrite_id: usize) -> Self {
        Self {
            pf_id,
            dendrite_id,
            weight: (rand::random::<f64>() - 0.5) * 0.1,
            plasticity_trace: 0.0,
            quantum_coupling: Complex64::new(0.1, 0.0),
            activity_trace: 0.0,
            eligibility: 0.0,
        }
    }
    
    /// Process parallel fiber input
    pub fn process_input(&mut self, pf_activity: f64) -> f64 {
        // Update activity trace
        self.activity_trace = self.activity_trace * 0.9 + pf_activity;
        
        // Update eligibility trace
        if pf_activity > 0.1 {
            self.eligibility = 1.0;
        } else {
            self.eligibility *= 0.95;
        }
        
        // Synaptic transmission
        pf_activity * self.weight
    }
    
    /// Apply climbing fiber modulation for plasticity
    pub fn apply_cf_modulation(
        &mut self,
        cf_strength: f64,
        calcium: f64,
        config: &PurkinjeCellConfig,
    ) {
        if self.eligibility > 0.1 && cf_strength.abs() > 0.1 {
            // LTD (Long-term depression) - characteristic of cerebellar learning
            let ltd_strength = cf_strength * self.eligibility * calcium * config.plasticity_rate;
            self.weight -= ltd_strength;
            
            // LTP component for recent activity
            if self.activity_trace > 0.5 {
                let ltp_strength = self.activity_trace * config.plasticity_rate * 0.1;
                self.weight += ltp_strength;
            }
            
            // Quantum plasticity modulation
            if config.quantum_plasticity {
                let quantum_modulation = self.quantum_coupling.norm() * 0.1;
                self.weight += quantum_modulation * cf_strength.signum();
            }
        }
        
        // Weight bounds
        self.weight = self.weight.clamp(-1.0, 1.0);
        
        // Update quantum coupling
        if config.quantum_plasticity {
            let phase_shift = cf_strength * std::f64::consts::PI / 4.0;
            self.quantum_coupling *= Complex64::new(0.0, phase_shift).exp();
        }
    }
}

/// Climbing fiber input for error signaling
#[derive(Debug, Clone)]
pub struct ClimbingFiberInput {
    /// Source inferior olive neuron
    pub source_id: usize,
    
    /// Current error signal
    pub error_signal: f64,
    
    /// Complex spike generation
    pub complex_spike_pending: bool,
    
    /// Teaching signal strength
    pub teaching_strength: f64,
    
    /// Quantum error encoding
    pub quantum_error_state: Complex64,
}

impl ClimbingFiberInput {
    pub fn new(source_id: usize) -> Self {
        Self {
            source_id,
            error_signal: 0.0,
            complex_spike_pending: false,
            teaching_strength: 1.0,
            quantum_error_state: Complex64::new(0.0, 0.0),
        }
    }
    
    /// Update with new error signal
    pub fn update_error(&mut self, error: f64, config: &PurkinjeCellConfig) {
        self.error_signal = error;
        
        // Generate complex spike if error exceeds threshold
        if error.abs() > config.complex_spike_threshold {
            self.complex_spike_pending = true;
            self.teaching_strength = error.abs();
        }
        
        // Quantum error encoding
        let error_magnitude = error.abs().min(1.0);
        let error_phase = if error > 0.0 { 0.0 } else { std::f64::consts::PI };
        self.quantum_error_state = Complex64::new(
            error_magnitude * error_phase.cos(),
            error_magnitude * error_phase.sin(),
        );
    }
    
    /// Generate complex spike
    pub fn generate_complex_spike(&mut self) -> f64 {
        if self.complex_spike_pending {
            self.complex_spike_pending = false;
            self.teaching_strength
        } else {
            0.0
        }
    }
}

/// Individual Purkinje cell with quantum enhancement
pub struct QuantumPurkinjeCell {
    /// Cell ID
    pub id: usize,
    
    /// Cell state
    pub state: CellState,
    
    /// Dendritic compartments
    pub dendrites: Vec<DendriticCompartment>,
    
    /// Climbing fiber input
    pub climbing_fiber: ClimbingFiberInput,
    
    /// Configuration
    config: PurkinjeCellConfig,
    
    /// Spike history
    pub spike_history: VecDeque<(f64, bool)>, // (time, is_complex_spike)
    
    /// Intrinsic excitability
    pub intrinsic_excitability: f64,
    
    /// Quantum circuit for complex integration
    quantum_circuit: Option<QuantumCircuitSimulator>,
    
    /// Performance metrics
    simple_spikes: u64,
    complex_spikes: u64,
}

impl QuantumPurkinjeCell {
    /// Create new quantum Purkinje cell
    pub fn new(
        id: usize,
        n_parallel_fibers: usize,
        config: &PurkinjeCellConfig,
    ) -> Result<Self> {
        let n_qubits = 8;
        let state = CellState::new(n_parallel_fibers, n_qubits);
        
        // Create dendritic compartments
        let mut dendrites = Vec::with_capacity(config.dendritic_branches);
        let pf_per_dendrite = n_parallel_fibers / config.dendritic_branches;
        
        for i in 0..config.dendritic_branches {
            dendrites.push(DendriticCompartment::new(i, pf_per_dendrite));
        }
        
        // Initialize climbing fiber
        let climbing_fiber = ClimbingFiberInput::new(id);
        
        // Create quantum circuit for integration
        let quantum_circuit = if config.quantum_dendritic_integration {
            Some(QuantumCircuitSimulator::new(4)?)
        } else {
            None
        };
        
        Ok(Self {
            id,
            state,
            dendrites,
            climbing_fiber,
            config: config.clone(),
            spike_history: VecDeque::with_capacity(1000),
            intrinsic_excitability: 1.0,
            quantum_circuit,
            simple_spikes: 0,
            complex_spikes: 0,
        })
    }
    
    /// Process parallel fiber inputs and generate output
    pub fn process(
        &mut self,
        parallel_fiber_activity: &[f64],
        climbing_fiber_signal: f64,
        current_time: f64,
    ) -> Result<QuantumSpikeTrain> {
        let mut spike_train = QuantumSpikeTrain::new(self.id, 1.0);
        
        // Update climbing fiber
        self.climbing_fiber.update_error(climbing_fiber_signal, &self.config);
        
        // Process dendritic integration
        let dendritic_inputs = self.integrate_dendritic_inputs(parallel_fiber_activity)?;
        
        // Apply quantum dendritic integration
        let quantum_enhancement = if let Some(ref mut qc) = self.quantum_circuit {
            self.apply_quantum_integration(qc, &dendritic_inputs)?
        } else {
            1.0
        };
        
        // Somatic integration
        let total_input = dendritic_inputs.sum() * quantum_enhancement;
        self.state.voltage += total_input * 0.1 * self.intrinsic_excitability;
        self.state.voltage *= 0.92; // Membrane leak
        
        // Generate spikes
        let mut spike_generated = false;
        
        // Complex spike from climbing fiber
        let cf_spike = self.climbing_fiber.generate_complex_spike();
        if cf_spike > 0.0 {
            let complex_amplitude = Complex64::new(cf_spike, std::f64::consts::PI);
            spike_train.add_spike(current_time, complex_amplitude, std::f64::consts::PI);
            
            self.state.reset(current_time, 10.0); // Long refractory for complex spike
            self.spike_history.push_back((current_time, true));
            self.complex_spikes += 1;
            spike_generated = true;
            
            // Apply climbing fiber modulation to all dendrites
            self.apply_climbing_fiber_modulation(cf_spike);
        }
        // Simple spike from parallel fiber integration
        else if !self.state.is_refractory(current_time) && 
                self.state.voltage > self.config.simple_spike_threshold {
            let spike_amplitude = Complex64::new(
                self.state.voltage - self.config.simple_spike_threshold,
                0.0
            );
            spike_train.add_spike(current_time, spike_amplitude, 0.0);
            
            self.state.reset(current_time, 2.0); // Short refractory for simple spike
            self.spike_history.push_back((current_time, false));
            self.simple_spikes += 1;
            spike_generated = true;
        }
        
        // Intrinsic plasticity
        if self.config.intrinsic_plasticity {
            self.update_intrinsic_plasticity(spike_generated);
        }
        
        // Cleanup old spike history
        while let Some(&(spike_time, _)) = self.spike_history.front() {
            if current_time - spike_time > 100.0 {
                self.spike_history.pop_front();
            } else {
                break;
            }
        }
        
        Ok(spike_train)
    }
    
    /// Integrate inputs across dendritic compartments
    fn integrate_dendritic_inputs(
        &mut self,
        pf_activity: &[f64],
    ) -> Result<DVector<f64>> {
        let mut dendritic_outputs = DVector::zeros(self.dendrites.len());
        
        for (i, dendrite) in self.dendrites.iter_mut().enumerate() {
            // Distribute parallel fiber inputs across dendrites
            let start_idx = i * (pf_activity.len() / self.dendrites.len());
            let end_idx = ((i + 1) * (pf_activity.len() / self.dendrites.len())).min(pf_activity.len());
            
            if start_idx < pf_activity.len() {
                let dendrite_input = &pf_activity[start_idx..end_idx];
                dendritic_outputs[i] = dendrite.integrate_pf_inputs(dendrite_input, &self.config);
            }
        }
        
        Ok(dendritic_outputs)
    }
    
    /// Apply quantum integration across dendritic tree
    fn apply_quantum_integration(
        &mut self,
        quantum_circuit: &mut QuantumCircuitSimulator,
        dendritic_inputs: &DVector<f64>,
    ) -> Result<f64> {
        quantum_circuit.reset();
        
        // Encode dendritic inputs in quantum state
        let input_amplitudes: Vec<f64> = dendritic_inputs.iter()
            .take(4)
            .map(|&x| x.abs().min(1.0))
            .collect();
        
        let input_phases: Vec<f64> = dendritic_inputs.iter()
            .take(4)
            .map(|&x| if x > 0.0 { 0.0 } else { std::f64::consts::PI })
            .collect();
        
        // Create quantum integration circuit
        CerebellarQuantumCircuits::create_spike_encoding_circuit(
            quantum_circuit,
            &input_amplitudes,
            &input_phases,
        )?;
        
        // Add dendritic correlations
        let correlations: Vec<(usize, usize, f64)> = (0..3)
            .map(|i| (i, i + 1, 0.3))
            .collect();
        
        CerebellarQuantumCircuits::create_entanglement_circuit(
            quantum_circuit,
            &correlations,
        )?;
        
        quantum_circuit.execute()?;
        
        // Measure quantum enhancement
        let probabilities = quantum_circuit.measure_probabilities(&[0, 1, 2, 3])?;
        let quantum_enhancement = probabilities.iter()
            .enumerate()
            .map(|(i, &p)| p * (i as f64 + 1.0))
            .sum::<f64>() / 4.0;
        
        Ok(quantum_enhancement.clamp(0.5, 2.0))
    }
    
    /// Apply climbing fiber modulation to all dendrites
    fn apply_climbing_fiber_modulation(&mut self, cf_strength: f64) {
        for dendrite in &mut self.dendrites {
            dendrite.climbing_fiber_modulation(cf_strength, &self.config);
        }
    }
    
    /// Update intrinsic plasticity based on activity
    fn update_intrinsic_plasticity(&mut self, spike_generated: bool) {
        let target_rate = 0.1; // Target 10% activity
        let current_activity = if spike_generated { 1.0 } else { 0.0 };
        
        // Simple rule: adjust excitability toward target
        let error = current_activity - target_rate;
        self.intrinsic_excitability -= error * self.config.plasticity_rate * 0.1;
        self.intrinsic_excitability = self.intrinsic_excitability.clamp(0.1, 3.0);
    }
    
    /// Get recent spike rate
    pub fn spike_rate(&self, time_window: f64, current_time: f64) -> f64 {
        let recent_spikes = self.spike_history.iter()
            .filter(|(spike_time, _)| current_time - spike_time < time_window)
            .count();
        
        recent_spikes as f64 / (time_window / 1000.0)
    }
    
    /// Get complex spike rate
    pub fn complex_spike_rate(&self, time_window: f64, current_time: f64) -> f64 {
        let recent_complex_spikes = self.spike_history.iter()
            .filter(|(spike_time, is_complex)| {
                *is_complex && current_time - spike_time < time_window
            })
            .count();
        
        recent_complex_spikes as f64 / (time_window / 1000.0)
    }
    
    /// Reset cell state
    pub fn reset(&mut self) {
        self.state.voltage = 0.0;
        self.state.last_spike_time = None;
        self.state.refractory_until = 0.0;
        self.spike_history.clear();
        self.intrinsic_excitability = 1.0;
        
        // Reset dendrites
        for dendrite in &mut self.dendrites {
            dendrite.voltage = 0.0;
            dendrite.calcium = 0.0;
            dendrite.ltp_tag = 0.0;
            dendrite.ltd_tag = 0.0;
            
            // Reset quantum state
            if dendrite.quantum_state.len() >= 2 {
                dendrite.quantum_state[0] = Complex64::new(1.0, 0.0);
                dendrite.quantum_state[1] = Complex64::new(0.0, 0.0);
            }
        }
        
        // Reset climbing fiber
        self.climbing_fiber.error_signal = 0.0;
        self.climbing_fiber.complex_spike_pending = false;
        self.climbing_fiber.quantum_error_state = Complex64::new(0.0, 0.0);
        
        if let Some(ref mut qc) = self.quantum_circuit {
            qc.reset();
        }
        
        self.simple_spikes = 0;
        self.complex_spikes = 0;
    }
}

/// Population of Purkinje cells
pub struct PurkinjeCellPopulation {
    /// Individual Purkinje cells
    pub purkinje_cells: Vec<QuantumPurkinjeCell>,
    
    /// Configuration
    config: PurkinjeCellConfig,
    
    /// Population activity pattern
    activity_pattern: DVector<f64>,
    
    /// Deep cerebellar nuclei connections
    dcn_weights: DMatrix<f64>,
    
    /// Performance metrics
    total_simple_spikes: u64,
    total_complex_spikes: u64,
    processing_time_ns: u64,
}

impl PurkinjeCellPopulation {
    /// Create new Purkinje cell population
    pub fn new(
        n_purkinje_cells: usize,
        n_parallel_fibers: usize,
        snn_config: &QuantumSNNConfig,
    ) -> Result<Self> {
        let config = PurkinjeCellConfig {
            n_parallel_fibers,
            ..Default::default()
        };
        
        let mut purkinje_cells = Vec::with_capacity(n_purkinje_cells);
        for i in 0..n_purkinje_cells {
            purkinje_cells.push(QuantumPurkinjeCell::new(i, n_parallel_fibers, &config)?);
        }
        
        // Initialize DCN connections (inhibitory)
        let dcn_weights = DMatrix::from_fn(n_purkinje_cells, 3, |_, _| -rand::random::<f64>() * 0.5);
        
        info!("Created Purkinje cell population: {} cells, {} PF inputs each", 
              n_purkinje_cells, n_parallel_fibers);
        
        Ok(Self {
            purkinje_cells,
            config,
            activity_pattern: DVector::zeros(n_purkinje_cells),
            dcn_weights,
            total_simple_spikes: 0,
            total_complex_spikes: 0,
            processing_time_ns: 0,
        })
    }
    
    /// Process parallel fiber inputs through population
    pub fn process(
        &mut self,
        parallel_fiber_activity: &[QuantumSpikeTrain],
        current_time: f64,
    ) -> Result<Vec<QuantumSpikeTrain>> {
        let start_time = std::time::Instant::now();
        
        // Convert spike trains to activity levels
        let pf_activities = self.convert_spike_trains_to_activity(parallel_fiber_activity, current_time);
        
        let mut output_spikes = Vec::new();
        let mut activities = Vec::new();
        
        // Process each Purkinje cell
        for pc in &mut self.purkinje_cells {
            // Default climbing fiber signal (would come from inferior olive)
            let cf_signal = 0.0; // No error signal by default
            
            let spike_train = pc.process(&pf_activities, cf_signal, current_time)?;
            
            // Record activity
            let activity = if spike_train.is_empty() { 0.0 } else { 1.0 };
            activities.push(activity);
            
            output_spikes.push(spike_train);
            
            // Update population statistics
            self.total_simple_spikes += pc.simple_spikes;
            self.total_complex_spikes += pc.complex_spikes;
        }
        
        // Update activity pattern
        self.activity_pattern = DVector::from_vec(activities);
        
        // Performance metrics
        self.processing_time_ns = start_time.elapsed().as_nanos() as u64;
        
        debug!("Purkinje population processed: {} active cells, {}μs", 
               self.activity_pattern.sum(), start_time.elapsed().as_micros());
        
        Ok(output_spikes)
    }
    
    /// Convert spike trains to activity levels
    fn convert_spike_trains_to_activity(
        &self,
        spike_trains: &[QuantumSpikeTrain],
        current_time: f64,
    ) -> Vec<f64> {
        let time_window = 5.0; // ms
        
        spike_trains.iter().map(|st| {
            let recent_spikes = st.times.iter()
                .filter(|&&spike_time| current_time - spike_time < time_window)
                .count();
            
            (recent_spikes as f64).min(1.0)
        }).collect()
    }
    
    /// Update plasticity based on error signals
    pub fn update_plasticity(
        &mut self,
        error_signals: &[f64],
        current_time: f64,
    ) -> Result<()> {
        for (i, pc) in self.purkinje_cells.iter_mut().enumerate() {
            if i < error_signals.len() {
                pc.climbing_fiber.update_error(error_signals[i], &self.config);
            }
        }
        
        Ok(())
    }
    
    /// Get population firing rate
    pub fn population_firing_rate(&self, time_window: f64, current_time: f64) -> f64 {
        let total_rate: f64 = self.purkinje_cells.iter()
            .map(|pc| pc.spike_rate(time_window, current_time))
            .sum();
        
        total_rate / self.purkinje_cells.len() as f64
    }
    
    /// Generate deep cerebellar nuclei output
    pub fn generate_dcn_output(&self) -> DVector<f64> {
        // Purkinje cells provide inhibitory input to DCN
        let pc_activities = &self.activity_pattern;
        let dcn_input = &self.dcn_weights.transpose() * pc_activities;
        
        // Apply activation function (ReLU with bias)
        dcn_input.map(|x| (x + 0.5).max(0.0))
    }
    
    /// Adapt DCN weights based on performance
    pub fn adapt_dcn_weights(&mut self, performance_error: f64) {
        let learning_rate = 0.001;
        
        for i in 0..self.dcn_weights.nrows() {
            for j in 0..self.dcn_weights.ncols() {
                let weight_update = learning_rate * performance_error * self.activity_pattern[i];
                self.dcn_weights[(i, j)] -= weight_update; // Inhibitory adaptation
            }
        }
        
        // Bound weights
        for i in 0..self.dcn_weights.nrows() {
            for j in 0..self.dcn_weights.ncols() {
                self.dcn_weights[(i, j)] = self.dcn_weights[(i, j)].clamp(-2.0, 0.0);
            }
        }
    }
    
    /// Reset population state
    pub fn reset(&mut self) {
        for pc in &mut self.purkinje_cells {
            pc.reset();
        }
        
        self.activity_pattern.fill(0.0);
        self.total_simple_spikes = 0;
        self.total_complex_spikes = 0;
    }
    
    /// Get population statistics
    pub fn get_statistics(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        
        stats.insert("total_simple_spikes".to_string(), self.total_simple_spikes as f64);
        stats.insert("total_complex_spikes".to_string(), self.total_complex_spikes as f64);
        stats.insert("processing_time_ms".to_string(), self.processing_time_ns as f64 / 1_000_000.0);
        stats.insert("active_cells".to_string(), self.activity_pattern.sum());
        stats.insert("population_size".to_string(), self.purkinje_cells.len() as f64);
        
        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_purkinje_cell_creation() {
        let config = PurkinjeCellConfig::default();
        let pc = QuantumPurkinjeCell::new(0, 50, &config).unwrap();
        
        assert_eq!(pc.id, 0);
        assert_eq!(pc.dendrites.len(), config.dendritic_branches);
        assert_eq!(pc.spike_history.len(), 0);
    }
    
    #[test]
    fn test_dendritic_integration() {
        let mut dendrite = DendriticCompartment::new(0, 10);
        let config = PurkinjeCellConfig::default();
        
        let pf_activity = vec![0.8, 0.3, 0.0, 0.9, 0.1, 0.6, 0.2, 0.7, 0.4, 0.5];
        let voltage = dendrite.integrate_pf_inputs(&pf_activity, &config);
        
        assert!(voltage >= 0.0);
    }
    
    #[test]
    fn test_parallel_fiber_synapse() {
        let mut synapse = ParallelFiberSynapse::new(0, 0);
        let config = PurkinjeCellConfig::default();
        
        let initial_weight = synapse.weight;
        
        // Process input
        let output = synapse.process_input(0.8);
        assert_relative_eq!(output, 0.8 * synapse.weight, epsilon = 1e-10);
        
        // Apply climbing fiber modulation
        synapse.apply_cf_modulation(1.0, 0.5, &config);
        
        // Weight should change due to LTD
        assert_ne!(synapse.weight, initial_weight);
    }
    
    #[test]
    fn test_climbing_fiber_input() {
        let mut cf = ClimbingFiberInput::new(0);
        let config = PurkinjeCellConfig::default();
        
        // Test error signal update
        cf.update_error(0.9, &config);
        assert!(cf.complex_spike_pending);
        
        // Test complex spike generation
        let spike_strength = cf.generate_complex_spike();
        assert!(spike_strength > 0.0);
        assert!(!cf.complex_spike_pending); // Should be reset
    }
    
    #[test]
    fn test_purkinje_cell_processing() {
        let config = PurkinjeCellConfig::default();
        let mut pc = QuantumPurkinjeCell::new(0, 20, &config).unwrap();
        
        let pf_activity = vec![0.7; 20];
        let cf_signal = 0.0;
        let current_time = 1.0;
        
        let spike_train = pc.process(&pf_activity, cf_signal, current_time).unwrap();
        
        // Should produce valid spike train
        assert_eq!(spike_train.neuron_id, 0);
    }
    
    #[test]
    fn test_complex_spike_generation() {
        let config = PurkinjeCellConfig::default();
        let mut pc = QuantumPurkinjeCell::new(0, 20, &config).unwrap();
        
        let pf_activity = vec![0.5; 20];
        let cf_signal = 1.0; // Strong climbing fiber signal
        let current_time = 1.0;
        
        let spike_train = pc.process(&pf_activity, cf_signal, current_time).unwrap();
        
        // Should generate complex spike
        if !spike_train.is_empty() {
            assert!(spike_train.phases[0] > 0.0); // Complex spike has phase information
        }
        
        assert_eq!(pc.complex_spikes, 1);
    }
    
    #[test]
    fn test_purkinje_population() {
        let snn_config = QuantumSNNConfig::default();
        let mut population = PurkinjeCellPopulation::new(5, 20, &snn_config).unwrap();
        
        // Create test spike trains
        let mut spike_trains = Vec::new();
        for i in 0..20 {
            let mut st = QuantumSpikeTrain::new(i, 1.0);
            st.add_spike(0.5, Complex64::new(0.7, 0.0), 0.0);
            spike_trains.push(st);
        }
        
        let output = population.process(&spike_trains, 1.0).unwrap();
        
        assert_eq!(output.len(), 5);
        assert!(population.activity_pattern.sum() >= 0.0);
    }
    
    #[test]
    fn test_dcn_output_generation() {
        let snn_config = QuantumSNNConfig::default();
        let population = PurkinjeCellPopulation::new(3, 10, &snn_config).unwrap();
        
        let dcn_output = population.generate_dcn_output();
        
        assert_eq!(dcn_output.len(), 3);
        // DCN output should be non-negative (ReLU activation)
        for &output in dcn_output.iter() {
            assert!(output >= 0.0);
        }
    }
    
    #[test]
    fn test_intrinsic_plasticity() {
        let config = PurkinjeCellConfig::default();
        let mut pc = QuantumPurkinjeCell::new(0, 10, &config).unwrap();
        
        let initial_excitability = pc.intrinsic_excitability;
        
        // Simulate high activity
        for _ in 0..10 {
            pc.update_intrinsic_plasticity(true);
        }
        
        // Excitability should decrease due to high activity
        assert!(pc.intrinsic_excitability <= initial_excitability);
    }
}