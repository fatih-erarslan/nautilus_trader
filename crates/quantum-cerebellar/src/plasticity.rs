//! Quantum-enhanced plasticity engine for cerebellar learning
//! 
//! Implements multiple forms of synaptic plasticity including LTD, LTP, STDP,
//! metaplasticity, and quantum-coherent plasticity mechanisms.

use std::collections::HashMap;
use nalgebra::{DMatrix, DVector};
use num_complex::Complex64;
use anyhow::{Result, anyhow};
use tracing::{debug, info, warn};
use serde::{Serialize, Deserialize};

use crate::{QuantumSNNConfig, QuantumSpikeTrain, QuantumSynapse};

/// Plasticity types supported by the engine
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum PlasticityType {
    /// Long-term Depression (characteristic of cerebellar PF-PC synapses)
    LTD,
    /// Long-term Potentiation
    LTP,
    /// Spike-timing Dependent Plasticity
    STDP,
    /// Homeostatic plasticity
    Homeostatic,
    /// Metaplasticity (plasticity of plasticity)
    Metaplasticity,
    /// Quantum-coherent plasticity
    QuantumCoherent,
}

/// Plasticity rule configuration
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PlasticityConfig {
    /// Learning rate
    pub learning_rate: f64,
    
    /// STDP time window (ms)
    pub stdp_window: f64,
    
    /// LTD/LTP balance
    pub ltd_ltp_ratio: f64,
    
    /// Metaplasticity threshold
    pub meta_threshold: f64,
    
    /// Quantum coherence contribution
    pub quantum_strength: f64,
    
    /// Homeostatic time constant
    pub homeostatic_tau: f64,
    
    /// Weight bounds
    pub weight_min: f64,
    pub weight_max: f64,
    
    /// Calcium dynamics
    pub calcium_decay: f64,
    pub calcium_threshold_ltd: f64,
    pub calcium_threshold_ltp: f64,
}

impl Default for PlasticityConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.01,
            stdp_window: 20.0,
            ltd_ltp_ratio: 3.0, // LTD stronger than LTP in cerebellum
            meta_threshold: 0.5,
            quantum_strength: 0.1,
            homeostatic_tau: 1000.0,
            weight_min: -2.0,
            weight_max: 2.0,
            calcium_decay: 0.95,
            calcium_threshold_ltd: 0.3,
            calcium_threshold_ltp: 0.8,
        }
    }
}

/// Calcium dynamics for plasticity induction
#[derive(Debug, Clone)]
pub struct CalciumDynamics {
    /// Current calcium concentration
    pub concentration: f64,
    
    /// Calcium trace for integration
    pub trace: f64,
    
    /// Calcium sources (NMDA, VGCC, internal stores)
    pub nmda_contribution: f64,
    pub vgcc_contribution: f64,
    pub internal_stores: f64,
    
    /// Calcium binding proteins
    pub buffer_capacity: f64,
    pub buffer_occupancy: f64,
}

impl CalciumDynamics {
    pub fn new() -> Self {
        Self {
            concentration: 0.0,
            trace: 0.0,
            nmda_contribution: 0.0,
            vgcc_contribution: 0.0,
            internal_stores: 0.0,
            buffer_capacity: 1.0,
            buffer_occupancy: 0.0,
        }
    }
    
    /// Update calcium based on neural activity
    pub fn update(
        &mut self,
        presynaptic_activity: f64,
        postsynaptic_activity: f64,
        climbing_fiber_activity: f64,
        config: &PlasticityConfig,
    ) {
        // NMDA-dependent calcium influx
        self.nmda_contribution = presynaptic_activity * postsynaptic_activity * 0.5;
        
        // Voltage-gated calcium channels
        self.vgcc_contribution = postsynaptic_activity * 0.3;
        
        // Climbing fiber-induced calcium release
        let cf_calcium = climbing_fiber_activity * 2.0;
        
        // Total calcium influx
        let influx = self.nmda_contribution + self.vgcc_contribution + cf_calcium;
        
        // Update concentration with buffering
        let effective_influx = influx * (1.0 - self.buffer_occupancy);
        self.concentration += effective_influx;
        
        // Calcium decay
        self.concentration *= config.calcium_decay;
        
        // Update trace (slower decay)
        self.trace = self.trace * 0.99 + self.concentration * 0.01;
        
        // Update buffer dynamics
        self.buffer_occupancy += (self.concentration - self.buffer_occupancy) * 0.1;
        self.buffer_occupancy = self.buffer_occupancy.clamp(0.0, 1.0);
    }
    
    /// Get plasticity signal based on calcium level
    pub fn plasticity_signal(&self, config: &PlasticityConfig) -> f64 {
        if self.concentration > config.calcium_threshold_ltp {
            // High calcium -> LTP
            (self.concentration - config.calcium_threshold_ltp) * 2.0
        } else if self.concentration > config.calcium_threshold_ltd {
            // Medium calcium -> LTD
            -(self.concentration - config.calcium_threshold_ltd) * config.ltd_ltp_ratio
        } else {
            // Low calcium -> no plasticity
            0.0
        }
    }
}

/// Metaplasticity state tracking
#[derive(Debug, Clone)]
pub struct MetaplasticityState {
    /// Recent plasticity history
    pub plasticity_history: Vec<f64>,
    
    /// Metaplasticity variable (BCM-like)
    pub theta: f64,
    
    /// Sliding threshold for plasticity induction
    pub modification_threshold: f64,
    
    /// Priming state
    pub priming_level: f64,
}

impl MetaplasticityState {
    pub fn new() -> Self {
        Self {
            plasticity_history: Vec::with_capacity(1000),
            theta: 0.5,
            modification_threshold: 0.3,
            priming_level: 0.0,
        }
    }
    
    /// Update metaplasticity based on recent activity
    pub fn update(&mut self, plasticity_amount: f64, config: &PlasticityConfig) {
        self.plasticity_history.push(plasticity_amount);
        
        // Keep only recent history
        if self.plasticity_history.len() > 1000 {
            self.plasticity_history.remove(0);
        }
        
        // Update sliding threshold (BCM rule)
        let recent_activity: f64 = self.plasticity_history.iter()
            .rev()
            .take(100)
            .sum::<f64>() / 100.0;
        
        self.theta += (recent_activity.powi(2) - self.theta) * 0.001;
        
        // Update modification threshold
        self.modification_threshold = self.theta * 0.6;
        
        // Update priming (facilitates future plasticity)
        if plasticity_amount.abs() > config.meta_threshold {
            self.priming_level += 0.1;
        }
        self.priming_level *= 0.995; // Slow decay
        self.priming_level = self.priming_level.clamp(0.0, 2.0);
    }
    
    /// Get metaplasticity modulation factor
    pub fn modulation_factor(&self) -> f64 {
        1.0 + self.priming_level
    }
}

/// Quantum coherence plasticity mechanism
#[derive(Debug, Clone)]
pub struct QuantumPlasticity {
    /// Quantum coherence between pre and post neurons
    pub coherence_matrix: DMatrix<Complex64>,
    
    /// Entanglement measure
    pub entanglement_strength: f64,
    
    /// Quantum phase relationships
    pub phase_relationships: Vec<f64>,
    
    /// Decoherence time
    pub coherence_time: f64,
}

impl QuantumPlasticity {
    pub fn new(n_neurons: usize) -> Self {
        Self {
            coherence_matrix: DMatrix::zeros(n_neurons, n_neurons),
            entanglement_strength: 0.0,
            phase_relationships: vec![0.0; n_neurons],
            coherence_time: 100.0,
        }
    }
    
    /// Update quantum coherence based on spike correlations
    pub fn update_coherence(
        &mut self,
        spike_trains: &[QuantumSpikeTrain],
        config: &PlasticityConfig,
    ) {
        let n = spike_trains.len().min(self.coherence_matrix.nrows());
        
        // Calculate pairwise coherence
        for i in 0..n {
            for j in (i + 1)..n {
                let coherence = self.calculate_spike_coherence(&spike_trains[i], &spike_trains[j]);
                self.coherence_matrix[(i, j)] = coherence;
                self.coherence_matrix[(j, i)] = coherence.conj();
            }
        }
        
        // Update entanglement strength
        self.entanglement_strength = self.calculate_entanglement_entropy();
        
        // Decoherence
        self.coherence_matrix *= Complex64::new(0.99, 0.0);
    }
    
    /// Calculate coherence between two spike trains
    fn calculate_spike_coherence(
        &self,
        train1: &QuantumSpikeTrain,
        train2: &QuantumSpikeTrain,
    ) -> Complex64 {
        if train1.amplitudes.is_empty() || train2.amplitudes.is_empty() {
            return Complex64::new(0.0, 0.0);
        }
        
        // Cross-correlation of quantum amplitudes
        let mut coherence = Complex64::new(0.0, 0.0);
        let mut count = 0;
        
        for (i, &amp1) in train1.amplitudes.iter().enumerate() {
            for (j, &amp2) in train2.amplitudes.iter().enumerate() {
                if i < train1.times.len() && j < train2.times.len() {
                    let time_diff = (train1.times[i] - train2.times[j]).abs();
                    if time_diff < 5.0 { // 5ms window
                        coherence += amp1.conj() * amp2;
                        count += 1;
                    }
                }
            }
        }
        
        if count > 0 {
            coherence / Complex64::new(count as f64, 0.0)
        } else {
            Complex64::new(0.0, 0.0)
        }
    }
    
    /// Calculate entanglement entropy
    fn calculate_entanglement_entropy(&self) -> f64 {
        // Simplified entanglement measure based on coherence matrix
        let eigenvalues = self.coherence_matrix.diagonal().map(|c| c.norm_sqr());
        let total = eigenvalues.sum();
        
        if total > 1e-10 {
            let normalized = eigenvalues.map(|x| x / total);
            -normalized.iter()
                .filter(|&&x| x > 1e-10)
                .map(|&x| x * x.ln())
                .sum::<f64>()
        } else {
            0.0
        }
    }
    
    /// Get quantum plasticity contribution
    pub fn quantum_contribution(&self, pre_idx: usize, post_idx: usize) -> f64 {
        if pre_idx < self.coherence_matrix.nrows() && post_idx < self.coherence_matrix.ncols() {
            self.coherence_matrix[(pre_idx, post_idx)].norm() * self.entanglement_strength
        } else {
            0.0
        }
    }
}

/// Main plasticity engine
pub struct PlasticityEngine {
    /// Configuration
    config: PlasticityConfig,
    
    /// Calcium dynamics for each synapse
    calcium_dynamics: HashMap<(usize, usize), CalciumDynamics>,
    
    /// Metaplasticity state for each neuron
    metaplasticity_states: HashMap<usize, MetaplasticityState>,
    
    /// Quantum plasticity mechanism
    quantum_plasticity: QuantumPlasticity,
    
    /// Weight change history
    weight_changes: HashMap<(usize, usize), Vec<f64>>,
    
    /// Performance statistics
    plasticity_events: u64,
    quantum_events: u64,
}

impl PlasticityEngine {
    /// Create new plasticity engine
    pub fn new(snn_config: &QuantumSNNConfig) -> Result<Self> {
        let config = PlasticityConfig {
            learning_rate: snn_config.learning_rate,
            stdp_window: snn_config.stdp_window,
            quantum_strength: snn_config.entanglement_strength,
            ..Default::default()
        };
        
        let quantum_plasticity = QuantumPlasticity::new(
            snn_config.n_granule_cells + snn_config.n_purkinje_cells
        );
        
        info!("Initialized plasticity engine with quantum enhancement");
        
        Ok(Self {
            config,
            calcium_dynamics: HashMap::new(),
            metaplasticity_states: HashMap::new(),
            quantum_plasticity,
            weight_changes: HashMap::new(),
            plasticity_events: 0,
            quantum_events: 0,
        })
    }
    
    /// Update all synapses based on activity
    pub fn update_all_synapses(
        &mut self,
        synapses: &mut [QuantumSynapse],
        presynaptic_spikes: &[QuantumSpikeTrain],
        postsynaptic_spikes: &[QuantumSpikeTrain],
        current_time: f64,
    ) -> Result<()> {
        // Update quantum coherence
        let all_spikes: Vec<QuantumSpikeTrain> = presynaptic_spikes.iter()
            .chain(postsynaptic_spikes.iter())
            .cloned()
            .collect();
        
        self.quantum_plasticity.update_coherence(&all_spikes, &self.config);
        
        // Update individual synapses
        for synapse in synapses.iter_mut() {
            self.update_synapse(
                synapse,
                presynaptic_spikes,
                postsynaptic_spikes,
                current_time,
            )?;
        }
        
        Ok(())
    }
    
    /// Update individual synapse
    fn update_synapse(
        &mut self,
        synapse: &mut QuantumSynapse,
        presynaptic_spikes: &[QuantumSpikeTrain],
        postsynaptic_spikes: &[QuantumSpikeTrain],
        current_time: f64,
    ) -> Result<()> {
        let synapse_id = (synapse.source, synapse.target);
        
        // Get or create calcium dynamics
        let calcium = self.calcium_dynamics.entry(synapse_id)
            .or_insert_with(CalciumDynamics::new);
        
        // Get or create metaplasticity state
        let metaplasticity = self.metaplasticity_states.entry(synapse.target)
            .or_insert_with(MetaplasticityState::new);
        
        // Calculate activities
        let pre_activity = self.calculate_recent_activity(
            presynaptic_spikes.get(synapse.source),
            current_time,
        );
        let post_activity = self.calculate_recent_activity(
            postsynaptic_spikes.get(synapse.target),
            current_time,
        );
        
        // Climbing fiber activity (simplified - would come from error signal)
        let cf_activity = 0.0;
        
        // Update calcium dynamics
        calcium.update(pre_activity, post_activity, cf_activity, &self.config);
        
        // Calculate plasticity
        let mut weight_change = 0.0;
        
        // Calcium-dependent plasticity
        let calcium_signal = calcium.plasticity_signal(&self.config);
        weight_change += calcium_signal * self.config.learning_rate;
        
        // STDP component
        if let (Some(pre_train), Some(post_train)) = (
            presynaptic_spikes.get(synapse.source),
            postsynaptic_spikes.get(synapse.target),
        ) {
            let stdp_change = self.calculate_stdp(pre_train, post_train, current_time);
            weight_change += stdp_change * self.config.learning_rate;
        }
        
        // Quantum plasticity contribution
        let quantum_contribution = self.quantum_plasticity.quantum_contribution(
            synapse.source, synapse.target
        );
        weight_change += quantum_contribution * self.config.quantum_strength;
        
        if quantum_contribution.abs() > 1e-6 {
            self.quantum_events += 1;
        }
        
        // Apply metaplasticity modulation
        weight_change *= metaplasticity.modulation_factor();
        
        // Update synapse weight
        if weight_change.abs() > 1e-6 {
            synapse.update_weight(weight_change, current_time);
            self.plasticity_events += 1;
            
            // Record weight change
            self.weight_changes.entry(synapse_id)
                .or_insert_with(Vec::new)
                .push(weight_change);
        }
        
        // Update metaplasticity
        metaplasticity.update(weight_change, &self.config);
        
        Ok(())
    }
    
    /// Calculate recent neural activity
    fn calculate_recent_activity(
        &self,
        spike_train: Option<&QuantumSpikeTrain>,
        current_time: f64,
    ) -> f64 {
        if let Some(train) = spike_train {
            let time_window = 10.0; // ms
            let recent_spikes = train.times.iter()
                .filter(|&&t| current_time - t < time_window)
                .count();
            
            recent_spikes as f64 / 10.0 // Normalize
        } else {
            0.0
        }
    }
    
    /// Calculate STDP weight change
    fn calculate_stdp(
        &self,
        pre_train: &QuantumSpikeTrain,
        post_train: &QuantumSpikeTrain,
        current_time: f64,
    ) -> f64 {
        let mut stdp_sum = 0.0;
        let window = self.config.stdp_window;
        
        // Consider recent spikes within STDP window
        for &pre_time in &pre_train.times {
            if current_time - pre_time < window {
                for &post_time in &post_train.times {
                    if current_time - post_time < window {
                        let dt = post_time - pre_time;
                        
                        if dt.abs() < window {
                            let stdp_strength = if dt > 0.0 {
                                // LTP: post after pre
                                0.1 * (-dt / (window / 4.0)).exp()
                            } else {
                                // LTD: pre after post (stronger in cerebellum)
                                -0.15 * (dt / (window / 4.0)).exp()
                            };
                            
                            stdp_sum += stdp_strength;
                        }
                    }
                }
            }
        }
        
        stdp_sum
    }
    
    /// Apply cerebellar-specific plasticity rules
    pub fn apply_cerebellar_plasticity(
        &mut self,
        pf_pc_synapses: &mut [QuantumSynapse],
        parallel_fiber_activity: &[QuantumSpikeTrain],
        purkinje_cell_activity: &[QuantumSpikeTrain],
        climbing_fiber_signals: &[f64],
        current_time: f64,
    ) -> Result<()> {
        for synapse in pf_pc_synapses.iter_mut() {
            let pf_id = synapse.source;
            let pc_id = synapse.target;
            
            // Get activities
            let pf_activity = parallel_fiber_activity.get(pf_id)
                .map(|train| self.calculate_recent_activity(Some(train), current_time))
                .unwrap_or(0.0);
            
            let pc_activity = purkinje_cell_activity.get(pc_id)
                .map(|train| self.calculate_recent_activity(Some(train), current_time))
                .unwrap_or(0.0);
            
            let cf_signal = climbing_fiber_signals.get(pc_id).copied().unwrap_or(0.0);
            
            // Cerebellar LTD rule: conjunctive PF + CF -> LTD
            if pf_activity > 0.1 && cf_signal > 0.1 {
                let ltd_strength = pf_activity * cf_signal * self.config.learning_rate * 2.0;
                synapse.update_weight(-ltd_strength, current_time);
                self.plasticity_events += 1;
            }
            // Weak LTP when PF active without CF
            else if pf_activity > 0.1 && cf_signal < 0.05 {
                let ltp_strength = pf_activity * self.config.learning_rate * 0.1;
                synapse.update_weight(ltp_strength, current_time);
            }
        }
        
        Ok(())
    }
    
    /// Homeostatic scaling to maintain target activity
    pub fn apply_homeostatic_scaling(
        &mut self,
        synapses: &mut [QuantumSynapse],
        target_activity: f64,
        current_activity: f64,
    ) {
        let activity_ratio = target_activity / (current_activity + 1e-6);
        let scaling_factor = ((activity_ratio - 1.0) * 0.001).tanh();
        
        for synapse in synapses.iter_mut() {
            synapse.weight *= 1.0 + scaling_factor;
            synapse.weight = synapse.weight.clamp(
                self.config.weight_min,
                self.config.weight_max,
            );
        }
    }
    
    /// Get plasticity statistics
    pub fn get_statistics(&self) -> HashMap<String, f64> {
        let mut stats = HashMap::new();
        
        stats.insert("plasticity_events".to_string(), self.plasticity_events as f64);
        stats.insert("quantum_events".to_string(), self.quantum_events as f64);
        stats.insert("entanglement_strength".to_string(), self.quantum_plasticity.entanglement_strength);
        
        // Weight change statistics
        let all_changes: Vec<f64> = self.weight_changes.values()
            .flatten()
            .copied()
            .collect();
        
        if !all_changes.is_empty() {
            let mean_change: f64 = all_changes.iter().sum::<f64>() / all_changes.len() as f64;
            let variance: f64 = all_changes.iter()
                .map(|&x| (x - mean_change).powi(2))
                .sum::<f64>() / all_changes.len() as f64;
            
            stats.insert("mean_weight_change".to_string(), mean_change);
            stats.insert("weight_change_variance".to_string(), variance);
        }
        
        stats
    }
    
    /// Reset plasticity state
    pub fn reset(&mut self) {
        self.calcium_dynamics.clear();
        self.metaplasticity_states.clear();
        self.weight_changes.clear();
        self.plasticity_events = 0;
        self.quantum_events = 0;
        
        // Reset quantum coherence
        self.quantum_plasticity.coherence_matrix.fill(Complex64::new(0.0, 0.0));
        self.quantum_plasticity.entanglement_strength = 0.0;
    }
}

/// Specialized plasticity rules for different synapse types
pub struct CerebellarPlasticityRules;

impl CerebellarPlasticityRules {
    /// Parallel fiber to Purkinje cell LTD
    pub fn pf_pc_ltd(
        pf_activity: f64,
        cf_activity: f64,
        calcium: f64,
        learning_rate: f64,
    ) -> f64 {
        if pf_activity > 0.1 && cf_activity > 0.1 {
            -learning_rate * pf_activity * cf_activity * (1.0 + calcium)
        } else {
            0.0
        }
    }
    
    /// Mossy fiber to granule cell plasticity
    pub fn mf_gc_plasticity(
        mf_activity: f64,
        gc_activity: f64,
        learning_rate: f64,
    ) -> f64 {
        if mf_activity > 0.1 && gc_activity > 0.1 {
            learning_rate * mf_activity * gc_activity * 0.1
        } else {
            0.0
        }
    }
    
    /// Golgi cell inhibitory plasticity
    pub fn golgi_inhibitory_plasticity(
        gc_activity: f64,
        golgi_activity: f64,
        population_activity: f64,
        learning_rate: f64,
    ) -> f64 {
        // Strengthen inhibition when population is too active
        if population_activity > 0.3 {
            learning_rate * gc_activity * 0.05
        } else {
            -learning_rate * gc_activity * 0.02
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_calcium_dynamics() {
        let mut calcium = CalciumDynamics::new();
        let config = PlasticityConfig::default();
        
        // Simulate synaptic activity
        calcium.update(0.8, 0.6, 0.5, &config);
        
        assert!(calcium.concentration > 0.0);
        assert!(calcium.nmda_contribution > 0.0);
        assert!(calcium.vgcc_contribution > 0.0);
        
        // Test plasticity signal
        let signal = calcium.plasticity_signal(&config);
        assert!(signal != 0.0); // Should induce some plasticity
    }
    
    #[test]
    fn test_metaplasticity() {
        let mut meta = MetaplasticityState::new();
        let config = PlasticityConfig::default();
        
        // Simulate repeated plasticity
        for _ in 0..50 {
            meta.update(0.1, &config);
        }
        
        assert!(meta.theta > 0.0);
        assert!(meta.modulation_factor() > 0.0);
    }
    
    #[test]
    fn test_quantum_plasticity() {
        let mut qp = QuantumPlasticity::new(4);
        
        // Create test spike trains
        let mut spike_trains = Vec::new();
        for i in 0..4 {
            let mut st = QuantumSpikeTrain::new(i, 10.0);
            st.add_spike(1.0, Complex64::new(0.7, 0.1), 0.0);
            spike_trains.push(st);
        }
        
        let config = PlasticityConfig::default();
        qp.update_coherence(&spike_trains, &config);
        
        assert!(qp.entanglement_strength >= 0.0);
        
        let contribution = qp.quantum_contribution(0, 1);
        assert!(contribution >= 0.0);
    }
    
    #[test]
    fn test_plasticity_engine() {
        let snn_config = QuantumSNNConfig::default();
        let mut engine = PlasticityEngine::new(&snn_config).unwrap();
        
        // Create test spike trains
        let pre_spikes = vec![{
            let mut st = QuantumSpikeTrain::new(0, 10.0);
            st.add_spike(1.0, Complex64::new(0.8, 0.0), 0.0);
            st
        }];
        
        let post_spikes = vec![{
            let mut st = QuantumSpikeTrain::new(0, 10.0);
            st.add_spike(2.0, Complex64::new(0.6, 0.0), 0.0);
            st
        }];
        
        let mut synapses = vec![QuantumSynapse::new(0, 0, 0.5)];
        
        engine.update_all_synapses(&mut synapses, &pre_spikes, &post_spikes, 3.0).unwrap();
        
        // Should have induced some plasticity
        assert_ne!(synapses[0].weight, 0.5);
    }
    
    #[test]
    fn test_stdp_calculation() {
        let snn_config = QuantumSNNConfig::default();
        let engine = PlasticityEngine::new(&snn_config).unwrap();
        
        // Create spike trains with specific timing
        let mut pre_train = QuantumSpikeTrain::new(0, 10.0);
        pre_train.add_spike(1.0, Complex64::new(1.0, 0.0), 0.0);
        
        let mut post_train = QuantumSpikeTrain::new(1, 10.0);
        post_train.add_spike(3.0, Complex64::new(1.0, 0.0), 0.0); // 2ms later
        
        let stdp_change = engine.calculate_stdp(&pre_train, &post_train, 5.0);
        
        // Should be positive (LTP) since post comes after pre
        assert!(stdp_change > 0.0);
    }
    
    #[test]
    fn test_cerebellar_plasticity_rules() {
        // Test PF-PC LTD
        let ltd_change = CerebellarPlasticityRules::pf_pc_ltd(0.8, 0.9, 0.5, 0.01);
        assert!(ltd_change < 0.0); // Should be negative (depression)
        
        // Test MF-GC plasticity
        let mf_gc_change = CerebellarPlasticityRules::mf_gc_plasticity(0.7, 0.6, 0.01);
        assert!(mf_gc_change > 0.0); // Should be positive
        
        // Test Golgi inhibitory plasticity
        let golgi_change = CerebellarPlasticityRules::golgi_inhibitory_plasticity(
            0.8, 0.3, 0.4, 0.01
        );
        assert!(golgi_change > 0.0); // Should strengthen inhibition when pop is active
    }
    
    #[test]
    fn test_homeostatic_scaling() {
        let snn_config = QuantumSNNConfig::default();
        let mut engine = PlasticityEngine::new(&snn_config).unwrap();
        
        let mut synapses = vec![
            QuantumSynapse::new(0, 0, 1.0),
            QuantumSynapse::new(1, 0, 0.5),
        ];
        
        // Apply homeostatic scaling
        engine.apply_homeostatic_scaling(&mut synapses, 0.2, 0.8); // Target < current
        
        // Weights should be scaled down
        assert!(synapses[0].weight < 1.0);
        assert!(synapses[1].weight < 0.5);
    }
}