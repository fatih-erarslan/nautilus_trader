//! Complete cerebellar network integration
//! 
//! Orchestrates the full cerebellar microcircuit including mossy fibers,
//! granule cells, Purkinje cells, climbing fibers, and deep cerebellar nuclei.

use std::collections::HashMap;
use tch::{Tensor, Device};
use nalgebra::{DMatrix, DVector};
use anyhow::{Result, anyhow};
use tracing::{debug, info, warn};

use crate::{
    QuantumSNNConfig, QuantumSpikeTrain, QuantumSynapse,
    QuantumSpikeEncoder, GranuleCellPopulation, PurkinjeCellPopulation,
    PlasticityEngine, QuantumReservoir, CerebellarMetrics,
};

/// Configuration for complete cerebellar network
#[derive(Debug, Clone)]
pub struct CerebellarNetworkConfig {
    /// Input preprocessing
    pub input_normalization: bool,
    pub input_gain: f64,
    
    /// Network topology
    pub mossy_fiber_fanout: usize,
    pub parallel_fiber_length: usize,
    pub climbing_fiber_coverage: f64,
    
    /// Learning parameters
    pub adaptation_rate: f64,
    pub error_threshold: f64,
    pub performance_window: usize,
    
    /// Output generation
    pub dcn_integration_time: f64,
    pub output_smoothing: f64,
    
    /// Real-time constraints
    pub max_processing_time_us: u64,
    pub spike_buffer_size: usize,
}

impl Default for CerebellarNetworkConfig {
    fn default() -> Self {
        Self {
            input_normalization: true,
            input_gain: 1.0,
            mossy_fiber_fanout: 10,
            parallel_fiber_length: 3000, // Biological realistic
            climbing_fiber_coverage: 0.1,
            adaptation_rate: 0.005,
            error_threshold: 0.1,
            performance_window: 100,
            dcn_integration_time: 20.0,
            output_smoothing: 0.8,
            max_processing_time_us: 10000, // 10ms limit
            spike_buffer_size: 10000,
        }
    }
}

/// Inferior olive for error signal generation
#[derive(Debug)]
pub struct InferiorOlive {
    /// Olive cell states
    olive_cells: Vec<f64>,
    
    /// Error integration
    error_integrator: DVector<f64>,
    
    /// Complex spike generation
    complex_spike_threshold: f64,
    
    /// Oscillatory dynamics
    oscillation_phase: f64,
    oscillation_frequency: f64,
}

impl InferiorOlive {
    pub fn new(n_cells: usize) -> Self {
        Self {
            olive_cells: vec![0.0; n_cells],
            error_integrator: DVector::zeros(n_cells),
            complex_spike_threshold: 0.5,
            oscillation_phase: 0.0,
            oscillation_frequency: 10.0, // 10 Hz
        }
    }
    
    /// Generate climbing fiber signals from error
    pub fn generate_climbing_fibers(
        &mut self,
        error_signals: &[f64],
        current_time: f64,
    ) -> Vec<f64> {
        let mut cf_signals = vec![0.0; self.olive_cells.len()];
        
        // Update oscillatory phase
        self.oscillation_phase += self.oscillation_frequency * 0.001; // Assuming 1ms time steps
        
        for i in 0..self.olive_cells.len().min(error_signals.len()) {
            // Integrate error signal
            self.error_integrator[i] = self.error_integrator[i] * 0.95 + error_signals[i];
            
            // Generate complex spike if error exceeds threshold
            if self.error_integrator[i].abs() > self.complex_spike_threshold {
                // Modulate by intrinsic oscillation
                let oscillation = (self.oscillation_phase + i as f64 * 0.1).sin() * 0.3 + 0.7;
                cf_signals[i] = self.error_integrator[i].signum() * oscillation;
                
                // Reset after spike
                self.error_integrator[i] *= 0.5;
            }
        }
        
        cf_signals
    }
    
    /// Reset olive state
    pub fn reset(&mut self) {
        self.olive_cells.fill(0.0);
        self.error_integrator.fill(0.0);
        self.oscillation_phase = 0.0;
    }
}

/// Deep cerebellar nuclei for output generation
#[derive(Debug)]
pub struct DeepCerebellarNuclei {
    /// DCN cell states
    dcn_cells: Vec<f64>,
    
    /// Purkinje cell inhibition weights
    pc_weights: DMatrix<f64>,
    
    /// Mossy fiber excitation weights
    mf_weights: DMatrix<f64>,
    
    /// Output integration
    output_integrator: DVector<f64>,
    
    /// Baseline activity
    baseline_activity: f64,
}

impl DeepCerebellarNuclei {
    pub fn new(n_dcn_cells: usize, n_purkinje_cells: usize, n_mossy_fibers: usize) -> Self {
        // Initialize with realistic connectivity
        let pc_weights = DMatrix::from_fn(n_dcn_cells, n_purkinje_cells, |_, _| {
            -rand::random::<f64>() * 0.8 // Inhibitory from PC
        });
        
        let mf_weights = DMatrix::from_fn(n_dcn_cells, n_mossy_fibers, |_, _| {
            rand::random::<f64>() * 0.3 // Excitatory from MF collaterals
        });
        
        Self {
            dcn_cells: vec![0.5; n_dcn_cells], // Baseline activity
            pc_weights,
            mf_weights,
            output_integrator: DVector::zeros(n_dcn_cells),
            baseline_activity: 0.5,
        }
    }
    
    /// Generate DCN output from PC inhibition and MF excitation
    pub fn generate_output(
        &mut self,
        pc_activity: &[f64],
        mf_activity: &[f64],
        integration_time: f64,
    ) -> DVector<f64> {
        let n_dcn = self.dcn_cells.len();
        let mut dcn_input = DVector::from_element(n_dcn, self.baseline_activity);
        
        // Purkinje cell inhibition
        if pc_activity.len() == self.pc_weights.ncols() {
            let pc_vec = DVector::from_column_slice(pc_activity);
            dcn_input += &self.pc_weights * &pc_vec;
        }
        
        // Mossy fiber excitation (collaterals)
        if mf_activity.len() == self.mf_weights.ncols() {
            let mf_vec = DVector::from_column_slice(mf_activity);
            dcn_input += &self.mf_weights * &mf_vec;
        }
        
        // Temporal integration
        let integration_factor = 1.0 / integration_time;
        self.output_integrator = &self.output_integrator * (1.0 - integration_factor) +
                                 &dcn_input * integration_factor;
        
        // Apply activation function (ReLU with saturation)
        self.output_integrator.map(|x| x.max(0.0).min(2.0))
    }
    
    /// Adapt DCN weights based on performance
    pub fn adapt_weights(&mut self, performance_error: &[f64], learning_rate: f64) {
        for i in 0..self.dcn_cells.len().min(performance_error.len()) {
            let error = performance_error[i];
            
            // Adapt PC weights (anti-Hebbian for inhibitory connections)
            for j in 0..self.pc_weights.ncols() {
                self.pc_weights[(i, j)] += learning_rate * error * 0.1;
                self.pc_weights[(i, j)] = self.pc_weights[(i, j)].clamp(-2.0, 0.0);
            }
            
            // Adapt MF weights (Hebbian for excitatory connections)
            for j in 0..self.mf_weights.ncols() {
                self.mf_weights[(i, j)] -= learning_rate * error * 0.05;
                self.mf_weights[(i, j)] = self.mf_weights[(i, j)].clamp(0.0, 1.0);
            }
        }
    }
    
    /// Reset DCN state
    pub fn reset(&mut self) {
        self.dcn_cells.fill(self.baseline_activity);
        self.output_integrator.fill(0.0);
    }
}

/// Complete cerebellar network orchestrator
pub struct CerebellarNetwork {
    /// Configuration
    config: CerebellarNetworkConfig,
    
    /// Network components
    spike_encoder: QuantumSpikeEncoder,
    granule_population: GranuleCellPopulation,
    purkinje_population: PurkinjeCellPopulation,
    plasticity_engine: PlasticityEngine,
    quantum_reservoir: QuantumReservoir,
    
    /// Subcortical structures
    inferior_olive: InferiorOlive,
    deep_cerebellar_nuclei: DeepCerebellarNuclei,
    
    /// Network state
    current_time: f64,
    mossy_fiber_activity: Vec<f64>,
    parallel_fiber_activity: Vec<QuantumSpikeTrain>,
    purkinje_activity: Vec<QuantumSpikeTrain>,
    climbing_fiber_signals: Vec<f64>,
    
    /// Performance tracking
    metrics: CerebellarMetrics,
    performance_history: Vec<f64>,
    
    /// Device for tensor operations
    device: Device,
}

impl CerebellarNetwork {
    /// Create complete cerebellar network
    pub fn new(snn_config: QuantumSNNConfig, network_config: CerebellarNetworkConfig) -> Result<Self> {
        info!("Initializing complete cerebellar network");
        
        let device = snn_config.device;
        
        // Initialize components
        let spike_encoder = QuantumSpikeEncoder::new(&snn_config)?;
        let granule_population = GranuleCellPopulation::new(
            snn_config.n_granule_cells,
            snn_config.n_mossy_fibers,
            &snn_config,
        )?;
        let purkinje_population = PurkinjeCellPopulation::new(
            snn_config.n_purkinje_cells,
            snn_config.n_granule_cells,
            &snn_config,
        )?;
        let plasticity_engine = PlasticityEngine::new(&snn_config)?;
        let quantum_reservoir = QuantumReservoir::new(&snn_config)?;
        
        // Initialize subcortical structures
        let inferior_olive = InferiorOlive::new(snn_config.n_purkinje_cells);
        let deep_cerebellar_nuclei = DeepCerebellarNuclei::new(
            3, // 3 DCN output dimensions
            snn_config.n_purkinje_cells,
            snn_config.n_mossy_fibers,
        );
        
        info!("Cerebellar network initialized with {} granule cells, {} Purkinje cells",
              snn_config.n_granule_cells, snn_config.n_purkinje_cells);
        
        Ok(Self {
            config: network_config,
            spike_encoder,
            granule_population,
            purkinje_population,
            plasticity_engine,
            quantum_reservoir,
            inferior_olive,
            deep_cerebellar_nuclei,
            current_time: 0.0,
            mossy_fiber_activity: vec![0.0; snn_config.n_mossy_fibers],
            parallel_fiber_activity: Vec::new(),
            purkinje_activity: Vec::new(),
            climbing_fiber_signals: vec![0.0; snn_config.n_purkinje_cells],
            metrics: CerebellarMetrics::default(),
            performance_history: Vec::with_capacity(1000),
            device,
        })
    }
    
    /// Process input through complete cerebellar network
    pub fn forward(&mut self, input: &Tensor, error_signal: Option<&Tensor>) -> Result<Tensor> {
        let start_time = std::time::Instant::now();
        
        // Check processing time constraint
        if start_time.elapsed().as_micros() > self.config.max_processing_time_us {
            warn!("Processing time exceeded limit, skipping complex processing");
            return self.generate_default_output(input);
        }
        
        // Stage 1: Input preprocessing and mossy fiber encoding
        let preprocessed_input = self.preprocess_input(input)?;
        let mossy_fiber_spikes = self.spike_encoder.encode(&preprocessed_input)?;
        
        // Update mossy fiber activity
        self.mossy_fiber_activity = self.convert_spikes_to_activity(&mossy_fiber_spikes);
        
        // Stage 2: Granule cell processing (sparse coding)
        self.parallel_fiber_activity = self.granule_population.process(
            &mossy_fiber_spikes,
            self.current_time,
        )?;
        
        // Stage 3: Error signal processing via inferior olive
        if let Some(error) = error_signal {
            let error_vec = self.tensor_to_vec(error)?;
            self.climbing_fiber_signals = self.inferior_olive.generate_climbing_fibers(
                &error_vec,
                self.current_time,
            );
            
            // Update Purkinje cell plasticity
            self.purkinje_population.update_plasticity(
                &self.climbing_fiber_signals,
                self.current_time,
            )?;
        }
        
        // Stage 4: Purkinje cell processing
        self.purkinje_activity = self.purkinje_population.process(
            &self.parallel_fiber_activity,
            self.current_time,
        )?;
        
        // Stage 5: Plasticity updates
        self.update_network_plasticity()?;
        
        // Stage 6: Quantum reservoir dynamics
        let reservoir_state = self.quantum_reservoir.update(
            &self.purkinje_activity,
            self.current_time,
        )?;
        
        // Stage 7: Deep cerebellar nuclei output generation
        let pc_activities = self.convert_spikes_to_activity(&self.purkinje_activity);
        let dcn_output = self.deep_cerebellar_nuclei.generate_output(
            &pc_activities,
            &self.mossy_fiber_activity,
            self.config.dcn_integration_time,
        );
        
        // Stage 8: Final output generation
        let final_output = self.generate_final_output(&dcn_output, &reservoir_state)?;
        
        // Update metrics
        let processing_time = start_time.elapsed();
        self.metrics.processing_time_ns = processing_time.as_nanos() as u64;
        self.metrics.total_spikes += self.count_total_spikes();
        
        // Update performance history
        if let Some(error) = error_signal {
            let performance = error.abs().mean(tch::Kind::Float).double_value(&[]);
            self.performance_history.push(performance);
            if self.performance_history.len() > self.config.performance_window {
                self.performance_history.remove(0);
            }
        }
        
        // Advance time
        self.current_time += 1.0;
        
        debug!("Cerebellar forward pass completed in {}μs", processing_time.as_micros());
        
        Ok(final_output)
    }
    
    /// Preprocess input tensor
    fn preprocess_input(&self, input: &Tensor) -> Result<Tensor> {
        let mut processed = input.to_device(self.device);
        
        if self.config.input_normalization {
            // Z-score normalization
            let mean = processed.mean(tch::Kind::Float);
            let std = processed.std(false);
            processed = (processed - mean) / (std + 1e-8);
        }
        
        // Apply input gain
        processed = processed * self.config.input_gain;
        
        Ok(processed)
    }
    
    /// Convert spike trains to activity levels
    fn convert_spikes_to_activity(&self, spike_trains: &[QuantumSpikeTrain]) -> Vec<f64> {
        spike_trains.iter().map(|st| {
            if st.is_empty() {
                0.0
            } else {
                // Recent spike strength
                let recent_spikes = st.times.iter()
                    .filter(|&&t| self.current_time - t < 5.0)
                    .count();
                (recent_spikes as f64 / 5.0).min(1.0)
            }
        }).collect()
    }
    
    /// Convert tensor to vector
    fn tensor_to_vec(&self, tensor: &Tensor) -> Result<Vec<f64>> {
        let data: Vec<f64> = tensor.to_device(Device::Cpu)
            .to_kind(tch::Kind::Double)
            .into();
        Ok(data)
    }
    
    /// Update network plasticity
    fn update_network_plasticity(&mut self) -> Result<()> {
        // Convert activities to spike trains for plasticity engine
        let granule_spikes = &self.parallel_fiber_activity;
        let purkinje_spikes = &self.purkinje_activity;
        
        // Update all network synapses (simplified - would need actual synapse lists)
        // self.plasticity_engine.update_all_synapses(...)?;
        
        // Apply cerebellar-specific plasticity rules
        // self.plasticity_engine.apply_cerebellar_plasticity(...)?;
        
        Ok(())
    }
    
    /// Generate final output from DCN and reservoir
    fn generate_final_output(
        &self,
        dcn_output: &DVector<f64>,
        reservoir_state: &Tensor,
    ) -> Result<Tensor> {
        // Combine DCN output with reservoir state
        let dcn_tensor = Tensor::from_slice(dcn_output.as_slice())
            .to_device(self.device)
            .unsqueeze(0);
        
        // Simple concatenation (could be more sophisticated)
        let reservoir_slice = reservoir_state.narrow(1, 0, 3);
        let combined = Tensor::cat(&[dcn_tensor, reservoir_slice], 1);
        
        // Apply output smoothing
        if self.performance_history.len() > 1 {
            let smoothing = self.config.output_smoothing;
            let prev_output = Tensor::from_slice(&[self.performance_history[self.performance_history.len()-1]])
                .to_device(self.device);
            Ok(combined * (1.0 - smoothing) + prev_output * smoothing)
        } else {
            Ok(combined)
        }
    }
    
    /// Generate default output when processing time is exceeded
    fn generate_default_output(&self, input: &Tensor) -> Result<Tensor> {
        // Simple passthrough with minimal processing
        let output = input.mean_dim(&[1], false, tch::Kind::Float);
        Ok(output.unsqueeze(1))
    }
    
    /// Count total spikes in network
    fn count_total_spikes(&self) -> u64 {
        let granule_spikes: usize = self.parallel_fiber_activity.iter()
            .map(|st| st.len())
            .sum();
        
        let purkinje_spikes: usize = self.purkinje_activity.iter()
            .map(|st| st.len())
            .sum();
        
        (granule_spikes + purkinje_spikes) as u64
    }
    
    /// Adapt network based on performance
    pub fn adapt_network(&mut self, performance_error: &[f64]) -> Result<()> {
        // Adapt DCN weights
        self.deep_cerebellar_nuclei.adapt_weights(
            performance_error,
            self.config.adaptation_rate,
        );
        
        // Adapt granule cell population sparsity
        let target_sparsity = 0.05; // 5% target sparsity
        self.granule_population.adapt_population(target_sparsity)?;
        
        // Apply homeostatic scaling
        let current_activity = self.performance_history.last().copied().unwrap_or(0.0);
        let target_activity = 0.3;
        // self.plasticity_engine.apply_homeostatic_scaling(..., target_activity, current_activity);
        
        info!("Network adaptation applied based on performance error");
        Ok(())
    }
    
    /// Get network performance metrics
    pub fn get_performance_metrics(&self) -> HashMap<String, f64> {
        let mut metrics = HashMap::new();
        
        metrics.insert("processing_time_us".to_string(), 
                      self.metrics.processing_time_ns as f64 / 1000.0);
        metrics.insert("total_spikes".to_string(), self.metrics.total_spikes as f64);
        metrics.insert("granule_sparsity".to_string(), self.granule_population.sparsity());
        metrics.insert("purkinje_firing_rate".to_string(), 
                      self.purkinje_population.population_firing_rate(100.0, self.current_time));
        metrics.insert("quantum_coherence".to_string(), 
                      self.granule_population.population_coherence());
        
        // Performance history statistics
        if !self.performance_history.is_empty() {
            let recent_perf: f64 = self.performance_history.iter()
                .rev()
                .take(10)
                .sum::<f64>() / 10.0.min(self.performance_history.len() as f64);
            metrics.insert("recent_performance".to_string(), recent_perf);
        }
        
        metrics
    }
    
    /// Reset entire network state
    pub fn reset(&mut self) {
        self.current_time = 0.0;
        self.mossy_fiber_activity.fill(0.0);
        self.parallel_fiber_activity.clear();
        self.purkinje_activity.clear();
        self.climbing_fiber_signals.fill(0.0);
        self.performance_history.clear();
        
        // Reset all components
        self.granule_population.reset();
        self.purkinje_population.reset();
        self.plasticity_engine.reset();
        self.quantum_reservoir.reset();
        self.inferior_olive.reset();
        self.deep_cerebellar_nuclei.reset();
        
        self.metrics = CerebellarMetrics::default();
        
        info!("Cerebellar network reset to initial state");
    }
    
    /// Save network state for checkpointing
    pub fn save_checkpoint(&self) -> Result<Vec<u8>> {
        // Simplified checkpoint - would serialize all component states
        let checkpoint = format!(
            "cerebellar_checkpoint:time={},spikes={}",
            self.current_time,
            self.metrics.total_spikes
        );
        Ok(checkpoint.into_bytes())
    }
    
    /// Load network state from checkpoint
    pub fn load_checkpoint(&mut self, data: &[u8]) -> Result<()> {
        let checkpoint = String::from_utf8_lossy(data);
        info!("Loading checkpoint: {}", checkpoint);
        // Would deserialize all component states
        Ok(())
    }
    
    /// Real-time processing mode with strict timing
    pub fn process_realtime(&mut self, input: &Tensor, max_latency_us: u64) -> Result<Tensor> {
        let start = std::time::Instant::now();
        
        // Set strict time limit
        let old_limit = self.config.max_processing_time_us;
        self.config.max_processing_time_us = max_latency_us;
        
        let result = self.forward(input, None);
        
        // Restore original limit
        self.config.max_processing_time_us = old_limit;
        
        let elapsed = start.elapsed().as_micros() as u64;
        if elapsed > max_latency_us {
            warn!("Real-time processing exceeded latency limit: {}μs > {}μs", 
                  elapsed, max_latency_us);
        }
        
        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_inferior_olive() {
        let mut olive = InferiorOlive::new(3);
        
        let errors = vec![0.8, -0.3, 0.6];
        let cf_signals = olive.generate_climbing_fibers(&errors, 1.0);
        
        assert_eq!(cf_signals.len(), 3);
        // Should generate strong signal for large errors
        assert!(cf_signals[0].abs() > 0.0);
    }
    
    #[test]
    fn test_deep_cerebellar_nuclei() {
        let mut dcn = DeepCerebellarNuclei::new(2, 3, 4);
        
        let pc_activity = vec![0.8, 0.3, 0.6];
        let mf_activity = vec![0.7, 0.4, 0.9, 0.2];
        
        let output = dcn.generate_output(&pc_activity, &mf_activity, 10.0);
        
        assert_eq!(output.len(), 2);
        // DCN should have some baseline activity
        assert!(output.iter().any(|&x| x > 0.0));
    }
    
    #[test]
    fn test_cerebellar_network_creation() {
        let snn_config = QuantumSNNConfig::default();
        let network_config = CerebellarNetworkConfig::default();
        
        let network = CerebellarNetwork::new(snn_config, network_config).unwrap();
        
        assert_eq!(network.current_time, 0.0);
        assert!(network.mossy_fiber_activity.len() > 0);
    }
    
    #[test]
    fn test_network_forward_pass() {
        let mut snn_config = QuantumSNNConfig::default();
        snn_config.n_granule_cells = 20;
        snn_config.n_purkinje_cells = 5;
        snn_config.n_mossy_fibers = 10;
        
        let network_config = CerebellarNetworkConfig::default();
        let mut network = CerebellarNetwork::new(snn_config, network_config).unwrap();
        
        let input = Tensor::randn(&[1, 8], tch::Kind::Float);
        let output = network.forward(&input, None).unwrap();
        
        assert!(output.size().len() > 0);
        assert!(network.current_time > 0.0);
    }
    
    #[test]
    fn test_network_with_error_signal() {
        let mut snn_config = QuantumSNNConfig::default();
        snn_config.n_granule_cells = 10;
        snn_config.n_purkinje_cells = 3;
        
        let network_config = CerebellarNetworkConfig::default();
        let mut network = CerebellarNetwork::new(snn_config, network_config).unwrap();
        
        let input = Tensor::randn(&[1, 8], tch::Kind::Float);
        let error = Tensor::randn(&[1, 3], tch::Kind::Float);
        
        let output = network.forward(&input, Some(&error)).unwrap();
        
        assert!(output.size().len() > 0);
        // Should have climbing fiber activity
        assert!(network.climbing_fiber_signals.iter().any(|&x| x.abs() > 0.0));
    }
    
    #[test]
    fn test_network_adaptation() {
        let snn_config = QuantumSNNConfig::default();
        let network_config = CerebellarNetworkConfig::default();
        let mut network = CerebellarNetwork::new(snn_config, network_config).unwrap();
        
        let performance_error = vec![0.5, -0.3, 0.8];
        network.adapt_network(&performance_error).unwrap();
        
        // Network should adapt without errors
        assert!(true);
    }
    
    #[test]
    fn test_realtime_processing() {
        let snn_config = QuantumSNNConfig::default();
        let network_config = CerebellarNetworkConfig::default();
        let mut network = CerebellarNetwork::new(snn_config, network_config).unwrap();
        
        let input = Tensor::randn(&[1, 8], tch::Kind::Float);
        let output = network.process_realtime(&input, 5000).unwrap(); // 5ms limit
        
        assert!(output.size().len() > 0);
    }
    
    #[test]
    fn test_performance_metrics() {
        let snn_config = QuantumSNNConfig::default();
        let network_config = CerebellarNetworkConfig::default();
        let network = CerebellarNetwork::new(snn_config, network_config).unwrap();
        
        let metrics = network.get_performance_metrics();
        
        assert!(metrics.contains_key("processing_time_us"));
        assert!(metrics.contains_key("total_spikes"));
        assert!(metrics.contains_key("granule_sparsity"));
    }
    
    #[test]
    fn test_network_reset() {
        let snn_config = QuantumSNNConfig::default();
        let network_config = CerebellarNetworkConfig::default();
        let mut network = CerebellarNetwork::new(snn_config, network_config).unwrap();
        
        // Process some input to change state
        let input = Tensor::randn(&[1, 8], tch::Kind::Float);
        let _ = network.forward(&input, None).unwrap();
        
        let time_before_reset = network.current_time;
        network.reset();
        
        assert_eq!(network.current_time, 0.0);
        assert!(network.current_time < time_before_reset);
    }
}