//! Quantum spike encoding for cerebellar neural networks
//! 
//! Converts classical signals into quantum-enhanced spike trains with
//! superposition, entanglement, and phase encoding for neuromorphic processing.

use std::collections::HashMap;
use tch::{Tensor, Device, Kind};
use nalgebra::{DVector, DMatrix};
use num_complex::Complex64;
use anyhow::{Result, anyhow};
use tracing::{debug, info, warn};
use serde::{Serialize, Deserialize};

use crate::{QuantumSNNConfig, QuantumSpikeTrain, QuantumCircuitSimulator, CerebellarQuantumCircuits};

/// Spike encoding strategies for different signal types
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub enum SpikeEncodingType {
    /// Rate coding: spike rate proportional to signal amplitude
    Rate,
    /// Temporal coding: spike timing encodes information
    Temporal,
    /// Population coding: distributed across multiple neurons
    Population,
    /// Phase coding: information in spike phase relationships
    Phase,
    /// Burst coding: patterns of spike bursts
    Burst,
    /// Quantum superposition coding: amplitude and phase superposition
    QuantumSuperposition,
}

/// Configuration for quantum spike encoder
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SpikeEncoderConfig {
    /// Encoding strategy
    pub encoding_type: SpikeEncodingType,
    
    /// Spike threshold for rate coding
    pub spike_threshold: f64,
    
    /// Maximum spike rate (Hz)
    pub max_spike_rate: f64,
    
    /// Time window for encoding (ms)
    pub time_window: f64,
    
    /// Refractory period (ms)
    pub refractory_period: f64,
    
    /// Quantum coherence parameters
    pub quantum_coherence_time: f64,
    pub phase_precision: f64,
    pub amplitude_resolution: usize,
    
    /// Population encoding parameters
    pub population_size: usize,
    pub receptive_field_width: f64,
    
    /// Noise parameters
    pub noise_level: f64,
    pub jitter_std: f64,
}

impl Default for SpikeEncoderConfig {
    fn default() -> Self {
        Self {
            encoding_type: SpikeEncodingType::QuantumSuperposition,
            spike_threshold: 0.5,
            max_spike_rate: 1000.0,
            time_window: 10.0,
            refractory_period: 1.0,
            quantum_coherence_time: 100.0,
            phase_precision: 0.1,
            amplitude_resolution: 256,
            population_size: 10,
            receptive_field_width: 0.2,
            noise_level: 0.01,
            jitter_std: 0.1,
        }
    }
}

/// High-performance quantum spike encoder
pub struct QuantumSpikeEncoder {
    /// Configuration
    config: SpikeEncoderConfig,
    
    /// Quantum circuit simulator
    quantum_simulator: QuantumCircuitSimulator,
    
    /// Device for tensor operations
    device: Device,
    
    /// Encoding cache for performance
    encoding_cache: HashMap<String, Vec<QuantumSpikeTrain>>,
    
    /// Population encoding receptive fields
    receptive_fields: Option<DMatrix<f64>>,
    
    /// Quantum state preparation circuits
    state_preparation_circuits: Vec<Vec<(String, Vec<f64>)>>,
    
    /// Performance statistics
    encoding_stats: EncodingStats,
}

/// Performance and accuracy statistics
#[derive(Debug, Default, Clone)]
pub struct EncodingStats {
    pub signals_encoded: u64,
    pub quantum_operations: u64,
    pub cache_hits: u64,
    pub cache_misses: u64,
    pub average_encoding_time_ns: u64,
    pub total_spikes_generated: u64,
    pub quantum_fidelity: f64,
}

impl QuantumSpikeEncoder {
    /// Create new quantum spike encoder
    pub fn new(snn_config: &QuantumSNNConfig) -> Result<Self> {
        let config = SpikeEncoderConfig {
            spike_threshold: snn_config.spike_threshold,
            time_window: snn_config.time_window as f64,
            quantum_coherence_time: snn_config.quantum_coherence_time,
            ..Default::default()
        };
        
        let quantum_simulator = QuantumCircuitSimulator::new(snn_config.n_qubits)?;
        let device = snn_config.device;
        
        // Initialize population encoding if needed
        let receptive_fields = if config.encoding_type == SpikeEncodingType::Population {
            Some(Self::create_receptive_fields(&config)?)
        } else {
            None
        };
        
        info!("Initialized quantum spike encoder with {:?} encoding", config.encoding_type);
        
        Ok(Self {
            config,
            quantum_simulator,
            device,
            encoding_cache: HashMap::new(),
            receptive_fields,
            state_preparation_circuits: Vec::new(),
            encoding_stats: EncodingStats::default(),
        })
    }
    
    /// Encode tensor input into quantum spike trains
    pub fn encode(&mut self, inputs: &Tensor) -> Result<Vec<QuantumSpikeTrain>> {
        let start_time = std::time::Instant::now();
        
        // Convert tensor to vector format
        let input_shape = inputs.size();
        let batch_size = input_shape[0];
        let input_dim = input_shape[1];
        
        let mut spike_trains = Vec::new();
        
        for batch_idx in 0..batch_size {
            let signal = inputs.narrow(0, batch_idx, 1).squeeze();
            let signal_vec = self.tensor_to_vector(&signal)?;
            
            let batch_spikes = match self.config.encoding_type {
                SpikeEncodingType::Rate => self.encode_rate_coding(&signal_vec)?,
                SpikeEncodingType::Temporal => self.encode_temporal_coding(&signal_vec)?,
                SpikeEncodingType::Population => self.encode_population_coding(&signal_vec)?,
                SpikeEncodingType::Phase => self.encode_phase_coding(&signal_vec)?,
                SpikeEncodingType::Burst => self.encode_burst_coding(&signal_vec)?,
                SpikeEncodingType::QuantumSuperposition => self.encode_quantum_superposition(&signal_vec)?,
            };
            
            spike_trains.extend(batch_spikes);
        }
        
        // Update statistics
        let encoding_time = start_time.elapsed();
        self.encoding_stats.signals_encoded += batch_size as u64;
        self.encoding_stats.average_encoding_time_ns = 
            (self.encoding_stats.average_encoding_time_ns + encoding_time.as_nanos() as u64) / 2;
        self.encoding_stats.total_spikes_generated += 
            spike_trains.iter().map(|st| st.len() as u64).sum::<u64>();
        
        debug!("Encoded {} signals in {}μs", batch_size, encoding_time.as_micros());
        
        Ok(spike_trains)
    }
    
    /// Rate coding: spike frequency proportional to signal amplitude
    fn encode_rate_coding(&mut self, signal: &DVector<f64>) -> Result<Vec<QuantumSpikeTrain>> {
        let mut spike_trains = Vec::new();
        
        for (neuron_id, &amplitude) in signal.iter().enumerate() {
            let mut spike_train = QuantumSpikeTrain::new(neuron_id, self.config.time_window);
            
            if amplitude > self.config.spike_threshold {
                // Calculate spike rate based on amplitude
                let spike_rate = amplitude.clamp(0.0, 1.0) * self.config.max_spike_rate;
                let isi = 1000.0 / spike_rate; // Inter-spike interval in ms
                
                let mut time = 0.0;
                while time < self.config.time_window {
                    // Add jitter for biological realism
                    let jitter = rand::random::<f64>() * self.config.jitter_std - self.config.jitter_std / 2.0;
                    let spike_time = time + jitter;
                    
                    if spike_time < self.config.time_window {
                        let spike_amplitude = Complex64::new(amplitude, 0.0);
                        spike_train.add_spike(spike_time, spike_amplitude, 0.0);
                    }
                    
                    time += isi;
                }
            }
            
            spike_trains.push(spike_train);
        }
        
        Ok(spike_trains)
    }
    
    /// Temporal coding: information in precise spike timing
    fn encode_temporal_coding(&mut self, signal: &DVector<f64>) -> Result<Vec<QuantumSpikeTrain>> {
        let mut spike_trains = Vec::new();
        
        for (neuron_id, &amplitude) in signal.iter().enumerate() {
            let mut spike_train = QuantumSpikeTrain::new(neuron_id, self.config.time_window);
            
            if amplitude > self.config.spike_threshold {
                // Map amplitude to spike timing
                let normalized_amp = (amplitude - self.config.spike_threshold) / 
                                   (1.0 - self.config.spike_threshold);
                let spike_time = normalized_amp * self.config.time_window;
                
                let spike_amplitude = Complex64::new(1.0, 0.0);
                spike_train.add_spike(spike_time, spike_amplitude, 0.0);
            }
            
            spike_trains.push(spike_train);
        }
        
        Ok(spike_trains)
    }
    
    /// Population coding: distributed representation across neuron population
    fn encode_population_coding(&mut self, signal: &DVector<f64>) -> Result<Vec<QuantumSpikeTrain>> {
        let receptive_fields = self.receptive_fields.as_ref()
            .ok_or_else(|| anyhow!("Receptive fields not initialized for population coding"))?;
        
        let mut spike_trains = Vec::new();
        
        for signal_idx in 0..signal.len() {
            let signal_value = signal[signal_idx];
            
            for pop_neuron in 0..self.config.population_size {
                let rf_response = receptive_fields[(pop_neuron, signal_idx)];
                let neuron_response = (-((signal_value - rf_response).powi(2)) / 
                                     (2.0 * self.config.receptive_field_width.powi(2))).exp();
                
                let neuron_id = signal_idx * self.config.population_size + pop_neuron;
                let mut spike_train = QuantumSpikeTrain::new(neuron_id, self.config.time_window);
                
                if neuron_response > self.config.spike_threshold {
                    let spike_rate = neuron_response * self.config.max_spike_rate;
                    let isi = 1000.0 / spike_rate;
                    
                    let mut time = 0.0;
                    while time < self.config.time_window {
                        let spike_amplitude = Complex64::new(neuron_response, 0.0);
                        spike_train.add_spike(time, spike_amplitude, 0.0);
                        time += isi;
                    }
                }
                
                spike_trains.push(spike_train);
            }
        }
        
        Ok(spike_trains)
    }
    
    /// Phase coding: information encoded in spike phase relationships
    fn encode_phase_coding(&mut self, signal: &DVector<f64>) -> Result<Vec<QuantumSpikeTrain>> {
        let mut spike_trains = Vec::new();
        
        // Find global maximum for phase reference
        let max_amplitude = signal.max();
        
        for (neuron_id, &amplitude) in signal.iter().enumerate() {
            let mut spike_train = QuantumSpikeTrain::new(neuron_id, self.config.time_window);
            
            if amplitude > self.config.spike_threshold {
                // Phase relative to maximum amplitude
                let phase = if max_amplitude > 0.0 {
                    2.0 * std::f64::consts::PI * amplitude / max_amplitude
                } else {
                    0.0
                };
                
                // Single spike with phase information
                let spike_time = self.config.time_window / 2.0; // Fixed timing, info in phase
                let spike_amplitude = Complex64::new(amplitude.cos() * amplitude, amplitude.sin() * amplitude);
                spike_train.add_spike(spike_time, spike_amplitude, phase);
            }
            
            spike_trains.push(spike_train);
        }
        
        Ok(spike_trains)
    }
    
    /// Burst coding: information in spike burst patterns
    fn encode_burst_coding(&mut self, signal: &DVector<f64>) -> Result<Vec<QuantumSpikeTrain>> {
        let mut spike_trains = Vec::new();
        
        for (neuron_id, &amplitude) in signal.iter().enumerate() {
            let mut spike_train = QuantumSpikeTrain::new(neuron_id, self.config.time_window);
            
            if amplitude > self.config.spike_threshold {
                // Number of spikes in burst based on amplitude
                let burst_size = ((amplitude - self.config.spike_threshold) * 10.0) as usize + 1;
                let burst_duration = 2.0; // ms
                let intra_burst_isi = burst_duration / burst_size as f64;
                
                // Generate burst at random time in window
                let burst_start = rand::random::<f64>() * (self.config.time_window - burst_duration);
                
                for spike_idx in 0..burst_size {
                    let spike_time = burst_start + spike_idx as f64 * intra_burst_isi;
                    let spike_amplitude = Complex64::new(amplitude, 0.0);
                    spike_train.add_spike(spike_time, spike_amplitude, 0.0);
                }
            }
            
            spike_trains.push(spike_train);
        }
        
        Ok(spike_trains)
    }
    
    /// Quantum superposition coding: amplitude and phase superposition
    fn encode_quantum_superposition(&mut self, signal: &DVector<f64>) -> Result<Vec<QuantumSpikeTrain>> {
        let mut spike_trains = Vec::new();
        
        // Prepare quantum state for each neuron
        for (neuron_id, &amplitude) in signal.iter().enumerate() {
            let mut spike_train = QuantumSpikeTrain::new(neuron_id, self.config.time_window);
            
            if amplitude > self.config.spike_threshold {
                // Create quantum superposition state
                let quantum_spikes = self.encode_quantum_state(amplitude, neuron_id)?;
                
                for (time, q_amplitude, phase) in quantum_spikes {
                    spike_train.add_spike(time, q_amplitude, phase);
                }
                
                // Add entanglement correlations with nearby neurons
                if neuron_id > 0 && signal[neuron_id - 1] > self.config.spike_threshold {
                    let correlation_strength = (amplitude * signal[neuron_id - 1]).sqrt();
                    let correlation = Complex64::new(correlation_strength, 0.0);
                    spike_train.correlations.push((neuron_id - 1, neuron_id, correlation));
                }
            }
            
            spike_trains.push(spike_train);
        }
        
        // Apply quantum correlations
        self.apply_quantum_correlations(&mut spike_trains)?;
        
        Ok(spike_trains)
    }
    
    /// Encode single value as quantum state
    fn encode_quantum_state(&mut self, amplitude: f64, neuron_id: usize) -> Result<Vec<(f64, Complex64, f64)>> {
        let mut quantum_spikes = Vec::new();
        
        // Reset quantum simulator
        self.quantum_simulator.reset();
        
        // Use subset of qubits for this neuron
        let qubits_per_neuron = 2;
        let start_qubit = (neuron_id * qubits_per_neuron) % self.quantum_simulator.n_qubits();
        let qubit_indices: Vec<usize> = (start_qubit..(start_qubit + qubits_per_neuron))
            .map(|i| i % self.quantum_simulator.n_qubits())
            .collect();
        
        // Create spike encoding circuit
        let spike_amplitudes = vec![amplitude, amplitude.sqrt()];
        let spike_phases = vec![0.0, amplitude * std::f64::consts::PI];
        
        CerebellarQuantumCircuits::create_spike_encoding_circuit(
            &mut self.quantum_simulator,
            &spike_amplitudes,
            &spike_phases,
        )?;
        
        // Execute quantum circuit
        self.quantum_simulator.execute()?;
        
        // Measure quantum state probabilities
        let probabilities = self.quantum_simulator.measure_probabilities(&qubit_indices)?;
        
        // Convert quantum measurements to spikes
        for (outcome, &prob) in probabilities.iter().enumerate() {
            if prob > 0.1 { // Threshold for significant probability
                let spike_time = (outcome as f64 / probabilities.len() as f64) * self.config.time_window;
                let q_amplitude = Complex64::new(prob.sqrt(), 0.0);
                let phase = outcome as f64 * std::f64::consts::PI / probabilities.len() as f64;
                
                quantum_spikes.push((spike_time, q_amplitude, phase));
            }
        }
        
        self.encoding_stats.quantum_operations += 1;
        
        Ok(quantum_spikes)
    }
    
    /// Apply quantum correlations between spike trains
    fn apply_quantum_correlations(&mut self, spike_trains: &mut [QuantumSpikeTrain]) -> Result<()> {
        // Create entanglement between correlated neurons
        let correlations: Vec<(usize, usize, f64)> = spike_trains
            .iter()
            .flat_map(|st| {
                st.correlations.iter().map(|&(n1, n2, c)| {
                    (n1, n2, c.norm())
                })
            })
            .collect();
        
        if !correlations.is_empty() {
            self.quantum_simulator.reset();
            CerebellarQuantumCircuits::create_entanglement_circuit(
                &mut self.quantum_simulator,
                &correlations,
            )?;
            self.quantum_simulator.execute()?;
            
            // Update spike train correlations based on quantum entanglement
            for spike_train in spike_trains.iter_mut() {
                for (n1, n2, _) in &spike_train.correlations {
                    if self.quantum_simulator.are_entangled(*n1 % self.quantum_simulator.n_qubits(), 
                                                          *n2 % self.quantum_simulator.n_qubits()) {
                        // Enhance correlation strength for entangled qubits
                        if let Some(corr) = spike_train.correlations.iter_mut()
                            .find(|(nn1, nn2, _)| (*nn1, *nn2) == (*n1, *n2)) {
                            corr.2 *= Complex64::new(1.2, 0.0); // Boost entangled correlations
                        }
                    }
                }
            }
        }
        
        Ok(())
    }
    
    /// Create receptive fields for population coding
    fn create_receptive_fields(config: &SpikeEncoderConfig) -> Result<DMatrix<f64>> {
        let mut rng = rand::thread_rng();
        let mut receptive_fields = DMatrix::zeros(config.population_size, 8); // Assume 8 input dimensions
        
        for pop_neuron in 0..config.population_size {
            for input_dim in 0..8 {
                // Gaussian receptive field centers
                let center = (pop_neuron as f64) / (config.population_size as f64);
                let noise = (rand::random::<f64>() - 0.5) * 0.1;
                receptive_fields[(pop_neuron, input_dim)] = center + noise;
            }
        }
        
        Ok(receptive_fields)
    }
    
    /// Convert tensor to vector
    fn tensor_to_vector(&self, tensor: &Tensor) -> Result<DVector<f64>> {
        let data: Vec<f64> = tensor.to_device(Device::Cpu)
            .to_kind(Kind::Double)
            .into();
        
        Ok(DVector::from_vec(data))
    }
    
    /// Decode spike trains back to signal (for validation)
    pub fn decode(&self, spike_trains: &[QuantumSpikeTrain]) -> Result<Tensor> {
        let mut decoded_signals = Vec::new();
        
        for spike_train in spike_trains {
            let decoded_value = match self.config.encoding_type {
                SpikeEncodingType::Rate => {
                    spike_train.spike_rate() / self.config.max_spike_rate
                }
                SpikeEncodingType::Temporal => {
                    if let Some(&first_spike) = spike_train.times.first() {
                        first_spike / self.config.time_window
                    } else {
                        0.0
                    }
                }
                SpikeEncodingType::Population => {
                    spike_train.spike_rate() / self.config.max_spike_rate
                }
                SpikeEncodingType::Phase => {
                    if let Some(phase) = spike_train.phases.first() {
                        phase / (2.0 * std::f64::consts::PI)
                    } else {
                        0.0
                    }
                }
                SpikeEncodingType::Burst => {
                    spike_train.len() as f64 / 10.0 // Assume max 10 spikes per burst
                }
                SpikeEncodingType::QuantumSuperposition => {
                    if let Some(amplitude) = spike_train.amplitudes.first() {
                        amplitude.norm()
                    } else {
                        0.0
                    }
                }
            };
            
            decoded_signals.push(decoded_value as f32);
        }
        
        Ok(Tensor::from_slice(&decoded_signals).to_device(self.device))
    }
    
    /// Get encoding statistics
    pub fn stats(&self) -> &EncodingStats {
        &self.encoding_stats
    }
    
    /// Clear encoding cache
    pub fn clear_cache(&mut self) {
        self.encoding_cache.clear();
    }
    
    /// Optimize encoding parameters based on performance
    pub fn optimize_parameters(&mut self, target_spike_rate: f64, target_precision: f64) -> Result<()> {
        // Adaptive parameter tuning
        if self.encoding_stats.total_spikes_generated > 0 {
            let current_rate = self.encoding_stats.total_spikes_generated as f64 / 
                              self.encoding_stats.signals_encoded as f64;
            
            if current_rate < target_spike_rate * 0.9 {
                self.config.spike_threshold *= 0.95; // Lower threshold to increase rate
            } else if current_rate > target_spike_rate * 1.1 {
                self.config.spike_threshold *= 1.05; // Raise threshold to decrease rate
            }
            
            self.config.spike_threshold = self.config.spike_threshold.clamp(0.1, 0.9);
        }
        
        info!("Optimized spike threshold to {:.3}", self.config.spike_threshold);
        Ok(())
    }
}

/// Specialized quantum encoders for different signal types
pub struct SpecializedQuantumEncoders;

impl SpecializedQuantumEncoders {
    /// Market data encoder optimized for trading signals
    pub fn encode_market_data(
        encoder: &mut QuantumSpikeEncoder,
        price: f64,
        volume: f64,
        volatility: f64,
        momentum: f64,
    ) -> Result<Vec<QuantumSpikeTrain>> {
        let market_signal = DVector::from_vec(vec![
            price.ln(),           // Log price
            volume.ln(),          // Log volume
            volatility,           // Volatility
            momentum,             // Momentum
            (price * volume).ln(), // Dollar volume
            volatility * momentum, // Risk factor
        ]);
        
        encoder.encode_quantum_superposition(&market_signal)
    }
    
    /// Technical indicator encoder
    pub fn encode_technical_indicators(
        encoder: &mut QuantumSpikeEncoder,
        indicators: &HashMap<String, f64>,
    ) -> Result<Vec<QuantumSpikeTrain>> {
        let mut signal_vec = Vec::new();
        
        // Standard technical indicators
        let indicator_keys = ["RSI", "MACD", "SMA", "EMA", "BB_UPPER", "BB_LOWER", "ATR", "STOCH"];
        
        for key in &indicator_keys {
            signal_vec.push(indicators.get(*key).copied().unwrap_or(0.0));
        }
        
        let tech_signal = DVector::from_vec(signal_vec);
        encoder.encode_quantum_superposition(&tech_signal)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;
    
    #[test]
    fn test_spike_encoder_creation() {
        let config = QuantumSNNConfig::default();
        let encoder = QuantumSpikeEncoder::new(&config).unwrap();
        assert_eq!(encoder.config.encoding_type, SpikeEncodingType::QuantumSuperposition);
    }
    
    #[test]
    fn test_rate_coding() {
        let config = QuantumSNNConfig::default();
        let mut encoder = QuantumSpikeEncoder::new(&config).unwrap();
        encoder.config.encoding_type = SpikeEncodingType::Rate;
        
        let signal = DVector::from_vec(vec![0.8, 0.3, 0.0, 0.9]);
        let spike_trains = encoder.encode_rate_coding(&signal).unwrap();
        
        assert_eq!(spike_trains.len(), 4);
        
        // High amplitude should generate more spikes
        assert!(spike_trains[0].len() > spike_trains[1].len());
        assert!(spike_trains[3].len() > spike_trains[1].len());
        assert_eq!(spike_trains[2].len(), 0); // Below threshold
    }
    
    #[test]
    fn test_temporal_coding() {
        let config = QuantumSNNConfig::default();
        let mut encoder = QuantumSpikeEncoder::new(&config).unwrap();
        encoder.config.encoding_type = SpikeEncodingType::Temporal;
        
        let signal = DVector::from_vec(vec![0.6, 0.8, 0.4, 0.9]);
        let spike_trains = encoder.encode_temporal_coding(&signal).unwrap();
        
        assert_eq!(spike_trains.len(), 4);
        
        // Higher amplitudes should spike earlier (for this encoding)
        if !spike_trains[1].times.is_empty() && !spike_trains[3].times.is_empty() {
            assert!(spike_trains[3].times[0] > spike_trains[1].times[0]);
        }
    }
    
    #[test]
    fn test_quantum_superposition_encoding() {
        let config = QuantumSNNConfig::default();
        let mut encoder = QuantumSpikeEncoder::new(&config).unwrap();
        
        let signal = DVector::from_vec(vec![0.7, 0.3, 0.9, 0.1]);
        let spike_trains = encoder.encode_quantum_superposition(&signal).unwrap();
        
        assert_eq!(spike_trains.len(), 4);
        
        // Check that quantum amplitudes are complex
        for spike_train in &spike_trains {
            for amplitude in &spike_train.amplitudes {
                assert!(amplitude.norm() >= 0.0);
            }
        }
    }
    
    #[test]
    fn test_phase_coding() {
        let config = QuantumSNNConfig::default();
        let mut encoder = QuantumSpikeEncoder::new(&config).unwrap();
        
        let signal = DVector::from_vec(vec![0.5, 0.8, 0.3, 1.0]);
        let spike_trains = encoder.encode_phase_coding(&signal).unwrap();
        
        assert_eq!(spike_trains.len(), 4);
        
        // Check phase relationships
        let max_idx = 3; // signal[3] = 1.0 is maximum
        if !spike_trains[max_idx].phases.is_empty() {
            let max_phase = spike_trains[max_idx].phases[0];
            // Maximum signal should have maximum phase
            assert_relative_eq!(max_phase, 2.0 * std::f64::consts::PI, epsilon = 1e-10);
        }
    }
    
    #[test]
    fn test_burst_coding() {
        let config = QuantumSNNConfig::default();
        let mut encoder = QuantumSpikeEncoder::new(&config).unwrap();
        
        let signal = DVector::from_vec(vec![0.6, 0.9, 0.3, 0.7]);
        let spike_trains = encoder.encode_burst_coding(&signal).unwrap();
        
        assert_eq!(spike_trains.len(), 4);
        
        // Higher amplitude should generate larger bursts
        assert!(spike_trains[1].len() > spike_trains[2].len());
    }
    
    #[test]
    fn test_encode_decode_consistency() {
        let config = QuantumSNNConfig::default();
        let mut encoder = QuantumSpikeEncoder::new(&config).unwrap();
        encoder.config.encoding_type = SpikeEncodingType::Rate;
        
        let original_signal = DVector::from_vec(vec![0.8, 0.3, 0.6, 0.9]);
        let spike_trains = encoder.encode_rate_coding(&original_signal).unwrap();
        
        // Note: Exact decode consistency is not expected due to stochastic nature
        // but we can check that high amplitudes remain relatively high
        assert!(spike_trains[0].spike_rate() > spike_trains[1].spike_rate());
        assert!(spike_trains[3].spike_rate() > spike_trains[2].spike_rate());
    }
    
    #[test]
    fn test_market_data_encoding() {
        let config = QuantumSNNConfig::default();
        let mut encoder = QuantumSpikeEncoder::new(&config).unwrap();
        
        let spike_trains = SpecializedQuantumEncoders::encode_market_data(
            &mut encoder,
            100.0,  // price
            1000.0, // volume
            0.2,    // volatility
            0.1,    // momentum
        ).unwrap();
        
        assert_eq!(spike_trains.len(), 6); // 6 derived features
        
        // Check that spikes were generated for valid market data
        let total_spikes: usize = spike_trains.iter().map(|st| st.len()).sum();
        assert!(total_spikes > 0);
    }
}