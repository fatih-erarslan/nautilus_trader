//! Input encoding and output decoding for cerebellar networks
//! 
//! Provides sophisticated spike-based encoding and decoding strategies
//! with comprehensive support for trading data and neural signals.

use candle_core::{Tensor, Device, DType, Result as CandleResult};
use candle_nn as nn;
use anyhow::{Result, anyhow};
use tracing::{debug, info, warn};
use std::collections::HashMap;
use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

use crate::compatibility::{TensorCompat, NeuralNetCompat, DTypeCompat};
use crate::{CerebellarNorseConfig, CircuitConfig};

/// Comprehensive input encoder for market data to spike conversion
#[derive(Debug)]
pub struct InputEncoder {
    /// Device for computation
    pub device: Device,
    /// Encoding strategy
    pub strategy: EncodingStrategy,
    /// Encoding parameters
    pub params: EncodingParameters,
    /// Random number generator for stochastic encoding
    pub rng: StdRng,
    /// Time step for temporal encoding
    pub dt: f64,
}

/// Encoding strategies for different data types
#[derive(Debug, Clone, Copy)]
pub enum EncodingStrategy {
    /// Rate-based encoding (firing rate proportional to value)
    Rate,
    /// Temporal encoding (precise spike timing)
    Temporal,
    /// Population encoding (distributed across multiple neurons)
    Population,
    /// Binary encoding (direct threshold-based)
    Binary,
    /// Hybrid encoding (combination of strategies)
    Hybrid,
}

/// Parameters for encoding configuration
#[derive(Debug, Clone)]
pub struct EncodingParameters {
    /// Maximum firing rate (Hz)
    pub max_rate: f64,
    /// Time window for encoding (ms)
    pub time_window: f64,
    /// Number of neurons per input dimension
    pub neurons_per_input: usize,
    /// Noise level for robustness
    pub noise_level: f64,
    /// Value range for normalization
    pub value_range: (f64, f64),
    /// Temporal precision (ms)
    pub temporal_precision: f64,
}

impl Default for EncodingParameters {
    fn default() -> Self {
        Self {
            max_rate: 100.0,
            time_window: 20.0,
            neurons_per_input: 10,
            noise_level: 0.01,
            value_range: (0.0, 1.0),
            temporal_precision: 1.0,
        }
    }
}

impl InputEncoder {
    /// Create new input encoder with configuration
    pub fn new(config: &CerebellarNorseConfig) -> Result<Self> {
        let strategy = EncodingStrategy::Hybrid;
        let params = EncodingParameters::default();
        let rng = StdRng::seed_from_u64(42);
        
        Ok(Self {
            device: config.device.clone(),
            strategy,
            params,
            rng,
            dt: 0.1, // 100 microseconds
        })
    }
    
    /// Create encoder with device only (compatibility)
    pub fn new_with_device(device: Device) -> Self {
        let strategy = EncodingStrategy::Rate;
        let params = EncodingParameters::default();
        let rng = StdRng::seed_from_u64(42);
        
        Self {
            device,
            strategy,
            params,
            rng,
            dt: 0.1,
        }
    }
    
    /// Encode input tensor to spike patterns using selected strategy
    pub fn encode(&mut self, input: &Tensor) -> Result<Vec<Tensor>> {
        match self.strategy {
            EncodingStrategy::Rate => self.rate_encode(input),
            EncodingStrategy::Temporal => self.temporal_encode(input),
            EncodingStrategy::Population => self.population_encode(input),
            EncodingStrategy::Binary => self.binary_encode(input),
            EncodingStrategy::Hybrid => self.hybrid_encode(input),
        }
    }
    
    /// Rate-based encoding: firing rate proportional to input value
    fn rate_encode(&mut self, input: &Tensor) -> Result<Vec<Tensor>> {
        let input_data = input.to_vec2::<f32>()?;
        let mut encoded_sequences = Vec::new();
        
        let time_steps = (self.params.time_window / self.dt) as usize;
        
        for batch_idx in 0..input_data.len() {
            let batch_input = &input_data[batch_idx];
            let mut time_sequence = Vec::new();
            
            for _t in 0..time_steps {
                let mut spike_pattern = Vec::new();
                
                for &value in batch_input {
                    // Normalize value to [0, 1]
                    let normalized = (value as f64 - self.params.value_range.0) 
                        / (self.params.value_range.1 - self.params.value_range.0);
                    let normalized = normalized.clamp(0.0, 1.0);
                    
                    // Convert to firing rate
                    let rate = normalized * self.params.max_rate;
                    let spike_prob = rate * self.dt / 1000.0; // Convert to probability
                    
                    // Generate spikes for population of neurons
                    for _ in 0..self.params.neurons_per_input {
                        let spike = if self.rng.gen::<f64>() < spike_prob {
                            1.0
                        } else {
                            0.0
                        };
                        spike_pattern.push(spike);
                    }
                }
                
                let time_tensor = Tensor::from_vec(
                    spike_pattern,
                    &[batch_input.len() * self.params.neurons_per_input],
                    &self.device
                )?;
                
                time_sequence.push(time_tensor);
            }
            
            // Combine time sequence into single tensor
            if !time_sequence.is_empty() {
                encoded_sequences.extend(time_sequence);
            }
        }
        
        debug!("Rate encoded input: {} time steps", encoded_sequences.len());
        
        if encoded_sequences.is_empty() {
            // Return zero tensor if encoding failed
            let zero_tensor = Tensor::zeros(&[input.dims()[1] * self.params.neurons_per_input], DType::F32, &self.device)?;
            encoded_sequences.push(zero_tensor);
        }
        
        Ok(encoded_sequences)
    }
    
    /// Temporal encoding: precise spike timing based on value
    fn temporal_encode(&mut self, input: &Tensor) -> Result<Vec<Tensor>> {
        let input_data = input.to_vec2::<f32>()?;
        let mut encoded_sequences = Vec::new();
        
        let time_steps = (self.params.time_window / self.params.temporal_precision) as usize;
        
        for batch_input in input_data {
            for &value in &batch_input {
                // Normalize value
                let normalized = (value as f64 - self.params.value_range.0) 
                    / (self.params.value_range.1 - self.params.value_range.0);
                let normalized = normalized.clamp(0.0, 1.0);
                
                // Convert to spike time (earlier spike = higher value)
                let spike_time = (1.0 - normalized) * self.params.time_window;
                let spike_step = (spike_time / self.params.temporal_precision) as usize;
                
                // Create temporal spike pattern
                let mut temporal_pattern = vec![0.0f32; time_steps];
                if spike_step < time_steps {
                    temporal_pattern[spike_step] = 1.0;
                }
                
                let temporal_tensor = Tensor::from_vec(
                    temporal_pattern,
                    &[time_steps],
                    &self.device
                )?;
                
                encoded_sequences.push(temporal_tensor);
            }
        }
        
        debug!("Temporal encoded input: {} sequences", encoded_sequences.len());
        Ok(encoded_sequences)
    }
    
    /// Population encoding: distributed representation across neurons
    fn population_encode(&mut self, input: &Tensor) -> Result<Vec<Tensor>> {
        let input_data = input.to_vec2::<f32>()?;
        let mut encoded_sequences = Vec::new();
        
        for batch_input in input_data {
            let mut population_pattern = Vec::new();
            
            for &value in &batch_input {
                // Normalize value
                let normalized = (value as f64 - self.params.value_range.0) 
                    / (self.params.value_range.1 - self.params.value_range.0);
                let normalized = normalized.clamp(0.0, 1.0);
                
                // Create population response with Gaussian tuning curves
                for i in 0..self.params.neurons_per_input {
                    let preferred_value = i as f64 / (self.params.neurons_per_input - 1) as f64;
                    let distance = (normalized - preferred_value).abs();
                    let response = (-distance * distance / 0.1).exp(); // Gaussian tuning
                    
                    // Convert response to spike probability
                    let spike_prob = response * self.params.max_rate * self.dt / 1000.0;
                    let spike = if self.rng.gen::<f64>() < spike_prob { 1.0 } else { 0.0 };
                    
                    population_pattern.push(spike);
                }
            }
            
            let population_tensor = Tensor::from_vec(
                population_pattern,
                &[batch_input.len() * self.params.neurons_per_input],
                &self.device
            )?;
            
            encoded_sequences.push(population_tensor);
        }
        
        debug!("Population encoded input: {} patterns", encoded_sequences.len());
        Ok(encoded_sequences)
    }
    
    /// Binary encoding: direct threshold-based conversion
    fn binary_encode(&mut self, input: &Tensor) -> Result<Vec<Tensor>> {
        let threshold = (self.params.value_range.0 + self.params.value_range.1) / 2.0;
        let binary_spikes = input.gt(threshold as f32)?;
        let spike_tensor = binary_spikes.to_dtype(DType::F32)?;
        
        debug!("Binary encoded input");
        Ok(vec![spike_tensor])
    }
    
    /// Hybrid encoding: combination of multiple strategies
    fn hybrid_encode(&mut self, input: &Tensor) -> Result<Vec<Tensor>> {
        // Combine rate and temporal encoding
        let mut rate_encoded = self.rate_encode(input)?;
        let temporal_encoded = self.temporal_encode(input)?;
        
        // Interleave the sequences
        rate_encoded.extend(temporal_encoded);
        
        debug!("Hybrid encoded input: {} total sequences", rate_encoded.len());
        Ok(rate_encoded)
    }
    
    /// Encode market data with trading-specific features
    pub fn encode_market_data(
        &mut self,
        price: f64,
        volume: f64, 
        timestamp: u64,
        indicators: &[f64]
    ) -> Result<Vec<Tensor>> {
        // Combine market features
        let mut features = vec![price, volume, timestamp as f64];
        features.extend_from_slice(indicators);
        
        // Normalize features
        let normalized_features: Vec<f32> = features.iter()
            .map(|&x| ((x - self.params.value_range.0) / 
                      (self.params.value_range.1 - self.params.value_range.0)).clamp(0.0, 1.0) as f32)
            .collect();
        
        let market_tensor = Tensor::from_vec(
            normalized_features,
            &[1, features.len()],
            &self.device
        )?;
        
        self.encode(&market_tensor)
    }
}

/// Comprehensive output decoder for spike patterns to trading signals
#[derive(Debug)]
pub struct OutputDecoder {
    /// Device for computation
    pub device: Device,
    /// Decoding strategy
    pub strategy: DecodingStrategy,
    /// Decoding parameters
    pub params: DecodingParameters,
    /// Time constant for exponential filtering
    pub tau: f64,
    /// Internal state for temporal decoding
    pub state: HashMap<String, Tensor>,
}

/// Decoding strategies for spike pattern interpretation
#[derive(Debug, Clone, Copy)]
pub enum DecodingStrategy {
    /// Rate-based decoding (average firing rate)
    Rate,
    /// Temporal decoding (spike timing analysis)
    Temporal,
    /// Population vector decoding
    PopulationVector,
    /// Maximum likelihood decoding
    MaximumLikelihood,
    /// Kernel-based decoding
    Kernel,
}

/// Parameters for decoding configuration
#[derive(Debug, Clone)]
pub struct DecodingParameters {
    /// Decoding time window (ms)
    pub time_window: f64,
    /// Smoothing factor
    pub smoothing_factor: f64,
    /// Output range
    pub output_range: (f64, f64),
    /// Confidence threshold
    pub confidence_threshold: f64,
}

impl Default for DecodingParameters {
    fn default() -> Self {
        Self {
            time_window: 20.0,
            smoothing_factor: 0.9,
            output_range: (0.0, 1.0),
            confidence_threshold: 0.5,
        }
    }
}

impl OutputDecoder {
    /// Create new output decoder with configuration
    pub fn new(config: &CerebellarNorseConfig) -> Result<Self> {
        let strategy = DecodingStrategy::Rate;
        let params = DecodingParameters::default();
        let state = HashMap::new();
        
        Ok(Self {
            device: config.device.clone(),
            strategy,
            params,
            tau: 10.0, // 10ms time constant
            state,
        })
    }
    
    /// Create decoder with device only (compatibility)
    pub fn new_with_device(device: Device) -> Self {
        let strategy = DecodingStrategy::Rate;
        let params = DecodingParameters::default();
        let state = HashMap::new();
        
        Self {
            device,
            strategy,
            params,
            tau: 10.0,
            state,
        }
    }
    
    /// Decode spike patterns to output values
    pub fn decode(&mut self, spikes: &Tensor) -> Result<Tensor> {
        match self.strategy {
            DecodingStrategy::Rate => self.rate_decode(spikes),
            DecodingStrategy::Temporal => self.temporal_decode(spikes),
            DecodingStrategy::PopulationVector => self.population_vector_decode(spikes),
            DecodingStrategy::MaximumLikelihood => self.maximum_likelihood_decode(spikes),
            DecodingStrategy::Kernel => self.kernel_decode(spikes),
        }
    }
    
    /// Rate-based decoding: average firing rate over time window
    fn rate_decode(&self, spikes: &Tensor) -> Result<Tensor> {
        // Compute mean firing rate
        let mean_rate = spikes.mean_all()?;
        
        // Scale to output range
        let rate_value = TensorCompat::sum_compat(&mean_rate)?;
        let scaled_value = (rate_value * (self.params.output_range.1 - self.params.output_range.0) 
                           + self.params.output_range.0) as f32;
        
        let output = Tensor::from_vec(
            vec![scaled_value],
            &[1],
            &self.device
        )?;
        
        debug!("Rate decoded to value: {:.4}", scaled_value);
        Ok(output)
    }
    
    /// Temporal decoding: analyze spike timing patterns
    fn temporal_decode(&self, spikes: &Tensor) -> Result<Tensor> {
        // Find first spike time (for temporal encoding)
        let spike_data = spikes.to_vec1::<f32>()?;
        
        let first_spike_time = spike_data.iter()
            .position(|&x| x > 0.5)
            .unwrap_or(spike_data.len()) as f64;
        
        // Convert spike time to value (earlier = higher value)
        let normalized_time = first_spike_time / spike_data.len() as f64;
        let decoded_value = (1.0 - normalized_time) * 
                           (self.params.output_range.1 - self.params.output_range.0) + 
                           self.params.output_range.0;
        
        let output = Tensor::from_vec(
            vec![decoded_value as f32],
            &[1],
            &self.device
        )?;
        
        debug!("Temporal decoded to value: {:.4}", decoded_value);
        Ok(output)
    }
    
    /// Population vector decoding: weighted sum of preferred values
    fn population_vector_decode(&self, spikes: &Tensor) -> Result<Tensor> {
        let spike_data = spikes.to_vec1::<f32>()?;
        
        let mut weighted_sum = 0.0;
        let mut total_weight = 0.0;
        
        for (i, &spike_rate) in spike_data.iter().enumerate() {
            let preferred_value = i as f64 / (spike_data.len() - 1) as f64;
            weighted_sum += spike_rate as f64 * preferred_value;
            total_weight += spike_rate as f64;
        }
        
        let decoded_value = if total_weight > 0.0 {
            weighted_sum / total_weight
        } else {
            0.5 // Default to middle value
        };
        
        // Scale to output range
        let scaled_value = decoded_value * (self.params.output_range.1 - self.params.output_range.0) 
                          + self.params.output_range.0;
        
        let output = Tensor::from_vec(
            vec![scaled_value as f32],
            &[1],
            &self.device
        )?;
        
        debug!("Population vector decoded to value: {:.4}", scaled_value);
        Ok(output)
    }
    
    /// Maximum likelihood decoding using Bayesian inference
    fn maximum_likelihood_decode(&self, spikes: &Tensor) -> Result<Tensor> {
        // Simplified ML decoding - find maximum spike rate
        let spike_data = spikes.to_vec1::<f32>()?;
        
        let max_idx = spike_data.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        let decoded_value = max_idx as f64 / (spike_data.len() - 1) as f64;
        let scaled_value = decoded_value * (self.params.output_range.1 - self.params.output_range.0) 
                          + self.params.output_range.0;
        
        let output = Tensor::from_vec(
            vec![scaled_value as f32],
            &[1],
            &self.device
        )?;
        
        debug!("ML decoded to value: {:.4}", scaled_value);
        Ok(output)
    }
    
    /// Kernel-based decoding using convolution with basis functions
    fn kernel_decode(&self, spikes: &Tensor) -> Result<Tensor> {
        // Use Gaussian kernel for smoothing
        let spike_data = spikes.to_vec1::<f32>()?;
        let kernel_width = 3;
        
        let mut smoothed = Vec::new();
        for i in 0..spike_data.len() {
            let mut kernel_sum = 0.0;
            let mut weight_sum = 0.0;
            
            for j in 0..spike_data.len() {
                let distance = (i as f64 - j as f64).abs();
                let weight = (-distance * distance / (2.0 * kernel_width as f64 * kernel_width as f64)).exp();
                
                kernel_sum += spike_data[j] as f64 * weight;
                weight_sum += weight;
            }
            
            smoothed.push((kernel_sum / weight_sum) as f32);
        }
        
        // Find peak of smoothed response
        let peak_idx = smoothed.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0);
        
        let decoded_value = peak_idx as f64 / (smoothed.len() - 1) as f64;
        let scaled_value = decoded_value * (self.params.output_range.1 - self.params.output_range.0) 
                          + self.params.output_range.0;
        
        let output = Tensor::from_vec(
            vec![scaled_value as f32],
            &[1],
            &self.device
        )?;
        
        debug!("Kernel decoded to value: {:.4}", scaled_value);
        Ok(output)
    }
    
    /// Decode spikes to trading signals (Buy/Sell/Hold)
    pub fn decode_to_trading_signals(&mut self, spikes: &Tensor) -> Result<TradingSignal> {
        let decoded = self.decode(spikes)?;
        let value = TensorCompat::sum_compat(&decoded)?;
        
        let action = if value > 0.6 {
            TradeAction::Buy
        } else if value < 0.4 {
            TradeAction::Sell
        } else {
            TradeAction::Hold
        };
        
        let confidence = (value - 0.5).abs() * 2.0; // Convert to [0, 1]
        
        Ok(TradingSignal {
            action,
            confidence,
            quantity: value * 100.0, // Scale to reasonable position size
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_millis() as u64,
        })
    }
}

/// Trading signal output from decoder
#[derive(Debug, Clone)]
pub struct TradingSignal {
    pub action: TradeAction,
    pub confidence: f64,
    pub quantity: f64,
    pub timestamp: u64,
}

/// Trading actions
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TradeAction {
    Buy,
    Sell,
    Hold,
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;
    
    #[test]
    fn test_rate_encoding() {
        let device = Device::Cpu;
        let mut encoder = InputEncoder::new_with_device(device.clone());
        
        let input = Tensor::from_vec(vec![0.5f32, 0.8f32], &[1, 2], &device).unwrap();
        let encoded = encoder.encode(&input).unwrap();
        
        assert!(!encoded.is_empty());
    }
    
    #[test]
    fn test_rate_decoding() {
        let device = Device::Cpu;
        let mut decoder = OutputDecoder::new_with_device(device.clone());
        
        let spikes = Tensor::from_vec(vec![1.0f32, 0.0f32, 1.0f32], &[3], &device).unwrap();
        let decoded = decoder.decode(&spikes).unwrap();
        
        assert_eq!(decoded.dims(), &[1]);
    }
    
    #[test]
    fn test_market_data_encoding() {
        let device = Device::Cpu;
        let mut encoder = InputEncoder::new_with_device(device);
        
        let encoded = encoder.encode_market_data(
            100.0,  // price
            1000.0, // volume
            1234567890, // timestamp
            &[0.5, 0.3, 0.8] // indicators
        ).unwrap();
        
        assert!(!encoded.is_empty());
    }
    
    #[test]
    fn test_trading_signal_decoding() {
        let device = Device::Cpu;
        let mut decoder = OutputDecoder::new_with_device(device.clone());
        
        let spikes = Tensor::from_vec(vec![0.8f32, 0.2f32, 0.1f32], &[3], &device).unwrap();
        let signal = decoder.decode_to_trading_signals(&spikes).unwrap();
        
        match signal.action {
            TradeAction::Buy | TradeAction::Sell | TradeAction::Hold => {},
        }
        assert!(signal.confidence >= 0.0 && signal.confidence <= 1.0);
    }
}