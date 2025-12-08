//! Test fixtures for cerebellar-norse tests
//! 
//! This module provides common test fixtures and data structures
//! used across multiple test suites.

use std::collections::HashMap;
use candle_core::{Tensor, Device, DType};
use candle_nn as nn;
use anyhow::Result;
use cerebellar_norse::*;

/// Standard test configuration fixtures
pub struct ConfigFixtures;

impl ConfigFixtures {
    /// Small configuration for unit tests
    pub fn small_config() -> CerebellarNorseConfig {
        CerebellarNorseConfig {
            input_dim: 4,
            output_dim: 1,
            n_granule: 20,
            n_purkinje: 5,
            n_golgi: 3,
            n_dcn: 2,
            time_steps: 10,
            dt: 1e-3,
            use_adex: HashMap::new(),
            seed: 42,
            device: Device::Cpu,
            max_processing_time_us: 1000,
        }
    }

    /// Medium configuration for integration tests
    pub fn medium_config() -> CerebellarNorseConfig {
        CerebellarNorseConfig {
            input_dim: 16,
            output_dim: 4,
            n_granule: 200,
            n_purkinje: 50,
            n_golgi: 25,
            n_dcn: 10,
            time_steps: 100,
            dt: 1e-3,
            use_adex: HashMap::new(),
            seed: 42,
            device: Device::Cpu,
            max_processing_time_us: 5000,
        }
    }

    /// Large configuration for performance tests
    pub fn large_config() -> CerebellarNorseConfig {
        CerebellarNorseConfig {
            input_dim: 64,
            output_dim: 16,
            n_granule: 4000,
            n_purkinje: 100,
            n_golgi: 50,
            n_dcn: 20,
            time_steps: 1000,
            dt: 1e-3,
            use_adex: HashMap::new(),
            seed: 42,
            device: Device::Cpu,
            max_processing_time_us: 10000,
        }
    }

    /// Trading-optimized configuration
    pub fn trading_config() -> CerebellarNorseConfig {
        CerebellarNorseConfig {
            input_dim: 32,
            output_dim: 8,
            n_granule: 2000,
            n_purkinje: 80,
            n_golgi: 40,
            n_dcn: 16,
            time_steps: 200,
            dt: 1e-3,
            use_adex: HashMap::new(),
            seed: 42,
            device: Device::Cpu,
            max_processing_time_us: 2000, // 2ms for trading
        }
    }

    /// Configuration with AdEx neurons
    pub fn adex_config() -> CerebellarNorseConfig {
        let mut use_adex = HashMap::new();
        use_adex.insert("purkinje".to_string(), true);
        use_adex.insert("dcn".to_string(), true);

        CerebellarNorseConfig {
            input_dim: 8,
            output_dim: 2,
            n_granule: 100,
            n_purkinje: 20,
            n_golgi: 10,
            n_dcn: 5,
            time_steps: 50,
            dt: 1e-3,
            use_adex,
            seed: 42,
            device: Device::Cpu,
            max_processing_time_us: 5000,
        }
    }
}

/// Layer configuration fixtures
pub struct LayerFixtures;

impl LayerFixtures {
    /// Small layer configurations for unit tests
    pub fn small_layers() -> HashMap<String, LayerConfig> {
        let mut configs = HashMap::new();
        
        configs.insert("granule".to_string(), LayerConfig {
            size: 20,
            neuron_type: NeuronType::LIF,
            tau_mem: 10.0,
            tau_syn_exc: 2.0,
            tau_syn_inh: 10.0,
            tau_adapt: Some(50.0),
            a: Some(2e-9),
            b: Some(1e-10),
        });
        
        configs.insert("purkinje".to_string(), LayerConfig {
            size: 5,
            neuron_type: NeuronType::LIF,
            tau_mem: 15.0,
            tau_syn_exc: 3.0,
            tau_syn_inh: 5.0,
            tau_adapt: Some(100.0),
            a: Some(4e-9),
            b: Some(5e-10),
        });
        
        configs.insert("golgi".to_string(), LayerConfig {
            size: 3,
            neuron_type: NeuronType::LIF,
            tau_mem: 30.0,
            tau_syn_exc: 5.0,
            tau_syn_inh: 10.0,
            tau_adapt: Some(200.0),
            a: Some(2e-9),
            b: Some(2e-10),
        });
        
        configs.insert("dcn".to_string(), LayerConfig {
            size: 2,
            neuron_type: NeuronType::LIF,
            tau_mem: 25.0,
            tau_syn_exc: 5.0,
            tau_syn_inh: 10.0,
            tau_adapt: Some(150.0),
            a: Some(1e-9),
            b: Some(5e-10),
        });
        
        configs
    }

    /// AdEx layer configurations
    pub fn adex_layers() -> HashMap<String, LayerConfig> {
        let mut configs = Self::small_layers();
        
        // Convert Purkinje and DCN to AdEx
        if let Some(purkinje) = configs.get_mut("purkinje") {
            purkinje.neuron_type = NeuronType::AdEx;
        }
        
        if let Some(dcn) = configs.get_mut("dcn") {
            dcn.neuron_type = NeuronType::AdEx;
        }
        
        configs
    }

    /// High-performance layer configurations
    pub fn performance_layers() -> HashMap<String, LayerConfig> {
        let mut configs = HashMap::new();
        
        configs.insert("granule".to_string(), LayerConfig {
            size: 1000,
            neuron_type: NeuronType::LIF,
            tau_mem: 8.0,    // Faster response
            tau_syn_exc: 1.5,
            tau_syn_inh: 8.0,
            tau_adapt: Some(40.0),
            a: Some(2e-9),
            b: Some(1e-10),
        });
        
        configs.insert("purkinje".to_string(), LayerConfig {
            size: 100,
            neuron_type: NeuronType::AdEx,
            tau_mem: 12.0,
            tau_syn_exc: 2.5,
            tau_syn_inh: 4.0,
            tau_adapt: Some(80.0),
            a: Some(4e-9),
            b: Some(5e-10),
        });
        
        configs.insert("golgi".to_string(), LayerConfig {
            size: 50,
            neuron_type: NeuronType::LIF,
            tau_mem: 25.0,
            tau_syn_exc: 4.0,
            tau_syn_inh: 8.0,
            tau_adapt: Some(150.0),
            a: Some(2e-9),
            b: Some(2e-10),
        });
        
        configs.insert("dcn".to_string(), LayerConfig {
            size: 20,
            neuron_type: NeuronType::AdEx,
            tau_mem: 20.0,
            tau_syn_exc: 3.0,
            tau_syn_inh: 6.0,
            tau_adapt: Some(100.0),
            a: Some(1e-9),
            b: Some(5e-10),
        });
        
        configs
    }
}

/// Test data fixtures
pub struct DataFixtures;

impl DataFixtures {
    /// Simple step input pattern
    pub fn step_input(batch_size: usize, input_dim: usize, device: Device) -> Tensor {
        let shape = vec![batch_size as i64, input_dim as i64];
        Tensor::ones(&shape, (DType::F32, &device)).unwrap()
    }

    /// Sinusoidal input pattern
    pub fn sinusoidal_input(batch_size: usize, input_dim: usize, device: Device) -> Tensor {
        let shape = vec![batch_size as i64, input_dim as i64];
        let mut data = Vec::new();
        
        for batch in 0..batch_size {
            for input in 0..input_dim {
                let t = batch as f32 * 0.1;
                let freq = (input + 1) as f32 * 0.5;
                let value = (2.0 * std::f32::consts::PI * freq * t).sin();
                data.push(value);
            }
        }
        
        Tensor::from_vec(data, &shape, &device).unwrap()
    }

    /// Noisy input pattern
    pub fn noisy_input(batch_size: usize, input_dim: usize, noise_level: f32, device: Device) -> Tensor {
        let shape = vec![batch_size as i64, input_dim as i64];
        let base = Self::step_input(batch_size, input_dim, device.clone());
        let noise = Tensor::randn(&shape, (DType::F32, &device)).unwrap() * noise_level;
        
        (&base + &noise).unwrap()
    }

    /// Spike train input
    pub fn spike_train_input(batch_size: usize, input_dim: usize, spike_rate: f64, device: Device) -> Tensor {
        let shape = vec![batch_size as i64, input_dim as i64];
        let random_vals = Tensor::rand(&shape, (DType::F32, &device)).unwrap();
        
        random_vals.lt(spike_rate).unwrap().to_dtype(DType::F32).unwrap()
    }

    /// Ramped input pattern
    pub fn ramp_input(batch_size: usize, input_dim: usize, device: Device) -> Tensor {
        let shape = vec![batch_size as i64, input_dim as i64];
        let mut data = Vec::new();
        
        for batch in 0..batch_size {
            for input in 0..input_dim {
                let value = (batch as f32 / batch_size as f32) * (input + 1) as f32 / input_dim as f32;
                data.push(value);
            }
        }
        
        Tensor::from_vec(data, &shape, &device).unwrap()
    }

    /// Binary classification targets
    pub fn binary_targets(batch_size: usize, device: Device) -> Tensor {
        let shape = vec![batch_size as i64, 1];
        let mut data = Vec::new();
        
        for batch in 0..batch_size {
            let value = if batch % 2 == 0 { 1.0 } else { 0.0 };
            data.push(value);
        }
        
        Tensor::from_vec(data, &shape, &device).unwrap()
    }

    /// Multi-class classification targets
    pub fn multiclass_targets(batch_size: usize, num_classes: usize, device: Device) -> Tensor {
        let shape = vec![batch_size as i64, num_classes as i64];
        let mut data = vec![0.0; batch_size * num_classes];
        
        for batch in 0..batch_size {
            let class_idx = batch % num_classes;
            data[batch * num_classes + class_idx] = 1.0;
        }
        
        Tensor::from_vec(data, &shape, &device).unwrap()
    }

    /// Regression targets
    pub fn regression_targets(batch_size: usize, output_dim: usize, device: Device) -> Tensor {
        let shape = vec![batch_size as i64, output_dim as i64];
        let mut data = Vec::new();
        
        for batch in 0..batch_size {
            for output in 0..output_dim {
                let value = (batch as f32 * 0.1).sin() * (output + 1) as f32 * 0.5;
                data.push(value);
            }
        }
        
        Tensor::from_vec(data, &shape, &device).unwrap()
    }

    /// Time series data
    pub fn time_series_data(seq_length: usize, batch_size: usize, input_dim: usize, device: Device) -> Tensor {
        let shape = vec![seq_length as i64, batch_size as i64, input_dim as i64];
        let mut data = Vec::new();
        
        for t in 0..seq_length {
            for batch in 0..batch_size {
                for input in 0..input_dim {
                    let time = t as f32 * 0.01;
                    let batch_offset = batch as f32 * 0.1;
                    let input_freq = (input + 1) as f32 * 0.2;
                    let value = (2.0 * std::f32::consts::PI * input_freq * (time + batch_offset)).sin();
                    data.push(value);
                }
            }
        }
        
        Tensor::from_vec(data, &shape, &device).unwrap()
    }
}

/// Circuit state fixtures
pub struct CircuitFixtures;

impl CircuitFixtures {
    /// Create a simple circuit for testing
    pub fn simple_circuit() -> Result<CerebellarCircuit> {
        let config = ConfigFixtures::small_config();
        let layer_configs = LayerFixtures::small_layers();
        let vs = nn::VarStore::new(config.device);
        
        CerebellarCircuit::new(&config, &layer_configs, &vs)
    }

    /// Create a circuit with AdEx neurons
    pub fn adex_circuit() -> Result<CerebellarCircuit> {
        let config = ConfigFixtures::adex_config();
        let layer_configs = LayerFixtures::adex_layers();
        let vs = nn::VarStore::new(config.device);
        
        CerebellarCircuit::new(&config, &layer_configs, &vs)
    }

    /// Create a performance-optimized circuit
    pub fn performance_circuit() -> Result<CerebellarCircuit> {
        let config = ConfigFixtures::large_config();
        let layer_configs = LayerFixtures::performance_layers();
        let vs = nn::VarStore::new(config.device);
        
        CerebellarCircuit::new(&config, &layer_configs, &vs)
    }

    /// Create a circuit with pre-trained weights
    pub fn pretrained_circuit() -> Result<CerebellarCircuit> {
        let mut circuit = Self::simple_circuit()?;
        
        // Initialize with specific weight patterns
        // This would load pre-trained weights in a real scenario
        circuit.reset();
        
        Ok(circuit)
    }
}

/// Encoder/Decoder fixtures
pub struct EncodingFixtures;

impl EncodingFixtures {
    /// Create a simple input encoder
    pub fn simple_encoder() -> Result<InputEncoder> {
        let config = ConfigFixtures::small_config();
        InputEncoder::new(&config)
    }

    /// Create a performance-optimized encoder
    pub fn performance_encoder() -> Result<InputEncoder> {
        let config = ConfigFixtures::large_config();
        InputEncoder::new(&config)
    }

    /// Create a simple output decoder
    pub fn simple_decoder() -> Result<OutputDecoder> {
        let config = ConfigFixtures::small_config();
        OutputDecoder::new(&config)
    }

    /// Create a trading-optimized decoder
    pub fn trading_decoder() -> Result<OutputDecoder> {
        let config = ConfigFixtures::trading_config();
        OutputDecoder::new(&config)
    }
}

/// Training configuration fixtures
pub struct TrainingFixtures;

impl TrainingFixtures {
    /// Basic training configuration
    pub fn basic_training_config() -> TrainingConfig {
        TrainingConfig {
            learning_rate: 1e-3,
            surrogate_alpha: 100.0,
            stdp_window: 20.0,
            ltp_strength: 0.01,
            ltd_strength: 0.005,
            weight_decay: 1e-5,
            gradient_clip: 1.0,
            use_biological_plasticity: false,
            parallel_batch_size: 16,
        }
    }

    /// Biological plasticity training configuration
    pub fn biological_training_config() -> TrainingConfig {
        TrainingConfig {
            learning_rate: 5e-4,
            surrogate_alpha: 50.0,
            stdp_window: 25.0,
            ltp_strength: 0.02,
            ltd_strength: 0.01,
            weight_decay: 5e-6,
            gradient_clip: 0.5,
            use_biological_plasticity: true,
            parallel_batch_size: 8,
        }
    }

    /// Fast training configuration
    pub fn fast_training_config() -> TrainingConfig {
        TrainingConfig {
            learning_rate: 5e-3,
            surrogate_alpha: 200.0,
            stdp_window: 10.0,
            ltp_strength: 0.05,
            ltd_strength: 0.02,
            weight_decay: 1e-4,
            gradient_clip: 2.0,
            use_biological_plasticity: false,
            parallel_batch_size: 32,
        }
    }
}

/// Optimization configuration fixtures
pub struct OptimizationFixtures;

impl OptimizationFixtures {
    /// Basic optimization configuration
    pub fn basic_optimization_config() -> OptimizationConfig {
        OptimizationConfig {
            use_cuda: false,
            use_simd: false,
            memory_pool_size: 128,
            batch_size: 16,
            num_threads: 2,
            cache_level: CacheLevel::Conservative,
            fusion_threshold: 512,
            memory_alignment: 32,
        }
    }

    /// Performance optimization configuration
    pub fn performance_optimization_config() -> OptimizationConfig {
        OptimizationConfig {
            use_cuda: cfg!(feature = "cuda"),
            use_simd: cfg!(feature = "simd"),
            memory_pool_size: 1024,
            batch_size: 64,
            num_threads: num_cpus::get(),
            cache_level: CacheLevel::Aggressive,
            fusion_threshold: 2048,
            memory_alignment: 64,
        }
    }

    /// Memory-optimized configuration
    pub fn memory_optimization_config() -> OptimizationConfig {
        OptimizationConfig {
            use_cuda: false,
            use_simd: true,
            memory_pool_size: 64,
            batch_size: 8,
            num_threads: 1,
            cache_level: CacheLevel::Conservative,
            fusion_threshold: 256,
            memory_alignment: 16,
        }
    }
}

/// Test scenario fixtures
pub struct ScenarioFixtures;

impl ScenarioFixtures {
    /// XOR learning scenario
    pub fn xor_scenario() -> (Tensor, Tensor) {
        let device = Device::Cpu;
        
        // XOR inputs
        let inputs = Tensor::from_vec(
            vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0],
            &[4, 2],
            &device
        ).unwrap();
        
        // XOR targets
        let targets = Tensor::from_vec(
            vec![0.0, 1.0, 1.0, 0.0],
            &[4, 1],
            &device
        ).unwrap();
        
        (inputs, targets)
    }

    /// Pattern recognition scenario
    pub fn pattern_recognition_scenario() -> (Tensor, Tensor) {
        let device = Device::Cpu;
        let batch_size = 100;
        let input_dim = 16;
        
        // Create patterns with different frequencies
        let mut input_data = Vec::new();
        let mut target_data = Vec::new();
        
        for batch in 0..batch_size {
            let pattern_type = batch % 4;
            
            // Generate pattern based on type
            for i in 0..input_dim {
                let value = match pattern_type {
                    0 => if i % 2 == 0 { 1.0 } else { 0.0 }, // Even positions
                    1 => if i % 2 == 1 { 1.0 } else { 0.0 }, // Odd positions
                    2 => if i < input_dim / 2 { 1.0 } else { 0.0 }, // First half
                    3 => if i >= input_dim / 2 { 1.0 } else { 0.0 }, // Second half
                    _ => 0.0,
                };
                input_data.push(value);
            }
            
            // One-hot target
            for i in 0..4 {
                target_data.push(if i == pattern_type { 1.0 } else { 0.0 });
            }
        }
        
        let inputs = Tensor::from_vec(input_data, &[batch_size, input_dim], &device).unwrap();
        let targets = Tensor::from_vec(target_data, &[batch_size, 4], &device).unwrap();
        
        (inputs, targets)
    }

    /// Time series prediction scenario
    pub fn time_series_scenario() -> (Tensor, Tensor) {
        let device = Device::Cpu;
        let seq_length = 50;
        let batch_size = 20;
        let input_dim = 1;
        
        // Generate sine wave with noise
        let mut input_data = Vec::new();
        let mut target_data = Vec::new();
        
        for batch in 0..batch_size {
            let phase = batch as f32 * 0.1;
            
            for t in 0..seq_length {
                let time = t as f32 * 0.1 + phase;
                let value = (2.0 * std::f32::consts::PI * 0.1 * time).sin();
                input_data.push(value);
            }
            
            // Predict next value
            let next_time = seq_length as f32 * 0.1 + phase;
            let next_value = (2.0 * std::f32::consts::PI * 0.1 * next_time).sin();
            target_data.push(next_value);
        }
        
        let inputs = Tensor::from_vec(
            input_data, 
            &[seq_length, batch_size, input_dim], 
            &device
        ).unwrap();
        let targets = Tensor::from_vec(
            target_data, 
            &[batch_size, 1], 
            &device
        ).unwrap();
        
        (inputs, targets)
    }

    /// Trading signal scenario
    pub fn trading_scenario() -> (Tensor, Tensor) {
        let device = Device::Cpu;
        let batch_size = 200;
        let input_dim = 8; // price, volume, indicators
        
        let mut input_data = Vec::new();
        let mut target_data = Vec::new();
        
        for batch in 0..batch_size {
            let t = batch as f32 * 0.01;
            
            // Generate synthetic market data
            let price = 100.0 + 10.0 * (t * 0.1).sin() + 2.0 * (t * 0.5).sin();
            let volume = 1000.0 + 500.0 * (t * 0.2).cos();
            let rsi = 0.5 + 0.3 * (t * 0.3).sin();
            let macd = 0.1 * (t * 0.15).sin();
            let bollinger = 0.05 * (t * 0.25).cos();
            let momentum = 0.02 * (t * 0.4).sin();
            let volatility = 0.1 + 0.05 * (t * 0.1).cos();
            let trend = 0.5 + 0.2 * (t * 0.05).sin();
            
            input_data.extend_from_slice(&[
                price / 100.0,
                volume / 1000.0,
                rsi,
                macd,
                bollinger,
                momentum,
                volatility,
                trend,
            ]);
            
            // Generate buy/sell/hold signals
            let signal = if rsi > 0.7 && macd > 0.05 {
                [1.0, 0.0, 0.0] // Buy
            } else if rsi < 0.3 && macd < -0.05 {
                [0.0, 1.0, 0.0] // Sell
            } else {
                [0.0, 0.0, 1.0] // Hold
            };
            
            target_data.extend_from_slice(&signal);
        }
        
        let inputs = Tensor::from_vec(input_data, &[batch_size, input_dim], &device).unwrap();
        let targets = Tensor::from_vec(target_data, &[batch_size, 3], &device).unwrap();
        
        (inputs, targets)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_config_fixtures() {
        let small = ConfigFixtures::small_config();
        let medium = ConfigFixtures::medium_config();
        let large = ConfigFixtures::large_config();
        
        assert!(small.n_granule < medium.n_granule);
        assert!(medium.n_granule < large.n_granule);
        assert!(small.time_steps < medium.time_steps);
        assert!(medium.time_steps < large.time_steps);
    }
    
    #[test]
    fn test_data_fixtures() {
        let device = Device::Cpu;
        let batch_size = 4;
        let input_dim = 3;
        
        let step = DataFixtures::step_input(batch_size, input_dim, device.clone());
        let sin = DataFixtures::sinusoidal_input(batch_size, input_dim, device.clone());
        let noisy = DataFixtures::noisy_input(batch_size, input_dim, 0.1, device.clone());
        
        assert_eq!(step.shape().dims(), &[batch_size, input_dim]);
        assert_eq!(sin.shape().dims(), &[batch_size, input_dim]);
        assert_eq!(noisy.shape().dims(), &[batch_size, input_dim]);
    }
    
    #[test]
    fn test_circuit_fixtures() {
        let circuit = CircuitFixtures::simple_circuit().unwrap();
        assert_eq!(circuit.config().n_granule, 20);
        assert_eq!(circuit.config().n_purkinje, 5);
    }
    
    #[test]
    fn test_scenario_fixtures() {
        let (inputs, targets) = ScenarioFixtures::xor_scenario();
        assert_eq!(inputs.shape().dims(), &[4, 2]);
        assert_eq!(targets.shape().dims(), &[4, 1]);
        
        let (inputs, targets) = ScenarioFixtures::pattern_recognition_scenario();
        assert_eq!(inputs.shape().dims(), &[100, 16]);
        assert_eq!(targets.shape().dims(), &[100, 4]);
    }
    
    #[test]
    fn test_encoding_fixtures() {
        let encoder = EncodingFixtures::simple_encoder().unwrap();
        let decoder = EncodingFixtures::simple_decoder().unwrap();
        
        // Test that fixtures are properly initialized
        assert_eq!(encoder.get_statistics().len(), 3);
        assert_eq!(decoder.get_statistics().len(), 4);
    }
}