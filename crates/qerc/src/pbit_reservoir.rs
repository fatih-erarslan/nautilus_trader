//! pBit Reservoir Computing
//!
//! Quantum-Enhanced Reservoir Computing (QERC) ported to pBit architecture.
//! Implements Echo State Network with pBit-based quantum kernel for
//! nonlinear temporal pattern recognition.
//!
//! ## Architecture
//!
//! ```text
//! Input u(t) → [W_in] → Reservoir x(t) → [pBit Kernel] → [W_out] → Output y(t)
//!                           ↓                  ↓
//!                    Leaking dynamics    Boltzmann sampling
//!                    x(t+1) = (1-α)x(t) + α·f(W·x + W_in·u)
//! ```
//!
//! ## Key Mappings (Quantum → pBit)
//!
//! - RY(θ) encoding → P(↑) = sin²(θ/2)
//! - CNOT entanglement → Ferromagnetic coupling J > 0
//! - RZ(φ) phase → Local bias field
//! - ⟨Z⟩ measurement → Magnetization ⟨σ⟩ = 2·P(↑) - 1

use quantum_core::{
    PBitState, PBitConfig, PBitCoupling, QuantumError, QuantumResult,
};
use ndarray::{Array1, Array2};
use rand::prelude::*;
use rand_chacha::ChaCha8Rng;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::f64::consts::PI;
use tracing::{debug, info, warn};

/// Configuration for pBit Reservoir Computing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PBitReservoirConfig {
    /// Size of the reservoir (number of nodes)
    pub reservoir_size: usize,
    /// Number of pBits for quantum kernel
    pub kernel_size: usize,
    /// Spectral radius for reservoir stability (should be < 1)
    pub spectral_radius: f64,
    /// Leaking rate for reservoir dynamics (0-1)
    pub leaking_rate: f64,
    /// Temporal window sizes for feature extraction
    pub temporal_windows: Vec<usize>,
    /// Input dimensionality
    pub input_dim: usize,
    /// Output dimensionality
    pub output_dim: usize,
    /// Reservoir connectivity (sparsity)
    pub connectivity: f64,
    /// pBit temperature for Boltzmann sampling
    pub pbit_temperature: f64,
    /// Number of pBit sweeps per kernel evaluation
    pub sweeps_per_kernel: usize,
    /// Random seed for reproducibility
    pub seed: Option<u64>,
}

impl Default for PBitReservoirConfig {
    fn default() -> Self {
        Self {
            reservoir_size: 500,
            kernel_size: 4,
            spectral_radius: 0.95,
            leaking_rate: 0.3,
            temporal_windows: vec![5, 15, 30, 60],
            input_dim: 16,
            output_dim: 4,
            connectivity: 0.1,
            pbit_temperature: 1.0,
            sweeps_per_kernel: 10,
            seed: None,
        }
    }
}

/// Output from reservoir processing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReservoirOutput {
    /// Trend signal (-1 to 1)
    pub trend: f64,
    /// Volatility signal (0 to 1)
    pub volatility: f64,
    /// Momentum signal (-1 to 1)
    pub momentum: f64,
    /// Regime signal (0 to 1)
    pub regime: f64,
    /// Temporal features extracted
    pub temporal_features: HashMap<String, f64>,
    /// Current reservoir state
    pub reservoir_state: Vec<f64>,
    /// Execution time in milliseconds
    pub execution_time_ms: f64,
}

/// pBit-based Quantum-Enhanced Reservoir Computing
pub struct PBitReservoir {
    /// Configuration
    config: PBitReservoirConfig,
    /// Input weight matrix W_in (reservoir_size × input_dim)
    w_in: Array2<f64>,
    /// Reservoir weight matrix W (reservoir_size × reservoir_size)
    w: Array2<f64>,
    /// Output weight matrix W_out (output_dim × reservoir_size)
    w_out: Array2<f64>,
    /// Current reservoir state
    reservoir_state: Array1<f64>,
    /// pBit state for quantum kernel
    pbit_state: PBitState,
    /// Random number generator
    rng: ChaCha8Rng,
    /// Initialization flag
    is_initialized: bool,
    /// Kernel cache for performance
    kernel_cache: HashMap<u64, f64>,
}

impl PBitReservoir {
    /// Create a new pBit reservoir
    pub fn new(config: PBitReservoirConfig) -> QuantumResult<Self> {
        let seed = config.seed.unwrap_or_else(|| rand::thread_rng().gen());
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        
        // Initialize pBit state for quantum kernel
        let pbit_config = PBitConfig {
            temperature: config.pbit_temperature,
            coupling_strength: 1.0,
            external_field: 0.0,
            seed: Some(seed),
        };
        let pbit_state = PBitState::with_config(config.kernel_size, pbit_config)?;
        
        // Initialize weight matrices
        let w_in = Self::init_input_weights(&config, &mut rng);
        let w = Self::init_reservoir_weights(&config, &mut rng);
        let w_out = Array2::zeros((config.output_dim, config.reservoir_size));
        
        // Initialize reservoir state
        let reservoir_state = Array1::zeros(config.reservoir_size);
        
        info!(
            "PBitReservoir initialized: reservoir_size={}, kernel_size={}, spectral_radius={}",
            config.reservoir_size, config.kernel_size, config.spectral_radius
        );
        
        Ok(Self {
            config,
            w_in,
            w,
            w_out,
            reservoir_state,
            pbit_state,
            rng,
            is_initialized: true,
            kernel_cache: HashMap::new(),
        })
    }
    
    /// Initialize input weight matrix W_in
    fn init_input_weights(config: &PBitReservoirConfig, rng: &mut ChaCha8Rng) -> Array2<f64> {
        let mut w_in = Array2::zeros((config.reservoir_size, config.input_dim));
        for i in 0..config.reservoir_size {
            for j in 0..config.input_dim {
                w_in[[i, j]] = rng.gen_range(-0.5..0.5);
            }
        }
        w_in
    }
    
    /// Initialize reservoir weight matrix W with spectral radius scaling
    fn init_reservoir_weights(config: &PBitReservoirConfig, rng: &mut ChaCha8Rng) -> Array2<f64> {
        let n = config.reservoir_size;
        let mut w = Array2::zeros((n, n));
        
        // Create sparse random matrix
        for i in 0..n {
            for j in 0..n {
                if rng.gen::<f64>() < config.connectivity {
                    w[[i, j]] = rng.gen_range(-0.5..0.5);
                }
            }
        }
        
        // Scale to desired spectral radius
        // Note: For large matrices, we approximate spectral radius
        let frobenius_norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        let approx_spectral_radius = frobenius_norm / (n as f64).sqrt();
        
        if approx_spectral_radius > 1e-10 {
            let scale = config.spectral_radius / approx_spectral_radius;
            w.mapv_inplace(|x| x * scale);
        }
        
        w
    }
    
    /// Encode input value to pBit probability using RY mapping
    /// RY(θ) → P(↑) = sin²(θ/2)
    fn encode_to_pbit_probability(&self, value: f64) -> f64 {
        let theta = value.clamp(-1.0, 1.0) * PI;
        (theta / 2.0).sin().powi(2)
    }
    
    /// pBit quantum kernel - replaces PennyLane circuit
    /// Simulates: RY(inputs) → CNOT chain → RZ(weights) → CNOT reverse → ⟨Z⟩
    fn pbit_kernel(&mut self, inputs: &[f64], weights: &[f64]) -> QuantumResult<f64> {
        let n = self.config.kernel_size.min(inputs.len()).min(weights.len());
        
        // Encode inputs as pBit probabilities (RY encoding)
        for i in 0..n {
            let prob = self.encode_to_pbit_probability(inputs[i]);
            if let Some(pbit) = self.pbit_state.get_pbit_mut(i) {
                pbit.probability_up = prob;
                pbit.bias = weights[i % weights.len()]; // RZ as bias
            }
        }
        
        // Add CNOT-like couplings (ferromagnetic chain)
        for i in 0..n.saturating_sub(1) {
            self.pbit_state.add_coupling(PBitCoupling::bell_coupling(i, i + 1, 1.0));
        }
        // Reverse chain (second CNOT layer)
        for i in (1..n).rev() {
            self.pbit_state.add_coupling(PBitCoupling::bell_coupling(i, i - 1, 0.5));
        }
        
        // Boltzmann sampling (sweeps)
        for _ in 0..self.config.sweeps_per_kernel {
            self.pbit_state.sweep();
        }
        
        // Return magnetization (⟨Z⟩ equivalent)
        Ok(self.pbit_state.magnetization())
    }
    
    /// Apply pBit nonlinearity to reservoir nodes
    fn apply_pbit_nonlinearity(&mut self, node_values: &[f64], weights: &[f64]) -> QuantumResult<Vec<f64>> {
        let batch_size = 10;
        let mut results = Vec::with_capacity(node_values.len());
        
        for chunk_start in (0..node_values.len()).step_by(batch_size) {
            let chunk_end = (chunk_start + batch_size).min(node_values.len());
            
            for i in chunk_start..chunk_end {
                // Prepare inputs for kernel
                let kernel_inputs: Vec<f64> = (0..self.config.kernel_size)
                    .map(|j| node_values[(i + j) % node_values.len()])
                    .collect();
                let kernel_weights: Vec<f64> = (0..self.config.kernel_size)
                    .map(|j| weights[(i + j) % weights.len()])
                    .collect();
                
                // Apply pBit kernel
                let kernel_result = self.pbit_kernel(&kernel_inputs, &kernel_weights)?;
                results.push(kernel_result);
            }
        }
        
        Ok(results)
    }
    
    /// Update reservoir state with new input
    fn update_reservoir_state(&mut self, input: &Array1<f64>) -> QuantumResult<()> {
        // Ensure input has correct dimension
        let input_padded = if input.len() < self.config.input_dim {
            let mut padded = Array1::zeros(self.config.input_dim);
            for (i, &v) in input.iter().enumerate() {
                padded[i] = v;
            }
            padded
        } else {
            input.slice(ndarray::s![..self.config.input_dim]).to_owned()
        };
        
        // Calculate input contribution: W_in · u(t)
        let input_contrib = self.w_in.dot(&input_padded);
        
        // Calculate internal dynamics: W · x(t)
        let internal_contrib = self.w.dot(&self.reservoir_state);
        
        // Combined pre-activation
        let pre_activation = &internal_contrib + &input_contrib;
        
        // Apply nonlinearity (tanh for most nodes, pBit for subset)
        let mut next_state: Array1<f64> = pre_activation.mapv(|x| x.tanh());
        
        // Apply pBit nonlinearity to ~20% of nodes (max 50)
        let num_pbit_nodes = (self.config.reservoir_size / 5).min(50);
        if num_pbit_nodes > 0 {
            // Select random nodes for pBit processing
            let node_indices: Vec<usize> = (0..self.config.reservoir_size)
                .choose_multiple(&mut self.rng, num_pbit_nodes)
                .into_iter()
                .collect();
            
            let node_values: Vec<f64> = node_indices.iter().map(|&i| next_state[i]).collect();
            let weights: Vec<f64> = node_indices.iter()
                .map(|&i| self.w[[i, 0]])
                .collect();
            
            if let Ok(pbit_results) = self.apply_pbit_nonlinearity(&node_values, &weights) {
                for (idx, &node_idx) in node_indices.iter().enumerate() {
                    if idx < pbit_results.len() {
                        next_state[node_idx] = pbit_results[idx];
                    }
                }
            }
        }
        
        // Apply leaking rate: x(t+1) = (1-α)·x(t) + α·f(...)
        let alpha = self.config.leaking_rate;
        self.reservoir_state = &self.reservoir_state * (1.0 - alpha) + &next_state * alpha;
        
        Ok(())
    }
    
    /// Extract temporal features from input data
    fn extract_temporal_features(&self, data: &[f64]) -> HashMap<String, f64> {
        let mut features = HashMap::new();
        let n = data.len();
        
        for &window in &self.config.temporal_windows {
            if n >= window {
                let window_data = &data[n - window..];
                
                // Mean
                let mean: f64 = window_data.iter().sum::<f64>() / window as f64;
                features.insert(format!("mean_{}", window), mean);
                
                // Standard deviation
                let variance: f64 = window_data.iter()
                    .map(|x| (x - mean).powi(2))
                    .sum::<f64>() / window as f64;
                let std = variance.sqrt();
                features.insert(format!("std_{}", window), std);
                
                // Trend (mean of differences)
                if window > 1 {
                    let diffs: Vec<f64> = window_data.windows(2)
                        .map(|w| w[1] - w[0])
                        .collect();
                    let trend = diffs.iter().sum::<f64>() / diffs.len() as f64;
                    features.insert(format!("trend_{}", window), trend);
                }
            } else {
                features.insert(format!("mean_{}", window), 0.0);
                features.insert(format!("std_{}", window), 0.0);
                features.insert(format!("trend_{}", window), 0.0);
            }
        }
        
        features
    }
    
    /// Calculate trend output
    fn calculate_trend(&self, prices: &[f64]) -> f64 {
        if prices.len() < 20 {
            return self.w_out.row(0).dot(&self.reservoir_state).tanh();
        }
        
        let n = prices.len();
        let short_ma: f64 = prices[n-5..].iter().sum::<f64>() / 5.0;
        let long_ma: f64 = prices[n-20..].iter().sum::<f64>() / 20.0;
        
        let raw_trend = if long_ma.abs() > 1e-10 {
            (short_ma - long_ma) / long_ma
        } else {
            0.0
        };
        
        let trend = (raw_trend * 10.0).tanh();
        let reservoir_output = self.w_out.row(0).dot(&self.reservoir_state).tanh();
        
        0.7 * trend + 0.3 * reservoir_output
    }
    
    /// Calculate volatility output
    fn calculate_volatility(&self, prices: &[f64]) -> f64 {
        if prices.len() < 21 {
            return 0.5 * (self.w_out.row(1).dot(&self.reservoir_state).tanh() + 1.0);
        }
        
        // Calculate returns
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| if w[0].abs() > 1e-10 { (w[1] - w[0]) / w[0] } else { 0.0 })
            .collect();
        
        let n = returns.len();
        let recent_returns = &returns[n.saturating_sub(20)..];
        let volatility = (recent_returns.iter().map(|r| r.powi(2)).sum::<f64>() / recent_returns.len() as f64).sqrt();
        
        let normalized = 1.0 - (-50.0 * volatility).exp();
        let reservoir_output = 0.5 * (self.w_out.row(1).dot(&self.reservoir_state).tanh() + 1.0);
        
        (0.7 * normalized + 0.3 * reservoir_output).clamp(0.0, 1.0)
    }
    
    /// Calculate momentum output
    fn calculate_momentum(&self, prices: &[f64]) -> f64 {
        if prices.len() < 60 {
            return self.w_out.row(2).dot(&self.reservoir_state).tanh();
        }
        
        let n = prices.len();
        let short_return = prices[n-1] / prices[n-5] - 1.0;
        let medium_return = prices[n-1] / prices[n-20] - 1.0;
        let long_return = prices[n-1] / prices[n-60] - 1.0;
        
        let weighted = 0.5 * short_return + 0.3 * medium_return + 0.2 * long_return;
        let normalized = (weighted * 10.0).tanh();
        let reservoir_output = self.w_out.row(2).dot(&self.reservoir_state).tanh();
        
        (0.7 * normalized + 0.3 * reservoir_output).clamp(-1.0, 1.0)
    }
    
    /// Calculate regime output
    fn calculate_regime(&self, prices: &[f64]) -> f64 {
        if prices.len() < 60 {
            return 0.5 * (self.w_out.row(3).dot(&self.reservoir_state).tanh() + 1.0);
        }
        
        let n = prices.len();
        
        // Calculate returns for regime detection
        let returns: Vec<f64> = prices.windows(2)
            .map(|w| if w[0].abs() > 1e-10 { (w[1] - w[0]) / w[0] } else { 0.0 })
            .collect();
        
        // Short-term volatility
        let short_vol: f64 = returns[returns.len()-10..].iter()
            .map(|r| r.powi(2)).sum::<f64>().sqrt();
        
        // Long-term volatility
        let long_vol: f64 = returns[returns.len()-60..].iter()
            .map(|r| r.powi(2)).sum::<f64>().sqrt();
        
        // Regime: high ratio = volatile/trending, low = stable
        let vol_ratio = if long_vol > 1e-10 { short_vol / long_vol } else { 1.0 };
        let regime = 1.0 / (1.0 + (-2.0 * (vol_ratio - 1.0)).exp()); // Sigmoid
        
        let reservoir_output = 0.5 * (self.w_out.row(3).dot(&self.reservoir_state).tanh() + 1.0);
        
        (0.6 * regime + 0.4 * reservoir_output).clamp(0.0, 1.0)
    }
    
    /// Process input data through the reservoir
    pub fn process(&mut self, prices: &[f64]) -> QuantumResult<ReservoirOutput> {
        let start = std::time::Instant::now();
        
        if !self.is_initialized {
            return Err(QuantumError::computation_error("process", "Reservoir not initialized"));
        }
        
        // Extract temporal features
        let temporal_features = self.extract_temporal_features(prices);
        
        // Convert features to input vector
        let mut input_vec: Vec<f64> = temporal_features.values().copied().collect();
        input_vec.resize(self.config.input_dim, 0.0);
        let input = Array1::from_vec(input_vec);
        
        // Update reservoir state
        self.update_reservoir_state(&input)?;
        
        // Calculate outputs
        let trend = self.calculate_trend(prices);
        let volatility = self.calculate_volatility(prices);
        let momentum = self.calculate_momentum(prices);
        let regime = self.calculate_regime(prices);
        
        let execution_time_ms = start.elapsed().as_secs_f64() * 1000.0;
        
        Ok(ReservoirOutput {
            trend,
            volatility,
            momentum,
            regime,
            temporal_features,
            reservoir_state: self.reservoir_state.to_vec(),
            execution_time_ms,
        })
    }
    
    /// Reset the reservoir state
    pub fn reset(&mut self) {
        self.reservoir_state = Array1::zeros(self.config.reservoir_size);
        self.kernel_cache.clear();
        debug!("Reservoir state reset");
    }
    
    /// Get current reservoir state
    pub fn state(&self) -> &Array1<f64> {
        &self.reservoir_state
    }
    
    /// Get reservoir size
    pub fn reservoir_size(&self) -> usize {
        self.config.reservoir_size
    }
    
    /// Get kernel size
    pub fn kernel_size(&self) -> usize {
        self.config.kernel_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_reservoir_creation() {
        let config = PBitReservoirConfig {
            reservoir_size: 100,
            kernel_size: 4,
            ..Default::default()
        };
        let reservoir = PBitReservoir::new(config).unwrap();
        assert_eq!(reservoir.reservoir_size(), 100);
        assert_eq!(reservoir.kernel_size(), 4);
    }
    
    #[test]
    fn test_pbit_encoding() {
        let config = PBitReservoirConfig::default();
        let reservoir = PBitReservoir::new(config).unwrap();
        
        // RY(0) → P(↑) = 0
        assert!((reservoir.encode_to_pbit_probability(0.0) - 0.0).abs() < 0.01);
        
        // RY(π/2) → P(↑) = 0.5
        assert!((reservoir.encode_to_pbit_probability(0.5) - 0.5).abs() < 0.1);
        
        // RY(π) → P(↑) = 1
        assert!((reservoir.encode_to_pbit_probability(1.0) - 1.0).abs() < 0.01);
    }
    
    #[test]
    fn test_reservoir_processing() {
        let config = PBitReservoirConfig {
            reservoir_size: 50,
            kernel_size: 4,
            ..Default::default()
        };
        let mut reservoir = PBitReservoir::new(config).unwrap();
        
        // Create synthetic price data
        let prices: Vec<f64> = (0..100)
            .map(|i| 100.0 + (i as f64 * 0.1).sin() * 5.0)
            .collect();
        
        let output = reservoir.process(&prices).unwrap();
        
        assert!(output.trend >= -1.0 && output.trend <= 1.0);
        assert!(output.volatility >= 0.0 && output.volatility <= 1.0);
        assert!(output.momentum >= -1.0 && output.momentum <= 1.0);
        assert!(output.regime >= 0.0 && output.regime <= 1.0);
    }
    
    #[test]
    fn test_temporal_features() {
        let config = PBitReservoirConfig::default();
        let reservoir = PBitReservoir::new(config).unwrap();
        
        let data: Vec<f64> = (0..100).map(|i| i as f64).collect();
        let features = reservoir.extract_temporal_features(&data);
        
        assert!(features.contains_key("mean_5"));
        assert!(features.contains_key("std_5"));
        assert!(features.contains_key("trend_5"));
    }
}
